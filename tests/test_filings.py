"""Unit tests for data/filings.py — NSE fully mocked via respx.

Network calls NEVER made. Integration test (off-VPN, marked) lives at the
bottom and skips gracefully when NSE is unreachable.
"""
from __future__ import annotations

import httpx
import pandas as pd
import pytest
import respx

from price_predictor.data.filings import (
    DEFAULT_KINDS,
    IST,
    NSE_BASE,
    NSE_HOMEPAGE,
    FilingsFetchError,
    _parse_announcement,
    _parse_board_meeting,
    _parse_corporate_action,
    _parse_financial_result,
    _parse_nse_datetime,
    _to_nse_date_param,
    _validate_inputs,
    fetch_filings,
    fetch_filings_batch,
)
from price_predictor.data.schema import Filing


# ─────────────────────────────────────────────────────────────
# Date helpers
# ─────────────────────────────────────────────────────────────
class TestParseNseDatetime:
    def test_full_timestamp(self):
        dt = _parse_nse_datetime("26-Apr-2026 18:30:00")
        assert dt is not None
        assert dt.year == 2026 and dt.month == 4 and dt.day == 26
        assert dt.hour == 18 and dt.minute == 30
        assert dt.tzinfo == IST

    def test_date_only(self):
        dt = _parse_nse_datetime("26-Apr-2026")
        assert dt is not None
        assert dt.tzinfo == IST

    def test_numeric_month(self):
        dt = _parse_nse_datetime("26-04-2026")
        assert dt is not None
        assert dt.month == 4

    def test_none(self):
        assert _parse_nse_datetime(None) is None

    def test_empty(self):
        assert _parse_nse_datetime("") is None

    def test_garbage(self):
        assert _parse_nse_datetime("not-a-date") is None

    def test_non_string(self):
        assert _parse_nse_datetime(12345) is None  # type: ignore[arg-type]


class TestToNseDateParam:
    def test_iso_to_nse(self):
        assert _to_nse_date_param("2026-01-15") == "15-01-2026"

    def test_invalid_iso_raises(self):
        with pytest.raises(ValueError):
            _to_nse_date_param("not-a-date")


# ─────────────────────────────────────────────────────────────
# _validate_inputs
# ─────────────────────────────────────────────────────────────
class TestValidateInputs:
    def test_happy(self):
        _validate_inputs("RELIANCE", "2026-01-01", "2026-01-31")

    def test_empty_symbol(self):
        with pytest.raises(ValueError, match="non-empty string"):
            _validate_inputs("", "2026-01-01", "2026-01-31")

    def test_ns_suffix_rejected(self):
        """NSE uses bare symbols, not .NS suffix."""
        with pytest.raises(ValueError, match="bare symbol"):
            _validate_inputs("RELIANCE.NS", "2026-01-01", "2026-01-31")

    def test_bad_start_date(self):
        with pytest.raises(ValueError, match="start"):
            _validate_inputs("RELIANCE", "01-01-2026", "2026-01-31")

    def test_bad_end_date(self):
        with pytest.raises(ValueError, match="end"):
            _validate_inputs("RELIANCE", "2026-01-01", "not-a-date")

    def test_start_after_end(self):
        with pytest.raises(ValueError, match=r"start.*<= end"):
            _validate_inputs("RELIANCE", "2026-01-31", "2026-01-01")


# ─────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────
class TestParseAnnouncement:
    def test_happy(self):
        item = {
            "an_dt": "26-Apr-2026 18:30:00",
            "desc": "Audited Financial Results for Q4",
            "smIndustry": "Refineries",
            "attchmntFile": "https://archives.nseindia.com/test.pdf",
            "attchmntText": "Detailed text here",
        }
        f = _parse_announcement(item, "RELIANCE")
        assert f is not None
        assert f.kind == "announcement"
        assert f.symbol == "RELIANCE"
        assert f.subject == "Audited Financial Results for Q4"
        assert f.event_type == "Refineries"
        assert str(f.attachment_url) == "https://archives.nseindia.com/test.pdf"
        assert f.event_at is None  # announcements have no separate event date

    def test_no_date_dropped(self):
        assert _parse_announcement({"desc": "X"}, "RELIANCE") is None

    def test_no_subject_dropped(self):
        item = {"an_dt": "26-Apr-2026", "desc": "", "attchmntText": ""}
        assert _parse_announcement(item, "RELIANCE") is None

    def test_extras_in_metadata(self):
        item = {
            "an_dt": "26-Apr-2026",
            "desc": "Test",
            "extra_field": "preserved",
        }
        f = _parse_announcement(item, "RELIANCE")
        assert f is not None
        assert f.metadata.get("extra_field") == "preserved"


class TestParseBoardMeeting:
    def test_happy_with_both_dates(self):
        item = {
            "bm_purpose": "Quarterly Results",
            "bm_desc": "Board to consider Q4 results",
            "bm_date": "10-May-2026",
            "bm_timestamp": "26-Apr-2026 14:00:00",
        }
        f = _parse_board_meeting(item, "RELIANCE")
        assert f is not None
        assert f.kind == "board_meeting"
        assert f.subject == "Quarterly Results"
        assert f.event_at is not None
        assert f.event_at.day == 10  # meeting date
        assert f.announced_at.day == 26  # filing date

    def test_only_event_date(self):
        """Meeting date present, announce timestamp missing — use event as both."""
        item = {"bm_purpose": "Audit", "bm_date": "10-May-2026"}
        f = _parse_board_meeting(item, "RELIANCE")
        assert f is not None
        assert f.event_at == f.announced_at

    def test_no_dates_dropped(self):
        assert _parse_board_meeting({"bm_purpose": "X"}, "RELIANCE") is None

    def test_default_subject_when_purpose_missing(self):
        item = {"bm_date": "10-May-2026"}
        f = _parse_board_meeting(item, "RELIANCE")
        assert f is not None
        assert f.subject == "Board Meeting"


class TestParseCorporateAction:
    def test_happy_dividend(self):
        item = {
            "subject": "Dividend - Rs 9 Per Share",
            "exDate": "08-Aug-2026",
        }
        f = _parse_corporate_action(item, "RELIANCE")
        assert f is not None
        assert f.kind == "corporate_action"
        assert f.event_type == "Dividend"
        assert f.event_at is not None
        assert f.event_at.month == 8

    def test_happy_split(self):
        item = {"subject": "Stock Split From Rs.10/- To Rs.2/- Per Share", "exDate": "15-Jun-2026"}
        f = _parse_corporate_action(item, "RELIANCE")
        assert f is not None
        assert f.event_type == "Stock"

    def test_no_ex_date_dropped(self):
        assert _parse_corporate_action({"subject": "Dividend"}, "RELIANCE") is None

    def test_no_subject_dropped(self):
        assert _parse_corporate_action({"exDate": "08-Aug-2026"}, "RELIANCE") is None

    def test_announced_equals_event(self):
        """Corp actions: NSE doesn't expose announce date; we use exDate as both."""
        item = {"subject": "Dividend - Rs 5", "exDate": "08-Aug-2026"}
        f = _parse_corporate_action(item, "RELIANCE")
        assert f is not None
        assert f.announced_at == f.event_at


class TestParseFinancialResult:
    def test_happy(self):
        item = {
            "broadCastDate": "26-Apr-2026 18:30:00",
            "fromDate": "01-Jan-2026",
            "toDate": "31-Mar-2026",
            "audited": "Audited",
            "consolidated": "Consolidated",
            "xbrlAttachment": "https://archives.nseindia.com/results.xml",
        }
        f = _parse_financial_result(item, "RELIANCE")
        assert f is not None
        assert f.kind == "financial_result"
        assert "Audited" in f.subject
        assert "Consolidated" in f.subject
        assert "01-Jan-2026" in f.subject
        assert f.event_at is not None
        assert f.event_at.month == 3  # toDate
        assert str(f.attachment_url).startswith("https://archives")

    def test_no_broadcast_date_dropped(self):
        assert _parse_financial_result({"fromDate": "01-Jan-2026"}, "RELIANCE") is None


# ─────────────────────────────────────────────────────────────
# fetch_filings (full integration of warmup + fan-out + parse)
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def mock_warmup(respx_mock):
    """Mock the NSE homepage warmup call."""
    respx_mock.get(NSE_HOMEPAGE).mock(
        return_value=httpx.Response(200, text="<html>OK</html>")
    )


@pytest.fixture
def announcement_payload():
    return [
        {
            "an_dt": "26-Apr-2026 18:30:00",
            "desc": "Audited Financial Results for Q4",
            "smIndustry": "Refineries",
            "attchmntFile": "https://archives.nseindia.com/test.pdf",
            "attchmntText": "",
        },
        {
            "an_dt": "20-Apr-2026 14:15:00",
            "desc": "Press release on JV announcement",
            "smIndustry": "Refineries",
        },
    ]


@pytest.fixture
def board_meeting_payload():
    return [
        {
            "bm_purpose": "Quarterly Results",
            "bm_desc": "Board to consider Q4 results",
            "bm_date": "10-May-2026",
            "bm_timestamp": "20-Apr-2026 14:00:00",
        }
    ]


@pytest.fixture
def corporate_action_payload():
    return [
        {"subject": "Dividend - Rs 9 Per Share", "exDate": "08-Aug-2026"},
        {"subject": "Annual General Meeting", "exDate": "15-Sep-2026"},
    ]


@pytest.fixture
def financial_result_payload():
    return [
        {
            "broadCastDate": "26-Apr-2026 18:30:00",
            "fromDate": "01-Jan-2026",
            "toDate": "31-Mar-2026",
            "audited": "Audited",
            "consolidated": "Consolidated",
            "xbrlAttachment": "https://archives.nseindia.com/results.xml",
        }
    ]


def _mock_all_endpoints(
    respx_mock, ann_payload, bm_payload, ca_payload, fr_payload,
):
    """Helper: register all 4 endpoint mocks."""
    respx_mock.get(url__startswith=f"{NSE_BASE}/api/corporate-announcements").mock(
        return_value=httpx.Response(200, json=ann_payload)
    )
    respx_mock.get(url__startswith=f"{NSE_BASE}/api/corporate-board-meetings").mock(
        return_value=httpx.Response(200, json=bm_payload)
    )
    respx_mock.get(url__startswith=f"{NSE_BASE}/api/corporates-corporateActions").mock(
        return_value=httpx.Response(200, json=ca_payload)
    )
    respx_mock.get(url__startswith=f"{NSE_BASE}/api/corporates-financial-results").mock(
        return_value=httpx.Response(200, json=fr_payload)
    )


class TestFetchFilings:
    @pytest.mark.asyncio
    @respx.mock
    async def test_happy_all_endpoints(
        self, mock_warmup, announcement_payload, board_meeting_payload,
        corporate_action_payload, financial_result_payload,
    ):
        _mock_all_endpoints(
            respx.mock,
            announcement_payload, board_meeting_payload,
            corporate_action_payload, financial_result_payload,
        )

        df = await fetch_filings("RELIANCE", "2026-04-01", "2026-09-30")

        assert isinstance(df, pd.DataFrame)
        # 2 announcements + 1 board meeting + 2 corp actions + 1 financial = 6
        assert len(df) == 6
        # All 4 kinds represented
        assert set(df["kind"].unique()) == {
            "announcement", "board_meeting", "corporate_action", "financial_result",
        }
        # Sorted by announced_at descending
        assert df["announced_at"].is_monotonic_decreasing

    @pytest.mark.asyncio
    @respx.mock
    async def test_subset_of_kinds(
        self, mock_warmup, announcement_payload,
    ):
        respx.mock.get(
            url__startswith=f"{NSE_BASE}/api/corporate-announcements"
        ).mock(return_value=httpx.Response(200, json=announcement_payload))

        df = await fetch_filings(
            "RELIANCE", "2026-04-01", "2026-04-30", kinds=["announcement"],
        )
        assert len(df) == 2
        assert set(df["kind"].unique()) == {"announcement"}

    @pytest.mark.asyncio
    @respx.mock
    async def test_one_endpoint_fails_others_succeed(
        self, mock_warmup, announcement_payload, board_meeting_payload,
        financial_result_payload,
    ):
        # Mock 3 endpoints OK, 1 fails with 500
        respx.mock.get(
            url__startswith=f"{NSE_BASE}/api/corporate-announcements"
        ).mock(return_value=httpx.Response(200, json=announcement_payload))
        respx.mock.get(
            url__startswith=f"{NSE_BASE}/api/corporate-board-meetings"
        ).mock(return_value=httpx.Response(200, json=board_meeting_payload))
        respx.mock.get(
            url__startswith=f"{NSE_BASE}/api/corporates-corporateActions"
        ).mock(return_value=httpx.Response(500, text="Server error"))
        respx.mock.get(
            url__startswith=f"{NSE_BASE}/api/corporates-financial-results"
        ).mock(return_value=httpx.Response(200, json=financial_result_payload))

        # Should NOT raise — partial failure is tolerated
        df = await fetch_filings("RELIANCE", "2026-04-01", "2026-09-30")
        # 2 ann + 1 bm + 1 fr = 4 (corp actions failed)
        assert len(df) == 4
        assert "corporate_action" not in df["kind"].unique()

    @pytest.mark.asyncio
    @respx.mock
    async def test_all_endpoints_fail_raises(self, mock_warmup):
        for url in (
            "/api/corporate-announcements", "/api/corporate-board-meetings",
            "/api/corporates-corporateActions", "/api/corporates-financial-results",
        ):
            respx.mock.get(url__startswith=NSE_BASE + url).mock(
                return_value=httpx.Response(500, text="Down")
            )

        with pytest.raises(FilingsFetchError, match="All 4 filings endpoints failed"):
            await fetch_filings("RELIANCE", "2026-04-01", "2026-09-30")

    @pytest.mark.asyncio
    @respx.mock
    async def test_warmup_failure_raises(self):
        respx.mock.get(NSE_HOMEPAGE).mock(return_value=httpx.Response(403))

        with pytest.raises(FilingsFetchError, match="warmup"):
            await fetch_filings("RELIANCE", "2026-04-01", "2026-04-30")

    @pytest.mark.asyncio
    @respx.mock
    async def test_empty_responses(self, mock_warmup):
        for url in (
            "/api/corporate-announcements", "/api/corporate-board-meetings",
            "/api/corporates-corporateActions", "/api/corporates-financial-results",
        ):
            respx.mock.get(url__startswith=NSE_BASE + url).mock(
                return_value=httpx.Response(200, json=[])
            )

        df = await fetch_filings("RELIANCE", "2026-04-01", "2026-04-30")
        # Empty result = success-with-0-rows (project rule)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        # Columns still present
        assert "kind" in df.columns
        assert "announced_at" in df.columns

    @pytest.mark.asyncio
    @respx.mock
    async def test_dict_wrapped_payload(self, mock_warmup, announcement_payload):
        """Some NSE endpoints wrap data in {'data': [...]} or {'rows': [...]}."""
        respx.mock.get(
            url__startswith=f"{NSE_BASE}/api/corporate-announcements"
        ).mock(return_value=httpx.Response(200, json={"data": announcement_payload}))

        df = await fetch_filings(
            "RELIANCE", "2026-04-01", "2026-04-30", kinds=["announcement"],
        )
        assert len(df) == 2

    @pytest.mark.asyncio
    @respx.mock
    async def test_non_json_response_raises(self, mock_warmup):
        respx.mock.get(
            url__startswith=f"{NSE_BASE}/api/corporate-announcements"
        ).mock(return_value=httpx.Response(200, text="<html>NOT JSON</html>"))

        with pytest.raises(FilingsFetchError, match=r"non-JSON|All.*failed"):
            await fetch_filings(
                "RELIANCE", "2026-04-01", "2026-04-30", kinds=["announcement"],
            )

    @pytest.mark.asyncio
    async def test_invalid_input_no_network_call(self):
        # Invalid before any network — no respx needed
        with pytest.raises(ValueError):
            await fetch_filings("RELIANCE.NS", "2026-04-01", "2026-04-30")

    @pytest.mark.asyncio
    async def test_empty_kinds_list_raises(self):
        with pytest.raises(ValueError, match="non-empty list"):
            await fetch_filings("RELIANCE", "2026-04-01", "2026-04-30", kinds=[])

    def test_default_kinds_includes_all_four(self):
        assert set(DEFAULT_KINDS) == {
            "announcement", "board_meeting", "corporate_action", "financial_result",
        }

    @pytest.mark.asyncio
    @respx.mock
    async def test_board_meeting_date_filter_applied(self, mock_warmup):
        """board_meeting endpoint doesn't accept date filter; we filter post-fetch."""
        # Two meetings: one inside window, one outside
        respx.mock.get(
            url__startswith=f"{NSE_BASE}/api/corporate-board-meetings"
        ).mock(return_value=httpx.Response(200, json=[
            {"bm_purpose": "In-window", "bm_date": "10-May-2026"},
            {"bm_purpose": "Out-of-window", "bm_date": "10-Dec-2026"},
        ]))

        df = await fetch_filings(
            "RELIANCE", "2026-05-01", "2026-05-31", kinds=["board_meeting"],
        )
        assert len(df) == 1
        assert df.iloc[0]["subject"] == "In-window"


# ─────────────────────────────────────────────────────────────
# fetch_filings_batch
# ─────────────────────────────────────────────────────────────
class TestFetchFilingsBatch:
    @pytest.mark.asyncio
    async def test_empty_list(self):
        result = await fetch_filings_batch([], "2026-04-01", "2026-04-30")
        assert result == {}

    @pytest.mark.asyncio
    @respx.mock
    async def test_all_succeed(self, mock_warmup, announcement_payload):
        respx.mock.get(
            url__startswith=f"{NSE_BASE}/api/corporate-announcements"
        ).mock(return_value=httpx.Response(200, json=announcement_payload))

        result = await fetch_filings_batch(
            ["RELIANCE", "TCS"],
            "2026-04-01", "2026-04-30",
            kinds=["announcement"],
        )
        assert len(result) == 2
        assert all(isinstance(v, pd.DataFrame) for v in result.values())
        assert len(result["RELIANCE"]) == 2
        assert len(result["TCS"]) == 2


# ─────────────────────────────────────────────────────────────
# DataFrame conversion edge cases
# ─────────────────────────────────────────────────────────────
class TestDataFrameConversion:
    @pytest.mark.asyncio
    @respx.mock
    async def test_attachment_url_serialized_as_string(
        self, mock_warmup, announcement_payload,
    ):
        respx.mock.get(
            url__startswith=f"{NSE_BASE}/api/corporate-announcements"
        ).mock(return_value=httpx.Response(200, json=announcement_payload))

        df = await fetch_filings(
            "RELIANCE", "2026-04-01", "2026-04-30", kinds=["announcement"],
        )
        # First row had attachment, second didn't
        assert isinstance(df.iloc[0]["attachment_url"], str)


# ─────────────────────────────────────────────────────────────
# Integration test (real NSE) — marked, skips gracefully
# ─────────────────────────────────────────────────────────────
@pytest.mark.integration
class TestIntegrationNse:
    @pytest.mark.asyncio
    async def test_real_nse_announcements(self):
        """Real NSE call for RELIANCE last 30 days. Skips on network failure.

        Walmart corp DNS blocks www.nseindia.com — run from non-corp network
        to verify. Like the GDELT test, this is environmental, not a code bug.
        """
        from datetime import datetime, timedelta

        end = (datetime.now().date() - timedelta(days=1)).strftime("%Y-%m-%d")
        start = (datetime.now().date() - timedelta(days=30)).strftime("%Y-%m-%d")

        try:
            df = await fetch_filings(
                "RELIANCE", start, end,
                kinds=["announcement"],  # Just one endpoint to keep test fast
            )
        except FilingsFetchError as e:
            pytest.skip(f"NSE unreachable (likely network/proxy): {e}")

        # Reliance is heavily covered — should have at least 1 filing in 30 days
        assert isinstance(df, pd.DataFrame)
        # Allow 0 if happens to be a quiet month, but verify shape
        if len(df) > 0:
            assert df["symbol"].iloc[0] == "RELIANCE"
            assert df["kind"].iloc[0] == "announcement"
            assert pd.notna(df["announced_at"].iloc[0])


# Keep imported names alive for ruff
_ = (Filing,)
