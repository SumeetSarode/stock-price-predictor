"""Unit tests for data/filings_bse.py — BSE fully mocked via respx.

Network calls NEVER made. Mirrors the testing style of test_filings.py
so that maintenance is uniform across the two providers.
"""
from __future__ import annotations

import httpx
import pytest
import respx

from price_predictor.data.filings_bse import (
    BSE_API,
    IST,
    BseFilingsFetchError,
    _bse_attachment_url,
    _build_url,
    _parse_bse_datetime,
    _parse_bse_row,
    _safe_url,
    _to_bse_date_param,
    _validate_inputs,
    fetch_bse_filings,
)


# ─────────────────────────────────────────────────────────────
# Date helpers
# ─────────────────────────────────────────────────────────────
class TestParseBseDatetime:
    def test_iso_with_t(self):
        dt = _parse_bse_datetime("2024-04-26T18:30:00")
        assert dt is not None
        assert (dt.year, dt.month, dt.day, dt.hour, dt.minute) == (
            2024, 4, 26, 18, 30,
        )
        assert dt.tzinfo == IST

    def test_iso_with_millis(self):
        dt = _parse_bse_datetime("2024-04-26T18:30:00.123")
        assert dt is not None
        assert dt.microsecond == 123000

    def test_iso_with_space(self):
        assert _parse_bse_datetime("2024-04-26 18:30:00") is not None

    def test_date_only(self):
        dt = _parse_bse_datetime("2024-04-26")
        assert dt is not None
        assert dt.hour == 0

    def test_human_format(self):
        assert _parse_bse_datetime("26 Apr 2024 18:30:00") is not None

    def test_none(self):
        assert _parse_bse_datetime(None) is None

    def test_empty(self):
        assert _parse_bse_datetime("") is None
        assert _parse_bse_datetime("   ") is None

    def test_garbage(self):
        assert _parse_bse_datetime("not-a-date") is None

    def test_non_string(self):
        # Defensive: BSE has shipped int dates in error before.
        assert _parse_bse_datetime(12345) is None  # type: ignore[arg-type]


class TestToBseDateParam:
    def test_iso_to_yyyymmdd(self):
        assert _to_bse_date_param("2024-04-26") == "20240426"

    def test_no_dashes_passthrough_safe(self):
        # Defensive: no dashes is already in target format; should still work.
        assert _to_bse_date_param("20240426") == "20240426"


class TestSafeUrl:
    """Mirror of the NSE safe-URL contract; BSE has the same junk shapes."""

    def test_valid_https(self):
        assert _safe_url("https://x.com/y.pdf") == "https://x.com/y.pdf"

    def test_strips_whitespace(self):
        assert _safe_url("  https://x.com/y.pdf  ") == "https://x.com/y.pdf"

    @pytest.mark.parametrize("junk", ["", "   ", "-", "n/a", "NA", "None", "null"])
    def test_junk_to_none(self, junk):
        assert _safe_url(junk) is None

    def test_relative_to_none(self):
        assert _safe_url("/some/path.pdf") is None

    def test_non_string_to_none(self):
        assert _safe_url(123) is None
        assert _safe_url(None) is None


class TestBseAttachmentUrl:
    def test_plain_filename_gets_prefix(self):
        url = _bse_attachment_url("abc123.pdf")
        assert url is not None
        assert url.startswith("https://www.bseindia.com/xml-data/corpfiling/AttachLive/")
        assert url.endswith("abc123.pdf")

    def test_empty_to_none(self):
        assert _bse_attachment_url("") is None
        assert _bse_attachment_url(None) is None
        assert _bse_attachment_url("   ") is None


# ─────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────
class TestValidateInputs:
    def test_happy(self):
        # Should NOT raise
        _validate_inputs("500325", "2024-01-01", "2024-12-31")

    @pytest.mark.parametrize("bad", ["", None, 12345])
    def test_bad_scrip_type(self, bad):
        with pytest.raises(ValueError):
            _validate_inputs(bad, "2024-01-01", "2024-12-31")  # type: ignore[arg-type]

    def test_non_numeric_scrip_rejected(self):
        with pytest.raises(ValueError, match="numeric digits"):
            _validate_inputs("RELIANCE", "2024-01-01", "2024-12-31")

    def test_bad_date_format(self):
        with pytest.raises(ValueError, match="ISO"):
            _validate_inputs("500325", "01-01-2024", "2024-12-31")

    def test_inverted_dates(self):
        with pytest.raises(ValueError, match="must be <="):
            _validate_inputs("500325", "2024-12-31", "2024-01-01")


# ─────────────────────────────────────────────────────────────
# URL builder
# ─────────────────────────────────────────────────────────────
class TestBuildUrl:
    def test_url_has_all_required_params(self):
        url = _build_url("500325", "2024-04-01", "2024-04-30")
        assert url.startswith(BSE_API)
        # Params we care about
        assert "strscrip=500325" in url
        assert "strPrevDate=20240401" in url
        assert "strToDate=20240430" in url
        assert "strCat=-1" in url
        assert "strType=C" in url

    def test_lowercase_strscrip_quirk_preserved(self):
        # Documented quirk: BSE accepts strscrip (lowercase), NOT strScrip.
        url = _build_url("500325", "2024-04-01", "2024-04-30")
        assert "strscrip=" in url
        assert "strScrip=" not in url


# ─────────────────────────────────────────────────────────────
# Row parser
# ─────────────────────────────────────────────────────────────
class TestParseBseRow:
    def test_full_row(self):
        row = {
            "SCRIP_CD": 500325,
            "NEWSID": "20240426-1234",
            "NEWS_DT": "2024-04-26T18:30:00",
            "BroadCastDate": "2024-04-26T18:35:00",
            "HEADLINE": "Outcome of Board Meeting",
            "CATEGORYNAME": "Board Meeting",
            "SUBCATNAME": "Outcome of Board Meeting",
            "MORE": "Detailed text here.",
            "ATTACHMENTNAME": "abc.pdf",
            "SLONGNAME": "Reliance Industries Ltd",
        }
        f = _parse_bse_row(row, "RELIANCE")
        assert f is not None
        assert f.symbol == "RELIANCE"
        assert f.kind == "announcement"
        assert f.subject == "Outcome of Board Meeting"
        assert f.event_type == "Board Meeting"
        assert f.description == "Detailed text here."
        assert f.metadata["source"] == "bse"
        assert f.metadata["scrip_code"] == "500325"
        assert f.metadata["news_id"] == "20240426-1234"
        assert f.metadata["company_name"] == "Reliance Industries Ltd"
        assert str(f.attachment_url).endswith("abc.pdf")
        assert f.event_at is not None  # BroadCastDate parsed

    def test_missing_headline_drops(self):
        row = {"SCRIP_CD": 500325, "NEWS_DT": "2024-04-26T18:30:00"}
        assert _parse_bse_row(row, "RELIANCE") is None

    def test_missing_date_drops(self):
        row = {"SCRIP_CD": 500325, "HEADLINE": "Some news"}
        assert _parse_bse_row(row, "RELIANCE") is None

    def test_falls_back_to_newssub(self):
        row = {
            "NEWSSUB": "Press Release",
            "NEWS_DT": "2024-04-26T18:30:00",
            "SCRIP_CD": 500325,
        }
        f = _parse_bse_row(row, "RELIANCE")
        assert f is not None
        assert f.subject == "Press Release"

    def test_blank_optional_fields_omitted_from_metadata(self):
        row = {
            "SCRIP_CD": 500325,
            "HEADLINE": "X",
            "NEWS_DT": "2024-04-26T18:30:00",
            "CATEGORYNAME": "",
            "SUBCATNAME": "",
            "SLONGNAME": "",
            "NEWSID": "",
        }
        f = _parse_bse_row(row, "RELIANCE")
        assert f is not None
        # Only `source` + `scrip_code` should remain.
        assert set(f.metadata.keys()) == {"source", "scrip_code"}

    def test_no_attachment(self):
        row = {
            "SCRIP_CD": 500325, "HEADLINE": "X",
            "NEWS_DT": "2024-04-26T18:30:00",
        }
        f = _parse_bse_row(row, "RELIANCE")
        assert f is not None
        assert f.attachment_url is None


# ─────────────────────────────────────────────────────────────
# fetch_bse_filings — full async flow with respx
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def bse_payload():
    """Two-row Table response in BSE's actual shape."""
    return {
        "Table": [
            {
                "SCRIP_CD": 500325,
                "NEWSID": "20240426-1",
                "NEWS_DT": "2024-04-26T18:30:00",
                "HEADLINE": "Outcome of Board Meeting",
                "CATEGORYNAME": "Board Meeting",
                "SUBCATNAME": "Outcome of Board Meeting",
                "MORE": "Q4 results approved.",
                "ATTACHMENTNAME": "outcome.pdf",
                "SLONGNAME": "Reliance Industries Ltd",
            },
            {
                "SCRIP_CD": 500325,
                "NEWSID": "20240420-1",
                "NEWS_DT": "2024-04-20T14:15:00",
                "HEADLINE": "Press Release on JV",
                "CATEGORYNAME": "Company Update",
                "SUBCATNAME": "",
                "MORE": "",
                "ATTACHMENTNAME": "",
                "SLONGNAME": "Reliance Industries Ltd",
            },
        ],
        "Table1": [{"ROWCNT": 2}],
    }


class TestFetchBseFilings:
    @pytest.mark.asyncio
    @respx.mock
    async def test_happy_path(self, bse_payload):
        respx.get(url__startswith=BSE_API).mock(
            return_value=httpx.Response(200, json=bse_payload)
        )
        filings = await fetch_bse_filings(
            "500325", "2024-04-01", "2024-04-30", symbol="RELIANCE",
        )
        assert len(filings) == 2
        # Newest-first ordering
        assert filings[0].announced_at > filings[1].announced_at
        for f in filings:
            assert f.symbol == "RELIANCE"
            assert f.kind == "announcement"
            assert f.metadata["source"] == "bse"

    @pytest.mark.asyncio
    @respx.mock
    async def test_symbol_defaults_to_scrip_code(self, bse_payload):
        respx.get(url__startswith=BSE_API).mock(
            return_value=httpx.Response(200, json=bse_payload)
        )
        filings = await fetch_bse_filings("500325", "2024-04-01", "2024-04-30")
        assert filings
        for f in filings:
            assert f.symbol == "500325"

    @pytest.mark.asyncio
    @respx.mock
    async def test_empty_table_returns_empty_list(self):
        respx.get(url__startswith=BSE_API).mock(
            return_value=httpx.Response(200, json={"Table": [], "Table1": []})
        )
        filings = await fetch_bse_filings(
            "500325", "2024-04-01", "2024-04-30", symbol="RELIANCE",
        )
        assert filings == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_missing_table_field_returns_empty_list(self):
        # Defensive: BSE has been known to return {} on no-results endpoints.
        respx.get(url__startswith=BSE_API).mock(
            return_value=httpx.Response(200, json={})
        )
        filings = await fetch_bse_filings(
            "500325", "2024-04-01", "2024-04-30", symbol="RELIANCE",
        )
        assert filings == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_table_not_a_list_raises(self):
        respx.get(url__startswith=BSE_API).mock(
            return_value=httpx.Response(200, json={"Table": "oops"})
        )
        with pytest.raises(BseFilingsFetchError, match="not a list"):
            await fetch_bse_filings(
                "500325", "2024-04-01", "2024-04-30", symbol="RELIANCE",
            )

    @pytest.mark.asyncio
    @respx.mock
    async def test_http_500_raises(self):
        respx.get(url__startswith=BSE_API).mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        with pytest.raises(BseFilingsFetchError, match="HTTP error"):
            await fetch_bse_filings(
                "500325", "2024-04-01", "2024-04-30", symbol="RELIANCE",
            )

    @pytest.mark.asyncio
    @respx.mock
    async def test_network_error_raises(self):
        respx.get(url__startswith=BSE_API).mock(
            side_effect=httpx.ConnectError("DNS hosed")
        )
        with pytest.raises(BseFilingsFetchError, match="network error"):
            await fetch_bse_filings(
                "500325", "2024-04-01", "2024-04-30", symbol="RELIANCE",
            )

    @pytest.mark.asyncio
    @respx.mock
    async def test_non_json_payload_raises(self):
        respx.get(url__startswith=BSE_API).mock(
            return_value=httpx.Response(200, text="<html>blocked</html>")
        )
        with pytest.raises(BseFilingsFetchError, match="non-JSON"):
            await fetch_bse_filings(
                "500325", "2024-04-01", "2024-04-30", symbol="RELIANCE",
            )

    @pytest.mark.asyncio
    @respx.mock
    async def test_bad_rows_in_table_dropped_not_fatal(self):
        # A mix of valid + invalid rows. We expect 1 valid filing returned,
        # and zero exceptions.
        payload = {
            "Table": [
                {"SCRIP_CD": 500325, "HEADLINE": "Good", "NEWS_DT": "2024-04-26T18:30:00"},
                {"SCRIP_CD": 500325},                       # missing headline + date
                "totally-not-a-dict",                       # wrong type
                {"SCRIP_CD": 500325, "HEADLINE": "Bad date", "NEWS_DT": "garbage"},
            ],
        }
        respx.get(url__startswith=BSE_API).mock(
            return_value=httpx.Response(200, json=payload)
        )
        filings = await fetch_bse_filings(
            "500325", "2024-04-01", "2024-04-30", symbol="RELIANCE",
        )
        assert len(filings) == 1
        assert filings[0].subject == "Good"

    @pytest.mark.asyncio
    async def test_validation_runs_before_network(self):
        # Should raise WITHOUT touching the network — no respx mock means
        # any real network call would be intercepted by respx anyway, but
        # we don't even instantiate a respx mock for this test.
        with pytest.raises(ValueError):
            await fetch_bse_filings("RELIANCE", "2024-04-01", "2024-04-30")

    @pytest.mark.asyncio
    @respx.mock
    async def test_caller_supplied_client_not_closed(self, bse_payload):
        """If caller provides their own AsyncClient, we MUST NOT close it
        (caller owns lifecycle)."""
        respx.get(url__startswith=BSE_API).mock(
            return_value=httpx.Response(200, json=bse_payload)
        )
        async with httpx.AsyncClient() as client:
            await fetch_bse_filings(
                "500325", "2024-04-01", "2024-04-30",
                symbol="RELIANCE", client=client,
            )
            # Client must still be usable.
            assert not client.is_closed
