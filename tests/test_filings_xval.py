"""Unit tests for data/filings_xval.py — both providers fully mocked.

Cross-validator behavior under test:
- Subject fingerprint normalisation
- Dedup key collision (same date, paraphrased subjects)
- Merge logic with corroboration flags
- Partial failure tolerance (one source down → other still returns)
- Total failure (both down → CrossValidationError)
- DataFrame column shape (corroborated + sources promoted out of metadata)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd
import pytest
import respx

from price_predictor.data.filings import NSE_BASE, NSE_HOMEPAGE
from price_predictor.data.filings_bse import BSE_API
from price_predictor.data.filings_xval import (
    CrossValidationError,
    _dedup_key,
    _merge_with_corroboration,
    fetch_filings_cross_validated,
    subject_fingerprint,
)
from price_predictor.data.schema import Filing

IST = timezone(timedelta(hours=5, minutes=30))


# ─────────────────────────────────────────────────────────────
# subject_fingerprint
# ─────────────────────────────────────────────────────────────
class TestSubjectFingerprint:
    def test_identical_strings_match(self):
        assert subject_fingerprint("Board Meeting Outcome") == \
               subject_fingerprint("Board Meeting Outcome")

    def test_case_insensitive(self):
        assert subject_fingerprint("Board Meeting Outcome") == \
               subject_fingerprint("BOARD MEETING OUTCOME")

    def test_punctuation_invariant(self):
        # The whole point: NSE adds " - 26-Apr-2024" suffix that BSE doesn't.
        assert subject_fingerprint("Board Meeting Outcome - 26-Apr-2024") == \
               subject_fingerprint("Board Meeting Outcome 26 Apr 2024")

    def test_whitespace_collapsed(self):
        assert subject_fingerprint("Board   Meeting\tOutcome") == \
               subject_fingerprint("Board Meeting Outcome")

    def test_empty_string_returns_empty(self):
        assert subject_fingerprint("") == ""

    def test_whitespace_only_returns_empty(self):
        assert subject_fingerprint("   ") == ""

    def test_returns_short_hex_string(self):
        fp = subject_fingerprint("Board Meeting")
        assert len(fp) == 12
        assert all(c in "0123456789abcdef" for c in fp)

    def test_different_subjects_different_fingerprints(self):
        assert subject_fingerprint("Board Meeting") != \
               subject_fingerprint("Annual General Meeting")


# ─────────────────────────────────────────────────────────────
# _dedup_key
# ─────────────────────────────────────────────────────────────
def _make_filing(*, announced_at: datetime, subject: str, kind: str = "announcement",
                 metadata: dict | None = None, symbol: str = "RELIANCE") -> Filing:
    """Helper to build minimal Filings without typing out every field."""
    return Filing(
        symbol=symbol,
        kind=kind,  # type: ignore[arg-type]
        announced_at=announced_at,
        subject=subject,
        metadata=metadata or {},
    )


class TestDedupKey:
    def test_same_day_same_subject_match(self):
        f1 = _make_filing(
            announced_at=datetime(2024, 4, 26, 10, 0, tzinfo=IST),
            subject="Board Meeting",
        )
        f2 = _make_filing(
            announced_at=datetime(2024, 4, 26, 14, 30, tzinfo=IST),
            subject="Board Meeting",
        )
        # Same calendar date, same fingerprint → same key, despite different
        # times of day.
        assert _dedup_key(f1) == _dedup_key(f2)

    def test_different_day_different_key(self):
        f1 = _make_filing(
            announced_at=datetime(2024, 4, 26, 10, 0, tzinfo=IST),
            subject="Board Meeting",
        )
        f2 = _make_filing(
            announced_at=datetime(2024, 4, 27, 10, 0, tzinfo=IST),
            subject="Board Meeting",
        )
        assert _dedup_key(f1) != _dedup_key(f2)


# ─────────────────────────────────────────────────────────────
# _merge_with_corroboration
# ─────────────────────────────────────────────────────────────
class TestMergeWithCorroboration:
    def test_match_marks_both_corroborated(self):
        nse = _make_filing(
            announced_at=datetime(2024, 4, 26, 10, 0, tzinfo=IST),
            subject="Outcome of Board Meeting",
            metadata={"source": "nse"},
        )
        bse = _make_filing(
            announced_at=datetime(2024, 4, 26, 14, 0, tzinfo=IST),
            subject="outcome of board meeting!",  # punctuation+case differs
            metadata={"source": "bse", "news_id": "BSE-123"},
        )
        out = _merge_with_corroboration([nse], [bse])
        assert len(out) == 1
        assert out[0].metadata["corroborated"] is True
        assert out[0].metadata["sources"] == ["nse", "bse"]
        assert out[0].metadata["bse_news_id"] == "BSE-123"
        # NSE wins canonicality (subject preserved from NSE).
        assert out[0].subject == "Outcome of Board Meeting"

    def test_nse_only_marked_uncorroborated(self):
        nse = _make_filing(
            announced_at=datetime(2024, 4, 26, 10, 0, tzinfo=IST),
            subject="Press Release JV",
        )
        out = _merge_with_corroboration([nse], [])
        assert len(out) == 1
        assert out[0].metadata["corroborated"] is False
        assert out[0].metadata["sources"] == ["nse"]

    def test_bse_only_marked_uncorroborated(self):
        bse = _make_filing(
            announced_at=datetime(2024, 4, 26, 10, 0, tzinfo=IST),
            subject="Press Release JV",
        )
        out = _merge_with_corroboration([], [bse])
        assert len(out) == 1
        assert out[0].metadata["corroborated"] is False
        assert out[0].metadata["sources"] == ["bse"]

    def test_mixed_corpus(self):
        # Two corroborated, one NSE-only, one BSE-only
        d1 = datetime(2024, 4, 26, 10, 0, tzinfo=IST)
        d2 = datetime(2024, 4, 25, 10, 0, tzinfo=IST)
        d3 = datetime(2024, 4, 24, 10, 0, tzinfo=IST)
        d4 = datetime(2024, 4, 23, 10, 0, tzinfo=IST)
        nse = [
            _make_filing(announced_at=d1, subject="Board Meeting"),
            _make_filing(announced_at=d2, subject="Q4 Results"),
            _make_filing(announced_at=d3, subject="NSE-Only Press Release"),
        ]
        bse = [
            _make_filing(announced_at=d1, subject="board meeting"),
            _make_filing(announced_at=d2, subject="Q4 Results."),
            _make_filing(announced_at=d4, subject="BSE-Only Update"),
        ]
        out = _merge_with_corroboration(nse, bse)
        # 4 unique keys total: d1 (matched), d2 (matched), d3 (NSE-only), d4 (BSE-only)
        assert len(out) == 4
        # d1 + d2 matched (case / punctuation diffs only)
        d1_hit = next(f for f in out if f.announced_at.date() == d1.date())
        d2_hit = next(f for f in out if f.announced_at.date() == d2.date())
        assert d1_hit.metadata["corroborated"] is True
        assert d2_hit.metadata["corroborated"] is True
        # d3 NSE-only, d4 BSE-only
        d3_hit = next(f for f in out if f.announced_at.date() == d3.date())
        d4_hit = next(f for f in out if f.announced_at.date() == d4.date())
        assert d3_hit.metadata["sources"] == ["nse"]
        assert d4_hit.metadata["sources"] == ["bse"]

    def test_paraphrased_subjects_match_thanks_to_fingerprint(self):
        # Punctuation/whitespace diffs match; ADDED CONTENT doesn't (by design).
        nse = _make_filing(
            announced_at=datetime(2024, 4, 26, 10, 0, tzinfo=IST),
            subject="Outcome of Board Meeting",
        )
        bse = _make_filing(
            announced_at=datetime(2024, 4, 26, 14, 0, tzinfo=IST),
            subject="outcome   of\tBoard, Meeting!",
        )
        out = _merge_with_corroboration([nse], [bse])
        assert len(out) == 1
        assert out[0].metadata["corroborated"] is True

    def test_caller_metadata_not_mutated(self):
        # Defensive: merging must not mutate the input filings' metadata
        # dict in place — pydantic copies, but we double-check.
        original_meta = {"source": "nse"}
        nse = _make_filing(
            announced_at=datetime(2024, 4, 26, 10, 0, tzinfo=IST),
            subject="Board Meeting",
            metadata=original_meta,
        )
        _merge_with_corroboration([nse], [])
        assert "corroborated" not in original_meta

    def test_output_sorted_newest_first(self):
        d_old = datetime(2024, 4, 20, 10, 0, tzinfo=IST)
        d_new = datetime(2024, 4, 26, 10, 0, tzinfo=IST)
        nse = [_make_filing(announced_at=d_old, subject="Old")]
        bse = [_make_filing(announced_at=d_new, subject="New")]
        out = _merge_with_corroboration(nse, bse)
        assert out[0].announced_at == d_new
        assert out[1].announced_at == d_old


# ─────────────────────────────────────────────────────────────
# fetch_filings_cross_validated — async respx integration
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def mock_warmup(respx_mock):
    """NSE warmup response."""
    respx_mock.get(NSE_HOMEPAGE).mock(
        return_value=httpx.Response(200, text="<html>OK</html>")
    )


def _mock_nse_endpoints(respx_mock, ann=None, bm=None, ca=None, fr=None):
    """Wire all NSE endpoints to empty/given payloads. Empty by default."""
    respx_mock.get(url__startswith=f"{NSE_BASE}/api/corporate-announcements").mock(
        return_value=httpx.Response(200, json=ann or [])
    )
    respx_mock.get(url__startswith=f"{NSE_BASE}/api/corporate-board-meetings").mock(
        return_value=httpx.Response(200, json=bm or [])
    )
    respx_mock.get(url__startswith=f"{NSE_BASE}/api/corporates-corporateActions").mock(
        return_value=httpx.Response(200, json=ca or [])
    )
    respx_mock.get(url__startswith=f"{NSE_BASE}/api/corporates-financial-results").mock(
        return_value=httpx.Response(200, json=fr or [])
    )


def _bse_payload(rows: list[dict] | None = None) -> dict:
    return {"Table": rows or [], "Table1": [{"ROWCNT": len(rows or [])}]}


class TestFetchFilingsCrossValidated:
    @pytest.mark.asyncio
    @respx.mock
    async def test_happy_path_corroboration(self, mock_warmup):
        # NSE returns one announcement; BSE returns the same headline on
        # the same date → should be marked corroborated=True.
        nse_payload = [
            {
                "an_dt": "26-Apr-2024 10:00:00",
                "desc": "Outcome of Board Meeting",
                "smIndustry": "Refineries",
            }
        ]
        bse_payload_obj = _bse_payload([
            {
                "SCRIP_CD": 500325,
                "NEWSID": "BSE-1",
                "NEWS_DT": "2024-04-26T14:00:00",
                "HEADLINE": "outcome of board meeting!",  # case+punct diff
                "CATEGORYNAME": "Board Meeting",
            }
        ])
        _mock_nse_endpoints(respx.mock, ann=nse_payload)
        respx.get(url__startswith=BSE_API).mock(
            return_value=httpx.Response(200, json=bse_payload_obj)
        )

        df = await fetch_filings_cross_validated(
            "RELIANCE", "500325", "2024-04-01", "2024-04-30",
        )
        # We get 1 unique merged row (NSE canonical, BSE corroborated)
        assert len(df) == 1
        row = df.iloc[0]
        assert bool(row["corroborated"]) is True
        assert row["sources"] == ["nse", "bse"]
        assert row["metadata"]["bse_news_id"] == "BSE-1"
    @pytest.mark.asyncio
    @respx.mock
    async def test_nse_only_marked_uncorroborated(self, mock_warmup):
        nse_payload = [
            {
                "an_dt": "26-Apr-2024 10:00:00",
                "desc": "NSE-only filing",
                "smIndustry": "Refineries",
            }
        ]
        _mock_nse_endpoints(respx.mock, ann=nse_payload)
        respx.get(url__startswith=BSE_API).mock(
            return_value=httpx.Response(200, json=_bse_payload([]))
        )
        df = await fetch_filings_cross_validated(
            "RELIANCE", "500325", "2024-04-01", "2024-04-30",
        )
        assert len(df) == 1
        assert bool(df.iloc[0]["corroborated"]) is False
        assert df.iloc[0]["sources"] == ["nse"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_bse_only_marked_uncorroborated(self, mock_warmup):
        bse_payload_obj = _bse_payload([
            {
                "SCRIP_CD": 500325,
                "NEWSID": "BSE-1",
                "NEWS_DT": "2024-04-26T14:00:00",
                "HEADLINE": "BSE-only headline",
            }
        ])
        _mock_nse_endpoints(respx.mock)  # all empty
        respx.get(url__startswith=BSE_API).mock(
            return_value=httpx.Response(200, json=bse_payload_obj)
        )
        df = await fetch_filings_cross_validated(
            "RELIANCE", "500325", "2024-04-01", "2024-04-30",
        )
        assert len(df) == 1
        assert bool(df.iloc[0]["corroborated"]) is False
        assert df.iloc[0]["sources"] == ["bse"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_partial_failure_nse_down_bse_up(self, mock_warmup):
        # NSE 500s on every endpoint; BSE returns a row. We should get the
        # BSE row back as a single-source result, not an exception.
        respx.get(
            url__startswith=f"{NSE_BASE}/api/corporate-announcements"
        ).mock(return_value=httpx.Response(500))
        respx.get(
            url__startswith=f"{NSE_BASE}/api/corporate-board-meetings"
        ).mock(return_value=httpx.Response(500))
        respx.get(
            url__startswith=f"{NSE_BASE}/api/corporates-corporateActions"
        ).mock(return_value=httpx.Response(500))
        respx.get(
            url__startswith=f"{NSE_BASE}/api/corporates-financial-results"
        ).mock(return_value=httpx.Response(500))
        respx.get(url__startswith=BSE_API).mock(
            return_value=httpx.Response(200, json=_bse_payload([
                {"SCRIP_CD": 500325, "HEADLINE": "BSE only",
                 "NEWS_DT": "2024-04-26T14:00:00", "NEWSID": "B1"},
            ]))
        )
        df = await fetch_filings_cross_validated(
            "RELIANCE", "500325", "2024-04-01", "2024-04-30",
        )
        assert len(df) == 1
        assert df.iloc[0]["sources"] == ["bse"]
        assert bool(df.iloc[0]["corroborated"]) is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_partial_failure_bse_down_nse_up(self, mock_warmup):
        _mock_nse_endpoints(respx.mock, ann=[
            {"an_dt": "26-Apr-2024 10:00:00", "desc": "NSE only",
             "smIndustry": "Refineries"},
        ])
        respx.get(url__startswith=BSE_API).mock(
            return_value=httpx.Response(500)
        )
        df = await fetch_filings_cross_validated(
            "RELIANCE", "500325", "2024-04-01", "2024-04-30",
        )
        assert len(df) == 1
        assert df.iloc[0]["sources"] == ["nse"]
        assert bool(df.iloc[0]["corroborated"]) is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_total_failure_raises(self, mock_warmup):
        # NSE 500s, BSE 500s. CrossValidationError expected.
        for ep in (
            "/api/corporate-announcements", "/api/corporate-board-meetings",
            "/api/corporates-corporateActions", "/api/corporates-financial-results",
        ):
            respx.get(url__startswith=f"{NSE_BASE}{ep}").mock(
                return_value=httpx.Response(500)
            )
        respx.get(url__startswith=BSE_API).mock(
            return_value=httpx.Response(500)
        )
        with pytest.raises(CrossValidationError, match="Both NSE and BSE"):
            await fetch_filings_cross_validated(
                "RELIANCE", "500325", "2024-04-01", "2024-04-30",
            )

    @pytest.mark.asyncio
    @respx.mock
    async def test_both_empty_returns_empty_df_with_xval_columns(
        self, mock_warmup,
    ):
        _mock_nse_endpoints(respx.mock)
        respx.get(url__startswith=BSE_API).mock(
            return_value=httpx.Response(200, json=_bse_payload([]))
        )
        df = await fetch_filings_cross_validated(
            "RELIANCE", "500325", "2024-04-01", "2024-04-30",
        )
        assert df.empty
        # Schema sanity
        assert "corroborated" in df.columns
        assert "sources" in df.columns

    @pytest.mark.asyncio
    async def test_input_validation_runs_first(self):
        # Empty symbol must short-circuit before any network attempt.
        with pytest.raises(ValueError, match="symbol"):
            await fetch_filings_cross_validated(
                "", "500325", "2024-04-01", "2024-04-30",
            )

    @pytest.mark.asyncio
    @respx.mock
    async def test_kinds_passthrough_to_nse(self, mock_warmup):
        # Restrict NSE to announcement only; the other endpoints must NOT
        # be hit. respx tracks called routes — we can assert by setting
        # the others up to fail loudly.
        announcement_route = respx.get(
            url__startswith=f"{NSE_BASE}/api/corporate-announcements"
        ).mock(return_value=httpx.Response(200, json=[]))
        # Set up the other 3 to fail if hit. respx default is to raise.
        respx.get(url__startswith=BSE_API).mock(
            return_value=httpx.Response(200, json=_bse_payload([]))
        )
        await fetch_filings_cross_validated(
            "RELIANCE", "500325", "2024-04-01", "2024-04-30",
            kinds=["announcement"],
        )
        assert announcement_route.called

    @pytest.mark.asyncio
    @respx.mock
    async def test_dataframe_columns_shape(self, mock_warmup):
        """Cross-validated df has the expected column set, including the
        two promoted xval columns."""
        _mock_nse_endpoints(respx.mock)
        respx.get(url__startswith=BSE_API).mock(
            return_value=httpx.Response(200, json=_bse_payload([
                {"SCRIP_CD": 500325, "HEADLINE": "Test",
                 "NEWS_DT": "2024-04-26T14:00:00", "NEWSID": "B1"},
            ]))
        )
        df = await fetch_filings_cross_validated(
            "RELIANCE", "500325", "2024-04-01", "2024-04-30",
        )
        expected = {
            "symbol", "kind", "announced_at", "event_at", "event_type",
            "subject", "description", "attachment_url",
            "corroborated", "sources", "metadata",
        }
        assert set(df.columns) == expected


# ─────────────────────────────────────────────────────────────
# Defensive corner case: empty fingerprint shouldn't blow up merge
# ─────────────────────────────────────────────────────────────
def test_merge_handles_empty_subject_filings_safely():
    # Pydantic enforces subject min_length=1 so we can't construct one,
    # but the dedup_key path uses `.subject` defensively. Verify with a
    # valid (single-char) subject that produces a tiny fingerprint.
    nse = _make_filing(
        announced_at=datetime(2024, 4, 26, 10, 0, tzinfo=IST),
        subject="X",
    )
    out = _merge_with_corroboration([nse], [])
    assert len(out) == 1
    assert out[0].metadata["corroborated"] is False


# Keep imports alive
_ = pd
