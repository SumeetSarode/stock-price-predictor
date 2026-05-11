"""Unit tests for data/bhavcopy.py — the per-day NSE bhavcopy fetcher.

We never touch real NSE: every HTTP call is mocked via respx and we
inject a respx-bound httpx.Client into the public function.
"""
from __future__ import annotations

from datetime import date, timedelta

import httpx
import pandas as pd
import pytest
import respx

from price_predictor.data.bhavcopy import (
    DAILY_REPORTS_URL,
    LEGACY_CUTOVER_DATE,
    LEGACY_URL_TEMPLATE,
    BhavcopyError,
    _coerce_numeric_and_finalize,
    _find_udiff_csv_url,
    _parse_legacy_csv,
    _parse_udiff_csv,
    fetch_nse_bhavcopy,
)


# ─────────────────────────────────────────────────────────────
# Test fixtures — sample CSV bodies
# ─────────────────────────────────────────────────────────────
def _legacy_csv(rows: list[dict] | None = None) -> str:
    """Build a legacy bhavcopy CSV body. Headers DELIBERATELY include the
    leading-space pattern NSE actually ships — verifying the parser strips."""
    rows = rows or [
        {
            "SYMBOL": "RELIANCE", "SERIES": " EQ", "DATE1": "25-APR-2024",
            "PREV_CLOSE": "2895.30", "OPEN_PRICE": "2900.00",
            "HIGH_PRICE": "2925.50", "LOW_PRICE": "2890.10",
            "LAST_PRICE": "2915.00", "CLOSE_PRICE": "2918.45",
            "AVG_PRICE": "2910.10", "TTL_TRD_QNTY": "5300100",
        },
        {
            "SYMBOL": "INFY", "SERIES": " EQ", "DATE1": "25-APR-2024",
            "PREV_CLOSE": "1410.0", "OPEN_PRICE": "1412.00",
            "HIGH_PRICE": "1420.50", "LOW_PRICE": "1405.10",
            "LAST_PRICE": "1418.00", "CLOSE_PRICE": "1417.20",
            "AVG_PRICE": "1413.10", "TTL_TRD_QNTY": "1200000",
        },
    ]
    headers = list(rows[0].keys())
    # Pad header names with a leading space (the real NSE quirk).
    padded_headers = [f" {h}" if h != "SYMBOL" else h for h in headers]
    lines = [",".join(padded_headers)]
    for r in rows:
        lines.append(",".join(str(r[h]) for h in headers))
    return "\n".join(lines) + "\n"


def _udiff_csv(rows: list[dict] | None = None) -> str:
    """Build a UDiff bhavcopy CSV body (cash + a futures row to verify
    the F&O-filter works)."""
    rows = rows or [
        {
            "TradDt": "2024-08-15", "BizDt": "2024-08-15",
            "Sgmt": "CM", "Src": "EOD", "FinInstrmTp": "EQ",
            "TckrSymb": "RELIANCE", "SctySrs": "EQ",
            "OpnPric": "3000.00", "HghPric": "3025.50",
            "LwPric": "2990.10", "ClsPric": "3018.45",
            "TtlTradgVol": "5300100",
        },
        {
            "TradDt": "2024-08-15", "BizDt": "2024-08-15",
            "Sgmt": "CM", "Src": "EOD", "FinInstrmTp": "EQ",
            "TckrSymb": "INFY", "SctySrs": "EQ",
            "OpnPric": "1500.00", "HghPric": "1520.00",
            "LwPric": "1495.00", "ClsPric": "1517.20",
            "TtlTradgVol": "1200000",
        },
        # F&O row — MUST be filtered out by the parser.
        {
            "TradDt": "2024-08-15", "BizDt": "2024-08-15",
            "Sgmt": "FO", "Src": "EOD", "FinInstrmTp": "STF",
            "TckrSymb": "RELIANCE", "SctySrs": "FU",
            "OpnPric": "3010.00", "HghPric": "3030.00",
            "LwPric": "2995.00", "ClsPric": "3020.00",
            "TtlTradgVol": "100000",
        },
    ]
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for r in rows:
        lines.append(",".join(str(r[h]) for h in headers))
    return "\n".join(lines) + "\n"


def _legacy_url(d: date) -> str:
    return LEGACY_URL_TEMPLATE.format(
        dd=f"{d.day:02d}", mm=f"{d.month:02d}", yyyy=d.year,
    )


# ─────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────
class TestInputValidation:
    def test_non_date_raises(self):
        with pytest.raises(ValueError, match="must be a date"):
            fetch_nse_bhavcopy("2024-04-25")  # type: ignore[arg-type]

    def test_future_date_raises(self):
        future = date.today() + timedelta(days=365)
        with pytest.raises(ValueError, match="future"):
            fetch_nse_bhavcopy(future)


# ─────────────────────────────────────────────────────────────
# Date routing — legacy vs UDiff
# ─────────────────────────────────────────────────────────────
class TestDateRouting:
    @respx.mock
    def test_date_before_cutover_uses_legacy_url(self):
        d = LEGACY_CUTOVER_DATE - timedelta(days=1)
        route = respx.get(_legacy_url(d)).mock(
            return_value=httpx.Response(200, text=_legacy_csv()),
        )
        df = fetch_nse_bhavcopy(d, client=httpx.Client())
        assert route.called
        assert len(df) == 2
        assert "RELIANCE" in df["SYMBOL"].values

    @respx.mock
    def test_date_on_cutover_uses_udiff(self):
        d = LEGACY_CUTOVER_DATE  # exact cutover
        # Daily-reports JSON returns the CSV link.
        csv_url = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_X.csv"
        respx.get(DAILY_REPORTS_URL).mock(
            return_value=httpx.Response(200, json=[
                {"name": "BhavCopy CM Foo", "link": csv_url},
            ]),
        )
        respx.get(csv_url).mock(
            return_value=httpx.Response(200, text=_udiff_csv()),
        )
        df = fetch_nse_bhavcopy(d, client=httpx.Client())
        # F&O row filtered out → 2 rows remain (RELIANCE + INFY).
        assert len(df) == 2

    @respx.mock
    def test_date_after_cutover_uses_udiff(self):
        d = LEGACY_CUTOVER_DATE + timedelta(days=180)
        csv_url = "https://example.invalid/BhavCopy_NSE_CM_0_0_0_Y.csv"
        respx.get(DAILY_REPORTS_URL).mock(
            return_value=httpx.Response(200, json=[
                {"name": "BhavCopy CM File", "link": csv_url},
            ]),
        )
        respx.get(csv_url).mock(
            return_value=httpx.Response(200, text=_udiff_csv()),
        )
        df = fetch_nse_bhavcopy(d, client=httpx.Client())
        assert len(df) == 2


# ─────────────────────────────────────────────────────────────
# Owns-client lifecycle
# ─────────────────────────────────────────────────────────────
class TestClientLifecycle:
    @respx.mock
    def test_provided_client_not_closed(self):
        """If caller passes a client we MUST NOT close it."""
        d = LEGACY_CUTOVER_DATE - timedelta(days=1)
        respx.get(_legacy_url(d)).mock(
            return_value=httpx.Response(200, text=_legacy_csv()),
        )
        client = httpx.Client()
        fetch_nse_bhavcopy(d, client=client)
        assert not client.is_closed
        client.close()

    @respx.mock
    def test_no_client_creates_and_closes_one(self):
        """When client=None we own it. Hard to assert on the internal
        instance directly; this test just verifies the call SUCCEEDS
        end-to-end without leaks (would warn at GC if leaked)."""
        d = LEGACY_CUTOVER_DATE - timedelta(days=1)
        respx.get(_legacy_url(d)).mock(
            return_value=httpx.Response(200, text=_legacy_csv()),
        )
        df = fetch_nse_bhavcopy(d)  # no client kwarg
        assert len(df) == 2


# ─────────────────────────────────────────────────────────────
# Legacy CSV parser
# ─────────────────────────────────────────────────────────────
class TestLegacyParser:
    def test_happy_path(self):
        df = _parse_legacy_csv(_legacy_csv(), trading_date=date(2024, 4, 25))
        assert list(df.columns) == [
            "SYMBOL", "SERIES", "DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME",
        ]
        assert df["SYMBOL"].tolist() == ["RELIANCE", "INFY"]
        # SERIES values are stripped of leading space.
        assert df["SERIES"].tolist() == ["EQ", "EQ"]
        assert df["OPEN"].dtype.kind == "f"
        assert df["VOLUME"].dtype.kind in ("i", "u")
        # DATE is the trading-date arg, tz-aware Asia/Kolkata.
        assert str(df["DATE"].dt.tz) == "Asia/Kolkata"
        assert df["DATE"].iloc[0].date() == date(2024, 4, 25)

    def test_empty_body_raises(self):
        with pytest.raises(BhavcopyError, match="empty body"):
            _parse_legacy_csv("", trading_date=date(2024, 4, 25))

    def test_whitespace_body_raises(self):
        with pytest.raises(BhavcopyError, match="empty body"):
            _parse_legacy_csv("   \n  \t  \n", trading_date=date(2024, 4, 25))

    def test_missing_columns_raises(self):
        bad = "SYMBOL,SERIES\nRELIANCE,EQ\n"
        with pytest.raises(BhavcopyError, match="missing columns"):
            _parse_legacy_csv(bad, trading_date=date(2024, 4, 25))

    def test_unparseable_prices_dropped(self):
        # A row with '-' in the price columns (NSE's "suspended scrip"
        # convention) should be dropped, not crash.
        rows = [
            {
                "SYMBOL": "RELIANCE", "SERIES": " EQ", "DATE1": "25-APR-2024",
                "PREV_CLOSE": "2895.30", "OPEN_PRICE": "2900.00",
                "HIGH_PRICE": "2925.50", "LOW_PRICE": "2890.10",
                "LAST_PRICE": "2915.00", "CLOSE_PRICE": "2918.45",
                "AVG_PRICE": "2910.10", "TTL_TRD_QNTY": "5300100",
            },
            {
                "SYMBOL": "DEADCO", "SERIES": " BE", "DATE1": "25-APR-2024",
                "PREV_CLOSE": "-", "OPEN_PRICE": "-",
                "HIGH_PRICE": "-", "LOW_PRICE": "-",
                "LAST_PRICE": "-", "CLOSE_PRICE": "-",
                "AVG_PRICE": "-", "TTL_TRD_QNTY": "0",
            },
        ]
        df = _parse_legacy_csv(_legacy_csv(rows), trading_date=date(2024, 4, 25))
        assert df["SYMBOL"].tolist() == ["RELIANCE"]


# ─────────────────────────────────────────────────────────────
# UDiff CSV parser
# ─────────────────────────────────────────────────────────────
class TestUdiffParser:
    def test_happy_path_filters_fno(self):
        df = _parse_udiff_csv(_udiff_csv(), trading_date=date(2024, 8, 15))
        assert list(df.columns) == [
            "SYMBOL", "SERIES", "DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME",
        ]
        # 3 input rows; F&O row dropped → 2 left.
        assert df["SYMBOL"].tolist() == ["RELIANCE", "INFY"]

    def test_empty_body_raises(self):
        with pytest.raises(BhavcopyError, match="empty body"):
            _parse_udiff_csv("", trading_date=date(2024, 8, 15))

    def test_missing_columns_raises(self):
        bad = "TckrSymb,SctySrs\nRELIANCE,EQ\n"
        with pytest.raises(BhavcopyError, match="missing columns"):
            _parse_udiff_csv(bad, trading_date=date(2024, 8, 15))

    def test_missing_optional_segment_columns_no_filter(self):
        # If Sgmt / FinInstrmTp absent, the parser must NOT crash and
        # also must NOT filter (defensive — old snapshots may lack them).
        rows = [
            {
                "TckrSymb": "RELIANCE", "SctySrs": "EQ",
                "OpnPric": "100.0", "HghPric": "110.0",
                "LwPric": "95.0", "ClsPric": "108.0",
                "TtlTradgVol": "1000",
            },
        ]
        text = ",".join(rows[0].keys()) + "\n" + ",".join(str(v) for v in rows[0].values())
        df = _parse_udiff_csv(text, trading_date=date(2024, 8, 15))
        assert df["SYMBOL"].tolist() == ["RELIANCE"]


# ─────────────────────────────────────────────────────────────
# UDiff link discovery
# ─────────────────────────────────────────────────────────────
class TestFindUdiffCsvUrl:
    def test_picks_first_matching_csv(self):
        listing = [
            {"name": "Some other report", "link": "https://x/foo.zip"},
            {"name": "BhavCopy NSE CM", "link": "https://x/bhavcopy.csv"},
            {"name": "BhavCopy NSE CM Backup", "link": "https://x/backup.csv"},
        ]
        url = _find_udiff_csv_url(listing, trading_date=date(2024, 8, 15))
        assert url == "https://x/bhavcopy.csv"

    def test_filePath_field_used_when_link_missing(self):
        listing = [{"name": "BhavCopy CM", "filePath": "https://x/v.csv"}]
        url = _find_udiff_csv_url(listing, trading_date=date(2024, 8, 15))
        assert url == "https://x/v.csv"

    def test_non_list_raises(self):
        with pytest.raises(BhavcopyError, match="expected a JSON list"):
            _find_udiff_csv_url({"items": []}, trading_date=date(2024, 8, 15))

    def test_no_match_raises(self):
        listing = [
            {"name": "Other Report", "link": "https://x/other.csv"},
            {"name": "BhavCopy CM", "link": "https://x/wrong.zip"},  # wrong ext
        ]
        with pytest.raises(BhavcopyError, match="no CSV entry"):
            _find_udiff_csv_url(listing, trading_date=date(2024, 8, 15))

    def test_skips_non_dict_entries(self):
        listing = [
            "garbage",
            {"name": "BhavCopy CM", "link": "https://x/ok.csv"},
        ]
        assert _find_udiff_csv_url(
            listing, trading_date=date(2024, 8, 15)
        ) == "https://x/ok.csv"


# ─────────────────────────────────────────────────────────────
# HTTP error paths
# ─────────────────────────────────────────────────────────────
class TestHttpErrors:
    @respx.mock
    def test_legacy_500_raises(self):
        d = date(2024, 4, 25)
        respx.get(_legacy_url(d)).mock(return_value=httpx.Response(500, text="boom"))
        with pytest.raises(BhavcopyError, match="HTTP 500"):
            fetch_nse_bhavcopy(d, client=httpx.Client())

    @respx.mock
    def test_legacy_404_raises(self):
        d = date(2024, 4, 25)
        respx.get(_legacy_url(d)).mock(return_value=httpx.Response(404))
        with pytest.raises(BhavcopyError, match="HTTP 404"):
            fetch_nse_bhavcopy(d, client=httpx.Client())

    @respx.mock
    def test_legacy_network_error_raises(self):
        d = date(2024, 4, 25)
        respx.get(_legacy_url(d)).mock(side_effect=httpx.ConnectError("dns"))
        with pytest.raises(BhavcopyError, match="GET .* failed"):
            fetch_nse_bhavcopy(d, client=httpx.Client())

    @respx.mock
    def test_udiff_daily_reports_500_raises(self):
        d = LEGACY_CUTOVER_DATE + timedelta(days=10)
        respx.get(DAILY_REPORTS_URL).mock(return_value=httpx.Response(500, text="x"))
        with pytest.raises(BhavcopyError, match="HTTP 500"):
            fetch_nse_bhavcopy(d, client=httpx.Client())

    @respx.mock
    def test_udiff_daily_reports_non_json_raises(self):
        d = LEGACY_CUTOVER_DATE + timedelta(days=10)
        respx.get(DAILY_REPORTS_URL).mock(
            return_value=httpx.Response(200, text="<html>not json</html>"),
        )
        with pytest.raises(BhavcopyError, match="not valid JSON"):
            fetch_nse_bhavcopy(d, client=httpx.Client())

    @respx.mock
    def test_udiff_csv_404_after_link_found(self):
        d = LEGACY_CUTOVER_DATE + timedelta(days=10)
        csv_url = "https://x/missing.csv"
        respx.get(DAILY_REPORTS_URL).mock(
            return_value=httpx.Response(200, json=[
                {"name": "BhavCopy CM", "link": csv_url},
            ]),
        )
        respx.get(csv_url).mock(return_value=httpx.Response(404))
        with pytest.raises(BhavcopyError, match="HTTP 404"):
            fetch_nse_bhavcopy(d, client=httpx.Client())

    @respx.mock
    def test_udiff_daily_reports_network_error_raises(self):
        d = LEGACY_CUTOVER_DATE + timedelta(days=10)
        respx.get(DAILY_REPORTS_URL).mock(side_effect=httpx.ConnectError("dns"))
        with pytest.raises(BhavcopyError, match="GET .* failed"):
            fetch_nse_bhavcopy(d, client=httpx.Client())


# ─────────────────────────────────────────────────────────────
# Numeric coercion finalizer
# ─────────────────────────────────────────────────────────────
class TestCoerceFinalize:
    def test_all_unparseable_raises(self):
        df = pd.DataFrame({
            "SYMBOL": ["X"],
            "SERIES": ["EQ"],
            "DATE": [pd.Timestamp("2024-04-25", tz="Asia/Kolkata")],
            "OPEN": ["-"], "HIGH": ["-"], "LOW": ["-"], "CLOSE": ["-"],
            "VOLUME": ["0"],
        })
        with pytest.raises(BhavcopyError, match="zero rows after cleaning"):
            _coerce_numeric_and_finalize(df, date(2024, 4, 25))

    def test_unparseable_volume_becomes_zero(self):
        df = pd.DataFrame({
            "SYMBOL": ["X"], "SERIES": ["EQ"],
            "DATE": [pd.Timestamp("2024-04-25", tz="Asia/Kolkata")],
            "OPEN": ["100"], "HIGH": ["110"], "LOW": ["95"], "CLOSE": ["108"],
            "VOLUME": ["NA"],
        })
        out = _coerce_numeric_and_finalize(df, date(2024, 4, 25))
        assert out["VOLUME"].iloc[0] == 0
        assert out["VOLUME"].dtype == "int64"
