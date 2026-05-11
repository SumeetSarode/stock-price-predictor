"""Tests for price_predictor.data.prices.fetch_ohlcv.

Layout:
    - Unit tests   → mock yfinance.download, run fast, no network
    - Integration  → real yfinance call, marked @pytest.mark.integration,
                     skipped by default (uv run pytest -m "not integration")

The mock helper `_make_yf_response()` builds a DataFrame shaped like what
yfinance ACTUALLY returns post-0.2 (MultiIndex columns, Title Case names,
tz-naive DatetimeIndex for daily data). If yfinance changes their response
shape, update this helper — every unit test depends on it.
"""
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from price_predictor.data.prices import (
    PriceFetchError,
    fetch_ohlcv,
    reset_default_fetcher,
)


@pytest.fixture(autouse=True)
def _pin_chain_to_yfinance(monkeypatch):
    """Lock the resilient chain to YFINANCE-ONLY for every test in this file.

    Why: tests here mock `yf.download` to assert yfinance-specific behavior.
    With the new default chain (`jugaad,nse_bhavcopy,yfinance`) jugaad runs
    FIRST and — if the dev's local `.cache/` happens to contain real bars —
    serves them before yfinance is ever called, making the mock useless.

    Pinning the chain here keeps each test focused on the provider it claims
    to test, and makes the suite robust to future default-chain changes.
    """
    monkeypatch.setenv("PRICE_CHAIN", "yfinance")
    monkeypatch.delenv("USE_PAID_PRICES", raising=False)
    # Build a Settings that IGNORES the .env file (otherwise pydantic-settings
    # reads .env's PRICE_CHAIN and ignores our setenv — same trick used in
    # test_settings.py's _build_settings helper). Then patch the binding
    # `prices.py` actually reads from.
    from price_predictor.config.settings import Settings as _SettingsCls
    from price_predictor.data import prices as _prices_mod
    fresh = _SettingsCls(_env_file=None)  # type: ignore[call-arg]
    monkeypatch.setattr(_prices_mod, "settings", fresh)
    reset_default_fetcher()
    yield
    reset_default_fetcher()


# ─────────────────────────────────────────────────────────────
# Mock helper — mimics real yfinance.download() output shape
# ─────────────────────────────────────────────────────────────
def _make_yf_response(
    ticker: str = "RELIANCE.NS",
    rows: int = 5,
) -> pd.DataFrame:
    """Build a DataFrame shaped like yfinance.download() returns.

    Real yfinance returns:
        - Index: DatetimeIndex, tz-naive for daily data, tz-aware (UTC) for intraday
        - Columns: MultiIndex with two levels:
              level 0 = field ("Open", "High", "Low", "Close", "Adj Close", "Volume")
              level 1 = ticker (e.g. "RELIANCE.NS")
    """
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    columns = pd.MultiIndex.from_product([fields, [ticker]])

    # Deterministic fake data (won't match reality — fine for unit tests)
    data = {
        ("Open", ticker):      [2400.0, 2410.0, 2420.0, 2430.0, 2440.0][:rows],
        ("High", ticker):      [2450.0, 2460.0, 2470.0, 2480.0, 2490.0][:rows],
        ("Low", ticker):       [2380.0, 2390.0, 2400.0, 2410.0, 2420.0][:rows],
        ("Close", ticker):     [2420.0, 2430.0, 2440.0, 2450.0, 2460.0][:rows],
        ("Adj Close", ticker): [2400.0, 2410.0, 2420.0, 2430.0, 2440.0][:rows],
        ("Volume", ticker):    [1000000, 1100000, 1200000, 1300000, 1400000][:rows],
    }
    return pd.DataFrame(data, index=idx, columns=columns)


# ─────────────────────────────────────────────────────────────
# Unit tests (mocked yfinance — fast, no network)
# ─────────────────────────────────────────────────────────────
def test_fetch_ohlcv_happy_path():
    """Mocked fetch returns a clean, normalized DataFrame."""
    with patch("price_predictor.data.providers.yfinance_provider.yf.download") as mock_dl:
        mock_dl.return_value = _make_yf_response()
        result = fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 5))

    assert isinstance(result, pd.DataFrame), f"Expected DataFrame, got {type(result).__name__}"
    assert result.columns.tolist() == ["open", "high", "low", "close", "adj_close", "volume"], (
        f"Unexpected columns: {result.columns.tolist()}"
    )
    assert len(result) == 5, f"Expected 5 rows from mock, got {len(result)}"
    assert str(result.index.tz) == "Asia/Kolkata", (
        f"Expected Asia/Kolkata tz, got {result.index.tz}"
    )
    # Sanity: data flowed through correctly
    assert result["close"].iloc[0] == 2420.0, (
        f"Expected close=2420.0 in first row, got {result['close'].iloc[0]}"
    )


def test_fetch_ohlcv_empty_raises():
    """When yfinance returns an empty DataFrame, fetch_ohlcv raises PriceFetchError."""
    with patch("price_predictor.data.providers.yfinance_provider.yf.download") as mock_dl:
        mock_dl.return_value = pd.DataFrame()  # empty
        with pytest.raises(PriceFetchError) as exc_info:
            fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 5))

    assert "RELIANCE.NS" in str(exc_info.value), (
        f"Error should mention the ticker for debuggability, got: {exc_info.value}"
    )


def test_fetch_ohlcv_start_after_end_raises():
    """start > end raises ValueError BEFORE yfinance is even called."""
    with pytest.raises(ValueError) as exc_info:
        fetch_ohlcv("RELIANCE.NS", date(2024, 1, 5), date(2024, 1, 1))

    msg = str(exc_info.value)
    assert "2024-01-05" in msg and "2024-01-01" in msg, (
        f"Error should mention both dates, got: {msg}"
    )


@pytest.mark.parametrize("bad_ticker", ["", "   ", "\t"])
def test_fetch_ohlcv_empty_ticker_raises(bad_ticker: str):
    """Empty / whitespace ticker is rejected up-front."""
    with pytest.raises(ValueError) as exc_info:
        fetch_ohlcv(bad_ticker, date(2024, 1, 1), date(2024, 1, 5))

    assert "non-empty" in str(exc_info.value), (
        f"Error should explain the requirement, got: {exc_info.value}"
    )


def test_fetch_ohlcv_column_normalization():
    """Title-case + multi-level columns from yfinance get flattened to lowercase."""
    with patch("price_predictor.data.providers.yfinance_provider.yf.download") as mock_dl:
        mock_dl.return_value = _make_yf_response()
        result = fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 5))

    # MultiIndex was flattened
    assert not isinstance(result.columns, pd.MultiIndex), (
        "Columns should be flat, not MultiIndex"
    )
    # All names are lowercase + snake_case
    for col in result.columns:
        assert col == col.lower(), f"Column {col!r} is not lowercase"
        assert " " not in col, f"Column {col!r} contains space (should be snake_case)"
    # adj_close specifically (catches the "Adj Close" → "adj_close" rename)
    assert "adj_close" in result.columns, "adj_close column missing — rename failed?"
    # Order matches documented contract
    assert result.columns.tolist() == ["open", "high", "low", "close", "adj_close", "volume"]


# ────────────────────────────────────────────────────────
# YFinance auto-suffix behaviour (.NS for bare symbols)
#
# These tests exist because of a real bug surfaced by the smoke test:
# the new default chain passes BARE symbols ('RELIANCE') to yfinance,
# which used to treat them as US tickers and return empty data.
# YFinanceProvider now appends `.NS` (configurable) to bare symbols.
# ────────────────────────────────────────────────────────
class TestYFinanceAutoSuffix:
    def test_bare_symbol_gets_ns_suffix(self):
        """'RELIANCE' → yfinance gets called with 'RELIANCE.NS'."""
        with patch("price_predictor.data.providers.yfinance_provider.yf.download") as mock_dl:
            mock_dl.return_value = _make_yf_response(ticker="RELIANCE.NS")
            fetch_ohlcv("RELIANCE", date(2024, 1, 1), date(2024, 1, 5))
        # Inspect the kwargs yfinance was called with
        assert mock_dl.call_args.kwargs["tickers"] == "RELIANCE.NS", (
            f"Expected bare 'RELIANCE' to be auto-suffixed to 'RELIANCE.NS', "
            f"got tickers={mock_dl.call_args.kwargs['tickers']!r}"
        )

    def test_already_ns_suffixed_passes_through(self):
        """'RELIANCE.NS' → yfinance gets called with 'RELIANCE.NS' (no double-suffix)."""
        with patch("price_predictor.data.providers.yfinance_provider.yf.download") as mock_dl:
            mock_dl.return_value = _make_yf_response(ticker="RELIANCE.NS")
            fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 5))
        assert mock_dl.call_args.kwargs["tickers"] == "RELIANCE.NS"

    def test_other_exchange_suffix_passes_through(self):
        """'AAPL.US' / 'BARC.L' / 'TYO.T' must NOT get '.NS' appended."""
        for input_ticker in ["AAPL.US", "BARC.L", "7203.T"]:
            with patch("price_predictor.data.providers.yfinance_provider.yf.download") as mock_dl:
                mock_dl.return_value = _make_yf_response(ticker=input_ticker)
                fetch_ohlcv(input_ticker, date(2024, 1, 1), date(2024, 1, 5))
            assert mock_dl.call_args.kwargs["tickers"] == input_ticker, (
                f"Existing exchange suffix on {input_ticker!r} should be preserved"
            )

    def test_dotted_us_ticker_passes_through(self):
        """'BRK.B' (Berkshire B-shares on NYSE) is a real dotted US ticker.
        The auto-suffix regex matches 1-5 letter suffix, so '.B' counts as
        an existing suffix and the ticker passes through unchanged — which
        is correct yfinance behaviour for BRK.B."""
        with patch("price_predictor.data.providers.yfinance_provider.yf.download") as mock_dl:
            mock_dl.return_value = _make_yf_response(ticker="BRK.B")
            fetch_ohlcv("BRK.B", date(2024, 1, 1), date(2024, 1, 5))
        assert mock_dl.call_args.kwargs["tickers"] == "BRK.B"

    def test_default_market_none_disables_suffixing(self):
        """Caller can opt out by constructing YFinanceProvider(default_market=None).
        Bare 'AAPL' then stays bare — useful if you're using yfinance for
        US-only and don't want any market munging."""
        from price_predictor.data.providers.yfinance_provider import YFinanceProvider
        prov = YFinanceProvider(default_market=None)
        with patch("price_predictor.data.providers.yfinance_provider.yf.download") as mock_dl:
            mock_dl.return_value = _make_yf_response(ticker="AAPL")
            prov.fetch_ohlcv("AAPL", date(2024, 1, 1), date(2024, 1, 5))
        assert mock_dl.call_args.kwargs["tickers"] == "AAPL"

    def test_default_market_with_leading_dot_normalised(self):
        """Constructor accepts both 'NS' and '.NS' — should be equivalent."""
        from price_predictor.data.providers.yfinance_provider import YFinanceProvider
        prov = YFinanceProvider(default_market=".NS")
        with patch("price_predictor.data.providers.yfinance_provider.yf.download") as mock_dl:
            mock_dl.return_value = _make_yf_response(ticker="RELIANCE.NS")
            prov.fetch_ohlcv("RELIANCE", date(2024, 1, 1), date(2024, 1, 5))
        assert mock_dl.call_args.kwargs["tickers"] == "RELIANCE.NS"

    def test_custom_default_market_lse(self):
        """Configurable: a UK-focused caller can default to '.L'."""
        from price_predictor.data.providers.yfinance_provider import YFinanceProvider
        prov = YFinanceProvider(default_market="L")
        with patch("price_predictor.data.providers.yfinance_provider.yf.download") as mock_dl:
            mock_dl.return_value = _make_yf_response(ticker="BARC.L")
            prov.fetch_ohlcv("BARC", date(2024, 1, 1), date(2024, 1, 5))
        assert mock_dl.call_args.kwargs["tickers"] == "BARC.L"

    def test_whitespace_stripped_before_suffix_check(self):
        """'  RELIANCE  ' should suffix to 'RELIANCE.NS', not '  RELIANCE  .NS'."""
        with patch("price_predictor.data.providers.yfinance_provider.yf.download") as mock_dl:
            mock_dl.return_value = _make_yf_response(ticker="RELIANCE.NS")
            fetch_ohlcv("  RELIANCE  ", date(2024, 1, 1), date(2024, 1, 5))
        assert mock_dl.call_args.kwargs["tickers"] == "RELIANCE.NS"

    def test_error_message_includes_resolved_and_input(self):
        """Empty-result error mentions BOTH the resolved ticker (what we
        actually queried) AND the input (what the caller passed) — makes
        debugging suffix issues trivial."""
        with patch("price_predictor.data.providers.yfinance_provider.yf.download") as mock_dl:
            mock_dl.return_value = pd.DataFrame()
            with pytest.raises(PriceFetchError) as exc_info:
                fetch_ohlcv("RELIANCE", date(2024, 1, 1), date(2024, 1, 5))
        msg = str(exc_info.value)
        assert "RELIANCE.NS" in msg, f"resolved ticker should appear: {msg}"
        assert "RELIANCE" in msg, f"input ticker should appear: {msg}"


# ─────────────────────────────────────────────────────────────
# Integration test — REAL yfinance call (working example)
# ─────────────────────────────────────────────────────────────
@pytest.mark.integration
def test_fetch_ohlcv_real_reliance():
    """Real fetch of RELIANCE.NS over a fixed historical month.

    Run with:   uv run pytest -m integration tests/test_prices.py
    Skipped by: uv run pytest -m "not integration"

    Uses a fixed date range so the test is deterministic — same data every run.
    """
    df = fetch_ohlcv(
        ticker="RELIANCE.NS",
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
    )

    assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df).__name__}"
    assert not df.empty, "Real fetch returned empty — yfinance broken or VPN issue?"
    assert df.columns.tolist() == ["open", "high", "low", "close", "adj_close", "volume"], (
        f"Unexpected columns: {df.columns.tolist()}"
    )
    assert str(df.index.tz) == "Asia/Kolkata", f"Expected Asia/Kolkata tz, got {df.index.tz}"
    # Jan 2024 had ~21 trading days. Allow some slack.
    assert 18 <= len(df) <= 25, f"Expected ~21 trading days, got {len(df)}"
