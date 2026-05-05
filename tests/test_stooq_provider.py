"""Tests for StooqProvider — ticker translation, parsing, error handling.

Strategy:
    - Mock httpx.get at the boundary so we never hit the real Stooq.
    - Test ticker translation as a pure function (no HTTP).
    - Test successful parse, empty-response, malformed CSV, HTTP errors.
    - Verify the contract: tz-aware index, lowercase snake_case columns,
      both close and adj_close present.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from price_predictor.data.providers.base import PriceFetchError
from price_predictor.data.providers.stooq_provider import StooqProvider


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _mock_response(text: str, status: int = 200) -> MagicMock:
    """Build a fake httpx.Response with the given body and status."""
    resp = MagicMock(spec=httpx.Response)
    resp.text = text
    resp.status_code = status
    if status >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status}", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


_VALID_CSV = (
    "Date,Open,High,Low,Close,Volume\n"
    "2024-01-01,100.0,105.0,99.0,103.0,1000000\n"
    "2024-01-02,103.5,108.0,102.0,107.0,1100000\n"
    "2024-01-03,107.0,110.0,106.0,109.0,950000\n"
)


# ─────────────────────────────────────────────────────────────
# Ticker translation -- pure function, no HTTP needed
# ─────────────────────────────────────────────────────────────
class TestTickerTranslation:
    """Each provider owns ticker-format translation. Caller stays
    yfinance-canonical."""

    @pytest.mark.parametrize(
        "yf_ticker,expected_stooq",
        [
            ("RELIANCE.NS", "reliance.in"),
            ("TCS.NS", "tcs.in"),
            ("HDFCBANK.NS", "hdfcbank.in"),
            ("RELIANCE.BO", "reliance.in"),  # BSE also maps to .in
            ("AAPL", "aapl.us"),
            ("MSFT", "msft.us"),
        ],
    )
    def test_translates_known_suffixes(self, yf_ticker: str, expected_stooq: str):
        assert StooqProvider._to_stooq_ticker(yf_ticker) == expected_stooq

    def test_strips_whitespace_and_lowercases(self):
        assert StooqProvider._to_stooq_ticker("  RELIANCE.NS  ") == "reliance.in"


# ─────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────
class TestFetchHappyPath:
    def test_returns_dataframe_with_expected_columns(self):
        p = StooqProvider(api_key="real_key")
        with patch("price_predictor.data.providers.stooq_provider.httpx.get") as m:
            m.return_value = _mock_response(_VALID_CSV)
            df = p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 3))

        assert list(df.columns) == [
            "open", "high", "low", "close", "adj_close", "volume",
        ]

    def test_index_is_tz_aware_kolkata(self):
        """Contract: every provider returns Asia/Kolkata-localized index."""
        p = StooqProvider(api_key="real_key")
        with patch("price_predictor.data.providers.stooq_provider.httpx.get") as m:
            m.return_value = _mock_response(_VALID_CSV)
            df = p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 3))

        assert df.index.tz is not None
        assert str(df.index.tz) == "Asia/Kolkata"

    def test_values_match_csv_input(self):
        p = StooqProvider(api_key="real_key")
        with patch("price_predictor.data.providers.stooq_provider.httpx.get") as m:
            m.return_value = _mock_response(_VALID_CSV)
            df = p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 3))

        assert len(df) == 3
        assert df.iloc[0]["close"] == 103.0
        assert df.iloc[-1]["close"] == 109.0
        assert df.iloc[0]["volume"] == 1000000

    def test_adj_close_falls_back_to_close_when_missing(self):
        """Stooq's CSV doesn't include Adj Close. Provider mirrors close
        into adj_close to satisfy the DataFrame contract."""
        p = StooqProvider(api_key="real_key")
        with patch("price_predictor.data.providers.stooq_provider.httpx.get") as m:
            m.return_value = _mock_response(_VALID_CSV)
            df = p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 3))

        # adj_close MUST equal close when source has no separate adj column
        assert (df["adj_close"] == df["close"]).all()

    def test_passes_correct_query_params(self):
        """Verify we hit the right URL with the right ticker translation
        AND include the apikey (Stooq's 2024 requirement)."""
        p = StooqProvider(api_key="real_key_xyz")
        with patch("price_predictor.data.providers.stooq_provider.httpx.get") as m:
            m.return_value = _mock_response(_VALID_CSV)
            p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 31))

        m.assert_called_once()
        call_kwargs = m.call_args.kwargs
        params = call_kwargs["params"]
        assert params["s"] == "reliance.in"
        assert params["i"] == "d"
        assert params["d1"] == "20240101"
        assert params["d2"] == "20240131"
        assert params["apikey"] == "real_key_xyz"


# ──────────────────────────────────────────────────
# Key handling -- lazy validation (mirrors AlphaVantage)
# ──────────────────────────────────────────────────
class TestKeyHandling:
    """Stooq added an apikey requirement in 2024. Lazy validation: empty
    is OK at construction (factory-friendly), fails clearly at fetch time."""

    def test_empty_key_at_construction_does_not_raise(self):
        StooqProvider(api_key="")  # MUST NOT raise

    def test_empty_key_at_fetch_raises_price_fetch_error(self):
        """Empty key at fetch -> PriceFetchError so resilient layer falls back.
        The error message must include both STOOQ_API_KEY and the captcha URL
        so the user can self-serve."""
        p = StooqProvider(api_key="")
        with pytest.raises(PriceFetchError, match="STOOQ_API_KEY"):
            p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 5))

    def test_empty_key_error_includes_captcha_url(self):
        """User-friendly: tell them exactly where to get a key."""
        p = StooqProvider(api_key="")
        with pytest.raises(PriceFetchError, match="get_apikey"):
            p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 5))

    def test_whitespace_key_treated_as_empty(self):
        p = StooqProvider(api_key="   ")
        with pytest.raises(PriceFetchError, match="STOOQ_API_KEY"):
            p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 5))


# ─────────────────────────────────────────────────────────────
# Error paths
# ─────────────────────────────────────────────────────────────
class TestErrorPaths:
    def test_empty_ticker_raises_value_error_no_http(self):
        """ValueError is caller-side -- must NOT trigger an HTTP call."""
        p = StooqProvider(api_key="real_key")
        with patch("price_predictor.data.providers.stooq_provider.httpx.get") as m:
            with pytest.raises(ValueError, match="non-empty"):
                p.fetch_ohlcv("", date(2024, 1, 1), date(2024, 1, 3))
            m.assert_not_called()

    def test_inverted_date_range_raises_value_error_no_http(self):
        p = StooqProvider(api_key="real_key")
        with patch("price_predictor.data.providers.stooq_provider.httpx.get") as m:
            with pytest.raises(ValueError, match="must be <="):
                p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 31), date(2024, 1, 1))
            m.assert_not_called()

    def test_unsupported_interval_raises_price_fetch_error(self):
        """Stooq is daily-only. PriceFetchError (not ValueError) so the
        resilient layer can fall back to a provider that handles weekly."""
        p = StooqProvider(api_key="real_key")
        with pytest.raises(PriceFetchError, match="only supports interval='1d'"):
            p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 3),
                          interval="1wk")

    def test_no_data_response_raises_price_fetch_error(self):
        """Stooq's quirky 'No data' string is HTTP 200 but means failure."""
        p = StooqProvider(api_key="real_key")
        with patch("price_predictor.data.providers.stooq_provider.httpx.get") as m:
            m.return_value = _mock_response("No data\n")
            with pytest.raises(PriceFetchError, match="no data"):
                p.fetch_ohlcv("BOGUS.NS", date(2024, 1, 1), date(2024, 1, 3))

    def test_empty_body_raises_price_fetch_error(self):
        p = StooqProvider(api_key="real_key")
        with patch("price_predictor.data.providers.stooq_provider.httpx.get") as m:
            m.return_value = _mock_response("")
            with pytest.raises(PriceFetchError, match="no data"):
                p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 3))

    def test_http_error_wrapped_as_price_fetch_error(self):
        """httpx.HTTPError must NOT bubble up raw -- the resilient layer
        only knows PriceFetchError / ValueError."""
        p = StooqProvider(api_key="real_key")
        with patch("price_predictor.data.providers.stooq_provider.httpx.get") as m:
            m.side_effect = httpx.ConnectError("connection refused")
            with pytest.raises(PriceFetchError, match="HTTP failure"):
                p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 3))

    def test_500_status_wrapped_as_price_fetch_error(self):
        p = StooqProvider(api_key="real_key")
        with patch("price_predictor.data.providers.stooq_provider.httpx.get") as m:
            m.return_value = _mock_response("server error", status=500)
            with pytest.raises(PriceFetchError, match="HTTP failure"):
                p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 3))

    def test_malformed_csv_wrapped_as_price_fetch_error(self):
        """A response that isn't valid CSV (e.g., HTML error page) must
        become a clean PriceFetchError, not a pandas parse exception."""
        p = StooqProvider(api_key="real_key")
        with patch("price_predictor.data.providers.stooq_provider.httpx.get") as m:
            # Stooq sometimes returns HTML when overloaded -- just non-parsable garbage
            m.return_value = _mock_response("\x00\x01\x02\xff\xfe garbage \x00")
            # Either malformed CSV or empty CSV -- both are PriceFetchError
            with pytest.raises(PriceFetchError):
                p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 3))


# ─────────────────────────────────────────────────────────────
# Provider name (logging contract)
# ─────────────────────────────────────────────────────────────
def test_provider_name_is_stooq():
    """Resilient layer keys cooldowns by name -- must be stable."""
    assert StooqProvider(api_key="real_key").name == "stooq"
