"""Tests for AlphaVantageProvider — JSON parsing, error envelopes, key handling.

Strategy:
    - Mock httpx.get at the boundary; never hit real AV (would burn quota).
    - Cover AV's three quirky error envelopes (Note / Information / Error Message).
    - Verify lazy-key behavior: empty key allowed at construction, fails clearly
      at fetch time with PriceFetchError (so the resilient layer falls back).
    - Verify date-range filtering, ticker translation, contract conformance.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from price_predictor.data.providers.alpha_vantage_provider import AlphaVantageProvider
from price_predictor.data.providers.base import PriceFetchError


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _mock_json_response(payload: dict, status: int = 200) -> MagicMock:
    """Build a fake httpx.Response with a JSON body."""
    resp = MagicMock(spec=httpx.Response)
    resp.json.return_value = payload
    resp.status_code = status
    if status >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status}", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _success_payload(start_date: str = "2024-01-01", n_days: int = 5) -> dict:
    """Build a realistic AV TIME_SERIES_DAILY response."""
    start = pd.Timestamp(start_date)
    series = {}
    for i in range(n_days):
        d = (start + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        # AV ships descending in real life; we ship in any order — provider sorts
        series[d] = {
            "1. open": str(100.0 + i),
            "2. high": str(105.0 + i),
            "3. low": str(99.0 + i),
            "4. close": str(103.0 + i),
            "5. volume": str(1_000_000 + i * 1000),
        }
    return {
        "Meta Data": {"1. Information": "Daily Prices", "2. Symbol": "RELIANCE.BSE"},
        "Time Series (Daily)": series,
    }


# ─────────────────────────────────────────────────────────────
# Ticker translation -- pure function
# ─────────────────────────────────────────────────────────────
class TestTickerTranslation:
    @pytest.mark.parametrize(
        "yf_ticker,expected_av",
        [
            ("RELIANCE.NS", "RELIANCE.BSE"),
            ("TCS.NS", "TCS.BSE"),
            ("RELIANCE.BO", "RELIANCE.BSE"),  # already BSE, normalized
            ("AAPL", "AAPL"),                  # US passthrough
            ("MSFT", "MSFT"),
        ],
    )
    def test_translates_known_suffixes(self, yf_ticker, expected_av):
        assert AlphaVantageProvider._to_av_ticker(yf_ticker) == expected_av

    def test_uppercases_and_strips(self):
        assert AlphaVantageProvider._to_av_ticker("  reliance.ns  ") == "RELIANCE.BSE"


# ─────────────────────────────────────────────────────────────
# Key handling -- the "lazy validation" design contract
# ─────────────────────────────────────────────────────────────
class TestKeyHandling:
    def test_empty_key_at_construction_does_not_raise(self):
        """Eager construction with lazy validation: factory builds every
        registered provider, even ones the user won't actually use."""
        AlphaVantageProvider(api_key="")  # MUST NOT raise

    def test_empty_key_at_fetch_raises_price_fetch_error(self):
        """Empty key at fetch -> PriceFetchError so resilient layer falls back."""
        p = AlphaVantageProvider(api_key="")
        with pytest.raises(PriceFetchError, match="ALPHA_VANTAGE_API_KEY"):
            p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 31))

    def test_whitespace_key_treated_as_empty(self):
        """A user who types '   ' in .env shouldn't get a confusing 401 from AV."""
        p = AlphaVantageProvider(api_key="   ")
        with pytest.raises(PriceFetchError, match="ALPHA_VANTAGE_API_KEY"):
            p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 31))


# ─────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────
class TestFetchHappyPath:
    def test_returns_dataframe_with_contract_columns(self):
        p = AlphaVantageProvider(api_key="real_key")
        with patch("price_predictor.data.providers.alpha_vantage_provider.httpx.get") as m:
            m.return_value = _mock_json_response(_success_payload(n_days=5))
            df = p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 5))

        assert list(df.columns) == [
            "open", "high", "low", "close", "adj_close", "volume",
        ]

    def test_index_is_tz_aware_kolkata(self):
        p = AlphaVantageProvider(api_key="real_key")
        with patch("price_predictor.data.providers.alpha_vantage_provider.httpx.get") as m:
            m.return_value = _mock_json_response(_success_payload(n_days=5))
            df = p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 5))

        assert df.index.tz is not None
        assert str(df.index.tz) == "Asia/Kolkata"

    def test_index_sorted_ascending(self):
        """AV ships descending; provider must sort to match contract."""
        p = AlphaVantageProvider(api_key="real_key")
        with patch("price_predictor.data.providers.alpha_vantage_provider.httpx.get") as m:
            m.return_value = _mock_json_response(_success_payload(n_days=5))
            df = p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 5))

        assert df.index.is_monotonic_increasing

    def test_date_range_filter_applied(self):
        """If AV returns 100 days but caller wants 5, we trim before returning."""
        p = AlphaVantageProvider(api_key="real_key")
        with patch("price_predictor.data.providers.alpha_vantage_provider.httpx.get") as m:
            m.return_value = _mock_json_response(_success_payload(n_days=10))
            df = p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 3), date(2024, 1, 6))

        # Inclusive bounds: Jan 3, 4, 5, 6 = 4 rows
        assert len(df) == 4
        assert df.index[0].date() == date(2024, 1, 3)
        assert df.index[-1].date() == date(2024, 1, 6)

    def test_string_values_cast_to_numeric(self):
        """AV returns everything as strings; provider must cast cleanly."""
        p = AlphaVantageProvider(api_key="real_key")
        with patch("price_predictor.data.providers.alpha_vantage_provider.httpx.get") as m:
            m.return_value = _mock_json_response(_success_payload(n_days=3))
            df = p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 3))

        assert df["close"].dtype == float
        assert df["volume"].dtype == int

    def test_adj_close_mirrors_close(self):
        """AV free tier omits adj_close; provider mirrors close to satisfy contract."""
        p = AlphaVantageProvider(api_key="real_key")
        with patch("price_predictor.data.providers.alpha_vantage_provider.httpx.get") as m:
            m.return_value = _mock_json_response(_success_payload(n_days=3))
            df = p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 3))

        assert (df["adj_close"] == df["close"]).all()

    def test_passes_correct_query_params(self):
        p = AlphaVantageProvider(api_key="real_key_xyz")
        with patch("price_predictor.data.providers.alpha_vantage_provider.httpx.get") as m:
            m.return_value = _mock_json_response(_success_payload(n_days=3))
            p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 3))

        m.assert_called_once()
        params = m.call_args.kwargs["params"]
        assert params["function"] == "TIME_SERIES_DAILY"
        assert params["symbol"] == "RELIANCE.BSE"
        assert params["apikey"] == "real_key_xyz"
        assert params["datatype"] == "json"

    def test_compact_outputsize_for_short_range(self):
        """Range <= 100 days -> compact (faster, 100 days payload)."""
        p = AlphaVantageProvider(api_key="real_key")
        with patch("price_predictor.data.providers.alpha_vantage_provider.httpx.get") as m:
            m.return_value = _mock_json_response(_success_payload(n_days=5))
            p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 30))

        assert m.call_args.kwargs["params"]["outputsize"] == "compact"

    def test_full_outputsize_for_long_range(self):
        """Range > 100 days -> full (20yr payload, slower but only path that
        covers it)."""
        p = AlphaVantageProvider(api_key="real_key")
        with patch("price_predictor.data.providers.alpha_vantage_provider.httpx.get") as m:
            m.return_value = _mock_json_response(_success_payload(n_days=5))
            p.fetch_ohlcv("RELIANCE.NS", date(2023, 1, 1), date(2024, 1, 1))

        assert m.call_args.kwargs["params"]["outputsize"] == "full"


# ─────────────────────────────────────────────────────────────
# AV's quirky error envelopes (HTTP 200 but with error fields)
# ─────────────────────────────────────────────────────────────
class TestAVErrorEnvelopes:
    def test_note_field_signals_rate_limit(self):
        """'Note' field = free-tier rate limit hit. Must be PriceFetchError so
        the resilient layer cools us down for 60s and stops hammering."""
        p = AlphaVantageProvider(api_key="real_key")
        with patch("price_predictor.data.providers.alpha_vantage_provider.httpx.get") as m:
            m.return_value = _mock_json_response({
                "Note": "Thank you for using Alpha Vantage! Our standard API "
                        "rate limit is 25 requests per day."
            })
            with pytest.raises(PriceFetchError, match="rate limit"):
                p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 5))

    def test_information_field_signals_premium_required(self):
        p = AlphaVantageProvider(api_key="real_key")
        with patch("price_predictor.data.providers.alpha_vantage_provider.httpx.get") as m:
            m.return_value = _mock_json_response({
                "Information": "This is a premium endpoint. Please subscribe."
            })
            with pytest.raises(PriceFetchError, match="premium"):
                p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 5))

    def test_error_message_field_signals_bad_ticker(self):
        p = AlphaVantageProvider(api_key="real_key")
        with patch("price_predictor.data.providers.alpha_vantage_provider.httpx.get") as m:
            m.return_value = _mock_json_response({
                "Error Message": "Invalid API call. Please retry or visit..."
            })
            with pytest.raises(PriceFetchError, match="rejected ticker"):
                p.fetch_ohlcv("BOGUS.NS", date(2024, 1, 1), date(2024, 1, 5))

    def test_missing_time_series_block_raises(self):
        """Unexpected response shape (neither error envelope nor data) = fetch error."""
        p = AlphaVantageProvider(api_key="real_key")
        with patch("price_predictor.data.providers.alpha_vantage_provider.httpx.get") as m:
            m.return_value = _mock_json_response({"Meta Data": {"foo": "bar"}})
            with pytest.raises(PriceFetchError, match="no time-series block"):
                p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 5))


# ─────────────────────────────────────────────────────────────
# Standard error paths (network, validation)
# ─────────────────────────────────────────────────────────────
class TestStandardErrors:
    def test_empty_ticker_raises_value_error_no_http(self):
        p = AlphaVantageProvider(api_key="real_key")
        with patch("price_predictor.data.providers.alpha_vantage_provider.httpx.get") as m:
            with pytest.raises(ValueError, match="non-empty"):
                p.fetch_ohlcv("", date(2024, 1, 1), date(2024, 1, 5))
            m.assert_not_called()

    def test_inverted_date_range_raises_value_error_no_http(self):
        p = AlphaVantageProvider(api_key="real_key")
        with patch("price_predictor.data.providers.alpha_vantage_provider.httpx.get") as m:
            with pytest.raises(ValueError, match="must be <="):
                p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 31), date(2024, 1, 1))
            m.assert_not_called()

    def test_unsupported_interval_raises_price_fetch_error(self):
        """v1 daily-only. Resilient layer can fall back to yfinance for weekly."""
        p = AlphaVantageProvider(api_key="real_key")
        with pytest.raises(PriceFetchError, match="only supports interval='1d'"):
            p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 5),
                          interval="1wk")

    def test_http_error_wrapped_as_price_fetch_error(self):
        p = AlphaVantageProvider(api_key="real_key")
        with patch("price_predictor.data.providers.alpha_vantage_provider.httpx.get") as m:
            m.side_effect = httpx.ConnectError("connection refused")
            with pytest.raises(PriceFetchError, match="HTTP failure"):
                p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 5))

    def test_non_json_body_wrapped_as_price_fetch_error(self):
        """AV occasionally returns HTML when overloaded -- must not crash."""
        p = AlphaVantageProvider(api_key="real_key")
        with patch("price_predictor.data.providers.alpha_vantage_provider.httpx.get") as m:
            resp = MagicMock(spec=httpx.Response)
            resp.raise_for_status.return_value = None
            resp.json.side_effect = ValueError("Expecting value")
            m.return_value = resp
            with pytest.raises(PriceFetchError, match="non-JSON"):
                p.fetch_ohlcv("RELIANCE.NS", date(2024, 1, 1), date(2024, 1, 5))

    def test_date_range_outside_returned_data_raises(self):
        """AV returned data, but none in our requested window -> PriceFetchError."""
        p = AlphaVantageProvider(api_key="real_key")
        with patch("price_predictor.data.providers.alpha_vantage_provider.httpx.get") as m:
            # Returned data is for January, caller asked for March
            m.return_value = _mock_json_response(_success_payload("2024-01-01", n_days=5))
            with pytest.raises(PriceFetchError, match="none in requested range"):
                p.fetch_ohlcv("RELIANCE.NS", date(2024, 3, 1), date(2024, 3, 5))


# ─────────────────────────────────────────────────────────────
# Provider name (logging + cooldown contract)
# ─────────────────────────────────────────────────────────────
def test_provider_name_is_alpha_vantage():
    assert AlphaVantageProvider(api_key="x").name == "alpha_vantage"
