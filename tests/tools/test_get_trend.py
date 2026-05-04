"""Tests for the get_trend ADK tool.

Uses a mock PriceCache (injected via set_cache) so no Yahoo calls happen.
"""
from __future__ import annotations

import asyncio
from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from price_predictor.agents.technical_agent.tools.get_trend import (
    _normalize_ticker,
    get_trend,
)
from price_predictor.data import _shared_cache
from price_predictor.data.prices import PriceFetchError


# ─────────────────────────────────────────────────────────────────
# Fixtures: synthetic OHLCV + a fake cache
# ─────────────────────────────────────────────────────────────────


def _build_uptrend_df(n: int = 400) -> pd.DataFrame:
    """Linearly rising series. Trend cluster should call this bullish."""
    closes = np.linspace(100, 200, n)
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 1,
            "low": closes - 1,
            "close": closes,
            "adj_close": closes,
            "volume": np.full(n, 1000),
        },
        index=dates,
    )


def _build_downtrend_df(n: int = 400) -> pd.DataFrame:
    closes = np.linspace(200, 100, n)
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 1,
            "low": closes - 1,
            "close": closes,
            "adj_close": closes,
            "volume": np.full(n, 1000),
        },
        index=dates,
    )


class _FakeCache:
    """Mimics PriceCache.get() but returns a pre-baked DataFrame."""

    def __init__(self, df_to_return: pd.DataFrame, *, raise_exc: Exception | None = None):
        self.df = df_to_return
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    async def get(self, ticker: str, start: date, end: date, interval: str = "1d"):
        self.calls.append(
            {"ticker": ticker, "start": start, "end": end, "interval": interval}
        )
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.df.copy()


@pytest.fixture(autouse=True)
def _reset_cache():
    """Ensure each test gets a clean cache slot."""
    _shared_cache.set_cache(None)
    yield
    _shared_cache.set_cache(None)


# ─────────────────────────────────────────────────────────────────
# _normalize_ticker
# ─────────────────────────────────────────────────────────────────
class TestNormalizeTicker:
    def test_already_has_ns_suffix(self):
        assert _normalize_ticker("RELIANCE.NS") == "RELIANCE.NS"
        assert _normalize_ticker("reliance.ns") == "RELIANCE.NS"

    def test_known_indian_stock_gets_ns_suffix(self):
        # RELIANCE is in our KB; should resolve to .NS
        assert _normalize_ticker("RELIANCE") == "RELIANCE.NS"

    def test_unknown_ticker_passes_through(self):
        # AAPL isn't in our KB; pass through bare
        assert _normalize_ticker("AAPL") == "AAPL"

    def test_whitespace_stripped(self):
        assert _normalize_ticker("  RELIANCE  ") == "RELIANCE.NS"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _normalize_ticker("")
        with pytest.raises(ValueError):
            _normalize_ticker("   ")


# ─────────────────────────────────────────────────────────────────
# get_trend happy paths
# ─────────────────────────────────────────────────────────────────
class TestGetTrendHappy:
    def test_uptrend_returns_bullish(self):
        cache = _FakeCache(_build_uptrend_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_trend("RELIANCE", sensitivity="standard"))
        assert result["status"] == "success"
        assert result["ticker"] == "RELIANCE.NS"
        assert result["signal"] == "bullish"
        assert result["strength"] in ("moderate", "strong")
        assert result["preset"] == "standard"
        assert "indicators" in result
        assert result["indicators"]["close"] is not None
        assert result["indicators"]["sma_20"] is not None
        assert result["indicators"]["sma_200"] is not None
        assert result["indicators"]["adx"] is not None
        assert "rationale" in result and len(result["rationale"]) > 0
        assert result["warnings"] == []

    def test_downtrend_returns_bearish(self):
        cache = _FakeCache(_build_downtrend_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_trend("RELIANCE", sensitivity="standard"))
        assert result["status"] == "success"
        assert result["signal"] == "bearish"

    def test_only_one_cache_call_per_invocation(self):
        cache = _FakeCache(_build_uptrend_df())
        _shared_cache.set_cache(cache)
        asyncio.run(get_trend("RELIANCE", sensitivity="standard"))
        # One tool call => exactly one cache fetch
        assert len(cache.calls) == 1

    def test_sensitive_preset_uses_shorter_smas(self):
        cache = _FakeCache(_build_uptrend_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_trend("RELIANCE", sensitivity="sensitive"))
        assert result["status"] == "success"
        # Sensitive preset uses SMA(10, 30, 100) -- different keys than standard
        assert "sma_10" in result["indicators"]
        assert "sma_30" in result["indicators"]
        assert "sma_100" in result["indicators"]
        assert "sma_20" not in result["indicators"]

    def test_smooth_preset_uses_longer_smas(self):
        cache = _FakeCache(_build_uptrend_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_trend("RELIANCE", sensitivity="smooth"))
        assert result["status"] == "success"
        assert "sma_30" in result["indicators"]
        assert "sma_70" in result["indicators"]
        assert "sma_200" in result["indicators"]

    def test_as_of_is_last_bar_date(self):
        cache = _FakeCache(_build_uptrend_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_trend("RELIANCE", sensitivity="standard"))
        # Last bar in our synthetic df is at index date (2024-01-01 + 399 days)
        assert result["as_of"] == "2025-02-03"


# ─────────────────────────────────────────────────────────────────
# get_trend error paths
# ─────────────────────────────────────────────────────────────────
class TestGetTrendErrors:
    def test_empty_ticker_returns_error(self):
        result = asyncio.run(get_trend("", sensitivity="standard"))
        assert result["status"] == "error"
        assert "non-empty" in result["error_message"].lower()

    def test_invalid_sensitivity_returns_error(self):
        result = asyncio.run(get_trend("RELIANCE", sensitivity="bananas"))
        assert result["status"] == "error"
        assert "sensitivity" in result["error_message"].lower()

    def test_fetch_error_returns_error(self):
        cache = _FakeCache(
            _build_uptrend_df(),
            raise_exc=PriceFetchError("yahoo went boom"),
        )
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_trend("RELIANCE", sensitivity="standard"))
        assert result["status"] == "error"
        assert "yahoo went boom" in result["error_message"]

    def test_empty_dataframe_returns_error(self):
        cache = _FakeCache(
            pd.DataFrame(columns=["open", "high", "low", "close", "adj_close", "volume"])
        )
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_trend("RELIANCE", sensitivity="standard"))
        assert result["status"] == "error"
        assert "no price data" in result["error_message"].lower()


# ─────────────────────────────────────────────────────────────────
# Insufficient history -> warnings, not crash
# ─────────────────────────────────────────────────────────────────
class TestGetTrendInsufficientHistory:
    def test_short_history_emits_warning(self):
        # Only 30 bars -- not enough for SMA-200
        short_df = _build_uptrend_df(n=30)
        cache = _FakeCache(short_df)
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_trend("RELIANCE", sensitivity="standard"))
        assert result["status"] == "success"
        assert "insufficient_history" in result["warnings"]
        # SMA-200 should be None
        assert result["indicators"]["sma_200"] is None
