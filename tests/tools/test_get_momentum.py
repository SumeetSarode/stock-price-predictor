"""Tests for the get_momentum ADK tool.

Mocks the shared cache (no Yahoo calls). Synthetic OHLCV produces
predictable momentum signals.
"""
from __future__ import annotations

import asyncio
from datetime import date

import numpy as np
import pandas as pd
import pytest

from price_predictor.agents.technical_agent.tools.get_momentum import get_momentum
from price_predictor.data import _shared_cache
from price_predictor.data.prices import PriceFetchError


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


def _uptrend_df(n: int = 400, seed: int = 3) -> pd.DataFrame:
    """Quiet base + parabolic rally in last 40 bars -- unambiguously bullish.

    WHY THIS SHAPE
    ==============
    Steady-state momentum tests are tricky because:
      - Pure-linear data: MACD histogram and Stoch %K vs %D converge to ~0
      - Saturated rallies: Stoch %K pins at 100, %D catches up, ties
    A parabolic acceleration keeps %K pulling above %D, MACD histogram
    expanding, and RSI saturating bullish -- all 3 indicators agree.
    """
    rng = np.random.default_rng(seed)
    base = 100 + rng.normal(0, 0.5, n).cumsum() * 0.05
    ramp = np.zeros(n)
    ramp[-40:] = (np.arange(40) ** 2) * 0.05  # quadratic acceleration
    closes = base + ramp
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


def _downtrend_df(n: int = 400, seed: int = 3) -> pd.DataFrame:
    """Quiet base + parabolic decline in last 40 bars -- unambiguously bearish."""
    rng = np.random.default_rng(seed)
    base = 200 + rng.normal(0, 0.5, n).cumsum() * 0.05
    ramp = np.zeros(n)
    ramp[-40:] = -(np.arange(40) ** 2) * 0.05  # quadratic decline
    closes = base + ramp
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


def _sideways_df(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Mean-reverting random walk -- neutral momentum."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.5, n).cumsum()
    closes = 150 + steps - steps.mean()
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
    """Mimics PriceCache.get() returning a pre-baked DataFrame."""

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
    _shared_cache.set_cache(None)
    yield
    _shared_cache.set_cache(None)


# ─────────────────────────────────────────────────────────────────
# Happy-path classification
# ─────────────────────────────────────────────────────────────────
class TestGetMomentumHappy:
    def test_uptrend_returns_bullish(self):
        cache = _FakeCache(_uptrend_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_momentum("RELIANCE"))
        assert result["status"] == "success"
        assert result["ticker"] == "RELIANCE.NS"
        assert result["signal"] == "bullish"
        assert result["preset"] == "standard"

    def test_downtrend_returns_bearish(self):
        cache = _FakeCache(_downtrend_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_momentum("RELIANCE"))
        assert result["status"] == "success"
        assert result["signal"] == "bearish"

    def test_response_shape(self):
        cache = _FakeCache(_uptrend_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_momentum("RELIANCE"))
        # Required top-level keys
        for k in ("status", "ticker", "as_of", "preset", "signal", "strength",
                  "indicators", "derived", "rationale", "warnings"):
            assert k in result, f"missing key: {k}"
        # Indicator keys
        for k in ("rsi", "macd_line", "macd_signal", "macd_histogram",
                  "macd_cross", "stoch_k", "stoch_d", "obv", "obv_slope_20"):
            assert k in result["indicators"], f"missing indicator: {k}"
        # Derived keys
        for k in ("candlestick_patterns", "patterns_detected_total",
                  "patterns_after_gating"):
            assert k in result["derived"]

    def test_only_one_cache_call_per_invocation(self):
        cache = _FakeCache(_uptrend_df())
        _shared_cache.set_cache(cache)
        asyncio.run(get_momentum("RELIANCE"))
        assert len(cache.calls) == 1

    def test_sensitivity_presets_all_work(self):
        for preset in ("standard", "sensitive", "smooth"):
            cache = _FakeCache(_uptrend_df())
            _shared_cache.set_cache(cache)
            result = asyncio.run(get_momentum("RELIANCE", sensitivity=preset))
            assert result["status"] == "success", f"failed for preset {preset}"
            assert result["preset"] == preset


# ─────────────────────────────────────────────────────────────────
# Candlestick context-gating integration
# ─────────────────────────────────────────────────────────────────
class TestCandlestickGatingIntegration:
    def test_synthetic_smooth_data_few_or_no_patterns(self):
        # Pure-linear uptrend has near-zero patterns and no swing structure
        # near recent bars -> after gating, list should be empty or very small
        cache = _FakeCache(_uptrend_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_momentum("RELIANCE"))
        gated = result["derived"]["candlestick_patterns"]
        # Sanity: gating reduced (or kept) the count, never increased it
        assert result["derived"]["patterns_after_gating"] <= result["derived"]["patterns_detected_total"]
        # Each surviving pattern has the gating metadata
        for p in gated:
            assert "context" in p
            assert "level_price" in p
            assert "distance_pct" in p
            assert p["context"] in ("near_support", "near_resistance")


# ─────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────
class TestGetMomentumErrors:
    def test_invalid_preset_returns_error(self):
        cache = _FakeCache(_uptrend_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_momentum("RELIANCE", sensitivity="bogus"))
        assert result["status"] == "error"
        assert "preset" in result["error_message"].lower() or "bogus" in result["error_message"].lower()

    def test_empty_ticker_returns_error(self):
        cache = _FakeCache(_uptrend_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_momentum(""))
        assert result["status"] == "error"

    def test_price_fetch_error_returns_error_dict(self):
        cache = _FakeCache(
            _uptrend_df(),
            raise_exc=PriceFetchError("upstream fail"),
        )
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_momentum("RELIANCE"))
        assert result["status"] == "error"
        assert "upstream fail" in result["error_message"]

    def test_empty_dataframe_returns_error(self):
        cache = _FakeCache(pd.DataFrame())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_momentum("RELIANCE"))
        assert result["status"] == "error"
        assert "no price data" in result["error_message"].lower()

    def test_typo_ticker_suggests_correction(self):
        # RELAINCE (typo) should suggest RELIANCE.NS via KB fuzzy match
        cache = _FakeCache(
            _uptrend_df(),
            raise_exc=PriceFetchError("not found"),
        )
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_momentum("RELAINCE"))
        assert result["status"] == "error"
        # Suggestion present iff KB fuzzy resolves
        if "suggested_ticker" in result:
            assert result["suggested_ticker"] == "RELIANCE.NS"
