"""Tests for the get_levels ADK tool.

Mocks shared cache. Synthetic OHLCV designed to exercise:
  - Breakout/breakdown detection
  - Near-support / near-resistance
  - Chart pattern integration
"""
from __future__ import annotations

import asyncio
from datetime import date

import numpy as np
import pandas as pd
import pytest

from price_predictor.agents.technical_agent.tools.get_levels import (
    get_levels,
)
from price_predictor.data import _shared_cache
from price_predictor.data.prices import PriceFetchError


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


def _ranged_then_breakout_df(n: int = 400) -> pd.DataFrame:
    """350 bars range-bound, then 50 bars rocketing to a new high."""
    rng = np.random.default_rng(42)
    ranged = 100 + rng.normal(0, 1, 350).cumsum() * 0.1
    breakout_tail = ranged[-1] + np.linspace(0, 50, 50)
    closes = np.concatenate([ranged, breakout_tail])
    return _wrap(closes, n)


def _ranged_then_breakdown_df(n: int = 400) -> pd.DataFrame:
    """350 bars range-bound, then 50 bars crashing to a new low."""
    rng = np.random.default_rng(42)
    ranged = 100 + rng.normal(0, 1, 350).cumsum() * 0.1
    crash_tail = ranged[-1] - np.linspace(0, 50, 50)
    closes = np.concatenate([ranged, crash_tail])
    return _wrap(closes, n)


def _stable_range_df(n: int = 400) -> pd.DataFrame:
    """No breakout/breakdown -- just a noisy random walk."""
    rng = np.random.default_rng(42)
    closes = 100 + rng.normal(0, 1, n).cumsum() * 0.1
    return _wrap(closes, n)


def _wrap(closes: np.ndarray, n: int) -> pd.DataFrame:
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
    def __init__(self, df, *, raise_exc: Exception | None = None):
        self.df = df
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    async def get(self, ticker: str, start: date, end: date, interval: str = "1d"):
        self.calls.append({"ticker": ticker, "start": start, "end": end, "interval": interval})
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.df.copy()


@pytest.fixture(autouse=True)
def _reset_cache():
    _shared_cache.set_cache(None)
    yield
    _shared_cache.set_cache(None)


# ─────────────────────────────────────────────────────────────────
# Happy paths
# ─────────────────────────────────────────────────────────────────
class TestGetLevelsHappy:
    def test_breakout_setup_detected(self):
        cache = _FakeCache(_ranged_then_breakout_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_levels("RELIANCE"))
        assert result["status"] == "success"
        assert result["signal"] == "bullish"
        assert result["derived"]["breakout_state"] == "breakout"

    def test_breakdown_setup_detected(self):
        cache = _FakeCache(_ranged_then_breakdown_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_levels("RELIANCE"))
        assert result["status"] == "success"
        assert result["signal"] == "bearish"
        assert result["derived"]["breakout_state"] == "breakdown"

    def test_response_shape(self):
        cache = _FakeCache(_ranged_then_breakout_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_levels("RELIANCE"))
        for k in ("status", "ticker", "as_of", "preset", "signal", "strength",
                  "indicators", "derived", "rationale", "warnings"):
            assert k in result, f"missing key: {k}"
        # Indicator keys
        for k in ("close", "swing_high", "swing_low", "high_52w", "low_52w",
                  "pp", "r1", "r2", "s1", "s2",
                  "distance_pct_swing_high", "distance_pct_swing_low",
                  "distance_pct_52w_high", "distance_pct_52w_low"):
            assert k in result["indicators"], f"missing indicator: {k}"
        # Derived keys
        for k in ("breakout_state", "near_level", "atr",
                  "chart_patterns", "pattern_count"):
            assert k in result["derived"], f"missing derived: {k}"

    def test_only_one_cache_call_per_invocation(self):
        cache = _FakeCache(_ranged_then_breakout_df())
        _shared_cache.set_cache(cache)
        asyncio.run(get_levels("RELIANCE"))
        assert len(cache.calls) == 1

    def test_sensitivity_presets_all_work(self):
        for preset in ("standard", "sensitive", "smooth"):
            cache = _FakeCache(_ranged_then_breakout_df())
            _shared_cache.set_cache(cache)
            result = asyncio.run(get_levels("RELIANCE", sensitivity=preset))
            assert result["status"] == "success", f"failed for preset {preset}"
            assert result["preset"] == preset

    def test_chart_patterns_list_is_returned(self):
        cache = _FakeCache(_ranged_then_breakout_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_levels("RELIANCE"))
        # Chart patterns may or may not detect on synthetic data; just verify shape
        assert isinstance(result["derived"]["chart_patterns"], list)
        assert isinstance(result["derived"]["pattern_count"], int)
        assert result["derived"]["pattern_count"] == len(result["derived"]["chart_patterns"])


# ─────────────────────────────────────────────────────────────────
# Range-bound (no breakout)
# ─────────────────────────────────────────────────────────────────
class TestStableRange:
    def test_stable_range_is_not_breakout(self):
        cache = _FakeCache(_stable_range_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_levels("RELIANCE"))
        assert result["status"] == "success"
        # Should NOT be a breakout/breakdown -- price oscillates within range
        assert result["derived"]["breakout_state"] == "none"


# ─────────────────────────────────────────────────────────────────
# 52-week strength upgrade
# ─────────────────────────────────────────────────────────────────
class TestStrength:
    def test_strong_breakout_above_52w_high(self):
        # Breakout setup goes from ~100 to ~150, easily blowing past
        # both swing-high AND 52w high
        cache = _FakeCache(_ranged_then_breakout_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_levels("RELIANCE"))
        # Should be strong since we're at/above 52w high
        assert result["strength"] == "strong"


# ─────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────
class TestGetLevelsErrors:
    def test_invalid_preset_returns_error(self):
        cache = _FakeCache(_ranged_then_breakout_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_levels("RELIANCE", sensitivity="bogus"))
        assert result["status"] == "error"

    def test_empty_ticker_returns_error(self):
        cache = _FakeCache(_ranged_then_breakout_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_levels(""))
        assert result["status"] == "error"

    def test_price_fetch_error_returns_error_dict(self):
        cache = _FakeCache(
            _ranged_then_breakout_df(),
            raise_exc=PriceFetchError("upstream fail"),
        )
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_levels("RELIANCE"))
        assert result["status"] == "error"
        assert "upstream fail" in result["error_message"]

    def test_empty_dataframe_returns_error(self):
        cache = _FakeCache(pd.DataFrame())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_levels("RELIANCE"))
        assert result["status"] == "error"
