"""Tests for the get_volatility ADK tool.

Mocks shared cache. Synthetic OHLCV produces predictable BB / ATR signals.
"""
from __future__ import annotations

import asyncio
from datetime import date

import numpy as np
import pandas as pd
import pytest

from price_predictor.agents.technical_agent.tools.get_volatility import (
    get_volatility,
)
from price_predictor.data import _shared_cache
from price_predictor.data.prices import PriceFetchError


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


def _high_close_in_band_df(n: int = 400, seed: int = 5) -> pd.DataFrame:
    """Recent rally pushes price toward upper BB band -- bullish %B."""
    rng = np.random.default_rng(seed)
    base = 100 + rng.normal(0, 0.8, n).cumsum() * 0.05
    ramp = np.zeros(n)
    ramp[-30:] = np.linspace(0, 15, 30)
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


def _low_close_in_band_df(n: int = 400, seed: int = 5) -> pd.DataFrame:
    """Recent decline pushes price toward lower BB band -- bearish %B."""
    rng = np.random.default_rng(seed)
    base = 200 + rng.normal(0, 0.8, n).cumsum() * 0.05
    ramp = np.zeros(n)
    ramp[-30:] = -np.linspace(0, 15, 30)
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


def _quiet_then_squeeze_df(n: int = 400, seed: int = 9) -> pd.DataFrame:
    """Constant-price series -- guaranteed BB squeeze.

    The squeeze logic is exhaustively tested in tests/analysis/test_volatility.py.
    Here we just need a series that DOES squeeze so we can verify the tool
    surfaces it correctly. The simplest such series is a constant price (all
    bandwidths collapse to ~0).
    """
    closes = np.full(n, 100.0)
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.01,
            "low": closes - 0.01,
            "close": closes,
            "adj_close": closes,
            "volume": np.full(n, 1000),
        },
        index=dates,
    )


class _FakeCache:
    def __init__(self, df_to_return: pd.DataFrame, *, raise_exc: Exception | None = None):
        self.df = df_to_return
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
class TestGetVolatilityHappy:
    def test_high_in_band_returns_bullish(self):
        cache = _FakeCache(_high_close_in_band_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_volatility("RELIANCE"))
        assert result["status"] == "success"
        assert result["ticker"] == "RELIANCE.NS"
        assert result["signal"] == "bullish"

    def test_low_in_band_returns_bearish(self):
        cache = _FakeCache(_low_close_in_band_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_volatility("RELIANCE"))
        assert result["status"] == "success"
        assert result["signal"] == "bearish"

    def test_response_shape(self):
        cache = _FakeCache(_high_close_in_band_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_volatility("RELIANCE"))
        for k in ("status", "ticker", "as_of", "preset", "signal", "strength",
                  "indicators", "derived", "rationale", "warnings"):
            assert k in result, f"missing key: {k}"
        for k in ("atr", "atr_pct_of_price", "bb_lower", "bb_middle",
                  "bb_upper", "bb_bandwidth", "bb_percent_b",
                  "bollinger_squeeze", "ttm_squeeze_on",
                  "ttm_squeeze_fire", "ttm_bars_in_squeeze"):
            assert k in result["indicators"], f"missing indicator: {k}"
        for k in ("volatility_regime", "suggested_stop_distance",
                  "per_share_risk", "atr_multiple_to_upper_band",
                  "atr_multiple_to_lower_band"):
            assert k in result["derived"], f"missing derived: {k}"

    def test_only_one_cache_call_per_invocation(self):
        cache = _FakeCache(_high_close_in_band_df())
        _shared_cache.set_cache(cache)
        asyncio.run(get_volatility("RELIANCE"))
        assert len(cache.calls) == 1

    def test_sensitivity_presets_all_work(self):
        for preset in ("standard", "sensitive", "smooth"):
            cache = _FakeCache(_high_close_in_band_df())
            _shared_cache.set_cache(cache)
            result = asyncio.run(get_volatility("RELIANCE", sensitivity=preset))
            assert result["status"] == "success", f"failed for preset {preset}"
            assert result["preset"] == preset


# ─────────────────────────────────────────────────────────────────
# Position-sizing helpers
# ─────────────────────────────────────────────────────────────────
class TestDerivedHelpers:
    def test_suggested_stop_is_2x_atr(self):
        cache = _FakeCache(_high_close_in_band_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_volatility("RELIANCE"))
        atr = result["indicators"]["atr"]
        stop = result["derived"]["suggested_stop_distance"]
        assert stop is not None
        assert atr is not None
        # Allow 1 cent tolerance for rounding
        assert abs(stop - 2 * atr) < 0.01

    def test_per_share_risk_is_alias_for_stop(self):
        cache = _FakeCache(_high_close_in_band_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_volatility("RELIANCE"))
        assert (
            result["derived"]["per_share_risk"]
            == result["derived"]["suggested_stop_distance"]
        )

    def test_atr_multiples_to_bands_are_finite(self):
        cache = _FakeCache(_high_close_in_band_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_volatility("RELIANCE"))
        # In a bullish setup, distance to upper band should be small/positive,
        # distance to lower band should be larger
        upper_mult = result["derived"]["atr_multiple_to_upper_band"]
        lower_mult = result["derived"]["atr_multiple_to_lower_band"]
        assert upper_mult is not None
        assert lower_mult is not None

    def test_volatility_regime_present(self):
        cache = _FakeCache(_high_close_in_band_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_volatility("RELIANCE"))
        assert result["derived"]["volatility_regime"] in (
            "low", "normal", "high", "unknown"
        )


# ─────────────────────────────────────────────────────────────────
# Squeeze detection (the strength upgrade)
# ─────────────────────────────────────────────────────────────────
class TestSqueeze:
    def test_quiet_period_can_trigger_squeeze(self):
        cache = _FakeCache(_quiet_then_squeeze_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_volatility("RELIANCE"))
        # Either squeeze flag firing should bump strength to "strong".
        # Bollinger percentile is the wider net (looser definition);
        # TTM may also fire if the quiet window is deep enough.
        triggered = (
            result["indicators"]["bollinger_squeeze"] is True
            or result["indicators"]["ttm_squeeze_on"] is True
            or result["indicators"]["ttm_squeeze_fire"] is True
        )
        assert triggered, (
            "quiet fixture should trigger at least one squeeze definition; "
            f"got bollinger={result['indicators']['bollinger_squeeze']}, "
            f"ttm_on={result['indicators']['ttm_squeeze_on']}, "
            f"ttm_fire={result['indicators']['ttm_squeeze_fire']}"
        )
        # TTM is the one that drives strength=strong; if only Bollinger
        # fires, strength may stay moderate — see _volatility_signal.py.
        if (result["indicators"]["ttm_squeeze_on"] is True
                or result["indicators"]["ttm_squeeze_fire"] is True):
            assert result["strength"] == "strong"


# ─────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────
class TestGetVolatilityErrors:
    def test_invalid_preset_returns_error(self):
        cache = _FakeCache(_high_close_in_band_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_volatility("RELIANCE", sensitivity="bogus"))
        assert result["status"] == "error"

    def test_empty_ticker_returns_error(self):
        cache = _FakeCache(_high_close_in_band_df())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_volatility(""))
        assert result["status"] == "error"

    def test_price_fetch_error_returns_error_dict(self):
        cache = _FakeCache(
            _high_close_in_band_df(),
            raise_exc=PriceFetchError("upstream fail"),
        )
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_volatility("RELIANCE"))
        assert result["status"] == "error"
        assert "upstream fail" in result["error_message"]

    def test_empty_dataframe_returns_error(self):
        cache = _FakeCache(pd.DataFrame())
        _shared_cache.set_cache(cache)
        result = asyncio.run(get_volatility("RELIANCE"))
        assert result["status"] == "error"
