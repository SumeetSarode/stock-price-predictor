"""Tests for analysis.volatility -- ATR, Bollinger Bands, two squeezes."""
from __future__ import annotations

from price_predictor.analysis.volatility import (
    bollinger_squeeze,
    latest_atr,
    latest_bbands,
    ttm_squeeze,
    volatility_snapshot,
)
from tests.analysis.conftest import (
    insufficient_history,
    linear_uptrend,
    sideways,
)


class TestLatestATR:
    def test_uptrend_atr_close_to_typical_range(self):
        df = linear_uptrend(n=200, start=100, slope=1)
        # Our synthetic high = close+1, low = close-1, range = 2
        atr = latest_atr(df, length=14)
        assert atr is not None
        assert 1.5 < atr < 2.5  # close to range of 2

    def test_short_history_none(self):
        # Wilder warmup is now 5*length = 70 bars; 5 is far short.
        assert latest_atr(insufficient_history(5), length=14) is None

    def test_three_length_no_longer_enough(self):
        # H7 fix: 3*length = 42 bars used to suffice. We now require 5*length.
        df = linear_uptrend(n=42, start=100, slope=1)
        assert latest_atr(df, length=14) is None

    def test_five_length_is_enough(self):
        df = linear_uptrend(n=70, start=100, slope=1)
        assert latest_atr(df, length=14) is not None


class TestLatestBBands:
    def test_sideways_price_between_bands(self):
        df = sideways(n=100)
        bb = latest_bbands(df, length=20, std=2)
        assert bb["lower"] is not None
        assert bb["middle"] is not None
        assert bb["upper"] is not None
        # Middle ~ mean of last 20 bars
        assert abs(bb["middle"] - 100) < 5
        # Upper > middle > lower
        assert bb["upper"] > bb["middle"] > bb["lower"]
        # %B should be sensible
        assert bb["percent_b"] is not None

    def test_short_history_none(self):
        bb = latest_bbands(insufficient_history(5))
        assert bb["lower"] is None
        assert bb["middle"] is None


class TestBollingerSqueeze:
    """Bollinger (2001) bandwidth-percentile squeeze."""

    def test_constant_price_should_squeeze(self):
        # Bandwidth collapses to ~0 on a flat series -- definitely a squeeze
        df = sideways(n=200, mean=100, amplitude=0.0)
        result = bollinger_squeeze(df, length=20, std=2, lookback=60)
        # On a perfectly flat series the function may return None or True
        # -- both are acceptable; the "false" case is the bug we want to avoid.
        assert result in (True, None)

    def test_short_history_none(self):
        assert bollinger_squeeze(insufficient_history(5)) is None


class TestTTMSqueeze:
    """Carter (2009) TTM Squeeze: BB inside Keltner."""

    def test_constant_price_squeeze_on(self):
        # Truly flat series: BB collapses tighter than Keltner's ATR-channel,
        # so squeeze must be on.
        df = sideways(n=200, mean=100, amplitude=0.0)
        result = ttm_squeeze(df)
        assert result["on"] is True
        assert result["bars_in_squeeze"] is not None
        assert result["bars_in_squeeze"] >= 1

    def test_violent_swings_squeeze_off(self):
        # Big oscillations -> BB explodes outside Keltner -> squeeze OFF.
        # `sideways` with large amplitude keeps mean stable but cranks BB width.
        df = sideways(n=200, mean=100, amplitude=15.0)
        result = ttm_squeeze(df)
        assert result["on"] is False
        assert result["bars_in_squeeze"] == 0

    def test_short_history_returns_none(self):
        result = ttm_squeeze(insufficient_history(5))
        assert result["on"] is None
        assert result["fire"] is None
        assert result["bars_in_squeeze"] is None


class TestVolatilitySnapshot:
    def test_snapshot_shape(self):
        df = linear_uptrend(n=200, start=100, slope=1)
        snap = volatility_snapshot(df, atr_length=14, bb_params=(20, 2.0))
        assert snap["atr"] is not None
        assert snap["atr_pct_of_price"] is not None
        assert 0 < snap["atr_pct_of_price"] < 5  # ~range/price = ~0.6%
        assert snap["bbands"]["middle"] is not None
        # Both squeezes are surfaced, not just one.
        assert "bollinger_squeeze" in snap
        assert "ttm_squeeze" in snap
        assert isinstance(snap["ttm_squeeze"], dict)
        assert {"on", "fire", "bars_in_squeeze"} <= set(snap["ttm_squeeze"].keys())

    def test_short_history_graceful(self):
        df = insufficient_history(n=10)
        snap = volatility_snapshot(df, atr_length=14, bb_params=(20, 2.0))
        assert snap["atr"] is None
        assert snap["bbands"]["middle"] is None
        assert snap["bollinger_squeeze"] is None
        assert snap["ttm_squeeze"]["on"] is None
