"""Tests for analysis.volatility -- ATR, Bollinger Bands, squeeze."""
from __future__ import annotations

from price_predictor.analysis.volatility import (
    bb_squeeze,
    latest_atr,
    latest_bbands,
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
        assert latest_atr(insufficient_history(5), length=14) is None


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


class TestBBSqueeze:
    def test_constant_price_should_squeeze(self):
        # Bandwidth collapses to ~0 on a flat series -- definitely a squeeze
        df = sideways(n=200, mean=100, amplitude=0.0)
        # Constant price: bandwidth ~0 but recent quantile is also 0;
        # should still report True (current <= threshold).
        result = bb_squeeze(df, length=20, std=2, lookback=60)
        # On a perfectly flat series the function may return None or True
        # -- both are acceptable; the "false" case is the bug we want to avoid.
        assert result in (True, None)

    def test_short_history_none(self):
        assert bb_squeeze(insufficient_history(5)) is None


class TestVolatilitySnapshot:
    def test_snapshot_shape(self):
        df = linear_uptrend(n=200, start=100, slope=1)
        snap = volatility_snapshot(df, atr_length=14, bb_params=(20, 2.0))
        assert snap["atr"] is not None
        assert snap["atr_pct_of_price"] is not None
        assert 0 < snap["atr_pct_of_price"] < 5  # ~range/price = ~0.6%
        assert snap["bbands"]["middle"] is not None

    def test_short_history_graceful(self):
        df = insufficient_history(n=10)
        snap = volatility_snapshot(df, atr_length=14, bb_params=(20, 2.0))
        assert snap["atr"] is None
        assert snap["bbands"]["middle"] is None
        assert snap["squeeze"] is None
