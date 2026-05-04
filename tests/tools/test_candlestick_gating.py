"""Unit tests for the candlestick context-gating helper (pure function)."""
from __future__ import annotations

import pandas as pd

from price_predictor.agents.technical_agent.tools._candlestick_gating import (
    BEARISH_PATTERNS,
    BULLISH_PATTERNS,
    NEUTRAL_PATTERNS,
    gate_patterns,
)


def _df(highs: list[float], lows: list[float]) -> pd.DataFrame:
    """Build a tiny OHLCV df from highs/lows (open/close = midpoint)."""
    n = len(highs)
    closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "adj_close": closes,
            "volume": [1000] * n,
        },
        index=idx,
    )


# ─────────────────────────────────────────────────────────────────
# Pattern classification (sanity check the constants)
# ─────────────────────────────────────────────────────────────────
class TestPatternClassification:
    def test_bullish_set(self):
        assert BULLISH_PATTERNS == {"hammer", "bullish_engulfing", "morning_star"}

    def test_bearish_set(self):
        assert BEARISH_PATTERNS == {"shooting_star", "bearish_engulfing", "evening_star"}

    def test_neutral_set(self):
        assert NEUTRAL_PATTERNS == {"doji"}

    def test_no_overlap(self):
        assert not (BULLISH_PATTERNS & BEARISH_PATTERNS)
        assert not (BULLISH_PATTERNS & NEUTRAL_PATTERNS)


# ─────────────────────────────────────────────────────────────────
# Bullish patterns -- gated by proximity to swing_low
# ─────────────────────────────────────────────────────────────────
class TestBullishGating:
    def test_hammer_near_support_surfaces(self):
        # Bar low = 100, swing_low = 99, ATR = 2 -> distance 1 <= ATR 2 -> surface
        df = _df(highs=[105, 102], lows=[103, 100])
        patterns = [{"name": "hammer", "bar_date": "2024-01-02", "bar_index": -1}]
        result = gate_patterns(patterns, df, swing_high=110, swing_low=99, atr=2)
        assert len(result) == 1
        assert result[0]["context"] == "near_support"
        assert result[0]["level_price"] == 99.0

    def test_hammer_far_from_support_filtered_out(self):
        # Bar low = 150, swing_low = 99, ATR = 2 -> distance 51 >> ATR 2 -> drop
        df = _df(highs=[155, 152], lows=[153, 150])
        patterns = [{"name": "hammer", "bar_date": "2024-01-02", "bar_index": -1}]
        result = gate_patterns(patterns, df, swing_high=160, swing_low=99, atr=2)
        assert result == []

    def test_bullish_engulfing_gated(self):
        df = _df(highs=[105, 102], lows=[103, 100])
        patterns = [{"name": "bullish_engulfing", "bar_date": "2024-01-02", "bar_index": -1}]
        result = gate_patterns(patterns, df, swing_high=110, swing_low=99, atr=2)
        assert len(result) == 1
        assert result[0]["context"] == "near_support"

    def test_no_swing_low_drops_bullish_pattern(self):
        df = _df(highs=[105, 102], lows=[103, 100])
        patterns = [{"name": "hammer", "bar_date": "2024-01-02", "bar_index": -1}]
        result = gate_patterns(patterns, df, swing_high=110, swing_low=None, atr=2)
        assert result == []


# ─────────────────────────────────────────────────────────────────
# Bearish patterns -- gated by proximity to swing_high
# ─────────────────────────────────────────────────────────────────
class TestBearishGating:
    def test_shooting_star_near_resistance_surfaces(self):
        # Bar high = 110, swing_high = 111, ATR = 2 -> distance 1 <= 2 -> surface
        df = _df(highs=[108, 110], lows=[105, 107])
        patterns = [{"name": "shooting_star", "bar_date": "2024-01-02", "bar_index": -1}]
        result = gate_patterns(patterns, df, swing_high=111, swing_low=100, atr=2)
        assert len(result) == 1
        assert result[0]["context"] == "near_resistance"
        assert result[0]["level_price"] == 111.0

    def test_shooting_star_far_from_resistance_filtered(self):
        df = _df(highs=[60, 62], lows=[58, 60])
        patterns = [{"name": "shooting_star", "bar_date": "2024-01-02", "bar_index": -1}]
        result = gate_patterns(patterns, df, swing_high=111, swing_low=50, atr=2)
        assert result == []

    def test_bearish_engulfing_gated(self):
        df = _df(highs=[108, 110], lows=[105, 107])
        patterns = [{"name": "bearish_engulfing", "bar_date": "2024-01-02", "bar_index": -1}]
        result = gate_patterns(patterns, df, swing_high=111, swing_low=100, atr=2)
        assert len(result) == 1


# ─────────────────────────────────────────────────────────────────
# Neutral pattern (doji) -- near EITHER level
# ─────────────────────────────────────────────────────────────────
class TestDojiGating:
    def test_doji_near_support_surfaces(self):
        df = _df(highs=[105, 102], lows=[103, 100])
        patterns = [{"name": "doji", "bar_date": "2024-01-02", "bar_index": -1}]
        result = gate_patterns(patterns, df, swing_high=120, swing_low=99, atr=2)
        assert len(result) == 1
        assert result[0]["context"] == "near_support"

    def test_doji_near_resistance_surfaces(self):
        df = _df(highs=[108, 110], lows=[105, 107])
        patterns = [{"name": "doji", "bar_date": "2024-01-02", "bar_index": -1}]
        result = gate_patterns(patterns, df, swing_high=111, swing_low=80, atr=2)
        assert len(result) == 1
        assert result[0]["context"] == "near_resistance"

    def test_doji_in_middle_filtered_out(self):
        df = _df(highs=[105, 102], lows=[103, 100])
        patterns = [{"name": "doji", "bar_date": "2024-01-02", "bar_index": -1}]
        result = gate_patterns(patterns, df, swing_high=200, swing_low=50, atr=2)
        assert result == []


# ─────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_no_atr_returns_empty(self):
        df = _df(highs=[105, 102], lows=[103, 100])
        patterns = [{"name": "hammer", "bar_date": "2024-01-02", "bar_index": -1}]
        result = gate_patterns(patterns, df, swing_high=110, swing_low=99, atr=None)
        assert result == []

    def test_zero_atr_returns_empty(self):
        df = _df(highs=[105, 102], lows=[103, 100])
        patterns = [{"name": "hammer", "bar_date": "2024-01-02", "bar_index": -1}]
        result = gate_patterns(patterns, df, swing_high=110, swing_low=99, atr=0)
        assert result == []

    def test_no_swing_levels_returns_empty(self):
        df = _df(highs=[105, 102], lows=[103, 100])
        patterns = [{"name": "hammer", "bar_date": "2024-01-02", "bar_index": -1}]
        result = gate_patterns(patterns, df, swing_high=None, swing_low=None, atr=2)
        assert result == []

    def test_unknown_pattern_dropped(self):
        df = _df(highs=[105, 102], lows=[103, 100])
        patterns = [{"name": "mystery_pattern", "bar_date": "2024-01-02", "bar_index": -1}]
        result = gate_patterns(patterns, df, swing_high=110, swing_low=99, atr=2)
        assert result == []

    def test_empty_pattern_list(self):
        df = _df(highs=[105], lows=[103])
        result = gate_patterns([], df, swing_high=110, swing_low=99, atr=2)
        assert result == []

    def test_multiple_patterns_filtered_independently(self):
        # Hammer near support (surface), shooting_star NOT near resistance (drop)
        df = _df(highs=[105, 102], lows=[103, 100])
        patterns = [
            {"name": "hammer", "bar_date": "2024-01-02", "bar_index": -1},
            {"name": "shooting_star", "bar_date": "2024-01-02", "bar_index": -1},
        ]
        result = gate_patterns(patterns, df, swing_high=200, swing_low=99, atr=2)
        assert len(result) == 1
        assert result[0]["name"] == "hammer"
