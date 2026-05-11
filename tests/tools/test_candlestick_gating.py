"""Unit tests for the candlestick context-gating helper (pure function).

Direction is now sourced from the pattern dict itself (set by the detector
per TA-Lib's signed signal), so test inputs must carry a "direction" field.
The legacy static BULLISH_PATTERNS / BEARISH_PATTERNS / NEUTRAL_PATTERNS
sets are gone — the gate reads `direction` directly.
"""
from __future__ import annotations

import pandas as pd

from price_predictor.agents.technical_agent.tools._candlestick_gating import (
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


def _pat(name: str, direction: str, bar_index: int = -1) -> dict:
    """Build a detector-shaped pattern dict for tests."""
    return {
        "name": name,
        "bar_date": "2024-01-02",
        "bar_index": bar_index,
        "direction": direction,
        "confidence": 100,
    }


# ─────────────────────────────────────────────────────────────────
# Bullish patterns -- gated by proximity to swing_low
# ─────────────────────────────────────────────────────────────────
class TestBullishGating:
    def test_hammer_near_support_surfaces(self):
        # Bar low = 100, swing_low = 99, ATR = 2 -> distance 1 <= ATR 2 -> surface
        df = _df(highs=[105, 102], lows=[103, 100])
        patterns = [_pat("hammer", "bullish")]
        result = gate_patterns(patterns, df, swing_high=110, swing_low=99, atr=2)
        assert len(result) == 1
        assert result[0]["context"] == "near_support"
        assert result[0]["level_price"] == 99.0

    def test_hammer_far_from_support_filtered_out(self):
        # Bar low = 150, swing_low = 99, ATR = 2 -> distance 51 >> ATR 2 -> drop
        df = _df(highs=[155, 152], lows=[153, 150])
        patterns = [_pat("hammer", "bullish")]
        result = gate_patterns(patterns, df, swing_high=160, swing_low=99, atr=2)
        assert result == []

    def test_bullish_engulfing_gated(self):
        df = _df(highs=[105, 102], lows=[103, 100])
        patterns = [_pat("engulfing", "bullish")]
        result = gate_patterns(patterns, df, swing_high=110, swing_low=99, atr=2)
        assert len(result) == 1
        assert result[0]["context"] == "near_support"

    def test_no_swing_low_drops_bullish_pattern(self):
        df = _df(highs=[105, 102], lows=[103, 100])
        patterns = [_pat("hammer", "bullish")]
        result = gate_patterns(patterns, df, swing_high=110, swing_low=None, atr=2)
        assert result == []


# ─────────────────────────────────────────────────────────────────
# Bearish patterns -- gated by proximity to swing_high
# ─────────────────────────────────────────────────────────────────
class TestBearishGating:
    def test_shooting_star_near_resistance_surfaces(self):
        # Bar high = 110, swing_high = 111, ATR = 2 -> distance 1 <= 2 -> surface
        df = _df(highs=[108, 110], lows=[105, 107])
        patterns = [_pat("shooting_star", "bearish")]
        result = gate_patterns(patterns, df, swing_high=111, swing_low=100, atr=2)
        assert len(result) == 1
        assert result[0]["context"] == "near_resistance"
        assert result[0]["level_price"] == 111.0

    def test_shooting_star_far_from_resistance_filtered(self):
        df = _df(highs=[60, 62], lows=[58, 60])
        patterns = [_pat("shooting_star", "bearish")]
        result = gate_patterns(patterns, df, swing_high=111, swing_low=50, atr=2)
        assert result == []

    def test_bearish_engulfing_gated(self):
        df = _df(highs=[108, 110], lows=[105, 107])
        patterns = [_pat("engulfing", "bearish")]
        result = gate_patterns(patterns, df, swing_high=111, swing_low=100, atr=2)
        assert len(result) == 1


# ─────────────────────────────────────────────────────────────────
# Neutral pattern (doji family) -- near EITHER level
# ─────────────────────────────────────────────────────────────────
class TestNeutralGating:
    def test_doji_near_support_surfaces(self):
        df = _df(highs=[105, 102], lows=[103, 100])
        patterns = [_pat("doji", "neutral")]
        result = gate_patterns(patterns, df, swing_high=120, swing_low=99, atr=2)
        assert len(result) == 1
        assert result[0]["context"] == "near_support"

    def test_doji_near_resistance_surfaces(self):
        df = _df(highs=[108, 110], lows=[105, 107])
        patterns = [_pat("doji", "neutral")]
        result = gate_patterns(patterns, df, swing_high=111, swing_low=80, atr=2)
        assert len(result) == 1
        assert result[0]["context"] == "near_resistance"

    def test_doji_in_middle_filtered_out(self):
        df = _df(highs=[105, 102], lows=[103, 100])
        patterns = [_pat("doji", "neutral")]
        result = gate_patterns(patterns, df, swing_high=200, swing_low=50, atr=2)
        assert result == []

    def test_spinning_top_treated_as_neutral(self):
        # Verifies that NEW TA-Lib neutral patterns (not just "doji") flow through.
        df = _df(highs=[105, 102], lows=[103, 100])
        patterns = [_pat("spinning_top", "neutral")]
        result = gate_patterns(patterns, df, swing_high=120, swing_low=99, atr=2)
        assert len(result) == 1
        assert result[0]["context"] == "near_support"


# ─────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_no_atr_returns_empty(self):
        df = _df(highs=[105, 102], lows=[103, 100])
        patterns = [_pat("hammer", "bullish")]
        result = gate_patterns(patterns, df, swing_high=110, swing_low=99, atr=None)
        assert result == []

    def test_zero_atr_returns_empty(self):
        df = _df(highs=[105, 102], lows=[103, 100])
        patterns = [_pat("hammer", "bullish")]
        result = gate_patterns(patterns, df, swing_high=110, swing_low=99, atr=0)
        assert result == []

    def test_no_swing_levels_returns_empty(self):
        df = _df(highs=[105, 102], lows=[103, 100])
        patterns = [_pat("hammer", "bullish")]
        result = gate_patterns(patterns, df, swing_high=None, swing_low=None, atr=2)
        assert result == []

    def test_unknown_direction_dropped(self):
        # Defensive: a pattern with a bogus direction is silently dropped.
        df = _df(highs=[105, 102], lows=[103, 100])
        patterns = [{"name": "mystery", "bar_date": "x", "bar_index": -1,
                     "direction": "unknown", "confidence": 100}]
        result = gate_patterns(patterns, df, swing_high=110, swing_low=99, atr=2)
        assert result == []

    def test_missing_direction_dropped(self):
        # Pattern dicts without a `direction` key are dropped (defensive).
        df = _df(highs=[105, 102], lows=[103, 100])
        patterns = [{"name": "hammer", "bar_date": "x", "bar_index": -1}]
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
            _pat("hammer", "bullish"),
            _pat("shooting_star", "bearish"),
        ]
        result = gate_patterns(patterns, df, swing_high=200, swing_low=99, atr=2)
        assert len(result) == 1
        assert result[0]["name"] == "hammer"
