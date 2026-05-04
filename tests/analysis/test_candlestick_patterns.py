"""Tests for analysis.candlestick_patterns -- hand-crafted bars per pattern."""
from __future__ import annotations

import pandas as pd

from price_predictor.analysis.candlestick_patterns import (
    detect_recent_patterns,
    is_bearish_engulfing,
    is_bullish_engulfing,
    is_doji,
    is_evening_star,
    is_hammer,
    is_morning_star,
    is_shooting_star,
)


def _bar(o: float, h: float, l: float, c: float) -> pd.Series:
    """Build a Series shaped like one OHLC row."""
    return pd.Series({"open": o, "high": h, "low": l, "close": c})


# ─────────────────────────────────────────────────────────────────
# Single-bar
# ─────────────────────────────────────────────────────────────────
class TestDoji:
    def test_classic_doji(self):
        # Body tiny, range moderate
        assert is_doji(_bar(100, 102, 98, 100.05))

    def test_long_body_not_doji(self):
        assert not is_doji(_bar(100, 102, 98, 101.5))


class TestHammer:
    def test_classic_hammer(self):
        # Open=100, close=101, low=95, high=101.2 -> long lower shadow, small body
        assert is_hammer(_bar(100, 101.2, 95, 101))

    def test_no_lower_shadow_not_hammer(self):
        assert not is_hammer(_bar(100, 102, 99.9, 101.5))

    def test_long_upper_shadow_disqualifies(self):
        # Long upper shadow = shooting star, NOT hammer
        assert not is_hammer(_bar(100, 105, 99.9, 100.5))


class TestShootingStar:
    def test_classic_shooting_star(self):
        # Open=100, close=99, high=105, low=98.8 -> long upper shadow, small body
        assert is_shooting_star(_bar(100, 105, 98.8, 99))

    def test_long_lower_shadow_not_shooting_star(self):
        assert not is_shooting_star(_bar(100, 100.2, 95, 99.5))


# ─────────────────────────────────────────────────────────────────
# Two-bar
# ─────────────────────────────────────────────────────────────────
class TestBullishEngulfing:
    def test_classic_bullish_engulfing(self):
        prev = _bar(105, 106, 100, 101)   # bearish, body 101..105
        curr = _bar(100, 108, 99, 107)    # bullish, body 100..107 (engulfs)
        assert is_bullish_engulfing(prev, curr)

    def test_inside_bar_not_engulfing(self):
        prev = _bar(105, 106, 100, 101)
        curr = _bar(102, 104, 101, 103)
        assert not is_bullish_engulfing(prev, curr)


class TestBearishEngulfing:
    def test_classic_bearish_engulfing(self):
        prev = _bar(100, 106, 99, 105)    # bullish, body 100..105
        curr = _bar(106, 107, 98, 99)     # bearish, body 99..106 (engulfs)
        assert is_bearish_engulfing(prev, curr)


# ─────────────────────────────────────────────────────────────────
# Three-bar
# ─────────────────────────────────────────────────────────────────
class TestMorningStar:
    def test_classic_morning_star(self):
        b1 = _bar(110, 111, 100, 101)   # long bearish body 101..110
        b2 = _bar(100, 102, 99, 100.5)  # small body (indecision)
        b3 = _bar(101, 110, 100, 109)   # long bullish body 101..109; closes above midpoint of b1 (105.5)
        assert is_morning_star(b1, b2, b3)


class TestEveningStar:
    def test_classic_evening_star(self):
        b1 = _bar(100, 110, 99, 109)    # long bullish body 100..109
        b2 = _bar(110, 112, 109, 110.5) # small body
        b3 = _bar(109, 110, 100, 101)   # long bearish closing below midpoint of b1 (104.5)
        assert is_evening_star(b1, b2, b3)


# ─────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────
class TestDetectRecentPatterns:
    def test_finds_hammer_in_recent_bars(self):
        # Build 10 bars; place a hammer at index -2
        rows = []
        for i in range(10):
            if i == 8:
                rows.append({"open": 100, "high": 101.2, "low": 95, "close": 101})
            else:
                rows.append({"open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100.5 + i})
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame(rows, index=idx)
        patterns = detect_recent_patterns(df, lookback=5)
        names = {p["name"] for p in patterns}
        assert "hammer" in names

    def test_short_df_returns_empty(self):
        df = pd.DataFrame(
            [{"open": 100, "high": 101, "low": 99, "close": 100}],
            index=pd.date_range("2024-01-01", periods=1, freq="D"),
        )
        assert detect_recent_patterns(df) == []
