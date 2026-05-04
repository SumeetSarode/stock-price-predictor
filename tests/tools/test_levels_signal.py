"""Tests for the levels signal classifier."""
from __future__ import annotations

import pytest

from price_predictor.agents.technical_agent.tools._levels_signal import (
    BEARISH_CHART_PATTERNS,
    BULLISH_CHART_PATTERNS,
    NEUTRAL_CHART_PATTERNS,
    classify_levels,
)


def _snap(close: float, swing_h=110.0, swing_l=90.0,
          high_52w=120.0, low_52w=80.0) -> dict:
    """Tiny helper for building a synthetic snapshot."""
    return {
        "close": close,
        "swing": {"swing_high": swing_h, "swing_low": swing_l},
        "fifty_two_week": {"high_52w": high_52w, "low_52w": low_52w},
        "pivots": {"pp": 100.0, "r1": 105.0, "r2": 110.0, "s1": 95.0, "s2": 90.0},
        "distance_pct": {
            "swing_high": round((swing_h - close) / close * 100, 2),
            "swing_low":  round((swing_l - close) / close * 100, 2),
            "high_52w":   round((high_52w - close) / close * 100, 2),
            "low_52w":    round((low_52w - close) / close * 100, 2),
        },
    }


class TestBreakoutDetection:
    def test_close_above_prior_swing_high_is_breakout(self):
        snap = _snap(close=112.0)  # close just above prior swing-high
        sig, _str, _r, _w, derived = classify_levels(
            snapshot=snap, prior_swing_high=110.0, prior_swing_low=90.0,
            prior_52w_high=None, prior_52w_low=None,
            atr=2.0, chart_patterns=[],
        )
        assert sig == "bullish"
        assert derived["breakout_state"] == "breakout"

    def test_close_below_prior_swing_low_is_breakdown(self):
        snap = _snap(close=88.0)
        sig, _, _, _, derived = classify_levels(
            snapshot=snap, prior_swing_high=110.0, prior_swing_low=90.0,
            prior_52w_high=None, prior_52w_low=None,
            atr=2.0, chart_patterns=[],
        )
        assert sig == "bearish"
        assert derived["breakout_state"] == "breakdown"

    def test_breaking_52w_level_is_strong(self):
        snap = _snap(close=125.0, high_52w=120.0)  # broke 52w high
        _sig, strength, _r, _w, _d = classify_levels(
            snapshot=snap, prior_swing_high=110.0, prior_swing_low=90.0,
            prior_52w_high=120.0, prior_52w_low=80.0,
            atr=2.0, chart_patterns=[],
        )
        assert strength == "strong"

    def test_breaking_swing_but_not_52w_is_moderate(self):
        # Broke swing-high (110) but NOT 52w high (200) -> moderate, not strong
        snap = _snap(close=112.0, high_52w=200.0)
        _sig, strength, _r, _w, _d = classify_levels(
            snapshot=snap, prior_swing_high=110.0, prior_swing_low=90.0,
            prior_52w_high=200.0, prior_52w_low=10.0,
            atr=2.0, chart_patterns=[],
        )
        assert strength == "moderate"

class TestNearLevel:
    def test_near_swing_low_is_bullish_bounce(self):
        # ATR=2, close=91, swing_low=90 -> distance 1 <= ATR -> near
        snap = _snap(close=91.0)
        sig, _, _, _, derived = classify_levels(
            snapshot=snap, prior_swing_high=200.0, prior_swing_low=50.0,
            prior_52w_high=None, prior_52w_low=None,
            atr=2.0, chart_patterns=[],
        )
        assert sig == "bullish"
        assert derived["near_level"] == "support"

    def test_near_swing_high_is_bearish_rejection(self):
        snap = _snap(close=109.0)  # 1 below 110, ATR=2
        sig, _, _, _, derived = classify_levels(
            snapshot=snap, prior_swing_high=200.0, prior_swing_low=50.0,
            prior_52w_high=None, prior_52w_low=None,
            atr=2.0, chart_patterns=[],
        )
        assert sig == "bearish"
        assert derived["near_level"] == "resistance"

    def test_near_52w_high_is_strong(self):
        # Set swing levels far away so only 52w high is near
        snap = _snap(close=119.5, swing_h=200.0, swing_l=10.0, high_52w=120.0, low_52w=5.0)
        _, strength, _, _, _ = classify_levels(
            snapshot=snap, prior_swing_high=300.0, prior_swing_low=1.0,
            prior_52w_high=None, prior_52w_low=None,
            atr=1.0, chart_patterns=[],
        )
        assert strength == "strong"

    def test_middle_of_range_is_neutral_weak(self):
        snap = _snap(close=100.0)  # smack in middle
        sig, strength, _, _, derived = classify_levels(
            snapshot=snap, prior_swing_high=200.0, prior_swing_low=50.0,
            prior_52w_high=None, prior_52w_low=None,
            atr=1.0, chart_patterns=[],
        )
        assert sig == "neutral"
        assert strength == "weak"
        assert derived["near_level"] == "none"
        assert derived["breakout_state"] == "none"


class TestChartPatternIntegration:
    def test_bullish_pattern_in_neutral_zone_no_conflict(self):
        snap = _snap(close=100.0)
        patterns = [{"name": "double_bottom", "confidence": 0.85, "key_levels": {}, "bar_indices": []}]
        _, _, rationale, warnings, _ = classify_levels(
            snapshot=snap, prior_swing_high=200.0, prior_swing_low=50.0,
            prior_52w_high=None, prior_52w_low=None,
            atr=1.0, chart_patterns=patterns,
        )
        assert any("double_bottom" in r for r in rationale)
        assert "pattern_signal_conflict" not in warnings

    def test_bearish_pattern_with_bullish_signal_warns(self):
        # Bullish breakout, bearish pattern -> conflict
        snap = _snap(close=112.0)
        patterns = [{"name": "head_shoulders", "confidence": 0.85, "key_levels": {}, "bar_indices": []}]
        sig, _, _, warnings, _ = classify_levels(
            snapshot=snap, prior_swing_high=110.0, prior_swing_low=90.0,
            prior_52w_high=None, prior_52w_low=None,
            atr=2.0, chart_patterns=patterns,
        )
        assert sig == "bullish"
        assert "pattern_signal_conflict" in warnings

    def test_bullish_pattern_with_bearish_signal_warns(self):
        snap = _snap(close=88.0)  # breakdown
        patterns = [{"name": "inverse_head_shoulders", "confidence": 0.85, "key_levels": {}, "bar_indices": []}]
        sig, _, _, warnings, _ = classify_levels(
            snapshot=snap, prior_swing_high=110.0, prior_swing_low=90.0,
            prior_52w_high=None, prior_52w_low=None,
            atr=2.0, chart_patterns=patterns,
        )
        assert sig == "bearish"
        assert "pattern_signal_conflict" in warnings

    def test_neutral_pattern_does_not_warn(self):
        snap = _snap(close=112.0)
        patterns = [{"name": "symmetric_triangle", "confidence": 0.85, "key_levels": {}, "bar_indices": []}]
        _, _, _, warnings, _ = classify_levels(
            snapshot=snap, prior_swing_high=110.0, prior_swing_low=90.0,
            prior_52w_high=None, prior_52w_low=None,
            atr=2.0, chart_patterns=patterns,
        )
        assert "pattern_signal_conflict" not in warnings

    def test_pattern_classifications_are_disjoint(self):
        all_patterns = (
            BULLISH_CHART_PATTERNS | BEARISH_CHART_PATTERNS | NEUTRAL_CHART_PATTERNS
        )
        # No pattern should appear in two sets
        assert (
            len(all_patterns)
            == len(BULLISH_CHART_PATTERNS)
            + len(BEARISH_CHART_PATTERNS)
            + len(NEUTRAL_CHART_PATTERNS)
        )


class TestEdgeCases:
    def test_no_atr_returns_neutral_weak(self):
        snap = _snap(close=100.0)
        sig, strength, _, warnings, _ = classify_levels(
            snapshot=snap, prior_swing_high=110.0, prior_swing_low=90.0,
            prior_52w_high=None, prior_52w_low=None,
            atr=None, chart_patterns=[],
        )
        assert sig == "neutral"
        assert strength == "weak"
        assert "insufficient_history" in warnings

    def test_no_close_returns_neutral_weak(self):
        snap = _snap(close=100.0)
        snap["close"] = None  # type: ignore
        sig, strength, _, warnings, _ = classify_levels(
            snapshot=snap, prior_swing_high=110.0, prior_swing_low=90.0,
            prior_52w_high=None, prior_52w_low=None,
            atr=2.0, chart_patterns=[],
        )
        assert sig == "neutral"
        assert strength == "weak"

    def test_zero_atr_returns_neutral_weak(self):
        snap = _snap(close=100.0)
        sig, _, _, warnings, _ = classify_levels(
            snapshot=snap, prior_swing_high=110.0, prior_swing_low=90.0,
            prior_52w_high=None, prior_52w_low=None,
            atr=0.0, chart_patterns=[],
        )
        assert sig == "neutral"
        assert "insufficient_history" in warnings

    def test_no_prior_swings_falls_back_to_near_level_logic(self):
        # No prior swing data -> can't detect breakout, but near-level still works
        snap = _snap(close=109.5)  # near swing-high
        sig, _, _, _, derived = classify_levels(
            snapshot=snap, prior_swing_high=None, prior_swing_low=None,
            prior_52w_high=None, prior_52w_low=None,
            atr=2.0, chart_patterns=[],
        )
        assert sig == "bearish"
        assert derived["near_level"] == "resistance"
        assert derived["breakout_state"] == "none"
