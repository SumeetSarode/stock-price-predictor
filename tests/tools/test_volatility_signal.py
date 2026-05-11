"""Unit tests for the volatility signal classifier (pure function)."""
from __future__ import annotations

from price_predictor.agents.technical_agent.tools._volatility_signal import (
    classify_volatility,
    classify_volatility_regime,
)


def _snapshot(
    *,
    atr=2.5,
    atr_pct=2.0,
    bb_lower=95.0,
    bb_middle=100.0,
    bb_upper=105.0,
    bandwidth=10.0,
    percent_b=0.6,
    bollinger_squeeze=False,
    ttm_on=False,
    ttm_fire=False,
    ttm_bars=0,
):
    """Build a synthetic volatility snapshot for classifier tests."""
    return {
        "atr": atr,
        "atr_pct_of_price": atr_pct,
        "bbands": {
            "lower": bb_lower,
            "middle": bb_middle,
            "upper": bb_upper,
            "bandwidth": bandwidth,
            "percent_b": percent_b,
        },
        "bollinger_squeeze": bollinger_squeeze,
        "ttm_squeeze": {"on": ttm_on, "fire": ttm_fire, "bars_in_squeeze": ttm_bars},
    }


# ─────────────────────────────────────────────────────────────────
# Direction signal from %B
# ─────────────────────────────────────────────────────────────────
class TestSignalDirection:
    def test_high_percent_b_is_bullish(self):
        snap = _snapshot(percent_b=0.85)
        signal, _, _, _ = classify_volatility(snap)
        assert signal == "bullish"

    def test_low_percent_b_is_bearish(self):
        snap = _snapshot(percent_b=0.15)
        signal, _, _, _ = classify_volatility(snap)
        assert signal == "bearish"

    def test_middle_percent_b_is_neutral(self):
        snap = _snapshot(percent_b=0.5)
        signal, _, _, _ = classify_volatility(snap)
        assert signal == "neutral"

    def test_just_above_threshold_is_bullish(self):
        snap = _snapshot(percent_b=0.56)
        signal, _, _, _ = classify_volatility(snap)
        assert signal == "bullish"

    def test_just_below_threshold_is_bearish(self):
        snap = _snapshot(percent_b=0.44)
        signal, _, _, _ = classify_volatility(snap)
        assert signal == "bearish"

    def test_neutral_zone_no_strong_bias(self):
        snap = _snapshot(percent_b=0.5)
        _, _, rationale, _ = classify_volatility(snap)
        assert any("no directional bias" in r.lower() for r in rationale)


# ─────────────────────────────────────────────────────────────────
# %B extremes (price stretched outside bands)
# ─────────────────────────────────────────────────────────────────
class TestPercentBExtremes:
    def test_above_upper_band_emits_warning(self):
        snap = _snapshot(percent_b=1.15)
        _, _, rationale, warnings = classify_volatility(snap)
        assert "price_above_upper_band" in warnings
        assert any("⚠" in r and "above upper" in r for r in rationale)

    def test_below_lower_band_emits_warning(self):
        snap = _snapshot(percent_b=-0.10)
        _, _, _, warnings = classify_volatility(snap)
        assert "price_below_lower_band" in warnings

    def test_normal_range_no_warning(self):
        snap = _snapshot(percent_b=0.6)
        _, _, _, warnings = classify_volatility(snap)
        assert "price_above_upper_band" not in warnings
        assert "price_below_lower_band" not in warnings


# ─────────────────────────────────────────────────────────────────
# Strength: squeeze beats regime
# ─────────────────────────────────────────────────────────────────
class TestStrength:
    def test_ttm_squeeze_on_yields_strong_regardless_of_regime(self):
        # Even with low volatility, an active TTM squeeze is STRONG (breakout coming)
        snap = _snapshot(atr_pct=0.5, ttm_on=True, ttm_bars=8)
        _, strength, rationale, _ = classify_volatility(snap)
        assert strength == "strong"
        assert any("TTM SQUEEZE active" in r for r in rationale)

    def test_ttm_squeeze_fire_yields_strong(self):
        # Squeeze JUST released -- Carter's trade trigger.
        snap = _snapshot(atr_pct=0.5, ttm_fire=True, ttm_bars=12)
        _, strength, rationale, _ = classify_volatility(snap)
        assert strength == "strong"
        assert any("FIRED" in r for r in rationale)

    def test_bollinger_squeeze_alone_does_not_force_strong(self):
        # Bollinger flag firing without TTM → surfaced as diagnostic only.
        snap = _snapshot(atr_pct=2.5, bollinger_squeeze=True, ttm_on=False)
        _, strength, rationale, _ = classify_volatility(snap)
        assert strength == "moderate"   # ATR is normal, no TTM bump
        assert any("Bollinger bandwidth" in r for r in rationale)

    def test_dead_quiet_is_weak(self):
        snap = _snapshot(atr_pct=0.5, ttm_on=False)
        _, strength, _, _ = classify_volatility(snap)
        assert strength == "weak"

    def test_normal_volatility_is_moderate(self):
        snap = _snapshot(atr_pct=2.5, ttm_on=False)
        _, strength, _, _ = classify_volatility(snap)
        assert strength == "moderate"

    def test_manic_volatility_is_weak_with_warning(self):
        snap = _snapshot(atr_pct=8.0, ttm_on=False)
        _, strength, _, warnings = classify_volatility(snap)
        assert strength == "weak"
        assert "high_volatility" in warnings

    def test_elevated_but_not_manic_is_weak(self):
        # 4-6% range: elevated but not manic
        snap = _snapshot(atr_pct=5.0, ttm_on=False)
        _, strength, _, _ = classify_volatility(snap)
        assert strength == "weak"


# ─────────────────────────────────────────────────────────────────
# Insufficient data
# ─────────────────────────────────────────────────────────────────
class TestInsufficientData:
    def test_all_none_returns_neutral_with_warning(self):
        snap = {
            "atr": None,
            "atr_pct_of_price": None,
            "bbands": {
                "lower": None, "middle": None, "upper": None,
                "bandwidth": None, "percent_b": None,
            },
            "bollinger_squeeze": None,
            "ttm_squeeze": {"on": None, "fire": None, "bars_in_squeeze": None},
        }
        signal, strength, rationale, warnings = classify_volatility(snap)
        assert signal == "neutral"
        assert strength == "weak"
        assert "insufficient_history" in warnings

    def test_missing_pct_b_only_yields_neutral_signal(self):
        # ATR available but BB missing -> can't determine direction
        snap = _snapshot(atr=2.5, atr_pct=2.0)
        snap["bbands"]["percent_b"] = None
        signal, strength, _, _ = classify_volatility(snap)
        assert signal == "neutral"
        assert strength == "moderate"  # ATR-pct in normal range


# ─────────────────────────────────────────────────────────────────
# classify_volatility_regime helper
# ─────────────────────────────────────────────────────────────────
class TestRegimeClassifier:
    def test_low_regime(self):
        assert classify_volatility_regime(0.5) == "low"

    def test_normal_regime(self):
        assert classify_volatility_regime(2.0) == "normal"
        assert classify_volatility_regime(1.0) == "normal"
        assert classify_volatility_regime(4.0) == "normal"

    def test_high_regime_elevated(self):
        # 4-6%: elevated -> high
        assert classify_volatility_regime(5.0) == "high"

    def test_high_regime_manic(self):
        assert classify_volatility_regime(8.0) == "high"

    def test_unknown_for_none(self):
        assert classify_volatility_regime(None) == "unknown"
