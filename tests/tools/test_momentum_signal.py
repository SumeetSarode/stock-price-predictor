"""Unit tests for the momentum signal classifier (pure function, no I/O)."""
from __future__ import annotations

from price_predictor.agents.technical_agent.tools._momentum_signal import (
    _rsi_strength,
    classify_momentum,
)


def _snapshot(
    *,
    rsi=55.0,
    macd_line=1.0,
    macd_signal=0.5,
    macd_hist=0.5,
    macd_cross=None,
    stoch_k=60.0,
    stoch_d=55.0,
    obv=10000.0,
    obv_slope=1.0,
):
    """Build a synthetic momentum snapshot for classifier tests."""
    return {
        "rsi": rsi,
        "macd": {
            "macd": macd_line,
            "signal": macd_signal,
            "histogram": macd_hist,
            "cross": macd_cross,
        },
        "stoch": {"k": stoch_k, "d": stoch_d},
        "obv": {"obv": obv, "slope_20": obv_slope},
    }


# ─────────────────────────────────────────────────────────────────
# _rsi_strength helper
# ─────────────────────────────────────────────────────────────────
class TestRsiStrength:
    def test_overbought(self):
        assert _rsi_strength(75) == "strong"

    def test_oversold(self):
        assert _rsi_strength(25) == "strong"

    def test_moderate_zone(self):
        assert _rsi_strength(50) == "moderate"
        assert _rsi_strength(40) == "moderate"
        assert _rsi_strength(60) == "moderate"

    def test_weak_transition(self):
        # 60-70 and 30-40 are transition zones (neither extreme nor neutral)
        assert _rsi_strength(65) == "weak"
        assert _rsi_strength(35) == "weak"

    def test_none_returns_weak(self):
        assert _rsi_strength(None) == "weak"


# ─────────────────────────────────────────────────────────────────
# classify_momentum -- bullish cases
# ─────────────────────────────────────────────────────────────────
class TestClassifyMomentumBullish:
    def test_classic_bullish(self):
        # RSI >50, MACD hist >0, Stoch %K > %D
        snap = _snapshot(rsi=60, macd_hist=1.5, stoch_k=70, stoch_d=60)
        signal, strength, rationale, warnings = classify_momentum(snap)
        assert signal == "bullish"
        assert strength == "moderate"  # RSI 60 is in moderate zone
        assert any("RSI" in r and "bullish" in r.lower() for r in rationale)
        assert warnings == []

    def test_strong_bullish_overbought(self):
        snap = _snapshot(rsi=75, macd_hist=2.0, stoch_k=85, stoch_d=80)
        signal, strength, _, _ = classify_momentum(snap)
        assert signal == "bullish"
        assert strength == "strong"  # overbought = strong momentum

    def test_macd_cross_mentioned_in_rationale(self):
        snap = _snapshot(rsi=55, macd_hist=0.1, macd_cross="bullish",
                         stoch_k=60, stoch_d=55)
        _, _, rationale, _ = classify_momentum(snap)
        assert any("crossed positive" in r.lower() for r in rationale)


# ─────────────────────────────────────────────────────────────────
# classify_momentum -- bearish cases
# ─────────────────────────────────────────────────────────────────
class TestClassifyMomentumBearish:
    def test_classic_bearish(self):
        snap = _snapshot(rsi=40, macd_hist=-1.5, stoch_k=30, stoch_d=40)
        signal, strength, rationale, warnings = classify_momentum(snap)
        assert signal == "bearish"
        assert strength == "moderate"
        assert any("RSI" in r and "bearish" in r.lower() for r in rationale)

    def test_strong_bearish_oversold(self):
        snap = _snapshot(rsi=20, macd_hist=-2.0, stoch_k=15, stoch_d=20)
        signal, strength, _, _ = classify_momentum(snap)
        assert signal == "bearish"
        assert strength == "strong"  # oversold = strong (downward) momentum


# ─────────────────────────────────────────────────────────────────
# classify_momentum -- neutral cases
# ─────────────────────────────────────────────────────────────────
class TestClassifyMomentumNeutral:
    def test_mixed_signals(self):
        # RSI bullish, MACD bearish, Stoch bullish -> mixed
        snap = _snapshot(rsi=55, macd_hist=-0.5, stoch_k=60, stoch_d=55)
        signal, _, _, _ = classify_momentum(snap)
        assert signal == "neutral"

    def test_all_at_threshold_returns_neutral(self):
        # RSI exactly 50 -> neither bullish nor bearish
        snap = _snapshot(rsi=50, macd_hist=0, stoch_k=50, stoch_d=50)
        signal, _, _, _ = classify_momentum(snap)
        assert signal == "neutral"


# ─────────────────────────────────────────────────────────────────
# classify_momentum -- OBV divergence detection
# ─────────────────────────────────────────────────────────────────
class TestObvDivergence:
    def test_bullish_signal_falling_obv_emits_warning(self):
        # Price momentum says bullish but OBV slope strongly negative
        snap = _snapshot(rsi=60, macd_hist=1.5, stoch_k=70, stoch_d=60,
                         obv_slope=-5.0)
        signal, _, rationale, warnings = classify_momentum(snap)
        assert signal == "bullish"
        assert "obv_divergence" in warnings
        assert any("⚠" in r and "diverging" in r for r in rationale)

    def test_bearish_signal_rising_obv_emits_warning(self):
        snap = _snapshot(rsi=40, macd_hist=-1.5, stoch_k=30, stoch_d=40,
                         obv_slope=5.0)
        signal, _, _, warnings = classify_momentum(snap)
        assert signal == "bearish"
        assert "obv_divergence" in warnings

    def test_aligned_obv_no_warning(self):
        # Bullish signal + rising OBV -> no warning
        snap = _snapshot(rsi=60, macd_hist=1.5, stoch_k=70, stoch_d=60,
                         obv_slope=3.0)
        _, _, _, warnings = classify_momentum(snap)
        assert "obv_divergence" not in warnings

    def test_small_obv_slope_not_divergent(self):
        # OBV slope tiny (-0.5%) -> below threshold, no warning even if direction differs
        snap = _snapshot(rsi=60, macd_hist=1.5, stoch_k=70, stoch_d=60,
                         obv_slope=-0.5)
        _, _, _, warnings = classify_momentum(snap)
        assert "obv_divergence" not in warnings


# ─────────────────────────────────────────────────────────────────
# Insufficient data
# ─────────────────────────────────────────────────────────────────
class TestInsufficientData:
    def test_all_none_returns_neutral_with_warning(self):
        snap = {
            "rsi": None,
            "macd": {"macd": None, "signal": None, "histogram": None, "cross": None},
            "stoch": {"k": None, "d": None},
            "obv": {"obv": None, "slope_20": None},
        }
        signal, strength, rationale, warnings = classify_momentum(snap)
        assert signal == "neutral"
        assert strength == "weak"
        assert "insufficient_history" in warnings
        assert any("insufficient" in r.lower() for r in rationale)

    def test_partial_data_still_classifies(self):
        # Only RSI and MACD available; Stoch missing
        snap = _snapshot(rsi=60, macd_hist=1.5, stoch_k=None, stoch_d=None)
        signal, _, _, _ = classify_momentum(snap)
        # 2 of 2 available components agree -> bullish (with available=2,
        # we need 2 - 1 = 1, so either single bullish or both is enough)
        assert signal == "bullish"
