"""Unit tests for the trend signal classifier (pure function, no I/O)."""
from __future__ import annotations

from price_predictor.agents.technical_agent.tools._trend_signal import (
    _adx_strength,
    _stack_score,
    classify_trend,
)


def _snapshot(
    *,
    close=1000.0,
    sma=None,
    above_sma=None,
    pct_above_sma=None,
    ema=1000.0,
    adx=25.0,
    di_plus=25.0,
    di_minus=15.0,
):
    """Build a synthetic snapshot dict for classifier testing."""
    sma = sma or {20: 990, 50: 980, 200: 950}
    if above_sma is None:
        above_sma = {
            n: (close > v if (close is not None and v is not None) else None)
            for n, v in sma.items()
        }
    if pct_above_sma is None:
        pct_above_sma = {
            n: (round((close - v) / v * 100, 2)
                if (close is not None and v is not None) else None)
            for n, v in sma.items()
        }
    return {
        "close": close,
        "sma": sma,
        "ema": ema,
        "above_sma": above_sma,
        "pct_above_sma": pct_above_sma,
        "adx": {"adx": adx, "di_plus": di_plus, "di_minus": di_minus},
    }


# ─────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────
class TestStackScore:
    def test_all_above(self):
        assert _stack_score({20: True, 50: True, 200: True}) == 3

    def test_all_below(self):
        assert _stack_score({20: False, 50: False, 200: False}) == 0

    def test_mixed(self):
        assert _stack_score({20: True, 50: False, 200: True}) == 2

    def test_none_values_ignored(self):
        assert _stack_score({20: True, 50: None, 200: True}) == 2


class TestAdxStrength:
    def test_strong(self):
        assert _adx_strength(45) == "strong"
        assert _adx_strength(40) == "strong"

    def test_moderate(self):
        assert _adx_strength(30) == "moderate"
        assert _adx_strength(25) == "moderate"

    def test_weak(self):
        assert _adx_strength(15) == "weak"
        assert _adx_strength(0) == "weak"

    def test_none_returns_weak(self):
        assert _adx_strength(None) == "weak"


# ─────────────────────────────────────────────────────────────────
# classify_trend
# ─────────────────────────────────────────────────────────────────
class TestClassifyTrendBullish:
    def test_classic_bullish_setup(self):
        # Close above all SMAs, ADX 30 (developing trend), +DI > -DI
        snap = _snapshot(adx=30, di_plus=30, di_minus=15)
        signal, strength, rationale = classify_trend(snap)
        assert signal == "bullish"
        assert strength == "moderate"
        assert any("above" in r.lower() for r in rationale)
        assert any("+DI" in r and "bullish" in r.lower() for r in rationale)

    def test_strong_bullish(self):
        snap = _snapshot(adx=50, di_plus=35, di_minus=10)
        signal, strength, _ = classify_trend(snap)
        assert signal == "bullish"
        assert strength == "strong"

    def test_two_of_three_smas_above_still_bullish(self):
        # Above SMA-20 and SMA-50, below SMA-200 (recovering downtrend)
        snap = _snapshot(
            close=1000,
            sma={20: 990, 50: 980, 200: 1100},
            adx=27, di_plus=28, di_minus=15,
        )
        signal, _, _ = classify_trend(snap)
        assert signal == "bullish"


class TestClassifyTrendBearish:
    def test_classic_bearish_setup(self):
        # Close below all SMAs, -DI > +DI
        snap = _snapshot(
            close=900,
            sma={20: 950, 50: 1000, 200: 1100},
            adx=30, di_plus=12, di_minus=28,
        )
        signal, strength, rationale = classify_trend(snap)
        assert signal == "bearish"
        assert strength == "moderate"
        assert any("below" in r.lower() for r in rationale)
        assert any("-DI" in r and "bearish" in r.lower() for r in rationale)

    def test_strong_bearish(self):
        snap = _snapshot(
            close=900,
            sma={20: 950, 50: 1000, 200: 1100},
            adx=50, di_plus=10, di_minus=35,
        )
        signal, strength, _ = classify_trend(snap)
        assert signal == "bearish"
        assert strength == "strong"


class TestClassifyTrendNeutral:
    def test_mixed_smas_neutral(self):
        # Above SMA-20, below SMA-50 and SMA-200; mixed DI
        snap = _snapshot(
            close=995,
            sma={20: 990, 50: 1000, 200: 1050},
            adx=15, di_plus=20, di_minus=20,
        )
        signal, strength, _ = classify_trend(snap)
        assert signal == "neutral"
        assert strength == "weak"

    def test_di_disagrees_with_smas_neutral(self):
        # All SMAs above (would be bullish) but DI strongly bearish
        snap = _snapshot(adx=25, di_plus=10, di_minus=30)
        signal, _, _ = classify_trend(snap)
        # DI disagreement -> we should NOT call it bullish
        assert signal != "bullish"


class TestClassifyTrendInsufficientData:
    def test_no_close_returns_neutral(self):
        snap = _snapshot(close=None)
        signal, strength, rationale = classify_trend(snap)
        assert signal == "neutral"
        assert strength == "weak"
        assert any("insufficient" in r.lower() for r in rationale)

    def test_all_smas_none_returns_neutral(self):
        snap = _snapshot()
        snap["sma"] = {20: None, 50: None, 200: None}
        snap["above_sma"] = {20: None, 50: None, 200: None}
        signal, strength, rationale = classify_trend(snap)
        assert signal == "neutral"
        assert strength == "weak"
        assert any("insufficient" in r.lower() for r in rationale)

    def test_missing_adx_still_classifies(self):
        # SMAs all above + ADX missing -> can still tell direction from stack
        snap = _snapshot()
        snap["adx"] = {"adx": None, "di_plus": None, "di_minus": None}
        signal, strength, _ = classify_trend(snap)
        # Without DI we can still call it bullish if all SMAs are above
        assert signal == "bullish"
        assert strength == "weak"  # No ADX = unknown strength
