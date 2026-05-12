"""Unit tests for the trend signal classifier (pure function, no I/O)."""
from __future__ import annotations

import pytest

from price_predictor.agents.technical_agent.tools._trend_signal import (
    MA_CROSS_FRESH_BARS,
    _adx_strength,
    _cross_label,
    _cross_vote_weight,
    _ma_cross_vote,
    _stack_score,
    classify_trend,
)


def _ma_struct(
    *, current="above", last_event=None, bars_since_event=None,
    short_ma=100.0, long_ma=99.0,
):
    """Convenience builder for an MA-cross sub-struct."""
    return {
        "current": current,
        "last_event": last_event,
        "bars_since_event": bars_since_event,
        "short_ma": short_ma,
        "long_ma": long_ma,
    }


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
    ma_crosses=None,
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
        "ma_crosses": ma_crosses if ma_crosses is not None else {},
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


# ─────────────────────────────────────────────────────────────────
# MA-cross helpers (Q1: bullish/bearish labels in code,
# Q2: SMA-50/200 + EMA-9/21, Q3: ±0.5 / ±0.3 fresh-only)
# ─────────────────────────────────────────────────────────────────
class TestCrossLabel:
    def test_sma_50_200_bullish_is_golden_cross(self):
        assert _cross_label("sma_50_200", "bullish") == "Golden Cross"

    def test_sma_50_200_bearish_is_death_cross(self):
        assert _cross_label("sma_50_200", "bearish") == "Death Cross"

    def test_ema_pair_uses_generic_label(self):
        # EMA-9/21 should NOT be called Golden Cross -- only the canonical
        # SMA-50/200 gets that name (Murphy 1999).
        assert _cross_label("ema_9_21", "bullish") == "bullish EMA-9/21 cross"

    def test_unknown_pair_falls_back(self):
        assert _cross_label("sma_3_8", "bearish") == "bearish SMA-3/8 cross"


class TestCrossVoteWeight:
    def test_sma_50_200_weight(self):
        assert _cross_vote_weight("sma_50_200") == 0.5

    def test_ema_9_21_weight(self):
        assert _cross_vote_weight("ema_9_21") == 0.3

    def test_unknown_pair_zero_weight(self):
        # Custom pairs surface in rationale but do not vote.
        assert _cross_vote_weight("sma_20_50") == 0.0


class TestMaCrossVote:
    def test_no_crosses_zero_vote(self):
        net, rationale = _ma_cross_vote({})
        assert net == 0.0
        assert rationale == []

    def test_fresh_golden_cross_positive_vote(self):
        crosses = {
            "sma_50_200": _ma_struct(last_event="bullish", bars_since_event=2),
        }
        net, rationale = _ma_cross_vote(crosses)
        assert net == pytest.approx(0.5)
        assert any("Golden Cross fired 2 bars ago" in r for r in rationale)

    def test_fresh_death_cross_negative_vote(self):
        crosses = {
            "sma_50_200": _ma_struct(
                current="below", last_event="bearish", bars_since_event=0,
            ),
        }
        net, rationale = _ma_cross_vote(crosses)
        assert net == pytest.approx(-0.5)
        assert any("Death Cross fired today" in r for r in rationale)

    def test_stale_cross_does_not_vote(self):
        crosses = {
            "sma_50_200": _ma_struct(
                last_event="bullish",
                bars_since_event=MA_CROSS_FRESH_BARS + 1,
            ),
        }
        net, rationale = _ma_cross_vote(crosses)
        assert net == 0.0
        assert any("stale" in r.lower() for r in rationale)

    def test_freshness_boundary_inclusive(self):
        # Exactly MA_CROSS_FRESH_BARS bars old should still vote.
        crosses = {
            "sma_50_200": _ma_struct(
                last_event="bullish",
                bars_since_event=MA_CROSS_FRESH_BARS,
            ),
        }
        net, _ = _ma_cross_vote(crosses)
        assert net == pytest.approx(0.5)

    def test_conflicting_fresh_crosses_partial_cancel(self):
        # SMA-50/200 bullish (+0.5), EMA-9/21 bearish (-0.3) -> net +0.2
        crosses = {
            "sma_50_200": _ma_struct(last_event="bullish", bars_since_event=1),
            "ema_9_21":   _ma_struct(
                current="below", last_event="bearish", bars_since_event=0,
            ),
        }
        net, _ = _ma_cross_vote(crosses)
        assert net == pytest.approx(0.2)

    def test_pair_with_no_event_in_history_emits_rationale(self):
        crosses = {
            "sma_50_200": _ma_struct(
                current="above", last_event=None, bars_since_event=None,
            ),
        }
        net, rationale = _ma_cross_vote(crosses)
        assert net == 0.0
        assert any("No sma-50-200 cross" in r for r in rationale)


class TestClassifyTrendWithCrosses:
    def test_fresh_golden_cross_nudges_neutral_to_bullish(self):
        # SMA stack mixed (1 above, 2 below) and ADX low -> would be neutral
        # without the cross. Fresh golden cross + EMA cross = +0.8 vote ->
        # nudges bullish.
        snap = _snapshot(
            close=1000,
            sma={20: 1010, 50: 990, 200: 1020},
            adx=18, di_plus=22, di_minus=18,
            ma_crosses={
                "sma_50_200": _ma_struct(
                    last_event="bullish", bars_since_event=2,
                ),
                "ema_9_21": _ma_struct(
                    last_event="bullish", bars_since_event=0,
                ),
            },
        )
        signal, _, rationale = classify_trend(snap)
        assert signal == "bullish"
        assert any("nudged bullish" in r for r in rationale)

    def test_fresh_death_cross_nudges_neutral_to_bearish(self):
        snap = _snapshot(
            close=1000,
            sma={20: 990, 50: 1010, 200: 980},
            adx=18, di_plus=18, di_minus=22,
            ma_crosses={
                "sma_50_200": _ma_struct(
                    current="below", last_event="bearish", bars_since_event=1,
                ),
                "ema_9_21": _ma_struct(
                    current="below", last_event="bearish", bars_since_event=0,
                ),
            },
        )
        signal, _, rationale = classify_trend(snap)
        assert signal == "bearish"
        assert any("nudged bearish" in r for r in rationale)

    def test_cross_does_not_override_locked_bullish(self):
        # SMA stack fully aligned + DI bullish -> already bullish.
        # A bearish cross should NOT flip it (verdict locked).
        snap = _snapshot(
            adx=30, di_plus=30, di_minus=15,
            ma_crosses={
                "sma_50_200": _ma_struct(
                    current="below", last_event="bearish", bars_since_event=1,
                ),
            },
        )
        signal, _, _ = classify_trend(snap)
        assert signal == "bullish"

    def test_cross_does_not_nudge_against_di(self):
        # Mixed SMA stack but DI is strongly bearish.
        # A bullish golden cross should NOT nudge to bullish in that case.
        snap = _snapshot(
            close=1000,
            sma={20: 1010, 50: 990, 200: 1020},
            adx=18, di_plus=10, di_minus=30,
            ma_crosses={
                "sma_50_200": _ma_struct(
                    last_event="bullish", bars_since_event=2,
                ),
            },
        )
        signal, _, _ = classify_trend(snap)
        # DI says bearish, cross says bullish -> classifier refuses to nudge
        assert signal != "bullish"

    def test_stale_cross_keeps_neutral_neutral(self):
        snap = _snapshot(
            close=1000,
            sma={20: 1010, 50: 990, 200: 1020},
            adx=18, di_plus=22, di_minus=18,
            ma_crosses={
                "sma_50_200": _ma_struct(
                    last_event="bullish",
                    bars_since_event=MA_CROSS_FRESH_BARS + 5,
                ),
            },
        )
        signal, _, rationale = classify_trend(snap)
        assert signal == "neutral"
        assert any("stale" in r.lower() for r in rationale)

    def test_classifier_works_with_no_ma_crosses_key(self):
        """Backward-compat: classifier must not crash on snapshots without
        the ma_crosses key (e.g. legacy callers / partial mocks)."""
        snap = _snapshot()
        del snap["ma_crosses"]
        # should not raise
        signal, _, _ = classify_trend(snap)
        assert signal in ("bullish", "neutral", "bearish")
