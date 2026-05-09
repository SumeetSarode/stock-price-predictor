"""Tests for prediction.calibration.

Covers:
- Empty input safety (no DivisionByZero)
- All hit-rate variants compute correctly under known inputs
- Direction accuracy excludes inconclusives
- Brier score arithmetic on hand-checkable cases
- Mean / median return
- compute_breakdown groups correctly and produces stable order
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from price_predictor.prediction.calibration import (
    CalibrationReport,
    compute_breakdown,
    compute_calibration,
)
from price_predictor.prediction.grading import GradeOutcome, GradedPrediction
from price_predictor.prediction.schema import (
    AnalysisBasis,
    Prediction,
    PredictionDirection,
    PredictionHorizon,
    PriceLevel,
)


# ─────────────────────────────────────────────────────────────
# Factories
# ─────────────────────────────────────────────────────────────
_IST = ZoneInfo("Asia/Kolkata")


def _make_pred(
    confidence: float = 0.7,
    direction: PredictionDirection = PredictionDirection.BULLISH,
    horizon: PredictionHorizon = PredictionHorizon.WEEKLY,
    ticker: str = "TEST.NS",
    as_of: datetime | None = None,
) -> Prediction:
    """Build a minimal Prediction. Most callers tweak only confidence."""
    return Prediction(
        ticker=ticker,
        as_of=as_of or datetime(2026, 4, 28, 10, 30, tzinfo=_IST),
        horizon=horizon,
        model_chain=("synthesizer:test",),
        direction=direction,
        confidence=confidence,
        entry_zone=(99.0, 101.0),
        target=PriceLevel(value=110.0, rationale="test"),
        stop_loss=PriceLevel(value=95.0, rationale="test"),
        rationale="for testing",
        contributing_signals=("signal",),
        conflicting_signals=(),
        analysis_basis=AnalysisBasis(
            close_price_at_prediction=100.0,
            bars_used=400,
            technical_summary="ok",
        ),
    )


def _grade(
    *,
    outcome: GradeOutcome,
    realized_return: float = 0.0,
    direction_correct: bool | None = None,
    confidence: float = 0.7,
    direction: PredictionDirection = PredictionDirection.BULLISH,
    **pred_kwargs,
) -> GradedPrediction:
    """Build a GradedPrediction directly. Bypasses grade_one for
    aggregation tests - we want to control inputs precisely."""
    return GradedPrediction(
        prediction=_make_pred(
            confidence=confidence, direction=direction, **pred_kwargs
        ),
        outcome=outcome,
        realized_return=realized_return,
        direction_correct=direction_correct,
        days_to_resolution=None if outcome in (
            GradeOutcome.EXPIRED, GradeOutcome.INCONCLUSIVE,
            GradeOutcome.NOT_APPLICABLE,
        ) else 1,
        bars_examined=0 if outcome == GradeOutcome.INCONCLUSIVE else 5,
        close_at_window_end=None if outcome == GradeOutcome.INCONCLUSIVE else 105.0,
    )


# ─────────────────────────────────────────────────────────────
# Empty / degenerate inputs
# ─────────────────────────────────────────────────────────────
class TestEmptyInputs:
    def test_empty_input_safe_zeros(self):
        report = compute_calibration([])
        assert report.n_predictions == 0
        assert report.n_judged == 0
        assert report.hit_rate_strict == 0.0
        assert report.hit_rate_resolved == 0.0
        assert report.hit_rate_optimistic == 0.0
        assert report.direction_accuracy == 0.0
        assert report.brier_score is None
        assert report.mean_confidence is None
        assert report.mean_return == 0.0
        assert report.median_return == 0.0

    def test_all_inconclusive_safe(self):
        # Brier and mean_confidence undefined when no judgements exist.
        graded = [
            _grade(outcome=GradeOutcome.INCONCLUSIVE),
            _grade(outcome=GradeOutcome.INCONCLUSIVE),
        ]
        report = compute_calibration(graded)
        assert report.n_predictions == 2
        assert report.n_inconclusive == 2
        assert report.n_judged == 0
        assert report.brier_score is None
        assert report.mean_confidence is None


# ─────────────────────────────────────────────────────────────
# Hit-rate flavours
# ─────────────────────────────────────────────────────────────
class TestHitRates:
    def _mixed_set(self):
        # 3 wins, 2 clean stops, 1 ambiguous, 1 expired, 1 NA, 1 inconclusive
        return [
            _grade(outcome=GradeOutcome.TARGET_HIT, direction_correct=True, realized_return=0.1),
            _grade(outcome=GradeOutcome.TARGET_HIT, direction_correct=True, realized_return=0.08),
            _grade(outcome=GradeOutcome.TARGET_HIT, direction_correct=True, realized_return=0.12),
            _grade(outcome=GradeOutcome.STOP_HIT, direction_correct=False, realized_return=-0.05),
            _grade(outcome=GradeOutcome.STOP_HIT, direction_correct=False, realized_return=-0.04),
            _grade(outcome=GradeOutcome.STOP_HIT_AMBIGUOUS, direction_correct=False, realized_return=-0.03),
            _grade(outcome=GradeOutcome.EXPIRED, direction_correct=True, realized_return=0.02),
            _grade(outcome=GradeOutcome.NOT_APPLICABLE, direction_correct=True, realized_return=0.01,
                   direction=PredictionDirection.NEUTRAL),
            _grade(outcome=GradeOutcome.INCONCLUSIVE),
        ]

    def test_strict_hit_rate(self):
        # 3 wins / (3 wins + 2 stops + 1 ambig + 1 expired + 1 NA) = 3/8
        report = compute_calibration(self._mixed_set())
        assert report.hit_rate_strict == pytest.approx(3 / 8)

    def test_resolved_hit_rate(self):
        # 3 wins / (3 wins + 2 stops + 1 ambig) = 3/6 = 0.5
        report = compute_calibration(self._mixed_set())
        assert report.hit_rate_resolved == pytest.approx(0.5)

    def test_optimistic_hit_rate(self):
        # 3 wins / (3 wins + 2 clean stops) = 3/5 = 0.6 (excludes ambig)
        report = compute_calibration(self._mixed_set())
        assert report.hit_rate_optimistic == pytest.approx(0.6)

    def test_hit_rate_ordering(self):
        # By construction: strict <= resolved <= optimistic when there
        # are any expired / NA / ambiguous. This monotonicity is a
        # built-in property worth asserting.
        report = compute_calibration(self._mixed_set())
        assert report.hit_rate_strict <= report.hit_rate_resolved <= report.hit_rate_optimistic


# ─────────────────────────────────────────────────────────────
# Direction accuracy
# ─────────────────────────────────────────────────────────────
class TestDirectionAccuracy:
    def test_excludes_inconclusive(self):
        graded = [
            _grade(outcome=GradeOutcome.TARGET_HIT, direction_correct=True),
            _grade(outcome=GradeOutcome.STOP_HIT, direction_correct=False),
            _grade(outcome=GradeOutcome.INCONCLUSIVE),  # not counted
        ]
        report = compute_calibration(graded)
        # 1 correct out of 2 judged = 0.5 (NOT 1/3)
        assert report.n_with_direction_judgement == 2
        assert report.direction_accuracy == 0.5

    def test_all_correct(self):
        graded = [
            _grade(outcome=GradeOutcome.TARGET_HIT, direction_correct=True),
            _grade(outcome=GradeOutcome.EXPIRED, direction_correct=True),
        ]
        report = compute_calibration(graded)
        assert report.direction_accuracy == 1.0


# ─────────────────────────────────────────────────────────────
# Brier score
# ─────────────────────────────────────────────────────────────
class TestBrierScore:
    def test_perfect_calibration(self):
        # 100% confident + always right -> Brier = 0
        graded = [
            _grade(outcome=GradeOutcome.TARGET_HIT, direction_correct=True, confidence=1.0),
            _grade(outcome=GradeOutcome.TARGET_HIT, direction_correct=True, confidence=1.0),
        ]
        report = compute_calibration(graded)
        assert report.brier_score == pytest.approx(0.0)

    def test_pathological_calibration(self):
        # 100% confident but always wrong -> Brier = 1
        graded = [
            _grade(outcome=GradeOutcome.STOP_HIT, direction_correct=False, confidence=1.0),
            _grade(outcome=GradeOutcome.STOP_HIT, direction_correct=False, confidence=1.0),
        ]
        report = compute_calibration(graded)
        assert report.brier_score == pytest.approx(1.0)

    def test_50_percent_baseline(self):
        # Always 50% confident, half right -> Brier = 0.25
        # Each contribution: (0.5 - 1)^2 or (0.5 - 0)^2 = 0.25 either way.
        graded = [
            _grade(outcome=GradeOutcome.TARGET_HIT, direction_correct=True, confidence=0.5),
            _grade(outcome=GradeOutcome.STOP_HIT, direction_correct=False, confidence=0.5),
        ]
        report = compute_calibration(graded)
        assert report.brier_score == pytest.approx(0.25)

    def test_hand_computed_mixed(self):
        # confidence=0.8 right -> (0.8-1)^2 = 0.04
        # confidence=0.6 wrong -> (0.6-0)^2 = 0.36
        # mean = (0.04 + 0.36) / 2 = 0.20
        graded = [
            _grade(outcome=GradeOutcome.TARGET_HIT, direction_correct=True, confidence=0.8),
            _grade(outcome=GradeOutcome.STOP_HIT, direction_correct=False, confidence=0.6),
        ]
        report = compute_calibration(graded)
        assert report.brier_score == pytest.approx(0.20)
        # Mean confidence over judged
        assert report.mean_confidence == pytest.approx(0.7)


# ─────────────────────────────────────────────────────────────
# Returns
# ─────────────────────────────────────────────────────────────
class TestReturns:
    def test_mean_and_median_judged_only(self):
        # Inconclusive entries contribute 0.0 in the model but should
        # NOT pull down the mean (they're excluded from judged).
        graded = [
            _grade(outcome=GradeOutcome.TARGET_HIT, direction_correct=True, realized_return=0.10),
            _grade(outcome=GradeOutcome.STOP_HIT, direction_correct=False, realized_return=-0.04),
            _grade(outcome=GradeOutcome.EXPIRED, direction_correct=True, realized_return=0.02),
            _grade(outcome=GradeOutcome.INCONCLUSIVE),  # excluded
        ]
        report = compute_calibration(graded)
        # Mean of 0.10, -0.04, 0.02 = 0.0266...
        assert report.mean_return == pytest.approx((0.10 - 0.04 + 0.02) / 3)
        # Median of [-0.04, 0.02, 0.10] = 0.02
        assert report.median_return == pytest.approx(0.02)

    def test_median_even_count(self):
        graded = [
            _grade(outcome=GradeOutcome.TARGET_HIT, direction_correct=True, realized_return=0.10),
            _grade(outcome=GradeOutcome.TARGET_HIT, direction_correct=True, realized_return=0.20),
            _grade(outcome=GradeOutcome.STOP_HIT, direction_correct=False, realized_return=-0.05),
            _grade(outcome=GradeOutcome.STOP_HIT, direction_correct=False, realized_return=-0.10),
        ]
        report = compute_calibration(graded)
        # Sorted: [-0.10, -0.05, 0.10, 0.20] -> median = (-0.05 + 0.10)/2 = 0.025
        assert report.median_return == pytest.approx(0.025)


# ─────────────────────────────────────────────────────────────
# compute_breakdown
# ─────────────────────────────────────────────────────────────
class TestBreakdown:
    def test_groups_by_horizon(self):
        graded = [
            _grade(outcome=GradeOutcome.TARGET_HIT, direction_correct=True,
                   horizon=PredictionHorizon.WEEKLY),
            _grade(outcome=GradeOutcome.TARGET_HIT, direction_correct=True,
                   horizon=PredictionHorizon.WEEKLY),
            _grade(outcome=GradeOutcome.STOP_HIT, direction_correct=False,
                   horizon=PredictionHorizon.BIWEEKLY),
        ]
        breakdown = compute_breakdown(graded, lambda g: g.prediction.horizon)
        assert PredictionHorizon.WEEKLY in breakdown
        assert PredictionHorizon.BIWEEKLY in breakdown
        assert breakdown[PredictionHorizon.WEEKLY].n_predictions == 2
        assert breakdown[PredictionHorizon.WEEKLY].hit_rate_optimistic == 1.0
        assert breakdown[PredictionHorizon.BIWEEKLY].hit_rate_optimistic == 0.0

    def test_groups_by_ticker(self):
        graded = [
            _grade(outcome=GradeOutcome.TARGET_HIT, direction_correct=True, ticker="A"),
            _grade(outcome=GradeOutcome.STOP_HIT, direction_correct=False, ticker="B"),
            _grade(outcome=GradeOutcome.TARGET_HIT, direction_correct=True, ticker="A"),
        ]
        breakdown = compute_breakdown(graded, lambda g: g.prediction.ticker)
        assert breakdown["A"].n_predictions == 2
        assert breakdown["B"].n_predictions == 1

    def test_groups_by_month(self):
        # Custom key: by year-month string
        graded = [
            _grade(outcome=GradeOutcome.TARGET_HIT, direction_correct=True,
                   as_of=datetime(2026, 4, 5, tzinfo=_IST)),
            _grade(outcome=GradeOutcome.STOP_HIT, direction_correct=False,
                   as_of=datetime(2026, 5, 12, tzinfo=_IST)),
        ]
        breakdown = compute_breakdown(
            graded, lambda g: g.prediction.as_of.strftime("%Y-%m")
        )
        assert "2026-04" in breakdown
        assert "2026-05" in breakdown

    def test_empty_input_empty_breakdown(self):
        assert compute_breakdown([], lambda g: g.prediction.ticker) == {}


# ─────────────────────────────────────────────────────────────
# Report immutability
# ─────────────────────────────────────────────────────────────
class TestImmutability:
    def test_report_is_frozen(self):
        report = compute_calibration([])
        with pytest.raises(Exception):
            report.n_predictions = 999  # type: ignore[misc]

    def test_round_trips_through_json(self):
        graded = [
            _grade(outcome=GradeOutcome.TARGET_HIT, direction_correct=True, confidence=0.9),
            _grade(outcome=GradeOutcome.STOP_HIT, direction_correct=False, confidence=0.6),
        ]
        report = compute_calibration(graded)
        blob = report.model_dump_json()
        restored = CalibrationReport.model_validate_json(blob)
        assert restored == report
