"""Calibration: aggregate metrics over many GradedPredictions.

WHY THIS MODULE
===============
grade_one tells us 'what happened to ONE prediction.' Calibration tells
us 'is the LLM, on average, any good?' That's the only question that
matters for deciding whether to trust the system.

Three classes of metric live here:
  1. Hit-rate variants  - how often did target hit vs stop?
  2. Direction accuracy - was the directional call right?
  3. Brier score        - was the LLM's CONFIDENCE well-calibrated?

WHY THREE HIT-RATE FLAVOURS?
============================
The same-bar ambiguity problem from grade_one bubbles up here. We
report three numbers so the user can pick the framing:

  - hit_rate_strict     = wins / (wins + losses + ambiguous + expired)
                          Treats expired and ambiguous as misses.
                          Most pessimistic; floors the LLM's claim.
  - hit_rate_resolved   = wins / (wins + losses + ambiguous)
                          Of the trades that actually closed at T or S,
                          what fraction won? Industry-standard 'win rate.'
  - hit_rate_optimistic = wins / (wins + clean_losses)
                          Excludes ambiguous days entirely. Useful when
                          ambiguity is rare and you want to see the
                          'unambiguous skill.'

By REPORTING all three (not picking one) we let the reader draw their
own conclusions. Cherry-picking the best one would be a sin.

WHY BRIER SCORE?
================
Hit-rate tells you 'is the LLM right on average?' but NOT 'is the LLM's
self-reported confidence honest?' A model that says 90% confident and
only hits 50% is overconfident; a model that says 60% and hits 80% is
under-confident. Both are 'wrong' in different ways.

Brier score = mean((confidence - actual_binary_outcome)^2)
  - Range [0, 1], lower is better.
  - 0 = perfect (every confident-bullish call ended bullish).
  - 0.25 = a model that always predicts p=0.5 regardless of outcome.
  - 1.0 = pathologically wrong (says 100% bullish, always bearish).

We compute it ONLY on predictions where direction_correct is known
(skip INCONCLUSIVE). For NEUTRAL we use the |return|-within-tolerance
binary as the 'event happened' signal.

DESIGN: REPORTS COMPOSE VIA compute_breakdown
=============================================
Rather than baking by_horizon / by_direction nesting into the report
class, we expose a generic compute_breakdown(graded, key_fn) helper.
Callers can group by anything: ticker, horizon, direction, week-of-
month, model_chain, whatever. Each group yields its own
CalibrationReport. Composable, testable, and avoids recursive schemas.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Hashable, Iterable, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from price_predictor.prediction.grading import (
    GradedPrediction,
    GradeOutcome,
)

# ─────────────────────────────────────────────────────────────
# The report
# ─────────────────────────────────────────────────────────────
class CalibrationReport(BaseModel):
    """Aggregate metrics over a set of GradedPredictions.

    Frozen + extra='forbid' so a report once computed is immutable
    evidence; any tweak requires re-computing from raw grades.

    All metrics are SAFE TO READ even when n_predictions == 0:
    rates default to 0.0, brier_score is None, etc. This avoids
    DivisionByZero in callers and lets templates render empty
    states without special-casing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ── Counts ──────────────────────────────────────────────
    n_predictions: int = Field(..., ge=0, description="Total grades in the set")
    n_inconclusive: int = Field(..., ge=0)
    n_target_hit: int = Field(..., ge=0)
    n_stop_hit: int = Field(..., ge=0, description="Clean stops (no T-touch)")
    n_stop_hit_ambiguous: int = Field(..., ge=0)
    n_expired: int = Field(..., ge=0)
    n_not_applicable: int = Field(..., ge=0, description="NEUTRAL preds")

    # ── Hit-rate variants (see module docstring for semantics) ──
    hit_rate_strict: float = Field(..., ge=0.0, le=1.0)
    hit_rate_resolved: float = Field(..., ge=0.0, le=1.0)
    hit_rate_optimistic: float = Field(..., ge=0.0, le=1.0)

    # ── Direction accuracy (excludes inconclusive) ──
    n_with_direction_judgement: int = Field(..., ge=0)
    direction_accuracy: float = Field(..., ge=0.0, le=1.0)

    # ── Confidence calibration ──
    brier_score: float | None = Field(
        ...,
        description=(
            "Mean squared error between confidence and binary "
            "direction-correct outcome. None if no judgements available."
        ),
    )
    mean_confidence: float | None = Field(
        ...,
        description="Mean LLM-reported confidence over judged predictions",
    )

    # ── Realized return ──
    mean_return: float = Field(
        ...,
        description=(
            "Mean realized_return over JUDGED predictions (excludes "
            "INCONCLUSIVE so we don't tank the average with zeros)."
        ),
    )
    median_return: float

    @property
    def n_judged(self) -> int:
        """Number of predictions with a measurable outcome."""
        return self.n_predictions - self.n_inconclusive


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _safe_div(numerator: float, denominator: float) -> float:
    """Return 0.0 instead of raising on zero denominator.

    Centralizes the 'no data yet' edge case so every metric handles
    it consistently.
    """
    return numerator / denominator if denominator > 0 else 0.0


def _median(values: list[float]) -> float:
    """Inline median to avoid a numpy dep just for this. Empty -> 0.0."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


# ─────────────────────────────────────────────────────────────
# Main aggregation
# ─────────────────────────────────────────────────────────────
def compute_calibration(graded: Iterable[GradedPrediction]) -> CalibrationReport:
    """Aggregate a sequence of GradedPredictions into a single report.

    Idempotent + pure: same input -> same report. No I/O.

    The function accepts any iterable (list, generator, etc.) and
    materializes once internally. We pay the O(N) memory cost for
    code clarity; for the 'thousands of predictions' regime this is
    fine.
    """
    items = list(graded)
    n = len(items)

    # Outcome counts via dict-of-counters; reads cleaner than a giant
    # if/elif chain and leaves room for new outcomes without code
    # changes here.
    counts: dict[GradeOutcome, int] = defaultdict(int)
    for g in items:
        counts[g.outcome] += 1

    n_target = counts[GradeOutcome.TARGET_HIT]
    n_stop = counts[GradeOutcome.STOP_HIT]
    n_amb = counts[GradeOutcome.STOP_HIT_AMBIGUOUS]
    n_expired = counts[GradeOutcome.EXPIRED]
    n_na = counts[GradeOutcome.NOT_APPLICABLE]
    n_inc = counts[GradeOutcome.INCONCLUSIVE]

    # ── Hit-rate variants ─────────────────────────────────────
    # Strict: count expired + ambiguous + losses + na as 'not a win.'
    # NA included because neutral predictions don't 'win' in T/S sense;
    # they're handled separately via direction_accuracy.
    hr_strict = _safe_div(n_target, n_target + n_stop + n_amb + n_expired + n_na)
    # Resolved: only trades that closed at T or S (incl. ambiguous).
    hr_resolved = _safe_div(n_target, n_target + n_stop + n_amb)
    # Optimistic: drop ambiguous from the denominator (they're noise).
    hr_optimistic = _safe_div(n_target, n_target + n_stop)

    # ── Direction accuracy (skip inconclusive) ────────────────
    judged = [g for g in items if g.direction_correct is not None]
    n_judged = len(judged)
    n_correct = sum(1 for g in judged if g.direction_correct is True)
    direction_acc = _safe_div(n_correct, n_judged)

    # ── Brier score ───────────────────────────────────────────
    # mean((confidence - actual)^2) where actual = 1 if correct else 0.
    if n_judged > 0:
        brier = sum(
            (g.prediction.confidence - (1.0 if g.direction_correct else 0.0)) ** 2
            for g in judged
        ) / n_judged
        mean_conf = sum(g.prediction.confidence for g in judged) / n_judged
    else:
        brier = None
        mean_conf = None

    # ── Realized returns (judged only) ────────────────────────
    returns = [g.realized_return for g in judged]
    mean_ret = sum(returns) / len(returns) if returns else 0.0
    median_ret = _median(returns)

    return CalibrationReport(
        n_predictions=n,
        n_inconclusive=n_inc,
        n_target_hit=n_target,
        n_stop_hit=n_stop,
        n_stop_hit_ambiguous=n_amb,
        n_expired=n_expired,
        n_not_applicable=n_na,
        hit_rate_strict=hr_strict,
        hit_rate_resolved=hr_resolved,
        hit_rate_optimistic=hr_optimistic,
        n_with_direction_judgement=n_judged,
        direction_accuracy=direction_acc,
        brier_score=brier,
        mean_confidence=mean_conf,
        mean_return=mean_ret,
        median_return=median_ret,
    )


# ─────────────────────────────────────────────────────────────
# Breakdowns (group-by helper)
# ─────────────────────────────────────────────────────────────
K = TypeVar("K", bound=Hashable)


def compute_breakdown(
    graded: Iterable[GradedPrediction],
    key_fn: Callable[[GradedPrediction], K],
) -> dict[K, CalibrationReport]:
    """Group graded predictions by a key, return a report per group.

    Args:
        graded:  Sequence of GradedPredictions.
        key_fn:  Function mapping a GradedPrediction to a hashable
                 group key. Common choices:
                   lambda g: g.prediction.horizon
                   lambda g: g.prediction.direction
                   lambda g: g.prediction.ticker
                   lambda g: g.prediction.as_of.strftime('%Y-%m')

    Returns:
        Dict {group_key: CalibrationReport}. Insertion-ordered by
        first occurrence (Python 3.7+ dict semantics) so callers can
        stream stable output without re-sorting.

    Why a generic function over baked-in by_X fields:
        Composability. Any grouping the user cares about works without
        schema changes. Schema-baked breakdowns (by_horizon, by_ticker,
        by_direction etc.) would each duplicate this code and force
        anyone wanting a new grouping (by-week, by-model-chain) to
        modify the report class.
    """
    groups: dict[K, list[GradedPrediction]] = defaultdict(list)
    for g in graded:
        groups[key_fn(g)].append(g)
    return {k: compute_calibration(v) for k, v in groups.items()}
