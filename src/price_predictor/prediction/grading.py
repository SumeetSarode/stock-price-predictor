"""Grading: how did a Prediction actually play out?

WHY THIS MODULE EXISTS
======================
We can generate predictions all day, but until we MEASURE whether they
came true we have no idea if the LLM is any good. Without measurement,
'the system' is just confident-sounding noise.

This module answers ONE question per prediction:
  Did target hit, stop hit, or did the window expire?

Aggregation across many graded predictions (hit-rate, Brier score,
direction accuracy) lives in calibration.py - kept separate so this
module stays a PURE pricing logic file.

DESIGN: grade_one is a PURE FUNCTION
====================================
Inputs: a Prediction + a DataFrame of bars after the prediction.
Outputs: a GradedPrediction.
NO I/O. NO network. NO file reads.

This matters because:
- Tests are trivial: build a tiny DataFrame inline, assert the outcome.
- The expensive part (fetching post-prediction OHLCV) lives in a
  separate orchestration layer that wraps grade_one.
- Backtesting on historical data uses the same function as live grading.

OUTCOME DETECTION SEMANTICS (the meat)
======================================
For a BULLISH prediction with target T and stop S:
  - target hit when bar.high >= T (price reached up to target)
  - stop hit   when bar.low  <= S (price drew down to stop)

For a BEARISH prediction (mirror):
  - target hit when bar.low  <= T (price dropped to target)
  - stop hit   when bar.high >= S (price rallied to stop)

For NEUTRAL: there's no target/stop semantics (range-bound call).
  - Outcome is always EXPIRED.
  - direction_correct: True iff |realized_return| <= NEUTRAL_TOLERANCE.

THE 'BOTH HIT SAME BAR' AMBIGUITY
=================================
Daily bars only tell us the day's high and low, not the SEQUENCE in which
they were touched. If a bullish trade has BOTH high>=T and low<=S on the
same day, we genuinely don't know if you got out at target or got
stopped first.

Industry standard: assume the WORST (stop hit first). Reason:
1. Conservative for paper-trading hit-rate claims.
2. Real execution often gets the bad fill (slippage, gap risk).
3. We surface AMBIGUOUS so the user knows the data was uncertain,
   not just the outcome.

We split this into two outcomes:
  STOP_HIT_AMBIGUOUS  - both touched same bar, called as stop (worst-case)
  STOP_HIT            - stop touched on a bar where target wasn't
This lets calibration give weight or filter ambiguous days separately.

REALIZED RETURN SEMANTICS
=========================
Always computed against close_price_at_prediction (anchor frozen at
prediction time) using the LAST close in the grading window, not the
target/stop value. Why: the user wants 'what would I have made if I
held to expiry,' which is independent of whether they got out earlier.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from price_predictor.prediction.schema import (
    Prediction,
    PredictionDirection,
    PredictionHorizon,
)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
# How many trading days each horizon implies. Used as the WINDOW for
# grading: we look at the next N bars. Beyond N, the prediction is
# 'expired' regardless of what price does after.
#
# Mapping rationale:
#   intraday: 1  (today only - if intraday made before close, same-day bar)
#   short:    5  (1 trading week)
#   medium:   20 (~1 trading month)
#   long:     60 (~3 trading months)
_HORIZON_TRADING_DAYS: dict[PredictionHorizon, int] = {
    PredictionHorizon.INTRADAY: 1,
    PredictionHorizon.SHORT: 5,
    PredictionHorizon.MEDIUM: 20,
    PredictionHorizon.LONG: 60,
}

# Tolerance for NEUTRAL direction correctness: if the LLM said neutral
# and price moved less than this fraction either way, it was right.
# 2% is a deliberate choice - tighter would punish neutral calls in
# any volatile name; looser would count almost everything as 'neutral
# success'.
NEUTRAL_TOLERANCE: float = 0.02


def horizon_window(horizon: PredictionHorizon) -> int:
    """Public lookup for trading-day window per horizon.

    Exposed so the orchestration layer (which fetches the post-
    prediction bars) knows how many days of OHLCV to ask for.
    """
    return _HORIZON_TRADING_DAYS[horizon]


# ─────────────────────────────────────────────────────────────
# Outcome enum
# ─────────────────────────────────────────────────────────────
class GradeOutcome(str, Enum):
    """What happened to a prediction within its horizon window.

    Why these specific values:
      TARGET_HIT          - clean win
      STOP_HIT            - clean loss
      STOP_HIT_AMBIGUOUS  - same-bar tie, called as loss (conservative)
      EXPIRED             - neither hit; window closed. Outcome judged
                            by realized return + direction.
      INCONCLUSIVE        - we couldn't grade (no post-prediction bars
                            available, e.g. weekend prediction with no
                            trading data yet). Different from EXPIRED:
                            no judgement made.
      NOT_APPLICABLE      - NEUTRAL prediction; target/stop semantics
                            don't apply. Use direction_correct instead.
    """

    TARGET_HIT = "target_hit"
    STOP_HIT = "stop_hit"
    STOP_HIT_AMBIGUOUS = "stop_hit_ambiguous"
    EXPIRED = "expired"
    INCONCLUSIVE = "inconclusive"
    NOT_APPLICABLE = "not_applicable"


# ─────────────────────────────────────────────────────────────
# Result model
# ─────────────────────────────────────────────────────────────
class GradedPrediction(BaseModel):
    """The original prediction + how it actually played out.

    Frozen because grading is a one-shot, immutable record. If grading
    logic changes (e.g. new outcome class), we re-grade and produce a
    NEW GradedPrediction; we don't mutate.

    Why embed the full Prediction (not just an ID): grades are studied
    long after the prediction was made. Carrying the prediction with
    the grade means a single GradedPrediction object is fully
    self-describing - no joins or stores required to read it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prediction: Prediction
    outcome: GradeOutcome
    realized_return: float = Field(
        ...,
        description=(
            "(close_at_window_end - close_at_prediction) / close_at_prediction. "
            "Negative for losses. Always present (even for INCONCLUSIVE we "
            "report 0.0)."
        ),
    )
    direction_correct: Optional[bool] = Field(
        ...,
        description=(
            "Did the predicted direction match what happened?\n"
            "  - bullish:  True iff realized_return > 0\n"
            "  - bearish:  True iff realized_return < 0\n"
            "  - neutral:  True iff |realized_return| <= NEUTRAL_TOLERANCE\n"
            "None if INCONCLUSIVE (can't judge what we couldn't measure)."
        ),
    )
    days_to_resolution: Optional[int] = Field(
        ...,
        description=(
            "Trading days from prediction to outcome bar. "
            "None for EXPIRED (used the full window) and INCONCLUSIVE."
        ),
    )
    bars_examined: int = Field(
        ..., ge=0,
        description="How many post-prediction bars were inspected.",
    )
    close_at_window_end: Optional[float] = Field(
        ...,
        description="Close of the LAST bar in the window. None if INCONCLUSIVE.",
    )

    @property
    def hit(self) -> bool:
        """Convenience: was this a target hit (vs stop or expiry)?

        Used by aggregation code to compute hit-rate. Ambiguous stops
        count as misses (conservative, matches outcome semantics).
        """
        return self.outcome == GradeOutcome.TARGET_HIT


# ─────────────────────────────────────────────────────────────
# Core grading function
# ─────────────────────────────────────────────────────────────
def _validate_bars(bars: pd.DataFrame) -> None:
    """Cheap precondition check: required columns present.

    Raises ValueError with a useful message rather than letting
    KeyError surface later from deep inside the loop.
    """
    needed = {"high", "low", "close"}
    missing = needed - set(bars.columns)
    if missing:
        raise ValueError(
            f"future_bars missing required columns: {sorted(missing)}. "
            f"Got: {sorted(bars.columns)}"
        )


def _grade_directional(
    prediction: Prediction, bars: pd.DataFrame,
) -> tuple[GradeOutcome, Optional[int]]:
    """Walk forward bar-by-bar; return (outcome, day_index_or_None).

    PURE: depends only on inputs. Returns the index (0-based) of the
    bar where resolution happened, or None if no resolution within
    the window (caller will mark EXPIRED).
    """
    target = prediction.target.value
    stop = prediction.stop_loss.value
    is_bullish = prediction.direction == PredictionDirection.BULLISH

    for i, (_, bar) in enumerate(bars.iterrows()):
        high = float(bar["high"])
        low = float(bar["low"])

        if is_bullish:
            target_touched = high >= target
            stop_touched = low <= stop
        else:  # bearish
            target_touched = low <= target
            stop_touched = high >= stop

        if target_touched and stop_touched:
            # Same-bar ambiguity. Conservative call: stop hit first.
            return GradeOutcome.STOP_HIT_AMBIGUOUS, i
        if target_touched:
            return GradeOutcome.TARGET_HIT, i
        if stop_touched:
            return GradeOutcome.STOP_HIT, i

    return GradeOutcome.EXPIRED, None


def _direction_correct(
    direction: PredictionDirection, realized_return: float,
) -> bool:
    """Did the predicted direction match reality?

    Pure function with three branches matching PredictionDirection.
    Extracted so it's separately testable and so its semantics are
    visible at a glance.
    """
    if direction == PredictionDirection.BULLISH:
        return realized_return > 0
    if direction == PredictionDirection.BEARISH:
        return realized_return < 0
    # NEUTRAL: 'right' means price stayed approximately flat
    return abs(realized_return) <= NEUTRAL_TOLERANCE


def grade_one(
    prediction: Prediction,
    future_bars: pd.DataFrame,
) -> GradedPrediction:
    """Grade a single prediction against the bars that came after it.

    Args:
        prediction:   The Prediction to evaluate.
        future_bars:  DataFrame with columns 'high', 'low', 'close',
                      indexed chronologically. Should contain ONLY bars
                      AFTER prediction.as_of (caller's responsibility -
                      we don't filter to avoid silent bugs from the
                      caller's perspective).

    Returns:
        GradedPrediction with outcome + realized metrics.

    Raises:
        ValueError: future_bars missing required columns.

    Note:
        We slice future_bars to the horizon window internally, so callers
        can pass more bars than needed (e.g. always pass 90 days; we use
        only the relevant subset based on horizon).
    """
    _validate_bars(future_bars)

    window = horizon_window(prediction.horizon)
    bars = future_bars.head(window)
    n_bars = len(bars)

    # ── Empty post-prediction window: can't grade anything ──
    if n_bars == 0:
        return GradedPrediction(
            prediction=prediction,
            outcome=GradeOutcome.INCONCLUSIVE,
            realized_return=0.0,
            direction_correct=None,
            days_to_resolution=None,
            bars_examined=0,
            close_at_window_end=None,
        )

    close_at_pred = prediction.analysis_basis.close_price_at_prediction
    close_at_end = float(bars["close"].iloc[-1])
    realized_return = (close_at_end - close_at_pred) / close_at_pred

    # ── NEUTRAL: target/stop semantics don't apply ──
    if prediction.direction == PredictionDirection.NEUTRAL:
        return GradedPrediction(
            prediction=prediction,
            outcome=GradeOutcome.NOT_APPLICABLE,
            realized_return=realized_return,
            direction_correct=_direction_correct(prediction.direction, realized_return),
            days_to_resolution=None,  # neutral never 'resolves' to T or S
            bars_examined=n_bars,
            close_at_window_end=close_at_end,
        )

    # ── BULLISH / BEARISH: walk the bars looking for resolution ──
    outcome, resolved_at = _grade_directional(prediction, bars)

    # If resolved early, days_to_resolution is the bar index + 1 (1-indexed
    # for human readability: 'hit on day 3' not 'hit on bar index 2').
    days_to_resolution = (resolved_at + 1) if resolved_at is not None else None

    return GradedPrediction(
        prediction=prediction,
        outcome=outcome,
        realized_return=realized_return,
        direction_correct=_direction_correct(prediction.direction, realized_return),
        days_to_resolution=days_to_resolution,
        bars_examined=n_bars,
        close_at_window_end=close_at_end,
    )
