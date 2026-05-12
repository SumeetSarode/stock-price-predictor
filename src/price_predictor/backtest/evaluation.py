"""Backtest evaluation -- grade a BacktestRun + compute calibration breakdowns.

WHY THIS EXISTS
===============
Step 2.1's run_backtest() produces a flat list of Predictions across
(ticker x as_of x horizon). To answer the actual question -- "is the
LLM any good?" -- we need to:

  1. GRADE each prediction: did target/stop hit? what was the realized
     return? was the direction call right? (existing grade_many)
  2. AGGREGATE across all grades into hit-rate, Brier, BSS, etc.
     (existing compute_calibration)
  3. SLICE by useful axes -- horizon, ticker, direction, month --
     so users can see "weekly works but daily is noise" or "RELIANCE
     calls are great but TCS is awful."

This module wires those three layers into ONE call:

    eval = evaluate_backtest(run)
    print(eval.overall.hit_rate_resolved)
    print(eval.by_horizon[PredictionHorizon.WEEKLY].direction_accuracy)
    print(eval.by_ticker["RELIANCE.NS"].brier_skill_score)

DESIGN CHOICES
==============
1. PURE WRAPPER, NO NEW LOGIC.
   grade_many already handles fetch + per-prediction grading + error
   capture. compute_calibration / compute_breakdown already do the
   stats. This file is ~40 lines of orchestration in a 200-line file
   of docstrings. DRY-as-can-be.

2. BREAKDOWNS COMPUTED EAGERLY.
   We slice by horizon, ticker, direction, AND month in one shot.
   They're cheap (O(N) per axis, N is few hundred typically), and
   the HTML report wants all four. Computing on-demand would force
   the report layer to know about CalibrationReport internals --
   leaky abstraction.

3. EVALUATION IS IMMUTABLE.
   Same reason BacktestRun is frozen: a graded run is evidence. To
   re-grade with different fetcher / today, you build a NEW
   BacktestEvaluation. No accidental mutation between read and write.

4. OPTIONAL fetch_ohlcv / today INJECTION.
   Real callers leave them None (use defaults). Tests pin them to
   avoid network + nondeterminism. Same pattern as grade_many.

WHAT'S NOT HERE
===============
- HTML rendering -- step 2.3.
- CLI command -- step 2.3.
- Comparison across runs ("did the v2 prompt beat v1?") -- not yet
  needed; YAGNI until we have multiple runs to compare.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from loguru import logger

from price_predictor.backtest.runner import BacktestRun
from price_predictor.prediction.calibration import (
    CalibrationReport,
    compute_breakdown,
    compute_calibration,
)
from price_predictor.prediction.grading import (
    FetchOHLCV,
    GradedPrediction,
    grade_many,
)
from price_predictor.prediction.schema import (
    PredictionDirection,
    PredictionHorizon,
)


# Month-key format. ISO YYYY-MM is sortable as a string + human readable
# in chart legends. Day-level grouping would be too sparse for stats;
# month-level shows seasonality without falling apart on small N.
_MONTH_KEY_FMT = "%Y-%m"


@dataclass(frozen=True)
class BacktestEvaluation:
    """A graded backtest with calibration breakdowns by useful axes.

    Frozen so once you've evaluated a run, the result is immutable
    evidence -- aligns with BacktestRun's contract.

    Attributes are READ-FRIENDLY for both programmatic use and HTML
    rendering. The HTML layer can iterate `by_horizon.items()` etc.
    without needing to know how the breakdown was computed.

    SHAPE NOTE
    ==========
    `graded` is a FLAT list (one entry per prediction) for drill-down
    views in the HTML report (the per-trade table). The breakdown
    dicts hold AGGREGATED CalibrationReports (one per group key).
    """
    # Source -- so a saved evaluation is fully self-describing.
    run: BacktestRun

    # Per-prediction outcomes (drill-down view).
    graded: list[GradedPrediction]

    # Aggregate metrics.
    overall: CalibrationReport
    by_horizon: dict[PredictionHorizon, CalibrationReport]
    by_ticker: dict[str, CalibrationReport]
    by_direction: dict[PredictionDirection, CalibrationReport]
    # YYYY-MM keys (string, sortable). Useful for time-series charts in
    # the HTML report ("did accuracy decay over Q2?").
    by_month: dict[str, CalibrationReport]

    @property
    def n_predictions(self) -> int:
        """Total predictions evaluated. Convenience for templates."""
        return len(self.graded)

    @property
    def n_judged(self) -> int:
        """Predictions with a measurable outcome (excludes INCONCLUSIVE)."""
        return self.overall.n_judged


def evaluate_backtest(
    run: BacktestRun,
    *,
    fetch_ohlcv: Optional[FetchOHLCV] = None,
    today: Optional[date] = None,
) -> BacktestEvaluation:
    """Grade every prediction in `run` and compute calibration breakdowns.

    Args:
        run: A completed BacktestRun (from run_backtest()).
        fetch_ohlcv: Optional injection for grading. Defaults to the
            production fetcher (yfinance + provider chain). Tests pin
            this to a fake to avoid network.
        today: Optional 'as-of-now' date for grading. Defaults to
            date.today(). Predictions whose horizons haven't elapsed
            get whatever bars exist (often INCONCLUSIVE). Tests use
            this to freeze time for determinism.

    Returns:
        BacktestEvaluation with overall + per-horizon/ticker/direction/
        month breakdowns. Safe to render directly into HTML.

    Raises:
        ValueError: run.predictions is empty -- nothing to evaluate.
            Caller-bug path; an empty backtest produces an empty
            evaluation, which is almost certainly a typo upstream.

    PERFORMANCE
    ===========
    grade_many is the dominant cost (one OHLCV fetch per prediction).
    With provider-chain caching, a 100-prediction evaluation typically
    takes 5-30s on a warm cache, longer cold. Calibration math is
    microseconds.
    """
    if not run.predictions:
        raise ValueError(
            "evaluate_backtest: run has no predictions to grade -- "
            "did the backtest fail entirely? Check run.errors."
        )

    logger.info(
        f"evaluate_backtest: grading {len(run.predictions)} predictions "
        f"({len(run.tickers)} ticker(s), {len(run.horizons)} horizon(s))"
    )

    # ── Step 1: GRADE every prediction (fetches OHLCV per prediction).
    # grade_many handles fetch failures internally (returns INCONCLUSIVE),
    # so we never lose position information in the output list.
    graded = grade_many(
        run.predictions,
        fetch_ohlcv=fetch_ohlcv,
        today=today,
    )

    # ── Step 2: OVERALL calibration -- the headline numbers.
    overall = compute_calibration(graded)

    # ── Step 3: BREAKDOWNS by axes the HTML report wants.
    # All call the same compute_breakdown helper -- no duplication of
    # grouping logic. Each lambda is the ONLY thing distinguishing them.
    by_horizon = compute_breakdown(graded, lambda g: g.prediction.horizon)
    by_ticker = compute_breakdown(graded, lambda g: g.prediction.ticker)
    by_direction = compute_breakdown(graded, lambda g: g.prediction.direction)
    by_month = compute_breakdown(
        graded,
        lambda g: g.prediction.as_of.strftime(_MONTH_KEY_FMT),
    )

    logger.info(
        f"evaluate_backtest: done -- overall hit_rate_resolved="
        f"{overall.hit_rate_resolved:.2%}, "
        f"direction_accuracy={overall.direction_accuracy:.2%}, "
        f"BSS={overall.brier_skill_score}"
    )

    return BacktestEvaluation(
        run=run,
        graded=graded,
        overall=overall,
        by_horizon=by_horizon,
        by_ticker=by_ticker,
        by_direction=by_direction,
        by_month=by_month,
    )
