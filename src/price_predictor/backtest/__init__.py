"""Backtest harness -- run predict() across historical date grids.

PUBLIC API
==========
Date sampling:
    trading_days_in_range(start, end, *, stride)

Runner:
    run_backtest(tickers, as_of_dates, horizons, **opts) -> BacktestRun
    BacktestRun         -- structured result (predictions + errors + metadata)
    BacktestError       -- one (ticker, as_of) pair that failed
    BacktestProgress    -- progress snapshot for callbacks

Evaluation (grading + calibration):
    evaluate_backtest(run, **opts) -> BacktestEvaluation
    BacktestEvaluation  -- graded run + overall + per-axis breakdowns

Report (HTML + insights):
    render_html_report(eval) -> str
    write_html_report(eval, path) -> Path
    generate_insights(eval) -> list[Insight]

USAGE
=====
Basic backtest of one ticker over Q1 2024, daily horizon, every
trading day:

    from datetime import date
    from price_predictor.backtest import (
        run_backtest, trading_days_in_range,
    )
    from price_predictor.prediction import PredictionHorizon, PredictionStore

    dates = trading_days_in_range(date(2024,1,1), date(2024,3,31))
    store = PredictionStore("./backtest-runs/q1-2024")
    run = await run_backtest(
        ["RELIANCE.NS"],
        dates,
        [PredictionHorizon.DAILY],
        store=store,
    )
    print(f"{run.n_pairs_succeeded}/{run.n_pairs_attempted} succeeded")

WHAT THIS LAYER OWNS
====================
- Iteration over (ticker x as_of) grids.
- Concurrency capping (semaphore).
- Eager persistence to PredictionStore (crash resilience).
- Per-pair error capture (one bad day doesn't kill the run).

WHAT IT EXPLICITLY DOESN'T DO
=============================
- HTML/CLI rendering -- see Step 2.3 (CLI command + report).
- Resume-after-crash -- YAGNI; eager save makes manual resume trivial.
"""
from price_predictor.backtest.dates import trading_days_in_range
from price_predictor.backtest.evaluation import (
    BacktestEvaluation,
    evaluate_backtest,
)
from price_predictor.backtest.html_report import (
    render_html_report,
    write_html_report,
)
from price_predictor.backtest.insights import (
    Insight,
    InsightLevel,
    generate_insights,
)
from price_predictor.backtest.runner import (
    BacktestError,
    BacktestProgress,
    BacktestRun,
    ProgressCallback,
    run_backtest,
    run_backtest_grid,
)

__all__ = [
    "BacktestError",
    "BacktestEvaluation",
    "BacktestProgress",
    "BacktestRun",
    "Insight",
    "InsightLevel",
    "ProgressCallback",
    "evaluate_backtest",
    "generate_insights",
    "render_html_report",
    "run_backtest",
    "run_backtest_grid",
    "trading_days_in_range",
    "write_html_report",
]
