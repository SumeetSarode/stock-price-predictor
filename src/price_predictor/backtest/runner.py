"""Backtest runner -- orchestrate predict() across (ticker x as_of) grid.

WHY THIS EXISTS
===============
Step 1.5 made `predict(as_of=X)` honest: technicals + news both
respect the historical date. But running it ONE call at a time
gets old fast when you want to backtest "RELIANCE.NS, every
trading day in 2024, all four horizons" -- that's ~245 trading
days x 4 horizons = ~980 predictions per ticker.

This module gives you ONE function call:

    run = await run_backtest(
        tickers=["RELIANCE.NS", "TCS.NS"],
        as_of_dates=trading_days_in_range(date(2024,1,1), date(2024,6,30)),
        horizons=[PredictionHorizon.DAILY, PredictionHorizon.WEEKLY],
    )

...which returns every prediction + every error in one structured
result, persists each prediction to disk as it completes (crash
resilient), and caps concurrency so we don't hammer the LLM.

DESIGN CHOICES
==============
1. ONE predict() call per (ticker, as_of) pair, NOT per horizon.
   `predict()` already accepts a list[PredictionHorizon] and returns
   them all in one shared gather phase (technicals + news fetched
   ONCE). Looping per-horizon would do redundant LLM calls for the
   same cluster signals -- cost goes up 4x for no signal gain.

2. Concurrency is over (ticker, as_of) PAIRS, not tickers.
   Otherwise a backtest of 1 ticker x 245 dates would run serially
   despite a concurrency=3 hint. We semaphore-cap the cartesian
   product fan-out instead.

3. Errors are PER-PAIR, not fatal.
   One bad as_of (e.g. holiday slipped through, snapshot fetch
   exploded) shouldn't kill the other 244 days. We capture
   exceptions into BacktestError objects and continue. Caller
   inspects run.errors to triage afterwards.

4. Persist EAGERLY (each save is atomic via PredictionStore).
   If the run crashes after 2 hours at prediction #800, you don't
   lose 799 predictions -- they're already on disk. Resume by
   diffing existing store contents against the requested grid.

5. Progress callback is OPTIONAL and fire-and-forget.
   The CLI/HTML wants a Rich progress bar; tests don't. Decoupling
   via a callback keeps the runner library-safe (no terminal deps).

NOT IN SCOPE FOR STEP 2.1
==========================
- Resume-after-crash (diff existing store -> skip already-done pairs).
  YAGNI for now; eager save makes manual resume trivial via re-run.
- Grading the predictions (Step 2.2 -- evaluation.py).
- HTML report (Step 2.3 -- backtest CLI command).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Awaitable, Callable, Literal

from loguru import logger

from price_predictor.prediction.predictor import predict
from price_predictor.prediction.schema import Prediction, PredictionHorizon
from price_predictor.prediction.store import (
    PredictionStore,
    PredictionStoreError,
)

Sensitivity = Literal["standard", "sensitive", "smooth"]


# ─────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class BacktestError:
    """One (ticker, as_of) pair failed.

    Frozen dataclass (not Pydantic) so it can carry the original
    exception object -- useful for callers who want to re-raise or
    pattern-match on type. Pydantic's strict typing would force us
    to stringify, losing the traceback.

    NOTE: `horizons` is plural because predict() is multi-horizon;
    a failure at the gather phase (technicals/news) takes ALL
    requested horizons down with it.
    """
    ticker: str
    as_of: date
    horizons: tuple[PredictionHorizon, ...]
    error: BaseException
    error_type: str  # convenience: type(error).__name__
    error_message: str  # convenience: str(error)

    @classmethod
    def from_exception(
        cls,
        ticker: str,
        as_of: date,
        horizons: list[PredictionHorizon],
        exc: BaseException,
    ) -> "BacktestError":
        return cls(
            ticker=ticker,
            as_of=as_of,
            horizons=tuple(horizons),
            error=exc,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


@dataclass(frozen=True)
class BacktestProgress:
    """Snapshot of the run's progress for callbacks.

    Sent to the progress_callback after EACH (ticker, as_of) pair
    completes (success or failure). Counts are cumulative.

    `current` is human-readable -- intended for log lines or progress
    bars, not for programmatic dispatch.
    """
    completed: int   # pairs finished (success OR failure)
    total: int       # total pairs in the run
    successes: int   # pairs that returned at least one prediction
    failures: int    # pairs in BacktestRun.errors
    current: str     # e.g. "RELIANCE.NS @ 2024-06-14 [2 horizons]"


@dataclass(frozen=True)
class BacktestRun:
    """The full structured result of a backtest.

    Frozen because a completed run is evidence -- mutating it after
    the fact would corrupt downstream calibration analysis. To slice
    or filter, build a derived object.

    `predictions` is FLAT (one entry per (ticker, as_of, horizon)),
    not nested -- downstream tools (grade_many, compute_calibration)
    take flat lists.
    """
    predictions: list[Prediction]
    errors: list[BacktestError]
    started_at: datetime
    finished_at: datetime
    # Echo of the configuration so a saved run is fully self-describing
    # (audit / reproducibility):
    tickers: tuple[str, ...]
    as_of_dates: tuple[date, ...]
    horizons: tuple[PredictionHorizon, ...]
    sensitivity: str
    concurrency: int

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def n_pairs_attempted(self) -> int:
        """Total (ticker, as_of) pairs the runner tried."""
        return len(self.tickers) * len(self.as_of_dates)

    @property
    def n_pairs_succeeded(self) -> int:
        """Pairs where at least one horizon's prediction completed."""
        # A successful pair contributes len(horizons) predictions.
        return self.n_pairs_attempted - len(self.errors)


# Type alias for the progress callback. Sync (not async) so callers
# can use plain print()/Rich without ceremony. Returning anything
# is ignored; raising propagates and aborts the run.
ProgressCallback = Callable[[BacktestProgress], None]


# ─────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────
async def run_backtest(
    tickers: list[str],
    as_of_dates: list[date],
    horizons: list[PredictionHorizon],
    *,
    sensitivity: Sensitivity = "standard",
    store: PredictionStore | None = None,
    concurrency: int = 3,
    progress_callback: ProgressCallback | None = None,
) -> BacktestRun:
    """Sweep predict() across (ticker x as_of) and collect everything.

    Args:
        tickers: Stock symbols to backtest. Duplicates removed,
            order-preserving (same convention as predict_many).
        as_of_dates: Historical dates to anchor predictions at.
            MUST be in the past -- predict() rejects future dates.
            Use trading_days_in_range() to build this.
        horizons: Which horizons to predict at each as_of. ALL of
            them are gathered in one predict() call per pair, so
            adding horizons is cheap (no extra technical fetches).
        sensitivity: Indicator preset, applied uniformly.
        store: Optional. If provided, every successful prediction
            is saved EAGERLY (atomic write). Crash resilience: a
            run that dies at pair #800 still has predictions 1-799
            on disk. If None, results live only in the returned
            BacktestRun.
        concurrency: Maximum number of in-flight predict() calls.
            Each call is ~30-60s (LLM-bound), so 3 is sane for
            free-tier Gemini. Bump for paid tiers.
        progress_callback: Optional sync callable invoked after
            each pair completes. Use for Rich progress bars or
            log lines. Exceptions in the callback ABORT the run
            (intentional -- a broken progress bar shouldn't be
            silently swallowed).

    Returns:
        BacktestRun with flat predictions list + per-pair errors +
        timing + config echo.

    Raises:
        ValueError: tickers/as_of_dates/horizons empty, concurrency<1.
            These are caller bugs (not data problems): an empty
            backtest is a no-op and probably means a typo, so we
            fail loud.
    """
    # ── Validation: caller-bug paths fail loud, data-problem paths
    #    show up as BacktestErrors (handled per-pair below).
    if not tickers:
        raise ValueError("run_backtest: tickers list is empty")
    if not as_of_dates:
        raise ValueError("run_backtest: as_of_dates list is empty")
    if not horizons:
        raise ValueError("run_backtest: horizons list is empty")
    if concurrency < 1:
        raise ValueError(
            f"run_backtest: concurrency must be >= 1 (got {concurrency})"
        )

    # ── Dedupe inputs. Preserves first-occurrence order via dict trick.
    unique_tickers = list(dict.fromkeys(tickers))
    unique_dates = list(dict.fromkeys(as_of_dates))
    unique_horizons = list(dict.fromkeys(horizons))
    if (len(unique_tickers) < len(tickers)
        or len(unique_dates) < len(as_of_dates)
        or len(unique_horizons) < len(horizons)):
        logger.info(
            f"run_backtest: deduplicated inputs "
            f"(tickers {len(tickers)}->{len(unique_tickers)}, "
            f"dates {len(as_of_dates)}->{len(unique_dates)}, "
            f"horizons {len(horizons)}->{len(unique_horizons)})"
        )

    # ── Build the work grid: cartesian (ticker x as_of). Order is
    #    ticker-major so callers can read logs as "all dates for
    #    ticker A, then all dates for ticker B" -- easier to grep.
    pairs: list[tuple[str, date]] = [
        (t, d) for t in unique_tickers for d in unique_dates
    ]
    total_pairs = len(pairs)
    started_at = datetime.now(timezone.utc)
    logger.info(
        f"run_backtest: starting -- {len(unique_tickers)} ticker(s) x "
        f"{len(unique_dates)} date(s) = {total_pairs} pair(s), "
        f"{len(unique_horizons)} horizon(s) per pair, "
        f"concurrency={concurrency}"
    )

    sem = asyncio.Semaphore(concurrency)
    # Atomic counters (mutable state shared across tasks); accessed
    # only inside the semaphore-held block + progress emission, so
    # contention is bounded.
    completed = 0
    successes = 0
    failures = 0
    predictions: list[Prediction] = []
    errors: list[BacktestError] = []

    async def _one_pair(ticker: str, as_of: date) -> None:
        """Run predict() for one (ticker, as_of), update shared state."""
        nonlocal completed, successes, failures

        async with sem:
            try:
                results = await predict(
                    ticker,
                    unique_horizons,
                    sensitivity=sensitivity,
                    as_of=as_of,
                )
            except BaseException as exc:  # noqa: BLE001 (intentional)
                # Capture, don't propagate -- one bad pair shouldn't
                # kill the run. KeyboardInterrupt etc. are still in
                # `errors` so the caller can detect + re-raise.
                err = BacktestError.from_exception(
                    ticker, as_of, unique_horizons, exc,
                )
                errors.append(err)
                failures += 1
                logger.warning(
                    f"backtest: {ticker} @ {as_of} FAILED "
                    f"({err.error_type}: {err.error_message})"
                )
            else:
                # results is dict[PredictionHorizon, Prediction] --
                # flatten and persist each.
                for horizon in unique_horizons:
                    pred = results[horizon]
                    predictions.append(pred)
                    if store is not None:
                        try:
                            store.save(pred)
                        except PredictionStoreError as e:
                            # Persistence failure is recoverable: the
                            # prediction is still in the in-memory
                            # list, just not on disk. Log + continue
                            # rather than upgrade to a hard failure.
                            logger.warning(
                                f"backtest: {ticker} @ {as_of} "
                                f"horizon={horizon.value}: store.save "
                                f"failed ({e}); kept in-memory only."
                            )
                successes += 1
                logger.debug(
                    f"backtest: {ticker} @ {as_of} OK "
                    f"({len(unique_horizons)} horizon(s))"
                )

            completed += 1
            if progress_callback is not None:
                progress_callback(BacktestProgress(
                    completed=completed,
                    total=total_pairs,
                    successes=successes,
                    failures=failures,
                    current=(
                        f"{ticker} @ {as_of} "
                        f"[{len(unique_horizons)} horizon(s)]"
                    ),
                ))

    # Fan out all pairs. asyncio.gather respects the semaphore;
    # we don't need return_exceptions because _one_pair captures
    # everything internally (so its return is always None).
    await asyncio.gather(*(_one_pair(t, d) for t, d in pairs))

    finished_at = datetime.now(timezone.utc)
    logger.info(
        f"run_backtest: done in {(finished_at - started_at).total_seconds():.1f}s "
        f"-- {successes} pair(s) succeeded, {failures} failed, "
        f"{len(predictions)} prediction(s) total"
    )

    return BacktestRun(
        predictions=predictions,
        errors=errors,
        started_at=started_at,
        finished_at=finished_at,
        tickers=tuple(unique_tickers),
        as_of_dates=tuple(unique_dates),
        horizons=tuple(unique_horizons),
        sensitivity=sensitivity,
        concurrency=concurrency,
    )
