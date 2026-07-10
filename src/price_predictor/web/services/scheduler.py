"""Background grading scheduler — closes the prediction feedback loop.

Why this exists
===============
Predictions are graded *lazily*: the outcome (HIT / STOPPED / EXPIRED)
is recomputed on-the-fly from the price cache every time someone opens
a stock's detail page. That's great for correctness (always fresh) but
it means:

  1. A PENDING prediction only *becomes* resolved when a human happens
     to reload the page after enough forward bars exist.
  2. The first page-load after a cold cache pays the full OHLCV-fetch
     latency.

This scheduler fixes both by running a periodic *grading pass*: it pulls
recent history, runs the same read-only `grade_rows()` engine (which
fetches + caches OHLCV as a side effect), and logs a scorecard. After a
pass, the cache is warm and every prediction reflects the latest bars —
so the loop closes on its own instead of waiting for a click.

Design
======
- ``run_grading_pass()`` is the *unit of work*: fully synchronous in
  intent, trivially testable, catches its own errors so one bad pass
  never kills the loop.
- ``grading_loop()`` is the *cadence*: sleep → pass → repeat, until
  cancelled. Nothing fancy (no cron) — an interval timer is honest and
  YAGNI-appropriate for a local-first single-user app.
- Wiring lives in ``app.py``'s lifespan, gated behind
  ``settings.enable_scheduler`` (off by default). Tests and
  ``create_app()`` stay side-effect-free.
"""
from __future__ import annotations

import asyncio

from loguru import logger

from price_predictor.web.services.grading_service import (
    Scorecard,
    build_scorecard,
    grade_rows,
)
from price_predictor.web.services.history_service import list_history


async def run_grading_pass(*, limit: int = 500) -> Scorecard:
    """Grade the most recent `limit` predictions across all tickers.

    Side effect: warms the shared OHLCV cache (grade_rows fetches bars).
    Returns the aggregate Scorecard so callers (and tests) can assert on
    the outcome. Never raises — a failed pass logs and returns an empty
    scorecard so the surrounding loop stays alive.
    """
    try:
        rows, total = list_history(limit=limit, offset=0)
    except Exception:  # pragma: no cover - defensive
        logger.opt(exception=True).error(
            "grading pass: failed to read history — skipping this pass"
        )
        return build_scorecard([])

    if not rows:
        logger.info("grading pass: no predictions in history yet — nothing to grade")
        return build_scorecard([])

    try:
        graded = await grade_rows(rows)
    except Exception:  # pragma: no cover - defensive
        logger.opt(exception=True).error(
            "grading pass: grade_rows() failed — skipping this pass"
        )
        return build_scorecard([])

    scorecard = build_scorecard(graded)
    logger.info(
        "grading pass complete: graded {}/{} rows — "
        "hits={} stops={} expired={} pending={} skipped={} "
        "hit_rate={} avg_r={}",
        len(graded), total,
        scorecard.hits, scorecard.stops, scorecard.expired,
        scorecard.pending, scorecard.skipped,
        scorecard.hit_rate, scorecard.avg_r,
    )
    return scorecard


async def grading_loop(
    *,
    interval_seconds: float,
    startup_delay_seconds: float = 0.0,
    limit: int = 500,
) -> None:
    """Run ``run_grading_pass`` forever, once per ``interval_seconds``.

    Cancellable: awaiting a cancellation during either the sleep or the
    pass exits cleanly (re-raises CancelledError so the task finishes).
    An unexpected error inside a pass is already swallowed by
    ``run_grading_pass``; this loop only has to survive cancellation.
    """
    logger.info(
        "grading scheduler: starting (interval={}s, startup_delay={}s, limit={})",
        interval_seconds, startup_delay_seconds, limit,
    )
    try:
        if startup_delay_seconds > 0:
            await asyncio.sleep(startup_delay_seconds)
        while True:
            await run_grading_pass(limit=limit)
            await asyncio.sleep(interval_seconds)
    except asyncio.CancelledError:
        logger.info("grading scheduler: cancelled — shutting down cleanly")
        raise
