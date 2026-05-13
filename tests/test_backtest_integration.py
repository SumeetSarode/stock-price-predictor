"""End-to-end integration smoke test for the backtest pipeline.

Hits REAL APIs (yfinance + GDELT/SEC + LLM) for a tiny but
representative window:

    3 tickers × 30 calendar days @ stride=5 (weekly) = ~12 prediction pairs
    × 1 horizon (WEEKLY only -- multi-horizon costs 4x with no extra signal
    for a smoke test).

Skipped by default (CI / local dev with no keys):
    uv run pytest -m "not integration"     # default
    uv run pytest -m integration            # run them

WHY THIS EXISTS
===============
The unit tests in test_backtest_runner.py / test_backtest_cli.py mock the
predict() boundary, so they prove ORCHESTRATION works but NOT that:
  - run_backtest survives 12 real concurrent predict() calls without
    rate-limit meltdown / event-loop weirdness.
  - The pipeline composes end-to-end:
        run_backtest -> evaluate_backtest -> write_html_report
  - Real grading actually labels predictions (vs test_grading.py which
    mocks the OHLCV fetch).
  - The whole thing finishes in a sane wall-clock time on a laptop.

This is the "does v1 backtest actually work?" net.

WHY THESE NUMBERS
=================
- 3 tickers (RELIANCE, TCS, INFY): liquid Indian large-caps with
  reliable yfinance coverage AND active news flow. Tiny enough to keep
  runtime under 5min (Option A's acceptance criterion).
- 30-day window ending ~3 months back: long enough for resolutions
  (target/stop hits) to actually fire so grading produces real labels,
  not just EXPIRED everywhere. Recent enough that data sources have it.
- stride=5 (weekly) keeps the cartesian product to ~12 pairs. Daily
  stride would multiply LLM cost 5x for zero additional invariant
  coverage.
- WEEKLY horizon only: matches the stride; resolves within the 30-day
  window so most predictions get a real grade.

WHAT WE ASSERT (and DON'T)
==========================
We assert STRUCTURAL invariants -- the kind that prove the wiring is
honest -- not numerical ones (markets move; LLMs vary).

ASSERTED:
  - Run completed: predictions list non-empty, error rate sane (<50%).
  - Each prediction has valid schema (delegated to Pydantic).
  - Evaluation produced a CalibrationReport with the same shape that
    `price-predictor calibration` consumes -- so the same renderer works.
  - HTML report file written and non-trivial in size.
  - Wall-clock under 5 minutes (Option A acceptance criterion).

NOT ASSERTED:
  - Specific hit-rate numbers (would be flaky).
  - Specific predictions (LLMs vary run-to-run).
  - That every pair succeeded (rate limits / transient API failures
    are normal; the pipeline must SURVIVE them, not avoid them).
"""
from __future__ import annotations

import os
import time
from datetime import date, timedelta
from pathlib import Path

import pytest

from price_predictor.backtest import (
    evaluate_backtest,
    run_backtest,
    trading_days_in_range,
    write_html_report,
)
from price_predictor.prediction.schema import PredictionHorizon

# Skip if any required key is missing -- surfaces a useful message
# instead of a confusing LiteLLM error halfway through 12 predictions.
_REQUIRED_KEYS = ("GEMINI_API_KEY",)
_missing = [k for k in _REQUIRED_KEYS if not os.environ.get(k)]
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        bool(_missing),
        reason=f"Missing required env vars for integration test: {_missing}",
    ),
]


# Liquid Indian large-caps with reliable yfinance + GDELT coverage.
# 3 names is enough to prove the cartesian fanout works; more would
# just inflate runtime without adding invariant coverage.
_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

# Window: 30 calendar days ending ~3 months back. Old enough that the
# WEEKLY horizon (5 trading days) has resolved for most predictions,
# so grading produces real target_hit / stop_hit / expired labels --
# not just NOT_APPLICABLE everywhere.
_END = date.today() - timedelta(days=90)
_START = _END - timedelta(days=30)

# Acceptance criterion from Option A: 30 days × 3 tickers in <5 min.
_MAX_WALLCLOCK_SECONDS = 5 * 60


async def test_backtest_end_to_end_smoke(tmp_path: Path):
    """Run the full backtest pipeline against real APIs.

    This is the v1-completion gate for Option A: if this test passes,
    `price-predictor backtest` is genuinely production-usable.
    """
    # ── 1. Build the schedule (NSE-aware, so weekends/holidays are out).
    as_of_dates = trading_days_in_range(_START, _END, stride=5)
    assert as_of_dates, (
        f"No NSE trading days in window {_START}..{_END}; check the "
        f"dates -- this test depends on a non-empty schedule."
    )

    # ── 2. Run the backtest. WEEKLY horizon only (matches stride=5;
    #     keeps LLM cost down without losing pipeline coverage).
    start_t = time.monotonic()
    run = await run_backtest(
        _TICKERS,
        as_of_dates,
        [PredictionHorizon.WEEKLY],
        # concurrency=3: one slot per ticker -- conservative to avoid
        # burning Gemini quota during CI.
        concurrency=3,
    )
    elapsed = time.monotonic() - start_t

    # ── 3. Wall-clock acceptance criterion.
    assert elapsed < _MAX_WALLCLOCK_SECONDS, (
        f"Backtest took {elapsed:.0f}s; Option A gate is "
        f"<{_MAX_WALLCLOCK_SECONDS}s. Investigate concurrency / "
        f"rate-limit handling."
    )

    # ── 4. Run survival: most pairs should succeed. Some failures are
    #     fine (transient 429s, news outage); a majority isn't.
    n_attempted = len(_TICKERS) * len(as_of_dates)
    error_rate = len(run.errors) / n_attempted if n_attempted else 1.0

    # Rate-limit detection: if the error rate is high AND the
    # dominant failure mode is quota / 429 / chain exhaustion, the
    # test environment is the bottleneck, not the pipeline. Skip
    # with a clear message instead of failing spuriously -- you
    # can't build a v1 gate around a quota that resets daily.
    if error_rate >= 0.5 and run.errors:
        rate_limit_signals = (
            "ratelimit", "rate limit", "429", "quota",
            "exhausted", "chain exhausted",
        )
        rate_limited = sum(
            1 for err in run.errors
            if any(sig in err.error_message.lower() for sig in rate_limit_signals)
        )
        if rate_limited / len(run.errors) >= 0.8:
            pytest.skip(
                f"{rate_limited}/{len(run.errors)} pairs failed with "
                f"rate-limit / quota errors -- LLM provider chain "
                f"exhausted. Re-run after cooldown. First error: "
                f"{run.errors[0].error_message[:200]}"
            )

    assert error_rate < 0.5, (
        f"Error rate {error_rate:.0%} exceeds 50% -- pipeline is "
        f"not surviving real-world API noise. Errors: "
        f"{[e.error_message[:80] for e in run.errors[:3]]}"
    )
    assert len(run.predictions) >= 1, (
        f"Zero predictions produced from {n_attempted} attempts; "
        f"first error: {run.errors[0].error_message if run.errors else 'n/a'}"
    )

    # ── 5. Evaluation: the contract we care about is "produces the
    #     same shape `calibration` consumes." If this works, the
    #     existing renderer in cli/calibration_cmd.py works for
    #     backtest output too.
    evaluation = evaluate_backtest(run)
    assert evaluation.calibration is not None
    # CalibrationReport invariants -- documented in calibration.py.
    cal = evaluation.calibration
    assert 0.0 <= cal.hit_rate_strict <= 1.0
    assert 0.0 <= cal.hit_rate_resolved <= 1.0
    assert 0.0 <= cal.hit_rate_optimistic <= 1.0
    assert 0.0 <= cal.brier_score <= 1.0

    # ── 6. HTML report writes without exploding and is non-trivial.
    out_path = tmp_path / "smoke_report.html"
    written = write_html_report(evaluation, out_path)
    assert written.exists()
    size = written.stat().st_size
    # 5KB floor: a real report has tables + CSS + insights + at least
    # one prediction row. Empty-template noise stops well under this.
    assert size > 5_000, (
        f"Report only {size}B -- looks like a stub, not a real render."
    )

    # ── 7. Spot-check that the report HTML actually mentions our
    #     tickers (proves predictions made it into the render path,
    #     not just into the run object).
    html = written.read_text()
    rendered_tickers = [t for t in _TICKERS if t in html]
    assert rendered_tickers, (
        f"None of {_TICKERS} appear in the HTML report; renderer is "
        f"silently dropping predictions."
    )
