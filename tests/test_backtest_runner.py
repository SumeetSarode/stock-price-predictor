"""Unit tests for backtest.runner -- predict() orchestration over a grid.

WHAT WE TEST
============
The contract that backtest runs depend on:
  - Exactly one predict() call per (ticker x as_of) pair.
  - All horizons returned in one call get flattened into predictions.
  - Per-pair errors are CAPTURED, not raised -- one bad day mustn't
    kill the run.
  - Concurrency cap is honored (semaphore math).
  - Eager save: each successful prediction hits the store immediately.
  - Progress callback fires once per pair, monotonically increasing.
  - Validation rejects empty inputs / bad concurrency (caller bugs).
  - Dedup: duplicates in tickers/dates/horizons are removed.

WHAT WE DON'T TEST (mocked out)
================================
- The actual predict() call -- it has its own 30+ test suite.
- Real LLM/network calls -- those are integration tests.
- PredictionStore disk format -- has its own tests.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from price_predictor.backtest import (
    BacktestError,
    BacktestProgress,
    BacktestRun,
    run_backtest,
)
from price_predictor.prediction.schema import (
    AnalysisBasis,
    Prediction,
    PredictionDirection,
    PredictionHorizon,
    PriceLevel,
)
from price_predictor.prediction.store import PredictionStore


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _sample_prediction(
    ticker: str = "RELIANCE.NS",
    as_of: date = date(2024, 6, 14),
    horizon: PredictionHorizon = PredictionHorizon.WEEKLY,
) -> Prediction:
    """Minimal valid Prediction for runner tests.

    We don't care about content -- just that it's well-formed enough
    to flow through PredictionStore.save() without ValidationError.
    """
    return Prediction(
        ticker=ticker,
        as_of=datetime(
            as_of.year, as_of.month, as_of.day, 15, 30,
            tzinfo=ZoneInfo("Asia/Kolkata"),
        ),
        horizon=horizon,
        model_chain=("news_impact:agentic_replay", "synthesizer:agentic"),
        direction=PredictionDirection.BULLISH,
        confidence=0.65,
        entry_zone=(199.0, 201.0),
        target=PriceLevel(value=210.0, rationale="x"*20),
        stop_loss=PriceLevel(value=195.0, rationale="x"*20),
        rationale="x" * 60,
        contributing_signals=("trend bullish",),
        conflicting_signals=(),
        analysis_basis=AnalysisBasis(
            close_price_at_prediction=200.0,
            bars_used=400,
            technical_summary="trend bullish, momentum bullish",
            news_sentiment_score=0.5,
            news_articles_considered=3,
            filings_considered=0,
        ),
    )


def _multi_horizon_result(
    ticker: str, as_of: date, horizons: list[PredictionHorizon],
) -> dict[PredictionHorizon, Prediction]:
    """Build the dict that predict() returns for a multi-horizon call."""
    return {h: _sample_prediction(ticker, as_of, h) for h in horizons}


def _run(coro):
    """asyncio.run shorthand -- pytest-asyncio felt like overkill for
    a few thin async tests in this file.
    """
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────
# Validation -- caller bugs fail loud
# ─────────────────────────────────────────────────────────────
class TestValidation:
    def test_empty_tickers_raises(self):
        with pytest.raises(ValueError, match="tickers list is empty"):
            _run(run_backtest(
                [], [date(2024, 6, 14)], [PredictionHorizon.WEEKLY],
            ))

    def test_empty_dates_raises(self):
        with pytest.raises(ValueError, match="as_of_dates list is empty"):
            _run(run_backtest(
                ["RELIANCE.NS"], [], [PredictionHorizon.WEEKLY],
            ))

    def test_empty_horizons_raises(self):
        with pytest.raises(ValueError, match="horizons list is empty"):
            _run(run_backtest(
                ["RELIANCE.NS"], [date(2024, 6, 14)], [],
            ))

    def test_concurrency_zero_raises(self):
        with pytest.raises(ValueError, match="concurrency must be >= 1"):
            _run(run_backtest(
                ["RELIANCE.NS"],
                [date(2024, 6, 14)],
                [PredictionHorizon.WEEKLY],
                concurrency=0,
            ))


# ─────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────
class TestHappyPath:
    @patch("price_predictor.backtest.runner.predict", new_callable=AsyncMock)
    def test_one_predict_call_per_pair(self, mock_predict):
        """One ticker x 3 dates x 2 horizons -> 3 predict() calls
        (one per pair), 6 predictions (one per horizon per pair).

        WHY THIS MATTERS: if we instead called predict() per horizon,
        we'd do 6 calls, each re-fetching technicals + news. That's
        4x cost for zero signal gain.
        """
        horizons = [PredictionHorizon.DAILY, PredictionHorizon.WEEKLY]
        dates = [date(2024, 6, 12), date(2024, 6, 13), date(2024, 6, 14)]

        mock_predict.side_effect = lambda t, h, **kw: _multi_horizon_result(
            t, kw["as_of"], h,
        )

        run = _run(run_backtest(["RELIANCE.NS"], dates, horizons))

        assert mock_predict.await_count == 3  # one per (ticker, as_of) pair
        assert len(run.predictions) == 6  # 3 pairs x 2 horizons
        assert len(run.errors) == 0

    @patch("price_predictor.backtest.runner.predict", new_callable=AsyncMock)
    def test_passes_all_horizons_in_one_call(self, mock_predict):
        """Each predict() call must receive THE FULL horizon list,
        not one horizon at a time.
        """
        horizons = [PredictionHorizon.DAILY, PredictionHorizon.WEEKLY]
        mock_predict.side_effect = lambda t, h, **kw: _multi_horizon_result(
            t, kw["as_of"], h,
        )

        _run(run_backtest(
            ["RELIANCE.NS"], [date(2024, 6, 14)], horizons,
        ))

        # The horizons argument (positional or kwarg) must contain all.
        call_args = mock_predict.call_args
        passed_horizons = call_args.args[1]
        assert list(passed_horizons) == horizons

    @patch("price_predictor.backtest.runner.predict", new_callable=AsyncMock)
    def test_passes_as_of_to_predict(self, mock_predict):
        """as_of MUST flow through to predict() -- the whole reason
        run_backtest exists is point-in-time honesty.
        """
        mock_predict.side_effect = lambda t, h, **kw: _multi_horizon_result(
            t, kw["as_of"], h,
        )
        target = date(2024, 6, 14)

        _run(run_backtest(
            ["RELIANCE.NS"], [target], [PredictionHorizon.WEEKLY],
        ))

        assert mock_predict.call_args.kwargs["as_of"] == target

    @patch("price_predictor.backtest.runner.predict", new_callable=AsyncMock)
    def test_records_run_metadata(self, mock_predict):
        """BacktestRun must echo the inputs so saved runs are
        self-describing for audit.
        """
        mock_predict.side_effect = lambda t, h, **kw: _multi_horizon_result(
            t, kw["as_of"], h,
        )

        run = _run(run_backtest(
            ["RELIANCE.NS", "TCS.NS"],
            [date(2024, 6, 12), date(2024, 6, 14)],
            [PredictionHorizon.WEEKLY],
            sensitivity="sensitive",
            concurrency=5,
        ))

        assert run.tickers == ("RELIANCE.NS", "TCS.NS")
        assert run.as_of_dates == (date(2024, 6, 12), date(2024, 6, 14))
        assert run.horizons == (PredictionHorizon.WEEKLY,)
        assert run.sensitivity == "sensitive"
        assert run.concurrency == 5
        assert run.n_pairs_attempted == 4  # 2 x 2
        assert run.n_pairs_succeeded == 4
        assert run.duration_seconds >= 0
        assert run.started_at <= run.finished_at


# ─────────────────────────────────────────────────────────────
# Error handling -- one bad pair shouldn't kill the run
# ─────────────────────────────────────────────────────────────
class TestErrorHandling:
    @patch("price_predictor.backtest.runner.predict", new_callable=AsyncMock)
    def test_single_failure_captured_not_raised(self, mock_predict):
        """One failing as_of MUST NOT cancel the other 2."""
        good_dates = [date(2024, 6, 12), date(2024, 6, 14)]
        bad_date = date(2024, 6, 13)

        async def _side(t, h, **kw):
            if kw["as_of"] == bad_date:
                raise RuntimeError("simulated GDELT outage")
            return _multi_horizon_result(t, kw["as_of"], h)

        mock_predict.side_effect = _side

        run = _run(run_backtest(
            ["RELIANCE.NS"],
            good_dates + [bad_date],
            [PredictionHorizon.WEEKLY],
        ))

        # 2 successful pairs (1 prediction each) + 1 failed pair.
        assert len(run.predictions) == 2
        assert len(run.errors) == 1
        err = run.errors[0]
        assert err.ticker == "RELIANCE.NS"
        assert err.as_of == bad_date
        assert err.error_type == "RuntimeError"
        assert "GDELT outage" in err.error_message

    @patch("price_predictor.backtest.runner.predict", new_callable=AsyncMock)
    def test_all_failures_returns_empty_predictions(self, mock_predict):
        """Worst case: every pair fails. Should return cleanly with
        empty predictions + populated errors, NOT raise.
        """
        mock_predict.side_effect = RuntimeError("everything is broken")

        run = _run(run_backtest(
            ["RELIANCE.NS", "TCS.NS"],
            [date(2024, 6, 14)],
            [PredictionHorizon.WEEKLY],
        ))

        assert run.predictions == []
        assert len(run.errors) == 2
        assert run.n_pairs_succeeded == 0


# ─────────────────────────────────────────────────────────────
# Dedup -- duplicates in inputs are removed
# ─────────────────────────────────────────────────────────────
class TestDedup:
    @patch("price_predictor.backtest.runner.predict", new_callable=AsyncMock)
    def test_duplicate_tickers_run_once(self, mock_predict):
        """Calling with the same ticker twice MUST NOT predict twice
        -- otherwise costs (LLM tokens) double for nothing.
        """
        mock_predict.side_effect = lambda t, h, **kw: _multi_horizon_result(
            t, kw["as_of"], h,
        )

        run = _run(run_backtest(
            ["RELIANCE.NS", "RELIANCE.NS", "TCS.NS"],
            [date(2024, 6, 14)],
            [PredictionHorizon.WEEKLY],
        ))

        assert mock_predict.await_count == 2  # not 3
        assert run.tickers == ("RELIANCE.NS", "TCS.NS")  # order preserved

    @patch("price_predictor.backtest.runner.predict", new_callable=AsyncMock)
    def test_duplicate_dates_run_once(self, mock_predict):
        mock_predict.side_effect = lambda t, h, **kw: _multi_horizon_result(
            t, kw["as_of"], h,
        )
        target = date(2024, 6, 14)

        run = _run(run_backtest(
            ["RELIANCE.NS"],
            [target, target, target],
            [PredictionHorizon.WEEKLY],
        ))

        assert mock_predict.await_count == 1
        assert run.as_of_dates == (target,)


# ─────────────────────────────────────────────────────────────
# Persistence -- eager save for crash resilience
# ─────────────────────────────────────────────────────────────
class TestPersistence:
    @patch("price_predictor.backtest.runner.predict", new_callable=AsyncMock)
    def test_eager_save_to_store(self, mock_predict, tmp_path: Path):
        """Each successful prediction must be persisted IMMEDIATELY,
        not at the end. This is what makes a 2-hour run resilient
        to crash at minute 90.
        """
        store = PredictionStore(tmp_path / "runs")
        mock_predict.side_effect = lambda t, h, **kw: _multi_horizon_result(
            t, kw["as_of"], h,
        )

        run = _run(run_backtest(
            ["RELIANCE.NS"],
            [date(2024, 6, 12), date(2024, 6, 14)],
            [PredictionHorizon.DAILY, PredictionHorizon.WEEKLY],
            store=store,
        ))

        # 2 dates x 2 horizons = 4 predictions on disk.
        assert store.count() == 4
        # And in-memory result matches.
        assert len(run.predictions) == 4

    @patch("price_predictor.backtest.runner.predict", new_callable=AsyncMock)
    def test_no_store_means_no_disk_writes(self, mock_predict, tmp_path: Path):
        """If store=None (default), nothing should hit disk. Live-mode
        ad-hoc backtests shouldn't litter the filesystem.
        """
        mock_predict.side_effect = lambda t, h, **kw: _multi_horizon_result(
            t, kw["as_of"], h,
        )

        # Build a watch-store but DON'T pass it.
        store = PredictionStore(tmp_path / "runs")
        run = _run(run_backtest(
            ["RELIANCE.NS"],
            [date(2024, 6, 14)],
            [PredictionHorizon.WEEKLY],
            store=None,
        ))

        assert store.count() == 0  # nothing was written
        assert len(run.predictions) == 1  # but in-memory is fine

    @patch("price_predictor.backtest.runner.predict", new_callable=AsyncMock)
    def test_store_save_failure_doesnt_lose_prediction(
        self, mock_predict, tmp_path: Path,
    ):
        """If store.save() raises, the prediction stays in-memory.
        Persistence is best-effort; losing the value entirely would
        be worse than just losing the on-disk copy.
        """
        bad_store = MagicMock(spec=PredictionStore)
        from price_predictor.prediction.store import PredictionStoreError
        bad_store.save.side_effect = PredictionStoreError("disk full")

        mock_predict.side_effect = lambda t, h, **kw: _multi_horizon_result(
            t, kw["as_of"], h,
        )

        run = _run(run_backtest(
            ["RELIANCE.NS"],
            [date(2024, 6, 14)],
            [PredictionHorizon.WEEKLY],
            store=bad_store,
        ))

        # Save was attempted...
        bad_store.save.assert_called_once()
        # ...but the prediction is still in the result list.
        assert len(run.predictions) == 1
        assert len(run.errors) == 0  # save failure isn't a "pair failure"


# ─────────────────────────────────────────────────────────────
# Progress callback
# ─────────────────────────────────────────────────────────────
class TestProgress:
    @patch("price_predictor.backtest.runner.predict", new_callable=AsyncMock)
    def test_callback_invoked_once_per_pair(self, mock_predict):
        mock_predict.side_effect = lambda t, h, **kw: _multi_horizon_result(
            t, kw["as_of"], h,
        )
        events: list[BacktestProgress] = []

        _run(run_backtest(
            ["RELIANCE.NS"],
            [date(2024, 6, 12), date(2024, 6, 13), date(2024, 6, 14)],
            [PredictionHorizon.WEEKLY],
            progress_callback=events.append,
        ))

        assert len(events) == 3
        # `completed` increases monotonically.
        completed = [e.completed for e in events]
        assert completed == sorted(completed)
        # Final event reports the total.
        assert events[-1].completed == 3
        assert events[-1].total == 3
        assert events[-1].successes == 3

    @patch("price_predictor.backtest.runner.predict", new_callable=AsyncMock)
    def test_callback_sees_failure_count(self, mock_predict):
        async def _flaky(t, h, **kw):
            if kw["as_of"].day == 13:
                raise ValueError("boom")
            return _multi_horizon_result(t, kw["as_of"], h)

        mock_predict.side_effect = _flaky
        events: list[BacktestProgress] = []

        _run(run_backtest(
            ["RELIANCE.NS"],
            [date(2024, 6, 12), date(2024, 6, 13), date(2024, 6, 14)],
            [PredictionHorizon.WEEKLY],
            progress_callback=events.append,
        ))

        # Final event must reflect 1 failure / 2 successes.
        assert events[-1].successes == 2
        assert events[-1].failures == 1


# ─────────────────────────────────────────────────────────────
# Concurrency cap
# ─────────────────────────────────────────────────────────────
class TestConcurrency:
    @patch("price_predictor.backtest.runner.predict", new_callable=AsyncMock)
    def test_semaphore_caps_in_flight_calls(self, mock_predict):
        """At any moment, no more than `concurrency` predict() calls
        should be in flight. Verified with a peak-concurrency counter
        that the mock increments on entry, decrements on exit.
        """
        in_flight = 0
        peak = 0
        lock = asyncio.Lock()

        async def _side(t, h, **kw):
            nonlocal in_flight, peak
            async with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            try:
                # Yield so other tasks have a chance to enter.
                await asyncio.sleep(0.01)
                return _multi_horizon_result(t, kw["as_of"], h)
            finally:
                async with lock:
                    in_flight -= 1

        mock_predict.side_effect = _side

        _run(run_backtest(
            ["RELIANCE.NS"],
            [date(2024, 6, 1) for _ in range(10)],  # dedup -> 1
            [PredictionHorizon.WEEKLY],
            concurrency=2,
        ))

        # After dedup we only have 1 pair; not a concurrency test.
        # Use distinct dates so we get real fanout.
        in_flight = peak = 0
        from price_predictor.backtest.dates import trading_days_in_range
        dates = trading_days_in_range(date(2024, 6, 1), date(2024, 6, 30))
        assert len(dates) >= 5  # sanity

        _run(run_backtest(
            ["RELIANCE.NS"],
            dates,
            [PredictionHorizon.WEEKLY],
            concurrency=2,
        ))

        assert peak <= 2, f"concurrency cap violated: peak={peak}"
        assert peak >= 1  # at least one call ran (sanity)


# ─────────────────────────────────────────────────────────────
# run_backtest_grid -- explicit (ticker, as_of) pairs
#
# These tests exercise the API used by survivorship-aware index
# backtests where the ticker set varies per date. The thin wrapper
# (`run_backtest`) is exercised by the suite above; here we only
# test the new explicit-pairs surface.
# ─────────────────────────────────────────────────────────────
from price_predictor.backtest import run_backtest_grid  # noqa: E402


class TestGridValidation:
    def test_empty_pairs_raises(self):
        with pytest.raises(ValueError, match="pairs list is empty"):
            _run(run_backtest_grid([], [PredictionHorizon.WEEKLY]))

    def test_empty_horizons_raises(self):
        with pytest.raises(ValueError, match="horizons list is empty"):
            _run(run_backtest_grid(
                [("RELIANCE.NS", date(2024, 6, 14))], [],
            ))

    def test_concurrency_zero_raises(self):
        with pytest.raises(ValueError, match="concurrency must be >= 1"):
            _run(run_backtest_grid(
                [("RELIANCE.NS", date(2024, 6, 14))],
                [PredictionHorizon.WEEKLY],
                concurrency=0,
            ))


class TestGridSparseShape:
    """The whole point of grid mode: NOT cartesian."""

    @patch("price_predictor.backtest.runner.predict", new_callable=AsyncMock)
    def test_sparse_pairs_only_those_pairs_called(self, mock_predict):
        """Pairs [(A, d1), (B, d2)] -- A is NOT predicted on d2 and
        B is NOT predicted on d1. This is what survivorship-aware
        index backtests need: a ticker only runs on dates it was
        actually a member.
        """
        mock_predict.side_effect = lambda t, h, **kw: _multi_horizon_result(
            t, kw["as_of"], h,
        )
        d1, d2 = date(2024, 6, 13), date(2024, 6, 14)
        pairs = [("A.NS", d1), ("B.NS", d2)]

        run = _run(run_backtest_grid(pairs, [PredictionHorizon.WEEKLY]))

        assert mock_predict.await_count == 2
        called = {
            (c.args[0], c.kwargs["as_of"])
            for c in mock_predict.await_args_list
        }
        assert called == {("A.NS", d1), ("B.NS", d2)}
        # Crucially, A was NOT called on d2 and B was NOT called on d1.
        assert ("A.NS", d2) not in called
        assert ("B.NS", d1) not in called

    @patch("price_predictor.backtest.runner.predict", new_callable=AsyncMock)
    def test_run_metadata_carries_unions_not_cartesian(self, mock_predict):
        """BacktestRun.tickers / .as_of_dates echo the UNION of what
        appeared in pairs -- the cartesian shape is gone (which is
        correct for sparse runs)."""
        mock_predict.side_effect = lambda t, h, **kw: _multi_horizon_result(
            t, kw["as_of"], h,
        )
        d1 = date(2024, 6, 12)
        d2 = date(2024, 6, 13)
        d3 = date(2024, 6, 14)
        # A on d1, d2; B on d2, d3.  Union: {A, B} x {d1, d2, d3}
        # would be 6 cartesian pairs, but we only run 4 here.
        pairs = [
            ("A.NS", d1), ("A.NS", d2),
            ("B.NS", d2), ("B.NS", d3),
        ]

        run = _run(run_backtest_grid(pairs, [PredictionHorizon.WEEKLY]))

        assert set(run.tickers) == {"A.NS", "B.NS"}
        assert set(run.as_of_dates) == {d1, d2, d3}
        # The legacy n_pairs_attempted property uses
        # len(tickers) * len(dates) -- on a sparse grid that
        # OVER-counts vs reality. Sparse callers should use
        # len(run.predictions) / len(horizons) instead.
        assert len(run.predictions) == 4  # 4 pairs x 1 horizon

    @patch("price_predictor.backtest.runner.predict", new_callable=AsyncMock)
    def test_dedup_pairs(self, mock_predict):
        """Duplicate pairs collapse; predict() called once per unique pair."""
        mock_predict.side_effect = lambda t, h, **kw: _multi_horizon_result(
            t, kw["as_of"], h,
        )
        d = date(2024, 6, 14)
        pairs = [("A.NS", d), ("A.NS", d), ("B.NS", d)]

        run = _run(run_backtest_grid(pairs, [PredictionHorizon.WEEKLY]))

        assert mock_predict.await_count == 2
        assert len(run.predictions) == 2


class TestGridBackwardCompat:
    """`run_backtest` should still behave identically; it now just
    delegates to run_backtest_grid internally. Catch any regressions
    where the wrapper drifts from the underlying contract."""

    @patch("price_predictor.backtest.runner.predict", new_callable=AsyncMock)
    def test_wrapper_produces_same_predictions_as_explicit_grid(
        self, mock_predict,
    ):
        mock_predict.side_effect = lambda t, h, **kw: _multi_horizon_result(
            t, kw["as_of"], h,
        )
        tickers = ["A.NS", "B.NS"]
        dates = [date(2024, 6, 13), date(2024, 6, 14)]
        horizons = [PredictionHorizon.WEEKLY]

        run_a = _run(run_backtest(tickers, dates, horizons))

        # Reset mock; run the explicit-grid equivalent.
        mock_predict.reset_mock()
        explicit_pairs = [(t, d) for t in tickers for d in dates]
        run_b = _run(run_backtest_grid(explicit_pairs, horizons))

        def pair_set(r):
            return {(p.ticker, p.as_of.date()) for p in r.predictions}
        assert len(run_a.predictions) == len(run_b.predictions) == 4
        assert pair_set(run_a) == pair_set(run_b)
