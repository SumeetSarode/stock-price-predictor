"""Unit tests for backtest.evaluation -- grading + calibration wiring.

WHAT WE TEST
============
The contract that the HTML report depends on:
  - Grading is invoked for every prediction (delegated to grade_many).
  - Overall calibration matches what compute_calibration would produce.
  - Breakdowns are computed for ALL four axes (horizon, ticker,
    direction, month).
  - Empty run raises (caller-bug guard).
  - Injected fetcher + today flow through to grade_many.
  - Frozen result -- can't be mutated post-construction.

WHAT WE DON'T TEST (already covered elsewhere)
==============================================
- grade_many internals -- has its own ~20 tests.
- compute_calibration metric math -- has its own ~30 tests.
- compute_breakdown grouping logic -- tested via calibration tests.
- Real OHLCV fetching -- integration tests; we inject fakes.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from price_predictor.backtest.evaluation import (
    BacktestEvaluation,
    evaluate_backtest,
)
from price_predictor.backtest.runner import BacktestRun
from price_predictor.prediction.calibration import CalibrationReport
from price_predictor.prediction.schema import (
    AnalysisBasis,
    Prediction,
    PredictionDirection,
    PredictionHorizon,
    PriceLevel,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────
def _prediction(
    ticker: str = "RELIANCE.NS",
    as_of_d: date = date(2024, 6, 14),
    horizon: PredictionHorizon = PredictionHorizon.WEEKLY,
    direction: PredictionDirection = PredictionDirection.BULLISH,
    confidence: float = 0.7,
    close: float = 200.0,
) -> Prediction:
    """Minimal valid Prediction. Direction/levels are consistent so
    the synthesizer's invariants don't reject it.
    """
    if direction == PredictionDirection.BULLISH:
        target_v, stop_v = close * 1.05, close * 0.97
    elif direction == PredictionDirection.BEARISH:
        target_v, stop_v = close * 0.95, close * 1.03
    else:
        # NEUTRAL: target/stop still required by schema; values are
        # symmetric around close. Direction logic in grader handles this.
        target_v, stop_v = close * 1.02, close * 0.98

    return Prediction(
        ticker=ticker,
        as_of=datetime(
            as_of_d.year, as_of_d.month, as_of_d.day, 15, 30,
            tzinfo=ZoneInfo("Asia/Kolkata"),
        ),
        horizon=horizon,
        model_chain=("synthesizer:agentic",),
        direction=direction,
        confidence=confidence,
        entry_zone=(close * 0.995, close * 1.005),
        target=PriceLevel(value=target_v, rationale="x" * 20),
        stop_loss=PriceLevel(value=stop_v, rationale="x" * 20),
        rationale="x" * 60,
        contributing_signals=("trend",),
        conflicting_signals=(),
        analysis_basis=AnalysisBasis(
            close_price_at_prediction=close,
            bars_used=400,
            technical_summary="trend bullish, momentum bullish",
            news_sentiment_score=0.5,
            news_articles_considered=3,
            filings_considered=0,
        ),
    )


def _empty_run() -> BacktestRun:
    """A BacktestRun with no predictions (e.g. all pairs failed)."""
    now = datetime.now(ZoneInfo("UTC"))
    return BacktestRun(
        predictions=[],
        errors=[],
        started_at=now,
        finished_at=now,
        tickers=("RELIANCE.NS",),
        as_of_dates=(date(2024, 6, 14),),
        horizons=(PredictionHorizon.WEEKLY,),
        sensitivity="standard",
        concurrency=3,
    )


def _run_with(predictions: list[Prediction]) -> BacktestRun:
    """Build a BacktestRun wrapping the supplied predictions.

    The metadata fields aren't critical for evaluation tests -- we
    just need a well-formed BacktestRun.
    """
    now = datetime.now(ZoneInfo("UTC"))
    tickers = tuple(dict.fromkeys(p.ticker for p in predictions))
    horizons = tuple(dict.fromkeys(p.horizon for p in predictions))
    as_ofs = tuple(dict.fromkeys(p.as_of.date() for p in predictions))
    return BacktestRun(
        predictions=predictions,
        errors=[],
        started_at=now,
        finished_at=now,
        tickers=tickers,
        as_of_dates=as_ofs,
        horizons=horizons,
        sensitivity="standard",
        concurrency=3,
    )


def _bullish_uptrend_bars(close_at_pred: float, n: int = 30) -> pd.DataFrame:
    """Bars that walk price UP past target (close * 1.05).

    Used for happy-path bullish grading: target should hit early, no
    stop touch.
    """
    closes = [close_at_pred * (1.0 + 0.01 * (i + 1)) for i in range(n)]
    return pd.DataFrame({
        "open":  closes,
        "high":  [c * 1.005 for c in closes],
        "low":   [c * 0.995 for c in closes],
        "close": closes,
        "volume": [1_000_000] * n,
    })


def _bearish_drop_bars(close_at_pred: float, n: int = 30) -> pd.DataFrame:
    """Bars that walk price DOWN past target (close * 0.95)."""
    closes = [close_at_pred * (1.0 - 0.01 * (i + 1)) for i in range(n)]
    return pd.DataFrame({
        "open":  closes,
        "high":  [c * 1.005 for c in closes],
        "low":   [c * 0.995 for c in closes],
        "close": closes,
        "volume": [1_000_000] * n,
    })


def _make_fetcher(default_bars: pd.DataFrame):
    """Return a fetch_ohlcv stub that returns the same bars for all
    (ticker, start, end) inputs. Sufficient for grading-orchestration
    tests where we don't care about per-ticker variation.
    """
    def _fetch(ticker: str, start: date, end: date) -> pd.DataFrame:
        return default_bars.copy()
    return _fetch


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────
class TestValidation:
    def test_empty_run_raises(self):
        """Caller-bug guard: an empty backtest is almost certainly a
        typo upstream. Fail loud rather than return a useless empty
        evaluation that the HTML layer would render as a blank page.
        """
        with pytest.raises(ValueError, match="no predictions to grade"):
            evaluate_backtest(_empty_run())


# ─────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────
class TestHappyPath:
    def test_returns_evaluation_with_all_axes(self):
        """The HTML report wants overall + 4 breakdowns. Verify they
        ALL exist (not None) on a well-formed result.
        """
        preds = [
            _prediction(ticker="RELIANCE.NS", horizon=PredictionHorizon.WEEKLY),
            _prediction(ticker="TCS.NS", horizon=PredictionHorizon.DAILY),
        ]
        run = _run_with(preds)
        bars = _bullish_uptrend_bars(close_at_pred=200.0)

        ev = evaluate_backtest(
            run,
            fetch_ohlcv=_make_fetcher(bars),
            today=date(2024, 8, 1),  # well past horizon windows
        )

        assert isinstance(ev, BacktestEvaluation)
        assert isinstance(ev.overall, CalibrationReport)
        assert ev.by_horizon is not None
        assert ev.by_ticker is not None
        assert ev.by_direction is not None
        assert ev.by_month is not None

    def test_grades_every_prediction(self):
        """No prediction silently dropped -- length of graded matches
        length of input. The HTML drill-down table needs this 1:1.
        """
        preds = [
            _prediction(as_of_d=date(2024, 6, d))
            for d in [10, 11, 12, 13, 14]
        ]
        run = _run_with(preds)
        bars = _bullish_uptrend_bars(close_at_pred=200.0)

        ev = evaluate_backtest(
            run,
            fetch_ohlcv=_make_fetcher(bars),
            today=date(2024, 8, 1),
        )

        assert len(ev.graded) == 5
        assert ev.n_predictions == 5

    def test_overall_matches_compute_calibration(self):
        """ev.overall MUST equal compute_calibration(ev.graded) -- it's
        the 'authoritative' headline number. If these drift, the HTML
        report would show contradictory numbers in different sections.
        """
        from price_predictor.prediction.calibration import compute_calibration

        preds = [_prediction() for _ in range(3)]
        run = _run_with(preds)
        bars = _bullish_uptrend_bars(close_at_pred=200.0)

        ev = evaluate_backtest(
            run,
            fetch_ohlcv=_make_fetcher(bars),
            today=date(2024, 8, 1),
        )

        recomputed = compute_calibration(ev.graded)
        assert ev.overall == recomputed


# ─────────────────────────────────────────────────────────────
# Breakdowns -- the slicing-and-dicing the HTML report needs
# ─────────────────────────────────────────────────────────────
class TestBreakdowns:
    def test_by_horizon_groups_correctly(self):
        """Two horizons in the run -> two keys in by_horizon."""
        preds = [
            _prediction(horizon=PredictionHorizon.DAILY),
            _prediction(horizon=PredictionHorizon.WEEKLY),
            _prediction(horizon=PredictionHorizon.WEEKLY),
        ]
        run = _run_with(preds)
        bars = _bullish_uptrend_bars(close_at_pred=200.0)

        ev = evaluate_backtest(
            run,
            fetch_ohlcv=_make_fetcher(bars),
            today=date(2024, 8, 1),
        )

        assert PredictionHorizon.DAILY in ev.by_horizon
        assert PredictionHorizon.WEEKLY in ev.by_horizon
        assert ev.by_horizon[PredictionHorizon.DAILY].n_predictions == 1
        assert ev.by_horizon[PredictionHorizon.WEEKLY].n_predictions == 2

    def test_by_ticker_groups_correctly(self):
        preds = [
            _prediction(ticker="RELIANCE.NS"),
            _prediction(ticker="RELIANCE.NS"),
            _prediction(ticker="TCS.NS"),
        ]
        run = _run_with(preds)
        bars = _bullish_uptrend_bars(close_at_pred=200.0)

        ev = evaluate_backtest(
            run,
            fetch_ohlcv=_make_fetcher(bars),
            today=date(2024, 8, 1),
        )

        assert ev.by_ticker["RELIANCE.NS"].n_predictions == 2
        assert ev.by_ticker["TCS.NS"].n_predictions == 1

    def test_by_direction_groups_correctly(self):
        preds = [
            _prediction(direction=PredictionDirection.BULLISH),
            _prediction(direction=PredictionDirection.BEARISH),
            _prediction(direction=PredictionDirection.BULLISH),
        ]
        run = _run_with(preds)
        bars = _bullish_uptrend_bars(close_at_pred=200.0)

        ev = evaluate_backtest(
            run,
            fetch_ohlcv=_make_fetcher(bars),
            today=date(2024, 8, 1),
        )

        assert ev.by_direction[PredictionDirection.BULLISH].n_predictions == 2
        assert ev.by_direction[PredictionDirection.BEARISH].n_predictions == 1

    def test_by_month_uses_iso_format(self):
        """Month keys MUST be YYYY-MM strings for sortability + chart
        legends. Locking the format prevents accidental drift.
        """
        preds = [
            _prediction(as_of_d=date(2024, 5, 14)),
            _prediction(as_of_d=date(2024, 6, 14)),
            _prediction(as_of_d=date(2024, 6, 28)),
        ]
        run = _run_with(preds)
        bars = _bullish_uptrend_bars(close_at_pred=200.0)

        ev = evaluate_backtest(
            run,
            fetch_ohlcv=_make_fetcher(bars),
            today=date(2024, 8, 1),
        )

        # ISO format, sortable as strings.
        assert "2024-05" in ev.by_month
        assert "2024-06" in ev.by_month
        assert ev.by_month["2024-05"].n_predictions == 1
        assert ev.by_month["2024-06"].n_predictions == 2
        assert sorted(ev.by_month.keys()) == ["2024-05", "2024-06"]


# ─────────────────────────────────────────────────────────────
# Hit-rate sanity -- confirms we're wired to grade_many properly
# ─────────────────────────────────────────────────────────────
class TestHitRateWiring:
    def test_all_bullish_uptrend_yields_high_hit_rate(self):
        """If every bullish prediction sees an uptrend that hits target,
        hit_rate_resolved should be ~100%. Sanity check that grading
        flows through to calibration.
        """
        preds = [
            _prediction(direction=PredictionDirection.BULLISH)
            for _ in range(5)
        ]
        run = _run_with(preds)
        bars = _bullish_uptrend_bars(close_at_pred=200.0)

        ev = evaluate_backtest(
            run,
            fetch_ohlcv=_make_fetcher(bars),
            today=date(2024, 8, 1),
        )

        # Bars walk up past 5% target by bar #5; weekly window is 5
        # bars; should ALL hit target.
        assert ev.overall.n_target_hit == 5
        assert ev.overall.hit_rate_resolved == pytest.approx(1.0)
        assert ev.overall.direction_accuracy == pytest.approx(1.0)

    def test_bullish_predictions_against_drop_yields_low_accuracy(self):
        """Bullish call + price drops = direction wrong, stop hits."""
        preds = [
            _prediction(direction=PredictionDirection.BULLISH)
            for _ in range(5)
        ]
        run = _run_with(preds)
        bars = _bearish_drop_bars(close_at_pred=200.0)

        ev = evaluate_backtest(
            run,
            fetch_ohlcv=_make_fetcher(bars),
            today=date(2024, 8, 1),
        )

        # Direction WRONG every time.
        assert ev.overall.direction_accuracy == pytest.approx(0.0)
        # Stops hit (price drops past close * 0.97).
        assert ev.overall.n_stop_hit + ev.overall.n_stop_hit_ambiguous == 5


# ─────────────────────────────────────────────────────────────
# Frozen-result invariant
# ─────────────────────────────────────────────────────────────
class TestFrozen:
    def test_evaluation_is_immutable(self):
        """BacktestEvaluation is frozen -- can't accidentally mutate
        a graded run between read and report-generation.
        """
        preds = [_prediction()]
        run = _run_with(preds)
        bars = _bullish_uptrend_bars(close_at_pred=200.0)

        ev = evaluate_backtest(
            run,
            fetch_ohlcv=_make_fetcher(bars),
            today=date(2024, 8, 1),
        )

        with pytest.raises((AttributeError, Exception)):
            ev.overall = None  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────
# Convenience properties
# ─────────────────────────────────────────────────────────────
class TestConvenienceProps:
    def test_n_judged_excludes_inconclusive(self):
        """today=as_of+0 means no bars elapsed -> all INCONCLUSIVE.
        n_judged should reflect that.
        """
        preds = [_prediction(as_of_d=date(2024, 6, 14))]
        run = _run_with(preds)

        # today == prediction date -> no future bars yet -> INCONCLUSIVE
        ev = evaluate_backtest(
            run,
            fetch_ohlcv=_make_fetcher(pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )),
            today=date(2024, 6, 14),
        )

        assert ev.n_predictions == 1
        assert ev.n_judged == 0
