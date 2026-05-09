"""CLI tests for price-predictor (Step 3.4.3 commit 3).

Uses typer's CliRunner to invoke commands without spawning subprocesses.
The actual predict() / predict_many() calls are mocked so tests stay
fast and don't require API keys.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from typer.testing import CliRunner

from price_predictor.cli.main import (
    _BREAKDOWN_KEYS,
    _render_batch,
    _render_calibration,
    _render_grades,
    _render_prediction,
    app,
)
from price_predictor.prediction import (
    BatchError,
    GradeOutcome,
    GradedPrediction,
    Prediction,
    PredictionError,
    PredictionStore,
    compute_calibration,
)
from price_predictor.prediction.schema import (
    AnalysisBasis,
    PredictionDirection,
    PredictionHorizon,
    PriceLevel,
)


# ─────────────────────────────────────────────────────────────
# Factory + fixtures
# ─────────────────────────────────────────────────────────────
def _make_pred(
    ticker: str = "RELIANCE.NS",
    direction: PredictionDirection = PredictionDirection.BULLISH,
) -> Prediction:
    is_bull = direction == PredictionDirection.BULLISH
    return Prediction(
        ticker=ticker,
        as_of=datetime(2026, 4, 28, 10, 30, 45, tzinfo=ZoneInfo("Asia/Kolkata")),
        horizon=PredictionHorizon.WEEKLY,
        model_chain=("news_impact:agentic", "synthesizer:agentic"),
        direction=direction,
        confidence=0.7,
        entry_zone=(1453.0, 1457.0),
        target=PriceLevel(
            value=1500.0 if is_bull else 1410.0,
            rationale="swing high" if is_bull else "swing low",
        ),
        stop_loss=PriceLevel(
            value=1425.0 if is_bull else 1480.0,
            rationale="below recent low" if is_bull else "above recent high",
        ),
        rationale="Multi-timeframe alignment.",
        contributing_signals=("trend bullish",),
        conflicting_signals=(),
        analysis_basis=AnalysisBasis(
            close_price_at_prediction=1455.0,
            bars_used=400,
            technical_summary="trend ok",
            news_sentiment_score=0.6,
            news_articles_considered=5,
            filings_considered=0,
        ),
    )


@pytest.fixture
def runner() -> CliRunner:
    # mix_stderr=False - keep stderr separate so we can assert error
    # messages independently from normal output.
    return CliRunner(mix_stderr=False)


@pytest.fixture
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point settings.predictions_dir at a tmp dir for the duration of test.

    predictions_dir is a @property derived from data_dir, so we patch
    data_dir on the singleton settings instance. This isolates tests
    from each other AND from the user's real ~/predictions folder.
    """
    monkeypatch.setattr(
        "price_predictor.cli.main.settings.data_dir",
        tmp_path,
    )
    # The property returns data_dir / 'predictions' - return that
    # so callers know exactly where files will land.
    return tmp_path / "predictions"


# ─────────────────────────────────────────────────────────────
# Renderer unit tests (pure functions, no I/O)
# ─────────────────────────────────────────────────────────────
class TestRenderers:
    def test_render_prediction_includes_key_fields(self):
        pred = _make_pred()
        table = _render_prediction(pred)
        # Table is a Rich Renderable - rendering to plain text gives
        # us something to grep without depending on Rich internals.
        from rich.console import Console
        from io import StringIO
        buf = StringIO()
        Console(file=buf, width=200, force_terminal=False).print(table)
        out = buf.getvalue()
        assert "RELIANCE.NS" in out
        assert "BULLISH" in out
        assert "70%" in out  # confidence
        assert "1500" in out  # target
        assert "1425" in out  # stop

    def test_render_batch_marks_failures(self):
        results = [_make_pred("A"), BatchError.from_exception("BAD", ValueError("boom"))]
        table = _render_batch(results, ["A", "BAD"])
        from rich.console import Console
        from io import StringIO
        buf = StringIO()
        Console(file=buf, width=200, force_terminal=False).print(table)
        out = buf.getvalue()
        assert "A" in out
        assert "BAD" in out
        assert "ValueError" in out  # error type shown


# ─────────────────────────────────────────────────────────────
# `predict` command
# ─────────────────────────────────────────────────────────────
class TestPredictCommand:
    @patch("price_predictor.cli.main._predict", new_callable=AsyncMock)
    def test_happy_path(self, mock_predict, runner):
        mock_predict.return_value = _make_pred()
        result = runner.invoke(app, ["predict", "RELIANCE.NS"])
        assert result.exit_code == 0, result.output
        assert "RELIANCE.NS" in result.output
        assert "BULLISH" in result.output
        # Default horizon = short
        mock_predict.assert_awaited_once()
        assert mock_predict.call_args.args == ("RELIANCE.NS", "weekly")

    @patch("price_predictor.cli.main._predict", new_callable=AsyncMock)
    def test_horizon_flag(self, mock_predict, runner):
        mock_predict.return_value = _make_pred()
        result = runner.invoke(app, ["predict", "AAPL", "--horizon", "biweekly"])
        assert result.exit_code == 0, result.output
        assert mock_predict.call_args.args == ("AAPL", "biweekly")

    @patch("price_predictor.cli.main._predict", new_callable=AsyncMock)
    def test_failure_exits_nonzero(self, mock_predict, runner):
        mock_predict.side_effect = PredictionError("everything's broken")
        result = runner.invoke(app, ["predict", "RELIANCE.NS"])
        assert result.exit_code == 1
        assert "everything's broken" in result.output

    @patch("price_predictor.cli.main._predict", new_callable=AsyncMock)
    def test_save_flag_writes_to_store(
        self, mock_predict, runner, isolated_store,
    ):
        mock_predict.return_value = _make_pred()
        result = runner.invoke(app, ["predict", "RELIANCE.NS", "--save"])
        assert result.exit_code == 0, result.output
        # Verify a file landed in the isolated store
        store = PredictionStore(isolated_store)
        assert store.count() == 1
        assert "Saved to" in result.output


# ─────────────────────────────────────────────────────────────
# `predict-many` command
# ─────────────────────────────────────────────────────────────
class TestPredictManyCommand:
    @patch("price_predictor.cli.main._predict_many", new_callable=AsyncMock)
    def test_happy_path(self, mock_pm, runner):
        mock_pm.return_value = [_make_pred("A"), _make_pred("B")]
        result = runner.invoke(app, ["predict-many", "A", "B"])
        assert result.exit_code == 0, result.output
        assert "A" in result.output and "B" in result.output

    @patch("price_predictor.cli.main._predict_many", new_callable=AsyncMock)
    def test_partial_failure_exits_nonzero(self, mock_pm, runner):
        """Non-zero exit lets cron/CI flag broken runs."""
        mock_pm.return_value = [
            _make_pred("A"),
            BatchError.from_exception("BAD", RuntimeError("x")),
        ]
        result = runner.invoke(app, ["predict-many", "A", "BAD"])
        assert result.exit_code == 1
        assert "RuntimeError" in result.output

    @patch("price_predictor.cli.main._predict_many", new_callable=AsyncMock)
    def test_save_flag_skips_failed(self, mock_pm, runner, isolated_store):
        mock_pm.return_value = [
            _make_pred("A"),
            BatchError.from_exception("BAD", RuntimeError("x")),
        ]
        runner.invoke(app, ["predict-many", "A", "BAD", "--save"])
        # Only 1 saved (the success), not 2
        assert PredictionStore(isolated_store).count() == 1


# ─────────────────────────────────────────────────────────────
# `history` command
# ─────────────────────────────────────────────────────────────
class TestHistoryCommand:
    def test_empty_history_prints_message(self, runner, isolated_store):
        result = runner.invoke(app, ["history", "RELIANCE.NS"])
        assert result.exit_code == 0
        assert "No stored predictions" in result.output

    def test_lists_stored_predictions(self, runner, isolated_store):
        store = PredictionStore(isolated_store)
        store.save(_make_pred())
        result = runner.invoke(app, ["history", "RELIANCE.NS"])
        assert result.exit_code == 0
        assert "RELIANCE.NS" in result.output
        assert "BULLISH" in result.output

    def test_limit_flag(self, runner, isolated_store):
        store = PredictionStore(isolated_store)
        ist = ZoneInfo("Asia/Kolkata")
        for day in range(20, 25):
            pred = _make_pred()
            # Mutate as_of via model_copy (frozen)
            pred = pred.model_copy(
                update={"as_of": datetime(2026, 4, day, 10, 0, 0, tzinfo=ist)}
            )
            store.save(pred)
        result = runner.invoke(app, ["history", "RELIANCE.NS", "--limit", "2"])
        assert result.exit_code == 0
        # Check the title says "5 predictions" total but only 2 in range
        # Easier: check that 2026-04-23 (3rd from end) is NOT in output
        # while 2026-04-24 (last) IS.
        assert "2026-04-24" in result.output
        assert "2026-04-22" not in result.output


# ─────────────────────────────────────────────────────────────
# Help / discoverability
# ─────────────────────────────────────────────────────────────
class TestHelp:
    def test_app_help_lists_all_commands(self, runner):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "predict" in result.output
        assert "predict-many" in result.output
        assert "history" in result.output

    def test_predict_help_shows_options(self, runner):
        result = runner.invoke(app, ["predict", "--help"])
        assert result.exit_code == 0
        assert "--horizon" in result.output
        assert "--save" in result.output


# ─────────────────────────────────────────────────────────────
# Step 3.5: grade + calibration commands
# ─────────────────────────────────────────────────────────────
def _make_graded(
    *,
    ticker: str = "RELIANCE.NS",
    outcome: GradeOutcome = GradeOutcome.TARGET_HIT,
    direction_correct: bool | None = True,
    realized_return: float = 0.05,
    confidence: float = 0.7,
    direction: PredictionDirection = PredictionDirection.BULLISH,
    horizon: PredictionHorizon = PredictionHorizon.WEEKLY,
) -> GradedPrediction:
    pred = Prediction(
        ticker=ticker,
        as_of=datetime(2026, 4, 28, 10, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
        horizon=horizon,
        model_chain=("synthesizer:test",),
        direction=direction,
        confidence=confidence,
        entry_zone=(99.0, 101.0),
        target=PriceLevel(value=110.0, rationale="t"),
        stop_loss=PriceLevel(value=95.0, rationale="s"),
        rationale="test",
        contributing_signals=("sig",),
        conflicting_signals=(),
        analysis_basis=AnalysisBasis(
            close_price_at_prediction=100.0,
            bars_used=400,
            technical_summary="ok",
        ),
    )
    return GradedPrediction(
        prediction=pred,
        outcome=outcome,
        realized_return=realized_return,
        direction_correct=direction_correct,
        days_to_resolution=2 if outcome == GradeOutcome.TARGET_HIT else None,
        bars_examined=5,
        close_at_window_end=105.0,
    )


class TestGradeRenderers:
    def test_render_grades_includes_outcome_and_return(self):
        from io import StringIO
        from rich.console import Console
        graded = [_make_graded()]
        buf = StringIO()
        Console(file=buf, width=200, force_terminal=False).print(_render_grades(graded))
        out = buf.getvalue()
        assert "RELIANCE.NS" in out
        assert "target_hit" in out
        assert "+5.00%" in out  # formatted realized return

    def test_render_calibration_includes_all_metrics(self):
        from io import StringIO
        from rich.console import Console
        report = compute_calibration([_make_graded(), _make_graded()])
        buf = StringIO()
        Console(file=buf, width=200, force_terminal=False).print(_render_calibration(report))
        out = buf.getvalue()
        # Spot-check that hit-rate variants AND brier all show up
        assert "strict" in out
        assert "resolved" in out
        assert "optimistic" in out
        assert "Brier" in out
        assert "Direction accuracy" in out


class TestGradeCommand:
    @patch("price_predictor.cli.main.grade_many")
    def test_no_predictions_prints_helpful_message(
        self, mock_grade, runner, isolated_store,
    ):
        result = runner.invoke(app, ["grade"])
        assert result.exit_code == 0
        assert "No predictions" in result.output
        # Should NOT have called grade_many - nothing to grade.
        mock_grade.assert_not_called()

    @patch("price_predictor.cli.main.grade_many")
    def test_grades_stored_predictions(
        self, mock_grade, runner, isolated_store,
    ):
        # Save a prediction; mock grade_many to return a known result.
        store = PredictionStore(isolated_store)
        store.save(_make_graded().prediction)
        mock_grade.return_value = [_make_graded()]

        result = runner.invoke(app, ["grade"])
        assert result.exit_code == 0, result.output
        assert "RELIANCE.NS" in result.output
        assert "target_hit" in result.output
        mock_grade.assert_called_once()

    @patch("price_predictor.cli.main.grade_many")
    def test_ticker_filter(self, mock_grade, runner, isolated_store):
        store = PredictionStore(isolated_store)
        # Save predictions for two different tickers
        store.save(_make_graded(ticker="AAA.NS").prediction)
        store.save(_make_graded(ticker="BBB.NS").prediction)
        mock_grade.return_value = [_make_graded(ticker="AAA.NS")]

        result = runner.invoke(app, ["grade", "--ticker", "AAA.NS"])
        assert result.exit_code == 0
        # Only one prediction should have been passed to grade_many
        passed_preds = mock_grade.call_args.args[0]
        assert len(passed_preds) == 1
        assert passed_preds[0].ticker == "AAA.NS"

    def test_invalid_date_exits_nonzero(self, runner, isolated_store):
        result = runner.invoke(app, ["grade", "--since", "not-a-date"])
        assert result.exit_code == 1
        assert "Invalid" in result.output


class TestCalibrationCommand:
    @patch("price_predictor.cli.main.grade_many")
    def test_no_predictions_short_circuits(
        self, mock_grade, runner, isolated_store,
    ):
        result = runner.invoke(app, ["calibration"])
        assert result.exit_code == 0
        assert "No predictions" in result.output
        mock_grade.assert_not_called()

    @patch("price_predictor.cli.main.grade_many")
    def test_aggregate_report(self, mock_grade, runner, isolated_store):
        store = PredictionStore(isolated_store)
        store.save(_make_graded().prediction)
        store.save(_make_graded().prediction)  # dedup -> 1 saved
        mock_grade.return_value = [_make_graded(), _make_graded()]

        result = runner.invoke(app, ["calibration"])
        assert result.exit_code == 0, result.output
        # Spot-check key metric labels appear
        assert "Hit rate" in result.output
        assert "Brier" in result.output

    @patch("price_predictor.cli.main.grade_many")
    def test_breakdown_by_horizon(self, mock_grade, runner, isolated_store):
        store = PredictionStore(isolated_store)
        # Save 1 prediction (mocked grades override anyway)
        store.save(_make_graded().prediction)
        mock_grade.return_value = [
            _make_graded(horizon=PredictionHorizon.WEEKLY),
            _make_graded(horizon=PredictionHorizon.BIWEEKLY, outcome=GradeOutcome.STOP_HIT,
                         direction_correct=False, realized_return=-0.04),
        ]

        result = runner.invoke(app, ["calibration", "--by", "horizon"])
        assert result.exit_code == 0, result.output
        assert "breakdown by horizon" in result.output.lower()
        assert "weekly" in result.output
        assert "biweekly" in result.output

    def test_invalid_by_axis_exits_nonzero(self, runner, isolated_store):
        result = runner.invoke(app, ["calibration", "--by", "banana"])
        assert result.exit_code == 1
        assert "Unknown --by" in result.output

    def test_breakdown_keys_registered(self):
        # Sanity: the dispatch map exposes the expected axes. Catches
        # someone accidentally renaming a key without updating help text.
        assert set(_BREAKDOWN_KEYS) == {"horizon", "ticker", "direction", "month"}
