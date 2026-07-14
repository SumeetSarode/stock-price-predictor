"""Unit tests for backtest.html_report -- HTML rendering.

WHAT WE TEST
============
- render_html_report returns valid-ish HTML containing the right
  structural markers (headers, sections, charts).
- Numbers from the evaluation appear in the rendered output.
- write_html_report creates the file at the right path with content.
- HTML escaping defends against malicious ticker names.
- Empty/edge cases don't crash (no insights, no months, etc.).
- Palette colors appear in the output.

WHAT WE DON'T TEST (intentionally)
==================================
- Pixel-perfect layout -- that's a visual concern, not a logic one.
- Chart.js render correctness -- that's the library's job.
- Tailwind CDN availability -- runtime, not unit-testable.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from price_predictor.backtest.evaluation import BacktestEvaluation
from price_predictor.backtest.html_report import (
    render_html_report,
    write_html_report,
)
from price_predictor.backtest.runner import BacktestRun
from price_predictor.prediction.calibration import CalibrationReport
from price_predictor.prediction.grading import GradedPrediction, GradeOutcome
from price_predictor.prediction.schema import (
    AnalysisBasis,
    Prediction,
    PredictionDirection,
    PredictionHorizon,
    PriceLevel,
)
from zoneinfo import ZoneInfo


# ─────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────
def _prediction(
    ticker: str = "RELIANCE.NS",
    as_of_d: date = date(2024, 6, 14),
    horizon: PredictionHorizon = PredictionHorizon.WEEKLY,
    direction: PredictionDirection = PredictionDirection.BULLISH,
    confidence: float = 0.7,
    close: float = 200.0,
) -> Prediction:
    if direction == PredictionDirection.BULLISH:
        target_v, stop_v = close * 1.05, close * 0.97
    elif direction == PredictionDirection.BEARISH:
        target_v, stop_v = close * 0.95, close * 1.03
    else:
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


def _graded(
    pred: Prediction | None = None,
    outcome: GradeOutcome = GradeOutcome.TARGET_HIT,
    realized_return: float = 0.05,
    direction_correct: bool | None = True,
) -> GradedPrediction:
    return GradedPrediction(
        prediction=pred or _prediction(),
        outcome=outcome,
        realized_return=realized_return,
        direction_correct=direction_correct,
        days_to_resolution=3,
        bars_examined=5,
        close_at_window_end=210.0,
    )


def _report(
    *,
    n_predictions: int = 50,
    direction_accuracy: float = 0.6,
    n_judged: int = 50,
) -> CalibrationReport:
    return CalibrationReport(
        n_predictions=n_predictions,
        n_inconclusive=max(0, n_predictions - n_judged),
        n_target_hit=20, n_stop_hit=10, n_stop_hit_ambiguous=2,
        n_expired=0, n_not_applicable=0,
        hit_rate_strict=0.40,
        hit_rate_resolved=0.625,
        hit_rate_optimistic=0.667,
        n_with_direction_judgement=n_judged,
        direction_accuracy=direction_accuracy,
        brier_score=0.20, brier_skill_score=0.10,
        base_rate=0.52, mean_confidence=0.65,
        mean_return=0.012, median_return=0.008,
    )


def _make_run(
    tickers: tuple[str, ...] = ("RELIANCE.NS",),
    horizons: tuple[PredictionHorizon, ...] = (PredictionHorizon.WEEKLY,),
) -> BacktestRun:
    now = datetime.now(timezone.utc)
    return BacktestRun(
        predictions=[],
        errors=[],
        started_at=now,
        finished_at=now,
        tickers=tickers,
        as_of_dates=(date(2024, 6, 14),),
        horizons=horizons,
        sensitivity="standard",
        concurrency=3,
    )


def _make_eval(
    *,
    graded: list[GradedPrediction] | None = None,
    overall: CalibrationReport | None = None,
    by_horizon: dict | None = None,
    by_ticker: dict | None = None,
    by_direction: dict | None = None,
    by_month: dict | None = None,
    tickers: tuple[str, ...] = ("RELIANCE.NS",),
) -> BacktestEvaluation:
    return BacktestEvaluation(
        run=_make_run(tickers=tickers),
        graded=graded or [_graded()],
        overall=overall or _report(),
        by_horizon=by_horizon or {PredictionHorizon.WEEKLY: _report()},
        by_ticker=by_ticker or {tickers[0]: _report()},
        by_direction=by_direction or {PredictionDirection.BULLISH: _report()},
        by_month=by_month or {"2024-06": _report()},
    )


# ─────────────────────────────────────────────────────────────
# Structural sanity
# ─────────────────────────────────────────────────────────────
class TestStructure:
    def test_html_starts_with_doctype(self):
        html = render_html_report(_make_eval())
        assert html.startswith("<!DOCTYPE html>")

    def test_html_ends_cleanly(self):
        html = render_html_report(_make_eval())
        assert html.rstrip().endswith("</html>")

    def test_includes_chartjs_cdn(self):
        html = render_html_report(_make_eval())
        assert "chart.js" in html.lower() or "chart.umd" in html.lower()

    def test_includes_tailwind_cdn(self):
        html = render_html_report(_make_eval())
        assert "tailwindcss.com" in html

    def test_has_executive_summary_section(self):
        """Insights MUST appear at the top -- the ops mandate."""
        html = render_html_report(_make_eval())
        assert "Executive Summary" in html

    def test_has_key_takeaways_section(self):
        """Insights MUST also appear at the bottom -- the ops mandate.

        This is the 'mirror' so a busy reader doesn't have to scroll
        back up to see what mattered.
        """
        html = render_html_report(_make_eval())
        assert "Key Takeaways" in html

    def test_has_overall_section(self):
        html = render_html_report(_make_eval())
        assert "Overall Metrics" in html

    def test_has_horizon_section(self):
        html = render_html_report(_make_eval())
        assert "By Horizon" in html

    def test_has_ticker_section(self):
        html = render_html_report(_make_eval())
        assert "By Ticker" in html

    def test_has_drilldown_section(self):
        html = render_html_report(_make_eval())
        assert "Per-Prediction Detail" in html


# ─────────────────────────────────────────────────────────────
# Data presence -- numbers from evaluation must reach the HTML
# ─────────────────────────────────────────────────────────────
class TestDataPresence:
    def test_renders_ticker_in_hero(self):
        ev = _make_eval(tickers=("UNIQUE_TICKER.NS",))
        html = render_html_report(ev)
        assert "UNIQUE_TICKER.NS" in html

    def test_renders_direction_accuracy(self):
        """A run with 60% DA should show '60.0%' somewhere."""
        ev = _make_eval(overall=_report(direction_accuracy=0.60))
        html = render_html_report(ev)
        assert "60.0%" in html

    def test_renders_chart_canvases(self):
        """One <canvas> per breakdown chart (horizon, ticker, direction, month)."""
        ev = _make_eval()
        html = render_html_report(ev)
        canvas_ids = re.findall(r'id="(chart-by-\w+)"', html)
        assert "chart-by-horizon" in canvas_ids
        assert "chart-by-ticker" in canvas_ids
        assert "chart-by-direction" in canvas_ids
        assert "chart-by-month" in canvas_ids

    def test_chart_has_fixed_height_wrapper(self):
        """Chart.js needs a fixed-height parent div (responsive:true ignores canvas height)."""
        html = render_html_report(_make_eval())
        # Search for the inline-style div pattern wrapping any canvas.
        assert "height:320px" in html or 'height: 320px' in html


# ───────────────────────────────────────────────────
# Palette presence
# ───────────────────────────────────────────────────
class TestPalette:
    def test_uses_accent_blue(self):
        html = render_html_report(_make_eval())
        assert "#0053e2" in html  # accent blue

    def test_uses_accent_yellow(self):
        html = render_html_report(_make_eval())
        assert "#ffc220" in html  # accent yellow

    def test_uses_loss_red(self):
        """Even on a happy run, the red is referenced via CSS class
        for stop-hit badges + critical insights' palette.
        """
        html = render_html_report(_make_eval(graded=[
            _graded(outcome=GradeOutcome.STOP_HIT, direction_correct=False),
        ]))
        assert "#ea1100" in html  # loss red

    def test_uses_gain_green(self):
        html = render_html_report(_make_eval())
        assert "#2a8703" in html  # gain green


# ─────────────────────────────────────────────────────────────
# Security: HTML escaping
# ─────────────────────────────────────────────────────────────
class TestEscaping:
    def test_escapes_evil_ticker_name(self):
        """A ticker with HTML metacharacters MUST NOT inject script.

        Defensive even though tickers come from yfinance -- prevents
        a malicious yfinance response from XSSing the report.
        """
        evil = "<script>alert('xss')</script>"
        ev = _make_eval(tickers=(evil,))
        html = render_html_report(ev)
        # Raw unescaped tag must NOT appear.
        assert "<script>alert" not in html
        # Escaped form SHOULD appear (proves escaping ran, not silently dropped).
        assert "&lt;script&gt;alert" in html


# ─────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_renders_with_no_months(self):
        """A backtest spanning <1 month may have an empty by_month -- shouldn't crash."""
        ev = _make_eval(by_month={})
        html = render_html_report(ev)
        # The section should be omitted or replaced cleanly, not error.
        assert "<html" in html

    def test_renders_with_single_prediction(self):
        ev = _make_eval(graded=[_graded()])
        html = render_html_report(ev)
        assert "Per-Prediction Detail" in html

    def test_truncates_huge_drilldown(self):
        """1000-row table would bloat the HTML; we cap at 500."""
        many_graded = [_graded() for _ in range(600)]
        ev = _make_eval(graded=many_graded)
        html = render_html_report(ev)
        assert "Showing first 500" in html


# ─────────────────────────────────────────────────────────────
# write_html_report -- file I/O
# ─────────────────────────────────────────────────────────────
class TestWriteHtmlReport:
    def test_creates_file_at_path(self, tmp_path: Path):
        out = tmp_path / "subdir" / "report.html"
        result = write_html_report(_make_eval(), out)
        assert result.exists()
        assert result.is_file()
        # Returns absolute path.
        assert result.is_absolute()

    def test_creates_parent_dirs(self, tmp_path: Path):
        """Auto-mkdir -p so callers don't have to pre-create the tree."""
        deep = tmp_path / "a" / "b" / "c" / "report.html"
        write_html_report(_make_eval(), deep)
        assert deep.exists()

    def test_overwrites_existing(self, tmp_path: Path):
        """A re-run of the same backtest should overwrite the old report."""
        out = tmp_path / "report.html"
        out.write_text("OLD CONTENT")
        write_html_report(_make_eval(), out)
        content = out.read_text()
        assert "OLD CONTENT" not in content
        assert "<!DOCTYPE html>" in content
