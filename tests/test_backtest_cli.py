"""CLI tests for `price-predictor backtest` (Step 2.4).

Strategy
========
- Pure parser helpers tested in isolation (no CliRunner, no patching).
- Command-level behavior tested via Typer's CliRunner with the heavy
  bits (`_run_with_progress`, `evaluate_backtest`, `write_html_report`)
  monkey-patched on the module so we never touch network / LLM / disk
  beyond the temp output path.

Why patch on the module (not the original location)
===================================================
`backtest_command` does `from price_predictor.backtest import ...` at
import time, so the names live on `cli.backtest_cmd`. Patching
`price_predictor.backtest.evaluate_backtest` would leave the CLI
holding the original reference. Standard Python-patching gotcha.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from price_predictor.cli import backtest_cmd
from price_predictor.cli.backtest_cmd import (
    _default_report_path,
    _make_progress_callback,
    _open_in_browser,
    _parse_csv_list,
    _parse_horizons,
    _parse_iso_date,
    _should_open_browser,
)
from price_predictor.cli.main import app


# ─────────────────────────────────────────────────────────────
# Fixtures + factories
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def runner() -> CliRunner:
    # Keep stderr separate so we can assert error messages
    # independently of the normal Rich output stream.
    return CliRunner(mix_stderr=False)


def _fake_run(
    *, n_predictions: int = 2, n_errors: int = 0,
) -> MagicMock:
    """Build a stand-in BacktestRun -- only the attributes the CLI reads."""
    run = MagicMock(name="BacktestRun")
    run.predictions = [MagicMock(name=f"pred-{i}") for i in range(n_predictions)]
    run.errors = []
    for i in range(n_errors):
        err = MagicMock(name=f"err-{i}")
        err.error_message = f"boom-{i}"
        run.errors.append(err)
    return run


def _fake_evaluation() -> MagicMock:
    return MagicMock(name="BacktestEvaluation")


# ─────────────────────────────────────────────────────────────
# Pure parser helpers
# ─────────────────────────────────────────────────────────────
class TestParseIsoDate:
    def test_parses_valid_date(self):
        assert _parse_iso_date("2024-06-14", label="start") == date(2024, 6, 14)

    def test_invalid_date_exits(self):
        import typer
        with pytest.raises(typer.Exit) as exc_info:
            _parse_iso_date("nope", label="start")
        assert exc_info.value.exit_code == 1

    def test_wrong_format_exits(self):
        import typer
        with pytest.raises(typer.Exit):
            _parse_iso_date("06/14/2024", label="end")


class TestParseCsvList:
    def test_basic_csv(self):
        assert _parse_csv_list("A,B,C", label="x") == ["A", "B", "C"]

    def test_strips_whitespace(self):
        assert _parse_csv_list(" A , B ,C ", label="x") == ["A", "B", "C"]

    def test_dedupes_preserving_order(self):
        assert _parse_csv_list("A,B,A,C,B", label="x") == ["A", "B", "C"]

    def test_empty_string_exits(self):
        import typer
        with pytest.raises(typer.Exit):
            _parse_csv_list("", label="tickers")

    def test_whitespace_only_exits(self):
        import typer
        with pytest.raises(typer.Exit):
            _parse_csv_list("   ", label="tickers")

    def test_only_separators_exits(self):
        import typer
        with pytest.raises(typer.Exit):
            _parse_csv_list(",,,", label="tickers")


class TestParseHorizons:
    def test_single_horizon(self):
        from price_predictor.prediction.schema import PredictionHorizon
        assert _parse_horizons("weekly") == [PredictionHorizon.WEEKLY]

    def test_multiple_horizons(self):
        from price_predictor.prediction.schema import PredictionHorizon
        result = _parse_horizons("daily,weekly,monthly")
        assert result == [
            PredictionHorizon.DAILY,
            PredictionHorizon.WEEKLY,
            PredictionHorizon.MONTHLY,
        ]

    def test_unknown_horizon_exits(self):
        import typer
        with pytest.raises(typer.Exit):
            _parse_horizons("hourly")

    def test_dedupes(self):
        from price_predictor.prediction.schema import PredictionHorizon
        result = _parse_horizons("weekly,weekly,daily")
        assert result == [PredictionHorizon.WEEKLY, PredictionHorizon.DAILY]


class TestDefaultReportPath:
    def test_returns_html_in_cwd(self):
        path = _default_report_path()
        assert path.suffix == ".html"
        assert path.parent == Path.cwd()
        assert path.name.startswith("backtest_")

    def test_includes_seconds_in_stamp(self):
        # Two back-to-back calls in the same second can collide; that's
        # fine because the user would have asked for --out in that case.
        # We just guarantee the format includes seconds (HHMMSS = 6 digits).
        path = _default_report_path()
        # backtest_YYYY-MM-DD_HHMMSS.html
        stem = path.stem  # "backtest_2026-05-12_142233"
        time_part = stem.split("_")[-1]
        assert len(time_part) == 6
        assert time_part.isdigit()


class TestShouldOpenBrowser:
    def test_no_open_flag_suppresses(self):
        assert _should_open_browser(no_open_flag=True) is False

    def test_non_tty_suppresses(self):
        with patch("price_predictor.cli.backtest_cmd.sys") as mock_sys:
            mock_sys.stdout.isatty.return_value = False
            assert _should_open_browser(no_open_flag=False) is False

    def test_tty_and_no_flag_opens(self):
        with patch("price_predictor.cli.backtest_cmd.sys") as mock_sys:
            mock_sys.stdout.isatty.return_value = True
            assert _should_open_browser(no_open_flag=False) is True

    def test_isatty_attribute_error_treated_as_non_tty(self):
        # Some test runners replace stdout with objects that lack isatty.
        with patch("price_predictor.cli.backtest_cmd.sys") as mock_sys:
            mock_sys.stdout.isatty.side_effect = AttributeError
            assert _should_open_browser(no_open_flag=False) is False


class TestOpenInBrowser:
    def test_calls_webbrowser_with_file_uri(self, tmp_path: Path):
        target = tmp_path / "report.html"
        target.write_text("hi")
        with patch("price_predictor.cli.backtest_cmd.webbrowser") as mock_wb:
            mock_wb.open.return_value = True
            ok = _open_in_browser(target)
        assert ok is True
        url_arg = mock_wb.open.call_args[0][0]
        assert url_arg.startswith("file://")
        assert url_arg.endswith("report.html")

    def test_swallows_exceptions(self, tmp_path: Path):
        target = tmp_path / "x.html"
        target.write_text("x")
        with patch("price_predictor.cli.backtest_cmd.webbrowser") as mock_wb:
            mock_wb.open.side_effect = RuntimeError("no display")
            assert _open_in_browser(target) is False


class TestProgressCallback:
    def test_updates_task_with_snapshot(self):
        from price_predictor.backtest.runner import BacktestProgress
        progress = MagicMock()
        cb = _make_progress_callback(progress, task_id=42)
        snap = BacktestProgress(
            completed=3, total=10, successes=3, failures=0,
            current="RELIANCE.NS @ 2024-06-14",
        )
        cb(snap)
        progress.update.assert_called_once_with(
            42, completed=3, description="[cyan]RELIANCE.NS @ 2024-06-14",
        )


# ─────────────────────────────────────────────────────────────
# Command-level tests (CliRunner)
# ─────────────────────────────────────────────────────────────
class TestBacktestCommandValidation:
    """Inputs that should fail loud BEFORE we touch any data."""

    def test_bad_start_date(self, runner: CliRunner):
        result = runner.invoke(app, [
            "backtest", "--start", "not-a-date", "--end", "2024-06-14",
            "--tickers", "RELIANCE.NS",
        ])
        assert result.exit_code == 1
        assert "Invalid --start" in result.stdout

    def test_bad_end_date(self, runner: CliRunner):
        result = runner.invoke(app, [
            "backtest", "--start", "2024-06-01", "--end", "bogus",
            "--tickers", "RELIANCE.NS",
        ])
        assert result.exit_code == 1
        assert "Invalid --end" in result.stdout

    def test_future_end_rejected(self, runner: CliRunner):
        future = date.today() + timedelta(days=30)
        result = runner.invoke(app, [
            "backtest", "--start", "2024-06-01",
            "--end", future.isoformat(),
            "--tickers", "RELIANCE.NS",
        ])
        assert result.exit_code == 1
        assert "must be in the past" in result.stdout

    def test_start_after_end_rejected(self, runner: CliRunner):
        # trading_days_in_range raises ValueError; we surface it.
        result = runner.invoke(app, [
            "backtest", "--start", "2024-06-30", "--end", "2024-06-01",
            "--tickers", "RELIANCE.NS",
        ])
        assert result.exit_code == 1
        assert "must be <=" in result.stdout

    def test_empty_tickers_rejected(self, runner: CliRunner):
        result = runner.invoke(app, [
            "backtest", "--start", "2024-06-01", "--end", "2024-06-14",
            "--tickers", "",
        ])
        assert result.exit_code == 1
        assert "tickers cannot be empty" in result.stdout

    def test_unknown_horizon_rejected(self, runner: CliRunner):
        result = runner.invoke(app, [
            "backtest", "--start", "2024-06-01", "--end", "2024-06-14",
            "--tickers", "RELIANCE.NS", "--horizons", "yearly",
        ])
        assert result.exit_code == 1
        assert "Unknown horizon" in result.stdout

    def test_bad_stride_rejected(self, runner: CliRunner):
        # stride<1 surfaced from trading_days_in_range
        result = runner.invoke(app, [
            "backtest", "--start", "2024-06-01", "--end", "2024-06-14",
            "--tickers", "RELIANCE.NS", "--stride", "0",
        ])
        assert result.exit_code == 1
        assert "stride must be >= 1" in result.stdout


class TestBacktestCommandHappyPath:
    """End-to-end with the heavy bits patched out."""

    def _patch_runtime(
        self,
        run: MagicMock,
        evaluation: MagicMock,
        out_path: Path,
        *,
        webbrowser_open: bool = True,
        is_tty: bool = True,
    ):
        """Stack the patches the success-path tests all need.

        Returned context-manager bundle: enter to activate, exit to clean.
        Tests that need access to individual mocks should use patch.multiple
        directly, but most just want "everything mocked and return X".
        """
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(
            backtest_cmd, "_run_with_progress", return_value=run,
        ))
        stack.enter_context(patch.object(
            backtest_cmd, "evaluate_backtest", return_value=evaluation,
        ))
        m_write = stack.enter_context(patch.object(
            backtest_cmd, "write_html_report", return_value=out_path,
        ))
        m_wb = stack.enter_context(patch.object(backtest_cmd, "webbrowser"))
        m_wb.open.return_value = webbrowser_open
        m_sys = stack.enter_context(patch.object(backtest_cmd, "sys"))
        m_sys.stdout.isatty.return_value = is_tty
        return stack, m_write, m_wb

    def test_success_writes_report_and_opens_browser(
        self, runner: CliRunner, tmp_path: Path,
    ):
        out = tmp_path / "report.html"
        run = _fake_run(n_predictions=3)
        ev = _fake_evaluation()

        stack, m_write, m_wb = self._patch_runtime(run, ev, out)
        with stack:
            result = runner.invoke(app, [
                "backtest", "--start", "2024-06-01", "--end", "2024-06-14",
                "--tickers", "RELIANCE.NS",
                "--out", str(out),
            ])

        assert result.exit_code == 0, result.stdout
        m_write.assert_called_once_with(ev, out)
        # Default behaviour: open in browser (TTY simulated above).
        m_wb.open.assert_called_once()

    def test_no_open_flag_suppresses_browser(
        self, runner: CliRunner, tmp_path: Path,
    ):
        out = tmp_path / "r.html"
        stack, _m_write, m_wb = self._patch_runtime(
            _fake_run(), _fake_evaluation(), out,
        )
        with stack:
            result = runner.invoke(app, [
                "backtest", "--start", "2024-06-01", "--end", "2024-06-14",
                "--tickers", "RELIANCE.NS",
                "--out", str(out), "--no-open",
            ])
        assert result.exit_code == 0
        m_wb.open.assert_not_called()

    def test_non_tty_suppresses_browser(
        self, runner: CliRunner, tmp_path: Path,
    ):
        out = tmp_path / "r.html"
        stack, _m_write, m_wb = self._patch_runtime(
            _fake_run(), _fake_evaluation(), out, is_tty=False,
        )
        with stack:
            result = runner.invoke(app, [
                "backtest", "--start", "2024-06-01", "--end", "2024-06-14",
                "--tickers", "RELIANCE.NS",
                "--out", str(out),
            ])
        assert result.exit_code == 0
        m_wb.open.assert_not_called()

    def test_default_out_path_used_when_no_flag(
        self, runner: CliRunner, tmp_path: Path,
    ):
        # Just verify write_html_report was called with SOME .html path in
        # cwd; the exact stamp is unstable across runs.
        run = _fake_run()
        ev = _fake_evaluation()
        with patch.object(backtest_cmd, "_run_with_progress", return_value=run), \
             patch.object(backtest_cmd, "evaluate_backtest", return_value=ev), \
             patch.object(backtest_cmd, "write_html_report") as m_write, \
             patch.object(backtest_cmd, "webbrowser"), \
             patch.object(backtest_cmd, "sys") as m_sys:
            m_write.return_value = Path("dummy.html")
            m_sys.stdout.isatty.return_value = False
            result = runner.invoke(app, [
                "backtest", "--start", "2024-06-01", "--end", "2024-06-14",
                "--tickers", "RELIANCE.NS",
            ])
        assert result.exit_code == 0
        path_arg = m_write.call_args[0][1]
        assert path_arg.suffix == ".html"
        assert path_arg.name.startswith("backtest_")

    def test_partial_failures_warn_but_succeed(
        self, runner: CliRunner, tmp_path: Path,
    ):
        out = tmp_path / "r.html"
        run = _fake_run(n_predictions=2, n_errors=1)
        stack, _m_write, _m_wb = self._patch_runtime(
            run, _fake_evaluation(), out,
        )
        with stack:
            result = runner.invoke(app, [
                "backtest", "--start", "2024-06-01", "--end", "2024-06-14",
                "--tickers", "RELIANCE.NS",
                "--out", str(out), "--no-open",
            ])
        assert result.exit_code == 0
        assert "1 of" in result.stdout and "failed" in result.stdout

    def test_all_failures_exit_1(
        self, runner: CliRunner, tmp_path: Path,
    ):
        run = _fake_run(n_predictions=0, n_errors=2)
        with patch.object(backtest_cmd, "_run_with_progress", return_value=run):
            result = runner.invoke(app, [
                "backtest", "--start", "2024-06-01", "--end", "2024-06-14",
                "--tickers", "RELIANCE.NS",
            ])
        assert result.exit_code == 1
        assert "All 2 pair(s) failed" in result.stdout

    def test_save_predictions_wires_store(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        # Isolate predictions_dir to tmp. The settings singleton is
        # imported lazily inside backtest_command, so we patch the
        # attribute on the cli.main re-export (same object identity).
        monkeypatch.setattr(
            "price_predictor.cli.main.settings.data_dir", tmp_path,
        )
        out = tmp_path / "r.html"
        run = _fake_run()
        stack, _m_write, _m_wb = self._patch_runtime(
            run, _fake_evaluation(), out,
        )
        with stack:
            # Capture the kwargs passed to _run_with_progress so we can
            # verify a non-None store landed in there.
            captured: dict = {}

            def _capture(*args, **kwargs):
                captured.update(kwargs)
                return run

            with patch.object(backtest_cmd, "_run_with_progress", side_effect=_capture):
                result = runner.invoke(app, [
                    "backtest", "--start", "2024-06-01", "--end", "2024-06-14",
                    "--tickers", "RELIANCE.NS",
                    "--out", str(out), "--no-open", "--save-predictions",
                ])

        assert result.exit_code == 0, result.stdout
        assert captured.get("store") is not None

    def test_no_save_predictions_means_no_store(
        self, runner: CliRunner, tmp_path: Path,
    ):
        out = tmp_path / "r.html"
        run = _fake_run()
        captured: dict = {}

        def _capture(*args, **kwargs):
            captured.update(kwargs)
            return run

        with patch.object(backtest_cmd, "_run_with_progress", side_effect=_capture), \
             patch.object(backtest_cmd, "evaluate_backtest", return_value=_fake_evaluation()), \
             patch.object(backtest_cmd, "write_html_report", return_value=out), \
             patch.object(backtest_cmd, "webbrowser"), \
             patch.object(backtest_cmd, "sys") as m_sys:
            m_sys.stdout.isatty.return_value = False
            result = runner.invoke(app, [
                "backtest", "--start", "2024-06-01", "--end", "2024-06-14",
                "--tickers", "RELIANCE.NS",
                "--out", str(out),
            ])

        assert result.exit_code == 0
        assert captured.get("store") is None

    def test_concurrency_flag_forwarded(
        self, runner: CliRunner, tmp_path: Path,
    ):
        out = tmp_path / "r.html"
        run = _fake_run()
        captured: dict = {}

        def _capture(*args, **kwargs):
            captured.update(kwargs)
            return run

        with patch.object(backtest_cmd, "_run_with_progress", side_effect=_capture), \
             patch.object(backtest_cmd, "evaluate_backtest", return_value=_fake_evaluation()), \
             patch.object(backtest_cmd, "write_html_report", return_value=out), \
             patch.object(backtest_cmd, "webbrowser"), \
             patch.object(backtest_cmd, "sys") as m_sys:
            m_sys.stdout.isatty.return_value = False
            runner.invoke(app, [
                "backtest", "--start", "2024-06-01", "--end", "2024-06-14",
                "--tickers", "RELIANCE.NS",
                "--out", str(out), "--concurrency", "7",
            ])
        assert captured.get("concurrency") == 7

    def test_help_includes_all_flags(self, runner: CliRunner):
        result = runner.invoke(app, ["backtest", "--help"])
        assert result.exit_code == 0
        for flag in (
            "--start", "--end", "--tickers", "--horizons", "--stride",
            "--sensitivity", "--concurrency", "--out", "--save-predictions",
            "--no-open",
        ):
            assert flag in result.stdout, f"missing {flag} in --help"
