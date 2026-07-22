"""Typer-based CLI for the price predictor.

Wired via [project.scripts] -> price_predictor:main, which delegates
to the typer app defined here. Commands:

    price-predictor predict <ticker> [--horizon=short] [--save]
    price-predictor predict-many <t1> <t2> ... [--horizon] [--save] [--concurrency]
    price-predictor history <ticker> [--limit=N]

WHY TYPER + RICH
================
- Typer gives us click-quality argparse with type hints + ergonomic
  defaults, and is already in our deps.
- Rich tables make terminal output actually readable (columns, colors,
  borders). Comes free as a typer transitive.
- Same import surface works under pytest's CliRunner for testing
  without spawning subprocesses.

DESIGN
======
1. ONE typer app, multiple commands. Standard 'verb on a noun' pattern.
2. Synchronous wrappers around async predict() / predict_many() via
   asyncio.run() at the command boundary. Typer doesn't natively
   support async commands; doing it once at the boundary keeps the
   library functions pure.
3. Default save location = settings.predictions_dir. CLI --save flag
   is opt-in (no surprise writes), but the path is consistent across
   commands so 'history' always reads from where 'predict --save'
   wrote.
4. Errors are caught at the top of each command and rendered via
   rich's red text, then sys.exit(1). Stack traces only with --verbose.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import date, datetime
from typing import Callable, Optional

import typer
from rich.console import Console
from rich.table import Table

from price_predictor.cli.backtest_cmd import backtest_command
from price_predictor.config.settings import settings
from price_predictor.llm.ollama_guard import warn_if_local_models_missing
from price_predictor.prediction import (
    BatchError,
    CalibrationReport,
    GradeOutcome,
    GradedPrediction,
    PredictionError,
    PredictionStore,
    compute_breakdown,
    compute_calibration,
    grade_many,
    predict as _predict,
    predict_many as _predict_many,
)
from price_predictor.prediction.schema import (
    Prediction,
    PredictionDirection,
    PredictionHorizon,
)

app = typer.Typer(
    name="price-predictor",
    help="Free, local Nifty50 trading prediction system.",
    no_args_is_help=True,
)


@app.callback()
def _startup(ctx: typer.Context) -> None:
    """Pre-command hook: verify the offline Ollama fallback model is ready.

    Only runs when an actual subcommand is invoked (skips bare `--help`).
    Non-fatal: logs a warning if the local fallback model isn't pulled,
    never blocks the command.
    """
    if ctx.invoked_subcommand is not None:
        warn_if_local_models_missing()

# Single shared Console - typer creates its own internally, but having
# our own lets us style consistently across commands.
console = Console()

# Color hints by direction. Note: bullish=green, bearish=red follow
# universal trading conventions; neutral=yellow is the cautious default.
_DIRECTION_STYLE = {
    PredictionDirection.BULLISH: "bold green",
    PredictionDirection.BEARISH: "bold red",
    PredictionDirection.NEUTRAL: "bold yellow",
}


# ─────────────────────────────────────────────────────────────
# Rendering helpers (pure, easy to test)
# ─────────────────────────────────────────────────────────────
def _render_prediction(p: Prediction) -> Table:
    """Pretty-print one prediction as a Rich table.

    Pure function returning a renderable - the caller decides whether
    to print or use it elsewhere (e.g. tests inspect the rows).
    """
    style = _DIRECTION_STYLE[p.direction]
    table = Table(
        title=f"[{style}]{p.ticker}[/{style}]  •  {p.horizon.value} horizon",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Field")
    table.add_column("Value")

    table.add_row("Direction", f"[{style}]{p.direction.value.upper()}[/{style}]")
    table.add_row("Confidence", f"{p.confidence:.0%}")
    table.add_row(
        "Entry zone",
        f"₹{p.entry_zone[0]:.2f} – ₹{p.entry_zone[1]:.2f}",
    )
    table.add_row("Target", f"₹{p.target.value:.2f}  [dim]({p.target.rationale})[/dim]")
    table.add_row(
        "Stop loss", f"₹{p.stop_loss.value:.2f}  [dim]({p.stop_loss.rationale})[/dim]",
    )
    table.add_row("As of", p.as_of.strftime("%Y-%m-%d %H:%M %Z"))
    table.add_row(
        "Close at prediction",
        f"₹{p.analysis_basis.close_price_at_prediction:.2f}",
    )
    return table


def _render_batch(results: list, tickers: list[str]) -> Table:
    """Compact summary table for a batch run."""
    table = Table(
        title=f"Batch results ({len(tickers)} tickers)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Ticker")
    table.add_column("Direction")
    table.add_column("Confidence")
    table.add_column("Target")
    table.add_column("Stop")
    table.add_column("Status")

    for ticker, r in zip(tickers, results):
        if isinstance(r, Prediction):
            style = _DIRECTION_STYLE[r.direction]
            table.add_row(
                r.ticker,
                f"[{style}]{r.direction.value.upper()}[/{style}]",
                f"{r.confidence:.0%}",
                f"₹{r.target.value:.2f}",
                f"₹{r.stop_loss.value:.2f}",
                "[green]ok[/green]",
            )
        else:  # BatchError
            table.add_row(
                ticker, "-", "-", "-", "-",
                f"[red]{r.error_type}[/red]",
            )
    return table


def _render_history(predictions: list[Prediction], ticker: str) -> Table:
    """Chronological history table for a single ticker."""
    table = Table(
        title=f"History for [bold]{ticker}[/bold] ({len(predictions)} predictions)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Date")
    table.add_column("Horizon")
    table.add_column("Direction")
    table.add_column("Confidence")
    table.add_column("Target")
    table.add_column("Close at pred")

    for p in predictions:
        style = _DIRECTION_STYLE[p.direction]
        table.add_row(
            p.as_of.strftime("%Y-%m-%d %H:%M"),
            p.horizon.value,
            f"[{style}]{p.direction.value.upper()}[/{style}]",
            f"{p.confidence:.0%}",
            f"₹{p.target.value:.2f}",
            f"₹{p.analysis_basis.close_price_at_prediction:.2f}",
        )
    return table


# ─────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────
@app.command(name="predict")
def predict_one(
    ticker: str = typer.Argument(..., help="Stock ticker (e.g. RELIANCE.NS, AAPL)"),
    horizon: str = typer.Option(
        "weekly", "--horizon", "-h",
        help="Prediction window: daily/weekly/biweekly/monthly",
    ),
    sensitivity: str = typer.Option(
        "standard", "--sensitivity", "-s",
        help="Indicator sensitivity preset",
    ),
    save: bool = typer.Option(
        False, "--save",
        help=f"Save to {settings.predictions_dir}",
    ),
) -> None:
    """Predict a single ticker. Exit 1 on failure.

    Single-horizon view. Multi-horizon (--horizons plural) lands in the
    CLI refactor commit; this wrapper extracts the lone Prediction from
    the dict that predict() now returns.
    """
    try:
        from price_predictor.prediction.schema import PredictionHorizon
        h_enum = PredictionHorizon(horizon)
        result_dict = asyncio.run(
            _predict(ticker, [h_enum], sensitivity=sensitivity)  # type: ignore[arg-type]
        )
        result = result_dict[h_enum]
    except PredictionError as e:
        console.print(f"[red]Prediction failed:[/red] {e}")
        sys.exit(1)

    console.print(_render_prediction(result))

    if save:
        store = PredictionStore(settings.predictions_dir)
        path = store.save(result)
        console.print(f"\n[dim]Saved to {path}[/dim]")


@app.command(name="predict-many")
def predict_many_cmd(
    tickers: list[str] = typer.Argument(..., help="One or more tickers"),
    horizon: str = typer.Option("weekly", "--horizon", "-h"),
    sensitivity: str = typer.Option("standard", "--sensitivity", "-s"),
    concurrency: int = typer.Option(3, "--concurrency", "-c"),
    save: bool = typer.Option(False, "--save"),
) -> None:
    """Predict multiple tickers in parallel."""
    try:
        results = asyncio.run(
            _predict_many(
                tickers, horizon,  # type: ignore[arg-type]
                sensitivity=sensitivity,  # type: ignore[arg-type]
                concurrency=concurrency,
            )
        )
    except ValueError as e:
        console.print(f"[red]Invalid input:[/red] {e}")
        sys.exit(1)

    console.print(_render_batch(results, tickers))

    if save:
        store = PredictionStore(settings.predictions_dir)
        n_saved = 0
        for r in results:
            if isinstance(r, Prediction):
                store.save(r)
                n_saved += 1
        console.print(f"\n[dim]Saved {n_saved} prediction(s) to {settings.predictions_dir}[/dim]")

    # Exit non-zero if any failed - useful for CI/cron pipelines.
    n_failed = sum(1 for r in results if isinstance(r, BatchError))
    if n_failed:
        sys.exit(1)


@app.command()
def history(
    ticker: str = typer.Argument(...),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n",
        help="Show only the most recent N (default: all)",
    ),
) -> None:
    """Show stored predictions for a ticker."""
    store = PredictionStore(settings.predictions_dir)
    preds = store.list_for_ticker(ticker)
    if not preds:
        console.print(f"[yellow]No stored predictions for {ticker}[/yellow]")
        console.print(f"[dim](looking in {settings.predictions_dir})[/dim]")
        return
    if limit is not None and limit > 0:
        preds = preds[-limit:]
    console.print(_render_history(preds, ticker))


# ─────────────────────────────────────────────────────────────
# Step 3.5: grade + calibration commands
# ─────────────────────────────────────────────────────────────
_OUTCOME_STYLE = {
    GradeOutcome.TARGET_HIT:         "bold green",
    GradeOutcome.STOP_HIT:           "bold red",
    GradeOutcome.STOP_HIT_AMBIGUOUS: "red",
    GradeOutcome.EXPIRED:            "yellow",
    GradeOutcome.NOT_APPLICABLE:     "dim",
    GradeOutcome.INCONCLUSIVE:       "dim",
}

# Maps the --by CLI flag to a key function for compute_breakdown. Adding
# a new grouping is a one-line dict entry - no schema/CLI surgery needed.
_BREAKDOWN_KEYS: dict[str, Callable[[GradedPrediction], object]] = {
    "horizon":   lambda g: g.prediction.horizon.value,
    "ticker":    lambda g: g.prediction.ticker,
    "direction": lambda g: g.prediction.direction.value,
    "month":     lambda g: g.prediction.as_of.strftime("%Y-%m"),
}


# ─────────────────────────────────────────────────────────────
# Shared loader (DRY: grade + calibration BOTH need this exact pipeline)
# ─────────────────────────────────────────────────────────────
def _parse_iso_date(s: str | None, *, label: str) -> date | None:
    """Parse YYYY-MM-DD or exit 1 with a friendly message."""
    if s is None:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        console.print(f"[red]Invalid --{label} date '{s}'. Use YYYY-MM-DD.[/red]")
        sys.exit(1)


def _load_predictions(
    *, ticker: str | None, since: date | None, until: date | None,
) -> list[Prediction]:
    """Read predictions from the store, filtering by ticker / date range.

    Pure 'load + filter' - no grading, no I/O beyond store reads. Each
    command builds its own pipeline on top of this so the loader stays
    one job.
    """
    store = PredictionStore(settings.predictions_dir)

    if ticker is not None:
        preds = store.list_for_ticker(ticker)
    elif since is not None or until is not None:
        # Date-range query when no ticker filter is given.
        # Defaults: open-ended on either side fills with sensible bounds.
        start = since or date(1970, 1, 1)
        end = until or date.today()
        preds = store.list_in_date_range(start, end)
    else:
        # No filters at all -> grade EVERYTHING the store knows about.
        preds = store.list_in_date_range(date(1970, 1, 1), date.today())

    # Apply secondary date filter when ticker AND since/until are both set:
    # list_for_ticker doesn't take a date range, so trim afterwards.
    if ticker is not None and (since is not None or until is not None):
        s = since or date(1970, 1, 1)
        e = until or date.today()
        preds = [p for p in preds if s <= p.as_of.date() <= e]

    return preds


# ─────────────────────────────────────────────────────────────
# Renderers (pure - return Table; tests assert on rendered text)
# ─────────────────────────────────────────────────────────────
def _render_grades(graded: list[GradedPrediction]) -> Table:
    """Per-prediction outcome table."""
    table = Table(
        title=f"Graded predictions ({len(graded)})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Date")
    table.add_column("Ticker")
    table.add_column("Dir")
    table.add_column("Outcome")
    table.add_column("Conf")
    table.add_column("Return")
    table.add_column("Days")

    for g in graded:
        p = g.prediction
        dir_style = _DIRECTION_STYLE[p.direction]
        out_style = _OUTCOME_STYLE[g.outcome]
        # Realized return only meaningful when judged.
        ret_str = "-" if g.outcome == GradeOutcome.INCONCLUSIVE else f"{g.realized_return:+.2%}"
        days_str = str(g.days_to_resolution) if g.days_to_resolution else "-"
        table.add_row(
            p.as_of.strftime("%Y-%m-%d"),
            p.ticker,
            f"[{dir_style}]{p.direction.value}[/{dir_style}]",
            f"[{out_style}]{g.outcome.value}[/{out_style}]",
            f"{p.confidence:.0%}",
            ret_str,
            days_str,
        )
    return table


def _render_calibration(report: CalibrationReport, title: str = "Calibration") -> Table:
    """Single-report summary table.

    Lays out the metrics in a 'metric | value | how-to-read-it' format
    so a non-stats user gets context inline. The 'how-to-read-it' column
    embeds the docstring summary - DRY with the report module's docs.
    """
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_column("Note")

    table.add_row("Total predictions", str(report.n_predictions), "")
    table.add_row("Judged (excl. inconclusive)", str(report.n_judged), "")
    table.add_row(
        "Hit rate (strict)", f"{report.hit_rate_strict:.1%}",
        "[dim]wins / (wins + losses + ambig + expired + na)[/dim]",
    )
    table.add_row(
        "Hit rate (resolved)", f"{report.hit_rate_resolved:.1%}",
        "[dim]wins / (wins + losses + ambig)  ← industry std[/dim]",
    )
    table.add_row(
        "Hit rate (optimistic)", f"{report.hit_rate_optimistic:.1%}",
        "[dim]wins / (wins + clean losses)[/dim]",
    )
    table.add_row(
        "Direction accuracy", f"{report.direction_accuracy:.1%}",
        "[dim]correct directional calls / judged[/dim]",
    )
    brier_str = f"{report.brier_score:.3f}" if report.brier_score is not None else "-"
    table.add_row(
        "Brier score", brier_str,
        "[dim]0=perfect, ~base_rate*(1-base_rate)=naive, 1=pathological[/dim]",
    )
    bss_str = (
        f"{report.brier_skill_score:+.3f}"
        if report.brier_skill_score is not None else "-"
    )
    table.add_row(
        "Brier Skill Score", bss_str,
        "[dim]>0 beats base-rate guess, 0=tied, <0 worse[/dim]",
    )
    base_rate_str = (
        f"{report.base_rate:.1%}" if report.base_rate is not None else "-"
    )
    table.add_row(
        "Base rate", base_rate_str,
        "[dim]empirical fraction correct — BSS reference[/dim]",
    )
    conf_str = f"{report.mean_confidence:.0%}" if report.mean_confidence is not None else "-"
    table.add_row("Mean confidence", conf_str, "")
    table.add_row("Mean return", f"{report.mean_return:+.2%}", "")
    table.add_row("Median return", f"{report.median_return:+.2%}", "")
    return table


def _render_breakdown(
    breakdown: dict, by: str,
) -> Table:
    """Compact multi-row breakdown table for `--by` queries."""
    table = Table(
        title=f"Calibration breakdown by {by}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column(by.capitalize())
    table.add_column("N")
    table.add_column("Hit rate (resolved)")
    table.add_column("Direction acc")
    table.add_column("Brier")
    table.add_column("BSS")
    table.add_column("Mean return")

    for key, report in breakdown.items():
        brier_str = f"{report.brier_score:.3f}" if report.brier_score is not None else "-"
        bss_str = (
            f"{report.brier_skill_score:+.3f}"
            if report.brier_skill_score is not None else "-"
        )
        table.add_row(
            str(key),
            str(report.n_predictions),
            f"{report.hit_rate_resolved:.1%}",
            f"{report.direction_accuracy:.1%}",
            brier_str,
            bss_str,
            f"{report.mean_return:+.2%}",
        )
    return table


# ─────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────
@app.command()
def grade(
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t"),
    since: Optional[str] = typer.Option(None, "--since", help="YYYY-MM-DD lower bound"),
    until: Optional[str] = typer.Option(None, "--until", help="YYYY-MM-DD upper bound"),
) -> None:
    """Grade stored predictions against realized OHLCV."""
    since_d = _parse_iso_date(since, label="since")
    until_d = _parse_iso_date(until, label="until")

    preds = _load_predictions(ticker=ticker, since=since_d, until=until_d)
    if not preds:
        console.print("[yellow]No predictions matched the filters.[/yellow]")
        console.print(f"[dim](looking in {settings.predictions_dir})[/dim]")
        return

    console.print(f"[dim]Grading {len(preds)} prediction(s) - this fetches OHLCV...[/dim]")
    graded = grade_many(preds)
    console.print(_render_grades(graded))


@app.command()
def calibration(
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t"),
    since: Optional[str] = typer.Option(None, "--since"),
    until: Optional[str] = typer.Option(None, "--until"),
    by: Optional[str] = typer.Option(
        None, "--by",
        help=f"Breakdown axis: one of {sorted(_BREAKDOWN_KEYS)}",
    ),
) -> None:
    """Compute calibration metrics over stored + graded predictions."""
    if by is not None and by not in _BREAKDOWN_KEYS:
        console.print(
            f"[red]Unknown --by '{by}'. Valid: {sorted(_BREAKDOWN_KEYS)}[/red]"
        )
        sys.exit(1)

    since_d = _parse_iso_date(since, label="since")
    until_d = _parse_iso_date(until, label="until")

    preds = _load_predictions(ticker=ticker, since=since_d, until=until_d)
    if not preds:
        console.print("[yellow]No predictions matched the filters.[/yellow]")
        return

    console.print(f"[dim]Grading {len(preds)} prediction(s)...[/dim]")
    graded = grade_many(preds)

    if by is None:
        report = compute_calibration(graded)
        console.print(_render_calibration(report))
        return

    breakdown = compute_breakdown(graded, _BREAKDOWN_KEYS[by])
    console.print(_render_breakdown(breakdown, by))


# ─────────────────────────────────────────────────────────────
# External commands (kept in their own modules for cohesion / size)
# ─────────────────────────────────────────────────────────────
app.command(name="backtest")(backtest_command)


# ─────────────────────────────────────────────────────────────
# Entry point — wired from src/price_predictor/__init__.py
# ─────────────────────────────────────────────────────────────
def main() -> None:
    """Entry point for [project.scripts] price-predictor = ...:main."""
    app()
