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
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from price_predictor.config.settings import settings
from price_predictor.prediction import (
    BatchError,
    PredictionStore,
    PredictionError,
    predict as _predict,
    predict_many as _predict_many,
)
from price_predictor.prediction.schema import Prediction, PredictionDirection

app = typer.Typer(
    name="price-predictor",
    help="Free, local Nifty50 trading prediction system.",
    no_args_is_help=True,
)

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
        "short", "--horizon", "-h",
        help="Prediction window: intraday/short/medium/long",
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
    """Predict a single ticker. Exit 1 on failure."""
    try:
        result = asyncio.run(_predict(ticker, horizon, sensitivity=sensitivity))  # type: ignore[arg-type]
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
    horizon: str = typer.Option("short", "--horizon", "-h"),
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
# Entry point — wired from src/price_predictor/__init__.py
# ─────────────────────────────────────────────────────────────
def main() -> None:
    """Entry point for [project.scripts] price-predictor = ...:main."""
    app()
