"""`price-predictor backtest` -- end-to-end backtest CLI command.

WHY A SEPARATE MODULE
=====================
cli/main.py was approaching the 600-line cap. Backtest is a chunky,
self-contained feature (date parsing + tickers/horizons CSV + Rich
progress wiring + browser open) that earns its own module on cohesion
grounds, not just line-count.

WIRING
======
This module exposes one public function -- ``backtest_command`` -- with
typer-style defaults baked in. ``cli/main.py`` registers it via
``app.command(name="backtest")(backtest_command)``. Keeping the typer
``app`` out of this file avoids a circular import.

DESIGN
======
1. Validation up front, in the order the user typed flags. Bad date /
   bad horizon / future end-date all fail with a single red message
   and exit 1 BEFORE we touch any data. Caller-bug discipline.
2. The wrapper is sync (typer doesn't natively await). asyncio.run()
   sits at the boundary -- everything inside the command body is sync
   and easy to test.
3. Browser open uses stdlib ``webbrowser`` -- works on Mac/Win/Linux,
   no new deps. Suppressed when stdout isn't a TTY (cron, CI) or when
   ``--no-open`` is passed.
4. Progress bar uses Rich's ``Progress`` -- already a transitive dep.
   The callback is a closure that updates the bar's task; the closure
   pattern keeps the Progress instance lifecycle inside the command.
"""
from __future__ import annotations

import asyncio
import sys
import webbrowser
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from price_predictor.backtest import (
    BacktestProgress,
    BacktestRun,
    evaluate_backtest,
    run_backtest,
    trading_days_in_range,
    write_html_report,
)
from price_predictor.prediction.schema import PredictionHorizon

console = Console()


# ─────────────────────────────────────────────────────────────
# Parsing helpers (pure -- no I/O, easy to unit-test)
# ─────────────────────────────────────────────────────────────
def _parse_iso_date(s: str, *, label: str) -> date:
    """Parse YYYY-MM-DD or exit 1 with a friendly message.

    Lives here (not imported from cli.main) so this module has zero
    intra-CLI deps -- main can be refactored without breaking us.
    """
    try:
        return date.fromisoformat(s)
    except ValueError:
        console.print(
            f"[red]Invalid --{label} date '{s}'. Use YYYY-MM-DD.[/red]"
        )
        raise typer.Exit(code=1)


def _parse_csv_list(raw: str, *, label: str) -> list[str]:
    """Split 'A,B,C' (or 'A , B , C') into ['A','B','C'], stripping blanks.

    Returns a deduped list preserving first-occurrence order. Empty
    input is a caller bug -- exit 1.
    """
    if not raw or not raw.strip():
        console.print(f"[red]--{label} cannot be empty.[/red]")
        raise typer.Exit(code=1)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        console.print(
            f"[red]--{label} '{raw}' has no usable entries after parsing.[/red]"
        )
        raise typer.Exit(code=1)
    # Dedupe preserving order.
    return list(dict.fromkeys(parts))


def _parse_horizons(raw: str) -> list[PredictionHorizon]:
    """CSV string -> list[PredictionHorizon]. Exits 1 on unknown values.

    The error message lists valid options so the user isn't guessing.
    """
    names = _parse_csv_list(raw, label="horizons")
    valid = {h.value for h in PredictionHorizon}
    out: list[PredictionHorizon] = []
    for n in names:
        if n not in valid:
            console.print(
                f"[red]Unknown horizon '{n}'. Valid: {sorted(valid)}[/red]"
            )
            raise typer.Exit(code=1)
        out.append(PredictionHorizon(n))
    return out


def _default_report_path() -> Path:
    """Auto-named report in the CWD: backtest_YYYY-MM-DD_HHMMSS.html.

    CWD (not a temp dir) so the user immediately sees the file in
    their `ls`. Filename includes seconds to avoid collisions on
    rapid back-to-back runs.
    """
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return Path.cwd() / f"backtest_{stamp}.html"


def _open_in_browser(path: Path) -> bool:
    """Open ``path`` in the default browser. Returns True on success.

    Failure is non-fatal -- the file already exists; we just couldn't
    auto-open it. The caller should still print the path so the user
    can click it manually.
    """
    try:
        return webbrowser.open(path.resolve().as_uri())
    except Exception as exc:  # noqa: BLE001 -- defensive UX shim
        console.print(f"[yellow]Could not auto-open browser: {exc}[/yellow]")
        return False


def _should_open_browser(*, no_open_flag: bool) -> bool:
    """Decide whether to attempt browser-open.

    Suppressed when:
    - User passed ``--no-open`` (explicit).
    - stdout isn't a TTY (piped/cron/CI -- nobody's watching).

    Both checks together mean ``--no-open`` is rarely needed in
    practice -- piping already does the right thing.
    """
    if no_open_flag:
        return False
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        # Some test runners replace stdout with objects that lack isatty.
        return False


# ─────────────────────────────────────────────────────────────
# Progress callback factory
# ─────────────────────────────────────────────────────────────
def _make_progress_callback(
    progress: Progress, task_id: int,
) -> Callable[[BacktestProgress], None]:
    """Bind a Rich Progress task to the run_backtest callback signature.

    Closure (not a class) because the state is just two refs and the
    callback is a one-liner -- a class would be ceremony.
    """
    def _cb(snapshot: BacktestProgress) -> None:
        progress.update(
            task_id,
            completed=snapshot.completed,
            description=f"[cyan]{snapshot.current}",
        )
    return _cb


# ─────────────────────────────────────────────────────────────
# The command itself
# ─────────────────────────────────────────────────────────────
def backtest_command(
    start: str = typer.Option(
        ..., "--start", help="Backtest window start, YYYY-MM-DD (inclusive).",
    ),
    end: str = typer.Option(
        ..., "--end", help="Backtest window end, YYYY-MM-DD (inclusive).",
    ),
    tickers: str = typer.Option(
        ..., "--tickers",
        help="Comma-separated tickers, e.g. 'RELIANCE.NS,TCS.NS'.",
    ),
    horizons: str = typer.Option(
        "weekly", "--horizons",
        help="Comma-separated horizons. Default: weekly.",
    ),
    stride: int = typer.Option(
        5, "--stride",
        help="Trading-day stride (1=daily, 5=weekly, 21=monthly).",
    ),
    sensitivity: str = typer.Option(
        "standard", "--sensitivity", "-s",
        help="Indicator sensitivity preset.",
    ),
    concurrency: int = typer.Option(
        3, "--concurrency", "-c",
        help="Max in-flight predict() calls.",
    ),
    out: Optional[str] = typer.Option(
        None, "--out",
        help="Report output path. Default: ./backtest_<timestamp>.html.",
    ),
    save_predictions: bool = typer.Option(
        False, "--save-predictions",
        help="Eagerly persist each prediction to settings.predictions_dir.",
    ),
    no_open: bool = typer.Option(
        False, "--no-open",
        help="Don't auto-open the report in the browser.",
    ),
) -> None:
    """Run a backtest over a date range and write an HTML report.

    Wires run_backtest -> evaluate_backtest -> write_html_report into
    a single command. Predictions are ephemeral by default; pass
    --save-predictions to persist them for follow-up `grade` /
    `calibration` queries.
    """
    # ── 1. Validate user inputs (fail loud BEFORE any data fetch).
    start_d = _parse_iso_date(start, label="start")
    end_d = _parse_iso_date(end, label="end")
    today = date.today()
    if end_d > today:
        console.print(
            f"[red]--end ({end_d}) must be in the past; "
            f"predict() rejects future as_of dates.[/red]"
        )
        raise typer.Exit(code=1)

    ticker_list = _parse_csv_list(tickers, label="tickers")
    horizon_list = _parse_horizons(horizons)

    # ── 2. Build the trading-day schedule. trading_days_in_range
    #     handles start>end and stride<1 -- we just translate its
    #     ValueErrors to friendly red text.
    try:
        as_of_dates = trading_days_in_range(start_d, end_d, stride=stride)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    if not as_of_dates:
        console.print(
            f"[yellow]No NSE trading days between {start_d} and {end_d}.[/yellow]"
        )
        raise typer.Exit(code=1)

    n_pairs = len(ticker_list) * len(as_of_dates)
    console.print(
        f"[dim]Scheduling {n_pairs} prediction(s): "
        f"{len(ticker_list)} ticker(s) x {len(as_of_dates)} as-of date(s) "
        f"x {len(horizon_list)} horizon(s) "
        f"(stride={stride}).[/dim]"
    )

    # ── 3. Optional eager-save store (matches existing CLI pattern).
    store = None
    if save_predictions:
        # Lazy import: avoids paying the prediction package's deps when
        # the user doesn't ask to persist.
        from price_predictor.config.settings import settings
        from price_predictor.prediction import PredictionStore
        store = PredictionStore(settings.predictions_dir)
        console.print(
            f"[dim]Saving each prediction to {settings.predictions_dir}.[/dim]"
        )

    # ── 4. Run the backtest with a Rich progress bar.
    run = _run_with_progress(
        ticker_list, as_of_dates, horizon_list,
        sensitivity=sensitivity,
        concurrency=concurrency,
        store=store,
        total_pairs=n_pairs,
    )

    # ── 5. Bail out if the whole run failed (no predictions to grade).
    if not run.predictions:
        console.print(
            f"[red]All {len(run.errors)} pair(s) failed. "
            f"First error: {run.errors[0].error_message}[/red]"
        )
        raise typer.Exit(code=1)

    if run.errors:
        console.print(
            f"[yellow]{len(run.errors)} of {n_pairs} pair(s) failed; "
            f"see report errors section.[/yellow]"
        )

    # ── 6. Grade + calibrate.
    console.print(f"[dim]Grading {len(run.predictions)} prediction(s)...[/dim]")
    with console.status("[dim]Computing calibration...[/dim]", spinner="dots"):
        evaluation = evaluate_backtest(run)

    # ── 7. Write the HTML report and (maybe) open it.
    out_path = Path(out) if out else _default_report_path()
    written = write_html_report(evaluation, out_path)
    console.print(f"[green]Wrote report to {written}.[/green]")

    if _should_open_browser(no_open_flag=no_open):
        opened = _open_in_browser(written)
        if opened:
            console.print("[dim]Opening in browser...[/dim]")


# ─────────────────────────────────────────────────────────────
# Async runner wrapper (extracted so tests can patch the boundary)
# ─────────────────────────────────────────────────────────────
def _run_with_progress(
    tickers: list[str],
    as_of_dates: list[date],
    horizons: list[PredictionHorizon],
    *,
    sensitivity: str,
    concurrency: int,
    store: object | None,
    total_pairs: int,
) -> BacktestRun:
    """Drive run_backtest under a Rich progress bar.

    Extracted so tests can monkeypatch ``backtest_cmd._run_with_progress``
    with a synchronous stub instead of mocking the whole asyncio +
    Rich + run_backtest stack.
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Backtesting"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("•"),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )
    with progress:
        task_id = progress.add_task("[dim]starting...", total=total_pairs)
        callback = _make_progress_callback(progress, task_id)
        return asyncio.run(
            run_backtest(
                tickers,
                as_of_dates,
                horizons,
                sensitivity=sensitivity,  # type: ignore[arg-type]
                concurrency=concurrency,
                store=store,  # type: ignore[arg-type]
                progress_callback=callback,
            )
        )
