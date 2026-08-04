"""Benchmark Ollama configs to settle the two NON-sure-shot speedups.

WHAT THIS ANSWERS
=================
The three flash-attention/KV/keep-alive flags are sure-shot (applied in
ensure_ollama.py). These two are NOT -- they need measurement on OUR task
(no external benchmark measures directional equity prediction):

    1. Lowering OLLAMA_NUM_CTX   -- free speedup ONLY if prompts never
       overflow it. We log the ACTUAL prompt-token sizes and recommend the
       smallest safe num_ctx.
    2. think=off / qwen3:4b      -- faster, but do they keep prediction
       SKILL? We run the real backtest per config and compare
       direction-accuracy + Brier Skill Score against tok/s + latency.

DESIGN: SUBPROCESS-PER-CONFIG
=============================
predictor.py builds its agents at IMPORT time from settings, so you can't
switch model/thinking in-process. And keeping two qwen3 models resident on
a 16 GB box would thrash swap and poison the speed numbers. So each config
runs in a FRESH subprocess with env vars (CHAIN_AGENTIC, OLLAMA_REASONING_
EFFORT, OLLAMA_NUM_CTX, USE_PAID) set BEFORE import. Clean RAM, clean
numbers, zero cross-contamination. The child prints one JSON line; the
parent renders the comparison table.

STATISTICAL HONESTY
===================
Direction accuracy on ~30 predictions is NOISY. The default grid is a
SMOKE TEST to prove the harness + get a rough read. For a decision you can
defend, widen the grid (more --tickers, smaller --stride, higher
--max-dates) to a few HUNDRED predictions and run overnight. The tool
prints n_judged so you know how much to trust the deltas.

USAGE
=====
    # default smoke sweep (4 configs: qwen3:8b/4b x think on/off)
    python scripts/bench_ollama_configs.py

    # a real, defensible run
    python scripts/bench_ollama_configs.py \\
        --tickers RELIANCE.NS TCS.NS HDFCBANK.NS INFY.NS ITC.NS \\
        --start 2026-01-01 --end 2026-03-31 --stride 2 --max-dates 40
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Pure helpers are import-safe (no settings / network at module load).
from price_predictor.backtest.perf_bench import (
    BenchConfig,
    CallCollector,
    ConfigResult,
    aggregate_speed,
    default_configs,
    format_comparison_table,
)

_RESULT_MARKER = "BENCH_RESULT_JSON:"


# ─────────────────────────────────────────────────────────────
# Shared arg plumbing
# ─────────────────────────────────────────────────────────────
def _default_dates() -> tuple[str, str]:
    """A safe past window whose horizons have all elapsed for grading.

    end = ~90 days ago (even monthly horizons have resolved), start 30
    days before that.
    """
    end = date.today() - timedelta(days=90)
    start = end - timedelta(days=30)
    return start.isoformat(), end.isoformat()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ollama config benchmark.")
    ds, de = _default_dates()
    p.add_argument("--tickers", nargs="+",
                   default=["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"])
    p.add_argument("--start", default=ds, help="ISO date (inclusive).")
    p.add_argument("--end", default=de, help="ISO date (inclusive).")
    p.add_argument("--stride", type=int, default=5,
                   help="Every Nth trading day (5 ~= weekly).")
    p.add_argument("--max-dates", type=int, default=6,
                   help="Cap dates for a quick smoke run (0 = no cap).")
    p.add_argument("--horizons", nargs="+", default=["daily", "weekly"])
    p.add_argument("--concurrency", type=int, default=1,
                   help="Keep 1 for clean CPU speed numbers.")
    p.add_argument("--models", nargs="+", default=["qwen3:8b", "qwen3:4b"])
    p.add_argument("--num-ctx", type=int, default=16384,
                   help="Held FIXED across configs so speed is comparable; "
                        "the tool recommends a lower value from observed prompts.")
    p.add_argument("--out", default="diagnostics",
                   help="Directory for the JSON + table report.")
    # Hidden worker flag: run exactly ONE config in this process.
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_label", default="", help=argparse.SUPPRESS)
    return p


# ─────────────────────────────────────────────────────────────
# WORKER: run one config in an isolated process, print JSON line
# ─────────────────────────────────────────────────────────────
def _feed(collector: CallCollector, response_obj, start_time, end_time) -> None:
    """Extract (prompt, completion, latency) from a litellm success event."""
    usage = getattr(response_obj, "usage", None)
    pt = getattr(usage, "prompt_tokens", 0) if usage else 0
    ct = getattr(usage, "completion_tokens", 0) if usage else 0
    try:
        lat = (end_time - start_time).total_seconds()
    except Exception:
        lat = 0.0
    collector.add(pt, ct, lat)


def _run_worker(args: argparse.Namespace) -> int:
    # Env (CHAIN_AGENTIC / OLLAMA_REASONING_EFFORT / OLLAMA_NUM_CTX /
    # USE_PAID) was set by the parent BEFORE this process imported
    # price_predictor, so settings + the import-time agents already
    # reflect this config.
    import litellm
    from litellm.integrations.custom_logger import CustomLogger

    from price_predictor.backtest.dates import trading_days_in_range
    from price_predictor.backtest.evaluation import evaluate_backtest
    from price_predictor.backtest.runner import run_backtest
    from price_predictor.config.settings import settings
    from price_predictor.prediction.schema import PredictionHorizon

    collector = CallCollector()

    class _Sink(CustomLogger):
        # Only ONE of these fires per call (litellm routes sync vs async
        # by the completion path), so no double counting. predict() is
        # async -> async_log_success_event is the live one.
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            _feed(collector, response_obj, start_time, end_time)

        async def async_log_success_event(
            self, kwargs, response_obj, start_time, end_time
        ):
            _feed(collector, response_obj, start_time, end_time)

    litellm.callbacks = [_Sink()]

    horizons = [PredictionHorizon(h) for h in args.horizons]
    dates = trading_days_in_range(
        date.fromisoformat(args.start),
        date.fromisoformat(args.end),
        stride=args.stride,
    )
    if args.max_dates and len(dates) > args.max_dates:
        dates = dates[: args.max_dates]

    cfg = BenchConfig(
        label=args._label,
        model=settings.effective_chain("agentic")[0],
        reasoning_effort=settings.ollama_reasoning_effort,
    )

    t0 = time.monotonic()
    run = asyncio.run(
        run_backtest(
            args.tickers, dates, horizons, concurrency=args.concurrency
        )
    )
    wall = time.monotonic() - t0

    ev = evaluate_backtest(run)
    speed = aggregate_speed(collector.snapshot())
    result = ConfigResult(
        config=cfg,
        n_predictions=ev.n_predictions,
        n_judged=ev.n_judged,
        direction_accuracy=ev.overall.direction_accuracy,
        hit_rate_resolved=ev.overall.hit_rate_resolved,
        brier_skill_score=ev.overall.brier_skill_score,
        speed=speed,
        wall_clock_s=wall,
        n_errors=len(run.errors),
    )
    # One machine-readable line the parent greps for.
    print(_RESULT_MARKER + json.dumps(result.as_dict()))
    return 0


# ─────────────────────────────────────────────────────────────
# ORCHESTRATOR: pull models, spawn one worker per config, render
# ─────────────────────────────────────────────────────────────
def _ollama(*cli_args: str) -> None:
    """Best-effort `ollama <args>` (pull/stop). Never fatal."""
    try:
        subprocess.run(["ollama", *cli_args], check=False)
    except Exception as exc:
        print(f"[bench] ollama {' '.join(cli_args)} failed: {exc}")


def _run_one_config(cfg: BenchConfig, args: argparse.Namespace) -> dict | None:
    """Spawn a worker subprocess for `cfg`; return its parsed JSON dict."""
    env = dict(os.environ)
    env.update({
        "CHAIN_AGENTIC": cfg.model,
        "OLLAMA_REASONING_EFFORT": cfg.reasoning_effort,
        "OLLAMA_NUM_CTX": str(args.num_ctx),
        "USE_PAID": "false",
    })
    cmd = [
        sys.executable, os.path.abspath(__file__), "--_worker",
        "--_label", cfg.label,
        "--tickers", *args.tickers,
        "--start", args.start, "--end", args.end,
        "--stride", str(args.stride), "--max-dates", str(args.max_dates),
        "--horizons", *args.horizons,
        "--concurrency", str(args.concurrency),
        "--num-ctx", str(args.num_ctx),
    ]
    proc = subprocess.run(env=env, args=cmd, capture_output=True, text=True)
    # Surface child logs so a failure is debuggable, then parse the marker.
    if proc.stderr.strip():
        sys.stderr.write(proc.stderr)
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER):])
    print(f"[bench] {cfg.label}: no result line (exit {proc.returncode}). "
          "Child stdout tail:")
    print("\n".join(proc.stdout.splitlines()[-15:]))
    return None


def _write_report(
    out_dir: str, results: list[ConfigResult], table: str
) -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    path = Path(out_dir) / f"ollama_bench_{ts}.json"
    payload = {
        "generated_utc": ts,
        "results": [r.as_dict() for r in results],
        "table": table,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _run_orchestrator(args: argparse.Namespace) -> int:
    configs = default_configs(tuple(args.models))
    print(f"[bench] {len(configs)} configs x "
          f"{len(args.tickers)} tickers, dates {args.start}..{args.end} "
          f"(stride {args.stride}, max {args.max_dates}), "
          f"horizons {args.horizons}\n")

    # Pull each unique model once, up front.
    for tag in dict.fromkeys(c.model_tag for c in configs):
        print(f"[bench] ensuring model '{tag}' is pulled...")
        _ollama("pull", tag)

    results: list[ConfigResult] = []
    for cfg in configs:
        print(f"\n=== running [{cfg.label}] "
              f"(model={cfg.model}, effort={cfg.reasoning_effort}) ===")
        d = _run_one_config(cfg, args)
        if d is not None:
            results.append(ConfigResult.from_dict(d))
        # Free the model from RAM before the next config so a fresh
        # config isn't measured against a swap-thrashing box.
        _ollama("stop", cfg.model_tag)

    if not results:
        print("[bench] No configs produced results. See child logs above.")
        return 1

    table = format_comparison_table(results)
    print("\n" + "=" * 72)
    print(table)
    print("=" * 72)
    report = _write_report(args.out, results, table)
    print(f"\n[bench] Full report written to {report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args._worker:
        return _run_worker(args)
    return _run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
