"""Model scorecard: of the models that WORK in our pipeline, which are actually any GOOD?

The sweep (free_model_sweep_offvpn.py) answers "does it run?". This answers "is it right?": it takes the sweep's
USEFUL models, runs each over the SAME historical cases (same technicals, same cached news), has the app's own
grader score the predictions against what the price really did, and ranks models by the worst plausible accuracy.

Normally you do not run this directly: run_offvpn.py does it after the sweep. But you can:

    uv run python scripts/model_scorecard_offvpn.py            # run (or continue) the scorecard
    uv run python scripts/model_scorecard_offvpn.py --dry-run  # the plan, no network

It resumes automatically and re-uses the same cases, so every run adds evidence. Output:
reports/model_scorecard_<UTC>.md (and model_scorecard_latest.md) plus the raw .jsonl that --resume reads.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import time
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path

import free_model_sweep_offvpn as FM
import model_survey_offvpn as S
import scorecard_core as SC
import scorecard_report as REP
import sweep_data as D
import sweep_probe as PR
import sweep_sources as SRC

DEFAULT_TICKERS = "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS"
SCORABLE_ROUTES = ("direct", "anon", "strata")     # the app can call these itself; gateway-only models cannot be scored
CALLS_PER_CASE = 6          # news-impact (a few tool turns) + one synthesizer call per horizon, rounded up
LATEST_MD = "model_scorecard_latest.md"
REPO, REPORTS = S.REPO, S.REPORTS


def load_shortlist(verdicts: Path, keys: dict[str, str], top_n: int) -> tuple[list[D.Candidate], list[str]]:
    """The sweep's USEFUL models the app can call directly, spread across providers; plus what was left out and why."""
    rows = [json.loads(ln) for ln in verdicts.read_text(encoding="utf-8").splitlines() if ln.strip()]
    useful = [r for r in rows if r["verdict"] == "USEFUL"]
    scorable = [D.Candidate.from_row(r, keys) for r in useful if r["route"] in SCORABLE_ROUTES]
    left_out = [f"{r['provider']}/{r['model']} ({r['route']}: reachable only through a gateway)"
                for r in useful if r["route"] not in SCORABLE_ROUTES]
    return SC.interleave(scorable)[:top_n], left_out


def choose_days(today: date, lag_days: int, window_days: int) -> list[date]:
    """Trading days far enough back that every horizon has resolved, recent enough that news archives still reach."""
    from price_predictor.backtest.dates import trading_days_in_range
    end = today - timedelta(days=lag_days)
    return trading_days_in_range(end - timedelta(days=window_days), end)


class Live:
    """The real app, behind the Plumbing interface. Every collaborator is injectable so tests need no network."""

    def __init__(self, P, horizons: list, timeout: float = S.STEP_TIMEOUT_S, compose=None, make_input=None) -> None:
        self.P, self.horizons, self.timeout = P, horizons, timeout
        self._compose, self._make_input = compose, make_input

    async def shared(self, case: SC.Case):
        """Technicals as of the case date, gathered ONCE and handed to every model."""
        compose = self._compose
        if compose is None:
            from price_predictor.prediction.inputs import compose_technical_view as compose
        return await compose(self.P._resolve_ticker(case.ticker), as_of=case.as_of)

    def _input(self, **kw):
        if self._make_input:
            return self._make_input(**kw)
        from price_predictor.prediction.inputs import SynthesisInput
        return SynthesisInput(**kw)

    async def predict(self, cand: D.Candidate, case: SC.Case, tv) -> list:
        """The production two-step on ONE model: news-impact under replay, then a synthesizer call per horizon."""
        P = self.P
        model = S.build_model(FM.group_for(cand), cand.test_id)
        P._news_impact_agent.model = model
        P._synthesizer_agent.model = model
        P._ensure_news_snapshot_installed()
        canonical = P._resolve_ticker(case.ticker)
        with P.replay_context(case.as_of):               # point-in-time news, served from the snapshot store
            impact = await asyncio.wait_for(P.run_news_impact_agent(canonical), self.timeout)
        as_of_dt = datetime.combine(case.as_of, dtime(15, 30), tzinfo=P.IST)
        out = []
        for h in self.horizons:
            si = self._input(ticker=canonical, horizon=h.value, as_of=as_of_dt, technical_view=tv,
                             impact_assessment=impact, model_chain=(P._NEWS_BACKTEST_REPLAY_TAG,))
            pred = await asyncio.wait_for(P.synthesize_with_guardrails(si), self.timeout)
            out.append(P._finalize_prediction(pred, news_degraded=False))
        return out


def plan(args: argparse.Namespace) -> str:
    return "\n".join([
        "SCORECARD PLAN",
        f"  models: up to {args.models} USEFUL models from the latest sweep ({args.from_sweep})",
        f"  cases: {args.cases} (tickers {args.tickers}), spread over the last {args.window_days} days ending "
        f"{args.lag_days} days ago; the same cases every run, so results accumulate",
        f"  horizons: {args.horizons}; about {CALLS_PER_CASE} LLM calls per model per case",
        f"  budget: {args.calls_per_provider} calls per provider per run (~{args.calls_per_provider // CALLS_PER_CASE} cases); "
        "re-running continues automatically",
        "  inputs: technicals gathered once per case; news cached on first fetch and replayed for every later model"])


async def amain(args: argparse.Namespace, live_factory=None) -> int:
    print(plan(args))
    if args.dry_run:
        print("\n--dry-run: nothing started, no network calls.")
        return 0
    verdicts = Path(args.from_sweep) if Path(args.from_sweep).is_absolute() else REPORTS / args.from_sweep
    if not verdicts.exists():
        print(f"\nNo sweep results at {verdicts}. Run the sweep first (run_offvpn.py does it for you).")
        return 2
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    REPORTS.mkdir(exist_ok=True)
    from dotenv import dotenv_values
    keys = SRC.read_env_keys(dotenv_values(REPO / ".env"), [p for p, s in D.SPECS.items() if not s.anonymous])
    models, left_out = load_shortlist(verdicts, keys, args.models)
    resume = [f] if args.resume and (f := latest_scorecard(REPORTS)) else []
    store = SC.CaseStore(REPORTS / f"model_scorecard_{stamp}.jsonl", tuple(resume))
    started = time.monotonic()
    meta = {"started_utc": stamp, "git_head": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                                              capture_output=True, text=True).stdout.strip(),
            "horizons": args.horizons, "models scored": len(models), "resumed_from": resume[0].name if resume else "-",
            "not scored": "; ".join(left_out) or "-", "outcome": "in progress"}

    def write(outcome: str) -> Path:
        meta["outcome"] = outcome
        meta["minutes"] = round((time.monotonic() - started) / 60, 1)
        scores, baseline, n_out = SC.aggregate(models, store)
        md = REPORTS / f"model_scorecard_{stamp}.md"
        md.write_text(REP.render(scores, baseline, n_out, store.cases or [], meta), encoding="utf-8")
        shutil.copyfile(md, REPORTS / LATEST_MD)
        return md

    code = 0
    try:
        if not store.cases:
            days = choose_days(date.today(), args.lag_days, args.window_days)
            store.set_cases(SC.pick_cases([t for t in args.tickers.split(",") if t], days, args.cases))
        if models and store.cases:
            live = (live_factory or make_live)(args)
            plumb = SC.Plumbing(live.shared, live.predict, FM.pace_for, S.classify)
            deadline = started + args.max_minutes * 60 if args.max_minutes else float("inf")
            budget = PR.Budget({}, args.calls_per_provider)
            await SC.run_scorecard(models, store.cases, store, budget, plumb, calls_per_case=CALLS_PER_CASE,
                                   deadline=deadline)
        outcome = "completed" if models else "nothing to score yet: the sweep found no USEFUL model the app can call"
    except KeyboardInterrupt:
        outcome, code = "INTERRUPTED (partial results saved; the next run continues)", 130
    except Exception as e:
        outcome, code = f"CRASHED: {type(e).__name__}: {e} (partial results saved; the next run continues)", 1
    md = write(outcome)
    print(f"\n{outcome}\nscorecard: {md}\nlatest:    {REPORTS / LATEST_MD}")
    return code


def make_live(args: argparse.Namespace) -> Live:
    from loguru import logger
    logger.remove()
    from price_predictor.prediction import predictor as P
    return Live(P, P._normalize_horizons([h.strip() for h in args.horizons.split(",") if h.strip()]))


def latest_scorecard(reports: Path) -> Path | None:
    files = sorted(reports.glob("model_scorecard_*.jsonl"))
    return files[-1] if files else None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=int, default=12, help="score at most this many models (default 12)")
    ap.add_argument("--cases", type=int, default=20, help="size of the fixed case grid (default 20)")
    ap.add_argument("--tickers", default=DEFAULT_TICKERS)
    ap.add_argument("--horizons", default="daily,weekly")
    ap.add_argument("--window-days", type=int, default=70, help="how far back the cases reach (news archives are ~3 months)")
    ap.add_argument("--lag-days", type=int, default=14, help="newest case is this old, so every horizon has resolved")
    ap.add_argument("--calls-per-provider", type=int, default=120, help="LLM-call budget per provider per run")
    ap.add_argument("--from-sweep", default="free_model_sweep_latest_verdicts.jsonl")
    ap.add_argument("--resume", default="latest")
    ap.add_argument("--fresh", action="store_true", help="ignore earlier results and start over")
    ap.add_argument("--max-minutes", type=int, default=240)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def normalize(args: argparse.Namespace) -> argparse.Namespace:
    args.resume = None if args.fresh else args.resume
    return args


def main() -> int:
    return asyncio.run(amain(normalize(build_parser().parse_args())))


if __name__ == "__main__":
    raise SystemExit(main())
