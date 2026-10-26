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
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
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
NEWS_ATTEMPTS, NEWS_RETRY_S, NEWS_TIMEOUT_S = 3, 20.0, 240.0     # GDELT answers one request per 5 s, and 429s often
LATEST_MD = "model_scorecard_latest.md"
REPO, REPORTS = S.REPO, S.REPORTS
GUARDRAIL = re.compile(r"\[(grounding|consistency)")      # the app's own rules refused the model's answer 3 times


def classify_case(message: str) -> str:
    """Why a (model, case) failed. A prediction the app's rules REFUSED is the model's own bad output: final, not
    worth retrying (it would only cost time to be refused again); everything else is classified as in the sweep."""
    return "rejected" if GUARDRAIL.search(message) else S.classify(message)


def load_shortlist(verdicts: Path, keys: dict[str, str], top_n: int) -> tuple[list[D.Candidate], list[str]]:
    """The sweep's USEFUL models the app can call directly, spread across providers; plus what was left out and why."""
    rows = [json.loads(ln) for ln in verdicts.read_text(encoding="utf-8").splitlines() if ln.strip()]
    useful = [r for r in rows if r["verdict"] == "USEFUL"]
    scorable = [D.Candidate.from_row(r, keys) for r in useful if r["route"] in SCORABLE_ROUTES]
    left_out = [f"{r['provider']}/{r['model']} ({r['route']}: reachable only through a gateway)"
                for r in useful if r["route"] not in SCORABLE_ROUTES]
    return SC.interleave(scorable)[:top_n], left_out


def universe() -> list[str]:
    """All 50 NIFTY stocks the app knows (with company names, which is what news is searched by)."""
    data = json.loads((REPO / "data" / "kb" / "stocks.json").read_text(encoding="utf-8"))
    return [f"{row['ticker']}.NS" for row in (data if isinstance(data, list) else data["stocks"])]


def parse_tickers(spec: str) -> list[str]:
    return universe() if spec.strip().lower() == "nifty50" else [t.strip() for t in spec.split(",") if t.strip()]


def choose_days(today: date, lag_days: int, window_days: int) -> list[date]:
    """Trading days far enough back that every horizon has resolved, recent enough that news archives still reach."""
    from price_predictor.backtest.dates import trading_days_in_range
    end = today - timedelta(days=lag_days)
    return trading_days_in_range(end - timedelta(days=window_days), end)


def live_moves(tickers: list[str], days: list[date], horizons: str) -> dict[tuple[str, date], float]:
    """How far each ticker's price really moved after each day, from the app's own price chain and its own
    horizon lengths and tolerances (one source of truth: the grader), so "a real move" means what grading means."""
    from price_predictor.data.prices import fetch_ohlcv
    from price_predictor.prediction.grading import horizon_window, neutral_tolerance
    from price_predictor.prediction.schema import PredictionHorizon
    hs = [PredictionHorizon(h.strip()) for h in horizons.split(",") if h.strip()]
    windows = [(horizon_window(h), neutral_tolerance(h)) for h in hs]
    reach = timedelta(days=int(max(w for w, _ in windows) * 1.7) + 7)       # calendar days that cover the longest horizon
    out: dict[tuple[str, date], float] = {}
    for ticker in tickers:
        bars = fetch_ohlcv(ticker, days[0] - timedelta(days=7), days[-1] + reach)
        closes = [(ts.date(), float(c)) for ts, c in bars["close"].items() if c == c]       # c == c drops NaN
        out.update(SC.case_moves(ticker, closes, days, windows))
    return out


class NewsUnavailable(Exception):
    """The news for a case could not be fetched; the case is skipped (and retried next run), nobody's fault."""


class Live:
    """The real app, behind the Plumbing interface. Every collaborator is injectable so tests need no network.

    `gather` is the app's news-input gatherer; with it, each case's news is fetched ONCE and every model is handed
    those very inputs (None = leave the app to fetch for itself, which unit tests of other behaviour use). Without
    this, a throttled news source could give the first model a whole week of headlines and the next model none,
    and the app would quietly answer the latter with a neutral stub: not the same exam.
    """

    def __init__(self, P, horizons: list, timeout: float = S.STEP_TIMEOUT_S, compose=None, make_input=None,
                 gather=None, sleep=asyncio.sleep) -> None:
        self.P, self.horizons, self.timeout = P, horizons, timeout
        self._compose, self._make_input, self._gather, self._sleep = compose, make_input, gather, sleep
        self._news: dict[str, object] = {}

    async def shared(self, case: SC.Case):
        """Technicals as of the case date, and the case's news, each gathered ONCE and handed to every model."""
        compose = self._compose
        if compose is None:
            from price_predictor.prediction.inputs import compose_technical_view as compose
        tv = await compose(self.P._resolve_ticker(case.ticker), as_of=case.as_of)
        await self._pin_news(case)
        return tv

    async def _pin_news(self, case: SC.Case) -> None:
        if self._gather is None or case.key in self._news:
            return
        self.P._ensure_news_snapshot_installed()
        ticker = self.P._resolve_ticker(case.ticker)
        for attempt in range(1, NEWS_ATTEMPTS + 1):
            with self.P.replay_context(case.as_of):
                inputs = await asyncio.wait_for(self._gather(ticker), NEWS_TIMEOUT_S)
            failed = [e for e in inputs.errors if "news unavailable" in e]       # "no news" is fine, "news failed" is not
            if not failed:
                self._news[case.key] = inputs
                return
            if attempt < NEWS_ATTEMPTS:
                await self._sleep(NEWS_RETRY_S)
        raise NewsUnavailable(f"news failed {NEWS_ATTEMPTS} times: {failed[0][:200]}")

    @contextmanager
    def _pinned(self, case: SC.Case):
        """For the duration, the app's news gatherer returns this case's pinned inputs instead of fetching again."""
        inputs = self._news.get(case.key)
        if inputs is None:
            yield
            return

        async def same(ticker, **kw):
            return inputs
        original = self.P.gather_news_impact_inputs
        self.P.gather_news_impact_inputs = same
        try:
            yield
        finally:
            self.P.gather_news_impact_inputs = original

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
        with P.replay_context(case.as_of), self._pinned(case):      # point-in-time news, the same for every model
            impact = await asyncio.wait_for(P.run_news_impact_agent(canonical), self.timeout)
        as_of_dt = datetime.combine(case.as_of, dtime(15, 30), tzinfo=P.IST)
        out = []
        for h in self.horizons:
            si = self._input(ticker=canonical, horizon=h.value, as_of=as_of_dt, technical_view=tv,
                             impact_assessment=impact, model_chain=(P._NEWS_BACKTEST_REPLAY_TAG,))
            pred = await asyncio.wait_for(P.synthesize_with_guardrails(si), self.timeout)
            out.append(P._finalize_prediction(pred, news_degraded=False))
        return out


def calls_per_model(args: argparse.Namespace) -> int:
    """Default: exactly enough for one model to see every case this run covers."""
    return args.calls_per_model or (args.first_cases or args.cases) * CALLS_PER_CASE


def plan(args: argparse.Namespace) -> str:
    return "\n".join([
        "SCORECARD PLAN",
        f"  models: {('only ' + args.only) if args.only else f'up to {args.models} USEFUL models'} "
        f"from the latest sweep ({args.from_sweep})",
        f"  cases: {args.cases} (tickers {args.tickers}), spread over the last {args.window_days} days ending "
        f"{args.lag_days} days ago; {args.mover_share:.0%} are days the price really moved, the rest calm, one case per "
        "move; the same cases every run, so results accumulate"
        + (f"; this run covers the first {args.first_cases}" if args.first_cases else ""),
        f"  horizons: {args.horizons}; about {CALLS_PER_CASE} LLM calls per model per case",
        f"  budget: {calls_per_model(args)} calls per MODEL per run (~{calls_per_model(args) // CALLS_PER_CASE} cases); "
        "re-running continues automatically",
        "  inputs: technicals gathered once per case; news cached on first fetch and replayed for every later model"])


def shortlist_for(args: argparse.Namespace) -> tuple[list[D.Candidate], list[str]]:
    """The models this run covers: the sweep's USEFUL ones, narrowed by --only. FileNotFoundError if no sweep ran."""
    verdicts = Path(args.from_sweep) if Path(args.from_sweep).is_absolute() else REPORTS / args.from_sweep
    if not verdicts.exists():
        raise FileNotFoundError(f"No sweep results at {verdicts}. Run the sweep first (run_offvpn.py does it for you).")
    from dotenv import dotenv_values
    keys = SRC.read_env_keys(dotenv_values(REPO / ".env"), [p for p, s in D.SPECS.items() if not s.anonymous])
    models, left_out = load_shortlist(verdicts, keys, args.models)
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        models = [m for m in models if m.model in wanted]
    return models, left_out


async def amain(args: argparse.Namespace, live_factory=None, moves_fn=None) -> int:
    print(plan(args))
    if args.dry_run:
        print("\n--dry-run: nothing started, no network calls.")
        return 0
    try:
        models, left_out = shortlist_for(args)
    except FileNotFoundError as e:
        print(f"\n{e}")
        return 2
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    REPORTS.mkdir(exist_ok=True)
    resume = [f] if args.resume and (f := latest_scorecard(REPORTS)) else []
    store = SC.CaseStore(REPORTS / f"model_scorecard_{stamp}.jsonl", tuple(resume))
    if store.cases and store.picker != SC.PICKER:
        print(f"\nThe earlier cases ({resume[0].name}) were chosen by an older method (not one independent real move "
              "per case), so they cannot be mixed with the new ones; starting a new case grid. The old file is "
              "left untouched.")
        store.path.unlink()                 # the constructor already copied the old history into this file
        store, resume = SC.CaseStore(store.path, ()), []
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
            tickers = parse_tickers(args.tickers)
            try:
                moves = (moves_fn or live_moves)(tickers, days, args.horizons)
            except Exception as e:
                print(f"\nWARNING: could not look up how far prices moved ({type(e).__name__}: {e}). Cases are spread "
                      "evenly instead, which mostly picks calm days and cannot rank models well; run again later.")
                moves = None
            store.set_cases(SC.pick_cases(tickers, days, args.cases, moves, args.mover_share),
                            SC.PICKER if moves else "spread")
        if models and store.cases:
            live = (live_factory or make_live)(args)
            plumb = SC.Plumbing(live.shared, live.predict, FM.pace_for, classify_case)
            deadline = started + args.max_minutes * 60 if args.max_minutes else float("inf")
            budget = PR.Budget({}, calls_per_model(args))
            await SC.run_scorecard(models, store.cases[:args.first_cases] if args.first_cases else store.cases,
                                   store, budget, plumb, calls_per_case=CALLS_PER_CASE, deadline=deadline)
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
    return Live(P, P._normalize_horizons([h.strip() for h in args.horizons.split(",") if h.strip()]),
                gather=P.gather_news_impact_inputs)


def latest_scorecard(reports: Path) -> Path | None:
    files = sorted(reports.glob("model_scorecard_*.jsonl"))
    return files[-1] if files else None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=int, default=12, help="score at most this many models (default 12)")
    ap.add_argument("--cases", type=int, default=30, help="size of the fixed case grid (default 30)")
    ap.add_argument("--tickers", default=DEFAULT_TICKERS, help="comma list, or 'nifty50' for all 50 the app knows")
    ap.add_argument("--mover-share", type=float, default=0.5, help="share of cases that are real moves (default 0.5)")
    ap.add_argument("--only", default="", help="comma list of model names to run (default: every shortlisted model)")
    ap.add_argument("--first-cases", type=int, default=0,
                    help="run only the first N cases of the grid (it is ordered so any prefix is representative)")
    ap.add_argument("--horizons", default="daily,weekly")
    ap.add_argument("--window-days", type=int, default=70, help="how far back the cases reach (news archives are ~3 months)")
    ap.add_argument("--lag-days", type=int, default=14, help="newest case is this old, so every horizon has resolved")
    ap.add_argument("--calls-per-model", type=int, default=0,
                    help="LLM-call budget per MODEL per run (default 0 = enough for the whole case grid)")
    ap.add_argument("--from-sweep", default="free_model_sweep_latest_verdicts.jsonl")
    ap.add_argument("--resume", default="latest")
    ap.add_argument("--fresh", action="store_true", help="ignore earlier results and start over")
    ap.add_argument("--max-minutes", type=int, default=0,
                    help="stop starting new cases after this many minutes (default 0 = no limit)")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def normalize(args: argparse.Namespace) -> argparse.Namespace:
    args.resume = None if args.fresh else args.resume
    return args


def main() -> int:
    return asyncio.run(amain(normalize(build_parser().parse_args())))


if __name__ == "__main__":
    raise SystemExit(main())
