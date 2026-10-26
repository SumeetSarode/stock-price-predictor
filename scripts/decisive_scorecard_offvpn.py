"""Decisive test: which free models are ACTUALLY accurate at our predictions, run in the order that wastes least.

A handful of cases cannot tell a 55% model from a 60% one; that takes about 100 graded real-move calls per model.
This runs that test in STAGES over one fixed case grid (any first K cases are representative), drops a model the
moment it is hopeless, stops spending on one the moment it is proven, and ends with one verdict per model.

    uv run python scripts/decisive_scorecard_offvpn.py --dry-run   # the order, the stages, the ETA; nothing runs
    uv run python scripts/decisive_scorecard_offvpn.py --probe     # connection checks + 6 real cases per model, then stop
    uv run python scripts/decisive_scorecard_offvpn.py             # the whole test; stop and re-run any time, it resumes

The ORDER comes from what earlier scorecards already showed: cheapest per valid answer first. A model that was already
hopeless (its best case below the baseline), or that a free key cannot serve at volume (its quota ran dry), is skipped
and the plan says why; --include-skipped overrides that. Everything is saved to reports/ as it happens.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import model_scorecard_offvpn as MS
import model_survey_offvpn as S
import scorecard_core as SC
import scorecard_stats as ST

RUN, SKIP = "RUN", "SKIP"
PROBE_CASES = 6
OVERHEAD_S = 90.0            # per case, once: technicals plus the news download (measured: two GDELT queries, ~70 s)
OWN_FAULT = {"rejected", "bad_output", "other"}      # the model's answer was refused; every other failure is the world's
DEFAULT_SEC = 40.0           # a model nobody has timed yet
QUOTA_SHARE = 0.5            # a model whose attempts were mostly rate-limited could not be served by a free key
PRIOR_MIN_JUDGED = 5         # fewest real-move calls before earlier evidence may rule a model out
REQUIRED_HOSTS = (S.REQUIRED_HOST, "api.gdeltproject.org")          # prices (also the VPN canary) and news
PROVIDER_HOSTS = {"google": "generativelanguage.googleapis.com", "mistral": "api.mistral.ai"}


@dataclass(frozen=True)
class Prior:
    """What an earlier scorecard already showed about one model."""
    name: str
    provider: str
    sec_per_case: float | None = None
    completion: float | None = None
    attempted: int = 0
    ok: int = 0
    rate_limited: int = 0
    k: int = 0
    n: int = 0
    baseline: float = 0.6


@dataclass(frozen=True)
class PlanRow:
    model: str
    provider: str
    action: str
    reason: str
    sec_per_case: float
    cost: float                # seconds per VALID answer: what a case really costs once failures are counted


@dataclass(frozen=True)
class Snapshot:
    scores: list
    baseline: float
    n_out: int
    no_inputs: list            # cases no model could be asked about (their news or prices never arrived)


def priors_from(scores: list, baseline: float) -> list[Prior]:
    return [Prior(s.cand.model, s.cand.provider, s.avg_seconds, s.reliability, s.attempted, s.ok,
                  s.fail_classes.get("rate_limited", 0), s.k_moves, s.n_moves, baseline) for s in scores]


def skip_reason(p: Prior) -> str:
    if p.attempted and p.rate_limited / p.attempted >= QUOTA_SHARE:
        return (f"quota-bound: a free key ran dry, only {p.ok} of {p.attempted} cases completed "
                f"({p.rate_limited} rate-limited); it cannot be tested or served at volume")
    if p.n >= PRIOR_MIN_JUDGED:
        high = ST.wilson(p.k, p.n)[1]
        if high < p.baseline:
            return f"already hopeless: {p.k}/{p.n} on real moves, its best case ({high:.0%}) is below the {p.baseline:.0%} baseline"
    return ""


def plan_models(priors: list[Prior], include_skipped: bool = False) -> list[PlanRow]:
    """Models to run, cheapest per valid answer first, then the skipped ones with the reason."""
    rows = []
    for p in priors:
        sec, why = p.sec_per_case or DEFAULT_SEC, skip_reason(p)
        skipped = bool(why) and not include_skipped
        rows.append(PlanRow(p.name, p.provider, SKIP if skipped else RUN, why or "no evidence against it", sec,
                            sec / max(p.completion if p.completion is not None else 1.0, 0.1)))
    return sorted((r for r in rows if r.action == RUN), key=lambda r: r.cost) + [r for r in rows if r.action == SKIP]


def stage_cuts(cases: int, floor: int = PROBE_CASES) -> list[int]:
    """Cumulative stage sizes: a screen, a middle look, then the whole grid; none smaller than the probe."""
    return sorted({min(cases, max(floor, round(cases * 0.27))), min(cases, max(floor, round(cases * 0.67))), cases})


def eta_minutes(rows: list[PlanRow], cases: int, overhead_s: float = OVERHEAD_S) -> float:
    """The next case's news downloads while this case's models answer, so a case costs the slower of the two."""
    return cases * max(overhead_s, sum(r.sec_per_case for r in rows if r.action == RUN)) / 60


def clock_time(minutes_from_now: float, now: datetime | None = None) -> str:
    return ((now or datetime.now().astimezone()) + timedelta(minutes=minutes_from_now)).strftime("%a %H:%M %Z")


def format_plan(rows: list[PlanRow], args: argparse.Namespace, cuts: list[int], partial: list[str]) -> str:
    n_move = round(args.cases * args.mover_share)
    mins = eta_minutes(rows, args.cases)
    lines = ["DECISIVE TEST PLAN",
             f"  grid: {args.cases} cases ({n_move} real moves, {args.cases - n_move} calm), {args.tickers} stocks, one case per move",
             f"  stages: {' -> '.join(str(c) for c in cuts)} cases; a model leaves when proven or hopeless "
             f"(interim looks use a 99% interval, the last a 95% one)",
             f"  needed for +/-{ST.TARGET_HALF_WIDTH:.0%}: about {ST.judged_needed()} graded real-move calls per model",
             "  priority (cheapest per valid answer first; within a case the models alternate by provider to hide pacing delays):"]
    n = 0
    for r in rows:
        n += r.action == RUN
        tag = f"{n:2d}." if r.action == RUN else " -."
        lines.append(f"    {tag} {r.action:4s} {r.provider:8s} {r.model:32s} {r.sec_per_case:4.0f}s/case  {r.reason}")
    lines += [f"     -. SKIP  {p}" for p in partial]
    lines.append(f"  time if no model is dropped: about {mins / 60:.1f} h (finishing around {clock_time(mins)}); less as models leave")
    return "\n".join(lines)


def partial_models(args: argparse.Namespace) -> list[str]:
    """Sweep models that passed only part of the time: never shortlisted, listed so the plan covers every model."""
    path = Path(args.from_sweep) if Path(args.from_sweep).is_absolute() else MS.REPORTS / args.from_sweep
    if not path.exists():
        return []
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [f"{r['provider']:8s} {r['model']:32s} passed only part of the sweep's runs, so it is not shortlisted"
            for r in rows if r["verdict"] == "PARTIAL"]


def stage_args(args: argparse.Namespace, alive: list[str], first_cases: int) -> argparse.Namespace:
    ns = MS.normalize(MS.build_parser().parse_args([]))
    vars(ns).update(cases=args.cases, mover_share=args.mover_share, tickers=args.tickers, horizons=args.horizons,
                    window_days=args.window_days, lag_days=args.lag_days, from_sweep=args.from_sweep,
                    only=",".join(alive), first_cases=first_cases)
    return ns


def prior_evidence(args: argparse.Namespace) -> list[Prior]:
    """The newest EARLIER scorecard (a different case grid), so the plan is the same however often this is re-run."""
    models, _ = MS.shortlist_for(stage_args(args, [], 0))
    for path in sorted(MS.REPORTS.glob("model_scorecard_2*.jsonl"), reverse=True):
        with tempfile.TemporaryDirectory() as tmp:
            store = SC.CaseStore(Path(tmp) / "old.jsonl", (path,))
            if store.picker == SC.PICKER or not store.rows:
                continue
            scores, baseline, _ = SC.aggregate(models, store)
        return priors_from(scores, baseline)
    return [Prior(m.model, m.provider) for m in models]


def score_latest(args: argparse.Namespace, names: list[str]) -> Snapshot:
    models, _ = MS.shortlist_for(stage_args(args, names, 0))
    with tempfile.TemporaryDirectory() as tmp:
        store = SC.CaseStore(Path(tmp) / "view.jsonl", (MS.latest_scorecard(MS.REPORTS),))
        scores, baseline, n_out = SC.aggregate(models, store)
        no_inputs = [c.key for c in store.cases or [] if ("*", c.key) in store.rows
                     and not any((m.uid, c.key) in store.rows for m in models)]
    return Snapshot(scores, baseline, n_out, no_inputs)


def check_hosts(providers: set[str], say=print, reachable=S.host_reachable, sleep=time.sleep) -> tuple[list[str], set[str]]:
    """-> (problems that stop everything, providers that cannot be reached right now)."""
    problems, down = [], set()
    for host in REQUIRED_HOSTS:
        for attempt in range(3):
            ok, why = reachable(host)
            if ok or attempt == 2:
                break
            sleep(10)
        say(f"  {host:34s} {'ok' if ok else why}")
        if not ok:
            problems.append(f"{host} unreachable ({why})" + ("; the VPN looks ON, disconnect it" if host == S.REQUIRED_HOST else ""))
    for p in sorted(providers):
        host = PROVIDER_HOSTS.get(p)
        if host:
            ok, why = reachable(host)
            say(f"  {host:34s} {'ok' if ok else why}")
            if not ok:
                down.add(p)
    return problems, down


def verdict_table(snap: Snapshot, verdicts: dict[str, str]) -> str:
    rows = ["| model | now | on real moves | 95% interval | on calm days | completed |", "|---|---|---|---|---|---|"]
    for s in snap.scores:
        if s.cand.model not in verdicts:
            continue
        lo, hi = ST.wilson(s.k_moves, s.n_moves) if s.n_moves else (0.0, 1.0)
        acc = f"{s.k_moves / s.n_moves:.0%} ({s.k_moves}/{s.n_moves})" if s.n_moves else "-"
        calm = f"{s.k_calm / s.n_calm:.0%} ({s.k_calm}/{s.n_calm})" if s.n_calm else "-"
        rows.append(f"| {s.cand.provider}/{s.cand.model} | {verdicts[s.cand.model]} | {acc} | "
                    f"{f'{lo:.0%} to {hi:.0%}' if s.n_moves else '-'} | {calm} | {s.ok}/{s.attempted} |")
    return "\n".join(rows)


def recommend(verdicts: dict[str, str], snap: Snapshot) -> str:
    low = {s.cand.model: ST.wilson(s.k_moves, s.n_moves)[0] if s.n_moves else 0.0 for s in snap.scores}
    proven = sorted((m for m, v in verdicts.items() if v == ST.BEATS), key=lambda m: -low[m])
    if proven:
        return (f"{', '.join(proven)} beat{'s' if len(proven) == 1 else ''} the {snap.baseline:.0%} baseline with confidence. "
                f"Put {proven[0]} first in CHAIN_AGENTIC.")
    undecided = [m for m, v in verdicts.items() if v in (ST.CANT_TELL, ST.TOO_FEW, ST.KEEP)]
    if undecided:
        return f"Not decided yet: {', '.join(undecided)} need more cases. Run again; it resumes."
    return (f"No model shows skill beyond a free constant strategy ({snap.baseline:.0%}, always bullish or always bearish). "
            "Choose by reliability and speed, and expect more from improving the pipeline (prompts, rules) than from swapping models.")


def write_report(rows: list[PlanRow], snap: Snapshot, verdicts: dict[str, str], stage: int, args: argparse.Namespace) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=MS.REPO, capture_output=True, text=True).stdout.strip()
    skipped = [f"- {r.provider}/{r.model}: {r.reason}" for r in rows if r.action == SKIP]
    text = "\n".join([
        "# Decisive model test", "",
        f"- **run:** {stamp} (git {head}); grid of {args.cases} cases, last stage reached: {stage}",
        f"- **baseline:** {snap.baseline:.0%} on {snap.n_out} real-move outcomes (the best of always bullish / always bearish)",
        f"- **cases whose inputs never arrived:** {len(snap.no_inputs)}", "",
        "## Verdicts", "", verdict_table(snap, verdicts), "",
        "NO_EDGE = enough data to say it is not better than the baseline. BEATS_BASELINE = proven. "
        "WORSE_THAN_BASELINE = hopeless. KEEP_TESTING / CANT_TELL_YET = needs more cases.", "",
        "## Recommendation", "", recommend(verdicts, snap), "",
        "## Not run, and why", "", *(skipped or ["- none"]), ""])
    MS.REPORTS.mkdir(exist_ok=True)
    path = MS.REPORTS / f"decisive_verdict_{stamp}.md"
    path.write_text(text, encoding="utf-8")
    (MS.REPORTS / "decisive_verdict_latest.md").write_text(text, encoding="utf-8")
    return path


def probe_findings(snap: Snapshot, args: argparse.Namespace, plan_secs: dict[str, float]) -> tuple[bool, str]:
    """Did the probe cases get real inputs and real answers, and how many graded answers will the full grid yield?

    -> (ok, what to tell the user). It only FAILS for a broken set-up (inputs never arrived, or nobody answered at
    all). A model whose answers are often refused by the app's own rules is not broken: the projection says how short
    of the ~97 graded real-move calls it will end up, which is the thing that decides whether the test can be decisive.
    """
    need, move_cases = ST.judged_needed(), args.cases * args.mover_share
    secs = {s.cand.model: s.avg_seconds or plan_secs.get(s.cand.model, DEFAULT_SEC) for s in snap.scores}
    per_case = max(OVERHEAD_S, sum(secs.values()))
    lines = [f"  probe: {args.probe_cases} real cases, {len(snap.scores)} models; about {per_case:.0f} s per case "
             f"(news {OVERHEAD_S:.0f} s overlapping the models), so {args.cases} cases is about "
             f"{args.cases * per_case / 3600:.1f} h before models leave"]
    for s in snap.scores:
        own = sum(n for c, n in s.fail_classes.items() if c in OWN_FAULT)
        completion = s.ok / (s.ok + own) if s.ok + own else None
        world = sum(s.fail_classes.values()) - own
        proj = f"projects ~{move_cases * completion:.0f} of {need} needed" if completion is not None else "no answer or refusal yet"
        lines.append(f"    {s.cand.provider:8s} {s.cand.model:32s} answered {s.ok}/{s.attempted}  refused {own}  "
                     f"world-failures {world}  {proj}")
    ok = not snap.no_inputs and any(s.ok for s in snap.scores)
    if snap.no_inputs:
        lines.append(f"  PROBE PROBLEM: news or prices never arrived for {snap.no_inputs}")
    elif not ok:
        lines.append("  PROBE PROBLEM: no model produced an answer the app accepted")
    return ok, "\n".join(lines)


async def run(args: argparse.Namespace, *, amain=MS.amain, priors_fn=prior_evidence, score_fn=score_latest,
              hosts_fn=check_hosts, partial_fn=partial_models, say=print) -> int:
    rows = plan_models(priors_fn(args), include_skipped=args.include_skipped)
    alive = [r.model for r in rows if r.action == RUN]
    cuts = stage_cuts(args.cases, args.probe_cases)
    say(format_plan(rows, args, cuts, partial_fn(args)))
    if args.dry_run or not alive:
        return 0 if alive else 2
    say("\nCONNECTIONS")
    problems, down = hosts_fn({r.provider for r in rows if r.action == RUN}, say)
    if problems:
        say("\nSTOPPED before spending anything:\n  " + "\n  ".join(problems))
        return 2
    gone = [r.model for r in rows if r.provider in down and r.model in alive]
    if gone:
        say(f"  not reachable now, left out of this run: {', '.join(gone)}")
        alive = [m for m in alive if m not in gone]
    planned = list(alive)

    code = await amain(stage_args(args, alive, args.probe_cases))
    if code:
        return code
    ok, text = probe_findings(score_fn(args, planned), args, {r.model: r.sec_per_case for r in rows})
    say("\n" + text)
    if args.probe:
        say("\nPROBE OK: run again without --probe for the whole test." if ok else "\nPROBE FAILED: fix the above first.")
        return 0 if ok else 3
    if not ok:
        say("\nSTOPPED: the probe cases did not work (see above); nothing more was spent.")
        return 3

    verdicts: dict[str, str] = {}
    for i, k in enumerate(cuts):
        final = i == len(cuts) - 1
        say(f"\nSTAGE {i + 1}/{len(cuts)}: first {k} cases, models {', '.join(alive)}")
        code = await amain(stage_args(args, alive, k))
        if code:
            return code
        snap = score_fn(args, planned)
        for s in snap.scores:
            if s.cand.model in alive:
                verdicts[s.cand.model] = ST.stage_verdict(s.k_moves, s.n_moves, snap.baseline, snap.n_out, final=final)
        say(verdict_table(snap, verdicts))
        alive = [m for m in alive if verdicts.get(m) == ST.KEEP]
        if not alive:
            break
    path = write_report(rows, snap, verdicts, i + 1, args)
    say(f"\nVERDICT\n  {recommend(verdicts, snap)}\n  report: {path}\n  latest: {MS.REPORTS / 'decisive_verdict_latest.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=int, default=150, help="size of the case grid (default 150)")
    ap.add_argument("--mover-share", type=float, default=0.73, help="share of cases that are real moves (default 0.73)")
    ap.add_argument("--tickers", default="nifty50")
    ap.add_argument("--horizons", default="daily,weekly")
    ap.add_argument("--window-days", type=int, default=70)
    ap.add_argument("--lag-days", type=int, default=14)
    ap.add_argument("--from-sweep", default="free_model_sweep_latest_verdicts.jsonl")
    ap.add_argument("--include-skipped", action="store_true", help="also run models earlier evidence ruled out")
    ap.add_argument("--probe", action="store_true", help="connection checks + real cases per model, then stop")
    ap.add_argument("--probe-cases", type=int, default=PROBE_CASES, help="cases the probe runs (default 6)")
    ap.add_argument("--dry-run", action="store_true", help="show the plan and ETA, run nothing")
    return ap


def main() -> int:
    try:
        return asyncio.run(run(build_parser().parse_args()))
    except KeyboardInterrupt:
        print("\nINTERRUPTED: everything so far is saved; run the same command again to continue.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
