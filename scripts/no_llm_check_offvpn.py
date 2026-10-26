"""No-LLM check: does a plain rule on the SAME technicals beat what the models did?

The decisive test found every free model below the always-bullish / always-bearish baseline. Before blaming the
models or the prompt, this asks the cheaper question: what do simple, fixed rules score on the very same cases,
graded by the very same grader and judged by the very same statistics? No model call, no news call: only Yahoo prices.

Rules read the four cluster signals (trend, momentum, volatility, levels) the synthesizer was shown. If a rule
beats the models, the LLM adds nothing over the indicators. If no rule beats the baseline either, the indicators
themselves carry no signal in this window, and no prompt or model change can fix that.

HOW IT GRADES: the app's grader scores one bullish, one bearish and one neutral call per (case, horizon); a rule's
score is then which of those three it would have made. Direction correctness depends only on the direction called and
the realized return, so this is exactly what grading a rule-made prediction would give.

    uv run python scripts/no_llm_check_offvpn.py                        # the latest case grid, all of it
    uv run python scripts/no_llm_check_offvpn.py --restrict-to codestral-latest   # only cases that model answered

Writes reports/no_llm_check_<UTC>.md (and no_llm_check_latest.md).
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import model_scorecard_offvpn as MS
import scorecard_core as SC
import scorecard_stats as ST
from price_predictor.prediction.schema import (
    AnalysisBasis,
    Prediction,
    PredictionDirection,
    PredictionHorizon,
    PriceLevel,
)

D = PredictionDirection
IST = ZoneInfo("Asia/Kolkata")
CONCURRENCY = 4                        # technical-view downloads at once
LATEST_MD = "no_llm_check_latest.md"
SIGNAL_TO_DIRECTION = {"bullish": D.BULLISH, "bearish": D.BEARISH}
FLIP = {D.BULLISH: D.BEARISH, D.BEARISH: D.BULLISH, D.NEUTRAL: D.NEUTRAL}


# --- the rules: a technical view in, a direction out ---------------------------------------------------------------

def cluster(name: str) -> Callable:
    """Follow ONE cluster: bullish or bearish as it says, neutral otherwise."""
    return lambda tv: SIGNAL_TO_DIRECTION.get(getattr(tv, name).signal, D.NEUTRAL)


def majority(tv, tie: D = D.NEUTRAL) -> D:
    """More bullish clusters than bearish ones -> bullish, and the other way round; a tie gets `tie`."""
    signals = [tv.trend.signal, tv.momentum.signal, tv.volatility.signal, tv.levels.signal]
    bull, bear = signals.count("bullish"), signals.count("bearish")
    return D.BULLISH if bull > bear else D.BEARISH if bear > bull else tie


RULES: dict[str, Callable] = {
    "always bullish": lambda tv: D.BULLISH,
    "always bearish": lambda tv: D.BEARISH,
    "always neutral": lambda tv: D.NEUTRAL,
    "majority of the 4 clusters (tie = neutral)": majority,
    "majority of the 4 clusters (tie follows trend)": lambda tv: majority(tv, tie=cluster("trend")(tv)),
    "trend cluster alone": cluster("trend"),
    "momentum cluster alone": cluster("momentum"),
    "INVERSE of the majority (diagnostic only)": lambda tv: FLIP[majority(tv)],
}


# --- scoring: pure, no network -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Outcome:
    """What happened for one (case, horizon), and whether a call in each direction would have been right."""
    case: str
    horizon: PredictionHorizon
    ret: float
    right: dict[PredictionDirection, bool | None]       # None = could not be graded (no prices yet)


@dataclass(frozen=True)
class Row:
    name: str
    verdict: str
    k_moves: int
    n_moves: int
    low: float
    high: float
    k_calm: int
    n_calm: int
    neutral_share: float


def outcomes_from_graded(graded: list) -> list[Outcome]:
    """Fold the app grader's results (one per case, horizon and direction) into one Outcome per case and horizon."""
    folded: dict[tuple[str, PredictionHorizon], tuple[float, dict]] = {}
    for g in graded:
        p = g.prediction
        key = (f"{p.ticker}@{p.as_of.date().isoformat()}", p.horizon)
        _, right = folded.setdefault(key, (g.realized_return, {}))
        right[p.direction] = g.direction_correct
    return [Outcome(case, h, ret, right) for (case, h), (ret, right) in folded.items()]


def is_real_move(o: Outcome, tolerance: Callable[[PredictionHorizon], float]) -> bool:
    return ST.is_move(o.ret, tolerance(o.horizon))


def baseline_of(outcomes: list[Outcome], tolerance: Callable) -> tuple[float, int]:
    """What the best constant strategy scores on the real moves among `outcomes`, and how many outcomes that is."""
    moves = [(o.ret, tolerance(o.horizon)) for o in outcomes if o.right[D.BULLISH] is not None and is_real_move(o, tolerance)]
    return ST.baseline_accuracy(moves), len(moves)


def score_rule(name: str, picks: dict[tuple[str, PredictionHorizon], D], outcomes: list[Outcome], tolerance: Callable,
               baseline: float, baseline_n: int) -> Row:
    """Judge one rule exactly as a model is judged: its graded calls on real-move days, against the baseline."""
    moves = calm = k_moves = k_calm = neutral = 0
    for o in outcomes:
        direction = picks.get((o.case, o.horizon))
        if direction is None or o.right.get(direction) is None:
            continue
        neutral += direction is D.NEUTRAL
        right = bool(o.right[direction])
        if is_real_move(o, tolerance):
            moves, k_moves = moves + 1, k_moves + right
        else:
            calm, k_calm = calm + 1, k_calm + right
    verdict, low, high = ST.judge(k_moves, moves, baseline, baseline_n)
    return Row(name, verdict, k_moves, moves, low, high, k_calm, calm, neutral / ((moves + calm) or 1))


def render(rows: list[Row], baseline: float, baseline_n: int, meta: dict) -> str:
    head = "\n".join(f"- **{k}:** {v}" for k, v in meta.items())
    table = "\n".join(
        f"| {r.name} | {r.verdict} | {r.k_moves / r.n_moves:.0%} ({r.k_moves}/{r.n_moves}) | {r.low:.0%} to {r.high:.0%} | "
        f"{(r.k_calm / r.n_calm if r.n_calm else 0):.0%} ({r.k_calm}/{r.n_calm}) | {r.neutral_share:.0%} |"
        if r.n_moves else f"| {r.name} | {r.verdict} | n/a | n/a | n/a | {r.neutral_share:.0%} |" for r in rows)
    return (f"# No-LLM check\n\n{head}\n- **baseline:** {baseline:.0%} on {baseline_n} real-move outcomes (the best of always "
            f"bullish / always bearish)\n\n| rule | verdict | on real moves | 95% interval | on calm days | neutral calls |\n"
            f"|---|---|---|---|---|---|\n{table}\n\nA coin flip between bullish and bearish scores 50% on real moves by "
            "construction. BEATS_BASELINE needs the interval's LOWER end above the baseline. The INVERSE rule is a "
            "diagnostic chosen AFTER seeing results: if it scores well, that is a lead to test on new data, not a finding.\n")


# --- the live part: Yahoo prices only ----------------------------------------------------------------------------

def placeholder(ticker: str, case: SC.Case, horizon: PredictionHorizon, direction: D, tv) -> Prediction:
    """A valid prediction to hand the grader. Its levels are irrelevant to direction grading, so they are plain."""
    close, up = tv.close_price, direction is not D.BEARISH
    level = lambda x, why: PriceLevel(value=round(x, 2), rationale=why)       # noqa: E731
    return Prediction(
        ticker=ticker, as_of=datetime.combine(case.as_of, dtime(15, 30), tzinfo=IST), horizon=horizon,
        model_chain=("no-llm-check",), direction=direction, confidence=0.5, entry_zone=(close, close),
        target=level(close * (1.05 if up else 0.95), "placeholder"), stop_loss=level(close * (0.95 if up else 1.05), "placeholder"),
        rationale="Rule check: no model involved.", contributing_signals=("rule check",), conflicting_signals=("rule check",),
        analysis_basis=AnalysisBasis(close_price_at_prediction=close, bars_used=tv.bars_used, technical_summary="rule check",
                                     news_sentiment_score=None, news_articles_considered=0, filings_considered=0))


async def collect_views(cases: list[SC.Case], compose, resolve, say=print) -> tuple[dict[str, object], list[str]]:
    """The technical view as of each case date (what the synthesizer saw), a few at a time. -> (views, failures)."""
    views, failed, gate = {}, [], asyncio.Semaphore(CONCURRENCY)

    async def one(case: SC.Case) -> None:
        async with gate:
            try:
                views[case.key] = await compose(resolve(case.ticker), as_of=case.as_of)
            except Exception as e:
                failed.append(f"{case.key}: {type(e).__name__}: {str(e)[:120]}")
    await asyncio.gather(*(one(c) for c in cases))
    say(f"  technicals gathered for {len(views)} of {len(cases)} cases")
    return views, failed


def answered_cases(store: SC.CaseStore, model: str) -> set[str]:
    """Case keys a model produced an accepted answer for."""
    return {r.case for r in store.rows.values() if r.status == "OK" and r.model_uid.split("|")[-1] == model}


def run_check(cases: list[SC.Case], views: dict, horizons: list[PredictionHorizon], resolve, grade_many, tolerance: Callable
              ) -> tuple[list[Row], float, int, list[Outcome]]:
    usable = [c for c in cases if c.key in views]
    preds = [placeholder(resolve(c.ticker), c, h, d, views[c.key]) for c in usable for h in horizons for d in D]
    outcomes = outcomes_from_graded(grade_many(preds))
    baseline, baseline_n = baseline_of(outcomes, tolerance)
    picks = {name: {(c.key, h): rule(views[c.key]) for c in usable for h in horizons} for name, rule in RULES.items()}
    rows = [score_rule(name, picks[name], outcomes, tolerance, baseline, baseline_n) for name in RULES]
    return rows, baseline, baseline_n, outcomes


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", help="scorecard .jsonl whose case grid to use (default: the latest)")
    ap.add_argument("--horizons", default="daily,weekly")
    ap.add_argument("--restrict-to", metavar="MODEL", help="only the cases this model produced an accepted answer for")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from price_predictor.prediction import predictor as P
    from price_predictor.prediction.grading import grade_many, neutral_tolerance
    from price_predictor.prediction.inputs import compose_technical_view

    grid = Path(args.grid) if args.grid else MS.latest_scorecard(MS.REPORTS)
    if not grid or not grid.exists():
        print("No scorecard to take the case grid from. Run the decisive test (or the scorecard) first.")
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        store = SC.CaseStore(Path(tmp) / "grid.jsonl", (grid,))
    cases = store.cases or []
    if args.restrict_to:
        keep = answered_cases(store, args.restrict_to)
        cases = [c for c in cases if c.key in keep]
    if not cases:
        print("No cases to check.")
        return 2
    horizons = list(P._normalize_horizons([h.strip() for h in args.horizons.split(",") if h.strip()]))
    print(f"NO-LLM CHECK: {len(cases)} cases from {grid.name}, horizons {args.horizons}; Yahoo prices only, no model calls")
    views, failed = asyncio.run(collect_views(cases, compose_technical_view, P._resolve_ticker))
    rows, baseline, baseline_n, _ = run_check(cases, views, horizons, P._resolve_ticker, grade_many, neutral_tolerance)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    meta = {"run": stamp, "case grid": grid.name, "cases": f"{len(views)} of {len(cases)} had technicals"
            + (f"; restricted to cases {args.restrict_to} answered" if args.restrict_to else ""),
            "failed to gather": "; ".join(failed[:5]) or "none"}
    text = render(rows, baseline, baseline_n, meta)
    MS.REPORTS.mkdir(exist_ok=True)
    out = MS.REPORTS / f"no_llm_check_{stamp}.md"
    out.write_text(text, encoding="utf-8")
    shutil.copyfile(out, MS.REPORTS / LATEST_MD)
    print("\n" + text + f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
