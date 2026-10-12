"""The scorecard report: plain-English, ranked by the WORST plausible accuracy, never by a lucky headline number."""
from __future__ import annotations

from collections import Counter

import scorecard_stats as ST
from scorecard_core import Case, ModelScore
from sweep_report import how_to_add

MIN_RELIABILITY = 0.8      # a model that fails a fifth of the cases is not chain material, however accurate


def pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.0f}%"


def _table(head: tuple[str, ...], rows: list[tuple]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head), *("| " + " | ".join(map(str, r)) + " |" for r in rows)]


def chain_order(scores: list[ModelScore]) -> list[ModelScore]:
    """Provisional chain order: proven first, then undecided models that at least run reliably. Never WORSE/TOO_FEW."""
    keep = [s for s in scores if s.verdict == ST.BEATS or (s.verdict == ST.CANT_TELL and (s.reliability or 0) >= MIN_RELIABILITY)]
    return sorted(keep, key=lambda s: (ST.VERDICT_RANK[s.verdict], -s.low))


def render(scores: list[ModelScore], baseline: float, n_outcomes: int, cases: list[Case], meta: dict) -> str:
    counts = Counter(s.verdict for s in scores)
    winners = [s for s in scores if s.verdict == ST.BEATS]
    out = ["# Model scorecard: how accurate is each model at OUR predictions?", "",
           *(f"- **{k}:** {v}" for k, v in meta.items()), "",
           "## How to read this", "",
           "Each shortlisted model ran our real pipeline (news-impact, then synthesizer) on the SAME historical cases, "
           "with the same technicals and the same cached news, and the app's own grader scored the predictions "
           "against what the price actually did. **Accuracy** = of the calls that could be graded, how many were "
           "right: bullish is right if the price rose, bearish if it fell, and neutral if it barely moved.", "",
           f"**Baseline: {pct(baseline)}.** That is what the best no-skill strategy scores on these {n_outcomes} "
           "distinct outcomes: always bullish, always bearish, or always neutral. Beating a coin flip is not enough "
           "in a market that drifts, and 'always neutral' is a free win in a calm one.", "",
           "Every accuracy comes with a 95% interval. A model only **BEATS_BASELINE** if even the low end of its "
           "interval clears the baseline. With few cases the honest answer is **CANT_TELL_YET**, and "
           f"about {ST.judged_needed()} graded calls are needed to pin accuracy to +/-10 points. Re-running adds "
           "cases, so the picture sharpens by itself. **TOO_FEW** = fewer than "
           f"{ST.MIN_JUDGED} graded calls. **WORSE_THAN_BASELINE** = even the high end is below it.", "",
           "## Headline", "",
           f"{len(scores)} models, {len(cases)} cases. " + ", ".join(f"{v}: {counts[v]}" for v in ST.VERDICT_RANK), ""]
    out += ([f"**{len(winners)} model(s) beat the baseline:** " + ", ".join(f"`{s.cand.provider}/{s.cand.model}`" for s in winners), ""]
            if winners else ["**No model has beaten the baseline yet.** That is expected while the sample is small: "
                             "it does not mean they are bad, only that this much evidence cannot tell. Re-run to add cases.", ""])
    out += ["## Ranked (best worst-case first)", ""]
    out += [*_table(("#", "model", "verdict", "accuracy", "95% interval", "target-before-stop", "skill vs base rate",
                   "neutral share", "reliability", "sec/case"),
                  [(i, f"{s.cand.provider}/{s.cand.model}", s.verdict,
                    f"{pct(s.accuracy)} ({s.k_correct}/{s.n_judged})", f"{pct(s.low)} to {pct(s.high)}" if s.n_judged >= ST.MIN_JUDGED else "-",
                    pct(s.hit_rate_resolved), "-" if s.bss is None else f"{s.bss:+.2f}", pct(s.neutral_share),
                    f"{s.ok}/{s.attempted}", "-" if s.avg_seconds is None else f"{s.avg_seconds:.0f}")
                   for i, s in enumerate(scores, 1)]), ""]
    order = chain_order(scores)
    out += ["## Suggested chain order (provisional)", "",
            "Proven models first, then undecided ones that at least run reliably "
            f"(>= {MIN_RELIABILITY:.0%} of cases). Models that scored worse than the baseline or have too little "
            "evidence are left out. **Treat this as a starting point, not a verdict.**", ""]
    out += ([f"{i}. `{s.cand.provider}/{s.cand.model}` ({s.verdict}; {how_to_add(s.cand)})" for i, s in enumerate(order, 1)]
            or ["_nothing qualifies yet_"]) + [""]
    fails = [s for s in scores if s.fail_classes]
    out += ["## Why cases failed", ""]
    out += ([f"- `{s.cand.provider}/{s.cand.model}`: " + ", ".join(f"{k} x{v}" for k, v in sorted(s.fail_classes.items()))
             + (f"; {s.skipped} not reached (budget)" if s.skipped else "") for s in fails]
            or ["_no failures_"]) + [""]
    out += ["## The cases", "", ", ".join(f"{c.ticker} {c.as_of.isoformat()}" for c in cases) or "_none_", "",
            "## Caveats", "",
            "- **Luck.** With many models, a few will beat the baseline by chance. Trust a model that beats it in two "
            "separate runs, not one.",
            "- **Look-ahead.** A model trained after these dates may have seen what happened. The cases are recent to "
            "reduce this, but it cannot be ruled out.",
            "- **Replay is not live.** Backtests have no analyst-estimates evidence and news is limited to what was "
            "indexed by the date, so live results can differ.",
            "- **Neutral calls count.** A neutral call is right when the price barely moved, and the baseline "
            "includes 'always neutral', so hiding behind neutral calls cannot beat it. Check the neutral share anyway.",
            "- **Accuracy is not profit.** This scores direction and target/stop outcomes, not trading returns."]
    return "\n".join(out) + "\n"
