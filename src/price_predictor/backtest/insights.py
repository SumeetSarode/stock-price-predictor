"""Rule-based executive insights generated from a BacktestEvaluation.

WHY THIS EXISTS
===============
The HTML report has metric cards everywhere, but a busy reader needs
the "so what" in one paragraph at the top AND bottom (per the
"executive insights at top & bottom" mandate). Hand-writing those
takeaways for every backtest is tedious; an LLM call would be slow
and nondeterministic.

This module derives plain-English insights from the calibration
metrics deterministically. Same eval -> same insights. Cheap.

DESIGN
======
- ONE function per insight TYPE (overall verdict, best/worst horizon,
  ticker outliers, time-series trend). Each is pure: input is a
  BacktestEvaluation, output is a (level, headline, detail) triple
  used by the HTML layer to color/format the card.
- An `Insight` dataclass with severity LEVEL (positive/neutral/warning/
  critical) so the HTML can color-code (green/blue/yellow/red, per
  Walmart palette).
- `generate_insights(eval)` orchestrator returns a list ordered by
  severity (criticals first), so the most actionable items render at
  the top of the insight section.

WHY RULE-BASED, NOT LLM
=======================
- Determinism: same numbers always give same words. Auditors love this.
- Cost: zero LLM tokens per report.
- Speed: <1ms vs 30s for an LLM round-trip.
- The insights are EXACTLY the kinds of statements a domain expert
  would make ("hit rate beats base rate by 8 pts; meaningful skill")
  -- no creative writing needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from price_predictor.backtest.evaluation import BacktestEvaluation


class InsightLevel(str, Enum):
    """Severity / sentiment of an insight, drives HTML color.

    Mapped in the HTML layer to Walmart palette:
        POSITIVE -> green.100  (#2a8703)  "things are working"
        NEUTRAL  -> blue.100   (#0053e2)  "informational"
        WARNING  -> spark.140  (#995213)  "watch this"
        CRITICAL -> red.100    (#ea1100)  "act on this"
    """
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Insight:
    """One observation about a backtest evaluation.

    Frozen so insights composed at top of report are guaranteed
    identical to the same insight rendered at bottom (no sneaky
    mutation between sections).
    """
    level: InsightLevel
    headline: str    # 1-line summary for card title (~50 chars max)
    detail: str      # 1-2 sentence explanation with numbers


# ─────────────────────────────────────────────────────────────
# Tunable thresholds (centralized so a single change updates ALL
# insights consistently). Numbers chosen from common trading lit:
#   - Brier Skill Score > 0 = real skill (Murphy 1973)
#   - Direction accuracy >= 55% on equities is generally considered
#     edge (Lo & MacKinlay '88 random-walk null is ~52%)
#   - Sample sizes < 20 are too noisy for individual-axis claims
# ─────────────────────────────────────────────────────────────
_MIN_SAMPLE_FOR_AXIS_CLAIM = 20
_DIRECTION_ACC_GOOD = 0.55
_DIRECTION_ACC_BAD = 0.45
_BSS_GOOD = 0.05
_BSS_BAD = -0.05
_HIT_RATE_GOOD = 0.55
_HIT_RATE_BAD = 0.40


# ─────────────────────────────────────────────────────────────
# Individual insight generators
# ─────────────────────────────────────────────────────────────
def _overall_verdict(ev: "BacktestEvaluation") -> Insight:
    """Headline verdict on the model's directional skill.

    Anchored on direction_accuracy (the most interpretable single
    metric) with BSS as confirmation that the LLM's confidence is
    also calibrated -- not just lucky.
    """
    o = ev.overall
    n = o.n_with_direction_judgement

    if n == 0:
        return Insight(
            level=InsightLevel.WARNING,
            headline="No judged predictions",
            detail=(
                f"All {o.n_predictions} predictions were INCONCLUSIVE "
                "(usually means horizons hadn't elapsed by grading time). "
                "Re-run with later --today, or wait for more bars."
            ),
        )

    da = o.direction_accuracy
    bss = o.brier_skill_score

    # Combined verdict: direction accuracy AND BSS must agree.
    if da >= _DIRECTION_ACC_GOOD and (bss is not None and bss >= _BSS_GOOD):
        level = InsightLevel.POSITIVE
        headline = f"Real skill: {da:.1%} direction accuracy"
        detail = (
            f"Across {n} judged predictions, the model picked the right "
            f"direction {da:.1%} of the time (above the {_DIRECTION_ACC_GOOD:.0%} "
            f"edge threshold) AND its confidence was well-calibrated "
            f"(BSS={bss:+.3f} > 0). Both conditions met."
        )
    elif da <= _DIRECTION_ACC_BAD or (bss is not None and bss <= _BSS_BAD):
        level = InsightLevel.CRITICAL
        bss_str = f"BSS={bss:+.3f}" if bss is not None else "BSS=n/a"
        headline = f"Anti-skill: {da:.1%} direction accuracy"
        detail = (
            f"Across {n} judged predictions, direction accuracy ({da:.1%}) "
            f"and confidence calibration ({bss_str}) suggest the model is "
            f"WORSE than guessing the base rate. Don't trade on this; "
            "investigate prompt + data drift."
        )
    else:
        level = InsightLevel.NEUTRAL
        bss_str = f"BSS={bss:+.3f}" if bss is not None else "BSS=n/a"
        headline = f"Marginal: {da:.1%} direction accuracy"
        detail = (
            f"Direction accuracy {da:.1%} and {bss_str} are within the "
            "noise band. More data needed before claiming or denying skill."
        )

    return Insight(level=level, headline=headline, detail=detail)


def _hit_rate_takeaway(ev: "BacktestEvaluation") -> Insight:
    """Hit-rate framing: target/stop economics, separate from direction."""
    o = ev.overall
    n_resolved = o.n_target_hit + o.n_stop_hit + o.n_stop_hit_ambiguous
    if n_resolved == 0:
        return Insight(
            level=InsightLevel.NEUTRAL,
            headline="No T/S resolutions",
            detail=(
                "No predictions hit target or stop within their windows; "
                "all directional calls expired. Hit-rate metrics not "
                "meaningful for this run -- look at direction accuracy."
            ),
        )

    hr = o.hit_rate_resolved
    if hr >= _HIT_RATE_GOOD:
        level = InsightLevel.POSITIVE
        headline = f"{hr:.0%} of resolved trades hit target"
    elif hr <= _HIT_RATE_BAD:
        level = InsightLevel.WARNING
        headline = f"Only {hr:.0%} of resolved trades hit target"
    else:
        level = InsightLevel.NEUTRAL
        headline = f"{hr:.0%} of resolved trades hit target"

    return Insight(
        level=level,
        headline=headline,
        detail=(
            f"Of {n_resolved} predictions that hit either target or stop, "
            f"{o.n_target_hit} hit target, {o.n_stop_hit} cleanly stopped out, "
            f"{o.n_stop_hit_ambiguous} ambiguous (same-bar T+S touch, called "
            "as stop conservatively)."
        ),
    )


def _best_horizon(ev: "BacktestEvaluation") -> Insight | None:
    """Highlight the horizon with the strongest direction accuracy.

    Only fires if we have multiple horizons AND the leader has enough
    samples to be a non-noise claim.
    """
    if len(ev.by_horizon) < 2:
        return None  # nothing to compare

    # Filter out under-sampled horizons before ranking.
    candidates = [
        (h, r) for h, r in ev.by_horizon.items()
        if r.n_with_direction_judgement >= _MIN_SAMPLE_FOR_AXIS_CLAIM
    ]
    if not candidates:
        return None

    best_h, best_r = max(candidates, key=lambda kv: kv[1].direction_accuracy)
    worst_h, worst_r = min(candidates, key=lambda kv: kv[1].direction_accuracy)

    if best_h == worst_h:
        return None  # only one horizon had enough data

    spread = best_r.direction_accuracy - worst_r.direction_accuracy
    # Don't bother surfacing if the spread is tiny (likely noise).
    if spread < 0.05:
        return None

    return Insight(
        level=InsightLevel.NEUTRAL,
        headline=(
            f"{best_h.value.capitalize()} horizon outperforms "
            f"{worst_h.value} by {spread:.1%}"
        ),
        detail=(
            f"{best_h.value.capitalize()}: {best_r.direction_accuracy:.1%} "
            f"direction accuracy ({best_r.n_with_direction_judgement} judged). "
            f"{worst_h.value.capitalize()}: {worst_r.direction_accuracy:.1%} "
            f"({worst_r.n_with_direction_judgement} judged). Consider "
            f"weighting {best_h.value} signals more heavily."
        ),
    )


def _ticker_outliers(ev: "BacktestEvaluation") -> Insight | None:
    """Flag tickers that are dramatically better or worse than overall.

    Only fires if we have multiple tickers AND at least one has enough
    samples to make a credible claim.
    """
    if len(ev.by_ticker) < 2:
        return None

    overall_da = ev.overall.direction_accuracy
    candidates = [
        (t, r) for t, r in ev.by_ticker.items()
        if r.n_with_direction_judgement >= _MIN_SAMPLE_FOR_AXIS_CLAIM
    ]
    if not candidates:
        return None

    # Find the most-deviant ticker (signed distance from overall).
    most_extreme_ticker, most_extreme_report = max(
        candidates,
        key=lambda kv: abs(kv[1].direction_accuracy - overall_da),
    )
    delta = most_extreme_report.direction_accuracy - overall_da

    if abs(delta) < 0.10:
        return None  # within normal variation

    if delta > 0:
        return Insight(
            level=InsightLevel.POSITIVE,
            headline=f"{most_extreme_ticker} is {delta:+.1%} above overall",
            detail=(
                f"{most_extreme_ticker} hits "
                f"{most_extreme_report.direction_accuracy:.1%} direction "
                f"accuracy vs the overall {overall_da:.1%} "
                f"({most_extreme_report.n_with_direction_judgement} judged). "
                "Worth investigating what's special -- coverage, news flow, "
                "volatility regime."
            ),
        )
    else:
        return Insight(
            level=InsightLevel.WARNING,
            headline=f"{most_extreme_ticker} is {delta:+.1%} vs overall",
            detail=(
                f"{most_extreme_ticker} only hits "
                f"{most_extreme_report.direction_accuracy:.1%} direction "
                f"accuracy vs overall {overall_da:.1%} "
                f"({most_extreme_report.n_with_direction_judgement} judged). "
                "Consider excluding or down-weighting this ticker."
            ),
        )


def _sample_size_warning(ev: "BacktestEvaluation") -> Insight | None:
    """Flag total sample size if it's too small for confident claims."""
    n = ev.overall.n_with_direction_judgement
    if n >= _MIN_SAMPLE_FOR_AXIS_CLAIM * 2:
        return None  # plenty of data, no warning needed

    if n == 0:
        return None  # already covered by overall_verdict

    return Insight(
        level=InsightLevel.WARNING,
        headline=f"Small sample: {n} judged predictions",
        detail=(
            f"Direction accuracy and Brier metrics are noisy at n={n}. "
            f"Aim for >= {_MIN_SAMPLE_FOR_AXIS_CLAIM * 2} judged predictions "
            "before drawing conclusions. Extend the date range or add tickers."
        ),
    )


# ─────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────
# Severity sort order for top-of-report rendering. Critical first
# (it's the action item), positive last (it's the reward).
_LEVEL_ORDER: dict[InsightLevel, int] = {
    InsightLevel.CRITICAL: 0,
    InsightLevel.WARNING: 1,
    InsightLevel.NEUTRAL: 2,
    InsightLevel.POSITIVE: 3,
}


def generate_insights(ev: "BacktestEvaluation") -> list[Insight]:
    """Build the list of insights for the report's exec-summary sections.

    Args:
        ev: A completed BacktestEvaluation.

    Returns:
        Insights sorted CRITICAL -> WARNING -> NEUTRAL -> POSITIVE
        so the most actionable items render first. Empty if the
        evaluation has no judged predictions (handled gracefully by
        the HTML layer with a placeholder).

    Why this is a flat list vs. a structured object:
        The HTML layer iterates the list and renders each card the
        same way (color from level, two text fields). A nested
        structure would force the renderer to know about each
        insight type -- leaky.
    """
    raw: list[Insight | None] = [
        _overall_verdict(ev),
        _hit_rate_takeaway(ev),
        _best_horizon(ev),
        _ticker_outliers(ev),
        _sample_size_warning(ev),
    ]

    # Drop the Nones (insights that decided not to fire) and sort.
    insights = [i for i in raw if i is not None]
    insights.sort(key=lambda i: _LEVEL_ORDER[i.level])
    return insights
