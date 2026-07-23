"""Hallucination guardrails for synthesizer output (Step 3.4.2 commit 5).

THE PROBLEM
===========
The synthesizer LLM emits a Prediction. ADK + Pydantic guarantee:
  - The JSON shape is right (output_schema).
  - Cross-field math holds (level invariants in schema.py).

What they DON'T guarantee:
  - The numbers are anchored in the input (LLM could invent a target
    that has nothing to do with any real swing level or ATR multiple).
  - The cited evidence exists (LLM could write contributing_signals=
    ('RSI=72',) when no cluster mentions RSI=72 anywhere).
  - The direction matches the evidence (LLM could call BULLISH against
    3 bearish clusters and neutral news).
  - The confidence is humble for the horizon (LLM could claim 0.95
    on a monthly call, where nobody can be that sure).

This module catches those failures BEFORE the prediction ships.

FOUR TIERS
==========
Tier 1 - validate_grounding    : numbers must trace to real input values
Tier 2 - validate_citations    : evidence strings must reference real input
Tier 3 - validate_consistency  : direction must match cluster majority
                                 or be backed by news
Tier 4 - validate_calibration  : confidence must respect per-horizon cap

PER-HORIZON BEHAVIOR (added in multi-horizon refactor commit B)
===============================================================
Stops, target ATR distances, entry-zone widths, and confidence caps
are per-horizon. The actual numbers live in `horizon_constants.py`
(single source of truth, also consumed by the synthesizer prompt in
commit C). Tier 1 reads stops + entry-zone + ATR-derived target
anchors per horizon. Tier 4 reads the confidence cap per horizon.

FAILURE FLOW
============
Any tier raising HallucinationError -> the predictor catches it,
re-invokes the synthesizer ONCE with the error message appended to the
prompt ("your previous attempt failed: <reason>, try again"), and re-
validates. Second failure -> bubble up as PredictionError.

WHY ONE-SHOT RETRY (not infinite)
=================================
Two failures in a row almost always means the LLM is genuinely confused
by the input (e.g. truly conflicting signals it can't resolve cleanly).
More retries waste tokens without improving outcomes. Better to fail
loud and let the caller decide.
"""
from __future__ import annotations

import re
from typing import Iterable

from price_predictor.prediction.horizon_constants import (
    confidence_cap,
    entry_zone_pct,
    stop_atr_range,
    target_atr_range,
)
from price_predictor.prediction.inputs import ClusterView, SynthesisInput
from price_predictor.prediction.schema import (
    Prediction,
    PredictionDirection,
    PredictionHorizon,
)

# Anchor-match precision tolerance — flat across horizons. This is
# "how close to an anchor counts as a match," which is a precision
# question (rounding tolerance), NOT a horizon question. See the
# horizon-specific knobs in horizon_constants.py for the per-horizon
# stop / target / entry-zone / confidence rules.
TARGET_ANCHOR_TOLERANCE_PCT = 0.005   # 0.5% — target must be within this
                                       # of a known anchor value

# Multiplicative grace factor applied to horizon-derived percentage caps
# (entry-zone-pct, stop ATR band, target ATR band) when validating LLM
# output. The LLM rounds to 2 decimals; the validator uses float math.
# Without this grace, a prediction that lands EXACTLY at the cap can be
# rejected by sub-paisa rounding (e.g. 0.5% cap on a ₹1000 stock → ₹5
# cap; LLM emits ₹5.02 due to 2-decimal rounding → rejected by 0.4%).
# 1.02 = a 2% slack, generous enough to absorb realistic LLM rounding
# while still preventing meaningful violations (a 2% slack on a 0.5%
# cap is 0.01% in absolute terms — noise).
_GROUNDING_GRACE = 1.02

# Number of ATR-derived target anchors generated per horizon. Three
# evenly-spaced points across the per-horizon target_atr_range
# (endpoints + midpoint) — dense enough to pin a reasonable target
# without flooding the anchor list. Compare to pre-commit-B behavior
# of 5 flat multipliers (1.0, 1.5, 2.0, 2.5, 3.0) used for ALL
# horizons.
_ATR_ANCHOR_COUNT = 3

# Tokens too generic to count as "real evidence" in citation checks.
# Lowercased.
#
# DELIBERATELY EXCLUDED from stopwords (used to be in here; were causing
# Tier 2 false positives):
#   above / below / near / over / under / high / low
#   These are CORE TA vocabulary, not filler. "RSI below 50" carries
#   real meaning. Treating them as stopwords meant citations like
#   "RSI 39.6 is below 50" tokenized to {} after filtering, so the
#   guardrail rejected perfectly-grounded LLM output.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "with", "for", "from", "into",
    "is", "are", "was", "were", "be", "been", "being", "to", "of", "in",
    "on", "at", "by", "as", "this", "that", "it", "its", "has", "have",
    "had",
    "price", "stock", "share", "shares",
})


# ─────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────
class HallucinationError(ValueError):
    """Raised when synthesizer output fails grounding/citation/consistency.

    Carries the tier name so the retry loop can include it in feedback.
    Subclass of ValueError so it surfaces naturally if not caught.
    """

    def __init__(self, tier: str, message: str):
        self.tier = tier
        super().__init__(f"[{tier}] {message}")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _atr_from_input(si: SynthesisInput) -> float:
    """Pull the ATR value from the input. Levels cluster has it in
    `derived["atr"]`, volatility cluster has it in `indicators["atr"]`.
    Either is fine (same number); we prefer levels.derived first.
    """
    levels_atr = si.technical_view.levels.derived.get("atr")
    if isinstance(levels_atr, (int, float)) and levels_atr > 0:
        return float(levels_atr)
    vol_atr = si.technical_view.volatility.indicators.get("atr")
    if isinstance(vol_atr, (int, float)) and vol_atr > 0:
        return float(vol_atr)
    raise HallucinationError(
        "grounding",
        "Input has no usable ATR (levels.derived nor volatility.indicators).",
    )


def _atr_anchors_for_horizon(horizon: PredictionHorizon) -> tuple[float, ...]:
    """Per-horizon ATR multipliers used to derive target anchor candidates.

    Returns three evenly-spaced points across the horizon's
    target_atr_range: (min, midpoint, max). Three is the smallest count
    that covers both endpoints AND the interior of the band, mirroring
    the density of the pre-commit-B flat 5-tuple while keeping the band
    horizon-specific.

    Example: for WEEKLY where target_atr_range = (1.0, 2.0), returns
    (1.0, 1.5, 2.0) — anchors at close + 1×ATR, +1.5×ATR, +2×ATR.
    """
    lo, hi = target_atr_range(horizon)
    mid = (lo + hi) / 2.0
    return (lo, mid, hi)


def _bullish_anchors(
    si: SynthesisInput, close: float, atr: float, horizon: PredictionHorizon,
) -> list[float]:
    """Valid target anchors for a BULLISH prediction.

    A bullish target should reach for resistance: swing high, pivot
    resistances, 52w high, or per-horizon ATR multiples above close.

    Real-level anchors (swing_high, r1, r2, high_52w) are NOT filtered
    by horizon — if the LLM picks a swing high for a daily call that's
    actually too far, the per-horizon stop check will catch the
    impossible R:R. Filtering real levels by horizon would require a
    judgment call ("how far is too far for daily?") not captured in
    the dossier; YAGNI for v1.
    """
    keys = ("swing_high", "r1", "r2", "high_52w")
    anchors = [
        v for v in (si.technical_view.levels.indicators.get(k) for k in keys)
        if isinstance(v, (int, float)) and v > 0
    ]
    anchors.extend(close + k * atr for k in _atr_anchors_for_horizon(horizon))
    return anchors


def _bearish_anchors(
    si: SynthesisInput, close: float, atr: float, horizon: PredictionHorizon,
) -> list[float]:
    """Mirror of _bullish_anchors for BEARISH predictions."""
    keys = ("swing_low", "s1", "s2", "low_52w")
    anchors = [
        v for v in (si.technical_view.levels.indicators.get(k) for k in keys)
        if isinstance(v, (int, float)) and v > 0
    ]
    anchors.extend(close - k * atr for k in _atr_anchors_for_horizon(horizon))
    # Filter out non-positive (a stock can't have a negative price target)
    return [a for a in anchors if a > 0]


def _within_tolerance(value: float, anchors: Iterable[float], tol_pct: float) -> bool:
    """True iff `value` is within `tol_pct` of at least one anchor."""
    for a in anchors:
        if a > 0 and abs(value - a) / a <= tol_pct:
            return True
    return False


def _tokenize(text: str) -> set[str]:
    """Lowercased set of significant tokens for citation matching.

    Two token classes are kept:
      - alphabetic tokens of length >= 3, not in _STOPWORDS
        (RSI, ATR, ADX, SMA, EMA, MACD, bullish, golden, cross, ...)
      - numeric tokens that are either >= 3 chars ("200", "1500") OR
        contain a decimal point ("39.6", "1.5", "0.20")

    Why include numerics: in technical analysis, the NUMBERS are the
    evidence. "RSI 39.6" is grounded by the literal value 39.6 appearing
    in the cluster rationale; throwing away "39.6" throws away the
    strongest signal that the LLM is actually reading the input.

    Why >= 3 not > 3: short indicator names (RSI, ATR, ADX, SMA, EMA)
    are 3 chars exactly. The old `> 3` bound silently dropped every
    short-named indicator from both the LLM's citation tokens AND the
    input vocabulary, making citations of those indicators unverifiable.

    Generic short numerics ("50", "70", "30") are dropped because they
    match too easily — RSI thresholds, percentages, you name it — and
    citing "below 50" alone shouldn't count as grounding evidence.
    """
    raw = re.findall(r"[a-zA-Z][a-zA-Z0-9_]*|\d+\.?\d*", text.lower())
    out: set[str] = set()
    for t in raw:
        if t[0].isdigit():
            # numeric: keep if has a decimal point OR is >= 3 chars
            if "." in t or len(t) >= 3:
                out.add(t)
        else:
            # alphabetic: keep if >= 3 chars and not a stopword
            if len(t) >= 3 and t not in _STOPWORDS:
                out.add(t)
    return out


# ─────────────────────────────────────────────────────────────
# Tier 1: numeric grounding
# ─────────────────────────────────────────────────────────────
def validate_grounding(pred: Prediction, si: SynthesisInput) -> None:
    """Ensure target/stop/entry are anchored to real input values.

    This is the load-bearing hallucination check: it's the difference
    between "LLM picked a real swing high" and "LLM made up a number
    that sounds plausible."

    Raises:
        HallucinationError(tier='grounding'): on any anchor violation.
    """
    close = si.technical_view.close_price
    atr = _atr_from_input(si)
    horizon = pred.horizon

    # ── Entry zone: must hug close_price (per-horizon width) ──
    # _GROUNDING_GRACE absorbs 2-decimal LLM rounding at the cap boundary.
    entry_tol = entry_zone_pct(horizon)
    entry_tol_eff = entry_tol * _GROUNDING_GRACE
    for tag, val in (("entry_low", pred.entry_zone[0]),
                     ("entry_high", pred.entry_zone[1])):
        if abs(val - close) / close > entry_tol_eff:
            raise HallucinationError(
                "grounding",
                f"{tag}={val:.2f} drifted from close={close:.2f} "
                f"by more than {entry_tol*100:.1f}% (horizon={horizon.value}). "
                f"Entry zones MUST hug close price.",
            )

    # ── Stop loss: distance must be in per-horizon ATR multiple band ──
    # Grace applied symmetrically to widen both bounds slightly.
    stop_min, stop_max = stop_atr_range(horizon)
    stop_dist = abs(pred.stop_loss.value - close)
    stop_lo_eff = stop_min / _GROUNDING_GRACE
    stop_hi_eff = stop_max * _GROUNDING_GRACE
    if not (stop_lo_eff * atr <= stop_dist <= stop_hi_eff * atr):
        raise HallucinationError(
            "grounding",
            f"stop_loss={pred.stop_loss.value:.2f} is "
            f"{stop_dist/atr:.2f}×ATR from close={close:.2f}. "
            f"Must be in [{stop_min}, {stop_max}]×ATR for horizon={horizon.value} "
            f"(ATR={atr:.2f}).",
        )

    # ── Target: must be near a real anchor (per-horizon ATR distances) ──
    if pred.direction == PredictionDirection.BULLISH:
        anchors = _bullish_anchors(si, close, atr, horizon)
        atr_lo, atr_hi = target_atr_range(horizon)
        anchor_desc = (
            f"swing_high / r1 / r2 / high_52w / close+({atr_lo}-{atr_hi})×ATR"
        )
    elif pred.direction == PredictionDirection.BEARISH:
        anchors = _bearish_anchors(si, close, atr, horizon)
        atr_lo, atr_hi = target_atr_range(horizon)
        anchor_desc = (
            f"swing_low / s1 / s2 / low_52w / close-({atr_lo}-{atr_hi})×ATR"
        )
    else:  # NEUTRAL
        # For neutral, target should be near close (range-bound). Allow
        # ±1×ATR around close as the "neutral target zone." This band is
        # NOT per-horizon — NEUTRAL by definition means "not really
        # moving," so the same tight band applies regardless of horizon.
        anchors = [close + k * atr for k in (-1.0, -0.5, 0.0, 0.5, 1.0)]
        anchor_desc = "close ± 0-1×ATR"

    if not _within_tolerance(
        pred.target.value, anchors, TARGET_ANCHOR_TOLERANCE_PCT * _GROUNDING_GRACE,
    ):
        raise HallucinationError(
            "grounding",
            f"target={pred.target.value:.2f} (direction={pred.direction.value}) "
            f"is not within {TARGET_ANCHOR_TOLERANCE_PCT*100:.1f}% of any "
            f"valid anchor. Valid anchors: {anchor_desc}.",
        )


# ─────────────────────────────────────────────────────────────
# Tier 2: citation traceability
# ─────────────────────────────────────────────────────────────
def _build_input_vocabulary(si: SynthesisInput) -> set[str]:
    """All tokens that count as 'real evidence' from the input.

    Includes:
      - cluster names (trend, momentum, volatility, levels)
      - signal words (bullish, bearish, neutral)
      - keys from each cluster's indicators dict (rsi, macd, atr, swing_high...)
      - cluster strength descriptors
      - tokens from rationale strings of each cluster
      - tokens from catalyst descriptions and reasoning
    """
    vocab: set[str] = {
        "trend", "momentum", "volatility", "levels",
        "bullish", "bearish", "neutral",
    }
    for cv in (si.technical_view.trend, si.technical_view.momentum,
               si.technical_view.volatility, si.technical_view.levels):
        # Indicator names like 'rsi', 'macd', 'atr', 'swing_high', 'sma_20'.
        # >= 3 (not > 3) because RSI/ATR/ADX/SMA/EMA are exactly 3 chars —
        # dropping them broke citation matching for every short-named
        # indicator. See _tokenize() docstring for the full story.
        vocab.update(k.lower() for k in cv.indicators.keys() if len(k) >= 3)
        vocab.update(k.lower() for k in cv.derived.keys() if len(k) >= 3)
        if cv.strength:
            vocab.update(_tokenize(cv.strength))
        for line in cv.rationale:
            vocab.update(_tokenize(line))
    # News side
    vocab.update(_tokenize(si.impact_assessment.reasoning))
    for cat in si.impact_assessment.catalysts:
        vocab.update(_tokenize(cat.description))
    return vocab


def validate_citations(pred: Prediction, si: SynthesisInput) -> None:
    """Each contributing/conflicting signal must reference real input.

    "Reference" = share at least one significant token (>3 chars,
    non-stopword) with the input vocabulary.

    Catches: LLM fabricating evidence (e.g. citing 'RSI=72' when no
    momentum indicator was actually 72).

    Raises:
        HallucinationError(tier='citation')
    """
    vocab = _build_input_vocabulary(si)

    for label, signals in (("contributing", pred.contributing_signals),
                           ("conflicting", pred.conflicting_signals)):
        for signal_str in signals:
            tokens = _tokenize(signal_str)
            if not tokens:
                # Empty or all-stopwords — uninformative, treat as fabricated.
                raise HallucinationError(
                    "citation",
                    f"{label}_signal {signal_str!r} has no informative tokens.",
                )
            if not (tokens & vocab):
                raise HallucinationError(
                    "citation",
                    f"{label}_signal {signal_str!r} shares no tokens with "
                    f"input evidence. Tokens={sorted(tokens)}. "
                    f"Cite real cluster signals or catalyst descriptions.",
                )


# ─────────────────────────────────────────────────────────────
# Tier 3: cross-cluster + news consistency
# ─────────────────────────────────────────────────────────────
def _count_cluster_signals(si: SynthesisInput) -> tuple[int, int, int]:
    """Return (bull_count, bear_count, neutral_count) across 4 clusters."""
    clusters: tuple[ClusterView, ...] = (
        si.technical_view.trend, si.technical_view.momentum,
        si.technical_view.volatility, si.technical_view.levels,
    )
    bull = sum(1 for c in clusters if c.signal == "bullish")
    bear = sum(1 for c in clusters if c.signal == "bearish")
    neutral = sum(1 for c in clusters if c.signal == "neutral")
    return bull, bear, neutral


def validate_consistency(pred: Prediction, si: SynthesisInput) -> None:
    """Direction must align with majority of clusters OR be backed by news.

    Rules:
      BULLISH  : need bull_count >= 2  OR  (bull >= 1 AND news=='bullish')
      BEARISH  : need bear_count >= 2  OR  (bear >= 1 AND news=='bearish')
      NEUTRAL  : always allowed (it's the safe default)

    Catches: LLM ignoring the evidence entirely. A bullish call against
    3 bearish clusters and neutral news is almost always a hallucination
    or sloppy reasoning.

    Raises:
        HallucinationError(tier='consistency')
    """
    bull, bear, _ = _count_cluster_signals(si)
    news = si.impact_assessment.sentiment

    if pred.direction == PredictionDirection.BULLISH:
        if bull < 2 and not (bull >= 1 and news == "bullish"):
            raise HallucinationError(
                "consistency",
                f"BULLISH call but only {bull}/4 clusters bullish and "
                f"news sentiment is {news!r}. Need ≥2 bullish clusters "
                f"OR ≥1 bullish cluster + bullish news.",
            )
    elif pred.direction == PredictionDirection.BEARISH:
        if bear < 2 and not (bear >= 1 and news == "bearish"):
            raise HallucinationError(
                "consistency",
                f"BEARISH call but only {bear}/4 clusters bearish and "
                f"news sentiment is {news!r}. Need ≥2 bearish clusters "
                f"OR ≥1 bearish cluster + bearish news.",
            )
    # NEUTRAL: no constraint — neutral is always defensible.


# ────────────────────────────────────────────
# Tier 4: per-horizon confidence cap (the humility check)
# ────────────────────────────────────────────
def validate_calibration(pred: Prediction, si: SynthesisInput) -> None:
    """Confidence must respect the per-horizon cap.

    Long-horizon predictions are inherently more uncertain (more events
    can intervene between as_of and target_datetime). The LLM should
    not be allowed to claim 0.95 confidence on a monthly call. The
    synthesizer prompt (commit C) tells the LLM about these caps;
    this is the enforcement.

    Caps are defined per horizon in `horizon_constants.py`:
      DAILY    ≤ 0.90
      WEEKLY   ≤ 0.85
      BIWEEKLY ≤ 0.80
      MONTHLY  ≤ 0.75

    The `si` argument is unused today (cap depends only on horizon),
    but kept for signature consistency with the other validators — a
    future calibration that depends on input richness (e.g. "only allow
    high confidence when news + technicals both confirm") would need it.

    Raises:
        HallucinationError(tier='calibration')
    """
    cap = confidence_cap(pred.horizon)
    if pred.confidence > cap:
        raise HallucinationError(
            "calibration",
            f"confidence {pred.confidence:.2f} exceeds cap {cap:.2f} "
            f"for horizon={pred.horizon.value}. Long-horizon predictions "
            f"are inherently more uncertain; lower confidence required.",
        )


# ────────────────────────────────────────────
# Public: run all four
# ────────────────────────────────────────────
def validate_all(pred: Prediction, si: SynthesisInput) -> None:
    """Run all four guardrails and report EVERY failure at once.

    Why collect-all instead of fail-fast: the retry loop feeds the error
    back to the LLM so it can fix its output. If we raised on the FIRST
    failing tier, the LLM only ever learns about one problem at a time --
    it fixes the target (grounding), which shifts the direction and trips
    consistency on the next attempt, which it fixes by re-breaking the
    target... a whack-a-mole that burns the whole retry budget without
    ever converging. By collecting every violation into a single error,
    the LLM sees the complete constraint set and can satisfy them all in
    one shot.

    Tiers still run in the same order (grounding, citation, consistency,
    calibration) so the combined message reads most-substantive-first.

    Raises:
        HallucinationError: if any tier(s) failed. When multiple failed,
            ``tier`` is a comma-joined list and the message enumerates
            every violation.
    """
    validators = (
        validate_grounding,
        validate_citations,
        validate_consistency,
        validate_calibration,
    )
    failures: list[HallucinationError] = []
    for validator in validators:
        try:
            validator(pred, si)
        except HallucinationError as e:
            failures.append(e)

    if not failures:
        return
    if len(failures) == 1:
        raise failures[0]

    # Multiple tiers failed -- combine so the LLM fixes them together.
    tiers = ",".join(e.tier for e in failures)
    combined = "\n".join(f"  - {e}" for e in failures)
    raise HallucinationError(
        tiers,
        f"{len(failures)} checks failed simultaneously. Fix ALL of them "
        f"in your next attempt (fixing one must not break another):\n"
        f"{combined}",
    )
