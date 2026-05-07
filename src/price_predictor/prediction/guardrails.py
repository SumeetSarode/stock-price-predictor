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

This module catches those failures BEFORE the prediction ships.

THREE TIERS
===========
Tier 1 - validate_grounding   : numbers must trace to real input values
Tier 2 - validate_citations   : evidence strings must reference real input
Tier 3 - validate_consistency : direction must match cluster majority
                                or be backed by news

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

from price_predictor.prediction.inputs import ClusterView, SynthesisInput
from price_predictor.prediction.schema import Prediction, PredictionDirection

# Tunable thresholds — strict by design (commit 5 is "max guardrails" mode).
# If real-world calibration shows these are too tight, loosen here, NOT
# in the validators (keep the policy in one place).
TARGET_ANCHOR_TOLERANCE_PCT = 0.005   # 0.5% — target must be within this
                                       # of a known anchor value
STOP_MIN_ATR = 0.7                     # stop must be ≥ 0.7×ATR from close
STOP_MAX_ATR = 1.8                     # stop must be ≤ 1.8×ATR from close
ENTRY_ZONE_TOLERANCE_PCT = 0.01        # entry zone must be within 1% of close
ATR_MULTIPLIERS = (1.0, 1.5, 2.0, 2.5, 3.0)  # close ± k×ATR are valid targets

# Tokens too short or too generic to count as "real evidence" in
# citation checks. Lowercased.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "with", "for", "from", "into",
    "is", "are", "was", "were", "be", "been", "being", "to", "of", "in",
    "on", "at", "by", "as", "this", "that", "it", "its", "has", "have",
    "had", "above", "below", "near", "over", "under", "high", "low",
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


def _bullish_anchors(si: SynthesisInput, close: float, atr: float) -> list[float]:
    """Valid target anchors for a BULLISH prediction.

    A bullish target should reach for resistance: swing high, pivot
    resistances, 52w high, or 1-3 ATRs above close.
    """
    keys = ("swing_high", "r1", "r2", "high_52w")
    anchors = [
        v for v in (si.technical_view.levels.indicators.get(k) for k in keys)
        if isinstance(v, (int, float)) and v > 0
    ]
    anchors.extend(close + k * atr for k in ATR_MULTIPLIERS)
    return anchors


def _bearish_anchors(si: SynthesisInput, close: float, atr: float) -> list[float]:
    """Mirror of _bullish_anchors for BEARISH predictions."""
    keys = ("swing_low", "s1", "s2", "low_52w")
    anchors = [
        v for v in (si.technical_view.levels.indicators.get(k) for k in keys)
        if isinstance(v, (int, float)) and v > 0
    ]
    anchors.extend(close - k * atr for k in ATR_MULTIPLIERS)
    # Filter out non-positive (a stock can't have a negative price target)
    return [a for a in anchors if a > 0]


def _within_tolerance(value: float, anchors: Iterable[float], tol_pct: float) -> bool:
    """True iff `value` is within `tol_pct` of at least one anchor."""
    for a in anchors:
        if a > 0 and abs(value - a) / a <= tol_pct:
            return True
    return False


def _tokenize(text: str) -> set[str]:
    """Lowercased set of significant tokens (>3 chars, non-stopword)."""
    raw = re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", text.lower())
    return {t for t in raw if len(t) > 3 and t not in _STOPWORDS}


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

    # ── Entry zone: must hug close_price ────────────────────
    for tag, val in (("entry_low", pred.entry_zone[0]),
                     ("entry_high", pred.entry_zone[1])):
        if abs(val - close) / close > ENTRY_ZONE_TOLERANCE_PCT:
            raise HallucinationError(
                "grounding",
                f"{tag}={val:.2f} drifted from close={close:.2f} "
                f"by more than {ENTRY_ZONE_TOLERANCE_PCT*100:.1f}%. "
                f"Entry zones MUST hug close price.",
            )

    # ── Stop loss: distance must be in valid ATR multiple ───
    stop_dist = abs(pred.stop_loss.value - close)
    if not (STOP_MIN_ATR * atr <= stop_dist <= STOP_MAX_ATR * atr):
        raise HallucinationError(
            "grounding",
            f"stop_loss={pred.stop_loss.value:.2f} is "
            f"{stop_dist/atr:.2f}×ATR from close={close:.2f}. "
            f"Must be in [{STOP_MIN_ATR}, {STOP_MAX_ATR}]×ATR (ATR={atr:.2f}).",
        )

    # ── Target: must be near a real anchor ──────────────────
    if pred.direction == PredictionDirection.BULLISH:
        anchors = _bullish_anchors(si, close, atr)
        anchor_desc = "swing_high / r1 / r2 / high_52w / close+(1-3)×ATR"
    elif pred.direction == PredictionDirection.BEARISH:
        anchors = _bearish_anchors(si, close, atr)
        anchor_desc = "swing_low / s1 / s2 / low_52w / close-(1-3)×ATR"
    else:  # NEUTRAL
        # For neutral, target should be near close (range-bound). Allow
        # ±1×ATR around close as the "neutral target zone."
        anchors = [close + k * atr for k in (-1.0, -0.5, 0.0, 0.5, 1.0)]
        anchor_desc = "close ± 0-1×ATR"

    if not _within_tolerance(
        pred.target.value, anchors, TARGET_ANCHOR_TOLERANCE_PCT,
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
        # Indicator names like 'rsi', 'macd', 'atr', 'swing_high', 'sma_20'
        vocab.update(k.lower() for k in cv.indicators.keys() if len(k) > 3)
        vocab.update(k.lower() for k in cv.derived.keys() if len(k) > 3)
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


# ─────────────────────────────────────────────────────────────
# Public: run all three
# ─────────────────────────────────────────────────────────────
def validate_all(pred: Prediction, si: SynthesisInput) -> None:
    """Run all three guardrails. Raises HallucinationError on first fail.

    Order matters: grounding (cheapest) first, then citations, then
    consistency. Earlier tiers catch the most common failures.
    """
    validate_grounding(pred, si)
    validate_citations(pred, si)
    validate_consistency(pred, si)
