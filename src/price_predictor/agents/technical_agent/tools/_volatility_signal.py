"""Volatility signal classifier -- pure function, no I/O.

WHY VOLATILITY HAS A 'SIGNAL'
=============================
Volatility is direction-agnostic in classical TA. But for the LLM's
benefit we still emit a bullish/neutral/bearish signal based on
position WITHIN the Bollinger Band:
  - %B > 0.5: price in upper half of band (bullish location)
  - %B < 0.5: price in lower half of band (bearish location)

Strength is the more important field here:
  - 'strong': BB squeeze active (low bandwidth -> breakout incoming)
  - 'moderate': normal volatility regime
  - 'weak': dead-quiet OR violently chaotic (neither tradeable)
"""
from __future__ import annotations

from price_predictor.agents.technical_agent.tools._types import Signal, Strength

# %B thresholds for direction classification
PCT_B_BULLISH = 0.55
PCT_B_BEARISH = 0.45

# ATR-as-%-of-price thresholds for regime classification
ATR_PCT_DEAD_QUIET = 1.0   # below = dead quiet (untradeable)
ATR_PCT_NORMAL_LOW = 1.0
ATR_PCT_NORMAL_HIGH = 4.0
ATR_PCT_MANIC = 6.0        # above = chaotic (avoid)


def classify_volatility(
    snapshot: dict,
) -> tuple[Signal, Strength, list[str], list[str]]:
    """Turn a volatility_snapshot into (signal, strength, rationale, warnings).

    Args:
        snapshot: output of analysis.volatility.volatility_snapshot()
                  shape: {atr, atr_pct_of_price, bbands: {...}, squeeze: bool}
    """
    rationale: list[str] = []
    warnings: list[str] = []

    atr = snapshot.get("atr")
    atr_pct = snapshot.get("atr_pct_of_price")
    bbands = snapshot.get("bbands", {})
    squeeze = snapshot.get("squeeze")

    pct_b = bbands.get("percent_b")
    bandwidth = bbands.get("bandwidth")
    upper = bbands.get("upper")
    lower = bbands.get("lower")
    middle = bbands.get("middle")

    # ── Insufficient-data short circuit ─────────────────────────
    if atr is None and pct_b is None:
        return (
            "neutral",
            "weak",
            ["Insufficient history for volatility analysis"],
            ["insufficient_history"],
        )

    # ── Direction signal from %B position ───────────────────────
    if pct_b is None:
        signal: Signal = "neutral"
    elif pct_b > PCT_B_BULLISH:
        signal = "bullish"
        rationale.append(
            f"BB %B = {pct_b:.2f} (price in upper half of band: bullish location)"
        )
    elif pct_b < PCT_B_BEARISH:
        signal = "bearish"
        rationale.append(
            f"BB %B = {pct_b:.2f} (price in lower half of band: bearish location)"
        )
    else:
        signal = "neutral"
        rationale.append(
            f"BB %B = {pct_b:.2f} (price near middle band: no directional bias)"
        )

    # ── %B extreme warnings (touching/exceeding bands) ──────────
    if pct_b is not None:
        if pct_b > 1.0:
            warnings.append("price_above_upper_band")
            rationale.append(f"⚠ Price above upper band (%B = {pct_b:.2f}): stretched")
        elif pct_b < 0.0:
            warnings.append("price_below_lower_band")
            rationale.append(f"⚠ Price below lower band (%B = {pct_b:.2f}): stretched")

    # ── ATR rationale ───────────────────────────────────────────
    if atr is not None and atr_pct is not None:
        rationale.append(
            f"ATR-14 = {atr:.2f} ({atr_pct:.2f}% of price)"
        )

    # ── Strength: squeeze > regime ──────────────────────────────
    # A squeeze is the highest-priority signal -- it predicts breakouts
    # regardless of current direction.
    if squeeze is True:
        strength: Strength = "strong"
        rationale.append(
            "⚡ BB SQUEEZE active: bandwidth in lowest 20% of recent range -- "
            "breakout likely incoming"
        )
    elif atr_pct is None:
        strength = "weak"
    elif atr_pct < ATR_PCT_DEAD_QUIET:
        strength = "weak"
        rationale.append(
            f"Volatility dead-quiet (ATR < {ATR_PCT_DEAD_QUIET}% of price): "
            "low conviction, hard to trade"
        )
    elif atr_pct > ATR_PCT_MANIC:
        strength = "weak"
        warnings.append("high_volatility")
        rationale.append(
            f"Volatility manic (ATR > {ATR_PCT_MANIC}% of price): "
            "wide stops needed, choppy"
        )
    elif ATR_PCT_NORMAL_LOW <= atr_pct <= ATR_PCT_NORMAL_HIGH:
        strength = "moderate"
    else:
        strength = "weak"  # 4-6% range: elevated but not manic

    # ── Bandwidth rationale ────────────────────────────────────
    if bandwidth is not None and squeeze is not True:
        rationale.append(f"BB bandwidth = {bandwidth:.2f}% of middle band")

    return signal, strength, rationale, warnings


def classify_volatility_regime(atr_pct: float | None) -> str:
    """Classify the volatility regime as 'low' | 'normal' | 'high' | 'unknown'.

    Surfaced separately in the tool's `derived` block so the LLM can quote
    the regime explicitly without re-deriving it.
    """
    if atr_pct is None:
        return "unknown"
    if atr_pct < ATR_PCT_DEAD_QUIET:
        return "low"
    if atr_pct > ATR_PCT_MANIC:
        return "high"
    if atr_pct > ATR_PCT_NORMAL_HIGH:
        return "high"
    return "normal"
