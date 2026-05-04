"""Momentum signal classifier -- pure function, no I/O.

Input: momentum_snapshot output (rsi, macd dict, stoch dict, obv dict).
Output: (signal, strength, rationale_bullets, warnings).

Kept separate from get_momentum.py so we can unit-test the rules without
mocking fetch/cache.
"""
from __future__ import annotations

from price_predictor.agents.technical_agent.tools._types import Signal, Strength

# RSI extremes for strength classification
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_NEUTRAL_LOW = 40
RSI_NEUTRAL_HIGH = 60

# OBV slope magnitude (%) above which we call divergence "meaningful"
OBV_DIVERGENCE_THRESHOLD_PCT = 2.0


def _rsi_strength(rsi: float | None) -> Strength:
    """Map RSI to strength label.

    >70 or <30 -> strong (extremes have momentum)
    40-60      -> moderate (active range)
    otherwise  -> weak (transition zones)
    """
    if rsi is None:
        return "weak"
    if rsi > RSI_OVERBOUGHT or rsi < RSI_OVERSOLD:
        return "strong"
    if RSI_NEUTRAL_LOW <= rsi <= RSI_NEUTRAL_HIGH:
        return "moderate"
    return "weak"


def classify_momentum(
    snapshot: dict,
) -> tuple[Signal, Strength, list[str], list[str]]:
    """Turn a momentum_snapshot into (signal, strength, rationale, warnings).

    RULES
    =====
    Signal (3-of-3 alignment, otherwise neutral):
      bullish: RSI > 50 AND MACD histogram > 0 AND Stoch %K > %D
      bearish: RSI < 50 AND MACD histogram < 0 AND Stoch %K < %D
      neutral: mixed signals or insufficient data

    Strength: derived from RSI extremes (independent of direction).

    Warnings:
      - "obv_divergence": OBV trend direction contradicts the signal
      - "insufficient_history": all core indicators are None
    """
    rationale: list[str] = []
    warnings: list[str] = []

    rsi = snapshot.get("rsi")
    macd = snapshot.get("macd", {})
    stoch = snapshot.get("stoch", {})
    obv = snapshot.get("obv", {})

    macd_hist = macd.get("histogram")
    stoch_k = stoch.get("k")
    stoch_d = stoch.get("d")
    obv_slope = obv.get("slope_20")

    # ── Insufficient data short circuit ─────────────────────────
    if rsi is None and macd_hist is None and stoch_k is None:
        return "neutral", "weak", ["Insufficient history for momentum analysis"], ["insufficient_history"]

    # ── Component direction checks ──────────────────────────────
    rsi_bullish = rsi is not None and rsi > 50
    rsi_bearish = rsi is not None and rsi < 50

    macd_bullish = macd_hist is not None and macd_hist > 0
    macd_bearish = macd_hist is not None and macd_hist < 0

    stoch_bullish = (
        stoch_k is not None and stoch_d is not None and stoch_k > stoch_d
    )
    stoch_bearish = (
        stoch_k is not None and stoch_d is not None and stoch_k < stoch_d
    )

    # ── Build rationale (one bullet per indicator) ──────────────
    if rsi is not None:
        if rsi > RSI_OVERBOUGHT:
            rationale.append(f"RSI {rsi:.1f} is overbought (>70)")
        elif rsi < RSI_OVERSOLD:
            rationale.append(f"RSI {rsi:.1f} is oversold (<30)")
        elif rsi > 50:
            rationale.append(f"RSI {rsi:.1f} is above 50: bullish bias")
        else:
            rationale.append(f"RSI {rsi:.1f} is below 50: bearish bias")

    if macd_hist is not None:
        cross = macd.get("cross")
        if cross == "bullish":
            rationale.append("MACD histogram crossed positive on latest bar")
        elif cross == "bearish":
            rationale.append("MACD histogram crossed negative on latest bar")
        elif macd_hist > 0:
            rationale.append(f"MACD histogram {macd_hist:.2f} is positive")
        else:
            rationale.append(f"MACD histogram {macd_hist:.2f} is negative")

    if stoch_k is not None and stoch_d is not None:
        if stoch_k > stoch_d:
            rationale.append(f"Stochastic %K ({stoch_k:.1f}) > %D ({stoch_d:.1f}): bullish")
        else:
            rationale.append(f"Stochastic %K ({stoch_k:.1f}) < %D ({stoch_d:.1f}): bearish")

    if obv_slope is not None:
        sign = "+" if obv_slope >= 0 else ""
        rationale.append(f"OBV 20-bar slope {sign}{obv_slope:.1f}% (volume {'rising' if obv_slope > 0 else 'falling'})")

    # ── Combine into final signal ──────────────────────────────
    bullish_count = sum([rsi_bullish, macd_bullish, stoch_bullish])
    bearish_count = sum([rsi_bearish, macd_bearish, stoch_bearish])

    # Need majority (2 of 3 available components, or 3 of 3 if all present)
    available = sum([
        rsi is not None,
        macd_hist is not None,
        stoch_k is not None and stoch_d is not None,
    ])

    if available == 0:
        signal: Signal = "neutral"
    elif bullish_count >= max(2, available - 1) and bearish_count == 0:
        signal = "bullish"
    elif bearish_count >= max(2, available - 1) and bullish_count == 0:
        signal = "bearish"
    else:
        signal = "neutral"

    # ── OBV divergence warning ─────────────────────────────────
    if obv_slope is not None and abs(obv_slope) >= OBV_DIVERGENCE_THRESHOLD_PCT:
        if signal == "bullish" and obv_slope < 0:
            warnings.append("obv_divergence")
            rationale.append(
                f"⚠ OBV diverging: price momentum bullish but volume falling ({obv_slope:.1f}%)"
            )
        elif signal == "bearish" and obv_slope > 0:
            warnings.append("obv_divergence")
            rationale.append(
                f"⚠ OBV diverging: price momentum bearish but volume rising (+{obv_slope:.1f}%)"
            )

    # ── Strength ────────────────────────────────────────────────
    strength = _rsi_strength(rsi)

    return signal, strength, rationale, warnings
