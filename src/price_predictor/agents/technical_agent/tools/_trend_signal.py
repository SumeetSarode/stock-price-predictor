"""Trend signal classifier -- pure function, no I/O.

Input: trend_snapshot output (close, sma dict, ema, adx dict, derived).
Output: (signal, strength, rationale_bullets).

Kept separate from get_trend.py so we can unit-test the classification
rules without mocking fetch/cache.
"""
from __future__ import annotations

from price_predictor.agents.technical_agent.tools._types import Signal, Strength

# ADX thresholds for trend strength (TA convention)
ADX_STRONG = 40
ADX_MODERATE = 25
ADX_WEAK_FLOOR = 20


def _stack_score(above_sma: dict[int, bool | None]) -> int:
    """Count how many SMAs the close is above. None == 0 (uncountable)."""
    return sum(1 for v in above_sma.values() if v is True)


def _adx_strength(adx: float | None) -> Strength:
    """Map ADX value to strength label.

    >=40 strong, >=25 moderate, otherwise weak (no real trend).
    """
    if adx is None:
        return "weak"
    if adx >= ADX_STRONG:
        return "strong"
    if adx >= ADX_MODERATE:
        return "moderate"
    return "weak"


def classify_trend(snapshot: dict) -> tuple[Signal, Strength, list[str]]:
    """Turn a trend_snapshot into (signal, strength, rationale).

    RULES
    =====
    Signal:
      bullish  -- close above >=2 of the 3 SMAs AND +DI > -DI
      bearish  -- close below >=2 of the 3 SMAs AND -DI > +DI
      neutral  -- everything else (mixed signals or insufficient data)

    Strength: derived from ADX value alone (independent of direction).
      A strong-but-mixed setup is still 'neutral' for signal but reports
      its ADX strength. A clear direction with low ADX is 'bullish'/'bearish'
      but with 'weak' strength (the trend hasn't asserted itself yet).

    Rationale: human-readable bullets explaining each component. The LLM
    quotes these to avoid inventing reasoning.
    """
    rationale: list[str] = []

    close = snapshot.get("close")
    sma = snapshot.get("sma", {})
    above = snapshot.get("above_sma", {})
    pct_above = snapshot.get("pct_above_sma", {})
    adx_dict = snapshot.get("adx", {})
    adx_val = adx_dict.get("adx")
    di_plus = adx_dict.get("di_plus")
    di_minus = adx_dict.get("di_minus")

    # ── Insufficient-data short circuit ─────────────────────────
    if close is None or all(v is None for v in sma.values()):
        return "neutral", "weak", ["Insufficient price history for trend analysis"]

    # ── SMA stack ───────────────────────────────────────────────
    score = _stack_score(above)
    total_smas = sum(1 for v in above.values() if v is not None)
    sma_lengths_sorted = sorted(sma.keys())

    above_list = [n for n in sma_lengths_sorted if above.get(n) is True]
    below_list = [n for n in sma_lengths_sorted if above.get(n) is False]
    if above_list:
        rationale.append(
            f"Close above SMA-{', SMA-'.join(str(n) for n in above_list)}"
        )
    if below_list:
        rationale.append(
            f"Close below SMA-{', SMA-'.join(str(n) for n in below_list)}"
        )

    # Add the most informative pct distance (longest SMA we have)
    if pct_above:
        longest = max(pct_above.keys())
        if pct_above[longest] is not None:
            sign = "+" if pct_above[longest] >= 0 else ""
            rationale.append(
                f"Close is {sign}{pct_above[longest]:.2f}% relative to SMA-{longest}"
            )

    # ── ADX direction & strength ────────────────────────────────
    di_direction: str | None = None
    if di_plus is not None and di_minus is not None:
        if di_plus > di_minus:
            di_direction = "bullish"
            rationale.append(
                f"+DI ({di_plus:.1f}) > -DI ({di_minus:.1f}): bullish directional bias"
            )
        elif di_minus > di_plus:
            di_direction = "bearish"
            rationale.append(
                f"-DI ({di_minus:.1f}) > +DI ({di_plus:.1f}): bearish directional bias"
            )

    strength = _adx_strength(adx_val)
    if adx_val is not None:
        if strength == "strong":
            rationale.append(f"ADX {adx_val:.1f} indicates a strong trend")
        elif strength == "moderate":
            rationale.append(f"ADX {adx_val:.1f} indicates a developing trend")
        else:
            rationale.append(f"ADX {adx_val:.1f} indicates choppy/range-bound action")

    # ── Combine into final signal ──────────────────────────────
    # Need majority of SMAs aligned AND DI direction agreeing.
    if total_smas == 0:
        signal: Signal = "neutral"
    elif score >= max(2, total_smas - 1) and di_direction == "bullish":
        signal = "bullish"
    elif score <= min(1, total_smas - 2) and di_direction == "bearish":
        signal = "bearish"
    elif score == total_smas and di_direction != "bearish":
        # All SMAs above, even if DI is unavailable -> still bullish
        signal = "bullish"
    elif score == 0 and di_direction != "bullish":
        signal = "bearish"
    else:
        signal = "neutral"

    return signal, strength, rationale
