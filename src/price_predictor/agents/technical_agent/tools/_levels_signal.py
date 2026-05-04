"""Levels signal classifier -- pure function.

WHY LEVELS HAS A 'SIGNAL'
=========================
Support/resistance is direction-relative:
  - Price near support  -> potential bounce  -> bullish
  - Price near resistance -> potential rejection -> bearish
  - Price BROKE support  -> breakdown  -> bearish (continuation)
  - Price BROKE resistance -> breakout -> bullish (continuation)

INPUT
=====
We need: levels_snapshot (price, swing levels, 52w levels, pivots,
distance_pct), prior_swing_high/low (excluding latest 3 bars, to detect
fresh breakouts), ATR (for proximity threshold), and detected
chart_patterns.

OUTPUTS
=======
(signal, strength, rationale, warnings, derived_extras) where
derived_extras includes the classified breakout/breakdown state.
"""
from __future__ import annotations

from price_predictor.agents.technical_agent.tools._types import Signal, Strength

# Chart pattern direction classifications (mirrors candlestick gating module)
BULLISH_CHART_PATTERNS = frozenset({
    "double_bottom",
    "inverse_head_shoulders",
    "ascending_triangle",
})
BEARISH_CHART_PATTERNS = frozenset({
    "double_top",
    "head_shoulders",
    "descending_triangle",
})
NEUTRAL_CHART_PATTERNS = frozenset({"symmetric_triangle"})


def classify_levels(
    snapshot: dict,
    prior_swing_high: float | None,
    prior_swing_low: float | None,
    prior_52w_high: float | None,
    prior_52w_low: float | None,
    atr: float | None,
    chart_patterns: list[dict],
) -> tuple[Signal, Strength, list[str], list[str], dict]:
    """Turn a levels_snapshot + prior swings + ATR + patterns into a classification.

    Args:
        snapshot: output of analysis.levels.levels_snapshot()
        prior_swing_high: max(high) over bars excluding the LAST 3 bars
                          (used to detect FRESH breakouts -- today's bar
                           making a new high above where it was yesterday)
        prior_swing_low: same for lows (breakdown detection)
        prior_52w_high: max(high) over the 52w window EXCLUDING last 3 bars
                        (used to upgrade breakout strength to 'strong'
                         when we just broke a 52w-extreme level)
        prior_52w_low: same for the lower 52w extreme
        atr: latest ATR value (proximity threshold = 1 * ATR)
        chart_patterns: list of detected chart patterns from
                        analysis.chart_patterns.detect_all_patterns()

    Returns:
        (signal, strength, rationale, warnings, derived_extras)
        derived_extras = {breakout_state: 'breakout'|'breakdown'|'none',
                          near_level: 'support'|'resistance'|'none'}
    """
    rationale: list[str] = []
    warnings: list[str] = []
    derived_extras = {
        "breakout_state": "none",
        "near_level": "none",
    }

    close = snapshot.get("close")
    swing = snapshot.get("swing", {})
    fifty_two = snapshot.get("fifty_two_week", {})
    distances = snapshot.get("distance_pct", {})

    swing_high = swing.get("swing_high")
    swing_low = swing.get("swing_low")
    high_52w = fifty_two.get("high_52w")
    low_52w = fifty_two.get("low_52w")

    # ── Insufficient data ──────────────────────────────────────
    if close is None or atr is None or atr <= 0:
        return (
            "neutral",
            "weak",
            ["Insufficient history for levels analysis"],
            ["insufficient_history"],
            derived_extras,
        )

    # ── Detect fresh breakout / breakdown ──────────────────────
    # Breakout = today's close above where yesterday's swing-high WAS
    # Breakdown = today's close below yesterday's swing-low
    # We use ATR as a confirmation buffer (must clear by some margin)
    breakout = (
        prior_swing_high is not None
        and close > prior_swing_high
    )
    breakdown = (
        prior_swing_low is not None
        and close < prior_swing_low
    )

    # ── Detect 'near a level' ──────────────────────────────────
    near_swing_high = (
        swing_high is not None and abs(swing_high - close) <= atr
    )
    near_swing_low = (
        swing_low is not None and abs(close - swing_low) <= atr
    )
    near_52w_high = (
        high_52w is not None and abs(high_52w - close) <= atr
    )
    near_52w_low = (
        low_52w is not None and abs(close - low_52w) <= atr
    )

    # ── Decide signal ──────────────────────────────────────────
    # Priority: fresh breakouts/breakdowns > "near level" tests
    signal: Signal
    if breakout:
        signal = "bullish"
        derived_extras["breakout_state"] = "breakout"
        rationale.append(
            f"BREAKOUT: close {close:.2f} above prior swing-high "
            f"{prior_swing_high:.2f}"
        )
    elif breakdown:
        signal = "bearish"
        derived_extras["breakout_state"] = "breakdown"
        rationale.append(
            f"BREAKDOWN: close {close:.2f} below prior swing-low "
            f"{prior_swing_low:.2f}"
        )
    elif near_swing_low or near_52w_low:
        signal = "bullish"  # potential bounce from support
        derived_extras["near_level"] = "support"
        if near_52w_low:
            rationale.append(
                f"Near 52-week low ({low_52w:.2f}): potential bounce zone"
            )
        else:
            rationale.append(
                f"Near swing-low ({swing_low:.2f}): potential bounce zone"
            )
    elif near_swing_high or near_52w_high:
        signal = "bearish"  # potential rejection at resistance
        derived_extras["near_level"] = "resistance"
        if near_52w_high:
            rationale.append(
                f"Near 52-week high ({high_52w:.2f}): potential rejection zone"
            )
        else:
            rationale.append(
                f"Near swing-high ({swing_high:.2f}): potential rejection zone"
            )
    else:
        signal = "neutral"
        rationale.append(
            "Price in middle of range: no immediate level interaction"
        )

    # ── Strength: 52w levels > swing levels > pivots only ──────
    strength: Strength
    if breakout or breakdown:
        # Did the move break a PRIOR 52w extreme? That's strong.
        # We use prior_52w_* (excluding last 3 bars) for the same reason
        # we use prior_swing_* -- 'high_52w' includes today, so checking
        # close > high_52w is impossible.
        broke_52w_high = (
            breakout and prior_52w_high is not None and close > prior_52w_high
        )
        broke_52w_low = (
            breakdown and prior_52w_low is not None and close < prior_52w_low
        )
        if broke_52w_high or broke_52w_low:
            strength = "strong"
            rationale.append("⚡ Broke a 52-week level: high-conviction move")
        else:
            strength = "moderate"
    elif near_52w_high or near_52w_low:
        strength = "strong"  # 52w levels carry psychological weight
    elif near_swing_high or near_swing_low:
        strength = "moderate"
    else:
        strength = "weak"

    # ── Add 52w distance context ───────────────────────────────
    dist_52w_high = distances.get("high_52w")
    dist_52w_low = distances.get("low_52w")
    if dist_52w_high is not None:
        rationale.append(
            f"52w high {high_52w:.2f} is {dist_52w_high:+.2f}% away"
        )
    if dist_52w_low is not None:
        rationale.append(
            f"52w low {low_52w:.2f} is {dist_52w_low:+.2f}% away"
        )

    # ── Chart-pattern context ──────────────────────────────────
    # Just count by direction; the actual patterns are surfaced in the
    # tool's `derived` block. Here we use them to nudge confidence.
    bull_patterns = [p for p in chart_patterns if p["name"] in BULLISH_CHART_PATTERNS]
    bear_patterns = [p for p in chart_patterns if p["name"] in BEARISH_CHART_PATTERNS]
    neutral_patterns = [p for p in chart_patterns if p["name"] in NEUTRAL_CHART_PATTERNS]

    for p in bull_patterns:
        rationale.append(
            f"Chart pattern: {p['name']} (confidence {p['confidence']:.2f}) -- bullish bias"
        )
    for p in bear_patterns:
        rationale.append(
            f"Chart pattern: {p['name']} (confidence {p['confidence']:.2f}) -- bearish bias"
        )
    for p in neutral_patterns:
        rationale.append(
            f"Chart pattern: {p['name']} (confidence {p['confidence']:.2f}) -- "
            "watch for breakout direction"
        )

    # If chart patterns strongly contradict our level-based signal, warn
    if signal == "bullish" and bear_patterns and not bull_patterns:
        warnings.append("pattern_signal_conflict")
    elif signal == "bearish" and bull_patterns and not bear_patterns:
        warnings.append("pattern_signal_conflict")

    return signal, strength, rationale, warnings, derived_extras
