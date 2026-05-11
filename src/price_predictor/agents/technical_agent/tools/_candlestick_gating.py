"""Candlestick pattern context-gating — pure function.

WHY THIS EXISTS
===============
Raw candlestick patterns fire constantly and pollute the LLM's reasoning.
The trading wisdom is: a hammer ONLY matters near support; a shooting
star ONLY matters near resistance.

This module takes the raw pattern list from
`analysis.candlestick_patterns.detect_recent_patterns` and filters it
down to "patterns that occur near a level". Only survivors get surfaced
to the LLM.

GATING RULES (direction-driven)
===============================
Each pattern dict from the detector now carries a `direction` field
("bullish" | "bearish" | "neutral") sourced from TA-Lib's signed signal.
We read that directly instead of maintaining a static name → side map —
that keeps the gate trivially correct as new TA-Lib patterns ship.

- Bullish patterns: surface only if the bar's LOW is within 1*ATR of
  the swing_low.
- Bearish patterns: surface only if the bar's HIGH is within 1*ATR of
  the swing_high.
- Neutral patterns (doji family + spinning top + harami cross etc.):
  surface if near EITHER level (indecision at any pivot is meaningful).

Patterns missing a `direction` key are dropped silently — they cannot
be context-gated correctly without it. (This is defensive; the current
detector always emits one.)

OUTPUT SHAPE
============
Each surviving pattern dict gets enriched with:
  - "context": "near_support" | "near_resistance"
  - "level_price": the swing level it's near
  - "distance_pct": how close (% of the bar's level-side price)
"""
from __future__ import annotations

import pandas as pd

# Direction values we know how to handle. Anything else gets dropped.
_BULLISH = "bullish"
_BEARISH = "bearish"
_NEUTRAL = "neutral"
_KNOWN_DIRECTIONS = frozenset({_BULLISH, _BEARISH, _NEUTRAL})


def _pct_distance(a: float, b: float) -> float:
    """Absolute % distance between two prices, relative to b."""
    if b == 0:
        return float("inf")
    return abs(a - b) / abs(b) * 100


def gate_patterns(
    patterns: list[dict],
    df: pd.DataFrame,
    swing_high: float | None,
    swing_low: float | None,
    atr: float | None,
) -> list[dict]:
    """Filter raw patterns to those occurring near a relevant level.

    Args:
        patterns: list from `detect_recent_patterns()`. Each dict must have
                  "name", "bar_index" (negative), and "direction" keys.
                  "direction" must be one of {"bullish","bearish","neutral"}.
        df: the OHLCV DataFrame the patterns were detected from.
        swing_high: latest swing-high price (resistance).
        swing_low:  latest swing-low price (support).
        atr: latest ATR — proximity threshold = 1 * ATR.

    Returns:
        Filtered list, each surviving dict enriched with "context",
        "level_price", and "distance_pct". Returns [] if atr is missing
        or both swing levels are missing — without those we have no way
        to context-gate.
    """
    if atr is None or atr <= 0:
        return []
    if swing_high is None and swing_low is None:
        return []

    surviving: list[dict] = []
    for p in patterns:
        direction = p.get("direction")
        if direction not in _KNOWN_DIRECTIONS:
            continue

        idx = p["bar_index"]  # negative; -1 = latest bar
        # Defensive: malformed bar_index should never reach here, but the
        # detector contract isn't enforced at type-check time.
        if idx >= 0 or abs(idx) > len(df):
            continue
        bar = df.iloc[idx]
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])

        if direction == _BULLISH:
            if swing_low is None:
                continue
            if abs(bar_low - swing_low) <= atr:
                surviving.append({
                    **p,
                    "context": "near_support",
                    "level_price": round(swing_low, 2),
                    "distance_pct": round(_pct_distance(bar_low, swing_low), 2),
                })

        elif direction == _BEARISH:
            if swing_high is None:
                continue
            if abs(bar_high - swing_high) <= atr:
                surviving.append({
                    **p,
                    "context": "near_resistance",
                    "level_price": round(swing_high, 2),
                    "distance_pct": round(_pct_distance(bar_high, swing_high), 2),
                })

        else:  # _NEUTRAL — surface if near EITHER level
            near_low = swing_low is not None and abs(bar_low - swing_low) <= atr
            near_high = swing_high is not None and abs(bar_high - swing_high) <= atr
            if near_low and near_high:
                # Pick whichever is closer
                d_low = _pct_distance(bar_low, swing_low)
                d_high = _pct_distance(bar_high, swing_high)
                if d_low <= d_high:
                    context, level, dist = "near_support", swing_low, d_low
                else:
                    context, level, dist = "near_resistance", swing_high, d_high
                surviving.append({
                    **p,
                    "context": context,
                    "level_price": round(level, 2),
                    "distance_pct": round(dist, 2),
                })
            elif near_low:
                surviving.append({
                    **p,
                    "context": "near_support",
                    "level_price": round(swing_low, 2),
                    "distance_pct": round(_pct_distance(bar_low, swing_low), 2),
                })
            elif near_high:
                surviving.append({
                    **p,
                    "context": "near_resistance",
                    "level_price": round(swing_high, 2),
                    "distance_pct": round(_pct_distance(bar_high, swing_high), 2),
                })
            # else: neutral pattern far from both levels → drop.

    return surviving
