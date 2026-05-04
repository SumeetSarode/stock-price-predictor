"""Candlestick pattern context-gating -- pure function.

WHY THIS EXISTS
===============
Raw candlestick patterns fire constantly and pollute the LLM's reasoning.
The trading wisdom is: a hammer ONLY matters near support; a shooting
star ONLY matters near resistance.

This module takes the raw pattern list from analysis/candlestick_patterns
and filters it down to "patterns that occur near a level". Only survivors
get surfaced to the LLM.

GATING RULES
============
- Bullish patterns (hammer, bullish_engulfing, morning_star):
    surface only if the bar's LOW is within `1 * ATR` of the swing_low
- Bearish patterns (shooting_star, bearish_engulfing, evening_star):
    surface only if the bar's HIGH is within `1 * ATR` of the swing_high
- Neutral pattern (doji): surface if near EITHER level (indecision at
  any pivot is meaningful)

OUTPUT SHAPE
============
Each surviving pattern gets enriched with:
  - "context": "near_support" | "near_resistance" | "near_either"
  - "level_price": the swing level it's near
  - "distance_pct": how close (% of price)
"""
from __future__ import annotations

import pandas as pd

# Pattern direction classifications
BULLISH_PATTERNS = frozenset({"hammer", "bullish_engulfing", "morning_star"})
BEARISH_PATTERNS = frozenset({"shooting_star", "bearish_engulfing", "evening_star"})
NEUTRAL_PATTERNS = frozenset({"doji"})


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
        patterns: list from detect_recent_patterns(); each is
                  {"name": str, "bar_date": str, "bar_index": int (negative)}
        df: the OHLCV DataFrame the patterns were detected from
        swing_high: latest swing-high price (resistance)
        swing_low: latest swing-low price (support)
        atr: latest ATR value (proximity threshold = 1 * ATR)

    Returns:
        Filtered list, each pattern enriched with "context", "level_price",
        and "distance_pct".

    If atr is None or both swing levels are None, returns [] -- we have
    no way to context-gate without those.
    """
    if atr is None or atr <= 0:
        return []
    if swing_high is None and swing_low is None:
        return []

    surviving: list[dict] = []
    for p in patterns:
        name = p["name"]
        idx = p["bar_index"]  # negative index, e.g. -1 = latest bar
        # Guard against malformed inputs
        if idx >= 0 or abs(idx) > len(df):
            continue
        bar = df.iloc[idx]
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])

        if name in BULLISH_PATTERNS:
            if swing_low is None:
                continue
            if abs(bar_low - swing_low) <= atr:
                surviving.append({
                    **p,
                    "context": "near_support",
                    "level_price": round(swing_low, 2),
                    "distance_pct": round(_pct_distance(bar_low, swing_low), 2),
                })
        elif name in BEARISH_PATTERNS:
            if swing_high is None:
                continue
            if abs(bar_high - swing_high) <= atr:
                surviving.append({
                    **p,
                    "context": "near_resistance",
                    "level_price": round(swing_high, 2),
                    "distance_pct": round(_pct_distance(bar_high, swing_high), 2),
                })
        elif name in NEUTRAL_PATTERNS:
            # Doji: surface if near either level
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
        # Unknown pattern names get dropped silently

    return surviving
