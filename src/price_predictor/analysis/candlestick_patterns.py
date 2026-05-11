"""Hand-rolled candlestick patterns for the momentum cluster.

WHY HAND-ROLLED
===============
The installed pandas-ta only ships 3 candlestick functions (doji, inside,
generic dispatcher). Most real ones (hammer, shooting star, engulfing,
morning/evening star) need ta-lib, which we explicitly avoided due to
install pain. The math for the 6 patterns we want is trivial -- shape
of one to three consecutive bars.

CONTEXT-GATING
==============
These detectors return per-bar booleans. The TOOL layer (get_momentum)
filters them: a hammer is only surfaced if it occurs near a level
(within 1*ATR of swing high/low). Without context, candlestick patterns
fire constantly on random bars and pollute the LLM's reasoning.

PATTERN DEFINITIONS USED
========================
Body  = |close - open|
Range = high - low
Upper shadow = high - max(open, close)
Lower shadow = min(open, close) - low
"Small body"   = body <= 0.30 * range  (one-third rule)
"Long body"    = body >= 0.60 * range
"""
from __future__ import annotations

import pandas as pd

# ── Helpers ─────────────────────────────────────────────────────────


def _bar_metrics(row: pd.Series) -> dict[str, float]:
    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    rng = h - l
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    return {
        "open": float(o), "high": float(h), "low": float(l), "close": float(c),
        "range": float(rng), "body": float(body),
        "upper": float(upper), "lower": float(lower),
        "bullish": c > o, "bearish": c < o,
    }


def _is_small_body(m: dict, ratio: float = 0.30) -> bool:
    return m["range"] > 0 and m["body"] <= ratio * m["range"]


def _is_long_body(m: dict, ratio: float = 0.60) -> bool:
    return m["range"] > 0 and m["body"] >= ratio * m["range"]


# ── Single-bar patterns ─────────────────────────────────────────────


def is_doji(row: pd.Series, body_ratio: float = 0.10) -> bool:
    """Open ~ close. Body is <= 10% of range. Indecision."""
    m = _bar_metrics(row)
    return m["range"] > 0 and m["body"] <= body_ratio * m["range"]


def is_hammer(row: pd.Series) -> bool:
    """Small body at top of range, long lower shadow (>=2x body),
    little/no upper shadow. Bullish reversal at support."""
    m = _bar_metrics(row)
    if not _is_small_body(m, ratio=0.35):
        return False
    if m["body"] == 0:
        return False
    return m["lower"] >= 2 * m["body"] and m["upper"] <= 0.3 * m["body"]


def is_shooting_star(row: pd.Series) -> bool:
    """Mirror of hammer. Small body at bottom, long upper shadow.
    Bearish reversal at resistance."""
    m = _bar_metrics(row)
    if not _is_small_body(m, ratio=0.35):
        return False
    if m["body"] == 0:
        return False
    return m["upper"] >= 2 * m["body"] and m["lower"] <= 0.3 * m["body"]


# ── Two-bar patterns ────────────────────────────────────────────────


# Minimum body size for a bar to be "engulfable" (as fraction of range).
# Nison's original definition (Japanese Candlestick Charting Techniques,
# 1991, ch. 6) requires the engulfed bar to have a real body — a doji or
# near-doji can be "engulfed" by almost anything, which produces noise.
_ENGULF_MIN_PREV_BODY_RATIO = 0.10


def is_bullish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    """Prev bearish (with a real body), current bullish, current body strictly
    engulfs prev body.

    Per Nison (1991) ch. 6: strict engulfing means current open BELOW prev
    close AND current close ABOVE prev open (not equal). Plus the prev bar
    must have a real body (>=10% of its range) so we don't fire on dojis.
    """
    pm, cm = _bar_metrics(prev), _bar_metrics(curr)
    if not pm["bearish"] or not cm["bullish"]:
        return False
    # Reject doji-as-prev (no real body to engulf)
    if pm["range"] == 0 or pm["body"] < _ENGULF_MIN_PREV_BODY_RATIO * pm["range"]:
        return False
    # STRICT inequality (Nison) — touching counts as inside, not engulfing
    return cm["open"] < pm["close"] and cm["close"] > pm["open"]


def is_bearish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    """Prev bullish (with a real body), current bearish, current body strictly
    engulfs prev body. Mirror of bullish engulfing — see that function for the
    Nison (1991) citation rationale.
    """
    pm, cm = _bar_metrics(prev), _bar_metrics(curr)
    if not pm["bullish"] or not cm["bearish"]:
        return False
    if pm["range"] == 0 or pm["body"] < _ENGULF_MIN_PREV_BODY_RATIO * pm["range"]:
        return False
    return cm["open"] > pm["close"] and cm["close"] < pm["open"]


# ── Three-bar patterns ─────────────────────────────────────────────


def is_morning_star(b1: pd.Series, b2: pd.Series, b3: pd.Series) -> bool:
    """Bearish long body, small-body bar (often gapping down), bullish long
    body that closes above midpoint of the first bar. Bullish reversal."""
    m1, m2, m3 = _bar_metrics(b1), _bar_metrics(b2), _bar_metrics(b3)
    if not (m1["bearish"] and _is_long_body(m1)):
        return False
    if not _is_small_body(m2):
        return False
    if not (m3["bullish"] and _is_long_body(m3)):
        return False
    midpoint_first = (m1["open"] + m1["close"]) / 2
    return m3["close"] > midpoint_first


def is_evening_star(b1: pd.Series, b2: pd.Series, b3: pd.Series) -> bool:
    """Mirror of morning star. Bullish long body, small body, bearish long
    body closing below midpoint of first."""
    m1, m2, m3 = _bar_metrics(b1), _bar_metrics(b2), _bar_metrics(b3)
    if not (m1["bullish"] and _is_long_body(m1)):
        return False
    if not _is_small_body(m2):
        return False
    if not (m3["bearish"] and _is_long_body(m3)):
        return False
    midpoint_first = (m1["open"] + m1["close"]) / 2
    return m3["close"] < midpoint_first


# ── Detection driver ──────────────────────────────────────────────


def detect_recent_patterns(df: pd.DataFrame, lookback: int = 5) -> list[dict]:
    """Find all candlestick patterns in the last `lookback` bars.

    Returns a list of dicts:
        [{"name": "hammer", "bar_date": "2025-04-25", "bar_index": -3}, ...]

    Sorted by bar_index ascending (oldest first).
    """
    patterns: list[dict] = []
    n = len(df)
    if n < 3:
        return patterns

    # Look at the last `lookback` bars (but ensure we have enough history
    # for 3-bar patterns).
    start = max(0, n - lookback)
    for i in range(start, n):
        row = df.iloc[i]
        bar_date = str(row.name.date()) if hasattr(row.name, "date") else str(i)

        # Single-bar
        if is_doji(row):
            patterns.append({"name": "doji", "bar_date": bar_date, "bar_index": i - n})
        if is_hammer(row):
            patterns.append({"name": "hammer", "bar_date": bar_date, "bar_index": i - n})
        if is_shooting_star(row):
            patterns.append({"name": "shooting_star", "bar_date": bar_date, "bar_index": i - n})

        # Two-bar
        if i >= 1:
            prev = df.iloc[i - 1]
            if is_bullish_engulfing(prev, row):
                patterns.append({"name": "bullish_engulfing", "bar_date": bar_date, "bar_index": i - n})
            if is_bearish_engulfing(prev, row):
                patterns.append({"name": "bearish_engulfing", "bar_date": bar_date, "bar_index": i - n})

        # Three-bar
        if i >= 2:
            b1 = df.iloc[i - 2]
            b2 = df.iloc[i - 1]
            if is_morning_star(b1, b2, row):
                patterns.append({"name": "morning_star", "bar_date": bar_date, "bar_index": i - n})
            if is_evening_star(b1, b2, row):
                patterns.append({"name": "evening_star", "bar_date": bar_date, "bar_index": i - n})

    return patterns
