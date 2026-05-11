"""Candlestick pattern detection — TA-Lib full 61-pattern dispatcher.

WHY TA-LIB
==========
Earlier versions hand-rolled 7 patterns (doji, hammer, shooting_star,
bullish/bearish engulfing, morning/evening star) because TA-Lib's install
pain was a perceived deal-breaker. With ta-lib 0.6 + Homebrew the install
is a one-liner (`brew install ta-lib`) and the Python wrapper is on PyPI.
The "pain" cost has been bought down to roughly zero; the upside (61
canonically-implemented patterns vs. 7 home-grown ones) is overwhelming.

Reference:
    Nison, Steve. *Japanese Candlestick Charting Techniques.* Prentice
    Hall, 1991. The TA-Lib `CDL*` family implements the formal recognizers
    derived from Nison plus several Bulkowski/Morris extensions documented
    in the TA-Lib source (`src/ta_func/ta_CDL*.c`).

PATTERN OUTPUT CONVENTION (TA-Lib)
==================================
Each `CDL*` function returns an int32 array same length as input:
    +200 / +100 / 0 / -100 / -200
- Sign indicates direction (+ bullish, - bearish, 0 = no signal).
- |magnitude| 100 = standard signal; 200 = high-confidence variant
  (e.g. CDLABANDONEDBABY at the close vs. CDLEVENINGSTAR generic — the
  former is rarer and stronger; TA-Lib reports it at 200).
- A handful of patterns are inherently directionally-neutral (the doji
  family + spinning top + high wave + harami cross + tristar). Those
  always emit +100 by convention; we override `direction` to "neutral"
  via `_NEUTRAL_PATTERNS` below.

PUBLIC API (UNCHANGED — backward-compatible)
============================================
- `detect_recent_patterns(df, lookback=5) -> list[dict]`
  Each dict: {"name", "bar_date", "bar_index", "direction", "confidence"}
- `is_doji`, `is_hammer`, `is_shooting_star`, `is_bullish_engulfing`,
  `is_bearish_engulfing`, `is_morning_star`, `is_evening_star`
  Single/multi-bar predicates retained for tests + thin-call use.
  These wrap the corresponding `CDL*` calls on a minimal-window slice.

GATING (still happens upstream)
===============================
Raw candlestick patterns fire constantly. The TOOL layer
(`technical_agent.tools._candlestick_gating`) filters by proximity to
swing levels; that's where actionability comes from. This module just
emits truthful detections.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import talib

# ─────────────────────────────────────────────────────────────────────
# Pattern registry — pretty_name → TA-Lib CDL function
# ─────────────────────────────────────────────────────────────────────
# Pretty-name keys keep our public API readable ("hammer" not "CDLHAMMER")
# and stable should we ever swap implementations again. Order is
# alphabetical so iteration is reproducible.
CDL_PATTERNS: dict[str, str] = {
    "two_crows":              "CDL2CROWS",
    "three_black_crows":      "CDL3BLACKCROWS",
    "three_inside":           "CDL3INSIDE",
    "three_line_strike":      "CDL3LINESTRIKE",
    "three_outside":          "CDL3OUTSIDE",
    "three_stars_in_south":   "CDL3STARSINSOUTH",
    "three_white_soldiers":   "CDL3WHITESOLDIERS",
    "abandoned_baby":         "CDLABANDONEDBABY",
    "advance_block":          "CDLADVANCEBLOCK",
    "belt_hold":              "CDLBELTHOLD",
    "breakaway":              "CDLBREAKAWAY",
    "closing_marubozu":       "CDLCLOSINGMARUBOZU",
    "concealing_baby_swallow":"CDLCONCEALBABYSWALL",
    "counterattack":          "CDLCOUNTERATTACK",
    "dark_cloud_cover":       "CDLDARKCLOUDCOVER",
    "doji":                   "CDLDOJI",
    "doji_star":              "CDLDOJISTAR",
    "dragonfly_doji":         "CDLDRAGONFLYDOJI",
    "engulfing":              "CDLENGULFING",
    "evening_doji_star":      "CDLEVENINGDOJISTAR",
    "evening_star":           "CDLEVENINGSTAR",
    "gap_side_by_side_white": "CDLGAPSIDESIDEWHITE",
    "gravestone_doji":        "CDLGRAVESTONEDOJI",
    "hammer":                 "CDLHAMMER",
    "hanging_man":            "CDLHANGINGMAN",
    "harami":                 "CDLHARAMI",
    "harami_cross":           "CDLHARAMICROSS",
    "high_wave":              "CDLHIGHWAVE",
    "hikkake":                "CDLHIKKAKE",
    "hikkake_modified":       "CDLHIKKAKEMOD",
    "homing_pigeon":          "CDLHOMINGPIGEON",
    "identical_three_crows":  "CDLIDENTICAL3CROWS",
    "in_neck":                "CDLINNECK",
    "inverted_hammer":        "CDLINVERTEDHAMMER",
    "kicking":                "CDLKICKING",
    "kicking_by_length":      "CDLKICKINGBYLENGTH",
    "ladder_bottom":          "CDLLADDERBOTTOM",
    "long_legged_doji":       "CDLLONGLEGGEDDOJI",
    "long_line":              "CDLLONGLINE",
    "marubozu":               "CDLMARUBOZU",
    "matching_low":           "CDLMATCHINGLOW",
    "mat_hold":               "CDLMATHOLD",
    "morning_doji_star":      "CDLMORNINGDOJISTAR",
    "morning_star":           "CDLMORNINGSTAR",
    "on_neck":                "CDLONNECK",
    "piercing":               "CDLPIERCING",
    "rickshaw_man":           "CDLRICKSHAWMAN",
    "rise_fall_three_methods":"CDLRISEFALL3METHODS",
    "separating_lines":       "CDLSEPARATINGLINES",
    "shooting_star":          "CDLSHOOTINGSTAR",
    "short_line":             "CDLSHORTLINE",
    "spinning_top":           "CDLSPINNINGTOP",
    "stalled_pattern":        "CDLSTALLEDPATTERN",
    "stick_sandwich":         "CDLSTICKSANDWICH",
    "takuri":                 "CDLTAKURI",
    "tasuki_gap":             "CDLTASUKIGAP",
    "thrusting":              "CDLTHRUSTING",
    "tristar":                "CDLTRISTAR",
    "unique_three_river":     "CDLUNIQUE3RIVER",
    "upside_gap_two_crows":   "CDLUPSIDEGAP2CROWS",
    "x_side_gap_three_methods":"CDLXSIDEGAP3METHODS",
}

# Sanity check at import time — if Homebrew/ta-lib ever adds/removes a
# CDL function and we drift, the next test run will tell us immediately
# rather than silently.
_TALIB_CDL = {f for f in dir(talib) if f.startswith("CDL")}
_REGISTERED = set(CDL_PATTERNS.values())
assert _REGISTERED == _TALIB_CDL, (
    f"TA-Lib CDL function registry drift. "
    f"Missing from CDL_PATTERNS: {_TALIB_CDL - _REGISTERED}. "
    f"Stale in CDL_PATTERNS: {_REGISTERED - _TALIB_CDL}."
)

# Patterns whose direction is inherently neutral (indecision / pause).
# TA-Lib emits +100 for every signal in these, but treating them as
# "bullish" would mislead the gating layer. These get direction="neutral"
# and the gating layer surfaces them if near EITHER swing level.
_NEUTRAL_PATTERNS: frozenset[str] = frozenset({
    "doji",
    "doji_star",
    "long_legged_doji",
    "dragonfly_doji",   # nominally bullish at bottoms but emitted unsigned
    "gravestone_doji",  # nominally bearish at tops but emitted unsigned
    "rickshaw_man",
    "spinning_top",
    "high_wave",
    "harami_cross",
    "tristar",
})

# ─────────────────────────────────────────────────────────────────────
# Public dispatcher
# ─────────────────────────────────────────────────────────────────────
@dataclass
class _PatternHit:
    """One (pattern, bar) detection. Internal — flattened to dict for output."""
    name: str
    bar_index: int      # NEGATIVE index from end of df (-1 = latest)
    bar_date: str
    direction: str      # "bullish" | "bearish" | "neutral"
    confidence: int     # 100 (standard) or 200 (high-confidence variant)


def _bar_date_str(idx_label) -> str:
    """Render the index label as YYYY-MM-DD if it has a .date(), else str(it)."""
    return str(idx_label.date()) if hasattr(idx_label, "date") else str(idx_label)


def _direction_for(name: str, raw: int) -> str:
    """Map (pattern_name, raw_signal) → direction string.

    Neutral-family patterns override the raw sign (TA-Lib emits +100 for
    them by convention but the meaning is indecision, not bullish).
    """
    if name in _NEUTRAL_PATTERNS:
        return "neutral"
    if raw > 0:
        return "bullish"
    if raw < 0:
        return "bearish"
    # raw == 0 should never reach here (caller filters), but be defensive
    return "neutral"


def detect_recent_patterns(df: pd.DataFrame, lookback: int = 5) -> list[dict]:
    """Find every TA-Lib candlestick pattern firing in the last `lookback` bars.

    Args:
        df: OHLCV DataFrame with columns 'open','high','low','close'. The
            full df is fed to TA-Lib (so multi-bar patterns have their
            necessary prior context); only hits inside the last `lookback`
            bars are returned.
        lookback: how many trailing bars to scan. Default 5 (≈ one trading
            week — matches the legacy hand-rolled implementation).

    Returns:
        List of dicts, each with:
            name          — pretty pattern name (e.g. "morning_star")
            bar_date      — ISO YYYY-MM-DD if df has a datetime index
            bar_index     — NEGATIVE index (-1 = latest bar)
            direction     — "bullish" | "bearish" | "neutral"
            confidence    — 100 (standard) or 200 (high-confidence variant)

        Sorted by (bar_index, name) — oldest bar first; alphabetical within
        each bar — for deterministic test output.

        Returns [] if df has fewer than 3 bars (insufficient history for
        the longest 3-bar patterns).
    """
    n = len(df)
    if n < 3:
        return []

    # Convert to contiguous float64 arrays once; TA-Lib copies them into C
    # land each call, so doing this up front is a single allocation.
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)

    # The window of indices we report on (always positive, df-relative)
    start_idx = max(0, n - lookback)
    hits: list[_PatternHit] = []

    for pretty_name, talib_name in CDL_PATTERNS.items():
        signal = getattr(talib, talib_name)(o, h, l, c)
        # signal is np.int32 array of len(n). Scan the lookback window.
        for i in range(start_idx, n):
            raw = int(signal[i])
            if raw == 0:
                continue
            hits.append(_PatternHit(
                name=pretty_name,
                bar_index=i - n,                       # negative
                bar_date=_bar_date_str(df.index[i]),
                direction=_direction_for(pretty_name, raw),
                confidence=abs(raw),                   # 100 or 200
            ))

    # Stable sort: by bar (oldest first), then alphabetical pattern name.
    hits.sort(key=lambda h: (h.bar_index, h.name))
    return [
        {
            "name": h.name,
            "bar_date": h.bar_date,
            "bar_index": h.bar_index,
            "direction": h.direction,
            "confidence": h.confidence,
        }
        for h in hits
    ]


# ─────────────────────────────────────────────────────────────────────
# Backward-compatible single/multi-bar predicates
# ─────────────────────────────────────────────────────────────────────
# These are RETAINED so existing call sites (and especially the test
# suite at tests/analysis/test_candlestick_patterns.py) continue to work
# unchanged. Each is a thin wrapper that builds the minimal OHLC array
# TA-Lib needs and returns the bool of the last-bar signal being non-zero.
#
# Why not just delete them? They guarantee that anything testing the
# legacy 7-pattern behavior still gets exercised, AND they double as a
# small consistency oracle: if TA-Lib's "hammer" disagrees with our
# previous home-grown definition on the test corpus, we'd see it
# instantly during a test run and decide whether to update the test or
# tighten the wrapper.
#
# IMPORTANT: TA-Lib's hammer/hanging-man/etc. need PRIOR CONTEXT (a
# downtrend or uptrend) to fire. For the legacy unit tests that pass a
# single isolated bar, we synthesize that context by prepending a short
# trending stub — just enough to satisfy the recognizer. Each predicate
# documents the stub it injects.

def _signal_from_arrays(
    fn_name: str,
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
) -> int:
    """Run a TA-Lib CDL function and return the LAST element (the focus bar)."""
    fn = getattr(talib, fn_name)
    return int(fn(o, h, l, c)[-1])


def _row_to_arrays(row: pd.Series) -> tuple[np.ndarray, ...]:
    """Single-row → length-1 OHLC arrays."""
    return (
        np.array([row["open"]], dtype=np.float64),
        np.array([row["high"]], dtype=np.float64),
        np.array([row["low"]], dtype=np.float64),
        np.array([row["close"]], dtype=np.float64),
    )


# A meaningful downtrend stub for bullish reversal patterns (hammer,
# inverted hammer). TA-Lib's recognizers compare the focus bar's body
# size to the AVERAGE body of the prior `BodyShort` bars (default 10),
# so the stub bodies must be visibly LARGER than the focus bar's body
# (~1 in the legacy unit tests) for the focus bar to register as
# "short-bodied". 15 bars trending down 1.5/bar with body~5 satisfies
# both "trend" and "average body" pre-checks across all reversal
# recognizers we expose via the legacy predicates.
def _downtrend_stub(n: int = 15, body: float = 5.0, slope: float = 1.5,
                    end_close: float = 100.0) -> tuple[np.ndarray, ...]:
    """Build a downtrending OHLC stub of `n` bars ending just above end_close."""
    closes = np.array([end_close + (n - i) * slope for i in range(n)], dtype=np.float64)
    opens = closes + body         # bearish bars: open > close
    highs = opens + 0.3
    lows = closes - 0.3
    return opens, highs, lows, closes


def _uptrend_stub(n: int = 15, body: float = 5.0, slope: float = 1.5,
                  end_close: float = 100.0) -> tuple[np.ndarray, ...]:
    """Build an uptrending OHLC stub of `n` bars ending just below end_close."""
    closes = np.array([end_close - (n - i) * slope for i in range(n)], dtype=np.float64)
    opens = closes - body         # bullish bars: open < close
    highs = closes + 0.3
    lows = opens - 0.3
    return opens, highs, lows, closes


def _with_downtrend(row: pd.Series) -> tuple[np.ndarray, ...]:
    """Prepend a downtrend stub to a single bar so reversal patterns can fire."""
    o0, h0, l0, c0 = _row_to_arrays(row)
    so, sh, sl, sc = _downtrend_stub(end_close=float(row["open"]))
    return (
        np.concatenate([so, o0]),
        np.concatenate([sh, h0]),
        np.concatenate([sl, l0]),
        np.concatenate([sc, c0]),
    )


def _with_uptrend(row: pd.Series) -> tuple[np.ndarray, ...]:
    """Prepend an uptrend stub for bearish-reversal patterns (shooting star)."""
    o0, h0, l0, c0 = _row_to_arrays(row)
    so, sh, sl, sc = _uptrend_stub(end_close=float(row["open"]))
    return (
        np.concatenate([so, o0]),
        np.concatenate([sh, h0]),
        np.concatenate([sl, l0]),
        np.concatenate([sc, c0]),
    )


def is_doji(row: pd.Series, body_ratio: float = 0.10) -> bool:
    """True if the bar is a doji (open ≈ close).

    Hand-rolled: TA-Lib's CDLDOJI uses a percentile-of-recent-bodies
    threshold which is undefined for a single isolated bar. The classic
    Nison rule "body ≤ 10% of range" is what the test suite expects.
    """
    o, h, l, c = (row["open"], row["high"], row["low"], row["close"])
    rng = h - l
    body = abs(c - o)
    return rng > 0 and body <= body_ratio * rng


def is_hammer(row: pd.Series) -> bool:
    """True if the bar is a hammer (small body at top, long lower shadow).

    Single-bar test → we prepend a 5-bar downtrend stub so TA-Lib's
    `CDLHAMMER` recognizer (which requires a prior downtrend) can fire.
    """
    return _signal_from_arrays("CDLHAMMER", *_with_downtrend(row)) > 0


def is_shooting_star(row: pd.Series) -> bool:
    """Mirror of hammer — small body at bottom, long upper shadow.

    Single-bar test → we prepend a 5-bar uptrend stub so TA-Lib's
    `CDLSHOOTINGSTAR` recognizer (which requires a prior uptrend) fires.
    """
    return _signal_from_arrays("CDLSHOOTINGSTAR", *_with_uptrend(row)) < 0


# ─── Engulfing real-body guard ────────────────────────────────────────
# Nison (1991, ch. 4 "Engulfing patterns") explicitly requires that the
# *prior* bar has a real body to be "engulfable" — "the second day's
# real body must engulf the first day's REAL BODY". TA-Lib's
# CDLENGULFING is shape-permissive: a (near-)doji prev with a sufficiently
# large current body can still emit a signal, because TA-Lib's body-size
# averaging gives marginal-doji bars a non-zero notional body. That's the
# M5 ambiguity called out in pred_logic_review §H/M.
#
# Our wrapper enforces an explicit Nison-aligned floor on the prior bar's
# body before consulting TA-Lib: prev body must be ≥ 10% of prev range.
# 10% is a deliberate choice (mirrors our own doji-detection threshold:
# `body / range < 0.1` is a doji, so anything ≥ 0.1 is "not a doji").
# Source: Nison 1991 ch. 4; threshold = our doji cutoff for consistency.
_ENGULFING_PREV_BODY_MIN_RATIO = 0.10


def _prev_has_real_body(prev: pd.Series) -> bool:
    """Nison real-body guard: prev body ≥ 10% of prev range.

    Returns True for normal bars, False for dojis / near-dojis where the
    prev bar has effectively no body to engulf.
    """
    body = abs(float(prev["close"]) - float(prev["open"]))
    rng = float(prev["high"]) - float(prev["low"])
    if rng <= 0:
        # degenerate four-equal bar — nothing to engulf
        return False
    return (body / rng) >= _ENGULFING_PREV_BODY_MIN_RATIO


def _two_bar_engulfing(prev: pd.Series, curr: pd.Series, *,
                       prior_trend: str) -> int:
    """Internal — returns TA-Lib's signed engulfing signal for the pair.

    `prior_trend` must be "down" (for bullish-engulfing tests) or "up"
    (for bearish-engulfing tests). TA-Lib's CDLENGULFING uses the average
    of recent body sizes when normalizing the engulfing magnitude, so we
    prepend a 15-bar trending stub to give it sane context.

    M5 GUARD: We short-circuit to 0 (no signal) when the prior bar has no
    real body (body < 10% of range). See `_prev_has_real_body` and the
    module-level note above for Nison's rationale.
    """
    if not _prev_has_real_body(prev):
        return 0
    stub_fn = _downtrend_stub if prior_trend == "down" else _uptrend_stub
    so, sh, sl, sc = stub_fn(end_close=float(prev["open"]))
    o = np.concatenate([so, [prev["open"],  curr["open"]]])
    h = np.concatenate([sh, [prev["high"],  curr["high"]]])
    l = np.concatenate([sl, [prev["low"],   curr["low"]]])
    c = np.concatenate([sc, [prev["close"], curr["close"]]])
    return int(talib.CDLENGULFING(o, h, l, c)[-1])


def is_bullish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    """True if (prev, curr) form a bullish engulfing per Nison."""
    return _two_bar_engulfing(prev, curr, prior_trend="down") > 0


def is_bearish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    """True if (prev, curr) form a bearish engulfing per Nison."""
    return _two_bar_engulfing(prev, curr, prior_trend="up") < 0


def _three_bar_signal(
    fn_name: str, b1: pd.Series, b2: pd.Series, b3: pd.Series, *,
    prior_trend: str,
) -> int:
    """Internal — runs a 3-bar TA-Lib pattern recognizer with a trend stub.

    `prior_trend` must be "down" (morning star is a bullish reversal
    requiring a prior downtrend) or "up" (evening star is the bearish
    mirror).
    """
    stub_fn = _downtrend_stub if prior_trend == "down" else _uptrend_stub
    so, sh, sl, sc = stub_fn(end_close=float(b1["open"]))
    o = np.concatenate([so, [b1["open"],  b2["open"],  b3["open"]]])
    h = np.concatenate([sh, [b1["high"],  b2["high"],  b3["high"]]])
    l = np.concatenate([sl, [b1["low"],   b2["low"],   b3["low"]]])
    c = np.concatenate([sc, [b1["close"], b2["close"], b3["close"]]])
    return int(getattr(talib, fn_name)(o, h, l, c)[-1])


def is_morning_star(b1: pd.Series, b2: pd.Series, b3: pd.Series) -> bool:
    """True if (b1, b2, b3) form a morning star reversal."""
    return _three_bar_signal("CDLMORNINGSTAR", b1, b2, b3, prior_trend="down") > 0


def is_evening_star(b1: pd.Series, b2: pd.Series, b3: pd.Series) -> bool:
    """True if (b1, b2, b3) form an evening star reversal."""
    return _three_bar_signal("CDLEVENINGSTAR", b1, b2, b3, prior_trend="up") < 0


__all__ = [
    "CDL_PATTERNS",
    "detect_recent_patterns",
    "is_bearish_engulfing",
    "is_bullish_engulfing",
    "is_doji",
    "is_evening_star",
    "is_hammer",
    "is_morning_star",
    "is_shooting_star",
]
