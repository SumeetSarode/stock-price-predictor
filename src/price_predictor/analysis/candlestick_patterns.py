"""Candlestick pattern detection — TA-Lib 61 + 4 hand-rolled = 65 patterns.

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
  Each dict: {"name", "bar_date", "bar_index", "direction", "confidence"}.
  Reversal-pattern hits (hammer / hanging_man / inverted_hammer /
  shooting_star) ALSO carry a `confirmed: bool | None` key (see Nison
  confirmation gate below).
- `is_doji`, `is_hammer`, `is_shooting_star`, `is_bullish_engulfing`,
  `is_bearish_engulfing`, `is_morning_star`, `is_evening_star`
  Single/multi-bar predicates retained for tests + thin-call use.
  These wrap the corresponding `CDL*` calls on a minimal-window slice.

HAND-ROLLED EXTENSIONS (TA-Lib has no equivalent)
=================================================
Four patterns are hand-rolled because TA-Lib doesn't ship them:

- `tweezer_top` / `tweezer_bottom` — two consecutive bars with
  near-matching highs (top, bearish) or lows (bottom, bullish), AND
  opposite colors. Tolerance is 0.05% of the bars' average close (a
  strict match — matching highs/lows are only useful if they're really
  matching). Reference: Nison 1991, ch. 6.
- `rising_window` / `falling_window` — a true gap with no overlap
  between adjacent bars. Bullish if `curr.low > prev.high`, bearish
  if `curr.high < prev.low`. Filtered by gap size > 0.5 x ATR(14)
  to suppress micro-gaps that are just noise on illiquid days.
  Reference: Nison 1991, ch. 5; ATR filter mirrors the gating layer
  proximity threshold for consistency.

NISON CONFIRMATION GATE (single-bar reversals)
==============================================
Nison (1991, ch. 3 "Reversal Patterns") is explicit: a hammer or
shooting star is a *candidate* reversal that requires next-day
confirmation to act on. Without confirmation, you trade noise.

For each hit in {hammer, hanging_man, inverted_hammer, shooting_star},
we attach `confirmed`:
  - `True`  — next bar's close moves IN the reversal's direction
              (bullish patterns: next_close > pattern_close;
               bearish patterns: next_close < pattern_close).
  - `False` — next bar's close moves AGAINST the reversal.
  - `None`  — no next bar exists (pattern is the latest bar).

The gating layer (technical_agent/tools/_candlestick_gating.py) is the
actionable filter; this field arms it with the Nison signal so the
synthesizer can downweight unconfirmed reversals.

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

# ──────────────────────────────────────────────────────────────
# Public dispatcher
# ──────────────────────────────────────────────────────────────
@dataclass
class _PatternHit:
    """One (pattern, bar) detection. Internal — flattened to dict for output."""
    name: str
    bar_index: int      # NEGATIVE index from end of df (-1 = latest)
    bar_date: str
    direction: str      # "bullish" | "bearish" | "neutral"
    confidence: int     # 100 (standard) or 200 (high-confidence variant)
    # Set ONLY for the 4 single-bar reversal patterns (Nison gate). For
    # all other patterns this stays None and is omitted from the output
    # dict so existing consumers see an unchanged schema.
    confirmed: bool | None = None


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

    # Hand-rolled extensions (TA-Lib has no equivalent recognizers).
    hits.extend(_scan_tweezers(o, h, l, c, df.index, start_idx, n))
    hits.extend(_scan_windows(o, h, l, c, df.index, start_idx, n))

    # Nison next-day confirmation gate for single-bar reversals. Mutates
    # in place rather than building a new list — cheaper, and the field
    # is purely additive.
    _apply_nison_confirmation(hits, c, n)

    # Stable sort: by bar (oldest first), then alphabetical pattern name.
    hits.sort(key=lambda h: (h.bar_index, h.name))
    return [_hit_to_dict(h) for h in hits]


# ──────────────────────────────────────────────────────────────
# Hand-rolled pattern detectors (not in TA-Lib)
# ──────────────────────────────────────────────────────────────
# Tweezer tolerance: highs/lows must match within 0.05% of the bars'
# average close. Strict on purpose — "matching highs" only matters when
# they're really matching. Reference: solutions doc; threshold cross-checked
# against Bulkowski 2008 Encyclopedia of Candlestick Charts which reports
# best edge when matching is exact at one-tick resolution.
_TWEEZER_TOL_RATIO = 0.0005

# Rising/falling window: gap size must exceed this multiple of ATR(14)
# to count as a real "window". 0.5 mirrors the candlestick gating layer's
# proximity threshold semantics (1.0 ATR for proximity, 0.5 ATR for
# gap-significance — a gap is a stronger signal so the bar can be smaller).
_WINDOW_ATR_MULT = 0.5
_WINDOW_ATR_PERIOD = 14

# Patterns that get a Nison next-day confirmation field. These are the
# four single-bar reversal patterns where the candle on its own is
# inconclusive; Nison (1991, ch. 3) explicitly requires next-bar
# confirmation before treating them as actionable.
_REVERSAL_PATTERNS_NEEDING_CONFIRMATION: frozenset[str] = frozenset({
    "hammer",
    "hanging_man",
    "inverted_hammer",
    "shooting_star",
})


def _scan_tweezers(
    o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray,
    index, start_idx: int, n: int,
) -> list[_PatternHit]:
    """Detect tweezer top / bottom 2-bar patterns in [start_idx, n).

    Tweezer top  (bearish): bullish prev + bearish curr + matching highs.
    Tweezer bot  (bullish): bearish prev + bullish curr + matching lows.

    Match tolerance is 0.05% of the avg close of the two bars, so the
    test scales naturally with price (works for both ₹100 and ₹5000 stocks).
    """
    out: list[_PatternHit] = []
    # Need at least one prior bar to form a pair, so the first scannable
    # current-bar index is max(start_idx, 1).
    first = max(start_idx, 1)
    for i in range(first, n):
        prev_open, prev_close = o[i - 1], c[i - 1]
        curr_open, curr_close = o[i], c[i]
        avg_close = (prev_close + curr_close) / 2.0
        if avg_close <= 0:
            continue
        tol = avg_close * _TWEEZER_TOL_RATIO

        prev_bullish = prev_close > prev_open
        prev_bearish = prev_close < prev_open
        curr_bullish = curr_close > curr_open
        curr_bearish = curr_close < curr_open

        # Tweezer top: matching highs at a high, color reversal down.
        if (
            prev_bullish and curr_bearish
            and abs(h[i - 1] - h[i]) <= tol
        ):
            out.append(_PatternHit(
                name="tweezer_top",
                bar_index=i - n,
                bar_date=_bar_date_str(index[i]),
                direction="bearish",
                confidence=100,
            ))
        # Tweezer bottom: matching lows at a low, color reversal up.
        elif (
            prev_bearish and curr_bullish
            and abs(lo[i - 1] - lo[i]) <= tol
        ):
            out.append(_PatternHit(
                name="tweezer_bottom",
                bar_index=i - n,
                bar_date=_bar_date_str(index[i]),
                direction="bullish",
                confidence=100,
            ))
    return out


def _scan_windows(
    o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray,
    index, start_idx: int, n: int,
) -> list[_PatternHit]:
    """Detect rising / falling windows (gaps) in [start_idx, n), ATR-filtered.

    Rising window (bullish): curr.low > prev.high  (gap up, no overlap).
    Falling window (bearish): curr.high < prev.low  (gap down, no overlap).

    The gap size must exceed `_WINDOW_ATR_MULT * ATR(14)` evaluated at the
    current bar to count, which suppresses dust-sized gaps on illiquid days.
    Returns [] until enough history exists to compute ATR.
    """
    if n < _WINDOW_ATR_PERIOD + 1:
        # Not enough history for ATR(14). Skipping is correct — the
        # alternative (using NaN-padded ATR) would silently fire on the
        # earliest bars where we have no scale to filter against.
        return []

    atr = talib.ATR(h, lo, c, timeperiod=_WINDOW_ATR_PERIOD)
    out: list[_PatternHit] = []
    first = max(start_idx, 1)
    for i in range(first, n):
        atr_i = float(atr[i])
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue
        threshold = _WINDOW_ATR_MULT * atr_i

        gap_up = lo[i] - h[i - 1]
        gap_dn = lo[i - 1] - h[i]
        if gap_up > threshold:
            out.append(_PatternHit(
                name="rising_window",
                bar_index=i - n,
                bar_date=_bar_date_str(index[i]),
                direction="bullish",
                confidence=100,
            ))
        elif gap_dn > threshold:
            out.append(_PatternHit(
                name="falling_window",
                bar_index=i - n,
                bar_date=_bar_date_str(index[i]),
                direction="bearish",
                confidence=100,
            ))
    return out


def _apply_nison_confirmation(
    hits: list[_PatternHit], c: np.ndarray, n: int,
) -> None:
    """Set `confirmed` on hammer/hanging_man/inverted_hammer/shooting_star.

    Mutates `hits` in place. For each eligible hit:
      - bullish reversal -> confirmed = next_close > pattern_close
      - bearish reversal -> confirmed = next_close < pattern_close
      - no next bar      -> confirmed = None  (latest bar, can't gate yet)

    Patterns not in `_REVERSAL_PATTERNS_NEEDING_CONFIRMATION` are left
    untouched (their `confirmed` stays None and is omitted from output).
    """
    for hit in hits:
        if hit.name not in _REVERSAL_PATTERNS_NEEDING_CONFIRMATION:
            continue
        # bar_index is negative; convert to positive df index.
        pos = n + hit.bar_index
        next_pos = pos + 1
        if next_pos >= n:
            hit.confirmed = None  # explicit; pattern is the latest bar
            continue
        pattern_close = float(c[pos])
        next_close = float(c[next_pos])
        if hit.direction == "bullish":
            hit.confirmed = next_close > pattern_close
        elif hit.direction == "bearish":
            hit.confirmed = next_close < pattern_close
        else:
            # Defensive — these 4 patterns shouldn't ever be neutral, but
            # if TA-Lib ever surprises us, leave it None rather than guess.
            hit.confirmed = None


def _hit_to_dict(h: _PatternHit) -> dict:
    """Flatten a _PatternHit to the public output dict.

    `confirmed` is included ONLY when the pattern was eligible for
    Nison confirmation (i.e. one of the 4 single-bar reversals). For
    every other pattern the key is omitted to keep the schema lean and
    backward-compatible with consumers that don't expect it.
    """
    out = {
        "name": h.name,
        "bar_date": h.bar_date,
        "bar_index": h.bar_index,
        "direction": h.direction,
        "confidence": h.confidence,
    }
    if h.name in _REVERSAL_PATTERNS_NEEDING_CONFIRMATION:
        out["confirmed"] = h.confirmed
    return out


# ──────────────────────────────────────────────────────────────
# Backward-compatible single/multi-bar predicates (re-exported)
# ──────────────────────────────────────────────────────────────
# Lifted to candlestick_legacy in C7 (file-size + cohesion). Importers
# of `candlestick_patterns` continue to see is_doji / is_hammer / etc.
# at the original location — zero churn for callers and tests.
from price_predictor.analysis.candlestick_legacy import (  # noqa: E402
    is_bearish_engulfing,
    is_bullish_engulfing,
    is_doji,
    is_evening_star,
    is_hammer,
    is_morning_star,
    is_shooting_star,
)

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

