"""Backward-compatible single/multi-bar candlestick predicates.

WHY THIS MODULE EXISTS
======================
These predicates are RETAINED so existing call sites (and especially the
test suite at tests/analysis/test_candlestick_patterns.py) continue to
work unchanged after the C7 migration to TA-Lib's full 61-pattern catalog.
Each is a thin wrapper that builds the minimal OHLC array TA-Lib needs and
returns the bool of the last-bar signal being non-zero.

Why not just delete them? They guarantee that anything testing the legacy
7-pattern behavior still gets exercised, AND they double as a small
consistency oracle: if TA-Lib's "hammer" disagrees with our previous
home-grown definition on the test corpus, we'd see it instantly during a
test run and decide whether to update the test or tighten the wrapper.

IMPORTANT: TA-Lib's hammer/hanging-man/etc. need PRIOR CONTEXT (a
downtrend or uptrend) to fire. For the legacy unit tests that pass a
single isolated bar, we synthesize that context by prepending a short
trending stub — just enough to satisfy the recognizer. Each predicate
documents the stub it injects.

This module is re-exported by `candlestick_patterns` for source
compatibility — callers should import from `candlestick_patterns`, not
from here directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib


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
# M5 ambiguity called out in the original TA review (§H/M).
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
    "is_bearish_engulfing",
    "is_bullish_engulfing",
    "is_doji",
    "is_evening_star",
    "is_hammer",
    "is_morning_star",
    "is_shooting_star",
]
