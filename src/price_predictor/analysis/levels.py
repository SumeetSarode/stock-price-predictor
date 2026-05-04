"""Price levels: swing high/low, 52-week high/low, classic pivot points.

Levels are PRICES, not scores. The signal layer in the tool turns
proximity-to-level into 'near_support', 'near_resistance', 'breakout', etc.
"""
from __future__ import annotations

import pandas as pd

from price_predictor.analysis.trend import _safe_float

# 52 weeks ≈ 252 trading days for NSE
TRADING_DAYS_PER_YEAR = 252


def swing_high_low(df: pd.DataFrame, lookback: int = 30) -> dict[str, float | None]:
    """Highest high and lowest low over the last `lookback` bars.

    Used as the most recent actionable resistance / support.
    """
    if df.empty:
        return {"swing_high": None, "swing_low": None}
    window = df.iloc[-lookback:]
    return {
        "swing_high": _safe_float(window["high"].max()),
        "swing_low":  _safe_float(window["low"].min()),
    }


def fifty_two_week_high_low(df: pd.DataFrame) -> dict[str, float | None]:
    """Highest high and lowest low over the last 252 trading days.

    Psychological levels every Indian-market trader watches.
    """
    if df.empty:
        return {"high_52w": None, "low_52w": None}
    window = df.iloc[-TRADING_DAYS_PER_YEAR:]
    return {
        "high_52w": _safe_float(window["high"].max()),
        "low_52w":  _safe_float(window["low"].min()),
    }


def classic_pivots(df: pd.DataFrame) -> dict[str, float | None]:
    """Classic floor-trader pivots from the most recent COMPLETED bar.

    Formulas:
        PP = (H + L + C) / 3
        R1 = 2*PP - L,          S1 = 2*PP - H
        R2 = PP + (H - L),      S2 = PP - (H - L)
    """
    keys = {"pp": None, "r1": None, "r2": None, "s1": None, "s2": None}
    if df.empty:
        return keys
    last = df.iloc[-1]
    h, l, c = (
        _safe_float(last["high"]),
        _safe_float(last["low"]),
        _safe_float(last["close"]),
    )
    if h is None or l is None or c is None:
        return keys
    pp = (h + l + c) / 3
    return {
        "pp": round(pp, 2),
        "r1": round(2 * pp - l, 2),
        "s1": round(2 * pp - h, 2),
        "r2": round(pp + (h - l), 2),
        "s2": round(pp - (h - l), 2),
    }


def levels_snapshot(df: pd.DataFrame, swing_lookback: int = 30) -> dict:
    """One-shot levels snapshot for the get_levels tool.

    Adds derived 'distance from current price' for each level so the signal
    layer can answer 'near support?', 'near resistance?'.
    """
    close = _safe_float(df["close"].iloc[-1]) if not df.empty else None
    swing = swing_high_low(df, lookback=swing_lookback)
    fifty_two = fifty_two_week_high_low(df)
    pivots = classic_pivots(df)

    def _pct_distance(level: float | None) -> float | None:
        if close is None or level is None or close == 0:
            return None
        return round((level - close) / close * 100, 2)

    return {
        "close": close,
        "swing": swing,
        "fifty_two_week": fifty_two,
        "pivots": pivots,
        "distance_pct": {
            "swing_high":   _pct_distance(swing["swing_high"]),
            "swing_low":    _pct_distance(swing["swing_low"]),
            "high_52w":     _pct_distance(fifty_two["high_52w"]),
            "low_52w":      _pct_distance(fifty_two["low_52w"]),
        },
    }
