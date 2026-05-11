"""Price levels: swing high/low, 52-week high/low, classic pivot points.

Levels are PRICES, not scores. The signal layer in the tool turns
proximity-to-level into 'near_support', 'near_resistance', 'breakout', etc.
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from price_predictor.analysis.trend import _safe_float

# 52 calendar weeks. We slice the dataframe by date (not by row count) so
# that NSE holidays in the lookback period are handled automatically:
# whatever trading days exist within the last 365 calendar days from the
# most recent bar are what we consider. This replaces the previous
# hardcoded 252-trading-day approximation, which silently drifted by
# +/- 2 bars per year depending on holiday calendar.
#
# We use the dataframe's own date index rather than calling
# `pandas_market_calendars` here because the dataframe is already
# guaranteed to contain only trading-day bars (yfinance/jugaad both
# emit one row per NSE trading day). A calendar lookup would give the
# same result for two more imports.
_FIFTY_TWO_WEEKS = timedelta(weeks=52)


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
    """Highest high and lowest low over the last 52 calendar weeks.

    Calendar-aware (date-sliced from the dataframe's index) rather than
    bar-count-aware. This means NSE holidays within the window are
    naturally handled and the lookback is exactly "one year ago from
    the latest bar" regardless of how many trading days that contains.

    Psychological levels every Indian-market trader watches.
    """
    if df.empty:
        return {"high_52w": None, "low_52w": None}
    last_date = df.index[-1].date()
    cutoff = last_date - _FIFTY_TWO_WEEKS
    # df.index is tz-aware datetime; .date attr extracts pure dates
    window = df[df.index.date >= cutoff]
    if window.empty:
        # Edge case: only the latest bar is within 52 weeks (e.g. very
        # short history). Fall back to using whatever we have.
        window = df
    return {
        "high_52w": _safe_float(window["high"].max()),
        "low_52w":  _safe_float(window["low"].min()),
    }


def prior_fifty_two_week_window(
    df: pd.DataFrame, exclude_last_bars: int = 0,
) -> pd.DataFrame:
    """Slice df to bars within the prior 52 calendar weeks, excluding the
    most recent `exclude_last_bars` rows.

    Used for breakout detection: callers want "the 52w high computed from
    everything BEFORE today" so they can detect today's bar AS the breakout.

    Args:
        df: OHLCV with tz-aware datetime index.
        exclude_last_bars: Number of trailing rows to exclude (typically
            1-3 to give the breakout candle room to be the new extreme).

    Returns:
        Filtered DataFrame. Empty if not enough history. Calendar-aware,
        same logic as fifty_two_week_high_low but excluding the tail.
    """
    if df.empty or exclude_last_bars >= len(df):
        return df.iloc[0:0]  # empty slice preserving columns
    base = df.iloc[:-exclude_last_bars] if exclude_last_bars > 0 else df
    if base.empty:
        return base
    last_date = base.index[-1].date()
    cutoff = last_date - _FIFTY_TWO_WEEKS
    return base[base.index.date >= cutoff]


def classic_pivots(df: pd.DataFrame) -> dict[str, float | None]:
    """Classic floor-trader pivots from the most recent COMPLETED bar.

    Formulas (Person, "A Complete Guide to Technical Trading Tactics",
    2004; standard floor-trader pivot set):
        PP = (H + L + C) / 3
        R1 = 2*PP - L,            S1 = 2*PP - H
        R2 = PP + (H - L),        S2 = PP - (H - L)
        R3 = H + 2*(PP - L),      S3 = L - 2*(H - PP)

    R3/S3 are the breakout-extension targets used when price decisively
    breaks R2/S2 \u2014 standard inclusion in any complete pivot stack
    (CME Group floor-trader pivots, StockCharts ChartSchool reference).
    """
    keys = {
        "pp": None,
        "r1": None, "r2": None, "r3": None,
        "s1": None, "s2": None, "s3": None,
    }
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
        "r3": round(h + 2 * (pp - l), 2),
        "s3": round(l - 2 * (h - pp), 2),
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
