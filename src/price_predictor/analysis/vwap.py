"""VWAP (Volume-Weighted Average Price) on daily OHLCV bars.

WHAT THIS MODULE PROVIDES
=========================
Two flavors of VWAP that are well-defined on daily bars (we have no
intraday data on the free-tier providers):

  * `anchored_vwap(df, anchor_date)` -- Brian Shannon's "anchored VWAP":
    cumulative VWAP from a chosen anchor date forward. Useful around
    catalysts (Budget day, post-earnings gap, RBI policy day).

  * `rolling_vwap(df, window=N)` -- N-day rolling VWAP. Behaves as a
    smoothed dynamic support/resistance line.

WHY VWAP ON DAILY BARS IS AN APPROXIMATION
==========================================
True intraday VWAP weights each tick by traded volume. On a daily bar
we only know (open, high, low, close, volume) -- we don't know what
fraction of the day's volume traded at each price. The standard daily
proxy is to use the **typical price** TP = (H + L + C) / 3 as the
single price for that bar's whole volume, then cumulate.

  daily_vwap_anchored[t] = Σ(TP_i * V_i, i=anchor..t) / Σ(V_i, i=anchor..t)

This proxy is what every retail charting platform (TradingView,
StockCharts) does for daily-bar VWAP. It's biased by intra-day skew
(when the close is far from the average of H/L), but on diversified
liquid names the bias is small (< 0.5% on NIFTY 50 in our spot checks)
and consistent enough to be useful as a relative level.

CITATIONS
=========
  * Brian Shannon, *Maximum Trading Gains with Anchored VWAP*, Wiley
    2023. ISBN 1394196687.
  * StockCharts ChartSchool -- Anchored VWAP:
    https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/anchored-vwap
  * Berkowitz, Logue & Noser (1988), "The Total Cost of Transactions
    on the NYSE," *Journal of Finance* 43(1): 97-112 -- the original
    institutional-benchmark definition of VWAP.

WHY NOT IN levels.py
====================
VWAP needs the *volume* column (the other levels don't) and the
anchored variant carries a date parameter. Keeping it standalone keeps
`levels.py` purely price-geometry. The integration point is
`levels_snapshot()`, which calls `latest_vwap()` for the rolling default.
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from price_predictor.analysis.trend import _safe_float

# Default rolling window. 20 trading days ~= one calendar month, which is
# the same horizon as our Bollinger Band and the standard rolling VWAP
# window on StockCharts / TradingView.
DEFAULT_ROLLING_WINDOW = 20


def _typical_price(df: pd.DataFrame) -> pd.Series:
    """Standard daily VWAP proxy price per bar: TP = (H + L + C) / 3.

    See module docstring for why this is the right daily-bar proxy.
    """
    return (df["high"] + df["low"] + df["close"]) / 3.0


def anchored_vwap(
    df: pd.DataFrame, anchor_date: date | datetime | pd.Timestamp,
) -> pd.Series:
    """Cumulative VWAP from `anchor_date` forward (Brian Shannon, 2023).

    Args:
        df: OHLCV with a tz-aware DatetimeIndex (NSE bars). Must include
            'high', 'low', 'close', 'volume' columns.
        anchor_date: The date to anchor from. May be a `date`, `datetime`,
            or `pd.Timestamp`. The first bar on or AFTER this date is the
            first bar of the cumulative window.

    Returns:
        A pd.Series indexed identically to `df`. Entries BEFORE the
        anchor date are NaN; entries on/after carry the running VWAP.
        If the volume in the anchor window is zero, those entries are NaN.

    Raises:
        Nothing -- malformed inputs return an all-NaN series matching df.

    Example:
        >>> # Anchor to a post-earnings gap day
        >>> avwap = anchored_vwap(df, anchor_date=date(2025, 7, 28))
        >>> latest = avwap.iloc[-1]
    """
    if df.empty or "volume" not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    # Normalize anchor to a tz-naive date for index slicing. df.index is
    # tz-aware (Asia/Kolkata in our pipeline); .date attribute strips tz.
    if isinstance(anchor_date, pd.Timestamp):
        anchor = anchor_date.date()
    elif isinstance(anchor_date, datetime):
        anchor = anchor_date.date()
    else:
        anchor = anchor_date

    mask = df.index.date >= anchor
    if not mask.any():
        # Anchor is after the last bar -- nothing to cumulate.
        return pd.Series(np.nan, index=df.index, dtype=float)

    sub = df.loc[mask]
    tp = _typical_price(sub)
    pv = (tp * sub["volume"]).cumsum()
    vol = sub["volume"].cumsum()

    # Avoid div-by-zero on zero-volume halts. NaN where vol == 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap_sub = pv / vol.replace(0, np.nan)

    # Reindex back onto the full df index. Pre-anchor bars stay NaN.
    return vwap_sub.reindex(df.index)


def rolling_vwap(
    df: pd.DataFrame, window: int = DEFAULT_ROLLING_WINDOW,
) -> pd.Series:
    """Rolling N-day VWAP -- dynamic support/resistance smoother.

    Same TP * V / V cumulation as anchored VWAP, but over a moving
    `window`-bar window instead of from a fixed anchor.

    Args:
        df: OHLCV with 'high', 'low', 'close', 'volume' columns.
        window: Number of bars in the rolling window. Defaults to 20
            (~ one trading month, matching our Bollinger Band window).

    Returns:
        pd.Series indexed identically to df. NaN for the first
        `window - 1` bars (insufficient lookback).
    """
    if df.empty or "volume" not in df.columns or window < 1:
        return pd.Series(index=df.index, dtype=float)

    tp = _typical_price(df)
    pv = tp * df["volume"]
    num = pv.rolling(window).sum()
    den = df["volume"].rolling(window).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den.replace(0, np.nan)


def vwap_snapshot(
    df: pd.DataFrame,
    *,
    anchor_date: date | datetime | pd.Timestamp | None = None,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
) -> dict[str, float | None]:
    """One-shot VWAP snapshot for the levels-tool integration.

    Returns the latest values of:
      * rolling_vwap(window=rolling_window)
      * anchored_vwap(anchor_date)  -- only if `anchor_date` is provided

    Args:
        df: OHLCV bars (tz-aware DatetimeIndex).
        anchor_date: Optional anchor for the anchored VWAP. If None,
            the `vwap_anchored` field is None and only the rolling
            variant is computed.
        rolling_window: Window size for the rolling VWAP.

    Returns:
        {
          "vwap_rolling": float | None,    -- latest N-day rolling VWAP
          "vwap_anchored": float | None,   -- latest anchored VWAP (or None)
          "anchor_date": str | None,       -- ISO date of the anchor used
          "rolling_window": int,           -- the N used (echoed back)
        }
    """
    out: dict[str, float | None] = {
        "vwap_rolling": None,
        "vwap_anchored": None,
        "anchor_date": None,
        "rolling_window": rolling_window,
    }

    if df.empty:
        return out

    rolling = rolling_vwap(df, window=rolling_window)
    if not rolling.empty:
        out["vwap_rolling"] = _safe_float(rolling.iloc[-1])

    if anchor_date is not None:
        anchored = anchored_vwap(df, anchor_date=anchor_date)
        if not anchored.empty:
            out["vwap_anchored"] = _safe_float(anchored.iloc[-1])
        # Echo the anchor we used as a string for downstream/logging.
        if isinstance(anchor_date, (pd.Timestamp, datetime)):
            out["anchor_date"] = anchor_date.date().isoformat()
        else:
            out["anchor_date"] = anchor_date.isoformat()

    return out
