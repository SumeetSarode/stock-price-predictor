"""Ichimoku Kinko Hyo ("one-glance equilibrium chart") on daily bars.

WHAT THIS MODULE PROVIDES
=========================
The five classic Ichimoku lines plus a one-shot regime snapshot:

  * `ichimoku(df, ...)` -- the full five-line DataFrame (tenkan, kijun,
    senkou_a, senkou_b, chikou), aligned to df's index.
  * `ichimoku_snapshot(df, ...)` -- the latest values + the cluster
    signals a trend classifier actually cares about (price-vs-cloud,
    tenkan/kijun cross, and the forward "kumo twist").

THE FIVE LINES
==============
  * Tenkan-sen (conversion, 9):   (max high + min low) / 2 over 9 bars.
  * Kijun-sen (base, 26):         same midline over 26 bars.
  * Senkou Span A (leading A):    ((tenkan + kijun) / 2), plotted 26 bars
                                  AHEAD (shift +displacement).
  * Senkou Span B (leading B):    52-bar midline, plotted 26 bars ahead.
  * Chikou Span (lagging):        close, plotted 26 bars BEHIND
                                  (shift -displacement).

The "cloud" (kumo) is the band between Senkou A and B. Price above the
cloud = bullish regime; below = bearish; inside = no-man's-land.

A NOTE ON THE SHIFTED SPANS
===========================
Because Senkou A/B are shifted FORWARD by `displacement`, the value at
the latest index is the projection that was computed `displacement` bars
ago -- i.e. it IS the cloud sitting under today's price. Reading
`.iloc[-1]` therefore gives the correct "current cloud" for the price/
cloud comparison. The genuinely *future* cloud (used for the twist
signal) is the un-shifted projection over the last `displacement` bars,
computed separately in `_forward_cloud`.

WHY DAILY BARS ARE FINE
=======================
Ichimoku is defined on any timeframe; the canonical (9, 26, 52, 26)
parameters come from Hosoda's original work on daily charts (26 ~= a
Japanese trading month when Saturdays traded). No intraday data needed.

CITATIONS
=========
  * Goichi Hosoda, *Ichimoku Kinko Hyo* (1969).
  * Manesh Patel, *Trading with Ichimoku Clouds*, Wiley 2010.
    ISBN 978-0470609361.
  * StockCharts ChartSchool -- Ichimoku Cloud:
    https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/ichimoku-cloud
"""
from __future__ import annotations

import pandas as pd

from price_predictor.analysis.trend import _safe_float

# Canonical Hosoda parameters. Exposed as module constants so callers
# (and the technical agent's trend tool) can echo/override them.
DEFAULT_TENKAN = 9
DEFAULT_KIJUN = 26
DEFAULT_SENKOU_B = 52
DEFAULT_DISPLACEMENT = 26


def _midline(df: pd.DataFrame, window: int) -> pd.Series:
    """Ichimoku midline: (rolling max high + rolling min low) / 2."""
    return (df["high"].rolling(window).max() + df["low"].rolling(window).min()) / 2.0


def ichimoku(
    df: pd.DataFrame,
    *,
    tenkan_n: int = DEFAULT_TENKAN,
    kijun_n: int = DEFAULT_KIJUN,
    senkou_b_n: int = DEFAULT_SENKOU_B,
    displacement: int = DEFAULT_DISPLACEMENT,
) -> pd.DataFrame:
    """Compute the five Ichimoku lines, aligned to `df.index`.

    Args:
        df: OHLCV with 'high', 'low', 'close' columns and a DatetimeIndex.
        tenkan_n / kijun_n / senkou_b_n / displacement: standard params.

    Returns:
        A DataFrame with columns tenkan, kijun, senkou_a, senkou_b,
        chikou -- same index as df. Leading/trailing entries are NaN
        where the rolling window or the shift has no data. An empty or
        malformed df yields an empty (columns-only) DataFrame.
    """
    cols = ["tenkan", "kijun", "senkou_a", "senkou_b", "chikou"]
    if df.empty or not {"high", "low", "close"}.issubset(df.columns):
        return pd.DataFrame(columns=cols, dtype=float)

    tenkan = _midline(df, tenkan_n)
    kijun = _midline(df, kijun_n)
    senkou_a = ((tenkan + kijun) / 2.0).shift(displacement)
    senkou_b = _midline(df, senkou_b_n).shift(displacement)
    chikou = df["close"].shift(-displacement)

    return pd.DataFrame(
        {
            "tenkan": tenkan,
            "kijun": kijun,
            "senkou_a": senkou_a,
            "senkou_b": senkou_b,
            "chikou": chikou,
        },
        index=df.index,
    )


def _forward_cloud(
    df: pd.DataFrame,
    *,
    tenkan_n: int,
    kijun_n: int,
    senkou_b_n: int,
    displacement: int,
) -> tuple[pd.Series, pd.Series]:
    """The genuinely-future cloud: un-shifted Senkou A/B over the last
    `displacement` bars (what will sit under price over the next
    `displacement` sessions). Used only for the kumo-twist signal.
    """
    tenkan = _midline(df, tenkan_n)
    kijun = _midline(df, kijun_n)
    fwd_a = ((tenkan + kijun) / 2.0).iloc[-displacement:]
    fwd_b = _midline(df, senkou_b_n).iloc[-displacement:]
    return fwd_a, fwd_b


def ichimoku_snapshot(
    df: pd.DataFrame,
    *,
    tenkan_n: int = DEFAULT_TENKAN,
    kijun_n: int = DEFAULT_KIJUN,
    senkou_b_n: int = DEFAULT_SENKOU_B,
    displacement: int = DEFAULT_DISPLACEMENT,
) -> dict:
    """One-shot Ichimoku regime snapshot for the trend cluster.

    Returns:
      {
        "tenkan": float|None, "kijun": float|None,
        "senkou_a": float|None, "senkou_b": float|None,  # CURRENT cloud
        "chikou": float|None,                            # last known lag span
        "cloud_top": float|None, "cloud_bottom": float|None,
        "price_vs_cloud": "above"|"below"|"inside"|None,
        "tk_signal": "bullish"|"bearish"|"neutral"|None, # tenkan vs kijun
        "kumo_twist_ahead": bool|None,   # future Senkou A/B cross projected
        "params": {tenkan, kijun, senkou_b, displacement},
      }

    All numeric fields are NaN-safe (None when history is too short).
    """
    out: dict = {
        "tenkan": None, "kijun": None,
        "senkou_a": None, "senkou_b": None, "chikou": None,
        "cloud_top": None, "cloud_bottom": None,
        "price_vs_cloud": None, "tk_signal": None,
        "kumo_twist_ahead": None,
        "params": {
            "tenkan": tenkan_n, "kijun": kijun_n,
            "senkou_b": senkou_b_n, "displacement": displacement,
        },
    }
    if df.empty or not {"high", "low", "close"}.issubset(df.columns):
        return out

    lines = ichimoku(
        df, tenkan_n=tenkan_n, kijun_n=kijun_n,
        senkou_b_n=senkou_b_n, displacement=displacement,
    )
    if lines.empty:
        return out

    tenkan = _safe_float(lines["tenkan"].iloc[-1])
    kijun = _safe_float(lines["kijun"].iloc[-1])
    senkou_a = _safe_float(lines["senkou_a"].iloc[-1])
    senkou_b = _safe_float(lines["senkou_b"].iloc[-1])
    close = _safe_float(df["close"].iloc[-1])

    out["tenkan"] = tenkan
    out["kijun"] = kijun
    out["senkou_a"] = senkou_a
    out["senkou_b"] = senkou_b

    # Chikou: last non-NaN lagging-span value (the tail is NaN by design).
    chikou_series = lines["chikou"].dropna()
    if not chikou_series.empty:
        out["chikou"] = _safe_float(chikou_series.iloc[-1])

    # Cloud band + price position.
    if senkou_a is not None and senkou_b is not None:
        top = max(senkou_a, senkou_b)
        bottom = min(senkou_a, senkou_b)
        out["cloud_top"] = top
        out["cloud_bottom"] = bottom
        if close is not None:
            if close > top:
                out["price_vs_cloud"] = "above"
            elif close < bottom:
                out["price_vs_cloud"] = "below"
            else:
                out["price_vs_cloud"] = "inside"

    # Tenkan/Kijun signal.
    if tenkan is not None and kijun is not None:
        if tenkan > kijun:
            out["tk_signal"] = "bullish"
        elif tenkan < kijun:
            out["tk_signal"] = "bearish"
        else:
            out["tk_signal"] = "neutral"

    # Kumo twist: does the FUTURE cloud (Senkou A vs B, un-shifted, last
    # `displacement` bars) change sign? A twist flags a possible regime
    # change ~displacement bars out.
    fwd_a, fwd_b = _forward_cloud(
        df, tenkan_n=tenkan_n, kijun_n=kijun_n,
        senkou_b_n=senkou_b_n, displacement=displacement,
    )
    diff = (fwd_a - fwd_b).dropna()
    if len(diff) >= 2:
        signs = diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        nonzero = signs[signs != 0]
        out["kumo_twist_ahead"] = bool(
            nonzero.nunique() > 1
        ) if not nonzero.empty else False

    return out
