"""India VIX regime gate (H9d) — pure, I/O-free.

WHAT THIS MODULE PROVIDES
=========================
India VIX is NSE's implied-volatility index (the "fear gauge"): one
free, daily number that summarises the market's expected 30-day
volatility. We use it as a **regime gate**, NOT a directional signal:

  * A bullish setup during a *high-vol* regime deserves smaller size /
    wider stops; the same setup in a *low-vol* regime is cleaner.
  * VIX says nothing about direction — only about the weather.

  `vix_regime(series)` -> "low_vol" | "normal" | "high_vol" | "unknown"
  `vix_snapshot(series)` -> the regime + the numbers behind it.

WHY A MEDIAN, NOT A FIXED THRESHOLD
===================================
Absolute VIX levels drift across market epochs (a "high" 18 in a calm
year is a "low" 18 in a turbulent one). Comparing the latest value to
its own rolling median makes the gate self-calibrating: > 1.15× median
= elevated, < 0.85× median = compressed. This mirrors how squeeze /
percentile indicators elsewhere in this package normalise against
recent history rather than magic constants.

The fetcher lives in ``data/vix.py`` — this module stays pure math so
it obeys the analysis-package "no I/O" contract and is trivially
testable.

CITATIONS
=========
  * NSE India VIX page:
    https://www.nseindia.com/products-services/indices-indiavix-index
  * NSE India VIX methodology:
    https://www.niftyindices.com/Methodology/Method_India_VIX.pdf
"""
from __future__ import annotations

import pandas as pd

from price_predictor.analysis.trend import _safe_float

# Regime bands, expressed as multiples of the rolling median. Chosen to
# match the ±15% band from the H9d spec.
LOW_VOL_MULT = 0.85
HIGH_VOL_MULT = 1.15
DEFAULT_LOOKBACK = 60  # ~ one quarter of trading days


def vix_regime(vix_close: pd.Series, lookback: int = DEFAULT_LOOKBACK) -> str:
    """Classify the latest VIX reading against its rolling median.

    Args:
        vix_close: Chronological series of India VIX closes.
        lookback: Rolling-median window (trading days). Default 60.

    Returns:
        "low_vol"  — latest < LOW_VOL_MULT × median (compressed)
        "high_vol" — latest > HIGH_VOL_MULT × median (elevated)
        "normal"   — in between
        "unknown"  — not enough history, or a NaN/degenerate median
    """
    if vix_close is None or len(vix_close) < lookback:
        return "unknown"

    median = _safe_float(vix_close.rolling(lookback).median().iloc[-1])
    cur = _safe_float(vix_close.iloc[-1])
    if median is None or cur is None or median <= 0:
        return "unknown"

    if cur < LOW_VOL_MULT * median:
        return "low_vol"
    if cur > HIGH_VOL_MULT * median:
        return "high_vol"
    return "normal"


def vix_snapshot(
    vix_close: pd.Series, lookback: int = DEFAULT_LOOKBACK,
) -> dict:
    """One-shot VIX snapshot: the regime + the numbers behind it.

    Returns:
      {
        "value": float|None,     # latest VIX close
        "median": float|None,    # rolling-median reference
        "regime": str,           # low_vol | normal | high_vol | unknown
        "lookback": int,         # window used (echoed)
      }
    """
    out: dict = {
        "value": None, "median": None,
        "regime": "unknown", "lookback": lookback,
    }
    if vix_close is None or len(vix_close) == 0:
        return out

    out["value"] = _safe_float(vix_close.iloc[-1])
    if len(vix_close) >= lookback:
        out["median"] = _safe_float(
            vix_close.rolling(lookback).median().iloc[-1]
        )
    out["regime"] = vix_regime(vix_close, lookback=lookback)
    return out
