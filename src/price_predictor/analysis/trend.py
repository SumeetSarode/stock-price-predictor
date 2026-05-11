"""Trend indicators: SMA family, EMA, ADX.

INPUT
=====
All functions accept a pd.DataFrame with at minimum a 'close' column.
ADX additionally requires 'high' and 'low'.

OUTPUT CONTRACT
===============
Every function returns either a float (single latest value) or a small
dict of floats. NaN-safe: returns None when there isn't enough history
to produce a meaningful value.
"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta


def _safe_float(value) -> float | None:
    """Coerce a pandas/numpy value to float; return None if NaN/None/missing."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return f


def latest_sma(df: pd.DataFrame, length: int) -> float | None:
    """Latest simple moving average of close. None if not enough history."""
    if len(df) < length:
        return None
    series = ta.sma(df["close"], length=length)
    if series is None or series.empty:
        return None
    return _safe_float(series.iloc[-1])


def latest_ema(df: pd.DataFrame, length: int) -> float | None:
    """Latest exponential moving average of close."""
    if len(df) < length:
        return None
    series = ta.ema(df["close"], length=length)
    if series is None or series.empty:
        return None
    return _safe_float(series.iloc[-1])


def latest_adx(df: pd.DataFrame, length: int = 14) -> dict[str, float | None]:
    """Latest ADX + directional indicators.

    Returns dict with keys:
      - adx:  trend strength (0-100; >25 trending, <20 chop)
      - di_plus / di_minus: directional indicators

    Wilder warmup guard (H7): ADX is doubly Wilder-smoothed — first the
    True Range and Directional Movement are RMA-smoothed (length=N), then
    DX is RMA-smoothed AGAIN (length=N) to give ADX. The first valid bar
    is at 2N (Wilder 1978), but seed-bias only falls below 1% by ~10N
    bars. We require ≥ 10*length bars (= 140 for ADX-14) per the
    convergence-guard derivation in pred_logic_solutions §H7. The previous
    `2 * length = 28` minimum left massive seed bias on the second
    smoothing pass and was the largest accuracy gap in the trend cluster.
    """
    if len(df) < 10 * length:
        return {"adx": None, "di_plus": None, "di_minus": None}

    out = ta.adx(df["high"], df["low"], df["close"], length=length)
    if out is None or out.empty:
        return {"adx": None, "di_plus": None, "di_minus": None}

    return {
        "adx":      _safe_float(out[f"ADX_{length}"].iloc[-1]),
        "di_plus":  _safe_float(out[f"DMP_{length}"].iloc[-1]),
        "di_minus": _safe_float(out[f"DMN_{length}"].iloc[-1]),
    }


def trend_snapshot(
    df: pd.DataFrame,
    sma_lengths: list[int],
    ema_length: int,
    adx_length: int,
) -> dict:
    """One-shot trend snapshot bundling everything the trend cluster needs.

    Returns:
      {
        "close":    latest close,
        "sma":      {20: 1450.3, 50: 1420.1, 200: 1380.7},
        "ema":      1455.8,  "adx":      {"adx": 28.4, "di_plus": 22.1, "di_minus": 14.5},
        "above_sma": {20: True, 50: True, 200: True},
        "pct_above_sma": {20: 0.4, 50: 2.1, 200: 5.0},
      }
    """
    close = _safe_float(df["close"].iloc[-1]) if not df.empty else None
    sma_values = {n: latest_sma(df, n) for n in sma_lengths}
    ema_value = latest_ema(df, ema_length)
    adx_values = latest_adx(df, adx_length)

    # Derived: position relative to each SMA
    above_sma: dict[int, bool | None] = {}
    pct_above_sma: dict[int, float | None] = {}
    for n, sma in sma_values.items():
        if close is None or sma is None:
            above_sma[n] = None
            pct_above_sma[n] = None
        else:
            above_sma[n] = close > sma
            pct_above_sma[n] = round((close - sma) / sma * 100, 2)

    return {
        "close": close,
        "sma": sma_values,
        "ema": ema_value,
        "adx": adx_values,
        "above_sma": above_sma,
        "pct_above_sma": pct_above_sma,
    }
