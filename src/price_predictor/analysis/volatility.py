"""Volatility indicators: ATR, Bollinger Bands.

ATR is the secret hero of the predictor -- it sets stop-loss sizing in
Step C. Every other indicator is about direction; ATR is about position
size. Make sure the math is right.
"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from price_predictor.analysis.trend import _safe_float


def latest_atr(df: pd.DataFrame, length: int = 14) -> float | None:
    """Latest ATR (Average True Range) in the same units as price."""
    if len(df) < 2 * length:
        return None
    series = ta.atr(df["high"], df["low"], df["close"], length=length)
    if series is None or series.empty:
        return None
    return _safe_float(series.iloc[-1])


def latest_bbands(
    df: pd.DataFrame,
    length: int = 20,
    std: float = 2.0,
) -> dict[str, float | None]:
    """Latest Bollinger Bands.

    Returns:
        {
          "lower": float, "middle": float, "upper": float,
          "bandwidth": float (%),  -- width as % of middle band
          "percent_b": float,      -- 0=lower band, 1=upper band
        }
    """
    keys = {"lower": None, "middle": None, "upper": None,
            "bandwidth": None, "percent_b": None}
    if len(df) < length:
        return keys

    bb = ta.bbands(df["close"], length=length, std=std)
    if bb is None or bb.empty:
        return keys

    # pandas-ta column convention: BBL_<len>_<std>_<dstd>, BBM_..., BBU_...,
    # BBB_... (bandwidth as %), BBP_... (%B).
    # The version installed yields BBL_20_2.0_2.0 (note duplicate trailing).
    # Locate columns dynamically to be tolerant of version drift.
    def _find(prefix: str) -> str | None:
        for col in bb.columns:
            if col.startswith(prefix):
                return col
        return None

    lower_col  = _find("BBL_")
    middle_col = _find("BBM_")
    upper_col  = _find("BBU_")
    bandw_col  = _find("BBB_")
    pctb_col   = _find("BBP_")

    if lower_col is None or middle_col is None or upper_col is None:
        return keys

    return {
        "lower":     _safe_float(bb[lower_col].iloc[-1]),
        "middle":    _safe_float(bb[middle_col].iloc[-1]),
        "upper":     _safe_float(bb[upper_col].iloc[-1]),
        "bandwidth": _safe_float(bb[bandw_col].iloc[-1]) if bandw_col else None,
        "percent_b": _safe_float(bb[pctb_col].iloc[-1]) if pctb_col else None,
    }


def bb_squeeze(
    df: pd.DataFrame,
    length: int = 20,
    std: float = 2.0,
    lookback: int = 60,
) -> bool | None:
    """Detect a Bollinger Band 'squeeze': bandwidth in the lowest 20% of
    its recent range. Often precedes a breakout.

    Returns None if not enough history.
    """
    if len(df) < max(length, lookback) + length:
        return None
    bb = ta.bbands(df["close"], length=length, std=std)
    if bb is None or bb.empty:
        return None

    bandw_col = next((c for c in bb.columns if c.startswith("BBB_")), None)
    if bandw_col is None:
        return None

    recent = bb[bandw_col].dropna().iloc[-lookback:]
    if recent.empty:
        return None

    current = _safe_float(recent.iloc[-1])
    threshold = _safe_float(recent.quantile(0.20))
    if current is None or threshold is None:
        return None
    return current <= threshold


def volatility_snapshot(
    df: pd.DataFrame,
    atr_length: int,
    bb_params: tuple[int, float],
) -> dict:
    """One-shot volatility snapshot for the get_volatility tool."""
    bb_length, bb_std = bb_params
    close = _safe_float(df["close"].iloc[-1]) if not df.empty else None
    atr = latest_atr(df, length=atr_length)
    atr_pct = None
    if close is not None and atr is not None and close != 0:
        atr_pct = round(atr / close * 100, 2)
    return {
        "atr": atr,
        "atr_pct_of_price": atr_pct,
        "bbands": latest_bbands(df, length=bb_length, std=bb_std),
        "squeeze": bb_squeeze(df, length=bb_length, std=bb_std),
    }
