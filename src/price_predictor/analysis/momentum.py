"""Momentum indicators: RSI, MACD, Stochastic, OBV.

OBV is folded into momentum per the design discussion -- it's conceptually
a momentum/divergence indicator (volume-weighted price direction).
"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from price_predictor.analysis.trend import _safe_float


def latest_rsi(df: pd.DataFrame, length: int = 14) -> float | None:
    """Latest RSI.

    Wilder warmup guard (H7): RSI uses Wilder's RMA. Seed bias decays as
    `(1−1/N)^k`; only after ~5N bars does it fall below 1%. We require
    ≥ 5*length bars (= 70 for RSI-14). See `latest_atr` for the same
    derivation. The previous 2*length = 28 minimum left ~10–15% bias
    — enough to flip RSI from 58 to 65 (false bullish vote).
    """
    if len(df) < 5 * length:
        return None
    series = ta.rsi(df["close"], length=length)
    if series is None or series.empty:
        return None
    return _safe_float(series.iloc[-1])


def latest_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, float | None]:
    """Latest MACD (line + signal + histogram).

    Returns:
        {"macd": -0.5, "signal": -0.3, "histogram": -0.2,
         "cross": "bullish" | "bearish" | None}
    The 'cross' field detects a crossover ON THE LATEST BAR -- i.e. the
    histogram changed sign.
    Warmup guard (H8): MACD = EMA(close, fast) − EMA(close, slow), signal
    = EMA(MACD, signal). Each EMA is an EWMA; the signal line is
    EMA-of-EMA which compounds seed bias. We require ≥ 5*slow bars
    (= 130 for the default 12/26/9) per the conservative warmup band
    in pred_logic_solutions §H8 / pred_logic_review §H8. Earlier code
    used `slow + signal = 35` which left ~30% seed bias in the signal
    line — cross detection was effectively noise on small histories.
    """
    out_keys = {"macd": None, "signal": None, "histogram": None, "cross": None}
    if len(df) < 5 * slow:
        return out_keys

    macd_df = ta.macd(df["close"], fast=fast, slow=slow, signal=signal)
    if macd_df is None or macd_df.empty:
        return out_keys

    macd_col = f"MACD_{fast}_{slow}_{signal}"
    sig_col  = f"MACDs_{fast}_{slow}_{signal}"
    hist_col = f"MACDh_{fast}_{slow}_{signal}"

    hist_now = _safe_float(macd_df[hist_col].iloc[-1])
    hist_prev = (
        _safe_float(macd_df[hist_col].iloc[-2]) if len(macd_df) >= 2 else None
    )
    cross: str | None = None
    if hist_now is not None and hist_prev is not None:
        if hist_prev <= 0 < hist_now:
            cross = "bullish"
        elif hist_prev >= 0 > hist_now:
            cross = "bearish"

    return {
        "macd":      _safe_float(macd_df[macd_col].iloc[-1]),
        "signal":    _safe_float(macd_df[sig_col].iloc[-1]),
        "histogram": hist_now,
        "cross":     cross,
    }


def latest_stoch(
    df: pd.DataFrame,
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3,
) -> dict[str, float | None]:
    """Latest Stochastic oscillator.

    Returns:
        {"k": <0-100>, "d": <0-100>}
    """
    if len(df) < k + d:
        return {"k": None, "d": None}

    out = ta.stoch(df["high"], df["low"], df["close"], k=k, d=d, smooth_k=smooth_k)
    if out is None or out.empty:
        return {"k": None, "d": None}

    k_col = f"STOCHk_{k}_{d}_{smooth_k}"
    d_col = f"STOCHd_{k}_{d}_{smooth_k}"
    return {
        "k": _safe_float(out[k_col].iloc[-1]),
        "d": _safe_float(out[d_col].iloc[-1]),
    }


def latest_obv(df: pd.DataFrame) -> dict[str, float | None]:
    """Latest On-Balance Volume + recent slope.

    Returns:
        {"obv": <cumulative>, "slope_20": <pct change over last 20 bars>}
    """
    if df.empty or "volume" not in df.columns:
        return {"obv": None, "slope_20": None}

    series = ta.obv(df["close"], df["volume"])
    if series is None or series.empty:
        return {"obv": None, "slope_20": None}

    obv_now = _safe_float(series.iloc[-1])
    slope = None
    if len(series) >= 20:
        obv_then = _safe_float(series.iloc[-20])
        if obv_now is not None and obv_then is not None and obv_then != 0:
            slope = round((obv_now - obv_then) / abs(obv_then) * 100, 2)

    return {"obv": obv_now, "slope_20": slope}


def momentum_snapshot(
    df: pd.DataFrame,
    rsi_length: int,
    macd_params: tuple[int, int, int],
    stoch_params: tuple[int, int, int],
) -> dict:
    """One-shot momentum snapshot for the get_momentum tool."""
    fast, slow, signal = macd_params
    k, d, smooth_k = stoch_params
    return {
        "rsi": latest_rsi(df, length=rsi_length),
        "macd": latest_macd(df, fast=fast, slow=slow, signal=signal),
        "stoch": latest_stoch(df, k=k, d=d, smooth_k=smooth_k),
        "obv": latest_obv(df),
    }
