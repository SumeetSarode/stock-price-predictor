"""Volatility indicators: ATR, Bollinger Bands, two distinct squeezes.

ATR is the secret hero of the predictor — it sets stop-loss sizing in
Step C. Every other indicator is about direction; ATR is about position
size. Make sure the math is right.

TWO SQUEEZE DEFINITIONS (DO NOT CONFLATE)
=========================================
The trading literature uses the word "squeeze" for two distinct,
unrelated constructs that we now expose under separate names:

1. BOLLINGER (BANDWIDTH-PERCENTILE) SQUEEZE — `bollinger_squeeze()`
   --------------------------------------------------------------
   Bollinger's own recommendation in *Bollinger on Bollinger Bands*
   (Wiley 2001), Ch. 11 "The Squeeze", p. 121-127:
       "A six-month low in bandwidth is a Squeeze."
   We operationalise this as: bandwidth in the lowest 20% of its last
   `lookback` (default 60) bars. Returns a continuous "is volatility
   currently at a historical low?" boolean.

2. TTM SQUEEZE (CARTER) — `ttm_squeeze()`
   --------------------------------------
   John Carter, *Mastering the Trade* (McGraw-Hill 2006/2009), Ch. 11:
   the TTM Squeeze fires when the upper Bollinger Band is INSIDE the
   upper Keltner Channel AND the lower BB is INSIDE the lower KC.
   Defaults: BB(20, 2.0), Keltner(20, 1.5*ATR). Carter calls the
   reverse transition (BBs popping back outside the KCs) the "fire"
   event — that's the breakout trigger.

   References:
   - Carter (2009), Ch. 11
   - StockCharts ChartSchool, "TTM Squeeze":
     https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/ttm-squeeze

DOWNSTREAM USAGE
----------------
- The volatility_signal classifier reads `ttm_squeeze.on` for the
  "squeeze regime" (volatility historically compressed AND coiled
  against trend channels — both conditions, not just BB width).
- The `bollinger_squeeze` boolean is exposed alongside as a sanity
  cross-check; the two usually agree but disagree in trending markets
  (BB width can be low because the trend is steady, even though the
  TTM compression hasn't engaged).
"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta

from price_predictor.analysis.trend import _safe_float


def latest_atr(df: pd.DataFrame, length: int = 14) -> float | None:
    """Latest ATR (Average True Range) in the same units as price.

    Wilder warmup guard (H7): pandas-ta uses Wilder's RMA, an EWMA with
    α = 1/N. Seed-value bias decays as `(1 − 1/N)^k`; only after ~5N bars
    does it fall below 1%. We require ≥ 5*length bars (= 70 bars for the
    default ATR-14) before publishing a value, per the convergence-guard
    derivation in Skoglund (2017) and Kirkpatrick & Dahlquist (2016).
    The previous 2*length = 28 minimum left ~10–15% seed bias in ATR —
    enough to push 2-ATR stop-loss sizing off by ~25%.
    """
    if len(df) < 5 * length:
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


def bollinger_squeeze(
    df: pd.DataFrame,
    length: int = 20,
    std: float = 2.0,
    lookback: int = 60,
    quantile: float = 0.20,
) -> bool | None:
    """Bollinger's bandwidth-percentile squeeze (Bollinger 2001, Ch. 11).

    Returns True when current bandwidth is in the lowest `quantile` (default
    20%) of its last `lookback` bars. Bollinger's own original heuristic was
    "a six-month low" (~125 trading days); we use a 60-bar / 20%-quantile
    relaxation that shows up in nearly every modern broker platform.

    Returns None if not enough history (need ≥ length + lookback bars).
    """
    if len(df) < length + lookback:
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
    threshold = _safe_float(recent.quantile(quantile))
    if current is None or threshold is None:
        return None
    return current <= threshold


def ttm_squeeze(
    df: pd.DataFrame,
    bb_length: int = 20,
    bb_std: float = 2.0,
    kc_length: int = 20,
    kc_scalar: float = 1.5,
) -> dict[str, bool | None]:
    """TTM Squeeze per John Carter (2009), *Mastering the Trade*, Ch. 11.

    The squeeze is "ON" when the Bollinger Bands are entirely INSIDE the
    Keltner Channels — i.e., normal-distribution-implied volatility is
    LOWER than ATR-based channel volatility. Coiling spring, breakout
    pending. The squeeze "FIRES" on the bar that BBs pop back outside KCs
    — Carter's recommended trade trigger.

    Args:
        bb_length, bb_std: Bollinger Band period and stdev (default 20, 2.0)
        kc_length, kc_scalar: Keltner period and ATR multiplier (default 20, 1.5)

    Returns:
        {
          "on":    bool | None — squeeze currently active?
          "fire":  bool | None — squeeze JUST released this bar?
          "bars_in_squeeze": int | None — how long it has been on (0 if off)
        }
        All None if df has insufficient history.
    """
    out = {"on": None, "fire": None, "bars_in_squeeze": None}
    n_needed = max(bb_length, kc_length) + 1   # need 1 prior bar for `fire`
    if len(df) < n_needed:
        return out

    bb = ta.bbands(df["close"], length=bb_length, std=bb_std)
    kc = ta.kc(df["high"], df["low"], df["close"],
               length=kc_length, scalar=kc_scalar)
    if bb is None or bb.empty or kc is None or kc.empty:
        return out

    bb_l_col = next((c for c in bb.columns if c.startswith("BBL_")), None)
    bb_u_col = next((c for c in bb.columns if c.startswith("BBU_")), None)
    kc_l_col = next((c for c in kc.columns if c.startswith("KCL")), None)
    kc_u_col = next((c for c in kc.columns if c.startswith("KCU")), None)
    if not all([bb_l_col, bb_u_col, kc_l_col, kc_u_col]):
        return out

    on_series = (bb[bb_u_col] < kc[kc_u_col]) & (bb[bb_l_col] > kc[kc_l_col])
    on_series = on_series.dropna()
    if on_series.empty:
        return out

    on_now = bool(on_series.iloc[-1])
    fire = (not on_now) and bool(on_series.iloc[-2]) if len(on_series) >= 2 else False

    # Count consecutive bars the squeeze has been active (0 if currently off)
    bars_in = 0
    if on_now:
        for v in reversed(on_series.tolist()):
            if v:
                bars_in += 1
            else:
                break

    return {"on": on_now, "fire": fire, "bars_in_squeeze": bars_in}


def volatility_snapshot(
    df: pd.DataFrame,
    atr_length: int,
    bb_params: tuple[int, float],
) -> dict:
    """One-shot volatility snapshot for the get_volatility tool.

    Surfaces BOTH squeeze definitions — see module docstring for why.
    Downstream `_volatility_signal` reads `ttm_squeeze.on` as the
    primary "squeeze regime" flag; `bollinger_squeeze` is exposed
    alongside for diagnostic transparency.
    """
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
        "bollinger_squeeze": bollinger_squeeze(df, length=bb_length, std=bb_std),
        "ttm_squeeze": ttm_squeeze(df, bb_length=bb_length, bb_std=bb_std),
    }
