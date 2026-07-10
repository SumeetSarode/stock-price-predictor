"""Trend indicators: SMA family, EMA, ADX, MA crossovers.

INPUT
=====
All functions accept a pd.DataFrame with at minimum a 'close' column.
ADX additionally requires 'high' and 'low'.

OUTPUT CONTRACT
===============
Every function returns either a float (single latest value) or a small
dict of floats / strings. NaN-safe: returns None when there isn't enough
history to produce a meaningful value.

MA CROSSOVERS
=============
`detect_ma_cross()` returns the L3 "regime + last event" struct discussed
in pred_logic.md §3.2 MA Crossover. Always answers truthfully even when
no fresh cross fired -- the consumer (technical_agent) gets a regime
state plus a `bars_since_event` field, never a misleading None.
"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta

# Default pairs computed by trend_snapshot. SMA-50/200 is the textbook
# Golden Cross (Murphy 1999); EMA-9/21 is the swing-trader's faster pair
# (Pring 2002). See pred_logic.md §3.2 for sourcing and the
# data-snooping caveat (Sullivan-Timmermann-White 1999).
DEFAULT_MA_CROSS_PAIRS: list[tuple[str, int, int]] = [
    ("sma", 50, 200),
    ("ema", 9, 21),
]


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


def detect_ma_cross(
    df: pd.DataFrame,
    short: int,
    long: int,
    kind: str = "sma",
) -> dict[str, object]:
    """Detect moving-average crossover state for a given (short, long) pair.

    Returns the L3 struct from pred_logic.md §3.2:
      {
        "current":           "above" | "below" | None,
        "last_event":        "bullish" | "bearish" | None,
        "bars_since_event":  int | None,
        "short_ma":          float | None,
        "long_ma":           float | None,
      }

    Semantics:
      - "current" describes whether the SHORT MA sits above or below the LONG MA
        on the latest bar. None when either MA can't be computed.
      - "last_event" is the direction of the most recent crossover within the
        available history. None means no cross has occurred (always above or
        always below since the first valid bar of the longer MA).
      - "bars_since_event" is 0 when the cross fired on the latest bar.
      - "short_ma" / "long_ma" are the latest values for downstream display.

    `bullish` = short MA crossed from <= long up through > long.
    `bearish` = short MA crossed from >= long down through < long.

    Args:
        df: OHLCV DataFrame with at least a 'close' column.
        short: Period of the short MA (must be < long).
        long:  Period of the long MA.
        kind:  'sma' or 'ema'.

    Raises:
        ValueError: if short >= long, periods are non-positive, or kind is
                    not 'sma'/'ema'. We fail fast on caller misuse so that
                    bad calls don't silently return all-None structs.
    """
    if kind not in ("sma", "ema"):
        raise ValueError(f"kind must be 'sma' or 'ema', got {kind!r}")
    if short <= 0 or long <= 0:
        raise ValueError(f"periods must be positive, got short={short} long={long}")
    if short >= long:
        raise ValueError(f"short ({short}) must be < long ({long})")

    empty: dict[str, object] = {
        "current": None,
        "last_event": None,
        "bars_since_event": None,
        "short_ma": None,
        "long_ma": None,
    }

    # Need at least `long` bars before the long MA produces its first value.
    if len(df) < long:
        return empty

    ma_func = ta.sma if kind == "sma" else ta.ema
    short_series = ma_func(df["close"], length=short)
    long_series = ma_func(df["close"], length=long)
    if short_series is None or long_series is None:
        return empty

    diff = (short_series - long_series).dropna()
    if diff.empty:
        return empty
    # Current regime from the latest valid diff value.
    current = "above" if diff.iloc[-1] > 0 else "below"

    # Walk backwards looking for the most recent sign change. We compare
    # each bar to its predecessor; the first sign change found is the
    # most recent cross.
    last_event: str | None = None
    bars_since_event: int | None = None
    diff_values = diff.to_numpy()
    last_idx = len(diff_values) - 1
    for i in range(last_idx, 0, -1):
        prev = diff_values[i - 1]
        curr = diff_values[i]
        # bullish cross: was <= 0, now > 0
        if prev <= 0 < curr:
            last_event = "bullish"
            bars_since_event = last_idx - i
            break
        # bearish cross: was >= 0, now < 0
        if prev >= 0 > curr:
            last_event = "bearish"
            bars_since_event = last_idx - i
            break

    return {
        "current": current,
        "last_event": last_event,
        "bars_since_event": bars_since_event,
        "short_ma": _safe_float(short_series.iloc[-1]),
        "long_ma": _safe_float(long_series.iloc[-1]),
    }


def _ma_cross_key(kind: str, short: int, long: int) -> str:
    """Stable string key for a (kind, short, long) triple.

    e.g. ('sma', 50, 200) -> 'sma_50_200'. Used to namespace
    trend_snapshot()['ma_crosses'] entries so downstream code can read
    them by name without juggling tuples.
    """
    return f"{kind}_{short}_{long}"


def detect_ma_crosses(
    df: pd.DataFrame,
    pairs: list[tuple[str, int, int]] | None = None,
) -> dict[str, dict[str, object]]:
    """Run detect_ma_cross() for each (kind, short, long) triple in `pairs`.

    `pairs` defaults to DEFAULT_MA_CROSS_PAIRS (SMA-50/200 + EMA-9/21).
    Returned dict is keyed by `_ma_cross_key()` so trend_snapshot consumers
    can do snap['ma_crosses']['sma_50_200']['last_event'].
    """
    if pairs is None:
        pairs = DEFAULT_MA_CROSS_PAIRS
    return {
        _ma_cross_key(kind, short, long): detect_ma_cross(df, short, long, kind=kind)
        for (kind, short, long) in pairs
    }


def trend_snapshot(
    df: pd.DataFrame,
    sma_lengths: list[int],
    ema_length: int,
    adx_length: int,
    ma_cross_pairs: list[tuple[str, int, int]] | None = None,
) -> dict:
    """One-shot trend snapshot bundling everything the trend cluster needs.

    Returns:
      {
        "close":    latest close,
        "sma":      {20: 1450.3, 50: 1420.1, 200: 1380.7},
        "ema":      1455.8,  "adx":      {"adx": 28.4, "di_plus": 22.1, "di_minus": 14.5},
        "above_sma": {20: True, 50: True, 200: True},
        "pct_above_sma": {20: 0.4, 50: 2.1, 200: 5.0},
        "ma_crosses": {
            "sma_50_200": {"current": "above", "last_event": "bullish",
                            "bars_since_event": 12, "short_ma": 1420, "long_ma": 1380},
            "ema_9_21":   {...},
        },
      }

    `ma_cross_pairs` defaults to DEFAULT_MA_CROSS_PAIRS (SMA-50/200 +
    EMA-9/21). Pass an explicit list to query custom pairs (each tuple
    is `(kind, short, long)` where kind is 'sma' or 'ema').
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

    ma_crosses = detect_ma_crosses(df, pairs=ma_cross_pairs)

    # Ichimoku cloud regime (H9b). Lazy import avoids a circular
    # dependency: ichimoku.py imports `_safe_float` from this module.
    from price_predictor.analysis.ichimoku import ichimoku_snapshot

    return {
        "close": close,
        "sma": sma_values,
        "ema": ema_value,
        "adx": adx_values,
        "above_sma": above_sma,
        "pct_above_sma": pct_above_sma,
        "ma_crosses": ma_crosses,
        "ichimoku": ichimoku_snapshot(df),
    }
