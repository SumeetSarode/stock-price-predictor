"""Live technical analysis service — read-only adapter over `analysis/`.

Purpose
=======
The stock detail page exposes a "show me the receipts" tab strip:
indicators, candlestick patterns, and chart patterns. This service
computes those LIVE (re-fetching current OHLCV) so the user always
sees what the market looks like RIGHT NOW, not what it looked like
at prediction time.

Boundary
========
- READ-ONLY consumer of `analysis/*` and `data/_shared_cache`.
- Never mutates the prediction pipeline, schema, cache, or store.
- Never imports from agents/, prediction/, or pred_logic_solution/.

Performance
===========
- Uses the process-wide PriceCache singleton from data/_shared_cache,
  so successive calls for the same ticker reuse warm bars (the same
  cache the agent pipeline uses).
- All indicator snapshots are pure-Python pandas math: <100ms total
  on a typical 400-bar frame. Pattern detectors are a bit heavier
  (TA-Lib + scipy peak detection) but still well under 1s.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from price_predictor.analysis import (
    LEVELS_PRESETS,
    MOMENTUM_PRESETS,
    TREND_PRESETS,
    VOLATILITY_PRESETS,
)
from price_predictor.analysis.candlestick_patterns import detect_recent_patterns
from price_predictor.analysis.chart_patterns import detect_all_patterns
from price_predictor.analysis.levels import levels_snapshot
from price_predictor.analysis.momentum import momentum_snapshot
from price_predictor.analysis.trend import trend_snapshot
from price_predictor.analysis.volatility import volatility_snapshot
from price_predictor.data._shared_cache import get_cache

# ── Tuning constants ─────────────────────────────────────────────────
# 400 calendar days ≈ 250 trading days, enough headroom for the 200-day
# SMA and 52-week levels to compute without NaNs.
_LOOKBACK_DAYS = 400
# "standard" preset matches what the production agent uses by default.
_DEFAULT_PRESET = "standard"
# Last 5 trading days for candlestick scan — one trading week of context.
_CANDLESTICK_LOOKBACK = 5
# Floor below which most indicators are noise (matches detector contracts).
_MIN_BARS = 20


@dataclass(frozen=True, slots=True)
class LiveAnalysis:
    """Bundle of every analysis result the detail tabs render.

    Frozen because the page treats this as an immutable snapshot —
    if the user wants fresher numbers they hit refresh, which re-runs
    the service.
    """

    ticker: str
    as_of: datetime          # tz-naive local-ish; only used for display
    bars_used: int
    trend: dict[str, Any]
    momentum: dict[str, Any]
    volatility: dict[str, Any]
    levels: dict[str, Any]
    candlesticks: list[dict[str, Any]]
    chart_patterns: list[dict[str, Any]]


class AnalysisServiceError(Exception):
    """Raised when the service can't deliver a usable LiveAnalysis.

    Carries a user-facing `message` and optional `hint` so the route
    can render a friendly error partial without leaking stack traces.
    """

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


def _normalize(ticker: str) -> str:
    """Match the cache/store/watchlist normalization: UPPER + .NS suffix."""
    t = ticker.strip().upper()
    if not t.endswith(".NS"):
        t = f"{t}.NS"
    return t


async def _fetch_bars(ticker: str) -> pd.DataFrame:
    """Pull `_LOOKBACK_DAYS` of daily bars via the shared cache."""
    today = date.today()
    start = today - timedelta(days=_LOOKBACK_DAYS)
    cache = get_cache()
    return await cache.get(ticker, start, today, "1d")


async def compute_live_analysis(ticker: str) -> LiveAnalysis:
    """Fetch fresh OHLCV for `ticker` and run every analyzer.

    Returns a single `LiveAnalysis` snapshot. Raises
    `AnalysisServiceError` on fetch failure or insufficient history.
    """
    t = _normalize(ticker)

    try:
        df = await _fetch_bars(t)
    except Exception as exc:  # noqa: BLE001 — re-raise as service error
        raise AnalysisServiceError(
            f"Couldn't fetch price history for {t}.",
            hint=str(exc),
        ) from exc

    if df.empty or len(df) < _MIN_BARS:
        raise AnalysisServiceError(
            f"Not enough price history for {t} "
            f"(need ≥{_MIN_BARS} bars, got {len(df)}).",
            hint="Yahoo Finance may not have data for this ticker yet.",
        )

    trend_p = TREND_PRESETS[_DEFAULT_PRESET]
    mom_p = MOMENTUM_PRESETS[_DEFAULT_PRESET]
    vol_p = VOLATILITY_PRESETS[_DEFAULT_PRESET]
    lvl_p = LEVELS_PRESETS[_DEFAULT_PRESET]

    trend = trend_snapshot(
        df,
        sma_lengths=trend_p["sma"],
        ema_length=trend_p["ema"],
        adx_length=trend_p["adx"],
    )
    momentum = momentum_snapshot(
        df,
        rsi_length=mom_p["rsi"],
        macd_params=mom_p["macd"],
        stoch_params=mom_p["stoch"],
    )
    volatility = volatility_snapshot(
        df,
        atr_length=vol_p["atr"],
        bb_params=vol_p["bb"],
    )
    levels = levels_snapshot(df, swing_lookback=lvl_p["swing_lookback"])
    candles = detect_recent_patterns(df, lookback=_CANDLESTICK_LOOKBACK)
    charts = detect_all_patterns(df)

    return LiveAnalysis(
        ticker=t,
        as_of=datetime.now(),
        bars_used=len(df),
        trend=trend,
        momentum=momentum,
        volatility=volatility,
        levels=levels,
        candlesticks=candles,
        chart_patterns=charts,
    )
