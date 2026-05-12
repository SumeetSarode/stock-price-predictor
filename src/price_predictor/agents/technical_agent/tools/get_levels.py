"""ADK tool: get_levels(ticker, sensitivity) -> dict.

The fourth and final cluster tool. The MOST integrative one -- it pulls
from THREE areas:

  1. analysis.levels -- swing high/low, 52w high/low, classic pivots
  2. analysis.chart_patterns -- double-top/bottom, H&S, triangles
  3. analysis.volatility -- ATR (used as proximity threshold for 'near level')

WHY ATR FOR PROXIMITY
=====================
"Near support" in absolute price terms is meaningless across stocks
(₹10 distance is huge for a ₹50 stock, nothing for a ₹5000 stock).
1 * ATR is the universal "one bar's typical move" yardstick.

BREAKOUT DETECTION
==================
We compare today's close against the swing-high computed EXCLUDING the
last 3 bars. This way a fresh new-high IS detected as a breakout
(otherwise swing_high would already include today's move and nothing
would ever count as a breakout).

CHART PATTERN INTEGRATION
=========================
detect_all_patterns() filters internally to confidence >= 0.7. We
surface them under derived.chart_patterns and use their bullish/bearish
direction to nudge rationale (and warn on conflicts).
"""
from __future__ import annotations

from datetime import date, timedelta

from price_predictor.agents.technical_agent.tools._levels_signal import (
    classify_levels,
)
from price_predictor.agents.technical_agent.tools._types import (
    ToolErrorResponse,
    ToolSuccessResponse,
)
from price_predictor.agents.technical_agent.tools.get_trend import (
    _normalize_ticker,
)
from price_predictor.analysis import LEVELS_PRESETS, validate_preset
from price_predictor.analysis.chart_patterns import detect_all_patterns
from price_predictor.analysis.levels import (
    levels_snapshot,
    prior_fifty_two_week_window,
)
from price_predictor.analysis.volatility import latest_atr
from price_predictor.data._shared_cache import get_cache
from price_predictor.data.prices import PriceFetchError
from price_predictor.kb.stocks import lookup as resolve_stock

LOOKBACK_DAYS = 750  # H7: ≥500 trading days for Wilder warmup
# Bars to exclude from "prior" swing/52w calculation when detecting fresh
# breakouts. 3 = today + 2 prior bars, so a 3-day breakout still counts.
BREAKOUT_EXCLUDE_BARS = 3


async def get_levels(
    ticker: str,
    sensitivity: str = "standard",
    *,
    as_of: date | None = None,
) -> dict:
    """Levels-cluster analysis for a ticker.

    ``as_of`` (keyword-only, default ``None``) pins the fetch window to a
    past trading date for honest backtest replay; ``None`` means "today".
    See ``get_trend`` for the full backtest contract.

    Args:
        ticker: Stock symbol. Indian stocks resolve to .NS automatically.
        sensitivity: 'standard' | 'sensitive' | 'smooth'.
                     Controls swing_lookback (15 / 30 / 60 bars).

    Returns:
        On success:
            {
              "status": "success",
              "ticker": "RELIANCE.NS",
              "as_of": "2026-04-28",
              "preset": "standard",
              "signal": "bullish" | "neutral" | "bearish",
              "strength": "strong" | "moderate" | "weak",
              "indicators": {
                  "close": float,
                  "swing_high": float, "swing_low": float,
                  "high_52w": float, "low_52w": float,
                  "pp": float, "r1": float, "r2": float, "s1": float, "s2": float,
                  "distance_pct_swing_high": float, ...
              },
              "derived": {
                  "breakout_state": "breakout" | "breakdown" | "none",
                  "near_level": "support" | "resistance" | "none",
                  "atr": float,                 -- proximity yardstick used
                  "chart_patterns": [           -- high-confidence only (>=0.7)
                      {"name": "double_bottom", "confidence": 0.85, ...}
                  ],
                  "pattern_count": int,
              },
              "rationale": [str, ...],
              "warnings": [str, ...]
            }
        On error:
            {"status": "error", "error_message": str, "ticker": str,
             "suggested_ticker"?: str}
    """
    # ── Validate args ──────────────────────────────────────────
    try:
        normalized = _normalize_ticker(ticker)
    except ValueError as e:
        return ToolErrorResponse(
            status="error", error_message=str(e), ticker=ticker
        )
    try:
        validate_preset(sensitivity)
    except ValueError as e:
        return ToolErrorResponse(
            status="error", error_message=str(e), ticker=normalized
        )

    # ── Fetch via shared cache ─────────────────────────────────
    end = as_of if as_of is not None else date.today()
    start = end - timedelta(days=LOOKBACK_DAYS)
    try:
        df = await get_cache().get(
            ticker=normalized, start=start, end=end, interval="1d"
        )
    except PriceFetchError as e:
        resolved = resolve_stock(normalized.replace(".NS", "").replace(".BO", ""))
        err: dict = {
            "status": "error",
            "error_message": str(e),
            "ticker": normalized,
        }
        if resolved is not None and resolved.yfinance_symbol != normalized:
            err["suggested_ticker"] = resolved.yfinance_symbol
        return err
    except (ValueError, KeyError) as e:
        return ToolErrorResponse(
            status="error", error_message=f"Data error: {e}", ticker=normalized
        )

    if df.empty:
        return ToolErrorResponse(
            status="error",
            error_message=f"No price data returned for {normalized}",
            ticker=normalized,
        )

    # ── Run primitives ─────────────────────────────────────────
    params = LEVELS_PRESETS[sensitivity]
    swing_lookback = params["swing_lookback"]

    snapshot = levels_snapshot(df, swing_lookback=swing_lookback)
    atr = latest_atr(df)

    # Prior swing high/low for breakout detection: exclude last N bars
    # so today's bar can BE the breakout
    prior_swing_high: float | None = None
    prior_swing_low: float | None = None
    if len(df) > swing_lookback + BREAKOUT_EXCLUDE_BARS:
        prior_window = df.iloc[
            -(swing_lookback + BREAKOUT_EXCLUDE_BARS) : -BREAKOUT_EXCLUDE_BARS
        ]
        prior_swing_high = float(prior_window["high"].max())
        prior_swing_low = float(prior_window["low"].min())

    # Same trick for the 52-week extremes -- needed so 52w breakouts
    # can be DETECTED at all (high_52w in snapshot includes today).
    # Calendar-aware: 52 calendar weeks back from the bar at
    # position -(BREAKOUT_EXCLUDE_BARS+1), so today's candle is free to
    # set the new extreme.
    prior_52w_high: float | None = None
    prior_52w_low: float | None = None
    win_52w = prior_fifty_two_week_window(df, exclude_last_bars=BREAKOUT_EXCLUDE_BARS)
    if not win_52w.empty:
        prior_52w_high = float(win_52w["high"].max())
        prior_52w_low = float(win_52w["low"].min())

    chart_patterns = detect_all_patterns(df)

    # ── Classify ─────────────────────────────────────────
    signal, strength, rationale, warnings, derived_extras = classify_levels(
        snapshot=snapshot,
        prior_swing_high=prior_swing_high,
        prior_swing_low=prior_swing_low,
        prior_52w_high=prior_52w_high,
        prior_52w_low=prior_52w_low,
        atr=atr,
        chart_patterns=chart_patterns,
    )

    # ── Flatten indicators for the LLM ─────────────────────────
    pivots = snapshot["pivots"]
    distances = snapshot["distance_pct"]
    indicators = {
        "close": snapshot["close"],
        "swing_high": snapshot["swing"]["swing_high"],
        "swing_low": snapshot["swing"]["swing_low"],
        "high_52w": snapshot["fifty_two_week"]["high_52w"],
        "low_52w": snapshot["fifty_two_week"]["low_52w"],
        "pp": pivots.get("pp"),
        "r1": pivots.get("r1"),
        "r2": pivots.get("r2"),
        "s1": pivots.get("s1"),
        "s2": pivots.get("s2"),
        "distance_pct_swing_high": distances.get("swing_high"),
        "distance_pct_swing_low": distances.get("swing_low"),
        "distance_pct_52w_high": distances.get("high_52w"),
        "distance_pct_52w_low": distances.get("low_52w"),
    }

    # ── Derived: breakout state + chart patterns ───────────────
    derived = {
        **derived_extras,  # breakout_state, near_level
        "atr": atr,
        "chart_patterns": chart_patterns,
        "pattern_count": len(chart_patterns),
    }

    as_of_str = (
        df.index[-1].date().isoformat()
        if hasattr(df.index[-1], "date")
        else end.isoformat()
    )

    return ToolSuccessResponse(
        status="success",
        ticker=normalized,
        as_of=as_of_str,
        preset=sensitivity,  # type: ignore[typeddict-item]
        signal=signal,
        strength=strength,
        indicators=indicators,
        derived=derived,
        rationale=rationale,
        warnings=warnings,
    )
