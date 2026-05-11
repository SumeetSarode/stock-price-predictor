"""ADK tool: get_momentum(ticker, sensitivity) -> dict.

Second cluster tool. Same pattern as get_trend, with one wrinkle:
candlestick patterns are detected, then context-gated by proximity to
swing levels (within 1*ATR). Only "patterns near a level" get surfaced
to the LLM.

CROSS-CLUSTER IMPORTS
=====================
This tool reaches into THREE analysis modules:
  - analysis/momentum.py    (the cluster's own primitives)
  - analysis/levels.py      (swing high/low for gating)
  - analysis/volatility.py  (ATR for proximity threshold)
That's by design. Tools COMPOSE primitives across clusters; primitives
remain blissfully unaware of each other.
"""
from __future__ import annotations

from datetime import date, timedelta

from price_predictor.agents.technical_agent.tools._candlestick_gating import (
    gate_patterns,
)
from price_predictor.agents.technical_agent.tools._momentum_signal import (
    classify_momentum,
)
from price_predictor.agents.technical_agent.tools._types import (
    ToolErrorResponse,
    ToolSuccessResponse,
)
from price_predictor.agents.technical_agent.tools.get_trend import (
    _normalize_ticker,
)
from price_predictor.analysis import (
    LEVELS_PRESETS,
    MOMENTUM_PRESETS,
    VOLATILITY_PRESETS,
    validate_preset,
)
from price_predictor.analysis.candlestick_patterns import detect_recent_patterns
from price_predictor.analysis.levels import swing_high_low
from price_predictor.analysis.momentum import momentum_snapshot
from price_predictor.analysis.volatility import latest_atr
from price_predictor.data._shared_cache import get_cache
from price_predictor.data.prices import PriceFetchError
from price_predictor.kb.stocks import lookup as resolve_stock

# Same lookback as get_trend -- shared cache means no duplicate fetch
LOOKBACK_DAYS = 750  # H7: ≥500 trading days for Wilder warmup

# How many recent bars to scan for candlestick patterns
PATTERN_LOOKBACK_BARS = 5


async def get_momentum(ticker: str, sensitivity: str = "standard") -> dict:
    """Momentum-cluster analysis for a ticker.

    Args:
        ticker: Stock symbol. Indian stocks resolve to .NS automatically.
        sensitivity: 'standard' | 'sensitive' | 'smooth'.
                     - standard:  RSI-14, MACD(12,26,9), Stoch(14,3,3)
                     - sensitive: RSI-9,  MACD(8,17,9),  Stoch(9,3,3)   (faster)
                     - smooth:    RSI-21, MACD(19,39,9), Stoch(21,5,5)  (slower)

    Returns:
        On success:
            {
              "status": "success",
              "ticker": "RELIANCE.NS",
              "as_of": "2026-04-28",
              "preset": "standard",
              "signal": "bullish" | "neutral" | "bearish",
              "strength": "weak" | "moderate" | "strong",
              "indicators": {rsi, macd_line, macd_signal, macd_histogram,
                             macd_cross, stoch_k, stoch_d, obv, obv_slope_20},
              "derived": {
                "candlestick_patterns": [
                  {"name": "hammer", "bar_date": "2026-04-25",
                   "context": "near_support", "level_price": 1200.0,
                   "distance_pct": 0.4},
                  ...
                ]
              },
              "rationale": [...],
              "warnings": [],   # may include "obv_divergence", "insufficient_history"
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
    end = date.today()
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

    # ── Run momentum primitives ────────────────────────────────
    mom_params = MOMENTUM_PRESETS[sensitivity]
    snapshot = momentum_snapshot(
        df,
        rsi_length=mom_params["rsi"],
        macd_params=mom_params["macd"],
        stoch_params=mom_params["stoch"],
    )

    # ── Classify ───────────────────────────────────────────────
    signal, strength, rationale, warnings = classify_momentum(snapshot)

    # ── Candlestick context-gating ─────────────────────────────
    # We need swing levels (from levels cluster) and ATR (from volatility
    # cluster) to know what "near a level" means. Use the SAME sensitivity
    # preset across clusters so the user's preset choice is consistent.
    levels_params = LEVELS_PRESETS[sensitivity]
    vol_params = VOLATILITY_PRESETS[sensitivity]

    swing = swing_high_low(df, lookback=levels_params["swing_lookback"])
    atr = latest_atr(df, length=vol_params["atr"])

    raw_patterns = detect_recent_patterns(df, lookback=PATTERN_LOOKBACK_BARS)
    gated_patterns = gate_patterns(
        raw_patterns,
        df,
        swing_high=swing.get("swing_high"),
        swing_low=swing.get("swing_low"),
        atr=atr,
    )

    # Add gated patterns to rationale so the LLM mentions them
    for p in gated_patterns:
        rationale.append(
            f"Candlestick: {p['name']} on {p['bar_date']} "
            f"({p['context']}, {p['distance_pct']:.2f}% from level)"
        )

    # ── Flatten indicators for the LLM ─────────────────────────
    indicators = {
        "rsi": snapshot["rsi"],
        "macd_line": snapshot["macd"]["macd"],
        "macd_signal": snapshot["macd"]["signal"],
        "macd_histogram": snapshot["macd"]["histogram"],
        "macd_cross": snapshot["macd"]["cross"],
        "stoch_k": snapshot["stoch"]["k"],
        "stoch_d": snapshot["stoch"]["d"],
        "obv": snapshot["obv"]["obv"],
        "obv_slope_20": snapshot["obv"]["slope_20"],
    }

    derived = {
        "candlestick_patterns": gated_patterns,
        "patterns_detected_total": len(raw_patterns),
        "patterns_after_gating": len(gated_patterns),
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
