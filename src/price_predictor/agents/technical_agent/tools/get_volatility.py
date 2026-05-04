"""ADK tool: get_volatility(ticker, sensitivity) -> dict.

Third cluster tool. The 'boring but critical' one -- ATR drives stop-loss
sizing in Step D. Get the math right.

NO CROSS-CLUSTER IMPORTS
========================
Unlike get_momentum (which gates patterns by levels+ATR), get_volatility
is self-contained within the volatility cluster. Just ATR, BB, squeeze.

DERIVED HELPERS
===============
We surface position-sizing math so the LLM (and Step D) doesn't have
to re-derive it:
  - suggested_stop_distance: 2 * ATR (the '2 ATR rule')
  - per_share_risk: same as above (renamed for clarity)
  - volatility_regime: 'low' | 'normal' | 'high'
"""
from __future__ import annotations

from datetime import date, timedelta

from price_predictor.agents.technical_agent.tools._types import (
    ToolErrorResponse,
    ToolSuccessResponse,
)
from price_predictor.agents.technical_agent.tools._volatility_signal import (
    classify_volatility,
    classify_volatility_regime,
)
from price_predictor.agents.technical_agent.tools.get_trend import (
    _normalize_ticker,
)
from price_predictor.analysis import VOLATILITY_PRESETS, validate_preset
from price_predictor.analysis.volatility import volatility_snapshot
from price_predictor.data._shared_cache import get_cache
from price_predictor.data.prices import PriceFetchError
from price_predictor.kb.stocks import lookup as resolve_stock

LOOKBACK_DAYS = 400


async def get_volatility(ticker: str, sensitivity: str = "standard") -> dict:
    """Volatility-cluster analysis for a ticker.

    Args:
        ticker: Stock symbol. Indian stocks resolve to .NS automatically.
        sensitivity: 'standard' | 'sensitive' | 'smooth'.
                     - standard:  ATR-14, BB(20, 2.0)
                     - sensitive: ATR-9,  BB(10, 2.0)   (faster reaction)
                     - smooth:    ATR-21, BB(30, 2.0)   (slower reaction)

    Returns:
        On success:
            {
              "status": "success",
              "ticker": "RELIANCE.NS",
              "as_of": "2026-04-28",
              "preset": "standard",
              "signal": "bullish" | "neutral" | "bearish",  -- from BB %B position
              "strength": "weak" | "moderate" | "strong",   -- squeeze > regime
              "indicators": {atr, atr_pct_of_price, bb_lower, bb_middle,
                             bb_upper, bb_bandwidth, bb_percent_b, squeeze},
              "derived": {
                "volatility_regime": "low" | "normal" | "high" | "unknown",
                "suggested_stop_distance": 2 * ATR (price units),
                "per_share_risk": same as above (alias for clarity),
                "atr_multiple_to_upper_band": float | None,
                "atr_multiple_to_lower_band": float | None,
              },
              "rationale": [...],
              "warnings": [],   # may include "price_above_upper_band",
                                # "price_below_lower_band", "high_volatility",
                                # "insufficient_history"
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

    # ── Run volatility primitives ──────────────────────────────
    params = VOLATILITY_PRESETS[sensitivity]
    snapshot = volatility_snapshot(
        df,
        atr_length=params["atr"],
        bb_params=params["bb"],
    )

    # ── Classify ───────────────────────────────────────────────
    signal, strength, rationale, warnings = classify_volatility(snapshot)

    # ── Flatten indicators for the LLM ─────────────────────────
    bb = snapshot["bbands"]
    indicators = {
        "atr": snapshot["atr"],
        "atr_pct_of_price": snapshot["atr_pct_of_price"],
        "bb_lower": bb.get("lower"),
        "bb_middle": bb.get("middle"),
        "bb_upper": bb.get("upper"),
        "bb_bandwidth": bb.get("bandwidth"),
        "bb_percent_b": bb.get("percent_b"),
        "squeeze": snapshot["squeeze"],
    }

    # ── Derived helpers (the position-sizing math) ─────────────
    atr = snapshot["atr"]
    suggested_stop = round(2 * atr, 2) if atr is not None else None

    # How many ATRs is price away from upper/lower band?
    # Useful for "how stretched is this move?" reasoning.
    close = float(df["close"].iloc[-1]) if not df.empty else None
    atr_to_upper = None
    atr_to_lower = None
    if atr is not None and atr > 0 and close is not None:
        if bb.get("upper") is not None:
            atr_to_upper = round((bb["upper"] - close) / atr, 2)
        if bb.get("lower") is not None:
            atr_to_lower = round((close - bb["lower"]) / atr, 2)

    derived = {
        "volatility_regime": classify_volatility_regime(snapshot["atr_pct_of_price"]),
        "suggested_stop_distance": suggested_stop,
        "per_share_risk": suggested_stop,  # alias for downstream Step D clarity
        "atr_multiple_to_upper_band": atr_to_upper,
        "atr_multiple_to_lower_band": atr_to_lower,
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
