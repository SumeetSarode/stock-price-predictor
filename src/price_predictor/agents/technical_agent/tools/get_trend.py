"""ADK tool: get_trend(ticker, sensitivity) -> dict.

The first of four cluster tools. Validates the entire tool pattern that
get_momentum / get_volatility / get_levels will follow.

CALL FLOW
=========
    1. Normalize ticker (strip whitespace, uppercase, ensure .NS for Indian)
    2. Validate sensitivity preset
    3. Fetch ~1y of OHLCV via the shared cache (1 network hit per process
       per ticker per session)
    4. Call analysis.trend.trend_snapshot() with preset params
    5. Pass the snapshot to classify_trend() -> (signal, strength, rationale)
    6. Build the uniform tool response dict and return

ERRORS
======
Returns {"status": "error", ...} for ANY failure path. Never raises -- the
LLM needs to see the error to recover gracefully (apologize / suggest /
move on).
"""
from __future__ import annotations

from datetime import date, timedelta

from price_predictor.agents.technical_agent.tools._trend_signal import classify_trend
from price_predictor.agents.technical_agent.tools._types import (
    ToolErrorResponse,
    ToolSuccessResponse,
)
from price_predictor.analysis import TREND_PRESETS, validate_preset
from price_predictor.analysis.trend import trend_snapshot
from price_predictor.data._shared_cache import get_cache
from price_predictor.data.prices import PriceFetchError
from price_predictor.kb.stocks import lookup as resolve_stock

# How much history to fetch. Trend cluster needs at least 200+ bars for
# SMA-200; we ask for 400 days to give all indicators headroom.
LOOKBACK_DAYS = 750  # H7: ≥500 trading days for Wilder warmup


def _normalize_ticker(ticker: str) -> str:
    """Trim, uppercase, and add .NS suffix if it looks like an Indian symbol.

    Resolution rules:
      - Empty/None -> ValueError (caller catches)
      - Already has .NS or .BO -> uppercase as-is
      - Looks like an Indian name we know -> append .NS
      - Otherwise pass through (could be a US ticker like AAPL)
    """
    if not ticker or not ticker.strip():
        raise ValueError("ticker must be non-empty")
    t = ticker.strip().upper()
    if t.endswith(".NS") or t.endswith(".BO"):
        return t
    # If our KB knows it, prefer the canonical yfinance symbol
    resolved = resolve_stock(t)
    if resolved is not None:
        return resolved.yfinance_symbol
    return t


async def get_trend(
    ticker: str,
    sensitivity: str = "standard",
    *,
    as_of: date | None = None,
) -> dict:
    """Trend-cluster analysis for a ticker.

    Args:
        ticker: Stock symbol. Indian stocks resolve to .NS automatically;
                US/foreign stocks pass through (e.g., 'AAPL').
        sensitivity: One of 'standard' | 'sensitive' | 'smooth'. Picks
                     the parameter preset:
                       - standard:  SMA(20,50,200), EMA-20, ADX-14
                       - sensitive: SMA(10,30,100), EMA-10, ADX-9 (faster)
                       - smooth:    SMA(30,70,200), EMA-30, ADX-21 (slower)
        as_of:       Keyword-only. The trading date the analysis should
                     be anchored to. ``None`` (default) means "today" —
                     the live behavior. A past ``date`` is the backtest
                     mode: history is fetched up to and including this
                     date, no future data leaks in. Future dates are
                     rejected by the caller (predict()); this tool
                     does not re-validate (defence-in-depth lives one
                     layer up to keep the tool surface narrow).

    Returns:
        On success:
            {
              "status": "success",
              "ticker": "RELIANCE.NS",
              "as_of": "2026-04-28",
              "preset": "standard",
              "signal": "bullish" | "neutral" | "bearish",
              "strength": "weak" | "moderate" | "strong",
              "indicators": {close, sma_20, sma_50, sma_200, ema_20,
                             adx_14, di_plus, di_minus},
              "derived": {above_sma_20, above_sma_50, above_sma_200,
                          pct_above_sma_50, pct_above_sma_200},
              "rationale": ["Close above SMA-20, SMA-50", ...],
              "warnings": [],   # e.g. ["insufficient_history"]
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
    # `end` defaults to today (live behavior) but can be pinned to a
    # past trading date via `as_of` for honest backtest replay.
    end = as_of if as_of is not None else date.today()
    start = end - timedelta(days=LOOKBACK_DAYS)
    try:
        df = await get_cache().get(
            ticker=normalized, start=start, end=end, interval="1d"
        )
    except PriceFetchError as e:
        # Try fuzzy resolution -- maybe the user typed a ticker we know
        # but yfinance doesn't have (e.g. HDFC -> HDFCBANK post-merger).
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
    preset_params = TREND_PRESETS[sensitivity]
    snapshot = trend_snapshot(
        df,
        sma_lengths=preset_params["sma"],
        ema_length=preset_params["ema"],
        adx_length=preset_params["adx"],
    )

    # ── Classify ───────────────────────────────────────────────
    signal, strength, rationale = classify_trend(snapshot)

    # ── Flatten indicators for the LLM ─────────────────────────
    indicators = {
        "close": snapshot["close"],
        "ema": snapshot["ema"],
    }
    for n, v in snapshot["sma"].items():
        indicators[f"sma_{n}"] = v
    indicators["adx"] = snapshot["adx"]["adx"]
    indicators["di_plus"] = snapshot["adx"]["di_plus"]
    indicators["di_minus"] = snapshot["adx"]["di_minus"]

    derived: dict = {}
    for n, v in snapshot["above_sma"].items():
        derived[f"above_sma_{n}"] = v
    for n, v in snapshot["pct_above_sma"].items():
        derived[f"pct_above_sma_{n}"] = v
    # MA crossovers: surface as a top-level dict in `derived` so the LLM
    # quotes the L3 struct verbatim instead of inferring a cross from
    # static SMA position. See pred_logic.md §3.2 MA Crossover.
    derived["ma_crosses"] = snapshot.get("ma_crosses", {})

    # ── Warnings ───────────────────────────────────────────────
    warnings: list[str] = []
    if any(v is None for v in snapshot["sma"].values()):
        warnings.append("insufficient_history")
    # Use the LATEST bar's date as 'as_of' so the LLM doesn't claim it has
    # data it doesn't have (e.g. weekends, market holidays).
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
