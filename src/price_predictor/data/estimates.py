"""Analyst estimates fetcher backed by yfinance.

Iteration 3.1.2.5 — partial fix for the "no consensus surprise context"
caveat from iteration 3.1.3 design discussion.

WHAT THIS GIVES US
==================
Forward-looking analyst consensus per Indian stock:
- Earnings (EPS) estimates per upcoming quarter
- Revenue estimates per upcoming quarter
- Buy/hold/sell distribution (current + 3 months back, for drift detection)
- Price target consensus (low/mean/median/high)

The price-impact analyzer can then compute "surprise vs consensus" on results
day, which is the strongest single earnings-day price-move signal.

DESIGN
======
- Async-first via `asyncio.to_thread` wrap (yfinance is sync internally)
- One ticker per call (yfinance fetches per-ticker; no batch endpoint)
- Use fetch_estimates_batch for parallelism across stocks
- Coverage gracefully absent (yfinance returns None for non-covered stocks)
  → `Estimates.has_coverage` lets caller decide what to do
- yfinance API failure → raises EstimatesFetchError (your "errors first" rule)

KNOWN LIMIT
===========
Indian-stock coverage on yfinance is empirical and patchy. Run the spike at
`scripts/coverage_spike_estimates.py` from off-corp network to verify what
fraction of NSE stocks have data.
"""
from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import yfinance as yf
from loguru import logger

from price_predictor.data.schema import (
    Estimates,
    PriceTargets,
    QuarterlyEstimate,
    RecommendationDistribution,
)


class EstimatesFetchError(RuntimeError):
    """Raised when yfinance fails to fetch estimates (network/API/parsing)."""


# ─────────────────────────────────────────────────────────────
# Internal helpers — DataFrame → Pydantic
# ─────────────────────────────────────────────────────────────
def _safe_float(v: Any) -> float | None:
    """Convert to float, returning None for NaN / None / non-numeric."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> int | None:
    """Convert to int, returning None for NaN / None / non-numeric."""
    f = _safe_float(v)
    return int(f) if f is not None else None


def _parse_quarterly_df(df: pd.DataFrame | None) -> list[QuarterlyEstimate]:
    """Parse yfinance's earnings_estimate / revenue_estimate DataFrame.

    Expected shape: index = period labels ('0q', '+1q', ...), columns include
    'numberOfAnalysts', 'avg', 'low', 'high', 'yearAgoEps' (or 'yearAgoRevenue'),
    'growth'. yfinance is inconsistent about which columns exist, so we use
    .get() everywhere and tolerate missing columns.
    """
    if df is None or df.empty:
        return []

    # yfinance uses different "year ago" column names for EPS vs revenue
    year_ago_col = next(
        (c for c in ("yearAgoEps", "yearAgoRevenue") if c in df.columns),
        None,
    )

    out: list[QuarterlyEstimate] = []
    for period, row in df.iterrows():
        out.append(
            QuarterlyEstimate(
                period=str(period),
                num_analysts=_safe_int(row.get("numberOfAnalysts")),
                avg=_safe_float(row.get("avg")),
                low=_safe_float(row.get("low")),
                high=_safe_float(row.get("high")),
                year_ago=_safe_float(row.get(year_ago_col)) if year_ago_col else None,
                growth=_safe_float(row.get("growth")),
            )
        )
    return out


def _parse_recommendations_df(df: pd.DataFrame | None) -> list[RecommendationDistribution]:
    """Parse yfinance's recommendations DataFrame.

    Expected columns: period, strongBuy, buy, hold, sell, strongSell.
    Missing columns default to 0 (RecommendationDistribution defaults).
    """
    if df is None or df.empty:
        return []

    out: list[RecommendationDistribution] = []
    for _, row in df.iterrows():
        out.append(
            RecommendationDistribution(
                period=str(row.get("period", "unknown")),
                strong_buy=_safe_int(row.get("strongBuy")) or 0,
                buy=_safe_int(row.get("buy")) or 0,
                hold=_safe_int(row.get("hold")) or 0,
                sell=_safe_int(row.get("sell")) or 0,
                strong_sell=_safe_int(row.get("strongSell")) or 0,
            )
        )
    return out


def _parse_price_targets(targets: dict[str, Any] | None) -> PriceTargets | None:
    """Parse yfinance's analyst_price_targets dict.

    Returns None if the dict is empty/None or has no usable price fields.
    yfinance returns: {'current': ..., 'low': ..., 'mean': ..., 'median': ..., 'high': ...}
    """
    if not targets:
        return None

    pt = PriceTargets(
        current=_safe_float(targets.get("current")),
        low=_safe_float(targets.get("low")),
        mean=_safe_float(targets.get("mean")),
        median=_safe_float(targets.get("median")),
        high=_safe_float(targets.get("high")),
    )
    # If every field is None, treat as no coverage (return None instead of empty obj)
    if all(getattr(pt, f) is None for f in ("current", "low", "mean", "median", "high")):
        return None
    return pt


# ─────────────────────────────────────────────────────────────
# Sync core (called via asyncio.to_thread)
# ─────────────────────────────────────────────────────────────
def _fetch_estimates_sync(symbol: str) -> Estimates:
    """Sync core: hits yfinance and assembles Estimates.

    Wrapped by the public async function via asyncio.to_thread.
    Raises EstimatesFetchError on any yfinance/network failure.
    """
    try:
        ticker = yf.Ticker(symbol)
        # Each property call may trigger an HTTP request to Yahoo's quoteSummary endpoint.
        # We capture them all then assemble. yfinance returns None/empty for missing data
        # rather than raising — exceptions here mean genuine network/API failure.
        earnings = ticker.earnings_estimate
        revenue = ticker.revenue_estimate
        recs = ticker.recommendations
        targets = ticker.analyst_price_targets
    except Exception as e:
        raise EstimatesFetchError(
            f"yfinance fetch failed for {symbol!r}: {type(e).__name__}: {e}"
        ) from e

    return Estimates(
        symbol=symbol,
        fetched_at=datetime.now(UTC),
        earnings_estimates=_parse_quarterly_df(earnings),
        revenue_estimates=_parse_quarterly_df(revenue),
        recommendations=_parse_recommendations_df(recs),
        price_targets=_parse_price_targets(targets),
    )


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
async def fetch_estimates(symbol: str) -> Estimates:
    """Fetch analyst estimates for one ticker.

    Args:
        symbol: NSE ticker WITH suffix (e.g. 'RELIANCE.NS', 'TCS.NS').

    Returns:
        Estimates with whatever yfinance returned. If yfinance has no analyst
        coverage for this stock, fields will be empty/None and `has_coverage`
        will be False. NOT an error — it's a valid outcome.

    Raises:
        ValueError: On invalid symbol input (no network call made).
        EstimatesFetchError: On yfinance / network / parsing failure.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError(f"symbol must be a non-empty string, got {symbol!r}")

    return await asyncio.to_thread(_fetch_estimates_sync, symbol)


async def fetch_estimates_batch(
    symbols: list[str],
    *,
    concurrency: int = 5,
) -> dict[str, Estimates | Exception]:
    """Fetch estimates for many tickers in parallel.

    Args:
        symbols: List of NSE tickers.
        concurrency: Max in-flight yfinance calls. Default 5.

    Returns:
        Dict mapping each symbol → Estimates on success OR Exception on failure.
        One bad symbol never breaks the batch.
    """
    if not symbols:
        return {}

    sem = asyncio.Semaphore(concurrency)

    async def _one(sym: str) -> Estimates:
        async with sem:
            return await fetch_estimates(sym)

    results = await asyncio.gather(
        *(_one(s) for s in symbols),
        return_exceptions=True,
    )

    out: dict[str, Estimates | Exception] = {}
    for sym, r in zip(symbols, results, strict=True):
        if isinstance(r, BaseException) and not isinstance(r, Exception):
            # Re-raise truly-fatal exceptions (KeyboardInterrupt, SystemExit)
            raise r
        out[sym] = r  # type: ignore[assignment]
    return out


# ─────────────────────────────────────────────────────────────
# Coverage utility (used by spike script + future analyzer)
# ─────────────────────────────────────────────────────────────
def coverage_summary(est: Estimates) -> dict[str, Any]:
    """Quick stats about how much data yfinance returned for one ticker.

    Useful for the off-corp coverage spike and for runtime "is this stock
    worth running fundamentals analysis on?" checks.
    """
    return {
        "symbol": est.symbol,
        "has_coverage": est.has_coverage,
        "earnings_quarters": len(est.earnings_estimates),
        "revenue_quarters": len(est.revenue_estimates),
        "recommendation_snapshots": len(est.recommendations),
        "has_price_targets": est.price_targets is not None,
        "num_analysts_current_quarter": (
            est.earnings_estimates[0].num_analysts if est.earnings_estimates else None
        ),
    }


# Suppress unused-import warning — logger is for future tracing additions
_ = logger
