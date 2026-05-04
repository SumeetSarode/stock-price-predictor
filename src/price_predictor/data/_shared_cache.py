"""Process-wide PriceCache singleton + async wrapper around fetch_ohlcv.

WHY THIS EXISTS
===============
The 4 cluster tools (get_trend, get_momentum, get_volatility, get_levels)
will be called in the same agent turn for the same ticker. Each tool needs
historical OHLCV. Without a shared cache, that's 4 yfinance hits per turn.

This module exposes ONE cache instance, lazily constructed on first use.
Tests can override it via `set_cache()` so they never hit the network.

INTENTIONAL TRADE-OFFS
======================
- Module-level state is usually a code smell, but the cache is conceptually
  one-per-process. Trying to thread it through ADK's tool signatures buys
  nothing and costs clarity.
- The cache lives the lifetime of the Python process. Restart `adk web`
  and you get a fresh cache. That's fine for v1.
"""
from __future__ import annotations

import asyncio
from datetime import date
from typing import Optional

import pandas as pd

from price_predictor.data.cache import PriceCache
from price_predictor.data.prices import fetch_ohlcv

# Holds the singleton; None until first access.
_cache: Optional[PriceCache] = None


async def _async_fetch(
    ticker: str, start: date, end: date, interval: str
) -> pd.DataFrame:
    """Async adapter around the sync fetch_ohlcv.

    PriceCache requires an async fetcher (so it can hold an asyncio.Lock
    while waiting). Our underlying fetcher is sync; asyncio.to_thread keeps
    the event loop responsive while it runs.
    """
    return await asyncio.to_thread(fetch_ohlcv, ticker, start, end)


def get_cache() -> PriceCache:
    """Return the process-wide cache, building it lazily."""
    global _cache
    if _cache is None:
        _cache = PriceCache(fetch_fn=_async_fetch)
    return _cache


def set_cache(cache: PriceCache | None) -> None:
    """Override the singleton. Used by tests with a mock cache.

    Pass None to reset back to lazy construction.
    """
    global _cache
    _cache = cache
