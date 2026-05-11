"""Range-aware in-memory price cache.

WHY THIS EXISTS
===============
The 4 thematic technical tools (`get_trend`, `get_momentum`, `get_volatility`,
`get_levels`) will fan out for a single user question like "analyze HDFCBANK".
Without a cache, that's 4 yfinance hits for the same data. With this cache,
it's 1 hit per ticker per session.

DESIGN (locked in next_steps.md)
================================
- Range-aware: cache stores the widest date range fetched per ticker. A
  later request for a sub-range slices from cache without re-fetching. A
  later request for a wider range fetches the missing chunk and merges.
- Key: (ticker, interval) -- one entry per ticker per bar size.
- Storage: in-memory only. Process restart = fresh cache.
- Concurrency: one asyncio.Lock per (ticker, interval). Two parallel calls
  for the SAME ticker share one fetch; calls for DIFFERENT tickers fetch
  in parallel.
- Lifetime: process-scoped. "Today" doesn't change within a session.
- Eviction: none. 50 stocks x ~50KB = trivial memory.

COPY-ON-RETURN
==============
Slicing returns a defensive copy. Callers can mutate freely without
poisoning the cache for other callers.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
from loguru import logger

# Default proactive fetch window. We grab ~750 calendar days (= ~520 NSE
# trading bars, ~2 calendar years) so that even after the H7 Wilder-smoothing
# warmup discard (10 × ADX-length = 140 bars + 200 SMA cushion) we still
# have a healthy ~180+ usable bars for downstream indicators. Subsequent
# calls for narrower windows are slices (no extra network). Bumped from
# 365 in pred_logic_solutions §H7 ("Convergence guard").
DEFAULT_PROACTIVE_DAYS = 750


# Type alias for the underlying fetcher -- any callable that fetches OHLCV
# given (ticker, start, end, interval). Lets us inject mocks in tests
# without monkey-patching the resilient fetcher.
FetchFn = Callable[[str, date, date, str], Awaitable[pd.DataFrame]]


@dataclass
class _CacheEntry:
    """One ticker's cached bars + metadata about its coverage."""

    df: pd.DataFrame
    start: date  # earliest date we have bars for
    end: date    # latest date we have bars for


class PriceCache:
    """Range-aware in-memory cache of OHLCV bars.

    Usage:
        cache = PriceCache(fetch_fn=my_async_fetcher)
        df = await cache.get("RELIANCE.NS", date(2025, 1, 1), date(2025, 6, 30))

    The fetch_fn is injected so tests can supply a fake. Production callers
    pass an async wrapper around `data.prices.fetch_ohlcv`.
    """

    def __init__(
        self,
        fetch_fn: FetchFn,
        proactive_days: int = DEFAULT_PROACTIVE_DAYS,
    ) -> None:
        self._fetch = fetch_fn
        self._proactive_days = proactive_days
        self._store: dict[tuple[str, str], _CacheEntry] = {}
        # One lock per (ticker, interval). Created lazily on first access
        # so we don't pre-allocate locks for tickers we never query.
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

    # ── Public API ────────────────────────────────────────────────

    async def get(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return OHLCV bars for [start, end] inclusive.

        Hits cache when possible, fetches the missing portion otherwise.
        Returns a defensive copy -- safe to mutate.
        """
        if start > end:
            raise ValueError(f"start ({start}) must be <= end ({end})")

        key = (ticker, interval)
        lock = self._locks.setdefault(key, asyncio.Lock())

        async with lock:
            entry = self._store.get(key)
            if entry is None:
                # Cold cache -- proactively fetch a wider window so future
                # narrower asks are slices.
                fetch_start = min(start, end - timedelta(days=self._proactive_days))
                fetch_end = end
                logger.debug(
                    f"[price-cache] miss ticker={ticker} -- fetching "
                    f"[{fetch_start}..{fetch_end}]"
                )
                df = await self._fetch(ticker, fetch_start, fetch_end, interval)
                self._store[key] = _CacheEntry(df=df, start=fetch_start, end=fetch_end)
                return self._slice(self._store[key], start, end)

            # Warm cache -- check coverage
            if entry.start <= start and entry.end >= end:
                logger.debug(
                    f"[price-cache] hit ticker={ticker} "
                    f"[{start}..{end}] within [{entry.start}..{entry.end}]"
                )
                return self._slice(entry, start, end)

            # Partial coverage -- need to extend
            new_start = min(entry.start, start)
            new_end = max(entry.end, end)
            logger.debug(
                f"[price-cache] partial ticker={ticker} -- "
                f"have [{entry.start}..{entry.end}], need [{start}..{end}], "
                f"extending to [{new_start}..{new_end}]"
            )
            # Simple v1 strategy: re-fetch the whole new range. This is
            # wasteful when extending by a few days, but correct and trivial.
            # A delta-fetch optimization is parked for later.
            df = await self._fetch(ticker, new_start, new_end, interval)
            self._store[key] = _CacheEntry(df=df, start=new_start, end=new_end)
            return self._slice(self._store[key], start, end)

    # ── Internals ─────────────────────────────────────────────────

    def _slice(self, entry: _CacheEntry, start: date, end: date) -> pd.DataFrame:
        """Return a defensive copy of bars in [start, end] (inclusive).

        Filters by the DataFrame's tz-aware datetime index. We compare on
        date (not datetime) so a request for end=2025-04-28 includes bars
        timestamped 2025-04-28 09:15 (NSE open).
        """
        idx_dates = entry.df.index.date
        mask = (idx_dates >= start) & (idx_dates <= end)
        return entry.df.loc[mask].copy()

    # ── Test/debug helpers ────────────────────────────────────────

    def _coverage(self, ticker: str, interval: str = "1d") -> tuple[date, date] | None:
        """For tests: report what range we currently have cached."""
        entry = self._store.get((ticker, interval))
        return None if entry is None else (entry.start, entry.end)

    def _clear(self) -> None:
        """For tests: reset state."""
        self._store.clear()
        self._locks.clear()
