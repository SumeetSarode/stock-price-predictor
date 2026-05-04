"""Tests for PriceCache -- range-aware in-memory caching with async locks."""
from __future__ import annotations

import asyncio
from datetime import date

import pandas as pd
import pytest

from price_predictor.data.cache import PriceCache


def _make_bars(start: date, end: date) -> pd.DataFrame:
    """Build a tz-aware OHLCV DataFrame covering [start, end] inclusive."""
    dates = pd.date_range(start=start, end=end, freq="D", tz="Asia/Kolkata")
    n = len(dates)
    return pd.DataFrame(
        {
            "open":      [100.0 + i for i in range(n)],
            "high":      [101.0 + i for i in range(n)],
            "low":       [99.0 + i for i in range(n)],
            "close":     [100.5 + i for i in range(n)],
            "adj_close": [100.5 + i for i in range(n)],
            "volume":    [1000 + i for i in range(n)],
        },
        index=dates,
    )


class _RecordingFetcher:
    """Fake fetch_fn that records every call and returns synthetic bars."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, date, date, str]] = []

    async def __call__(self, ticker, start, end, interval):
        self.calls.append((ticker, start, end, interval))
        return _make_bars(start, end)


# ─────────────────────────────────────────────────────────────────
# Single-fetch caching
# ─────────────────────────────────────────────────────────────────
class TestSingleFetchCaches:
    @pytest.mark.asyncio
    async def test_first_call_fetches(self):
        fetcher = _RecordingFetcher()
        cache = PriceCache(fetch_fn=fetcher)
        df = await cache.get("X", date(2025, 1, 1), date(2025, 1, 10))
        assert not df.empty
        assert len(fetcher.calls) == 1

    @pytest.mark.asyncio
    async def test_second_identical_call_returns_cached(self):
        fetcher = _RecordingFetcher()
        cache = PriceCache(fetch_fn=fetcher)
        await cache.get("X", date(2025, 1, 1), date(2025, 1, 10))
        await cache.get("X", date(2025, 1, 1), date(2025, 1, 10))
        assert len(fetcher.calls) == 1  # second call hit cache

    @pytest.mark.asyncio
    async def test_invalid_range_rejected(self):
        cache = PriceCache(fetch_fn=_RecordingFetcher())
        with pytest.raises(ValueError, match="must be <="):
            await cache.get("X", date(2025, 1, 10), date(2025, 1, 1))


# ─────────────────────────────────────────────────────────────────
# Range slicing -- narrower request, no re-fetch
# ─────────────────────────────────────────────────────────────────
class TestRangeSlicing:
    @pytest.mark.asyncio
    async def test_narrower_range_slices_from_cache(self):
        fetcher = _RecordingFetcher()
        # Set proactive_days small so first fetch is bounded by the request
        cache = PriceCache(fetch_fn=fetcher, proactive_days=0)
        # First: fetch wide
        await cache.get("X", date(2025, 1, 1), date(2025, 1, 31))
        # Second: ask for narrower window inside the cached range
        df = await cache.get("X", date(2025, 1, 10), date(2025, 1, 20))
        assert len(fetcher.calls) == 1  # no re-fetch
        assert df.index.date.min() == date(2025, 1, 10)
        assert df.index.date.max() == date(2025, 1, 20)

    @pytest.mark.asyncio
    async def test_proactive_fetch_makes_subsequent_calls_slices(self):
        """First call with default proactive_days fetches wide; later
        narrower calls are slices."""
        fetcher = _RecordingFetcher()
        cache = PriceCache(fetch_fn=fetcher)  # default 365 proactive
        await cache.get("X", date(2025, 6, 1), date(2025, 6, 30))
        # The fetch should have started ~365 days before end
        first_call = fetcher.calls[0]
        assert (first_call[2] - first_call[1]).days >= 365
        # Now ask for a 5-day window inside that pre-fetched range -- no fetch
        await cache.get("X", date(2025, 5, 1), date(2025, 5, 5))
        assert len(fetcher.calls) == 1


# ─────────────────────────────────────────────────────────────────
# Range expansion -- wider/different request, fetches missing chunk
# ─────────────────────────────────────────────────────────────────
class TestRangeExpansion:
    @pytest.mark.asyncio
    async def test_expanding_range_re_fetches(self):
        fetcher = _RecordingFetcher()
        cache = PriceCache(fetch_fn=fetcher, proactive_days=0)
        # First: small range
        await cache.get("X", date(2025, 1, 10), date(2025, 1, 20))
        # Second: bigger range -- needs re-fetch
        df = await cache.get("X", date(2025, 1, 1), date(2025, 1, 31))
        assert len(fetcher.calls) == 2
        # The second fetch covered the union [2025-01-01..2025-01-31]
        second = fetcher.calls[1]
        assert second[1] == date(2025, 1, 1)
        assert second[2] == date(2025, 1, 31)
        # Returned df is the requested range
        assert df.index.date.min() == date(2025, 1, 1)
        assert df.index.date.max() == date(2025, 1, 31)

    @pytest.mark.asyncio
    async def test_request_extends_only_on_one_side(self):
        fetcher = _RecordingFetcher()
        cache = PriceCache(fetch_fn=fetcher, proactive_days=0)
        await cache.get("X", date(2025, 1, 10), date(2025, 1, 20))
        # Extend forward only
        await cache.get("X", date(2025, 1, 15), date(2025, 1, 25))
        assert len(fetcher.calls) == 2
        coverage = cache._coverage("X")
        assert coverage == (date(2025, 1, 10), date(2025, 1, 25))


# ─────────────────────────────────────────────────────────────────
# Concurrency -- locks per (ticker, interval)
# ─────────────────────────────────────────────────────────────────
class TestConcurrency:
    @pytest.mark.asyncio
    async def test_parallel_same_ticker_one_fetch(self):
        """Two concurrent calls for the same ticker share one fetch."""
        slow_fetcher_calls = 0

        async def slow_fetch(ticker, start, end, interval):
            nonlocal slow_fetcher_calls
            slow_fetcher_calls += 1
            await asyncio.sleep(0.05)  # simulate network
            return _make_bars(start, end)

        cache = PriceCache(fetch_fn=slow_fetch, proactive_days=0)
        # Fire 5 in parallel
        results = await asyncio.gather(
            *[cache.get("X", date(2025, 1, 1), date(2025, 1, 10)) for _ in range(5)]
        )
        assert len(results) == 5
        assert slow_fetcher_calls == 1  # lock serialized; first won, others cached

    @pytest.mark.asyncio
    async def test_parallel_different_tickers_fetch_independently(self):
        """Different tickers don't block each other."""
        events: list[str] = []

        async def slow_fetch(ticker, start, end, interval):
            events.append(f"start-{ticker}")
            await asyncio.sleep(0.05)
            events.append(f"end-{ticker}")
            return _make_bars(start, end)

        cache = PriceCache(fetch_fn=slow_fetch)
        await asyncio.gather(
            cache.get("A", date(2025, 1, 1), date(2025, 1, 10)),
            cache.get("B", date(2025, 1, 1), date(2025, 1, 10)),
        )
        # Both should have started before either finished -- proves parallelism
        assert events.index("start-A") < events.index("end-B")
        assert events.index("start-B") < events.index("end-A")


# ─────────────────────────────────────────────────────────────────
# Defensive copy -- mutating returned df doesn't affect cache
# ─────────────────────────────────────────────────────────────────
class TestImmutability:
    @pytest.mark.asyncio
    async def test_mutating_returned_df_does_not_affect_cache(self):
        fetcher = _RecordingFetcher()
        cache = PriceCache(fetch_fn=fetcher, proactive_days=0)
        df1 = await cache.get("X", date(2025, 1, 1), date(2025, 1, 10))
        df1.loc[df1.index[0], "close"] = -999.0  # vandalism
        df2 = await cache.get("X", date(2025, 1, 1), date(2025, 1, 10))
        assert df2.iloc[0]["close"] != -999.0  # cache wasn't poisoned


# ─────────────────────────────────────────────────────────────────
# Error propagation -- failures don't poison the cache
# ─────────────────────────────────────────────────────────────────
class TestErrorPropagation:
    @pytest.mark.asyncio
    async def test_fetch_error_propagates_and_no_entry_stored(self):
        async def failing_fetch(ticker, start, end, interval):
            raise RuntimeError("upstream broken")

        cache = PriceCache(fetch_fn=failing_fetch)
        with pytest.raises(RuntimeError, match="upstream broken"):
            await cache.get("X", date(2025, 1, 1), date(2025, 1, 10))
        # Cache should be empty -- no half-baked entry
        assert cache._coverage("X") is None
