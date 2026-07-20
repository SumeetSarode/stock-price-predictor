"""Nifty 50 dashboard service — fetches EOD prices in parallel,
caches per-day, exposes a render-ready list of rows.

Architecture:
  - Parallel fetch via asyncio.to_thread (jugaad-data is sync). Bounded
    concurrency (semaphore=10) to be polite to NSE infra.
  - In-memory cache, keyed by IST date. First call of the day takes
    ~5-10s; subsequent calls within the same trading day are instant.
  - Disk persistence + manual refresh button land in substep 2C.

Error handling: any per-ticker failure is captured into the row's
``error`` field rather than failing the whole batch. The template
renders a friendly em-dash for missing values. One bad ticker doesn't
break the dashboard.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
from loguru import logger

from price_predictor.data.prices import PriceFetchError, fetch_ohlcv
from price_predictor.web.services.search_service import Stock, all_nifty50
from price_predictor.web.services.watchlist_service import watchlist_tickers

# India Standard Time, UTC+5:30. We use IST for the cache key because
# the dashboard's freshness is gated by Indian trading days, not UTC.
_IST = timezone(timedelta(hours=5, minutes=30))

# Bound concurrent fetches. NSE doesn't publish a rate limit but
# hammering them with 50 simultaneous requests is rude.
_MAX_CONCURRENT_FETCHES = 10


# ── Public data model ───────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DashboardRow:
    """One row of the Nifty 50 dashboard table.

    Any field except ``ticker``, ``name``, ``sector`` may be ``None``
    if the per-ticker fetch failed. ``error`` carries the message in
    that case so the UI can show a friendly hint.

    ``is_watched`` is populated at snapshot-render time from the
    watchlist table; it's *not* part of the price-cache key.
    """

    ticker: str
    name: str
    sector: str
    close: float | None = None
    prev_close: float | None = None
    change_pct: float | None = None
    volume: int | None = None
    last_trading_day: date | None = None
    error: str | None = None
    is_watched: bool = False

    @property
    def direction(self) -> str:
        """Coarse direction for color-coding: bullish | bearish | neutral."""
        if self.change_pct is None:
            return "neutral"
        if self.change_pct > 0.05:
            return "bullish"
        if self.change_pct < -0.05:
            return "bearish"
        return "neutral"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "sector": self.sector,
            "close": self.close,
            "prev_close": self.prev_close,
            "change_pct": self.change_pct,
            "volume": self.volume,
            "direction": self.direction,
            "last_trading_day": self.last_trading_day.isoformat() if self.last_trading_day else None,
            "error": self.error,
            "is_watched": self.is_watched,
        }


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
    """A complete dashboard fetch result + the timestamp it was taken."""

    rows: tuple[DashboardRow, ...]
    fetched_at: datetime              # always tz-aware (IST)
    trading_day: date | None          # IST date the prices reflect

    @property
    def fetched_at_str(self) -> str:
        """e.g. '15 May 2026 18:45 IST'."""
        return self.fetched_at.strftime("%d %b %Y %H:%M IST")

    @property
    def is_market_open(self) -> bool:
        """True iff fetched_at falls within NSE market hours.

        NSE equity session: Mon-Fri 09:15-15:30 IST. We approximate
        (no holiday calendar) — the worst case is a 'LIVE' pill on a
        bank holiday, which is harmless.
        """
        ts = self.fetched_at  # already IST (tz-aware)
        if ts.weekday() >= 5:  # 5=Sat, 6=Sun
            return False
        # 09:15 ≤ hh:mm < 15:30
        minutes = ts.hour * 60 + ts.minute
        return 555 <= minutes < 930

    @property
    def fetched_age_label(self) -> str:
        """Relative friendly label — 'just now', '5 min ago', '2 hrs ago'.

        Drives the freshness microcopy next to the refresh button.
        """
        from datetime import datetime as _dt
        now = _dt.now(self.fetched_at.tzinfo)
        secs = (now - self.fetched_at).total_seconds()
        if secs < 60:    return "just now"
        if secs < 3600:  return f"{int(secs//60)} min ago"
        if secs < 86400: hrs = int(secs//3600); return f"{hrs} hr ago" if hrs == 1 else f"{hrs} hrs ago"
        return self.fetched_at.strftime("%d %b")


# ── In-memory cache ─────────────────────────────────────────────────
#
# Process-local. Lost on restart. Disk persistence comes in substep 2C.

_cache: DashboardSnapshot | None = None


def _today_ist() -> date:
    return datetime.now(_IST).date()


def _cache_is_fresh(snapshot: DashboardSnapshot | None) -> bool:
    """A snapshot is fresh iff it was fetched today (IST).

    Crude but correct for EOD data: prices don't change overnight,
    so as long as the cache was taken today, we trust it. Intraday
    refresh logic lands in substep 2C.
    """
    if snapshot is None:
        return False
    return snapshot.fetched_at.date() == _today_ist()


# ── Per-ticker fetch ────────────────────────────────────────────────


def _compute_row(stock: Stock, df: pd.DataFrame) -> DashboardRow:
    """Turn a 2-bar OHLCV DataFrame into a DashboardRow."""
    last = df.iloc[-1]
    if len(df) >= 2:
        prev = df.iloc[-2]
        prev_close = float(prev["close"])
        change_pct = ((float(last["close"]) - prev_close) / prev_close) * 100.0
    else:
        prev_close = None
        change_pct = None

    last_day = last.name  # DatetimeIndex
    last_day = last_day.date() if hasattr(last_day, "date") else None

    return DashboardRow(
        ticker=stock.ticker,
        name=stock.name,
        sector=stock.sector,
        close=float(last["close"]),
        prev_close=prev_close,
        change_pct=change_pct,
        volume=int(last["volume"]) if pd.notna(last["volume"]) else None,
        last_trading_day=last_day,
    )


def _fetch_one_sync(stock: Stock, lookback_days: int = 10) -> DashboardRow:
    """Synchronous per-ticker fetch. Wrapped in asyncio.to_thread."""
    end = _today_ist()
    start = end - timedelta(days=lookback_days)
    try:
        df = fetch_ohlcv(stock.ticker, start=start, end=end)
        if df is not None and len(df) > 0:
            # Drop rows missing a close — notably today's still-forming bar,
            # which some providers return with a NaN close. Without this the
            # "latest" bar is NaN, which blanks the price in the UI AND
            # crashes JSON endpoints (NaN is not JSON-serialisable). Mirrors
            # the guard already in analysis_service._fetch_bars.
            df = df.dropna(subset=["close"])
        if df is None or len(df) == 0:
            return DashboardRow(
                ticker=stock.ticker,
                name=stock.name,
                sector=stock.sector,
                error="No data returned",
            )
        return _compute_row(stock, df)
    except PriceFetchError as e:
        logger.warning("dashboard: fetch failed ticker={} err={}", stock.ticker, e)
        return DashboardRow(
            ticker=stock.ticker, name=stock.name, sector=stock.sector,
            error=str(e),
        )
    except Exception as e:  # defensive — anything we didn't anticipate
        logger.exception("dashboard: unexpected error ticker={}", stock.ticker)
        return DashboardRow(
            ticker=stock.ticker, name=stock.name, sector=stock.sector,
            error=f"Unexpected: {type(e).__name__}",
        )


# ── Public service API ──────────────────────────────────────────────


async def get_dashboard(*, force_refresh: bool = False) -> DashboardSnapshot:
    """Return the Nifty 50 dashboard snapshot.

    Cache hit → returns instantly.
    Cache miss → parallel-fetches all 50 tickers (~5-10s), caches result.

    Args:
        force_refresh: bypass the cache and refetch. Used by the
            manual-refresh button (substep 2C).
    """
    global _cache

    if not force_refresh and _cache_is_fresh(_cache):
        logger.debug("dashboard: cache hit (fetched_at={})", _cache.fetched_at)
        return _cache

    stocks = all_nifty50()
    logger.info("dashboard: fetching {} tickers in parallel", len(stocks))

    sem = asyncio.Semaphore(_MAX_CONCURRENT_FETCHES)

    async def _bounded_fetch(s: Stock) -> DashboardRow:
        async with sem:
            return await asyncio.to_thread(_fetch_one_sync, s)

    rows = await asyncio.gather(*[_bounded_fetch(s) for s in stocks])

    # Trading day = max last_trading_day across all successful rows.
    # Handles weekends / holidays correctly (prices might be from Friday
    # even if we're checking on Saturday).
    trading_days = [r.last_trading_day for r in rows if r.last_trading_day]
    trading_day = max(trading_days) if trading_days else None

    snapshot = DashboardSnapshot(
        rows=tuple(rows),
        fetched_at=datetime.now(_IST),
        trading_day=trading_day,
    )
    _cache = snapshot

    n_ok = sum(1 for r in rows if r.error is None)
    logger.info("dashboard: fetch done {}/{} ok trading_day={}",
                n_ok, len(rows), trading_day)
    return snapshot


def snapshot_with_watchlist(snapshot: DashboardSnapshot) -> DashboardSnapshot:
    """Return a copy of `snapshot` with `is_watched` populated per row.

    Kept separate from get_dashboard() because the price cache is keyed
    by date but watchlist state can change at any moment. Cheap (one
    SQL query + dataclass copy) so we just run it on every render.
    """
    watched = watchlist_tickers()
    if not watched:
        return snapshot  # short-circuit: no work to do

    new_rows = tuple(
        DashboardRow(
            ticker=r.ticker, name=r.name, sector=r.sector,
            close=r.close, prev_close=r.prev_close,
            change_pct=r.change_pct, volume=r.volume,
            last_trading_day=r.last_trading_day, error=r.error,
            is_watched=(r.ticker in watched),
        )
        for r in snapshot.rows
    )
    return DashboardSnapshot(
        rows=new_rows,
        fetched_at=snapshot.fetched_at,
        trading_day=snapshot.trading_day,
    )


def reset_cache_for_tests() -> None:
    """Drop the in-memory snapshot. Used by tests."""
    global _cache
    _cache = None
