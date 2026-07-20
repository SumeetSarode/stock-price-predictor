"""Chart data service — fetches historical closes for the price chart
on the stock detail page.

Keeps payloads tiny: just (date, close) pairs over the chosen window.
No OHLCV, no volume — those add nothing to a single-line price chart
and bloat the JSON unnecessarily.

The render layer (Chart.js, frontend/scripts/chart.js) overlays the
cached prediction's entry/target/stop levels as horizontal dashed
lines when present. That's added client-side using values already in
the prediction view dict — no extra server work.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, timedelta

from loguru import logger

from price_predictor.data.prices import PriceFetchError, fetch_ohlcv


# Default window for the chart. 90 calendar days ≈ ~62 trading days
# in India after weekends + holidays. Long enough to show trends,
# short enough to keep payloads sub-2KB and rendering snappy.
DEFAULT_WINDOW_DAYS = 90


@dataclass(frozen=True, slots=True)
class ChartSeries:
    """Lightweight container for a single ticker's price history."""

    ticker: str
    dates: list[str]    # ISO 8601 (YYYY-MM-DD)
    closes: list[float]

    @property
    def is_empty(self) -> bool:
        return len(self.closes) == 0


def _normalize(ticker: str) -> str:
    t = ticker.strip().upper()
    if not t.endswith(".NS"):
        t = f"{t}.NS"
    return t


async def get_chart_series(
    ticker: str,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> ChartSeries:
    """Return (dates, closes) for `ticker` over the last `window_days`.

    Async wrapper around the (sync) jugaad/yfinance fetcher; the
    underlying fetch runs in a worker thread via asyncio.to_thread so
    we don't block the event loop.

    Returns an empty series on any fetch failure (with a logged warning).
    The chart UI handles empty series gracefully ('chart unavailable').
    """
    t = _normalize(ticker)
    end = date.today()
    start = end - timedelta(days=window_days)

    try:
        df = await asyncio.to_thread(fetch_ohlcv, t, start, end, "1d")
    except PriceFetchError as exc:
        logger.warning("chart: fetch failed ticker={} err={}", t, exc)
        return ChartSeries(ticker=t, dates=[], closes=[])
    except Exception:
        logger.opt(exception=True).warning("chart: unexpected fetch error ticker={}", t)
        return ChartSeries(ticker=t, dates=[], closes=[])

    if df is None or df.empty:
        return ChartSeries(ticker=t, dates=[], closes=[])

    # Drop rows missing a close — notably today's still-forming bar, which
    # some providers return with a NaN close. NaN is not JSON-serialisable,
    # so leaving it in crashes the /api/chart endpoint with a 500. Mirrors
    # the guard in dashboard_service and analysis_service.
    df = df.dropna(subset=["close"])
    if df.empty:
        return ChartSeries(ticker=t, dates=[], closes=[])

    # Ensure we have close + a date index/column. The DataFrame schema
    # is established by data.prices; we defensively coerce anyway.
    try:
        # Index is a DatetimeIndex for jugaad/yfinance output.
        idx = df.index
        dates = [d.date().isoformat() if hasattr(d, "date") else str(d) for d in idx]
        closes = [float(v) for v in df["close"].tolist()]
    except Exception:
        logger.opt(exception=True).warning("chart: dataframe shape unexpected ticker={}", t)
        return ChartSeries(ticker=t, dates=[], closes=[])

    return ChartSeries(ticker=t, dates=dates, closes=closes)
