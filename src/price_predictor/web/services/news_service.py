"""Recent news service — read-only adapter over `data/news.fetch_news`.

Purpose
=======
The stock detail page exposes a "📰 News" tab showing the last N days
of GDELT headlines for the company. This service queries GDELT live
(or via the news_snapshot replay context if one's been pinned for
backtesting — same behavior as the news_impact agent).

Boundary
========
- READ-ONLY consumer of `data/news` and `web/services/search_service`.
- Never mutates anything; never imports prediction/agents code.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from loguru import logger

from price_predictor.data.news import NewsFetchError, fetch_news
from price_predictor.web.services.search_service import get_by_ticker

_DEFAULT_DAYS = 7
_MAX_HEADLINES = 25  # cap render volume; GDELT can return up to 250
# GDELT is flaky under load -- it often answers a rate-limited request
# with a non-JSON HTML page (surfaced as NewsFetchError). A couple of
# short retries with backoff turns most of those transient failures into
# successful loads. Backoff list length == number of RETRIES after the
# first attempt, so 2 entries = up to 3 total attempts.
_RETRY_BACKOFF_S = (0.6, 1.4)


@dataclass(frozen=True, slots=True)
class Headline:
    """One news article — render-ready primitives only."""

    title: str
    url: str
    source: str
    published_at: datetime   # tz-aware UTC
    age_label: str           # "3h ago", "2d ago", etc.


@dataclass(frozen=True, slots=True)
class NewsBundle:
    """Everything the news tab needs to render."""

    ticker: str
    query: str               # actual query string sent to GDELT
    days: int
    headlines: list[Headline]
    fetched_at: datetime     # tz-naive local for display
    error: str | None        # human-friendly fetch error, or None


class NewsServiceError(Exception):
    """Raised when GDELT lookup hard-fails and we have nothing to show."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


def _normalize(ticker: str) -> str:
    t = ticker.strip().upper()
    if not t.endswith(".NS"):
        t = f"{t}.NS"
    return t


def _resolve_query(ticker: str) -> str:
    """Prefer the company NAME (better GDELT recall) over the ticker.

    Falls back to the ticker stem if the search index doesn't know
    this stock — better than nothing for the long tail.
    """
    stock = get_by_ticker(ticker)
    if stock and stock.name:
        return stock.name
    return ticker.removesuffix(".NS")


def _age_label(published: datetime, now: datetime) -> str:
    """Compact relative time: '3h ago', '2d ago', '5m ago', 'just now'."""
    delta = now - published
    secs = int(delta.total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


async def _fetch_with_retry(query: str, start: str, end: str) -> Any:
    """Call GDELT, retrying transient failures with backoff.

    Returns the articles DataFrame on success. Re-raises the LAST
    NewsFetchError only after every attempt is exhausted, so the caller
    still gets a clean soft-error to show.
    """
    attempts = len(_RETRY_BACKOFF_S) + 1
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return await fetch_news(query, start, end, max_records=_MAX_HEADLINES)
        except (NewsFetchError, ValueError) as exc:
            last_exc = exc
            if i < len(_RETRY_BACKOFF_S):
                backoff = _RETRY_BACKOFF_S[i]
                logger.warning(
                    "news: GDELT attempt {}/{} failed for {!r} ({}); "
                    "retrying in {}s",
                    i + 1, attempts, query, type(exc).__name__, backoff,
                )
                await asyncio.sleep(backoff)
    assert last_exc is not None  # unreachable: loop always sets it on failure
    raise last_exc


async def fetch_recent_headlines(
    ticker: str,
    *,
    days: int = _DEFAULT_DAYS,
) -> NewsBundle:
    """Pull the last `days` of GDELT headlines for `ticker`.

    Returns an empty-headlines bundle (NOT an error) when the lookup
    succeeded but found nothing — that's a valid market outcome.
    Returns a bundle with `error` populated when GDELT itself failed,
    so the template can show a soft "couldn't load news right now"
    instead of crashing the page.
    """
    t = _normalize(ticker)
    days = max(1, min(days, 30))  # clamp to a sane window
    query = _resolve_query(t)

    today = date.today()
    start = (today - timedelta(days=days)).isoformat()
    end = today.isoformat()

    fetched_at = datetime.now()
    error: str | None = None
    headlines: list[Headline] = []

    try:
        df = await _fetch_with_retry(query, start, end)
    except (NewsFetchError, ValueError) as exc:
        error = f"Couldn't load news right now ({type(exc).__name__})."
        df = None

    if df is not None and not df.empty:
        # Newest first — GDELT sometimes orders oldest first.
        df = df.sort_values("published_at", ascending=False).head(_MAX_HEADLINES)
        now_utc = datetime.now(timezone.utc)
        for row in df.itertuples(index=False):
            published: Any = row.published_at
            # pandas may hand back numpy/Timestamp; coerce to datetime.
            if hasattr(published, "to_pydatetime"):
                published = published.to_pydatetime()
            headlines.append(
                Headline(
                    title=str(row.title),
                    url=str(row.url),
                    source=str(row.source or "unknown"),
                    published_at=published,
                    age_label=_age_label(published, now_utc),
                )
            )

    return NewsBundle(
        ticker=t,
        query=query,
        days=days,
        headlines=headlines,
        fetched_at=fetched_at,
        error=error,
    )
