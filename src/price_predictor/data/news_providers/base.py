"""News provider abstraction + coverage metadata.

WHY THIS EXISTS
===============
News discovery used to be GDELT-only (`data/news.py`). GDELT is keyless and
IP-rate-limited (~1 req/5s), so bursts return HTTP 429 and a run silently
loses its news. This package adds a small provider abstraction so a live
fallback (Google News RSS) can pick up when GDELT throttles -- mirroring the
`data/providers/` pattern already used for price data.

CRITICAL: LOOK-AHEAD SAFETY
===========================
GDELT is the ONLY historical source (coverage back to 2017). RSS feeds return
RECENT news only. The prediction pipeline serves BOTH live predictions and
backtests through the same code path (`NewsSnapshotStore.get_or_fetch`),
distinguished only by `as_of`. A live-only provider MUST therefore declare a
freshness horizon via `coverage`, and the resilient layer MUST refuse to use
it for windows older than that horizon -- otherwise a backtest for 2019 could
be contaminated with today's articles (a look-ahead bug). GDELT remains the
sole provider for anything historical; RSS is strictly a live fallback.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

# The output contract every provider must satisfy -- identical to
# data/news.py's fetch_news() so callers never need to know the source.
NEWS_DF_COLUMNS = ["title", "url", "published_at", "source", "language"]


@dataclass(frozen=True)
class NewsProviderCoverage:
    """What time range a provider can honestly answer for.

    Attributes:
        historical: True if the provider can serve arbitrary past date
            ranges (GDELT). False if it only returns recent news (RSS).
        freshness_days: For live-only providers, how many days back from
            *now* the provider's results are still meaningful. Ignored when
            ``historical`` is True. A request whose ``end`` date is older
            than this many days must NOT be served by this provider.
    """

    historical: bool
    freshness_days: int = 0

    def can_serve_end(self, end: datetime, *, now: datetime | None = None) -> bool:
        """Can this provider honestly answer for a window ending at ``end``?

        Historical providers: always yes. Live-only providers: only if
        ``end`` is within ``freshness_days`` of now -- otherwise using it
        would fabricate 'current' news for a past date (look-ahead).
        """
        if self.historical:
            return True
        now = now or datetime.now(UTC)
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)
        age_days = (now - end).total_seconds() / 86400.0
        return age_days <= self.freshness_days


class NewsFetchError(RuntimeError):
    """Raised when a provider fails to return usable data (HTTP/parse/timeout).

    Distinct from ValueError (caller's bad input), which providers raise for
    empty queries / bad dates and which must NOT trigger fallback.
    """


class NewsProvider(ABC):
    """Contract every concrete news provider implements.

    REQUIRED OUTPUT SHAPE
    =====================
    fetch() returns a DataFrame with columns NEWS_DF_COLUMNS:
        title, url, published_at (tz-aware UTC), source, language.
    An empty DataFrame (0 rows) is a VALID result meaning "found nothing" --
    NOT an error.

    REQUIRED ERROR BEHAVIOR
    =======================
    - Empty/whitespace query or bad dates -> ValueError (caller's fault;
      no fallback).
    - Network / HTTP / parse failure      -> NewsFetchError (upstream's
      fault; triggers fallback).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short id for logging, e.g. 'gdelt', 'google_news_rss'."""

    @property
    @abstractmethod
    def coverage(self) -> NewsProviderCoverage:
        """Time-range this provider can honestly answer for."""

    @abstractmethod
    async def fetch(
        self,
        query: str,
        start: str,
        end: str,
        *,
        lang: str = "eng",
        max_records: int = 250,
        exact_phrase: bool = True,
    ) -> pd.DataFrame:
        """Fetch article metadata. See class docstring for the contract."""


def empty_news_df() -> pd.DataFrame:
    """Canonical empty result with the right columns + dtypes."""
    df = pd.DataFrame(columns=NEWS_DF_COLUMNS)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df
