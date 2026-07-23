"""Google News RSS provider -- live-only news fallback for when GDELT 429s.

Google News exposes a per-query RSS search feed:

    https://news.google.com/rss/search?q=<query>&hl=en-IN&gl=IN&ceid=IN:en

Unlike GDELT it is NOT rate-limited in practice and has excellent Indian
coverage (gl=IN / ceid=IN:en localise it). What it CANNOT do is historical
date-range queries -- it only returns recent articles. That is why its
`coverage` is live-only with a freshness horizon; the resilient layer refuses
to use it for backtest windows (see news_providers/base.py for the look-ahead
rationale).

Parsing uses the stdlib (xml.etree + email.utils) -- deliberately NO new
dependency (no feedparser). The feed is small, well-formed RSS 2.0.
"""
from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

import httpx
import pandas as pd
from loguru import logger

from price_predictor.data.news_providers.base import (
    NewsFetchError,
    NewsProvider,
    NewsProviderCoverage,
    empty_news_df,
)

GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
DEFAULT_TIMEOUT_S = 15.0
DEFAULT_USER_AGENT = "price-predictor/0.1 (+https://github.com/sumeet-s2597)"
# How recent an article must be for this live-only source to be usable. A
# request whose window ends before (now - this) is a backtest and must NOT
# be served from a live feed.
DEFAULT_FRESHNESS_DAYS = 30


def _parse_pubdate(raw: str | None) -> datetime | None:
    """Parse an RFC-2822 pubDate into tz-aware UTC. None on failure."""
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:  # RFC 2822 allows missing tz -> assume UTC
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class GoogleNewsRssProvider(NewsProvider):
    """Per-query Google News RSS search. Live-only, India-localised."""

    def __init__(
        self,
        *,
        freshness_days: int = DEFAULT_FRESHNESS_DAYS,
        timeout: float = DEFAULT_TIMEOUT_S,
        hl: str = "en-IN",
        gl: str = "IN",
        ceid: str = "IN:en",
    ) -> None:
        self._freshness_days = freshness_days
        self._timeout = timeout
        self._hl = hl
        self._gl = gl
        self._ceid = ceid

    @property
    def name(self) -> str:
        return "google_news_rss"

    @property
    def coverage(self) -> NewsProviderCoverage:
        return NewsProviderCoverage(
            historical=False, freshness_days=self._freshness_days
        )

    async def fetch(
        self,
        query: str,
        start: str,
        end: str,
        *,
        lang: str = "eng",
        max_records: int = 250,
        exact_phrase: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> pd.DataFrame:
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"query must be a non-empty string, got {query!r}")
        try:
            start_dt = datetime.fromisoformat(start).replace(tzinfo=UTC)
            end_dt = datetime.fromisoformat(end).replace(
                hour=23, minute=59, second=59, tzinfo=UTC
            )
        except ValueError as e:
            raise ValueError(f"bad date input start={start!r} end={end!r}: {e}") from e

        q = f'"{query.strip()}"' if exact_phrase else query.strip()
        params = {"q": q, "hl": self._hl, "gl": self._gl, "ceid": self._ceid}
        headers = {"User-Agent": DEFAULT_USER_AGENT}

        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=self._timeout, headers=headers)
        try:
            resp = await client.get(GOOGLE_NEWS_RSS_URL, params=params)
            resp.raise_for_status()
            raw_xml = resp.text
        except httpx.HTTPError as e:
            raise NewsFetchError(
                f"Google News RSS request failed for query={query!r}: {e}"
            ) from e
        finally:
            if owns_client:
                await client.aclose()

        return self._parse(raw_xml, start_dt, end_dt, max_records, query=query)

    def _parse(
        self,
        raw_xml: str,
        start_dt: datetime,
        end_dt: datetime,
        max_records: int,
        *,
        query: str,
    ) -> pd.DataFrame:
        try:
            root = ET.fromstring(raw_xml)
        except ET.ParseError as e:
            raise NewsFetchError(
                f"Google News RSS returned unparseable XML for query={query!r}: {e}"
            ) from e

        rows: list[dict] = []
        for item in root.iterfind(".//item"):
            published = _parse_pubdate(item.findtext("pubDate"))
            # Respect the [start, end] contract. Undated items are dropped:
            # an article we can't timestamp can't be placed in a point-in-time
            # view. (The snapshot layer post-filters again as belt-and-braces.)
            if published is None or not (start_dt <= published <= end_dt):
                continue
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            source_el = item.find("source")
            if source_el is not None and (source_el.text or "").strip():
                source = source_el.text.strip()
            elif " - " in title:  # Google appends ' - Source' to titles
                source = title.rsplit(" - ", 1)[-1].strip()
            else:
                source = ""
            if not title or not link:
                continue
            rows.append(
                {
                    "title": title,
                    "url": link,
                    "published_at": published,
                    "source": source,
                    "language": "eng",
                }
            )
            if len(rows) >= max_records:
                break

        if not rows:
            logger.debug(
                f"[google_news_rss] no in-window items for query={query!r} "
                f"({start_dt.date()}..{end_dt.date()})"
            )
            return empty_news_df()

        df = pd.DataFrame(rows)
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
        return df.sort_values("published_at").reset_index(drop=True)
