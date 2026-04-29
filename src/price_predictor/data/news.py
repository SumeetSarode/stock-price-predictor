"""News fetcher backed by GDELT Doc API 2.0 + trafilatura body extraction.

DESIGN
======

Two concerns, two functions:

1. DISCOVERY (`fetch_news`, `fetch_news_batch`)
   GDELT Doc API 2.0 — free, no auth, 7-day rolling window.
   Returns metadata only: title, url, published_at, source, language.
   On HTTP / JSON failure → raises NewsFetchError.
   On empty results       → returns an empty DataFrame (success-with-0-rows).

2. EXTRACTION (`fetch_article_body`)
   Per-URL HTML fetch + trafilatura text extraction.
   Returns ArticleBody (status="success" with body, or status="error" with
   error_message). Never raises — extraction failure is *normal at scale*
   (paywalls, JS-only sites, bot blocks), so we surface it as data instead
   of exceptions. The status field keeps "successful empty body" (rare but
   possible) distinguishable from "extraction failed" (common).

Async-first because:
- News fetching scales to dozens of stocks per run.
- Body extraction = one HTTP call per article = naturally I/O-bound.
- httpx.AsyncClient + asyncio.Semaphore = polite parallelism.

Date handling:
- Input:  ISO YYYY-MM-DD strings (consistent with prices module).
- Output: tz-aware UTC datetimes (GDELT reports in UTC).
- GDELT wants YYYYMMDDHHMMSS UTC — converted internally.
- start gets 000000, end gets 235959 (end-inclusive, mirrors prices).
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import httpx
import pandas as pd
import trafilatura
from loguru import logger

from price_predictor.data.schema import ArticleBody

# ─────────────────────────────────────────────────────────────
# Module constants
# ─────────────────────────────────────────────────────────────
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_TIMEOUT_S = 15.0
DEFAULT_USER_AGENT = "price-predictor/0.1 (+https://github.com/sumeet-s2597)"

NEWS_DF_COLUMNS = ["title", "url", "published_at", "source", "language"]


class NewsFetchError(RuntimeError):
    """Raised when GDELT discovery fails (HTTP / JSON / timeout)."""


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────
def _validate_inputs(query: str, start: str, end: str) -> tuple[datetime, datetime]:
    """Validate query + parse dates. Raises ValueError on bad input."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"query must be a non-empty string, got {query!r}")

    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
        end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as e:
        raise ValueError(
            f"Invalid date format: {e}. Dates must be ISO format YYYY-MM-DD "
            "(e.g., '2024-01-31')."
        ) from e

    if start_dt > end_dt:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    return start_dt, end_dt


def _to_gdelt_datetime(dt: datetime, *, end_of_day: bool) -> str:
    """Convert tz-aware UTC datetime → GDELT YYYYMMDDHHMMSS string."""
    suffix = "235959" if end_of_day else "000000"
    return dt.strftime("%Y%m%d") + suffix


def _build_params(
    query: str,
    start_dt: datetime,
    end_dt: datetime,
    lang: str,
    max_records: int,
) -> dict[str, str]:
    """Build the GDELT query-string params dict.

    Note: we APPEND `sourcelang:<lang>` to the query (GDELT operator syntax)
    rather than passing it as a separate param — that's how their API works.
    """
    full_query = f"{query} sourcelang:{lang}"
    return {
        "query": full_query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(max_records),
        "startdatetime": _to_gdelt_datetime(start_dt, end_of_day=False),
        "enddatetime": _to_gdelt_datetime(end_dt, end_of_day=True),
    }


def _parse_seendate(seendate: str) -> datetime:
    """Parse GDELT seendate (YYYYMMDDTHHMMSSZ) → tz-aware UTC datetime."""
    return datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)


def _normalize_articles(articles: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize raw GDELT article dicts → DataFrame with our 5 canonical columns."""
    if not articles:
        return pd.DataFrame(columns=NEWS_DF_COLUMNS)

    rows = []
    for a in articles:
        # GDELT sometimes omits fields on weird articles; skip rather than crash.
        if not a.get("title") or not a.get("url") or not a.get("seendate"):
            logger.debug("Skipping article with missing required fields: {}", a)
            continue
        try:
            published_at = _parse_seendate(a["seendate"])
        except ValueError:
            logger.debug("Skipping article with bad seendate {!r}", a.get("seendate"))
            continue
        rows.append(
            {
                "title": a["title"],
                "url": a["url"],
                "published_at": published_at,
                "source": a.get("domain", ""),
                "language": a.get("language", ""),
            }
        )

    return pd.DataFrame(rows, columns=NEWS_DF_COLUMNS)


# ─────────────────────────────────────────────────────────────
# Public API: discovery
# ─────────────────────────────────────────────────────────────
async def fetch_news(
    query: str,
    start: str,
    end: str,
    *,
    lang: str = "eng",
    max_records: int = 250,
    timeout: float = DEFAULT_TIMEOUT_S,
    client: httpx.AsyncClient | None = None,
) -> pd.DataFrame:
    """Fetch news article metadata from GDELT Doc API 2.0.

    Args:
        query: Free-text query (caller-crafted; e.g., "Reliance Industries").
        start: ISO YYYY-MM-DD UTC start date (inclusive, 00:00:00).
        end:   ISO YYYY-MM-DD UTC end date (inclusive, 23:59:59).
        lang:  3-letter ISO language code (GDELT format). Default "eng".
        max_records: GDELT cap is 250.
        timeout: Per-request timeout in seconds.
        client: Optional shared AsyncClient (for batch reuse). If None, a
                temporary client is created per call.

    Returns:
        DataFrame with columns: title, url, published_at (tz-aware UTC),
        source, language. May be empty if the search legitimately found
        nothing (NOT an error — that's a valid outcome).

    Raises:
        ValueError: On invalid query or date inputs (no network call made).
        NewsFetchError: On HTTP / JSON / timeout failures.
    """
    start_dt, end_dt = _validate_inputs(query, start, end)
    params = _build_params(query, start_dt, end_dt, lang, max_records)
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=timeout, headers=headers)

    try:
        resp = await client.get(GDELT_DOC_URL, params=params)
        resp.raise_for_status()
        try:
            payload = resp.json()
        except ValueError as e:
            raise NewsFetchError(
                f"GDELT returned non-JSON response (status={resp.status_code}): "
                f"{resp.text[:200]!r}"
            ) from e
    except httpx.HTTPError as e:
        raise NewsFetchError(
            f"GDELT request failed for query={query!r} "
            f"({start}..{end}): {type(e).__name__}: {e}"
        ) from e
    finally:
        if owns_client:
            await client.aclose()

    articles = payload.get("articles", [])
    if not isinstance(articles, list):
        raise NewsFetchError(
            f"GDELT payload['articles'] is not a list: {type(articles).__name__}"
        )

    return _normalize_articles(articles)


async def fetch_news_batch(
    queries: list[str],
    start: str,
    end: str,
    *,
    lang: str = "eng",
    max_records: int = 250,
    concurrency: int = 5,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> dict[str, pd.DataFrame | Exception]:
    """Fetch news for many queries in parallel.

    Uses a single shared AsyncClient (connection pooling) and a Semaphore
    to cap concurrent in-flight requests (polite to GDELT).

    Args:
        queries: List of query strings — e.g., ["Reliance Industries", "TCS"].
        start, end, lang, max_records, timeout: Same as fetch_news.
        concurrency: Max in-flight requests at once. Default 5.

    Returns:
        Dict mapping each input query → DataFrame on success OR Exception
        on failure. One bad query never breaks the batch — caller decides
        how to handle individual failures.
    """
    if not queries:
        return {}

    sem = asyncio.Semaphore(concurrency)
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:

        async def _one(q: str) -> pd.DataFrame:
            async with sem:
                return await fetch_news(
                    q, start, end,
                    lang=lang, max_records=max_records,
                    timeout=timeout, client=client,
                )

        results = await asyncio.gather(
            *(_one(q) for q in queries),
            return_exceptions=True,
        )

    return dict(zip(queries, results, strict=True))


# ─────────────────────────────────────────────────────────────
# Public API: extraction
# ─────────────────────────────────────────────────────────────
async def fetch_article_body(
    url: str,
    *,
    timeout: float = 10.0,
    client: httpx.AsyncClient | None = None,
) -> ArticleBody:
    """Fetch + extract clean article body text from a URL.

    Returns ArticleBody (status-tagged) instead of raising. Extraction
    failure is *normal at scale* (paywalls, bot blocks, JS-only pages),
    so we surface failures as data, not exceptions.

    Args:
        url: Article URL.
        timeout: Per-request timeout in seconds.
        client: Optional shared AsyncClient.

    Returns:
        ArticleBody with status="success" + body (non-empty) on success,
        OR status="error" + error_message on any failure (HTTP error,
        timeout, trafilatura returning None/empty).
    """
    if not isinstance(url, str) or not url.strip():
        return ArticleBody(status="error", error_message=f"Invalid URL: {url!r}")

    headers = {"User-Agent": DEFAULT_USER_AGENT}
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True)

    try:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
        except httpx.HTTPError as e:
            msg = f"HTTP fetch failed: {type(e).__name__}: {e}"
            logger.warning("Article body fetch failed for {}: {}", url, msg)
            return ArticleBody(status="error", error_message=msg)

        try:
            extracted = trafilatura.extract(html)
        except Exception as e:  # trafilatura's internal failures
            msg = f"trafilatura extract raised: {type(e).__name__}: {e}"
            logger.warning("Body extraction failed for {}: {}", url, msg)
            return ArticleBody(status="error", error_message=msg)

        if not extracted:
            msg = "trafilatura returned no content (paywall, JS-only, or unsupported layout)"
            logger.warning("Empty extraction for {}", url)
            return ArticleBody(status="error", error_message=msg)

        return ArticleBody(status="success", body=extracted)
    finally:
        if owns_client:
            await client.aclose()
