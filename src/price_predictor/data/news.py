"""News fetcher backed by GDELT Doc API 2.0 + trafilatura body extraction.

DESIGN
======

Two concerns, two functions:

1. DISCOVERY (`fetch_news`, `fetch_news_batch`, `fetch_news_paginated`)
   GDELT Doc API 2.0 — free, no auth.
   Coverage: Feb 18, 2017 — present (verified empirically + per GDELT's
   official launch announcement at https://blog.gdeltproject.org/announcing-
   the-next-generation-of-gdelt-2-0/). Earlier prototype docstrings claimed
   a "7-day rolling window" — that's wrong; multi-year queries succeed.
   Returns metadata only: title, url, published_at, source, language.
   On HTTP / JSON failure → raises NewsFetchError.
   On empty results       → returns an empty DataFrame (success-with-0-rows).
   For long ranges that would exceed GDELT's 250-record per-call cap,
   `fetch_news_paginated` slides a per-day (configurable) window with
   polite sleeps between calls.

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
import os
from datetime import UTC, datetime, timedelta
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

# GDELT is keyless and IP-rate-limited (~1 request / 5s); a burst returns
# HTTP 429. A prediction fires company + sector news near-simultaneously, so
# an unlucky collision (or two nearby predictions) can 429 and silently lose
# that run's news. Retry a 429 a few times with linear backoff; every other
# failure still raises immediately. Backoff is deliberately >= GDELT's ~5s
# window so the retry actually lands after the limit resets.
#
# Env-tunable (GDELT_RATE_LIMIT_RETRIES / GDELT_NETWORK_RETRIES): dial GDELT's
# patience without touching code. NOTE: for LIVE predictions the snapshot layer
# OVERRIDES these to 0 when the RSS fallback can catch us -- see
# news_snapshot._rss_fallback -- so a 429 falls over to RSS in <1s instead of
# sleeping ~15s. These defaults still apply to backtests (RSS-forbidden by the
# look-ahead guard) and any direct fetch_news() caller.
GDELT_RATE_LIMIT_RETRIES = int(os.getenv("GDELT_RATE_LIMIT_RETRIES", "2"))
GDELT_RATE_LIMIT_BACKOFF_S = 5.0

# Transient network faults (ConnectTimeout / ReadTimeout / ConnectError) are
# distinct from a 429: they're usually a momentary local blip, not GDELT
# throttling. Retry them a couple times with a SHORT backoff (no point waiting
# 5s for a network hiccup). Seen in the wild: a machine's connection blinked
# and BOTH company + sector news timed out simultaneously.
GDELT_NETWORK_RETRIES = int(os.getenv("GDELT_NETWORK_RETRIES", "2"))
GDELT_NETWORK_BACKOFF_S = 2.0

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


# GDELT's Doc API rejects any quoted phrase below a minimum length with a
# plain-text "The specified phrase is too short." body (which then fails
# JSON parsing). Short NSE names are common: ITC, MRF, DLF, ACC, PVR, IEX,
# IOC ... all 3 chars.
#
# PROVEN LIVE (scripts/diagnose.py floor probe, off-VPN):
#   "IT"                      -> REJECTED (too short)
#   "ITCXYZ"                  -> ACCEPTED
#   ("ITC" OR "ITC Limited")  -> REJECTED (too short)   <-- key finding
# GDELT validates the SHORTEST term, so a bare 3-char name is unusable even
# inside an OR group. We therefore must NOT emit the bare name at all for
# short tickers. Instead we OR together finance-QUALIFIED phrases that are
# each safely above the floor AND actually occur in Indian market coverage
# ("ITC Limited", "ITC shares", ...). This clears the floor and, as a bonus,
# filters out junk (a bare "ITC" would also match unrelated 'IT' chatter).
_MIN_GDELT_QUERY_CHARS = 6
_SHORT_NAME_QUALIFIERS = ("Limited", "Ltd", "shares", "stock", "share price")


def _gdelt_keyword(name: str, *, exact_phrase: bool) -> str:
    """Build the GDELT keyword expression for a company name.

    Normal-length names: a quoted phrase (exact tier) or bare tokens
    (loose tier). Short names (below GDELT's per-term floor): an OR group
    of finance-qualified phrases -- the bare name is NEVER emitted alone
    because GDELT rejects any sub-floor term, even inside an OR.
    """
    n = name.strip()
    if len(n) >= _MIN_GDELT_QUERY_CHARS or n.lower().endswith(("ltd", "limited")):
        return f'"{n}"' if exact_phrase else n
    # Short ticker: only qualified phrases, each safely above the floor.
    variants = " OR ".join(f'"{n} {q}"' for q in _SHORT_NAME_QUALIFIERS)
    return f"({variants})"


def _build_params(
    query: str,
    start_dt: datetime,
    end_dt: datetime,
    lang: str,
    max_records: int,
    *,
    exact_phrase: bool = True,
    source_country: str | None = None,
) -> dict[str, str]:
    """Build the GDELT query-string params dict.

    Relevance controls (GDELT operator syntax, all APPENDED to the query
    string — that's how their API works, there are no separate params):

    - ``exact_phrase`` (default True): wrap the query in double quotes so
      GDELT matches the whole phrase, not each loose token. This is the
      fix for junk like a bare ticker 'INFY' matching a WrestleMania
      recap — '"Infosys"' only matches articles actually containing that
      phrase.
    - ``source_country`` (e.g. 'IN'): bias to a source country via
      ``sourcecountry:``. Left off by default; the relevance ladder in
      ``fetch_news_relevant`` adds it as the most-specific first tier.
    """
    phrase = _gdelt_keyword(query, exact_phrase=exact_phrase)
    parts = [phrase, f"sourcelang:{lang}"]
    if source_country:
        parts.append(f"sourcecountry:{source_country}")
    full_query = " ".join(parts)
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
async def _gdelt_get_json(
    client: httpx.AsyncClient,
    params: dict[str, str],
    *,
    query: str,
    start: str,
    end: str,
    rate_limit_retries: int = GDELT_RATE_LIMIT_RETRIES,
    network_retries: int = GDELT_NETWORK_RETRIES,
) -> Any:
    """GET the GDELT Doc API with bounded retry on transient failures.

    Two independent retry paths, both bounded:

    * HTTP 429 (rate-limit): GDELT throttles bursts to ~1 req/5s. A single
      429 otherwise costs a whole run its news (company + sector fire
      together). Retried up to ``GDELT_RATE_LIMIT_RETRIES`` times with a
      5s/10s backoff (>= GDELT's window so the retry lands after reset).
    * Network faults (ConnectTimeout / ReadTimeout / ConnectError): a
      momentary local blip -- retried up to ``GDELT_NETWORK_RETRIES`` times
      with a short 2s backoff. Seen live: the machine's connection blinked
      and both news calls timed out at once.

    A non-retryable HTTP error, or exhausting either budget, raises
    ``NewsFetchError``. Non-JSON bodies (GDELT's 'too short' plain text)
    also raise ``NewsFetchError``. gather.py soft-fails that to neutral, so
    a prediction never crashes on news.
    """
    network_attempts = 0
    rate_limit_attempts = 0
    while True:
        try:
            resp = await client.get(GDELT_DOC_URL, params=params)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            is_429 = e.response.status_code == 429
            if is_429 and rate_limit_attempts < rate_limit_retries:
                rate_limit_attempts += 1
                backoff = GDELT_RATE_LIMIT_BACKOFF_S * rate_limit_attempts
                logger.debug(
                    f"GDELT 429 for query={query!r} "
                    f"(retry {rate_limit_attempts}); waiting {backoff:.0f}s"
                )
                await asyncio.sleep(backoff)
                continue
            raise NewsFetchError(
                f"GDELT request failed for query={query!r} "
                f"({start}..{end}): {type(e).__name__}: {e}"
            ) from e
        except httpx.TransportError as e:
            # ConnectTimeout / ReadTimeout / ConnectError -- transient blip.
            if network_attempts < network_retries:
                network_attempts += 1
                backoff = GDELT_NETWORK_BACKOFF_S * network_attempts
                logger.debug(
                    f"GDELT network fault ({type(e).__name__}) for "
                    f"query={query!r} (retry {network_attempts}); "
                    f"waiting {backoff:.0f}s"
                )
                await asyncio.sleep(backoff)
                continue
            raise NewsFetchError(
                f"GDELT request failed for query={query!r} "
                f"({start}..{end}): {type(e).__name__}: {e}"
            ) from e
        except httpx.HTTPError as e:
            raise NewsFetchError(
                f"GDELT request failed for query={query!r} "
            f"({start}..{end}): {type(e).__name__}: {e}"
            ) from e
        try:
            return resp.json()
        except ValueError as e:
            raise NewsFetchError(
                f"GDELT returned non-JSON response (status={resp.status_code}): "
                f"{resp.text[:200]!r}"
            ) from e


async def fetch_news(
    query: str,
    start: str,
    end: str,
    *,
    lang: str = "eng",
    max_records: int = 250,
    timeout: float = DEFAULT_TIMEOUT_S,
    client: httpx.AsyncClient | None = None,
    exact_phrase: bool = True,
    source_country: str | None = None,
    rate_limit_retries: int | None = None,
    network_retries: int | None = None,
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
        exact_phrase: Wrap the query in quotes for whole-phrase matching
                (default True — kills loose-token false positives).
        source_country: Optional GDELT sourcecountry bias (e.g. 'IN').
        rate_limit_retries: Override the 429 retry budget. None => module
                default (GDELT_RATE_LIMIT_RETRIES, env-tunable). Pass 0 to
                fail FAST on the first 429 -- used by the snapshot layer for
                live windows where RSS can catch us in <1s (no point sleeping
                ~15s hoping GDELT recovers when a fallback is ready).
        network_retries: Override the network-fault retry budget. None =>
                module default (GDELT_NETWORK_RETRIES, env-tunable).

    Returns:
        DataFrame with columns: title, url, published_at (tz-aware UTC),
        source, language. May be empty if the search legitimately found
        nothing (NOT an error — that's a valid outcome).

    Raises:
        ValueError: On invalid query or date inputs (no network call made).
        NewsFetchError: On HTTP / JSON / timeout failures.
    """
    start_dt, end_dt = _validate_inputs(query, start, end)
    params = _build_params(
        query, start_dt, end_dt, lang, max_records,
        exact_phrase=exact_phrase, source_country=source_country,
    )
    headers = {"User-Agent": DEFAULT_USER_AGENT}

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=timeout, headers=headers)

    try:
        payload = await _gdelt_get_json(
            client, params, query=query, start=start, end=end,
            rate_limit_retries=(
                GDELT_RATE_LIMIT_RETRIES if rate_limit_retries is None
                else rate_limit_retries
            ),
            network_retries=(
                GDELT_NETWORK_RETRIES if network_retries is None
                else network_retries
            ),
        )
    finally:
        if owns_client:
            await client.aclose()

    articles = payload.get("articles", [])
    if not isinstance(articles, list):
        raise NewsFetchError(
            f"GDELT payload['articles'] is not a list: {type(articles).__name__}"
        )

    return _normalize_articles(articles)


# Relevance ladder: most-specific first, loosened only if a tier finds
# nothing. Each tuple is (exact_phrase, source_country). This is the fix
# for both halves of the relevance problem: exact-phrase quoting kills
# loose-token junk (INFY -> WrestleMania), and the IN source bias favours
# Indian coverage — while the fallbacks guarantee we never over-filter to
# zero (dropping the country, then the quotes, before giving up).
_RELEVANCE_LADDER: tuple[tuple[bool, str | None], ...] = (
    (True, "IN"),    # "Infosys" from Indian sources
    (True, None),    # "Infosys" from anywhere (Reuters/Bloomberg on India)
    (False, None),   # loose tokens — last resort so we return *something*
)


async def fetch_news_relevant(
    query: str,
    start: str,
    end: str,
    *,
    lang: str = "eng",
    max_records: int = 250,
    timeout: float = DEFAULT_TIMEOUT_S,
    client: httpx.AsyncClient | None = None,
) -> pd.DataFrame:
    """Fetch news, trading precision for recall only when needed.

    Walks `_RELEVANCE_LADDER`: tries the strictest query first (exact
    phrase, Indian sources) and relaxes one notch at a time ONLY when a
    tier returns zero rows. Returns the first non-empty result, or an
    empty DataFrame if even the loosest tier finds nothing.

    This is the live-path wrapper around `fetch_news`. `fetch_news` itself
    stays a single deterministic call (exact-phrase by default) so the
    backtest snapshot store records one faithful query per (name, date).

    Same args/raises as `fetch_news`. A NewsFetchError on any tier
    propagates immediately (we don't mask real API failures as 'empty').
    """
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(
            timeout=timeout, headers={"User-Agent": DEFAULT_USER_AGENT},
        )
    try:
        df = pd.DataFrame(columns=NEWS_DF_COLUMNS)
        for exact_phrase, country in _RELEVANCE_LADDER:
            df = await fetch_news(
                query, start, end,
                lang=lang, max_records=max_records, timeout=timeout,
                client=client, exact_phrase=exact_phrase,
                source_country=country,
            )
            if not df.empty:
                return df
        return df  # exhausted the ladder — legitimately nothing
    finally:
        if owns_client:
            await client.aclose()


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
# Public API: pagination
# ─────────────────────────────────────────────────────────────
DEFAULT_POLITE_SLEEP_S = 5.0  # GDELT public guidance: ≤ 1 request / 5 seconds.


async def fetch_news_paginated(
    query: str,
    start: str,
    end: str,
    *,
    window_days: int = 1,
    polite_sleep_s: float = DEFAULT_POLITE_SLEEP_S,
    lang: str = "eng",
    max_records: int = 250,
    timeout: float = DEFAULT_TIMEOUT_S,
    client: httpx.AsyncClient | None = None,
    dedupe: bool = True,
) -> pd.DataFrame:
    """Fetch GDELT news across a long date range by sliding a window.

    GDELT's Doc API hard-caps results at `max_records` (≤ 250) per call.
    For multi-day searches on high-volume queries that would exceed that
    cap, this helper chunks `[start, end]` into `window_days`-sized
    sub-windows, calls `fetch_news` per window, sleeps `polite_sleep_s`
    between calls (GDELT's published rate-limit guidance), then
    concatenates the per-window DataFrames into one.

    WHY a separate function (vs a flag on `fetch_news`):
        `fetch_news` is the single-call primitive — one HTTP request, one
        DataFrame, one set of failure semantics. Pagination is a different
        concern (looping, sleeping, deduping) layered on top. Keeping them
        separate matches the existing `fetch_news` / `fetch_news_batch`
        split and keeps each function easy to reason about.

    Args:
        query: Free-text query (same semantics as fetch_news).
        start: ISO YYYY-MM-DD UTC start (inclusive).
        end:   ISO YYYY-MM-DD UTC end   (inclusive).
        window_days: Width of each sub-window in days. Default 1
            (per-day chunking — safest for high-volume queries).
            Must be >= 1.
        polite_sleep_s: Seconds to sleep BETWEEN windows. Not slept
            after the last window. Pass 0.0 to skip sleeping (use in
            tests). Default 5.0 per GDELT's published guidance.
        lang, max_records, timeout: Same as fetch_news.
        client: Optional shared AsyncClient (for connection pooling).
            If None, a temporary client is created for the run.
        dedupe: If True (default), drop duplicate URLs across windows.
            Adjacent windows DON'T overlap by date so duplicates are
            rare, but GDELT can re-surface the same article under
            different `seendate` timestamps if it's re-crawled — dedupe
            keeps the FIRST occurrence (oldest seendate).

    Returns:
        Single DataFrame with the same 5 columns as `fetch_news`,
        sorted by `published_at` ascending. Empty DataFrame if all
        windows legitimately found nothing (NOT an error).

    Raises:
        ValueError: Bad inputs (empty query, bad dates, window_days < 1,
            polite_sleep_s < 0). No network call made.
        NewsFetchError: Any window's fetch fails. Fail-fast — the partial
            results from earlier windows are discarded so the caller never
            sees an inconsistent half-fetched DataFrame. Wrap the call in
            try/except if you want partial-success semantics.
    """
    start_dt, end_dt = _validate_inputs(query, start, end)
    if window_days < 1:
        raise ValueError(f"window_days must be >= 1, got {window_days}")
    if polite_sleep_s < 0:
        raise ValueError(f"polite_sleep_s must be >= 0, got {polite_sleep_s}")

    headers = {"User-Agent": DEFAULT_USER_AGENT}
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=timeout, headers=headers)

    try:
        windows = list(_iter_windows(start_dt, end_dt, window_days))
        frames: list[pd.DataFrame] = []
        for i, (w_start, w_end) in enumerate(windows):
            df = await fetch_news(
                query,
                w_start.strftime("%Y-%m-%d"),
                w_end.strftime("%Y-%m-%d"),
                lang=lang,
                max_records=max_records,
                timeout=timeout,
                client=client,
            )
            frames.append(df)
            is_last = i == len(windows) - 1
            if not is_last and polite_sleep_s > 0:
                await asyncio.sleep(polite_sleep_s)
    finally:
        if owns_client:
            await client.aclose()

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=NEWS_DF_COLUMNS
    )
    if dedupe and not combined.empty:
        combined = combined.drop_duplicates(subset="url", keep="first")
    if not combined.empty:
        combined = combined.sort_values("published_at", kind="stable").reset_index(
            drop=True
        )
    return combined


def _iter_windows(
    start_dt: datetime, end_dt: datetime, window_days: int
) -> list[tuple[datetime, datetime]]:
    """Yield non-overlapping (window_start, window_end) pairs covering [start, end].

    Both bounds are date-aligned (HH:MM:SS is dropped — GDELT day-resolution
    queries already cover full days via _to_gdelt_datetime). Each window is
    inclusive on both ends. The final window is clipped at end_dt.

    Examples (window_days=1):
        start=2024-01-01, end=2024-01-01 -> [(2024-01-01, 2024-01-01)]
        start=2024-01-01, end=2024-01-03 -> [(2024-01-01, 2024-01-01),
                                             (2024-01-02, 2024-01-02),
                                             (2024-01-03, 2024-01-03)]
    """
    out: list[tuple[datetime, datetime]] = []
    cursor = start_dt
    step = timedelta(days=window_days)
    one_day = timedelta(days=1)
    while cursor <= end_dt:
        # Window inclusive at both ends => width is (window_days - 1) days.
        window_end = min(cursor + step - one_day, end_dt)
        out.append((cursor, window_end))
        cursor = window_end + one_day
    return out


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
