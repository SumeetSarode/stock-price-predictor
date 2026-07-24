"""Resilient live news fetch: GDELT primary, Google-News-RSS fallback.

WHY THIS MODULE EXISTS
======================
The GDELT->RSS fallback used to live *only* inside
``NewsSnapshot.get_or_fetch`` -- which is called *only* in backtest/replay
mode, where the look-ahead guard then forbids RSS from ever firing. So the
fallback was dead code on every LIVE path (the News tab and the news_impact
gather step both called raw ``fetch_news`` and had no idea RSS existed).

This module is the ONE place live callers go for news. It:

1. Tries GDELT first (the richer, historical-capable source).
2. If GDELT fails AND the window is fresh enough for RSS to serve
   (the look-ahead guard -- RSS is live-only), falls over to Google News
   RSS in <1s.
3. When RSS can catch us, tells GDELT to FAIL FAST (0 retries) so we don't
   burn ~15s of GDELT backoff before the fallback even starts.

Backtests keep using ``NewsSnapshot.get_or_fetch`` (deterministic caching +
point-in-time semantics). ``rss_can_catch`` is the shared predicate both this
module and the snapshot layer use, so the "is RSS allowed here?" decision can
never disagree between the two.
"""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd
from loguru import logger

from price_predictor.data.news import (
    DEFAULT_TIMEOUT_S,
    NewsFetchError,
    fetch_news,
    fetch_news_relevant,
)
from price_predictor.data.news_providers import GoogleNewsRssProvider
from price_predictor.data.news_providers import NewsFetchError as RssFetchError
from price_predictor.data.news_providers.base import NewsProviderCoverage


def rss_can_catch(end: str | date | datetime) -> bool:
    """Will the live RSS fallback be able to serve a window ending at ``end``?

    True iff the fallback is enabled AND ``end`` is within RSS's freshness
    horizon (the look-ahead guard -- RSS returns RECENT news only, so it must
    never answer for a backtest date).

    This is the SINGLE source of truth shared by:
      * the live ``fetch_news_resilient`` fast-fail + fallback decision, and
      * ``NewsSnapshot._rss_can_catch`` (backtest layer),
    so the two can never disagree about when RSS is permitted.
    """
    from price_predictor.config.settings import settings

    if not settings.news_rss_fallback_enabled:
        return False
    cov = NewsProviderCoverage(
        historical=False, freshness_days=settings.news_rss_freshness_days
    )
    if isinstance(end, str):
        end_dt = pd.Timestamp(end, tz="UTC").to_pydatetime()
    elif isinstance(end, datetime):
        end_dt = end if end.tzinfo else pd.Timestamp(end, tz="UTC").to_pydatetime()
    else:  # date
        end_dt = pd.Timestamp(end, tz="UTC").to_pydatetime()
    return cov.can_serve_end(end_dt)


async def fetch_news_resilient(
    query: str,
    start: str,
    end: str,
    *,
    lang: str = "eng",
    max_records: int = 250,
    timeout: float = DEFAULT_TIMEOUT_S,
    use_ladder: bool = True,
    exact_phrase: bool = True,
    source_country: str | None = "IN",
) -> pd.DataFrame:
    """Fetch live news with a GDELT->RSS fallback.

    Args:
        query: Free-text news query (e.g. "Reliance Industries").
        start, end: ISO YYYY-MM-DD window (end inclusive).
        lang: 3-letter language code.
        max_records: GDELT cap (<=250).
        timeout: Per-request timeout.
        use_ladder: True (company news) -> use the relevance ladder
            (``fetch_news_relevant``: exact-phrase + India bias, relaxed only
            if a tier finds nothing). False (sector / tab) -> a single
            ``fetch_news`` call with the given exact_phrase/source_country.
        exact_phrase / source_country: only used when ``use_ladder`` is False.

    Returns:
        Articles DataFrame. Empty is a VALID outcome (source found nothing).

    Raises:
        NewsFetchError: GDELT failed AND RSS couldn't catch us (disabled,
            out-of-horizon backtest window, or RSS also failed/empty).
        ValueError: bad inputs (no fallback would help).
    """
    can_catch = rss_can_catch(end)
    # Fast-fail GDELT when RSS is standing by; keep full patience otherwise.
    retries = 0 if can_catch else None

    try:
        if use_ladder:
            return await fetch_news_relevant(
                query, start, end, lang=lang, max_records=max_records,
                timeout=timeout,
                rate_limit_retries=retries, network_retries=retries,
            )
        return await fetch_news(
            query, start, end, lang=lang, max_records=max_records,
            timeout=timeout, exact_phrase=exact_phrase,
            source_country=source_country,
            rate_limit_retries=retries, network_retries=retries,
        )
    except ValueError:
        raise  # caller's bad input -- no fallback helps
    except NewsFetchError as gdelt_err:
        return await _rss_fallback(
            query, start, end, lang=lang,
            exact_phrase=exact_phrase, gdelt_err=gdelt_err,
        )


async def _rss_fallback(
    query: str,
    start: str,
    end: str,
    *,
    lang: str,
    exact_phrase: bool,
    gdelt_err: NewsFetchError,
) -> pd.DataFrame:
    """Live-only fallback when GDELT fails. Re-raises ``gdelt_err`` unless a
    fresh, non-empty RSS result is available (look-ahead guard via
    ``rss_can_catch``)."""
    from price_predictor.config.settings import settings

    if not rss_can_catch(end):
        logger.warning(
            f"news RSS fallback SKIPPED (disabled, or window ends {end} older "
            f"than {settings.news_rss_freshness_days}d horizon); re-raising "
            f"GDELT error for query={query!r}"
        )
        raise gdelt_err

    provider = GoogleNewsRssProvider(
        freshness_days=settings.news_rss_freshness_days
    )
    logger.info(
        f"news RSS fallback: GDELT failed ({gdelt_err}); trying "
        f"{provider.name} for query={query!r}"
    )
    try:
        df = await provider.fetch(
            query, start, end, lang=lang, exact_phrase=exact_phrase,
        )
    except (ValueError, RssFetchError) as rss_err:
        logger.warning(
            f"news RSS fallback ALSO failed ({rss_err}); re-raising original "
            f"GDELT error for query={query!r}"
        )
        raise gdelt_err from rss_err

    if df.empty:
        logger.warning(
            f"news RSS fallback returned no articles for query={query!r}; "
            f"re-raising GDELT error"
        )
        raise gdelt_err

    logger.info(
        f"news RSS fallback SUCCEEDED: {len(df)} article(s) from "
        f"{provider.name} for query={query!r}"
    )
    return df
