"""Deterministic, LLM-free gathering of news-impact inputs.

WHY THIS EXISTS ("gather in code, reason once")
================================================
The original news_impact agent let the LLM drive a tool loop: it decided
which of 4 tools to call, we paid tokens for those decisions, and the
growing tool transcript was re-sent every turn (2-4x token cost). But
*fetching* data needs no intelligence — it's pure code. Only the final
*judgment* (bullish? how much? why?) needs the model.

So this module does all the fetching deterministically and in parallel:
company news, sector news, corporate filings, analyst estimates, and
recent prices. The agent layer then makes ONE synthesis call over the
gathered result. No tool loop, no re-sent context.

LOOK-AHEAD DEFENSE (ported 1:1 from the old tools)
==================================================
Everything is anchored to `as_of` (the backtest date), which defaults to
the replay contextvar so existing `replay_context(...)` plumbing keeps
working unchanged:
  - news / sector news → served from the snapshot store in replay
  - filings / prices   → date window ends at as_of (never past it)
  - estimates          → SKIPPED in replay (yfinance has no PIT archive)

SOFT-FAIL PER SOURCE
====================
Each fetch is independent: a rate-limited estimates call or an empty
filings feed doesn't sink the others. Failures are recorded in
`errors` so the synthesizer can note what was missing.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from loguru import logger

from price_predictor.agents.news_impact.sectors import (
    sector_for,
    sector_query_for,
)
from price_predictor.agents.price_agent.agent import fetch_prices_tool
from price_predictor.data.estimates import EstimatesFetchError, fetch_estimates
from price_predictor.data.filings import FilingsFetchError, fetch_filings
from price_predictor.data.news import NewsFetchError, fetch_news
from price_predictor.data.news_snapshot import (
    NewsSnapshotError,
    get_news_snapshot,
)
from price_predictor.prediction.replay_context import get_as_of
from price_predictor.web.services import search_service

IST = timezone(timedelta(hours=5, minutes=30))

# Defaults for each source's look-back window (days).
_COMPANY_NEWS_DAYS = 7
_SECTOR_NEWS_DAYS = 7
_FILINGS_DAYS = 30
_PRICES_DAYS = 30

# Row caps — keep the single synthesis prompt tight.
_MAX_COMPANY_ARTICLES = 20
_MAX_SECTOR_ARTICLES = 8
_MAX_FILINGS = 20


@dataclass(slots=True)
class NewsImpactInputs:
    """Everything the synthesizer needs, gathered deterministically.

    Plain data — no behavior. `errors` collects soft-fail notes so the
    synthesis prompt can honestly say what was unavailable.
    """

    ticker: str
    company_name: str
    sector: str | None
    window_start: str
    window_end: str
    company_news: list[dict] = field(default_factory=list)
    sector_news: list[dict] = field(default_factory=list)
    filings: list[dict] = field(default_factory=list)
    estimates: dict | None = None
    prices: dict | None = None
    errors: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _window(days_back: int, as_of) -> tuple[str, str]:
    """(start_iso, end_iso) ending at as_of (replay) or today (live)."""
    end = as_of or datetime.now(IST).date()
    start = end - timedelta(days=days_back)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _company_name_for(ticker: str) -> str:
    """Human company name for news search; fall back to the bare symbol.

    GDELT searches article text, so the company's common name returns
    far better hits than a ticker symbol.
    """
    stock = search_service.get_by_ticker(ticker)
    if stock is not None and stock.name.strip():
        return stock.name.strip()
    return ticker.removesuffix(".NS")


async def _fetch_news_rows(query: str, days_back: int, as_of, cap: int) -> list[dict]:
    """Fetch + shape news rows (snapshot in replay, live otherwise)."""
    start, end = _window(days_back, as_of)
    snapshot = get_news_snapshot()
    if as_of is not None and snapshot is not None:
        df = await snapshot.get_or_fetch(query, as_of, days_back)
    else:
        df = await fetch_news(query, start, end)
    rows = df.head(cap)
    return [
        {
            "title": str(r["title"])[:250],
            "url": str(r["url"]),
            "published_at": str(r["published_at"]),
            "source": str(r.get("source", "")),
        }
        for _, r in rows.iterrows()
    ]


async def _fetch_filings_rows(nse_symbol: str, as_of) -> list[dict]:
    """Fetch + shape corporate filings (date window honors as_of)."""
    start, end = _window(_FILINGS_DAYS, as_of)
    df = await fetch_filings(nse_symbol, start, end)
    rows = df.head(_MAX_FILINGS)
    return [
        {
            "kind": str(r["kind"]),
            "announced_at": str(r["announced_at"]),
            "event_type": r["event_type"],
            "subject": str(r["subject"])[:300],
        }
        for _, r in rows.iterrows()
    ]


async def _fetch_estimates_summary(yf_ticker: str, as_of) -> dict | None:
    """Analyst estimates summary, or None in replay (no PIT archive)."""
    if as_of is not None:
        # yfinance only has *current* consensus; using it in a backtest
        # would leak the future. Degrade to 'no coverage'.
        return None
    est = await fetch_estimates(yf_ticker)
    next_eps = next((e for e in est.earnings_estimates
                     if e.period in ("0q", "+1q")), None)
    return {
        "has_coverage": est.has_coverage,
        "next_quarter_eps_consensus": next_eps.avg if next_eps else None,
        "current_price": est.price_targets.current if est.price_targets else None,
        "price_target_mean": est.price_targets.mean if est.price_targets else None,
    }


async def _fetch_prices_summary(yf_ticker: str, as_of) -> dict:
    """Recent OHLCV summary via the price_agent tool (sync → thread)."""
    start, end = _window(_PRICES_DAYS, as_of)
    return await asyncio.to_thread(
        fetch_prices_tool, yf_ticker, start, end, include_bars=False,
    )


async def gather_news_impact_inputs(
    ticker: str,
    *,
    as_of=None,
) -> NewsImpactInputs:
    """Gather all impact inputs for a ticker in parallel, LLM-free.

    Args:
        ticker: yfinance ticker, e.g. 'RELIANCE.NS'.
        as_of: backtest date; defaults to the replay contextvar (None
            in live mode → today). Explicit arg wins if given.

    Returns:
        A populated NewsImpactInputs. Individual sources that fail are
        left empty/None and noted in `.errors`; the call never raises
        for a data-source failure.
    """
    if as_of is None:
        as_of = get_as_of()

    nse_symbol = ticker.removesuffix(".NS").upper()
    company_name = _company_name_for(ticker)
    sector = sector_for(ticker)
    sector_query = sector_query_for(ticker)
    start, end = _window(_COMPANY_NEWS_DAYS, as_of)

    result = NewsImpactInputs(
        ticker=ticker,
        company_name=company_name,
        sector=sector,
        window_start=start,
        window_end=end,
    )

    # Build the task list. Sector news is conditional — only when we have
    # a mapped sector phrase (otherwise it's company news only).
    tasks: list = [
        _fetch_news_rows(company_name, _COMPANY_NEWS_DAYS, as_of,
                         _MAX_COMPANY_ARTICLES),
        _fetch_filings_rows(nse_symbol, as_of),
        _fetch_estimates_summary(ticker, as_of),
        _fetch_prices_summary(ticker, as_of),
    ]
    if sector_query:
        tasks.append(
            _fetch_news_rows(sector_query, _SECTOR_NEWS_DAYS, as_of,
                             _MAX_SECTOR_ARTICLES)
        )

    gathered = await asyncio.gather(*tasks, return_exceptions=True)
    company_news, filings, estimates, prices = gathered[:4]
    sector_news = gathered[4] if sector_query else []

    result.company_news = _unwrap(result, company_news, "company news", [])
    result.filings = _unwrap(result, filings, "filings", [])
    result.estimates = _unwrap(result, estimates, "estimates", None)
    result.prices = _unwrap(result, prices, "prices", None)
    result.sector_news = _unwrap(result, sector_news, "sector news", [])
    return result


def _unwrap(result: NewsImpactInputs, value, label: str, default):
    """Record soft-fails; return `default` on exception, else the value."""
    if isinstance(value, Exception):
        msg = f"{label} unavailable: {type(value).__name__}: {value}"
        logger.debug(msg)
        result.errors.append(msg)
        return default
    return value


# Re-exported so callers can catch the same errors these fetchers raise.
__all__ = [
    "NewsImpactInputs",
    "gather_news_impact_inputs",
    "NewsFetchError",
    "NewsSnapshotError",
    "FilingsFetchError",
    "EstimatesFetchError",
]
