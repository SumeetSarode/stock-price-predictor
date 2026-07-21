"""Sector → news-query cheat-sheet for sector-aware impact analysis.

WHY
===
A great company in a sinking sector is still at risk (and vice-versa).
To let the impact analyzer weigh the whole industry, we fetch a little
sector news alongside the company's own news. GDELT is a free-text
search, so each yfinance sector label is mapped to a concise,
India-contextual search phrase that actually returns relevant Indian
market coverage (e.g. 'Technology' → 'Indian IT sector').

The labels on the LEFT are yfinance's GICS-style sectors — the same
taxonomy the Phase-0 backfill writes into the bundled CSV, so lookups
line up exactly. Anything not in this map (notably the 'NSE Listed'
fallback for stocks yfinance had no sector for) yields None → the
analyzer simply skips sector news for that stock and leans on company
news alone. Graceful degradation, no special-casing downstream.
"""
from __future__ import annotations

from price_predictor.web.services import search_service

# yfinance sector label → GDELT-friendly, India-contextual search phrase.
# Kept deliberately small (one row per real sector). If yfinance ever
# introduces a new sector label, add a row here; until then it degrades
# to "no sector news" rather than searching a bad phrase.
SECTOR_NEWS_QUERY: dict[str, str] = {
    "Technology": "Indian IT sector",
    "Financial Services": "Indian banking and financial sector",
    "Energy": "Indian energy oil and gas sector",
    "Healthcare": "Indian pharmaceutical and healthcare sector",
    "Consumer Cyclical": "Indian auto and consumer discretionary sector",
    "Consumer Defensive": "Indian FMCG sector",
    "Basic Materials": "Indian metals and materials sector",
    "Industrials": "Indian industrial and manufacturing sector",
    "Real Estate": "Indian real estate sector",
    "Communication Services": "Indian telecom and media sector",
    "Utilities": "Indian power and utilities sector",
}


def sector_for(ticker: str) -> str | None:
    """Return the yfinance sector label for a ticker, or None if unknown.

    Looks the ticker up in the bundled search index. Returns None when
    the ticker isn't in the index or carries the 'NSE Listed' fallback
    (i.e. no real sector was resolved).
    """
    stock = search_service.get_by_ticker(ticker)
    if stock is None:
        return None
    sector = stock.sector.strip()
    if not sector or sector == "NSE Listed":
        return None
    return sector


def sector_query_for(ticker: str) -> str | None:
    """Return a sector news search phrase for a ticker, or None.

    None means "don't fetch sector news for this stock" — either its
    sector is unknown/unresolved, or it's a sector we don't have a
    curated phrase for. Callers treat None as "company news only".
    """
    sector = sector_for(ticker)
    if sector is None:
        return None
    return SECTOR_NEWS_QUERY.get(sector)
