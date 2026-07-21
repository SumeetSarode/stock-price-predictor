"""Sector news query for sector-aware impact analysis.

WHY
===
A great company in a sinking sector is still at risk (and vice-versa).
To let the impact analyzer weigh the whole industry, we fetch a little
sector news alongside the company's own news.

The query is built straight from the stock's yfinance sector label (e.g.
"Technology", "Basic Materials") with NO geographic qualifier. That's
deliberate: searching the bare sector, with no India country bias in the
fetch, naturally surfaces coverage from wherever the sector's news lives.
Indian IT is moved by US client spending, Indian metals by Chinese
demand, energy by OPEC — a geography-neutral search picks all of that up
without us hardcoding which foreign market matters for which sector.

The sector label comes from the bundled search index (populated by the
Phase-0 backfill). Stocks with no resolved sector (the 'NSE Listed'
fallback) yield an empty list → the analyzer skips sector news and leans
on company news alone. Graceful degradation, no special-casing.
"""
from __future__ import annotations

from price_predictor.web.services import search_service


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
    """Return a geography-neutral sector news query, or None.

    None means "don't fetch sector news for this stock" — its sector is
    unknown/unresolved. The query is a CONCEPT query: search it with loose
    token matching (exact_phrase=False) and no India country bias so it
    picks up global sector coverage naturally.
    """
    sector = sector_for(ticker)
    if sector is None:
        return None
    return f"{sector} sector"
