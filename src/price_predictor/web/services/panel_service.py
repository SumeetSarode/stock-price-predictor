"""Watchlist panel service — assembles the data for the left-side
predictions panel on the home page.

Joins three sources:
  1. Watchlist (SQLite)        → which tickers user has starred
  2. Search index (CSV)        → company names + sectors + N50 flag
  3. Dashboard cache (memory)  → current close + day change %

Phase 2 scope: cards show price context + "View details" link.
Phase 3 will add cached prediction data (entry / target / stop / RR).

This service is intentionally lightweight: it does NOT trigger LLM
calls. The 30-60s prediction work is decoupled into the prediction
cache layer (Phase 3).
"""
from __future__ import annotations

from dataclasses import dataclass

from price_predictor.web.services.dashboard_service import (
    DashboardRow,
    get_dashboard,
)
from price_predictor.web.services.search_service import Stock, get_by_ticker
from price_predictor.web.services.watchlist_service import list_watchlist


@dataclass(frozen=True, slots=True)
class PanelCard:
    """One card in the watchlist predictions panel.

    Phase 2 only has the 'context' fields populated. Phase 3 will add
    a ``prediction`` field carrying the cached PredictionView.
    """

    ticker: str           # 'RELIANCE.NS'
    name: str             # 'Reliance Industries'
    sector: str           # 'Oil, Gas & Consumable Fuels'
    is_nifty50: bool
    close: float | None = None
    change_pct: float | None = None
    direction: str = "neutral"  # bullish | bearish | neutral

    @property
    def display_ticker(self) -> str:
        """Show 'RELIANCE' not 'RELIANCE.NS' — the .NS noise hurts scannability."""
        return self.ticker.removesuffix(".NS")


async def get_panel_cards(horizon: str = "weekly") -> list[PanelCard]:
    """Return all watchlist cards for the given horizon, in watchlist order.

    The ``horizon`` parameter is accepted now even though Phase 2 doesn't
    use it — Phase 3 will key the prediction cache by (ticker, horizon).
    Keeping the signature forward-compatible avoids touching callers later.
    """
    entries = list_watchlist()
    if not entries:
        return []

    # Use the dashboard snapshot for current prices. Don't force_refresh —
    # if the dashboard is cold for this process, we'd block for ~5-10s.
    # Better: render cards instantly with None close and let the user
    # refresh manually. (In practice the dashboard cache is almost always
    # warm because the user opens the home page first.)
    snapshot = await get_dashboard()
    price_lookup: dict[str, DashboardRow] = {r.ticker: r for r in snapshot.rows}

    cards: list[PanelCard] = []
    for entry in entries:
        stock: Stock | None = get_by_ticker(entry.ticker)
        # Fall back to the ticker itself if it's a non-N500 unknown — the
        # search index is bounded; user could have starred via the
        # detail page for any NSE symbol.
        name = stock.name if stock else entry.ticker.removesuffix(".NS")
        sector = stock.sector if stock else "—"
        is_n50 = stock.is_nifty50 if stock else False

        price_row = price_lookup.get(entry.ticker)
        cards.append(PanelCard(
            ticker=entry.ticker,
            name=name,
            sector=sector,
            is_nifty50=is_n50,
            close=price_row.close if price_row else None,
            change_pct=price_row.change_pct if price_row else None,
            direction=price_row.direction if price_row else "neutral",
        ))

    return cards
