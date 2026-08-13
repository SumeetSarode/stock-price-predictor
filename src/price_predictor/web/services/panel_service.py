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
    get_quote,
    get_quotes,
)
from price_predictor.web.services.prediction_cache import (
    CachedPrediction,
    get_latest_many,
)
from price_predictor.web.services.search_service import Stock, get_by_ticker
from price_predictor.web.services.watchlist_service import list_watchlist


@dataclass(frozen=True, slots=True)
class PanelCard:
    """One card in the watchlist predictions panel.

    Three states, ordered by progressive enhancement:
      1. No prediction cached      → prediction is None,  "Run prediction" CTA
      2. Fresh cached prediction   → prediction populated, normal age label
      3. Stale cached prediction   → prediction populated + is_stale=True
    """

    ticker: str           # 'RELIANCE.NS'
    name: str             # 'Reliance Industries'
    sector: str           # 'Oil, Gas & Consumable Fuels'
    is_nifty50: bool
    close: float | None = None
    change_pct: float | None = None
    price_direction: str = "neutral"  # day's move direction (separate from prediction!)
    prediction: CachedPrediction | None = None

    @property
    def display_ticker(self) -> str:
        """Show 'RELIANCE' not 'RELIANCE.NS' — the .NS noise hurts scannability."""
        return self.ticker.removesuffix(".NS")

    @property
    def card_direction(self) -> str:
        """Which direction class to tint the whole card with.

        Prefers the prediction direction (the headline) over the day's
        move (the context). Falls back to today's price move when no
        prediction is cached — keeps the card meaningful in either state.
        """
        if self.prediction is not None:
            return self.prediction.direction
        return self.price_direction


async def get_panel_cards(horizon: str = "weekly") -> list[PanelCard]:
    """Return all watchlist cards for the given horizon, in watchlist order.

    Joins:
      - watchlist (SQLite)         → which tickers user has starred
      - search index (CSV)         → names + sectors + N50 flag
      - dashboard cache (memory)   → today's close + day change (context)
      - predictions_cache (SQLite) → latest cached prediction per ticker

    The prediction lookup is a single bulk query (get_latest_many) —
    not N queries — so this is O(1) regardless of watchlist size.
    """
    entries = list_watchlist()
    if not entries:
        return []

    # Current prices for every watched ticker in one call. get_quotes()
    # piggybacks the N50 dashboard snapshot when a ticker is in it (free)
    # and fetches non-N50 names on demand, cached per IST trading day --
    # so this stays correct for the ~450 stocks outside the Nifty 50.
    tickers = [e.ticker for e in entries]
    price_lookup: dict[str, DashboardRow] = await get_quotes(tickers)

    # Bulk-fetch cached predictions for all watched tickers, one DB hit.
    prediction_lookup = get_latest_many(tickers, horizon)

    cards: list[PanelCard] = []
    for entry in entries:
        stock: Stock | None = get_by_ticker(entry.ticker)
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
            price_direction=price_row.direction if price_row else "neutral",
            prediction=prediction_lookup.get(entry.ticker),
        ))

    return cards


async def get_one_card(ticker: str, horizon: str = "weekly") -> PanelCard:
    """Build a single PanelCard — used by POST /api/predictions/run to
    swap one card after a fresh prediction without re-rendering the
    entire panel.

    Reuses the same data sources as get_panel_cards() but skips the
    list_watchlist() filter — we just need the single ticker's view
    even if the user has un-starred it mid-request (graceful degrade).
    """
    # Normalize to match dashboard / cache conventions.
    t = ticker.strip().upper()
    if not t.endswith(".NS"):
        t = f"{t}.NS"

    stock: Stock | None = get_by_ticker(t)
    name = stock.name if stock else t.removesuffix(".NS")
    sector = stock.sector if stock else "—"
    is_n50 = stock.is_nifty50 if stock else False

    price_row = await get_quote(t)

    cached = get_latest_many([t], horizon).get(t)

    return PanelCard(
        ticker=t,
        name=name,
        sector=sector,
        is_nifty50=is_n50,
        close=price_row.close if price_row else None,
        change_pct=price_row.change_pct if price_row else None,
        price_direction=price_row.direction if price_row else "neutral",
        prediction=cached,
    )
