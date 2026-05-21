"""Stock detail service — assembles everything the detail page needs.

Joins:
  - search index (CSV)              → name + sector + N50 flag
  - watchlist (SQLite)              → is_watched for ⭐ button state
  - dashboard cache (memory)        → today's close + change % (context)
  - predictions_cache (SQLite)      → cached prediction for the chosen horizon

The full prediction view dict (with rationale, signals, technical summary)
is reconstructed from the cache's view_json blob — no LLM call needed
when rendering a cached prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from price_predictor.web.services.dashboard_service import (
    DashboardRow,
    get_dashboard,
)
from price_predictor.web.services.prediction_cache import (
    CachedPrediction,
    get_latest,
)
from price_predictor.web.services.search_service import Stock, get_by_ticker
from price_predictor.web.services.watchlist_service import is_watched


def _normalize(ticker: str) -> str:
    """Match watchlist/cache normalization: UPPER + .NS suffix."""
    t = ticker.strip().upper()
    if not t.endswith(".NS"):
        t = f"{t}.NS"
    return t


@dataclass(frozen=True, slots=True)
class StockDetail:
    """Everything the stock detail page needs to render.

    Two layers:
      - 'header' fields (always present): ticker, name, sector, is_n50,
        is_watched, close, change_pct, price_direction.
      - 'body' fields (horizon-specific): the cached prediction for the
        chosen horizon, plus the view dict for the rich card render.
        Both are None when no prediction is cached.
    """

    # Header — stable across horizon switches
    ticker: str
    name: str
    sector: str
    is_nifty50: bool
    is_watched: bool
    close: float | None
    change_pct: float | None
    price_direction: str    # bullish / bearish / neutral

    # Body — depends on the active horizon
    horizon: str
    prediction: CachedPrediction | None
    view: dict[str, Any] | None   # the full render-ready view dict; None if no cache

    @property
    def display_ticker(self) -> str:
        return self.ticker.removesuffix(".NS")


async def get_stock_detail(ticker: str, horizon: str = "weekly") -> StockDetail:
    """Build a StockDetail bundle for `ticker` at `horizon`.

    Cheap — does NOT trigger LLM calls. Cache miss simply leaves
    prediction/view as None, and the template renders a "Run prediction"
    CTA. Dashboard cache is queried without force_refresh to avoid
    blocking on a 5-10s NSE fetch.
    """
    t = _normalize(ticker)
    h = horizon.lower().strip()

    stock: Stock | None = get_by_ticker(t)
    name = stock.name if stock else t.removesuffix(".NS")
    sector = stock.sector if stock else "Unknown"
    is_n50 = stock.is_nifty50 if stock else False

    snapshot = await get_dashboard()
    price_row: DashboardRow | None = next(
        (r for r in snapshot.rows if r.ticker == t), None
    )

    cached = get_latest(t, h)
    view = cached.view if cached else None

    return StockDetail(
        ticker=t,
        name=name,
        sector=sector,
        is_nifty50=is_n50,
        is_watched=is_watched(t),
        close=price_row.close if price_row else None,
        change_pct=price_row.change_pct if price_row else None,
        price_direction=price_row.direction if price_row else "neutral",
        horizon=h,
        prediction=cached,
        view=view,
    )
