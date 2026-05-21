"""Market summary service — derives top-level dashboard widgets from
the existing DashboardSnapshot. No new fetches; pure aggregation.

Powers:
  - Market index summary bar  (Nifty 50 ↑0.4% · 32 up / 18 down)
  - Top 5 gainers strip
  - Top 5 losers strip

Reads from the in-memory dashboard cache so the home page renders all
three widgets from a single fetch (no extra HTTP round-trips).
"""
from __future__ import annotations

from dataclasses import dataclass

from price_predictor.web.services.dashboard_service import (
    DashboardRow,
    DashboardSnapshot,
)


@dataclass(frozen=True, slots=True)
class MarketSummary:
    """Top-of-page market mood-ring.

    Index proxy: equal-weighted mean of all Nifty 50 daily % changes.
    Not the official index value (which is float-cap weighted), but
    a perfectly fine signal of breadth at a glance. We label it as
    'Nifty 50 (avg)' to avoid implying authority.
    """

    avg_change_pct: float | None
    n_advancing: int
    n_declining: int
    n_unchanged: int
    n_total: int
    n_errors: int

    @property
    def direction(self) -> str:
        if self.avg_change_pct is None:
            return "neutral"
        if self.avg_change_pct > 0.05:  return "bullish"
        if self.avg_change_pct < -0.05: return "bearish"
        return "neutral"

    @property
    def breadth_ratio(self) -> float | None:
        """advancers / (advancers + decliners). None when no live rows."""
        denom = self.n_advancing + self.n_declining
        return self.n_advancing / denom if denom else None


@dataclass(frozen=True, slots=True)
class MoversStrip:
    """Top-N gainers and losers, both newest-first."""

    gainers: tuple[DashboardRow, ...]
    losers: tuple[DashboardRow, ...]


def summarize_market(snapshot: DashboardSnapshot) -> MarketSummary:
    """Compute index-proxy + breadth counters from a snapshot.

    Excludes rows where the fetch errored out (close is None) — those
    would skew the average toward zero.
    """
    live: list[DashboardRow] = [r for r in snapshot.rows if r.change_pct is not None]
    errors = len(snapshot.rows) - len(live)

    if not live:
        return MarketSummary(
            avg_change_pct=None,
            n_advancing=0, n_declining=0, n_unchanged=0,
            n_total=len(snapshot.rows), n_errors=errors,
        )

    avg = sum(r.change_pct for r in live) / len(live)  # type: ignore[misc]
    up = sum(1 for r in live if (r.change_pct or 0) > 0.05)
    down = sum(1 for r in live if (r.change_pct or 0) < -0.05)
    flat = len(live) - up - down

    return MarketSummary(
        avg_change_pct=avg,
        n_advancing=up,
        n_declining=down,
        n_unchanged=flat,
        n_total=len(snapshot.rows),
        n_errors=errors,
    )


def get_movers(snapshot: DashboardSnapshot, *, top_n: int = 5) -> MoversStrip:
    """Return top-N gainers and losers from a snapshot.

    Sort key is `change_pct`. Rows with no change_pct (fetch errors)
    are excluded — they're noise here.
    """
    live = [r for r in snapshot.rows if r.change_pct is not None]
    if not live:
        return MoversStrip(gainers=(), losers=())

    by_pct = sorted(live, key=lambda r: r.change_pct or 0.0)
    losers = tuple(by_pct[:top_n])              # most negative first
    gainers = tuple(reversed(by_pct[-top_n:]))  # most positive first
    return MoversStrip(gainers=gainers, losers=losers)
