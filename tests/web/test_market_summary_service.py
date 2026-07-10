"""Smoke tests for market_summary_service — pure aggregation over a
DashboardSnapshot (index-proxy, breadth counters, top movers)."""
from __future__ import annotations

from datetime import datetime, timezone

from price_predictor.web.services.dashboard_service import (
    DashboardRow,
    DashboardSnapshot,
)
from price_predictor.web.services.market_summary_service import (
    get_movers,
    summarize_market,
)


def _row(ticker, change_pct):
    return DashboardRow(
        ticker=ticker, name=ticker.removesuffix(".NS"), sector="X",
        close=100.0, prev_close=100.0, change_pct=change_pct,
    )


def _snapshot(rows):
    return DashboardSnapshot(
        rows=tuple(rows),
        fetched_at=datetime.now(timezone.utc),
        trading_day=None,
    )


class TestSummarizeMarket:
    def test_empty_snapshot(self):
        s = summarize_market(_snapshot([]))
        assert s.avg_change_pct is None
        assert s.direction == "neutral"
        assert s.breadth_ratio is None
        assert s.n_total == 0

    def test_all_errored_rows_excluded(self):
        # change_pct=None means the fetch failed for that ticker.
        s = summarize_market(_snapshot([_row("A.NS", None), _row("B.NS", None)]))
        assert s.avg_change_pct is None
        assert s.n_errors == 2
        assert s.n_total == 2

    def test_breadth_and_direction_bullish(self):
        rows = [_row("A.NS", 2.0), _row("B.NS", 1.0), _row("C.NS", -0.5)]
        s = summarize_market(_snapshot(rows))
        assert s.n_advancing == 2
        assert s.n_declining == 1
        assert s.direction == "bullish"
        assert s.avg_change_pct > 0
        assert abs(s.breadth_ratio - (2 / 3)) < 1e-9

    def test_direction_bearish(self):
        rows = [_row("A.NS", -2.0), _row("B.NS", -1.0)]
        s = summarize_market(_snapshot(rows))
        assert s.direction == "bearish"

    def test_flat_counted_as_unchanged(self):
        rows = [_row("A.NS", 0.0), _row("B.NS", 0.01)]  # both within ±0.05 band
        s = summarize_market(_snapshot(rows))
        assert s.n_unchanged == 2
        assert s.direction == "neutral"

    def test_mixed_live_and_errored(self):
        rows = [_row("A.NS", 1.0), _row("B.NS", None), _row("C.NS", -1.0)]
        s = summarize_market(_snapshot(rows))
        assert s.n_total == 3
        assert s.n_errors == 1
        assert s.n_advancing == 1 and s.n_declining == 1


class TestGetMovers:
    def test_empty(self):
        m = get_movers(_snapshot([]))
        assert m.gainers == () and m.losers == ()

    def test_top_gainers_and_losers_sorted(self):
        rows = [
            _row("A.NS", 5.0), _row("B.NS", 3.0), _row("C.NS", -1.0),
            _row("D.NS", -4.0), _row("E.NS", 0.5),
        ]
        m = get_movers(_snapshot(rows), top_n=2)
        # Gainers: most positive first.
        assert [r.ticker for r in m.gainers] == ["A.NS", "B.NS"]
        # Losers: most negative first.
        assert [r.ticker for r in m.losers] == ["D.NS", "C.NS"]

    def test_errored_rows_excluded_from_movers(self):
        rows = [_row("A.NS", 2.0), _row("B.NS", None)]
        m = get_movers(_snapshot(rows), top_n=5)
        tickers = {r.ticker for r in m.gainers} | {r.ticker for r in m.losers}
        assert "B.NS" not in tickers
