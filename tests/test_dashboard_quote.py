"""Regression tests for single-ticker quotes (get_quote / get_quotes).

Bug: stocks OUTSIDE the Nifty 50 showed no price. The panel and detail
pages read prices from the Nifty-50-only dashboard snapshot, so any of
the ~450 non-N50 names in nifty500.csv had close=None. get_quote() fixes
this by piggybacking the snapshot for N50 and fetching non-N50 tickers on
demand. These tests pin that contract.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from unittest.mock import patch

import pytest

from price_predictor.web.services import dashboard_service as ds
from price_predictor.web.services.dashboard_service import (
    DashboardRow,
    DashboardSnapshot,
    get_quote,
    get_quotes,
    reset_cache_for_tests,
)
from price_predictor.web.services.search_service import Stock

_IST = timezone.utc  # tz value is irrelevant to these tests


@pytest.fixture(autouse=True)
def _clean_caches():
    reset_cache_for_tests()
    yield
    reset_cache_for_tests()


def _non_n50_stock() -> Stock:
    return Stock(
        ticker="DIXON.NS",
        name="Dixon Technologies",
        sector="Consumer Durables",
        is_nifty50=False,  # the whole point: NOT in the Nifty 50
    )


def _row(ticker: str, close: float) -> DashboardRow:
    return DashboardRow(
        ticker=ticker, name="x", sector="y",
        close=close, prev_close=close - 1, change_pct=1.0,
        last_trading_day=date(2026, 7, 10),
    )


class TestGetQuote:
    def test_non_nifty50_ticker_returns_price(self):
        """The core regression: a non-N50 stock DOES get a price now."""
        stock = _non_n50_stock()
        with patch.object(ds, "get_by_ticker", return_value=stock), \
             patch.object(ds, "_fetch_one_sync",
                          return_value=_row("DIXON.NS", 15000.0)) as fetch:
            row = asyncio.run(get_quote("DIXON.NS"))

        assert row is not None
        assert row.ticker == "DIXON.NS"
        assert row.close == 15000.0
        fetch.assert_called_once()  # actually fetched (not in N50 snapshot)

    def test_snapshot_fast_path_avoids_fetch(self):
        """If the ticker is already in the N50 snapshot, no extra fetch."""
        ds._cache = DashboardSnapshot(
            rows=(_row("RELIANCE.NS", 3000.0),),
            fetched_at=datetime.now(_IST),
            trading_day=date(2026, 7, 10),
        )
        with patch.object(ds, "_fetch_one_sync") as fetch:
            row = asyncio.run(get_quote("RELIANCE.NS"))

        assert row is not None and row.close == 3000.0
        fetch.assert_not_called()  # served free from the snapshot

    def test_per_ticker_cache_hits_second_call(self):
        """Second call for the same non-N50 ticker uses the cache."""
        stock = _non_n50_stock()
        with patch.object(ds, "get_by_ticker", return_value=stock), \
             patch.object(ds, "_fetch_one_sync",
                          return_value=_row("DIXON.NS", 15000.0)) as fetch:
            asyncio.run(get_quote("DIXON.NS"))
            asyncio.run(get_quote("DIXON.NS"))

        fetch.assert_called_once()  # only ONE fetch across two calls

    def test_unknown_ticker_returns_none(self):
        """No Stock metadata -> None (caller renders em-dash, no crash)."""
        with patch.object(ds, "get_by_ticker", return_value=None):
            row = asyncio.run(get_quote("NOTAREAL.NS"))
        assert row is None


class TestGetQuotes:
    def test_bulk_mixes_n50_and_non_n50(self):
        """N50 comes free from snapshot; non-N50 is fetched on demand."""
        ds._cache = DashboardSnapshot(
            rows=(_row("RELIANCE.NS", 3000.0),),
            fetched_at=datetime.now(_IST),
            trading_day=date(2026, 7, 10),
        )
        stock = _non_n50_stock()
        with patch.object(ds, "get_by_ticker", return_value=stock), \
             patch.object(ds, "_fetch_one_sync",
                          return_value=_row("DIXON.NS", 15000.0)) as fetch:
            out = asyncio.run(get_quotes(["RELIANCE.NS", "DIXON.NS"]))

        assert set(out) == {"RELIANCE.NS", "DIXON.NS"}
        assert out["RELIANCE.NS"].close == 3000.0  # from snapshot
        assert out["DIXON.NS"].close == 15000.0     # fetched
        fetch.assert_called_once()  # only the non-N50 one triggered a fetch
