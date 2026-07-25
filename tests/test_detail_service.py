"""Regression tests for the stock detail page price display.

Directly answers 'will the details page show prices for all 2400+
stocks?': the search index holds ~2364 names (50 N50 + ~2314 non-N50),
and get_stock_detail() must surface a price for the non-N50 ones -- which
previously blanked because detail read the Nifty-50-only snapshot.
"""
from __future__ import annotations

import asyncio
from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from price_predictor.web.services import detail_service as det
from price_predictor.web.services.dashboard_service import DashboardRow
from price_predictor.web.services.detail_service import get_stock_detail
from price_predictor.web.services.search_service import Stock


def _row(ticker: str, close: float) -> DashboardRow:
    return DashboardRow(
        ticker=ticker, name="x", sector="y",
        close=close, prev_close=close - 5, change_pct=1.2,
        last_trading_day=date(2026, 7, 10),
    )


class TestDetailPageShowsPriceForNonNifty50:
    def test_non_nifty50_stock_shows_price(self):
        """The detail page must show a price for a non-N50 stock."""
        stock = Stock(
            ticker="DIXON.NS", name="Dixon Technologies",
            sector="Consumer Durables", is_nifty50=False,
        )
        with patch.object(det, "get_by_ticker", return_value=stock), \
             patch.object(det, "get_quote", new=AsyncMock(
                 return_value=_row("DIXON.NS", 15000.0))), \
             patch.object(det, "get_latest", return_value=None), \
             patch.object(det, "is_watched", return_value=False):
            detail = asyncio.run(get_stock_detail("DIXON.NS", "weekly"))

        assert detail.ticker == "DIXON.NS"
        assert detail.name == "Dixon Technologies"
        assert detail.is_nifty50 is False
        assert detail.close == 15000.0          # <-- the fix: price present
        assert detail.change_pct == 1.2
        assert detail.last_trading_day == date(2026, 7, 10)

    def test_price_fetch_failure_degrades_gracefully(self):
        """If the on-demand fetch yields nothing, header still renders."""
        stock = Stock(
            ticker="SUZLON.NS", name="Suzlon Energy",
            sector="Electrical Equipment", is_nifty50=False,
        )
        with patch.object(det, "get_by_ticker", return_value=stock), \
             patch.object(det, "get_quote", new=AsyncMock(return_value=None)), \
             patch.object(det, "get_latest", return_value=None), \
             patch.object(det, "is_watched", return_value=False):
            detail = asyncio.run(get_stock_detail("SUZLON.NS", "weekly"))

        assert detail.name == "Suzlon Energy"   # metadata still there
        assert detail.close is None             # em-dash in UI, no crash
        assert detail.price_direction == "neutral"
