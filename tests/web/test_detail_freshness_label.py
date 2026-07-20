"""Tests for the 'as of <date>' price freshness label on the stock detail.

Off-VPN the app uses yfinance, whose EOD close can lag a session or two.
The header used to hardcode 'today's close', which is a lie when the price
is actually from a previous session. `StockDetail.close_as_of` surfaces the
real trading day so the lag is obvious at a glance.
"""
from __future__ import annotations

from datetime import date

from price_predictor.web.services.detail_service import StockDetail


def _detail(last_trading_day):
    return StockDetail(
        ticker="RELIANCE.NS", name="Reliance", sector="Oil & Gas",
        is_nifty50=True, is_watched=False,
        close=1327.20, change_pct=2.36, price_direction="bullish",
        last_trading_day=last_trading_day,
        horizon="weekly", prediction=None, view=None,
    )


def test_close_as_of_formats_human_date():
    assert _detail(date(2026, 7, 17)).close_as_of == "17 Jul 2026"


def test_close_as_of_strips_leading_zero_day():
    # '07 Jul' -> '7 Jul' reads naturally, no zero-padded day.
    assert _detail(date(2026, 7, 7)).close_as_of == "7 Jul 2026"


def test_close_as_of_is_none_without_trading_day():
    # Price fetch failed -> no date -> template falls back to 'latest close'.
    assert _detail(None).close_as_of is None
