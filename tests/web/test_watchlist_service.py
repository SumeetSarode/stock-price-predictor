"""Smoke tests for watchlist_service — CRUD over the SQLite watchlist.

All tests use the `tmp_db` fixture (isolated temp DB per test).
"""
from __future__ import annotations

import pytest

from price_predictor.web.services import watchlist_service as ws


class TestWatchlistCrud:
    def test_empty_by_default(self, tmp_db):
        assert ws.list_watchlist() == []
        assert ws.watchlist_tickers() == set()
        assert ws.count_watchlist() == 0

    def test_add_normalizes_ticker(self, tmp_db):
        entry = ws.add("reliance")
        assert entry.ticker == "RELIANCE.NS"
        assert ws.is_watched("RELIANCE") is True
        assert ws.is_watched("reliance.ns") is True

    def test_add_is_idempotent(self, tmp_db):
        first = ws.add("TCS")
        second = ws.add("tcs.ns")
        assert ws.count_watchlist() == 1
        # Re-add returns the existing entry (same added_at), never raises.
        assert first.ticker == second.ticker == "TCS.NS"

    def test_remove(self, tmp_db):
        ws.add("INFY")
        assert ws.remove("infy") is True
        assert ws.is_watched("INFY") is False
        # Removing a non-existent ticker returns False.
        assert ws.remove("INFY") is False

    def test_toggle(self, tmp_db):
        now_watched, was_full = ws.toggle("WIPRO")
        assert now_watched is True and was_full is False
        now_watched, was_full = ws.toggle("WIPRO")
        assert now_watched is False and was_full is False

    def test_full_cap_raises(self, tmp_db):
        for i in range(ws.MAX_WATCHLIST_SIZE):
            ws.add(f"STOCK{i}")
        assert ws.count_watchlist() == ws.MAX_WATCHLIST_SIZE
        with pytest.raises(ws.WatchlistFullError):
            ws.add("ONEMORE")

    def test_toggle_reports_full_attempt(self, tmp_db):
        for i in range(ws.MAX_WATCHLIST_SIZE):
            ws.add(f"STOCK{i}")
        now_watched, was_full = ws.toggle("OVERFLOW")
        assert now_watched is False and was_full is True

    def test_list_ordered_newest_first(self, tmp_db):
        ws.add("AAA")
        ws.add("BBB")
        tickers = [e.ticker for e in ws.list_watchlist()]
        # Newest-added first (BBB added after AAA).
        assert tickers.index("BBB.NS") < tickers.index("AAA.NS")
