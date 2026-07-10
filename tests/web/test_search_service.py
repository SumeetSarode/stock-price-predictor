"""Smoke tests for search_service — in-memory autocomplete over the
bundled Nifty 500 CSV (frontend/data/nifty500.csv, ~197 rows)."""
from __future__ import annotations

from price_predictor.web.services import search_service as ss


class TestIndex:
    def test_index_loads(self):
        stats = ss.index_stats()
        assert stats["total"] > 100          # bundled CSV has ~197 rows
        assert stats["nifty50"] >= 45         # ~50 N50 members
        assert stats["total"] == stats["nifty50"] + stats["extras"]

    def test_all_nifty50_flagged(self):
        n50 = ss.all_nifty50()
        assert len(n50) >= 45
        assert all(s.is_nifty50 for s in n50)


class TestSearch:
    def test_empty_query_returns_empty(self):
        assert ss.search("") == []
        assert ss.search("   ") == []

    def test_exact_ticker_ranks_first(self):
        results = ss.search("RELIANCE")
        assert results
        assert results[0].ticker == "RELIANCE.NS"

    def test_prefix_match(self):
        results = ss.search("REL")
        tickers = [s.ticker for s in results]
        assert "RELIANCE.NS" in tickers

    def test_limit_respected(self):
        results = ss.search("a", limit=3)
        assert len(results) <= 3

    def test_case_insensitive(self):
        assert ss.search("reliance")[0].ticker == "RELIANCE.NS"

    def test_no_match_returns_empty(self):
        assert ss.search("zzzznotarealticker") == []


class TestGetByTicker:
    def test_found_with_and_without_suffix(self):
        a = ss.get_by_ticker("RELIANCE")
        b = ss.get_by_ticker("RELIANCE.NS")
        c = ss.get_by_ticker("reliance")
        assert a is not None
        assert a.ticker == b.ticker == c.ticker == "RELIANCE.NS"

    def test_not_found_returns_none(self):
        assert ss.get_by_ticker("NOTAREALTICKER") is None
