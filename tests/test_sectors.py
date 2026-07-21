"""Tests for the sector news query helper (news_impact/sectors.py)."""
from __future__ import annotations

from price_predictor.agents.news_impact import sectors
from price_predictor.web.services.search_service import Stock


def _stock(ticker="RELIANCE.NS", sector="Energy"):
    return Stock(ticker=ticker, name="Test Co", sector=sector, is_nifty50=True)


class TestSectorFor:
    def test_returns_yfinance_sector(self, monkeypatch):
        monkeypatch.setattr(sectors.search_service, "get_by_ticker",
                            lambda t: _stock(sector="Technology"))
        assert sectors.sector_for("INFY.NS") == "Technology"

    def test_unknown_ticker_returns_none(self, monkeypatch):
        monkeypatch.setattr(sectors.search_service, "get_by_ticker", lambda t: None)
        assert sectors.sector_for("WHO.NS") is None

    def test_nse_listed_fallback_returns_none(self, monkeypatch):
        monkeypatch.setattr(sectors.search_service, "get_by_ticker",
                            lambda t: _stock(sector="NSE Listed"))
        assert sectors.sector_for("SMALL.NS") is None

    def test_blank_sector_returns_none(self, monkeypatch):
        monkeypatch.setattr(sectors.search_service, "get_by_ticker",
                            lambda t: _stock(sector="   "))
        assert sectors.sector_for("X.NS") is None


class TestSectorQueryFor:
    def test_builds_query_from_label(self, monkeypatch):
        monkeypatch.setattr(sectors.search_service, "get_by_ticker",
                            lambda t: _stock(sector="Technology"))
        assert sectors.sector_query_for("INFY.NS") == "Technology sector"

    def test_no_geographic_qualifier(self, monkeypatch):
        """Deliberately geography-neutral: no 'Indian' / 'global' words —
        the fetch runs with no country bias so global coverage surfaces
        naturally."""
        monkeypatch.setattr(sectors.search_service, "get_by_ticker",
                            lambda t: _stock(sector="Basic Materials"))
        q = sectors.sector_query_for("TATASTEEL.NS")
        assert q == "Basic Materials sector"
        assert "indian" not in q.lower()
        assert "global" not in q.lower()

    def test_unknown_ticker_none(self, monkeypatch):
        monkeypatch.setattr(sectors.search_service, "get_by_ticker", lambda t: None)
        assert sectors.sector_query_for("WHO.NS") is None

    def test_nse_listed_none(self, monkeypatch):
        monkeypatch.setattr(sectors.search_service, "get_by_ticker",
                            lambda t: _stock(sector="NSE Listed"))
        assert sectors.sector_query_for("SMALL.NS") is None
