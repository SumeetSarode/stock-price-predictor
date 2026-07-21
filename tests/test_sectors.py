"""Tests for the sector → news-query cheat-sheet (news_impact/sectors.py)."""
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
    def test_maps_sector_to_phrase(self, monkeypatch):
        monkeypatch.setattr(sectors.search_service, "get_by_ticker",
                            lambda t: _stock(sector="Technology"))
        assert sectors.sector_query_for("INFY.NS") == "Indian IT sector"

    def test_energy_phrase(self, monkeypatch):
        monkeypatch.setattr(sectors.search_service, "get_by_ticker",
                            lambda t: _stock(sector="Energy"))
        assert "energy" in sectors.sector_query_for("RELIANCE.NS").lower()

    def test_unknown_ticker_none(self, monkeypatch):
        monkeypatch.setattr(sectors.search_service, "get_by_ticker", lambda t: None)
        assert sectors.sector_query_for("WHO.NS") is None

    def test_unmapped_sector_none(self, monkeypatch):
        # An old NSE-style label we don't have a phrase for → None.
        monkeypatch.setattr(sectors.search_service, "get_by_ticker",
                            lambda t: _stock(sector="Diversified"))
        assert sectors.sector_query_for("X.NS") is None

    def test_every_mapped_sector_has_india_phrase(self):
        # Guard: all phrases mention India so GDELT returns local coverage.
        for sector, phrase in sectors.SECTOR_NEWS_QUERY.items():
            assert "indian" in phrase.lower(), sector
            assert len(phrase) > 8, sector

    def test_all_eleven_yfinance_sectors_mapped(self):
        # The 11 GICS-style sectors yfinance emits for NSE stocks.
        expected = {
            "Technology", "Financial Services", "Energy", "Healthcare",
            "Consumer Cyclical", "Consumer Defensive", "Basic Materials",
            "Industrials", "Real Estate", "Communication Services", "Utilities",
        }
        assert expected == set(sectors.SECTOR_NEWS_QUERY)
