"""Tests for the deterministic gather layer (news_impact/gather.py).

All network fetchers are monkeypatched — no GDELT/NSE/yfinance calls.
"""
from __future__ import annotations

import types
from datetime import date

import pandas as pd
import pytest

from price_predictor.agents.news_impact import gather as g
from price_predictor.web.services.search_service import Stock


# ── fixtures / fakes ────────────────────────────────────────────────
def _news_df(titles):
    return pd.DataFrame([
        {"title": t, "url": f"http://x/{i}", "published_at": "2026-01-01",
         "source": "ET"}
        for i, t in enumerate(titles)
    ])


def _filings_df(subjects):
    return pd.DataFrame([
        {"kind": "announcement", "announced_at": "2026-01-01",
         "event_at": None, "event_type": "results", "subject": s}
        for s in subjects
    ])


def _fake_estimates(has_coverage=True):
    ns = types.SimpleNamespace
    return ns(
        has_coverage=has_coverage,
        earnings_estimates=[ns(period="+1q", avg=12.5)],
        revenue_estimates=[],
        recommendations=[],
        price_targets=ns(current=100.0, mean=120.0, high=140.0, low=90.0),
    )


@pytest.fixture(autouse=True)
def _stub_index(monkeypatch):
    """Default: ticker resolves to a Technology stock named 'Test Co'."""
    monkeypatch.setattr(
        g.search_service, "get_by_ticker",
        lambda t: Stock(ticker="INFY.NS", name="Infosys",
                        sector="Technology", is_nifty50=True),
    )


@pytest.fixture
def happy_fetchers(monkeypatch):
    """Wire all fetchers to succeed. Records calls for assertions.

    Company news flows through fetch_news_relevant (exact-phrase India
    ladder); sector news flows through fetch_news directly (loose, no
    country bias). We patch both and record them separately.
    """
    calls = {"news": [], "sector": [], "filings": [], "estimates": [],
             "prices": [], "snapshot": []}

    async def fake_news(query, start, end):
        calls["news"].append((query, start, end))
        return _news_df([f"{query} headline"])

    async def fake_sector_news(query, start, end, *, exact_phrase, source_country):
        calls["sector"].append((query, exact_phrase, source_country))
        return _news_df([f"{query} headline"])

    async def fake_filings(sym, start, end):
        calls["filings"].append((sym, start, end))
        return _filings_df(["Q3 results"])

    async def fake_estimates(tk):
        calls["estimates"].append(tk)
        return _fake_estimates()

    def fake_prices(tk, start, end, include_bars=False):
        calls["prices"].append((tk, start, end))
        return {"status": "success", "last_close": 100.0}

    monkeypatch.setattr(g, "fetch_news_relevant", fake_news)
    monkeypatch.setattr(g, "fetch_news", fake_sector_news)
    monkeypatch.setattr(g, "fetch_filings", fake_filings)
    monkeypatch.setattr(g, "fetch_estimates", fake_estimates)
    monkeypatch.setattr(g, "fetch_prices_tool", fake_prices)
    monkeypatch.setattr(g, "get_news_snapshot", lambda: None)
    return calls


# ── live-mode happy path ────────────────────────────────────────────
class TestHappyPath:
    @pytest.mark.asyncio
    async def test_all_sources_populated(self, happy_fetchers):
        out = await g.gather_news_impact_inputs("INFY.NS")
        assert out.ticker == "INFY.NS"
        assert out.company_name == "Infosys"
        assert out.sector == "Technology"
        assert out.company_news and out.sector_news
        assert out.filings and out.estimates and out.prices
        assert out.errors == []

    @pytest.mark.asyncio
    async def test_company_news_uses_company_name(self, happy_fetchers):
        await g.gather_news_impact_inputs("INFY.NS")
        queries = [c[0] for c in happy_fetchers["news"]]
        assert "Infosys" in queries  # not "INFY.NS"

    @pytest.mark.asyncio
    async def test_sector_news_uses_sector_phrase(self, happy_fetchers):
        await g.gather_news_impact_inputs("INFY.NS")
        sector_queries = [c[0] for c in happy_fetchers["sector"]]
        assert "Technology sector" in sector_queries

    @pytest.mark.asyncio
    async def test_sector_news_is_loose_and_unbiased(self, happy_fetchers):
        """Sector queries must be loose-token (exact_phrase=False) with no
        country bias so global coverage surfaces naturally."""
        await g.gather_news_impact_inputs("INFY.NS")
        _, exact_phrase, country = happy_fetchers["sector"][0]
        assert exact_phrase is False
        assert country is None

    @pytest.mark.asyncio
    async def test_filings_use_bare_symbol(self, happy_fetchers):
        await g.gather_news_impact_inputs("INFY.NS")
        assert happy_fetchers["filings"][0][0] == "INFY"

    @pytest.mark.asyncio
    async def test_estimates_and_prices_use_yf_ticker(self, happy_fetchers):
        await g.gather_news_impact_inputs("INFY.NS")
        assert happy_fetchers["estimates"] == ["INFY.NS"]
        assert happy_fetchers["prices"][0][0] == "INFY.NS"


# ── sector handling ─────────────────────────────────────────────────
class TestSectorHandling:
    @pytest.mark.asyncio
    async def test_no_sector_skips_sector_news(self, happy_fetchers, monkeypatch):
        monkeypatch.setattr(
            g.search_service, "get_by_ticker",
            lambda t: Stock(ticker="X.NS", name="X Ltd",
                            sector="NSE Listed", is_nifty50=False),
        )
        out = await g.gather_news_impact_inputs("X.NS")
        assert out.sector is None
        assert out.sector_news == []
        # company news fetched once; sector news never fetched
        assert len(happy_fetchers["news"]) == 1
        assert happy_fetchers["sector"] == []


# ── soft-fail isolation ─────────────────────────────────────────────
class TestSoftFail:
    @pytest.mark.asyncio
    async def test_one_source_failure_does_not_sink_others(self, happy_fetchers,
                                                           monkeypatch):
        async def boom_filings(sym, start, end):
            raise g.FilingsFetchError("NSE 403")

        monkeypatch.setattr(g, "fetch_filings", boom_filings)
        out = await g.gather_news_impact_inputs("INFY.NS")
        assert out.filings == []
        assert any("filings unavailable" in e for e in out.errors)
        # others still fine
        assert out.company_news and out.prices

    @pytest.mark.asyncio
    async def test_all_failures_recorded_not_raised(self, monkeypatch,
                                                    happy_fetchers):
        async def boom(*a, **k):
            raise RuntimeError("down")

        def boom_sync(*a, **k):
            raise RuntimeError("down")

        monkeypatch.setattr(g, "fetch_news_relevant", boom)
        monkeypatch.setattr(g, "fetch_news", boom)
        monkeypatch.setattr(g, "fetch_filings", boom)
        monkeypatch.setattr(g, "fetch_estimates", boom)
        monkeypatch.setattr(g, "fetch_prices_tool", boom_sync)
        out = await g.gather_news_impact_inputs("INFY.NS")
        assert out.company_news == [] and out.prices is None
        assert len(out.errors) >= 4


# ── look-ahead defense ──────────────────────────────────────────────
class TestLookAhead:
    @pytest.mark.asyncio
    async def test_estimates_skipped_in_replay(self, happy_fetchers):
        out = await g.gather_news_impact_inputs("INFY.NS", as_of=date(2025, 1, 1))
        assert out.estimates is None
        assert happy_fetchers["estimates"] == []  # never called

    @pytest.mark.asyncio
    async def test_uses_snapshot_in_replay(self, happy_fetchers, monkeypatch):
        snap_calls = []

        class FakeSnap:
            async def get_or_fetch(self, query, as_of, days, *, exact_phrase=True):
                snap_calls.append((query, as_of, days, exact_phrase))
                return _news_df([f"snap {query}"])

        monkeypatch.setattr(g, "get_news_snapshot", lambda: FakeSnap())
        out = await g.gather_news_impact_inputs("INFY.NS", as_of=date(2025, 1, 1))
        # snapshot used for BOTH company and sector news
        assert len(snap_calls) == 2
        # company = exact phrase; sector = loose
        exacts = {q: ep for q, _, _, ep in snap_calls}
        assert exacts["Infosys"] is True
        assert exacts["Technology sector"] is False
        # live fetch NOT called
        assert happy_fetchers["news"] == []
        assert happy_fetchers["sector"] == []
        assert out.company_news

    @pytest.mark.asyncio
    async def test_window_ends_at_as_of(self, happy_fetchers):
        await g.gather_news_impact_inputs("INFY.NS", as_of=date(2025, 6, 15))
        # filings window end == as_of
        assert happy_fetchers["filings"][0][2] == "2025-06-15"

    @pytest.mark.asyncio
    async def test_as_of_defaults_from_replay_context(self, happy_fetchers):
        from price_predictor.prediction.replay_context import replay_context
        with replay_context(date(2024, 3, 3)):
            out = await g.gather_news_impact_inputs("INFY.NS")
        # replay active → estimates skipped
        assert out.estimates is None
        assert out.window_end == "2024-03-03"

    @pytest.mark.asyncio
    async def test_live_mode_fetches_estimates(self, happy_fetchers):
        out = await g.gather_news_impact_inputs("INFY.NS")
        assert out.estimates is not None
        assert happy_fetchers["estimates"] == ["INFY.NS"]


# ── company-name fallback ───────────────────────────────────────────
class TestCompanyName:
    @pytest.mark.asyncio
    async def test_falls_back_to_bare_symbol(self, happy_fetchers, monkeypatch):
        monkeypatch.setattr(g.search_service, "get_by_ticker", lambda t: None)
        out = await g.gather_news_impact_inputs("WEIRD.NS")
        assert out.company_name == "WEIRD"


# ── has_news_evidence gate (LLM only when there's something to reason) ─
class TestHasNewsEvidence:
    def _blank(self):
        return g.NewsImpactInputs(
            ticker="X.NS", company_name="X", sector=None,
            window_start="2026-01-01", window_end="2026-01-08",
        )

    def test_empty_is_false(self):
        assert self._blank().has_news_evidence is False

    def test_prices_alone_is_false(self):
        inp = self._blank()
        inp.prices = {"status": "success", "last_close": 100.0}
        assert inp.has_news_evidence is False

    def test_company_news_is_true(self):
        inp = self._blank()
        inp.company_news = [{"title": "t"}]
        assert inp.has_news_evidence is True

    def test_sector_news_is_true(self):
        inp = self._blank()
        inp.sector_news = [{"title": "t"}]
        assert inp.has_news_evidence is True

    def test_filings_is_true(self):
        inp = self._blank()
        inp.filings = [{"subject": "results"}]
        assert inp.has_news_evidence is True

    def test_covered_estimates_is_true(self):
        inp = self._blank()
        inp.estimates = {"has_coverage": True}
        assert inp.has_news_evidence is True

    def test_uncovered_estimates_is_false(self):
        inp = self._blank()
        inp.estimates = {"has_coverage": False}
        assert inp.has_news_evidence is False
