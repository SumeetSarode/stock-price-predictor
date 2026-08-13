"""Regression tests for panel_service.get_panel_cards().

WHY THIS FILE EXISTS
====================
get_panel_cards() had ZERO test coverage, and shipped broken: commit
f9f45bf ("show prices for stocks outside the Nifty 50") swapped the
price source from the N50-only dashboard snapshot to get_quotes(), but
in get_panel_cards() it DELETED the old lines without writing the
replacements:

    -    snapshot = await get_dashboard()
    -    price_lookup = {r.ticker: r for r in snapshot.rows}
    -    tickers = [e.ticker for e in entries]
    -    prediction_lookup = get_latest_many(tickers, horizon)
    +

...while the loop below still referenced both names. Any user with a
non-empty watchlist hit NameError: name 'price_lookup' is not defined
on the home page. The sibling function get_one_card() WAS fixed
correctly in the same commit, which is exactly why this slipped: the
bug lived in the one code path nothing executed.

The full suite stayed green at 1998 passed the entire time, because
`pytest -k panel` matched literally nothing. A green suite is not the
same as a working app -- these tests close that specific gap by
actually CALLING get_panel_cards() end to end.
"""
from __future__ import annotations

import pytest

from price_predictor.web.services import panel_service
from price_predictor.web.services.dashboard_service import DashboardRow
from price_predictor.web.services.panel_service import (
    PanelCard,
    get_panel_cards,
)


class _Entry:
    """Minimal stand-in for a watchlist entry (only .ticker is read)."""

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker


def _row(ticker: str, close: float, change_pct: float) -> DashboardRow:
    """A DashboardRow built via the real class, so field drift breaks here."""
    return DashboardRow(
        ticker=ticker,
        name=ticker.removesuffix(".NS"),
        sector="Test Sector",
        close=close,
        change_pct=change_pct,
    )


@pytest.fixture
def _watchlist(monkeypatch):
    """Two tickers: one Nifty-50, one deliberately NOT in the N50.

    The non-N50 name is the whole point of f9f45bf -- it must still get
    a price via get_quotes() rather than rendering close=None.
    """
    entries = [_Entry("RELIANCE.NS"), _Entry("SOMESMALLCAP.NS")]
    monkeypatch.setattr(panel_service, "list_watchlist", lambda: entries)
    return entries


class TestGetPanelCardsDefinedNames:
    """The NameError regression itself -- the home page must render."""

    @pytest.mark.asyncio
    async def test_returns_cards_without_nameerror(
        self, monkeypatch, _watchlist
    ):
        """THE regression: this raised NameError('price_lookup') before.

        Deliberately asserts nothing clever -- if the function body
        references an unbound name, we never get past the await.
        """
        async def _fake_get_quotes(tickers):
            return {t: _row(t, 100.0, 1.5) for t in tickers}

        monkeypatch.setattr(panel_service, "get_quotes", _fake_get_quotes)
        monkeypatch.setattr(panel_service, "get_latest_many", lambda *a: {})
        monkeypatch.setattr(panel_service, "get_by_ticker", lambda t: None)

        cards = await get_panel_cards()

        assert len(cards) == 2
        assert all(isinstance(c, PanelCard) for c in cards)

    @pytest.mark.asyncio
    async def test_prices_are_actually_populated(
        self, monkeypatch, _watchlist
    ):
        """price_lookup must be WIRED, not merely defined.

        Guards the lazy 'fix' of assigning an empty dict to silence the
        NameError -- that would leave every card priceless (close=None)
        and still look green here without this assertion.
        """
        async def _fake_get_quotes(tickers):
            return {
                "RELIANCE.NS": _row("RELIANCE.NS", 1234.5, 2.0),
                "SOMESMALLCAP.NS": _row("SOMESMALLCAP.NS", 42.0, -3.25),
            }

        monkeypatch.setattr(panel_service, "get_quotes", _fake_get_quotes)
        monkeypatch.setattr(panel_service, "get_latest_many", lambda *a: {})
        monkeypatch.setattr(panel_service, "get_by_ticker", lambda t: None)

        cards = {c.ticker: c for c in await get_panel_cards()}

        assert cards["RELIANCE.NS"].close == 1234.5
        assert cards["RELIANCE.NS"].change_pct == 2.0
        assert cards["RELIANCE.NS"].price_direction == "bullish"
        # The non-N50 name -- the exact case f9f45bf set out to fix.
        assert cards["SOMESMALLCAP.NS"].close == 42.0
        assert cards["SOMESMALLCAP.NS"].price_direction == "bearish"

    @pytest.mark.asyncio
    async def test_predictions_are_actually_wired(
        self, monkeypatch, _watchlist
    ):
        """prediction_lookup must reach the card too (same failure mode)."""
        sentinel = object()

        async def _fake_get_quotes(tickers):
            return {t: _row(t, 100.0, 0.0) for t in tickers}

        monkeypatch.setattr(panel_service, "get_quotes", _fake_get_quotes)
        monkeypatch.setattr(
            panel_service,
            "get_latest_many",
            lambda tickers, horizon: {"RELIANCE.NS": sentinel},
        )
        monkeypatch.setattr(panel_service, "get_by_ticker", lambda t: None)

        cards = {c.ticker: c for c in await get_panel_cards()}

        assert cards["RELIANCE.NS"].prediction is sentinel
        assert cards["SOMESMALLCAP.NS"].prediction is None

    @pytest.mark.asyncio
    async def test_horizon_is_forwarded_to_prediction_lookup(
        self, monkeypatch, _watchlist
    ):
        """The horizon arg must reach get_latest_many, not be dropped."""
        seen: dict[str, object] = {}

        async def _fake_get_quotes(tickers):
            return {t: _row(t, 1.0, 0.0) for t in tickers}

        def _spy(tickers, horizon):
            seen["tickers"] = list(tickers)
            seen["horizon"] = horizon
            return {}

        monkeypatch.setattr(panel_service, "get_quotes", _fake_get_quotes)
        monkeypatch.setattr(panel_service, "get_latest_many", _spy)
        monkeypatch.setattr(panel_service, "get_by_ticker", lambda t: None)

        await get_panel_cards(horizon="monthly")

        assert seen["horizon"] == "monthly"
        assert seen["tickers"] == ["RELIANCE.NS", "SOMESMALLCAP.NS"]

    @pytest.mark.asyncio
    async def test_missing_quote_degrades_gracefully(
        self, monkeypatch, _watchlist
    ):
        """get_quotes() omits unknown tickers -- cards must not explode.

        Its docstring says 'Missing tickers omitted', so .get() returning
        None is a REAL runtime state, not a hypothetical.
        """
        async def _fake_get_quotes(tickers):
            return {"RELIANCE.NS": _row("RELIANCE.NS", 10.0, 0.5)}

        monkeypatch.setattr(panel_service, "get_quotes", _fake_get_quotes)
        monkeypatch.setattr(panel_service, "get_latest_many", lambda *a: {})
        monkeypatch.setattr(panel_service, "get_by_ticker", lambda t: None)

        cards = {c.ticker: c for c in await get_panel_cards()}

        assert cards["SOMESMALLCAP.NS"].close is None
        assert cards["SOMESMALLCAP.NS"].change_pct is None
        assert cards["SOMESMALLCAP.NS"].price_direction == "neutral"

    @pytest.mark.asyncio
    async def test_empty_watchlist_short_circuits(self, monkeypatch):
        """No watchlist -> no cards, and NO price fetch at all."""
        called = False

        async def _boom(tickers):
            nonlocal called
            called = True
            return {}

        monkeypatch.setattr(panel_service, "list_watchlist", lambda: [])
        monkeypatch.setattr(panel_service, "get_quotes", _boom)

        assert await get_panel_cards() == []
        assert called is False
