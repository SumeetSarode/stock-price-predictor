"""Tests for kb.stocks -- stock registry + lookup.

What we're pinning here:
- The registry loads from data/kb/stocks.json correctly.
- Lookup resolves ticker, exact name, substring, fuzzy in that order.
- The HDFC merger problem solves itself organically (no hardcoded alias).
- Common-typos recover via fuzzy matching.
- Garbage input + empty input cleanly return None.
- by_index() filters to a single index.
- Stock.yfinance_symbol appends `.NS`.

We DON'T mock the data file -- the registry committed to the repo IS
test fodder. If the registry shape changes, these tests should catch it.
"""
from __future__ import annotations

import pytest

from price_predictor.kb.stocks import (
    Stock,
    all_stocks,
    by_index,
    lookup,
)


# ─────────────────────────────────────────────────────────────────
# Stock model
# ─────────────────────────────────────────────────────────────────
class TestStockModel:
    def test_yfinance_symbol_appends_ns(self):
        s = Stock(ticker="RELIANCE", company_name="Reliance Industries")
        assert s.yfinance_symbol == "RELIANCE.NS"

    def test_indices_default_empty(self):
        s = Stock(ticker="X", company_name="Y")
        assert s.indices == []

    def test_ticker_required_nonempty(self):
        with pytest.raises(ValueError):
            Stock(ticker="", company_name="X")

    def test_company_name_required_nonempty(self):
        with pytest.raises(ValueError):
            Stock(ticker="X", company_name="")


# ─────────────────────────────────────────────────────────────────
# Registry loading
# ─────────────────────────────────────────────────────────────────
class TestRegistry:
    def test_all_stocks_returns_50_for_nifty50(self):
        """The committed registry should have all 50 Nifty constituents."""
        stocks = all_stocks()
        assert len(stocks) == 50

    def test_all_stocks_is_immutable(self):
        """Returned tuple prevents callers from corrupting the cache."""
        assert isinstance(all_stocks(), tuple)

    def test_every_stock_has_nifty50_membership(self):
        """For v1, every stock in the registry is in NIFTY50.
        Once we add another index this expectation widens, but the
        invariant 'every stock belongs to >=1 index' must hold forever."""
        for s in all_stocks():
            assert s.indices, f"{s.ticker} has no index membership"

    def test_well_known_constituents_present(self):
        """Spot-check tickers we KNOW are in Nifty50 (won't churn weekly)."""
        tickers = {s.ticker for s in all_stocks()}
        for expected in ["RELIANCE", "HDFCBANK", "INFY", "TCS", "LT"]:
            assert expected in tickers, f"{expected} missing from registry"


# ─────────────────────────────────────────────────────────────────
# by_index
# ─────────────────────────────────────────────────────────────────
class TestByIndex:
    def test_nifty50_returns_all_50(self):
        assert len(by_index("NIFTY50")) == 50

    def test_case_insensitive(self):
        assert len(by_index("nifty50")) == 50
        assert len(by_index("Nifty50")) == 50

    def test_unknown_index_returns_empty(self):
        """Caller decides if empty == error. We don't raise -- predictable."""
        assert by_index("BANKNIFTY") == []  # not yet bootstrapped


# ─────────────────────────────────────────────────────────────────
# lookup -- the meat of the module
# ─────────────────────────────────────────────────────────────────
class TestLookupExactTicker:
    """Tier 1: exact ticker match wins immediately."""

    @pytest.mark.parametrize("query,expected_ticker", [
        ("RELIANCE", "RELIANCE"),
        ("reliance", "RELIANCE"),       # case-insensitive
        ("RELIANCE.NS", "RELIANCE"),    # yfinance suffix tolerated
        ("reliance.ns", "RELIANCE"),    # both
        ("HDFCBANK", "HDFCBANK"),
        ("INFY", "INFY"),
        ("TCS", "TCS"),
        ("LT", "LT"),                   # short ticker
    ])
    def test_exact_ticker_resolves(self, query, expected_ticker):
        result = lookup(query)
        assert result is not None
        assert result.ticker == expected_ticker


class TestLookupExactName:
    """Tier 2: exact company name match (case-insensitive)."""

    @pytest.mark.parametrize("query,expected_ticker", [
        ("Infosys", "INFY"),
        ("infosys", "INFY"),
        ("Reliance Industries", "RELIANCE"),
        ("HDFC Bank", "HDFCBANK"),
    ])
    def test_exact_name_resolves(self, query, expected_ticker):
        result = lookup(query)
        assert result is not None
        assert result.ticker == expected_ticker


class TestLookupSubstring:
    """Tier 3: substring on company name. Shortest name wins (most specific)."""

    def test_hdfc_resolves_to_hdfc_bank_not_hdfc_life(self):
        """The merger problem, solved without a hardcoded alias.
        'HDFC' substring-matches 'HDFC Bank' AND 'HDFC Life'; we prefer
        the SHORTER name (more specific match)."""
        result = lookup("HDFC")
        assert result is not None
        assert result.ticker == "HDFCBANK"

    def test_larsen_resolves_to_lt(self):
        result = lookup("Larsen")
        assert result is not None
        assert result.ticker == "LT"

    def test_lt_special_chars_preserved(self):
        """'L&T' must keep its ampersand to match 'Larsen & Toubro'."""
        result = lookup("L&T")
        assert result is not None
        # Either substring match on '&' or fuzzy will land on LT.
        assert result.ticker == "LT"


class TestLookupFuzzyTicker:
    """Tier 4: typo recovery on tickers via SequenceMatcher."""

    @pytest.mark.parametrize("typo,expected", [
        ("RELIENCE", "RELIANCE"),       # transposition
        ("RELAINCE", "RELIANCE"),
        ("INFOSY", "INFY"),             # close to name 'Infosys' too
    ])
    def test_typo_recovers(self, typo, expected):
        result = lookup(typo)
        assert result is not None
        assert result.ticker == expected


class TestLookupRejection:
    """Garbage in, None out -- don't invent matches."""

    @pytest.mark.parametrize("garbage", [
        "",
        "   ",
        "asdfgh",
        "QWERTY99",
        "!!!!",
    ])
    def test_garbage_returns_none(self, garbage):
        assert lookup(garbage) is None


class TestLookupWithIndexFilter:
    """Optional `index` arg restricts the search pool."""

    def test_known_index_filter_works(self):
        """RELIANCE is in NIFTY50, so filtering to NIFTY50 still finds it."""
        result = lookup("RELIANCE", index="NIFTY50")
        assert result is not None
        assert result.ticker == "RELIANCE"

    def test_unknown_index_means_empty_pool_means_none(self):
        """No stocks in BANKNIFTY (yet) -> nothing to match against -> None."""
        result = lookup("HDFCBANK", index="BANKNIFTY")
        assert result is None
