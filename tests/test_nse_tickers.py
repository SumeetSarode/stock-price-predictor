"""Tests for the NSE ticker alias resolver.

These pin the contract: known aliases redirect, unknowns stay None,
canonical tickers don't get redirected to themselves.
"""
from __future__ import annotations

import pytest

from price_predictor.data.nse_tickers import suggest_alternative


class TestSuggestAlternative:
    # ── Hot-path: the bug that motivated this whole module ─────────
    def test_hdfc_redirects_to_hdfcbank(self):
        """Regression: HDFC Ltd merged into HDFC Bank in 2023-07-01.
        HDFC.NS is delisted -- agent must learn to use HDFCBANK.NS."""
        assert suggest_alternative("HDFC") == "HDFCBANK"

    @pytest.mark.parametrize(
        "user_input",
        ["HDFC", "hdfc", "  HDFC  ", "HDFC.NS", "hdfc.ns", "Hdfc"],
    )
    def test_hdfc_normalization_handles_real_user_input(self, user_input):
        """Users type 'HDFC', 'hdfc', or 'HDFC.NS' interchangeably."""
        assert suggest_alternative(user_input) == "HDFCBANK"

    # ── Other documented aliases ───────────────────────────────────
    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("LARSEN", "LT"),
            ("LarsenToubro", "LT"),
            ("MAHINDRA", "M&M"),
            ("INFOSYS", "INFY"),
            ("TATAMOTOR", "TATAMOTORS"),
        ],
    )
    def test_known_aliases_redirect(self, alias, canonical):
        assert suggest_alternative(alias) == canonical

    # ── Negative space: don't suggest when no smart suggestion exists ─
    def test_unknown_ticker_returns_none(self):
        """Caller treats None as 'no opinion, use input as-is'."""
        assert suggest_alternative("ZZZZZ") is None

    def test_canonical_ticker_returns_none_not_self(self):
        """RELIANCE -> RELIANCE would be a useless redirect that could
        loop the agent. Return None to mean 'this is already canonical'."""
        assert suggest_alternative("RELIANCE") is None
        assert suggest_alternative("TCS") is None
        assert suggest_alternative("INFY") is None  # canonical, even if INFOSYS->INFY

    def test_empty_string_returns_none(self):
        assert suggest_alternative("") is None

    def test_canonical_in_table_no_self_redirect(self):
        """LT is in the alias table mapping to itself (so 'LT' input is
        a valid lookup). But suggest_alternative('LT') must NOT return 'LT'
        -- that's a no-op redirect."""
        assert suggest_alternative("LT") is None
