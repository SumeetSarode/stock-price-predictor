"""Tests for data.vix -- the India VIX fetcher (provider mocked)."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from price_predictor.data import vix as vix_mod
from price_predictor.data.providers import PriceFetchError


def _ohlcv(closes) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "open": closes, "high": closes, "low": closes,
            "close": closes, "adj_close": closes, "volume": [0] * n,
        },
        index=idx,
    )


class TestFetchIndiaVix:
    def test_returns_close_series(self, monkeypatch):
        captured = {}

        def _fake_fetch(ticker, start, end, interval):
            captured["ticker"] = ticker
            captured["interval"] = interval
            return _ohlcv([15.0, 16.0, 17.0])

        monkeypatch.setattr(vix_mod._vix_provider, "fetch_ohlcv", _fake_fetch)

        s = vix_mod.fetch_india_vix(date(2025, 1, 1), date(2025, 1, 3))
        assert list(s) == [15.0, 16.0, 17.0]
        assert s.name == "india_vix"
        # Uses the index symbol untouched (no .NS suffix).
        assert captured["ticker"] == "^INDIAVIX"
        assert captured["interval"] == "1d"

    def test_empty_df_returns_empty_series(self, monkeypatch):
        monkeypatch.setattr(
            vix_mod._vix_provider, "fetch_ohlcv",
            lambda *a, **k: pd.DataFrame(),
        )
        s = vix_mod.fetch_india_vix(date(2025, 1, 1), date(2025, 1, 3))
        assert s.empty
        assert s.name == "india_vix"

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError):
            vix_mod.fetch_india_vix(date(2025, 1, 5), date(2025, 1, 1))

    def test_provider_error_propagates(self, monkeypatch):
        def _boom(*a, **k):
            raise PriceFetchError("yfinance down")

        monkeypatch.setattr(vix_mod._vix_provider, "fetch_ohlcv", _boom)
        with pytest.raises(PriceFetchError):
            vix_mod.fetch_india_vix(date(2025, 1, 1), date(2025, 1, 3))

    def test_provider_is_suffix_free(self):
        # The dedicated provider must NOT append a market suffix, or
        # "^INDIAVIX" would become "^INDIAVIX.NS".
        assert vix_mod._vix_provider._default_market is None
