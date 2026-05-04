"""Tests for ResilientPriceFetcher -- the fallback/cooldown logic.

We use a `FakeProvider` instead of mocking YFinanceProvider, because the
resilient layer's behavior is independent of which providers it wraps.
A fake we control completely makes failure-mode tests trivial.
"""
from __future__ import annotations

from datetime import UTC, date, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from price_predictor.data.providers import (
    PriceFetchError,
    PriceProvider,
    ResilientPriceFetcher,
)
from price_predictor.data.providers.resilient import AllProvidersExhaustedError


# ─────────────────────────────────────────────────────────────────
# Fake provider -- programmable success/failure for tests
# ─────────────────────────────────────────────────────────────────
class FakeProvider(PriceProvider):
    """Provider whose behavior is dictated by constructor flags.

    Attributes:
        _name:        name to expose (lets tests build distinguishable chains)
        behavior:     'success' | 'fetch_error' | 'value_error' | 'unexpected'
        call_count:   how many times fetch_ohlcv was called (for assertions)
    """

    def __init__(self, name: str, behavior: str = "success") -> None:
        self._name = name
        self.behavior = behavior
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    def fetch_ohlcv(self, ticker, start, end, interval="1d"):
        self.call_count += 1
        if self.behavior == "success":
            return pd.DataFrame(
                {"open": [1.0], "high": [2.0], "low": [0.5],
                 "close": [1.5], "adj_close": [1.5], "volume": [100]},
                index=pd.DatetimeIndex(["2025-01-01"], tz="Asia/Kolkata"),
            )
        if self.behavior == "fetch_error":
            raise PriceFetchError(f"{self._name} simulated fetch error")
        if self.behavior == "value_error":
            raise ValueError(f"{self._name} simulated value error")
        if self.behavior == "unexpected":
            raise RuntimeError(f"{self._name} simulated unexpected error")
        raise AssertionError(f"unknown behavior {self.behavior!r}")


# ─────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────
class TestConstruction:
    def test_empty_providers_rejected(self):
        with pytest.raises(ValueError, match="at least one provider"):
            ResilientPriceFetcher(providers=[])

    def test_chain_property_lists_provider_names_in_order(self):
        f = ResilientPriceFetcher(
            providers=[FakeProvider("a"), FakeProvider("b"), FakeProvider("c")]
        )
        assert f.chain == ["a", "b", "c"]


# ─────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────
class TestHappyPath:
    def test_single_provider_success(self):
        p = FakeProvider("yfinance")
        f = ResilientPriceFetcher(providers=[p])
        df = f.fetch_ohlcv("RELIANCE.NS", date(2025, 1, 1), date(2025, 1, 2))
        assert not df.empty
        assert p.call_count == 1

    def test_first_provider_succeeds_second_not_called(self):
        p1 = FakeProvider("yfinance", behavior="success")
        p2 = FakeProvider("stooq", behavior="success")
        f = ResilientPriceFetcher(providers=[p1, p2])
        f.fetch_ohlcv("X", date(2025, 1, 1), date(2025, 1, 2))
        assert p1.call_count == 1
        assert p2.call_count == 0  # short-circuit


# ─────────────────────────────────────────────────────────────────
# Fallback on transient failure
# ─────────────────────────────────────────────────────────────────
class TestFallback:
    def test_fetch_error_falls_back_to_next_provider(self):
        p1 = FakeProvider("yfinance", behavior="fetch_error")
        p2 = FakeProvider("stooq", behavior="success")
        f = ResilientPriceFetcher(providers=[p1, p2])
        df = f.fetch_ohlcv("X", date(2025, 1, 1), date(2025, 1, 2))
        assert not df.empty
        assert p1.call_count == 1
        assert p2.call_count == 1

    def test_unexpected_error_falls_back_to_next_provider(self):
        """Even non-PriceFetchError exceptions trigger fallback (defensive)."""
        p1 = FakeProvider("yfinance", behavior="unexpected")
        p2 = FakeProvider("stooq", behavior="success")
        f = ResilientPriceFetcher(providers=[p1, p2])
        df = f.fetch_ohlcv("X", date(2025, 1, 1), date(2025, 1, 2))
        assert not df.empty
        assert p1.call_count == 1
        assert p2.call_count == 1

    def test_all_providers_fail_raises_exhausted(self):
        p1 = FakeProvider("yfinance", behavior="fetch_error")
        p2 = FakeProvider("stooq", behavior="fetch_error")
        f = ResilientPriceFetcher(providers=[p1, p2])
        with pytest.raises(AllProvidersExhaustedError) as exc:
            f.fetch_ohlcv("X", date(2025, 1, 1), date(2025, 1, 2))
        assert exc.value.chain == ["yfinance", "stooq"]
        assert "stooq simulated fetch error" in str(exc.value)

    def test_exhausted_error_is_subclass_of_price_fetch_error(self):
        """Callers can `except PriceFetchError` to catch both."""
        p = FakeProvider("yfinance", behavior="fetch_error")
        f = ResilientPriceFetcher(providers=[p])
        with pytest.raises(PriceFetchError):  # broader except still catches it
            f.fetch_ohlcv("X", date(2025, 1, 1), date(2025, 1, 2))


# ─────────────────────────────────────────────────────────────────
# ValueError -- caller bug, must NOT fall back
# ─────────────────────────────────────────────────────────────────
class TestValueErrorPropagation:
    def test_value_error_propagates_immediately(self):
        """Falling back would just hit the same ValueError -- waste of time."""
        p1 = FakeProvider("yfinance", behavior="value_error")
        p2 = FakeProvider("stooq", behavior="success")
        f = ResilientPriceFetcher(providers=[p1, p2])
        with pytest.raises(ValueError, match="simulated value error"):
            f.fetch_ohlcv("X", date(2025, 1, 1), date(2025, 1, 2))
        assert p1.call_count == 1
        assert p2.call_count == 0  # never reached

    def test_value_error_does_not_trigger_cooldown(self):
        """Caller bugs shouldn't penalize the provider."""
        p = FakeProvider("yfinance", behavior="value_error")
        f = ResilientPriceFetcher(providers=[p])
        with pytest.raises(ValueError):
            f.fetch_ohlcv("X", date(2025, 1, 1), date(2025, 1, 2))
        # No cooldown set -> provider is still available
        assert p.name not in f._cooled_until


# ─────────────────────────────────────────────────────────────────
# Cooldown behavior
# ─────────────────────────────────────────────────────────────────
class TestCooldown:
    def test_failed_provider_skipped_within_cooldown(self):
        """After a transient failure, subsequent calls skip that provider."""
        p1 = FakeProvider("yfinance", behavior="fetch_error")
        p2 = FakeProvider("stooq", behavior="success")
        f = ResilientPriceFetcher(providers=[p1, p2], cooldown=timedelta(seconds=60))

        # First call: p1 fails, p2 succeeds, p1 cooled down
        f.fetch_ohlcv("X", date(2025, 1, 1), date(2025, 1, 2))
        assert p1.call_count == 1

        # Second call: p1 still cooled, only p2 is tried
        f.fetch_ohlcv("Y", date(2025, 1, 1), date(2025, 1, 2))
        assert p1.call_count == 1  # NOT incremented
        assert p2.call_count == 2

    def test_cooldown_expires_and_provider_retried(self):
        """After cooldown elapses, provider is back in rotation."""
        p1 = FakeProvider("yfinance", behavior="fetch_error")
        p2 = FakeProvider("stooq", behavior="success")
        f = ResilientPriceFetcher(providers=[p1, p2], cooldown=timedelta(seconds=60))

        # Trip the cooldown
        f.fetch_ohlcv("X", date(2025, 1, 1), date(2025, 1, 2))
        assert p1.call_count == 1

        # Manually expire the cooldown by rewinding the cooled-until timestamp
        from datetime import datetime
        f._cooled_until[p1.name] = datetime.now(UTC) - timedelta(seconds=1)

        # Next call: p1 is tried again (still fails, but the point is it was tried)
        f.fetch_ohlcv("Y", date(2025, 1, 1), date(2025, 1, 2))
        assert p1.call_count == 2

    def test_all_providers_cooled_still_tries_them(self):
        """Last-resort behavior: if everything is cooled, try anyway."""
        p1 = FakeProvider("yfinance", behavior="fetch_error")
        p2 = FakeProvider("stooq", behavior="fetch_error")
        f = ResilientPriceFetcher(providers=[p1, p2])

        # First call: both fail, both get cooled
        with pytest.raises(AllProvidersExhaustedError):
            f.fetch_ohlcv("X", date(2025, 1, 1), date(2025, 1, 2))
        assert p1.call_count == 1
        assert p2.call_count == 1

        # Second call: both still cooled, but we try them as last resort
        with pytest.raises(AllProvidersExhaustedError):
            f.fetch_ohlcv("Y", date(2025, 1, 1), date(2025, 1, 2))
        assert p1.call_count == 2
        assert p2.call_count == 2


# ─────────────────────────────────────────────────────────────────
# Integration with the default fetcher in data/prices.py
# ─────────────────────────────────────────────────────────────────
class TestDefaultFetcherShim:
    """Sanity: data.prices.fetch_ohlcv delegates correctly."""

    def test_prices_module_fetch_ohlcv_uses_resilient_chain(self):
        """A successful yfinance mock at the bottom of the chain reaches the top."""
        from price_predictor.data.prices import fetch_ohlcv

        mock_df = pd.DataFrame(
            {("Open", "X"): [1.0], ("High", "X"): [2.0], ("Low", "X"): [0.5],
             ("Close", "X"): [1.5], ("Adj Close", "X"): [1.5],
             ("Volume", "X"): [100]},
            index=pd.DatetimeIndex(["2025-01-01"]),
        )
        with patch(
            "price_predictor.data.providers.yfinance_provider.yf.download"
        ) as mock_dl:
            mock_dl.return_value = mock_df
            df = fetch_ohlcv("X.NS", date(2025, 1, 1), date(2025, 1, 1))

        assert not df.empty
        assert "close" in df.columns  # snake_case rename happened
        assert df.index.tz is not None  # tz localization happened
