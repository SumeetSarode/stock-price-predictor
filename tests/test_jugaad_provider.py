"""Unit tests for data/providers/jugaad_provider.py.

We never hit the real NSE network: the provider's `stock_df_fn` constructor
arg is the injection seam. Tests pass either a fake function or a mocker
to validate every branch of the wrapper.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from price_predictor.data.providers.base import PriceFetchError
from price_predictor.data.providers.jugaad_provider import (
    JugaadDataProvider,
    _normalise_symbol,
)


# ─────────────────────────────────────────────────────────────
# Symbol normalisation
# ─────────────────────────────────────────────────────────────
class TestNormaliseSymbol:
    @pytest.mark.parametrize("raw,expected", [
        ("RELIANCE", "RELIANCE"),
        ("RELIANCE.NS", "RELIANCE"),
        ("RELIANCE.BO", "RELIANCE"),
        ("RELIANCE.NSE", "RELIANCE"),
        ("RELIANCE.BSE", "RELIANCE"),
        ("reliance.ns", "reliance"),    # case-insensitive suffix
        ("RELIANCE.Ns", "RELIANCE"),
        ("HDFCBANK", "HDFCBANK"),
        ("  RELIANCE.NS  ", "RELIANCE"),  # whitespace trimmed
    ])
    def test_strip_suffixes(self, raw, expected):
        assert _normalise_symbol(raw) == expected

    def test_no_change_for_bare_symbol(self):
        assert _normalise_symbol("INFY") == "INFY"

    def test_does_not_strip_mid_string(self):
        # Defensive: ".NS" inside the symbol body should NOT be stripped.
        # (Hypothetical; no real NSE symbol has this, but the regex must
        # be anchored to end-of-string.)
        assert _normalise_symbol("FOO.NSBAR") == "FOO.NSBAR"


# ─────────────────────────────────────────────────────────────
# Helpers — build fake jugaad-data responses
# ─────────────────────────────────────────────────────────────
def _fake_raw(rows: int = 3, *, symbol: str = "RELIANCE") -> pd.DataFrame:
    """Build a DataFrame matching jugaad-data's stock_df shape."""
    dates = pd.date_range("2024-04-01", periods=rows, freq="D")
    return pd.DataFrame({
        "DATE": dates,
        "SERIES": ["EQ"] * rows,
        "OPEN": [100.0 + i for i in range(rows)],
        "HIGH": [102.0 + i for i in range(rows)],
        "LOW": [99.0 + i for i in range(rows)],
        "PREV. CLOSE": [99.5 + i for i in range(rows)],
        "LTP": [101.0 + i for i in range(rows)],
        "CLOSE": [101.0 + i for i in range(rows)],
        "VWAP": [100.5 + i for i in range(rows)],
        "VOLUME": [1_000_000 + i * 1000 for i in range(rows)],
        "VALUE": [1.0e8 + i * 1e6 for i in range(rows)],
        "NO OF TRADES": [10000 + i for i in range(rows)],
        "DELIVERY QTY": [500_000 + i * 100 for i in range(rows)],
        "DELIVERY %": [50.0] * rows,
        "SYMBOL": [symbol] * rows,
    })


def _fake_stock_df(return_value=None, raises: Exception | None = None):
    """Factory for a fake stock_df callable used as the injection seam."""
    captured: dict = {}

    def _impl(symbol, from_date, to_date, series="EQ"):
        captured["symbol"] = symbol
        captured["from_date"] = from_date
        captured["to_date"] = to_date
        captured["series"] = series
        if raises is not None:
            raise raises
        return return_value if return_value is not None else _fake_raw()

    _impl.captured = captured  # type: ignore[attr-defined]
    return _impl


# ─────────────────────────────────────────────────────────────
# Basic behavior
# ─────────────────────────────────────────────────────────────
class TestProviderBasics:
    def test_name(self):
        assert JugaadDataProvider().name == "jugaad-data"

    def test_default_series_eq(self):
        fake = _fake_stock_df()
        prov = JugaadDataProvider(stock_df_fn=fake)
        prov.fetch_ohlcv("RELIANCE", date(2024, 4, 1), date(2024, 4, 3))
        assert fake.captured["series"] == "EQ"  # type: ignore[attr-defined]

    def test_custom_series_passed_through(self):
        fake = _fake_stock_df()
        prov = JugaadDataProvider(series="BE", stock_df_fn=fake)
        prov.fetch_ohlcv("RELIANCE", date(2024, 4, 1), date(2024, 4, 3))
        assert fake.captured["series"] == "BE"  # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────
class TestInputValidation:
    def setup_method(self):
        self.prov = JugaadDataProvider(stock_df_fn=_fake_stock_df())

    @pytest.mark.parametrize("bad", ["", "   ", None])
    def test_empty_ticker_raises_valueerror(self, bad):
        with pytest.raises(ValueError, match="non-empty"):
            self.prov.fetch_ohlcv(bad, date(2024, 4, 1), date(2024, 4, 3))  # type: ignore[arg-type]

    def test_inverted_dates_raise_valueerror(self):
        with pytest.raises(ValueError, match="must be <="):
            self.prov.fetch_ohlcv("RELIANCE", date(2024, 4, 3), date(2024, 4, 1))

    def test_unsupported_interval_raises_valueerror(self):
        with pytest.raises(ValueError, match="only supports interval='1d'"):
            self.prov.fetch_ohlcv(
                "RELIANCE", date(2024, 4, 1), date(2024, 4, 3), interval="1h",
            )

    def test_suffix_only_ticker_raises_valueerror(self):
        # ".NS" alone strips to "" — must reject explicitly, not silently
        # call jugaad with empty symbol.
        with pytest.raises(ValueError, match="empty string after stripping"):
            self.prov.fetch_ohlcv(".NS", date(2024, 4, 1), date(2024, 4, 3))


# ─────────────────────────────────────────────────────────────
# Suffix normalisation reaches the upstream call
# ─────────────────────────────────────────────────────────────
class TestSymbolPassthrough:
    def test_yfinance_style_suffix_stripped_before_upstream(self):
        fake = _fake_stock_df()
        prov = JugaadDataProvider(stock_df_fn=fake)
        prov.fetch_ohlcv("RELIANCE.NS", date(2024, 4, 1), date(2024, 4, 3))
        assert fake.captured["symbol"] == "RELIANCE"  # type: ignore[attr-defined]

    def test_bare_symbol_passed_unchanged(self):
        fake = _fake_stock_df()
        prov = JugaadDataProvider(stock_df_fn=fake)
        prov.fetch_ohlcv("INFY", date(2024, 4, 1), date(2024, 4, 3))
        assert fake.captured["symbol"] == "INFY"  # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────
# DataFrame contract
# ─────────────────────────────────────────────────────────────
class TestDataFrameContract:
    def test_columns_match_contract(self):
        fake = _fake_stock_df(return_value=_fake_raw(rows=5))
        prov = JugaadDataProvider(stock_df_fn=fake)
        df = prov.fetch_ohlcv("RELIANCE", date(2024, 4, 1), date(2024, 4, 5))
        assert list(df.columns) == [
            "open", "high", "low", "close", "adj_close", "volume",
        ]

    def test_index_is_tz_aware_kolkata(self):
        prov = JugaadDataProvider(stock_df_fn=_fake_stock_df())
        df = prov.fetch_ohlcv("RELIANCE", date(2024, 4, 1), date(2024, 4, 3))
        assert df.index.tz is not None
        assert str(df.index.tz) == "Asia/Kolkata"

    def test_index_sorted_ascending(self):
        # Build raw data in REVERSE order; provider must sort.
        raw = _fake_raw(rows=3)
        raw = raw.iloc[::-1].reset_index(drop=True)
        prov = JugaadDataProvider(stock_df_fn=_fake_stock_df(return_value=raw))
        df = prov.fetch_ohlcv("RELIANCE", date(2024, 4, 1), date(2024, 4, 3))
        assert df.index.is_monotonic_increasing

    def test_adj_close_equals_close(self):
        prov = JugaadDataProvider(stock_df_fn=_fake_stock_df())
        df = prov.fetch_ohlcv("RELIANCE", date(2024, 4, 1), date(2024, 4, 3))
        assert (df["adj_close"] == df["close"]).all()

    def test_row_count_matches_upstream(self):
        prov = JugaadDataProvider(
            stock_df_fn=_fake_stock_df(return_value=_fake_raw(rows=10)),
        )
        df = prov.fetch_ohlcv("RELIANCE", date(2024, 4, 1), date(2024, 4, 10))
        assert len(df) == 10

    def test_tz_aware_dates_are_converted_not_localized(self):
        # Defensive: if jugaad-data ever ships tz-aware DATE rows (UTC,
        # say), provider must CONVERT to Asia/Kolkata, not localize
        # (which would double-shift the wallclock).
        raw = _fake_raw(rows=3)
        raw["DATE"] = pd.to_datetime(raw["DATE"]).dt.tz_localize("UTC")
        prov = JugaadDataProvider(stock_df_fn=_fake_stock_df(return_value=raw))
        df = prov.fetch_ohlcv("RELIANCE", date(2024, 4, 1), date(2024, 4, 3))
        assert str(df.index.tz) == "Asia/Kolkata"
        # 2024-04-01 00:00 UTC -> 2024-04-01 05:30 IST (NOT 00:00).
        assert df.index[0].hour == 5 and df.index[0].minute == 30


# ─────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────
class TestErrorHandling:
    def test_upstream_exception_wrapped_as_pricefetcherror(self):
        prov = JugaadDataProvider(
            stock_df_fn=_fake_stock_df(raises=ConnectionError("nse hosed")),
        )
        with pytest.raises(PriceFetchError, match="jugaad-data failed"):
            prov.fetch_ohlcv("RELIANCE", date(2024, 4, 1), date(2024, 4, 3))

    def test_upstream_exception_chained(self):
        original = RuntimeError("boom")
        prov = JugaadDataProvider(stock_df_fn=_fake_stock_df(raises=original))
        with pytest.raises(PriceFetchError) as excinfo:
            prov.fetch_ohlcv("RELIANCE", date(2024, 4, 1), date(2024, 4, 3))
        assert excinfo.value.__cause__ is original

    def test_empty_df_raises_pricefetcherror(self):
        prov = JugaadDataProvider(
            stock_df_fn=_fake_stock_df(return_value=_fake_raw(rows=0)),
        )
        with pytest.raises(PriceFetchError, match="returned no data"):
            prov.fetch_ohlcv("RELIANCE", date(2024, 4, 1), date(2024, 4, 3))

    def test_none_response_raises_pricefetcherror(self):
        # Defensive: stock_df returning None (rare; happens on some
        # internal jugaad-data parse failures we want to surface).
        def _none_fn(*args, **kwargs):
            return None
        prov = JugaadDataProvider(stock_df_fn=_none_fn)
        with pytest.raises(PriceFetchError, match="returned no data"):
            prov.fetch_ohlcv("RELIANCE", date(2024, 4, 1), date(2024, 4, 3))

    def test_missing_columns_raises_pricefetcherror(self):
        # Upstream API change: drop a required column.
        bad = _fake_raw(rows=3).drop(columns=["VOLUME"])
        prov = JugaadDataProvider(stock_df_fn=_fake_stock_df(return_value=bad))
        with pytest.raises(PriceFetchError, match="missing expected columns"):
            prov.fetch_ohlcv("RELIANCE", date(2024, 4, 1), date(2024, 4, 3))

    def test_unparseable_dates_raise_pricefetcherror(self):
        bad = _fake_raw(rows=3)
        # Mix in a row with a totally-broken date. Build via dict assignment
        # to dodge pandas' incompatible-dtype FutureWarning.
        bad["DATE"] = bad["DATE"].astype(object)
        bad.loc[1, "DATE"] = "totally not a date"
        prov = JugaadDataProvider(stock_df_fn=_fake_stock_df(return_value=bad))
        with pytest.raises(PriceFetchError, match="unparseable DATE rows"):
            prov.fetch_ohlcv("RELIANCE", date(2024, 4, 1), date(2024, 4, 3))


# ─────────────────────────────────────────────────────────────
# Lazy import of jugaad_data
# ─────────────────────────────────────────────────────────────
class TestLazyImport:
    def test_real_module_resolved_when_no_injection(self, monkeypatch):
        """Without an injected stock_df_fn, the provider lazy-imports the
        real jugaad_data.nse.stock_df. Verify by patching the module."""
        prov = JugaadDataProvider()  # no injection

        sentinel_calls: list = []

        def fake_stock_df(symbol, from_date, to_date, series="EQ"):
            sentinel_calls.append((symbol, from_date, to_date, series))
            return _fake_raw(rows=2)

        # Patch the function as imported within jugaad_data.nse — the
        # provider lazy-imports `from jugaad_data.nse import stock_df`.
        import jugaad_data.nse as jdn
        monkeypatch.setattr(jdn, "stock_df", fake_stock_df)

        df = prov.fetch_ohlcv("INFY", date(2024, 4, 1), date(2024, 4, 2))
        assert len(df) == 2
        assert sentinel_calls == [("INFY", date(2024, 4, 1), date(2024, 4, 2), "EQ")]

    def test_missing_jugaad_data_raises_pricefetcherror(self, monkeypatch):
        """If jugaad_data.nse is not importable, the lazy import path must
        surface a PriceFetchError with an actionable message."""
        prov = JugaadDataProvider()  # no injection

        # Simulate the import failing — the cleanest way is to remove the
        # cached module + intercept the import.
        import sys
        monkeypatch.setitem(sys.modules, "jugaad_data", None)
        monkeypatch.setitem(sys.modules, "jugaad_data.nse", None)

        with pytest.raises(PriceFetchError, match="not installed"):
            prov.fetch_ohlcv("INFY", date(2024, 4, 1), date(2024, 4, 2))


# ─────────────────────────────────────────────────────────────
# Registry integration
# ─────────────────────────────────────────────────────────────
class TestRegistryIntegration:
    def test_registered_under_jugaad_short_name(self):
        from price_predictor.data.providers import (
            PROVIDER_REGISTRY,
            JugaadDataProvider as Cls,
            build_provider,
        )
        assert "jugaad" in PROVIDER_REGISTRY
        instance = build_provider("jugaad")
        assert isinstance(instance, Cls)
        assert instance.name == "jugaad-data"

    def test_unknown_provider_still_raises(self):
        from price_predictor.data.providers import build_provider
        with pytest.raises(ValueError, match="Unknown price provider"):
            build_provider("not-a-real-provider")
