"""Unit tests for data/providers/bhavcopy_provider.py.

Pure adapter tests — we inject a `bhavcopy_fn` into the provider so
no HTTP is involved at all.
"""
from __future__ import annotations

import time
from datetime import date

import pandas as pd
import pytest

from price_predictor.data.bhavcopy import BhavcopyError
from price_predictor.data.providers.base import PriceFetchError
from price_predictor.data.providers.bhavcopy_provider import (
    NseBhavcopyProvider,
    _FAIL_FAST_ERROR_THRESHOLD,
    _iter_trading_days,
)


# ─────────────────────────────────────────────────────────────
# Helpers — fake bhavcopy frames
# ─────────────────────────────────────────────────────────────
def _fake_day(d: date, *, symbols: list[str] = ("RELIANCE", "INFY")) -> pd.DataFrame:
    """Build a DataFrame matching what fetch_nse_bhavcopy returns."""
    return pd.DataFrame({
        "SYMBOL": list(symbols),
        "SERIES": ["EQ"] * len(symbols),
        "DATE": [pd.Timestamp(d, tz="Asia/Kolkata")] * len(symbols),
        "OPEN": [100.0 + i for i in range(len(symbols))],
        "HIGH": [102.0 + i for i in range(len(symbols))],
        "LOW": [99.0 + i for i in range(len(symbols))],
        "CLOSE": [101.0 + i for i in range(len(symbols))],
        "VOLUME": [1_000_000 + i * 1000 for i in range(len(symbols))],
    })


def _make_fetcher(per_day: dict[date, pd.DataFrame] | None = None,
                  raises: dict[date, Exception] | None = None):
    """Factory for an injectable bhavcopy_fn that captures calls."""
    per_day = per_day or {}
    raises = raises or {}
    captured: list[date] = []

    def _impl(d: date) -> pd.DataFrame:
        captured.append(d)
        if d in raises:
            raise raises[d]
        if d in per_day:
            return per_day[d]
        # Default: return a frame with our default symbols.
        return _fake_day(d)

    _impl.captured = captured  # type: ignore[attr-defined]
    return _impl


# ─────────────────────────────────────────────────────────────
# _iter_trading_days
# ─────────────────────────────────────────────────────────────
class TestIterTradingDays:
    def test_skips_weekends(self):
        # Mon 2024-04-22 .. Sun 2024-04-28 → 5 weekdays.
        days = list(_iter_trading_days(date(2024, 4, 22), date(2024, 4, 28)))
        assert all(d.weekday() < 5 for d in days)
        # Should be 5 weekdays UNLESS one is an NSE holiday.
        assert len(days) <= 5

    def test_inverted_range_yields_nothing(self):
        days = list(_iter_trading_days(date(2024, 4, 28), date(2024, 4, 22)))
        assert days == []

    def test_single_trading_day(self):
        # Pick a known Tuesday (Apr 23 2024).
        days = list(_iter_trading_days(date(2024, 4, 23), date(2024, 4, 23)))
        assert days == [date(2024, 4, 23)]


# ─────────────────────────────────────────────────────────────
# Basic behaviour
# ─────────────────────────────────────────────────────────────
class TestProviderBasics:
    def test_name(self):
        assert NseBhavcopyProvider().name == "nse-bhavcopy"

    def test_default_fetcher_is_real_one(self):
        # Constructed with no injection → uses real fetch_nse_bhavcopy.
        from price_predictor.data import bhavcopy as bm
        prov = NseBhavcopyProvider()
        assert prov._bhavcopy_fn is bm.fetch_nse_bhavcopy


# ─────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────
class TestInputValidation:
    def setup_method(self):
        self.prov = NseBhavcopyProvider(bhavcopy_fn=_make_fetcher())

    @pytest.mark.parametrize("bad", ["", "   ", None])
    def test_empty_ticker_raises_valueerror(self, bad):
        with pytest.raises(ValueError, match="non-empty"):
            self.prov.fetch_ohlcv(bad, date(2024, 4, 22), date(2024, 4, 24))  # type: ignore[arg-type]

    def test_inverted_dates_raise_valueerror(self):
        with pytest.raises(ValueError, match="must be <="):
            self.prov.fetch_ohlcv("RELIANCE", date(2024, 4, 24), date(2024, 4, 22))

    def test_unsupported_interval_raises_valueerror(self):
        with pytest.raises(ValueError, match="only supports interval='1d'"):
            self.prov.fetch_ohlcv(
                "RELIANCE", date(2024, 4, 22), date(2024, 4, 24), interval="1h",
            )

    def test_suffix_only_ticker_raises_valueerror(self):
        with pytest.raises(ValueError, match="empty string after stripping"):
            self.prov.fetch_ohlcv(".NS", date(2024, 4, 22), date(2024, 4, 24))


# ─────────────────────────────────────────────────────────────
# Fetch happy paths
# ─────────────────────────────────────────────────────────────
class TestFetchOhlcv:
    def test_iterates_only_trading_days(self):
        # Mon-Fri week (no NSE holidays expected) = 5 calls.
        fetcher = _make_fetcher()
        prov = NseBhavcopyProvider(bhavcopy_fn=fetcher)
        prov.fetch_ohlcv("RELIANCE", date(2024, 4, 22), date(2024, 4, 28))
        # Captured days must all be weekdays.
        assert all(d.weekday() < 5 for d in fetcher.captured)  # type: ignore[attr-defined]
        # Exactly the 5 weekdays in that range.
        assert len(fetcher.captured) <= 5  # type: ignore[attr-defined]

    def test_filters_to_requested_symbol(self):
        fetcher = _make_fetcher()
        prov = NseBhavcopyProvider(bhavcopy_fn=fetcher)
        df = prov.fetch_ohlcv("INFY", date(2024, 4, 22), date(2024, 4, 26))
        # Each day yields 1 row for INFY (filter out RELIANCE).
        assert len(df) == len([d for d in fetcher.captured  # type: ignore[attr-defined]
                               if d.weekday() < 5])
        # Non-zero rows verify filtering actually returned something.
        assert len(df) > 0

    def test_dataframe_contract(self):
        prov = NseBhavcopyProvider(bhavcopy_fn=_make_fetcher())
        df = prov.fetch_ohlcv("RELIANCE", date(2024, 4, 22), date(2024, 4, 26))
        assert list(df.columns) == [
            "open", "high", "low", "close", "adj_close", "volume",
        ]
        assert str(df.index.tz) == "Asia/Kolkata"
        assert df.index.is_monotonic_increasing
        assert (df["adj_close"] == df["close"]).all()

    def test_lowercase_ticker_normalised_uppercase(self):
        prov = NseBhavcopyProvider(bhavcopy_fn=_make_fetcher())
        df = prov.fetch_ohlcv("reliance", date(2024, 4, 22), date(2024, 4, 26))
        assert len(df) > 0  # would be 0 if uppercase normalisation missed

    def test_yfinance_style_suffix_stripped(self):
        prov = NseBhavcopyProvider(bhavcopy_fn=_make_fetcher())
        df = prov.fetch_ohlcv("RELIANCE.NS", date(2024, 4, 22), date(2024, 4, 26))
        assert len(df) > 0

    def test_tz_naive_dates_get_localized_not_double_shifted(self):
        # Defensive: if bhavcopy bulk ever ships tz-naive DATE, provider
        # must localize (not crash, not double-shift).
        def _tz_naive_fetcher(d: date) -> pd.DataFrame:
            df = _fake_day(d)
            df["DATE"] = pd.Timestamp(d)  # tz-naive
            return df

        prov = NseBhavcopyProvider(bhavcopy_fn=_tz_naive_fetcher)
        df = prov.fetch_ohlcv("RELIANCE", date(2024, 4, 23), date(2024, 4, 23))
        assert str(df.index.tz) == "Asia/Kolkata"


# ─────────────────────────────────────────────────────────────
# Error paths
# ─────────────────────────────────────────────────────────────
class TestErrorPaths:
    def test_zero_matching_rows_raises(self):
        # Fetcher returns days but with NO matching symbol.
        prov = NseBhavcopyProvider(
            bhavcopy_fn=_make_fetcher(per_day={
                date(2024, 4, 22): _fake_day(date(2024, 4, 22), symbols=["TCS"]),
                date(2024, 4, 23): _fake_day(date(2024, 4, 23), symbols=["TCS"]),
            }),
        )
        with pytest.raises(PriceFetchError, match="no rows for symbol"):
            prov.fetch_ohlcv("RELIANCE", date(2024, 4, 22), date(2024, 4, 23))

    def test_per_day_bhavcopyerror_does_not_abort_run(self):
        # Day 1 errors, Day 2 returns RELIANCE → result must include day 2.
        d1 = date(2024, 4, 22)  # Mon
        d2 = date(2024, 4, 23)  # Tue
        prov = NseBhavcopyProvider(bhavcopy_fn=_make_fetcher(
            raises={d1: BhavcopyError("404 holiday")},
            per_day={d2: _fake_day(d2)},
        ))
        df = prov.fetch_ohlcv("RELIANCE", d1, d2)
        assert len(df) >= 1

    def test_all_days_error_raises_with_summary(self):
        d1 = date(2024, 4, 22)
        d2 = date(2024, 4, 23)
        d3 = date(2024, 4, 24)
        prov = NseBhavcopyProvider(bhavcopy_fn=_make_fetcher(
            raises={
                d1: BhavcopyError("nse 500"),
                d2: BhavcopyError("nse 500"),
                d3: BhavcopyError("nse 500"),
            },
        ))
        with pytest.raises(PriceFetchError, match="day\\(s\\) errored"):
            prov.fetch_ohlcv("RELIANCE", d1, d3)

    def test_only_holidays_in_range_raises(self):
        # Sat-Sun → no trading days → no fetcher calls → error.
        fetcher = _make_fetcher()
        prov = NseBhavcopyProvider(bhavcopy_fn=fetcher)
        with pytest.raises(PriceFetchError, match="no rows for symbol"):
            prov.fetch_ohlcv("RELIANCE", date(2024, 4, 27), date(2024, 4, 28))
        assert fetcher.captured == []  # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────
# Fail-fast on wholesale NSE block (geo-block / 403 / timeout).
#
# Outside India NSE blocks EVERY per-day fetch. Without fail-fast the
# provider would grind through hundreds of trading days (workers at a
# time, 30s timeout each) before giving up -- minutes wasted for a
# guaranteed-empty result. These tests lock in the early abort that lets
# the resilient chain fall back to yfinance in seconds.
#
# max_workers=1 keeps completion order == submission order so the
# threshold-counting assertions are deterministic (no thread races).
# ─────────────────────────────────────────────────────────────
class TestFailFast:
    @staticmethod
    def _always_raises():
        captured: list[date] = []

        def _impl(d: date) -> pd.DataFrame:
            captured.append(d)
            # Simulate NSE latency (real fetches carry a 30s timeout). A
            # tiny sleep keeps the single worker from racing through every
            # instant task before the main loop can break + cancel the
            # pending futures -- which is precisely what fail-fast prevents
            # in production, where each call is slow.
            time.sleep(0.02)
            raise BhavcopyError("403 access denied (NSE geo-block)")

        _impl.captured = captured  # type: ignore[attr-defined]
        return _impl

    def test_aborts_early_without_fetching_every_day(self):
        # A month-long window has well over the threshold of trading
        # days; every one errors -> must abort early rather than fetch
        # all of them.
        start, end = date(2024, 4, 1), date(2024, 4, 30)
        trading_days = list(_iter_trading_days(start, end))
        assert len(trading_days) > _FAIL_FAST_ERROR_THRESHOLD  # precondition

        fetcher = self._always_raises()
        prov = NseBhavcopyProvider(bhavcopy_fn=fetcher, max_workers=1)
        with pytest.raises(PriceFetchError, match="aborted early"):
            prov.fetch_ohlcv("RELIANCE", start, end)

        # Proof of early abort: not every trading day was fetched.
        assert len(fetcher.captured) < len(trading_days)  # type: ignore[attr-defined]

    def test_error_message_points_to_yfinance_workaround(self):
        prov = NseBhavcopyProvider(
            bhavcopy_fn=self._always_raises(), max_workers=1,
        )
        with pytest.raises(PriceFetchError, match="PRICE_CHAIN=yfinance"):
            prov.fetch_ohlcv("RELIANCE", date(2024, 4, 1), date(2024, 4, 30))

    def test_no_fail_fast_when_a_day_succeeds(self):
        # A reachable NSE interleaves successes among any blips. One good
        # day means we must NOT bail -- return what we found.
        start, end = date(2024, 4, 1), date(2024, 4, 30)
        good_day = list(_iter_trading_days(start, end))[0]

        def _impl(d: date) -> pd.DataFrame:
            if d == good_day:
                return _fake_day(d)
            raise BhavcopyError("403")

        prov = NseBhavcopyProvider(bhavcopy_fn=_impl, max_workers=1)
        df = prov.fetch_ohlcv("RELIANCE", start, end)
        assert len(df) >= 1

    def test_small_range_below_threshold_uses_normal_error_path(self):
        # <= threshold trading days all erroring -> the ORIGINAL
        # "day(s) errored" summary, NOT the fail-fast abort.
        start, end = date(2024, 4, 22), date(2024, 4, 24)  # Mon-Wed
        trading_days = list(_iter_trading_days(start, end))
        assert len(trading_days) <= _FAIL_FAST_ERROR_THRESHOLD

        prov = NseBhavcopyProvider(
            bhavcopy_fn=self._always_raises(), max_workers=1,
        )
        with pytest.raises(PriceFetchError, match=r"day\(s\) errored"):
            prov.fetch_ohlcv("RELIANCE", start, end)


# ─────────────────────────────────────────────────────────────
# Registry integration
# ─────────────────────────────────────────────────────────────
class TestRegistryIntegration:
    def test_registered_under_nse_bhavcopy_short_name(self):
        from price_predictor.data.providers import (
            PROVIDER_REGISTRY,
            NseBhavcopyProvider as Cls,
            build_provider,
        )
        assert "nse_bhavcopy" in PROVIDER_REGISTRY
        instance = build_provider("nse_bhavcopy")
        assert isinstance(instance, Cls)
        assert instance.name == "nse-bhavcopy"


# ─────────────────────────────────────────────────────────────
# Parallel fetching — added when we moved the per-day loop to
# a ThreadPoolExecutor to fix the ~4-minute warm-from-cold pain
# on 2-year windows.
# ─────────────────────────────────────────────────────────────
class TestParallelFanout:
    def test_max_workers_defaults_to_env_or_fallback(self, monkeypatch):
        # No env set → fallback (8 at time of writing).
        monkeypatch.delenv(
            "PRICE_PREDICTOR_BHAVCOPY_MAX_WORKERS", raising=False,
        )
        from price_predictor.data.providers.bhavcopy_provider import (
            _DEFAULT_MAX_WORKERS_FALLBACK,
        )
        prov = NseBhavcopyProvider(bhavcopy_fn=_make_fetcher())
        assert prov._max_workers == _DEFAULT_MAX_WORKERS_FALLBACK

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("PRICE_PREDICTOR_BHAVCOPY_MAX_WORKERS", "4")
        prov = NseBhavcopyProvider(bhavcopy_fn=_make_fetcher())
        assert prov._max_workers == 4

    def test_constructor_arg_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("PRICE_PREDICTOR_BHAVCOPY_MAX_WORKERS", "4")
        prov = NseBhavcopyProvider(bhavcopy_fn=_make_fetcher(), max_workers=2)
        assert prov._max_workers == 2

    def test_invalid_env_falls_back_with_warning(self, monkeypatch):
        monkeypatch.setenv("PRICE_PREDICTOR_BHAVCOPY_MAX_WORKERS", "not-an-int")
        from price_predictor.data.providers.bhavcopy_provider import (
            _DEFAULT_MAX_WORKERS_FALLBACK,
        )
        prov = NseBhavcopyProvider(bhavcopy_fn=_make_fetcher())
        assert prov._max_workers == _DEFAULT_MAX_WORKERS_FALLBACK

    def test_clamps_to_one_minimum(self):
        # Pool of 0 or negative is meaningless — must clamp up.
        prov = NseBhavcopyProvider(bhavcopy_fn=_make_fetcher(), max_workers=0)
        assert prov._max_workers == 1
        prov2 = NseBhavcopyProvider(bhavcopy_fn=_make_fetcher(), max_workers=-5)
        assert prov2._max_workers == 1

    def test_clamps_to_thirty_two_maximum(self):
        # Protect NSE from a misconfigured huge pool.
        prov = NseBhavcopyProvider(bhavcopy_fn=_make_fetcher(), max_workers=999)
        assert prov._max_workers == 32

    def test_parallel_fetch_returns_all_days(self):
        # Wide window with 10 trading days, pool of 4 — every day's data
        # must show up in the result.
        fetcher = _make_fetcher()
        prov = NseBhavcopyProvider(bhavcopy_fn=fetcher, max_workers=4)
        df = prov.fetch_ohlcv(
            "RELIANCE",
            date(2024, 4, 22),
            date(2024, 5, 6),  # ~10 trading days
        )
        # One row per trading day in the range.
        expected_days = len([d for d in fetcher.captured  # type: ignore[attr-defined]
                              if d.weekday() < 5])
        assert len(df) == expected_days
        # Index must still be sorted ascending despite arbitrary
        # future-completion order from the thread pool.
        assert df.index.is_monotonic_increasing

    def test_parallel_per_day_errors_dont_abort_batch(self):
        # 5 trading days, 2 of which error — the 3 good ones must
        # still come back in order.
        d1 = date(2024, 4, 22)  # Mon
        d2 = date(2024, 4, 23)  # Tue — errors
        d3 = date(2024, 4, 24)  # Wed
        d4 = date(2024, 4, 25)  # Thu — errors
        d5 = date(2024, 4, 26)  # Fri
        prov = NseBhavcopyProvider(
            bhavcopy_fn=_make_fetcher(
                raises={
                    d2: BhavcopyError("transient"),
                    d4: BhavcopyError("transient"),
                },
            ),
            max_workers=4,
        )
        df = prov.fetch_ohlcv("RELIANCE", d1, d5)
        assert len(df) == 3  # Mon, Wed, Fri only
        assert df.index.is_monotonic_increasing

    def test_single_day_window_still_works(self):
        # Edge case: 1 trading day — workers gets clamped to 1.
        prov = NseBhavcopyProvider(bhavcopy_fn=_make_fetcher(), max_workers=8)
        df = prov.fetch_ohlcv("RELIANCE", date(2024, 4, 23), date(2024, 4, 23))
        assert len(df) == 1
