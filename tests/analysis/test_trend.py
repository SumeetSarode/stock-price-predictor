"""Tests for analysis.trend -- SMAs, EMA, ADX, snapshot."""
from __future__ import annotations

from price_predictor.analysis.trend import (
    latest_adx,
    latest_ema,
    latest_sma,
    trend_snapshot,
)
from tests.analysis.conftest import (
    insufficient_history,
    linear_downtrend,
    linear_uptrend,
    sideways,
)


class TestLatestSMA:
    def test_uptrend_sma_close_to_recent_mean(self):
        df = linear_uptrend(n=250, start=100, slope=1)
        sma20 = latest_sma(df, 20)
        # Last 20 closes go from ~331 to ~350 -> mean ~340.5
        assert sma20 is not None
        assert 335 < sma20 < 345

    def test_insufficient_history_returns_none(self):
        df = insufficient_history(n=5)
        assert latest_sma(df, 20) is None


class TestLatestEMA:
    def test_uptrend_ema_close_to_recent_mean(self):
        df = linear_uptrend(n=200, start=100, slope=1)
        ema = latest_ema(df, 20)
        assert ema is not None
        assert 270 < ema < 305

    def test_insufficient_history_returns_none(self):
        assert latest_ema(insufficient_history(5), 20) is None


class TestLatestADX:
    def test_strong_uptrend_high_adx(self):
        df = linear_uptrend(n=200, start=100, slope=1)
        adx = latest_adx(df, length=14)
        assert adx["adx"] is not None
        # Pure linear uptrend -> ADX should be very high
        assert adx["adx"] > 40
        assert adx["di_plus"] is not None
        assert adx["di_minus"] is not None
        # +DI should dominate -DI in an uptrend
        assert adx["di_plus"] > adx["di_minus"]

    def test_sideways_low_adx(self):
        df = sideways(n=200)
        adx = latest_adx(df, length=14)
        assert adx["adx"] is not None
        # Chop should be < 25 (the standard "trending" threshold)
        assert adx["adx"] < 25

    def test_insufficient_history_all_none(self):
        adx = latest_adx(insufficient_history(5), length=14)
        assert adx == {"adx": None, "di_plus": None, "di_minus": None}

    def test_old_warmup_no_longer_enough_h7(self):
        # H7 fix: warmup bumped from 2*length=28 to 10*length=140.
        df = linear_uptrend(n=28, start=100, slope=1)
        adx = latest_adx(df, length=14)
        assert adx["adx"] is None

    def test_ten_length_is_enough(self):
        df = linear_uptrend(n=140, start=100, slope=1)
        adx = latest_adx(df, length=14)
        assert adx["adx"] is not None


class TestTrendSnapshot:
    def test_uptrend_snapshot_all_above_sma(self):
        df = linear_uptrend(n=250, start=100, slope=1)
        snap = trend_snapshot(df, sma_lengths=[20, 50, 200], ema_length=20, adx_length=14)
        assert snap["close"] is not None
        # Latest close > all SMAs in a clean uptrend
        for n in (20, 50, 200):
            assert snap["above_sma"][n] is True
            assert snap["pct_above_sma"][n] > 0
        assert snap["adx"]["adx"] > 40

    def test_downtrend_snapshot_all_below_sma(self):
        df = linear_downtrend(n=250, start=200, slope=0.5)
        snap = trend_snapshot(df, sma_lengths=[20, 50, 200], ema_length=20, adx_length=14)
        for n in (20, 50, 200):
            assert snap["above_sma"][n] is False
            assert snap["pct_above_sma"][n] < 0

    def test_short_history_graceful_degradation(self):
        df = insufficient_history(n=10)
        snap = trend_snapshot(df, sma_lengths=[20, 50, 200], ema_length=20, adx_length=14)
        assert snap["sma"][20] is None
        assert snap["sma"][50] is None
        assert snap["above_sma"][200] is None
        assert snap["pct_above_sma"][200] is None
