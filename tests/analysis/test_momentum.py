"""Tests for analysis.momentum -- RSI, MACD, Stochastic, OBV."""
from __future__ import annotations

from price_predictor.analysis.momentum import (
    latest_macd,
    latest_obv,
    latest_rsi,
    latest_stoch,
    momentum_snapshot,
)
from tests.analysis.conftest import (
    insufficient_history,
    linear_downtrend,
    linear_uptrend,
    sideways,
)


class TestLatestRSI:
    def test_pure_uptrend_rsi_overbought(self):
        df = linear_uptrend(n=200, start=100, slope=1)
        rsi = latest_rsi(df, length=14)
        assert rsi is not None
        # Strict uptrend -> RSI should pin near 100
        assert rsi > 90

    def test_pure_downtrend_rsi_oversold(self):
        df = linear_downtrend(n=200, start=200, slope=0.5)
        rsi = latest_rsi(df, length=14)
        assert rsi is not None
        assert rsi < 10

    def test_sideways_rsi_neutral(self):
        df = sideways(n=200)
        rsi = latest_rsi(df, length=14)
        assert rsi is not None
        assert 30 < rsi < 70

    def test_short_history_returns_none(self):
        assert latest_rsi(insufficient_history(5), length=14) is None


class TestLatestMACD:
    def test_uptrend_macd_above_signal(self):
        df = linear_uptrend(n=200, start=100, slope=1)
        macd = latest_macd(df)
        assert macd["macd"] is not None
        assert macd["signal"] is not None
        # On a perfectly linear uptrend the MACD and signal lines
        # mathematically converge to the same value (EMAs of EMAs catch up).
        # Allow tiny floating-point difference but assert they're effectively
        # equal-or-above.
        assert macd["macd"] >= macd["signal"] - 1e-6
        assert macd["histogram"] >= -1e-6

    def test_downtrend_macd_below_signal(self):
        df = linear_downtrend(n=200, start=200, slope=0.5)
        macd = latest_macd(df)
        assert macd["macd"] < macd["signal"]
        assert macd["histogram"] < 0

    def test_short_history_all_none(self):
        macd = latest_macd(insufficient_history(5))
        assert macd["macd"] is None
        assert macd["signal"] is None
        assert macd["histogram"] is None
        assert macd["cross"] is None


class TestLatestStoch:
    def test_uptrend_stoch_high(self):
        df = linear_uptrend(n=100, start=100, slope=1)
        stoch = latest_stoch(df)
        assert stoch["k"] is not None
        # Pure uptrend -> %K pinned near 100
        assert stoch["k"] > 80

    def test_downtrend_stoch_low(self):
        df = linear_downtrend(n=100, start=200, slope=0.5)
        stoch = latest_stoch(df)
        assert stoch["k"] < 20

    def test_short_history_none(self):
        assert latest_stoch(insufficient_history(5))["k"] is None


class TestLatestOBV:
    def test_uptrend_with_volume_obv_positive(self):
        df = linear_uptrend(n=100, start=100, slope=1)
        obv = latest_obv(df)
        assert obv["obv"] is not None
        # All up days with positive volume -> OBV grows
        assert obv["obv"] > 0
        assert obv["slope_20"] is not None

    def test_no_volume_column_returns_none(self):
        df = linear_uptrend(n=100, start=100, slope=1).drop(columns=["volume"])
        obv = latest_obv(df)
        assert obv["obv"] is None
        assert obv["slope_20"] is None


class TestMomentumSnapshot:
    def test_uptrend_snapshot_shape(self):
        df = linear_uptrend(n=200, start=100, slope=1)
        snap = momentum_snapshot(
            df, rsi_length=14, macd_params=(12, 26, 9), stoch_params=(14, 3, 3)
        )
        assert snap["rsi"] is not None
        assert snap["macd"]["macd"] is not None
        assert snap["stoch"]["k"] is not None
        assert snap["obv"]["obv"] is not None

    def test_short_history_graceful(self):
        df = insufficient_history(n=10)
        snap = momentum_snapshot(
            df, rsi_length=14, macd_params=(12, 26, 9), stoch_params=(14, 3, 3)
        )
        assert snap["rsi"] is None
        assert snap["macd"]["macd"] is None
        assert snap["stoch"]["k"] is None
