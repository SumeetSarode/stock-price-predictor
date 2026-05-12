"""Tests for analysis.trend -- SMAs, EMA, ADX, snapshot, MA crossovers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from price_predictor.analysis.trend import (
    DEFAULT_MA_CROSS_PAIRS,
    detect_ma_cross,
    detect_ma_crosses,
    latest_adx,
    latest_ema,
    latest_sma,
    trend_snapshot,
)
from tests.analysis.conftest import (
    _ohlc_from_close,
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

    def test_snapshot_includes_default_ma_crosses(self):
        """trend_snapshot ships the default pairs (SMA-50/200 + EMA-9/21)
        without the caller having to know the keys."""
        df = linear_uptrend(n=250, start=100, slope=1)
        snap = trend_snapshot(
            df, sma_lengths=[20, 50, 200], ema_length=20, adx_length=14
        )
        assert "ma_crosses" in snap
        assert set(snap["ma_crosses"].keys()) == {"sma_50_200", "ema_9_21"}

    def test_snapshot_custom_ma_pairs_override(self):
        df = linear_uptrend(n=250, start=100, slope=1)
        snap = trend_snapshot(
            df, sma_lengths=[20, 50, 200], ema_length=20, adx_length=14,
            ma_cross_pairs=[("sma", 20, 50)],
        )
        assert set(snap["ma_crosses"].keys()) == {"sma_20_50"}


# ─────────────────────────────────────────────────────────────────
# detect_ma_cross — the L3 "regime + last event" struct (Q3 design)
# ─────────────────────────────────────────────────────────────────
class TestDetectMaCrossValidation:
    def test_short_must_be_less_than_long(self):
        df = linear_uptrend(n=300)
        with pytest.raises(ValueError, match="short.*must be < long"):
            detect_ma_cross(df, 200, 50, kind="sma")

    def test_short_equal_long_raises(self):
        df = linear_uptrend(n=300)
        with pytest.raises(ValueError, match="short.*must be < long"):
            detect_ma_cross(df, 50, 50, kind="sma")

    def test_unknown_kind_raises(self):
        df = linear_uptrend(n=300)
        with pytest.raises(ValueError, match="kind must be"):
            detect_ma_cross(df, 50, 200, kind="wma")

    def test_non_positive_period_raises(self):
        df = linear_uptrend(n=300)
        with pytest.raises(ValueError, match="periods must be positive"):
            detect_ma_cross(df, 0, 50, kind="sma")
        with pytest.raises(ValueError, match="periods must be positive"):
            detect_ma_cross(df, 10, -1, kind="sma")


class TestDetectMaCrossEmptyStates:
    def test_insufficient_history_returns_all_none(self):
        # < long bars -> entire struct is None except keys present
        df = insufficient_history(n=50)
        result = detect_ma_cross(df, 50, 200, kind="sma")
        assert result == {
            "current": None,
            "last_event": None,
            "bars_since_event": None,
            "short_ma": None,
            "long_ma": None,
        }

    def test_pure_uptrend_no_cross_in_window(self):
        """A monotonic uptrend never crosses; current='above', last_event=None."""
        df = linear_uptrend(n=300, start=100, slope=1)
        result = detect_ma_cross(df, 50, 200, kind="sma")
        assert result["current"] == "above"
        assert result["last_event"] is None
        assert result["bars_since_event"] is None
        assert result["short_ma"] is not None
        assert result["long_ma"] is not None

    def test_pure_downtrend_no_cross_in_window(self):
        df = linear_downtrend(n=300, start=400, slope=1)
        result = detect_ma_cross(df, 50, 200, kind="sma")
        assert result["current"] == "below"
        assert result["last_event"] is None


class TestDetectMaCrossEvents:
    def test_engineered_bullish_cross(self):
        """Down then up -> bullish cross detected somewhere in the second half."""
        n_down, n_up = 200, 100
        close = np.concatenate([
            np.linspace(200, 100, n_down),
            np.linspace(100, 250, n_up),
        ])
        df = _ohlc_from_close(close)
        result = detect_ma_cross(df, 50, 200, kind="sma")
        assert result["current"] == "above"
        assert result["last_event"] == "bullish"
        assert isinstance(result["bars_since_event"], int)
        assert result["bars_since_event"] >= 0

    def test_engineered_bearish_cross(self):
        """Up then down -> bearish cross detected."""
        n_up, n_down = 200, 100
        close = np.concatenate([
            np.linspace(100, 250, n_up),
            np.linspace(250, 100, n_down),
        ])
        df = _ohlc_from_close(close)
        result = detect_ma_cross(df, 50, 200, kind="sma")
        assert result["current"] == "below"
        assert result["last_event"] == "bearish"

    def test_ema_pair_works(self):
        """EMA-9/21 is the second default pair; sanity-check it's plumbed."""
        df = linear_uptrend(n=200, start=100, slope=1)
        result = detect_ma_cross(df, 9, 21, kind="ema")
        assert result["current"] == "above"
        assert result["short_ma"] is not None

    def test_bars_since_event_never_negative(self):
        """Sanity: bars_since_event >= 0 in any non-None case."""
        n_down, n_up = 100, 100
        close = np.concatenate([
            np.linspace(200, 100, n_down),
            np.linspace(100, 200, n_up),
        ])
        df = _ohlc_from_close(close)
        for kind, short, long in [("sma", 50, 200), ("ema", 9, 21)]:
            r = detect_ma_cross(df, short, long, kind=kind)
            if r["bars_since_event"] is not None:
                assert r["bars_since_event"] >= 0


class TestDetectMaCrossesPlural:
    def test_default_pairs_returned(self):
        df = linear_uptrend(n=250, start=100, slope=1)
        out = detect_ma_crosses(df)
        assert set(out.keys()) == {"sma_50_200", "ema_9_21"}
        for v in out.values():
            assert {"current", "last_event", "bars_since_event",
                    "short_ma", "long_ma"} == set(v.keys())

    def test_default_pairs_constant_matches_returned_keys(self):
        # Defensive: catches drift between DEFAULT_MA_CROSS_PAIRS and what
        # detect_ma_crosses actually emits.
        from price_predictor.analysis.trend import _ma_cross_key
        df = linear_uptrend(n=250)
        out = detect_ma_crosses(df)
        expected = {_ma_cross_key(k, s, l) for (k, s, l) in DEFAULT_MA_CROSS_PAIRS}
        assert set(out.keys()) == expected

    def test_custom_pairs_override(self):
        df = linear_uptrend(n=250)
        out = detect_ma_crosses(df, pairs=[("sma", 20, 50), ("ema", 12, 26)])
        assert set(out.keys()) == {"sma_20_50", "ema_12_26"}
