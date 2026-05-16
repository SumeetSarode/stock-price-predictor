"""Tests for analysis.vwap -- anchored + rolling VWAP."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from price_predictor.analysis.vwap import (
    DEFAULT_ROLLING_WINDOW,
    _typical_price,
    anchored_vwap,
    rolling_vwap,
    vwap_snapshot,
)
from tests.analysis.conftest import linear_uptrend


# ── Helpers ────────────────────────────────────────────────────────


def _flat_constant_ohlcv(
    n: int = 30, price: float = 100.0, volume: float = 1000.0,
) -> pd.DataFrame:
    """Constant-price, constant-volume frame. VWAP must equal `price`."""
    dates = pd.date_range("2025-01-01", periods=n, freq="D", tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "open":      np.full(n, price),
            "high":      np.full(n, price),
            "low":       np.full(n, price),
            "close":     np.full(n, price),
            "adj_close": np.full(n, price),
            "volume":    np.full(n, volume),
        },
        index=dates,
    )


# ── _typical_price ─────────────────────────────────────────────────


class TestTypicalPrice:
    def test_average_of_hlc(self):
        df = pd.DataFrame({"high": [110.0], "low": [90.0], "close": [100.0]})
        tp = _typical_price(df)
        assert tp.iloc[0] == pytest.approx(100.0)

    def test_uneven_hlc(self):
        df = pd.DataFrame({"high": [120.0], "low": [60.0], "close": [90.0]})
        # (120 + 60 + 90) / 3 = 90
        assert _typical_price(df).iloc[0] == pytest.approx(90.0)


# ── rolling_vwap ───────────────────────────────────────────────────


class TestRollingVwap:
    def test_constant_price_equals_price(self):
        df = _flat_constant_ohlcv(n=30, price=100.0)
        rv = rolling_vwap(df, window=20)
        # First 19 bars NaN, rest should all be exactly 100
        assert rv.iloc[-1] == pytest.approx(100.0)
        assert rv.iloc[-10] == pytest.approx(100.0)
        assert pd.isna(rv.iloc[0])
        assert pd.isna(rv.iloc[18])
        assert not pd.isna(rv.iloc[19])

    def test_volume_weighted_not_price_weighted(self):
        """If one bar has 100x the volume at price 200, VWAP tilts toward 200."""
        df = _flat_constant_ohlcv(n=20, price=100.0, volume=1.0)
        # Spike volume on the last bar at a different price
        idx = df.index[-1]
        df.loc[idx, ["open", "high", "low", "close"]] = 200.0
        df.loc[idx, "volume"] = 100.0  # 100x the volume

        rv = rolling_vwap(df, window=20)
        # 19 bars * 1 unit * 100 = 1900, plus 1 bar * 100 units * 200 = 20000
        # numerator = 21900; denominator = 19 + 100 = 119; vwap = ~184.03
        expected = (19 * 1 * 100 + 1 * 100 * 200) / (19 * 1 + 1 * 100)
        assert rv.iloc[-1] == pytest.approx(expected)
        # Critical: VWAP is well above 150 (simple average would be 105)
        assert rv.iloc[-1] > 180

    def test_window_larger_than_history_returns_all_nan(self):
        df = _flat_constant_ohlcv(n=10, price=100.0)
        rv = rolling_vwap(df, window=20)
        assert rv.isna().all()

    def test_empty_df_returns_empty_series(self):
        empty = pd.DataFrame(columns=["high", "low", "close", "volume"])
        rv = rolling_vwap(empty)
        assert rv.empty

    def test_zero_volume_window_yields_nan(self):
        df = _flat_constant_ohlcv(n=25, price=100.0, volume=0.0)
        rv = rolling_vwap(df, window=20)
        # Division by zero volume -> NaN, not inf
        assert pd.isna(rv.iloc[-1])
        assert not np.isinf(rv.iloc[-1])

    def test_invalid_window_returns_aligned_all_nan(self):
        """Invalid window shouldn't crash; should yield an aligned all-NaN series."""
        df = _flat_constant_ohlcv(n=10, price=100.0)
        rv_zero = rolling_vwap(df, window=0)
        rv_neg = rolling_vwap(df, window=-5)
        # Same index alignment as df (callers can join without surprises)
        assert len(rv_zero) == len(df)
        assert len(rv_neg) == len(df)
        assert rv_zero.isna().all()
        assert rv_neg.isna().all()


# ── anchored_vwap ──────────────────────────────────────────────────


class TestAnchoredVwap:
    def test_constant_price_equals_price(self):
        df = _flat_constant_ohlcv(n=30, price=150.0)
        anchor = df.index[10].date()
        av = anchored_vwap(df, anchor_date=anchor)
        # Pre-anchor bars NaN; anchored bars all 150
        assert pd.isna(av.iloc[0])
        assert pd.isna(av.iloc[9])
        assert av.iloc[10] == pytest.approx(150.0)
        assert av.iloc[-1] == pytest.approx(150.0)

    def test_anchor_after_last_bar_yields_all_nan(self):
        df = _flat_constant_ohlcv(n=10, price=100.0)
        # Anchor in the future
        av = anchored_vwap(df, anchor_date=date(2099, 1, 1))
        assert av.isna().all()

    def test_anchor_accepts_timestamp_datetime_date(self):
        df = _flat_constant_ohlcv(n=20, price=100.0)
        anchor_dt = df.index[5]  # tz-aware Timestamp
        av_ts = anchored_vwap(df, anchor_date=anchor_dt)
        av_py = anchored_vwap(df, anchor_date=anchor_dt.to_pydatetime())
        av_dt = anchored_vwap(df, anchor_date=anchor_dt.date())
        # All three must produce identical series
        pd.testing.assert_series_equal(av_ts, av_py, check_names=False)
        pd.testing.assert_series_equal(av_py, av_dt, check_names=False)

    def test_anchor_index_aligned_to_input(self):
        df = linear_uptrend(n=50, start=100.0, slope=1.0)
        anchor = df.index[20].date()
        av = anchored_vwap(df, anchor_date=anchor)
        # Must match df length and index
        assert len(av) == len(df)
        assert (av.index == df.index).all()

    def test_running_vwap_monotonic_for_uptrend(self):
        """For a monotonically rising series, anchored VWAP must also rise."""
        df = linear_uptrend(n=50, start=100.0, slope=1.0)
        anchor = df.index[10].date()
        av = anchored_vwap(df, anchor_date=anchor).dropna()
        # Each subsequent VWAP should be > the previous (rising input).
        diffs = av.diff().dropna()
        assert (diffs > 0).all()

    def test_volume_weighted_first_bar_equals_typical_price(self):
        """At the anchor bar itself, anchored VWAP == TP of that bar."""
        df = _flat_constant_ohlcv(n=10, price=100.0)
        anchor_idx = 3
        # Bump that one bar's prices to confirm TP is used (not close)
        df.iloc[anchor_idx, df.columns.get_loc("high")] = 110.0
        df.iloc[anchor_idx, df.columns.get_loc("low")] = 90.0
        df.iloc[anchor_idx, df.columns.get_loc("close")] = 105.0
        anchor = df.index[anchor_idx].date()
        av = anchored_vwap(df, anchor_date=anchor)
        # TP at anchor = (110 + 90 + 105)/3 = 101.6667
        assert av.iloc[anchor_idx] == pytest.approx((110 + 90 + 105) / 3)

    def test_missing_volume_column_yields_aligned_all_nan(self):
        df = pd.DataFrame(
            {"high": [1.0], "low": [1.0], "close": [1.0]},
            index=pd.date_range("2025-01-01", periods=1, tz="Asia/Kolkata"),
        )
        av = anchored_vwap(df, anchor_date=date(2025, 1, 1))
        # No volume column -> aligned all-NaN (defensive, not a crash)
        assert len(av) == len(df)
        assert av.isna().all()

    def test_empty_df_returns_empty_series(self):
        empty = pd.DataFrame(columns=["high", "low", "close", "volume"])
        av = anchored_vwap(empty, anchor_date=date(2025, 1, 1))
        assert av.empty


# ── vwap_snapshot ──────────────────────────────────────────────────


class TestVwapSnapshot:
    def test_rolling_only_snapshot(self):
        df = _flat_constant_ohlcv(n=30, price=100.0)
        snap = vwap_snapshot(df)
        assert snap["vwap_rolling"] == pytest.approx(100.0)
        assert snap["vwap_anchored"] is None
        assert snap["anchor_date"] is None
        assert snap["rolling_window"] == DEFAULT_ROLLING_WINDOW

    def test_anchored_snapshot_records_anchor_date(self):
        df = _flat_constant_ohlcv(n=30, price=100.0)
        anchor = df.index[5].date()
        snap = vwap_snapshot(df, anchor_date=anchor)
        assert snap["vwap_anchored"] == pytest.approx(100.0)
        assert snap["anchor_date"] == anchor.isoformat()

    def test_anchor_accepts_timestamp(self):
        df = _flat_constant_ohlcv(n=30, price=100.0)
        anchor = df.index[5]  # Timestamp
        snap = vwap_snapshot(df, anchor_date=anchor)
        assert snap["vwap_anchored"] == pytest.approx(100.0)
        assert snap["anchor_date"] == anchor.date().isoformat()

    def test_empty_df_returns_all_none(self):
        empty = pd.DataFrame(columns=["high", "low", "close", "volume"])
        snap = vwap_snapshot(empty)
        assert snap["vwap_rolling"] is None
        assert snap["vwap_anchored"] is None

    def test_short_history_rolling_none(self):
        # 5 bars but default window is 20 -> rolling VWAP not yet defined
        df = _flat_constant_ohlcv(n=5, price=100.0)
        snap = vwap_snapshot(df)
        assert snap["vwap_rolling"] is None

    def test_custom_rolling_window_used(self):
        df = _flat_constant_ohlcv(n=10, price=100.0)
        snap = vwap_snapshot(df, rolling_window=5)
        assert snap["vwap_rolling"] == pytest.approx(100.0)
        assert snap["rolling_window"] == 5
