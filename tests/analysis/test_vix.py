"""Tests for analysis.vix -- the pure India VIX regime gate."""
from __future__ import annotations

import numpy as np
import pandas as pd

from price_predictor.analysis.vix import (
    DEFAULT_LOOKBACK,
    vix_regime,
    vix_snapshot,
)


def _series(values) -> pd.Series:
    n = len(values)
    idx = pd.date_range("2025-01-01", periods=n, freq="D", tz="Asia/Kolkata")
    return pd.Series(values, index=idx, name="india_vix")


class TestVixRegime:
    def test_insufficient_history_unknown(self):
        assert vix_regime(_series([15.0] * 10), lookback=60) == "unknown"

    def test_none_series_unknown(self):
        assert vix_regime(None) == "unknown"

    def test_flat_series_is_normal(self):
        # Constant VIX → latest == median → normal.
        assert vix_regime(_series([16.0] * 80), lookback=60) == "normal"

    def test_spike_is_high_vol(self):
        vals = [15.0] * 79 + [30.0]  # latest 2x the flat median
        assert vix_regime(_series(vals), lookback=60) == "high_vol"

    def test_crush_is_low_vol(self):
        vals = [20.0] * 79 + [10.0]  # latest half the flat median
        assert vix_regime(_series(vals), lookback=60) == "low_vol"

    def test_band_edges(self):
        # Just inside the ±15% band → still normal.
        base = [20.0] * 79
        assert vix_regime(_series(base + [22.0]), lookback=60) == "normal"  # 1.10x
        assert vix_regime(_series(base + [18.0]), lookback=60) == "normal"  # 0.90x

    def test_custom_lookback(self):
        vals = [15.0] * 29 + [30.0]
        # lookback 30 → enough history; spike detected.
        assert vix_regime(_series(vals), lookback=30) == "high_vol"


class TestVixSnapshot:
    def test_empty_series(self):
        snap = vix_snapshot(pd.Series(dtype=float))
        assert snap["value"] is None
        assert snap["median"] is None
        assert snap["regime"] == "unknown"
        assert snap["lookback"] == DEFAULT_LOOKBACK

    def test_short_series_value_but_no_median(self):
        snap = vix_snapshot(_series([15.0] * 10), lookback=60)
        assert snap["value"] == 15.0
        assert snap["median"] is None
        assert snap["regime"] == "unknown"

    def test_full_snapshot(self):
        vals = [15.0] * 79 + [30.0]
        snap = vix_snapshot(_series(vals), lookback=60)
        assert snap["value"] == 30.0
        assert snap["median"] == 15.0
        assert snap["regime"] == "high_vol"
        assert snap["lookback"] == 60
