"""Tests for analysis.levels -- swing/52w/pivots/snapshot."""
from __future__ import annotations

import pandas as pd

from price_predictor.analysis.levels import (
    classic_pivots,
    fifty_two_week_high_low,
    levels_snapshot,
    swing_high_low,
)
from tests.analysis.conftest import insufficient_history, linear_uptrend


class TestSwingHighLow:
    def test_uptrend_swing_high_is_recent_high(self):
        df = linear_uptrend(n=100, start=100, slope=1)
        sw = swing_high_low(df, lookback=30)
        # Last 30 bars: closes ~71..200, last close = 200 -> high = 201
        assert sw["swing_high"] is not None
        assert sw["swing_high"] == 201.0
        assert sw["swing_low"] is not None

    def test_empty_df_none(self):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "adj_close", "volume"])
        sw = swing_high_low(empty, lookback=30)
        assert sw["swing_high"] is None
        assert sw["swing_low"] is None


class TestFiftyTwoWeekHighLow:
    def test_full_year_high_is_max_of_window(self):
        df = linear_uptrend(n=300, start=100, slope=1)
        # We have 300 bars; function looks at last 252
        ft = fifty_two_week_high_low(df)
        assert ft["high_52w"] is not None
        assert ft["low_52w"] is not None
        assert ft["high_52w"] > ft["low_52w"]

    def test_short_history_uses_what_it_has(self):
        df = linear_uptrend(n=10, start=100, slope=1)
        ft = fifty_two_week_high_low(df)
        # With <252 bars, we just use whatever we have
        assert ft["high_52w"] is not None


class TestClassicPivots:
    def test_pivots_from_known_bar(self):
        # Hand-crafted last bar: H=110, L=100, C=105
        df = linear_uptrend(n=10)
        df.iloc[-1, df.columns.get_loc("high")] = 110.0
        df.iloc[-1, df.columns.get_loc("low")] = 100.0
        df.iloc[-1, df.columns.get_loc("close")] = 105.0
        p = classic_pivots(df)
        # PP = (110+100+105)/3 = 105
        assert p["pp"] == 105.0
        # R1 = 2*PP - L = 210 - 100 = 110
        assert p["r1"] == 110.0
        # S1 = 2*PP - H = 210 - 110 = 100
        assert p["s1"] == 100.0


class TestLevelsSnapshot:
    def test_snapshot_includes_distance_pct(self):
        df = linear_uptrend(n=100, start=100, slope=1)
        snap = levels_snapshot(df, swing_lookback=30)
        assert snap["close"] is not None
        assert snap["swing"]["swing_high"] is not None
        assert snap["distance_pct"]["swing_high"] is not None
        # Latest close is at the swing high in a strict uptrend -> distance ~0
        assert abs(snap["distance_pct"]["swing_high"]) < 2

    def test_short_history_graceful(self):
        df = insufficient_history(n=5)
        snap = levels_snapshot(df, swing_lookback=30)
        # Even with short history, swing_high uses what it can
        assert snap["close"] is not None
        assert snap["swing"]["swing_high"] is not None
