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
        # We have 300 calendar days; function looks at last 52 weeks (~365 cal days)
        ft = fifty_two_week_high_low(df)
        assert ft["high_52w"] is not None
        assert ft["low_52w"] is not None
        assert ft["high_52w"] > ft["low_52w"]

    def test_short_history_uses_what_it_has(self):
        df = linear_uptrend(n=10, start=100, slope=1)
        ft = fifty_two_week_high_low(df)
        # With <52 weeks of bars, we just use whatever we have
        assert ft["high_52w"] is not None

    def test_calendar_window_excludes_old_bars(self):
        # Build a 500-cal-day series. The 52w window should exclude
        # the oldest ~135 bars; max of last 52w should be close to the
        # most recent (highest in an uptrend), NOT the global max
        # (which it would be if we used .iloc[-252:] and n=500).
        df = linear_uptrend(n=500, start=100, slope=1)
        ft = fifty_two_week_high_low(df)
        # Last bar is index 499, close=600, high=601
        # 52 weeks back from last_date: cutoff bar = ~bar 135
        # So the lowest in the 52w window should be > the global low
        global_low = float(df["low"].min())
        assert ft["low_52w"] is not None
        assert ft["low_52w"] > global_low, (
            f"calendar-window low ({ft['low_52w']}) should exclude the "
            f"first ~135 bars; global low is {global_low}"
        )


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
        # R2 = PP + (H - L) = 105 + 10 = 115
        assert p["r2"] == 115.0
        # S2 = PP - (H - L) = 105 - 10 = 95
        assert p["s2"] == 95.0
        # R3 = H + 2*(PP - L) = 110 + 2*5 = 120  (Person 2004)
        assert p["r3"] == 120.0
        # S3 = L - 2*(H - PP) = 100 - 2*5 = 90
        assert p["s3"] == 90.0

    def test_pivots_ladder_ordering(self):
        # R3 > R2 > R1 > PP > S1 > S2 > S3 should hold for any non-degenerate bar
        df = linear_uptrend(n=10)
        df.iloc[-1, df.columns.get_loc("high")] = 250.0
        df.iloc[-1, df.columns.get_loc("low")] = 200.0
        df.iloc[-1, df.columns.get_loc("close")] = 230.0
        p = classic_pivots(df)
        assert p["r3"] > p["r2"] > p["r1"] > p["pp"] > p["s1"] > p["s2"] > p["s3"]


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

    def test_snapshot_includes_vwap_block(self):
        df = linear_uptrend(n=100, start=100, slope=1)
        snap = levels_snapshot(df, swing_lookback=30)
        assert "vwap" in snap
        assert "vwap_rolling" in snap["vwap"]
        # 100 bars >> rolling default 20, so VWAP must be populated
        assert snap["vwap"]["vwap_rolling"] is not None
        # Anchor not provided -> anchored stays None
        assert snap["vwap"]["vwap_anchored"] is None
        # And the per-level distance % is exposed for both VWAP variants
        assert "vwap_rolling" in snap["distance_pct"]
        assert "vwap_anchored" in snap["distance_pct"]
        assert snap["distance_pct"]["vwap_rolling"] is not None
        assert snap["distance_pct"]["vwap_anchored"] is None

    def test_snapshot_with_anchor_populates_anchored_vwap(self):
        df = linear_uptrend(n=100, start=100, slope=1)
        anchor = df.index[50].date()
        snap = levels_snapshot(df, swing_lookback=30, vwap_anchor=anchor)
        assert snap["vwap"]["vwap_anchored"] is not None
        assert snap["vwap"]["anchor_date"] == anchor.isoformat()
        assert snap["distance_pct"]["vwap_anchored"] is not None
