"""Unit tests for the candlestick-chart SVG geometry builder."""
from __future__ import annotations

import math

from price_predictor.web.utils.candle_chart import (
    CandleChart,
    build_candle_chart,
)


def _bar(o, h, low, c, highlight=False):
    return {"open": o, "high": h, "low": low, "close": c, "highlight": highlight}


class TestBuildCandleChart:
    def test_empty_returns_none(self):
        assert build_candle_chart([]) is None

    def test_nan_value_returns_none(self):
        bars = [_bar(10, 12, 9, 11), _bar(11, float("nan"), 10, 12)]
        assert build_candle_chart(bars) is None

    def test_missing_key_returns_none(self):
        assert build_candle_chart([{"open": 1, "high": 2, "low": 0}]) is None

    def test_single_bar_ok(self):
        chart = build_candle_chart([_bar(10, 12, 9, 11)])
        assert isinstance(chart, CandleChart)
        assert len(chart.candles) == 1

    def test_bullish_flag(self):
        chart = build_candle_chart([_bar(10, 12, 9, 11)])  # close > open
        assert chart.candles[0].bullish is True

    def test_bearish_flag(self):
        chart = build_candle_chart([_bar(11, 12, 9, 10)])  # close < open
        assert chart.candles[0].bullish is False

    def test_highlight_propagates(self):
        chart = build_candle_chart([
            _bar(10, 12, 9, 11, highlight=False),
            _bar(11, 13, 10, 12, highlight=True),
        ])
        assert chart.candles[0].highlighted is False
        assert chart.candles[1].highlighted is True

    def test_flat_range_no_div_by_zero(self):
        # All identical -> span 0 -> everything centered vertically.
        chart = build_candle_chart([_bar(10, 10, 10, 10), _bar(10, 10, 10, 10)])
        assert chart is not None
        mid = chart.height / 2
        for c in chart.candles:
            assert math.isclose(c.wick_top, mid) or c.wick_top == mid

    def test_coordinates_within_canvas(self):
        bars = [_bar(10, 15, 8, 12), _bar(12, 18, 11, 9), _bar(9, 11, 7, 10)]
        chart = build_candle_chart(bars, width=200, height=100, pad=6)
        for c in chart.candles:
            assert 0 <= c.x <= 200
            assert 0 <= c.wick_top <= 100
            assert 0 <= c.wick_bottom <= 100
            assert c.body_x >= 0
            assert c.body_w > 0

    def test_body_height_floored(self):
        # A doji (open == close) should still get a visible (>=1) body.
        chart = build_candle_chart([_bar(10, 12, 8, 10)])
        assert chart.candles[0].body_h >= 1.0

    def test_high_maps_above_low(self):
        # SVG y grows downward, so the high (wick_top) must have a SMALLER
        # y than the low (wick_bottom).
        chart = build_candle_chart([_bar(10, 20, 5, 15)])
        c = chart.candles[0]
        assert c.wick_top < c.wick_bottom

    def test_levels_included_and_expand_range(self):
        bars = [_bar(100, 102, 99, 101)]
        # A level far below the candle range must still render and pull the
        # min down (so it stays visible on-canvas).
        chart = build_candle_chart(bars, levels={"support": 50.0})
        assert len(chart.levels) == 1
        assert chart.levels[0].label == "support"
        assert chart.levels[0].value == 50.0
        assert 0 <= chart.levels[0].y <= chart.height

    def test_level_label_underscores_spaced(self):
        chart = build_candle_chart([_bar(10, 12, 9, 11)], levels={"neck_line": 10.5})
        assert chart.levels[0].label == "neck line"

    def test_nan_level_skipped(self):
        chart = build_candle_chart([_bar(10, 12, 9, 11)], levels={"bad": float("nan")})
        assert chart.levels == []


class TestXAxis:
    def _dbar(self, d, o, h, low, c):
        return {"date": d, "open": o, "high": h, "low": low, "close": c}

    def test_no_ticks_without_dates(self):
        chart = build_candle_chart([_bar(10, 12, 9, 11), _bar(11, 13, 10, 12)])
        assert chart.x_ticks == []

    def test_ticks_present_with_dates(self):
        bars = [
            self._dbar("2026-07-01", 10, 12, 9, 11),
            self._dbar("2026-07-02", 11, 13, 10, 12),
            self._dbar("2026-07-03", 12, 14, 11, 13),
        ]
        chart = build_candle_chart(bars)
        assert len(chart.x_ticks) == 3
        assert chart.x_ticks[0].label == "01/07"
        assert chart.x_ticks[-1].label == "03/07"
        # Ticks sit under their candle (same x).
        assert chart.x_ticks[0].x == chart.candles[0].x

    def test_ticks_capped_and_include_ends(self):
        bars = [self._dbar(f"2026-07-{i:02d}", 10, 12, 9, 11) for i in range(1, 21)]
        chart = build_candle_chart(bars, width=380, height=142)
        assert 0 < len(chart.x_ticks) <= 6
        # First and last bars are always labelled.
        assert chart.x_ticks[0].x == chart.candles[0].x
        assert chart.x_ticks[-1].x == chart.candles[-1].x

    def test_dates_reserve_axis_space(self):
        # With dates, candles must stay ABOVE the axis strip (never render
        # into the bottom ~14px reserved for labels).
        bars = [self._dbar("2026-07-01", 10, 20, 5, 15)]
        chart = build_candle_chart(bars, height=110)
        assert chart.candles[0].wick_bottom <= 110 - 14
