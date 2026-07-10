"""Smoke tests for the pure sparkline geometry helper (web/utils)."""
from __future__ import annotations

from price_predictor.web.utils.sparkline import Sparkline, build_sparkline


class TestBuildSparkline:
    def test_too_few_points_returns_none(self):
        assert build_sparkline([]) is None
        assert build_sparkline([1.0]) is None

    def test_basic_shape(self):
        sp = build_sparkline([1.0, 2.0, 3.0], width=120, height=28)
        assert isinstance(sp, Sparkline)
        assert sp.width == 120
        assert sp.height == 28
        # 3 points → 3 "x,y" tokens.
        assert len(sp.points.split(" ")) == 3
        # Last point sits at the right edge.
        assert sp.last_x == 120

    def test_all_positive_no_baseline(self):
        sp = build_sparkline([0.5, 1.0, 2.0])
        assert sp.has_baseline is False
        assert sp.sign == "pos"

    def test_all_negative_sign_neg(self):
        sp = build_sparkline([-0.5, -1.0, -2.0])
        assert sp.has_baseline is False
        assert sp.sign == "neg"

    def test_range_crossing_zero_has_baseline(self):
        sp = build_sparkline([-1.0, 0.5, 2.0])
        assert sp.has_baseline is True
        # Baseline y is within the drawable canvas.
        assert 0 <= sp.baseline_y <= sp.height

    def test_last_value_zero_is_neutral(self):
        sp = build_sparkline([1.0, -1.0, 0.0])
        assert sp.sign == "neutral"

    def test_flat_line_centers_vertically(self):
        sp = build_sparkline([2.0, 2.0, 2.0], height=28, pad=3)
        # span == 0 → every y is the vertical center (pad + inner_h/2).
        ys = {tok.split(",")[1] for tok in sp.points.split(" ")}
        assert len(ys) == 1  # all identical

    def test_higher_value_has_smaller_y(self):
        # SVG y grows downward → the max value should map to the smallest y.
        sp = build_sparkline([1.0, 5.0])
        y_first = float(sp.points.split(" ")[0].split(",")[1])
        y_last = float(sp.points.split(" ")[1].split(",")[1])
        assert y_last < y_first
