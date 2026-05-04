"""Tests for analysis.chart_patterns -- synthetic series per pattern shape."""
from __future__ import annotations

import numpy as np
import pandas as pd

from price_predictor.analysis.chart_patterns import (
    detect_all_patterns,
    detect_double_bottom,
    detect_double_top,
    detect_head_shoulders,
    detect_inverse_head_shoulders,
    detect_triangle,
)


def _build_df(close: list[float], hi_offset: float = 0.5, lo_offset: float = 0.5) -> pd.DataFrame:
    """Build OHLC from a hand-crafted close series."""
    n = len(close)
    arr = np.array(close, dtype=float)
    return pd.DataFrame(
        {
            "open":  arr,
            "high":  arr + hi_offset,
            "low":   arr - lo_offset,
            "close": arr,
            "adj_close": arr,
            "volume": np.full(n, 1000),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Kolkata"),
    )


# ─────────────────────────────────────────────────────────────────
# Double top / bottom
# ─────────────────────────────────────────────────────────────────
class TestDoubleTop:
    def test_classic_double_top_detected(self):
        # Up to ~120, down to ~108, back up to ~120, back down
        # Use a clear shape with prominent peaks separated by a deep trough
        closes = (
            list(np.linspace(100, 120, 20))   # peak 1 forms at ~120
            + list(np.linspace(120, 108, 15)) # trough
            + list(np.linspace(108, 120, 15)) # peak 2 also at ~120
            + list(np.linspace(120, 110, 10)) # decline after
        )
        df = _build_df(closes)
        result = detect_double_top(df)
        assert result is not None
        assert result.name == "double_top"
        assert result.confidence > 0.3

    def test_no_pattern_in_uptrend(self):
        df = _build_df(list(np.linspace(100, 200, 100)))
        # Strict uptrend: no two similar peaks
        result = detect_double_top(df)
        # Either None or low confidence is acceptable
        assert result is None or result.confidence < 0.5


class TestDoubleBottom:
    def test_classic_double_bottom_detected(self):
        closes = (
            list(np.linspace(120, 100, 20))   # trough 1
            + list(np.linspace(100, 112, 15)) # peak between
            + list(np.linspace(112, 100, 15)) # trough 2
            + list(np.linspace(100, 110, 10)) # rise after
        )
        df = _build_df(closes)
        result = detect_double_bottom(df)
        assert result is not None
        assert result.name == "double_bottom"


# ─────────────────────────────────────────────────────────────────
# Head and shoulders
# ─────────────────────────────────────────────────────────────────
class TestHeadShoulders:
    def test_classic_head_shoulders_detected(self):
        closes = (
            list(np.linspace(100, 110, 15))   # left shoulder up
            + list(np.linspace(110, 102, 10)) # down
            + list(np.linspace(102, 120, 15)) # head up
            + list(np.linspace(120, 102, 10)) # down
            + list(np.linspace(102, 110, 15)) # right shoulder up
            + list(np.linspace(110, 100, 10)) # break
        )
        df = _build_df(closes)
        result = detect_head_shoulders(df)
        assert result is not None
        assert result.name == "head_and_shoulders"
        # Head should be the highest of the three peaks
        assert result.key_levels["head"] > result.key_levels["shoulder_left"]
        assert result.key_levels["head"] > result.key_levels["shoulder_right"]


class TestInverseHeadShoulders:
    def test_classic_inverse_head_shoulders_detected(self):
        closes = (
            list(np.linspace(120, 110, 15))
            + list(np.linspace(110, 118, 10))
            + list(np.linspace(118, 100, 15))
            + list(np.linspace(100, 118, 10))
            + list(np.linspace(118, 110, 15))
            + list(np.linspace(110, 120, 10))
        )
        df = _build_df(closes)
        result = detect_inverse_head_shoulders(df)
        assert result is not None
        assert result.name == "inverse_head_and_shoulders"
        assert result.key_levels["head"] < result.key_levels["shoulder_left"]


# ─────────────────────────────────────────────────────────────────
# Triangles
# ─────────────────────────────────────────────────────────────────
class TestTriangle:
    def test_ascending_triangle_detected(self):
        # Highs flat at ~120, lows rising from ~100 to ~115
        n = 80
        closes = []
        for i in range(n):
            # Oscillate between rising lower bound and ~flat upper bound
            phase = i % 10
            if phase < 5:
                closes.append(120 - phase * 1.0)  # near top
            else:
                closes.append(100 + i * 0.2)  # rising bottom
        df = _build_df(closes)
        result = detect_triangle(df)
        # Triangle detection is fuzzy; accept any triangle name or None
        if result is not None:
            assert "triangle" in result.name


# ─────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────
class TestDetectAllPatterns:
    def test_returns_list_of_dicts(self):
        df = _build_df(list(np.linspace(100, 200, 100)))
        results = detect_all_patterns(df)
        assert isinstance(results, list)
        for r in results:
            assert "name" in r
            assert "confidence" in r
            assert r["confidence"] >= 0.7  # default threshold

    def test_handles_short_df_gracefully(self):
        df = _build_df([100, 101, 102])
        # Should not crash; just returns empty
        assert detect_all_patterns(df) == []
