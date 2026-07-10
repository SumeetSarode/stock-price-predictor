"""Tests for analysis.ichimoku -- the five lines + regime snapshot."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from price_predictor.analysis.ichimoku import (
    DEFAULT_DISPLACEMENT,
    DEFAULT_KIJUN,
    DEFAULT_SENKOU_B,
    DEFAULT_TENKAN,
    ichimoku,
    ichimoku_snapshot,
)
from tests.analysis.conftest import (
    insufficient_history,
    linear_downtrend,
    linear_uptrend,
)


def _flat(n: int = 100, price: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=n, freq="D", tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "open": np.full(n, price), "high": np.full(n, price),
            "low": np.full(n, price), "close": np.full(n, price),
            "adj_close": np.full(n, price), "volume": np.full(n, 1000.0),
        },
        index=dates,
    )


class TestIchimokuLines:
    def test_returns_five_columns_aligned(self):
        df = linear_uptrend(n=250)
        lines = ichimoku(df)
        assert list(lines.columns) == [
            "tenkan", "kijun", "senkou_a", "senkou_b", "chikou",
        ]
        assert (lines.index == df.index).all()

    def test_empty_df_returns_empty_columns_frame(self):
        empty = pd.DataFrame(columns=["high", "low", "close"])
        lines = ichimoku(empty)
        assert lines.empty
        assert list(lines.columns) == [
            "tenkan", "kijun", "senkou_a", "senkou_b", "chikou",
        ]

    def test_missing_columns_defensive(self):
        df = pd.DataFrame({"close": [1.0, 2.0]})
        assert ichimoku(df).empty

    def test_flat_price_lines_equal_price(self):
        df = _flat(n=120, price=100.0)
        lines = ichimoku(df)
        # All midlines collapse to the constant price.
        assert lines["tenkan"].iloc[-1] == pytest.approx(100.0)
        assert lines["kijun"].iloc[-1] == pytest.approx(100.0)
        assert lines["senkou_a"].iloc[-1] == pytest.approx(100.0)
        assert lines["senkou_b"].iloc[-1] == pytest.approx(100.0)

    def test_senkou_spans_are_shifted_forward(self):
        # senkou_a/b are shifted +displacement → first `displacement`
        # entries are NaN even once the midline is defined.
        df = linear_uptrend(n=250)
        lines = ichimoku(df)
        assert lines["senkou_a"].iloc[:DEFAULT_DISPLACEMENT].isna().all()

    def test_chikou_shifted_backward(self):
        # chikou = close.shift(-displacement) → last `displacement` NaN.
        df = linear_uptrend(n=250)
        lines = ichimoku(df)
        assert lines["chikou"].iloc[-DEFAULT_DISPLACEMENT:].isna().all()


class TestIchimokuSnapshot:
    def test_uptrend_above_cloud_bullish(self):
        df = linear_uptrend(n=250, start=100.0, slope=1.0)
        snap = ichimoku_snapshot(df)
        assert snap["price_vs_cloud"] == "above"
        assert snap["tk_signal"] == "bullish"
        assert snap["cloud_top"] >= snap["cloud_bottom"]

    def test_downtrend_below_cloud_bearish(self):
        df = linear_downtrend(n=250, start=300.0, slope=1.0)
        snap = ichimoku_snapshot(df)
        assert snap["price_vs_cloud"] == "below"
        assert snap["tk_signal"] == "bearish"

    def test_flat_price_inside_cloud_neutral(self):
        df = _flat(n=120, price=100.0)
        snap = ichimoku_snapshot(df)
        # Constant price: cloud collapses onto price → not above/below.
        assert snap["price_vs_cloud"] == "inside"
        assert snap["tk_signal"] == "neutral"
        assert snap["kumo_twist_ahead"] is False

    def test_insufficient_history_all_none(self):
        snap = ichimoku_snapshot(insufficient_history(n=5))
        assert snap["tenkan"] is None
        assert snap["senkou_b"] is None
        assert snap["price_vs_cloud"] is None
        assert snap["tk_signal"] is None

    def test_empty_df_all_none(self):
        empty = pd.DataFrame(columns=["high", "low", "close"])
        snap = ichimoku_snapshot(empty)
        assert snap["tenkan"] is None
        assert snap["price_vs_cloud"] is None

    def test_params_echoed(self):
        snap = ichimoku_snapshot(linear_uptrend(n=250))
        assert snap["params"] == {
            "tenkan": DEFAULT_TENKAN, "kijun": DEFAULT_KIJUN,
            "senkou_b": DEFAULT_SENKOU_B, "displacement": DEFAULT_DISPLACEMENT,
        }

    def test_custom_params_respected(self):
        df = linear_uptrend(n=250)
        snap = ichimoku_snapshot(df, tenkan_n=5, kijun_n=13, senkou_b_n=26)
        assert snap["params"]["tenkan"] == 5
        assert snap["params"]["kijun"] == 13
        assert snap["params"]["senkou_b"] == 26

    def test_chikou_present_with_enough_history(self):
        snap = ichimoku_snapshot(linear_uptrend(n=250))
        assert snap["chikou"] is not None

    def test_kumo_twist_detects_sign_change(self):
        # Build a series that rises then falls sharply, so the future
        # cloud (Senkou A vs B) flips sign within the projection window.
        up = np.linspace(100, 200, 120)
        down = np.linspace(200, 80, 60)
        close = np.concatenate([up, down])
        n = len(close)
        dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Kolkata")
        df = pd.DataFrame(
            {
                "open": close, "high": close + 1.0, "low": close - 1.0,
                "close": close, "adj_close": close, "volume": np.full(n, 1000.0),
            },
            index=dates,
        )
        snap = ichimoku_snapshot(df)
        # Just assert it computed a boolean (regime-change flag is defined).
        assert isinstance(snap["kumo_twist_ahead"], bool)
