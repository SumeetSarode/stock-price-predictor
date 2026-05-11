"""Tests for the C7 hand-rolled extensions in candlestick_patterns:

- Tweezer Top / Tweezer Bottom (2-bar matched-extreme reversals)
- Rising Window / Falling Window (ATR-filtered gaps)
- Nison next-day confirmation gate (hammer / hanging_man / inverted_hammer
  / shooting_star)

These extensions piggyback on `detect_recent_patterns`, so the tests
build small OHLCV DataFrames and assert the right hits surface (with the
right direction, confirmed flag, etc.).

We use SHORT df fixtures (5-25 bars) plus an ATR-prep stub when window
detection is being exercised (ATR(14) requires 15+ bars of history).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from price_predictor.analysis.candlestick_patterns import detect_recent_patterns


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _make_df(bars: list[dict]) -> pd.DataFrame:
    """Build a daily-indexed OHLC DataFrame from a list of bar dicts."""
    idx = pd.date_range("2024-01-01", periods=len(bars), freq="D")
    return pd.DataFrame(bars, index=idx)


def _flat_history(n: int, price: float = 100.0, body: float = 0.5) -> list[dict]:
    """Generate `n` neutral bars around `price`. Used to seed ATR with a
    stable scale so window-gap thresholds are predictable.

    Each bar is a small bullish doji-ish bar: open=price-body/2,
    close=price+body/2, range=2*body. ATR(14) settles to ~2*body.
    """
    return [
        {
            "open": price - body / 2,
            "high": price + body,
            "low": price - body,
            "close": price + body / 2,
        }
        for _ in range(n)
    ]


def _names(patterns: list[dict]) -> set[str]:
    return {p["name"] for p in patterns}


def _by_name(patterns: list[dict], name: str) -> list[dict]:
    return [p for p in patterns if p["name"] == name]


# ─────────────────────────────────────────────────────────────────────
# Tweezer Top / Bottom
# ─────────────────────────────────────────────────────────────────────
class TestTweezerTop:
    def test_classic_tweezer_top(self):
        bars = _flat_history(5, price=100, body=0.5)
        # Bullish bar with high=110
        bars.append({"open": 100, "high": 110, "low": 99.5, "close": 109})
        # Bearish bar with high=110.0 (exact match), small bullish reversal
        bars.append({"open": 109, "high": 110, "low": 100, "close": 101})
        df = _make_df(bars)
        patterns = detect_recent_patterns(df, lookback=3)
        tops = _by_name(patterns, "tweezer_top")
        assert len(tops) == 1
        assert tops[0]["direction"] == "bearish"
        assert tops[0]["confidence"] == 100
        assert tops[0]["bar_index"] == -1

    def test_tweezer_top_within_tolerance_fires(self):
        # Highs differ by 0.04 on ~100-priced bars (0.04% < 0.05% tol).
        bars = _flat_history(5)
        bars.append({"open": 100, "high": 110.00, "low": 99.5, "close": 109})
        bars.append({"open": 109, "high": 110.04, "low": 100, "close": 101})
        df = _make_df(bars)
        assert "tweezer_top" in _names(detect_recent_patterns(df, lookback=3))

    def test_tweezer_top_outside_tolerance_does_not_fire(self):
        # Highs differ by 0.20 on ~100-priced bars (0.20% > 0.05% tol).
        bars = _flat_history(5)
        bars.append({"open": 100, "high": 110.00, "low": 99.5, "close": 109})
        bars.append({"open": 109, "high": 110.20, "low": 100, "close": 101})
        df = _make_df(bars)
        assert "tweezer_top" not in _names(detect_recent_patterns(df, lookback=3))

    def test_tweezer_top_requires_color_reversal(self):
        # Both bars bullish (open<close) — even with matching highs, no signal.
        bars = _flat_history(5)
        bars.append({"open": 100, "high": 110, "low": 99.5, "close": 109})
        bars.append({"open": 100.5, "high": 110, "low": 100, "close": 109.5})
        df = _make_df(bars)
        assert "tweezer_top" not in _names(detect_recent_patterns(df, lookback=3))

    def test_tweezer_top_scales_with_price(self):
        # Same 0.04% relative match on ~5000-priced bars (high stocks).
        bars = _flat_history(5, price=5000, body=10)
        bars.append({"open": 5000, "high": 5500.0, "low": 4990, "close": 5450})
        bars.append({"open": 5450, "high": 5502.0, "low": 5000, "close": 5050})
        # diff=2.0 on 5500 ~= 0.036% < 0.05% tol
        df = _make_df(bars)
        assert "tweezer_top" in _names(detect_recent_patterns(df, lookback=3))


class TestTweezerBottom:
    def test_classic_tweezer_bottom(self):
        bars = _flat_history(5)
        # Bearish bar with low=90
        bars.append({"open": 100, "high": 100.5, "low": 90, "close": 91})
        # Bullish bar with low=90.0 (exact match)
        bars.append({"open": 91, "high": 100, "low": 90, "close": 99})
        df = _make_df(bars)
        bottoms = _by_name(detect_recent_patterns(df, lookback=3), "tweezer_bottom")
        assert len(bottoms) == 1
        assert bottoms[0]["direction"] == "bullish"
        assert bottoms[0]["confidence"] == 100

    def test_tweezer_bottom_requires_color_reversal(self):
        # Both bearish — no signal even with matching lows.
        bars = _flat_history(5)
        bars.append({"open": 100, "high": 100.5, "low": 90, "close": 91})
        bars.append({"open": 92, "high": 92.5, "low": 90, "close": 90.5})
        df = _make_df(bars)
        assert "tweezer_bottom" not in _names(detect_recent_patterns(df, lookback=3))


# ─────────────────────────────────────────────────────────────────────
# Rising / Falling Window (ATR-filtered gaps)
# ─────────────────────────────────────────────────────────────────────
class TestRisingWindow:
    def test_classic_rising_window(self):
        # 14 stable bars (ATR settles to ~1.0), then a clean gap up of 5.
        bars = _flat_history(14, price=100, body=0.5)
        bars.append({"open": 100, "high": 100.5, "low": 99.5, "close": 100})
        bars.append({"open": 106, "high": 107, "low": 105, "close": 106.5})
        df = _make_df(bars)
        windows = _by_name(detect_recent_patterns(df, lookback=3), "rising_window")
        assert len(windows) == 1
        assert windows[0]["direction"] == "bullish"
        assert windows[0]["bar_index"] == -1

    def test_micro_gap_below_atr_threshold_does_not_fire(self):
        # ATR ~ 1.0; gap = 0.05 (< 0.5 * ATR). Should NOT fire.
        bars = _flat_history(14, price=100, body=0.5)
        bars.append({"open": 100, "high": 100.5, "low": 99.5, "close": 100})
        bars.append({"open": 100.6, "high": 101.0, "low": 100.55, "close": 100.8})
        df = _make_df(bars)
        assert "rising_window" not in _names(detect_recent_patterns(df, lookback=3))

    def test_overlapping_bars_no_window(self):
        # No gap at all; bars overlap. No rising_window.
        bars = _flat_history(14, price=100, body=0.5)
        bars.append({"open": 100, "high": 102, "low": 99, "close": 101})
        bars.append({"open": 101, "high": 103, "low": 100, "close": 102})
        df = _make_df(bars)
        assert "rising_window" not in _names(detect_recent_patterns(df, lookback=3))


class TestFallingWindow:
    def test_classic_falling_window(self):
        bars = _flat_history(14, price=100, body=0.5)
        bars.append({"open": 100, "high": 100.5, "low": 99.5, "close": 100})
        bars.append({"open": 94, "high": 94.5, "low": 93, "close": 93.5})
        df = _make_df(bars)
        windows = _by_name(detect_recent_patterns(df, lookback=3), "falling_window")
        assert len(windows) == 1
        assert windows[0]["direction"] == "bearish"


class TestWindowATRGate:
    def test_returns_no_windows_when_history_too_short(self):
        # Only 10 bars total < ATR(14) period; window scan must skip
        # silently rather than fire on uncomputable ATR.
        bars = [
            {"open": 100, "high": 100.5, "low": 99.5, "close": 100} for _ in range(8)
        ]
        bars.append({"open": 100, "high": 100.5, "low": 99.5, "close": 100})
        bars.append({"open": 110, "high": 111, "low": 109, "close": 110})  # huge gap
        df = _make_df(bars)
        # We don't blow up; we just don't surface a window.
        names = _names(detect_recent_patterns(df, lookback=3))
        assert "rising_window" not in names
        assert "falling_window" not in names


# ─────────────────────────────────────────────────────────────────────
# Nison next-day confirmation
# ─────────────────────────────────────────────────────────────────────
class TestNisonConfirmation:
    """Hammer / shooting-star flow: build a trend stub + the reversal bar +
    a confirming OR contradicting next bar, then check `confirmed`."""

    def _hammer_setup(self, next_close: float) -> pd.DataFrame:
        """Downtrend stub + hammer at index -2 + next-bar at index -1.

        Hammer close = 101. next_close > 101 should set confirmed=True.
        """
        rows = []
        for i in range(15):
            close = 130 - i * 1.5
            rows.append({"open": close + 3, "high": close + 3.5,
                         "low": close - 0.3, "close": close})
        rows.append({"open": 100, "high": 101.2, "low": 95, "close": 101})
        rows.append({"open": 101, "high": next_close + 1, "low": 100,
                     "close": next_close})
        return _make_df(rows)

    def test_confirmed_true_when_next_close_higher(self):
        df = self._hammer_setup(next_close=103)  # 103 > 101 -> confirmed
        hammers = _by_name(detect_recent_patterns(df, lookback=3), "hammer")
        assert len(hammers) == 1
        assert hammers[0]["confirmed"] is True

    def test_confirmed_false_when_next_close_lower(self):
        df = self._hammer_setup(next_close=99)  # 99 < 101 -> NOT confirmed
        hammers = _by_name(detect_recent_patterns(df, lookback=3), "hammer")
        assert len(hammers) == 1
        assert hammers[0]["confirmed"] is False

    def test_confirmed_none_when_pattern_is_last_bar(self):
        # Hammer on the very last bar => no next bar to confirm.
        rows = []
        for i in range(15):
            close = 130 - i * 1.5
            rows.append({"open": close + 3, "high": close + 3.5,
                         "low": close - 0.3, "close": close})
        rows.append({"open": 100, "high": 101.2, "low": 95, "close": 101})
        df = _make_df(rows)
        hammers = _by_name(detect_recent_patterns(df, lookback=2), "hammer")
        assert len(hammers) == 1
        assert hammers[0]["confirmed"] is None

    def test_shooting_star_bearish_confirmation(self):
        # Uptrend stub + shooting star at -2 + bar that closes lower at -1.
        rows = []
        for i in range(15):
            close = 70 + i * 1.5
            rows.append({"open": close - 3, "high": close + 0.3,
                         "low": close - 3.5, "close": close})
        # Shooting star: small body at bottom of bar, long upper shadow.
        rows.append({"open": 100, "high": 105, "low": 99.8, "close": 99.9})
        # Confirming next bar: closes BELOW the star's close (99.9).
        rows.append({"open": 99.5, "high": 99.6, "low": 97, "close": 97.5})
        df = _make_df(rows)
        stars = _by_name(detect_recent_patterns(df, lookback=3), "shooting_star")
        assert len(stars) == 1
        assert stars[0]["confirmed"] is True

    def test_non_reversal_pattern_has_no_confirmed_key(self):
        """Doji (or any non-{hammer,hanging_man,inverted_hammer,shooting_star})
        must NOT carry a `confirmed` key in the output dict."""
        # A simple doji series — body-tiny bars; CDLDOJI fires when body is
        # small relative to recent average body.
        rows = _flat_history(15, price=100, body=2.0)
        # Insert a clear doji bar.
        rows.append({"open": 100, "high": 102, "low": 98, "close": 100.05})
        df = _make_df(rows)
        patterns = detect_recent_patterns(df, lookback=3)
        for p in patterns:
            if p["name"] not in (
                "hammer", "hanging_man", "inverted_hammer", "shooting_star"
            ):
                assert "confirmed" not in p, (
                    f"non-reversal pattern {p['name']} leaked a confirmed key"
                )


# ─────────────────────────────────────────────────────────────────────
# Integration: dispatcher still emits the existing TA-Lib patterns AND
# the new C7 ones, sorted, with the expected dict shape.
# ─────────────────────────────────────────────────────────────────────
class TestDispatcherIntegration:
    def test_output_dict_shape_for_new_patterns(self):
        bars = _flat_history(14, price=100, body=0.5)
        bars.append({"open": 100, "high": 100.5, "low": 99.5, "close": 100})
        bars.append({"open": 106, "high": 107, "low": 105, "close": 106.5})
        df = _make_df(bars)
        windows = _by_name(detect_recent_patterns(df, lookback=3), "rising_window")
        assert len(windows) == 1
        w = windows[0]
        assert set(w.keys()) == {
            "name", "bar_date", "bar_index", "direction", "confidence",
        }
        assert w["confidence"] == 100
        assert w["direction"] == "bullish"

    def test_results_are_sorted_oldest_first(self):
        # Build a series with both a tweezer and a window appearing on
        # different bars. Verify dispatcher sort order is preserved.
        bars = _flat_history(14, price=100, body=0.5)
        # Bar index -3: tweezer top setup pair starts here
        bars.append({"open": 100, "high": 110, "low": 99, "close": 109})  # -3 bullish
        bars.append({"open": 109, "high": 110, "low": 100, "close": 101})  # -2 bearish (tweezer hits at -2)
        # Bar -1: gap up rising window
        bars.append({"open": 110, "high": 111, "low": 108, "close": 110})  # -1 (gap from 101 close, low 108 > prev high 110? no)
        # Adjust: previous bar high was 110, so we need low > 110.
        bars[-1] = {"open": 115, "high": 116, "low": 114, "close": 115}
        df = _make_df(bars)
        patterns = detect_recent_patterns(df, lookback=4)
        # Indices in patterns must be monotonically non-decreasing.
        idxs = [p["bar_index"] for p in patterns]
        assert idxs == sorted(idxs), f"not sorted: {idxs}"


# ─────────────────────────────────────────────────────────────────────
# Defensive: scanners gracefully handle pathological data
# ─────────────────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_zero_avg_close_skipped_in_tweezer(self):
        """Defensive: if avg_close <= 0 (theoretical / corrupted data), skip."""
        bars = _flat_history(5)
        # Two bars with negative+positive prices that average to zero.
        # Realistically this never happens, but the guard exists.
        bars.append({"open": -1.0, "high": 0.0, "low": -2.0, "close": 1.0})
        bars.append({"open": 1.0, "high": 0.0, "low": -2.0, "close": -1.0})
        df = _make_df(bars)
        # No crash.
        patterns = detect_recent_patterns(df, lookback=3)
        assert isinstance(patterns, list)

    def test_constant_bars_no_atr_no_window(self):
        """All-equal bars => ATR=0, no window can fire."""
        bars = [{"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}
                for _ in range(20)]
        df = _make_df(bars)
        names = _names(detect_recent_patterns(df, lookback=5))
        assert "rising_window" not in names
        assert "falling_window" not in names


# ─────────────────────────────────────────────────────────────────────
# Property-style: our pattern names are stable + not colliding with TA-Lib
# ─────────────────────────────────────────────────────────────────────
def test_handrolled_names_dont_collide_with_talib_registry():
    """The 4 hand-rolled pattern names must NOT clash with any TA-Lib
    pretty name in CDL_PATTERNS — otherwise downstream consumers can't
    tell hand-rolled hits apart from TA-Lib hits."""
    from price_predictor.analysis.candlestick_patterns import CDL_PATTERNS
    handrolled = {"tweezer_top", "tweezer_bottom", "rising_window", "falling_window"}
    overlap = handrolled & set(CDL_PATTERNS.keys())
    assert overlap == set(), f"name collision: {overlap}"


@pytest.mark.parametrize(
    "name,expected_direction",
    [
        ("tweezer_top", "bearish"),
        ("tweezer_bottom", "bullish"),
        ("rising_window", "bullish"),
        ("falling_window", "bearish"),
    ],
)
def test_handrolled_directions_are_pinned(name: str, expected_direction: str):
    """Compile-time assertion that we never accidentally flip a direction
    through a refactor. Direction is part of the public contract; flipping
    it silently would mis-feed the gating layer."""
    # Trigger each pattern with a bespoke fixture and assert direction.
    if name == "tweezer_top":
        bars = _flat_history(5)
        bars.append({"open": 100, "high": 110, "low": 99.5, "close": 109})
        bars.append({"open": 109, "high": 110, "low": 100, "close": 101})
    elif name == "tweezer_bottom":
        bars = _flat_history(5)
        bars.append({"open": 100, "high": 100.5, "low": 90, "close": 91})
        bars.append({"open": 91, "high": 100, "low": 90, "close": 99})
    elif name == "rising_window":
        bars = _flat_history(14, body=0.5)
        bars.append({"open": 100, "high": 100.5, "low": 99.5, "close": 100})
        bars.append({"open": 106, "high": 107, "low": 105, "close": 106.5})
    else:  # falling_window
        bars = _flat_history(14, body=0.5)
        bars.append({"open": 100, "high": 100.5, "low": 99.5, "close": 100})
        bars.append({"open": 94, "high": 94.5, "low": 93, "close": 93.5})

    df = _make_df(bars)
    hits = _by_name(detect_recent_patterns(df, lookback=3), name)
    assert len(hits) == 1
    assert hits[0]["direction"] == expected_direction


def test_atr_propagation_into_array_is_finite():
    """Sanity: the ATR series inside _scan_windows shouldn't ever produce
    NaN signals that leak. With clean float64 input and >= 15 bars, ATR
    should be finite from index 14 onward; earlier indices are nan but
    the scan starts at first=max(start_idx, 1) so we touch bar 1+ which
    is nan-clean by the time we actually look (we only look from start_idx).

    This test makes sure that even when we DO touch a nan-ATR bar (small
    df, lookback bigger than ATR-warmup window), no false signals fire."""
    # 14 bars of history -> ATR(14) needs 14 bars warmup so atr[14] is
    # the FIRST non-nan value. With n=15 and lookback=10, indices 5..14
    # are scanned; only i=14 has finite ATR.
    bars = _flat_history(13, body=0.5)
    bars.append({"open": 100, "high": 100.5, "low": 99.5, "close": 100})
    bars.append({"open": 110, "high": 111, "low": 109, "close": 110})
    df = _make_df(bars)
    # No exception, no false-fire on nan bars.
    patterns = detect_recent_patterns(df, lookback=10)
    # Final bar (i=14) IS a real gap with finite ATR -> rising_window allowed.
    rising = _by_name(patterns, "rising_window")
    # We just want *finite* (no nan, no inf) results everywhere.
    for p in patterns:
        assert isinstance(p["bar_index"], int)
        assert isinstance(p["confidence"], int)
    # And the gap bar gets surfaced once finite ATR is available:
    assert any(p["bar_index"] == -1 for p in rising)


# Suppress unused-import warning when np is only imported for clarity.
_ = np
