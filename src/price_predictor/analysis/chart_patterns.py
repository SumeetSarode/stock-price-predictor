"""Hand-rolled chart pattern detectors for the levels cluster.

THREE PATTERNS ONLY (per design discussion)
===========================================
- Double top / double bottom -- twin peaks/troughs at similar price
- Head and shoulders (regular + inverse) -- 3-peak with center taller
- Triangles (ascending / descending / symmetric) -- converging trendlines

Skipped: cup & handle, flags, pennants, wedges. Too noisy / unreliable
for v1.

CONFIDENCE
==========
Every detection returns a confidence in [0, 1]. The TOOL layer filters
to confidence >= 0.7 before surfacing to the LLM. Below that the noise-
to-signal ratio is too high to help.

APPROACH
========
- Use scipy.signal.find_peaks for swing point detection.
- Geometric checks for each pattern (symmetry, neckline angle, etc.).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# Confidence threshold below which patterns are NOT surfaced
DEFAULT_CONFIDENCE_THRESHOLD = 0.7


@dataclass
class ChartPattern:
    """One detected pattern with metadata."""
    name: str
    confidence: float
    key_levels: dict[str, float]
    bar_indices: list[int]  # which bars in df form the pattern (positive indices)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Helpers ─────────────────────────────────────────────────────────


def _find_swing_highs(df: pd.DataFrame, distance: int = 5) -> np.ndarray:
    """Indices of swing-high bars (local maxima)."""
    if len(df) < distance * 2 + 1:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(df["high"].values, distance=distance)
    return peaks


def _find_swing_lows(df: pd.DataFrame, distance: int = 5) -> np.ndarray:
    """Indices of swing-low bars (local minima)."""
    if len(df) < distance * 2 + 1:
        return np.array([], dtype=int)
    # find_peaks finds maxima -- negate to find minima
    peaks, _ = find_peaks(-df["low"].values, distance=distance)
    return peaks


def _similarity_score(a: float, b: float, tolerance: float = 0.03) -> float:
    """1.0 if a == b, decreasing linearly to 0 at `tolerance` relative diff."""
    if a == 0:
        return 0.0
    diff_pct = abs(a - b) / abs(a)
    if diff_pct >= tolerance:
        return 0.0
    return 1.0 - diff_pct / tolerance


# ── Double top / bottom ────────────────────────────────────────────


def detect_double_top(df: pd.DataFrame) -> ChartPattern | None:
    """Two consecutive swing highs at similar price, separated by a trough."""
    highs = _find_swing_highs(df)
    if len(highs) < 2:
        return None

    # Use the two most recent swing highs
    h1_idx, h2_idx = highs[-2], highs[-1]
    h1, h2 = df.iloc[h1_idx]["high"], df.iloc[h2_idx]["high"]

    # Need a trough between them
    between = df.iloc[h1_idx + 1:h2_idx]
    if between.empty:
        return None
    trough = float(between["low"].min())

    # Confidence: how similar are the peaks, and how deep is the trough
    peak_similarity = _similarity_score(h1, h2, tolerance=0.03)
    if peak_similarity == 0:
        return None
    avg_peak = (h1 + h2) / 2
    trough_depth = (avg_peak - trough) / avg_peak if avg_peak > 0 else 0
    depth_score = min(trough_depth / 0.05, 1.0)  # >=5% drop scores 1.0
    confidence = round(peak_similarity * depth_score, 2)
    if confidence < 0.3:
        return None

    return ChartPattern(
        name="double_top",
        confidence=confidence,
        key_levels={
            "resistance": round(avg_peak, 2),
            "neckline":   round(trough, 2),
            "target":     round(trough - (avg_peak - trough), 2),
        },
        bar_indices=[int(h1_idx), int(h2_idx)],
    )


def detect_double_bottom(df: pd.DataFrame) -> ChartPattern | None:
    """Mirror of double top."""
    lows = _find_swing_lows(df)
    if len(lows) < 2:
        return None
    l1_idx, l2_idx = lows[-2], lows[-1]
    l1, l2 = df.iloc[l1_idx]["low"], df.iloc[l2_idx]["low"]

    between = df.iloc[l1_idx + 1:l2_idx]
    if between.empty:
        return None
    peak = float(between["high"].max())

    trough_similarity = _similarity_score(l1, l2, tolerance=0.03)
    if trough_similarity == 0:
        return None
    avg_trough = (l1 + l2) / 2
    peak_height = (peak - avg_trough) / avg_trough if avg_trough > 0 else 0
    height_score = min(peak_height / 0.05, 1.0)
    confidence = round(trough_similarity * height_score, 2)
    if confidence < 0.3:
        return None

    return ChartPattern(
        name="double_bottom",
        confidence=confidence,
        key_levels={
            "support":  round(avg_trough, 2),
            "neckline": round(peak, 2),
            "target":   round(peak + (peak - avg_trough), 2),
        },
        bar_indices=[int(l1_idx), int(l2_idx)],
    )


# ── Head and shoulders ─────────────────────────────────────────────


def detect_head_shoulders(df: pd.DataFrame) -> ChartPattern | None:
    """Three consecutive swing highs: middle (head) is tallest, two shoulders
    are shorter and roughly equal."""
    highs = _find_swing_highs(df)
    if len(highs) < 3:
        return None
    s1_idx, h_idx, s2_idx = highs[-3], highs[-2], highs[-1]
    s1, head, s2 = (
        df.iloc[s1_idx]["high"],
        df.iloc[h_idx]["high"],
        df.iloc[s2_idx]["high"],
    )

    if not (head > s1 and head > s2):
        return None
    shoulder_similarity = _similarity_score(s1, s2, tolerance=0.05)
    if shoulder_similarity == 0:
        return None

    # Head should be meaningfully taller than shoulders (>=3% above max shoulder)
    max_shoulder = max(s1, s2)
    head_prominence = (head - max_shoulder) / max_shoulder if max_shoulder > 0 else 0
    prominence_score = min(head_prominence / 0.03, 1.0)

    # Neckline = average of the two troughs between (s1, head) and (head, s2)
    trough1 = float(df.iloc[s1_idx + 1:h_idx]["low"].min())
    trough2 = float(df.iloc[h_idx + 1:s2_idx]["low"].min())
    neckline = (trough1 + trough2) / 2

    confidence = round(shoulder_similarity * prominence_score, 2)
    if confidence < 0.3:
        return None

    return ChartPattern(
        name="head_and_shoulders",
        confidence=confidence,
        key_levels={
            "head":     round(float(head), 2),
            "shoulder_left":  round(float(s1), 2),
            "shoulder_right": round(float(s2), 2),
            "neckline": round(neckline, 2),
            "target":   round(neckline - (head - neckline), 2),
        },
        bar_indices=[int(s1_idx), int(h_idx), int(s2_idx)],
    )


def detect_inverse_head_shoulders(df: pd.DataFrame) -> ChartPattern | None:
    """Mirror: three swing lows with middle (head) lowest."""
    lows = _find_swing_lows(df)
    if len(lows) < 3:
        return None
    s1_idx, h_idx, s2_idx = lows[-3], lows[-2], lows[-1]
    s1, head, s2 = (
        df.iloc[s1_idx]["low"],
        df.iloc[h_idx]["low"],
        df.iloc[s2_idx]["low"],
    )

    if not (head < s1 and head < s2):
        return None
    shoulder_similarity = _similarity_score(s1, s2, tolerance=0.05)
    if shoulder_similarity == 0:
        return None

    min_shoulder = min(s1, s2)
    head_prominence = (min_shoulder - head) / min_shoulder if min_shoulder > 0 else 0
    prominence_score = min(head_prominence / 0.03, 1.0)

    peak1 = float(df.iloc[s1_idx + 1:h_idx]["high"].max())
    peak2 = float(df.iloc[h_idx + 1:s2_idx]["high"].max())
    neckline = (peak1 + peak2) / 2

    confidence = round(shoulder_similarity * prominence_score, 2)
    if confidence < 0.3:
        return None

    return ChartPattern(
        name="inverse_head_and_shoulders",
        confidence=confidence,
        key_levels={
            "head":            round(float(head), 2),
            "shoulder_left":   round(float(s1), 2),
            "shoulder_right":  round(float(s2), 2),
            "neckline":        round(neckline, 2),
            "target":          round(neckline + (neckline - head), 2),
        },
        bar_indices=[int(s1_idx), int(h_idx), int(s2_idx)],
    )


# ── Triangles ──────────────────────────────────────────────────────


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (slope, intercept) of a least-squares line through (x, y)."""
    if len(x) < 2:
        return 0.0, float(y[0]) if len(y) > 0 else 0.0
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def detect_triangle(df: pd.DataFrame, min_pivots: int = 4) -> ChartPattern | None:
    """Detect ascending / descending / symmetric triangle from recent pivots.

    Approach:
      - Get the last 4+ swing highs and 4+ swing lows
      - Fit upper trendline through highs, lower trendline through lows
      - Classify by slopes:
          ascending  = upper~flat (slope~0), lower rising
          descending = upper falling, lower~flat
          symmetric  = upper falling, lower rising
    """
    highs = _find_swing_highs(df)
    lows = _find_swing_lows(df)
    if len(highs) < min_pivots // 2 or len(lows) < min_pivots // 2:
        return None

    # Use up to last 5 pivots of each
    h_idx = highs[-5:]
    l_idx = lows[-5:]
    upper_slope, upper_int = _fit_line(
        h_idx.astype(float), df.iloc[h_idx]["high"].values
    )
    lower_slope, lower_int = _fit_line(
        l_idx.astype(float), df.iloc[l_idx]["low"].values
    )

    # Normalize slopes by mean price so threshold is unit-free
    mean_price = float(df["close"].iloc[-len(df) // 4:].mean()) or 1.0
    upper_norm = upper_slope / mean_price
    lower_norm = lower_slope / mean_price

    flat_threshold = 1e-4   # ~0.01% of price per bar = essentially flat
    rising_threshold = 5e-4

    name: str | None = None
    if abs(upper_norm) < flat_threshold and lower_norm > rising_threshold:
        name = "ascending_triangle"
    elif upper_norm < -rising_threshold and abs(lower_norm) < flat_threshold:
        name = "descending_triangle"
    elif upper_norm < -rising_threshold and lower_norm > rising_threshold:
        name = "symmetric_triangle"
    if name is None:
        return None

    # Confidence: how cleanly the slopes match the pattern signature
    confidence = 0.7  # base for matching the shape
    if name == "symmetric_triangle":
        # Bonus if slopes are roughly symmetric in magnitude
        ratio = min(abs(upper_norm), abs(lower_norm)) / max(
            abs(upper_norm), abs(lower_norm)
        )
        confidence = round(0.5 + 0.5 * ratio, 2)
    confidence = max(0.5, min(confidence, 1.0))

    last_x = float(len(df) - 1)
    apex_upper = upper_slope * last_x + upper_int
    apex_lower = lower_slope * last_x + lower_int

    return ChartPattern(
        name=name,
        confidence=confidence,
        key_levels={
            "upper_trendline_now": round(apex_upper, 2),
            "lower_trendline_now": round(apex_lower, 2),
        },
        bar_indices=[int(i) for i in list(h_idx) + list(l_idx)],
    )


# ── Detection driver ───────────────────────────────────────────────


def detect_all_patterns(
    df: pd.DataFrame,
    min_confidence: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[dict]:
    """Run all detectors, filter by confidence, return as list of dicts."""
    results: list[ChartPattern] = []
    for fn in (
        detect_double_top,
        detect_double_bottom,
        detect_head_shoulders,
        detect_inverse_head_shoulders,
        detect_triangle,
    ):
        try:
            r = fn(df)
        except (ValueError, IndexError):
            r = None
        if r is not None and r.confidence >= min_confidence:
            results.append(r)
    return [r.to_dict() for r in results]
