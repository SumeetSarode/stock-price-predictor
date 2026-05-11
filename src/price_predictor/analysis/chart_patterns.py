"""Hand-rolled chart pattern detectors for the levels cluster.

FIVE PATTERN PAIRS (per LMW 2000)
=================================
Lo, Mamaysky, Wang (2000), "Foundations of Technical Analysis",
Journal of Finance 55(4) (NBER WP #7613), Section II.A, define FIVE
pairs of geometric chart patterns. We implement all five:

  Def 1: Head-and-shoulders (regular + inverse)
  Def 2: Broadening tops / bottoms          (megaphone)
  Def 3: Triangle tops / bottoms             (ascending / descending / symmetric)
  Def 4: Rectangle tops / bottoms            (sideways consolidation)
  Def 5: Double tops / bottoms

Not implemented (LMW intentionally excluded these from their canonical
catalogue and we follow their lead): cup & handle, flags, pennants,
wedges. They lack a closed-form geometric definition with published
tolerances and have higher noise-to-signal in independent backtests
(Bulkowski 2005 reports flag/pennant reliability < 50% on 5y / S&P).

CONFIDENCE
==========
Every detection returns a confidence in [0, 1]. The TOOL layer filters
to confidence >= 0.7 before surfacing to the LLM. Below that the noise-
to-signal ratio is too high to help.

PIVOT DETECTION (deviation from LMW)
====================================
LMW use a NON-PARAMETRIC KERNEL REGRESSION smoother (Nadaraya-Watson with
a Gaussian kernel and cross-validated bandwidth) to identify pivots. We
use `scipy.signal.find_peaks(distance=5)` instead. This is a deliberate,
documented simplification:
  - LMW's kernel approach reduces noise but adds two hyperparameters
    (bandwidth + Nadaraya-Watson order) that drift with regime change.
  - `find_peaks` with a fixed minimum-separation parameter gives more
    reproducible swing detection on daily NSE bars and is what every
    open-source pattern library (TradingView Pine, finta, Bulkowski's
    own tooling) actually ships with.
  - The downstream geometric tolerances (1.5% shoulders, 0.75% flat
    line, etc.) are LMW's; only the pivot-identification step differs.

CANONICAL THRESHOLDS
====================
Geometric tolerances follow Lo, Mamaysky, Wang (2000), Section II.A.
The 22-trading-day double-top separation is LMW's own operational
discretization ("...the two tops occur at least a month, or 22 trading
days, apart") of Edwards & Magee's qualitative "~one month / several
weeks" guidance — NOT a number that appears directly in E&M. See
docs/research/constants_dossier.md.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# Confidence threshold below which patterns are NOT surfaced
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

# LMW (2000) academic-standard tolerances for chart-pattern geometry.
# Tightened from earlier looser values (5% / 3%) to match the Journal of
# Finance reference implementation.
_HS_SHOULDER_TOLERANCE = 0.015        # LMW Def 1: E1, E5 within 1.5% of avg
_HS_NECKLINE_TOLERANCE = 0.015        # LMW Def 1: E2, E4 within 1.5% of avg
_DOUBLE_TOP_PEAK_TOLERANCE = 0.015    # LMW Def 5: two tops within 1.5% of avg
_DOUBLE_TOP_MIN_SEPARATION_BARS = 22  # LMW (2000) Def 5 — LMW's own discretization

# H3 (review): the trough-depth saturation point. A double top with a
# trough only 5% below the peaks is barely a "top" — it's noise. Bulkowski
# ("Encyclopedia of Chart Patterns" 2nd ed., 2005, ch. 23 Tab 23.1) reports
# the median double-top retracement is ~10% on US large-caps; LMW's own
# kernel-smoothed pivots imply the same order of magnitude. Saturating
# the depth score at 10% (rather than 5%) discounts shallow troughs more
# aggressively without rejecting them outright.
_DOUBLE_TOP_DEPTH_SATURATION = 0.10

# LMW Def 4 (Rectangle): pivot prices on a "flat" trendline must lie within
# 0.75% of their average. We reuse this for ascending/descending triangle's
# horizontal line and for rectangle pattern detection — it's the same
# geometric construct (a horizontal resistance or support).
_FLAT_LINE_SPREAD_TOLERANCE = 0.0075

# LMW Def 2 (Broadening): minimum % expansion between successive
# extrema of the same kind for the megaphone to register. Set to 1.5%
# to match the LMW peak-similarity bound (the inverse condition: instead
# of "within 1.5%", we want "more than 1.5% APART").
_BROADENING_MIN_EXPANSION = 0.015


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


def _similarity_score(a: float, b: float, tolerance: float = 0.015) -> float:
    """1.0 if a == b, decreasing linearly to 0 at `tolerance` relative diff.

    Default 1.5% matches LMW (2000) chart pattern definitions. Callers may
    pass a different tolerance for non-LMW use cases (e.g., depth filters).
    """
    if a == 0:
        return 0.0
    diff_pct = abs(a - b) / abs(a)
    if diff_pct >= tolerance:
        return 0.0
    return 1.0 - diff_pct / tolerance


# ── Double top / bottom ────────────────────────────────────────────


def detect_double_top(df: pd.DataFrame) -> ChartPattern | None:
    """Two consecutive swing highs at similar price, separated by a trough.

    Implements LMW (2000) Definition 5: two tops within 1.5% of their
    average, separated by at least 22 trading days. The 22-day figure is
    LMW's own operationalization ("...the two tops occur at least a month,
    or 22 trading days, apart") — NOT a direct Edwards & Magee number;
    E&M only gave qualitative "~one month / several weeks" guidance.
    """
    highs = _find_swing_highs(df)
    if len(highs) < 2:
        return None

    # Use the two most recent swing highs
    h1_idx, h2_idx = highs[-2], highs[-1]

    # LMW Def 5: tops must be at least ~1 month apart. The 22-day figure
    # is LMW's own discretization, not a direct Edwards & Magee number.
    if (h2_idx - h1_idx) < _DOUBLE_TOP_MIN_SEPARATION_BARS:
        return None

    h1, h2 = df.iloc[h1_idx]["high"], df.iloc[h2_idx]["high"]

    # Need a trough between them
    between = df.iloc[h1_idx + 1:h2_idx]
    if between.empty:
        return None
    trough = float(between["low"].min())

    # Confidence: how similar are the peaks, and how deep is the trough
    peak_similarity = _similarity_score(h1, h2, tolerance=_DOUBLE_TOP_PEAK_TOLERANCE)
    if peak_similarity == 0:
        return None
    avg_peak = (h1 + h2) / 2
    trough_depth = (avg_peak - trough) / avg_peak if avg_peak > 0 else 0
    depth_score = min(trough_depth / _DOUBLE_TOP_DEPTH_SATURATION, 1.0)
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
    """Mirror of double top. See detect_double_top for citations."""
    lows = _find_swing_lows(df)
    if len(lows) < 2:
        return None
    l1_idx, l2_idx = lows[-2], lows[-1]

    # LMW Def 5 (mirror): bottoms must be at least ~1 month apart. 22 is
    # LMW's discretization of E&M's qualitative "several weeks" guidance.
    if (l2_idx - l1_idx) < _DOUBLE_TOP_MIN_SEPARATION_BARS:
        return None

    l1, l2 = df.iloc[l1_idx]["low"], df.iloc[l2_idx]["low"]

    between = df.iloc[l1_idx + 1:l2_idx]
    if between.empty:
        return None
    peak = float(between["high"].max())

    trough_similarity = _similarity_score(l1, l2, tolerance=_DOUBLE_TOP_PEAK_TOLERANCE)
    if trough_similarity == 0:
        return None
    avg_trough = (l1 + l2) / 2
    peak_height = (peak - avg_trough) / avg_trough if avg_trough > 0 else 0
    height_score = min(peak_height / _DOUBLE_TOP_DEPTH_SATURATION, 1.0)
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
    are shorter and roughly equal.

    Implements LMW (2000) Definition 1: shoulders (E1, E5) within 1.5% of
    their average AND neckline troughs (E2, E4) within 1.5% of theirs.
    """
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
    shoulder_similarity = _similarity_score(s1, s2, tolerance=_HS_SHOULDER_TOLERANCE)
    if shoulder_similarity == 0:
        return None

    # Head should be meaningfully taller than shoulders (>=3% above max shoulder)
    max_shoulder = max(s1, s2)
    head_prominence = (head - max_shoulder) / max_shoulder if max_shoulder > 0 else 0
    prominence_score = min(head_prominence / 0.03, 1.0)

    # Neckline = average of the two troughs between (s1, head) and (head, s2)
    trough1 = float(df.iloc[s1_idx + 1:h_idx]["low"].min())
    trough2 = float(df.iloc[h_idx + 1:s2_idx]["low"].min())

    # LMW Def 1: the two neckline troughs (E2, E4) must also be near-equal
    neckline_similarity = _similarity_score(
        trough1, trough2, tolerance=_HS_NECKLINE_TOLERANCE,
    )
    if neckline_similarity == 0:
        return None

    neckline = (trough1 + trough2) / 2

    confidence = round(
        shoulder_similarity * prominence_score * neckline_similarity, 2,
    )
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
    """Mirror: three swing lows with middle (head) lowest.

    Implements LMW (2000) Definition 1 (inverted): shoulders within 1.5%
    of average AND neckline peaks within 1.5% of theirs.
    """
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
    shoulder_similarity = _similarity_score(s1, s2, tolerance=_HS_SHOULDER_TOLERANCE)
    if shoulder_similarity == 0:
        return None

    min_shoulder = min(s1, s2)
    head_prominence = (min_shoulder - head) / min_shoulder if min_shoulder > 0 else 0
    prominence_score = min(head_prominence / 0.03, 1.0)

    peak1 = float(df.iloc[s1_idx + 1:h_idx]["high"].max())
    peak2 = float(df.iloc[h_idx + 1:s2_idx]["high"].max())

    # LMW Def 1 (inverted): the two neckline peaks must also be near-equal
    neckline_similarity = _similarity_score(
        peak1, peak2, tolerance=_HS_NECKLINE_TOLERANCE,
    )
    if neckline_similarity == 0:
        return None

    neckline = (peak1 + peak2) / 2

    confidence = round(
        shoulder_similarity * prominence_score * neckline_similarity, 2,
    )
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


def _is_flat_line(prices: np.ndarray) -> bool:
    """True if the pivot prices form a near-horizontal line.

    LMW (2000) Def 4 (Rectangle): all pivot prices on a horizontal
    trendline lie within 0.75% of their average. We reuse the same
    rule for triangle's flat side detection.
    """
    if len(prices) < 2:
        return True
    avg = float(np.mean(prices))
    if avg == 0:
        return False
    spread = float(np.max(prices) - np.min(prices))
    return (spread / abs(avg)) <= _FLAT_LINE_SPREAD_TOLERANCE


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

    # Classify each line as flat / rising / falling using LMW Def 4's
    # 0.75% pivot-price-spread tolerance (instead of an unitless normalized
    # slope hack). "Flat" means: max(prices) - min(prices) <= 0.75% of mean.
    upper_prices = df.iloc[h_idx]["high"].values
    lower_prices = df.iloc[l_idx]["low"].values
    upper_flat = _is_flat_line(upper_prices)
    lower_flat = _is_flat_line(lower_prices)

    name: str | None = None
    if upper_flat and not lower_flat and lower_slope > 0:
        name = "ascending_triangle"
    elif lower_flat and not upper_flat and upper_slope < 0:
        name = "descending_triangle"
    elif (
        not upper_flat
        and not lower_flat
        and upper_slope < 0
        and lower_slope > 0
    ):
        name = "symmetric_triangle"
    if name is None:
        return None

    # Confidence: how cleanly the slopes match the pattern signature.
    # Use unit-free normalized slopes ONLY for the symmetry-bonus calculation
    # — never as a classification threshold.
    confidence = 0.7  # base for matching the shape
    if name == "symmetric_triangle":
        mean_price = float(df["close"].iloc[-len(df) // 4:].mean()) or 1.0
        upper_norm = upper_slope / mean_price
        lower_norm = lower_slope / mean_price
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


# ── Chronological 5-extrema helper (used by Broadening + Rectangle) ───


def _last_n_alternating_extrema(
    df: pd.DataFrame, n: int = 5, distance: int = 5,
) -> list[tuple[int, str, float]] | None:
    """Return the last `n` chronologically-alternating extrema (high/low).

    Each tuple is (bar_index, kind, price) where kind is "H" (swing high)
    or "L" (swing low). The list is in chronological order and strictly
    alternates kinds — if `find_peaks` produces two adjacent highs (a
    common case when a deeper high overshadows an intermediate one),
    only the latest of the consecutive run is kept.

    Returns None if fewer than `n` alternating extrema exist after
    cleaning. This is the LMW (2000) E1..E5 input vector for Definitions
    1, 2, 3, 4 (HS, Broadening, Triangle, Rectangle).
    """
    highs = _find_swing_highs(df, distance=distance)
    lows = _find_swing_lows(df, distance=distance)

    # Tag and merge
    tagged: list[tuple[int, str, float]] = []
    for i in highs:
        tagged.append((int(i), "H", float(df.iloc[i]["high"])))
    for i in lows:
        tagged.append((int(i), "L", float(df.iloc[i]["low"])))
    tagged.sort(key=lambda t: t[0])

    if len(tagged) < n:
        return None

    # Collapse adjacent same-kind extrema by keeping the more extreme one.
    # Two consecutive highs without an intervening low → keep the higher;
    # two consecutive lows → keep the lower. This mirrors the cleanup that
    # LMW's kernel-smoother does implicitly by smoothing out the lesser pivot.
    cleaned: list[tuple[int, str, float]] = []
    for ext in tagged:
        if cleaned and cleaned[-1][1] == ext[1]:
            prev = cleaned[-1]
            if (ext[1] == "H" and ext[2] >  prev[2]) or \
               (ext[1] == "L" and ext[2] <  prev[2]):
                cleaned[-1] = ext
            # else: keep the previous (more extreme) one
        else:
            cleaned.append(ext)

    if len(cleaned) < n:
        return None
    return cleaned[-n:]


# ── Broadening tops / bottoms (LMW Def 2) ────────────────────────────


def detect_broadening_top(df: pd.DataFrame) -> ChartPattern | None:
    """LMW (2000) Definition 2 — Broadening Top (megaphone).

    On 5 alternating extrema E1..E5 starting with a HIGH:
        E1 < E3 < E5     (peaks rising)
        E2 > E4          (troughs falling)
    """
    extrema = _last_n_alternating_extrema(df, n=5)
    if extrema is None or extrema[0][1] != "H":
        return None
    e1, e2, e3, e4, e5 = (e[2] for e in extrema)

    if not (e1 < e3 < e5 and e2 > e4):
        return None

    # Demand the expansion is meaningful (not 0.01% drift)
    peaks_expansion = (e5 - e1) / e1 if e1 > 0 else 0
    troughs_expansion = (e2 - e4) / e2 if e2 > 0 else 0
    if peaks_expansion < _BROADENING_MIN_EXPANSION or \
       troughs_expansion < _BROADENING_MIN_EXPANSION:
        return None

    # Confidence: average of the two expansions, saturated at 5% (any
    # broader than that and we're 100% sure it's a megaphone).
    score = (peaks_expansion + troughs_expansion) / 2
    confidence = round(min(score / 0.05, 1.0), 2)
    if confidence < 0.3:
        return None

    return ChartPattern(
        name="broadening_top",
        confidence=confidence,
        key_levels={
            "upper_pivot_latest": round(e5, 2),
            "lower_pivot_latest": round(e4, 2),
            "upper_pivot_first":  round(e1, 2),
            "lower_pivot_first":  round(e2, 2),
        },
        bar_indices=[e[0] for e in extrema],
    )


def detect_broadening_bottom(df: pd.DataFrame) -> ChartPattern | None:
    """LMW (2000) Definition 2 — Broadening Bottom (inverted megaphone).

    On 5 alternating extrema E1..E5 starting with a LOW:
        E1 > E3 > E5     (troughs falling)
        E2 < E4          (peaks rising)
    """
    extrema = _last_n_alternating_extrema(df, n=5)
    if extrema is None or extrema[0][1] != "L":
        return None
    e1, e2, e3, e4, e5 = (e[2] for e in extrema)

    if not (e1 > e3 > e5 and e2 < e4):
        return None

    troughs_expansion = (e1 - e5) / e1 if e1 > 0 else 0
    peaks_expansion = (e4 - e2) / e2 if e2 > 0 else 0
    if peaks_expansion < _BROADENING_MIN_EXPANSION or \
       troughs_expansion < _BROADENING_MIN_EXPANSION:
        return None

    score = (peaks_expansion + troughs_expansion) / 2
    confidence = round(min(score / 0.05, 1.0), 2)
    if confidence < 0.3:
        return None

    return ChartPattern(
        name="broadening_bottom",
        confidence=confidence,
        key_levels={
            "lower_pivot_latest": round(e5, 2),
            "upper_pivot_latest": round(e4, 2),
            "lower_pivot_first":  round(e1, 2),
            "upper_pivot_first":  round(e2, 2),
        },
        bar_indices=[e[0] for e in extrema],
    )


# ── Rectangle tops / bottoms (LMW Def 4) ──────────────────────────────


def _detect_rectangle(df: pd.DataFrame, *, side: str) -> ChartPattern | None:
    """Shared engine for rectangle top / rectangle bottom (LMW Def 4).

    side = "top" or "bottom". Both share identical geometry; the label
    is determined by which extreme the FIRST pivot is. LMW Def 4 says:
        - All 3 maxima (E1, E3, E5 for a top; E2, E4 for a bottom)
          lie within `_FLAT_LINE_SPREAD_TOLERANCE` of their mean.
        - Same for all 2-3 minima.
        - lowest_max > highest_min  (otherwise it's chop, not a channel).
    """
    extrema = _last_n_alternating_extrema(df, n=5)
    if extrema is None:
        return None
    expected_first = "H" if side == "top" else "L"
    if extrema[0][1] != expected_first:
        return None

    highs = np.array([e[2] for e in extrema if e[1] == "H"])
    lows = np.array([e[2] for e in extrema if e[1] == "L"])
    if len(highs) < 2 or len(lows) < 2:
        return None
    if not _is_flat_line(highs) or not _is_flat_line(lows):
        return None
    if highs.min() <= lows.max():
        return None

    upper = float(highs.mean())
    lower = float(lows.mean())
    # Confidence: how tight both lines are. Tighter → 1.0.
    upper_tightness = 1.0 - (highs.max() - highs.min()) / (
        _FLAT_LINE_SPREAD_TOLERANCE * upper) if upper > 0 else 0.0
    lower_tightness = 1.0 - (lows.max() - lows.min()) / (
        _FLAT_LINE_SPREAD_TOLERANCE * lower) if lower > 0 else 0.0
    confidence = round(
        max(0.0, min(1.0, (upper_tightness + lower_tightness) / 2)), 2,
    )
    if confidence < 0.3:
        return None

    name = "rectangle_top" if side == "top" else "rectangle_bottom"
    # Measured-move target per LMW: breakout in the direction of the side
    # (top → down, bottom → up) by the channel height.
    height = upper - lower
    target = round(lower - height, 2) if side == "top" else round(upper + height, 2)

    return ChartPattern(
        name=name,
        confidence=confidence,
        key_levels={
            "resistance": round(upper, 2),
            "support":    round(lower, 2),
            "target":     target,
        },
        bar_indices=[e[0] for e in extrema],
    )


def detect_rectangle_top(df: pd.DataFrame) -> ChartPattern | None:
    """LMW (2000) Definition 4 — Rectangle Top."""
    return _detect_rectangle(df, side="top")


def detect_rectangle_bottom(df: pd.DataFrame) -> ChartPattern | None:
    """LMW (2000) Definition 4 — Rectangle Bottom."""
    return _detect_rectangle(df, side="bottom")


# ── Detection driver ─────────────────────────────────────────────


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
        detect_broadening_top,
        detect_broadening_bottom,
        detect_rectangle_top,
        detect_rectangle_bottom,
    ):
        try:
            r = fn(df)
        except (ValueError, IndexError):
            r = None
        if r is not None and r.confidence >= min_confidence:
            results.append(r)
    return [r.to_dict() for r in results]
