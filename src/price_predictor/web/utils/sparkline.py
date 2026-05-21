"""Pure-math helper that turns a list of numbers into SVG geometry.

Used by the prediction tracker (right rail) to draw a tiny trend line
of recent R-multiples next to the Avg R headline. Kept in `web/utils/`
because it's purely a presentation concern — no data fetching, no DB,
no I/O. Trivially unit-testable.

Design notes
------------
- Returns POINTS (a string for SVG <polyline points="...">) and the
  coordinates of the last point (so the template can drop a dot on
  the latest value). Renderers handle colors via CSS classes, not
  inline attributes — keeps theming pluggable.
- 0 or 1 values → returns None. Caller decides what to show
  (typically a muted dash placeholder).
- The viewBox is the canonical (0,0) → (width,height) box. We use
  preserveAspectRatio="none" in the template so the SVG can stretch
  responsively without distorting the data shape semantically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Sparkline:
    """SVG geometry for a tiny trend line. Coordinate space is 0..W × 0..H."""

    width: int
    height: int
    points: str             # space-separated "x,y x,y ..." for <polyline>
    last_x: float
    last_y: float
    baseline_y: float       # y-coordinate of "zero" line (if range crosses 0)
    has_baseline: bool      # False when all values share a sign
    sign: Literal["pos", "neg", "neutral"]  # color hint for the LAST value


def build_sparkline(
    values: list[float],
    *,
    width: int = 120,
    height: int = 28,
    pad: int = 3,
) -> Sparkline | None:
    """Compute SVG geometry for a sparkline of `values` (chronological order).

    Returns None if there isn't enough data to draw a line (n < 2). The
    caller renders a placeholder in that case. We refuse to draw a single
    point as a "line" because it's misleading — one bar of R is not a trend.
    """
    if len(values) < 2:
        return None

    vmin = min(values)
    vmax = max(values)
    span = vmax - vmin

    inner_h = height - 2 * pad
    inner_w = width  # x edges go right to 0 / width

    def y_for(v: float) -> float:
        # Flat line (all equal values) → center vertically.
        if span == 0:
            return pad + inner_h / 2
        # SVG y grows downward, so we invert the normalised value.
        return pad + (1 - (v - vmin) / span) * inner_h

    n = len(values)
    pts = []
    for i, v in enumerate(values):
        x = (i / (n - 1)) * inner_w
        y = y_for(v)
        # 3-decimal rounding keeps the rendered HTML compact without any
        # visible loss for a 120×28 canvas.
        pts.append(f"{x:.3f},{y:.3f}")

    last = values[-1]
    last_x = inner_w
    last_y = y_for(last)
    sign: Literal["pos", "neg", "neutral"] = (
        "pos" if last > 0 else "neg" if last < 0 else "neutral"
    )

    # Baseline only makes sense if 0 is actually inside the value range.
    # Otherwise drawing a "zero line" at the top/bottom edge is noise.
    has_baseline = vmin < 0 < vmax
    baseline_y = y_for(0.0) if has_baseline else 0.0

    return Sparkline(
        width=width,
        height=height,
        points=" ".join(pts),
        last_x=last_x,
        last_y=last_y,
        baseline_y=baseline_y,
        has_baseline=has_baseline,
        sign=sign,
    )
