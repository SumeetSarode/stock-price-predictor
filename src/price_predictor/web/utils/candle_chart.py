"""Pure-math helper that turns real OHLC bars into candlestick SVG geometry.

Used by the Patterns tab to draw a tiny, honest chart of the ACTUAL candles
where a pattern was detected — no external images, no idealized/made-up
diagrams. The candles you see are the real market data that triggered the
detection.

Design mirrors ``sparkline.py``: purely a presentation concern (no data
fetching, no DB, no I/O), returns geometry dataclasses, trivially
unit-testable. The template turns this geometry into ``<line>``/``<rect>``
SVG elements; colors come from CSS classes so theming stays pluggable.

Coordinate space is (0,0) top-left → (width,height) bottom-right, matching
SVG's downward-growing y-axis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Default canvas sizes. Candlesticks get a compact box (few bars of
# context); chart patterns get a wider box (they span more bars).
# Heights include the ~14px date-axis strip added below the candles.
CANDLE_W = 200
CANDLE_H = 110
CHART_W = 380
CHART_H = 142
_PAD = 6
# Fraction of each bar's horizontal slot taken up by the body rectangle.
_BODY_FRAC = 0.6
# Minimum body width so a body never vanishes to an invisible sliver.
_MIN_BODY_W = 1.0
# Vertical strip reserved at the bottom for the date (x) axis, in px.
# Only reserved when bars carry a ``date`` -- keeps date-less callers
# (and their geometry) unchanged.
_AXIS_H = 14
# Most date ticks we'll ever draw (avoid crowding on wide charts).
_MAX_X_TICKS = 6


@dataclass(frozen=True, slots=True)
class Candle:
    """SVG geometry for one OHLC bar."""

    x: float            # center x of the bar (wick + body share it)
    wick_top: float     # y of the high
    wick_bottom: float  # y of the low
    body_top: float     # y of max(open, close)
    body_bottom: float  # y of min(open, close)
    body_x: float       # left edge of the body rect
    body_w: float       # body rect width
    bullish: bool       # close >= open
    highlighted: bool   # True if this bar is part of the detected pattern

    @property
    def body_h(self) -> float:
        """Body rect height. Floored at ~1px so a doji is still visible."""
        return max(self.body_bottom - self.body_top, 1.0)


@dataclass(frozen=True, slots=True)
class LevelLine:
    """A horizontal reference line (e.g. neckline / support / resistance)."""

    y: float
    label: str
    value: float


@dataclass(frozen=True, slots=True)
class XTick:
    """One date label on the x-axis, positioned under its candle."""

    x: float
    y: float
    label: str


@dataclass(frozen=True, slots=True)
class CandleChart:
    """Full geometry bundle for one inline candlestick SVG."""

    width: int
    height: int
    candles: list[Candle]
    levels: list[LevelLine]
    x_ticks: list[XTick]


def _num(value: Any) -> float | None:
    """Coerce to float; return None for None/NaN/non-numeric."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    # NaN != NaN — the canonical NaN test without importing math/pandas.
    if f != f:
        return None
    return f


def _short_date(iso: Any) -> str:
    """Turn 'YYYY-MM-DD' into a compact 'DD/MM' axis label.

    Falls back to the raw string (trimmed) for anything unexpected.
    """
    s = str(iso)
    parts = s.split("-")
    if len(parts) == 3 and all(parts):
        return f"{parts[2]}/{parts[1]}"
    return s[:5]


def build_candle_chart(
    bars: list[dict[str, Any]],
    *,
    width: int = CANDLE_W,
    height: int = CANDLE_H,
    pad: int = _PAD,
    levels: dict[str, float] | None = None,
) -> CandleChart | None:
    """Compute SVG geometry for a candlestick chart of `bars`.

    Args:
        bars: chronological list of dicts, each with numeric
            ``open``/``high``/``low``/``close`` and an optional truthy
            ``highlight`` flag marking pattern bars.
        width, height, pad: canvas dimensions (px).
        levels: optional ``{label: price}`` map drawn as horizontal
            reference lines (chart-pattern key levels). Prices outside the
            candle range are still included so they stay visible.

    Returns:
        A ``CandleChart``, or ``None`` if there is nothing sane to draw
        (no bars, or any bar has a bad/NaN value). Callers render a text
        fallback in the None case.
    """
    if not bars:
        return None

    # Parse + validate every bar up front. One bad value → bail to the
    # text fallback rather than draw a misleading partial chart.
    parsed: list[tuple[float, float, float, float, bool]] = []
    dates: list[str | None] = []
    for b in bars:
        o = _num(b.get("open"))
        h = _num(b.get("high"))
        low = _num(b.get("low"))
        c = _num(b.get("close"))
        if None in (o, h, low, c):
            return None
        parsed.append((o, h, low, c, bool(b.get("highlight"))))
        dates.append(b.get("date"))

    has_dates = any(d for d in dates)
    axis_h = _AXIS_H if has_dates else 0

    lo_vals = [p[2] for p in parsed]
    hi_vals = [p[1] for p in parsed]
    vmin = min(lo_vals)
    vmax = max(hi_vals)

    level_lines_src = list((levels or {}).items())
    for _, lv in level_lines_src:
        fv = _num(lv)
        if fv is not None:
            vmin = min(vmin, fv)
            vmax = max(vmax, fv)

    span = vmax - vmin
    # Reserve a strip at the bottom for date labels (only when we have
    # dates). Candles map into the space ABOVE that strip.
    inner_h = height - 2 * pad - axis_h
    inner_w = width - 2 * pad
    n = len(parsed)

    def y_for(v: float) -> float:
        if span == 0:
            return pad + inner_h / 2
        return round(pad + (1 - (v - vmin) / span) * inner_h, 3)

    slot = inner_w / n
    body_w = max(slot * _BODY_FRAC, _MIN_BODY_W)

    candles: list[Candle] = []
    for i, (o, h, low, c, hl) in enumerate(parsed):
        cx = round(pad + slot * (i + 0.5), 3)
        body_top = y_for(max(o, c))
        body_bottom = y_for(min(o, c))
        candles.append(
            Candle(
                x=cx,
                wick_top=y_for(h),
                wick_bottom=y_for(low),
                body_top=body_top,
                body_bottom=body_bottom,
                body_x=round(cx - body_w / 2, 3),
                body_w=round(body_w, 3),
                bullish=c >= o,
                highlighted=hl,
            )
        )

    level_lines: list[LevelLine] = []
    for label, lv in level_lines_src:
        fv = _num(lv)
        if fv is None:
            continue
        level_lines.append(
            LevelLine(y=y_for(fv), label=label.replace("_", " "), value=fv)
        )

    # Date (x) axis: pick up to _MAX_X_TICKS evenly-spaced bars, always
    # including the first and last, and label them under their candle.
    x_ticks: list[XTick] = []
    if has_dates:
        if n <= _MAX_X_TICKS:
            tick_idxs = list(range(n))
        else:
            step = (n - 1) / (_MAX_X_TICKS - 1)
            tick_idxs = sorted({round(i * step) for i in range(_MAX_X_TICKS)})
        label_y = height - pad + 2
        for i in tick_idxs:
            d = dates[i]
            if not d:
                continue
            x_ticks.append(
                XTick(x=candles[i].x, y=round(label_y, 3), label=_short_date(d))
            )

    return CandleChart(
        width=width,
        height=height,
        candles=candles,
        levels=level_lines,
        x_ticks=x_ticks,
    )
