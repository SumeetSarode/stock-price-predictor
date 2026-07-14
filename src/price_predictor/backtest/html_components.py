"""Reusable HTML/Chart.js building blocks for backtest reports.

WHY THIS EXISTS
===============
The backtest report (html_report.py) hit ~640 lines because every
section function carried its own slice of presentation primitives
(escaping, percent-formatting, card markup, chart-canvas wrappers,
table layout). Pulling the GENERIC helpers here keeps html_report.py
focused on backtest-specific composition + makes the helpers
trivially reusable for any future palette-driven HTML reports.

WHAT'S IN HERE
==============
- Palette constants + level-style maps.
- Escape / format helpers (_esc, _pct, _num).
- Generic UI atoms (_metric_card, _chart_block, _table).
- Domain-aware badges that need report-specific palette knowledge
  (_outcome_badge, _direction_badge) but no aggregation logic.

WHAT'S NOT IN HERE
==================
- Anything that knows about BacktestEvaluation, CalibrationReport,
  or the orchestration of sections -- that lives in html_report.py.
- Tailwind CSS overrides specific to a particular report layout.
"""
from __future__ import annotations

import html
import json
from typing import Any

from price_predictor.backtest.insights import InsightLevel
from price_predictor.prediction.grading import GradeOutcome
from price_predictor.prediction.schema import PredictionDirection


# ─────────────────────────────────────────────────────────────
# Palette + level styling. Centralized so any tweak propagates.
# All combinations have been picked to clear WCAG AA (4.5:1 text,
# 3:1 UI). The dark "text" colors are darker tints chosen so heading
# text on white/light bg stays >= 4.5:1 even when the level color
# itself is borderline (e.g. spark.140 brown).
# ─────────────────────────────────────────────────────────────
INSIGHT_STYLE: dict[InsightLevel, dict[str, str]] = {
    InsightLevel.POSITIVE: {
        "bg": "bg-white", "border": "border-l-4 border-[#2a8703]",
        "text": "text-[#1c5102]", "tag_bg": "bg-[#2a8703]",
        "tag_label": "Positive",
    },
    InsightLevel.NEUTRAL: {
        "bg": "bg-white", "border": "border-l-4 border-[#0053e2]",
        "text": "text-[#003a9e]", "tag_bg": "bg-[#0053e2]",
        "tag_label": "Info",
    },
    InsightLevel.WARNING: {
        "bg": "bg-[#fff8eb]", "border": "border-l-4 border-[#995213]",
        "text": "text-[#995213]", "tag_bg": "bg-[#995213]",
        "tag_label": "Warning",
    },
    InsightLevel.CRITICAL: {
        "bg": "bg-white", "border": "border-l-4 border-[#ea1100]",
        "text": "text-[#9a0a00]", "tag_bg": "bg-[#ea1100]",
        "tag_label": "Critical",
    },
}

OUTCOME_LABEL: dict[GradeOutcome, str] = {
    GradeOutcome.TARGET_HIT: "Target hit",
    GradeOutcome.STOP_HIT: "Stop hit",
    GradeOutcome.STOP_HIT_AMBIGUOUS: "Stop (ambiguous)",
    GradeOutcome.EXPIRED: "Expired",
    GradeOutcome.INCONCLUSIVE: "Inconclusive",
    GradeOutcome.NOT_APPLICABLE: "N/A (neutral)",
}

OUTCOME_COLOR: dict[GradeOutcome, str] = {
    GradeOutcome.TARGET_HIT: "#2a8703",
    GradeOutcome.STOP_HIT: "#ea1100",
    GradeOutcome.STOP_HIT_AMBIGUOUS: "#995213",
    GradeOutcome.EXPIRED: "#6b7280",
    GradeOutcome.INCONCLUSIVE: "#9ca3af",
    GradeOutcome.NOT_APPLICABLE: "#0053e2",
}

DIRECTION_COLOR: dict[PredictionDirection, str] = {
    PredictionDirection.BULLISH: "#2a8703",
    PredictionDirection.BEARISH: "#ea1100",
    PredictionDirection.NEUTRAL: "#995213",
}


# ─────────────────────────────────────────────────────────────
# Escape / format primitives
# ─────────────────────────────────────────────────────────────
def esc(s: Any) -> str:
    """HTML-escape a value, coercing to str. Handles None safely.

    All user-controlled / model-controlled strings (ticker symbols,
    rationale text, etc.) MUST flow through this before landing in
    HTML. Defensive even though most data here is internal --
    cheap insurance against a future code path that pipes user input.
    """
    return html.escape("" if s is None else str(s))


def pct(v: float | None, *, decimals: int = 1) -> str:
    """Format a 0..1 fraction as a percent string. None -> em dash.

    Em dash (vs blank) makes "no data" visible at a glance -- a blank
    cell looks like a rendering bug.
    """
    if v is None:
        return "&mdash;"
    return f"{v * 100:.{decimals}f}%"


def num(v: float | None, *, decimals: int = 3) -> str:
    """Format a number with N decimals. None -> em dash."""
    if v is None:
        return "&mdash;"
    return f"{v:.{decimals}f}"


# ─────────────────────────────────────────────────────────────
# Generic UI atoms
# ─────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, *, accent: str = "#0053e2") -> str:
    """A small card with a label + big number. Used in metric grids.

    `accent` is a hex color for the value; defaults to the primary
    accent blue. Callers pick contextual colors (green for good, red for bad).
    """
    return f"""
    <div class="bg-white rounded-lg p-4 border border-gray-200 shadow-sm">
        <div class="text-xs uppercase tracking-wide text-gray-500 font-semibold">{esc(label)}</div>
        <div class="text-2xl font-bold mt-1" style="color: {accent}">{value}</div>
    </div>
    """


def chart_block(canvas_id: str, *, height_px: int = 320) -> str:
    """Wrap a Chart.js canvas in a fixed-height div.

    WHY: Chart.js with responsive:true IGNORES the canvas height
    attribute. Fixed-height parent is the workaround.
    """
    return (
        f'<div style="position:relative;height:{height_px}px;width:100%">'
        f'<canvas id="{esc(canvas_id)}"></canvas></div>'
    )


def table(headers: list[str], rows: list[list[str]],
          *, table_id: str | None = None) -> str:
    """Tailwind-styled table. Cell content is trusted (NOT escaped here).

    NOTE: callers are responsible for HTML-escaping cell content
    because some cells legitimately contain pre-formatted HTML
    (e.g. colored badges via outcome_badge). To keep this helper
    dumb, we trust input -- escape at the call site.
    """
    th = "".join(
        f'<th class="px-3 py-2 text-left text-xs font-semibold '
        f'text-gray-700 uppercase tracking-wide border-b border-gray-200">{h}</th>'
        for h in headers
    )
    body_rows = []
    for r in rows:
        tds = "".join(
            f'<td class="px-3 py-2 text-sm text-gray-800 '
            f'border-b border-gray-100">{c}</td>'
            for c in r
        )
        body_rows.append(f"<tr class='hover:bg-gray-50'>{tds}</tr>")
    id_attr = f' id="{esc(table_id)}"' if table_id else ""
    return (
        f'<div class="overflow-x-auto rounded-lg border border-gray-200">'
        f'<table{id_attr} class="min-w-full bg-white">'
        f'<thead class="bg-gray-50">{th}</thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        f'</table></div>'
    )


def chart_script(canvas_id: str, config: dict) -> str:
    """Render a <script> that instantiates a Chart.js chart.

    JSON-serializing the config (vs string-templating) is BOTH safer
    (no manual escaping) and clearer (the dict reads like Python).

    SECURITY: standard JSON encoding does NOT escape '<', so a label
    like "</script><script>evil()</script>" would break out of our
    <script> block. The standard recipe is to replace `<` with the
    JSON unicode escape `\\u003c` (and `&`, `>` for defense in depth).
    Same trick the Flask source recommends in its tojson filter.
    """
    safe_json = (
        json.dumps(config)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    return (
        '<script>new Chart(document.getElementById('
        f'{json.dumps(canvas_id)}), {safe_json});</script>'
    )


# ─────────────────────────────────────────────────────────────
# Domain badges (use palette but still presentation-only)
# ─────────────────────────────────────────────────────────────
def outcome_badge(o: GradeOutcome) -> str:
    """Colored pill for a grade outcome (target hit / stop / expired)."""
    color = OUTCOME_COLOR[o]
    label = OUTCOME_LABEL[o]
    return (
        f'<span class="inline-block px-2 py-0.5 rounded text-xs font-semibold '
        f'text-white" style="background:{color}">{esc(label)}</span>'
    )


def direction_badge(d: PredictionDirection) -> str:
    """Colored pill for a predicted direction (bull/bear/neutral)."""
    color = DIRECTION_COLOR[d]
    return (
        f'<span class="inline-block px-2 py-0.5 rounded text-xs font-semibold '
        f'text-white" style="background:{color}">{esc(d.value)}</span>'
    )
