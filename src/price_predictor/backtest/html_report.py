"""HTML report renderer for a BacktestEvaluation.

WHY THIS EXISTS
===============
We need a self-contained, opens-in-any-browser report with:
  - Executive insights at top AND bottom (per ops mandate)
  - Daily/monthly time-series breakdowns
  - Per-axis charts (horizon, ticker, direction, month)
  - Per-prediction drill-down table for forensic review
  - Palette + WCAG AA contrast

This module produces ONE HTML string from a BacktestEvaluation.
Save it anywhere, double-click, done. No server, no build step --
Tailwind + Chart.js via CDN, the rest is inline.

WHY NO JINJA2
=============
Jinja is a transitive dep here, not a direct one. Promoting it to
direct just for this one report is YAGNI. Pure-Python composition
of small section functions reads cleanly without it.

DESIGN
======
- Generic UI atoms (escape, tables, charts, badges) live in
  html_components.py -- reusable across any future report.
- This module composes those atoms into BACKTEST-SPECIFIC sections.
- ONE function per section: pure (eval -> html string), no I/O.
- ONE top-level orchestrator: render_html_report(eval) -> str.
- Convenience writer: write_html_report(eval, path) -> Path.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from price_predictor.backtest.html_components import (
    INSIGHT_STYLE,
    chart_block,
    chart_script,
    direction_badge,
    esc,
    metric_card,
    num,
    outcome_badge,
    pct,
    table,
)
from price_predictor.backtest.insights import Insight, generate_insights
from price_predictor.prediction.calibration import CalibrationReport

if TYPE_CHECKING:
    from price_predictor.backtest.evaluation import BacktestEvaluation


# ─────────────────────────────────────────────────────────────
# Section: page shell (head + foot)
# ─────────────────────────────────────────────────────────────
def _page_head(title: str) -> str:
    """HTML head: Tailwind CDN, Chart.js CDN, base styles.

    No HTMX -- this is a static report; HTMX would be ceremony for
    nothing. (HTMX shines for interactive apps; static reports may skip it.)
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{esc(title)}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        body {{ color: #1c1c1c; font-family: ui-sans-serif, system-ui, sans-serif; }}
        h1, h2, h3 {{ color: #1c1c1c; }}
        .section-h2 {{ font-size: 1.25rem; font-weight: 700; margin: 2rem 0 1rem;
                      padding-bottom: 0.5rem; border-bottom: 2px solid #0053e2; }}
    </style>
</head>
<body class="bg-gray-50">
"""


def _page_foot() -> str:
    return """
</body>
</html>"""


# ─────────────────────────────────────────────────────────────
# Section: hero (run metadata)
# ─────────────────────────────────────────────────────────────
def _section_hero(ev: "BacktestEvaluation") -> str:
    run = ev.run
    tickers_str = ", ".join(esc(t) for t in run.tickers)
    date_range = (
        f"{run.as_of_dates[0].isoformat()} &rarr; "
        f"{run.as_of_dates[-1].isoformat()}"
        if run.as_of_dates else "&mdash;"
    )
    horizons_str = ", ".join(h.value for h in run.horizons)
    duration = run.duration_seconds
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""
<header class="bg-[#0053e2] text-white">
    <div class="max-w-6xl mx-auto px-6 py-8">
        <div class="flex items-baseline justify-between flex-wrap gap-2">
            <h1 class="text-3xl font-bold">Backtest Report</h1>
            <div class="text-sm opacity-80">Generated {esc(generated)}</div>
        </div>
        <div class="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
                <div class="opacity-80 uppercase tracking-wide text-xs">Tickers</div>
                <div class="font-semibold mt-1">{tickers_str}</div>
            </div>
            <div>
                <div class="opacity-80 uppercase tracking-wide text-xs">Date range</div>
                <div class="font-semibold mt-1">{date_range}</div>
            </div>
            <div>
                <div class="opacity-80 uppercase tracking-wide text-xs">Horizons</div>
                <div class="font-semibold mt-1">{esc(horizons_str)}</div>
            </div>
            <div>
                <div class="opacity-80 uppercase tracking-wide text-xs">Predictions</div>
                <div class="font-semibold mt-1">
                    {ev.n_predictions} total &middot; {ev.n_judged} judged &middot;
                    {duration:.0f}s runtime
                </div>
            </div>
        </div>
    </div>
</header>
"""


# ─────────────────────────────────────────────────────────────
# Section: insight cards (used at TOP and BOTTOM of page)
# ─────────────────────────────────────────────────────────────
def _section_insights(insights: list[Insight], *, title: str) -> str:
    if not insights:
        body = (
            '<div class="bg-white rounded-lg p-6 border border-gray-200 '
            'text-gray-500 italic">No insights available '
            '(empty evaluation).</div>'
        )
    else:
        cards = []
        for ins in insights:
            style = INSIGHT_STYLE[ins.level]
            cards.append(f"""
            <div class="{style['bg']} {style['border']} rounded-lg p-4 shadow-sm">
                <div class="flex items-center gap-2 mb-2">
                    <span class="{style['tag_bg']} text-white text-xs font-semibold
                                 uppercase tracking-wide px-2 py-0.5 rounded">
                        {style['tag_label']}
                    </span>
                </div>
                <div class="font-semibold {style['text']} text-base">{esc(ins.headline)}</div>
                <div class="text-sm text-gray-700 mt-1 leading-snug">{esc(ins.detail)}</div>
            </div>
            """)
        body = (
            '<div class="grid grid-cols-1 md:grid-cols-2 gap-4">'
            + "".join(cards) + "</div>"
        )

    return f"""
<section class="max-w-6xl mx-auto px-6 mt-8">
    <h2 class="section-h2">{esc(title)}</h2>
    {body}
</section>
"""


# ─────────────────────────────────────────────────────────────
# Section: overall metrics grid
# ─────────────────────────────────────────────────────────────
def _section_overall(ev: "BacktestEvaluation") -> str:
    o = ev.overall
    # Color the headline numbers by quality (green good, red bad, blue neutral).
    da_color = (
        "#2a8703" if o.direction_accuracy >= 0.55
        else "#ea1100" if o.direction_accuracy <= 0.45
        else "#0053e2"
    )
    bss_color = (
        "#2a8703" if (o.brier_skill_score or 0) > 0.05
        else "#ea1100" if (o.brier_skill_score or 0) < -0.05
        else "#0053e2"
    )

    cards = [
        metric_card("Direction accuracy", pct(o.direction_accuracy), accent=da_color),
        metric_card("Hit rate (resolved)", pct(o.hit_rate_resolved)),
        metric_card("Hit rate (strict)", pct(o.hit_rate_strict)),
        metric_card("Brier Skill Score", num(o.brier_skill_score), accent=bss_color),
        metric_card("Brier score", num(o.brier_score)),
        metric_card("Mean confidence", pct(o.mean_confidence)),
        metric_card("Base rate", pct(o.base_rate)),
        metric_card("Mean return", pct(o.mean_return, decimals=2)),
        metric_card("Median return", pct(o.median_return, decimals=2)),
        metric_card("Target hits", str(o.n_target_hit), accent="#2a8703"),
        metric_card("Stop hits", str(o.n_stop_hit + o.n_stop_hit_ambiguous), accent="#ea1100"),
        metric_card("Expired / N/A / Inc.",
                    str(o.n_expired + o.n_not_applicable + o.n_inconclusive)),
    ]

    return f"""
<section class="max-w-6xl mx-auto px-6 mt-8">
    <h2 class="section-h2">Overall Metrics</h2>
    <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
        {"".join(cards)}
    </div>
</section>
"""


# ─────────────────────────────────────────────────────────────
# Section: breakdown chart + table (one fn, used per axis)
# ─────────────────────────────────────────────────────────────
def _breakdown_section(
    title: str,
    canvas_id: str,
    breakdown: dict[Any, CalibrationReport],
    *,
    label_fn=lambda k: str(k),
    chart_height: int = 320,
) -> str:
    """Generic 'bar chart + table' for any breakdown dict.

    DRY: ONE function handles by_horizon, by_ticker, by_direction.
    Differences are passed in as args (label formatter, canvas id).
    """
    if not breakdown:
        return (
            f'<section class="max-w-6xl mx-auto px-6 mt-8">'
            f'<h2 class="section-h2">{esc(title)}</h2>'
            f'<div class="text-gray-500 italic">No data.</div></section>'
        )

    items = list(breakdown.items())
    labels = [label_fn(k) for k in (i[0] for i in items)]
    da_values = [round(i[1].direction_accuracy * 100, 2) for i in items]
    hit_values = [round(i[1].hit_rate_resolved * 100, 2) for i in items]

    chart_config = {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": [
                {"label": "Direction accuracy %", "data": da_values,
                 "backgroundColor": "#0053e2"},
                {"label": "Hit rate (resolved) %", "data": hit_values,
                 "backgroundColor": "#ffc220"},
            ],
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "scales": {"y": {"beginAtZero": True, "max": 100,
                             "title": {"display": True, "text": "%"}}},
            "plugins": {"legend": {"position": "top"}},
        },
    }

    rows = [
        [
            esc(label_fn(k)), str(r.n_predictions),
            str(r.n_with_direction_judgement),
            pct(r.direction_accuracy), pct(r.hit_rate_resolved),
            num(r.brier_skill_score), pct(r.mean_return, decimals=2),
        ]
        for k, r in items
    ]

    table_html = table(
        headers=["Group", "n", "n judged", "Direction acc",
                 "Hit rate (resolved)", "BSS", "Mean return"],
        rows=rows,
    )

    return f"""
<section class="max-w-6xl mx-auto px-6 mt-8">
    <h2 class="section-h2">{esc(title)}</h2>
    {chart_block(canvas_id, height_px=chart_height)}
    {chart_script(canvas_id, chart_config)}
    <div class="mt-4">{table_html}</div>
</section>
"""


# ─────────────────────────────────────────────────────────────
# Section: by-month line chart (time-series view)
# ─────────────────────────────────────────────────────────────
def _section_by_month(ev: "BacktestEvaluation") -> str:
    """Time-series accuracy by month -- spots regime drift.

    Differs from generic _breakdown_section because we want a LINE
    chart (continuity over time) and the keys are pre-sortable
    YYYY-MM strings.
    """
    if not ev.by_month:
        return ""

    sorted_keys = sorted(ev.by_month.keys())
    da_values = [round(ev.by_month[k].direction_accuracy * 100, 2)
                 for k in sorted_keys]
    n_values = [ev.by_month[k].n_with_direction_judgement
                for k in sorted_keys]

    chart_config = {
        "type": "line",
        "data": {
            "labels": sorted_keys,
            "datasets": [
                {"label": "Direction accuracy %", "data": da_values,
                 "borderColor": "#0053e2", "backgroundColor": "#0053e2",
                 "tension": 0.2, "yAxisID": "y"},
                {"label": "n judged", "data": n_values,
                 "borderColor": "#ffc220", "backgroundColor": "#ffc220",
                 "tension": 0.2, "yAxisID": "y1", "type": "bar"},
            ],
        },
        "options": {
            "responsive": True, "maintainAspectRatio": False,
            "scales": {
                "y": {"beginAtZero": True, "max": 100, "position": "left",
                       "title": {"display": True, "text": "Direction acc %"}},
                "y1": {"beginAtZero": True, "position": "right",
                        "title": {"display": True, "text": "n judged"},
                        "grid": {"drawOnChartArea": False}},
            },
            "plugins": {"legend": {"position": "top"}},
        },
    }

    return f"""
<section class="max-w-6xl mx-auto px-6 mt-8">
    <h2 class="section-h2">Accuracy Over Time (Monthly)</h2>
    {chart_block("chart-by-month", height_px=320)}
    {chart_script("chart-by-month", chart_config)}
</section>
"""


# ─────────────────────────────────────────────────────────────
# Section: per-prediction drill-down table
# ─────────────────────────────────────────────────────────────
def _section_drilldown(ev: "BacktestEvaluation", *, max_rows: int = 500) -> str:
    """Per-prediction detail table.

    Capped at `max_rows` because a 1000-prediction backtest would
    blow up the HTML file size. Excessive rows are truncated with a
    note rather than silently omitted.
    """
    graded = ev.graded
    truncated = len(graded) > max_rows
    rows_to_render = graded[:max_rows] if truncated else graded

    rows = []
    for g in rows_to_render:
        p = g.prediction
        rows.append([
            esc(p.ticker),
            esc(p.as_of.date().isoformat()),
            esc(p.horizon.value),
            direction_badge(p.direction),
            pct(p.confidence, decimals=0),
            outcome_badge(g.outcome),
            pct(g.realized_return, decimals=2),
            "&check;" if g.direction_correct is True
            else "&cross;" if g.direction_correct is False
            else "&mdash;",
        ])

    truncation_note = ""
    if truncated:
        truncation_note = (
            f'<div class="text-sm text-gray-600 mt-2 italic">'
            f'Showing first {max_rows} of {len(graded)} predictions.</div>'
        )

    return f"""
<section class="max-w-6xl mx-auto px-6 mt-8">
    <h2 class="section-h2">Per-Prediction Detail</h2>
    {table(
        headers=["Ticker", "Date", "Horizon", "Direction", "Confidence",
                 "Outcome", "Realized return", "Direction OK"],
        rows=rows,
    )}
    {truncation_note}
</section>
"""


# ─────────────────────────────────────────────────────────────
# Section: footer
# ─────────────────────────────────────────────────────────────
def _section_footer(ev: "BacktestEvaluation") -> str:
    return f"""
<footer class="max-w-6xl mx-auto px-6 mt-12 mb-8 text-center text-xs text-gray-500">
    Generated by price_predictor backtest. Run sensitivity:
    <span class="font-mono">{esc(ev.run.sensitivity)}</span>,
    concurrency: <span class="font-mono">{ev.run.concurrency}</span>.
</footer>
"""


# ─────────────────────────────────────────────────────────────
# Top-level orchestrator
# ─────────────────────────────────────────────────────────────
def render_html_report(ev: "BacktestEvaluation") -> str:
    """Build the complete HTML report string.

    Composes every section in the canonical reading order:
      1. Hero (run metadata)
      2. Top insights (executive summary)
      3. Overall metrics
      4. By horizon / ticker / direction (chart + table each)
      5. By month (line chart)
      6. Per-prediction drill-down
      7. Bottom insights (mirror of #2 -- mandate)
      8. Footer
    """
    insights = generate_insights(ev)

    return (
        _page_head("Backtest Report")
        + _section_hero(ev)
        + _section_insights(insights, title="Executive Summary")
        + _section_overall(ev)
        + _breakdown_section(
            "By Horizon", "chart-by-horizon", ev.by_horizon,
            label_fn=lambda h: h.value,
        )
        + _breakdown_section(
            "By Ticker", "chart-by-ticker", ev.by_ticker,
        )
        + _breakdown_section(
            "By Direction", "chart-by-direction", ev.by_direction,
            label_fn=lambda d: d.value,
        )
        + _section_by_month(ev)
        + _section_drilldown(ev)
        + _section_insights(insights, title="Key Takeaways")
        + _section_footer(ev)
        + _page_foot()
    )


def write_html_report(ev: "BacktestEvaluation", path: Path) -> Path:
    """Write the report to `path`. Returns the absolute path written.

    Creates parent directories as needed. Overwrites existing files
    (a backtest report is always a fresh artifact -- no append mode
    makes sense).
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html_report(ev), encoding="utf-8")
    return path
