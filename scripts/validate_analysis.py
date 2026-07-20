#!/usr/bin/env python
"""Off-VPN validation harness for the Analysis tab.

Run this while OFF the Walmart VPN (so Yahoo Finance / GDELT are reachable).
It exercises the EXACT same service functions the web app uses --
`compute_live_analysis` (price feed -> resample -> every analyzer) and
`fetch_recent_headlines` (GDELT) -- across a basket of NSE tickers and all
three timeframes, then writes a copy-pasteable report.

What it validates
-----------------
1. Price feed actually reachable (no more connection errors).
2. Indicators populate on daily / weekly / monthly (SMA/EMA/RSI/ADX/OBV/
   pivots/Ichimoku present, bar counts sane and coarsening by timeframe).
3. Chart patterns finally SHOW -- counts per ticker/timeframe, with the
   0.5 (display) vs 0.7 (prediction) split so we can see the threshold
   impact on real data.
4. News fetch works (headline counts, or a clean soft-error).

Usage
-----
    cd practice_project/price_predictor
    .venv/bin/python scripts/validate_analysis.py
    # optional: custom tickers
    .venv/bin/python scripts/validate_analysis.py RELIANCE TCS INFY

Output
------
    - Prints a full report to the terminal.
    - Writes /tmp/pp_validation_report.txt  (paste this back to the agent)
    - Writes /tmp/pp_validation.json         (machine-readable)
"""
from __future__ import annotations

import asyncio
import json
import sys
import traceback
from datetime import datetime

from price_predictor.web.services.analysis_service import (
    AnalysisServiceError,
    compute_live_analysis,
    timeframe_options,
)
from price_predictor.web.services.news_service import fetch_recent_headlines

# A spread of liquid NSE names likely to show a variety of patterns.
DEFAULT_TICKERS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "SBIN", "ITC", "TATAMOTORS", "WIPRO", "AXISBANK",
]
NEWS_SAMPLE = 3  # how many of the tickers to also news-check

REPORT_TXT = "/tmp/pp_validation_report.txt"
REPORT_JSON = "/tmp/pp_validation.json"


def _fmt(v: float | None, prec: int = 2) -> str:
    return "—" if v is None else f"{v:.{prec}f}"


def _summarize_analysis(a) -> dict:
    """Pull a compact, comparable summary out of a LiveAnalysis."""
    trend = a.trend or {}
    mom = a.momentum or {}
    lvl = a.levels or {}
    crosses = trend.get("ma_crosses") or {}
    ich = trend.get("ichimoku") or {}
    obv = mom.get("obv") or {}
    pivots = lvl.get("pivots") or {}

    patterns = a.chart_patterns or []
    pat_05 = [p for p in patterns]  # already filtered at 0.5 for display
    pat_07 = [p for p in patterns if float(p.get("confidence", 0)) >= 0.7]

    return {
        "bars_used": a.bars_used,
        "close": trend.get("close"),
        "sma_keys": sorted(str(k) for k in (trend.get("sma") or {})),
        "rsi": mom.get("rsi"),
        "adx": (trend.get("adx") or {}).get("adx"),
        "ma_cross_keys": sorted(crosses.keys()),
        "has_ichimoku": bool(ich.get("price_vs_cloud")),
        "has_obv": obv.get("slope_20") is not None,
        "has_pivots": pivots.get("pp") is not None,
        "candlesticks": len(a.candlesticks or []),
        "patterns_ge_050": [
            (p["name"], round(float(p["confidence"]), 2)) for p in pat_05
        ],
        "patterns_ge_070": [
            (p["name"], round(float(p["confidence"]), 2)) for p in pat_07
        ],
    }


async def _run(tickers: list[str]) -> dict:
    tf_keys = [o["key"] for o in timeframe_options()]
    results: dict = {"generated_at": datetime.now().isoformat(),
                     "tickers": {}, "news": {}, "errors": []}

    for tkr in tickers:
        results["tickers"][tkr] = {}
        for tf in tf_keys:
            try:
                a = await compute_live_analysis(tkr, timeframe=tf)
                results["tickers"][tkr][tf] = _summarize_analysis(a)
            except AnalysisServiceError as exc:
                results["tickers"][tkr][tf] = {"error": exc.message}
            except Exception as exc:  # noqa: BLE001 -- want the full picture
                results["tickers"][tkr][tf] = {"error": f"UNEXPECTED: {exc}"}
                results["errors"].append(f"{tkr}/{tf}: {exc}\n{traceback.format_exc()}")

    for tkr in tickers[:NEWS_SAMPLE]:
        try:
            bundle = await fetch_recent_headlines(tkr, days=7)
            results["news"][tkr] = {
                "headlines": len(bundle.headlines),
                "error": bundle.error,
                "sample": [h.title for h in bundle.headlines[:3]],
            }
        except Exception as exc:  # noqa: BLE001
            results["news"][tkr] = {"error": f"UNEXPECTED: {exc}"}
            results["errors"].append(f"news/{tkr}: {exc}\n{traceback.format_exc()}")

    return results


def _render_report(r: dict, tickers: list[str]) -> str:
    tf_keys = [o["key"] for o in timeframe_options()]
    lines: list[str] = []
    add = lines.append

    add("=" * 72)
    add("PRICE PREDICTOR — ANALYSIS TAB VALIDATION")
    add(f"generated: {r['generated_at']}")
    add("=" * 72)

    # ---- 1. Feed reachability ----
    ok = fail = 0
    for tkr in tickers:
        for tf in tf_keys:
            cell = r["tickers"][tkr].get(tf, {})
            if "error" in cell:
                fail += 1
            else:
                ok += 1
    add(f"\n[1] PRICE FEED: {ok} ok / {fail} failed "
        f"across {len(tickers)} tickers × {len(tf_keys)} timeframes")

    # ---- 2. Per-ticker indicator + pattern table ----
    add("\n[2] INDICATORS & PATTERNS (per ticker × timeframe)")
    for tkr in tickers:
        add(f"\n  {tkr}")
        for tf in tf_keys:
            c = r["tickers"][tkr].get(tf, {})
            if "error" in c:
                add(f"    {tf:8} ERROR: {c['error']}")
                continue
            ind = []
            ind.append(f"bars={c['bars_used']}")
            ind.append(f"close={_fmt(c['close'])}")
            ind.append(f"rsi={_fmt(c['rsi'], 1)}")
            ind.append(f"adx={_fmt(c['adx'], 1)}")
            ind.append(f"MAx={len(c['ma_cross_keys'])}")
            ind.append("ich" if c["has_ichimoku"] else "no-ich")
            ind.append("obv" if c["has_obv"] else "no-obv")
            ind.append("piv" if c["has_pivots"] else "no-piv")
            add(f"    {tf:8} " + "  ".join(ind))
            p05, p07 = c["patterns_ge_050"], c["patterns_ge_070"]
            add(f"             patterns ≥0.5: {len(p05)}  "
                f"(≥0.7: {len(p07)})  candles={c['candlesticks']}")
            if p05:
                add("             " + ", ".join(
                    f"{n}({conf})" for n, conf in p05))

    # ---- 3. Pattern coverage roll-up ----
    add("\n[3] PATTERN COVERAGE ROLL-UP")
    for tf in tf_keys:
        with_pat = sum(
            1 for tkr in tickers
            if r["tickers"][tkr].get(tf, {}).get("patterns_ge_050")
        )
        add(f"    {tf:8} {with_pat}/{len(tickers)} tickers show ≥1 chart pattern")

    # ---- 4. News ----
    add("\n[4] NEWS FETCH")
    for tkr, n in r["news"].items():
        if n.get("error"):
            add(f"    {tkr:12} error: {n['error']}  (headlines={n.get('headlines', 0)})")
        else:
            add(f"    {tkr:12} {n['headlines']} headlines")
            for t in n.get("sample", []):
                add(f"                 · {t[:80]}")

    # ---- 5. Unexpected errors ----
    if r["errors"]:
        add("\n[5]  UNEXPECTED ERRORS (share these!)")
        for e in r["errors"]:
            add("    " + e.splitlines()[0])
    else:
        add("\n[5] no unexpected exceptions ")

    add("\n" + "=" * 72)
    add(f"Full JSON: {REPORT_JSON}")
    add("Paste this whole report back to the agent.")
    add("=" * 72)
    return "\n".join(lines)


def main() -> None:
    tickers = [t.upper() for t in sys.argv[1:]] or DEFAULT_TICKERS
    print(f"Validating {len(tickers)} tickers across daily/weekly/monthly "
          f"+ news… (this hits the live feed, give it a minute)\n")
    r = asyncio.run(_run(tickers))
    report = _render_report(r, tickers)
    print(report)
    with open(REPORT_TXT, "w") as f:
        f.write(report + "\n")
    with open(REPORT_JSON, "w") as f:
        json.dump(r, f, indent=2, default=str)
    print(f"\nWrote {REPORT_TXT} and {REPORT_JSON}")


if __name__ == "__main__":
    main()
