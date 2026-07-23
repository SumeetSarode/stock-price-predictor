#!/usr/bin/env python
"""Verify the news fallback (GDELT -> Google News RSS) against the REAL internet.

Run this OFF-VPN (corporate proxy can block GDELT / Google News). It makes
live HTTP calls -- no mocks -- and writes a timestamped results file so you
have a receipt of what actually happened.

WHAT IT CHECKS
==============
  1. Google News RSS provider ALONE  -- does the fallback source work + how
     fast? (This is the thing that must be "quick".)
  2. GDELT ALONE                      -- does the primary work / is it 429ing?
  3. End-to-end NATURAL               -- NewsSnapshot.get_or_fetch as the app
     really calls it (GDELT first, RSS only if GDELT fails).
  4. End-to-end FORCED-FALLBACK       -- GDELT monkeypatched to fail so we
     PROVE RSS takes over through the real snapshot code path, and time it.

Every step is timed. Results go to both the console and an output file
(--out, default: news_fallback_results_<UTC timestamp>.txt) plus a .json
sibling with the raw numbers.

USAGE
=====
    .venv/bin/python scripts/verify_news_fallback.py
    .venv/bin/python scripts/verify_news_fallback.py "Reliance Industries" "TCS"
    .venv/bin/python scripts/verify_news_fallback.py --out ~/Desktop/news.txt
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from datetime import UTC, date, datetime
from pathlib import Path

# Ensure the package is importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from price_predictor.data import news_snapshot as ns_mod
from price_predictor.data.news import NewsFetchError, fetch_news
from price_predictor.data.news_providers import (
    GoogleNewsRssProvider,
)
from price_predictor.data.news_snapshot import NewsSnapshot

DEFAULT_QUERIES = ["Reliance Industries", "Tata Consultancy Services", "Infosys"]
LOOKBACK_DAYS = 7


def _fmt_articles(df: pd.DataFrame, limit: int = 3) -> list[dict]:
    """Compact sample of articles for the report."""
    out = []
    for _, row in df.head(limit).iterrows():
        pub = row.get("published_at")
        out.append(
            {
                "title": str(row.get("title", ""))[:120],
                "source": str(row.get("source", "")),
                "published_at": str(pub),
                "url": str(row.get("url", ""))[:120],
            }
        )
    return out


async def _timed(coro):
    """Await a coroutine, returning (seconds, result_or_exception, ok)."""
    t0 = time.perf_counter()
    try:
        res = await coro
        return time.perf_counter() - t0, res, True
    except Exception as e:
        return time.perf_counter() - t0, e, False


async def check_rss(query: str) -> dict:
    provider = GoogleNewsRssProvider()
    today = date.today()
    start = (today - pd.Timedelta(days=LOOKBACK_DAYS)).isoformat()
    secs, res, ok = await _timed(
        provider.fetch(query, start, today.isoformat(), exact_phrase=True)
    )
    if ok:
        return {
            "query": query, "ok": True, "seconds": round(secs, 3),
            "article_count": len(res), "sample": _fmt_articles(res),
        }
    return {
        "query": query, "ok": False, "seconds": round(secs, 3),
        "error": f"{type(res).__name__}: {res}",
    }


async def check_gdelt(query: str) -> dict:
    today = date.today()
    start = (today - pd.Timedelta(days=LOOKBACK_DAYS)).isoformat()
    secs, res, ok = await _timed(
        fetch_news(query, start, today.isoformat(), exact_phrase=True)
    )
    if ok:
        return {
            "query": query, "ok": True, "seconds": round(secs, 3),
            "article_count": len(res), "sample": _fmt_articles(res),
        }
    return {
        "query": query, "ok": False, "seconds": round(secs, 3),
        "error": f"{type(res).__name__}: {res}",
    }


async def check_end_to_end_natural(query: str, root: Path) -> dict:
    """Real app path: GDELT first, RSS only if GDELT fails."""
    store = NewsSnapshot(root / "natural")
    secs, res, ok = await _timed(
        store.get_or_fetch(query, date.today(), LOOKBACK_DAYS)
    )
    if ok:
        return {
            "query": query, "ok": True, "seconds": round(secs, 3),
            "article_count": len(res), "sample": _fmt_articles(res),
        }
    return {
        "query": query, "ok": False, "seconds": round(secs, 3),
        "error": f"{type(res).__name__}: {res}",
    }


async def check_end_to_end_forced_fallback(query: str, root: Path) -> dict:
    """Force GDELT to fail so RSS MUST take over -- proves the fallback wiring
    and times the pure-RSS path (no GDELT retry backoff)."""
    store = NewsSnapshot(root / "forced")
    original = ns_mod.fetch_news

    async def _boom(*args, **kwargs):
        raise NewsFetchError("SIMULATED GDELT 429 (forced-fallback test)")

    ns_mod.fetch_news = _boom  # type: ignore[assignment]
    try:
        secs, res, ok = await _timed(
            store.get_or_fetch(query, date.today(), LOOKBACK_DAYS)
        )
    finally:
        ns_mod.fetch_news = original  # type: ignore[assignment]

    if ok:
        return {
            "query": query, "ok": True, "seconds": round(secs, 3),
            "article_count": len(res), "sample": _fmt_articles(res),
            "note": "RSS served this after GDELT was forced to fail",
        }
    return {
        "query": query, "ok": False, "seconds": round(secs, 3),
        "error": f"{type(res).__name__}: {res}",
    }


async def run(queries: list[str]) -> dict:
    results: dict = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "lookback_days": LOOKBACK_DAYS,
        "queries": queries,
        "rss_only": [],
        "gdelt_only": [],
        "end_to_end_natural": [],
        "end_to_end_forced_fallback": [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for q in queries:
            print(f"  [RSS]            {q} ...", flush=True)
            results["rss_only"].append(await check_rss(q))
            print(f"  [GDELT]          {q} ...", flush=True)
            results["gdelt_only"].append(await check_gdelt(q))
            print(f"  [E2E natural]    {q} ...", flush=True)
            results["end_to_end_natural"].append(
                await check_end_to_end_natural(q, root)
            )
            print(f"  [E2E forced RSS] {q} ...", flush=True)
            results["end_to_end_forced_fallback"].append(
                await check_end_to_end_forced_fallback(q, root)
            )
    return results


def _section(title: str, rows: list[dict]) -> str:
    lines = [f"\n### {title}", "-" * 60]
    for r in rows:
        status = "OK " if r.get("ok") else "FAIL"
        head = (
            f"[{status}] {r['query']:<30} {r['seconds']:>6.3f}s "
            f"articles={r.get('article_count', '-')}"
        )
        lines.append(head)
        if not r.get("ok"):
            lines.append(f"        error: {r.get('error')}")
        for a in r.get("sample", []):
            lines.append(f"        - ({a['source']}) {a['title']}")
            lines.append(f"          {a['published_at']}")
    return "\n".join(lines)


def render(results: dict) -> str:
    def _avg(rows):
        oks = [r["seconds"] for r in rows if r.get("ok")]
        return f"{sum(oks) / len(oks):.3f}s avg" if oks else "n/a (all failed)"

    out = [
        "=" * 60,
        "NEWS FALLBACK VERIFICATION",
        "=" * 60,
        f"generated_at_utc : {results['generated_at_utc']}",
        f"lookback_days    : {results['lookback_days']}",
        f"queries          : {', '.join(results['queries'])}",
        "",
        "SUMMARY (avg latency of successful calls)",
        "-" * 60,
        f"  RSS only                 : {_avg(results['rss_only'])}",
        f"  GDELT only               : {_avg(results['gdelt_only'])}",
        f"  End-to-end (natural)     : {_avg(results['end_to_end_natural'])}",
        f"  End-to-end (forced RSS)  : "
        f"{_avg(results['end_to_end_forced_fallback'])}",
    ]
    out.append(_section("Google News RSS ONLY", results["rss_only"]))
    out.append(_section("GDELT ONLY", results["gdelt_only"]))
    out.append(_section("END-TO-END (natural: GDELT->RSS)",
                        results["end_to_end_natural"]))
    out.append(_section("END-TO-END (forced fallback: RSS must serve)",
                        results["end_to_end_forced_fallback"]))
    out.append("\n" + "=" * 60)
    out.append("INTERPRETATION")
    out.append("-" * 60)
    out.append(
        "  * RSS 'OK' with articles + low latency  => fallback source works.\n"
        "  * 'forced RSS' OK with articles         => fallback WIRING works\n"
        "    (RSS really takes over when GDELT dies).\n"
        "  * GDELT FAIL with 429/timeout           => the very problem this\n"
        "    fallback exists for; RSS should still be OK above."
    )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("queries", nargs="*", default=None,
                    help="Company names to search (default: 3 NSE large-caps)")
    ap.add_argument("--out", default=None, help="Output file path")
    args = ap.parse_args()

    queries = args.queries or DEFAULT_QUERIES
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.out) if args.out else Path(
        f"news_fallback_results_{stamp}.txt"
    )

    print("Running news fallback verification (LIVE network)...")
    print("NOTE: run this OFF-VPN if GDELT/Google News are blocked.\n")
    results = asyncio.run(run(queries))

    report = render(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(report)
    print(f"\nSaved report -> {out_path}")
    print(f"Saved raw    -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
