"""Coverage spike for NSE filings endpoints + parser verification.

PURPOSE
=======
Iteration 3.1.3 built parsers for 4 NSE endpoints based on inferred JSON
shapes (community libs + NSE web UI). Real shapes may differ. This script
verifies our parsers against real NSE responses for a small set of stocks.

Some networks block www.nseindia.com; run this from one that can reach it.

USAGE
=====
1. cd practice_project/price_predictor
2. uv run python scripts/coverage_spike_filings.py
3. Report written to: reports/filings_coverage_<UTC_TIMESTAMP>.{md,json}
4. Ask the puppy to review the report

WHAT IT TESTS
=============
3 NSE stocks (small set, since we're testing SHAPES not COVERAGE):
- RELIANCE (large-cap, heavy filer)
- TCS (large-cap, heavy filer)
- HAPPSTMNDS (small-cap, lighter coverage)

For each stock x each of 4 endpoints, captures:
- HTTP status (does the endpoint exist? does cookie warmup work?)
- Raw item count (does NSE return data for this stock/window?)
- Parser success count (do OUR parsers correctly extract Filing objects?)
- Sample raw JSON item (what NSE actually returns)
- Sample parsed Filing (what we extracted)
- Field comparison: expected fields vs actual fields present

KEY OUTPUT: per-endpoint verdict
- ✅ All systems go: endpoint works, parser extracts everything
- ⚠️  Partial: endpoint works but parser misses some items (shape drift)
- ❌ Broken: endpoint 404/403/dead, OR parser extracts 0 from N items
"""
from __future__ import annotations

import asyncio
import json
import sys
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Suppress noisy logger output
warnings.filterwarnings("ignore")

# Make src/ importable when running this script directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402

from price_predictor.data.filings import (  # noqa: E402
    _BROWSER_HEADERS,
    _ENDPOINTS,
    FilingsFetchError,
    _warm_session,
)
from price_predictor.data.schema import FilingKind  # noqa: E402

# ─────────────────────────────────────────────────────────────
# Test universe (small — we're verifying shapes, not coverage)
# ─────────────────────────────────────────────────────────────
TICKERS: list[tuple[str, str]] = [
    ("RELIANCE", "large"),
    ("TCS", "large"),
    ("HAPPSTMNDS", "small"),
]

KINDS: list[FilingKind] = [
    "announcement",
    "board_meeting",
    "corporate_action",
    "financial_result",
]


# ─────────────────────────────────────────────────────────────
# Per-endpoint probe: hit raw, capture both raw and parsed
# ─────────────────────────────────────────────────────────────
async def _probe_endpoint(
    client: httpx.AsyncClient,
    symbol: str,
    kind: FilingKind,
    start: str,
    end: str,
) -> dict[str, Any]:
    """Probe one (symbol, kind) pair. Returns a structured result dict.

    Doesn't raise — captures everything (HTTP errors, JSON errors, parser
    errors) so the report can show them.
    """
    url_builder, parser = _ENDPOINTS[kind]
    url = url_builder(symbol, start, end)

    result: dict[str, Any] = {
        "symbol": symbol,
        "kind": kind,
        "url": url,
        "http_status": None,
        "content_encoding": None,  # debugging: catch brotli/gzip mismatches
        "content_length": None,
        "raw_item_count": 0,
        "parsed_count": 0,
        "sample_raw_item": None,
        "sample_parsed_filing": None,
        "raw_keys": [],
        "error": None,
    }

    # ── Step 1: HTTP fetch ──
    try:
        resp = await client.get(url, headers=_BROWSER_HEADERS)
        result["http_status"] = resp.status_code
        result["content_encoding"] = resp.headers.get("content-encoding")
        result["content_length"] = len(resp.content)
    except Exception as e:
        result["error"] = f"HTTP_ERROR: {type(e).__name__}: {str(e)[:200]}"
        return result

    if resp.status_code >= 400:
        result["error"] = f"HTTP_{resp.status_code}: {resp.text[:200]}"
        return result

    # ── Step 2: JSON parse ──
    try:
        payload = resp.json()
    except ValueError as e:
        result["error"] = f"JSON_PARSE: {e}; body[:200]={resp.text[:200]}"
        return result

    # ── Step 3: Extract items list (NSE wraps inconsistently) ──
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = (
            payload.get("data")
            or payload.get("rows")
            or payload.get("result")
            or []
        )
        if not isinstance(items, list):
            result["error"] = (
                f"WRAPPED_BUT_NOT_LIST: payload keys={list(payload.keys())[:10]}"
            )
            return result
    else:
        result["error"] = f"UNEXPECTED_PAYLOAD_TYPE: {type(payload).__name__}"
        return result

    result["raw_item_count"] = len(items)

    # ── Step 4: Capture sample raw item ──
    if items and isinstance(items[0], dict):
        first = items[0]
        result["raw_keys"] = sorted(first.keys())
        # Truncate long string values for report sanity
        sample = {
            k: (v[:120] + "…" if isinstance(v, str) and len(v) > 120 else v)
            for k, v in first.items()
        }
        result["sample_raw_item"] = sample

    # ── Step 5: Run through OUR parser ──
    parsed = []
    parse_errors: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            f = parser(item, symbol)
            if f is not None:
                parsed.append(f)
        except Exception as e:
            parse_errors.append(f"{type(e).__name__}: {str(e)[:100]}")

    result["parsed_count"] = len(parsed)
    if parsed:
        # Capture one parsed Filing for shape-spot-check
        result["sample_parsed_filing"] = json.loads(parsed[0].model_dump_json())
    if parse_errors:
        result["parser_errors"] = parse_errors[:3]  # first 3

    return result


# ─────────────────────────────────────────────────────────────
# Spike runner
# ─────────────────────────────────────────────────────────────
async def run_spike() -> dict[str, Any]:
    end = datetime.now(UTC).date() - timedelta(days=1)
    start = end - timedelta(days=180)  # 6-month window — generous
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    print(f"Spike starting at {datetime.now(UTC).isoformat()}")
    print(f"Date window: {start_str} to {end_str}")
    print(f"Testing {len(TICKERS)} tickers x {len(KINDS)} endpoints "
          f"= {len(TICKERS) * len(KINDS)} probes\n")

    probes: list[dict[str, Any]] = []
    warmup_status: dict[str, Any] = {"success": False, "error": None}

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        # ── Cookie warmup (one-shot, shared across all probes) ──
        print("  warming up NSE session...", end=" ", flush=True)
        try:
            await _warm_session(client)
            warmup_status["success"] = True
            print("OK")
        except FilingsFetchError as e:
            warmup_status["error"] = str(e)
            print(f"FAILED: {e}")
            print("\n⚠️  Warmup failed. Probes will likely all fail.")
            print("   Continuing anyway to capture per-endpoint behavior.\n")

        # ── Probe each (symbol, kind) ──
        for sym, _tier in TICKERS:
            for kind in KINDS:
                print(f"  probing {sym:<14} {kind:<18}...", end=" ", flush=True)
                # NSE rate-limits aggressively; small delay between calls
                await asyncio.sleep(0.5)
                result = await _probe_endpoint(client, sym, kind, start_str, end_str)
                probes.append(result)
                if result["error"]:
                    print(f"❌ {result['error'][:60]}")
                elif result["raw_item_count"] == 0:
                    print("⚠️  no data")
                elif result["parsed_count"] == 0:
                    print(f"❌ parser broken (0/{result['raw_item_count']})")
                elif result["parsed_count"] < result["raw_item_count"]:
                    print(f"⚠️  partial ({result['parsed_count']}/{result['raw_item_count']})")
                else:
                    print(f"✅ {result['parsed_count']}/{result['raw_item_count']}")

    return {
        "ran_at": datetime.now(UTC).isoformat(),
        "date_window": {"start": start_str, "end": end_str},
        "warmup": warmup_status,
        "probes": probes,
    }


# ─────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────
def _render_report(spike: dict) -> str:
    probes = spike["probes"]
    warmup = spike["warmup"]

    md: list[str] = []
    md.append("# NSE Filings — Coverage & Parser-Verification Spike Report")
    md.append("")
    md.append(f"- **Ran at:** {spike['ran_at']}")
    md.append(f"- **Date window:** {spike['date_window']['start']} to "
              f"{spike['date_window']['end']} (180 days)")
    md.append(f"- **Cookie warmup:** {'✅ OK' if warmup['success'] else '❌ FAILED'}")
    if not warmup["success"]:
        md.append(f"  - Error: `{warmup['error']}`")
    md.append("")

    # ── Per-endpoint verdict ────────────────────────────────
    md.append("## Per-endpoint verdict")
    md.append("")
    md.append("| Endpoint | HTTP success | Has data | Parser success | Verdict |")
    md.append("|---|---|---|---|---|")

    by_kind: dict[str, list[dict]] = {}
    for p in probes:
        by_kind.setdefault(p["kind"], []).append(p)

    for kind in KINDS:
        ps = by_kind.get(kind, [])
        if not ps:
            continue
        http_ok = sum(1 for p in ps if p["http_status"] and p["http_status"] < 400)
        any_data = sum(1 for p in ps if p["raw_item_count"] > 0)
        parser_full = sum(
            1 for p in ps
            if p["raw_item_count"] > 0 and p["parsed_count"] == p["raw_item_count"]
        )
        parser_partial = sum(
            1 for p in ps
            if p["raw_item_count"] > 0 and 0 < p["parsed_count"] < p["raw_item_count"]
        )
        parser_zero = sum(
            1 for p in ps
            if p["raw_item_count"] > 0 and p["parsed_count"] == 0
        )

        if http_ok == 0:
            verdict = "❌ Endpoint dead/blocked"
        elif any_data == 0:
            verdict = "⚠️ No data returned (window too narrow? stock too quiet?)"
        elif parser_zero > 0:
            verdict = f"❌ Parser broken ({parser_zero} stocks: 0 of N parsed)"
        elif parser_partial > 0:
            verdict = f"⚠️ Parser partial ({parser_partial} stocks lost some items)"
        elif parser_full == any_data:
            verdict = "✅ Endpoint + parser working"
        else:
            verdict = "❓ Mixed"

        md.append(
            f"| `{kind}` | {http_ok}/{len(ps)} | {any_data}/{len(ps)} | "
            f"{parser_full}+{parser_partial} full+partial | {verdict} |"
        )
    md.append("")

    # ── Decision criteria ───────────────────────────────────
    md.append("## Decision criteria for the puppy")
    md.append("")
    md.append("- **All 4 endpoints ✅** → ship as-is, proceed to iteration 3.2")
    md.append("- **1-2 endpoints ⚠️ (partial)** → fix parser shape mismatches, "
              "then ship")
    md.append("- **Any ❌ Endpoint dead** → URL/param wrong; check NSE web UI for "
              "current path")
    md.append("- **Any ❌ Parser broken (0 of N)** → JSON shape totally different "
              "from inferred; need to look at the raw sample below and rewrite parser")
    md.append("")

    # ── Per-probe detail ────────────────────────────────────
    md.append("## Per-probe detail")
    md.append("")
    md.append("| Symbol | Kind | HTTP | Encoding | Bytes | Raw items | Parsed | Error |")
    md.append("|---|---|---|---|---|---|---|---|")
    for p in probes:
        md.append(
            f"| {p['symbol']} | `{p['kind']}` | "
            f"{p['http_status'] or '-'} | "
            f"{p.get('content_encoding') or '-'} | "
            f"{p.get('content_length') or '-'} | "
            f"{p['raw_item_count']} | {p['parsed_count']} | "
            f"{(p['error'] or '')[:60]} |"
        )
    md.append("")

    # ── Raw shape inspection (most important for puppy review) ──
    md.append("## Raw JSON shape per endpoint (one sample each)")
    md.append("")
    md.append("**This is the critical section for the puppy:** compare the keys "
              "in `sample_raw_item` against what our parsers expect. Mismatches "
              "= parser bugs to fix.")
    md.append("")

    for kind in KINDS:
        # Find first probe for this kind that got data
        sample = next(
            (p for p in by_kind.get(kind, []) if p["sample_raw_item"]),
            None,
        )
        md.append(f"### `{kind}`")
        md.append("")
        if sample is None:
            md.append("_No samples — endpoint returned no data or failed for all stocks._")
            md.append("")
            continue

        md.append(f"**From:** `{sample['symbol']}` "
                  f"(HTTP {sample['http_status']}, "
                  f"{sample['raw_item_count']} items)")
        md.append("")
        md.append(f"**Keys present:** `{sample['raw_keys']}`")
        md.append("")
        md.append("**Sample raw item:**")
        md.append("```json")
        md.append(json.dumps(sample["sample_raw_item"], indent=2, default=str))
        md.append("```")
        md.append("")
        if sample["sample_parsed_filing"]:
            md.append("**Sample parsed Filing (our output):**")
            md.append("```json")
            md.append(json.dumps(sample["sample_parsed_filing"], indent=2, default=str))
            md.append("```")
            md.append("")
        if sample.get("parser_errors"):
            md.append("**Parser errors observed:**")
            for err in sample["parser_errors"]:
                md.append(f"- `{err}`")
            md.append("")

    return "\n".join(md)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    spike = asyncio.run(run_spike())

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_md = reports_dir / f"filings_coverage_{timestamp}.md"
    out_json = reports_dir / f"filings_coverage_{timestamp}.json"

    out_md.write_text(_render_report(spike), encoding="utf-8")
    out_json.write_text(json.dumps(spike, indent=2, default=str), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"✅ Markdown report: {out_md.relative_to(ROOT)}")
    print(f"✅ Raw JSON:        {out_json.relative_to(ROOT)}")
    print()
    print("Bring those files back on VPN and ask the puppy to review.")


if __name__ == "__main__":
    main()
