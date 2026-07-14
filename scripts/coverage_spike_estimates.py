"""Coverage spike for yfinance Indian-stock analyst data.

PURPOSE
=======
yfinance analyst-data endpoints (query1.finance.yahoo.com) may be blocked
on some networks. Run this from a network that can reach Yahoo Finance to
verify how many NSE stocks actually have analyst coverage in yfinance.

USAGE
=====
1. cd practice_project/price_predictor
2. uv run python scripts/coverage_spike_estimates.py
3. Report written to: reports/estimates_coverage_<UTC_TIMESTAMP>.md
4. Ask the puppy to review the report

WHAT IT TESTS
=============
20 NSE tickers across cap tiers:
- 8 large-cap (NIFTY 50 staples — should be best covered)
- 7 mid-cap (NIFTY Midcap 100 picks — coverage gets patchy here)
- 5 small-cap (NIFTY Smallcap 100 picks — likely worst coverage)

For each ticker, captures:
- Number of forward EPS quarters returned
- Number of forward revenue quarters returned
- Number of analyst-recommendation snapshots
- Whether price targets are present
- Sample data dump (for the puppy to inspect shape/realism)

OUTPUT
======
Markdown report with:
- Summary stats (overall coverage %, by tier)
- Per-ticker detail
- Sample data dumps for 1 large-cap, 1 mid-cap, 1 small-cap (for spot-check)
- Environment info (Python, yfinance version)
"""
from __future__ import annotations

import asyncio
import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

# Suppress yfinance's noisy warnings
warnings.filterwarnings("ignore")

# Make `src/` importable when running this script directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yfinance as yf  # noqa: E402

from price_predictor.data.estimates import (  # noqa: E402
    EstimatesFetchError,
    coverage_summary,
    fetch_estimates,
)

# ─────────────────────────────────────────────────────────────
# Test universe — 20 NSE stocks across cap tiers
# ─────────────────────────────────────────────────────────────
TICKERS: list[tuple[str, str]] = [
    # Large-cap (NIFTY 50 — should have best coverage)
    ("RELIANCE.NS", "large"),
    ("TCS.NS", "large"),
    ("HDFCBANK.NS", "large"),
    ("INFY.NS", "large"),
    ("ICICIBANK.NS", "large"),
    ("HINDUNILVR.NS", "large"),
    ("ITC.NS", "large"),
    ("BHARTIARTL.NS", "large"),
    # Mid-cap (NIFTY Midcap 100 — coverage drops off here)
    ("PERSISTENT.NS", "mid"),
    ("MPHASIS.NS", "mid"),
    ("LTIM.NS", "mid"),
    ("COFORGE.NS", "mid"),
    ("PIIND.NS", "mid"),
    ("OBEROIRLTY.NS", "mid"),
    ("CGPOWER.NS", "mid"),
    # Small-cap (likely worst coverage)
    ("KPITTECH.NS", "small"),
    ("HAPPSTMNDS.NS", "small"),
    ("MAPMYINDIA.NS", "small"),
    ("ANGELONE.NS", "small"),
    ("RADICO.NS", "small"),
]


# ─────────────────────────────────────────────────────────────
# Spike runner
# ─────────────────────────────────────────────────────────────
async def run_spike() -> dict:
    """Fetch estimates for every ticker; collect summaries + sample data."""
    print(f"Spike starting at {datetime.now(UTC).isoformat()}")
    print(f"Testing {len(TICKERS)} tickers...\n")

    results: list[dict] = []
    sample_dumps: dict[str, dict] = {}  # one sample per tier for spot-check

    for sym, tier in TICKERS:
        print(f"  fetching {sym} ({tier})...", end=" ", flush=True)
        try:
            est = await fetch_estimates(sym)
            summary = coverage_summary(est)
            summary["tier"] = tier
            summary["error"] = None
            results.append(summary)

            # Capture one full dump per tier for shape inspection
            if tier not in sample_dumps and est.has_coverage:
                sample_dumps[tier] = json.loads(est.model_dump_json())

            print(f"OK (coverage={est.has_coverage})")
        except (EstimatesFetchError, ValueError) as e:
            results.append(
                {
                    "symbol": sym,
                    "tier": tier,
                    "has_coverage": False,
                    "earnings_quarters": 0,
                    "revenue_quarters": 0,
                    "recommendation_snapshots": 0,
                    "has_price_targets": False,
                    "num_analysts_current_quarter": None,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            print(f"FAILED ({type(e).__name__})")

    return {
        "ran_at": datetime.now(UTC).isoformat(),
        "python_version": sys.version,
        "yfinance_version": yf.__version__,
        "results": results,
        "sample_dumps": sample_dumps,
    }


# ─────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────
def _render_report(spike: dict) -> str:
    rows = spike["results"]
    samples = spike["sample_dumps"]

    # Aggregate stats
    total = len(rows)
    with_coverage = sum(1 for r in rows if r["has_coverage"])
    with_eps = sum(1 for r in rows if r["earnings_quarters"] > 0)
    with_rev = sum(1 for r in rows if r["revenue_quarters"] > 0)
    with_pt = sum(1 for r in rows if r["has_price_targets"])
    errors = sum(1 for r in rows if r["error"])

    by_tier = {"large": [], "mid": [], "small": []}
    for r in rows:
        by_tier[r["tier"]].append(r)

    def tier_pct(tier: str) -> str:
        items = by_tier[tier]
        if not items:
            return "n/a"
        covered = sum(1 for r in items if r["has_coverage"])
        return f"{covered}/{len(items)} ({100 * covered / len(items):.0f}%)"

    md = []
    md.append("# yfinance Indian-Stock Analyst Coverage — Spike Report")
    md.append("")
    md.append(f"- **Ran at:** {spike['ran_at']}")
    md.append(f"- **yfinance version:** {spike['yfinance_version']}")
    md.append(f"- **Python:** {spike['python_version'].split()[0]}")
    md.append("")

    # ── Headline ────────────────────────────────────────────
    md.append("## Headline numbers")
    md.append("")
    md.append(f"- **Total tickers tested:** {total}")
    md.append(f"- **Any analyst coverage:** {with_coverage}/{total} "
              f"({100 * with_coverage / total:.0f}%)")
    md.append(f"- **Earnings estimates present:** {with_eps}/{total}")
    md.append(f"- **Revenue estimates present:** {with_rev}/{total}")
    md.append(f"- **Price targets present:** {with_pt}/{total}")
    md.append(f"- **Hard fetch failures:** {errors}/{total}")
    md.append("")

    # ── By tier ─────────────────────────────────────────────
    md.append("## Coverage by cap tier")
    md.append("")
    md.append("| Tier | Coverage rate |")
    md.append("|---|---|")
    md.append(f"| Large-cap | {tier_pct('large')} |")
    md.append(f"| Mid-cap | {tier_pct('mid')} |")
    md.append(f"| Small-cap | {tier_pct('small')} |")
    md.append("")

    # ── Decision criteria ───────────────────────────────────
    md.append("## Decision criteria for the puppy")
    md.append("")
    md.append("- **≥80% large-cap + ≥50% mid-cap covered** → ship the module as-is, "
              "fundamentals analysis viable for top-N stocks.")
    md.append("- **≥60% large-cap, <50% mid-cap** → ship with `has_coverage` gating; "
              "mid/small caps fall back to PDF parsing only.")
    md.append("- **<60% large-cap covered** → yfinance is insufficient for v1; "
              "defer estimates to a later iteration with a different source "
              "(Trendlyne scrape / Screener / paid API).")
    md.append("")

    # ── Per-ticker detail ───────────────────────────────────
    md.append("## Per-ticker detail")
    md.append("")
    md.append("| Symbol | Tier | Coverage | EPS Qs | Rev Qs | Recs | PriceTgt | #Analysts | Error |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| {r['symbol']} | {r['tier']} | "
            f"{'✅' if r['has_coverage'] else '❌'} | "
            f"{r['earnings_quarters']} | {r['revenue_quarters']} | "
            f"{r['recommendation_snapshots']} | "
            f"{'✅' if r['has_price_targets'] else '❌'} | "
            f"{r['num_analysts_current_quarter'] or '-'} | "
            f"{(r['error'] or '')[:50]} |"
        )
    md.append("")

    # ── Sample dumps ────────────────────────────────────────
    md.append("## Sample data dumps (for the puppy to inspect shape)")
    md.append("")
    for tier in ("large", "mid", "small"):
        if tier in samples:
            md.append(f"### {tier.capitalize()}-cap sample: `{samples[tier]['symbol']}`")
            md.append("")
            md.append("```json")
            md.append(json.dumps(samples[tier], indent=2, default=str))
            md.append("```")
            md.append("")
        else:
            md.append(f"### {tier.capitalize()}-cap sample")
            md.append("")
            md.append("_No covered ticker in this tier — nothing to dump._")
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
    out_path = reports_dir / f"estimates_coverage_{timestamp}.md"

    out_path.write_text(_render_report(spike), encoding="utf-8")

    # Also dump raw JSON next to the markdown for any deeper analysis
    json_path = reports_dir / f"estimates_coverage_{timestamp}.json"
    json_path.write_text(json.dumps(spike, indent=2, default=str), encoding="utf-8")

    print()
    print("=" * 60)
    print(f"✅ Markdown report: {out_path.relative_to(ROOT)}")
    print(f"✅ Raw JSON:        {json_path.relative_to(ROOT)}")
    print()
    print("Bring those files back on VPN and ask the puppy to review.")


if __name__ == "__main__":
    main()
