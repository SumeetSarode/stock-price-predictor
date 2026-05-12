"""End-to-end smoke for MA-cross wiring (Stage 1: data → tool).

Pulls real OHLCV via the existing shared cache, runs `get_trend()` against
a handful of liquid NSE/US tickers, and prints the new `ma_crosses` field
+ the rationale lines that mention the cross.

Usage:
    uv run python scripts/verify_ma_cross_e2e.py

This is a SMOKE script, not a test. It hits the network. Skip in CI.
"""
from __future__ import annotations

import asyncio
import json

from price_predictor.agents.technical_agent.tools.get_trend import get_trend

# Mix: 2 NSE blue chips + 1 US tech for breadth. Different timezones,
# different vol regimes, different cross histories.
TICKERS = ["RELIANCE.NS", "TCS.NS", "AAPL"]


async def _verify_one(ticker: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {ticker}")
    print(f"{'═' * 60}")

    resp = await get_trend(ticker, sensitivity="standard")

    if resp.get("status") != "success":
        print(f"  ✗ tool error: {resp}")
        return

    derived = resp.get("derived", {})
    ma_crosses = derived.get("ma_crosses")

    if ma_crosses is None:
        print("  ✗ ma_crosses field missing from derived!")
        return

    print(f"  ma_crosses keys: {sorted(ma_crosses.keys())}")
    print()
    for pair_key, struct in ma_crosses.items():
        print(f"  [{pair_key}]")
        print(f"    {json.dumps(struct, indent=4, default=str)}")

    print()
    print(f"  signal: {resp.get('signal')}  strength: {resp.get('strength')}")
    print()
    print("  rationale:")
    for line in resp.get("rationale", ()):
        marker = "  ⮕ " if any(
            tok in line.lower()
            for tok in ("cross", "golden", "death")
        ) else "    "
        print(f"  {marker}{line}")


async def main() -> None:
    for ticker in TICKERS:
        try:
            await _verify_one(ticker)
        except Exception as exc:  # noqa: BLE001 — smoke script
            print(f"\n  ✗ {ticker} failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
