"""Diagnose whether news fetching (GDELT) works on THIS machine.

News comes from GDELT's free Doc API at api.gdeltproject.org. If news isn't
loading, it's almost always network reachability -- not a code bug. Run:

    python scripts/check_news.py

It checks, in order:
    1. Is an HTTP(S) proxy configured? (corporate networks usually need one)
    2. Can we reach GDELT at all? (a live fetch for a well-known ticker)
    3. Prints a plain-English verdict + the fix if it's broken.

Read-only. Never raises -- it's a diagnostic, not part of the app.
"""
from __future__ import annotations

import asyncio
import os
from datetime import date, timedelta

from price_predictor.data.news import NewsFetchError, fetch_news


def _proxy_status() -> str:
    https = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    http = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    if https or http:
        return f"proxy IS set (HTTPS_PROXY={https or '-'}, HTTP_PROXY={http or '-'})"
    return "no proxy set (direct connection)"


async def _probe() -> tuple[bool, int, str]:
    """Try a tiny live GDELT fetch. Returns (ok, row_count, detail)."""
    end = date.today()
    start = end - timedelta(days=7)
    try:
        df = await fetch_news("Infosys", start.isoformat(), end.isoformat(), max_records=3)
        return True, len(df), "ok"
    except NewsFetchError as exc:
        return False, 0, f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # never raise from a diagnostic
        return False, 0, f"{type(exc).__name__}: {exc}"


def main() -> None:
    print("=== News (GDELT) connectivity check ===\n")
    print(f"1. Proxy: {_proxy_status()}")

    ok, rows, detail = asyncio.run(_probe())
    print(f"2. Live GDELT fetch for 'Infosys': {'OK' if ok else 'FAILED'} ({detail})")

    print("\n--- Verdict ---")
    if ok and rows > 0:
        print(f"News is WORKING. Got {rows} headline(s). Nothing to fix.")
    elif ok and rows == 0:
        print("Reached GDELT fine, but it returned 0 articles for the test query.")
        print("That's usually a quiet news week, not a fault. News is working.")
    else:
        print("Could NOT reach GDELT from this machine. The code is fine -- this")
        print("is a network issue. Most likely one of:")
        print("  * You're on a corporate network/VPN that blocks GDELT directly.")
        print("    Fix: set HTTPS_PROXY (and HTTP_PROXY) in your .env to your")
        print("    network's proxy, then relaunch. The app wires it into every")
        print("    HTTP client automatically.")
        print("  * GDELT is briefly rate-limiting or down. Fix: try again later.")
        print("  * No internet. Fix: check your connection.")
        print("\nNote: predictions still run without news -- news only *enriches*")
        print("them. A news outage never blocks a prediction.")


if __name__ == "__main__":
    main()
