"""Smoke test: fetch RELIANCE bars through the full resilient chain.

Prints which providers succeeded / fell back so we can SEE the new
default chain in action. Also writes a full transcript to
`scripts/smoke_chain_output.txt` so the result can be reviewed later
(useful when running on a different network than where you're reading
this from).

Run with:
    uv run python scripts/smoke_chain.py
"""
from __future__ import annotations

import contextlib
import io
import sys
import time
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

from loguru import logger

from price_predictor.config.settings import settings
from price_predictor.data.prices import fetch_ohlcv, reset_default_fetcher
from price_predictor.data.providers import (
    PriceFetchError,
    build_provider,
)

# Output file lives next to the script so it's easy to find and clean up.
OUTPUT_FILE = Path(__file__).parent / "smoke_chain_output.txt"


def _try_one(name: str, ticker: str, start: date, end: date) -> tuple[bool, str, float, int]:
    """Build a single provider, fetch, return (ok, msg, elapsed_s, rows)."""
    t0 = time.monotonic()
    try:
        prov = build_provider(name)
        df = prov.fetch_ohlcv(ticker, start, end)
        elapsed = time.monotonic() - t0
        return True, f"OK ({len(df)} rows, last close={df['close'].iloc[-1]:.2f})", elapsed, len(df)
    except PriceFetchError as e:
        elapsed = time.monotonic() - t0
        return False, f"PriceFetchError: {str(e)[:160]}", elapsed, 0
    except Exception as e:
        elapsed = time.monotonic() - t0
        return False, f"{type(e).__name__}: {str(e)[:160]}", elapsed, 0


def main() -> int:
    logger.remove()
    logger.add(sys.stderr, level="WARNING")  # quiet down providers' debug logs

    ticker = "RELIANCE"
    end = date.today()
    start = end - timedelta(days=14)  # ~10 trading days

    print("=" * 70)
    print(f"SMOKE TEST: {ticker} from {start} to {end}")
    print(f"Run at: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Configured chain: {settings.effective_price_chain()}")
    print("=" * 70)
    print()

    # Step 1: try each provider in isolation so we see WHO works
    print("--- Per-provider results (independent calls) ---")
    print(f"{'PROVIDER':<18} {'STATUS':<8} {'ELAPSED':<10} {'DETAIL'}")
    print("-" * 70)
    chain_results: list[tuple[str, bool]] = []
    for name in settings.effective_price_chain():
        # jugaad/yfinance want bare 'RELIANCE'; some providers might prefer
        # '.NS'. Pass bare; each provider normalises internally.
        ok, msg, elapsed, _rows = _try_one(name, ticker, start, end)
        status = "✅ OK" if ok else "❌ FAIL"
        print(f"{name:<18} {status:<8} {elapsed:>5.2f}s    {msg}")
        chain_results.append((name, ok))

    print()

    # Step 2: run the resilient chain end-to-end
    print("--- Resilient chain end-to-end ---")
    reset_default_fetcher()  # ensure we use the current settings
    t0 = time.monotonic()
    try:
        df = fetch_ohlcv(ticker, start, end)
        elapsed = time.monotonic() - t0
        print(f"✅ SUCCESS in {elapsed:.2f}s — {len(df)} bars")
        print()
        print(df.tail(3).to_string())
    except PriceFetchError as e:
        elapsed = time.monotonic() - t0
        print(f"❌ ALL PROVIDERS FAILED after {elapsed:.2f}s")
        print(f"   last error: {e}")
        return 1

    print()
    print("=" * 70)
    print("Summary:")
    print(f"  Working providers: {[n for n, ok in chain_results if ok]}")
    print(f"  Failed providers:  {[n for n, ok in chain_results if not ok]}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    # Capture every print() call so we can both show the user AND save
    # to a file for later review. Tee pattern via StringIO + redirect.
    buffer = io.StringIO()

    class _Tee:
        """Write to BOTH the real stdout and the in-memory buffer."""
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    real_stdout = sys.stdout
    tee = _Tee(real_stdout, buffer)
    exit_code = 1
    try:
        with contextlib.redirect_stdout(tee):
            exit_code = main()
    except Exception:
        # Capture full traceback into the report so we can debug from the file.
        with contextlib.redirect_stdout(tee):
            print("\n!!! UNCAUGHT EXCEPTION !!!")
            traceback.print_exc(file=tee)
    finally:
        OUTPUT_FILE.write_text(buffer.getvalue())
        real_stdout.write(f"\n\n[smoke_chain] Full transcript saved to: {OUTPUT_FILE}\n")
    sys.exit(exit_code)
