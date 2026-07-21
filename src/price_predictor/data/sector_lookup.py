"""yfinance sector lookup with a persistent, resumable cache.

WHY
===
Every NSE ticker belongs to a real industry, but NSE's own sector feed is
geo-blocked outside India. yfinance is the only per-stock sector source
reachable from a US laptop, and it returns a GICS-style sector for
essentially every listed name (e.g. 'Technology', 'Energy',
'Financial Services').

This module backfills a real sector for a big list of tickers. The catch
is scale: ~2,300 tickers means ~2,300 yfinance calls, and Yahoo rate-limits
hard. So the design is built around SURVIVING interruption:

  - Every resolved ticker is written to a JSON cache on disk.
  - A transient fetch failure (rate-limit, network) leaves the ticker
    UNCACHED, so a rerun retries only the ones that didn't resolve.
  - A genuine "yfinance has no sector for this name" is cached as
    UNKNOWN_SECTOR so we don't retry it forever.

Run it, let it get as far as it can, rerun until everything resolves.

TESTABILITY
===========
The network call is injected via the `fetcher` parameter, so tests never
touch yfinance — they pass a fake that returns/raises deterministically.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

from loguru import logger

# Placeholder for a name yfinance genuinely has no sector for. Matches the
# label the search-index builder used before this backfill existed, so the
# UI/search behaviour is unchanged for those (rare) stragglers.
UNKNOWN_SECTOR = "NSE Listed"


class SectorFetchError(Exception):
    """Transient failure fetching a sector (rate-limit, network, empty info).

    Distinct from 'yfinance returned data but had no sector field'
    (which is a legitimate None). Only transient failures are retryable,
    so the backfill leaves them uncached rather than poisoning the cache.
    """


def yf_sector(ticker: str) -> str | None:
    """Fetch the yfinance sector for one ticker.

    Returns:
        The sector string (e.g. 'Technology'), or None when yfinance
        returned data but genuinely has no sector for this ticker.

    Raises:
        SectorFetchError: transient failure (rate-limit/network/empty
            response) — the caller should retry later, NOT cache this.
    """
    import yfinance as yf

    try:
        info = yf.Ticker(ticker).info
    except Exception as e:  # yfinance raises a zoo of error types
        raise SectorFetchError(f"{ticker}: {type(e).__name__}: {e}") from e

    # A rate-limited / blocked call often comes back as None or a near-empty
    # dict rather than raising. Treat that as transient so we retry.
    if not info or len(info) < 3:
        raise SectorFetchError(f"{ticker}: empty info (likely rate-limited)")

    sector = info.get("sector")
    if isinstance(sector, str) and sector.strip():
        return sector.strip()
    return None  # real 'no sector' — cache as UNKNOWN so we stop retrying


def load_cache(path: Path) -> dict[str, str]:
    """Load the ticker→sector cache. Returns {} if missing/corrupt."""
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as fp:
            data = json.load(fp)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("sector cache unreadable ({}); starting fresh", e)
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def save_cache(path: Path, cache: dict[str, str]) -> None:
    """Atomically write the cache (tmp file + rename) so a crash mid-write
    can never corrupt the on-disk cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fp:
        json.dump(cache, fp, indent=0, sort_keys=True)
    tmp.replace(path)


def backfill_sectors(
    tickers: Iterable[str],
    *,
    cache_path: Path,
    fetcher: Callable[[str], str | None] = yf_sector,
    max_workers: int = 4,
    save_every: int = 25,
    progress: Callable[[int, int, int], None] | None = None,
) -> dict[str, str]:
    """Resolve sectors for `tickers`, resuming from and updating the cache.

    Args:
        tickers: ticker symbols (yfinance form, e.g. 'RELIANCE.NS').
        cache_path: JSON cache; already-resolved tickers are skipped.
        fetcher: injectable per-ticker fetch (defaults to real yfinance).
        max_workers: thread pool size. Kept low — Yahoo rate-limits hard.
        save_every: flush the cache to disk every N resolutions.
        progress: optional callback(done, total, failed) for a progress line.

    Returns:
        The full cache dict (ticker → sector), including prior entries.
        Tickers that hit a transient failure are absent (retry next run).
    """
    cache = load_cache(cache_path)
    # Preserve order, drop dups, skip already-cached.
    todo = [t for t in dict.fromkeys(tickers) if t not in cache]
    total = len(todo)
    if total == 0:
        return cache

    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetcher, t): t for t in todo}
        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                sector = fut.result()
            except SectorFetchError:
                failed += 1  # leave uncached → retried on the next run
                continue
            except Exception as e:  # unexpected — treat as transient too
                logger.debug("unexpected sector error {}: {}", ticker, e)
                failed += 1
                continue
            cache[ticker] = sector or UNKNOWN_SECTOR
            done += 1
            if progress is not None:
                progress(done, total, failed)
            if done % save_every == 0:
                save_cache(cache_path, cache)

    save_cache(cache_path, cache)
    return cache
