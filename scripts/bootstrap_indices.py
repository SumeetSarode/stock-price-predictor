"""Bootstrap (or extend) the stock-index knowledge base from Wikipedia.

WHAT THIS DOES
==============
For a given index (default: NIFTY50), fetches the constituents table from
Wikipedia, normalizes it, and MERGES the result into:
    data/kb/stocks.json    -- universal stock registry, one entry per ticker
    data/kb/indices.json   -- metadata per index (source URL, count, last_updated)

"Merge" = if `data/kb/stocks.json` already exists, an existing stock that's
also in this index gets its `indices` list extended (deduped). New stocks
are appended. Nothing is silently overwritten.

WHY THIS DESIGN (universal registry vs per-index files)
=======================================================
A stock can be in multiple indices simultaneously (HDFCBANK is in NIFTY50,
BANKNIFTY, NIFTY FINSERV, ...). Per-index JSON files would duplicate that
stock's metadata N times -> drift risk + dedupe pain at lookup. A single
registry keyed by ticker, with `indices: list[str]` membership, is the
clean Open/Closed shape: open for extension (add an index), closed for
modification (existing rows get appended-to, not rewritten).

HOW TO ADD A NEW INDEX
======================
1. Add an entry to `INDEX_SOURCES` below with the Wikipedia URL and
   expected constituent count.
2. Run: `uv run python scripts/bootstrap_indices.py --index BANKNIFTY`
3. Verify the diff in `data/kb/stocks.json` (existing stocks gain a new entry
   in their `indices` list; new stocks are appended).
4. Commit.

That's it. No code changes outside this script's INDEX_SOURCES dict.

USAGE
=====
    uv run python scripts/bootstrap_indices.py --index NIFTY50
    uv run python scripts/bootstrap_indices.py --index BANKNIFTY  # (future)

WHEN TO RE-RUN
==============
- After `git clone` (verify data is fresh)
- Quarterly, after index reconstitution
- When you add a new index to INDEX_SOURCES
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

# Wikipedia 403s the default Python urllib User-Agent. A real browser UA
# is required. Keep this current-ish; nothing about it is secret.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

DATA_DIR = Path("data") / "kb"  # matches settings.kb_dir; keep in sync
STOCKS_PATH = DATA_DIR / "stocks.json"
INDICES_PATH = DATA_DIR / "indices.json"

# Column-rename map. Wikipedia adds footnote refs like [15], [16] to
# headers; we strip those and normalize to snake_case.
# Different index pages may use slightly different headers (e.g. "Company"
# vs "Company name"); we match by substring rather than exact equality.
COLUMN_PATTERNS = {
    "company_name": ["company name", "company"],
    "ticker": ["symbol", "ticker"],
    "sector": ["sector", "industry"],
    "date_added": ["date added", "added"],  # often present, optional
}


@dataclass(frozen=True)
class IndexSource:
    """Metadata + parse hints for one Wikipedia-sourced index."""
    code: str                  # short code we use everywhere (NIFTY50, BANKNIFTY)
    display_name: str          # human-friendly name
    wikipedia_url: str         # source URL (must be Wikipedia for the parser)
    expected_count: int        # constituent count; bail if reality differs


# Adding a new index? Drop another entry here. That's the whole API.
INDEX_SOURCES: dict[str, IndexSource] = {
    "NIFTY50": IndexSource(
        code="NIFTY50",
        display_name="Nifty 50",
        wikipedia_url="https://en.wikipedia.org/wiki/NIFTY_50",
        expected_count=50,
    ),
    # "BANKNIFTY": IndexSource(
    #     code="BANKNIFTY",
    #     display_name="Nifty Bank",
    #     wikipedia_url="https://en.wikipedia.org/wiki/Nifty_Bank",
    #     expected_count=12,
    # ),
}


# ─────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────
def _strip_footnote_refs(text: object) -> str:
    """Remove Wikipedia footnote markers like '[15]' or '[h]' from a string."""
    return re.sub(r"\[[^\]]+\]", "", str(text)).strip()


def _resolve_column(df: pd.DataFrame, target: str) -> str | None:
    """Find a column in `df` whose stripped name matches any pattern for `target`.

    Returns the actual column name (so caller can index `df[name]`), or None
    if no column matches. Match is case-insensitive substring.
    """
    patterns = COLUMN_PATTERNS[target]
    for col in df.columns:
        clean = _strip_footnote_refs(col).lower()
        if any(p in clean for p in patterns):
            return col
    return None


def _parse_constituents(html: str, source: IndexSource) -> list[dict[str, str]]:
    """Find + parse the constituents table from Wikipedia HTML.

    Discriminator: a table whose row count == source.expected_count AND
    that has both a 'Symbol/Ticker' column and a 'Company' column. Shape
    alone matches sidebar tables that are coincidentally the same size.
    """
    all_tables = pd.read_html(StringIO(html))
    candidates = []
    for t in all_tables:
        if len(t) != source.expected_count:
            continue
        if _resolve_column(t, "ticker") and _resolve_column(t, "company_name"):
            candidates.append(t)

    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly 1 constituents table for {source.code} "
            f"({source.expected_count} rows + Symbol + Company columns); "
            f"found {len(candidates)}. Wikipedia layout may have changed."
        )

    raw = candidates[0]
    out: list[dict[str, str]] = []
    for _, row in raw.iterrows():
        record = {}
        for target in COLUMN_PATTERNS:
            col = _resolve_column(raw, target)
            # date_added is optional (some indices don't track it)
            if col is None:
                if target == "date_added":
                    record[target] = ""
                    continue
                raise ValueError(
                    f"Required column '{target}' not found in {source.code} table"
                )
            record[target] = _strip_footnote_refs(row[col])
        out.append(record)
    return out


def fetch_index_constituents(source: IndexSource) -> list[dict[str, str]]:
    """Fetch + parse one index's constituents from Wikipedia."""
    print(f"Fetching {source.wikipedia_url} ...")
    response = httpx.get(
        source.wikipedia_url,
        headers={"User-Agent": BROWSER_UA},
        timeout=30,
        follow_redirects=True,
    )
    response.raise_for_status()
    print(f"  status={response.status_code}, bytes={len(response.text):,}")

    constituents = _parse_constituents(response.text, source)
    print(f"  parsed {len(constituents)} constituents for {source.code}")
    return constituents


# ─────────────────────────────────────────────────────────────────
# Merge logic
# ─────────────────────────────────────────────────────────────────
def _load_existing_stocks(path: Path) -> dict[str, dict[str, Any]]:
    """Load existing stocks.json into a {ticker: stock_dict} map.

    Returns empty dict if file doesn't exist (first-run case).
    """
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {s["ticker"]: s for s in payload.get("stocks", [])}


def _load_existing_indices(path: Path) -> dict[str, dict[str, Any]]:
    """Load existing indices.json into a {code: metadata_dict} map."""
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def merge_index(
    constituents: list[dict[str, str]],
    source: IndexSource,
    existing_stocks: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], int, int]:
    """Merge a fetched index's constituents into the existing stock map.

    Returns:
        (merged_stocks, new_count, updated_count)
        - new_count: stocks that didn't exist before (added to map)
        - updated_count: stocks that existed and gained this index in their list
    """
    new_count = 0
    updated_count = 0
    for c in constituents:
        ticker = c["ticker"]
        if ticker in existing_stocks:
            stock = existing_stocks[ticker]
            indices = stock.setdefault("indices", [])
            if source.code not in indices:
                indices.append(source.code)
                indices.sort()
                updated_count += 1
        else:
            existing_stocks[ticker] = {
                "ticker": ticker,
                "company_name": c["company_name"],
                "sector": c["sector"],
                "date_added": c["date_added"],
                "indices": [source.code],
            }
            new_count += 1
    return existing_stocks, new_count, updated_count


def write_stocks(stocks: dict[str, dict[str, Any]], path: Path) -> None:
    """Write stocks.json, sorted by ticker for clean diffs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = sorted(stocks.values(), key=lambda s: s["ticker"])
    payload = {
        "count": len(records),
        "stocks": records,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_indices(indices: dict[str, dict[str, Any]], path: Path) -> None:
    """Write indices.json with metadata for each known index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sort keys for deterministic diffs.
    ordered = dict(sorted(indices.items()))
    path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────
def bootstrap(index_code: str) -> int:
    if index_code not in INDEX_SOURCES:
        print(
            f"ERROR: unknown index '{index_code}'. "
            f"Known: {list(INDEX_SOURCES)}",
            file=sys.stderr,
        )
        return 2

    source = INDEX_SOURCES[index_code]
    try:
        constituents = fetch_index_constituents(source)
    except Exception as e:
        print(f"ERROR fetching {index_code}: {e}", file=sys.stderr)
        return 1

    # Merge into existing universal registry.
    existing_stocks = _load_existing_stocks(STOCKS_PATH)
    merged, new_count, updated_count = merge_index(
        constituents, source, existing_stocks
    )
    write_stocks(merged, STOCKS_PATH)

    # Update index metadata.
    existing_indices = _load_existing_indices(INDICES_PATH)
    existing_indices[source.code] = {
        "display_name": source.display_name,
        "source_url": source.wikipedia_url,
        "constituent_count": source.expected_count,
        "last_updated": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_indices(existing_indices, INDICES_PATH)

    print(
        f"Merged {source.code}: {new_count} new stocks, "
        f"{updated_count} existing stocks gained membership"
    )
    print(f"  -> {STOCKS_PATH} now has {len(merged)} total stocks")
    print(f"  -> {INDICES_PATH} now tracks {len(existing_indices)} indices")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--index",
        default="NIFTY50",
        choices=list(INDEX_SOURCES),
        help="Which index to fetch + merge (see INDEX_SOURCES in this script).",
    )
    args = parser.parse_args()
    return bootstrap(args.index)


if __name__ == "__main__":
    raise SystemExit(main())
