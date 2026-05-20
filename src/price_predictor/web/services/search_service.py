"""Ticker search service — fast in-memory autocomplete over the bundled
Nifty 500 CSV.

Loaded once at import time (~200 rows, <1ms). Search is brute-force O(n)
substring matching — fine for n=200, would need a trie if we ever
expand to thousands of symbols.

Ranking rules (in order of priority — see ``_score`` for details):
  1. Exact ticker match     → score 1000  (e.g. "RELIANCE" -> RELIANCE.NS)
  2. Ticker prefix match    → score  900  (e.g. "REL" -> RELIANCE.NS)
  3. Name prefix match      → score  800  (e.g. "Reli" -> Reliance Industries)
  4. Ticker contains query  → score  600
  5. Name word starts with  → score  500  (e.g. "Bank" -> "...Bank of Baroda")
  6. Name contains query    → score  300
  Nifty 50 bonus            → +50 (tied scores: N50 wins)

Returns at most ``limit`` matches as plain dicts so the API layer can
serialize directly to JSON or pass to a template.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from price_predictor.web.settings import settings


# ── Data model ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Stock:
    """One row in the search index. Immutable & hashable."""

    ticker: str           # "RELIANCE.NS"
    name: str             # "Reliance Industries"
    sector: str           # "Oil, Gas & Consumable Fuels"
    is_nifty50: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "sector": self.sector,
            "is_nifty50": self.is_nifty50,
        }


# ── Index loading (singleton, cached) ───────────────────────────────


def _csv_path() -> Path:
    """Resolve the bundled CSV path. Indirected for test override."""
    return settings.frontend_dir / "data" / "nifty500.csv"


@lru_cache(maxsize=1)
def _load_index() -> tuple[Stock, ...]:
    """Read the bundled CSV once and return an immutable tuple.

    Cached so subsequent calls return the same object — no repeated
    file I/O. The tuple is returned (not a list) so callers can't
    accidentally mutate the shared index.
    """
    path = _csv_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Search index CSV not found at {path}. "
            "Run `python scripts/build_search_index.py` to generate it."
        )

    stocks: list[Stock] = []
    with path.open(encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            stocks.append(
                Stock(
                    ticker=row["ticker"],
                    name=row["name"],
                    sector=row["sector"],
                    is_nifty50=row["nifty50"].lower() == "true",
                )
            )
    return tuple(stocks)


# ── Ranking ─────────────────────────────────────────────────────────


def _score(stock: Stock, q: str) -> int:
    """Return a relevance score; 0 means no match.

    Higher = more relevant. See module docstring for the rubric.
    """
    ticker_base = stock.ticker.removesuffix(".NS").lower()
    name_lower = stock.name.lower()

    score = 0

    if ticker_base == q:
        score = 1000
    elif ticker_base.startswith(q):
        score = 900
    elif name_lower.startswith(q):
        score = 800
    elif q in ticker_base:
        score = 600
    elif any(word.startswith(q) for word in name_lower.split()):
        # Match any word boundary in the company name. E.g. searching
        # "bank" matches "...Bank of Baroda" via the second word.
        score = 500
    elif q in name_lower:
        score = 300
    else:
        return 0

    if stock.is_nifty50:
        score += 50

    return score


def search(query: str, limit: int = 8) -> list[Stock]:
    """Return up to ``limit`` matches ranked by relevance.

    Empty / whitespace-only query returns []. We don't show "all tickers"
    on empty input — that's the dropdown's caller's job to handle.
    """
    q = query.strip().lower()
    if not q:
        return []

    index = _load_index()
    scored: list[tuple[int, Stock]] = []
    for stock in index:
        s = _score(stock, q)
        if s > 0:
            scored.append((s, stock))

    # Sort by score descending. Stable sort means same-score entries
    # keep their CSV order (which already has N50 first alphabetically).
    scored.sort(key=lambda pair: pair[0], reverse=True)

    return [stock for _, stock in scored[:limit]]


def get_by_ticker(ticker: str) -> Stock | None:
    """Look up exactly one stock by ticker (case-insensitive, .NS optional).

    Returns None if not found — caller decides whether that's a 404
    or "let it through to the predictor as a free-form ticker".
    """
    needle = ticker.strip().upper()
    if not needle.endswith(".NS"):
        needle = f"{needle}.NS"
    for stock in _load_index():
        if stock.ticker.upper() == needle:
            return stock
    return None


def all_nifty50() -> list[Stock]:
    """Return all Nifty 50 stocks in alphabetic order.

    Used by the dashboard table in substep 2B.
    """
    return [s for s in _load_index() if s.is_nifty50]


# ── Convenience for diagnostics ─────────────────────────────────────


def index_stats() -> dict[str, int]:
    """Quick stats for the /api/health endpoint and tests."""
    index = _load_index()
    return {
        "total": len(index),
        "nifty50": sum(1 for s in index if s.is_nifty50),
        "extras": sum(1 for s in index if not s.is_nifty50),
    }


def reset_cache_for_tests() -> None:
    """Drop the cached index. Tests use this to inject fresh data."""
    _load_index.cache_clear()
