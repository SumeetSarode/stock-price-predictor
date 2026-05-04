"""Stock registry + lookup, backed by data/kb/stocks.json.

This module is the agent's "what stocks exist" knowledge. It loads the
universal registry (built by scripts/bootstrap_indices.py from Wikipedia)
and exposes a small API:

    Stock                 -- Pydantic model for one constituent
    all_stocks()          -- every stock in the registry
    by_index(code)        -- stocks belonging to a given index
    lookup(query, index)  -- resolve user input ("HDFC", "infosys", "RELIANCE.NS")

Why this isn't an ADK tool
==========================
Looking up "what's HDFC's canonical ticker?" returns the same answer every
time. That's knowledge, not action -- it belongs in a Python module
imported by tools (and used during their execution), not in a separate
LLM-callable tool that costs a round-trip per call.

ADK tools should be reserved for things with side effects (writes), live
data (fetches), or genuine non-determinism. KB lookup has none of those.

Why fuzzy matching (instead of an alias dict)
=============================================
Maintaining a hardcoded alias map (HDFC -> HDFCBANK, L&T -> LT) is fragile:
    - Every future merger / rename needs a manual entry.
    - Aliases drift from reality silently.

Fuzzy matching against the LIVE company-name field self-heals:
    - "HDFC" -> matches "HDFC Bank" -> HDFCBANK (because Wikipedia's current
      Nifty 50 has HDFC Bank, not HDFC Ltd which merged in 2023).
    - "L&T" -> matches "Larsen & Toubro" -> LT.
    - When SBIN is renamed (hypothetically), Wikipedia updates -> we pick
      up the new name on next bootstrap, no code change.
"""
from __future__ import annotations

import functools
import json
import re
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Final

from pydantic import BaseModel, Field

from price_predictor.config.settings import settings

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
STOCKS_FILE: Final[Path] = settings.kb_dir / "stocks.json"
INDICES_FILE: Final[Path] = settings.kb_dir / "indices.json"

# Fuzzy threshold. difflib's ratio is in [0, 1]; 0.5 is generous enough
# to catch "HDFC" -> "HDFC Bank" (~0.62) but tight enough to reject pure
# garbage. Tunable; covered by tests.
FUZZY_CUTOFF: Final[float] = 0.5


# ─────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────
class Stock(BaseModel):
    """One constituent of one or more stock indices.

    `ticker` is the bare NSE symbol (e.g. "HDFCBANK") -- no exchange
    suffix. Use `yfinance_symbol` when you need the suffixed form for
    yfinance API calls.
    """

    ticker: str = Field(min_length=1)
    company_name: str = Field(min_length=1)
    sector: str = ""
    date_added: str = ""
    indices: list[str] = Field(default_factory=list)

    @property
    def yfinance_symbol(self) -> str:
        """yfinance API form -- NSE tickers need a `.NS` suffix."""
        return f"{self.ticker}.NS"


# ─────────────────────────────────────────────────────────────────
# Loaders (cached)
# ─────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def all_stocks() -> tuple[Stock, ...]:
    """Load + cache the full registry. Tuple so callers can't mutate the cache."""
    if not STOCKS_FILE.exists():
        raise FileNotFoundError(
            f"Stock registry not found at {STOCKS_FILE}. "
            f"Run: uv run python scripts/bootstrap_indices.py --index NIFTY50"
        )
    payload = json.loads(STOCKS_FILE.read_text())
    return tuple(Stock(**s) for s in payload.get("stocks", []))


def _reset_cache() -> None:
    """Clear the lru_cache. Used by tests that point STOCKS_FILE elsewhere."""
    all_stocks.cache_clear()


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────
def by_index(index_code: str) -> list[Stock]:
    """All stocks belonging to a given index, e.g. 'NIFTY50'.

    Case-insensitive on the index code. Returns [] if the index is unknown
    -- callers can decide whether that's an error.
    """
    code = index_code.upper()
    return [s for s in all_stocks() if code in {i.upper() for i in s.indices}]


def _normalize_query(query: str) -> str:
    """Strip exchange suffix + non-alphanumeric noise, uppercase.

    Handles common user inputs: "RELIANCE.NS", "reliance ", "Reliance!"
    all collapse to "RELIANCE".
    """
    # Drop yfinance suffixes (.NS, .BO, .BSE, etc.)
    stripped = re.sub(r"\.[A-Za-z]+$", "", query.strip())
    # Keep only alphanumerics + spaces for the comparison form
    cleaned = re.sub(r"[^A-Za-z0-9\s&]", "", stripped).strip()
    return cleaned.upper()


def lookup(query: str, *, index: str | None = None) -> Stock | None:
    """Resolve a user query (ticker / name / partial / common alias) to a Stock.

    Strategy (first match wins):
        1. Exact ticker match  -- "RELIANCE", "HDFCBANK.NS"
        2. Exact company name  -- "Infosys", "HDFC Bank"
        3. Substring on company name  -- "infosys" matches "Infosys Ltd"
        4. Fuzzy match on company name  -- "HDFC" -> "HDFC Bank" (FUZZY_CUTOFF)

    Args:
        query: User input. Case-insensitive. `.NS` suffix tolerated.
        index: If set, restrict candidates to stocks in this index only.
            Useful for "give me a Bank Nifty stock matching HDFC".

    Returns:
        Best matching Stock, or None if nothing crosses the fuzzy threshold.
    """
    if not query or not query.strip():
        return None

    pool = by_index(index) if index else list(all_stocks())
    if not pool:
        return None

    norm_query = _normalize_query(query)
    # Bail if normalization stripped everything (e.g. "!!!!" -> "").
    # Without this guard, the substring-match tier below would match every
    # stock ("" is in every string) and return whichever has the shortest
    # company name. That's a fun bug, but very wrong.
    if not norm_query:
        return None

    # 1. Exact ticker match
    for stock in pool:
        if stock.ticker.upper() == norm_query:
            return stock

    # 2. Exact company name (case-insensitive)
    for stock in pool:
        if stock.company_name.upper() == norm_query:
            return stock

    # 3. Substring match on company name (e.g. "infosys" in "Infosys Ltd")
    substring_hits = [
        s for s in pool if norm_query in s.company_name.upper()
    ]
    if substring_hits:
        # Prefer the SHORTEST name (most specific match -- "HDFC" prefers
        # "HDFC Bank" over the longer "HDFC Asset Management").
        return min(substring_hits, key=lambda s: len(s.company_name))

    # 4. Fuzzy match on company names
    names = [s.company_name for s in pool]
    matches = get_close_matches(query, names, n=1, cutoff=FUZZY_CUTOFF)
    if matches:
        return next(s for s in pool if s.company_name == matches[0])

    # 4b. Last resort: fuzzy on ticker too. Helps with typos like
    # "RELIENCE" -> "RELIANCE". SequenceMatcher gives us the score directly.
    best_ticker_match: Stock | None = None
    best_ratio = FUZZY_CUTOFF  # threshold; only beat-it counts
    for s in pool:
        ratio = SequenceMatcher(None, norm_query, s.ticker.upper()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_ticker_match = s
    return best_ticker_match
