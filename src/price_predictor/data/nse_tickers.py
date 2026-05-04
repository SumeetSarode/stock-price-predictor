"""Indian market ticker resolution — single source of truth for aliases.

WHY THIS EXISTS
===============
The NSE ticker space has gotchas that no LLM training cut-off reliably knows:

    - HDFC Ltd merged into HDFC Bank on 2023-07-01. HDFC.NS is delisted.
      Users still say "HDFC" meaning HDFC Bank.
    - Common names diverge from tickers in non-obvious ways
      ("L&T" -> "LT", "M&M" -> "M&M", "TVS Motor" -> "TVSMOTOR").
    - Some tickers have DVR / dual-class quirks (TATAMOTORS vs TATAMOTORS-DVR).

Centralizing here means:
    - Agents don't hard-code these in their prompts (DRY).
    - When NSE has another merger / name change, ONE file gets edited.
    - Tools and prompts both read from the same source.

WHAT THIS IS NOT
================
Not a comprehensive Nifty50 universe — that lives in the KB. This is purely
the *aliases / corrections* layer. If an entry isn't here, the agent's normal
ticker resolution (company name -> NAME.NS) is correct.
"""
from __future__ import annotations

# Mapping of "what the user might say or use" -> canonical NSE symbol
# (no .NS suffix; callers add the suffix as needed).
#
# Keys are normalized: uppercase, no spaces, no .NS suffix.
# Add entries when a real failure surfaces a missing one (don't speculate).
TICKER_ALIASES: dict[str, str] = {
    # Mergers / corporate actions
    "HDFC": "HDFCBANK",          # HDFC Ltd merged into HDFC Bank, 2023-07-01
    "HDFCLTD": "HDFCBANK",       # alt form sometimes seen
    # Common name <-> ticker quirks (extend as users surface more)
    "LT": "LT",                  # "L&T" with the ampersand stripped
    "LARSEN": "LT",
    "LARSENTOUBRO": "LT",
    "MAHINDRA": "M&M",
    "MARUTI": "MARUTI",          # confirms canonical (Maruti Suzuki India)
    "TATAMOTOR": "TATAMOTORS",   # singular vs plural
    "INFY": "INFY",
    "INFOSYS": "INFY",
    "TCS": "TCS",
    "RELIANCE": "RELIANCE",
}


def _normalize(raw: str) -> str:
    """Normalize user input for lookup: strip, uppercase, remove .NS / spaces."""
    return raw.strip().upper().replace(".NS", "").replace(" ", "").replace("&", "&")


def suggest_alternative(ticker: str) -> str | None:
    """Return a canonical NSE symbol if `ticker` has a known alias, else None.

    Args:
        ticker: User-provided string. Any of: 'HDFC', 'hdfc', 'HDFC.NS',
                'Reliance Industries' (won't match — not in alias table),
                'INFOSYS'.

    Returns:
        Canonical bare NSE symbol (no .NS) if an alias exists, else None.
        Returning None means "no opinion — caller should treat the input
        as already canonical."

    Why None instead of returning the input unchanged:
        Lets callers distinguish "we have a redirect" from "we have nothing
        smart to say". Avoids the agent looping on already-correct tickers.
    """
    if not ticker:
        return None
    canonical = TICKER_ALIASES.get(_normalize(ticker))
    # Don't suggest the same ticker back — that's not a redirect, it's a no-op.
    if canonical and _normalize(canonical) != _normalize(ticker):
        return canonical
    return None
