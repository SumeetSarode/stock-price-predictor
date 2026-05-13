"""Bootstrap data/kb/index_membership.json from Wikipedia.

WHAT THIS DOES
==============
Scrapes Wikipedia's NIFTY 50 page for two tables:

1. The current constituents table (50 rows) -> the "anchor" of today's
   members + a name->symbol map for resolving historical references.
2. The "List of replacements since 2005" table (~70 rows) -> the
   add/remove event log.

It then:
- Filters events to those on/after HISTORY_STARTS (default 2015-01-01).
- Resolves every "Constituent excluded" / "Constituent included" company
  name to a yfinance ticker (e.g. "Tata Consultancy Services" -> "TCS.NS")
  using the current-50 map first, then HISTORICAL_NAME_TO_TICKER for
  delisted / merged companies that no longer appear in today's index.
- FAILS LOUD if any name can't be resolved -- print the unresolved set
  and ask the operator to add it to HISTORICAL_NAME_TO_TICKER. NO silent
  fallback; survivorship-bias defense that quietly drops events is worse
  than no defense.
- Sorts events DESC by (date, then included-before-excluded for stable
  same-day ordering) to match kb.membership's load-time invariant.
- Writes data/kb/index_membership.json.

WHY ONE FILE PER INDEX-CODE-IN-A-DICT (not multiple JSONs)
==========================================================
Same reasoning as data/kb/indices.json: a single file keyed by index
code lets us add NIFTY100, BANKNIFTY, etc. by extending a dict, not by
adding new file-discovery code.

USAGE
=====
    uv run python scripts/bootstrap_membership_history.py
    uv run python scripts/bootstrap_membership_history.py --history-starts 2010-01-01

WHEN TO RE-RUN
==============
- Quarterly, after NSE's semi-annual reconstitution settles. Wikipedia
  usually catches up within a week.
- After the Wikipedia layout changes (script will fail loud if so).
- NEVER between commits to "fix" a backtest -- the membership data
  defines the experiment; changing it mid-stream invalidates results.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/NIFTY_50"
INDEX_CODE = "NIFTY50"
INDEX_DISPLAY_NAME = "Nifty 50"
EXPECTED_CURRENT_COUNT = 50
DEFAULT_HISTORY_STARTS = date(2015, 1, 1)

# Wikipedia 403s the default Python urllib UA. Match the existing
# bootstrap_indices.py UA so any Wikipedia rate limiting is consistent.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

OUTPUT_PATH = Path("data") / "kb" / "index_membership.json"


# ─────────────────────────────────────────────────────────────────
# Manual override map: historical company names -> .NS tickers.
#
# Why this exists: the "List of replacements" table on Wikipedia uses
# full company names (e.g. "Bharat Petroleum"), and many of those
# companies are no longer in NIFTY 50 today, so their names don't
# appear in the current-constituents table that we use as the primary
# name->symbol map.
#
# Each entry is a name we've seen in the replacement log (at least once
# since HISTORY_STARTS) that needs an explicit ticker. Maintenance:
# if a future re-run reports an unresolved name, add it here with a
# brief verification note.
#
# Suffix: .NS (yfinance/NSE convention used everywhere in the
# predictor). Even for delisted companies we keep the .NS suffix --
# the membership module's symbol regex requires it, and it's the form
# the symbol HAD when it was an active NIFTY 50 member.
# ─────────────────────────────────────────────────────────────────
HISTORICAL_NAME_TO_TICKER: dict[str, str] = {
    # ── Cement / Materials ─────────────────────────────────────
    "ACC": "ACC.NS",
    "Ambuja Cements": "AMBUJACEM.NS",
    "Shree Cement": "SHREECEM.NS",
    # ── Pharma ─────────────────────────────────────────────────
    "Aurobindo Pharma": "AUROPHARMA.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "Lupin": "LUPIN.NS",
    # ── Energy / Oil & Gas ─────────────────────────────────────
    "Bharat Petroleum": "BPCL.NS",
    "Cairn India": "CAIRN.NS",     # merged into Vedanta 2017
    "GAIL": "GAIL.NS",
    "Hindustan Petroleum": "HINDPETRO.NS",
    "Indian Oil Corporation": "IOC.NS",
    # ── Banks / Financials ─────────────────────────────────────
    "Bank of Baroda": "BANKBARODA.NS",
    "HDFC": "HDFC.NS",             # merged into HDFC Bank 2023-07
    "IDFC": "IDFC.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Indiabulls Housing Finance": "IBULHSGFIN.NS",
    "Punjab National Bank": "PNB.NS",
    "Yes Bank": "YESBANK.NS",
    # ── Auto ───────────────────────────────────────────────────
    "Bosch India": "BOSCHLTD.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    # ── Metals / Mining ────────────────────────────────────────
    "BHEL": "BHEL.NS",
    "Jindal Steel & Power": "JINDALSTEL.NS",
    "NMDC": "NMDC.NS",
    "Tata Power": "TATAPOWER.NS",
    "Vedanta": "VEDL.NS",
    # ── Telecom / Media ────────────────────────────────────────
    "Bharti Infratel": "INFRATEL.NS",     # later merged into IndusTowers
    "Idea Cellular": "IDEA.NS",           # now Vodafone Idea, ticker unchanged
    "Zee Entertainment Enterprises": "ZEEL.NS",
    # ── IT services / consumer / misc ──────────────────────────
    "Britannia Industries": "BRITANNIA.NS",
    "DLF": "DLF.NS",
    "LTIMindtree": "LTIM.NS",     # post LTI + MINDTREE merger Oct 2022
    "UPL": "UPL.NS",
}


# ─────────────────────────────────────────────────────────────────
# Helpers (same shape as bootstrap_indices.py)
# ─────────────────────────────────────────────────────────────────
def _strip_footnote_refs(text: object) -> str:
    """Remove Wikipedia footnote markers like '[15]' or '[h]'."""
    return re.sub(r"\[[^\]]+\]", "", str(text)).strip()


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """If columns are a MultiIndex, take the most specific level."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            _strip_footnote_refs(c[-1]) for c in df.columns
        ]
    else:
        df = df.copy()
        df.columns = [_strip_footnote_refs(c) for c in df.columns]
    return df


def _parse_event_date(s: str) -> date:
    """Parse Wikipedia date strings ('25 February 2005', '1 April 2013')."""
    parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        raise ValueError(f"could not parse date {s!r}")
    return parsed.date()


# ─────────────────────────────────────────────────────────────────
# Fetching + parsing
# ─────────────────────────────────────────────────────────────────
def fetch_wikipedia_html() -> str:
    print(f"Fetching {WIKIPEDIA_URL} ...")
    response = httpx.get(
        WIKIPEDIA_URL,
        headers={"User-Agent": BROWSER_UA},
        timeout=30,
        follow_redirects=True,
    )
    response.raise_for_status()
    print(f"  status={response.status_code}, bytes={len(response.text):,}")
    return response.text


def parse_current_constituents(html: str) -> dict[str, str]:
    """Return {company_name -> .NS ticker} for today's 50 members.

    Discriminator: a 50-row table with both a Symbol-ish column and a
    Company-ish column. Identical pattern to bootstrap_indices.py so
    drift in either is caught the same way.
    """
    candidates = []
    for raw in pd.read_html(StringIO(html)):
        if len(raw) != EXPECTED_CURRENT_COUNT:
            continue
        df = _flatten_columns(raw)
        cols_lower = {c.lower(): c for c in df.columns}
        sym_col = next(
            (cols_lower[c] for c in cols_lower if "symbol" in c), None,
        )
        name_col = next(
            (cols_lower[c] for c in cols_lower if "company" in c), None,
        )
        if sym_col and name_col:
            candidates.append((df, sym_col, name_col))

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly 1 current-constituents table "
            f"({EXPECTED_CURRENT_COUNT} rows + Symbol + Company columns); "
            f"found {len(candidates)}. Wikipedia layout may have changed."
        )

    df, sym_col, name_col = candidates[0]
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        name = _strip_footnote_refs(row[name_col])
        sym = _strip_footnote_refs(row[sym_col]).upper()
        out[name] = f"{sym}.NS"
    if len(out) != EXPECTED_CURRENT_COUNT:
        raise RuntimeError(
            f"current-constituents map has {len(out)} entries; "
            f"expected {EXPECTED_CURRENT_COUNT} (duplicate names?)"
        )
    print(f"  parsed {len(out)} current constituents")
    return out


def parse_replacement_events(
    html: str,
) -> list[dict[str, str]]:
    """Return list of {excluded, included, date_str, reason} dicts.

    Discriminator: a multi-row table with both 'Constituent excluded'
    and 'Constituent included' columns. Doesn't filter by date here --
    that happens later so the bootstrap can warn about events outside
    the configured window.
    """
    candidates = []
    for raw in pd.read_html(StringIO(html)):
        df = _flatten_columns(raw)
        cols = set(df.columns)
        if {"Constituent excluded", "Constituent included"}.issubset(cols):
            candidates.append(df)

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly 1 replacements table; found {len(candidates)}. "
            f"Wikipedia layout may have changed."
        )

    df = candidates[0]
    date_col = next(
        (c for c in df.columns if "date" in c.lower()), None,
    )
    reason_col = next(
        (c for c in df.columns if "reason" in c.lower()), None,
    )
    if date_col is None:
        raise RuntimeError(
            "Replacements table missing a 'Date of replacement' column"
        )

    out: list[dict[str, str]] = []
    for _, row in df.iterrows():
        out.append({
            "excluded": _strip_footnote_refs(row["Constituent excluded"]),
            "included": _strip_footnote_refs(row["Constituent included"]),
            "date_str": _strip_footnote_refs(row[date_col]),
            "reason": (
                _strip_footnote_refs(row[reason_col])
                if reason_col else ""
            ),
        })
    print(f"  parsed {len(out)} total replacement events (all years)")
    return out


# ─────────────────────────────────────────────────────────────────
# Resolution + assembly
# ─────────────────────────────────────────────────────────────────
def resolve_name_to_ticker(
    name: str, current_map: dict[str, str],
) -> str | None:
    """Try current map first (today's 50), then historical override.

    Returns None if neither knows the name -- caller decides whether to
    crash or accumulate for batch reporting.
    """
    name = name.strip()
    if name in current_map:
        return current_map[name]
    if name in HISTORICAL_NAME_TO_TICKER:
        return HISTORICAL_NAME_TO_TICKER[name]
    return None


def build_payload(
    current_map: dict[str, str],
    raw_events: list[dict[str, str]],
    history_starts: date,
) -> dict[str, Any]:
    """Combine inputs into the JSON payload kb.membership expects.

    Two passes over the events:
    1. Resolve every name; collect failures so the operator sees them
       all at once instead of one-at-a-time-fix-and-rerun.
    2. Filter to >= history_starts and split each row into two events
       (one removed + one added) -- this matches the simpler add-or-
       remove primitive that members_on() uses.
    """
    today = date.today()

    # Pass 1: resolve all names from events we'd actually KEEP. We pre-
    # filter by date here so the operator isn't asked to map company
    # names from pre-history_starts events that we'll discard anyway.
    in_window: list[tuple[dict[str, str], date]] = []
    skipped_old = 0
    skipped_unparseable = 0
    for evt in raw_events:
        try:
            d = _parse_event_date(evt["date_str"])
        except ValueError as exc:
            print(f"  warning: {exc}; skipping row", file=sys.stderr)
            skipped_unparseable += 1
            continue
        if d < history_starts:
            skipped_old += 1
            continue
        in_window.append((evt, d))

    unresolved: set[str] = set()
    for evt, _ in in_window:
        for key in ("excluded", "included"):
            if resolve_name_to_ticker(evt[key], current_map) is None:
                unresolved.add(evt[key])
    if unresolved:
        print(
            "\nERROR: cannot resolve the following company names to "
            ".NS tickers (events on/after "
            f"{history_starts}):",
            file=sys.stderr,
        )
        for name in sorted(unresolved):
            print(f"  - {name!r}", file=sys.stderr)
        print(
            "\nAdd them to HISTORICAL_NAME_TO_TICKER in "
            "scripts/bootstrap_membership_history.py and re-run.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    # Pass 2: split each in-window row into two events.
    events: list[dict[str, Any]] = []
    for evt, d in in_window:
        reason = evt["reason"] or None

        # NB: a single Wikipedia row produces TWO events -- one removal
        # and one addition. Both share the same date and (optionally)
        # the same reason string; we copy the reason onto both for
        # symmetry with future indices that might track add reasons too.
        events.append({
            "symbol": resolve_name_to_ticker(evt["excluded"], current_map),
            "action": "removed",
            "date": d.isoformat(),
            "reason": reason,
        })
        events.append({
            "symbol": resolve_name_to_ticker(evt["included"], current_map),
            "action": "added",
            "date": d.isoformat(),
            "reason": reason,
        })

    # Sort DESC by (date, then "added" before "removed" for stable
    # same-day ordering -- pure aesthetic, doesn't affect members_on).
    events.sort(
        key=lambda e: (e["date"], 0 if e["action"] == "added" else 1),
        reverse=True,
    )

    print(
        f"  kept {len(events)} events from {len(raw_events) - skipped_old}"
        f" replacements >= {history_starts}"
        f" (skipped {skipped_old} pre-{history_starts.year}, "
        f"{skipped_unparseable} unparseable)"
    )

    current_members = sorted(current_map.values())
    return {
        INDEX_CODE: {
            "display_name": INDEX_DISPLAY_NAME,
            "source_url": WIKIPEDIA_URL,
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "history_starts": history_starts.isoformat(),
            "current_snapshot_date": today.isoformat(),
            "current_members": current_members,
            "events": events,
        }
    }


def write_payload(payload: dict[str, Any], path: Path) -> None:
    """Write JSON with deterministic formatting (2-space indent, sorted)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path}")


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────
def bootstrap(history_starts: date) -> int:
    try:
        html = fetch_wikipedia_html()
        current_map = parse_current_constituents(html)
        raw_events = parse_replacement_events(html)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    payload = build_payload(current_map, raw_events, history_starts)
    write_payload(payload, OUTPUT_PATH)

    # Quick summary so the operator can eyeball it before commit.
    nifty = payload[INDEX_CODE]
    print(
        f"\nSummary for {INDEX_CODE}:\n"
        f"  history_starts: {nifty['history_starts']}\n"
        f"  snapshot_date:  {nifty['current_snapshot_date']}\n"
        f"  current_members: {len(nifty['current_members'])}\n"
        f"  events: {len(nifty['events'])}\n"
        f"  oldest event: "
        f"{nifty['events'][-1]['date'] if nifty['events'] else '(none)'}\n"
        f"  newest event: "
        f"{nifty['events'][0]['date'] if nifty['events'] else '(none)'}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--history-starts",
        type=lambda s: date.fromisoformat(s),
        default=DEFAULT_HISTORY_STARTS,
        help=(
            "Earliest event date to include (YYYY-MM-DD). "
            f"Default {DEFAULT_HISTORY_STARTS}."
        ),
    )
    args = parser.parse_args()
    return bootstrap(args.history_starts)


if __name__ == "__main__":
    raise SystemExit(main())
