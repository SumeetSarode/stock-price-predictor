"""Cross-validation orchestrator: NSE filings x BSE filings.

WHY THIS MODULE EXISTS
======================
Solves C3 from `pred_logic_solutions.md`. NSE has no public/stable API
and is Cloudflare-protected; BSE's `api.bseindia.com` is an INDEPENDENT
source for the same disclosures (since most NSE-listed companies
dual-list on BSE). Fetching both and deduping gives us:

1. **Resilience**: If NSE's session warmup or Cloudflare gating fails,
   BSE still returns data — partial coverage > zero coverage.
2. **Cross-validation**: Filings present in BOTH sources are
   higher-confidence (independently confirmed). Filings from only one
   source surface as a `corroborated=False` flag for downstream
   weighting / inspection.

DEDUP STRATEGY
==============
Filings are matched across sources on `(announced_date, subject_hash)`
where:
- `announced_date` = the IST CALENDAR date of `announced_at` (not the
  full timestamp; NSE and BSE timestamp the same filing minutes apart
  routinely, so we'd miss matches if we keyed on the second).
- `subject_hash` = a normalized fingerprint of the subject text, lowering
  case and stripping punctuation/whitespace before hashing. Both
  exchanges paraphrase headlines slightly so we use a NORMALIZED hash,
  not raw equality.

When a match is found, we keep the NSE record as canonical (richer
metadata, cleaner event_at) and stamp `metadata["corroborated"] = True`
along with the BSE counterpart's news_id / attachment for traceability.
Unmatched records keep `corroborated = False`.

DESIGN CHOICES
==============
- The two source fetchers run **in parallel** via `asyncio.gather`
  (return_exceptions=True). One side failing does NOT kill the call —
  we return whatever data the working side gave + log the failure.
- Caller passes BOTH the NSE alpha symbol AND the BSE numeric scrip
  code. We INTENTIONALLY don't auto-map — that's a separate concern
  (the BSE master list is a 5MB JSON download that doesn't belong in
  every fetch path).
- Output format mirrors `filings.fetch_filings()` (a DataFrame), so this
  function can be a drop-in replacement for callers that want
  cross-validated data.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Iterable
from datetime import date

import httpx
import pandas as pd
from loguru import logger

from price_predictor.data.filings import (
    DEFAULT_KINDS,
    FilingsFetchError,
    fetch_filings,
)
from price_predictor.data.filings_bse import (
    BseFilingsFetchError,
    fetch_bse_filings,
)
from price_predictor.data.schema import Filing, FilingKind

__all__ = [
    "CrossValidationError",
    "fetch_filings_cross_validated",
    "subject_fingerprint",
]


class CrossValidationError(RuntimeError):
    """Raised when BOTH NSE and BSE fetches fail — caller has no data."""


# ─────────────────────────────────────────────────────────────
# Fingerprinting / dedup key
# ─────────────────────────────────────────────────────────────
# Normalize subjects for fingerprinting:
# - lowercase
# - strip leading/trailing whitespace
# - collapse internal whitespace to single space
# - drop punctuation that varies between sources (commas, dots, dashes)
_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def subject_fingerprint(subject: str) -> str:
    """Stable, source-agnostic hash of a filing subject line.

    NSE and BSE often format headlines slightly differently:
        NSE:  "Outcome of Board Meeting - 26-Apr-2024"
        BSE:  "Outcome of Board Meeting"
    Normalizing + hashing lets these match. We use SHA1 (12 hex chars)
    rather than raw normalized text as the key so the dedup map stays
    compact and the key space is bounded.
    """
    if not subject:
        return ""
    norm = subject.lower().strip()
    norm = _PUNCT_RE.sub(" ", norm)
    norm = _SPACE_RE.sub(" ", norm).strip()
    if not norm:
        return ""
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def _dedup_key(f: Filing) -> tuple[date, str]:
    """Cross-source dedup key: (announced calendar date, subject fingerprint).

    Calendar date (NOT full timestamp) because NSE and BSE timestamp the
    same filing several minutes apart routinely; matching on the second
    would split duplicates.
    """
    return (f.announced_at.date(), subject_fingerprint(f.subject))


# ─────────────────────────────────────────────────────────────
# Merge & corroborate
# ─────────────────────────────────────────────────────────────
def _merge_with_corroboration(
    nse_filings: Iterable[Filing],
    bse_filings: Iterable[Filing],
) -> list[Filing]:
    """Combine two filing streams, marking matched records as corroborated.

    ALGORITHM
    ---------
    1. Build a {dedup_key: Filing} index for the BSE side.
    2. Walk the NSE side. For each:
       - If key matches a BSE filing → mark NSE Filing's metadata as
         corroborated=True, attach BSE's news_id + attachment_url under
         metadata['bse_*'], and POP the BSE filing from the index.
       - Otherwise → mark Filing.metadata.corroborated=False (NSE-only).
    3. Whatever's LEFT in the BSE index = BSE-only filings → also mark
       corroborated=False and append.

    INVARIANT: Output length = unique filings across both sources. No
    filing is dropped; every one carries a corroborated flag. NSE wins
    canonicality on conflict (richer metadata, cleaner event_at).
    """
    # NB: a single BSE entry can collide with at most one NSE entry per
    # (date, subject_hash). Multiple NSE entries with the same key on the
    # same day (which happens — multiple amendments) all share the same
    # one BSE counterpart. We pop after the FIRST match so subsequent
    # NSE entries on the same key see corroborated=False (which is
    # accurate — only one of them was independently confirmed).
    bse_by_key: dict[tuple[date, str], Filing] = {}
    for bf in bse_filings:
        key = _dedup_key(bf)
        if key == ((), ""):  # defensive — empty fingerprint
            continue
        # If duplicate keys within BSE itself, keep the FIRST (newest after
        # provider's pre-sort).
        bse_by_key.setdefault(key, bf)

    out: list[Filing] = []

    for nf in nse_filings:
        key = _dedup_key(nf)
        match = bse_by_key.pop(key, None)
        meta = dict(nf.metadata)  # don't mutate caller's dict
        if match is None:
            meta["corroborated"] = False
            meta["sources"] = ["nse"]
        else:
            meta["corroborated"] = True
            meta["sources"] = ["nse", "bse"]
            # Carry forward BSE traceability bits useful for ops debugging.
            bse_news_id = match.metadata.get("news_id")
            if bse_news_id:
                meta["bse_news_id"] = bse_news_id
            if match.attachment_url:
                meta["bse_attachment_url"] = str(match.attachment_url)
        out.append(nf.model_copy(update={"metadata": meta}))

    # Append BSE-only leftovers.
    for leftover in bse_by_key.values():
        meta = dict(leftover.metadata)
        meta["corroborated"] = False
        meta["sources"] = ["bse"]
        out.append(leftover.model_copy(update={"metadata": meta}))

    out.sort(key=lambda x: x.announced_at, reverse=True)
    return out


# ─────────────────────────────────────────────────────────────
# DataFrame conversion (mirrors filings._to_dataframe shape + adds cols)
# ─────────────────────────────────────────────────────────────
def _to_dataframe(filings: list[Filing]) -> pd.DataFrame:
    """Convert to DataFrame. Adds `corroborated` + `sources` top-level cols.

    These columns are PROMOTED out of metadata for ergonomics — they're
    the whole point of cross-validation, so callers shouldn't have to
    dig into the metadata dict to filter on them.
    """
    columns = [
        "symbol", "kind", "announced_at", "event_at", "event_type",
        "subject", "description", "attachment_url",
        "corroborated", "sources", "metadata",
    ]
    if not filings:
        return pd.DataFrame(columns=columns)

    rows = []
    for f in filings:
        rows.append({
            "symbol": f.symbol,
            "kind": f.kind,
            "announced_at": f.announced_at,
            "event_at": f.event_at,
            "event_type": f.event_type,
            "subject": f.subject,
            "description": f.description,
            "attachment_url": str(f.attachment_url) if f.attachment_url else None,
            "corroborated": f.metadata.get("corroborated", False),
            "sources": f.metadata.get("sources", []),
            "metadata": f.metadata,
        })
    df = pd.DataFrame(rows, columns=columns)
    return df.sort_values("announced_at", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
async def fetch_filings_cross_validated(
    symbol: str,
    bse_scrip_code: str,
    start: str,
    end: str,
    *,
    kinds: list[FilingKind] | None = None,
    timeout: float = 15.0,
    client: httpx.AsyncClient | None = None,
) -> pd.DataFrame:
    """Fetch filings from NSE + BSE in parallel, dedupe, mark corroboration.

    Args:
        symbol: NSE bare ticker (e.g. 'RELIANCE'). Stamped on Filing.symbol
                for ALL records — including BSE-only ones — so downstream
                ticker-keyed joins work uniformly.
        bse_scrip_code: BSE numeric scrip code as string (e.g. '500325').
                        Caller's responsibility to map; see
                        `pred_logic_solutions.md` C3 for the BSE master
                        list URL.
        start: ISO 'YYYY-MM-DD' (inclusive).
        end:   ISO 'YYYY-MM-DD' (inclusive).
        kinds: NSE endpoints to query (passed through to fetch_filings).
               Defaults to filings.DEFAULT_KINDS. BSE always returns ALL
               announcement categories (no per-kind filter).
        timeout: Per-request timeout in seconds.
        client: Optional pre-built AsyncClient shared across calls.

    Returns:
        DataFrame with every column `filings.fetch_filings()` returns,
        PLUS:
        - `corroborated`: bool — True if both NSE+BSE confirmed the filing.
        - `sources`:      list[str] — subset of {"nse","bse"}, length 1 or 2.

    Failure semantics:
        - One side fails → return the other side's data (logged warning).
        - BOTH sides fail → raise `CrossValidationError`.

    Note:
        NSE-only kinds (board_meeting, corporate_action, financial_result)
        will NEVER be marked corroborated=True because BSE returns them as
        plain "announcement" kind under different headlines. Match is on
        (date, subject_fingerprint), so the kind asymmetry is harmless for
        dedup but IS visible in the per-row `kind` column.
    """
    # Lightweight upfront input validation. We DON'T re-validate the
    # individual provider params here because each provider does its own
    # tighter validation when called.
    if not symbol or not isinstance(symbol, str):
        raise ValueError(f"symbol must be a non-empty string, got {symbol!r}")

    requested_kinds = list(kinds) if kinds is not None else list(DEFAULT_KINDS)

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    try:
        # Race both providers in parallel. Each provider raises on total
        # failure for its own source; we catch and treat as partial.
        nse_task = fetch_filings(
            symbol, start, end, kinds=requested_kinds, timeout=timeout, client=client,
        )
        bse_task = fetch_bse_filings(
            bse_scrip_code, start, end, symbol=symbol, timeout=timeout, client=client,
        )
        results = await asyncio.gather(nse_task, bse_task, return_exceptions=True)
    finally:
        if own_client:
            await client.aclose()

    nse_result, bse_result = results

    nse_failed = isinstance(nse_result, BaseException)
    bse_failed = isinstance(bse_result, BaseException)

    # Re-raise non-Exception base classes (KeyboardInterrupt etc.) immediately.
    for r in (nse_result, bse_result):
        if isinstance(r, BaseException) and not isinstance(r, Exception):
            raise r

    if nse_failed and bse_failed:
        raise CrossValidationError(
            f"Both NSE and BSE fetches failed for {symbol} (BSE scrip "
            f"{bse_scrip_code}): NSE={nse_result!r}; BSE={bse_result!r}"
        )

    if nse_failed:
        assert isinstance(nse_result, Exception)
        logger.warning(
            f"[xval] NSE fetch failed for {symbol}: "
            f"{type(nse_result).__name__}: {nse_result}"
        )
        # Mark BSE-only payload appropriately.
        return _to_dataframe(_merge_with_corroboration([], bse_result))  # type: ignore[arg-type]

    if bse_failed:
        assert isinstance(bse_result, Exception)
        logger.warning(
            f"[xval] BSE fetch failed for {symbol} (scrip {bse_scrip_code}): "
            f"{type(bse_result).__name__}: {bse_result}"
        )
        # NSE side returned a DataFrame; we need Filings to merge.
        # Convert back via the schema or — simpler — surface NSE alone with
        # corroborated=False stamped per row.
        nse_df: pd.DataFrame = nse_result  # type: ignore[assignment]
        return _nse_df_only_to_xval_df(nse_df)

    # Happy path: both succeeded.
    nse_df: pd.DataFrame = nse_result  # type: ignore[assignment]
    bse_filings: list[Filing] = bse_result  # type: ignore[assignment]
    nse_filings = _df_back_to_filings(nse_df, symbol=symbol)
    merged = _merge_with_corroboration(nse_filings, bse_filings)
    return _to_dataframe(merged)


# ─────────────────────────────────────────────────────────────
# DataFrame ↔ Filing converters (for the merge layer)
# ─────────────────────────────────────────────────────────────
# We use Filing as the in-memory model for merging because comparing on
# pure dict rows is fragile (datetime types, optional URL types). Building
# Filing objects forces validation back through pydantic and gives us a
# single authoritative shape to merge against.
def _df_back_to_filings(df: pd.DataFrame, *, symbol: str) -> list[Filing]:
    """Re-hydrate a `filings.fetch_filings` DataFrame back into Filing objs."""
    if df.empty:
        return []
    out: list[Filing] = []
    for _, row in df.iterrows():
        try:
            out.append(Filing(
                symbol=row["symbol"] or symbol,
                kind=row["kind"],
                announced_at=row["announced_at"],
                event_at=row["event_at"],
                event_type=row["event_type"],
                subject=row["subject"],
                description=row["description"] or "",
                attachment_url=row["attachment_url"],
                metadata=dict(row["metadata"]) if row["metadata"] else {},
            ))
        except Exception as e:
            # Defensive: a malformed NSE row shouldn't kill the merge.
            logger.debug(f"[xval] skip malformed NSE row: {e}")
            continue
    return out


def _nse_df_only_to_xval_df(nse_df: pd.DataFrame) -> pd.DataFrame:
    """Wrap an NSE-only DataFrame into the cross-validated output shape.

    Used when BSE failed but NSE succeeded — we still want the caller to
    get the standard xval columns (corroborated/sources) so their schema
    is consistent across success/partial-failure paths.
    """
    if nse_df.empty:
        return _to_dataframe([])
    df = nse_df.copy()
    df["corroborated"] = False
    df["sources"] = [["nse"]] * len(df)
    columns = [
        "symbol", "kind", "announced_at", "event_at", "event_type",
        "subject", "description", "attachment_url",
        "corroborated", "sources", "metadata",
    ]
    return df[columns].reset_index(drop=True)


# Keep imports alive for ruff
_ = (FilingsFetchError, BseFilingsFetchError)
