"""BSE corporate filings fetcher (cross-validator for NSE filings).

WHY THIS MODULE EXISTS
======================
Solves C3: NSE has no public/stable API
and is Cloudflare-protected. BSE's `api.bseindia.com` is an INDEPENDENT
source for the same disclosures (since most NSE-listed companies
dual-list on BSE). We can fetch from both, dedupe, and cross-validate
that filings actually exist if both confirm.

KEY DIFFERENCE FROM NSE
=======================
- BSE's API needs **no cookie warmup** — it's a plain JSON endpoint that
  responds to a normal browser-like request. No session priming, no
  Cloudflare dance.
- BSE keys filings by NUMERIC scrip code (e.g. "500325" for Reliance),
  NOT by alpha symbol. Caller must supply the scrip code.
- Date format is `YYYYMMDD` (no dashes), differs from NSE's `dd-MMM-yyyy`.
- Field names use BSE's CamelCase / abbreviated style — different normaliser
  than NSE.

DESIGN CHOICES
==============
- Async via httpx.AsyncClient (consistent with `filings.py` and `news.py`)
- Returns a list of `Filing` objects normalized to the same schema as the
  NSE provider, so the cross-validator (`filings_xval.py`) can dedupe
  across both sources without source-aware branching.
- Source provenance is stamped into `metadata["source"] = "bse"` on every
  Filing — downstream can filter / weight by source if needed.
- `kind` is hard-coded to `"announcement"` because BSE's `AnnGetData`
  endpoint returns ALL announcement types in one stream, including what
  NSE splits into board_meeting / corporate_action / financial_result.
  The cross-validator uses subject-text fingerprinting (not `kind`) for
  dedup so this asymmetry doesn't cause spurious mismatches.
- We DELIBERATELY don't try to download + cache the full BSE scrip-code
  master list here. That belongs in a separate utility (or the caller).
  Single responsibility: this module does ONE thing — fetch + normalize.

KNOWN LIMITS
============
- BSE's `Categoryname` field is sometimes empty for filings where NSE
  would set a clear `event_type`. We surface whatever BSE provides; the
  cross-validator treats `event_type` as best-effort metadata.
- BSE doesn't ship a structured `event_at` (split / ex-div date). Some
  of that lives in the announcement TEXT body. We extract `BroadCastDate`
  as our best `event_at` proxy and acknowledge the gap in the docstring.
- Some networks may block `api.bseindia.com`. Unit tests fully mock httpx
  via respx so they pass on any network; integration tests will surface
  real connectivity status when run.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import httpx
from loguru import logger

from price_predictor.data.schema import Filing

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
BSE_API = "https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w"
IST = timezone(timedelta(hours=5, minutes=30))

# BSE accepts plain User-Agents in practice but it's polite + future-proof
# to send the same browser identity as NSE.
_BSE_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Origin": "https://www.bseindia.com",
    "Referer": "https://www.bseindia.com/",
    "Connection": "keep-alive",
}


class BseFilingsFetchError(RuntimeError):
    """Raised when BSE fetch fails (network/HTTP/JSON parsing)."""


# ─────────────────────────────────────────────────────────────
# Date helpers — BSE uses YYYYMMDD on input, ISO-ish on output
# ─────────────────────────────────────────────────────────────
def _to_bse_date_param(iso_date: str) -> str:
    """`YYYY-MM-DD` → `YYYYMMDD` (BSE's expected query param format)."""
    return iso_date.replace("-", "")


# BSE returns timestamps like:
#   "2024-04-26T18:30:00"            (no offset, IST implied)
#   "2024-04-26 18:30:00"            (some endpoints)
#   "2024-04-26T18:30:00.000"        (with millis)
#   "26 Apr 2024 18:30:00"           (occasional human format)
_BSE_DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%d %b %Y %H:%M:%S",
    "%d %b %Y",
)


def _parse_bse_datetime(raw: str | None) -> datetime | None:
    """Parse BSE's date strings into tz-aware IST datetimes.

    Returns None on missing / unparseable input — we'd rather drop a
    bad date than poison downstream queries with a fabricated timestamp.
    """
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not raw:
        return None
    for fmt in _BSE_DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=IST)
        except ValueError:
            continue
    logger.debug(f"[BSE] unparseable date: {raw!r}")
    return None


def _safe_url(raw: Any) -> str | None:
    """Return a clean URL string or None.

    Mirrors `_safe_url` in `filings.py` so the BSE provider's output
    matches NSE's None-vs-string contract for downstream consumers.
    """
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not raw or raw.lower() in {"-", "n/a", "na", "none", "null"}:
        return None
    if not raw.startswith(("http://", "https://")):
        return None
    return raw


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────
def _validate_inputs(scrip_code: str, start: str, end: str) -> None:
    """Raise ValueError on bad inputs BEFORE making any network call."""
    if not scrip_code or not isinstance(scrip_code, str):
        raise ValueError(f"scrip_code must be a non-empty string, got {scrip_code!r}")
    # BSE scrip codes are numeric (5-7 digits historically); enforce digits-only
    # rather than range so future expansion doesn't break us.
    if not scrip_code.isdigit():
        raise ValueError(
            f"scrip_code must be numeric digits only, got {scrip_code!r}"
        )

    for label, val in (("start", start), ("end", end)):
        try:
            datetime.strptime(val, "%Y-%m-%d")
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"{label}={val!r} must be ISO 'YYYY-MM-DD'"
            ) from e

    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")


# ─────────────────────────────────────────────────────────────
# Parser — BSE row → unified Filing
# ─────────────────────────────────────────────────────────────
# BSE's AnnGetData rows look roughly like:
#   {
#       "SCRIP_CD": 500325, "NEWSID": "...", "XBRLFLAG": "Y",
#       "NEWS_DT": "2024-04-26T18:30:00", "ATTACHMENTNAME": "abc.pdf",
#       "HEADLINE": "Board Meeting Outcome", "CATEGORYNAME": "Board Meeting",
#       "SUBCATNAME": "Outcome of Board Meeting", "MORE": "...",
#       "DT_TM": "2024-04-26T18:30:00", "NSURL": "...",
#       "SLONGNAME": "Reliance Industries Ltd",
#   }
# Field names we care about: HEADLINE, CATEGORYNAME, NEWS_DT/DT_TM,
# ATTACHMENTNAME, NSURL. Everything else lands in metadata.
_BSE_PDF_BASE = "https://www.bseindia.com/xml-data/corpfiling/AttachLive/"


def _bse_attachment_url(attachment_name: Any) -> str | None:
    """Build a clickable PDF URL from BSE's bare attachment file name.

    BSE's `ATTACHMENTNAME` is just the filename ("abc123.pdf"); the actual
    download URL needs the `/xml-data/corpfiling/AttachLive/` prefix.
    """
    if not attachment_name or not isinstance(attachment_name, str):
        return None
    name = attachment_name.strip()
    if not name:
        return None
    return _safe_url(_BSE_PDF_BASE + name)


def _parse_bse_row(item: dict[str, Any], symbol: str) -> Filing | None:
    """Map a BSE AnnGetData row to a unified `Filing`.

    Returns None if the row is missing the bare-minimum fields we need
    (subject + announced_at). We DROP rather than poison.

    `symbol` is the NSE-style alpha symbol the caller is correlating
    against — we stamp it on the Filing so cross-validation can match
    by symbol regardless of source.
    """
    headline = (item.get("HEADLINE") or item.get("NEWSSUB") or "").strip()
    if not headline:
        return None

    raw_announced = item.get("NEWS_DT") or item.get("DT_TM")
    announced_at = _parse_bse_datetime(raw_announced)
    if announced_at is None:
        return None

    # BSE has no separate "event_at"; BroadCastDate is a near-proxy for
    # when the announcement was actually broadcast (vs. submitted). Most
    # of the time it's identical to NEWS_DT; we still try it for parity
    # with NSE's structured event_at when it differs.
    event_at = _parse_bse_datetime(item.get("BroadCastDate"))

    category = (item.get("CATEGORYNAME") or "").strip() or None
    subcat = (item.get("SUBCATNAME") or "").strip() or None
    description = (item.get("MORE") or "").strip()

    # Scrip code lands in metadata so the cross-validator can verify it
    # actually matched the requested code (defensive against BSE returning
    # neighbour-code rows on a search-style query).
    metadata: dict[str, Any] = {
        "source": "bse",
        "scrip_code": str(item.get("SCRIP_CD") or "").strip() or None,
        "subcategory": subcat,
        "news_id": (str(item.get("NEWSID")).strip()
                    if item.get("NEWSID") not in (None, "") else None),
        "company_name": (item.get("SLONGNAME") or "").strip() or None,
    }
    # Drop None metadata keys to keep the dict tight (matches NSE provider).
    metadata = {k: v for k, v in metadata.items() if v is not None}

    return Filing(
        symbol=symbol,
        kind="announcement",  # see module docstring on this choice
        announced_at=announced_at,
        event_at=event_at,
        event_type=category,
        subject=headline,
        description=description,
        attachment_url=_bse_attachment_url(item.get("ATTACHMENTNAME")),
        metadata=metadata,
    )


# ─────────────────────────────────────────────────────────────
# Fetcher — async, single endpoint
# ─────────────────────────────────────────────────────────────
def _build_url(scrip_code: str, start: str, end: str) -> str:
    """Compose the BSE AnnGetData URL with all required query params."""
    p = {
        "strCat": "-1",                       # all categories
        "strType": "C",                       # Company filings
        "strSearch": "P",                     # Period-based query
        "strscrip": scrip_code,               # NB: lowercase 's' per BSE quirk
        "strPrevDate": _to_bse_date_param(start),
        "strToDate": _to_bse_date_param(end),
    }
    qs = "&".join(f"{k}={v}" for k, v in p.items())
    return f"{BSE_API}?{qs}"


async def fetch_bse_filings(
    scrip_code: str,
    start: str,
    end: str,
    *,
    symbol: str | None = None,
    timeout: float = 15.0,
    client: httpx.AsyncClient | None = None,
) -> list[Filing]:
    """Fetch BSE corporate filings for a scrip code over [start, end].

    Args:
        scrip_code: BSE numeric scrip code as a string (e.g. "500325" for
                    RELIANCE). Must be digits-only.
        start: ISO `YYYY-MM-DD` (inclusive).
        end:   ISO `YYYY-MM-DD` (inclusive).
        symbol: Alpha symbol to stamp on the Filing.symbol field for
                cross-source matching. Defaults to `scrip_code` if not
                provided — but callers SHOULD pass the NSE alpha symbol
                so the cross-validator can dedupe across sources.
        timeout: Per-request timeout in seconds.
        client: Optional pre-built AsyncClient (lets the cross-validator
                share one client across NSE + BSE calls).

    Returns:
        List of `Filing` objects, sorted newest-first by `announced_at`.
        Empty list on no results — NOT an error.

    Raises:
        ValueError: invalid input (no network call made).
        BseFilingsFetchError: HTTP / JSON / parse failure.
    """
    _validate_inputs(scrip_code, start, end)
    sym_for_filing = symbol if symbol else scrip_code
    url = _build_url(scrip_code, start, end)

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    try:
        try:
            resp = await client.get(url, headers=_BSE_HEADERS)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise BseFilingsFetchError(
                f"BSE HTTP error for scrip={scrip_code}: {e}"
            ) from e
        except httpx.RequestError as e:
            raise BseFilingsFetchError(
                f"BSE network error for scrip={scrip_code}: {e}"
            ) from e

        try:
            payload = resp.json()
        except ValueError as e:
            raise BseFilingsFetchError(
                f"BSE returned non-JSON for scrip={scrip_code}: {e}"
            ) from e

        # Successful payloads look like {"Table": [...], "Table1": [{"ROWCNT": N}]}
        # Empty / no-results pages are still 200 with {"Table": []}.
        rows = payload.get("Table") or []
        if not isinstance(rows, list):
            raise BseFilingsFetchError(
                f"BSE Table field is not a list for scrip={scrip_code}"
            )

        filings: list[Filing] = []
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            f = _parse_bse_row(raw, sym_for_filing)
            if f is not None:
                filings.append(f)

        filings.sort(key=lambda x: x.announced_at, reverse=True)
        return filings
    finally:
        if own_client:
            await client.aclose()


# Keep imported names alive for ruff (UTC used implicitly via tz-aware logic)
_ = UTC
