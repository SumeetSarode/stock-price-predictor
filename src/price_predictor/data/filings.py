"""NSE corporate filings fetcher (iteration 3.1.3).

WHAT THIS GIVES US
==================
Unified async access to 4 NSE corporate-events endpoints:
- announcement       — regulatory disclosures (M&A, fundraising, lawsuits, results filings)
- board_meeting      — scheduled board meetings (forward-looking earnings windows)
- corporate_action   — splits, dividends, bonuses, rights (with ex-dates)
- financial_result   — structured quarterly/annual results filings

Each endpoint returns its own quirky JSON shape; we normalize all into the
unified `Filing` model with endpoint-specific extras in `metadata`.

DESIGN
======
- Async-first via httpx.AsyncClient (consistent with news module)
- Cookie warmup: NSE blocks API calls without session cookies; we visit the
  homepage first to warm up before any /api/ request
- Browser-like headers: NSE blocks plain User-Agent strings
- Fan-out: one fetch_filings() call hits N endpoints in parallel via gather
- Partial failure tolerance: if one endpoint fails, others still return; the
  failure is logged but doesn't kill the batch
- Forward-looking ready: `event_at` field captured separately from `announced_at`
  so callers can query "splits effective in next 60 days" not just "filings
  in last 30 days"

KNOWN LIMITS / RISKS
====================
- NSE has NO official API. Endpoints are scraped from their UI's network calls.
  They can change without notice. Integration test verifies real behavior.
- Walmart corp DNS blocks www.nseindia.com (verified). Run integration test
  off-VPN. Unit tests fully mock httpx via respx.
- Endpoint JSON shapes inferred from community libraries (nsepython etc.) +
  NSE web UI. Real responses may differ; integration test will surface this.
- Some endpoints (e.g. board_meeting) may not accept date filters server-side;
  we filter post-fetch where needed.
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import httpx
import pandas as pd
from loguru import logger

from price_predictor.data.schema import Filing, FilingKind

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
NSE_BASE = "https://www.nseindia.com"
NSE_HOMEPAGE = f"{NSE_BASE}/"
IST = timezone(timedelta(hours=5, minutes=30))

# NSE blocks plain Python User-Agents. Browser-like headers are mandatory.
# Note: NO 'br' in Accept-Encoding -- httpx doesn't decode brotli without
# the optional `brotli` package installed. gzip/deflate are auto-handled.
_BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Referer": NSE_BASE + "/",
    "Connection": "keep-alive",
}


class FilingsFetchError(RuntimeError):
    """Raised when NSE fetch fails (network/HTTP/JSON parsing)."""


# ─────────────────────────────────────────────────────────────
# Date helpers — NSE uses "dd-MMM-yyyy" or "dd-MMM-yyyy HH:MM:SS" (IST)
# ─────────────────────────────────────────────────────────────
_NSE_DATE_FORMATS = (
    "%d-%b-%Y %H:%M:%S",   # "26-Apr-2026 18:30:00"
    "%d-%b-%Y",            # "26-Apr-2026"
    "%d-%m-%Y %H:%M:%S",   # "26-04-2026 18:30:00" (some endpoints)
    "%d-%m-%Y",            # "26-04-2026"
)


def _parse_nse_datetime(raw: str | None) -> datetime | None:
    """Parse NSE's date strings into tz-aware IST datetimes. Returns None on failure."""
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    for fmt in _NSE_DATE_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.replace(tzinfo=IST)
        except ValueError:
            continue
    return None


def _to_nse_date_param(iso_date: str) -> str:
    """Convert our 'YYYY-MM-DD' input into NSE's 'dd-MM-yyyy' query format."""
    dt = datetime.strptime(iso_date, "%Y-%m-%d")
    return dt.strftime("%d-%m-%Y")


def _validate_inputs(symbol: str, start: str, end: str) -> None:
    """Validate before any network call. Raises ValueError on bad input."""
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError(f"symbol must be a non-empty string, got {symbol!r}")
    # NSE uses bare symbols (RELIANCE, not RELIANCE.NS)
    if "." in symbol:
        raise ValueError(
            f"symbol must be NSE bare symbol (no .NS suffix), got {symbol!r}"
        )
    for label, val in (("start", start), ("end", end)):
        if not isinstance(val, str):
            raise ValueError(f"{label} must be 'YYYY-MM-DD' string, got {val!r}")
        try:
            datetime.strptime(val, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"{label}={val!r} not a valid 'YYYY-MM-DD' date") from e
    if datetime.strptime(start, "%Y-%m-%d") > datetime.strptime(end, "%Y-%m-%d"):
        raise ValueError(f"start ({start}) must be <= end ({end})")


# ─────────────────────────────────────────────────────────────
# Per-endpoint parsers (raw_json item → Filing)
# ─────────────────────────────────────────────────────────────
def _parse_announcement(item: dict[str, Any], symbol: str) -> Filing | None:
    """Parse one corporate-announcements row.

    Expected fields (NSE shape):
        an_dt, attchmntFile, attchmntText, desc, smIndustry, sort_date
    """
    announced = _parse_nse_datetime(item.get("an_dt") or item.get("sort_date"))
    if announced is None:
        return None  # Drop rows we can't date

    subject = (item.get("desc") or item.get("attchmntText") or "").strip()
    if not subject:
        return None  # Drop rows with no subject

    return Filing(
        symbol=symbol,
        kind="announcement",
        announced_at=announced,
        event_at=None,  # Announcements don't have a separate event date
        event_type=item.get("smIndustry") or None,
        subject=subject[:500],  # Cap to a sane length
        description=(item.get("attchmntText") or "")[:5000],
        attachment_url=item.get("attchmntFile") or None,
        metadata={k: v for k, v in item.items() if k not in {
            "an_dt", "attchmntFile", "attchmntText", "desc", "smIndustry", "sort_date",
        }},
    )


def _parse_board_meeting(item: dict[str, Any], symbol: str) -> Filing | None:
    """Parse one corporate-board-meetings row.

    Expected fields (NSE shape):
        bm_purpose, bm_desc, bm_date, attachment, bm_timestamp
    """
    # Board meetings: 'announced_at' = bm_timestamp (when filed), 'event_at' = bm_date (when meeting happens)
    announced = _parse_nse_datetime(item.get("bm_timestamp"))
    event = _parse_nse_datetime(item.get("bm_date"))

    # If we have neither timestamp, drop the row
    if announced is None and event is None:
        return None
    # If announced missing but event present, use event as both (best effort)
    if announced is None:
        announced = event

    purpose = (item.get("bm_purpose") or "").strip()
    desc = (item.get("bm_desc") or "").strip()
    subject = purpose or desc or "Board Meeting"

    return Filing(
        symbol=symbol,
        kind="board_meeting",
        announced_at=announced,
        event_at=event,
        event_type=purpose or None,
        subject=subject[:500],
        description=desc[:5000],
        attachment_url=item.get("attachment") or None,
        metadata={k: v for k, v in item.items() if k not in {
            "bm_purpose", "bm_desc", "bm_date", "attachment", "bm_timestamp",
        }},
    )


def _parse_corporate_action(item: dict[str, Any], symbol: str) -> Filing | None:
    """Parse one corporates-corporateActions row.

    Expected fields (NSE shape):
        subject, exDate, recDate, bcStartDate, bcEndDate, ndStartDate, ndEndDate
    """
    # Corp actions: 'announced_at' is rarely in the payload; 'exDate' is THE date
    # that matters (when action takes effect). Use exDate as both.
    ex = _parse_nse_datetime(item.get("exDate"))
    if ex is None:
        return None

    subject = (item.get("subject") or "").strip()
    if not subject:
        return None

    # Try to extract event_type from subject (e.g. "Dividend - Rs 9 Per Share" → "Dividend")
    event_type: str | None = None
    if " - " in subject:
        event_type = subject.split(" - ", 1)[0].strip()
    elif " " in subject:
        event_type = subject.split(" ", 1)[0].strip()

    return Filing(
        symbol=symbol,
        kind="corporate_action",
        announced_at=ex,    # Use exDate; NSE doesn't expose announcement date here
        event_at=ex,
        event_type=event_type,
        subject=subject[:500],
        description="",
        attachment_url=None,
        metadata={k: v for k, v in item.items() if k not in {"subject", "exDate"}},
    )


# Capture e.g. "1:5", "5:1" for splits; "Rs 9", "Rs. 4.50" for dividends
_RATIO_RE = re.compile(r"\b(\d+)\s*:\s*(\d+)\b")
_RUPEE_RE = re.compile(r"Rs\.?\s*([\d.]+)", re.IGNORECASE)


def _parse_financial_result(item: dict[str, Any], symbol: str) -> Filing | None:
    """Parse one corporates-financial-results row.

    Expected fields (NSE shape):
        fromDate, toDate, broadCastDate, audited, consolidated, xbrlAttachment,
        ind, params (often a nested dict with revenue/profit/EPS)
    """
    announced = _parse_nse_datetime(item.get("broadCastDate"))
    if announced is None:
        return None

    period_from = item.get("fromDate", "")
    period_to = item.get("toDate", "")
    audited = item.get("audited", "")
    consolidated = item.get("consolidated", "")

    # Subject like "Audited Consolidated Financial Results for 01-Jan-2026 to 31-Mar-2026"
    parts = [p for p in (audited, consolidated, "Financial Results") if p]
    subject = " ".join(parts).strip()
    if period_from and period_to:
        subject = f"{subject} for {period_from} to {period_to}"

    return Filing(
        symbol=symbol,
        kind="financial_result",
        announced_at=announced,
        event_at=_parse_nse_datetime(period_to),  # Period end — useful for joining to estimates
        event_type=audited or "Financial Results",
        subject=subject[:500],
        description="",
        attachment_url=item.get("xbrlAttachment") or None,
        metadata={k: v for k, v in item.items() if k not in {
            "broadCastDate", "fromDate", "toDate", "audited", "consolidated", "xbrlAttachment",
        }},
    )


# ─────────────────────────────────────────────────────────────
# Endpoint registry
# ─────────────────────────────────────────────────────────────
ParserFn = Callable[[dict[str, Any], str], "Filing | None"]


def _build_announcement_url(symbol: str, start: str, end: str) -> str:
    return (
        f"{NSE_BASE}/api/corporate-announcements"
        f"?index=equities&symbol={symbol}"
        f"&from_date={_to_nse_date_param(start)}&to_date={_to_nse_date_param(end)}"
    )


def _build_board_meeting_url(symbol: str, start: str, end: str) -> str:
    # Board-meetings endpoint doesn't accept date filters reliably; we filter post-fetch.
    return f"{NSE_BASE}/api/corporate-board-meetings?index=equities&symbol={symbol}"


def _build_corporate_action_url(symbol: str, start: str, end: str) -> str:
    return (
        f"{NSE_BASE}/api/corporates-corporateActions"
        f"?index=equities&symbol={symbol}"
        f"&from_date={_to_nse_date_param(start)}&to_date={_to_nse_date_param(end)}"
    )


def _build_financial_result_url(symbol: str, start: str, end: str) -> str:
    return (
        f"{NSE_BASE}/api/corporates-financial-results"
        f"?index=equities&symbol={symbol}&period=Quarterly"
        f"&from_date={_to_nse_date_param(start)}&to_date={_to_nse_date_param(end)}"
    )


# (kind, url_builder, parser)
_ENDPOINTS: dict[FilingKind, tuple[Callable[[str, str, str], str], ParserFn]] = {
    "announcement":     (_build_announcement_url,     _parse_announcement),
    "board_meeting":    (_build_board_meeting_url,    _parse_board_meeting),
    "corporate_action": (_build_corporate_action_url, _parse_corporate_action),
    "financial_result": (_build_financial_result_url, _parse_financial_result),
}


# ─────────────────────────────────────────────────────────────
# Cookie warmup
# ─────────────────────────────────────────────────────────────
async def _warm_session(client: httpx.AsyncClient) -> None:
    """Hit NSE homepage to populate session cookies.

    NSE often blocks /api/* without cookies; visiting the homepage usually
    sets them. Idempotent.

    BEST-EFFORT: NSE's homepage occasionally 403s plain HTTP clients even
    when the API endpoints themselves are accessible. We log a warning on
    non-2xx but don't raise -- the response often still includes Set-Cookie
    headers, and the API may work even without warmed cookies.

    Only raises on a true network error (DNS, connection refused, timeout).
    """
    try:
        resp = await client.get(NSE_HOMEPAGE, headers=_BROWSER_HEADERS)
    except (httpx.HTTPError, httpx.InvalidURL) as e:
        raise FilingsFetchError(f"NSE session warmup network error: {e}") from e

    if resp.status_code >= 400:
        logger.warning(
            f"NSE homepage warmup returned HTTP {resp.status_code} "
            f"(continuing -- cookies may still be set; API endpoints often "
            f"work without warmed session)"
        )


# ─────────────────────────────────────────────────────────────
# Per-endpoint fetch
# ─────────────────────────────────────────────────────────────
async def _fetch_one_kind(
    client: httpx.AsyncClient,
    symbol: str,
    start: str,
    end: str,
    kind: FilingKind,
) -> list[Filing]:
    """Fetch one endpoint, parse, and return list[Filing]. Empty on no data.

    Raises FilingsFetchError on HTTP/JSON failure.
    """
    url_builder, parser = _ENDPOINTS[kind]
    url = url_builder(symbol, start, end)

    try:
        resp = await client.get(url, headers=_BROWSER_HEADERS)
    except httpx.HTTPError as e:
        raise FilingsFetchError(f"{kind} HTTP error for {symbol}: {e}") from e

    if resp.status_code >= 400:
        raise FilingsFetchError(
            f"{kind} HTTP {resp.status_code} for {symbol}: {resp.text[:200]}"
        )

    try:
        payload = resp.json()
    except ValueError as e:
        raise FilingsFetchError(
            f"{kind} returned non-JSON for {symbol}: {resp.text[:200]}"
        ) from e

    # NSE responses vary: sometimes a bare list, sometimes wrapped in a key.
    # Be permissive: accept list, or dict with one of the common wrapper keys.
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = (
            payload.get("data")
            or payload.get("rows")
            or payload.get("result")
            or []
        )
    else:
        items = []

    if not isinstance(items, list):
        return []

    filings: list[Filing] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            f = parser(item, symbol)
        except Exception as e:  # parser bugs shouldn't kill the batch
            logger.warning(f"Parser failed for {kind} item: {e}")
            continue
        if f is not None:
            filings.append(f)

    # Apply date filtering in code for endpoints that don't support server-side filter
    # (board_meeting). Filter on event_at if present, else announced_at.
    if kind == "board_meeting":
        s_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=IST)
        e_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=IST) + timedelta(days=1)
        filings = [
            f for f in filings
            if s_dt <= (f.event_at or f.announced_at) < e_dt
        ]

    return filings


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
DEFAULT_KINDS: list[FilingKind] = [
    "announcement",
    "board_meeting",
    "corporate_action",
    "financial_result",
]


async def fetch_filings(
    symbol: str,
    start: str,
    end: str,
    *,
    kinds: list[FilingKind] | None = None,
    timeout: float = 15.0,
    client: httpx.AsyncClient | None = None,
) -> pd.DataFrame:
    """Fetch corporate filings across N NSE endpoints in parallel.

    Args:
        symbol: NSE bare ticker (e.g. 'RELIANCE', NOT 'RELIANCE.NS')
        start: ISO 'YYYY-MM-DD'
        end:   ISO 'YYYY-MM-DD' (inclusive)
        kinds: Which endpoints to hit. Defaults to all 4. Pass a subset to
               opt out (e.g. `['announcement']` for only the broadest catch-all).
        timeout: Per-request timeout in seconds.
        client: Optional pre-built AsyncClient (lets caller share warmed-up
                cookie session across many calls).

    Returns:
        DataFrame with columns matching Filing's fields. Empty (0 rows) on
        no results — NOT an error.

    Raises:
        ValueError: invalid input (no network call made).
        FilingsFetchError: every requested endpoint failed (partial failures
                           are tolerated and logged).
    """
    _validate_inputs(symbol, start, end)
    requested_kinds = list(kinds) if kinds is not None else DEFAULT_KINDS
    if not requested_kinds:
        raise ValueError("kinds must be a non-empty list when provided")

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    try:
        # Always warm up if we built our own client. Caller-supplied clients
        # are assumed to be already warmed (or will warm on first failed call).
        if own_client:
            await _warm_session(client)

        results = await asyncio.gather(
            *(_fetch_one_kind(client, symbol, start, end, k) for k in requested_kinds),
            return_exceptions=True,
        )

        all_filings: list[Filing] = []
        failures: list[tuple[FilingKind, Exception]] = []
        for kind, r in zip(requested_kinds, results, strict=True):
            if isinstance(r, BaseException) and not isinstance(r, Exception):
                raise r  # KeyboardInterrupt etc.
            if isinstance(r, Exception):
                failures.append((kind, r))
                logger.warning(f"{symbol} {kind} fetch failed: {r}")
                continue
            all_filings.extend(r)  # type: ignore[arg-type]

        # If EVERY endpoint failed, raise — caller has no data
        if failures and len(failures) == len(requested_kinds):
            raise FilingsFetchError(
                f"All {len(requested_kinds)} filings endpoints failed for {symbol}: "
                + "; ".join(f"{k}: {e}" for k, e in failures)
            )
    finally:
        if own_client:
            await client.aclose()

    return _to_dataframe(all_filings)


def _to_dataframe(filings: list[Filing]) -> pd.DataFrame:
    """Convert list[Filing] → DataFrame. Empty list → empty DF with right columns."""
    columns = [
        "symbol", "kind", "announced_at", "event_at", "event_type",
        "subject", "description", "attachment_url", "metadata",
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
            "metadata": f.metadata,
        })
    df = pd.DataFrame(rows, columns=columns)
    return df.sort_values("announced_at", ascending=False).reset_index(drop=True)


async def fetch_filings_batch(
    symbols: list[str],
    start: str,
    end: str,
    *,
    kinds: list[FilingKind] | None = None,
    concurrency: int = 3,
    timeout: float = 15.0,
) -> dict[str, pd.DataFrame | Exception]:
    """Fetch filings for many tickers, sharing a warmed-up session.

    Concurrency default is 3 (lower than news's 5) — NSE rate-limits aggressively.
    """
    if not symbols:
        return {}

    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        await _warm_session(client)  # Single warmup, shared across all symbols

        async def _one(sym: str) -> pd.DataFrame:
            async with sem:
                return await fetch_filings(
                    sym, start, end, kinds=kinds, client=client,
                )

        results = await asyncio.gather(
            *(_one(s) for s in symbols),
            return_exceptions=True,
        )

    out: dict[str, pd.DataFrame | Exception] = {}
    for sym, r in zip(symbols, results, strict=True):
        if isinstance(r, BaseException) and not isinstance(r, Exception):
            raise r
        out[sym] = r  # type: ignore[assignment]
    return out


# Keep imported names alive for ruff (used in parsers/runtime)
_ = (UTC, datetime)
