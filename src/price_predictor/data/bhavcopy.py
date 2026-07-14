"""NSE bhavcopy bulk fetcher — one CSV per trading day, all symbols.

WHY THIS EXISTS
===============
Solves the C1 secondary tier (NSE bhavcopy). The bhavcopy is
the exchange-of-record for a given trading day's EOD prices: every NSE
EQ symbol with OHLC + volume in one CSV. For backtests + bulk universe
fetches it's faster than per-symbol APIs (one HTTP call per day instead
of N calls per symbol per day).

DESIGN — ONE BULK FUNCTION + ONE PROVIDER ADAPTER (Option B)
=============================================================
This module exposes a SINGLE public function: `fetch_nse_bhavcopy(d)`
returning a normalized DataFrame for that day. It does NOT know about
the PriceProvider interface or symbol filtering — separation of
responsibilities. The thin provider wrapper lives next to its peers in
`data/providers/bhavcopy_provider.py`.

NSE BHAVCOPY FORMAT CHANGEOVER (2024-07-08)
============================================
NSE switched bhavcopy formats on 2024-07-08:
  - Pre-2024-07-08:  legacy `sec_bhavdata_full_DDMMYYYY.csv` at a stable
                     archives URL.
  - From 2024-07-08: new "UDiff" CSV (`BhavCopy_NSE_CM_0_0_0_<DDMMYYYY>_F_0000.csv`)
                     fetched via the daily-reports JSON API.

We route automatically by date and normalize both formats to the same
DataFrame shape so callers don't need to care.

SOURCE OF TRUTH
===============
- jugaad-data archives (legacy URL pattern):
  https://github.com/jugaad-py/jugaad-data/blob/master/jugaad_data/nse/archives.py
- NSE UDiff circular CMTR59722:
  https://nsearchives.nseindia.com/content/circulars/CMTR59722.pdf
- Daily reports API endpoint key="CM" returns CM-segment files including
  the new bhavcopy.

NETWORKING NOTES
================
The bhavcopy archives subdomain (`nsearchives.nseindia.com`) does NOT
require cookie warmup — unlike the main `www.nseindia.com` API. We
just need a real-browser User-Agent + Accept-Language header (NSE 403s
on python-requests/httpx defaults). The UDiff path goes through the
main API which DOES need cookies; we reuse `_warmup_session` from the
filings module rather than reimplement.

TESTING
=======
The HTTP layer is fully injectable via the `client` keyword. Tests pass
a `respx.mock`-bound httpx client and never touch real NSE.
"""
from __future__ import annotations

import io
from datetime import date

import httpx
import pandas as pd
from loguru import logger

# Reuse the filings module's CA-bundle handling; no point duplicating.
from price_predictor.data.providers._http import get_verify_setting


class BhavcopyError(Exception):
    """Raised when a bhavcopy fetch or parse fails for any reason."""


# ─────────────────────────────────────────────────────────────
# URLs + constants
# ─────────────────────────────────────────────────────────────
# NSE switched the bhavcopy CSV layout on 2024-07-08. Dates strictly BEFORE
# this use the legacy URL; dates ON or AFTER use the UDiff API path.
LEGACY_CUTOVER_DATE = date(2024, 7, 8)

# Legacy full bhavdata archive — stable, no auth/cookies required.
LEGACY_URL_TEMPLATE = (
    "https://nsearchives.nseindia.com/products/content/"
    "sec_bhavdata_full_{dd}{mm}{yyyy}.csv"
)

# UDiff path: NSE's daily-reports JSON gives the actual CSV URL for the day.
DAILY_REPORTS_URL = "https://www.nseindia.com/api/daily-reports?key=CM"

# UDiff endpoints live under www.nseindia.com which BLOCKS /api/* requests
# without session cookies. Hitting the homepage first sets them. Same
# pattern as data/filings.py's `_warm_session` (we keep them separate
# rather than share because filings.py is async and bhavcopy is sync,
# and merging them would force one side into the other's I/O model).
NSE_HOMEPAGE_URL = "https://www.nseindia.com/"

# The label NSE uses inside the daily-reports JSON for the new bhavcopy.
# Verified against multiple production responses: items have a "name" field
# whose value contains this substring.
UDIFF_REPORT_NAME_FRAGMENT = "BhavCopy"

# NSE archives 403 on default httpx User-Agent. A real-browser UA + a
# matching Accept-Language gets us through. These are NOT secrets.
NSE_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/144.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en-US;q=0.9",
    "Accept": "text/csv, application/json, */*",
}

# Final canonical column shape. Both legacy and UDiff parsers normalize
# to this so callers don't see the format split.
_CANONICAL_COLS = ["SYMBOL", "SERIES", "DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────
def fetch_nse_bhavcopy(
    d: date,
    *,
    client: httpx.Client | None = None,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch the NSE bhavcopy for a single trading day.

    Args:
        d:       The trading date.
        client:  Optional httpx.Client (tests inject a respx-bound one).
                 Production callers leave it None and we build + close
                 one per call.
        timeout: Seconds for the underlying GET; default 30s.

    Returns:
        DataFrame with columns:
            SYMBOL (str), SERIES (str), DATE (datetime64[ns, Asia/Kolkata]),
            OPEN (float), HIGH (float), LOW (float), CLOSE (float),
            VOLUME (int).

    Raises:
        ValueError: `d` is not a `date`, or in the future.
        BhavcopyError: NSE returned a non-200, response was empty, or
            the CSV could not be parsed into the canonical shape.
    """
    if not isinstance(d, date):
        raise ValueError(f"d must be a date, got {type(d).__name__}")
    today = date.today()
    if d > today:
        raise ValueError(f"d ({d}) is in the future (today is {today})")

    owns_client = client is None
    if owns_client:
        client = httpx.Client(
            timeout=timeout,
            verify=get_verify_setting(),
            headers=NSE_BROWSER_HEADERS,
        )
    try:
        if d < LEGACY_CUTOVER_DATE:
            df = _fetch_legacy_bhavcopy(d, client)
        else:
            df = _fetch_udiff_bhavcopy(d, client)
    finally:
        if owns_client:
            client.close()

    logger.debug(f"bhavcopy fetched date={d} rows={len(df)}")
    return df


# ─────────────────────────────────────────────────────────────
# Legacy format (pre-2024-07-08)
# ─────────────────────────────────────────────────────────────
def _fetch_legacy_bhavcopy(d: date, client: httpx.Client) -> pd.DataFrame:
    """Legacy `sec_bhavdata_full_DDMMYYYY.csv` path."""
    url = LEGACY_URL_TEMPLATE.format(
        dd=f"{d.day:02d}", mm=f"{d.month:02d}", yyyy=d.year,
    )
    text = _http_get_text(url, client)
    return _parse_legacy_csv(text, trading_date=d)


def _parse_legacy_csv(text: str, *, trading_date: date) -> pd.DataFrame:
    """Parse the legacy bhavcopy CSV body into the canonical shape.

    The legacy format ships with leading-space headers (' SERIES', ' OPEN_PRICE'
    etc.) — pandas reads those literally so we strip whitespace from headers
    before mapping. The DATE column in the file is a string ('25-APR-2024');
    we override it with the trading_date arg for two reasons:
      (1) some files use 'DD-MMM-YYYY', some use 'DD/MM/YYYY' depending on
          the era — the routing arg is unambiguous.
      (2) callers already know which day they asked for; trusting the
          file echo would let a server-side mistake silently mislabel rows.
    """
    if not text or not text.strip():
        raise BhavcopyError(
            f"Legacy bhavcopy for {trading_date} returned empty body"
        )
    try:
        raw = pd.read_csv(io.StringIO(text))
    except Exception as e:
        raise BhavcopyError(
            f"Legacy bhavcopy for {trading_date}: CSV parse failed: {e}"
        ) from e

    # Strip whitespace from headers — NSE pads them with leading spaces.
    raw.columns = [c.strip() for c in raw.columns]

    legacy_to_canon = {
        "SYMBOL": "SYMBOL",
        "SERIES": "SERIES",
        "OPEN_PRICE": "OPEN",
        "HIGH_PRICE": "HIGH",
        "LOW_PRICE": "LOW",
        "CLOSE_PRICE": "CLOSE",
        "TTL_TRD_QNTY": "VOLUME",
    }
    missing = set(legacy_to_canon) - set(raw.columns)
    if missing:
        raise BhavcopyError(
            f"Legacy bhavcopy for {trading_date} missing columns "
            f"{sorted(missing)}; got {sorted(raw.columns)}"
        )

    df = raw.rename(columns=legacy_to_canon)[list(legacy_to_canon.values())].copy()
    # SERIES values also pad with whitespace in the legacy file.
    df["SERIES"] = df["SERIES"].astype(str).str.strip()
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
    df["DATE"] = pd.Timestamp(trading_date, tz="Asia/Kolkata")
    return _coerce_numeric_and_finalize(df, trading_date)


# ─────────────────────────────────────────────────────────────
# UDiff format (2024-07-08 onwards)
# ─────────────────────────────────────────────────────────────
def _fetch_udiff_bhavcopy(d: date, client: httpx.Client) -> pd.DataFrame:
    """UDiff path — pull the per-day CSV URL from daily-reports JSON,
    then fetch and parse the CSV.

    The www.nseindia.com /api/* surface 403s on unwarmed sessions, so we
    visit the homepage first to populate cookies. The same `client` is
    reused for every subsequent call so the cookies persist across the
    JSON + CSV requests.
    """
    _warm_nse_session(client, trading_date=d)
    listing = _http_get_json(DAILY_REPORTS_URL, client)
    csv_url = _find_udiff_csv_url(listing, trading_date=d)
    text = _http_get_text(csv_url, client)
    return _parse_udiff_csv(text, trading_date=d)


def _warm_nse_session(client: httpx.Client, *, trading_date: date) -> None:
    """Hit the NSE homepage to populate session cookies.

    Best-effort: NSE's homepage occasionally 403s plain HTTP clients even
    when the API endpoints themselves are reachable. We log a warning on
    non-2xx but don't raise — the response usually still includes
    Set-Cookie headers, and a downstream API failure will surface its
    own clear error. Only raises on a true network error (DNS, connect
    refused, timeout).

    The `trading_date` arg is passed in only so the warning includes
    the context the caller cared about — makes log-tracing trivial.
    """
    try:
        resp = client.get(NSE_HOMEPAGE_URL)
    except httpx.HTTPError as e:
        raise BhavcopyError(
            f"NSE session warmup for {trading_date} failed: "
            f"{type(e).__name__}: {e}"
        ) from e
    if resp.status_code >= 400:
        logger.warning(
            f"[bhavcopy] NSE homepage warmup for {trading_date} returned "
            f"HTTP {resp.status_code}; continuing (cookies may still be "
            "set; UDiff endpoints sometimes work without a warmed session)"
        )


def _find_udiff_csv_url(listing: object, *, trading_date: date) -> str:
    """Walk the daily-reports payload for the bhavcopy CSV URL.

    The endpoint returns a JSON list of {name, link, ...} dicts. We
    look for an entry whose `name` includes UDIFF_REPORT_NAME_FRAGMENT
    AND whose `link` ends with `.csv`. Defensive — if NSE shuffles the
    schema, we surface a clear error instead of an IndexError.
    """
    if not isinstance(listing, list):
        raise BhavcopyError(
            f"UDiff daily-reports for {trading_date}: expected a JSON list, "
            f"got {type(listing).__name__}"
        )
    for item in listing:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        link = str(item.get("link") or item.get("filePath") or "")
        if UDIFF_REPORT_NAME_FRAGMENT in name and link.lower().endswith(".csv"):
            return link
    raise BhavcopyError(
        f"UDiff daily-reports for {trading_date}: no CSV entry matching "
        f"{UDIFF_REPORT_NAME_FRAGMENT!r} found in {len(listing)} items"
    )


def _parse_udiff_csv(text: str, *, trading_date: date) -> pd.DataFrame:
    """Parse the UDiff bhavcopy into the canonical shape.

    UDiff file mixes EQ + derivatives (futures/options) in one CSV. We
    keep ONLY rows where Sgmt == 'CM' AND FinInstrmTp ∈ {'EQ','BE','BL','ST'}
    (i.e. the cash-market equity instruments). This drops futures/options
    cleanly so callers get the same shape as the legacy file.
    """
    if not text or not text.strip():
        raise BhavcopyError(
            f"UDiff bhavcopy for {trading_date} returned empty body"
        )
    try:
        raw = pd.read_csv(io.StringIO(text))
    except Exception as e:
        raise BhavcopyError(
            f"UDiff bhavcopy for {trading_date}: CSV parse failed: {e}"
        ) from e

    raw.columns = [c.strip() for c in raw.columns]

    udiff_to_canon = {
        "TckrSymb": "SYMBOL",
        "SctySrs": "SERIES",
        "OpnPric": "OPEN",
        "HghPric": "HIGH",
        "LwPric": "LOW",
        "ClsPric": "CLOSE",
        "TtlTradgVol": "VOLUME",
    }
    missing = set(udiff_to_canon) - set(raw.columns)
    if missing:
        raise BhavcopyError(
            f"UDiff bhavcopy for {trading_date} missing columns "
            f"{sorted(missing)}; got {sorted(raw.columns)}"
        )

    # Filter to cash-market equities (drop F&O rows). 'Sgmt' / 'FinInstrmTp'
    # may be absent in older snapshots — be defensive and only filter when
    # the columns exist.
    if "Sgmt" in raw.columns:
        raw = raw[raw["Sgmt"].astype(str).str.strip() == "CM"]
    if "FinInstrmTp" in raw.columns:
        eq_kinds = {"EQ", "BE", "BL", "ST"}
        raw = raw[raw["FinInstrmTp"].astype(str).str.strip().isin(eq_kinds)]

    df = raw.rename(columns=udiff_to_canon)[list(udiff_to_canon.values())].copy()
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
    df["SERIES"] = df["SERIES"].astype(str).str.strip()
    df["DATE"] = pd.Timestamp(trading_date, tz="Asia/Kolkata")
    return _coerce_numeric_and_finalize(df, trading_date)


# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────
def _coerce_numeric_and_finalize(df: pd.DataFrame, trading_date: date) -> pd.DataFrame:
    """Coerce price + volume cols to numeric, drop rows with bad values,
    return canonical column order + reset index."""
    for col in ("OPEN", "HIGH", "LOW", "CLOSE"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["VOLUME"] = pd.to_numeric(df["VOLUME"], errors="coerce")

    # Drop rows where ANY of the price columns failed to coerce (these are
    # usually suspended scrips that ship with '-' in price fields).
    before = len(df)
    df = df.dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE"])
    dropped = before - len(df)
    if dropped:
        logger.debug(
            f"bhavcopy {trading_date}: dropped {dropped} rows with "
            "unparseable price fields (likely suspended scrips)"
        )

    if df.empty:
        raise BhavcopyError(
            f"bhavcopy for {trading_date}: zero rows after cleaning "
            "(all rows had unparseable prices)"
        )

    df["VOLUME"] = df["VOLUME"].fillna(0).astype("int64")
    return df[_CANONICAL_COLS].reset_index(drop=True)


def _http_get_text(url: str, client: httpx.Client) -> str:
    try:
        resp = client.get(url)
    except httpx.HTTPError as e:
        raise BhavcopyError(f"GET {url} failed: {type(e).__name__}: {e}") from e
    if resp.status_code != 200:
        raise BhavcopyError(
            f"GET {url} -> HTTP {resp.status_code} "
            f"(body head: {resp.text[:200]!r})"
        )
    return resp.text


def _http_get_json(url: str, client: httpx.Client) -> object:
    try:
        resp = client.get(url)
    except httpx.HTTPError as e:
        raise BhavcopyError(f"GET {url} failed: {type(e).__name__}: {e}") from e
    if resp.status_code != 200:
        raise BhavcopyError(
            f"GET {url} -> HTTP {resp.status_code} "
            f"(body head: {resp.text[:200]!r})"
        )
    try:
        return resp.json()
    except ValueError as e:
        raise BhavcopyError(
            f"GET {url}: response was not valid JSON: {e}"
        ) from e
