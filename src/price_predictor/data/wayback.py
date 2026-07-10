"""Point-in-time (PIT) article fetcher via the Wayback Machine.

WHY THIS EXISTS (look-ahead-bias hardening)
===========================================
For an honest backtest, the news the model reads at ``as_of`` must be
the news that *actually existed on or before ``as_of``* — never a later
revision or a headline written with hindsight. Fetching the live URL
today leaks the future. The Internet Archive's Wayback Machine lets us
retrieve the page *as it was archived* at a past moment, closing that
leak.

  ``article_body_pit(url, asof)`` -> clean article text as of ``asof``,
  or ``None`` if no snapshot ≤ ``asof`` exists.

THE HARD GUARANTEE
==================
We NEVER return content from a snapshot captured AFTER ``asof``. The
CDX query is bounded server-side (``to=<asof>``) and re-checked
client-side (``_is_pit_safe``). If no qualifying snapshot exists we
return ``None`` — we do NOT silently fall back to the live URL (that
would reintroduce the exact bias we're eliminating). Callers drop the
observation or mark it post-hoc.

WHY httpx + the CDX API (not waybackpy)
=======================================
pred_logic_solutions.md sketched this with ``waybackpy``. We already
depend on ``httpx`` and ``trafilatura``; the Wayback CDX Server API is a
plain HTTP endpoint, so we avoid adding another dependency. The
``&to=<asof>&limit=-1`` query returns the most-recent snapshot at/before
``asof`` directly.

POLITENESS
==========
The Internet Archive asks for a contact User-Agent and gentle pacing.
We send a descriptive UA and enforce a minimum interval between live
requests (module-level, thread-safe). Everything is cached in SQLite
keyed by ``(url, asof_date)`` so a re-run never re-hits Wayback.

CITATIONS
=========
  * Wayback CDX Server API:
    https://github.com/internetarchive/wayback/tree/master/wayback-cdx-server
  * Internet Archive ToS: https://archive.org/about/terms.php
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
import trafilatura
from loguru import logger

from price_predictor.config.settings import settings

USER_AGENT = "price-predictor-research/1.0 (backtest PIT fetcher; +https://github.com/)"
CDX_ENDPOINT = "http://web.archive.org/cdx/search/cdx"
_WAYBACK_TS_FMT = "%Y%m%d%H%M%S"

# Minimum seconds between live Wayback requests (politeness). Overridable
# for tests via set_min_request_interval().
_min_request_interval = 1.0
_last_request_at = 0.0
_rate_lock = threading.Lock()


class WaybackError(Exception):
    """Raised on unrecoverable Wayback interaction failures."""


@dataclass(frozen=True, slots=True)
class Snapshot:
    """A single Wayback capture that is PIT-safe for a given as_of."""

    timestamp: str      # 'YYYYMMDDHHMMSS' (the archived capture time)
    original_url: str   # the URL that was archived

    @property
    def raw_url(self) -> str:
        """The 'id_' raw-content URL (no Wayback toolbar injection)."""
        return f"https://web.archive.org/web/{self.timestamp}id_/{self.original_url}"


# ── Pure helpers (no I/O — trivially testable) ──────────────────────


def _asof_stamp(asof: datetime) -> str:
    """Format an as_of datetime as a Wayback CDX 'to=' timestamp."""
    return asof.strftime(_WAYBACK_TS_FMT)


def _is_pit_safe(snapshot_ts: str, asof: datetime) -> bool:
    """True iff the snapshot was captured on/before as_of.

    String comparison is valid because both are zero-padded
    'YYYYMMDDHHMMSS' — lexical order == chronological order.
    """
    return snapshot_ts <= _asof_stamp(asof)


def _pick_latest_pit_row(
    cdx_rows: list[list[str]], asof: datetime,
) -> Snapshot | None:
    """Parse a CDX JSON response into the newest PIT-safe Snapshot.

    Args:
        cdx_rows: The raw CDX rows. Row 0 is the column header; the rest
            are captures [urlkey, timestamp, original, mimetype,
            statuscode, digest, length].
        asof: The PIT boundary.

    Returns:
        The most-recent capture with timestamp ≤ asof, or None.
    """
    if not cdx_rows or len(cdx_rows) < 2:
        return None

    header = cdx_rows[0]
    try:
        ts_i = header.index("timestamp")
        orig_i = header.index("original")
    except ValueError:
        # Unexpected schema — refuse rather than guess.
        return None

    best: Snapshot | None = None
    for row in cdx_rows[1:]:
        if len(row) <= max(ts_i, orig_i):
            continue
        ts = row[ts_i]
        if not _is_pit_safe(ts, asof):
            continue  # defence-in-depth on top of the server 'to=' bound
        if best is None or ts > best.timestamp:
            best = Snapshot(timestamp=ts, original_url=row[orig_i])
    return best


# ── SQLite cache ────────────────────────────────────────────────────


class WaybackCache:
    """Persists (url, asof_date) -> (snapshot_ts, body).

    A ``body`` of NULL records a *negative* result (no PIT snapshot
    existed) so we don't re-query Wayback for a known miss. ``body`` of
    '' means an empty extraction (also cached — Wayback had it, but
    trafilatura found no article text).
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or (settings.cache_dir / "wayback_pit.sqlite")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS wayback_pit (
                    url         TEXT NOT NULL,
                    asof_date   TEXT NOT NULL,
                    snapshot_ts TEXT,
                    body        TEXT,
                    fetched_at  TEXT NOT NULL,
                    PRIMARY KEY (url, asof_date)
                )
                """
            )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get(self, url: str, asof_date: str) -> tuple[bool, str | None]:
        """Return (hit, body). ``hit`` False means not cached at all."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT body FROM wayback_pit WHERE url = ? AND asof_date = ?",
                (url, asof_date),
            ).fetchone()
        if row is None:
            return False, None
        return True, row["body"]

    def put(
        self, url: str, asof_date: str, snapshot_ts: str | None, body: str | None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO wayback_pit
                    (url, asof_date, snapshot_ts, body, fetched_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (url, asof_date, snapshot_ts, body, datetime.utcnow().isoformat()),
            )


# ── Network seams (mocked in tests) ─────────────────────────────────


def set_min_request_interval(seconds: float) -> None:
    """Override the politeness delay. Tests set this to 0."""
    global _min_request_interval
    _min_request_interval = max(0.0, seconds)


def _throttle() -> None:
    """Block until at least _min_request_interval has passed since the
    last live request. Thread-safe."""
    global _last_request_at
    with _rate_lock:
        now = time.monotonic()
        wait = _min_request_interval - (now - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


def _cdx_query(url: str, asof: datetime, *, timeout: float = 30.0) -> list[list[str]]:
    """Query the Wayback CDX API for captures of ``url`` at/before asof.

    Returns the raw JSON rows (header + captures). Empty list if none.
    """
    _throttle()
    params = {
        "url": url,
        "output": "json",
        "to": _asof_stamp(asof),
        "filter": "statuscode:200",
        "limit": "-1",        # last (=most recent) match within the 'to' bound
        "fastLatest": "true",
    }
    try:
        resp = httpx.get(
            CDX_ENDPOINT, params=params,
            headers={"User-Agent": USER_AGENT}, timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        raise WaybackError(f"CDX query failed for {url}: {exc}") from exc
    return data if isinstance(data, list) else []


def _fetch_raw(snapshot: Snapshot, *, timeout: float = 30.0) -> str | None:
    """Fetch the raw archived HTML for a snapshot. None on failure."""
    _throttle()
    try:
        resp = httpx.get(
            snapshot.raw_url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout, follow_redirects=True,
        )
        resp.raise_for_status()
        return resp.text
    except httpx.HTTPError as exc:
        logger.warning("[wayback] raw fetch failed for {}: {}", snapshot.raw_url, exc)
        return None


# ── Public API ──────────────────────────────────────────────────────


def find_snapshot(url: str, asof: datetime) -> Snapshot | None:
    """Return the newest PIT-safe Wayback snapshot for ``url``, or None."""
    rows = _cdx_query(url, asof)
    return _pick_latest_pit_row(rows, asof)


def article_body_pit(
    url: str,
    asof: datetime,
    *,
    cache: WaybackCache | None = None,
) -> str | None:
    """Return clean article text for ``url`` as it existed on/before asof.

    Returns None when no PIT-safe snapshot exists (never falls back to
    the live URL). Results (including negative results) are cached.

    Args:
        url: The article URL.
        asof: PIT boundary — nothing captured after this is eligible.
        cache: Optional WaybackCache. A fresh default one is used if None.
    """
    cache = cache or WaybackCache()
    asof_date = asof.strftime("%Y-%m-%d")

    hit, body = cache.get(url, asof_date)
    if hit:
        return body  # may be None (known miss) or '' (empty extraction)

    snapshot = find_snapshot(url, asof)
    if snapshot is None:
        cache.put(url, asof_date, None, None)
        return None

    html = _fetch_raw(snapshot)
    if html is None:
        # Transient fetch failure — do NOT cache as a permanent miss.
        return None

    extracted = trafilatura.extract(html) or ""
    cache.put(url, asof_date, snapshot.timestamp, extracted)
    return extracted
