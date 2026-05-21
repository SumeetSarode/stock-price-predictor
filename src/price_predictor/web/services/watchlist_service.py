"""Watchlist persistence — add / remove / list starred tickers.

Tiny CRUD layer on top of the watchlist SQLite table. All SQL is
contained here; callers (routes, dashboard_service) only see Python.

Conventions:
  - Tickers are stored UPPERCASE with .NS suffix (e.g. 'RELIANCE.NS').
  - Normalization happens on the boundary (add()) — internal helpers
    assume already-normalized tickers.
  - The 10-stock soft cap is enforced here; routes get a clean
    WatchlistFullError they can render into a friendly toast.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from price_predictor.web.services.db import get_connection


# Soft cap. Adjustable via the constant; no env var yet (YAGNI — wait
# until a real user complains).
MAX_WATCHLIST_SIZE = 10


class WatchlistError(Exception):
    """Base class for watchlist-specific errors."""


class WatchlistFullError(WatchlistError):
    """Raised when add() would exceed MAX_WATCHLIST_SIZE."""

    def __init__(self) -> None:
        super().__init__(
            f"Watchlist is full ({MAX_WATCHLIST_SIZE} stocks max). "
            f"Remove one to add another."
        )


# ── Data model ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class WatchlistEntry:
    """One row in the watchlist table."""

    ticker: str          # 'RELIANCE.NS'
    added_at: datetime   # tz-aware UTC


# ── Internal helpers ────────────────────────────────────────────────


def _normalize(ticker: str) -> str:
    """Upper-case + ensure .NS suffix. Idempotent."""
    t = ticker.strip().upper()
    if not t.endswith(".NS"):
        t = f"{t}.NS"
    return t


# ── Public API ──────────────────────────────────────────────────────


def list_watchlist() -> list[WatchlistEntry]:
    """Return all watchlist entries, newest-added first."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT ticker, added_at FROM watchlist ORDER BY added_at DESC"
        ).fetchall()

    return [
        WatchlistEntry(
            ticker=r["ticker"],
            added_at=datetime.fromisoformat(r["added_at"]),
        )
        for r in rows
    ]


def watchlist_tickers() -> set[str]:
    """Return just the set of starred tickers — for fast 'is this watched?' checks."""
    with get_connection() as conn:
        rows = conn.execute("SELECT ticker FROM watchlist").fetchall()
    return {r["ticker"] for r in rows}


def count_watchlist() -> int:
    """Return how many tickers are starred."""
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM watchlist").fetchone()
    return int(row["c"])


def is_watched(ticker: str) -> bool:
    """Return True iff the given ticker is starred."""
    t = _normalize(ticker)
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM watchlist WHERE ticker = ? LIMIT 1", (t,)
        ).fetchone()
    return row is not None


def add(ticker: str) -> WatchlistEntry:
    """Add a ticker to the watchlist. Idempotent — re-adding is a no-op.

    Raises:
        WatchlistFullError: if we're at the cap and the ticker isn't
            already watched (idempotent re-adds never raise).
    """
    t = _normalize(ticker)
    now = datetime.now(timezone.utc)

    with get_connection() as conn:
        existing = conn.execute(
            "SELECT added_at FROM watchlist WHERE ticker = ?", (t,)
        ).fetchone()
        if existing is not None:
            # Already watched — return existing entry, don't error.
            return WatchlistEntry(
                ticker=t,
                added_at=datetime.fromisoformat(existing["added_at"]),
            )

        count_row = conn.execute("SELECT COUNT(*) AS c FROM watchlist").fetchone()
        if int(count_row["c"]) >= MAX_WATCHLIST_SIZE:
            raise WatchlistFullError()

        conn.execute(
            "INSERT INTO watchlist (ticker, added_at) VALUES (?, ?)",
            (t, now.isoformat()),
        )

    return WatchlistEntry(ticker=t, added_at=now)


def remove(ticker: str) -> bool:
    """Remove a ticker. Returns True iff it was present."""
    t = _normalize(ticker)
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM watchlist WHERE ticker = ?", (t,))
        return cursor.rowcount > 0


def toggle(ticker: str) -> tuple[bool, bool]:
    """Star/unstar a ticker. Returns (now_watched, was_full_attempt).

    If the ticker was watched → unstar, return (False, False).
    If the ticker was not watched and we have capacity → star, return (True, False).
    If we'd exceed the cap → don't star, return (False, True).
    """
    if is_watched(ticker):
        remove(ticker)
        return False, False

    try:
        add(ticker)
        return True, False
    except WatchlistFullError:
        return False, True
