"""Local SQLite database for the web app.

Houses everything that needs to outlive a process restart:
  - watchlist          → user's starred tickers
  - predictions_cache  → cached prediction results (Phase 3)
  - prediction_history → user-visible history table (substep 2F)

Choice of stdlib sqlite3 over SQLAlchemy:
  - Tiny schema (3 tables, ~10 columns total)
  - No relationships needing eager-loading magic
  - Zero extra deps
  - Migrations are 'ALTER TABLE' or 'CREATE TABLE IF NOT EXISTS' —
    no Alembic needed at this scale

If the schema ever balloons, swapping to SQLAlchemy is a contained
refactor — the service layer (watchlist_service, etc.) hides all SQL.

Default DB location:  ~/.price_predictor/app.db
Override:             WEB_DB_PATH env var
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from loguru import logger

from price_predictor.web.settings import settings


# Module-level lock for schema initialization. SQLite is thread-safe
# for queries but we don't want two threads racing to CREATE TABLE on
# first boot. Once init completes, this lock is never touched again.
_init_lock = threading.Lock()
_initialized = False


def _resolve_db_path() -> Path:
    """Return the path the DB should live at, creating parents as needed."""
    db_path = settings.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create tables + indexes if they don't exist. Idempotent."""

    # Watchlist — one row per starred ticker. `added_at` is ISO 8601 UTC.
    # We index by ticker (lookup) and by added_at (chronological display).
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS watchlist (
            ticker      TEXT    PRIMARY KEY,
            added_at    TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS ix_watchlist_added_at
            ON watchlist(added_at);

        -- Predictions cache — append-only history of prediction results.
        -- We intentionally do NOT upsert: every run gets a new row, so we
        -- can grade past predictions against subsequent price moves and
        -- compute calibration metrics. "Current" prediction = latest row
        -- for (ticker, horizon).
        --
        -- view_json holds the full _to_view_dict() output so the stock
        -- detail page can re-render rich content without re-running the
        -- LLM. The denormalized columns (direction, entry_low, etc.) are
        -- there for fast queries / sort / filter without parsing JSON.
        CREATE TABLE IF NOT EXISTS predictions_cache (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT    NOT NULL,
            horizon         TEXT    NOT NULL,
            created_at      TEXT    NOT NULL,
            direction       TEXT    NOT NULL,
            confidence_pct  INTEGER NOT NULL,
            close_price     REAL    NOT NULL,
            entry_low       REAL    NOT NULL,
            entry_high      REAL    NOT NULL,
            target_value    REAL    NOT NULL,
            stop_value      REAL    NOT NULL,
            risk_reward     REAL,
            view_json       TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS ix_predictions_lookup
            ON predictions_cache(ticker, horizon, created_at DESC);
        """
    )
    conn.commit()


def _ensure_initialized() -> None:
    """Run schema init exactly once per process."""
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:  # double-check inside the lock
            return
        db_path = _resolve_db_path()
        logger.info("db: initializing schema at {}", db_path)
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA journal_mode = WAL")  # better concurrency
            conn.execute("PRAGMA foreign_keys = ON")
            _init_schema(conn)
        _initialized = True


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with sensible defaults.

    Usage::

        with get_connection() as conn:
            conn.execute("INSERT INTO watchlist ...", (...))
            conn.commit()

    The context manager commits on success, rolls back on exception.
    Row factory is set to sqlite3.Row so callers can access columns
    by name (row["ticker"]) instead of positional index.
    """
    _ensure_initialized()
    db_path = _resolve_db_path()
    conn = sqlite3.connect(db_path, isolation_level=None)  # autocommit-off
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute("BEGIN")
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def reset_for_tests() -> None:
    """Wipe the in-memory init flag so tests can swap DB paths."""
    global _initialized
    _initialized = False
