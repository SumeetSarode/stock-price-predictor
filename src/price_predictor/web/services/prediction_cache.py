"""Prediction cache — persists prediction results so the watchlist
panel can render them instantly across refreshes and restarts.

Design choices:
  - Append-only: every prediction run inserts a new row. We keep history
    for calibration / grading (which prediction was right vs wrong?).
    "Current" prediction = latest row for (ticker, horizon).
  - Denormalized columns + view_json blob. The columns let us query
    quickly without parsing JSON ("find all bullish weekly predictions
    with R/R > 2"). The blob preserves the full render-ready view dict
    so rich pages don't have to re-run the LLM.
  - Freshness is horizon-dependent. Daily predictions go stale fast
    (24h); monthly predictions stay fresh for ~30 days. Encoded once
    in HORIZON_FRESHNESS so callers don't repeat the rules.

What this module does NOT do:
  - It does not call predict() itself. The web layer's get_or_run
    helper composes this cache with the existing prediction_service.
    Keeps cache logic pure + trivially testable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from price_predictor.web.services.db import get_connection


# Per-horizon freshness rules. The values are tuned to be a little
# *shorter* than the prediction's own validity window — better to
# refresh slightly early than to show stale data on the home page.
HORIZON_FRESHNESS: dict[str, timedelta] = {
    "daily":    timedelta(hours=24),
    "weekly":   timedelta(days=5),
    "biweekly": timedelta(days=10),
    "monthly":  timedelta(days=25),
}


def _normalize_ticker(ticker: str) -> str:
    """Match watchlist_service normalization: UPPER + .NS suffix."""
    t = ticker.strip().upper()
    if not t.endswith(".NS"):
        t = f"{t}.NS"
    return t


# ── Data model ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CachedPrediction:
    """A prediction row loaded back out of the cache."""

    ticker: str
    horizon: str
    created_at: datetime           # tz-aware UTC
    direction: str                 # bullish | bearish | neutral
    confidence_pct: int
    close_price: float
    entry_low: float
    entry_high: float
    target_value: float
    stop_value: float
    risk_reward: float | None
    view: dict[str, Any]           # parsed view_json — full render-ready dict

    @property
    def is_stale(self) -> bool:
        """True iff this prediction is past its horizon-specific freshness."""
        ttl = HORIZON_FRESHNESS.get(self.horizon, timedelta(days=7))
        return (datetime.now(timezone.utc) - self.created_at) > ttl

    @property
    def age_label(self) -> str:
        """Human label: 'just now', '3 hrs ago', '2 days ago'."""
        delta = datetime.now(timezone.utc) - self.created_at
        secs = delta.total_seconds()
        if secs < 60:
            return "just now"
        if secs < 3600:
            mins = int(secs // 60)
            return f"{mins} min ago"
        if secs < 86400:
            hrs = int(secs // 3600)
            return f"{hrs} hr ago" if hrs == 1 else f"{hrs} hrs ago"
        days = int(secs // 86400)
        return f"{days} day ago" if days == 1 else f"{days} days ago"


# ── Public API ──────────────────────────────────────────────────────


def get_latest(ticker: str, horizon: str) -> CachedPrediction | None:
    """Return the most recent cached prediction for (ticker, horizon).

    Returns None if nothing has ever been cached for this pair.
    Stale entries are still returned — callers decide whether to use
    them, kick off a refresh, or both.
    """
    t = _normalize_ticker(ticker)
    h = horizon.lower().strip()
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT ticker, horizon, created_at, direction, confidence_pct,
                   close_price, entry_low, entry_high, target_value, stop_value,
                   risk_reward, view_json
              FROM predictions_cache
             WHERE ticker = ? AND horizon = ?
             ORDER BY created_at DESC
             LIMIT 1
            """,
            (t, h),
        ).fetchone()

    if row is None:
        return None

    return CachedPrediction(
        ticker=row["ticker"],
        horizon=row["horizon"],
        created_at=datetime.fromisoformat(row["created_at"]),
        direction=row["direction"],
        confidence_pct=int(row["confidence_pct"]),
        close_price=float(row["close_price"]),
        entry_low=float(row["entry_low"]),
        entry_high=float(row["entry_high"]),
        target_value=float(row["target_value"]),
        stop_value=float(row["stop_value"]),
        risk_reward=float(row["risk_reward"]) if row["risk_reward"] is not None else None,
        view=json.loads(row["view_json"]),
    )


def get_latest_many(tickers: list[str], horizon: str) -> dict[str, CachedPrediction]:
    """Bulk-fetch latest predictions for many tickers — one DB roundtrip.

    Used by the panel service to populate N cards at once without
    issuing N queries.
    """
    if not tickers:
        return {}

    h = horizon.lower().strip()
    normalized = [_normalize_ticker(t) for t in tickers]
    placeholders = ",".join("?" * len(normalized))

    # Window-function trick: rank rows newest-first per (ticker, horizon),
    # then keep only rank=1. Faster + simpler than a JOIN with a subquery
    # for our scale.
    sql = f"""
        WITH ranked AS (
          SELECT
            ticker, horizon, created_at, direction, confidence_pct,
            close_price, entry_low, entry_high, target_value, stop_value,
            risk_reward, view_json,
            ROW_NUMBER() OVER (
              PARTITION BY ticker, horizon
              ORDER BY created_at DESC
            ) AS rn
          FROM predictions_cache
          WHERE horizon = ? AND ticker IN ({placeholders})
        )
        SELECT * FROM ranked WHERE rn = 1
    """
    with get_connection() as conn:
        rows = conn.execute(sql, (h, *normalized)).fetchall()

    return {
        row["ticker"]: CachedPrediction(
            ticker=row["ticker"],
            horizon=row["horizon"],
            created_at=datetime.fromisoformat(row["created_at"]),
            direction=row["direction"],
            confidence_pct=int(row["confidence_pct"]),
            close_price=float(row["close_price"]),
            entry_low=float(row["entry_low"]),
            entry_high=float(row["entry_high"]),
            target_value=float(row["target_value"]),
            stop_value=float(row["stop_value"]),
            risk_reward=float(row["risk_reward"]) if row["risk_reward"] is not None else None,
            view=json.loads(row["view_json"]),
        )
        for row in rows
    }


def save(view: dict[str, Any]) -> None:
    """Persist a prediction view dict to the cache.

    Accepts the dict shape produced by ``prediction_service._to_view_dict``.
    The dict already has all the denormalized fields we need; this fn
    just plucks them out and INSERTs.

    No upsert — we keep history. Callers that want "latest only"
    behavior should use get_latest().
    """
    ticker = _normalize_ticker(view["ticker"])
    horizon = str(view["horizon"]).lower().strip()
    now = datetime.now(timezone.utc).isoformat()

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO predictions_cache (
                ticker, horizon, created_at, direction, confidence_pct,
                close_price, entry_low, entry_high, target_value, stop_value,
                risk_reward, view_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker, horizon, now,
                str(view["direction"]),
                int(view["confidence_pct"]),
                float(view["close_price"]),
                float(view["entry_low"]),
                float(view["entry_high"]),
                float(view["target_value"]),
                float(view["stop_value"]),
                float(view["risk_reward"]) if view.get("risk_reward") is not None else None,
                json.dumps(view, default=str),  # default=str → handle datetime/Decimal/Enum
            ),
        )
