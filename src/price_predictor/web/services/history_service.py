"""Prediction history service — read-only views over predictions_cache.

Backs both:
  - The 'recent predictions' table on /history (all tickers, paginated)
  - The 'history for this stock' table on /stock/<ticker> (single ticker)

Append-only cache means each row is a historical snapshot of what
the model thought at that moment. Once we add grading (substep 2H+),
we'll join against actual price movement to compute hit/miss outcomes.

Phase 1 (now): just list the predictions. Grading column shows '—'
for everything; the column is wired up so we can populate it in 2H
without changing the template.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from price_predictor.web.services.db import get_connection


@dataclass(frozen=True, slots=True)
class HistoryRow:
    """One row in the history table.

    Mirrors the denormalized columns of predictions_cache plus a few
    derived fields the template wants directly.
    """

    id: int
    ticker: str
    horizon: str
    created_at: datetime
    direction: str
    confidence_pct: int
    close_price: float
    entry_low: float
    entry_high: float
    target_value: float
    stop_value: float
    risk_reward: float | None
    grade: str | None  # "hit" / "miss" / "pending" / None — populated in 2H

    @property
    def display_ticker(self) -> str:
        return self.ticker.removesuffix(".NS")

    @property
    def age_label(self) -> str:
        """Friendly relative-time label."""
        from datetime import timezone
        delta = datetime.now(timezone.utc) - self.created_at
        secs = delta.total_seconds()
        if secs < 60:    return "just now"
        if secs < 3600:  return f"{int(secs//60)} min ago"
        if secs < 86400: hrs = int(secs//3600); return f"{hrs} hr ago" if hrs == 1 else f"{hrs} hrs ago"
        days = int(secs // 86400)
        return f"{days} day ago" if days == 1 else f"{days} days ago"


def _row_to_history(row) -> HistoryRow:
    return HistoryRow(
        id=int(row["id"]),
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
        grade=None,  # 2H will populate this from the grades table
    )


def list_history(
    *,
    ticker: str | None = None,
    horizon: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[HistoryRow], int]:
    """Return (rows, total_count) for the history table.

    Filters applied left-to-right; passing None for any filter means
    'no filter on that column'. Newest-first ordering.

    Returns ``total_count`` separately so the template can show "showing
    50 of 142" and render pagination controls.
    """
    where: list[str] = []
    params: list[str] = []
    if ticker:
        t = ticker.strip().upper()
        if not t.endswith(".NS"):
            t = f"{t}.NS"
        where.append("ticker = ?")
        params.append(t)
    if horizon:
        where.append("horizon = ?")
        params.append(horizon.lower().strip())
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    with get_connection() as conn:
        count_row = conn.execute(
            f"SELECT COUNT(*) AS c FROM predictions_cache {where_sql}",
            params,
        ).fetchone()
        total = int(count_row["c"])

        rows = conn.execute(
            f"""
            SELECT id, ticker, horizon, created_at, direction, confidence_pct,
                   close_price, entry_low, entry_high, target_value, stop_value,
                   risk_reward, view_json
              FROM predictions_cache
              {where_sql}
             ORDER BY created_at DESC
             LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        ).fetchall()

    return [_row_to_history(r) for r in rows], total
