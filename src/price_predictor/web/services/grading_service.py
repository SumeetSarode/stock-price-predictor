"""Prediction grading service — compute hit/stop/expired outcomes.

Purpose
=======
The stock detail page has a right-side "How are predictions doing?"
panel that grades every past prediction for the current ticker.
This service is the read-only engine behind it.

For each cached prediction we replay forward bars to determine:
  - HIT       : target reached before stop, within the horizon window
  - STOPPED   : stop hit before (or with) target — conservative tie-break
  - EXPIRED   : neither hit nor stop within the window; final R from close
  - PENDING   : not enough forward bars yet (too recent or weekend)

A scorecard aggregates across all graded predictions (HIT + STOPPED;
EXPIRED is reported separately so it doesn't inflate the hit-rate).

Boundary
========
- READ-ONLY consumer of `history_service` (which reads the cache DB)
  and the shared price cache.
- Never mutates anything; never imports prediction/agents code.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Literal

import pandas as pd

from price_predictor.data._shared_cache import get_cache
from price_predictor.web.services.history_service import HistoryRow, list_history

Outcome = Literal["hit", "stopped", "expired", "pending", "skipped"]

# Trading-bar budget per horizon — how many forward sessions we give a
# prediction to play out before grading it as EXPIRED. Approximations:
#   daily    = 1 session
#   weekly   = 5 sessions (a full trading week)
#   biweekly = 10 sessions
#   monthly  = 22 sessions (~1 calendar month of trading days)
_HORIZON_BARS: dict[str, int] = {
    "daily": 1,
    "weekly": 5,
    "biweekly": 10,
    "monthly": 22,
}
_DEFAULT_BARS = 5  # safety default for unknown horizons

# How many calendar days of OHLCV to pull when grading. Generous so
# weekends + holidays don't truncate the bar window.
_OHLCV_PAD_DAYS = 60


@dataclass(frozen=True, slots=True)
class GradedPrediction:
    """One historical prediction with its computed outcome.

    The original `row` is kept so the template can show entry/target/stop
    without us re-duplicating those fields here.
    """

    row: HistoryRow
    outcome: Outcome
    r_multiple: float | None     # realized R: hit=+R/R, stop=-1, expired=partial
    resolved_at: date | None     # date the outcome was determined
    bars_used: int               # forward bars consumed (0 for pending/skipped)
    note: str | None             # short human-readable detail


@dataclass(frozen=True, slots=True)
class Scorecard:
    """Aggregate stats across graded predictions for a ticker.

    `hit_rate` deliberately EXCLUDES expired/pending — they're not
    resolved outcomes. Reporting them separately keeps the headline
    metric honest.
    """

    total: int
    hits: int
    stops: int
    expired: int
    pending: int
    skipped: int
    avg_r: float | None          # mean R across hit+stop+expired
    hit_rate: float | None       # hits / (hits + stops); None if denom=0


@dataclass(frozen=True, slots=True)
class TickerGrading:
    """Full payload for the right-rail grading panel."""

    ticker: str
    scorecard: Scorecard
    graded: list[GradedPrediction]   # newest-first
    computed_at: datetime


# ─────────────────────────────────────────────────────────────────
# Internal: bar-replay engine
# ─────────────────────────────────────────────────────────────────


def _normalize(ticker: str) -> str:
    t = ticker.strip().upper()
    if not t.endswith(".NS"):
        t = f"{t}.NS"
    return t


def _bars_for_horizon(horizon: str) -> int:
    return _HORIZON_BARS.get(horizon.lower().strip(), _DEFAULT_BARS)


def _r_for_hit(row: HistoryRow) -> float:
    """Realized R when target is hit = the planned R:R (1.5×, 2×, etc).

    Use the prediction's own R:R when present (it's a planned metric),
    otherwise derive from entry midpoint vs target vs stop. Falling back
    to 1.0 when math is degenerate keeps the UI clean.
    """
    if row.risk_reward and row.risk_reward > 0:
        return float(row.risk_reward)
    entry_mid = (row.entry_low + row.entry_high) / 2
    risk = abs(entry_mid - row.stop_value)
    reward = abs(row.target_value - entry_mid)
    return reward / risk if risk > 0 else 1.0


def _r_for_close(row: HistoryRow, final_close: float) -> float:
    """Partial R when the window expires without hit/stop.

    R is measured against the planned risk distance (entry → stop).
    Bullish: positive R when price ended above entry; negative below.
    Bearish: flipped. Neutral predictions are skipped before this runs.
    """
    entry_mid = (row.entry_low + row.entry_high) / 2
    risk = abs(entry_mid - row.stop_value)
    if risk == 0:
        return 0.0
    delta = final_close - entry_mid
    if row.direction == "bearish":
        delta = -delta
    return round(delta / risk, 2)


def _find_anchor_bar_date(
    df: pd.DataFrame, ref_close: float, on_or_before: date,
) -> date | None:
    """Find the bar whose close matches the prediction's reference close.

    A prediction is anchored to the LAST CLOSE the model saw, not the
    moment it ran. If a user runs a prediction at 08:30 IST (before
    market open), the anchor is yesterday's close — today's bar still
    counts as a forward bar to grade against.

    Strategy:
      1. Restrict candidates to bars on/before `on_or_before` (the
         prediction's creation date in IST).
      2. Prefer the most recent bar whose close matches `ref_close`
         within ₹0.05 (handles float rounding).
      3. Fall back to the latest bar on/before the date if no close
         match — conservative, may under-count one bar in edge cases.
      4. Return None when no bars exist at all (rare — brand-new ticker).
    """
    if df.empty:
        return None
    # Compare via .date to dodge tz-awareness mismatches (cache returns
    # an Asia/Kolkata-localized DatetimeIndex; `on_or_before` is naive).
    candidates = df.loc[df.index.date <= on_or_before]
    if candidates.empty:
        return None
    close_matches = candidates[(candidates["close"] - ref_close).abs() < 0.05]
    if not close_matches.empty:
        return close_matches.index[-1].date()
    return candidates.index[-1].date()


def _forward_bars_from_anchor(
    df: pd.DataFrame, anchor: date, max_bars: int,
) -> pd.DataFrame:
    """Return up to `max_bars` bars strictly AFTER the anchor date."""
    if df.empty or anchor is None:
        return df.iloc[0:0]  # empty same-schema slice
    return df.loc[df.index.date > anchor].head(max_bars)


def _replay(row: HistoryRow, df: pd.DataFrame) -> GradedPrediction:
    """Walk forward bars and assign HIT / STOPPED / EXPIRED / PENDING.

    Tie-break rule: if both target and stop are reached in the SAME bar,
    we call it STOPPED. Conservative — without intraday data we can't
    know which came first; pessimistic accounting protects the user from
    over-claiming wins.
    """
    if row.direction == "neutral":
        return GradedPrediction(
            row=row, outcome="skipped", r_multiple=None,
            resolved_at=None, bars_used=0,
            note="Neutral prediction — no directional bet to grade.",
        )

    max_bars = _bars_for_horizon(row.horizon)
    anchor = _find_anchor_bar_date(df, row.close_price, row.created_at.date())

    if anchor is None:
        return GradedPrediction(
            row=row, outcome="pending", r_multiple=None,
            resolved_at=None, bars_used=0,
            note="No matching reference bar in price history yet.",
        )

    forward = _forward_bars_from_anchor(df, anchor, max_bars)

    if forward.empty:
        # Distinguish the two pending sub-cases so the UI tells the truth:
        #   - latest bar IS the anchor → truly waiting for next session
        #   - somehow we have bars but none after anchor (shouldn't happen)
        latest = df.index[-1].date()
        if latest <= anchor:
            note = (
                f"Anchored to {anchor.strftime('%d %b')} close — "
                "waiting for the next trading session to grade."
            )
        else:
            note = f"No bars after {anchor.strftime('%d %b')} in cache."
        return GradedPrediction(
            row=row, outcome="pending", r_multiple=None,
            resolved_at=None, bars_used=0, note=note,
        )

    bullish = row.direction == "bullish"
    target = row.target_value
    stop = row.stop_value

    for i, (ts, bar) in enumerate(forward.iterrows(), start=1):
        high = float(bar["high"])
        low = float(bar["low"])

        hit_target = (high >= target) if bullish else (low <= target)
        hit_stop = (low <= stop) if bullish else (high >= stop)

        if hit_stop:
            # Conservative tie-break: stop wins even when both touched.
            return GradedPrediction(
                row=row, outcome="stopped", r_multiple=-1.0,
                resolved_at=ts.date(), bars_used=i,
                note=f"Stop hit on bar {i} of {max_bars}.",
            )
        if hit_target:
            return GradedPrediction(
                row=row, outcome="hit", r_multiple=_r_for_hit(row),
                resolved_at=ts.date(), bars_used=i,
                note=f"Target reached on bar {i} of {max_bars}.",
            )

    # Window exhausted: grade from final close
    final_close = float(forward.iloc[-1]["close"])
    return GradedPrediction(
        row=row, outcome="expired", r_multiple=_r_for_close(row, final_close),
        resolved_at=forward.index[-1].date(), bars_used=len(forward),
        note=f"Window of {max_bars} bars expired without hit or stop.",
    )


def _build_scorecard(graded: list[GradedPrediction]) -> Scorecard:
    """Aggregate outcomes into the headline scorecard."""
    hits = sum(1 for g in graded if g.outcome == "hit")
    stops = sum(1 for g in graded if g.outcome == "stopped")
    expired = sum(1 for g in graded if g.outcome == "expired")
    pending = sum(1 for g in graded if g.outcome == "pending")
    skipped = sum(1 for g in graded if g.outcome == "skipped")

    resolved_rs = [
        g.r_multiple for g in graded
        if g.outcome in ("hit", "stopped", "expired") and g.r_multiple is not None
    ]
    avg_r = round(sum(resolved_rs) / len(resolved_rs), 2) if resolved_rs else None

    denom = hits + stops
    hit_rate = round(hits / denom, 3) if denom > 0 else None

    return Scorecard(
        total=len(graded),
        hits=hits, stops=stops, expired=expired,
        pending=pending, skipped=skipped,
        avg_r=avg_r, hit_rate=hit_rate,
    )


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────


async def grade_ticker(
    ticker: str,
    *,
    horizon: str | None = None,
    limit: int = 25,
) -> TickerGrading:
    """Grade the last `limit` predictions for `ticker` (optional horizon filter).

    Returns a `TickerGrading` bundle — always non-empty (empty `graded`
    list when there are no historical predictions). Bar fetch failures
    don't crash: predictions just stay PENDING with a soft note.
    """
    t = _normalize(ticker)
    rows, _ = list_history(
        ticker=t, horizon=horizon, limit=limit, offset=0,
    )

    if not rows:
        return TickerGrading(
            ticker=t,
            scorecard=Scorecard(0, 0, 0, 0, 0, 0, None, None),
            graded=[],
            computed_at=datetime.now(),
        )

    # Single OHLCV fetch covering the oldest prediction → today.
    # Cheaper than re-fetching per prediction; the cache layer dedupes
    # too. End date is clamped to TODAY (not today+1) because some
    # providers (e.g. bhavcopy) raise on future dates.
    oldest = min(r.created_at for r in rows).date()
    today = date.today()
    start = oldest - timedelta(days=2)  # tiny pad for index lookups
    end = today

    cache = get_cache()
    fetch_failed = False
    df: pd.DataFrame | None = None
    try:
        df = await cache.get(t, start, end, "1d")
    except Exception:
        # Don't crash — grading degrades gracefully to PENDING with a hint.
        fetch_failed = True

    if fetch_failed or df is None or df.empty:
        # We couldn't replay bars at all. Surface the actual reason so the
        # user knows whether to retry (network) or just wait (no data yet).
        graded_pending = [
            GradedPrediction(
                row=r, outcome="pending", r_multiple=None,
                resolved_at=None, bars_used=0,
                note="Couldn't fetch price history — try the refresh button.",
            )
            for r in rows
        ]
        return TickerGrading(
            ticker=t,
            scorecard=_build_scorecard(graded_pending),
            graded=graded_pending,
            computed_at=datetime.now(),
        )

    graded = [_replay(r, df) for r in rows]
    return TickerGrading(
        ticker=t,
        scorecard=_build_scorecard(graded),
        graded=graded,
        computed_at=datetime.now(),
    )
