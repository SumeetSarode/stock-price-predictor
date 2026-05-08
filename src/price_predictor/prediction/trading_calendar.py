"""NSE trading-calendar helpers for horizon → target-date math.

WHY THIS MODULE EXISTS
======================
Three layers — schema (validates predictions are well-formed), predictor
(emits predictions), and grading (evaluates them against actuals) — all
need to answer the same question:

    "Given a prediction made at time T with horizon H, when is the
     target evaluation moment?"

Centralizing that math here means there is ONE definition. If schema and
grading disagreed (e.g. schema says weekly = +5 trading days, grading
says +7 calendar days), we'd silently mis-grade every prediction. So
this module owns the truth and everyone imports from it.

WHY pandas-market-calendars
===========================
NSE has ~15 holidays/year that shift each year (Diwali, Holi, Eid follow
lunar calendars). Hardcoding them ourselves would mean a yearly chore +
chronic bugs every time we forget. `pandas-market-calendars` maintains
NSE's calendar (and 100+ others) and is the de-facto standard. Pure
Python, ~1MB on disk, no native deps. Worth the dep.

THE RULES (locked with user before building)
============================================
- Calendar windows, not trading-day windows. "Weekly" means "+7 calendar
  days," not "+5 trading days." Matches user mental model: "weekly on
  Thursday = next Thursday."
- DAILY horizon means "end of the next applicable NSE session." If we
  predict during market hours, target is today's 15:30 IST close. If
  after-hours/weekend/holiday, target is the next trading day's close.
- For weekly/biweekly/monthly: target_calendar_date = as_of + duration.
  If that calendar date is a non-trading day, fall BACK to the last
  trading day on or before it. This avoids silently extending the window.

PUBLIC API
==========
- IST                                : timezone constant (Asia/Kolkata, +05:30)
- MARKET_CLOSE_HOUR / MINUTE         : NSE close = 15:30 IST
- HORIZON_NAMES                      : ("daily", "weekly", "biweekly", "monthly")
- is_trading_day(d)                  : bool
- next_trading_session_close(now)    : datetime
- last_trading_day_on_or_before(d)   : date
- target_datetime_for_horizon(h, t)  : datetime  ← the one most callers want
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from typing import Literal

from dateutil.relativedelta import relativedelta

import pandas_market_calendars as mcal

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
# Why we build IST manually instead of zoneinfo("Asia/Kolkata"):
# - zoneinfo on macOS sometimes lacks the tzdata package by default
# - IST has no DST, so a fixed-offset tz is exactly correct (forever)
# - Zero ambiguity for tests across machines
IST = timezone(timedelta(hours=5, minutes=30))

# NSE regular session: 09:15 - 15:30 IST. We only care about the close
# (target evaluation moment for `daily` horizon). 15:30 = (15, 30).
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# Public horizon vocabulary. Mirrors the eventual PredictionHorizon enum
# values (commit 2). Defining the strings HERE first so this module has
# no upward dependency on the schema (which would create a cycle).
HorizonName = Literal["daily", "weekly", "biweekly", "monthly"]
HORIZON_NAMES: tuple[HorizonName, ...] = ("daily", "weekly", "biweekly", "monthly")

# How many calendar days to scan back/forward when finding the nearest
# trading day. NSE has at most ~3 consecutive non-trading days (long
# weekend + holiday). 14 is comfortably above the worst case.
_LOOKBACK_DAYS = 14


# ─────────────────────────────────────────────────────────────
# Internal: calendar singleton
# ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _nse():  # pragma: no cover (trivial wrapper)
    """Cached NSE calendar instance.

    Construction is non-trivial (parses regular hours + holiday list).
    Cache it module-wide; the calendar is stateless from our POV.
    """
    return mcal.get_calendar("NSE")


def _to_ist(dt: datetime) -> datetime:
    """Ensure dt is tz-aware and converted to IST.

    Naive datetimes are an error — caller must explicitly choose a tz.
    Silent assumption-of-local-tz is the kind of bug that causes
    off-by-12-hour grading errors in production.
    """
    if dt.tzinfo is None:
        raise ValueError(
            f"Naive datetime not allowed: {dt!r}. Pass a tz-aware datetime "
            "(e.g. datetime.now(IST))."
        )
    return dt.astimezone(IST)


def _ist_close_of(d: date) -> datetime:
    """The 15:30 IST market-close moment on calendar date `d`.

    Caller must have already verified `d` is a trading day; this is a
    pure construction helper, not a validator.
    """
    return datetime.combine(d, time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE), tzinfo=IST)


# ─────────────────────────────────────────────────────────────
# Public: trading-day predicates
# ─────────────────────────────────────────────────────────────
def is_trading_day(d: date) -> bool:
    """True iff `d` is an NSE regular trading day.

    Considers both weekends (handled by NSE's calendar) and the NSE
    holiday list (Republic Day, Diwali, etc.). Cheap — a single index
    lookup against the cached calendar.
    """
    nse = _nse()
    # valid_days returns a DatetimeIndex of trading days in the range.
    # For a single-day query we just check membership.
    valid = nse.valid_days(start_date=d, end_date=d)
    return len(valid) > 0


# ─────────────────────────────────────────────────────────────
# Public: horizon → target-datetime math
# ─────────────────────────────────────────────────────────────
def next_trading_session_close(now: datetime) -> datetime:
    """End of the next applicable NSE trading session.

    The DAILY-horizon target. Three branches by intent:

      Branch A — `now` is mid-session on a trading day:
          target = today's 15:30 IST close
          (Predicted at 10 AM today, evaluated at today's close.)

      Branch B — `now` is post-close on a trading day, OR `now` is on a
                 non-trading day:
          target = next trading day's 15:30 IST close
          (Predicted at 4 PM Mon → Tue close. Predicted Sat → Mon close.)

      Edge case — `now == today's 15:30 IST exactly`:
          Treated as Branch B (close just happened, "next session" is
          tomorrow). Avoids zero-length windows.

    Args:
        now: Tz-aware. Will be converted to IST internally.

    Returns:
        Tz-aware datetime in IST representing the target close.
    """
    now_ist = _to_ist(now)
    today = now_ist.date()
    today_close = _ist_close_of(today)

    # Branch A: still inside today's session
    if is_trading_day(today) and now_ist < today_close:
        return today_close

    # Branch B: find the next trading day STRICTLY after today.
    # Search a 14-day window — comfortably covers any holiday cluster.
    nse = _nse()
    forward = nse.valid_days(
        start_date=today + timedelta(days=1),
        end_date=today + timedelta(days=_LOOKBACK_DAYS),
    )
    if len(forward) == 0:  # pragma: no cover (impossible — NSE always has a day within 14)
        raise RuntimeError(
            f"No NSE trading day found in 14 days after {today}. "
            "Is the calendar data corrupt?"
        )
    next_day = forward[0].date()
    return _ist_close_of(next_day)


def last_trading_day_on_or_before(target: date) -> date:
    """Largest NSE trading day ≤ `target`.

    Used for weekly/biweekly/monthly horizons: we compute the calendar
    target date (as_of + 7 days, etc.), then snap it back to a real
    trading day. Snapping FORWARD would silently extend the prediction
    window — confidence math assumes a fixed window, so we always snap
    backward.

    Args:
        target: Calendar date (no tz needed — pure date math).

    Returns:
        Trading date on or before `target`. If `target` itself is a
        trading day, returns `target` unchanged.

    Raises:
        RuntimeError: If no trading day exists within 14 days before
            `target`. Indicates corrupt calendar data.
    """
    nse = _nse()
    window = nse.valid_days(
        start_date=target - timedelta(days=_LOOKBACK_DAYS),
        end_date=target,
    )
    if len(window) == 0:  # pragma: no cover
        raise RuntimeError(
            f"No NSE trading day found in 14 days ending {target}. "
            "Is the calendar data corrupt?"
        )
    return window[-1].date()


def target_datetime_for_horizon(horizon: HorizonName, as_of: datetime) -> datetime:
    """Compute the target evaluation moment for a (horizon, as_of) pair.

    THE PUBLIC ENTRY POINT. Schema, predictor, grading all call this.

    Mapping (calendar-anchored):

      daily     → next_trading_session_close(as_of)
      weekly    → last trading day ≤ (as_of_date + 7 calendar days)  @ 15:30 IST
      biweekly  → last trading day ≤ (as_of_date + 14 calendar days) @ 15:30 IST
      monthly   → last trading day ≤ (as_of_date + 1 calendar month) @ 15:30 IST

    Why calendar windows (not trading-day windows):
        Matches user mental model. "Weekly on Thursday" means "by next
        Thursday" — 7 calendar days — regardless of holidays in between.

    Why snap to last trading day ≤ target:
        We can only evaluate against real OHLCV bars. Snapping forward
        would silently widen the window and inflate confidence math.

    Args:
        horizon: One of HORIZON_NAMES.
        as_of: Tz-aware prediction-time anchor.

    Returns:
        Tz-aware datetime in IST. Always falls on a trading day's
        market close (15:30 IST), so OHLCV lookups are well-defined.

    Raises:
        ValueError: On naive `as_of` or unknown horizon.
    """
    as_of_ist = _to_ist(as_of)

    if horizon == "daily":
        return next_trading_session_close(as_of_ist)

    # Calendar offsets per horizon
    as_of_date = as_of_ist.date()
    if horizon == "weekly":
        target_cal = as_of_date + timedelta(days=7)
    elif horizon == "biweekly":
        target_cal = as_of_date + timedelta(days=14)
    elif horizon == "monthly":
        # relativedelta handles month-end correctly:
        # Aug 31 + 1 month = Sep 30 (not Sep 31, which doesn't exist)
        target_cal = as_of_date + relativedelta(months=1)
    else:
        raise ValueError(
            f"Unknown horizon: {horizon!r}. "
            f"Expected one of {HORIZON_NAMES}."
        )

    target_trading = last_trading_day_on_or_before(target_cal)
    return _ist_close_of(target_trading)
