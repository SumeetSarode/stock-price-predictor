"""Tests for prediction.trading_calendar — NSE horizon math.

Test anchors (all verified against the live pandas-market-calendars NSE
data for 2026):

  Holiday clusters used:
    Jan 26 2026 (Mon) — Republic Day    [Fri Jan 23 → Tue Jan 27]
    Apr 14 2026 (Tue) — Ambedkar Jayanti [Mon Apr 13 → Wed Apr 15]

  Month-end edge:
    Aug 31 2026 (Mon, trading) → Sep 30 2026 (Wed, trading)
    Jan 30 2026 (Fri, trading) + 1mo = Mar 1 (Sun) → snap back Feb 27 (Fri)

If pandas-market-calendars updates its NSE holiday list, these tests
will fail loudly — which is what we want. False positives here are a
real risk to grading correctness, so we anchor on specific calendar
truths rather than synthetic dates.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from price_predictor.prediction.trading_calendar import (
    HORIZON_NAMES,
    IST,
    MARKET_CLOSE_HOUR,
    MARKET_CLOSE_MINUTE,
    is_trading_day,
    last_trading_day_on_or_before,
    next_trading_session_close,
    target_datetime_for_horizon,
)


# ─────────────────────────────────────────────────────────────
# Sanity: module-level constants
# ─────────────────────────────────────────────────────────────
class TestConstants:
    def test_ist_is_plus_530(self):
        assert IST.utcoffset(None) == timedelta(hours=5, minutes=30)

    def test_market_close_is_1530(self):
        assert (MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE) == (15, 30)

    def test_horizon_names_locked(self):
        # Locked vocabulary — adding a new horizon means an explicit
        # update here AND in the schema enum. Catches accidental drift.
        assert HORIZON_NAMES == ("daily", "weekly", "biweekly", "monthly")


# ─────────────────────────────────────────────────────────────
# is_trading_day
# ─────────────────────────────────────────────────────────────
class TestIsTradingDay:
    def test_regular_weekday(self):
        # Wed Apr 15 2026 — verified trading
        assert is_trading_day(date(2026, 4, 15)) is True

    def test_saturday(self):
        # Sat Jan 24 2026
        assert is_trading_day(date(2026, 1, 24)) is False

    def test_sunday(self):
        # Sun Jan 25 2026
        assert is_trading_day(date(2026, 1, 25)) is False

    def test_republic_day_holiday(self):
        # Mon Jan 26 2026 — Republic Day. Weekday but NOT trading.
        # Catches both "weekend = non-trading" and "holiday = non-trading"
        # in one assertion: it's a Monday so weekend filter alone wouldn't
        # catch it.
        assert is_trading_day(date(2026, 1, 26)) is False

    def test_ambedkar_jayanti_holiday(self):
        # Tue Apr 14 2026 — Ambedkar Jayanti. Same point as above with
        # a different holiday — guards against "we hardcoded only Republic
        # Day" regressions.
        assert is_trading_day(date(2026, 4, 14)) is False

    def test_friday_before_holiday(self):
        # Fri Jan 23 2026 — last trading day before Republic Day cluster
        assert is_trading_day(date(2026, 1, 23)) is True

    def test_tuesday_after_holiday(self):
        # Tue Jan 27 2026 — first trading day after Republic Day cluster
        assert is_trading_day(date(2026, 1, 27)) is True


# ─────────────────────────────────────────────────────────────
# next_trading_session_close (DAILY horizon)
# ─────────────────────────────────────────────────────────────
class TestNextTradingSessionClose:
    """Per spec:

      Branch A: mid-session on a trading day  → today's 15:30 IST
      Branch B: post-close OR non-trading day → next trading day's 15:30 IST
      Edge:    exactly 15:30 IST on trading day → next trading day (close
               just happened; "next session" means tomorrow, not zero-length)
    """

    def test_mid_session_trading_day(self):
        # Wed Apr 15 2026 at 10:00 IST — market open
        now = datetime(2026, 4, 15, 10, 0, tzinfo=IST)
        result = next_trading_session_close(now)
        assert result == datetime(2026, 4, 15, 15, 30, tzinfo=IST)

    def test_pre_open_trading_day_targets_today(self):
        # Wed Apr 15 2026 at 8:00 IST — before market open but on a
        # trading day. Spec: target = today's close (we're inside the
        # "before today's close" window).
        now = datetime(2026, 4, 15, 8, 0, tzinfo=IST)
        result = next_trading_session_close(now)
        assert result == datetime(2026, 4, 15, 15, 30, tzinfo=IST)

    def test_post_close_trading_day_targets_next(self):
        # Wed Apr 15 2026 at 16:00 IST — after market close
        now = datetime(2026, 4, 15, 16, 0, tzinfo=IST)
        result = next_trading_session_close(now)
        # Next trading day = Thu Apr 16
        assert result == datetime(2026, 4, 16, 15, 30, tzinfo=IST)

    def test_exactly_at_close_targets_next(self):
        # Wed Apr 15 2026 at 15:30:00 IST exactly — the close moment.
        # Spec: NOT today (zero-length window); next trading day.
        now = datetime(2026, 4, 15, 15, 30, tzinfo=IST)
        result = next_trading_session_close(now)
        assert result == datetime(2026, 4, 16, 15, 30, tzinfo=IST)

    def test_saturday_targets_monday(self):
        # Sat Jan 24 2026 at any time — not a trading day, skip weekend.
        # Mon Jan 26 is Republic Day → Tue Jan 27 is the answer.
        # GREAT test: covers BOTH weekend AND holiday skip in one shot.
        now = datetime(2026, 1, 24, 10, 0, tzinfo=IST)
        result = next_trading_session_close(now)
        assert result == datetime(2026, 1, 27, 15, 30, tzinfo=IST)

    def test_friday_post_close_skips_holiday_cluster(self):
        # Fri Jan 23 2026 at 16:00 IST — post-close on Friday.
        # Sat (24), Sun (25), Mon Republic Day (26) all non-trading.
        # Next trading day = Tue Jan 27.
        now = datetime(2026, 1, 23, 16, 0, tzinfo=IST)
        result = next_trading_session_close(now)
        assert result == datetime(2026, 1, 27, 15, 30, tzinfo=IST)

    def test_monday_holiday_targets_tuesday(self):
        # Mon Jan 26 2026 (Republic Day) at 10:00 IST.
        # Today not a trading day → next is Tue Jan 27.
        now = datetime(2026, 1, 26, 10, 0, tzinfo=IST)
        result = next_trading_session_close(now)
        assert result == datetime(2026, 1, 27, 15, 30, tzinfo=IST)

    def test_returns_ist_tz(self):
        # Result must always be in IST regardless of input tz
        now_utc = datetime(2026, 4, 15, 4, 30, tzinfo=timezone.utc)  # = 10:00 IST
        result = next_trading_session_close(now_utc)
        assert result.tzinfo == IST

    def test_naive_datetime_rejected(self):
        with pytest.raises(ValueError, match="Naive datetime"):
            next_trading_session_close(datetime(2026, 4, 15, 10, 0))


# ─────────────────────────────────────────────────────────────
# last_trading_day_on_or_before
# ─────────────────────────────────────────────────────────────
class TestLastTradingDayOnOrBefore:
    def test_target_is_trading_day_returns_unchanged(self):
        # Wed Apr 15 is a trading day → returns itself
        assert last_trading_day_on_or_before(date(2026, 4, 15)) == date(2026, 4, 15)

    def test_saturday_snaps_to_friday(self):
        # Sat Jan 24 → Fri Jan 23
        assert last_trading_day_on_or_before(date(2026, 1, 24)) == date(2026, 1, 23)

    def test_sunday_snaps_to_friday(self):
        # Sun Jan 25 → Fri Jan 23
        assert last_trading_day_on_or_before(date(2026, 1, 25)) == date(2026, 1, 23)

    def test_holiday_snaps_to_prior_friday(self):
        # Mon Jan 26 (Republic Day) → Fri Jan 23 (skips Sat/Sun too)
        assert last_trading_day_on_or_before(date(2026, 1, 26)) == date(2026, 1, 23)

    def test_holiday_tuesday_snaps_to_prior_monday(self):
        # Tue Apr 14 (Ambedkar Jayanti) → Mon Apr 13
        # Different snap distance (1 day, not 3) — guards against
        # off-by-one in the lookback window logic.
        assert last_trading_day_on_or_before(date(2026, 4, 14)) == date(2026, 4, 13)


# ─────────────────────────────────────────────────────────────
# target_datetime_for_horizon — the public entry point
# ─────────────────────────────────────────────────────────────
class TestTargetDatetimeForHorizon:
    def test_daily_delegates_to_next_session(self):
        # Should match next_trading_session_close exactly
        now = datetime(2026, 4, 15, 10, 0, tzinfo=IST)
        assert (
            target_datetime_for_horizon("daily", now)
            == next_trading_session_close(now)
        )

    def test_weekly_simple_case(self):
        # Thu Apr 16 + 7 days = Thu Apr 23 (trading day, no snap)
        as_of = datetime(2026, 4, 16, 10, 0, tzinfo=IST)
        result = target_datetime_for_horizon("weekly", as_of)
        assert result == datetime(2026, 4, 23, 15, 30, tzinfo=IST)

    def test_weekly_target_lands_on_holiday_snaps_back(self):
        # Tue Apr 7 + 7 days = Tue Apr 14 (Ambedkar Jayanti).
        # Must snap to Mon Apr 13.
        # First verify Apr 7 itself is trading (so as_of is realistic).
        assert is_trading_day(date(2026, 4, 7))
        as_of = datetime(2026, 4, 7, 10, 0, tzinfo=IST)
        result = target_datetime_for_horizon("weekly", as_of)
        assert result == datetime(2026, 4, 13, 15, 30, tzinfo=IST)

    def test_biweekly_simple_case(self):
        # Wed Apr 15 + 14 days = Wed Apr 29 (verified trading)
        as_of = datetime(2026, 4, 15, 10, 0, tzinfo=IST)
        result = target_datetime_for_horizon("biweekly", as_of)
        assert result == datetime(2026, 4, 29, 15, 30, tzinfo=IST)

    def test_monthly_normal_case(self):
        # Mon Aug 31 2026 + 1 month = Wed Sep 30 2026 (both trading).
        # This is a critical test: relativedelta handles "Aug 31 + 1mo"
        # as Sep 30 (clamps to last day of month) rather than Oct 1.
        as_of = datetime(2026, 8, 31, 10, 0, tzinfo=IST)
        result = target_datetime_for_horizon("monthly", as_of)
        assert result == datetime(2026, 9, 30, 15, 30, tzinfo=IST)

    def test_monthly_target_lands_on_weekend_snaps_back(self):
        # Fri Jan 30 2026 + 1 month = Sun Mar 1 2026.
        # Must snap to Fri Feb 27 2026 (verified trading).
        as_of = datetime(2026, 1, 30, 10, 0, tzinfo=IST)
        result = target_datetime_for_horizon("monthly", as_of)
        assert result == datetime(2026, 2, 27, 15, 30, tzinfo=IST)

    def test_unknown_horizon_raises(self):
        as_of = datetime(2026, 4, 15, 10, 0, tzinfo=IST)
        with pytest.raises(ValueError, match="Unknown horizon"):
            target_datetime_for_horizon("yearly", as_of)  # type: ignore[arg-type]

    def test_naive_as_of_rejected(self):
        with pytest.raises(ValueError, match="Naive datetime"):
            target_datetime_for_horizon("daily", datetime(2026, 4, 15, 10, 0))

    def test_input_in_utc_normalized_to_ist(self):
        # 09:00 UTC on Apr 15 = 14:30 IST on Apr 15. Mid-session.
        # Target should be today's 15:30 IST close.
        as_of_utc = datetime(2026, 4, 15, 9, 0, tzinfo=timezone.utc)
        result = target_datetime_for_horizon("daily", as_of_utc)
        assert result == datetime(2026, 4, 15, 15, 30, tzinfo=IST)

    def test_all_horizons_return_ist_tz_at_market_close(self):
        # Property: every result lands at 15:30 IST exactly.
        # Catches accidental tz drift or wrong-time-of-day bugs.
        as_of = datetime(2026, 4, 15, 10, 0, tzinfo=IST)
        for h in HORIZON_NAMES:
            r = target_datetime_for_horizon(h, as_of)
            assert r.tzinfo == IST
            assert (r.hour, r.minute) == (15, 30)
