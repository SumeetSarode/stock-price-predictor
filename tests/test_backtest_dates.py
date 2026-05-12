"""Unit tests for backtest.dates -- trading-day iteration.

WHAT WE TEST
============
The contract that backtest runs depend on:
  - Only NSE trading days are emitted (weekends + holidays excluded).
  - Stride controls density (1=daily, 5=weekly-ish, 21=monthly-ish).
  - Edge cases: same-day range, empty trading-day range, stride > range.
  - Validation rejects caller bugs (start > end, stride < 1).

WHAT WE DON'T TEST
==================
- The NSE calendar itself -- that's a third-party library
  (pandas_market_calendars) with its own test suite. We trust it.
- Cache warm/cold timing -- not a correctness property.
"""
from __future__ import annotations

from datetime import date

import pytest

from price_predictor.backtest.dates import trading_days_in_range


class TestTradingDaysInRange:
    def test_excludes_weekends(self):
        """Sat 2024-06-15 and Sun 2024-06-16 must NOT appear.

        Note: 2024-06-17 is Eid al-Adha (NSE holiday), so we use a
        different week to assert Monday-inclusion separately. Mixing
        weekend + holiday checks here masks intent.
        """
        days = trading_days_in_range(date(2024, 6, 13), date(2024, 6, 18))
        assert date(2024, 6, 15) not in days  # Sat
        assert date(2024, 6, 16) not in days  # Sun
        assert date(2024, 6, 13) in days  # Thu
        assert date(2024, 6, 14) in days  # Fri
        assert date(2024, 6, 18) in days  # Tue

    def test_includes_regular_monday(self):
        """Pick a Monday with no holiday to confirm Mondays aren't
        accidentally being filtered out (sister to the weekend test).
        """
        # 2024-06-10 is a regular Monday (no NSE holiday).
        days = trading_days_in_range(date(2024, 6, 10), date(2024, 6, 10))
        assert days == [date(2024, 6, 10)]

    def test_chronological_order(self):
        """Results must be sorted ascending -- callers iterate them as
        as_of values, ordering matters for log readability + sanity.
        """
        days = trading_days_in_range(date(2024, 6, 1), date(2024, 6, 30))
        assert days == sorted(days)

    def test_inclusive_endpoints(self):
        """Both start and end are inclusive -- a 1-trading-day range
        emits exactly that day.
        """
        days = trading_days_in_range(date(2024, 6, 14), date(2024, 6, 14))
        # 2024-06-14 was a Friday, regular trading day.
        assert days == [date(2024, 6, 14)]

    def test_range_with_no_trading_days_returns_empty(self):
        """Saturday-Sunday-only range -> empty list, not an error.

        WHY: a calibration tool sweeping date(d, d+1) for d in some
        list might legitimately hit a weekend. Returning [] is the
        honest answer; raising would force every caller to wrap.
        """
        days = trading_days_in_range(date(2024, 6, 15), date(2024, 6, 16))
        assert days == []

    # ── Stride ──────────────────────────────────────────────
    def test_stride_1_is_every_trading_day(self):
        """Stride=1 must NOT skip anything -- it's the default."""
        d1 = trading_days_in_range(date(2024, 6, 1), date(2024, 6, 30))
        d2 = trading_days_in_range(date(2024, 6, 1), date(2024, 6, 30), stride=1)
        assert d1 == d2

    def test_stride_5_skips_to_weekly(self):
        """Stride=5 over a month yields ~4-5 dates.

        WHY ~4-5: NSE has ~21 trading days/month; stride=5 -> ~4-5
        samples. Exact count depends on holidays; we just check the
        stride is APPLIED (sample count is reasonable, dates are
        spaced).
        """
        days = trading_days_in_range(
            date(2024, 6, 1), date(2024, 6, 30), stride=5,
        )
        # Reasonable bounds: not full daily (~21), not just one.
        assert 3 <= len(days) <= 6
        # Dates monotonically increasing with at least 5 trading
        # days between consecutive samples (allow holiday slack).
        for prev, curr in zip(days, days[1:]):
            gap_days = (curr - prev).days
            assert gap_days >= 5, f"stride=5 violated: {prev} -> {curr}"

    def test_stride_starts_at_first_trading_day(self):
        """Predictability: result[0] is the first trading day on/after
        start, regardless of stride. Guards against off-by-one bugs
        in pipelines that rely on as_of_dates[0] being a known date.
        """
        # 2024-06-10 is a Monday.
        days = trading_days_in_range(date(2024, 6, 10), date(2024, 6, 30), stride=3)
        assert days[0] == date(2024, 6, 10)

    def test_stride_larger_than_range_returns_one(self):
        """Stride >> available trading days -> just the first one.

        Pythonic slicing semantics ([::stride] for stride > len),
        but we lock the contract so a future implementation can't
        regress to "" or raise.
        """
        days = trading_days_in_range(
            date(2024, 6, 13), date(2024, 6, 14), stride=100,
        )
        assert days == [date(2024, 6, 13)]

    # ── Validation ──────────────────────────────────────────
    def test_start_after_end_raises(self):
        """Caller-bug path: fail loud, don't silently return []."""
        with pytest.raises(ValueError, match="start.*must be <="):
            trading_days_in_range(date(2024, 6, 20), date(2024, 6, 10))

    def test_stride_zero_raises(self):
        """Stride is 1-indexed; 0 is degenerate ([::0] is a Python error)."""
        with pytest.raises(ValueError, match="stride must be >= 1"):
            trading_days_in_range(date(2024, 6, 1), date(2024, 6, 30), stride=0)

    def test_stride_negative_raises(self):
        """Negative stride would reverse the list -- almost certainly
        a caller bug, not a feature. Reject explicitly.
        """
        with pytest.raises(ValueError, match="stride must be >= 1"):
            trading_days_in_range(date(2024, 6, 1), date(2024, 6, 30), stride=-1)

    # ── Smoke against a known-holiday week ──────────────────
    def test_excludes_known_nse_holiday(self):
        """2024-08-15 is Indian Independence Day -- NSE closed.

        Pinning a real holiday makes regressions visible immediately
        if pandas_market_calendars ever drifts on Indian holidays.
        """
        days = trading_days_in_range(date(2024, 8, 14), date(2024, 8, 16))
        assert date(2024, 8, 15) not in days
        assert date(2024, 8, 14) in days  # Wed before
        # 2024-08-16 is Friday -- regular day
        assert date(2024, 8, 16) in days
