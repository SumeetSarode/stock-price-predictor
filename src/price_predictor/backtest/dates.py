"""Trading-day iteration helpers for backtest runs.

WHY THIS EXISTS
===============
A backtest sweeps `predict()` across many historical as_of dates.
But `predict()` only makes sense on TRADING days -- the technicals
need fresh OHLCV bars, and predictions stamped on market holidays
would be either degenerate (no new bars since Friday) or rejected
by the prediction code's date validation.

This module gives the runner an honest list of NSE trading days to
iterate over, with optional stride for cheap weekly/monthly backtests.

WHY A DEDICATED MODULE (vs inlining in runner.py)
=================================================
Date-sampling logic is INDEPENDENT of orchestration: the same
function is useful for ad-hoc Jupyter analysis, calibration tools,
"how many trading days in Q2?" sanity checks. Keeping it pure +
sync makes it trivially testable without touching async machinery.

Single Responsibility: this file does one thing -- map a calendar
date range to NSE trading days.
"""
from __future__ import annotations

from datetime import date, timedelta

from price_predictor.prediction.trading_calendar import is_trading_day


def trading_days_in_range(
    start: date,
    end: date,
    *,
    stride: int = 1,
) -> list[date]:
    """All NSE trading days in [start, end] inclusive, every Nth one.

    Args:
        start: First date to consider (inclusive).
        end: Last date to consider (inclusive).
        stride: Take every Nth trading day. Common values:
            1   -> daily backtest (every trading day)
            5   -> weekly-ish (one per trading week)
            21  -> monthly-ish (one per trading month)
            Stride is applied to the FILTERED trading-day list, NOT
            calendar days, so stride=5 always lands on a trading day
            even across long weekends/holidays.

    Returns:
        Trading days in chronological order. Empty if the range
        contains no trading days (rare but possible: e.g. Diwali
        cluster + weekend).

    Raises:
        ValueError: start > end (probable caller bug -- fail loud
            rather than silently return []), or stride < 1
            (degenerate; strides are 1-indexed).

    WHY STRIDE STARTS AT INDEX 0 (not arbitrary offset)
    ===================================================
    Predictability: trading_days_in_range(d, d+30, stride=5) always
    starts AT `d` (if d is a trading day) and walks forward by 5.
    No surprise off-by-one for callers building a pipeline.

    PERFORMANCE
    ===========
    O(N) calendar days. is_trading_day is a cached NSE lookup, so
    cost is dominated by the date arithmetic itself. A 1-year
    range (~365 calls) takes <50ms on cold cache, <1ms warm.
    """
    if start > end:
        raise ValueError(
            f"start ({start}) must be <= end ({end}) -- "
            "did you mix up the arguments?"
        )
    if stride < 1:
        raise ValueError(
            f"stride must be >= 1 (got {stride}); use 1 for daily, "
            "5 for weekly, 21 for monthly."
        )

    # Walk every calendar day, keep only trading days.
    all_trading: list[date] = []
    d = start
    while d <= end:
        if is_trading_day(d):
            all_trading.append(d)
        d += timedelta(days=1)

    # Apply stride. Slicing is O(N/stride) and crystal clear -- a
    # manual loop with index-tracking would be more code for zero gain.
    return all_trading[::stride]
