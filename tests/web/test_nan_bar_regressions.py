"""Regression tests for the NaN 'still-forming bar' class of bugs.

Some price providers return today's in-progress bar with a NaN close.
If that NaN reaches the UI it (a) blanks the displayed price and, worse,
(b) crashes any JSON endpoint because NaN is not JSON-serialisable
(`ValueError: Out of range float values are not JSON compliant: nan`).

These two bugs took down the whole app once: the dashboard showed no
prices and clicking any stock 500'd (the /api/predictions/detail and
/api/chart JSON responses choked on the NaN). The fix drops NaN-close
rows at the source in both dashboard_service and chart_service. These
tests lock that behaviour so it never regresses.
"""
from __future__ import annotations

import asyncio
import json
import math

import numpy as np
import pandas as pd

from price_predictor.web.services import chart_service, dashboard_service
from price_predictor.web.services.dashboard_service import Stock, _fetch_one_sync
from price_predictor.web.services.chart_service import get_chart_series


def _stock(ticker: str = "RELIANCE.NS", name: str = "Reliance") -> Stock:
    return Stock(ticker=ticker, name=name, sector="X", is_nifty50=True)


def _ohlcv_with_forming_bar() -> pd.DataFrame:
    """3 real daily bars + a trailing still-forming bar with NaN close."""
    idx = pd.to_datetime(
        ["2026-07-15", "2026-07-16", "2026-07-17", "2026-07-20"]
    )
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            # Last bar (today) has not closed yet -> NaN.
            "close": [100.5, 101.5, 102.5, np.nan],
            "volume": [1000, 1100, 1200, np.nan],
        },
        index=idx,
    )


class TestDashboardDropsFormingBar:
    def test_close_is_last_real_bar_not_nan(self, monkeypatch):
        monkeypatch.setattr(
            dashboard_service, "fetch_ohlcv",
            lambda *a, **k: _ohlcv_with_forming_bar(),
        )
        row = _fetch_one_sync(_stock())
        # The displayed close must be the last *real* bar (102.5), never NaN.
        assert row.close == 102.5
        assert row.error is None
        # change_pct is computed off the two most recent real bars.
        assert row.change_pct is not None
        assert not math.isnan(row.change_pct)

    def test_row_close_is_json_serialisable(self, monkeypatch):
        monkeypatch.setattr(
            dashboard_service, "fetch_ohlcv",
            lambda *a, **k: _ohlcv_with_forming_bar(),
        )
        row = _fetch_one_sync(_stock())
        # This json.dumps is exactly what the /api endpoints do — it must
        # not raise "Out of range float values are not JSON compliant".
        json.dumps({"close": row.close, "change_pct": row.change_pct})

    def test_all_nan_close_yields_error_row_not_crash(self, monkeypatch):
        df = _ohlcv_with_forming_bar()
        df["close"] = np.nan  # every bar missing a close
        monkeypatch.setattr(dashboard_service, "fetch_ohlcv", lambda *a, **k: df)
        row = _fetch_one_sync(_stock("X.NS", "X"))
        # Nothing usable -> treated as no data, not a hard crash.
        assert row.close is None
        assert row.error is not None


class TestChartDropsFormingBar:
    def test_closes_have_no_nan_and_are_json_serialisable(self, monkeypatch):
        monkeypatch.setattr(
            chart_service, "fetch_ohlcv",
            lambda *a, **k: _ohlcv_with_forming_bar(),
        )
        series = asyncio.run(get_chart_series("RELIANCE.NS", window_days=90))
        # The forming NaN bar must be dropped: 3 real closes, no NaN.
        assert series.closes == [100.5, 101.5, 102.5]
        assert len(series.dates) == len(series.closes)
        assert not any(math.isnan(c) for c in series.closes)
        # And the JSON the endpoint returns must serialise cleanly.
        json.dumps({"dates": series.dates, "closes": series.closes})

    def test_all_nan_close_yields_empty_series(self, monkeypatch):
        df = _ohlcv_with_forming_bar()
        df["close"] = np.nan
        monkeypatch.setattr(chart_service, "fetch_ohlcv", lambda *a, **k: df)
        series = asyncio.run(get_chart_series("X.NS"))
        assert series.is_empty
        assert series.closes == []
