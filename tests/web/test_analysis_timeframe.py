"""Unit tests for the analysis timeframe plumbing (daily/weekly/monthly).

These cover the pure, network-free bits: the timeframe option list and
the OHLC resampler. The full compute_live_analysis path needs a live data
feed, so it's exercised via the running app / manual QA instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from price_predictor.web.services.analysis_service import (
    DEFAULT_TIMEFRAME,
    _resample_ohlc,
    timeframe_options,
)


def _daily_df(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    price = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, n))
    return pd.DataFrame(
        {
            "open": price,
            "high": price + 1.5,
            "low": price - 1.5,
            "close": price,
            "adj_close": price,
            "volume": np.random.default_rng(1).integers(1e5, 1e6, n),
        },
        index=idx,
    )


class TestTimeframeOptions:
    def test_exposes_daily_weekly_monthly_in_order(self):
        opts = timeframe_options()
        assert [o["key"] for o in opts] == ["daily", "weekly", "monthly"]
        assert all("label" in o for o in opts)

    def test_default_timeframe_is_daily(self):
        assert DEFAULT_TIMEFRAME == "daily"


class TestResampleOhlc:
    def test_weekly_has_fewer_bars_than_daily(self):
        df = _daily_df(400)
        wk = _resample_ohlc(df, "W-FRI")
        assert 0 < len(wk) < len(df)

    def test_monthly_is_coarser_than_weekly(self):
        df = _daily_df(400)
        wk = _resample_ohlc(df, "W-FRI")
        mo = _resample_ohlc(df, "ME")
        assert len(mo) < len(wk)

    def test_ohlc_integrity_preserved(self):
        df = _daily_df(400)
        for rule in ("W-FRI", "ME"):
            out = _resample_ohlc(df, rule)
            assert (out["high"] >= out["low"]).all()
            assert (out["high"] >= out["open"]).all()
            assert (out["high"] >= out["close"]).all()
            assert (out["low"] <= out["open"]).all()
            assert (out["low"] <= out["close"]).all()

    def test_volume_is_summed_not_averaged(self):
        df = _daily_df(60)
        wk = _resample_ohlc(df, "W-FRI")
        # A resampled week's volume should equal the sum of its dailies.
        first_week_end = wk.index[0]
        daily_slice = df.loc[df.index <= first_week_end, "volume"]
        assert wk["volume"].iloc[0] == daily_slice.sum()

    def test_empty_periods_dropped(self):
        df = _daily_df(400)
        out = _resample_ohlc(df, "W-FRI")
        assert not out[["open", "high", "low", "close"]].isna().any().any()


class TestTrailingNanBar:
    """Regression: today's still-forming bar (NaN OHLC) must not blank out
    the latest-bar reads (close, RSI) on the daily view."""

    @pytest.mark.asyncio
    async def test_daily_close_survives_trailing_nan(self, monkeypatch):
        import price_predictor.web.services.analysis_service as svc

        base = _daily_df(260)
        # Simulate the provider's forming bar: last row NaN OHLC.
        base.loc[base.index[-1], ["open", "high", "low", "close"]] = np.nan

        async def _fake_fetch(ticker, lookback_days=400):
            return base

        monkeypatch.setattr(svc, "_fetch_bars", _fake_fetch)

        a = await svc.compute_live_analysis("RELIANCE", timeframe="daily")
        assert a.trend["close"] is not None, "close blanked by trailing NaN bar"
        assert a.momentum["rsi"] is not None, "RSI blanked by trailing NaN bar"
        # dropna should have removed exactly the one NaN row.
        assert a.bars_used == len(base) - 1
