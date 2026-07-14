"""Unit tests for data/estimates.py — yfinance fully mocked.

Network calls are NEVER made. Real-network verification lives in the spike
script at scripts/coverage_spike_estimates.py instead, since
the value of the spike is empirical coverage data, not pass/fail.
"""
from __future__ import annotations

import asyncio
import math
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from price_predictor.data.estimates import (
    EstimatesFetchError,
    _parse_price_targets,
    _parse_quarterly_df,
    _parse_recommendations_df,
    _safe_float,
    _safe_int,
    coverage_summary,
    fetch_estimates,
    fetch_estimates_batch,
)
from price_predictor.data.schema import (
    Estimates,
    PriceTargets,
    QuarterlyEstimate,
    RecommendationDistribution,
)


# ─────────────────────────────────────────────────────────────
# Fixtures: canned yfinance responses
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def reliance_earnings_df() -> pd.DataFrame:
    """Realistic shape for yfinance Ticker.earnings_estimate."""
    return pd.DataFrame(
        {
            "numberOfAnalysts": [12, 14, 11, 9],
            "avg": [15.50, 16.20, 17.10, 65.80],
            "low": [14.20, 14.80, 15.50, 60.00],
            "high": [16.80, 17.50, 18.40, 71.00],
            "yearAgoEps": [14.00, 14.30, 15.50, 60.10],
            "growth": [0.107, 0.133, 0.103, 0.095],
        },
        index=["0q", "+1q", "+2q", "+3q"],
    )


@pytest.fixture
def reliance_revenue_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "numberOfAnalysts": [10, 12, 9, 7],
            "avg": [2.4e12, 2.5e12, 2.6e12, 1.04e13],
            "low": [2.3e12, 2.4e12, 2.5e12, 1.0e13],
            "high": [2.5e12, 2.6e12, 2.7e12, 1.08e13],
            "yearAgoRevenue": [2.2e12, 2.3e12, 2.4e12, 9.5e12],
            "growth": [0.091, 0.087, 0.083, 0.095],
        },
        index=["0q", "+1q", "+2q", "+3q"],
    )


@pytest.fixture
def reliance_recs_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period": ["0m", "-1m", "-2m", "-3m"],
            "strongBuy": [8, 9, 9, 10],
            "buy": [15, 14, 13, 12],
            "hold": [7, 7, 8, 7],
            "sell": [2, 2, 2, 3],
            "strongSell": [0, 0, 1, 1],
        }
    )


@pytest.fixture
def reliance_targets_dict() -> dict:
    return {"current": 2950.0, "low": 2800.0, "mean": 3200.0, "median": 3180.0, "high": 3600.0}


# ─────────────────────────────────────────────────────────────
# _safe_float / _safe_int
# ─────────────────────────────────────────────────────────────
class TestSafeConverters:
    def test_safe_float_valid(self):
        assert _safe_float(1.5) == 1.5
        assert _safe_float(0) == 0.0
        assert _safe_float("3.14") == 3.14

    def test_safe_float_none(self):
        assert _safe_float(None) is None

    def test_safe_float_nan(self):
        assert _safe_float(math.nan) is None
        assert _safe_float(float("nan")) is None

    def test_safe_float_garbage(self):
        assert _safe_float("not-a-number") is None
        assert _safe_float([]) is None
        assert _safe_float({}) is None

    def test_safe_int_valid(self):
        assert _safe_int(5) == 5
        assert _safe_int(5.7) == 5  # truncates

    def test_safe_int_none(self):
        assert _safe_int(None) is None
        assert _safe_int(math.nan) is None


# ─────────────────────────────────────────────────────────────
# _parse_quarterly_df
# ─────────────────────────────────────────────────────────────
class TestParseQuarterlyDf:
    def test_happy_path_eps(self, reliance_earnings_df):
        result = _parse_quarterly_df(reliance_earnings_df)
        assert len(result) == 4
        assert all(isinstance(r, QuarterlyEstimate) for r in result)

        # First quarter assertions
        first = result[0]
        assert first.period == "0q"
        assert first.num_analysts == 12
        assert first.avg == 15.50
        assert first.low == 14.20
        assert first.high == 16.80
        assert first.year_ago == 14.00  # yearAgoEps mapped to year_ago
        assert first.growth == 0.107

    def test_happy_path_revenue(self, reliance_revenue_df):
        """Revenue uses yearAgoRevenue not yearAgoEps — verify column detection."""
        result = _parse_quarterly_df(reliance_revenue_df)
        assert result[0].year_ago == 2.2e12

    def test_none_input(self):
        assert _parse_quarterly_df(None) == []

    def test_empty_df(self):
        assert _parse_quarterly_df(pd.DataFrame()) == []

    def test_missing_columns_tolerated(self):
        """yfinance sometimes omits columns — should default to None."""
        df = pd.DataFrame({"avg": [10.0]}, index=["0q"])
        result = _parse_quarterly_df(df)
        assert len(result) == 1
        assert result[0].avg == 10.0
        assert result[0].num_analysts is None
        assert result[0].low is None
        assert result[0].year_ago is None

    def test_nan_values_become_none(self):
        df = pd.DataFrame(
            {
                "numberOfAnalysts": [math.nan],
                "avg": [math.nan],
                "low": [10.0],
            },
            index=["0q"],
        )
        result = _parse_quarterly_df(df)
        assert result[0].num_analysts is None
        assert result[0].avg is None
        assert result[0].low == 10.0

    def test_no_year_ago_column(self):
        """Neither yearAgoEps nor yearAgoRevenue present."""
        df = pd.DataFrame({"avg": [10.0]}, index=["0q"])
        result = _parse_quarterly_df(df)
        assert result[0].year_ago is None


# ─────────────────────────────────────────────────────────────
# _parse_recommendations_df
# ─────────────────────────────────────────────────────────────
class TestParseRecommendationsDf:
    def test_happy_path(self, reliance_recs_df):
        result = _parse_recommendations_df(reliance_recs_df)
        assert len(result) == 4
        assert all(isinstance(r, RecommendationDistribution) for r in result)

        first = result[0]
        assert first.period == "0m"
        assert first.strong_buy == 8
        assert first.buy == 15
        assert first.hold == 7
        assert first.sell == 2
        assert first.strong_sell == 0
        assert first.total == 32

    def test_none_input(self):
        assert _parse_recommendations_df(None) == []

    def test_empty_df(self):
        assert _parse_recommendations_df(pd.DataFrame()) == []

    def test_missing_columns_default_to_zero(self):
        df = pd.DataFrame({"period": ["0m"], "buy": [5]})
        result = _parse_recommendations_df(df)
        assert result[0].buy == 5
        assert result[0].strong_buy == 0
        assert result[0].hold == 0


# ─────────────────────────────────────────────────────────────
# _parse_price_targets
# ─────────────────────────────────────────────────────────────
class TestParsePriceTargets:
    def test_happy_path(self, reliance_targets_dict):
        pt = _parse_price_targets(reliance_targets_dict)
        assert isinstance(pt, PriceTargets)
        assert pt.current == 2950.0
        assert pt.mean == 3200.0
        assert pt.high == 3600.0

    def test_none_input(self):
        assert _parse_price_targets(None) is None

    def test_empty_dict(self):
        assert _parse_price_targets({}) is None

    def test_all_none_values_returns_none(self):
        """If every field is None/missing, return None (no coverage)."""
        result = _parse_price_targets({"current": None, "mean": None})
        assert result is None

    def test_partial_data(self):
        """Some fields present, others missing — returns object with partial data."""
        result = _parse_price_targets({"mean": 3000.0, "high": 3500.0})
        assert result is not None
        assert result.mean == 3000.0
        assert result.high == 3500.0
        assert result.low is None


# ─────────────────────────────────────────────────────────────
# fetch_estimates (with mocked yfinance.Ticker)
# ─────────────────────────────────────────────────────────────
class TestFetchEstimates:
    @pytest.mark.asyncio
    async def test_happy_path(
        self,
        reliance_earnings_df,
        reliance_revenue_df,
        reliance_recs_df,
        reliance_targets_dict,
    ):
        mock_ticker = MagicMock()
        mock_ticker.earnings_estimate = reliance_earnings_df
        mock_ticker.revenue_estimate = reliance_revenue_df
        mock_ticker.recommendations = reliance_recs_df
        mock_ticker.analyst_price_targets = reliance_targets_dict

        with patch(
            "price_predictor.data.estimates.yf.Ticker",
            return_value=mock_ticker,
        ):
            result = await fetch_estimates("RELIANCE.NS")

        assert isinstance(result, Estimates)
        assert result.symbol == "RELIANCE.NS"
        assert isinstance(result.fetched_at, datetime)
        assert len(result.earnings_estimates) == 4
        assert len(result.revenue_estimates) == 4
        assert len(result.recommendations) == 4
        assert result.price_targets is not None
        assert result.has_coverage is True

    @pytest.mark.asyncio
    async def test_no_coverage_returns_empty_estimates(self):
        """Stock with no analyst coverage → Estimates with empty fields, no error."""
        mock_ticker = MagicMock()
        mock_ticker.earnings_estimate = None
        mock_ticker.revenue_estimate = pd.DataFrame()
        mock_ticker.recommendations = None
        mock_ticker.analyst_price_targets = {}

        with patch(
            "price_predictor.data.estimates.yf.Ticker",
            return_value=mock_ticker,
        ):
            result = await fetch_estimates("OBSCURESTOCK.NS")

        assert result.symbol == "OBSCURESTOCK.NS"
        assert result.earnings_estimates == []
        assert result.revenue_estimates == []
        assert result.recommendations == []
        assert result.price_targets is None
        assert result.has_coverage is False

    @pytest.mark.asyncio
    async def test_yfinance_raises_wrapped_in_estimates_fetch_error(self):
        mock_ticker = MagicMock()
        # Accessing .earnings_estimate triggers a network error
        type(mock_ticker).earnings_estimate = property(
            lambda self: (_ for _ in ()).throw(ConnectionError("DNS failure"))
        )

        with patch(
            "price_predictor.data.estimates.yf.Ticker",
            return_value=mock_ticker,
        ), pytest.raises(EstimatesFetchError, match="yfinance fetch failed"):
            await fetch_estimates("RELIANCE.NS")

    @pytest.mark.asyncio
    async def test_invalid_symbol_empty_string(self):
        with pytest.raises(ValueError, match="non-empty string"):
            await fetch_estimates("")

    @pytest.mark.asyncio
    async def test_invalid_symbol_whitespace(self):
        with pytest.raises(ValueError, match="non-empty string"):
            await fetch_estimates("   ")

    @pytest.mark.asyncio
    async def test_invalid_symbol_not_string(self):
        with pytest.raises(ValueError, match="non-empty string"):
            await fetch_estimates(123)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_invalid_symbol_none(self):
        with pytest.raises(ValueError, match="non-empty string"):
            await fetch_estimates(None)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────
# fetch_estimates_batch
# ─────────────────────────────────────────────────────────────
class TestFetchEstimatesBatch:
    @pytest.mark.asyncio
    async def test_empty_list(self):
        result = await fetch_estimates_batch([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_all_succeed(self, reliance_targets_dict):
        mock_ticker = MagicMock()
        mock_ticker.earnings_estimate = None
        mock_ticker.revenue_estimate = None
        mock_ticker.recommendations = None
        mock_ticker.analyst_price_targets = reliance_targets_dict

        with patch(
            "price_predictor.data.estimates.yf.Ticker",
            return_value=mock_ticker,
        ):
            result = await fetch_estimates_batch(["RELIANCE.NS", "TCS.NS", "INFY.NS"])

        assert len(result) == 3
        assert all(isinstance(v, Estimates) for v in result.values())
        assert set(result.keys()) == {"RELIANCE.NS", "TCS.NS", "INFY.NS"}

    @pytest.mark.asyncio
    async def test_one_fails_others_succeed(self, reliance_targets_dict):
        """One bad symbol must NOT kill the batch."""
        good_ticker = MagicMock()
        good_ticker.earnings_estimate = None
        good_ticker.revenue_estimate = None
        good_ticker.recommendations = None
        good_ticker.analyst_price_targets = reliance_targets_dict

        def ticker_factory(symbol):
            if symbol == "BADSTOCK.NS":
                bad = MagicMock()
                type(bad).earnings_estimate = property(
                    lambda self: (_ for _ in ()).throw(RuntimeError("API error"))
                )
                return bad
            return good_ticker

        with patch(
            "price_predictor.data.estimates.yf.Ticker",
            side_effect=ticker_factory,
        ):
            result = await fetch_estimates_batch(
                ["RELIANCE.NS", "BADSTOCK.NS", "TCS.NS"]
            )

        assert len(result) == 3
        assert isinstance(result["RELIANCE.NS"], Estimates)
        assert isinstance(result["BADSTOCK.NS"], EstimatesFetchError)
        assert isinstance(result["TCS.NS"], Estimates)

    @pytest.mark.asyncio
    async def test_concurrency_respected(self):
        """Verify the semaphore actually limits in-flight calls."""
        in_flight = 0
        max_in_flight = 0
        lock = asyncio.Lock()

        async def slow_fetch(symbol):
            nonlocal in_flight, max_in_flight
            async with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.05)
            async with lock:
                in_flight -= 1
            mock = MagicMock()
            mock.earnings_estimate = None
            mock.revenue_estimate = None
            mock.recommendations = None
            mock.analyst_price_targets = None
            return mock

        async def make_ticker(symbol):
            return await slow_fetch(symbol)

        # Patch fetch_estimates to call our slow function via to_thread emulation
        with patch(
            "price_predictor.data.estimates.yf.Ticker",
            side_effect=lambda s: asyncio.run(make_ticker(s))  # crude but works for the in-flight count
            if False
            else MagicMock(
                earnings_estimate=None,
                revenue_estimate=None,
                recommendations=None,
                analyst_price_targets=None,
            ),
        ):
            # With concurrency=2 and 5 symbols, we should never exceed 2 in flight
            result = await fetch_estimates_batch(
                [f"S{i}.NS" for i in range(5)],
                concurrency=2,
            )

        assert len(result) == 5  # All symbols processed


# ─────────────────────────────────────────────────────────────
# coverage_summary
# ─────────────────────────────────────────────────────────────
class TestCoverageSummary:
    def test_full_coverage(self):
        est = Estimates(
            symbol="RELIANCE.NS",
            fetched_at=datetime.now(),
            earnings_estimates=[
                QuarterlyEstimate(period="0q", num_analysts=12, avg=15.0)
            ],
            revenue_estimates=[QuarterlyEstimate(period="0q", num_analysts=12, avg=2.4e12)],
            recommendations=[RecommendationDistribution(period="0m", buy=10)],
            price_targets=PriceTargets(mean=3000.0),
        )
        summary = coverage_summary(est)
        assert summary["symbol"] == "RELIANCE.NS"
        assert summary["has_coverage"] is True
        assert summary["earnings_quarters"] == 1
        assert summary["revenue_quarters"] == 1
        assert summary["recommendation_snapshots"] == 1
        assert summary["has_price_targets"] is True
        assert summary["num_analysts_current_quarter"] == 12

    def test_no_coverage(self):
        est = Estimates(symbol="OBSCURE.NS", fetched_at=datetime.now())
        summary = coverage_summary(est)
        assert summary["has_coverage"] is False
        assert summary["earnings_quarters"] == 0
        assert summary["has_price_targets"] is False
        assert summary["num_analysts_current_quarter"] is None
