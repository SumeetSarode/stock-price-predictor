"""Replay-mode behaviour for news_impact agent tools (Step 1.5).

Sister file to test_news_impact.py -- extracted because the parent
file was bumping the 600-line ceiling. These tests cover the
contextvar -> _date_window -> tools -> snapshot store pipe end to
end, complementing test_news_snapshot.py (which tests the store
in isolation).
"""
from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from price_predictor.agents.news_impact.agent import (
    _date_window,
    fetch_estimates_tool,
    fetch_recent_news_tool,
)
from price_predictor.data.news_snapshot import (
    NewsSnapshot,
    set_news_snapshot,
)
from price_predictor.data.schema import (
    Estimates,
    PriceTargets,
    QuarterlyEstimate,
    RecommendationDistribution,
)
from price_predictor.prediction.replay_context import replay_context


# ─────────────────────────────────────────────────────────────
# Local fixtures (mirrors test_news_impact.py shape; deliberately
# duplicated so this file can stand alone -- DRY tax not worth a
# shared conftest for two test files).
# ─────────────────────────────────────────────────────────────
def _news_df(rows: int = 3) -> pd.DataFrame:
    """Match fetch_news()'s normalized output shape (string published_at
    is the legacy contract -- the snapshot defensively parses it).
    """
    return pd.DataFrame([
        {
            "title": f"Headline #{i}",
            "url": f"https://example.com/article{i}",
            "published_at": f"2024-06-1{i} 10:00:00",
            "domain": "example.com",
            "snippet": f"Body for article {i}.",
        }
        for i in range(1, rows + 1)
    ])


def _estimates_with_coverage() -> Estimates:
    return Estimates(
        symbol="RELIANCE.NS",
        fetched_at=datetime.now(UTC),
        earnings_estimates=[
            QuarterlyEstimate(
                period="0q", num_analysts=18,
                avg=15.5, low=14.0, high=17.0, year_ago=13.0,
            ),
        ],
        revenue_estimates=[
            QuarterlyEstimate(
                period="0q", num_analysts=18,
                avg=2.5e12,
                low=2.4e12,
                high=2.6e12,
                year_ago=2.3e12,
            ),
        ],
        recommendations=[
            RecommendationDistribution(
                period="0m",
                strong_buy=10, buy=12, hold=8, sell=2, strong_sell=0,
            ),
        ],
        price_targets=PriceTargets(
            current=2900.0, mean=3200.0, high=3500.0, low=2800.0, median=3150.0,
        ),
    )


@pytest.fixture(autouse=True)
def _reset_snapshot_singleton():
    """Each test starts with a clean singleton -- otherwise a leak from
    one test's set_news_snapshot would silently drive another's.
    """
    set_news_snapshot(None)
    yield
    set_news_snapshot(None)


# ─────────────────────────────────────────────────────────────
# _date_window: the load-bearing contextvar consumer
# ─────────────────────────────────────────────────────────────
class TestDateWindow:
    def test_honors_replay_context(self):
        """End MUST be the as_of date when replay is active -- this is
        the single thing that makes ALL date-bounded tools (news,
        filings, prices) honest in backtest mode.
        """
        with replay_context(date(2024, 6, 14)):
            start, end = _date_window(7)
        assert end == "2024-06-14"
        assert start == "2024-06-07"

    def test_lives_today_without_context(self):
        ist = timezone(timedelta(hours=5, minutes=30))
        _, end = _date_window(7)
        assert end == datetime.now(ist).date().strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────
# News tool: snapshot store routing
# ─────────────────────────────────────────────────────────────
class TestNewsToolReplay:
    def test_uses_snapshot_in_replay_mode(self, tmp_path: Path):
        """Replay + snapshot installed -> fetch_news flows through the
        snapshot's get_or_fetch (cached, post-filtered), NOT the agent's
        direct fetch_news. This is what makes month-old backtests
        reproducible without re-hitting GDELT.
        """
        set_news_snapshot(NewsSnapshot(tmp_path / "snaps"))

        with patch(
            "price_predictor.data.news_snapshot.fetch_news",
            new_callable=AsyncMock,
        ) as mock_snap_fetch, patch(
            "price_predictor.agents.news_impact.agent.fetch_news",
            new_callable=AsyncMock,
        ) as mock_direct_fetch:
            mock_snap_fetch.return_value = _news_df(2)

            with replay_context(date(2024, 6, 14)):
                result = asyncio.run(
                    fetch_recent_news_tool("Reliance Industries", days_back=7)
                )

            mock_snap_fetch.assert_awaited_once()
            mock_direct_fetch.assert_not_awaited()
            assert result["status"] == "success"

    def test_falls_back_to_live_without_snapshot(self):
        """Defence in depth: replay context set, snapshot NOT installed
        -> live fetch with a clean degraded path, NOT a crash.
        Important so a misconfigured backtest doesn't blow up entirely.
        """
        with patch(
            "price_predictor.agents.news_impact.agent.fetch_news",
            new_callable=AsyncMock,
        ) as mock_direct:
            mock_direct.return_value = _news_df(1)
            with replay_context(date(2024, 6, 14)):
                result = asyncio.run(
                    fetch_recent_news_tool("Reliance", days_back=7)
                )
            mock_direct.assert_awaited_once()
            assert result["status"] == "success"


# ─────────────────────────────────────────────────────────────
# Estimates tool: short-circuit because there's no historical truth
# ─────────────────────────────────────────────────────────────
class TestEstimatesReplay:
    def test_short_circuits_in_replay_mode(self):
        """yfinance returns CURRENT analyst estimates with no historical
        archive. Returning today's consensus from a backtest would leak
        forward-looking info into the past. Replay mode MUST emit the
        no-coverage shape instead.
        """
        with patch(
            "price_predictor.agents.news_impact.agent.fetch_estimates",
            new_callable=AsyncMock,
        ) as mock_est:
            with replay_context(date(2024, 6, 14)):
                result = asyncio.run(fetch_estimates_tool("RELIANCE.NS"))

            mock_est.assert_not_awaited()  # short-circuited -- no live call
            assert result["status"] == "success"
            assert result["has_coverage"] is False
            assert result["summary"]["price_target_mean"] is None
            assert "replay_note" in result  # audit breadcrumb

    def test_unaffected_in_live_mode(self):
        """No replay -> the live fetch path is bit-for-bit unchanged."""
        with patch(
            "price_predictor.agents.news_impact.agent.fetch_estimates",
            new_callable=AsyncMock,
        ) as mock_est:
            mock_est.return_value = _estimates_with_coverage()
            result = asyncio.run(fetch_estimates_tool("RELIANCE.NS"))
            mock_est.assert_awaited_once()
            assert result["status"] == "success"
            assert result["has_coverage"] is True
