"""Tests for the news provider base contract + coverage look-ahead guard."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from price_predictor.data.news_providers.base import (
    NewsProviderCoverage,
    empty_news_df,
)


class TestCoverageCanServeEnd:
    def test_historical_always_serves(self):
        cov = NewsProviderCoverage(historical=True)
        assert cov.can_serve_end(datetime(2019, 1, 1, tzinfo=UTC)) is True
        assert cov.can_serve_end(datetime.now(UTC)) is True

    def test_live_serves_recent(self):
        cov = NewsProviderCoverage(historical=False, freshness_days=30)
        now = datetime(2026, 7, 10, tzinfo=UTC)
        recent = now - timedelta(days=5)
        assert cov.can_serve_end(recent, now=now) is True

    def test_live_refuses_stale_backtest_window(self):
        # THE look-ahead guard: a live-only source must not answer for an old
        # window (would fabricate 'current' news for a past date).
        cov = NewsProviderCoverage(historical=False, freshness_days=30)
        now = datetime(2026, 7, 10, tzinfo=UTC)
        old = datetime(2019, 3, 14, tzinfo=UTC)
        assert cov.can_serve_end(old, now=now) is False

    def test_live_boundary_is_inclusive(self):
        cov = NewsProviderCoverage(historical=False, freshness_days=30)
        now = datetime(2026, 7, 10, tzinfo=UTC)
        exactly_30d = now - timedelta(days=30)
        assert cov.can_serve_end(exactly_30d, now=now) is True

    def test_naive_end_treated_as_utc(self):
        cov = NewsProviderCoverage(historical=False, freshness_days=30)
        now = datetime(2026, 7, 10, tzinfo=UTC)
        naive_recent = datetime(2026, 7, 8)  # no tzinfo
        assert cov.can_serve_end(naive_recent, now=now) is True


class TestEmptyNewsDf:
    def test_columns_and_dtype(self):
        df = empty_news_df()
        assert list(df.columns) == [
            "title", "url", "published_at", "source", "language",
        ]
        assert len(df) == 0
        import pandas as pd
        assert pd.api.types.is_datetime64_any_dtype(df["published_at"])
