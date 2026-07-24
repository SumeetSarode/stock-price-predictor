"""Tests for the shared live GDELT->RSS resilient news fetcher."""
from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from price_predictor.data import news_resilient as nr
from price_predictor.data.news import NewsFetchError
from price_predictor.data.news_providers import NewsFetchError as RssFetchError

pytestmark = pytest.mark.asyncio


def _df(title="headline", published_at=None) -> pd.DataFrame:
    published_at = published_at or (date.today().isoformat())
    return pd.DataFrame(
        {
            "title": [title],
            "url": ["https://et.com/x"],
            "published_at": pd.to_datetime([published_at], utc=True),
            "source": ["Economic Times"],
            "language": ["eng"],
        }
    )


class _FakeRssProvider:
    def __init__(self, *, df=None, err=None):
        self._df = df if df is not None else pd.DataFrame()
        self._err = err
        self.fetch_called = False

    @property
    def name(self):
        return "google_news_rss"

    async def fetch(self, *a, **k):
        self.fetch_called = True
        if self._err is not None:
            raise self._err
        return self._df


# ── rss_can_catch predicate ──────────────────────────────────────────
class TestRssCanCatch:
    def test_fresh_iso_true(self):
        assert nr.rss_can_catch(date.today().isoformat()) is True

    def test_stale_iso_false(self):
        assert nr.rss_can_catch("2019-01-01") is False

    def test_disabled_false(self, monkeypatch):
        from price_predictor.config.settings import settings as s
        monkeypatch.setattr(s, "news_rss_fallback_enabled", False)
        assert nr.rss_can_catch(date.today().isoformat()) is False


# ── fetch_news_resilient ─────────────────────────────────────────────
class TestFetchResilient:
    async def test_gdelt_success_ladder_no_rss(self, monkeypatch):
        mock = AsyncMock(return_value=_df("gdelt"))
        monkeypatch.setattr(nr, "fetch_news_relevant", mock)
        rss = _FakeRssProvider(df=_df("rss"))
        monkeypatch.setattr(nr, "GoogleNewsRssProvider", lambda **k: rss)

        out = await nr.fetch_news_resilient(
            "Reliance", "2024-01-01", date.today().isoformat(),
        )
        assert out.iloc[0]["title"] == "gdelt"
        assert rss.fetch_called is False

    async def test_gdelt_fail_fresh_window_falls_to_rss(self, monkeypatch):
        monkeypatch.setattr(
            nr, "fetch_news_relevant",
            AsyncMock(side_effect=NewsFetchError("GDELT 429")),
        )
        rss = _FakeRssProvider(df=_df("rss"))
        monkeypatch.setattr(nr, "GoogleNewsRssProvider", lambda **k: rss)

        out = await nr.fetch_news_resilient(
            "Reliance", "2024-01-01", date.today().isoformat(),
        )
        assert out.iloc[0]["title"] == "rss"
        assert rss.fetch_called is True

    async def test_gdelt_fail_stale_window_reraises_no_rss(self, monkeypatch):
        monkeypatch.setattr(
            nr, "fetch_news_relevant",
            AsyncMock(side_effect=NewsFetchError("GDELT 429")),
        )
        rss = _FakeRssProvider(df=_df("rss"))
        monkeypatch.setattr(nr, "GoogleNewsRssProvider", lambda **k: rss)

        with pytest.raises(NewsFetchError, match="GDELT 429"):
            await nr.fetch_news_resilient("Reliance", "2019-01-01", "2019-01-31")
        assert rss.fetch_called is False

    async def test_gdelt_fail_rss_empty_reraises(self, monkeypatch):
        monkeypatch.setattr(
            nr, "fetch_news_relevant",
            AsyncMock(side_effect=NewsFetchError("GDELT 429")),
        )
        rss = _FakeRssProvider(df=pd.DataFrame())
        monkeypatch.setattr(nr, "GoogleNewsRssProvider", lambda **k: rss)

        with pytest.raises(NewsFetchError, match="GDELT 429"):
            await nr.fetch_news_resilient(
                "Reliance", "2024-01-01", date.today().isoformat(),
            )

    async def test_gdelt_fail_rss_also_fails_reraises_gdelt(self, monkeypatch):
        monkeypatch.setattr(
            nr, "fetch_news_relevant",
            AsyncMock(side_effect=NewsFetchError("GDELT 429")),
        )
        rss = _FakeRssProvider(err=RssFetchError("rss down"))
        monkeypatch.setattr(nr, "GoogleNewsRssProvider", lambda **k: rss)

        with pytest.raises(NewsFetchError, match="GDELT 429"):
            await nr.fetch_news_resilient(
                "Reliance", "2024-01-01", date.today().isoformat(),
            )

    async def test_valueerror_reraises_no_rss(self, monkeypatch):
        monkeypatch.setattr(
            nr, "fetch_news_relevant",
            AsyncMock(side_effect=ValueError("bad query")),
        )
        rss = _FakeRssProvider(df=_df("rss"))
        monkeypatch.setattr(nr, "GoogleNewsRssProvider", lambda **k: rss)

        with pytest.raises(ValueError, match="bad query"):
            await nr.fetch_news_resilient(
                "Reliance", "2024-01-01", date.today().isoformat(),
            )
        assert rss.fetch_called is False

    async def test_fresh_window_fast_fails_gdelt(self, monkeypatch):
        """Fresh window => GDELT called with 0 retries (fast fail -> RSS)."""
        mock = AsyncMock(return_value=_df("gdelt"))
        monkeypatch.setattr(nr, "fetch_news_relevant", mock)
        await nr.fetch_news_resilient(
            "Reliance", "2024-01-01", date.today().isoformat(),
        )
        kwargs = mock.call_args.kwargs
        assert kwargs["rate_limit_retries"] == 0
        assert kwargs["network_retries"] == 0

    async def test_stale_window_keeps_gdelt_retries(self, monkeypatch):
        mock = AsyncMock(return_value=_df("gdelt", "2019-01-05"))
        monkeypatch.setattr(nr, "fetch_news_relevant", mock)
        await nr.fetch_news_resilient("Reliance", "2019-01-01", "2019-01-31")
        kwargs = mock.call_args.kwargs
        assert kwargs["rate_limit_retries"] is None
        assert kwargs["network_retries"] is None

    async def test_use_ladder_false_calls_fetch_news(self, monkeypatch):
        """Sector path (use_ladder=False) hits fetch_news, not the ladder."""
        mock = AsyncMock(return_value=_df("sector"))
        monkeypatch.setattr(nr, "fetch_news", mock)
        ladder = AsyncMock(return_value=_df("ladder"))
        monkeypatch.setattr(nr, "fetch_news_relevant", ladder)

        out = await nr.fetch_news_resilient(
            "Energy sector", "2024-01-01", date.today().isoformat(),
            use_ladder=False, exact_phrase=False, source_country=None,
        )
        assert out.iloc[0]["title"] == "sector"
        assert ladder.await_count == 0
        assert mock.call_args.kwargs["exact_phrase"] is False
