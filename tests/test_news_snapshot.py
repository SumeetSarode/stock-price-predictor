"""Unit tests for NewsSnapshot - the disk-backed point-in-time GDELT cache.

WHAT WE TEST
============
The contract that makes backtest replay reproducible:
  - Path computation is deterministic (same input -> same path).
  - Different inputs produce different paths (no key collisions).
  - Cache HIT loads from disk without ever calling fetch_news.
  - Cache MISS calls fetch_news, post-filters by published_at <= as_of,
    and persists.
  - Atomic writes (temp file + rename, no half-written JSON).
  - Singleton get/set is symmetric; tests can swap stores.

WHAT WE DON'T TEST (intentionally)
==================================
  - Real GDELT API calls -- too slow + flaky for unit tests. The
    fetch_news shim is mocked and we trust its existing test suite.
  - Concurrency safety beyond "different keys = different files" --
    the docstring says last-writer-wins on same key is acceptable;
    we don't try to assert that here.
"""
from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from price_predictor.data.news import NewsFetchError
from price_predictor.data.news_providers import NewsFetchError as RssFetchError
from price_predictor.data.news_providers.base import NewsProviderCoverage
from price_predictor.data.news_snapshot import (
    NewsSnapshot,
    NewsSnapshotError,
    _hash_key,
    _safe_lang,
    get_news_snapshot,
    set_news_snapshot,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def tmp_snapshot(tmp_path: Path) -> NewsSnapshot:
    """Fresh snapshot rooted in a per-test tmpdir (hermetic isolation)."""
    return NewsSnapshot(tmp_path / "news_snapshots")


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset module-level singleton each test so cross-test state can't leak."""
    set_news_snapshot(None)
    yield
    set_news_snapshot(None)


def _sample_articles_df() -> pd.DataFrame:
    """3 articles spanning the as_of cutoff: 2 before, 1 after.

    Used by post-filter tests to assert the AFTER article is dropped
    even when GDELT returns it (seendate-vs-publishdate skew).
    """
    return pd.DataFrame(
        {
            "title": ["before-1", "before-2", "after-cutoff"],
            "url": ["http://a/1", "http://a/2", "http://a/3"],
            "published_at": pd.to_datetime(
                ["2024-06-13T12:00:00Z",
                 "2024-06-14T08:00:00Z",
                 "2024-06-15T01:00:00Z"],  # AFTER cutoff
                utc=True,
            ),
            "source": ["src1", "src2", "src3"],
            "language": ["eng", "eng", "eng"],
        }
    )


# ─────────────────────────────────────────────────────────────
# 1. Key computation -- determinism + collision-resistance
# ─────────────────────────────────────────────────────────────
class TestKeyHashing:
    def test_same_inputs_produce_same_key(self):
        """Replay needs deterministic keys -- same logical fetch hits
        the same file on every run.
        """
        k1 = _hash_key("Reliance Industries", "eng", 7)
        k2 = _hash_key("Reliance Industries", "eng", 7)
        assert k1 == k2

    def test_different_query_produces_different_key(self):
        k1 = _hash_key("Reliance", "eng", 7)
        k2 = _hash_key("TCS", "eng", 7)
        assert k1 != k2

    def test_different_lookback_produces_different_key(self):
        """Lookback is part of the cache key: a 7-day window and a 30-day
        window for the same query are different snapshots and must NOT
        collide on disk.
        """
        k1 = _hash_key("Reliance", "eng", 7)
        k2 = _hash_key("Reliance", "eng", 30)
        assert k1 != k2

    def test_safe_lang_strips_garbage(self):
        assert _safe_lang("eng") == "eng"
        assert _safe_lang("ENG") == "eng"
        assert _safe_lang("eng/x") == "engx"
        assert _safe_lang("../../../etc/passwd") == "etcpasswd"
        assert _safe_lang("") == "unk"  # never empty


# ─────────────────────────────────────────────────────────────
# 2. Path computation
# ─────────────────────────────────────────────────────────────
class TestPathComputation:
    def test_path_for_includes_date_dir(self, tmp_snapshot: NewsSnapshot):
        path = tmp_snapshot.path_for(
            "Reliance Industries", date(2024, 6, 14), 7,
        )
        assert path.parent.name == "2024-06-14"
        assert path.suffix == ".json"

    def test_path_for_is_deterministic(self, tmp_snapshot: NewsSnapshot):
        p1 = tmp_snapshot.path_for("Reliance", date(2024, 6, 14), 7)
        p2 = tmp_snapshot.path_for("Reliance", date(2024, 6, 14), 7)
        assert p1 == p2

    def test_different_as_of_different_path(self, tmp_snapshot: NewsSnapshot):
        p1 = tmp_snapshot.path_for("Reliance", date(2024, 6, 14), 7)
        p2 = tmp_snapshot.path_for("Reliance", date(2024, 6, 15), 7)
        assert p1 != p2
        assert p1.parent != p2.parent  # different day dirs

    def test_root_created_on_init(self, tmp_path: Path):
        root = tmp_path / "nonexistent" / "deep" / "snapshots"
        assert not root.exists()
        NewsSnapshot(root)
        assert root.exists()


# ─────────────────────────────────────────────────────────────
# 3. Cache miss -- fetches live, post-filters, persists
# ─────────────────────────────────────────────────────────────
class TestCacheMiss:
    def test_miss_calls_fetch_news_and_persists(
        self, tmp_snapshot: NewsSnapshot,
    ):
        """First call for a (query, as_of) tuple: fetch, filter, save."""
        with patch(
            "price_predictor.data.news_snapshot.fetch_news",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.return_value = _sample_articles_df()

            df = asyncio.run(tmp_snapshot.get_or_fetch(
                "Reliance Industries", date(2024, 6, 14), 7,
            ))

            mock_fetch.assert_awaited_once()
            # File now exists on disk.
            path = tmp_snapshot.path_for(
                "Reliance Industries", date(2024, 6, 14), 7,
            )
            assert path.exists()
            # Post-filter dropped the after-cutoff row.
            assert len(df) == 2
            assert "after-cutoff" not in df["title"].values

    def test_miss_post_filters_published_at(
        self, tmp_snapshot: NewsSnapshot,
    ):
        """The whole reason the snapshot exists: GDELT can return articles
        whose published_at > as_of (seendate-vs-publishdate skew). Drop
        them aggressively so the cache is honest.
        """
        with patch(
            "price_predictor.data.news_snapshot.fetch_news",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.return_value = _sample_articles_df()

            df = asyncio.run(tmp_snapshot.get_or_fetch(
                "Reliance", date(2024, 6, 14), 7,
            ))

            # Only the 2 articles published on or before 2024-06-14
            # survived. The 2024-06-15 row was dropped.
            assert set(df["title"]) == {"before-1", "before-2"}

    def test_empty_response_is_a_valid_cached_result(
        self, tmp_snapshot: NewsSnapshot,
    ):
        """An empty DataFrame is NOT a fetch failure -- the query may
        legitimately match nothing. We MUST cache it so the next call
        doesn't re-hit GDELT.
        """
        with patch(
            "price_predictor.data.news_snapshot.fetch_news",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.return_value = pd.DataFrame(
                columns=["title", "url", "published_at", "source", "language"]
            )

            df1 = asyncio.run(tmp_snapshot.get_or_fetch(
                "ObscureCompanyXYZ", date(2024, 6, 14), 7,
            ))
            df2 = asyncio.run(tmp_snapshot.get_or_fetch(
                "ObscureCompanyXYZ", date(2024, 6, 14), 7,
            ))

            assert df1.empty and df2.empty
            # Second call MUST be cache hit, not a re-fetch.
            assert mock_fetch.await_count == 1


# ─────────────────────────────────────────────────────────────
# 4. Cache HIT -- never calls fetch_news
# ─────────────────────────────────────────────────────────────
class TestCacheHit:
    def test_second_call_loads_from_disk(
        self, tmp_snapshot: NewsSnapshot,
    ):
        """The replay guarantee: a snapshot exists -> we never touch
        the network again. This is what makes month-old backtests
        reproducible without hammering GDELT.
        """
        with patch(
            "price_predictor.data.news_snapshot.fetch_news",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.return_value = _sample_articles_df()

            # First call: cache miss, hits the network.
            df1 = asyncio.run(tmp_snapshot.get_or_fetch(
                "Reliance", date(2024, 6, 14), 7,
            ))
            # Second call: cache hit, MUST NOT hit the network.
            df2 = asyncio.run(tmp_snapshot.get_or_fetch(
                "Reliance", date(2024, 6, 14), 7,
            ))

            assert mock_fetch.await_count == 1  # Only the first
            # And the loaded DataFrame is bit-for-bit equal (modulo
            # column order — JSON sort_keys=True alphabetises keys for
            # deterministic diffs, which is desirable on its own).
            pd.testing.assert_frame_equal(
                df1.reset_index(drop=True),
                df2.reset_index(drop=True),
                check_dtype=False,  # JSON round-trip nudges some dtypes
                check_like=True,    # column order doesn't matter
            )

    def test_hit_preserves_published_at_as_tz_aware(
        self, tmp_snapshot: NewsSnapshot,
    ):
        """published_at must come back as tz-aware UTC -- consumers
        (post-filters, prompt builders) rely on the dtype. JSON
        round-trip would naively give string columns; the loader
        is responsible for restoring the datetime type.
        """
        with patch(
            "price_predictor.data.news_snapshot.fetch_news",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.return_value = _sample_articles_df()
            asyncio.run(tmp_snapshot.get_or_fetch(
                "Reliance", date(2024, 6, 14), 7,
            ))
            # Reload from disk via a second call.
            df = asyncio.run(tmp_snapshot.get_or_fetch(
                "Reliance", date(2024, 6, 14), 7,
            ))

            assert pd.api.types.is_datetime64_any_dtype(df["published_at"])
            assert df["published_at"].dt.tz is not None


# ─────────────────────────────────────────────────────────────
# 5. Singleton get/set
# ─────────────────────────────────────────────────────────────
class TestSingleton:
    def test_default_is_none(self):
        assert get_news_snapshot() is None

    def test_set_and_get(self, tmp_snapshot: NewsSnapshot):
        set_news_snapshot(tmp_snapshot)
        assert get_news_snapshot() is tmp_snapshot

    def test_clear_with_none(self, tmp_snapshot: NewsSnapshot):
        set_news_snapshot(tmp_snapshot)
        set_news_snapshot(None)
        assert get_news_snapshot() is None


# ─────────────────────────────────────────────────────────────
# 6. On-disk format -- humans can grep it
# ─────────────────────────────────────────────────────────────
class TestPersistedFormat:
    def test_payload_includes_query_for_human_grep(
        self, tmp_snapshot: NewsSnapshot,
    ):
        """The query string lives inside the JSON so an operator
        can grep the cache without recomputing hashes.
        """
        with patch(
            "price_predictor.data.news_snapshot.fetch_news",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.return_value = _sample_articles_df()
            asyncio.run(tmp_snapshot.get_or_fetch(
                "Reliance Industries", date(2024, 6, 14), 7,
            ))

            path = tmp_snapshot.path_for(
                "Reliance Industries", date(2024, 6, 14), 7,
            )
            payload = json.loads(path.read_text())
            assert payload["query"] == "Reliance Industries"
            assert payload["as_of"] == "2024-06-14"
            assert payload["lookback_days"] == 7
            assert payload["lang"] == "eng"
            assert payload["article_count"] == 2  # post-filter applied
            assert "fetched_at" in payload  # for forensics

    def test_corrupt_file_raises_snapshot_error(
        self, tmp_snapshot: NewsSnapshot,
    ):
        """If the cache file is mangled, raise NewsSnapshotError so the
        caller can distinguish 'cache broken' from 'GDELT broken'.
        """
        path = tmp_snapshot.path_for("X", date(2024, 6, 14), 7)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ this is not valid json", encoding="utf-8")

        with pytest.raises(NewsSnapshotError, match="Cannot load snapshot"):
            tmp_snapshot._load(path)


# ────────────────────────────────────────────────────
# 7. RSS live fallback when GDELT fails (Scope A)
# ────────────────────────────────────────────────────
class _FakeRss:
    """Stand-in for GoogleNewsRssProvider with REAL coverage logic so the
    look-ahead freshness guard is exercised for real; only the network
    .fetch is faked."""

    def __init__(self, *, freshness_days=30, df=None, err=None):
        self._cov = NewsProviderCoverage(
            historical=False, freshness_days=freshness_days
        )
        self._df = df
        self._err = err
        self.fetch_called = False

    @property
    def name(self):
        return "google_news_rss"

    @property
    def coverage(self):
        return self._cov

    async def fetch(self, *args, **kwargs):
        self.fetch_called = True
        if self._err is not None:
            raise self._err
        return self._df


def _rss_df(published_at) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title": ["rss headline"],
            "url": ["https://et.com/live"],
            "published_at": pd.to_datetime([published_at], utc=True),
            "source": ["Economic Times"],
            "language": ["eng"],
        }
    )


class TestRssFallback:
    def test_fresh_window_uses_rss_when_gdelt_fails(self, tmp_snapshot):
        """GDELT 429s on a recent window -> RSS fallback supplies articles."""
        today = date.today()
        rss_df = _rss_df(pd.Timestamp(today, tz="UTC") - pd.Timedelta(days=1))
        fake = _FakeRss(df=rss_df)

        with patch(
            "price_predictor.data.news_snapshot.fetch_news",
            new_callable=AsyncMock, side_effect=NewsFetchError("GDELT 429"),
        ), patch(
            "price_predictor.data.news_snapshot.GoogleNewsRssProvider",
            return_value=fake,
        ):
            df = asyncio.run(tmp_snapshot.get_or_fetch("Reliance", today, 7))

        assert fake.fetch_called is True
        assert len(df) == 1
        assert df.iloc[0]["source"] == "Economic Times"

    def test_stale_backtest_window_skips_rss_and_reraises(self, tmp_snapshot):
        """Look-ahead guard: an old window must NOT be served by a live feed.
        RSS is never called; the original GDELT error propagates."""
        old = date(2019, 3, 14)
        fake = _FakeRss(df=_rss_df(pd.Timestamp("2026-07-09", tz="UTC")))

        with patch(
            "price_predictor.data.news_snapshot.fetch_news",
            new_callable=AsyncMock, side_effect=NewsFetchError("GDELT 429"),
        ), patch(
            "price_predictor.data.news_snapshot.GoogleNewsRssProvider",
            return_value=fake,
        ), pytest.raises(NewsFetchError, match="GDELT 429"):
            asyncio.run(tmp_snapshot.get_or_fetch("Reliance", old, 7))

        assert fake.fetch_called is False  # guard skipped it

    def test_disabled_flag_reraises_without_rss(self, tmp_snapshot, monkeypatch):
        from price_predictor.config.settings import settings as _settings
        monkeypatch.setattr(_settings, "news_rss_fallback_enabled", False)
        fake = _FakeRss(df=_rss_df(pd.Timestamp(date.today(), tz="UTC")))
        with patch(
            "price_predictor.data.news_snapshot.fetch_news",
            new_callable=AsyncMock, side_effect=NewsFetchError("GDELT 429"),
        ), patch(
            "price_predictor.data.news_snapshot.GoogleNewsRssProvider",
            return_value=fake,
        ), pytest.raises(NewsFetchError, match="GDELT 429"):
            asyncio.run(tmp_snapshot.get_or_fetch("X", date.today(), 7))
        assert fake.fetch_called is False

    def test_empty_rss_reraises_and_does_not_cache(self, tmp_snapshot):
        """GDELT ERRORED (not 'found nothing'). Empty RSS must re-raise so we
        don't cache an empty result and permanently neuter this key."""
        today = date.today()
        empty = pd.DataFrame(
            columns=["title", "url", "published_at", "source", "language"]
        )
        fake = _FakeRss(df=empty)
        with patch(
            "price_predictor.data.news_snapshot.fetch_news",
            new_callable=AsyncMock, side_effect=NewsFetchError("GDELT 429"),
        ), patch(
            "price_predictor.data.news_snapshot.GoogleNewsRssProvider",
            return_value=fake,
        ), pytest.raises(NewsFetchError, match="GDELT 429"):
            asyncio.run(tmp_snapshot.get_or_fetch("X", today, 7))
        # Nothing cached.
        assert not tmp_snapshot.path_for("X", today, 7).exists()

    def test_rss_also_fails_reraises_original_gdelt_error(self, tmp_snapshot):
        today = date.today()
        fake = _FakeRss(err=RssFetchError("RSS down"))
        with patch(
            "price_predictor.data.news_snapshot.fetch_news",
            new_callable=AsyncMock, side_effect=NewsFetchError("GDELT 429"),
        ), patch(
            "price_predictor.data.news_snapshot.GoogleNewsRssProvider",
            return_value=fake,
        ), pytest.raises(NewsFetchError, match="GDELT 429"):
            asyncio.run(tmp_snapshot.get_or_fetch("X", today, 7))
        assert fake.fetch_called is True

    def test_valueerror_from_gdelt_never_triggers_rss(self, tmp_snapshot):
        """Caller's bad input -> raise immediately, no fallback."""
        fake = _FakeRss(df=_rss_df(pd.Timestamp(date.today(), tz="UTC")))
        with patch(
            "price_predictor.data.news_snapshot.fetch_news",
            new_callable=AsyncMock, side_effect=ValueError("bad query"),
        ), patch(
            "price_predictor.data.news_snapshot.GoogleNewsRssProvider",
            return_value=fake,
        ), pytest.raises(ValueError, match="bad query"):
            asyncio.run(tmp_snapshot.get_or_fetch("X", date.today(), 7))
        assert fake.fetch_called is False

    def test_fresh_window_tells_gdelt_to_fail_fast(self, tmp_snapshot):
        """Live window + RSS enabled => GDELT called with 0 retries so a 429
        falls over to RSS in <1s instead of sleeping ~15s."""
        today = date.today()
        good = _rss_df(pd.Timestamp(today, tz="UTC") - pd.Timedelta(days=1))
        with patch(
            "price_predictor.data.news_snapshot.fetch_news",
            new_callable=AsyncMock, return_value=good,
        ) as mock_fetch:
            asyncio.run(tmp_snapshot.get_or_fetch("Reliance", today, 7))
        kwargs = mock_fetch.call_args.kwargs
        assert kwargs["rate_limit_retries"] == 0
        assert kwargs["network_retries"] == 0

    def test_stale_window_keeps_gdelt_retries(self, tmp_snapshot):
        """Backtest window (RSS can't help) => GDELT keeps its full patient
        retry budget (None => module default)."""
        old = date(2019, 3, 14)
        good = _rss_df(pd.Timestamp("2019-03-13", tz="UTC"))
        with patch(
            "price_predictor.data.news_snapshot.fetch_news",
            new_callable=AsyncMock, return_value=good,
        ) as mock_fetch:
            asyncio.run(tmp_snapshot.get_or_fetch("Reliance", old, 7))
        kwargs = mock_fetch.call_args.kwargs
        assert kwargs["rate_limit_retries"] is None
        assert kwargs["network_retries"] is None

    def test_disabled_fallback_keeps_gdelt_retries(
        self, tmp_snapshot, monkeypatch,
    ):
        """RSS disabled => no fast fallback to lean on => keep GDELT retries
        even on a fresh window."""
        from price_predictor.config.settings import settings as _settings
        monkeypatch.setattr(_settings, "news_rss_fallback_enabled", False)
        today = date.today()
        good = _rss_df(pd.Timestamp(today, tz="UTC") - pd.Timedelta(days=1))
        with patch(
            "price_predictor.data.news_snapshot.fetch_news",
            new_callable=AsyncMock, return_value=good,
        ) as mock_fetch:
            asyncio.run(tmp_snapshot.get_or_fetch("Reliance", today, 7))
        kwargs = mock_fetch.call_args.kwargs
        assert kwargs["rate_limit_retries"] is None
        assert kwargs["network_retries"] is None
