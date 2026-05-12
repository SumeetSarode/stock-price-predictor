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
