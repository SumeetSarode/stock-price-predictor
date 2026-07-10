"""Tests for data.wayback -- PIT article fetcher (network fully mocked)."""
from __future__ import annotations

from datetime import datetime

import pytest

from price_predictor.data import wayback
from price_predictor.data.wayback import (
    Snapshot,
    WaybackCache,
    _asof_stamp,
    _is_pit_safe,
    _pick_latest_pit_row,
    article_body_pit,
)


@pytest.fixture(autouse=True)
def _no_throttle():
    """Disable politeness delay so tests are instant."""
    wayback.set_min_request_interval(0.0)
    yield
    wayback.set_min_request_interval(1.0)


ASOF = datetime(2023, 6, 15, 12, 0, 0)


def _cdx(rows_after_header):
    header = [
        "urlkey", "timestamp", "original", "mimetype",
        "statuscode", "digest", "length",
    ]
    return [header, *rows_after_header]


def _row(ts, url="http://ex.com/a"):
    return ["ex", ts, url, "text/html", "200", "DIGEST", "1234"]


# ── Pure helpers ────────────────────────────────────────────────────


class TestPureHelpers:
    def test_asof_stamp(self):
        assert _asof_stamp(ASOF) == "20230615120000"

    def test_pit_safe_boundary(self):
        assert _is_pit_safe("20230615120000", ASOF) is True   # equal = safe
        assert _is_pit_safe("20230615115959", ASOF) is True   # before
        assert _is_pit_safe("20230615120001", ASOF) is False  # after

    def test_pick_empty(self):
        assert _pick_latest_pit_row([], ASOF) is None
        assert _pick_latest_pit_row(_cdx([]), ASOF) is None  # header only

    def test_pick_latest_among_safe(self):
        rows = _cdx([
            _row("20230101000000"),
            _row("20230601000000"),   # newest safe
            _row("20230301000000"),
        ])
        snap = _pick_latest_pit_row(rows, ASOF)
        assert snap.timestamp == "20230601000000"

    def test_pick_rejects_post_asof(self):
        rows = _cdx([
            _row("20230301000000"),   # safe
            _row("20240101000000"),   # AFTER asof — must be ignored
        ])
        snap = _pick_latest_pit_row(rows, ASOF)
        assert snap.timestamp == "20230301000000"

    def test_pick_all_post_asof_returns_none(self):
        rows = _cdx([_row("20240101000000"), _row("20250101000000")])
        assert _pick_latest_pit_row(rows, ASOF) is None

    def test_pick_bad_schema_returns_none(self):
        bad = [["not", "the", "expected", "header"], ["a", "b", "c", "d"]]
        assert _pick_latest_pit_row(bad, ASOF) is None

    def test_snapshot_raw_url(self):
        s = Snapshot(timestamp="20230601000000", original_url="http://ex.com/a")
        assert s.raw_url == (
            "https://web.archive.org/web/20230601000000id_/http://ex.com/a"
        )


# ── Cache ───────────────────────────────────────────────────────────


class TestWaybackCache:
    def test_miss_then_put_then_hit(self, tmp_path):
        cache = WaybackCache(db_path=tmp_path / "wb.sqlite")
        hit, body = cache.get("http://ex.com/a", "2023-06-15")
        assert hit is False and body is None

        cache.put("http://ex.com/a", "2023-06-15", "20230601000000", "article text")
        hit, body = cache.get("http://ex.com/a", "2023-06-15")
        assert hit is True and body == "article text"

    def test_negative_result_cached(self, tmp_path):
        cache = WaybackCache(db_path=tmp_path / "wb.sqlite")
        cache.put("http://ex.com/a", "2023-06-15", None, None)
        hit, body = cache.get("http://ex.com/a", "2023-06-15")
        assert hit is True and body is None  # known miss, distinct from uncached

    def test_replace_on_conflict(self, tmp_path):
        cache = WaybackCache(db_path=tmp_path / "wb.sqlite")
        cache.put("u", "2023-06-15", "t1", "v1")
        cache.put("u", "2023-06-15", "t2", "v2")
        _, body = cache.get("u", "2023-06-15")
        assert body == "v2"


# ── article_body_pit orchestration ──────────────────────────────────


class TestArticleBodyPit:
    def test_cache_hit_skips_network(self, tmp_path, monkeypatch):
        cache = WaybackCache(db_path=tmp_path / "wb.sqlite")
        cache.put("http://ex.com/a", "2023-06-15", "20230601000000", "cached body")

        def _boom(*a, **k):
            raise AssertionError("network must not be hit on cache hit")

        monkeypatch.setattr(wayback, "find_snapshot", _boom)
        out = article_body_pit("http://ex.com/a", ASOF, cache=cache)
        assert out == "cached body"

    def test_happy_path_fetch_extract_cache(self, tmp_path, monkeypatch):
        cache = WaybackCache(db_path=tmp_path / "wb.sqlite")
        snap = Snapshot("20230601000000", "http://ex.com/a")
        monkeypatch.setattr(wayback, "find_snapshot", lambda u, a: snap)
        monkeypatch.setattr(wayback, "_fetch_raw", lambda s: "<html>raw</html>")
        monkeypatch.setattr(wayback.trafilatura, "extract", lambda html: "clean text")

        out = article_body_pit("http://ex.com/a", ASOF, cache=cache)
        assert out == "clean text"
        # Persisted for next time.
        hit, body = cache.get("http://ex.com/a", "2023-06-15")
        assert hit is True and body == "clean text"

    def test_no_snapshot_returns_none_and_caches_miss(self, tmp_path, monkeypatch):
        cache = WaybackCache(db_path=tmp_path / "wb.sqlite")
        monkeypatch.setattr(wayback, "find_snapshot", lambda u, a: None)

        out = article_body_pit("http://ex.com/a", ASOF, cache=cache)
        assert out is None
        # Negative result cached (so we don't re-query).
        hit, body = cache.get("http://ex.com/a", "2023-06-15")
        assert hit is True and body is None

    def test_never_falls_back_to_live(self, tmp_path, monkeypatch):
        """The whole point: no snapshot => None, NOT the live page."""
        cache = WaybackCache(db_path=tmp_path / "wb.sqlite")
        monkeypatch.setattr(wayback, "find_snapshot", lambda u, a: None)
        # If _fetch_raw were called with a live URL it'd be a bias leak.
        monkeypatch.setattr(
            wayback, "_fetch_raw",
            lambda s: (_ for _ in ()).throw(AssertionError("no live fallback!")),
        )
        assert article_body_pit("http://ex.com/a", ASOF, cache=cache) is None

    def test_transient_fetch_failure_not_cached_as_miss(self, tmp_path, monkeypatch):
        cache = WaybackCache(db_path=tmp_path / "wb.sqlite")
        snap = Snapshot("20230601000000", "http://ex.com/a")
        monkeypatch.setattr(wayback, "find_snapshot", lambda u, a: snap)
        monkeypatch.setattr(wayback, "_fetch_raw", lambda s: None)  # transient fail

        out = article_body_pit("http://ex.com/a", ASOF, cache=cache)
        assert out is None
        # Must NOT be cached — a retry later should try again.
        hit, _ = cache.get("http://ex.com/a", "2023-06-15")
        assert hit is False

    def test_empty_extraction_cached_as_empty_string(self, tmp_path, monkeypatch):
        cache = WaybackCache(db_path=tmp_path / "wb.sqlite")
        snap = Snapshot("20230601000000", "http://ex.com/a")
        monkeypatch.setattr(wayback, "find_snapshot", lambda u, a: snap)
        monkeypatch.setattr(wayback, "_fetch_raw", lambda s: "<html></html>")
        monkeypatch.setattr(wayback.trafilatura, "extract", lambda html: None)

        out = article_body_pit("http://ex.com/a", ASOF, cache=cache)
        assert out == ""  # empty, but a real (cached) result
        hit, body = cache.get("http://ex.com/a", "2023-06-15")
        assert hit is True and body == ""


# ── find_snapshot wiring ────────────────────────────────────────────


class TestFindSnapshot:
    def test_uses_cdx_and_picks_latest(self, monkeypatch):
        rows = _cdx([_row("20230101000000"), _row("20230601000000")])
        monkeypatch.setattr(wayback, "_cdx_query", lambda u, a: rows)
        snap = wayback.find_snapshot("http://ex.com/a", ASOF)
        assert snap.timestamp == "20230601000000"

    def test_empty_cdx_returns_none(self, monkeypatch):
        monkeypatch.setattr(wayback, "_cdx_query", lambda u, a: [])
        assert wayback.find_snapshot("http://ex.com/a", ASOF) is None
