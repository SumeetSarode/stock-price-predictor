"""Tests for the resumable yfinance sector-backfill (data/sector_lookup.py).

Network is fully injected — these tests never touch yfinance.
"""
from __future__ import annotations

import json

import pytest

from price_predictor.data import sector_lookup as sl
from price_predictor.data.sector_lookup import (
    UNKNOWN_SECTOR,
    SectorFetchError,
    backfill_sectors,
    load_cache,
    save_cache,
)


class TestCacheRoundTrip:
    def test_load_missing_returns_empty(self, tmp_path):
        assert load_cache(tmp_path / "nope.json") == {}

    def test_save_then_load(self, tmp_path):
        p = tmp_path / "sec.json"
        save_cache(p, {"RELIANCE.NS": "Energy", "INFY.NS": "Technology"})
        assert load_cache(p) == {"RELIANCE.NS": "Energy", "INFY.NS": "Technology"}

    def test_corrupt_cache_starts_fresh(self, tmp_path):
        p = tmp_path / "sec.json"
        p.write_text("{not valid json", encoding="utf-8")
        assert load_cache(p) == {}

    def test_save_is_atomic_no_tmp_left(self, tmp_path):
        p = tmp_path / "sec.json"
        save_cache(p, {"A.NS": "X"})
        assert not (tmp_path / "sec.json.tmp").exists()


class TestBackfill:
    def test_resolves_all_and_writes_cache(self, tmp_path):
        p = tmp_path / "sec.json"
        sectors = {"RELIANCE.NS": "Energy", "INFY.NS": "Technology"}
        cache = backfill_sectors(
            sectors.keys(), cache_path=p,
            fetcher=lambda t: sectors[t], max_workers=2,
        )
        assert cache == sectors
        assert json.loads(p.read_text()) == sectors

    def test_skips_already_cached(self, tmp_path):
        p = tmp_path / "sec.json"
        save_cache(p, {"RELIANCE.NS": "Energy"})
        called = []

        def fetch(t):
            called.append(t)
            return "Technology"

        cache = backfill_sectors(
            ["RELIANCE.NS", "INFY.NS"], cache_path=p, fetcher=fetch,
        )
        # Only the uncached ticker hit the fetcher.
        assert called == ["INFY.NS"]
        assert cache == {"RELIANCE.NS": "Energy", "INFY.NS": "Technology"}

    def test_none_sector_cached_as_unknown(self, tmp_path):
        p = tmp_path / "sec.json"
        cache = backfill_sectors(
            ["WEIRD.NS"], cache_path=p, fetcher=lambda t: None,
        )
        assert cache == {"WEIRD.NS": UNKNOWN_SECTOR}

    def test_transient_failure_left_uncached_for_retry(self, tmp_path):
        p = tmp_path / "sec.json"

        def flaky(t):
            if t == "RATELIMITED.NS":
                raise SectorFetchError("rate limited")
            return "Energy"

        cache = backfill_sectors(
            ["RELIANCE.NS", "RATELIMITED.NS"], cache_path=p, fetcher=flaky,
        )
        # The good one is cached; the failed one is absent → retried next run.
        assert cache == {"RELIANCE.NS": "Energy"}
        assert "RATELIMITED.NS" not in load_cache(p)

    def test_rerun_retries_only_failures(self, tmp_path):
        p = tmp_path / "sec.json"
        state = {"fail": True}

        def flaky(t):
            if t == "FLAKY.NS" and state["fail"]:
                raise SectorFetchError("transient")
            return "Healthcare"

        backfill_sectors(["OK.NS", "FLAKY.NS"], cache_path=p,
                         fetcher=lambda t: "Energy" if t == "OK.NS" else flaky(t))
        assert load_cache(p) == {"OK.NS": "Energy"}
        # Second run: the transient issue is gone.
        state["fail"] = False
        cache = backfill_sectors(["OK.NS", "FLAKY.NS"], cache_path=p, fetcher=flaky)
        assert cache == {"OK.NS": "Energy", "FLAKY.NS": "Healthcare"}

    def test_unexpected_exception_treated_as_transient(self, tmp_path):
        p = tmp_path / "sec.json"

        def boom(t):
            raise ValueError("surprise")

        cache = backfill_sectors(["X.NS"], cache_path=p, fetcher=boom)
        assert cache == {}  # not cached, retryable

    def test_dedups_input(self, tmp_path):
        p = tmp_path / "sec.json"
        calls = []
        backfill_sectors(
            ["A.NS", "A.NS", "A.NS"], cache_path=p,
            fetcher=lambda t: calls.append(t) or "Energy",
        )
        assert calls == ["A.NS"]

    def test_empty_input_no_crash(self, tmp_path):
        assert backfill_sectors([], cache_path=tmp_path / "s.json") == {}

    def test_progress_callback_invoked(self, tmp_path):
        p = tmp_path / "sec.json"
        seen = []
        backfill_sectors(
            ["A.NS", "B.NS"], cache_path=p, fetcher=lambda t: "Energy",
            progress=lambda d, tot, f: seen.append((d, tot, f)),
        )
        assert len(seen) == 2
        assert seen[-1][1] == 2  # total


class TestYfSectorParsing:
    """yf_sector's parsing logic with a fake yfinance module."""

    def _patch_yf(self, monkeypatch, info):
        import types
        fake = types.ModuleType("yfinance")

        class _T:
            def __init__(self, t): ...
            @property
            def info(self_inner):
                if isinstance(info, Exception):
                    raise info
                return info

        fake.Ticker = _T
        monkeypatch.setitem(__import__("sys").modules, "yfinance", fake)

    def test_returns_sector(self, monkeypatch):
        self._patch_yf(monkeypatch, {"sector": "Energy", "a": 1, "b": 2})
        assert sl.yf_sector("RELIANCE.NS") == "Energy"

    def test_missing_sector_returns_none(self, monkeypatch):
        self._patch_yf(monkeypatch, {"a": 1, "b": 2, "c": 3})
        assert sl.yf_sector("X.NS") is None

    def test_empty_info_raises_transient(self, monkeypatch):
        self._patch_yf(monkeypatch, {})
        with pytest.raises(SectorFetchError):
            sl.yf_sector("X.NS")

    def test_exception_wrapped_as_transient(self, monkeypatch):
        self._patch_yf(monkeypatch, RuntimeError("429 Too Many Requests"))
        with pytest.raises(SectorFetchError):
            sl.yf_sector("X.NS")
