"""Tests for price_predictor.data.news.

Mocking strategy:
    - Network calls are mocked at the httpx layer using `respx`
    - trafilatura is patched directly in body-extraction tests
    - No real HTTP except the integration tests (marked @pytest.mark.integration)

Test layout:
    - Validation (no network)
    - Date conversion helpers
    - Article normalization
    - fetch_news (mocked)
    - fetch_news_batch (mocked)
    - fetch_article_body (mocked)
    - Schema round-trip
    - Integration (real GDELT, marked)
"""
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import httpx
import pytest
import respx

from price_predictor.data.news import (
    GDELT_DOC_URL,
    NewsFetchError,
    _build_params,
    _gdelt_keyword,
    _iter_windows,
    _normalize_articles,
    _parse_seendate,
    _to_gdelt_datetime,
    _validate_inputs,
    fetch_article_body,
    fetch_news,
    fetch_news_batch,
    fetch_news_paginated,
    fetch_news_relevant,
)
from price_predictor.data.schema import ArticleBody, NewsArticle


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _gdelt_response(articles: list[dict]) -> dict:
    """Build a fake GDELT JSON response."""
    return {"articles": articles}


def _article(
    title: str = "Reliance Q3 Results Beat Estimates",
    url: str = "https://reuters.com/article/reliance-q3",
    seendate: str = "20240115T103000Z",
    domain: str = "reuters.com",
    language: str = "English",
) -> dict:
    """Build a fake GDELT article dict."""
    return {
        "title": title,
        "url": url,
        "seendate": seendate,
        "domain": domain,
        "language": language,
    }


# ─────────────────────────────────────────────────────────────
# _validate_inputs
# ─────────────────────────────────────────────────────────────
class TestValidateInputs:
    def test_happy_path(self):
        s, e = _validate_inputs("Reliance", "2024-01-01", "2024-01-31")
        assert s == datetime(2024, 1, 1, tzinfo=UTC)
        assert e == datetime(2024, 1, 31, tzinfo=UTC)

    @pytest.mark.parametrize("bad_query", ["", "   ", None, 123, []])
    def test_bad_query(self, bad_query):
        with pytest.raises(ValueError, match="non-empty string"):
            _validate_inputs(bad_query, "2024-01-01", "2024-01-31")

    @pytest.mark.parametrize(
        "bad_date",
        ["01-01-2024", "2024/01/01", "Jan 1 2024", "20240101", "not-a-date"],
    )
    def test_bad_date_format(self, bad_date):
        with pytest.raises(ValueError, match="Invalid date format"):
            _validate_inputs("Reliance", bad_date, "2024-01-31")

    def test_start_after_end(self):
        with pytest.raises(ValueError, match=r"start.*must be <= end"):
            _validate_inputs("Reliance", "2024-02-01", "2024-01-01")


# ─────────────────────────────────────────────────────────────
# _to_gdelt_datetime
# ─────────────────────────────────────────────────────────────
class TestToGdeltDatetime:
    def test_start_of_day(self):
        dt = datetime(2024, 1, 15, tzinfo=UTC)
        assert _to_gdelt_datetime(dt, end_of_day=False) == "20240115000000"

    def test_end_of_day(self):
        dt = datetime(2024, 1, 15, tzinfo=UTC)
        assert _to_gdelt_datetime(dt, end_of_day=True) == "20240115235959"


# ─────────────────────────────────────────────────────────────
# _build_params
# ─────────────────────────────────────────────────────────────
class TestGdeltKeyword:
    def test_long_name_exact_is_quoted(self):
        assert _gdelt_keyword("Reliance Industries", exact_phrase=True) == '"Reliance Industries"'

    def test_long_name_loose_is_bare(self):
        assert _gdelt_keyword("Reliance Industries", exact_phrase=False) == "Reliance Industries"

    def test_short_name_keeps_bare_name_matchable_via_or_group(self):
        # PROVEN: GDELT rejects any sub-floor term even inside an OR, so the
        # bare 3-char name must NOT appear alone. We OR finance-qualified
        # phrases that each clear the floor.
        kw = _gdelt_keyword("ITC", exact_phrase=True)
        assert kw.startswith('("ITC Limited" OR ')
        assert '"ITC shares"' in kw
        assert '"ITC"' not in kw  # bare sub-floor term must never be emitted

    def test_short_name_or_group_regardless_of_tier(self):
        # Short names need qualified phrases on BOTH tiers (bare name is
        # unusable), so the OR group is used even when exact_phrase is False.
        kw = _gdelt_keyword("MRF", exact_phrase=False)
        assert kw.startswith('("MRF Limited" OR ')
        assert '"MRF stock"' in kw

    def test_whitespace_trimmed(self):
        assert _gdelt_keyword("  ITC  ", exact_phrase=True).startswith('("ITC Limited" OR ')

    def test_short_name_already_suffixed_left_alone(self):
        assert _gdelt_keyword("X Ltd", exact_phrase=True) == '"X Ltd"'


class TestBuildParams:
    def test_exact_phrase_quotes_query_by_default(self):
        params = _build_params(
            "Reliance Industries",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 31, tzinfo=UTC),
            lang="eng",
            max_records=250,
        )
        assert params["query"] == '"Reliance Industries" sourcelang:eng'
        assert params["mode"] == "ArtList"
        assert params["format"] == "json"
        assert params["maxrecords"] == "250"
        assert params["startdatetime"] == "20240101000000"
        assert params["enddatetime"] == "20240131235959"

    def test_exact_phrase_off_leaves_tokens_loose(self):
        params = _build_params(
            "Reliance", datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 31, tzinfo=UTC), lang="eng", max_records=250,
            exact_phrase=False,
        )
        assert params["query"] == "Reliance sourcelang:eng"

    def test_source_country_appended(self):
        params = _build_params(
            "Infosys", datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 31, tzinfo=UTC), lang="eng", max_records=250,
            source_country="IN",
        )
        assert params["query"] == '"Infosys" sourcelang:eng sourcecountry:IN'

    def test_short_name_is_padded_to_clear_gdelt_floor(self):
        """The ITC bug: emit qualified phrases, never the bare sub-floor name."""
        params = _build_params(
            "ITC", datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 31, tzinfo=UTC), lang="eng", max_records=250,
            source_country="IN",
        )
        assert params["query"].startswith('("ITC Limited" OR ')
        assert params["query"].endswith("sourcelang:eng sourcecountry:IN")
        assert '"ITC"' not in params["query"]

    def test_short_name_padded_even_when_loose(self):
        params = _build_params(
            "MRF", datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 31, tzinfo=UTC), lang="eng", max_records=250,
            exact_phrase=False,
        )
        assert params["query"].startswith('("MRF Limited" OR ')
        assert params["query"].endswith("sourcelang:eng")


# ─────────────────────────────────────────────────────────────
# fetch_news_relevant (relevance ladder)
# ─────────────────────────────────────────────────────────────
class TestFetchNewsRelevant:
    @pytest.mark.asyncio
    async def test_returns_first_tier_when_it_has_results(self, monkeypatch):
        import price_predictor.data.news as news_mod
        calls = []

        async def fake(query, start, end, *, exact_phrase, source_country, **kw):
            calls.append((exact_phrase, source_country))
            return _normalize_articles([_article()])  # non-empty

        monkeypatch.setattr(news_mod, "fetch_news", fake)
        df = await fetch_news_relevant("Infosys", "2024-01-01", "2024-01-07")
        assert len(df) == 1
        # Stopped after the strictest tier (exact phrase + IN).
        assert calls == [(True, "IN")]

    @pytest.mark.asyncio
    async def test_relaxes_when_earlier_tiers_empty(self, monkeypatch):
        import price_predictor.data.news as news_mod
        calls = []

        async def fake(query, start, end, *, exact_phrase, source_country, **kw):
            calls.append((exact_phrase, source_country))
            # Only the loosest tier (no quotes, no country) returns anything.
            if not exact_phrase and source_country is None:
                return _normalize_articles([_article()])
            return _normalize_articles([])

        monkeypatch.setattr(news_mod, "fetch_news", fake)
        df = await fetch_news_relevant("Infosys", "2024-01-01", "2024-01-07")
        assert len(df) == 1
        assert calls == [(True, "IN"), (True, None), (False, None)]

    @pytest.mark.asyncio
    async def test_all_tiers_empty_returns_empty(self, monkeypatch):
        import price_predictor.data.news as news_mod

        async def fake(query, start, end, *, exact_phrase, source_country, **kw):
            return _normalize_articles([])

        monkeypatch.setattr(news_mod, "fetch_news", fake)
        df = await fetch_news_relevant("Nothing", "2024-01-01", "2024-01-07")
        assert df.empty

    @pytest.mark.asyncio
    async def test_fetch_error_propagates(self, monkeypatch):
        import price_predictor.data.news as news_mod

        async def fake(query, start, end, *, exact_phrase, source_country, **kw):
            raise NewsFetchError("GDELT down")

        monkeypatch.setattr(news_mod, "fetch_news", fake)
        with pytest.raises(NewsFetchError):
            await fetch_news_relevant("Infosys", "2024-01-01", "2024-01-07")


# ─────────────────────────────────────────────────────────────
# _parse_seendate
# ─────────────────────────────────────────────────────────────
class TestParseSeendate:
    def test_happy_path(self):
        result = _parse_seendate("20240115T103000Z")
        assert result == datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        assert result.tzinfo is not None

    def test_bad_format_raises(self):
        with pytest.raises(ValueError):
            _parse_seendate("2024-01-15 10:30:00")


# ─────────────────────────────────────────────────────────────
# _normalize_articles
# ─────────────────────────────────────────────────────────────
class TestNormalizeArticles:
    def test_happy_path(self):
        df = _normalize_articles([_article(), _article(title="Another")])
        assert len(df) == 2
        assert list(df.columns) == ["title", "url", "published_at", "source", "language"]
        assert df.iloc[0]["title"] == "Reliance Q3 Results Beat Estimates"
        assert df.iloc[0]["source"] == "reuters.com"

    def test_empty_input(self):
        df = _normalize_articles([])
        assert len(df) == 0
        assert list(df.columns) == ["title", "url", "published_at", "source", "language"]

    def test_skips_articles_missing_required_fields(self):
        articles = [
            _article(),                                  # ok
            {"title": "no url"},                         # missing url, seendate
            {"url": "https://x.com", "title": "no seendate"},  # missing seendate
            _article(title="ok2"),                       # ok
        ]
        df = _normalize_articles(articles)
        assert len(df) == 2
        assert df.iloc[1]["title"] == "ok2"

    def test_skips_articles_with_bad_seendate(self):
        articles = [
            _article(),
            _article(title="bad", seendate="not-a-date"),
            _article(title="good"),
        ]
        df = _normalize_articles(articles)
        assert len(df) == 2
        assert "bad" not in df["title"].tolist()


# ─────────────────────────────────────────────────────────────
# fetch_news (mocked HTTP)
# ─────────────────────────────────────────────────────────────
class TestFetchNews:
    @respx.mock
    async def test_happy_path(self):
        respx.get(GDELT_DOC_URL).mock(
            return_value=httpx.Response(200, json=_gdelt_response([_article(), _article()]))
        )
        df = await fetch_news("Reliance", "2024-01-01", "2024-01-31")
        assert len(df) == 2
        assert list(df.columns) == ["title", "url", "published_at", "source", "language"]

    @respx.mock
    async def test_empty_results_returns_empty_df_not_error(self):
        """Empty article list = success-with-0-rows, NOT an error."""
        respx.get(GDELT_DOC_URL).mock(
            return_value=httpx.Response(200, json=_gdelt_response([]))
        )
        df = await fetch_news("ZZZ-no-such-stock", "2024-01-01", "2024-01-31")
        assert len(df) == 0
        assert list(df.columns) == ["title", "url", "published_at", "source", "language"]

    @respx.mock
    async def test_http_500_raises(self):
        respx.get(GDELT_DOC_URL).mock(return_value=httpx.Response(500))
        with pytest.raises(NewsFetchError, match="GDELT request failed"):
            await fetch_news("Reliance", "2024-01-01", "2024-01-31")

    @respx.mock
    async def test_429_then_success_retries(self, monkeypatch):
        """A transient 429 (GDELT rate-limit) is retried, not surfaced."""
        import price_predictor.data.news as news_mod

        monkeypatch.setattr(news_mod, "GDELT_RATE_LIMIT_BACKOFF_S", 0.0)
        route = respx.get(GDELT_DOC_URL)
        route.side_effect = [
            httpx.Response(429),
            httpx.Response(200, json=_gdelt_response([_article()])),
        ]
        df = await fetch_news("Reliance", "2024-01-01", "2024-01-31")
        assert len(df) == 1
        assert route.call_count == 2  # retried once

    @respx.mock
    async def test_429_exhausts_retries_then_raises(self, monkeypatch):
        """Persistent 429 raises after RETRIES+1 attempts (bounded, not infinite)."""
        import price_predictor.data.news as news_mod

        monkeypatch.setattr(news_mod, "GDELT_RATE_LIMIT_BACKOFF_S", 0.0)
        route = respx.get(GDELT_DOC_URL)
        route.side_effect = [httpx.Response(429)] * 5
        with pytest.raises(NewsFetchError, match="GDELT request failed"):
            await fetch_news("Reliance", "2024-01-01", "2024-01-31")
        assert route.call_count == news_mod.GDELT_RATE_LIMIT_RETRIES + 1

    @respx.mock
    async def test_non_429_http_error_is_not_retried(self, monkeypatch):
        """A 500 fails fast -- only 429 is retryable."""
        import price_predictor.data.news as news_mod

        monkeypatch.setattr(news_mod, "GDELT_RATE_LIMIT_BACKOFF_S", 0.0)
        route = respx.get(GDELT_DOC_URL)
        route.side_effect = [httpx.Response(500), httpx.Response(500)]
        with pytest.raises(NewsFetchError, match="GDELT request failed"):
            await fetch_news("Reliance", "2024-01-01", "2024-01-31")
        assert route.call_count == 1  # no retry on non-429

    @respx.mock
    async def test_connect_timeout_then_success_retries(self, monkeypatch):
        """A transient ConnectTimeout (network blip) is retried, not surfaced."""
        import price_predictor.data.news as news_mod

        monkeypatch.setattr(news_mod, "GDELT_NETWORK_BACKOFF_S", 0.0)
        route = respx.get(GDELT_DOC_URL)
        route.side_effect = [
            httpx.ConnectTimeout("connect timed out"),
            httpx.Response(200, json=_gdelt_response([_article()])),
        ]
        df = await fetch_news("Reliance", "2024-01-01", "2024-01-31")
        assert len(df) == 1
        assert route.call_count == 2  # retried once

    @respx.mock
    async def test_connect_timeout_exhausts_retries_then_raises(self, monkeypatch):
        """Persistent network faults raise after NETWORK_RETRIES+1 attempts."""
        import price_predictor.data.news as news_mod

        monkeypatch.setattr(news_mod, "GDELT_NETWORK_BACKOFF_S", 0.0)
        route = respx.get(GDELT_DOC_URL)
        route.side_effect = [httpx.ConnectTimeout("nope")] * 5
        with pytest.raises(NewsFetchError, match="GDELT request failed"):
            await fetch_news("Reliance", "2024-01-01", "2024-01-31")
        assert route.call_count == news_mod.GDELT_NETWORK_RETRIES + 1

    @respx.mock
    async def test_timeout_raises(self):
        respx.get(GDELT_DOC_URL).mock(side_effect=httpx.TimeoutException("timeout"))
        with pytest.raises(NewsFetchError, match="GDELT request failed"):
            await fetch_news("Reliance", "2024-01-01", "2024-01-31")

    @respx.mock
    async def test_malformed_json_raises(self):
        respx.get(GDELT_DOC_URL).mock(
            return_value=httpx.Response(200, text="<html>not json</html>")
        )
        with pytest.raises(NewsFetchError, match="non-JSON"):
            await fetch_news("Reliance", "2024-01-01", "2024-01-31")

    @respx.mock
    async def test_articles_not_a_list_raises(self):
        respx.get(GDELT_DOC_URL).mock(
            return_value=httpx.Response(200, json={"articles": "oops"})
        )
        with pytest.raises(NewsFetchError, match="not a list"):
            await fetch_news("Reliance", "2024-01-01", "2024-01-31")

    async def test_validation_errors_no_network(self):
        """Bad inputs raise ValueError without making any HTTP call."""
        with respx.mock(assert_all_called=False) as mock:
            mock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json={}))
            with pytest.raises(ValueError):
                await fetch_news("", "2024-01-01", "2024-01-31")
            with pytest.raises(ValueError):
                await fetch_news("X", "bad-date", "2024-01-31")
            assert mock.calls.call_count == 0


# ─────────────────────────────────────────────────────────────
# fetch_news_batch (mocked HTTP)
# ─────────────────────────────────────────────────────────────
class TestFetchNewsBatch:
    @respx.mock
    async def test_happy_path(self):
        respx.get(GDELT_DOC_URL).mock(
            return_value=httpx.Response(200, json=_gdelt_response([_article()]))
        )
        results = await fetch_news_batch(
            ["Reliance", "TCS"], "2024-01-01", "2024-01-31", concurrency=2
        )
        assert set(results.keys()) == {"Reliance", "TCS"}
        for df in results.values():
            assert len(df) == 1

    @respx.mock
    async def test_one_query_fails_others_succeed(self):
        """A failed query lands as an Exception in the dict; others still succeed."""
        # First call fails, second succeeds — respx replays mocks in order.
        route = respx.get(GDELT_DOC_URL)
        route.side_effect = [
            httpx.Response(500),
            httpx.Response(200, json=_gdelt_response([_article()])),
        ]
        results = await fetch_news_batch(
            ["BadQuery", "GoodQuery"], "2024-01-01", "2024-01-31", concurrency=1
        )
        # One must be exception, one must be DataFrame
        statuses = {k: type(v).__name__ for k, v in results.items()}
        assert "NewsFetchError" in statuses.values()
        assert "DataFrame" in statuses.values()

    async def test_empty_list(self):
        results = await fetch_news_batch([], "2024-01-01", "2024-01-31")
        assert results == {}


# ─────────────────────────────────────────────────────────────
# _iter_windows
# ─────────────────────────────────────────────────────────────
class TestIterWindows:
    def test_single_day_one_window(self):
        s = datetime(2024, 1, 1, tzinfo=UTC)
        e = datetime(2024, 1, 1, tzinfo=UTC)
        assert _iter_windows(s, e, window_days=1) == [(s, e)]

    def test_three_days_window_one(self):
        s = datetime(2024, 1, 1, tzinfo=UTC)
        e = datetime(2024, 1, 3, tzinfo=UTC)
        out = _iter_windows(s, e, window_days=1)
        assert out == [
            (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)),
            (datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)),
            (datetime(2024, 1, 3, tzinfo=UTC), datetime(2024, 1, 3, tzinfo=UTC)),
        ]

    def test_window_wider_than_range_one_window(self):
        s = datetime(2024, 1, 1, tzinfo=UTC)
        e = datetime(2024, 1, 3, tzinfo=UTC)
        out = _iter_windows(s, e, window_days=30)
        assert out == [(s, e)]

    def test_partial_last_window_clipped(self):
        """7 days with window=3 -> [1..3], [4..6], [7..7]."""
        s = datetime(2024, 1, 1, tzinfo=UTC)
        e = datetime(2024, 1, 7, tzinfo=UTC)
        out = _iter_windows(s, e, window_days=3)
        assert out == [
            (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 3, tzinfo=UTC)),
            (datetime(2024, 1, 4, tzinfo=UTC), datetime(2024, 1, 6, tzinfo=UTC)),
            (datetime(2024, 1, 7, tzinfo=UTC), datetime(2024, 1, 7, tzinfo=UTC)),
        ]

    def test_exact_multiple(self):
        """6 days with window=3 -> [1..3], [4..6] (no partial tail)."""
        s = datetime(2024, 1, 1, tzinfo=UTC)
        e = datetime(2024, 1, 6, tzinfo=UTC)
        out = _iter_windows(s, e, window_days=3)
        assert out == [
            (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 3, tzinfo=UTC)),
            (datetime(2024, 1, 4, tzinfo=UTC), datetime(2024, 1, 6, tzinfo=UTC)),
        ]

    def test_windows_are_non_overlapping_and_contiguous(self):
        """Property test: every day in [start, end] is in exactly one window."""
        s = datetime(2024, 1, 1, tzinfo=UTC)
        e = datetime(2024, 1, 31, tzinfo=UTC)
        for w in (1, 2, 3, 5, 7, 10, 30, 100):
            windows = _iter_windows(s, e, window_days=w)
            # Reconstruct day set; should equal [s..e] inclusive, no dupes.
            seen: set[datetime] = set()
            for ws, we in windows:
                assert ws <= we
                day = ws
                while day <= we:
                    assert day not in seen, f"day {day} in two windows for w={w}"
                    seen.add(day)
                    day += timedelta(days=1)
            expected = {s + timedelta(days=i) for i in range((e - s).days + 1)}
            assert seen == expected, f"coverage gap for window_days={w}"


# ─────────────────────────────────────────────────────────────
# fetch_news_paginated (mocked HTTP)
# ─────────────────────────────────────────────────────────────
class TestFetchNewsPaginated:
    @respx.mock
    async def test_single_window_short_circuits(self):
        """Range fits inside one window → exactly one HTTP call, no sleep."""
        route = respx.get(GDELT_DOC_URL).mock(
            return_value=httpx.Response(200, json=_gdelt_response([_article()]))
        )
        df = await fetch_news_paginated(
            "Reliance",
            "2024-01-01",
            "2024-01-01",
            window_days=1,
            polite_sleep_s=0.0,
        )
        assert route.call_count == 1
        assert len(df) == 1

    @respx.mock
    async def test_multi_window_concatenates(self):
        """3-day range with window=1 → 3 calls, results concatenated."""
        route = respx.get(GDELT_DOC_URL)
        route.side_effect = [
            httpx.Response(200, json=_gdelt_response([
                _article(title="day1", url="https://x.com/1", seendate="20240101T100000Z"),
            ])),
            httpx.Response(200, json=_gdelt_response([
                _article(title="day2", url="https://x.com/2", seendate="20240102T100000Z"),
            ])),
            httpx.Response(200, json=_gdelt_response([
                _article(title="day3", url="https://x.com/3", seendate="20240103T100000Z"),
            ])),
        ]
        df = await fetch_news_paginated(
            "Reliance",
            "2024-01-01",
            "2024-01-03",
            window_days=1,
            polite_sleep_s=0.0,
        )
        assert route.call_count == 3
        assert len(df) == 3
        # Sorted ascending by published_at.
        assert df["title"].tolist() == ["day1", "day2", "day3"]

    @respx.mock
    async def test_dedupe_across_windows(self):
        """Same URL appearing in two windows is collapsed (first kept)."""
        route = respx.get(GDELT_DOC_URL)
        route.side_effect = [
            httpx.Response(200, json=_gdelt_response([
                _article(title="first", url="https://x.com/dup", seendate="20240101T100000Z"),
            ])),
            httpx.Response(200, json=_gdelt_response([
                _article(title="dup-later", url="https://x.com/dup", seendate="20240102T100000Z"),
                _article(title="unique", url="https://x.com/u", seendate="20240102T110000Z"),
            ])),
        ]
        df = await fetch_news_paginated(
            "Reliance",
            "2024-01-01",
            "2024-01-02",
            window_days=1,
            polite_sleep_s=0.0,
        )
        assert len(df) == 2
        # First occurrence kept (oldest seendate)
        assert "first" in df["title"].tolist()
        assert "dup-later" not in df["title"].tolist()

    @respx.mock
    async def test_dedupe_disabled_keeps_duplicates(self):
        route = respx.get(GDELT_DOC_URL)
        route.side_effect = [
            httpx.Response(200, json=_gdelt_response([
                _article(url="https://x.com/dup", seendate="20240101T100000Z"),
            ])),
            httpx.Response(200, json=_gdelt_response([
                _article(url="https://x.com/dup", seendate="20240102T100000Z"),
            ])),
        ]
        df = await fetch_news_paginated(
            "Reliance",
            "2024-01-01",
            "2024-01-02",
            window_days=1,
            polite_sleep_s=0.0,
            dedupe=False,
        )
        assert len(df) == 2

    @respx.mock
    async def test_all_empty_returns_empty_df(self):
        """All windows empty → empty DataFrame, never an error."""
        respx.get(GDELT_DOC_URL).mock(
            return_value=httpx.Response(200, json=_gdelt_response([]))
        )
        df = await fetch_news_paginated(
            "NoSuchStock",
            "2024-01-01",
            "2024-01-03",
            window_days=1,
            polite_sleep_s=0.0,
        )
        assert len(df) == 0
        assert list(df.columns) == ["title", "url", "published_at", "source", "language"]

    @respx.mock
    async def test_failure_in_any_window_raises_fail_fast(self):
        """Window 2 fails → entire call raises (no half-fetched DataFrame)."""
        route = respx.get(GDELT_DOC_URL)
        route.side_effect = [
            httpx.Response(200, json=_gdelt_response([_article()])),
            httpx.Response(500),
        ]
        with pytest.raises(NewsFetchError, match="GDELT request failed"):
            await fetch_news_paginated(
                "Reliance",
                "2024-01-01",
                "2024-01-02",
                window_days=1,
                polite_sleep_s=0.0,
            )

    @respx.mock
    async def test_polite_sleep_called_between_but_not_after_last(self):
        """3 windows → sleep called exactly 2 times (between, not trailing)."""
        respx.get(GDELT_DOC_URL).mock(
            return_value=httpx.Response(200, json=_gdelt_response([_article()]))
        )
        with patch(
            "price_predictor.data.news.asyncio.sleep",
            new_callable=lambda: __import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock(),
        ) as mock_sleep:
            await fetch_news_paginated(
                "Reliance",
                "2024-01-01",
                "2024-01-03",
                window_days=1,
                polite_sleep_s=5.0,
            )
        assert mock_sleep.call_count == 2
        for c in mock_sleep.call_args_list:
            assert c.args == (5.0,)

    @respx.mock
    async def test_polite_sleep_zero_skips_sleep(self):
        """polite_sleep_s=0 → asyncio.sleep never called (fast tests)."""
        respx.get(GDELT_DOC_URL).mock(
            return_value=httpx.Response(200, json=_gdelt_response([_article()]))
        )
        with patch(
            "price_predictor.data.news.asyncio.sleep",
            new_callable=lambda: __import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock(),
        ) as mock_sleep:
            await fetch_news_paginated(
                "Reliance",
                "2024-01-01",
                "2024-01-03",
                window_days=1,
                polite_sleep_s=0.0,
            )
        assert mock_sleep.call_count == 0

    async def test_validation_bad_query_no_network(self):
        with respx.mock(assert_all_called=False) as mock:
            mock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json={}))
            with pytest.raises(ValueError, match="non-empty string"):
                await fetch_news_paginated("", "2024-01-01", "2024-01-03")
            assert mock.calls.call_count == 0

    async def test_validation_bad_dates_no_network(self):
        with respx.mock(assert_all_called=False) as mock:
            mock.get(GDELT_DOC_URL).mock(return_value=httpx.Response(200, json={}))
            with pytest.raises(ValueError, match="Invalid date format"):
                await fetch_news_paginated("X", "bad-date", "2024-01-03")
            assert mock.calls.call_count == 0

    async def test_validation_bad_window_days(self):
        with pytest.raises(ValueError, match="window_days must be >= 1"):
            await fetch_news_paginated(
                "X", "2024-01-01", "2024-01-03", window_days=0
            )
        with pytest.raises(ValueError, match="window_days must be >= 1"):
            await fetch_news_paginated(
                "X", "2024-01-01", "2024-01-03", window_days=-1
            )

    async def test_validation_bad_polite_sleep(self):
        with pytest.raises(ValueError, match="polite_sleep_s must be >= 0"):
            await fetch_news_paginated(
                "X", "2024-01-01", "2024-01-03", polite_sleep_s=-0.1
            )

    @respx.mock
    async def test_uses_shared_client_when_provided(self):
        """When caller passes a client, paginator MUST NOT close it."""
        respx.get(GDELT_DOC_URL).mock(
            return_value=httpx.Response(200, json=_gdelt_response([_article()]))
        )
        async with httpx.AsyncClient() as client:
            await fetch_news_paginated(
                "Reliance",
                "2024-01-01",
                "2024-01-02",
                window_days=1,
                polite_sleep_s=0.0,
                client=client,
            )
            # Client still usable after paginated call.
            assert not client.is_closed

    @respx.mock
    async def test_window_days_wider_than_range_makes_one_call(self):
        route = respx.get(GDELT_DOC_URL).mock(
            return_value=httpx.Response(200, json=_gdelt_response([_article()]))
        )
        await fetch_news_paginated(
            "Reliance",
            "2024-01-01",
            "2024-01-03",
            window_days=30,
            polite_sleep_s=0.0,
        )
        assert route.call_count == 1

# ─────────────────────────────────────────────────────────────
# fetch_article_body (mocked HTTP + trafilatura)
# ─────────────────────────────────────────────────────────────
class TestFetchArticleBody:
    @respx.mock
    async def test_happy_path(self):
        respx.get("https://example.com/article").mock(
            return_value=httpx.Response(200, text="<html><body>Hi</body></html>")
        )
        with patch(
            "price_predictor.data.news.trafilatura.extract",
            return_value="Clean article body text here.",
        ):
            result = await fetch_article_body("https://example.com/article")
        assert result.status == "success"
        assert result.body == "Clean article body text here."
        assert result.error_message is None

    @respx.mock
    async def test_http_403_returns_error_does_not_raise(self):
        respx.get("https://paywalled.com/article").mock(return_value=httpx.Response(403))
        result = await fetch_article_body("https://paywalled.com/article")
        assert result.status == "error"
        assert "HTTP fetch failed" in result.error_message
        assert result.body == ""

    @respx.mock
    async def test_timeout_returns_error_does_not_raise(self):
        respx.get("https://slow.com/article").mock(
            side_effect=httpx.TimeoutException("slow")
        )
        result = await fetch_article_body("https://slow.com/article")
        assert result.status == "error"
        assert "HTTP fetch failed" in result.error_message

    @respx.mock
    async def test_trafilatura_returns_none_returns_error(self):
        """Empty extraction → status=error (NOT success-with-empty-body).

        This is the contract that prevents callers from confusing
        'extraction failed' with 'article was genuinely empty'.
        """
        respx.get("https://js-only.com/article").mock(
            return_value=httpx.Response(200, text="<html></html>")
        )
        with patch("price_predictor.data.news.trafilatura.extract", return_value=None):
            result = await fetch_article_body("https://js-only.com/article")
        assert result.status == "error"
        assert "no content" in result.error_message
        assert result.body == ""

    @respx.mock
    async def test_trafilatura_raises_returns_error(self):
        respx.get("https://broken.com/article").mock(
            return_value=httpx.Response(200, text="<html>x</html>")
        )
        with patch(
            "price_predictor.data.news.trafilatura.extract",
            side_effect=RuntimeError("trafilatura blew up"),
        ):
            result = await fetch_article_body("https://broken.com/article")
        assert result.status == "error"
        assert "trafilatura extract raised" in result.error_message

    async def test_invalid_url(self):
        result = await fetch_article_body("")
        assert result.status == "error"
        assert "Invalid URL" in result.error_message


# ─────────────────────────────────────────────────────────────
# Schema round-trip
# ─────────────────────────────────────────────────────────────
class TestNewsArticleSchema:
    def test_roundtrip(self):
        original = NewsArticle(
            title="Reliance Q3",
            url="https://reuters.com/article",
            published_at=datetime(2024, 1, 15, 10, 30, tzinfo=UTC),
            source="reuters.com",
            language="eng",
        )
        as_json = original.model_dump_json()
        restored = NewsArticle.model_validate_json(as_json)
        assert restored.title == original.title
        assert str(restored.url) == str(original.url)
        assert restored.published_at == original.published_at

    def test_article_body_success_roundtrip(self):
        original = ArticleBody(status="success", body="Some body text")
        restored = ArticleBody.model_validate_json(original.model_dump_json())
        assert restored.status == "success"
        assert restored.body == "Some body text"

    def test_article_body_error_roundtrip(self):
        original = ArticleBody(status="error", error_message="HTTP 500")
        restored = ArticleBody.model_validate_json(original.model_dump_json())
        assert restored.status == "error"
        assert restored.error_message == "HTTP 500"


# ─────────────────────────────────────────────────────────────
# Integration tests (real network — marked, skipped by default)
# ─────────────────────────────────────────────────────────────
@pytest.mark.integration
class TestIntegrationGdelt:
    async def test_real_gdelt_fetch_returns_articles(self):
        """Real GDELT call for a high-volume Indian stock — should return ≥1 article.

        Uses a 7-day window ending yesterday (GDELT free tier = 7-day rolling).
        Skips gracefully if GDELT is unreachable (corporate proxy / DNS / outage).
        """
        from datetime import timedelta
        end = datetime.now(UTC).date() - timedelta(days=1)
        start = end - timedelta(days=6)

        try:
            df = await fetch_news(
                "Reliance Industries",
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
                max_records=50,
            )
        except NewsFetchError as e:
            # Network/DNS/proxy issues are environmental, not code bugs.
            # Real GDELT outages are rare; corporate-proxy DNS failures are common.
            pytest.skip(f"GDELT unreachable (likely network/proxy): {e}")

        # Reliance is heavily covered — should have at least 1 article in any 7-day window
        assert len(df) >= 1, f"Expected ≥1 article, got {len(df)}"
        assert all(col in df.columns for col in ["title", "url", "published_at", "source"])


@pytest.mark.integration
class TestIntegrationBodyFetch:
    async def test_real_article_body_fetch(self):
        """Real body extraction from a stable, public article.

        Uses example.com which is guaranteed-stable. trafilatura may not extract
        much from it (it's tiny), so we accept either success-with-body or a
        clean error — what matters is that we don't crash.
        """
        result = await fetch_article_body("https://example.com/")
        # Either it worked (status=success) or it failed cleanly (status=error)
        assert result.status in ("success", "error")
        if result.status == "success":
            assert result.body  # non-empty
        else:
            assert result.error_message  # non-empty
