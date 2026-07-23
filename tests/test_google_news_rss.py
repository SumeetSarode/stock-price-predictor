"""Tests for GoogleNewsRssProvider -- parsing, window filtering, errors."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pandas as pd
import pytest

from price_predictor.data.news_providers import (
    GoogleNewsRssProvider,
    NewsFetchError,
)


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _rss(items_xml: str) -> str:
    return (
        '<?xml version="1.0"?><rss version="2.0"><channel>'
        f"{items_xml}</channel></rss>"
    )


def _item(title, link, pubdate, source=None):
    src = f'<source url="http://x">{source}</source>' if source else ""
    return (
        f"<item><title>{title}</title><link>{link}</link>"
        f"<pubDate>{pubdate}</pubDate>{src}</item>"
    )


def _pub(dt: datetime) -> str:
    return dt.strftime("%a, %d %b %Y %H:%M:%S GMT")


@pytest.mark.asyncio
async def test_parses_in_window_items():
    now = datetime.now(UTC)
    recent = now - timedelta(days=1)
    xml = _rss(_item("Reliance up - ET", "https://et/a", _pub(recent), "ET"))

    client = _mock_client(lambda r: httpx.Response(200, text=xml))
    p = GoogleNewsRssProvider()
    df = await p.fetch(
        "Reliance",
        (now - timedelta(days=7)).date().isoformat(),
        now.date().isoformat(),
        client=client,
    )
    await client.aclose()

    assert list(df.columns) == [
        "title", "url", "published_at", "source", "language",
    ]
    assert len(df) == 1
    assert df.iloc[0]["source"] == "ET"
    assert df.iloc[0]["url"] == "https://et/a"
    assert df.iloc[0]["language"] == "eng"
    assert pd.api.types.is_datetime64_any_dtype(df["published_at"])


@pytest.mark.asyncio
async def test_drops_out_of_window_items():
    now = datetime.now(UTC)
    recent = now - timedelta(days=2)
    xml = _rss(
        _item("Fresh - ET", "https://et/fresh", _pub(recent), "ET")
        + _item("Ancient - Mint", "https://m/old",
                "Mon, 07 Jan 2019 10:00:00 GMT", "Mint")
    )
    client = _mock_client(lambda r: httpx.Response(200, text=xml))
    p = GoogleNewsRssProvider()
    df = await p.fetch(
        "Reliance",
        (now - timedelta(days=7)).date().isoformat(),
        now.date().isoformat(),
        client=client,
    )
    await client.aclose()
    assert len(df) == 1
    assert df.iloc[0]["url"] == "https://et/fresh"


@pytest.mark.asyncio
async def test_source_derived_from_title_suffix_when_no_source_el():
    now = datetime.now(UTC)
    recent = now - timedelta(days=1)
    xml = _rss(_item("Big move - Business Standard", "https://bs/a", _pub(recent)))
    client = _mock_client(lambda r: httpx.Response(200, text=xml))
    p = GoogleNewsRssProvider()
    df = await p.fetch(
        "TCS",
        (now - timedelta(days=7)).date().isoformat(),
        now.date().isoformat(),
        client=client,
    )
    await client.aclose()
    assert df.iloc[0]["source"] == "Business Standard"


@pytest.mark.asyncio
async def test_empty_feed_returns_empty_df():
    client = _mock_client(lambda r: httpx.Response(200, text=_rss("")))
    p = GoogleNewsRssProvider()
    df = await p.fetch("Nothing", "2026-07-01", "2026-07-10", client=client)
    await client.aclose()
    assert df.empty
    assert list(df.columns) == [
        "title", "url", "published_at", "source", "language",
    ]


@pytest.mark.asyncio
async def test_http_error_raises_newsfetcherror():
    client = _mock_client(lambda r: httpx.Response(503, text="down"))
    p = GoogleNewsRssProvider()
    with pytest.raises(NewsFetchError):
        await p.fetch("Reliance", "2026-07-01", "2026-07-10", client=client)
    await client.aclose()


@pytest.mark.asyncio
async def test_bad_xml_raises_newsfetcherror():
    client = _mock_client(lambda r: httpx.Response(200, text="<not xml"))
    p = GoogleNewsRssProvider()
    with pytest.raises(NewsFetchError):
        await p.fetch("Reliance", "2026-07-01", "2026-07-10", client=client)
    await client.aclose()


@pytest.mark.asyncio
async def test_empty_query_raises_valueerror():
    p = GoogleNewsRssProvider()
    with pytest.raises(ValueError):
        await p.fetch("   ", "2026-07-01", "2026-07-10")


@pytest.mark.asyncio
async def test_exact_phrase_wraps_query_in_quotes():
    captured = {}

    def handler(request):
        captured["q"] = dict(request.url.params).get("q")
        return httpx.Response(200, text=_rss(""))

    client = _mock_client(handler)
    p = GoogleNewsRssProvider()
    await p.fetch("Reliance Industries", "2026-07-01", "2026-07-10",
                  exact_phrase=True, client=client)
    await client.aclose()
    assert captured["q"] == '"Reliance Industries"'


@pytest.mark.asyncio
async def test_coverage_is_live_only():
    p = GoogleNewsRssProvider(freshness_days=45)
    assert p.name == "google_news_rss"
    assert p.coverage.historical is False
    assert p.coverage.freshness_days == 45
