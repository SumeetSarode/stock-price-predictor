"""News provider package: pluggable news discovery sources.

GDELT (data/news.py) stays the primary + only historical source. This
package adds live-only fallbacks (currently Google News RSS) that the
snapshot layer uses when GDELT throttles. See base.py for the look-ahead
safety contract that keeps live sources out of backtests.
"""
from __future__ import annotations

from price_predictor.data.news_providers.base import (
    NewsFetchError,
    NewsProvider,
    NewsProviderCoverage,
    empty_news_df,
)
from price_predictor.data.news_providers.google_news_rss import (
    GoogleNewsRssProvider,
)

__all__ = [
    "GoogleNewsRssProvider",
    "NewsFetchError",
    "NewsProvider",
    "NewsProviderCoverage",
    "empty_news_df",
]
