"""Price provider package -- pluggable data sources behind a common interface.

WHY THIS EXISTS
===============
yfinance is our v1 data source. It works, but it's community-maintained and
has known quirks (rate limits, occasional library breakage, silent empty
returns). To insulate the rest of the codebase from any one source, all
price-fetching code lives behind a `PriceProvider` interface.

A `ResilientPriceFetcher` wraps an ordered list of providers and falls back
on transient failures (rate limit, network blip, empty result). This is the
same pattern as `ResilientModel` for LLMs -- different domain, same shape.

Today the chain is just [YFinanceProvider]. Adding Stooq or NSE direct is
"write a new class + add to chain" -- callers don't change.

SHAPE
=====
    PriceProvider           -- abstract base class (the contract)
    YFinanceProvider        -- v1 implementation
    ResilientPriceFetcher   -- ordered fallback over multiple providers
    PriceFetchError         -- raised by any provider on a fetch failure
"""
from price_predictor.data.providers.base import PriceFetchError, PriceProvider
from price_predictor.data.providers.resilient import ResilientPriceFetcher
from price_predictor.data.providers.yfinance_provider import YFinanceProvider

__all__ = [
    "PriceFetchError",
    "PriceProvider",
    "ResilientPriceFetcher",
    "YFinanceProvider",
]
