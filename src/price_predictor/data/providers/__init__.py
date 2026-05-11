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

CURRENT REGISTRY
================
    yfinance       -- free, no key, but Yahoo throttles aggressively
    jugaad         -- free, NSE-native, primary tier per pred_logic_solutions C1
    stooq          -- free, no key, daily-only, NO India coverage (legacy)
    alpha_vantage  -- free tier (25/day) or paid (~$50/mo), unreliable for NSE

PROVIDER_REGISTRY maps short names (used in PRICE_CHAIN env var) to
factory callables. The factory pattern lets us pass per-provider config
(e.g., the AlphaVantage API key) without leaking that into Settings.

ADDING A NEW PROVIDER
=====================
1. Implement YourProvider(PriceProvider) in your_provider.py
2. Add `"your_name": lambda: YourProvider(...)` to PROVIDER_REGISTRY below
3. Add the short name to PRICE_CHAIN in .env.example

That's it. Tools, agents, tests don't change.

SHAPE
=====
    PriceProvider           -- abstract base class (the contract)
    YFinanceProvider        -- v1 implementation (Yahoo Finance)
    JugaadDataProvider      -- NSE-native primary tier (jugaad-data)
    StooqProvider           -- legacy: NO India coverage (kept for non-NSE)
    AlphaVantageProvider    -- final fallback / paid-tier option
    ResilientPriceFetcher   -- ordered fallback over multiple providers
    PriceFetchError         -- raised by any provider on a fetch failure
    build_provider          -- factory: short name -> provider instance
"""
from collections.abc import Callable

from price_predictor.config.settings import settings
from price_predictor.data.providers.alpha_vantage_provider import AlphaVantageProvider
from price_predictor.data.providers.base import PriceFetchError, PriceProvider
from price_predictor.data.providers.jugaad_provider import JugaadDataProvider
from price_predictor.data.providers.resilient import ResilientPriceFetcher
from price_predictor.data.providers.stooq_provider import StooqProvider
from price_predictor.data.providers.yfinance_provider import YFinanceProvider

# Map short-name (used in PRICE_CHAIN) to a zero-arg factory that builds
# a configured provider instance. Lambdas (not classes directly) so we can
# inject per-provider config like API keys at construction time.
PROVIDER_REGISTRY: dict[str, Callable[[], PriceProvider]] = {
    "yfinance": YFinanceProvider,
    "jugaad": JugaadDataProvider,
    "stooq": lambda: StooqProvider(
        api_key=settings.stooq_api_key.get_secret_value()
    ),
    "alpha_vantage": lambda: AlphaVantageProvider(
        api_key=settings.alpha_vantage_api_key.get_secret_value()
    ),
}


def build_provider(name: str) -> PriceProvider:
    """Build a provider instance by short name.

    Raises:
        ValueError: name not in PROVIDER_REGISTRY (typo in PRICE_CHAIN env var).
    """
    factory = PROVIDER_REGISTRY.get(name)
    if factory is None:
        raise ValueError(
            f"Unknown price provider {name!r}. "
            f"Registered providers: {sorted(PROVIDER_REGISTRY)}. "
            "Check your PRICE_CHAIN / PRICE_PAID env vars."
        )
    return factory()


__all__ = [
    "PROVIDER_REGISTRY",
    "AlphaVantageProvider",
    "JugaadDataProvider",
    "PriceFetchError",
    "PriceProvider",
    "ResilientPriceFetcher",
    "StooqProvider",
    "YFinanceProvider",
    "build_provider",
]
