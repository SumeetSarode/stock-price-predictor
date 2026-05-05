"""OHLCV price fetcher -- thin shim over the resilient provider chain.

WHY THIS FILE STILL EXISTS
==========================
The `fetch_ohlcv` function and `PriceFetchError` exception are imported
across the codebase (price_agent, news_impact, tests). Keeping the same
public symbols here -- as a thin delegate to the provider layer -- means
zero churn in callers when we add or swap providers.

Internals: a module-level singleton `_default_fetcher` is built lazily
on first use, reading PRICE_CHAIN / USE_PAID_PRICES from settings. To
add or reorder providers, edit `.env` and restart -- no code change.

WHY TWO CLOSE COLUMNS
=====================
yfinance is called with auto_adjust=False so we get BOTH:
    - close     -- unadjusted (what actually traded; what users see on
                   their broker tomorrow; what target/SL math uses)
    - adj_close -- adjusted for splits/dividends (what indicators like
                   SMA, RSI, MACD, ATR should consume to avoid jumps
                   on splits)

Downstream code MUST pick the right one for its job. This contract is
documented on PriceProvider and enforced by every provider implementation.
Providers without a separate adj_close (Stooq, AV free) mirror close into
adj_close -- imperfect but usable for technical analysis.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
from loguru import logger

from price_predictor.config.settings import settings
from price_predictor.data.providers import (
    PriceFetchError,
    ResilientPriceFetcher,
    build_provider,
)

# Lazy singleton -- built on first call to fetch_ohlcv(). Lazy so that
# tests can patch settings BEFORE the chain is materialized; eager build
# at import would freeze the chain to whatever the settings looked like
# at import time.
_default_fetcher: ResilientPriceFetcher | None = None


def _get_default_fetcher() -> ResilientPriceFetcher:
    """Build (or return cached) ResilientPriceFetcher per current settings."""
    global _default_fetcher
    if _default_fetcher is None:
        chain_names = settings.effective_price_chain()
        providers = [build_provider(name) for name in chain_names]
        _default_fetcher = ResilientPriceFetcher(providers=providers)
        logger.info(
            f"[prices] initialized resilient fetcher with chain={chain_names} "
            f"(use_paid_prices={settings.use_paid_prices})"
        )
    return _default_fetcher


def reset_default_fetcher() -> None:
    """Reset the lazy singleton. Test-only helper -- production code never calls this.

    WHY EXPOSED: tests sometimes need to flip env vars and rebuild the chain;
    making this a public-but-discouraged function is cleaner than monkey-
    patching a private name across test files.
    """
    global _default_fetcher
    _default_fetcher = None


def fetch_ohlcv(
    ticker: str,
    start: date,
    end: date,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV history for a ticker via the configured resilient chain.

    Args:
        ticker:   Provider-native ticker symbol (yfinance: 'RELIANCE.NS').
                  Each provider translates internally to its own format.
        start:    First trading day to include (inclusive).
        end:      Last trading day to include (inclusive).
        interval: yfinance interval string. '1d' (default), '1wk', '1mo', '1h'.
                  Stooq + AlphaVantage support '1d' only -- other intervals
                  trigger their PriceFetchError path so the chain falls
                  back to yfinance.

    Returns:
        DataFrame indexed by tz-aware datetime (Asia/Kolkata), columns:
            open, high, low, close, adj_close, volume

    Raises:
        ValueError:                    Empty/whitespace ticker or start > end.
        PriceFetchError:               All providers in the chain failed.
        AllProvidersExhaustedError:    (subclass of PriceFetchError) -- the
                                       chain was exhausted; check .last_error.
    """
    return _get_default_fetcher().fetch_ohlcv(ticker, start, end, interval)


# Re-export so existing `from price_predictor.data.prices import PriceFetchError`
# imports keep working without churn.
__all__ = ["PriceFetchError", "fetch_ohlcv", "reset_default_fetcher"]
