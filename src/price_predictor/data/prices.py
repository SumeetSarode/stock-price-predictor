"""OHLCV price fetcher -- thin shim over the resilient provider chain.

WHY THIS FILE STILL EXISTS
==========================
The `fetch_ohlcv` function and `PriceFetchError` exception are imported
across the codebase (price_agent, news_impact, tests). Keeping the same
public symbols here -- as a thin delegate to the provider layer -- means
zero churn in callers when we add or swap providers.

Internals: a module-level singleton `_default_fetcher` is built once with
the v1 chain (just YFinanceProvider). Adding Stooq / NSE / Alpha Vantage
later = extend the chain in this one place.

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
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from price_predictor.data.providers import (
    PriceFetchError,
    ResilientPriceFetcher,
    YFinanceProvider,
)

# ── Default fetcher chain ────────────────────────────────────────
# v1: yfinance only. When we add Stooq / NSE / etc., extend this list
# and that's the only change required anywhere in the codebase.
_default_fetcher = ResilientPriceFetcher(providers=[YFinanceProvider()])


def fetch_ohlcv(
    ticker: str,
    start: date,
    end: date,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV history for a ticker via the default resilient chain.

    Args:
        ticker:   Provider-native ticker symbol (yfinance: 'RELIANCE.NS').
        start:    First trading day to include (inclusive).
        end:      Last trading day to include (inclusive).
        interval: yfinance interval string. '1d' (default), '1wk', '1mo', '1h'.

    Returns:
        DataFrame indexed by tz-aware datetime (Asia/Kolkata), columns:
            open, high, low, close, adj_close, volume

    Raises:
        ValueError:                    Empty/whitespace ticker or start > end.
        PriceFetchError:               All providers in the chain failed.
        AllProvidersExhaustedError:    (subclass of PriceFetchError) -- the
                                       chain was exhausted; check .last_error.
    """
    return _default_fetcher.fetch_ohlcv(ticker, start, end, interval)


# Re-export so existing `from price_predictor.data.prices import PriceFetchError`
# imports keep working without churn.
__all__ = ["PriceFetchError", "fetch_ohlcv"]
