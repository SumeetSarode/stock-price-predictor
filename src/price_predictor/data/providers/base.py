"""Base interface every price provider must satisfy.

The contract is intentionally minimal: one method, fixed signature, fixed
return shape. Anything that satisfies this can plug into ResilientPriceFetcher.

The DataFrame contract is part of the interface -- providers MUST normalize
their raw data to match it. That's the whole point of the abstraction:
callers should never need to know which provider served the bars.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class PriceFetchError(Exception):
    """Raised when a provider fails to return usable data.

    Wraps the underlying cause (network error, rate limit, empty response,
    parser failure) so the resilient fetcher can decide whether to fall
    back to the next provider.
    """


class PriceProvider(ABC):
    """The contract every concrete provider implements.

    REQUIRED OUTPUT SHAPE
    =====================
    fetch_ohlcv() must return a DataFrame with:
        - Index: tz-aware DatetimeIndex in 'Asia/Kolkata'
        - Columns: open, high, low, close, adj_close, volume
            (lowercase snake_case; floats; volume as int or float)
        - Rows sorted ascending by date
        - Both 'close' (raw, what traded) and 'adj_close' (split/dividend
          adjusted) -- callers pick the right one for their use case.

    REQUIRED ERROR BEHAVIOR
    =======================
    - Empty/whitespace ticker         -> ValueError (not PriceFetchError)
    - start > end                     -> ValueError
    - Network/API failure             -> PriceFetchError (wrap the cause)
    - Empty result from upstream      -> PriceFetchError with explanation
    - Anything else unexpected        -> PriceFetchError (wrap)

    The split between ValueError (caller's fault) and PriceFetchError
    (upstream's fault) lets the resilient layer decide: ValueErrors
    propagate immediately (no point retrying), PriceFetchErrors trigger
    fallback to the next provider.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable name for logging, e.g. 'yfinance', 'stooq'."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV bars. See class docstring for output + error contract.

        Args:
            ticker:   Provider-native ticker symbol. Each provider documents
                      its own format (yfinance: 'RELIANCE.NS', stooq: 'reliance.in').
            start:    First trading day to include (inclusive).
            end:      Last trading day to include (inclusive).
          interval: Bar size. '1d' is required; '1wk', '1mo', '1h' optional
                      per provider.
        """
