"""yfinance implementation of PriceProvider.

Wraps yfinance.download() with the normalization our interface requires:
    - Flatten yfinance's MultiIndex columns
    - Rename Title Case -> snake_case
    - Localize index to Asia/Kolkata
    - Inclusive end-date semantics (yfinance is exclusive)
    - Wrap exceptions as PriceFetchError so the resilient layer can fall back

NOTHING ABOUT YFINANCE LEAKS PAST THIS FILE. The next provider we add
(Stooq, Alpha Vantage, etc.) gets its own class with the same interface;
this one stays untouched.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from loguru import logger

from price_predictor.data.providers.base import PriceFetchError, PriceProvider


class YFinanceProvider(PriceProvider):
    """Concrete provider backed by the yfinance library.

    Stateless -- safe to share a single instance across the whole process.
    """

    @property
    def name(self) -> str:
        return "yfinance"

    def fetch_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        # ── Input validation (caller's fault -> ValueError, no fallback) ──
        if not ticker or not ticker.strip():
            raise ValueError(f"ticker must be a non-empty string, got {ticker!r}")
        if start > end:
            raise ValueError(f"start ({start}) must be <= end ({end})")

        # ── Fetch from yfinance (upstream's fault -> PriceFetchError) ──
        # `end` shifted +1 day so our wrapper is inclusive (yfinance is exclusive).
        # auto_adjust=False preserves both `Close` (raw) and `Adj Close` (adjusted).
        try:
            df = yf.download(
                tickers=ticker,
                start=start,
                end=end + timedelta(days=1),
                interval=interval,
                auto_adjust=False,
                progress=False,
            )
        except Exception as e:
            raise PriceFetchError(
                f"yfinance failed for ticker={ticker!r} "
                f"start={start} end={end}: {e}"
            ) from e

        # ── Empty-result check ──
        if df is None or df.empty:
            raise PriceFetchError(
                f"yfinance returned no data for ticker={ticker!r} "
                f"in range {start}…{end} (interval={interval}). "
                f"Possible causes: delisted ticker, wrong suffix "
                f"(NSE needs '.NS'), weekend-only range, or upstream outage."
            )

        # ── Column normalization ──
        # yfinance >= 0.2 returns MultiIndex columns even for single tickers:
        #   level 0 = field ("Open", "High", ...), level 1 = ticker.
        # Flatten to single-level snake_case for downstream sanity.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        df = df[["open", "high", "low", "close", "adj_close", "volume"]]

        # ── Timezone normalization ──
        # Daily data comes back tz-naive; intraday comes back tz-aware (UTC).
        # Normalize to Asia/Kolkata for consistent as-of-date math.
        if df.index.tz is None:
            df.index = df.index.tz_localize("Asia/Kolkata")
        else:
            df.index = df.index.tz_convert("Asia/Kolkata")

        logger.debug(
            f"yfinance fetched ticker={ticker} rows={len(df)} "
            f"start={start} end={end} interval={interval}"
        )
        return df
