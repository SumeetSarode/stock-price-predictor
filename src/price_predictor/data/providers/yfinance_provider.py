"""yfinance implementation of PriceProvider.

Wraps yfinance.download() with the normalization our interface requires:
    - Flatten yfinance's MultiIndex columns
    - Rename Title Case -> snake_case
    - Localize index to Asia/Kolkata
    - Inclusive end-date semantics (yfinance is exclusive)
    - Auto-suffix bare symbols with the configured market (default '.NS')
      so this provider plays nice as the fallback in the NSE-default chain
      without breaking explicit non-NSE callers.
    - Wrap exceptions as PriceFetchError so the resilient layer can fall back

NOTHING ABOUT YFINANCE LEAKS PAST THIS FILE. The next provider we add
(Stooq, Alpha Vantage, etc.) gets its own class with the same interface;
this one stays untouched.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from loguru import logger

from price_predictor.data.providers.base import PriceFetchError, PriceProvider

# A ticker is considered "already suffixed" if it ends with `.XX` where
# XX is 1-5 letters (covers .NS, .BO, .L, .US, .DE, .HK, .NSE, .BSE, etc.).
# Anchored to end-of-string so internal dots in (hypothetical) tickers
# aren't confused for an exchange marker.
_HAS_SUFFIX_RE = re.compile(r"\.[A-Za-z]{1,5}$")


class YFinanceProvider(PriceProvider):
    """Concrete provider backed by the yfinance library.

    Stateless -- safe to share a single instance across the whole process.

    `default_market` controls the suffix appended to bare symbols (no
    exchange suffix). Defaults to 'NS' because this codebase is NSE-
    focused; pass `default_market=None` to disable auto-suffixing if
    you want the raw yfinance behaviour (where bare symbols are
    interpreted as US tickers).

    Examples:
        YFinanceProvider().fetch_ohlcv('RELIANCE', ...)      -> 'RELIANCE.NS'
        YFinanceProvider().fetch_ohlcv('RELIANCE.NS', ...)   -> 'RELIANCE.NS' (unchanged)
        YFinanceProvider().fetch_ohlcv('AAPL.US', ...)       -> 'AAPL.US' (unchanged)
        YFinanceProvider(default_market=None).fetch_ohlcv('AAPL', ...)
                                                              -> 'AAPL' (unchanged)
    """

    def __init__(self, *, default_market: str | None = "NS") -> None:
        # Strip any leading '.' the caller may have added (`'.NS'` and `'NS'`
        # should both work). Empty string treated as None for safety.
        self._default_market = (default_market or "").lstrip(".") or None

    @property
    def name(self) -> str:
        return "yfinance"

    def _resolve_ticker(self, ticker: str) -> str:
        """Apply default_market suffix iff ticker has no exchange suffix.

        Why a regex instead of a simple `'.' in ticker`: some tickers
        legitimately contain dots (e.g. 'BRK.B' on NYSE). The regex
        anchors to a 1-5 letter suffix at end-of-string — 'BRK.B' DOES
        match (.B is 1 letter), and 'BRK.B' is already a valid yfinance
        ticker, so leaving it unchanged is correct behaviour.
        """
        if self._default_market is None:
            return ticker
        if _HAS_SUFFIX_RE.search(ticker):
            return ticker
        return f"{ticker}.{self._default_market}"

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

        resolved_ticker = self._resolve_ticker(ticker.strip())

        # ── Fetch from yfinance (upstream's fault -> PriceFetchError) ──
        # `end` shifted +1 day so our wrapper is inclusive (yfinance is exclusive).
        # auto_adjust=False preserves both `Close` (raw) and `Adj Close` (adjusted).
        try:
            df = yf.download(
                tickers=resolved_ticker,
                start=start,
                end=end + timedelta(days=1),
                interval=interval,
                auto_adjust=False,
                progress=False,
            )
        except Exception as e:
            raise PriceFetchError(
                f"yfinance failed for ticker={resolved_ticker!r} "
                f"(input={ticker!r}) start={start} end={end}: {e}"
            ) from e

        # ── Empty-result check ──
        if df is None or df.empty:
            raise PriceFetchError(
                f"yfinance returned no data for ticker={resolved_ticker!r} "
                f"(input={ticker!r}) in range {start}…{end} (interval={interval}). "
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
            f"yfinance fetched ticker={resolved_ticker} (input={ticker}) "
            f"rows={len(df)} start={start} end={end} interval={interval}"
        )
        return df
