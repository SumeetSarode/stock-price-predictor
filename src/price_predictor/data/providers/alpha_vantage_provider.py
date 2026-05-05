"""Alpha Vantage implementation of PriceProvider.

Alpha Vantage is a JSON-based market-data API. Free tier: 25 requests per
day. Paid tier: lifts that to 75/min for ~$50/mo.

WHY ALPHA VANTAGE
=================
- Genuine REST API (vs. yfinance scraping or Stooq CSV download).
- Has both free AND paid tiers -- toggling USE_PAID_PRICES targets this.
- Reliable error reporting (clear JSON shape; no guessing if a 200 means
  success or "rate-limited but you wouldn't know it").
- Has intraday + daily for India + US.
- Drawback: free tier is brutal (25/day). That's WHY it sits LAST in the
  free chain -- only invoked when both yfinance and stooq are already cooled
  down, so we never burn the daily quota on routine fallback.

KEY HANDLING
============
The constructor takes an optional api_key. If empty, EVERY call raises
PriceFetchError with a clear "set ALPHA_VANTAGE_API_KEY" message.

We deliberately do NOT raise at construction. WHY: the prices.py factory
constructs every registered provider eagerly -- if a user has only yfinance
in PRICE_CHAIN, they shouldn't be forced to set an AV key just because the
class exists. Eager construction with lazy key validation = best of both.

TICKER TRANSLATION
==================
AV's Indian-market format: 'RELIANCE.BSE' (BSE-listed). They don't expose
a separate NSE namespace -- BSE coverage is the closest match for our
'.NS' tickers. For US tickers we pass through unchanged.

    yfinance       Alpha Vantage     What it is
    ---------      --------------    --------------------
    RELIANCE.NS    RELIANCE.BSE      Indian stock (mapped to BSE on AV)
    RELIANCE.BO    RELIANCE.BSE      already BSE
    AAPL           AAPL              US stock (passthrough)

SOURCES
=======
- API docs:        https://www.alphavantage.co/documentation/
- Rate limits:     https://www.alphavantage.co/premium/
- Status field:    every successful response has 'Time Series (Daily)' as
                   the data key; errors put a string under 'Note' (rate
                   limit) or 'Error Message' (bad ticker).
"""
from __future__ import annotations

from datetime import date

import httpx
import pandas as pd
from loguru import logger

from price_predictor.data.providers._http import get_verify_setting
from price_predictor.data.providers.base import PriceFetchError, PriceProvider

_AV_URL = "https://www.alphavantage.co/query"

# AV is HTTP-slow at ~3-5s per request even on a healthy day. 15s gives
# headroom without making the resilient-layer fallback feel sluggish.
_AV_TIMEOUT_SECS = 15.0

# AV's TIME_SERIES_DAILY supports two output sizes:
#   compact = last 100 trading days (faster, smaller payload)
#   full    = 20+ years of history (slower, larger payload)
# We pick based on requested range. 100 trading days ≈ 145 calendar days.
_COMPACT_THRESHOLD_DAYS = 100


class AlphaVantageProvider(PriceProvider):
    """Concrete provider backed by AlphaVantage's TIME_SERIES_DAILY endpoint.

    Stateless apart from the API key -- safe to share one instance across
    the process.
    """

    def __init__(self, api_key: str = "") -> None:
        """Construct with an API key (may be empty -- see lazy validation).

        Empty key is allowed at construction so that the prices.py factory
        can build every registered provider unconditionally. The key is
        re-checked at each fetch_ohlcv call; an empty key surfaces as a
        PriceFetchError so the resilient layer falls back cleanly.
        """
        self._api_key = api_key.strip()

    @property
    def name(self) -> str:
        return "alpha_vantage"

    # ───────────────────────────────────────────────────────────
    # Ticker translation
    # ───────────────────────────────────────────────────────────
    @staticmethod
    def _to_av_ticker(ticker: str) -> str:
        """Translate yfinance-canonical ticker to AlphaVantage format.

        AV doesn't have an NSE-specific namespace; both NSE-listed and
        BSE-listed Indian stocks resolve under '.BSE' on their side.
        """
        t = ticker.strip()
        # Case-insensitive suffix check: callers occasionally pass lowercase.
        upper = t.upper()
        if upper.endswith(".NS") or upper.endswith(".BO"):
            return upper.split(".")[0] + ".BSE"
        return upper

    # ───────────────────────────────────────────────────────────
    # Main fetch
    # ───────────────────────────────────────────────────────────
    def fetch_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        # Caller-side validation -- ValueError, no fallback worth attempting
        if not ticker or not ticker.strip():
            raise ValueError(f"ticker must be a non-empty string, got {ticker!r}")
        if start > end:
            raise ValueError(f"start ({start}) must be <= end ({end})")
        if interval != "1d":
            # We could implement TIME_SERIES_INTRADAY here, but YAGNI for
            # v1. PriceFetchError so the resilient layer can try yfinance.
            raise PriceFetchError(
                f"alpha_vantage provider only supports interval='1d' in v1, "
                f"got {interval!r}. (Intraday is supported by AV's API but "
                "not yet wired up here.)"
            )

        # Lazy key check: empty -> PriceFetchError so resilient layer falls back.
        # Includes ticker for debuggability so log scanners can trace which
        # symbol triggered the missing-key complaint.
        if not self._api_key:
            raise PriceFetchError(
                f"alpha_vantage requires ALPHA_VANTAGE_API_KEY to be set "
                f"(while fetching ticker={ticker!r}). "
                "Either set it in your .env (https://www.alphavantage.co/support/#api-key) "
                "or remove 'alpha_vantage' from PRICE_CHAIN / PRICE_PAID."
            )

        av_symbol = self._to_av_ticker(ticker)
        # Decide compact vs full output size. Compact is ~5x faster.
        days_requested = (end - start).days
        outputsize = "compact" if days_requested <= _COMPACT_THRESHOLD_DAYS else "full"

        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": av_symbol,
            "outputsize": outputsize,
            "datatype": "json",
            "apikey": self._api_key,
        }

        # ── HTTP fetch ──
        try:
            resp = httpx.get(
                _AV_URL,
                params=params,
                timeout=_AV_TIMEOUT_SECS,
                verify=get_verify_setting(),
            )
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPError as e:
            raise PriceFetchError(
                f"alpha_vantage HTTP failure for ticker={ticker!r} "
                f"(av_symbol={av_symbol!r}): {e}"
            ) from e
        except ValueError as e:
            # response.json() raises ValueError on non-JSON bodies. AV
            # occasionally returns HTML when overloaded.
            raise PriceFetchError(
                f"alpha_vantage returned non-JSON body for ticker={ticker!r}: {e}"
            ) from e

        # ── AV-specific error envelopes (HTTP 200 with error fields) ──
        # 'Note'         -> rate-limit hit (free tier: 25 req/day)
        # 'Information'  -> rate-limit on premium endpoints used without sub
        # 'Error Message'-> invalid ticker / bad params
        if "Note" in payload:
            raise PriceFetchError(
                f"alpha_vantage rate limit hit for ticker={ticker!r}: "
                f"{payload['Note']}"
            )
        if "Information" in payload:
            raise PriceFetchError(
                f"alpha_vantage premium-tier required for ticker={ticker!r}: "
                f"{payload['Information']}"
            )
        if "Error Message" in payload:
            raise PriceFetchError(
                f"alpha_vantage rejected ticker={ticker!r} "
                f"(av_symbol={av_symbol!r}): {payload['Error Message']}"
            )

        # ── Extract the time-series block ──
        # Successful TIME_SERIES_DAILY responses put data under this key.
        # If absent, the response is shaped unexpectedly -- treat as fetch error.
        series = payload.get("Time Series (Daily)")
        if not series:
            raise PriceFetchError(
                f"alpha_vantage returned no time-series block for ticker={ticker!r}. "
                f"Response keys: {list(payload.keys())}"
            )

        # ── Build DataFrame ──
        # AV's per-day shape:
        #   "1. open": "...", "2. high": "...", "3. low": "...",
        #   "4. close": "...", "5. volume": "..."
        # All values arrive as strings; we cast to float/int.
        records = []
        for date_str, bar in series.items():
            records.append({
                "date": date_str,
                "open": float(bar["1. open"]),
                "high": float(bar["2. high"]),
                "low": float(bar["3. low"]),
                "close": float(bar["4. close"]),
                "volume": int(bar["5. volume"]),
            })
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()  # AV ships descending

        # AV's free TIME_SERIES_DAILY (post-2024) doesn't include adjusted
        # close. Same approximation as Stooq: mirror close into adj_close.
        # See WHY-NO-ADJCLOSE in stooq_provider.py for the trade-off.
        df["adj_close"] = df["close"]
        df = df[["open", "high", "low", "close", "adj_close", "volume"]]

        # ── Date-range filter ──
        # AV always returns either 100 days (compact) or full history.
        # Trim to the caller's window so downstream sees inclusive boundaries.
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

        if df.empty:
            raise PriceFetchError(
                f"alpha_vantage returned data, but none in requested range "
                f"{start}…{end} for ticker={ticker!r} (response had "
                f"{len(series)} bars total)."
            )

        # ── Timezone normalization (contract: Asia/Kolkata) ──
        df.index = df.index.tz_localize("Asia/Kolkata")

        logger.debug(
            f"alpha_vantage fetched ticker={ticker} (av_symbol={av_symbol}) "
            f"rows={len(df)} outputsize={outputsize}"
        )
        return df
