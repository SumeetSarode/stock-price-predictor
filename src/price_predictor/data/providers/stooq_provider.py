"""Stooq implementation of PriceProvider.

Stooq is a free Polish-hosted financial data site that exposes daily OHLC
as CSV downloads.

WHY STOOQ
=========
- Genuinely free (an apikey is required as of 2024 but it's free to obtain
  via a one-time captcha challenge -- no signup, no email, no expiry).
- Stable: been around since 2004; rarely changes its CSV format.
- Diverse coverage: NSE/BSE Indian stocks, US stocks, FX, crypto, indices.
- Daily resolution only -- intraday isn't free.
- Slightly delayed (last updated bar is usually previous trading day's close,
  refreshed in the European morning).

GETTING AN APIKEY
=================
1. Open https://stooq.com/q/d/?s=aapl.us&get_apikey
2. Solve the captcha
3. Copy the apikey from the bottom of the resulting page
4. Drop it in your .env as STOOQ_API_KEY=...

No email, no account creation. Same key works forever (until Stooq
rotates server-side, very rare).

WHY IT'S A GOOD #2 IN THE CHAIN
================================
yfinance throttles aggressively when bursty (which happens easily in our
4-tools-per-question agent flow). Stooq doesn't have meaningful per-IP
rate limits in practice -- it's a static-CSV-download service. So if
yfinance is sulking, Stooq will almost always have an answer.

TICKER FORMAT TRANSLATION
=========================
Callers pass the YFINANCE-CANONICAL ticker (e.g., 'RELIANCE.NS') and we
translate internally. This keeps the ticker abstraction at the provider
boundary, NOT in the agent or tool code (Tell-Don't-Ask).

    yfinance       Stooq          What it is
    ---------      ------------   --------------------
    RELIANCE.NS    reliance.in    NSE-listed Indian stock
    AAPL           aapl.us        NYSE/NASDAQ US stock
    BTC-USD        btcusd         crypto pair (no dot)
"""
from __future__ import annotations

import io
from datetime import date

import httpx
import pandas as pd
from loguru import logger

from price_predictor.data.providers._http import get_verify_setting
from price_predictor.data.providers.base import PriceFetchError, PriceProvider

# Stooq's bulk CSV endpoint. d1 and d2 are inclusive YYYYMMDD.
_STOOQ_URL = "https://stooq.com/q/d/l/"

# How long to wait for Stooq before giving up. They're usually fast (<2s),
# but European peak hours can stretch this. 10s is the sweet spot between
# "snappy fallback" and "don't bail on a transient slowdown".
_STOOQ_TIMEOUT_SECS = 10.0


class StooqProvider(PriceProvider):
    """Concrete provider backed by Stooq's CSV download endpoint.

    Stateless apart from the optional API key -- safe to share one
    instance across the process.
    """

    def __init__(self, api_key: str = "") -> None:
        """Construct with an optional API key (see lazy validation).

        Empty key is allowed at construction so the prices.py factory can
        build every registered provider unconditionally. The key is
        re-checked at each fetch_ohlcv call; an empty key surfaces as a
        PriceFetchError so the resilient layer falls back cleanly.

        Stooq's CSV endpoint requires this key as of 2024. It's free to
        obtain via a one-time captcha; see module docstring for steps.
        """
        self._api_key = api_key.strip()

    @property
    def name(self) -> str:
        return "stooq"

    # ───────────────────────────────────────────────────────────
    # Ticker translation
    # ───────────────────────────────────────────────────────────
    @staticmethod
    def _to_stooq_ticker(ticker: str) -> str:
        """Translate a yfinance-canonical ticker to Stooq's format.

        Args:
            ticker: yfinance-style ticker. NSE: 'RELIANCE.NS'. US: 'AAPL'.

        Returns:
            Stooq-style ticker, lowercased with the right suffix.

        Why lowercased: Stooq's URL is case-insensitive in practice, but
        their downloads come back with lowercase symbols. Normalizing here
        means tests don't have to worry about case round-tripping.
        """
        t = ticker.strip()
        # Case-insensitive suffix check: callers occasionally pass lowercase.
        upper = t.upper()
        if upper.endswith(".NS") or upper.endswith(".BO"):
            # NSE or BSE Indian -- both map to Stooq's .in namespace.
            return upper.split(".")[0].lower() + ".in"
        # Otherwise assume US-listed (Stooq treats bare tickers as .us).
        return t.lower() + ".us"

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
            # Stooq's free tier is daily-only; '1wk' / '1mo' / '1h' aren't
            # supported. Raise PriceFetchError so the resilient layer can
            # try the next provider, which might handle that interval.
            raise PriceFetchError(
                f"stooq only supports interval='1d', got {interval!r}. "
                "For intraday/weekly/monthly bars use yfinance or alpha_vantage."
            )

        stooq_symbol = self._to_stooq_ticker(ticker)

        # Lazy key check: empty -> PriceFetchError so resilient layer falls back.
        # Includes ticker for log scanability.
        if not self._api_key:
            raise PriceFetchError(
                f"stooq requires STOOQ_API_KEY to be set "
                f"(while fetching ticker={ticker!r}). "
                f"Get a free apikey via captcha at "
                f"https://stooq.com/q/d/?s={stooq_symbol}&get_apikey then set "
                "STOOQ_API_KEY in .env. Or remove 'stooq' from PRICE_CHAIN."
            )

        params = {
            "s": stooq_symbol,
            "i": "d",  # daily
            "d1": start.strftime("%Y%m%d"),
            "d2": end.strftime("%Y%m%d"),
            "apikey": self._api_key,
        }

        # ── HTTP fetch ──
        try:
            resp = httpx.get(
                _STOOQ_URL,
                params=params,
                timeout=_STOOQ_TIMEOUT_SECS,
                verify=get_verify_setting(),
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise PriceFetchError(
                f"stooq HTTP failure for ticker={ticker!r} "
                f"(stooq_symbol={stooq_symbol!r}): {e}"
            ) from e

        body = resp.text.strip()

        # ── Empty / unknown-symbol detection ──
        # Stooq returns the literal string 'No data' (HTTP 200!) for unknown
        # symbols. That's not 'success with empty df'; it's a fetch failure.
        if not body or body.lower().startswith("no data"):
            raise PriceFetchError(
                f"stooq returned no data for ticker={ticker!r} "
                f"(stooq_symbol={stooq_symbol!r}) in range {start}…{end}. "
                "Possible causes: unknown symbol, weekend-only range, or "
                "ticker not covered by stooq."
            )

        # ── CSV parse ──
        # Stooq columns: Date,Open,High,Low,Close,Volume (no Adj Close).
        # See WHY-NO-ADJCLOSE comment below.
        try:
            df = pd.read_csv(io.StringIO(body))
        except Exception as e:
            raise PriceFetchError(
                f"stooq returned malformed CSV for ticker={ticker!r}: {e}\n"
                f"First 200 chars of body: {body[:200]!r}"
            ) from e

        if df.empty:
            raise PriceFetchError(
                f"stooq returned empty CSV for ticker={ticker!r} "
                f"in range {start}…{end} (header-only response)."
            )

        # ── Column normalization ──
        df.columns = df.columns.str.lower()
        # Stooq omits Adj Close. Best v1 approximation: use Close as both.
        # WHY-NO-ADJCLOSE: Stooq's CSV is split-adjusted but NOT dividend-
        # adjusted. For our use cases (technical indicators), this is fine --
        # SMAs/RSI/etc. care about price-action continuity, not absolute
        # dividend-reinvestment returns. If we ever need true total-return
        # math, the adj_close=close approximation here will be wrong; flag
        # it then, not now (YAGNI).
        if "adj close" in df.columns:
            df = df.rename(columns={"adj close": "adj_close"})
        else:
            df["adj_close"] = df["close"]

        df = df[["date", "open", "high", "low", "close", "adj_close", "volume"]]

        # ── Index normalization ──
        # Stooq dates are tz-naive YYYY-MM-DD strings. Localize to Asia/Kolkata
        # to match yfinance's normalized output (and our DataFrame contract).
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df.index = df.index.tz_localize("Asia/Kolkata")
        df = df.sort_index()  # Stooq ships ascending already, but be safe.

        logger.debug(
            f"stooq fetched ticker={ticker} (stooq_symbol={stooq_symbol}) "
            f"rows={len(df)} start={start} end={end}"
        )
        return df
