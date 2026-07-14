"""jugaad-data implementation of PriceProvider — NSE-native bars.

WHY THIS EXISTS
===============
Solves the C1 primary tier (NSE-native prices). Stooq ships zero
NSE coverage and was only ever in the chain because of a copy-pasted
US tutorial. `jugaad-data` is an actively-maintained NSE-native library
that pulls bars from NSE's own historical-trade endpoints, with built-in
session priming + per-year batching to satisfy NSE's 1-year query limit.

WHY NOT yfinance ALONE
======================
yfinance gets NSE bars via Yahoo's mirror, which:
- Is missing for some delisted / illiquid scrips,
- Lags reality by 1-2 trading sessions on results days,
- Quietly silently breaks ~2x/year when Yahoo changes their HTML.
jugaad-data hits NSE directly so it's the "exchange-of-record" tier
when you need to be sure the bars are real.

WHAT JUGAAD-DATA RETURNS
========================
`stock_df(symbol='RELIANCE', from_date=..., to_date=..., series='EQ')` -> DataFrame:
    DATE | SERIES | OPEN | HIGH | LOW | PREV. CLOSE | LTP | CLOSE |
    VWAP | VOLUME | VALUE | NO OF TRADES | DELIVERY QTY | DELIVERY % | SYMBOL

KEY MAPPINGS to our PriceProvider contract:
    DATE   -> index (tz-aware Asia/Kolkata)
    OPEN   -> open
    HIGH   -> high
    LOW    -> low
    CLOSE  -> close
    CLOSE  -> adj_close (NSE doesn't ship split/dividend-adjusted prices;
                         we copy raw close to satisfy the contract and
                         DOCUMENT this limitation. Callers needing true
                         adjusted prices should fall through to yfinance.)
    VOLUME -> volume

KNOWN LIMITS
============
- adj_close == close (no upstream adjusted feed). Documented in module
  docstring + at the assignment site. Not silently misleading.
- Symbol must be the BARE NSE trading symbol ('RELIANCE', NOT 'RELIANCE.NS'
  and NOT 'RELIANCE.BO'). We strip common suffixes defensively so callers
  using yfinance-style tickers still work.
- jugaad-data is synchronous + does I/O. We DO NOT add async here because
  the rest of the price stack is synchronous and the resilient layer runs
  providers serially anyway.
- Some networks may block nseindia.com. Unit tests fully mock
  `stock_df` via monkeypatch; integration tests degrade gracefully when
  NSE is unreachable (same pattern as the filings module).
"""
from __future__ import annotations

import re
from datetime import date

import pandas as pd
from loguru import logger

from price_predictor.data.providers.base import PriceFetchError, PriceProvider

# Default "EQ" = equity series. Other valid values: "BE" (book entry),
# "BL" (bulk deal), "ST" (segment). 99% of use cases want EQ; we expose
# it as a constructor arg for the rare ones that don't.
_DEFAULT_SERIES = "EQ"

# Strip trailing ".NS" / ".BO" / ".BSE" / ".NSE" so callers using
# yfinance-style suffixes don't have to remember to drop them.
_TICKER_SUFFIX_RE = re.compile(r"\.(NS|BO|NSE|BSE)$", re.IGNORECASE)


def _normalise_symbol(ticker: str) -> str:
    """RELIANCE.NS -> RELIANCE; RELIANCE -> RELIANCE.

    jugaad-data wants the bare NSE symbol. Leaving the suffix on causes
    a NoData response (NSE returns empty for 'RELIANCE.NS' as a literal
    symbol query), which would mask a fixable user error as an upstream
    failure. We strip it ourselves and document why.
    """
    return _TICKER_SUFFIX_RE.sub("", ticker.strip())


class JugaadDataProvider(PriceProvider):
    """NSE-native price provider via the jugaad-data library.

    Stateless across calls — safe to share a single instance.

    The `stock_df_fn` constructor arg is an injection seam used by tests
    (so we don't need real network in unit tests). In production callers
    leave it as the default and we import the real `jugaad_data.nse.stock_df`
    on first use (lazy import so a missing optional dep doesn't kill
    `from price_predictor.data.providers import ...`).
    """

    def __init__(
        self,
        *,
        series: str = _DEFAULT_SERIES,
        stock_df_fn=None,
    ) -> None:
        self._series = series
        self._stock_df_fn = stock_df_fn  # tests inject; prod resolves lazily

    @property
    def name(self) -> str:
        return "jugaad-data"

    def _resolve_stock_df(self):
        """Lazy-import jugaad_data.nse.stock_df.

        Lazy because:
          (1) keeps `from price_predictor.data.providers import ...` cheap
              when jugaad-data isn't installed (e.g. CI envs that don't
              need NSE),
          (2) lets tests inject a fake without monkeypatching imports.
        """
        if self._stock_df_fn is not None:
            return self._stock_df_fn
        try:
            from jugaad_data.nse import stock_df
        except ImportError as e:
            raise PriceFetchError(
                "jugaad-data is not installed. "
                "Add `jugaad-data>=0.33.1` to your project dependencies."
            ) from e
        return stock_df

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
        if interval != "1d":
            # jugaad-data's stock_df ONLY ships daily bars. We refuse other
            # intervals explicitly rather than silently returning daily.
            raise ValueError(
                f"JugaadDataProvider only supports interval='1d', got {interval!r}"
            )

        symbol = _normalise_symbol(ticker)
        if not symbol:
            raise ValueError(
                f"ticker {ticker!r} reduced to empty string after stripping "
                "exchange suffix; supply a non-empty bare NSE symbol."
            )

        # ── Fetch from jugaad-data (upstream's fault -> PriceFetchError) ──
        stock_df = self._resolve_stock_df()
        try:
            raw = stock_df(
                symbol=symbol,
                from_date=start,
                to_date=end,
                series=self._series,
            )
        except Exception as e:
            raise PriceFetchError(
                f"jugaad-data failed for symbol={symbol!r} "
                f"start={start} end={end}: {type(e).__name__}: {e}"
            ) from e

        # ── Empty-result check ──
        if raw is None or (hasattr(raw, "empty") and raw.empty):
            raise PriceFetchError(
                f"jugaad-data returned no data for symbol={symbol!r} "
                f"in range {start}..{end} (series={self._series}). "
                "Possible causes: invalid symbol, weekend-only range, "
                "delisted/suspended scrip, or NSE archive gap."
            )

        # ── Required columns sanity ──
        required = {"DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"}
        missing = required - set(raw.columns)
        if missing:
            raise PriceFetchError(
                f"jugaad-data response for {symbol!r} missing expected "
                f"columns {sorted(missing)}; got {sorted(raw.columns)}. "
                "Library API may have changed — pin jugaad-data version."
            )

        df = raw[["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]].copy()
        df.columns = ["date", "open", "high", "low", "close", "volume"]

        # NSE doesn't publish split/dividend-adjusted prices. Setting
        # adj_close = close keeps the contract intact AND is honest:
        # callers wanting true adjusted prices should chain through to
        # yfinance which gets adjusted feeds from Yahoo's data side.
        # See git history for the C1 production-readiness scorecard.
        df["adj_close"] = df["close"]

        # Reorder to the canonical contract column order.
        df = df[["date", "open", "high", "low", "close", "adj_close", "volume"]]

        # ── Index + tz normalization ──
        # jugaad-data's DATE column is parsed via its own dtype wrapper;
        # we coerce defensively to handle both date and datetime cases.
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            bad = df["date"].isna().sum()
            raise PriceFetchError(
                f"jugaad-data returned {bad} unparseable DATE rows for "
                f"{symbol!r}; refusing to silently drop trading days."
            )
        df = df.set_index("date").sort_index()

        # NSE's DATE column has no tz; localize to Asia/Kolkata to match
        # what yfinance/stooq emit (consistent with PriceProvider contract).
        if df.index.tz is None:
            df.index = df.index.tz_localize("Asia/Kolkata")
        else:
            df.index = df.index.tz_convert("Asia/Kolkata")

        logger.debug(
            f"jugaad-data fetched symbol={symbol} rows={len(df)} "
            f"start={start} end={end} series={self._series}"
        )
        return df
