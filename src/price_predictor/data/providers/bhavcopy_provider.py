"""NSE bhavcopy implementation of PriceProvider — bulk EOD fallback.

WHY THIS EXISTS
===============
Solves C1 secondary tier from `pred_logic_solutions.md`. This is a thin
adapter: the bulk fetch logic + format-routing lives in `data/bhavcopy.py`
(single-responsibility); this class just stitches per-day responses
together and filters to a single ticker.

WHEN TO PREFER THIS PROVIDER
============================
- jugaad-data is unreachable (NSE API rotation, etc.).
- You want exchange-of-record OHLC for a known trading day window.
- You don't need adjusted prices (NSE doesn't ship them; we set
  adj_close = close. Same approach as JugaadDataProvider — see its
  module docstring for the rationale.)

WHEN NOT TO USE
===============
- Long backtests on a single symbol over many years: this makes ONE
  HTTP call per trading day. Per-symbol providers (jugaad-data,
  yfinance) are dramatically faster for that shape.
- True backfill of the entire universe: prefer `fetch_nse_bhavcopy(d)`
  directly so you only pay one HTTP call per day instead of one per
  (symbol x day) pair.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta

import pandas as pd
from loguru import logger

from price_predictor.data.bhavcopy import BhavcopyError, fetch_nse_bhavcopy
from price_predictor.data.providers.base import PriceFetchError, PriceProvider
from price_predictor.data.providers.jugaad_provider import _normalise_symbol
from price_predictor.prediction.trading_calendar import is_trading_day

# Type alias for the injected bulk fetcher (same signature as
# `fetch_nse_bhavcopy(d) -> DataFrame`).
BhavcopyFetcher = Callable[[date], pd.DataFrame]


class NseBhavcopyProvider(PriceProvider):
    """PriceProvider adapter over the per-day bhavcopy bulk utility.

    Iterates trading days in [start, end], calls the bulk fetcher for
    each, filters to the requested ticker, and concatenates. Uses
    `is_trading_day` to skip weekends + NSE holidays so we don't
    waste HTTP calls on days we know have no data.

    The `bhavcopy_fn` constructor arg is the test-injection seam — same
    pattern as JugaadDataProvider.
    """

    def __init__(self, *, bhavcopy_fn: BhavcopyFetcher | None = None) -> None:
        # Resolve once at construction; keeps fetch_ohlcv branchless.
        self._bhavcopy_fn: BhavcopyFetcher = bhavcopy_fn or fetch_nse_bhavcopy

    @property
    def name(self) -> str:
        return "nse-bhavcopy"

    def fetch_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        # ── Input validation (caller's fault → ValueError, no fallback) ──
        if not ticker or not ticker.strip():
            raise ValueError(f"ticker must be a non-empty string, got {ticker!r}")
        if start > end:
            raise ValueError(f"start ({start}) must be <= end ({end})")
        if interval != "1d":
            # Bhavcopy IS the daily snapshot — no other interval is even
            # meaningful here.
            raise ValueError(
                f"NseBhavcopyProvider only supports interval='1d', got {interval!r}"
            )

        symbol = _normalise_symbol(ticker)
        if not symbol:
            raise ValueError(
                f"ticker {ticker!r} reduced to empty string after stripping "
                "exchange suffix; supply a non-empty bare NSE symbol."
            )
        # NSE bhavcopy SYMBOL column is upper-case; normalise to match so
        # callers using lowercase tickers don't silently get zero rows.
        symbol_upper = symbol.upper()

        # ── Iterate trading days ──
        per_day_frames: list[pd.DataFrame] = []
        per_day_errors: list[tuple[date, str]] = []
        for d in _iter_trading_days(start, end):
            try:
                day_df = self._bhavcopy_fn(d)
            except BhavcopyError as e:
                # Holidays NSE forgot to flag, occasional 404s on
                # newly-listed-day rows, etc. Don't kill the whole fetch
                # for one bad day; record + move on.
                per_day_errors.append((d, str(e)))
                continue

            hits = day_df[day_df["SYMBOL"].str.upper() == symbol_upper]
            if not hits.empty:
                per_day_frames.append(hits)

        if not per_day_frames:
            err_summary = (
                f"; {len(per_day_errors)} day(s) errored: "
                f"{per_day_errors[:3]}{'...' if len(per_day_errors) > 3 else ''}"
                if per_day_errors else ""
            )
            raise PriceFetchError(
                f"NSE bhavcopy returned no rows for symbol={symbol_upper!r} "
                f"in range {start}..{end}{err_summary}"
            )

        df = pd.concat(per_day_frames, ignore_index=True)

        # Reshape to the PriceProvider DataFrame contract: lowercase
        # snake_case OHLCV columns, tz-aware Asia/Kolkata DatetimeIndex,
        # adj_close = close (NSE doesn't publish adjusted prices).
        df = df.rename(columns={
            "DATE": "date",
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "CLOSE": "close",
            "VOLUME": "volume",
        })
        df["adj_close"] = df["close"]
        df = df[["date", "open", "high", "low", "close", "adj_close", "volume"]]
        df = df.set_index("date").sort_index()

        # `DATE` was built tz-aware in the bulk fetcher; convert defensively
        # in case a future change ships tz-naive (don't double-shift).
        if df.index.tz is None:
            df.index = df.index.tz_localize("Asia/Kolkata")
        else:
            df.index = df.index.tz_convert("Asia/Kolkata")

        logger.debug(
            f"nse-bhavcopy fetched symbol={symbol_upper} rows={len(df)} "
            f"start={start} end={end} per_day_errors={len(per_day_errors)}"
        )
        return df


def _iter_trading_days(start: date, end: date):
    """Yield each trading day in [start, end] inclusive, skipping weekends
    and NSE holidays. Pure generator — zero allocation for big windows.

    Lives at module scope (not as a method) because:
      (1) it has no instance state to depend on,
      (2) tests import it directly to verify holiday-skipping behaviour
          without spinning up a provider.
    """
    cur = start
    one_day = timedelta(days=1)
    while cur <= end:
        if is_trading_day(cur):
            yield cur
        cur += one_day
