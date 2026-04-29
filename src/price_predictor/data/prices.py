"""OHLCV price fetcher backed by yfinance.

WHY TWO CLOSE COLUMNS:
    yfinance is called with auto_adjust=False so we get BOTH:
      - close     — unadjusted (what actually traded; what users see on their
                    broker tomorrow; what target/SL math uses)
      - adj_close — adjusted for splits/dividends (what indicators like SMA,
                    RSI, MACD, ATR should consume to avoid jumps on splits)

    Downstream code MUST pick the right one for its job.
"""
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from loguru import logger


class PriceFetchError(Exception):
    """Raised when yfinance returns no data, errors out, or returns garbage."""


def fetch_ohlcv(
    ticker: str,
    start: date,
    end: date,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV history for a ticker.

    Args:
        ticker:   yfinance ticker symbol, e.g. "RELIANCE.NS" for NSE.
        start:    First trading day to include (inclusive).
        end:      Last trading day to include (inclusive — we shift +1 day
                  internally to handle yfinance's exclusive-end quirk).
        interval: yfinance interval string. "1d" (default), "1wk", "1mo", "1h".

    Returns:
        DataFrame indexed by tz-aware datetime (Asia/Kolkata), columns:
            open, high, low, close, adj_close, volume

    Raises:
        ValueError:      Empty/whitespace ticker, or start > end.
        PriceFetchError: yfinance returned no rows, or upstream API error.
    """
    # ── Input validation ─────────────────────────
    if not ticker or not ticker.strip():
        raise ValueError(f"ticker must be a non-empty string, got {ticker!r}")
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    # ── Fetch from yfinance ──────────────────────
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

    # ── Empty-result check ─────────────────────────
    if df is None or df.empty:
        raise PriceFetchError(
            f"No price data returned for ticker={ticker!r} "
            f"in range {start}…{end} (interval={interval}). "
            f"Possible causes: delisted ticker, wrong suffix "
            f"(NSE needs '.NS'), weekend-only range, or upstream outage."
        )

    # ── Column normalization ───────────────────────
    # yfinance >= 0.2 returns MultiIndex columns even for single tickers:
    #   level 0 = field ("Open", "High", ...), level 1 = ticker.
    # We flatten to single-level snake_case for downstream sanity.
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

    # ── Timezone normalization ──────────────────────
    # Daily data comes back tz-naive; intraday comes back tz-aware (UTC).
    # Normalize everything to Asia/Kolkata for consistent as-of-date math.
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")

    # ── Debug log ───────────────────────────────────
    logger.debug(
        f"fetch_ohlcv: ticker={ticker} rows={len(df)} "
        f"start={start} end={end} interval={interval}"
    )

    return df
