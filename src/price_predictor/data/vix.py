"""India VIX fetcher — the I/O half of the H9d regime gate.

Kept separate from ``analysis/vix.py`` (which is pure math) so the
analysis package stays I/O-free.

WHY yfinance, NOT nsepython
===========================
pred_logic_solutions.md §H9d suggested nsepython's ``index_history``.
But nsepython is GPL-licensed (flagged in the doc's own dependency
table), and we already depend on yfinance, which serves India VIX under
the symbol ``^INDIAVIX``. Reusing our existing provider avoids adding a
copyleft dependency for one number.

WHY A DEDICATED PROVIDER INSTANCE
=================================
``^INDIAVIX`` has no exchange suffix, so the default YFinanceProvider
(default_market="NS") would wrongly rewrite it to ``^INDIAVIX.NS``. We
build a suffix-free provider (default_market=None) so the index symbol
passes through untouched.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
from loguru import logger

from price_predictor.data.providers import PriceFetchError
from price_predictor.data.providers.yfinance_provider import YFinanceProvider

# yfinance's symbol for the NSE India VIX index.
INDIA_VIX_SYMBOL = "^INDIAVIX"

# Suffix-free provider so "^INDIAVIX" isn't rewritten to "^INDIAVIX.NS".
# Module-level singleton — the provider is stateless and cheap to reuse.
_vix_provider = YFinanceProvider(default_market=None)


def fetch_india_vix(start: date, end: date) -> pd.Series:
    """Fetch India VIX daily closes for [start, end] (inclusive).

    Args:
        start: First trading day (inclusive).
        end: Last trading day (inclusive).

    Returns:
        A chronological pd.Series of VIX closes named "india_vix",
        indexed by the provider's tz-aware DatetimeIndex. Empty series
        if the provider returned no rows.

    Raises:
        ValueError: start > end (caller's fault).
        PriceFetchError: the provider failed — caller decides whether a
            missing VIX is fatal (usually not: it's a regime gate, so
            "unknown" is an acceptable degrade).
    """
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")

    try:
        df = _vix_provider.fetch_ohlcv(INDIA_VIX_SYMBOL, start, end, "1d")
    except PriceFetchError:
        logger.warning("[vix] India VIX fetch failed for {}..{}", start, end)
        raise

    if df.empty or "close" not in df.columns:
        return pd.Series(dtype=float, name="india_vix")

    return df["close"].rename("india_vix")
