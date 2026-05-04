"""Test fixtures: synthetic OHLCV series with known properties."""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_TZ = "Asia/Kolkata"


def _ohlc_from_close(close: np.ndarray, vol: float = 1000.0) -> pd.DataFrame:
    """Build OHLCV from a close series with high=close+1, low=close-1, etc."""
    n = len(close)
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz=DEFAULT_TZ)
    return pd.DataFrame(
        {
            "open":      close.copy(),
            "high":      close + 1.0,
            "low":       close - 1.0,
            "close":     close,
            "adj_close": close,
            "volume":    np.full(n, vol),
        },
        index=dates,
    )


def linear_uptrend(n: int = 250, start: float = 100.0, slope: float = 1.0) -> pd.DataFrame:
    """Strictly rising series. ADX should be high; RSI elevated; bullish."""
    return _ohlc_from_close(np.linspace(start, start + slope * n, n))


def linear_downtrend(n: int = 250, start: float = 200.0, slope: float = 1.0) -> pd.DataFrame:
    """Strictly falling series. Mirror of uptrend."""
    return _ohlc_from_close(np.linspace(start, start - slope * n, n))


def sideways(n: int = 250, mean: float = 100.0, amplitude: float = 2.0) -> pd.DataFrame:
    """Random walk around `mean`. ADX should be low (chop), RSI ~50.

    Uses a fixed seed for reproducibility -- tests must be deterministic.
    """
    rng = np.random.default_rng(seed=42)
    # Mean-reverting random walk: each step is small noise + pull toward mean
    closes = np.zeros(n)
    closes[0] = mean
    for i in range(1, n):
        noise = rng.normal(0, amplitude * 0.3)
        pull = (mean - closes[i-1]) * 0.1
        closes[i] = closes[i-1] + noise + pull
    return _ohlc_from_close(closes)


def insufficient_history(n: int = 5) -> pd.DataFrame:
    """Tiny series: most indicators should return None."""
    return _ohlc_from_close(np.linspace(100, 105, n))
