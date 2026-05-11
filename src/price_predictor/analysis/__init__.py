"""Pure indicator math -- no ADK, no agents, no I/O.

PACKAGE LAYOUT
==============
    analysis/
    ├── __init__.py             -- PRESETS, common types, signal enums
    ├── trend.py                -- SMAs, EMA-20, ADX
    ├── momentum.py             -- RSI, MACD, Stochastic, OBV
    ├── volatility.py           -- ATR, Bollinger Bands, bollinger_squeeze + ttm_squeeze
    ├── levels.py               -- swing high/low, 52w high/low, pivots
    ├── candlestick_patterns.py -- TA-Lib full 61-pattern dispatcher
    └── chart_patterns.py       -- LMW chart patterns (HS, double top/bot,
                                   triangles, broadening, rectangles)

DESIGN
======
- Pure functions consume pd.DataFrame of OHLCV bars.
- Return floats / dicts / dataclasses -- never strings, never signal enums.
- Signal interpretation lives in the TOOL layer, not here.
- Each module exposes its own preset bundle when relevant; the tool layer
  selects a preset by name.
"""
from __future__ import annotations

from typing import Final

# ── Sensitivity preset bundles ──────────────────────────────────────
# Three semantic presets per the design discussion. The tool layer picks
# one by name; primitives consume the relevant slice.

TREND_PRESETS: Final[dict[str, dict]] = {
    "standard":  {"sma": [20, 50, 200], "ema": 20, "adx": 14},
    "sensitive": {"sma": [10, 30, 100], "ema": 10, "adx": 9},
    "smooth":    {"sma": [30, 70, 200], "ema": 30, "adx": 21},
}

MOMENTUM_PRESETS: Final[dict[str, dict]] = {
    "standard":  {"rsi": 14, "macd": (12, 26, 9), "stoch": (14, 3, 3)},
    "sensitive": {"rsi": 9,  "macd": (8, 17, 9),  "stoch": (9, 3, 3)},
    "smooth":    {"rsi": 21, "macd": (19, 39, 9), "stoch": (21, 5, 5)},
}

VOLATILITY_PRESETS: Final[dict[str, dict]] = {
    "standard":  {"atr": 14, "bb": (20, 2.0)},
    "sensitive": {"atr": 9,  "bb": (10, 2.0)},
    "smooth":    {"atr": 21, "bb": (30, 2.0)},
}

LEVELS_PRESETS: Final[dict[str, dict]] = {
    "standard":  {"swing_lookback": 30},
    "sensitive": {"swing_lookback": 15},
    "smooth":    {"swing_lookback": 60},
}

VALID_PRESETS: Final[tuple[str, ...]] = ("standard", "sensitive", "smooth")


def validate_preset(preset: str) -> None:
    """Raise ValueError if preset isn't one of the three valid names.

    Used at the top of every tool to fail fast on bad LLM-supplied args.
    """
    if preset not in VALID_PRESETS:
        raise ValueError(
            f"sensitivity must be one of {VALID_PRESETS}, got {preset!r}"
        )


__all__ = [
    "LEVELS_PRESETS",
    "MOMENTUM_PRESETS",
    "TREND_PRESETS",
    "VALID_PRESETS",
    "VOLATILITY_PRESETS",
    "validate_preset",
]
