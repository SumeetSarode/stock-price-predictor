"""Tests for price_predictor.data.schema.OHLCVBar.

Covers:
    - Valid bar constructs cleanly + round-trips through JSON
    - Each field-level constraint (gt=0 / ge=0) rejects bad input
    - The model_validator catches all 3 high/low consistency violations
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from price_predictor.data.schema import OHLCVBar


# ─────────────────────────────────────────────────────────────
# Helper: a known-valid bar (used as the base for negative tests)
# ─────────────────────────────────────────────────────────────
def _valid_bar_kwargs() -> dict:
    """Baseline kwargs for a valid bar. Override one field to test violations."""
    return {
        "timestamp": datetime(2024, 1, 1, tzinfo=ZoneInfo("Asia/Kolkata")),
        "open": 2400.0,
        "high": 2450.0,
        "low": 2380.0,
        "close": 2420.0,
        "adj_close": 2400.0,
        "volume": 1_000_000,
    }


# ─────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────
def test_ohlcv_bar_valid_construction():
    """A bar with all valid fields constructs without error."""
    bar = OHLCVBar(**_valid_bar_kwargs())
    assert bar.open == 2400.0
    assert bar.close == 2420.0
    assert bar.volume == 1_000_000
    assert str(bar.timestamp.tzinfo) == "Asia/Kolkata"


def test_ohlcv_bar_json_round_trip():
    """Bar -> JSON -> Bar produces an equal object."""
    original = OHLCVBar(**_valid_bar_kwargs())
    json_str = original.model_dump_json()
    rebuilt = OHLCVBar.model_validate_json(json_str)
    assert rebuilt == original


# ─────────────────────────────────────────────────────────────
# Field-level constraints (gt=0 / ge=0)
# ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("field", ["open", "high", "low", "close", "adj_close"])
def test_ohlcv_bar_negative_price_rejected(field: str):
    """Any negative price field is rejected (gt=0)."""
    kwargs = _valid_bar_kwargs()
    kwargs[field] = -1.0
    with pytest.raises(ValidationError) as exc_info:
        OHLCVBar(**kwargs)
    assert field in str(exc_info.value)


@pytest.mark.parametrize("field", ["open", "high", "low", "close", "adj_close"])
def test_ohlcv_bar_zero_price_rejected(field: str):
    """Zero is rejected for prices (gt=0, not ge=0)."""
    kwargs = _valid_bar_kwargs()
    kwargs[field] = 0.0
    with pytest.raises(ValidationError):
        OHLCVBar(**kwargs)


def test_ohlcv_bar_negative_volume_rejected():
    """Volume must be >= 0 (negative is nonsense)."""
    kwargs = _valid_bar_kwargs()
    kwargs["volume"] = -1
    with pytest.raises(ValidationError) as exc_info:
        OHLCVBar(**kwargs)
    assert "volume" in str(exc_info.value)


def test_ohlcv_bar_zero_volume_allowed():
    """Volume of 0 is valid (trading-halt day, illiquid stock)."""
    kwargs = _valid_bar_kwargs()
    kwargs["volume"] = 0
    bar = OHLCVBar(**kwargs)
    assert bar.volume == 0


# ─────────────────────────────────────────────────────────────
# High/low consistency (model_validator)
# ─────────────────────────────────────────────────────────────
def test_ohlcv_bar_high_below_low_rejected():
    """high < low is structurally impossible -- caught by validator."""
    kwargs = _valid_bar_kwargs()
    kwargs["high"] = 100.0
    kwargs["low"] = 200.0
    with pytest.raises(ValidationError) as exc_info:
        OHLCVBar(**kwargs)
    assert "high" in str(exc_info.value) and "low" in str(exc_info.value)


def test_ohlcv_bar_high_below_open_rejected():
    """high must be >= open."""
    kwargs = _valid_bar_kwargs()
    kwargs["high"] = 2400.0   # equal to open
    kwargs["open"] = 2500.0   # now open > high
    with pytest.raises(ValidationError):
        OHLCVBar(**kwargs)


def test_ohlcv_bar_low_above_close_rejected():
    """low must be <= close."""
    kwargs = _valid_bar_kwargs()
    kwargs["low"] = 2500.0    # higher than close (2420)
    with pytest.raises(ValidationError):
        OHLCVBar(**kwargs)
