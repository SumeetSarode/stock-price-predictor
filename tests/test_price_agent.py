"""Tests for price_predictor.agents.price_agent.

Strategy:
    - Mock fetch_ohlcv at the boundary (the tool's call site)
    - Verify the tool's response shape, error handling, math, and types
    - Structural smoke for make_price_agent() — no LLM behavior tests

We do NOT test LLM responses here. Those are verified manually via:
    uv run adk run price_predictor.agents.price_agent
"""
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from price_predictor.agents.price_agent import (
    fetch_prices_tool,
    make_price_agent,
    root_agent,
)
from price_predictor.data.prices import PriceFetchError


# ─────────────────────────────────────────────────────────────
# Helper: build a DataFrame in our normalized post-fetch shape
# ─────────────────────────────────────────────────────────────
def _make_normalized_df(rows: int = 5) -> pd.DataFrame:
    """Build a DataFrame matching what fetch_ohlcv() returns AFTER normalization.

    Columns: lowercase OHLCV, no MultiIndex.
    Index: tz-aware Asia/Kolkata.

    Used to mock the inside of fetch_prices_tool — we skip what fetch_ohlcv does
    and just hand back the already-normalized result.
    """
    idx = pd.DatetimeIndex(
        [
            datetime(2024, 1, d, tzinfo=ZoneInfo("Asia/Kolkata"))
            for d in range(1, rows + 1)
        ]
    )
    return pd.DataFrame(
        {
            "open":      [2400.0, 2410.0, 2420.0, 2430.0, 2440.0][:rows],
            "high":      [2450.0, 2460.0, 2470.0, 2480.0, 2490.0][:rows],
            "low":       [2380.0, 2390.0, 2400.0, 2410.0, 2420.0][:rows],
            "close":     [2420.0, 2430.0, 2440.0, 2450.0, 2460.0][:rows],
            "adj_close": [2400.0, 2410.0, 2420.0, 2430.0, 2440.0][:rows],
            "volume":    [1_000_000, 1_100_000, 1_200_000, 1_300_000, 1_400_000][:rows],
        },
        index=idx,
    )


# ─────────────────────────────────────────────────────────────
# Happy paths
# ─────────────────────────────────────────────────────────────
def test_fetch_prices_tool_happy_path_no_bars():
    """Default call returns success with summary, no bars."""
    with patch("price_predictor.agents.price_agent.agent.fetch_ohlcv") as mock_fetch:
        mock_fetch.return_value = _make_normalized_df(rows=5)
        result = fetch_prices_tool("RELIANCE.NS", "2024-01-01", "2024-01-05")

    assert result["status"] == "success"
    assert result["ticker"] == "RELIANCE.NS"
    assert result["rows"] == 5
    assert result["start"] == "2024-01-01"
    assert result["end"] == "2024-01-05"
    assert "bars" not in result, "bars should be omitted when include_bars=False"

    summary = result["summary"]
    assert summary["first_close"] == 2420.0
    assert summary["last_close"] == 2460.0
    assert summary["high"] == 2490.0
    assert summary["low"] == 2380.0


def test_fetch_prices_tool_happy_path_with_bars():
    """include_bars=True adds the full bars list."""
    with patch("price_predictor.agents.price_agent.agent.fetch_ohlcv") as mock_fetch:
        mock_fetch.return_value = _make_normalized_df(rows=3)
        result = fetch_prices_tool(
            "RELIANCE.NS", "2024-01-01", "2024-01-03", include_bars=True
        )

    assert result["status"] == "success"
    assert "bars" in result
    assert len(result["bars"]) == 3

    first_bar = result["bars"][0]
    assert first_bar == {
        "date": "2024-01-01",
        "open": 2400.0,
        "high": 2450.0,
        "low": 2380.0,
        "close": 2420.0,
        "adj_close": 2400.0,
        "volume": 1_000_000,
    }


# ─────────────────────────────────────────────────────────────
# adj_change_pct math
# ─────────────────────────────────────────────────────────────
def test_fetch_prices_tool_adj_change_pct_calculation():
    """adj_change_pct = (last_adj_close - first_adj_close) / first_adj_close * 100."""
    with patch("price_predictor.agents.price_agent.agent.fetch_ohlcv") as mock_fetch:
        # 5-row helper: adj_close = [2400, 2410, 2420, 2430, 2440]
        # Expected: (2440 - 2400) / 2400 * 100 = 1.6666... → rounded to 1.67
        mock_fetch.return_value = _make_normalized_df(rows=5)
        result = fetch_prices_tool("RELIANCE.NS", "2024-01-01", "2024-01-05")

    assert result["summary"]["adj_change_pct"] == 1.67


# ─────────────────────────────────────────────────────────────
# Bar serialization — native Python types, not numpy
# ─────────────────────────────────────────────────────────────
def test_fetch_prices_tool_bars_use_native_python_types():
    """Bars must contain native Python types so the dict is JSON-serializable."""
    with patch("price_predictor.agents.price_agent.agent.fetch_ohlcv") as mock_fetch:
        mock_fetch.return_value = _make_normalized_df(rows=2)
        result = fetch_prices_tool(
            "RELIANCE.NS", "2024-01-01", "2024-01-02", include_bars=True
        )

    bar = result["bars"][0]
    assert type(bar["open"]) is float, f"open should be Python float, got {type(bar['open'])}"
    assert type(bar["volume"]) is int, f"volume should be Python int, got {type(bar['volume'])}"
    assert type(bar["date"]) is str, f"date should be Python str, got {type(bar['date'])}"


# ─────────────────────────────────────────────────────────────
# Error paths
# ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "bad_date",
    ["01-01-2024", "2024/01/01", "Jan 1 2024", "20240101", "not-a-date"],
)
def test_fetch_prices_tool_bad_date_format(bad_date: str):
    """Bad date format returns status:error WITHOUT calling fetch_ohlcv."""
    with patch("price_predictor.agents.price_agent.agent.fetch_ohlcv") as mock_fetch:
        result = fetch_prices_tool("RELIANCE.NS", bad_date, "2024-01-31")

    assert result["status"] == "error"
    assert "Invalid date format" in result["error_message"]
    mock_fetch.assert_not_called(), "fetch_ohlcv should never run on bad dates"


def test_fetch_prices_tool_propagates_value_error():
    """ValueError from fetch_ohlcv (e.g., empty ticker) becomes status:error."""
    with patch("price_predictor.agents.price_agent.agent.fetch_ohlcv") as mock_fetch:
        mock_fetch.side_effect = ValueError("ticker must be a non-empty string, got ''")
        result = fetch_prices_tool("", "2024-01-01", "2024-01-05")

    assert result["status"] == "error"
    assert "non-empty" in result["error_message"]


def test_fetch_prices_tool_propagates_price_fetch_error():
    """PriceFetchError from fetch_ohlcv (e.g., upstream API failure) becomes status:error."""
    with patch("price_predictor.agents.price_agent.agent.fetch_ohlcv") as mock_fetch:
        mock_fetch.side_effect = PriceFetchError("yfinance is rate-limited")
        result = fetch_prices_tool("RELIANCE.NS", "2024-01-01", "2024-01-05")

    assert result["status"] == "error"
    assert "rate-limited" in result["error_message"]
    # Canonical ticker -- no alias to suggest.
    assert "suggested_ticker" not in result


def test_fetch_prices_tool_includes_alias_suggestion_on_error():
    """Regression for live UI bug: user asked about HDFC, got delisted error,
    agent had no way to know to retry HDFCBANK.NS. Tool now surfaces the
    suggestion so the agent can self-recover without user intervention."""
    with patch("price_predictor.agents.price_agent.agent.fetch_ohlcv") as mock_fetch:
        mock_fetch.side_effect = PriceFetchError("No price data: delisted")
        result = fetch_prices_tool("HDFC.NS", "2024-01-01", "2024-01-05")

    assert result["status"] == "error"
    assert result["suggested_ticker"] == "HDFCBANK.NS"
    assert "HDFCBANK" in result["suggestion_reason"]


def test_fetch_prices_tool_no_suggestion_for_unknown_ticker():
    """Don't invent suggestions when there's no known alias.
    Returning a bogus ticker would send the agent on a wild goose chase."""
    with patch("price_predictor.agents.price_agent.agent.fetch_ohlcv") as mock_fetch:
        mock_fetch.side_effect = PriceFetchError("Unknown ticker")
        result = fetch_prices_tool("ZZZZZ.NS", "2024-01-01", "2024-01-05")

    assert result["status"] == "error"
    assert "suggested_ticker" not in result


# ─────────────────────────────────────────────────────────────
# Agent factory — structural smoke (no LLM calls)
# ─────────────────────────────────────────────────────────────
def test_make_price_agent_structure():
    """Factory returns an LlmAgent configured correctly."""
    agent = make_price_agent()
    assert agent.name == "price_agent"
    assert "Indian stock prices" in agent.description
    assert len(agent.tools) == 1
    assert agent.tools[0] is fetch_prices_tool


def test_root_agent_module_level():
    """root_agent must exist at module level for ADK CLI discovery."""
    assert root_agent is not None
    assert root_agent.name == "price_agent"
