"""Price-fetching ADK agent — implementation module.

Provides one tool (`fetch_prices_tool`) that wraps the underlying
data/prices.fetch_ohlcv() function for LLM consumption.

Structure (ADK CLI convention):
    price_agent/
        __init__.py    re-exports public names
        agent.py       this file -- defines tool, factory, and root_agent
"""
from datetime import datetime

from google.adk.agents import LlmAgent

from price_predictor.config.settings import settings
from price_predictor.data.prices import PriceFetchError, fetch_ohlcv
from price_predictor.llm.factory import make_model


# ─────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────
def fetch_prices_tool(
    ticker: str,
    start_date: str,
    end_date: str,
    include_bars: bool = False,
) -> dict:
    """Fetch historical OHLCV price data for a ticker over a date range.

    Args:
        ticker: Yfinance ticker symbol. NSE stocks REQUIRE the '.NS' suffix
                (e.g., 'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS').
        start_date: Start of range, ISO format YYYY-MM-DD (inclusive).
        end_date: End of range, ISO format YYYY-MM-DD (inclusive).
        include_bars: If True, include the full list of OHLCV bars in the
                      response. Default False -- only summary stats are returned.
                      Set True ONLY if the user needs to inspect specific dates,
                      find chart patterns, or analyze day-by-day action.

    Returns:
        dict with one of two shapes.

        On success:
            {
                "status": "success",
                "ticker": str,
                "rows": int,
                "start": str,
                "end": str,
                "summary": {
                    "first_close": float,
                    "last_close": float,
                    "high": float,
                    "low": float,
                    "adj_change_pct": float,
                },
                "bars": list[dict],  # only present if include_bars=True
            }

        On error:
            {
                "status": "error",
                "error_message": str,
            }
    """
    # ── Parse dates ────────────────────────────────────────
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError as e:
        return {
            "status": "error",
            "error_message": (
                f"Invalid date format: {e}. "
                "Dates must be ISO format YYYY-MM-DD (e.g., '2024-01-31')."
            ),
        }

    # ── Fetch ──────────────────────────────────────────────
    try:
        df = fetch_ohlcv(ticker=ticker, start=start, end=end)
    except (ValueError, PriceFetchError) as e:
        return {"status": "error", "error_message": str(e)}

    # ── Build response ─────────────────────────────────────
    first_close = float(df["close"].iloc[0])
    last_close = float(df["close"].iloc[-1])
    first_adj_close = float(df["adj_close"].iloc[0])
    last_adj_close = float(df["adj_close"].iloc[-1])

    response = {
        "status": "success",
        "ticker": ticker,
        "rows": len(df),
        "start": start_date,
        "end": end_date,
        "summary": {
            "first_close": first_close,
            "last_close": last_close,
            "high": float(df["high"].max()),
            "low": float(df["low"].min()),
            "adj_change_pct": round(
                (last_adj_close - first_adj_close) / first_adj_close * 100, 2
            ),
        },
    }

    if include_bars:
        response["bars"] = [
            {
                "date": ts.strftime("%Y-%m-%d"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "adj_close": float(row["adj_close"]),
                "volume": int(row["volume"]),
            }
            for ts, row in df.iterrows()
        ]

    return response


# ─────────────────────────────────────────────────────────────
# Agent factory
# ─────────────────────────────────────────────────────────────
def make_price_agent() -> LlmAgent:
    """Build the price-fetching agent."""
    return LlmAgent(
        name="price_agent",
        description="An assistant that fetches and analyzes Indian stock prices.",
        model=make_model(settings.primary_model),
        instruction=(
            "You are a helpful assistant that answers questions about Indian "
            "stock prices using the fetch_prices_tool.\n\n"
            "TICKER FORMAT (CRITICAL):\n"
            "- NSE-listed stocks REQUIRE the '.NS' suffix.\n"
            "- Examples: 'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS'.\n"
            "- If the user says 'Reliance', call the tool with 'RELIANCE.NS'.\n"
            "- For US/foreign stocks, use the bare ticker (e.g., 'AAPL').\n\n"
            "DATE FORMAT:\n"
            "- All dates MUST be ISO format: YYYY-MM-DD.\n"
            "- 'last week' = compute the actual dates yourself based on today.\n"
            "- 'January 2024' = start_date='2024-01-01', end_date='2024-01-31'.\n\n"
            "INCLUDE_BARS RULE:\n"
            "- Default include_bars=False -- for trend / range / general questions.\n"
            "- Set include_bars=True ONLY if the user asks about:\n"
            "  - A specific date's price ('what was the close on Jan 15?')\n"
            "  - Day-by-day analysis or chart pattern detection.\n"
            "- Never set True 'just to be safe' -- it bloats the response.\n\n"
            "RESPONSE RULES:\n"
            "- After the tool returns, check the 'status' field.\n"
            "- On 'success': summarize the result conversationally. Mention\n"
            "  ticker, date range, and key numbers from 'summary'. Use the\n"
            "  Rupee symbol (Rs) for NSE prices and dollar ($) for US prices.\n"
            "- On 'error': apologize and explain the error_message clearly."
        ),
        tools=[fetch_prices_tool],
    )


# ─────────────────────────────────────────────────────────────
# ADK CLI entry point
# Module-level instance required by `adk run` / `adk web` / `adk api_server`.
# ─────────────────────────────────────────────────────────────
root_agent = make_price_agent()
