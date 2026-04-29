"""Price-fetching ADK agent — public package interface.

ADK CLI (`adk run`, `adk web`) discovers `root_agent` from this module.
Tests and other code import `fetch_prices_tool` and `make_price_agent` directly.
"""
from price_predictor.agents.price_agent.agent import (
    fetch_prices_tool,
    make_price_agent,
    root_agent,
)

__all__ = ["fetch_prices_tool", "make_price_agent", "root_agent"]
