"""News-impact analyzer ADK agent — public package interface.

ADK CLI (`adk run`, `adk web`) discovers `root_agent` from this module.
Tests and other code import the tools / schemas / factory directly.
"""
from price_predictor.agents.news_impact.agent import (
    Catalyst,
    ImpactAssessment,
    fetch_estimates_tool,
    fetch_recent_filings_tool,
    fetch_recent_news_tool,
    fetch_recent_prices_tool,
    make_news_impact_agent,
    root_agent,
)

__all__ = [
    "Catalyst",
    "ImpactAssessment",
    "fetch_estimates_tool",
    "fetch_recent_filings_tool",
    "fetch_recent_news_tool",
    "fetch_recent_prices_tool",
    "make_news_impact_agent",
    "root_agent",
]
