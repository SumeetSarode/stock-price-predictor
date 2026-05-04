"""technical_agent — multi-cluster technical-analysis agent.

Built across Step C in commits:
    C.1: get_trend tool
    C.2: get_momentum tool (+ candlestick context-gating)
    C.3: get_volatility tool
    C.4: get_levels tool (+ chart pattern integration)
    C.5: LlmAgent wiring  ← we are here
    C.6: Manual smoke test in adk web

ADK CLI (`adk run`, `adk web`) discovers `root_agent` from this module.
Tests and other code import `make_technical_agent` directly for fresh
instances.
"""
from price_predictor.agents.technical_agent.agent import (
    TECHNICAL_AGENT_INSTRUCTION,
    make_technical_agent,
    root_agent,
)

__all__ = [
    "TECHNICAL_AGENT_INSTRUCTION",
    "make_technical_agent",
    "root_agent",
]
