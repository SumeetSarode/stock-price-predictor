"""Hello-world ADK agent — public package interface.

ADK CLI (`adk run`, `adk web`) discovers `root_agent` from this module.
Tests and other code import `get_current_time` and `make_hello_agent` directly.
"""
from price_predictor.agents.hello_agent.agent import (
    get_current_time,
    make_hello_agent,
    root_agent,
)

__all__ = ["get_current_time", "make_hello_agent", "root_agent"]
