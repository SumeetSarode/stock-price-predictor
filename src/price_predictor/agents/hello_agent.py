"""Hello-world ADK agent demonstrating tool use.

Provides a single tool (`get_current_time`) that returns the current time
in a given IANA timezone. The agent decides when to call it based on user input.

This file is the canonical template for how an agent module is structured:
    1. Module docstring
    2. stdlib imports
    3. third-party imports
    4. first-party imports
    5. Tool functions
    6. Agent factory (`make_<name>_agent`)
"""
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from google.adk.agents import LlmAgent

from config.settings import settings
from price_predictor.llm.factory import make_model


# ─────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────
def get_current_time(timezone: str) -> dict:
    """Get the current date and time in the specified IANA timezone.

    Args:
        timezone: IANA timezone name (e.g., "Asia/Kolkata", "UTC",
                  "America/New_York"). If the user does not specify a
                  timezone, use "Asia/Kolkata" (Indian Standard Time).

    Returns:
        A dict with one of two shapes:

          On success:
            {
                "status": "success",
                "timezone": str,        # echoed back for confirmation
                "datetime": str,        # human-readable, e.g. "2026-04-28 22:35:12 IST"
                "iso": str,             # ISO-8601, e.g. "2026-04-28T22:35:12+05:30"
            }

          On error:
            {
                "status": "error",
                "error_message": str,   # human-readable explanation
            }
    """
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        return {
            "status": "error",
            "error_message": (
                f"Unknown timezone: {timezone!r}. "
                "Try IANA names like 'Asia/Kolkata', 'UTC', or 'America/New_York'."
            ),
        }

    now = datetime.now(tz)
    return {
        "status": "success",
        "timezone": timezone,
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "iso": now.isoformat(),
    }


# ─────────────────────────────────────────────────────────────
# Agent factory
# ─────────────────────────────────────────────────────────────
def make_hello_agent() -> LlmAgent:
    """Build the hello-world agent with the get_current_time tool."""
    return LlmAgent(
        name="hello_agent",
        description="A friendly assistant that answers questions about the current time.",
        model=make_model(settings.primary_model),
        instruction=(
            "You are a friendly assistant. When the user asks about the current "
            "time, call the get_current_time tool. If the user does not specify "
            "a timezone, pass timezone='Asia/Kolkata' (Indian Standard Time). "
            "After receiving the tool result, check its 'status' field: if "
            "'success', format the 'datetime' value naturally and conversationally; "
            "if 'error', apologize and use the 'error_message' to help the user."
        ),
        tools=[get_current_time],
    )
