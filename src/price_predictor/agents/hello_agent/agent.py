"""Hello-world ADK agent — implementation module.

Provides a single tool (`get_current_time`) that returns the current time
in a given IANA timezone. The agent decides when to call it based on user input.

Structure (ADK CLI convention):
    hello_agent/
        __init__.py    re-exports public names
        agent.py       this file — defines tool, factory, and root_agent
"""
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from google.adk.agents import LlmAgent

from price_predictor.config.settings import settings
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
            "You are a friendly assistant that answers questions about the "
            "current time.\n\n"
            "TOOL USE RULES (follow exactly):\n"
            "- When the user asks about the current time, call the "
            "get_current_time tool using the standard tool-call format. "
            "Do NOT write tool calls as plain text or XML.\n"
            "- The 'timezone' argument MUST be a valid IANA timezone name "
            "like 'Asia/Tokyo', 'America/New_York', 'Europe/London', or 'UTC'. "
            "Convert city / country names to their IANA equivalent before calling. "
            "Examples: 'Tokyo' → 'Asia/Tokyo', 'New York' → 'America/New_York', "
            "'London' → 'Europe/London', 'India' → 'Asia/Kolkata'.\n"
            "- If the user does not specify a location, pass "
            "timezone='Asia/Kolkata' (Indian Standard Time).\n\n"
            "RESPONSE RULES:\n"
            "- After the tool returns, check its 'status' field. If 'success', "
            "format the 'datetime' value naturally and conversationally. "
            "If 'error', apologize and use the 'error_message' to help the user."
        ),
        tools=[get_current_time],
    )


# ─────────────────────────────────────────────────────────────
# ADK CLI entry point
# Module-level instance required by `adk run` / `adk web` / `adk api_server`.
# ADK looks for `root_agent` in <agent_dir>/agent.py or <agent_dir>/__init__.py.
# Tests and other code should still use `make_hello_agent()` for fresh instances.
# ─────────────────────────────────────────────────────────────
root_agent = make_hello_agent()
