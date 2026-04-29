"""Unit tests for hello_agent's get_current_time tool.

These tests cover the pure function only — no LLM, no Runner.
Agent integration tests (which DO hit the LLM) will live in
test_hello_agent_e2e.py and be marked @pytest.mark.integration.
"""
import pytest

from price_predictor.agents.hello_agent import get_current_time


def test_get_current_time_success():
    """Valid timezone returns a success dict with all expected fields."""
    result = get_current_time("Asia/Kolkata")

    assert isinstance(result, dict), f"Expected dict, got {type(result).__name__}"
    assert result["status"] == "success", f"Expected success, got: {result}"
    assert result["timezone"] == "Asia/Kolkata", "Timezone should be echoed back unchanged"
    assert isinstance(result["datetime"], str) and result["datetime"], (
        f"Expected non-empty datetime string, got: {result.get('datetime')!r}"
    )
    assert isinstance(result["iso"], str) and result["iso"], (
        f"Expected non-empty iso string, got: {result.get('iso')!r}"
    )


def test_get_current_time_invalid_timezone():
    """Invalid timezone returns an error dict (does NOT raise)."""
    result = get_current_time("Mars/Olympus")

    assert isinstance(result, dict), f"Expected dict, got {type(result).__name__}"
    assert result["status"] == "error", f"Expected error, got: {result}"
    assert isinstance(result["error_message"], str) and result["error_message"], (
        "Expected non-empty error_message"
    )
    assert "Mars/Olympus" in result["error_message"], (
        f"Error should mention the bad timezone for debuggability, got: {result['error_message']}"
    )


@pytest.mark.parametrize(
    "timezone",
    ["Asia/Kolkata", "UTC", "America/New_York", "Europe/London"],
)
def test_get_current_time_multiple_timezones(timezone: str):
    """Multiple valid IANA timezones all return success with timezone echoed back."""
    result = get_current_time(timezone)

    assert result["status"] == "success", f"Failed for {timezone}: {result}"
    assert result["timezone"] == timezone, f"Timezone should be echoed back for {timezone}"
