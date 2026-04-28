"""Smoke tests for LLM provider connectivity.

Hits real APIs — marked as `integration` so they can be skipped:
    uv run pytest -m "not integration"
"""
import pytest
from litellm import completion

from config.settings import settings


@pytest.mark.integration
def test_groq_smoke():
    """Verify Groq API key works and we can reach the API."""
    response = completion(
        model="groq/llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say hi"}],
        api_key=settings.groq_api_key.get_secret_value(),
        max_tokens=10,
    )
    text = response.choices[0].message.content
    assert text and text.strip(), f"Empty Groq response: {response}"


@pytest.mark.integration
def test_gemini_smoke():
    """Verify Gemini API key works and we can reach the API."""
    response = completion(
        model="gemini/gemini-2.5-flash",
        messages=[{"role": "user", "content": "Say hi"}],
        api_key=settings.gemini_api_key.get_secret_value(),
        max_tokens=100,
    )
    text = response.choices[0].message.content
    assert text and text.strip(), f"Empty Gemini response: {response}"
