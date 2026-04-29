"""Model factory for ADK agents via the LiteLLM adapter.

Single entry point: make_model(model_name). Provider is inferred from the
LiteLLM-style 'provider/model' prefix.

Model selection lives in config (settings.primary_model / settings.secondary_model),
NOT here. This module only knows how to BUILD a model given its name.

To add a new provider:
    1. Add the API key field to config/settings.py + .env.example
    2. Add a (provider_name, getter) entry to _API_KEY_GETTERS below
"""
from collections.abc import Callable

from google.adk.models.lite_llm import LiteLlm

from price_predictor.config.settings import settings

# ─────────────────────────────────────────────────────────────
# Provider → API key getter (lazy: only unmasks SecretStr when called)
# ─────────────────────────────────────────────────────────────
_API_KEY_GETTERS: dict[str, Callable[[], str]] = {
    "groq": lambda: settings.groq_api_key.get_secret_value(),
    "gemini": lambda: settings.gemini_api_key.get_secret_value(),
}


def make_model(model_name: str) -> LiteLlm:
    """Build a LiteLLM-wrapped model for ADK use.

    Provider is auto-detected from the 'provider/model' prefix.

    Args:
        model_name: LiteLLM model string. Examples:
            - "groq/llama-3.3-70b-versatile"
            - "gemini/gemini-2.5-flash"
            - "openai/gpt-4o" (requires OPENAI_API_KEY in settings)

    Returns:
        Configured LiteLlm instance ready to pass to LlmAgent(model=...).

    Raises:
        ValueError: if the provider prefix is not registered in _API_KEY_GETTERS.
    """
    provider = model_name.split("/", 1)[0]
    if provider not in _API_KEY_GETTERS:
        raise ValueError(
            f"Unsupported provider: {provider!r}. "
            f"Supported: {sorted(_API_KEY_GETTERS)}. "
            f"Add an API key + getter to factory.py to enable."
        )
    return LiteLlm(model=model_name, api_key=_API_KEY_GETTERS[provider]())
