"""Model factory for ADK agents — single source of truth for LLM construction.

Two public entry points, by design (Single Responsibility):

    make_model(name)               — build ONE specific LiteLLM-wrapped model.
                                      For tests, scripts, smoke checks.

    make_resilient_model(profile)  — build a fallback chain wrapped as one
                                      model. THIS IS WHAT AGENTS USE.

CONVENTION
==========
Agents MUST use make_resilient_model(profile=...). Never call make_model()
directly from an agent — it bypasses the fallback / paid-toggle mechanism
the project relies on for resilience and cost control.

Adding a new provider:
    1. Add the API key field to config/settings.py + .env.example
    2. Add a (provider_name, getter) entry to _API_KEY_GETTERS below

Adding a new profile:
    1. Add CHAIN_<NAME> + PAID_<NAME> to .env / .env.example
    2. Add a field for each in config/settings.Settings
    3. Add an entry to settings._profile_map
    No changes needed in this file — make_resilient_model is profile-agnostic.
"""
from collections.abc import Callable

from google.adk.models.lite_llm import LiteLlm

from price_predictor.config.settings import settings
from price_predictor.llm.resilient import ResilientModel

# ─────────────────────────────────────────────────────────────
# Provider → API key getter (lazy: only unmasks SecretStr when called)
# ─────────────────────────────────────────────────────────────
_API_KEY_GETTERS: dict[str, Callable[[], str]] = {
    "groq": lambda: settings.groq_api_key.get_secret_value(),
    "gemini": lambda: settings.gemini_api_key.get_secret_value(),
}

# Providers that run LOCALLY and need NO API key -- they need an `api_base`
# (where the local server listens) instead. LiteLLM routes both
# 'ollama/<model>' and 'ollama_chat/<model>' to a local Ollama server.
# Used as the final OFFLINE fallback tier: when every hosted provider
# (Groq, Gemini) is rate-limited, the chain drops to the local model, which
# has no quota and no rate limit. See settings.ollama_api_base.
_KEYLESS_LOCAL_PROVIDERS: frozenset[str] = frozenset({"ollama", "ollama_chat"})


def make_model(model_name: str) -> LiteLlm:
    """Build a LiteLLM-wrapped model for ADK use. ONE specific model only.

    Use this from tests, scripts, or smoke checks — anywhere you need to
    pin a specific model. Agents should use make_resilient_model() instead.

    Args:
        model_name: LiteLLM model string. Examples:
            - "groq/openai/gpt-oss-120b"
            - "gemini/gemini-2.5-flash"
            - "ollama_chat/qwen3:8b"     (local, no key; uses ollama_api_base)

    Returns:
        Configured LiteLlm instance ready to pass to LlmAgent(model=...).

    Raises:
        ValueError: if the provider prefix is not registered in _API_KEY_GETTERS.
    """
    provider = model_name.split("/", 1)[0]

    # Local, keyless providers (Ollama): no API key, but a local api_base
    # pointing at the running Ollama server. This is what lets the resilient
    # chain fall over to an offline model when hosted providers are exhausted.
    if provider in _KEYLESS_LOCAL_PROVIDERS:
        # Disable qwen3-style "thinking": litellm maps reasoning_effort='none'
        # to Ollama's think=False, so the model emits the JSON answer directly
        # instead of narrating its reasoning as prose (which then fails
        # ImpactAssessment / Prediction JSON parsing). See llm/json_extract.py
        # for the defence-in-depth that recovers JSON if a model ignores this.
        return LiteLlm(
            model=model_name,
            api_base=settings.ollama_api_base,
            reasoning_effort="none",
        )

    if provider not in _API_KEY_GETTERS:
        supported = sorted(set(_API_KEY_GETTERS) | _KEYLESS_LOCAL_PROVIDERS)
        raise ValueError(
            f"Unsupported provider: {provider!r}. "
            f"Supported: {supported}. "
            f"Add an API key + getter to factory.py to enable."
        )
    return LiteLlm(model=model_name, api_key=_API_KEY_GETTERS[provider]())


def make_resilient_model(profile: str = "agentic") -> ResilientModel:
    """Build a fallback-chain model for an agent profile. THE DEFAULT FOR AGENTS.

    Resolves the profile to its ordered model chain via settings, builds each
    inner model via make_model(), and wraps them in a ResilientModel that
    transparently falls back on rate-limit / availability errors.

    When settings.use_paid=True, the chain collapses to the profile's single
    paid override — no fallback (paying = no rate limits worth handling).

    Args:
        profile: Profile name registered in settings._profile_map.
                 Currently only "agentic" exists. Add more as the project grows
                 (e.g. "fast" for the critic agent in iteration B).

    Returns:
        ResilientModel that quacks like a single LlmAgent-compatible model
        but internally manages a fallback chain.

    Raises:
        ValueError: unknown profile, empty chain, or unsupported provider.
    """
    chain = settings.effective_chain(profile)
    inner = [make_model(name) for name in chain]
    return ResilientModel(inner_models=inner)
