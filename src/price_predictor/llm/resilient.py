"""Resilient model wrapper: ordered fallback across a chain of LiteLLM models.

Wraps ADK's `BaseLlm` interface so it's a drop-in replacement for any model.
On transient errors (rate limit, provider outage, timeout) it tries the next
model in the chain. On structural errors (bad request, auth) it raises
immediately — those are bugs, not capacity issues, and fallback would mask them.

State (cooldowns) is per-instance and in-memory. A new instance starts fresh.
That's intentional: simple, no extra deps, fits the 'one wrapper per agent
process' usage pattern. If you ever need cross-process persistence, swap the
`_cooldowns` dict for a Redis client behind the same interface.

USAGE
=====
Don't construct this directly — use `make_resilient_model(profile=...)` from
`llm.factory`. The factory ensures consistent chain resolution across agents.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, time, timedelta

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    ContextWindowExceededError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from pydantic import ConfigDict, Field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Error taxonomy: which exceptions trigger fallback?
# ─────────────────────────────────────────────────────────────
# TRANSIENT = capacity / availability issues. Try the next model.
TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    RateLimitError,            # 429: per-minute or daily quota
    ServiceUnavailableError,   # 503: provider down
    APIConnectionError,        # network blip / DNS / TLS
    Timeout,                   # took too long
)

# STRUCTURAL = our bug or config issue. Don't fallback — raise immediately.
# Falling back would mask the real problem AND fail again on the next model.
STRUCTURAL_ERRORS: tuple[type[Exception], ...] = (
    BadRequestError,           # 400: malformed request
    AuthenticationError,       # 401/403: bad/missing API key
    ContextWindowExceededError,  # input too big — won't get smaller for next model
)

# Cooldown durations
SHORT_COOLDOWN = timedelta(seconds=60)   # per-minute rate limit


def _next_midnight_utc() -> datetime:
    """Compute the next UTC midnight (when daily quotas reset)."""
    now = datetime.now(UTC)
    tomorrow = now.date() + timedelta(days=1)
    return datetime.combine(tomorrow, time.min, tzinfo=UTC)


def _classify_cooldown(error: Exception) -> timedelta | datetime:
    """Decide cooldown duration for a transient error.

    Returns either:
        - timedelta: short cooldown from now (per-minute rate limit)
        - datetime: absolute expiry (daily quota exhausted, reset at midnight UTC)
    """
    msg = str(error).lower()
    # LiteLLM surfaces quota-exhausted with phrases like 'daily limit',
    # 'quota exceeded', 'tokens per day'. Per-minute uses 'rate_limit', 'tpm'.
    if any(phrase in msg for phrase in ("daily", "per day", "quota", "tpd", "rpd")):
        return _next_midnight_utc()
    return SHORT_COOLDOWN


# ─────────────────────────────────────────────────────────────
# Custom exception when the entire chain is exhausted
# ─────────────────────────────────────────────────────────────
class AllModelsExhaustedError(RuntimeError):
    """Raised when every model in the chain is cooled-down or has failed.

    Carries the last underlying error so callers can inspect what actually
    went wrong (vs. just knowing 'everything failed').
    """

    def __init__(self, chain: list[str], last_error: Exception | None) -> None:
        self.chain = chain
        self.last_error = last_error
        super().__init__(
            f"All {len(chain)} models in chain exhausted. "
            f"Chain: {chain}. Last error: {last_error!r}"
        )


# ─────────────────────────────────────────────────────────────
# The wrapper
# ─────────────────────────────────────────────────────────────
class ResilientModel(BaseLlm):
    """ADK BaseLlm that fans out to a chain of underlying models on failure.

    Tries each model in order. On a transient error, marks it cooled-down
    (60s for rate limits, until midnight UTC for daily quotas) and tries
    the next. On a structural error (bad request / auth), raises immediately.

    The `model` field (inherited from BaseLlm) is set to a synthetic name
    `'resilient[N]'` for ADK's logging/identification — actual calls dispatch
    to the wrapped models.
    """

    # Pydantic v2 config: allow non-Pydantic types (the wrapped models)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Wrapped models, ordered by preference
    inner_models: list[BaseLlm] = Field(default_factory=list)
    # name -> cooldown_expires_at (in-memory state)
    cooldowns: dict[str, datetime] = Field(default_factory=dict)

    def __init__(self, inner_models: list[BaseLlm], **kwargs) -> None:
        if not inner_models:
            raise ValueError("ResilientModel requires at least one inner model.")
        # Synthetic name shown in ADK logs
        synthetic_name = f"resilient[{len(inner_models)}]"
        super().__init__(
            model=synthetic_name,
            inner_models=inner_models,
            cooldowns={},
            **kwargs,
        )

    # ─────────────────────────────────────────────────────────
    # Cooldown helpers
    # ─────────────────────────────────────────────────────────
    def _is_available(self, model_name: str) -> bool:
        """True if `model_name` is not currently in cooldown."""
        expiry = self.cooldowns.get(model_name)
        if expiry is None:
            return True
        if datetime.now(UTC) >= expiry:
            # Cooldown expired — clean up so dict doesn't grow forever
            del self.cooldowns[model_name]
            return True
        return False

    def _set_cooldown(self, model_name: str, error: Exception) -> None:
        """Mark `model_name` cooled down based on the error's signal."""
        cooldown = _classify_cooldown(error)
        expiry = datetime.now(UTC) + cooldown if isinstance(cooldown, timedelta) else cooldown
        self.cooldowns[model_name] = expiry
        logger.warning(
            "[resilient] cooldown set: model=%s until=%s reason=%s",
            model_name, expiry.isoformat(), type(error).__name__,
        )

    def _available_models(self) -> list[BaseLlm]:
        """Return inner models that are NOT currently cooled-down, in order."""
        return [m for m in self.inner_models if self._is_available(m.model)]

    # ─────────────────────────────────────────────────────────
    # Main entry point — ADK calls this for every LLM turn
    # ─────────────────────────────────────────────────────────
    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse]:
        """Try each available model in order until one succeeds.

        On transient errors, sets cooldown and tries next.
        On structural errors, re-raises immediately (don't mask bugs).
        On total exhaustion, raises AllModelsExhaustedError.
        """
        chain = [m.model for m in self.inner_models]
        available = self._available_models()
        if not available:
            raise AllModelsExhaustedError(chain, last_error=None)

        last_error: Exception | None = None
        for model in available:
            try:
                logger.info("[resilient] trying model=%s", model.model)
                # Drive the inner generator. We yield as we go — this means
                # if a model fails MID-STREAM (after first yield) the error
                # propagates to the caller (we can't un-yield). Pre-first-yield
                # failures fall through to the next model. This matches user
                # intent: 'transparent fallback when possible.'
                async for response in model.generate_content_async(llm_request, stream):
                    yield response
                logger.info("[resilient] success model=%s", model.model)
                return
            except STRUCTURAL_ERRORS:
                # Bug / config issue — fail loud, don't try more models
                logger.error("[resilient] structural error model=%s — not falling back", model.model)
                raise
            except TRANSIENT_ERRORS as e:
                last_error = e
                self._set_cooldown(model.model, e)
                logger.warning(
                    "[resilient] transient failure model=%s err=%s — falling back",
                    model.model, type(e).__name__,
                )
                continue

        # If we exhausted the loop, every available model failed transiently
        raise AllModelsExhaustedError(chain, last_error)
