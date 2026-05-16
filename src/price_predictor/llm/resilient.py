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
    InternalServerError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from pydantic import ConfigDict, Field

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Error taxonomy: which exceptions trigger fallback?
# ──────────────────────────────────────────────────────────────
# Four buckets, in handling order:
#
# 1. MODEL_INCOMPATIBLE  — BadRequest where the *model* is at fault, not the
#    request. e.g. gpt-oss-120b emits reasoning_content that Groq's input
#    validator then rejects on the next turn; llama-3.3 emits XML-style tool
#    calls that fail Groq's tool_use validator. Other models in the chain
#    will likely succeed. Fall back + cooldown long (this model just doesn't
#    work for this conversation shape).
#
# 2. MODEL_UNAVAILABLE  — provider returned 404: model doesn't exist on this
#    account/tier (e.g. moonshotai/kimi-k2-instruct on a free Groq account).
#    Fall back + LONG cooldown (this won't fix itself this session).
#
# 3. TRANSIENT  — capacity / availability issues. Try the next model with
#    a short or daily cooldown depending on whether it's per-minute vs daily.
#
# 4. STRUCTURAL  — our bug or config issue. Don't fall back — raise
#    immediately. Falling back would mask the real problem AND fail again
#    on the next model with the same broken request.
TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    RateLimitError,            # 429: per-minute or daily quota
    ServiceUnavailableError,   # 503: provider down
    APIConnectionError,        # network blip / DNS / TLS
    Timeout,                   # took too long
    InternalServerError,       # 500: provider hiccup OR LiteLLM's catch-all
                               # for httpx/proxy ConnectError (e.g. DNS
                               # resolution failure for the proxy host).
                               # Same recovery path: try the next model.
)

MODEL_UNAVAILABLE_ERRORS: tuple[type[Exception], ...] = (
    NotFoundError,             # 404: model not on this account/provider
)

STRUCTURAL_ERRORS: tuple[type[Exception], ...] = (
    AuthenticationError,       # 401/403: bad/missing API key
    ContextWindowExceededError,  # input too big — won't shrink for next model
)

# Substrings that indicate "this specific model can't handle this conversation
# shape" rather than "your request is malformed". All come back wrapped in
# litellm.BadRequestError, so we have to string-sniff.
#
# CONVENTION: only add patterns we've actually seen and confirmed. False
# positives here mask real bugs in our request construction.
MODEL_INCOMPATIBILITY_PATTERNS: tuple[str, ...] = (
    "reasoning_content",   # gpt-oss-120b leaks reasoning into history; Groq rejects
    "tool_use_failed",     # llama-3.3 emits XML tool calls; Groq rejects
    "is unsupported",      # generic Groq "feature not supported on this model"
    "does not support",    # generic OpenAI/Gemini "this model lacks feature X"
    "json_validate_failed",  # Groq's structured-output validator rejects the
                             # model's JSON (seen: llama-3.x emitted
                             # "entry_zone": ["2254.942273.06"] -- one string
                             # with both floats smashed together, missing the
                             # comma between array elements). Recovery: fall
                             # back to Gemini, which has better JSON
                             # discipline for nested numeric arrays.
)

# Cooldown durations
SHORT_COOLDOWN = timedelta(seconds=60)    # per-minute rate limit
INCOMPAT_COOLDOWN = timedelta(hours=1)    # model can't handle this conversation


def _next_midnight_utc() -> datetime:
    """Compute the next UTC midnight (when daily quotas reset)."""
    now = datetime.now(UTC)
    tomorrow = now.date() + timedelta(days=1)
    return datetime.combine(tomorrow, time.min, tzinfo=UTC)


def _is_model_incompatibility(error: BadRequestError) -> bool:
    """True if this BadRequest is a model-specific quirk (worth falling back).

    See MODEL_INCOMPATIBILITY_PATTERNS for the substring list and rationale.
    """
    msg = str(error).lower()
    return any(p in msg for p in MODEL_INCOMPATIBILITY_PATTERNS)


def _is_groq_tool_validation_failure(error: Exception) -> bool:
    """True if this is the LiteLLM-MidStreamFallbackError construction bug.

    Failure shape:
        Llama-on-Groq emits a tool call with malformed args -> Groq returns
        `code: "tool_use_failed"` mid-stream -> LiteLLM tries to wrap it as
        a MidStreamFallbackError -> the constructor blindly does
        `int("tool_use_failed")` -> ValueError propagates raw.

    See litellm/exceptions.py:958 (no isdigit() guard around int(status_code)).

    Why we pattern-match a bare ValueError instead of catching the proper
    exception class: the proper class never finishes constructing. The
    ValueError IS the failure -- there's no MidStreamFallbackError to catch.

    Recovery is identical to TRANSIENT: fall back to next model. Gemini
    handles complex tool schemas more reliably than Llama-on-Groq, so the
    fallback typically succeeds where the original call could not.
    """
    if not isinstance(error, ValueError):
        return False
    msg = str(error)
    return "invalid literal for int()" in msg and "tool_use_failed" in msg


def _classify_cooldown(error: Exception) -> timedelta | datetime:
    """Decide cooldown duration for a fallback-triggering error.

    Returns either:
        - timedelta: short / 1h cooldown from now
        - datetime: absolute expiry (daily quota → midnight UTC)
    """
    if isinstance(error, BadRequestError) and _is_model_incompatibility(error):
        return INCOMPAT_COOLDOWN  # 1h: this model fundamentally can't help
    if isinstance(error, MODEL_UNAVAILABLE_ERRORS):
        return INCOMPAT_COOLDOWN  # 1h: model not on this account, won't appear
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

        # ADK's basic.py upstream sets llm_request.model = self.model
        # (= 'resilient[N]'). Inner LiteLlm prefers llm_request.model over
        # its own self.model, which would route the call with our synthetic
        # name into litellm and explode (no provider prefix). Save + restore
        # the original so we don't leak our patched value back to ADK.
        original_request_model = llm_request.model
        last_error: Exception | None = None
        try:
            for model in available:
                try:
                    logger.info("[resilient] trying model=%s", model.model)
                    # Point the request at the REAL inner model name
                    llm_request.model = model.model
                    # Drive the inner generator. We yield as we go -- this means
                    # if a model fails MID-STREAM (after first yield) the error
                    # propagates to the caller (we can't un-yield). Pre-first-yield
                    # failures fall through to the next model. Matches user intent.
                    async for response in model.generate_content_async(llm_request, stream):
                        yield response
                    logger.info("[resilient] success model=%s", model.model)
                    return
                except BadRequestError as e:
                    # Sub-classify: model-specific quirk → fall back; otherwise
                    # genuine bug → raise. (BadRequestError is NOT in
                    # STRUCTURAL_ERRORS for this reason — it needs sub-handling.)
                    if _is_model_incompatibility(e):
                        last_error = e
                        self._set_cooldown(model.model, e)
                        logger.warning(
                            "[resilient] model incompatibility model=%s -- "
                            "this model can't handle this conversation shape, "
                            "falling back. underlying error: %s",
                            model.model, str(e)[:300],
                        )
                        continue
                    logger.error(
                        "[resilient] genuine bad request model=%s -- not falling back",
                        model.model,
                    )
                    raise
                except STRUCTURAL_ERRORS:
                    # Auth / context-window: bug or config issue, fail loud
                    logger.error(
                        "[resilient] structural error model=%s -- not falling back",
                        model.model,
                    )
                    raise
                except MODEL_UNAVAILABLE_ERRORS as e:
                    # 404: model not on this account/tier (e.g. wrong model id
                    # in chain). Fall back + 1h cooldown.
                    last_error = e
                    self._set_cooldown(model.model, e)
                    logger.warning(
                        "[resilient] model unavailable model=%s -- "
                        "not on this account, falling back. underlying error: %s",
                        model.model, str(e)[:300],
                    )
                    continue
                except TRANSIENT_ERRORS as e:
                    last_error = e
                    self._set_cooldown(model.model, e)
                    logger.warning(
                        "[resilient] transient failure model=%s err=%s -- falling back",
                        model.model, type(e).__name__,
                    )
                    continue
                except ValueError as e:
                    # Surgical catch for the LiteLLM MidStreamFallbackError
                    # construction bug (see _is_groq_tool_validation_failure
                    # docstring). Re-raise any OTHER ValueError -- those are
                    # real coding bugs we want to surface, not silently swallow.
                    if not _is_groq_tool_validation_failure(e):
                        raise
                    last_error = e
                    self._set_cooldown(model.model, e)
                    logger.warning(
                        "[resilient] groq tool_use_failed (litellm bug) "
                        "model=%s -- falling back",
                        model.model,
                    )
                    continue

            # Exhausted all available models transiently
            raise AllModelsExhaustedError(chain, last_error)
        finally:
            llm_request.model = original_request_model
