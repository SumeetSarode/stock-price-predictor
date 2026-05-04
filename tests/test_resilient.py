"""Tests for price_predictor.llm.resilient.

Strategy:
    - Build ResilientModel with FAKE inner BaseLlm subclasses (no real LLM).
    - Each fake records its calls + raises on demand to simulate transient
      / structural errors.
    - Verify fallback behavior, cooldown state machine, error classification.

We also test cooldown expiry by manipulating the cooldowns dict directly
(no time-travel libs needed).
"""
from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from pydantic import ConfigDict

from price_predictor.llm.resilient import (
    SHORT_COOLDOWN,
    AllModelsExhaustedError,
    ResilientModel,
    _classify_cooldown,
    _next_midnight_utc,
)


# ─────────────────────────────────────────────────────────────
# Fake BaseLlm — records calls + raises on demand
# ─────────────────────────────────────────────────────────────
class FakeLlm(BaseLlm):
    """Test double for an inner LiteLlm. Records calls; raises if `error` set."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    error: Exception | None = None
    call_count: int = 0
    response_text: str = "ok"

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse]:
        self.call_count += 1
        if self.error is not None:
            raise self.error
        yield LlmResponse()  # empty but valid


def _make_fake(name: str, error: Exception | None = None) -> FakeLlm:
    return FakeLlm(model=name, error=error)


def _make_request() -> LlmRequest:
    """Minimal LlmRequest. Resilient never inspects it — fakes ignore it too."""
    return LlmRequest()


# ─────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────
class TestConstruction:
    def test_requires_at_least_one_model(self):
        with pytest.raises(ValueError, match="at least one"):
            ResilientModel(inner_models=[])

    def test_synthetic_model_name_reflects_chain_length(self):
        m = ResilientModel(inner_models=[_make_fake("a"), _make_fake("b"), _make_fake("c")])
        assert m.model == "resilient[3]"

    def test_starts_with_no_cooldowns(self):
        m = ResilientModel(inner_models=[_make_fake("a")])
        assert m.cooldowns == {}


# ─────────────────────────────────────────────────────────────
# Happy path — first model works
# ─────────────────────────────────────────────────────────────
class TestHappyPath:
    @pytest.mark.asyncio
    async def test_first_model_succeeds_others_not_called(self):
        a = _make_fake("a")
        b = _make_fake("b")
        c = _make_fake("c")
        m = ResilientModel(inner_models=[a, b, c])

        responses = [r async for r in m.generate_content_async(_make_request())]

        assert len(responses) == 1
        assert a.call_count == 1
        assert b.call_count == 0
        assert c.call_count == 0
        assert m.cooldowns == {}, "happy path should not set any cooldown"


# ─────────────────────────────────────────────────────────────
# Fallback on transient errors
# ─────────────────────────────────────────────────────────────
class TestTransientFallback:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "transient_error",
        [
            RateLimitError("rate limited", llm_provider="groq", model="x"),
            ServiceUnavailableError("503", llm_provider="groq", model="x"),
            APIConnectionError(message="conn dropped", llm_provider="groq", model="x"),
            Timeout(message="timed out", llm_provider="groq", model="x"),
        ],
    )
    async def test_falls_back_to_next_on_each_transient_error(self, transient_error):
        a = _make_fake("a", error=transient_error)
        b = _make_fake("b")  # succeeds
        m = ResilientModel(inner_models=[a, b])

        responses = [r async for r in m.generate_content_async(_make_request())]

        assert len(responses) == 1
        assert a.call_count == 1
        assert b.call_count == 1
        assert "a" in m.cooldowns, "failed model must be marked cooled-down"

    @pytest.mark.asyncio
    async def test_falls_back_through_multiple_failures(self):
        err = RateLimitError("rl", llm_provider="groq", model="x")
        a = _make_fake("a", error=err)
        b = _make_fake("b", error=err)
        c = _make_fake("c")  # finally succeeds
        m = ResilientModel(inner_models=[a, b, c])

        responses = [r async for r in m.generate_content_async(_make_request())]

        assert len(responses) == 1
        assert a.call_count == 1
        assert b.call_count == 1
        assert c.call_count == 1
        assert "a" in m.cooldowns
        assert "b" in m.cooldowns
        assert "c" not in m.cooldowns


# ─────────────────────────────────────────────────────────────
# Structural errors NEVER trigger fallback (would mask bugs)
# ─────────────────────────────────────────────────────────────
class TestStructuralRaises:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "structural_error",
        [
            BadRequestError("malformed", model="x", llm_provider="groq"),
            AuthenticationError("bad key", llm_provider="groq", model="x"),
        ],
    )
    async def test_structural_error_raises_without_fallback(self, structural_error):
        a = _make_fake("a", error=structural_error)
        b = _make_fake("b")  # would succeed but should NOT be tried
        m = ResilientModel(inner_models=[a, b])

        with pytest.raises(type(structural_error)):
            async for _ in m.generate_content_async(_make_request()):
                pass

        assert a.call_count == 1
        assert b.call_count == 0, "fallback must NOT happen on structural errors"
        assert m.cooldowns == {}, "structural errors must not set cooldowns"


# ─────────────────────────────────────────────────────────────
# Total exhaustion
# ─────────────────────────────────────────────────────────────
class TestExhaustion:
    @pytest.mark.asyncio
    async def test_all_models_failing_raises_exhausted(self):
        err = RateLimitError("rl", llm_provider="groq", model="x")
        a = _make_fake("a", error=err)
        b = _make_fake("b", error=err)
        m = ResilientModel(inner_models=[a, b])

        with pytest.raises(AllModelsExhaustedError) as exc_info:
            async for _ in m.generate_content_async(_make_request()):
                pass

        assert exc_info.value.chain == ["a", "b"]
        assert isinstance(exc_info.value.last_error, RateLimitError)

    @pytest.mark.asyncio
    async def test_all_in_cooldown_raises_immediately(self):
        """If every model is already cooled-down, fail fast without calling any."""
        a = _make_fake("a")
        b = _make_fake("b")
        m = ResilientModel(inner_models=[a, b])
        # Pre-populate cooldowns to simulate prior failures
        future = datetime.now(UTC) + timedelta(minutes=10)
        m.cooldowns = {"a": future, "b": future}

        with pytest.raises(AllModelsExhaustedError) as exc_info:
            async for _ in m.generate_content_async(_make_request()):
                pass

        assert exc_info.value.last_error is None, "no last_error when nothing was tried"
        assert a.call_count == 0
        assert b.call_count == 0


# ─────────────────────────────────────────────────────────────
# Cooldown state machine
# ─────────────────────────────────────────────────────────────
class TestCooldownStateMachine:
    @pytest.mark.asyncio
    async def test_cooldown_skips_model_on_subsequent_call(self):
        err = RateLimitError("rl", llm_provider="groq", model="x")
        a = _make_fake("a", error=err)
        b = _make_fake("b")
        m = ResilientModel(inner_models=[a, b])

        # First call: a fails, b succeeds, a marked cooled
        async for _ in m.generate_content_async(_make_request()):
            pass

        # Make a healthy now
        a.error = None
        a.call_count = 0
        b.call_count = 0

        # Second call: a is cooled-down → skipped; b called directly
        async for _ in m.generate_content_async(_make_request()):
            pass

        assert a.call_count == 0, "cooled-down model must be skipped"
        assert b.call_count == 1

    @pytest.mark.asyncio
    async def test_expired_cooldown_lets_model_be_tried_again(self):
        a = _make_fake("a")
        b = _make_fake("b")
        m = ResilientModel(inner_models=[a, b])
        # Set a cooldown that already expired
        m.cooldowns = {"a": datetime.now(UTC) - timedelta(seconds=1)}

        async for _ in m.generate_content_async(_make_request()):
            pass

        assert a.call_count == 1, "expired cooldown should let model be retried"
        assert "a" not in m.cooldowns, "expired cooldown should be cleaned up"

    def test_short_cooldown_for_per_minute_rate_limit(self):
        err = RateLimitError("tpm exceeded", llm_provider="groq", model="x")
        result = _classify_cooldown(err)
        assert isinstance(result, timedelta)
        assert result == SHORT_COOLDOWN

    def test_long_cooldown_for_daily_quota(self):
        err = RateLimitError(
            "daily limit exceeded", llm_provider="groq", model="x",
        )
        result = _classify_cooldown(err)
        assert isinstance(result, datetime)
        # Should be a future datetime at midnight UTC
        assert result == _next_midnight_utc()
        assert result.hour == 0 and result.minute == 0 and result.second == 0


# ─────────────────────────────────────────────────────────────
# Order preservation — chain must be tried in the declared order
# ─────────────────────────────────────────────────────────────
class TestChainOrder:
    @pytest.mark.asyncio
    async def test_chain_tried_in_order(self):
        call_log: list[str] = []
        err = RateLimitError("rl", llm_provider="groq", model="x")

        class Recorder(FakeLlm):
            async def generate_content_async(
                self, llm_request: LlmRequest, stream: bool = False,
            ) -> AsyncGenerator[LlmResponse]:
                call_log.append(self.model)
                if self.error is not None:
                    raise self.error
                yield LlmResponse()

        m = ResilientModel(inner_models=[
            Recorder(model="first", error=err),
            Recorder(model="second", error=err),
            Recorder(model="third"),
        ])

        async for _ in m.generate_content_async(_make_request()):
            pass

        assert call_log == ["first", "second", "third"]
