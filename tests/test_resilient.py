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
    InternalServerError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from pydantic import ConfigDict

from price_predictor.llm.resilient import (
    INCOMPAT_COOLDOWN,
    SHORT_COOLDOWN,
    AllModelsExhaustedError,
    ResilientModel,
    _classify_cooldown,
    _is_model_availability_400,
    _is_model_incompatibility,
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
            # Regression: live bug -- httpx ConnectError (DNS failure through
            # corp proxy) gets wrapped by LiteLLM as InternalServerError.
            # Used to bubble up uncaught -> 500 in UI -> never tried Gemini.
            InternalServerError(
                "GroqException - [Errno 8] nodename nor servname provided",
                llm_provider="groq", model="x",
            ),
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

    @pytest.mark.asyncio
    async def test_ollama_tail_catches_when_all_hosted_rate_limited(self):
        """Mirror the REAL chain: Gemini + both Groq models rate-limited,
        local Ollama tail (no quota) catches and succeeds.

        This is the exact scenario the offline fallback exists for: every
        hosted provider is 429'd, and the pipeline still returns an answer
        instead of raising AllModelsExhaustedError.
        """
        def rl(provider: str) -> RateLimitError:
            return RateLimitError("429", llm_provider=provider, model="x")

        gemini = _make_fake("gemini/gemini-2.5-flash", error=rl("gemini"))
        groq1 = _make_fake("groq/openai/gpt-oss-120b", error=rl("groq"))
        groq2 = _make_fake("groq/llama-3.3-70b-versatile", error=rl("groq"))
        ollama = _make_fake("ollama_chat/qwen3:8b")  # local, succeeds
        m = ResilientModel(inner_models=[gemini, groq1, groq2, ollama])

        responses = [r async for r in m.generate_content_async(_make_request())]

        assert len(responses) == 1, "Ollama tail should have produced a response"
        # all three hosted models were tried and cooled down
        assert gemini.call_count == 1
        assert groq1.call_count == 1
        assert groq2.call_count == 1
        # the local tail was reached and succeeded (never cooled down)
        assert ollama.call_count == 1
        assert "ollama_chat/qwen3:8b" not in m.cooldowns


# ──────────────────────────────────────────────────────────────
# Structural errors NEVER trigger fallback (would mask bugs)
# ──────────────────────────────────────────────────────────────
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

    @pytest.mark.asyncio
    async def test_unrelated_value_error_still_raises(self):
        """We catch ValueError ONLY for the Groq tool_use_failed bug.
        Any other ValueError must propagate -- it's a real coding bug,
        not something to silently retry."""
        a = _make_fake("a", error=ValueError("something completely unrelated"))
        b = _make_fake("b")
        m = ResilientModel(inner_models=[a, b])

        with pytest.raises(ValueError, match="completely unrelated"):
            async for _ in m.generate_content_async(_make_request()):
                pass

        assert b.call_count == 0, "unrelated ValueError must NOT trigger fallback"


# ──────────────────────────────────────────────────────────────
# Groq tool_use_failed -- LiteLLM construction-time bug.
# When Llama-on-Groq emits malformed tool calls, Groq returns
# code='tool_use_failed' which crashes litellm/exceptions.py:958
# while it tries to build a MidStreamFallbackError. We catch the
# resulting bare ValueError and fall back to the next model.
# ──────────────────────────────────────────────────────────────
class TestGroqToolUseFailed:
    @pytest.mark.asyncio
    async def test_groq_tool_use_failed_falls_back(self):
        """Live regression: agent crashed mid-stream with
        ValueError("invalid literal for int() with base 10: 'tool_use_failed'")
        because litellm's MidStreamFallbackError ctor blindly does
        int(status_code). Our resilient layer must recognize this exact
        shape and fall back to Gemini (which handles the schema better)."""
        groq_bug = ValueError(
            "invalid literal for int() with base 10: 'tool_use_failed'"
        )
        a = _make_fake("groq/llama", error=groq_bug)
        b = _make_fake("gemini/gemini-2.5-flash")  # should succeed on fallback
        m = ResilientModel(inner_models=[a, b])

        events = [ev async for ev in m.generate_content_async(_make_request())]

        assert a.call_count == 1
        assert b.call_count == 1, "must fall back to Gemini after Groq bug"
        assert events, "Gemini fallback should have produced events"
        # Cooldown applied so we don't immediately retry the broken model
        assert "groq/llama" in m.cooldowns


# ──────────────────────────────────────────────────────────────
# Model incompatibility — BadRequest with model-specific quirk patterns
# falls back instead of raising (the bug we just fixed for gpt-oss-120b)
# ──────────────────────────────────────────────────────────────
class TestModelIncompatibility:
    """Some 400s aren't bugs in OUR code — they're model-specific quirks.

    Examples (substring → cause):
        - 'reasoning_content'    : gpt-oss-120b leaks reasoning into history,
                                   Groq rejects on next turn.
        - 'tool_use_failed'      : llama-3.3 emits XML tool calls, not JSON.
        - 'is unsupported'       : feature-not-supported on this Groq model.
        - 'does not support'     : feature-not-supported elsewhere.
        - 'json_validate_failed' : Groq's structured-output validator rejects
                                   the model's JSON (e.g. llama-3.x emitted
                                   '"entry_zone": ["2254.942273.06"]' — one
                                   string with both floats smashed together).

    These should fall back (next model may handle it) with a LONG cooldown
    (this model fundamentally can't help, no point retrying soon).
    """

    def test_detects_known_incompatibility_substrings(self):
        for substring in [
            "property 'reasoning_content' is unsupported",
            "tool_use_failed: model emitted invalid xml",
            "feature is unsupported on this model",
            "this model does not support tool calling",
            # Regression: the exact Groq error shape from the TCS.NS prod
            # failure on 2026-05-16. Llama-on-Groq smashed the two
            # entry_zone floats into a single string with no comma.
            "GroqException - {\"error\":{\"message\":\"Generated JSON does "
            "not match the expected schema.\",\"type\":\"invalid_request_"
            "error\",\"code\":\"json_validate_failed\"}}",
        ]:
            err = BadRequestError(substring, model="x", llm_provider="groq")
            assert _is_model_incompatibility(err), f"should detect: {substring!r}"

    def test_does_not_detect_genuine_bad_requests(self):
        for substring in [
            "messages array is empty",
            "invalid json in tools parameter",
            "max_tokens must be positive",
        ]:
            err = BadRequestError(substring, model="x", llm_provider="groq")
            assert not _is_model_incompatibility(err), f"false positive: {substring!r}"

    @pytest.mark.asyncio
    async def test_incompatibility_falls_back_to_next_model(self):
        """The exact bug from the news_impact agent log."""
        err = BadRequestError(
            "GroqException - 'messages.2': property 'reasoning_content' is unsupported",
            model="groq/openai/gpt-oss-120b",
            llm_provider="groq",
        )
        a = _make_fake("groq/openai/gpt-oss-120b", error=err)
        b = _make_fake("gemini/gemini-2.5-flash")  # should be tried
        m = ResilientModel(inner_models=[a, b])

        responses = [r async for r in m.generate_content_async(_make_request())]

        assert len(responses) == 1
        assert a.call_count == 1
        assert b.call_count == 1, "incompatibility must trigger fallback"
        assert "groq/openai/gpt-oss-120b" in m.cooldowns

    @pytest.mark.asyncio
    async def test_incompatibility_sets_long_cooldown(self):
        """1h cooldown, not 60s -- this model can't help anytime soon."""
        err = BadRequestError(
            "reasoning_content is unsupported",
            model="x", llm_provider="groq",
        )
        a = _make_fake("a", error=err)
        b = _make_fake("b")
        m = ResilientModel(inner_models=[a, b])

        before = datetime.now(UTC)
        async for _ in m.generate_content_async(_make_request()):
            pass
        after = datetime.now(UTC)

        cooldown_expiry = m.cooldowns["a"]
        # Should be roughly now + INCOMPAT_COOLDOWN (1h), not now + 60s
        expected_min = before + INCOMPAT_COOLDOWN - timedelta(seconds=1)
        expected_max = after + INCOMPAT_COOLDOWN + timedelta(seconds=1)
        assert expected_min <= cooldown_expiry <= expected_max, (
            f"Expected ~1h cooldown, got expiry={cooldown_expiry}"
        )

    @pytest.mark.asyncio
    async def test_genuine_bad_request_still_raises(self):
        """Don't mask actual bugs in our request construction."""
        err = BadRequestError(
            "messages array cannot be empty",  # NOT a known incompat pattern
            model="x", llm_provider="groq",
        )
        a = _make_fake("a", error=err)
        b = _make_fake("b")  # must NOT be tried
        m = ResilientModel(inner_models=[a, b])

        with pytest.raises(BadRequestError, match="messages array"):
            async for _ in m.generate_content_async(_make_request()):
                pass

        assert a.call_count == 1
        assert b.call_count == 0
        assert m.cooldowns == {}, "genuine bad requests must not set cooldowns"

    def test_classify_cooldown_for_incompatibility_returns_long(self):
        err = BadRequestError(
            "reasoning_content not supported",
            model="x", llm_provider="groq",
        )
        result = _classify_cooldown(err)
        assert result == INCOMPAT_COOLDOWN
        assert result > SHORT_COOLDOWN, "incompat cooldown must outlast rate-limit"

    # ── Model-availability 400 (the gemini-flash-latest bug) ──────────
    def test_detects_gemini_availability_400_message(self):
        """The literal Gemini 400 that killed predictions must be detected."""
        err = BadRequestError(
            "litellm.BadRequestError: GeminiException - models/gemini-flash-latest "
            "is not found for API version v1beta, or is not supported for "
            "generateContent.",
            model="gemini/gemini-flash-latest", llm_provider="gemini",
        )
        assert _is_model_availability_400(err)
        # and it must NOT be mistaken for a conversation-shape incompatibility
        assert not _is_model_incompatibility(err)

    def test_availability_400_is_not_a_genuine_bug(self):
        """A real malformed-request 400 must NOT match availability patterns."""
        err = BadRequestError(
            "messages array cannot be empty",
            model="x", llm_provider="groq",
        )
        assert not _is_model_availability_400(err)

    @pytest.mark.asyncio
    async def test_gemini_availability_400_falls_back_and_succeeds(self):
        """EXACT repro of the user's bug: bad primary Gemini model must NOT
        kill the prediction -- it must fall back to Groq and succeed."""
        err = BadRequestError(
            "GeminiException - models/gemini-flash-latest is not found for API "
            "version v1beta, or is not supported for generateContent.",
            model="gemini/gemini-flash-latest", llm_provider="gemini",
        )
        primary = _make_fake("gemini/gemini-flash-latest", error=err)
        fallback = _make_fake("groq/openai/gpt-oss-120b")  # must be tried
        m = ResilientModel(inner_models=[primary, fallback])

        responses = [r async for r in m.generate_content_async(_make_request())]

        assert len(responses) == 1, "prediction must still succeed via fallback"
        assert primary.call_count == 1
        assert fallback.call_count == 1, "availability 400 must trigger fallback"
        assert "gemini/gemini-flash-latest" in m.cooldowns

    def test_classify_cooldown_for_availability_400_returns_long(self):
        err = BadRequestError(
            "is not found for API version v1beta",
            model="gemini/gemini-flash-latest", llm_provider="gemini",
        )
        result = _classify_cooldown(err)
        assert result == INCOMPAT_COOLDOWN


# ──────────────────────────────────────────────────────────────
# Model unavailable — NotFoundError (404: model not on this account/tier)
# falls back instead of bubbling up uncaught (the bug we just fixed for
# moonshotai/kimi-k2-instruct on a free Groq account).
# ──────────────────────────────────────────────────────────────
class TestModelUnavailable:
    """Provider returned 404 'model not found / no access on this account'.

    Different from incompatibility (which is a 400 about conversation shape).
    Same handling though: fall back + LONG cooldown (this won't fix itself).
    """

    @pytest.mark.asyncio
    async def test_not_found_falls_back_to_next_model(self):
        """The exact bug: Kimi-K2 returns 404 on free Groq tier."""
        err = NotFoundError(
            "The model `moonshotai/kimi-k2-instruct` does not exist or you do not have access to it.",
            model="groq/moonshotai/kimi-k2-instruct",
            llm_provider="groq",
        )
        a = _make_fake("groq/moonshotai/kimi-k2-instruct", error=err)
        b = _make_fake("gemini/gemini-2.5-flash")  # should be tried
        m = ResilientModel(inner_models=[a, b])

        responses = [r async for r in m.generate_content_async(_make_request())]

        assert len(responses) == 1
        assert a.call_count == 1
        assert b.call_count == 1, "NotFoundError must trigger fallback, not raise"
        assert "groq/moonshotai/kimi-k2-instruct" in m.cooldowns

    @pytest.mark.asyncio
    async def test_not_found_sets_long_cooldown(self):
        """1h cooldown -- model won't appear on the account this session."""
        err = NotFoundError("model_not_found", model="x", llm_provider="groq")
        a = _make_fake("a", error=err)
        b = _make_fake("b")
        m = ResilientModel(inner_models=[a, b])

        before = datetime.now(UTC)
        async for _ in m.generate_content_async(_make_request()):
            pass
        after = datetime.now(UTC)

        cooldown_expiry = m.cooldowns["a"]
        expected_min = before + INCOMPAT_COOLDOWN - timedelta(seconds=1)
        expected_max = after + INCOMPAT_COOLDOWN + timedelta(seconds=1)
        assert expected_min <= cooldown_expiry <= expected_max, (
            f"Expected ~1h cooldown for NotFoundError, got expiry={cooldown_expiry}"
        )

    def test_classify_cooldown_for_not_found_returns_long(self):
        err = NotFoundError("model_not_found", model="x", llm_provider="groq")
        result = _classify_cooldown(err)
        assert result == INCOMPAT_COOLDOWN
        assert result > SHORT_COOLDOWN


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


# ──────────────────────────────────────────────────────────────
# Order preservation — chain must be tried in the declared order
# ──────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────
# llm_request.model patching — critical for ADK + LiteLlm interop
# ──────────────────────────────────────────────────────────────
class TestLlmRequestModelPatching:
    """ADK upstream sets llm_request.model = our wrapper's synthetic name.

    LiteLlm reads llm_request.model first, falling back to self.model only if
    empty -- so without patching, our 'resilient[N]' name leaks into litellm
    and explodes ('LLM Provider NOT provided').

    The wrapper MUST overwrite llm_request.model with the real inner-model
    name before each delegated call, AND restore it on exit so we don't
    leak our patched value back to ADK.
    """

    @pytest.mark.asyncio
    async def test_inner_model_sees_its_own_name_in_request(self):
        """On delegation, llm_request.model must equal the inner model's name."""
        seen_models: list[str] = []

        class Inspector(FakeLlm):
            async def generate_content_async(
                self, llm_request: LlmRequest, stream: bool = False,
            ) -> AsyncGenerator[LlmResponse]:
                seen_models.append(llm_request.model)
                yield LlmResponse()

        m = ResilientModel(inner_models=[Inspector(model="groq/x/y")])
        req = _make_request()
        req.model = "resilient[1]"  # mimic ADK's upstream patching

        async for _ in m.generate_content_async(req):
            pass

        assert seen_models == ["groq/x/y"], (
            "Inner model must see its OWN name in llm_request.model, not 'resilient[N]'"
        )

    @pytest.mark.asyncio
    async def test_each_fallback_sees_its_own_name(self):
        """On fallback, the next inner sees ITS name (not the failed one's)."""
        seen_models: list[str] = []
        err = RateLimitError("rl", llm_provider="groq", model="x")

        class Inspector(FakeLlm):
            async def generate_content_async(
                self, llm_request: LlmRequest, stream: bool = False,
            ) -> AsyncGenerator[LlmResponse]:
                seen_models.append(llm_request.model)
                if self.error is not None:
                    raise self.error
                yield LlmResponse()

        m = ResilientModel(inner_models=[
            Inspector(model="groq/a", error=err),
            Inspector(model="gemini/b"),
        ])
        req = _make_request()
        req.model = "resilient[2]"

        async for _ in m.generate_content_async(req):
            pass

        assert seen_models == ["groq/a", "gemini/b"]

    @pytest.mark.asyncio
    async def test_request_model_restored_after_success(self):
        m = ResilientModel(inner_models=[_make_fake("groq/x")])
        req = _make_request()
        req.model = "resilient[1]"

        async for _ in m.generate_content_async(req):
            pass

        assert req.model == "resilient[1]", (
            "Wrapper must restore original llm_request.model after success"
        )

    @pytest.mark.asyncio
    async def test_request_model_restored_after_structural_error(self):
        err = BadRequestError("bad", model="x", llm_provider="groq")
        m = ResilientModel(inner_models=[_make_fake("groq/x", error=err)])
        req = _make_request()
        req.model = "resilient[1]"

        with pytest.raises(BadRequestError):
            async for _ in m.generate_content_async(req):
                pass

        assert req.model == "resilient[1]"

    @pytest.mark.asyncio
    async def test_request_model_restored_after_exhaustion(self):
        err = RateLimitError("rl", llm_provider="groq", model="x")
        m = ResilientModel(inner_models=[
            _make_fake("groq/a", error=err),
            _make_fake("groq/b", error=err),
        ])
        req = _make_request()
        req.model = "resilient[2]"

        with pytest.raises(AllModelsExhaustedError):
            async for _ in m.generate_content_async(req):
                pass

        assert req.model == "resilient[2]"
