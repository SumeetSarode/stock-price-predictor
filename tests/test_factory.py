"""Tests for the LLM model factory -- provider routing + local (Ollama) support.

Focus: the keyless-local-provider path added so the resilient chain can fall
over to a local Ollama model when hosted providers (Groq/Gemini) are exhausted.
No network / no live model: we only assert how models are CONSTRUCTED.
"""
from __future__ import annotations

import pytest

from price_predictor.config.settings import settings
from price_predictor.llm import factory
from price_predictor.llm.factory import make_model, make_resilient_model
from price_predictor.llm.resilient import ResilientModel


class TestKeylessLocalProvider:
    """Ollama models build with an api_base and NO api key."""

    def test_ollama_chat_sets_api_base_no_key(self):
        m = make_model("ollama_chat/qwen3:8b")
        assert m.model == "ollama_chat/qwen3:8b"
        # ADK's LiteLlm stashes extra kwargs in _additional_args.
        assert m._additional_args["api_base"] == settings.ollama_api_base
        # keyless: no api_key was injected.
        assert "api_key" not in m._additional_args

    def test_ollama_bare_prefix_also_supported(self):
        m = make_model("ollama/llama3.1:8b")
        assert m._additional_args["api_base"] == settings.ollama_api_base

    def test_ollama_keeps_thinking_on_for_quality(self):
        # The local tier is the de-facto workhorse once free-tier hosted
        # quotas are exhausted, so we DEFAULT qwen3 reasoning ON for
        # prediction quality. litellm maps reasoning_effort='high' ->
        # Ollama think=True. Parseability is guaranteed by json_extract
        # (strips <think> blocks) + predictor's penalize-and-fall-over.
        m = make_model("ollama_chat/qwen3:8b")
        assert m._additional_args["reasoning_effort"] == "high"

    def test_reasoning_effort_follows_settings_override(self, monkeypatch):
        # A speed-tuning workflow might flip this to 'none' to
        # measure the faster think-off mode. For qwen3, litellm maps any
        # value NOT in {low,medium,high} -> think=False. Verify the factory
        # threads the setting through rather than hardcoding 'high'.
        monkeypatch.setattr(settings, "ollama_reasoning_effort", "none")
        m = make_model("ollama_chat/qwen3:8b")
        assert m._additional_args["reasoning_effort"] == "none"

    def test_hosted_models_do_not_get_reasoning_effort(self):
        # Hosted models reason at their own provider defaults (gpt-oss reasons
        # at 'medium' out of the box). We don't override them: forcing 'high'
        # burns more tokens and trips Groq's free-tier rate/size limits sooner
        # (proven via live probe), causing needless fallover.
        m = make_model("groq/openai/gpt-oss-120b")
        assert "reasoning_effort" not in m._additional_args

    def test_api_base_follows_settings(self, monkeypatch):
        monkeypatch.setattr(settings, "ollama_api_base", "http://192.168.1.50:11434")
        m = make_model("ollama_chat/qwen3:8b")
        assert m._additional_args["api_base"] == "http://192.168.1.50:11434"

    def test_ollama_sets_num_ctx_from_settings(self):
        # Ollama defaults num_ctx to 2048 regardless of the model's real
        # capacity, so a big prediction prompt overflows it and litellm
        # raises ContextWindowExceededError -- surfaced as an "LLM token
        # limit" error even though Ollama has no quota. We MUST set num_ctx
        # explicitly for the offline fallback to be usable.
        m = make_model("ollama_chat/qwen3:8b")
        assert m._additional_args["num_ctx"] == settings.ollama_num_ctx
        assert settings.ollama_num_ctx >= 8192  # comfortably above a real prompt

    def test_num_ctx_follows_settings_override(self, monkeypatch):
        monkeypatch.setattr(settings, "ollama_num_ctx", 8192)
        m = make_model("ollama_chat/qwen3:8b")
        assert m._additional_args["num_ctx"] == 8192

    def test_hosted_models_do_not_get_num_ctx(self):
        # num_ctx is an Ollama-only knob; hosted providers manage their own
        # (much larger) context windows.
        m = make_model("groq/openai/gpt-oss-120b")
        assert "num_ctx" not in m._additional_args


class TestFailFastConfig:
    """LiteLLM must NOT retry internally -- we own fallback + pacing.

    Regression guard for the 'stuck on Retrying...' hang: without
    num_retries=0, LiteLLM retries transient errors (429/500/timeout)
    with exponential backoff BEFORE our ResilientModel ever sees the
    error, so a rate-limited Groq call blocks for many seconds and never
    falls over to Ollama. A bounded timeout stops a wedged connection
    from hanging the whole prediction.

    Timeout is a SPLIT openai.Timeout (connect vs read/write/pool), not one
    flat number -- see factory.py's _HOSTED_CONNECT_TIMEOUT_S comment for
    why: a dead/unreachable connection (connect leg) should fail fast, but
    a real non-streaming generation response (read leg) legitimately needs
    the full budget.
    """

    def test_hosted_model_disables_internal_retry_and_sets_timeout(self):
        m = make_model("groq/openai/gpt-oss-120b")
        assert m._additional_args["num_retries"] == 0
        t = m._additional_args["timeout"]
        assert t.connect == factory._HOSTED_CONNECT_TIMEOUT_S
        assert t.read == factory._HOSTED_TIMEOUT_S

    def test_gemini_model_disables_internal_retry(self):
        m = make_model("gemini/gemini-flash-latest")
        assert m._additional_args["num_retries"] == 0
        t = m._additional_args["timeout"]
        assert t.connect == factory._HOSTED_CONNECT_TIMEOUT_S
        assert t.read == factory._HOSTED_TIMEOUT_S

    def test_hosted_connect_timeout_is_much_shorter_than_read_timeout(self):
        # The whole point: a dead socket must fail MUCH faster than we'd
        # wait for a real (slow but alive) generation to finish.
        assert factory._HOSTED_CONNECT_TIMEOUT_S < factory._HOSTED_TIMEOUT_S

    def test_ollama_model_disables_internal_retry_with_generous_timeout(self):
        m = make_model("ollama_chat/qwen3:8b")
        assert m._additional_args["num_retries"] == 0
        t = m._additional_args["timeout"]
        # Local reasoning_effort='high' is slow -> generous READ timeout,
        # but connect (is Ollama even reachable?) still fails fast.
        assert t.connect == factory._OLLAMA_CONNECT_TIMEOUT_S
        assert t.read == factory._OLLAMA_TIMEOUT_S


class TestHostedProviderStillWorks:
    """The keyed path (groq/gemini) is unchanged."""

    def test_groq_builds_with_key(self):
        m = make_model("groq/openai/gpt-oss-120b")
        assert m.model == "groq/openai/gpt-oss-120b"

    def test_unsupported_provider_lists_ollama(self):
        with pytest.raises(ValueError, match="Unsupported provider") as exc:
            make_model("openai/gpt-4o")
        # error message advertises the local providers as now-supported.
        assert "ollama" in str(exc.value)


class TestOpenRouterProvider:
    """openrouter is a hosted, keyed provider -- same shape as groq/gemini.

    Its key is OPTIONAL at the settings layer (Settings.effective_chain
    silently drops an openrouter chain entry if unconfigured -- see
    test_settings.py::TestOptionalKeyProviders), but make_model() itself
    just builds whatever it's given, same as any other keyed provider.
    """

    def test_openrouter_builds_with_key(self):
        m = make_model("openrouter/nvidia/nemotron-3-ultra-550b-a55b:free")
        assert m.model == "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"

    def test_openrouter_uses_hosted_timeout_and_no_internal_retry(self):
        m = make_model("openrouter/nvidia/nemotron-3-ultra-550b-a55b:free")
        t = m._additional_args["timeout"]
        assert t.connect == factory._HOSTED_CONNECT_TIMEOUT_S
        assert t.read == factory._HOSTED_TIMEOUT_S
        assert m._additional_args["num_retries"] == factory._NO_INTERNAL_RETRY


class TestCohereProvider:
    """cohere_chat is a hosted, keyed provider -- same shape as openrouter.

    Promoted into the default chain ahead of Nemotron after a measured
    head-to-head (3/3 vs 1/3 production-valid JSON, see
    reports/free_tier_candidates_20260923T155343Z.md).
    """

    def test_cohere_chat_builds_with_key(self, monkeypatch):
        # Patch the GETTER, not the settings field: pydantic blocks
        # instance/class attr patching, and this keeps the test independent
        # of whatever happens to be in the developer's real .env.
        monkeypatch.setitem(
            factory._API_KEY_GETTERS, "cohere_chat", lambda: "test-cohere-key",
        )
        m = make_model("cohere_chat/command-r-08-2024")
        assert m.model == "cohere_chat/command-r-08-2024"

    def test_cohere_chat_uses_hosted_timeout_and_no_internal_retry(self, monkeypatch):
        monkeypatch.setitem(
            factory._API_KEY_GETTERS, "cohere_chat", lambda: "test-cohere-key",
        )
        m = make_model("cohere_chat/command-r-08-2024")
        t = m._additional_args["timeout"]
        assert t.connect == factory._HOSTED_CONNECT_TIMEOUT_S
        assert t.read == factory._HOSTED_TIMEOUT_S
        assert m._additional_args["num_retries"] == factory._NO_INTERNAL_RETRY

    def test_plain_cohere_prefix_is_rejected(self):
        """'cohere/' is litellm's EMBEDDINGS provider, not chat.

        Only 'cohere_chat/' is registered. A 'cohere/' entry is a genuine
        config error and must fail LOUDLY here rather than being built and
        then 404-ing at generation time.
        """
        with pytest.raises(ValueError, match="Unsupported provider"):
            make_model("cohere/command-r-08-2024")


class TestCohereSurvivesAdkResponseFormat:
    """Regression: Cohere never served a production request.

    Both agents set output_schema, so ADK adds `response_format` to every
    request, and litellm's cohere_chat route rejects it with
    UnsupportedParamsError. The chain filed that as 'model can't handle this
    conversation' (1h cooldown) so Cohere was silently dead -- while every
    earlier check passed, because they called litellm directly and skipped
    ADK's output_schema. These tests go through the REAL ADK path; only the
    network call is faked (litellm mock mode), so litellm's own parameter
    validation still runs.
    """

    @staticmethod
    def _client_with_mock_reply():
        from google.adk.models.lite_llm import LiteLLMClient

        class _Mocked(LiteLLMClient):
            async def acompletion(self, model, messages, tools, **kwargs):
                return await super().acompletion(
                    model, messages, tools,
                    mock_response='{"ticker":"RELIANCE.NS"}', **kwargs,
                )

        return _Mocked()

    @staticmethod
    def _request_with_schema():
        from google.adk.models.llm_request import LlmRequest
        from google.genai import types

        from price_predictor.agents.news_impact.agent import ImpactAssessment

        return LlmRequest(
            contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
            config=types.GenerateContentConfig(response_schema=ImpactAssessment),
        )

    @pytest.mark.asyncio
    async def test_control_without_drop_params_fails_as_in_production(self):
        """Proves this test can see the bug: a bare LiteLlm raises."""
        from google.adk.models.lite_llm import LiteLlm
        from litellm.exceptions import UnsupportedParamsError

        bare = LiteLlm(
            model="cohere_chat/command-r-08-2024", api_key="k",
            llm_client=self._client_with_mock_reply(),
        )
        with pytest.raises(UnsupportedParamsError):
            _ = [r async for r in bare.generate_content_async(self._request_with_schema())]

    @pytest.mark.asyncio
    async def test_make_model_cohere_drops_response_format_and_answers(self, monkeypatch):
        monkeypatch.setitem(factory._API_KEY_GETTERS, "cohere_chat", lambda: "k")
        m = make_model("cohere_chat/command-r-08-2024")
        m.llm_client = self._client_with_mock_reply()

        responses = [r async for r in m.generate_content_async(self._request_with_schema())]

        assert responses, "Cohere must answer, not raise UnsupportedParamsError"
        assert "RELIANCE.NS" in responses[-1].content.parts[0].text

    def test_other_providers_still_get_response_format(self, monkeypatch):
        """drop_params is scoped to Cohere: providers that support
        response_format must keep receiving it."""
        monkeypatch.setitem(factory._API_KEY_GETTERS, "groq", lambda: "k")
        assert "drop_params" not in make_model("groq/openai/gpt-oss-120b")._additional_args


class TestResilientChainWithLocalTail:
    """A chain ending in Ollama builds into a ResilientModel (offline tail)."""

    def test_chain_with_ollama_tail_builds(self, monkeypatch):
        # Patch the chain so the test needs NO hosted keys -- pure local.
        # Patch on the class (pydantic blocks instance-attr setattr).
        monkeypatch.setattr(
            type(factory.settings),
            "effective_chain",
            lambda self, profile: ["ollama_chat/qwen3:8b", "ollama/llama3.1:8b"],
        )
        model = make_resilient_model("agentic")
        assert isinstance(model, ResilientModel)
        names = [m.model for m in model.inner_models]
        assert names == ["ollama_chat/qwen3:8b", "ollama/llama3.1:8b"]
        # every inner local model carries the api_base.
        for inner in model.inner_models:
            assert inner._additional_args["api_base"] == settings.ollama_api_base
