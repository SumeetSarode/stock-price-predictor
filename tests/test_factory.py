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
