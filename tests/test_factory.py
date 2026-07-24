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
        # quotas are exhausted, so we keep qwen3 reasoning ON for prediction
        # quality. litellm maps reasoning_effort='high' -> Ollama think=True.
        # Parseability is guaranteed by json_extract (strips <think> blocks)
        # + predictor's penalize-and-fall-over on any still-unparseable reply.
        m = make_model("ollama_chat/qwen3:8b")
        assert m._additional_args["reasoning_effort"] == "high"

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


class TestFailFastConfig:
    """LiteLLM must NOT retry internally -- we own fallback + pacing.

    Regression guard for the 'stuck on Retrying...' hang: without
    num_retries=0, LiteLLM retries transient errors (429/500/timeout)
    with exponential backoff BEFORE our ResilientModel ever sees the
    error, so a rate-limited Groq call blocks for many seconds and never
    falls over to Ollama. A bounded timeout stops a wedged connection
    from hanging the whole prediction.
    """

    def test_hosted_model_disables_internal_retry_and_sets_timeout(self):
        m = make_model("groq/openai/gpt-oss-120b")
        assert m._additional_args["num_retries"] == 0
        assert m._additional_args["timeout"] == factory._HOSTED_TIMEOUT_S

    def test_gemini_model_disables_internal_retry(self):
        m = make_model("gemini/gemini-flash-latest")
        assert m._additional_args["num_retries"] == 0
        assert m._additional_args["timeout"] == factory._HOSTED_TIMEOUT_S

    def test_ollama_model_disables_internal_retry_with_generous_timeout(self):
        m = make_model("ollama_chat/qwen3:8b")
        assert m._additional_args["num_retries"] == 0
        # Local reasoning_effort='high' is slow -> generous timeout.
        assert m._additional_args["timeout"] == factory._OLLAMA_TIMEOUT_S


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
