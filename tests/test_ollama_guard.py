"""Tests for the Ollama startup guard. Fully mocked -- no real network."""
from __future__ import annotations

import httpx
import respx

from price_predictor.llm.ollama_guard import (
    _is_pulled,
    check_local_models,
    ollama_tags_in_chain,
    warn_if_local_models_missing,
)

_BASE = "http://localhost:11434"
_TAGS_URL = f"{_BASE}/api/tags"


class TestTagExtraction:
    def test_extracts_only_ollama_entries(self):
        chain = [
            "gemini/gemini-2.5-flash",
            "groq/openai/gpt-oss-120b",
            "ollama_chat/qwen3:8b",
            "ollama/llama3.1:8b",
        ]
        assert ollama_tags_in_chain(chain) == ["qwen3:8b", "llama3.1:8b"]

    def test_no_ollama_returns_empty(self):
        assert ollama_tags_in_chain(["gemini/x", "groq/y"]) == []

    def test_dedupes_preserving_order(self):
        chain = ["ollama_chat/qwen3:8b", "ollama/qwen3:8b"]
        assert ollama_tags_in_chain(chain) == ["qwen3:8b"]


class TestIsPulled:
    def test_exact_match(self):
        assert _is_pulled("qwen3:8b", {"qwen3:8b"})

    def test_latest_forgiveness(self):
        assert _is_pulled("qwen3", {"qwen3:latest"})

    def test_missing(self):
        assert not _is_pulled("qwen3:8b", {"llama3.1:8b"})


class TestCheckLocalModels:
    def test_no_local_model_skips_network(self):
        # No ollama entry -> must NOT hit the network at all.
        with respx.mock:
            route = respx.get(_TAGS_URL)
            warnings = check_local_models(["gemini/x", "groq/y"], _BASE)
        assert warnings == []
        assert not route.called

    @respx.mock
    def test_pulled_model_no_warning(self):
        respx.get(_TAGS_URL).mock(
            return_value=httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})
        )
        warnings = check_local_models(["gemini/x", "ollama_chat/qwen3:8b"], _BASE)
        assert warnings == []

    @respx.mock
    def test_missing_model_warns_with_pull_cmd(self):
        respx.get(_TAGS_URL).mock(
            return_value=httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})
        )
        warnings = check_local_models(["ollama_chat/qwen3:8b"], _BASE)
        assert len(warnings) == 1
        assert "ollama pull qwen3:8b" in warnings[0]

    @respx.mock
    def test_server_unreachable_warns_softly(self):
        respx.get(_TAGS_URL).mock(side_effect=httpx.ConnectError("refused"))
        warnings = check_local_models(["ollama_chat/qwen3:8b"], _BASE)
        assert len(warnings) == 1
        assert "Could not reach Ollama" in warnings[0]


class TestWarnConvenienceNeverRaises:
    @respx.mock
    def test_missing_model_logs_but_does_not_raise(self):
        # Whatever the real chain is, an unreachable/empty server must be safe.
        respx.get(_TAGS_URL).mock(side_effect=httpx.ConnectError("refused"))
        # Should complete silently (no exception) regardless of chain contents.
        warn_if_local_models_missing()
