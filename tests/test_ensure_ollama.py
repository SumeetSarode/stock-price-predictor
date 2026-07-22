"""Tests for scripts/ensure_ollama.py. All subprocess/network calls mocked."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "ensure_ollama",
    Path(__file__).resolve().parents[1] / "scripts" / "ensure_ollama.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)  # type: ignore[union-attr]


class TestModelsToPull:
    def test_returns_only_missing(self):
        assert mod.models_to_pull(["qwen3:8b", "llama3.1:8b"], {"qwen3:8b"}) == ["llama3.1:8b"]

    def test_empty_when_all_present(self):
        assert mod.models_to_pull(["qwen3:8b"], {"qwen3:8b"}) == []

    def test_latest_alias_satisfied(self):
        # _is_pulled forgives the implicit :latest tag.
        assert mod.models_to_pull(["qwen3"], {"qwen3:latest"}) == []


class TestEnsure:
    def test_noop_when_no_ollama_in_chain(self, monkeypatch):
        monkeypatch.setattr(mod, "ollama_tags_in_chain", lambda chain: [])
        calls = []
        monkeypatch.setattr(mod, "ollama_installed", lambda: calls.append("installed"))
        mod.ensure()
        assert calls == []  # never even checked install -> true no-op

    def test_ready_when_installed_and_pulled(self, monkeypatch, capsys):
        monkeypatch.setattr(mod, "ollama_tags_in_chain", lambda chain: ["qwen3:8b"])
        monkeypatch.setattr(mod, "ollama_installed", lambda: True)
        monkeypatch.setattr(mod, "_pulled_models", lambda url: {"qwen3:8b"})
        pulled_called = []
        monkeypatch.setattr(mod, "_pull", lambda tag: pulled_called.append(tag))
        mod.ensure()
        assert pulled_called == []  # nothing to pull
        assert "ready" in capsys.readouterr().out.lower()

    def test_pulls_missing_model(self, monkeypatch):
        monkeypatch.setattr(mod, "ollama_tags_in_chain", lambda chain: ["qwen3:8b"])
        monkeypatch.setattr(mod, "ollama_installed", lambda: True)
        # server up, but model not pulled -> then present after pull
        states = [set(), {"qwen3:8b"}]
        monkeypatch.setattr(mod, "_pulled_models", lambda url: states.pop(0) if states else {"qwen3:8b"})
        pulled = []
        monkeypatch.setattr(mod, "_pull", lambda tag: pulled.append(tag))
        mod.ensure()
        assert pulled == ["qwen3:8b"]

    def test_starts_server_when_unreachable(self, monkeypatch):
        monkeypatch.setattr(mod, "ollama_tags_in_chain", lambda chain: ["qwen3:8b"])
        monkeypatch.setattr(mod, "ollama_installed", lambda: True)
        # first probe: None (down); after start+wait: pulled set with model
        probes = [None]
        monkeypatch.setattr(mod, "_pulled_models", lambda url: probes.pop(0) if probes else {"qwen3:8b"})
        started = []
        monkeypatch.setattr(mod, "_start_server", lambda: started.append(True))
        monkeypatch.setattr(mod, "_wait_for_server", lambda url, timeout_s=20.0: {"qwen3:8b"})
        monkeypatch.setattr(mod, "_pull", lambda tag: None)
        mod.ensure()
        assert started == [True]

    def test_not_installed_prints_instructions_never_raises(self, monkeypatch, capsys):
        monkeypatch.setattr(mod, "ollama_tags_in_chain", lambda chain: ["qwen3:8b"])
        monkeypatch.setattr(mod, "ollama_installed", lambda: False)
        monkeypatch.setattr(mod, "_try_winget_install", lambda: False)
        mod.ensure()  # must not raise
        assert "ollama.com/download" in capsys.readouterr().out
