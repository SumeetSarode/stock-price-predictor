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


class TestPerfFlags:
    """The three accuracy-neutral server perf flags (flash attn, q8 KV,
    keep-alive) must be applied to the server we spawn AND persisted when
    a server is already running."""

    def test_perf_flags_are_the_expected_three(self):
        assert mod._OLLAMA_PERF_FLAGS == {
            "OLLAMA_FLASH_ATTENTION": "1",
            "OLLAMA_KV_CACHE_TYPE": "q8_0",
            "OLLAMA_KEEP_ALIVE": "-1",
        }

    def test_perf_env_overlays_flags_on_os_environ(self, monkeypatch):
        monkeypatch.setenv("PATH", "/usr/bin")  # some real inherited var
        env = mod._perf_env()
        assert env["PATH"] == "/usr/bin"  # inheritance preserved
        for key, value in mod._OLLAMA_PERF_FLAGS.items():
            assert env[key] == value  # flags overlaid

    def test_start_server_passes_perf_env(self, monkeypatch):
        captured = {}

        def fake_popen(args, **kwargs):
            captured["args"] = args
            captured["env"] = kwargs.get("env")
            return object()

        monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
        mod._start_server()
        assert captured["args"] == ["ollama", "serve"]
        assert captured["env"]["OLLAMA_FLASH_ATTENTION"] == "1"
        assert captured["env"]["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
        assert captured["env"]["OLLAMA_KEEP_ALIVE"] == "-1"

    def test_already_running_persists_on_windows(self, monkeypatch):
        monkeypatch.setattr(mod.platform, "system", lambda: "Windows")
        monkeypatch.setattr(mod.shutil, "which", lambda name: "C:/setx.exe")
        calls = []
        monkeypatch.setattr(
            mod.subprocess, "run",
            lambda args, **kw: calls.append(args) or None,
        )
        mod._apply_perf_to_running_server()
        set_keys = {c[1] for c in calls if c[0] == "setx"}
        assert set_keys == set(mod._OLLAMA_PERF_FLAGS)

    def test_already_running_non_windows_just_advises(self, monkeypatch, capsys):
        monkeypatch.setattr(mod.platform, "system", lambda: "Darwin")
        mod._apply_perf_to_running_server()  # must not raise / no setx
        out = capsys.readouterr().out
        assert "OLLAMA_FLASH_ATTENTION=1" in out
