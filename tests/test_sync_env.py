"""Tests for scripts/sync_env.py -- the add-missing-only .env merger.

The golden rule under test: existing keys (secrets, user tweaks) are NEVER
modified; only genuinely-new keys are appended.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

# scripts/ isn't a package -> load the module by path.
_SPEC = importlib.util.spec_from_file_location(
    "sync_env",
    Path(__file__).resolve().parents[1] / "scripts" / "sync_env.py",
)
sync_env_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sync_env_mod)  # type: ignore[union-attr]

active_keys = sync_env_mod.active_keys
merge = sync_env_mod.merge
sync_env = sync_env_mod.sync_env


class TestActiveKeys:
    def test_uncommented_keys_only(self):
        text = "A=1\n#B=2\n  C=3\n\nD=4\n"
        # A and D are active; #B is commented; '  C' is indented (not col 0).
        assert active_keys(text) == {"A", "D"}

    def test_empty(self):
        assert active_keys("") == set()


class TestMerge:
    def test_adds_missing_key(self):
        example = "GROQ_API_KEY=x\nOLLAMA_API_BASE=http://localhost:11434\n"
        env = "GROQ_API_KEY=my_real_secret\n"
        new_text, added = merge(example, env)
        assert added == ["OLLAMA_API_BASE"]
        assert "OLLAMA_API_BASE=http://localhost:11434" in new_text

    def test_never_touches_existing_values(self):
        example = "GROQ_API_KEY=your_groq_key_here\nNEW=1\n"
        env = "GROQ_API_KEY=sk-REAL-SECRET-123\n"
        new_text, added = merge(example, env)
        # The user's real secret must survive verbatim; placeholder NOT applied.
        assert "GROQ_API_KEY=sk-REAL-SECRET-123" in new_text
        assert "your_groq_key_here" not in new_text
        assert added == ["NEW"]

    def test_noop_when_nothing_missing(self):
        example = "A=1\nB=2\n"
        env = "A=99\nB=88\n"
        new_text, added = merge(example, env)
        assert added == []
        assert new_text == env  # unchanged

    def test_carries_preceding_comment(self):
        example = "# Local Ollama server URL\nOLLAMA_API_BASE=http://localhost:11434\n"
        env = "GROQ_API_KEY=x\n"
        new_text, added = merge(example, env)
        assert "# Local Ollama server URL" in new_text
        assert added == ["OLLAMA_API_BASE"]

    def test_commented_template_lines_are_not_added(self):
        # #HTTPS_PROXY is documentation, not an active key -> never injected.
        example = "A=1\n#HTTPS_PROXY=http://proxy:8080\n"
        env = "A=1\n"
        _, added = merge(example, env)
        assert added == []


class TestSyncEnvFileOps:
    def test_first_run_seeds_from_template(self, tmp_path: Path):
        example = tmp_path / ".env.example"
        example.write_text("A=1\nB=2\n", encoding="utf-8")
        env = tmp_path / ".env"  # does not exist yet
        added = sync_env(env, example)
        assert env.exists()
        assert env.read_text() == "A=1\nB=2\n"
        assert set(added) == {"A", "B"}

    def test_adds_missing_preserves_secret(self, tmp_path: Path):
        example = tmp_path / ".env.example"
        example.write_text("GROQ_API_KEY=placeholder\nOLLAMA_API_BASE=x\n", encoding="utf-8")
        env = tmp_path / ".env"
        env.write_text("GROQ_API_KEY=sk-SECRET\n", encoding="utf-8")
        added = sync_env(env, example)
        result = env.read_text()
        assert added == ["OLLAMA_API_BASE"]
        assert "sk-SECRET" in result
        assert "placeholder" not in result

    def test_idempotent_second_run_adds_nothing(self, tmp_path: Path):
        example = tmp_path / ".env.example"
        example.write_text("A=1\nNEW=2\n", encoding="utf-8")
        env = tmp_path / ".env"
        env.write_text("A=1\n", encoding="utf-8")
        first = sync_env(env, example)
        second = sync_env(env, example)
        assert first == ["NEW"]
        assert second == []  # nothing left to add

    def test_missing_template_is_noop(self, tmp_path: Path):
        env = tmp_path / ".env"
        env.write_text("A=1\n", encoding="utf-8")
        added = sync_env(env, tmp_path / "does_not_exist.example")
        assert added == []
        assert env.read_text() == "A=1\n"  # untouched
