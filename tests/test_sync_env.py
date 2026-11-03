"""Tests for scripts/sync_env.py -- the add-missing-only .env merger.

The golden rule under test: existing keys (secrets, user tweaks) are NEVER
modified; only genuinely-new keys are appended.
"""
from __future__ import annotations

import importlib.util
import io
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
apply_managed_updates = sync_env_mod.apply_managed_updates


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
        new_text, added, updated = merge(example, env)
        assert added == ["OLLAMA_API_BASE"]
        assert updated == []
        assert "OLLAMA_API_BASE=http://localhost:11434" in new_text

    def test_never_touches_existing_values(self):
        example = "GROQ_API_KEY=your_groq_key_here\nNEW=1\n"
        env = "GROQ_API_KEY=sk-REAL-SECRET-123\n"
        new_text, added, updated = merge(example, env)
        # The user's real secret must survive verbatim; placeholder NOT applied.
        assert "GROQ_API_KEY=sk-REAL-SECRET-123" in new_text
        assert "your_groq_key_here" not in new_text
        assert added == ["NEW"]
        assert updated == []

    def test_noop_when_nothing_missing(self):
        example = "A=1\nB=2\n"
        env = "A=99\nB=88\n"
        new_text, added, updated = merge(example, env)
        assert added == []
        assert updated == []
        assert new_text == env  # unchanged

    def test_carries_preceding_comment(self):
        example = "# Local Ollama server URL\nOLLAMA_API_BASE=http://localhost:11434\n"
        env = "GROQ_API_KEY=x\n"
        new_text, added, _updated = merge(example, env)
        assert "# Local Ollama server URL" in new_text
        assert added == ["OLLAMA_API_BASE"]

    def test_commented_template_lines_are_not_added(self):
        # #HTTPS_PROXY is documentation, not an active key -> never injected.
        example = "A=1\n#HTTPS_PROXY=http://proxy:8080\n"
        env = "A=1\n"
        _, added, _updated = merge(example, env)
        assert added == []


class TestManagedKeys:
    def test_managed_chain_value_is_updated(self):
        # CHAIN_AGENTIC is app-managed -> its value syncs to the template's.
        example = "CHAIN_AGENTIC=gemini/x,groq/y,ollama_chat/qwen3:8b\n"
        env = "CHAIN_AGENTIC=gemini/x,groq/y\n"
        new_text, added, updated = merge(example, env)
        assert updated == ["CHAIN_AGENTIC"]
        assert added == []
        assert "ollama_chat/qwen3:8b" in new_text

    def test_secrets_are_NOT_managed(self):
        # GROQ_API_KEY is NOT managed -> user's value stays, template ignored.
        example = "GROQ_API_KEY=placeholder\n"
        env = "GROQ_API_KEY=sk-REAL\n"
        new_text, _added, updated = merge(example, env)
        assert updated == []
        assert "sk-REAL" in new_text
        assert "placeholder" not in new_text

    def test_price_chain_is_NOT_managed(self):
        # PRICE_CHAIN is a user/geo tweak -> never overwritten.
        example = "PRICE_CHAIN=jugaad,nse_bhavcopy,yfinance\n"
        env = "PRICE_CHAIN=yfinance\n"
        new_text, _added, updated = merge(example, env)
        assert updated == []
        assert "PRICE_CHAIN=yfinance\n" in new_text

    def test_managed_noop_when_value_matches(self):
        example = "CHAIN_AGENTIC=a,b,c\n"
        env = "CHAIN_AGENTIC=a,b,c\n"
        _new, _added, updated = merge(example, env)
        assert updated == []

    def test_apply_managed_updates_preserves_line_order(self):
        example = "CHAIN_AGENTIC=new\n"
        env = "GROQ_API_KEY=sk\nCHAIN_AGENTIC=old\nPRICE_CHAIN=yfinance\n"
        new_text, updated = apply_managed_updates(env, example)
        assert updated == ["CHAIN_AGENTIC"]
        lines = new_text.splitlines()
        assert lines == ["GROQ_API_KEY=sk", "CHAIN_AGENTIC=new", "PRICE_CHAIN=yfinance"]


class TestSyncEnvFileOps:
    def test_first_run_seeds_from_template(self, tmp_path: Path):
        example = tmp_path / ".env.example"
        example.write_text("A=1\nB=2\n", encoding="utf-8")
        env = tmp_path / ".env"  # does not exist yet
        result = sync_env(env, example)
        assert env.exists()
        assert env.read_text() == "A=1\nB=2\n"
        assert set(result["added"]) == {"A", "B"}

    def test_adds_missing_preserves_secret(self, tmp_path: Path):
        example = tmp_path / ".env.example"
        example.write_text("GROQ_API_KEY=placeholder\nOLLAMA_API_BASE=x\n", encoding="utf-8")
        env = tmp_path / ".env"
        env.write_text("GROQ_API_KEY=sk-SECRET\n", encoding="utf-8")
        result = sync_env(env, example)
        content = env.read_text()
        assert result["added"] == ["OLLAMA_API_BASE"]
        assert "sk-SECRET" in content
        assert "placeholder" not in content

    def test_managed_key_updates_on_disk(self, tmp_path: Path):
        example = tmp_path / ".env.example"
        example.write_text("CHAIN_AGENTIC=gemini/x,ollama_chat/qwen3:8b\n", encoding="utf-8")
        env = tmp_path / ".env"
        env.write_text("CHAIN_AGENTIC=gemini/x\n", encoding="utf-8")
        result = sync_env(env, example)
        assert result["updated"] == ["CHAIN_AGENTIC"]
        assert "ollama_chat/qwen3:8b" in env.read_text()

    def test_idempotent_second_run_adds_nothing(self, tmp_path: Path):
        example = tmp_path / ".env.example"
        example.write_text("A=1\nNEW=2\n", encoding="utf-8")
        env = tmp_path / ".env"
        env.write_text("A=1\n", encoding="utf-8")
        first = sync_env(env, example)
        second = sync_env(env, example)
        assert first["added"] == ["NEW"]
        assert second["added"] == [] and second["updated"] == []  # nothing left

    def test_missing_template_is_noop(self, tmp_path: Path):
        env = tmp_path / ".env"
        env.write_text("A=1\n", encoding="utf-8")
        result = sync_env(env, tmp_path / "does_not_exist.example")
        assert result == {"added": [], "updated": []}
        assert env.read_text() == "A=1\n"  # untouched


class TestShipsToDeployedLaptops:
    """What the Windows launcher does on every start: merge the REAL template into an old .env."""

    REPO = Path(__file__).resolve().parents[1]
    OLD_ENV = (  # a laptop set up from the previous release template, keys written with spaces
        "GROQ_API_KEY = gsk_REAL_GROQ\n"
        "GEMINI_API_KEY = AIza_REAL_GEMINI\n"
        "OPENROUTER_API_KEY=sk-or-REAL\n"
        "CHAIN_AGENTIC=gemini/gemini-flash-latest,groq/openai/gpt-oss-120b,ollama_chat/qwen3:8b\n"
        "PAID_AGENTIC=groq/llama-3.3-70b-versatile\n"
        "PRICE_CHAIN=yfinance\n"
    )

    def _merged(self) -> str:
        example = (self.REPO / ".env.example").read_text(encoding="utf-8")
        new_text, _added, _updated = merge(example, self.OLD_ENV)
        return new_text

    def test_every_llm_key_slot_reaches_the_laptop_and_real_keys_survive(self):
        from dotenv import dotenv_values

        from price_predictor.config.settings import Settings
        values = dotenv_values(stream=io.StringIO(self._merged()))
        for _field, env in Settings.LLM_KEY_ENV.values():
            assert env in values, f"{env} must be offered in the laptop's .env"
        assert values["GEMINI_API_KEY"] == "AIza_REAL_GEMINI"   # not wiped by a blank duplicate
        assert values["GROQ_API_KEY"] == "gsk_REAL_GROQ"
        assert values["MISTRAL_API_KEY"] == "" and values["NVIDIA_NIM_API_KEY"] == ""
        assert values["PRICE_CHAIN"] == "yfinance"

    def test_the_chain_is_updated_to_the_template_chain(self):
        from dotenv import dotenv_values
        example = (self.REPO / ".env.example").read_text(encoding="utf-8")
        merged = dotenv_values(stream=io.StringIO(self._merged()))
        assert merged["CHAIN_AGENTIC"] == dotenv_values(stream=io.StringIO(example))["CHAIN_AGENTIC"]

    def test_merging_twice_changes_nothing(self):
        example = (self.REPO / ".env.example").read_text(encoding="utf-8")
        once = self._merged()
        twice, added, updated = merge(example, once)
        assert (twice, added, updated) == (once, [], [])

    def test_a_key_written_with_spaces_counts_as_present(self):
        _new, added, _upd = merge("GEMINI_API_KEY=\n", "GEMINI_API_KEY = real\n")
        assert added == []
