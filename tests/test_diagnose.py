"""Tests for the pure (non-network) helpers in scripts/diagnose.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "diagnose",
    Path(__file__).resolve().parents[1] / "scripts" / "diagnose.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)  # type: ignore[union-attr]


class TestMask:
    def test_masks_long_secret(self):
        out = mod._mask("sk-abcdefghijklmnop")
        assert "sk-a" in out and "mnop" in out and "..." in out
        assert "efghij" not in out  # middle hidden

    def test_short_secret_fully_hidden(self):
        assert mod._mask("abc") == "***"

    def test_empty(self):
        assert mod._mask("") == "(empty)"


class TestBuildVerdict:
    def test_recommends_first_working_gemini(self):
        r = {
            "model_probes": [
                {"model": "gemini/gemini-flash-latest", "ok": False, "error": "400 x"},
                {"model": "gemini/gemini-2.5-flash", "ok": True},
                {"model": "groq/openai/gpt-oss-120b", "ok": True},
            ],
            "gdelt_news": {"ok": True},
            "ollama": {"configured": True, "server_reachable": True,
                       "wanted": ["qwen3:8b"], "pulled": ["qwen3:8b"]},
            "prediction": {"skipped": True},
        }
        v = " ".join(mod._build_verdict(r))
        assert "recommend setting CHAIN_AGENTIC primary to: gemini/gemini-2.5-flash" in v
        assert "gemini/gemini-flash-latest" in v  # listed as broken

    def test_flags_no_working_gemini(self):
        r = {
            "model_probes": [{"model": "gemini/x", "ok": False, "error": "e"}],
            "gdelt_news": {"ok": False, "error": "conn"},
            "ollama": {"configured": False},
            "prediction": {"skipped": True},
        }
        v = " ".join(mod._build_verdict(r))
        assert "NO Gemini model worked" in v
        assert "News (GDELT): FAILED" in v


class TestRenderTxt:
    def test_renders_without_error(self):
        r = {
            "_meta": {"started_utc": "2026-07-22T00:00:00+00:00"},
            "environment": {"os": "Test", "python": "3.13"},
            "gemini_listmodels": {"ok": True, "flash_models": ["gemini/gemini-2.5-flash"]},
            "model_probes": [{"model": "gemini/gemini-2.5-flash", "ok": True, "latency_s": 0.4}],
            "gdelt_news": {"ok": True, "rows": 3, "query": "Infosys", "sample": "x"},
            "ollama": {"configured": False},
            "prediction": {"skipped": True},
            "verdict": ["all good"],
        }
        txt = mod._render_txt(r)
        assert "OFF-VPN DIAGNOSTIC RESULTS" in txt
        assert "gemini/gemini-2.5-flash" in txt
        assert "all good" in txt
