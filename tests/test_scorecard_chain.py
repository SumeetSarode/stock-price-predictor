"""--chain: score the production CHAIN_AGENTIC entries, called the way the app calls them."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import model_scorecard_offvpn as MS

CHAIN = "gemini/gemini-flash-latest, groq/openai/gpt-oss-120b,cohere_chat/command-r-08-2024,ollama_chat/qwen3:8b,"


def test_each_chain_entry_becomes_a_candidate_on_the_apps_own_model_path():
    cands = MS.chain_candidates(CHAIN)
    assert [c.test_id for c in cands] == ["gemini/gemini-flash-latest", "groq/openai/gpt-oss-120b",
                                          "cohere_chat/command-r-08-2024", "ollama_chat/qwen3:8b"]
    assert [c.provider for c in cands] == ["gemini", "groq", "cohere_chat", "ollama_chat"]
    assert cands[1].model == "openai/gpt-oss-120b" and len({c.uid for c in cands}) == 4
    assert MS.chain_candidates("") == []


def test_chain_mode_reads_the_env_chain_and_honours_only(monkeypatch):
    monkeypatch.setattr("dotenv.dotenv_values", lambda path: {"CHAIN_AGENTIC": CHAIN})
    args = argparse.Namespace(chain=True, only="", from_sweep="missing.jsonl", models=12)
    models, left_out = MS.shortlist_for(args)
    assert len(models) == 4 and left_out == []
    args.only = "groq/openai/gpt-oss-120b,command-r-08-2024"
    assert [m.model for m in MS.shortlist_for(args)[0]] == ["openai/gpt-oss-120b", "command-r-08-2024"]


def test_the_flag_parses():
    assert MS.build_parser().parse_args(["--chain"]).chain is True
    assert MS.build_parser().parse_args([]).chain is False
