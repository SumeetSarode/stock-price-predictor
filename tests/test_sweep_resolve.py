"""Offline tests for scripts/sweep_resolve.py: finding the current id of a model a provider says is not found."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sweep_data as D
import sweep_probe as PR
import sweep_resolve as R


def test_alternatives_keep_every_name_word_and_rank_the_closest_first():
    pool = ["moonshotai/kimi-k2.7", "moonshotai/kimi-k2.6-instruct", "meta/llama-3.3-70b", "kimi-k3"]
    assert R.alternatives("moonshotai/kimi-k2.6", pool)[:2] == ["moonshotai/kimi-k2.6-instruct", "moonshotai/kimi-k2.7"]
    assert "meta/llama-3.3-70b" not in R.alternatives("moonshotai/kimi-k2.6", pool)


def test_a_different_model_of_the_same_brand_is_not_a_replacement():
    """Seen live: gemini-2.5-pro was 'replaced' by gemini-2.5-flash and gemini-4-argon. Never again."""
    pool = ["gemini-2.5-flash", "gemini-4-argon", "gemini-2.5-pro-preview-06-05", "gemini-3.1-pro-preview"]
    assert R.alternatives("gemini-2.5-pro", pool) == ["gemini-2.5-pro-preview-06-05", "gemini-3.1-pro-preview"]


def test_alternatives_ignore_vendor_paths_and_route_suffixes():
    assert R.alternatives("@cf/google/gemma-3-12b-it", ["@cf/google/gemma-4-12b-it"]) == ["@cf/google/gemma-4-12b-it"]
    assert R.alternatives("x/qwen3-8b:free", ["y/qwen3-8b:free"]) == ["y/qwen3-8b:free"]
    assert R.alternatives("gemini-2.5-pro", []) == []


def test_catalog_ids_match_the_provider_under_its_models_dev_names():
    cat = {"ovhcloud": {"models": {"a": {}}}, "cloudflare-workers-ai": {"models": {"b": {}}}, "groq": {"models": {}}}
    assert R.catalog_ids(cat, "ovh") == {"a"} and R.catalog_ids(cat, "cloudflare") == {"b"}


def test_replacements_add_new_candidates_once_and_note_the_original(tmp_path):
    spec = D.SPECS["google"]
    gone = D.Candidate("direct", "google", "gemini-2.5-pro", base=spec.base, key="k", litellm_id="gemini/gemini-2.5-pro")
    live = D.Candidate("direct", "google", "gemini-2.5-pro-preview-06-05", base=spec.base, key="k")
    store = PR.Store(tmp_path / "r.jsonl")
    store.add(PR.Result(gone.uid, "chat", "FAIL", "not_found", "404"))
    new = R.replacements([gone, live], store, {"google": {"models": {"gemini-2.5-pro-latest": {}}}})
    assert [c.model for c in new] == ["gemini-2.5-pro-latest"]          # the live listed id is not added twice
    assert new[0].litellm_id == "gemini/gemini-2.5-pro-latest" and new[0].key == "k"
    assert "gemini-2.5-pro-preview-06-05" in gone.note
    assert R.replacements([gone, live, *new], store, {"google": {"models": {"gemini-2.5-pro-latest": {}}}}) == []
    assert gone.note.count(R.MARK) == 1                                 # resolved once, the note not repeated


def test_replacements_respect_free_only_providers(tmp_path):
    gone = D.Candidate("direct", "openrouter", "vendor/model-a:free", key="k")
    store = PR.Store(tmp_path / "r.jsonl")
    store.add(PR.Result(gone.uid, "chat", "FAIL", "not_found", "404"))
    cat = {"openrouter": {"models": {"vendor/model-b": {}, "vendor/model-b:free": {}}}}
    assert [c.model for c in R.replacements([gone], store, cat)] == ["vendor/model-b:free"]
