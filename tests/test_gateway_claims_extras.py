"""Offline tests for the models.dev and Strata parts of the claims script.

No Docker, no providers, no network: httpx.MockTransport plays every remote. These prove the
verdict RULES (an unknown is INCONCLUSIVE, never a failure; this-machine verdicts are labelled as
such) and that secrets never reach the saved ledger. The real behaviour of the real projects is
what the off-VPN run measures.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import gateway_claims_catalog as CAT  # noqa: E402
import gateway_claims_offvpn as G  # noqa: E402
import gateway_claims_strata as ST  # noqa: E402
import model_survey_offvpn as S  # noqa: E402

GROQ_KEY = "gsk_" + "k" * 30


@pytest.fixture(autouse=True)
def _tmp_reports(monkeypatch, tmp_path):
    for mod in (G, S):
        monkeypatch.setattr(mod, "REPORTS", tmp_path)
    monkeypatch.setattr(G, "_SECRETS", [])


def client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=5)


def verdicts(L) -> dict[str, str]:
    return {f"{f.claim_id}[{f.phase}]": f.verdict for f in L.findings}


def evidence(L, claim_phase: str) -> str:
    return next(f.evidence for f in L.findings if f"{f.claim_id}[{f.phase}]" == claim_phase)


# ── models.dev ───────────────────────────────────────────────────────────

def _catalog(groq_models: list[str], providers: int = 230, so: bool = True) -> dict:
    cat = {f"p{i}": {"models": {}} for i in range(providers)}
    cat["groq"] = {"models": {m: {"id": m, "structured_output": so, "last_updated": "2026-01-01",
                                  "modalities": {"output": ["text"]}} for m in groq_models}}
    cat["google"] = {"models": {"gemini-x": {"id": "gemini-x", "structured_output": False}}}
    return cat


def _router(catalog, groq_served, chat=None, google_served=("models/gemini-x",)):
    seen = {"chat_body": None}

    def handler(req: httpx.Request) -> httpx.Response:
        u = str(req.url)
        if u == CAT.CATALOG_URL:
            return httpx.Response(200, json=catalog) if catalog is not None else httpx.Response(500)
        if u.startswith("https://api.groq.com/openai/v1/models"):
            return httpx.Response(200, json={"data": [{"id": m} for m in groq_served]})
        if u.startswith("https://generativelanguage.googleapis.com/v1beta/models"):
            return httpx.Response(200, json={"models": [{"name": m} for m in google_served]})
        if u.startswith("https://api.groq.com/openai/v1/chat/completions"):
            seen["chat_body"] = json.loads(req.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": chat or ""}}]})
        return httpx.Response(500)

    handler.seen = seen
    return handler


def test_md1_counts_the_catalog_and_treats_an_unreachable_one_as_inconclusive():
    L = G.Ledger("md1")
    CAT.catalog_phase(L, client(_router(_catalog(["a"]), ["a"])), {"groq": GROQ_KEY}, G.call)
    assert verdicts(L)["MD1[-]"] == "CONFIRMED"
    L = G.Ledger("md1b")
    CAT.catalog_phase(L, client(_router(_catalog(["a"], providers=20), ["a"])), {}, G.call)
    assert verdicts(L)["MD1[-]"] == "NOT_CONFIRMED"
    L = G.Ledger("md1c")
    CAT.catalog_phase(L, client(_router(None, [])), {}, G.call)
    assert verdicts(L) == {"MD1[-]": "INCONCLUSIVE"}


def test_md2_flags_stale_catalog_entries_and_names_them():
    L = G.Ledger("md2")
    CAT.catalog_phase(L, client(_router(_catalog(["a", "b", "retired-c"]), ["a", "b"])), {"groq": GROQ_KEY}, G.call)
    assert verdicts(L)["MD2[groq]"] == "NOT_CONFIRMED"
    assert "retired-c" in evidence(L, "MD2[groq]")


def test_md2_confirms_when_catalog_matches_and_normalises_googles_models_prefix():
    L = G.Ledger("md2b")
    CAT.catalog_phase(L, client(_router(_catalog(["a", "b"]), ["a", "b"])), {"groq": GROQ_KEY, "google": "g" * 20}, G.call)
    v = verdicts(L)
    assert v["MD2[groq]"] == "CONFIRMED"
    assert v["MD2[google]"] == "CONFIRMED", "'models/gemini-x' must match catalog id 'gemini-x'"


def test_md2_without_a_key_is_inconclusive_not_a_failure():
    L = G.Ledger("md2c")
    CAT.catalog_phase(L, client(_router(_catalog(["a"]), ["a"])), {}, G.call)
    assert verdicts(L)["MD2[groq]"] == "INCONCLUSIVE"


def test_md3_checks_the_structured_output_flag_with_a_real_json_schema_call():
    handler = _router(_catalog(["good-model"]), ["good-model"], chat=json.dumps({"answer": "ok"}))
    L = G.Ledger("md3")
    CAT.catalog_phase(L, client(handler), {"groq": GROQ_KEY}, G.call)
    assert verdicts(L)["MD3[groq/good-model]"] == "CONFIRMED"
    assert handler.seen["chat_body"]["response_format"]["type"] == "json_schema"
    assert handler.seen["chat_body"]["model"] == "good-model"


def test_md3_catches_a_model_flagged_structured_that_answers_in_prose():
    L = G.Ledger("md3b")
    CAT.catalog_phase(L, client(_router(_catalog(["liar"]), ["liar"], chat="Sure! Here you go")), {"groq": GROQ_KEY}, G.call)
    assert verdicts(L)["MD3[groq/liar]"] == "NOT_CONFIRMED"


def test_md3_skips_models_the_provider_does_not_serve_or_the_catalog_does_not_flag():
    handler = _router(_catalog(["flagged-but-gone", "unflagged"], so=True), ["unflagged"], chat='{"answer":"ok"}')
    L = G.Ledger("md3c")
    CAT.catalog_phase(L, client(handler), {"groq": GROQ_KEY}, G.call)
    tested = [k for k in verdicts(L) if k.startswith("MD3")]
    assert tested == ["MD3[groq/unflagged]"], "a model the provider no longer serves must not be probed"


# ── Strata: hardware verdicts ────────────────────────────────────────────

def hw(system="Windows", machine="AMD64", ram=64, disk=200, gpus=None):
    return {"system": system, "machine": machine, "ram_gb": ram, "free_disk_gb": disk,
            "gpus": [{"name": "NVIDIA GeForce RTX 5070", "vram_gb": 12.0, "vendor": "nvidia"}] if gpus is None else gpus}


def test_this_mac_does_not_meet_the_requirements_and_the_verdict_says_why():
    mac = hw("Darwin", "arm64", 36, 326, [{"name": "Apple M3 Pro", "vram_gb": None, "vendor": "apple"}])
    verdict, text = ST.evaluate(mac)
    assert verdict == "NOT_CONFIRMED"
    assert "no NVIDIA/AMD GPU" in text and "OS is Darwin" in text
    assert "#518" in text and "#1449" in text, "the maintainer's stance and the unmerged PR must be cited"


def test_a_qualifying_gaming_pc_is_confirmed():
    assert ST.evaluate(hw())[0] == "CONFIRMED"


@pytest.mark.parametrize("case, expected, needle", [
    (hw(gpus=[{"name": "RTX 3060", "vram_gb": 8.0, "vendor": "nvidia"}]), "NOT_CONFIRMED", "8.0 GB VRAM"),
    (hw(ram=16), "NOT_CONFIRMED", "RAM 16 GB"),
    (hw(disk=40), "NOT_CONFIRMED", "free disk 40 GB"),
    (hw(gpus=[]), "NOT_CONFIRMED", "none detected"),
    (hw("Linux", gpus=[{"name": "Radeon RX 7900 XTX", "vram_gb": None, "vendor": "amd"}]), "INCONCLUSIVE", "VRAM"),
    (hw(ram=None), "INCONCLUSIVE", "RAM not detected"),
])
def test_requirement_edges(case, expected, needle):
    verdict, text = ST.evaluate(case)
    assert verdict == expected and needle in text


def test_unknown_values_are_never_reported_as_failures():
    verdict, _ = ST.evaluate(hw("Linux", gpus=[{"name": "Radeon RX 9070", "vram_gb": None, "vendor": "amd"}], ram=None, disk=500))
    assert verdict == "INCONCLUSIVE"


def test_detect_gpus_parses_each_platforms_tooling():
    nv = lambda *a: "NVIDIA GeForce RTX 5070, 12227\n" if a[0] == "nvidia-smi" else ""  # noqa: E731
    assert ST.detect_gpus("Windows", nv) == [{"name": "NVIDIA GeForce RTX 5070", "vram_gb": 11.9, "vendor": "nvidia"}]
    mac = lambda *a: "Chipset Model: Apple M3 Pro\n  Type: GPU\n" if a[0] == "system_profiler" else ""  # noqa: E731
    assert ST.detect_gpus("Darwin", mac)[0]["vendor"] == "apple"
    win = lambda *a: "AMD Radeon RX 9070 XT\nMicrosoft Basic Display\n" if a[0] == "powershell" else ""  # noqa: E731
    assert [g["vendor"] for g in ST.detect_gpus("Windows", win)] == ["amd", "unknown"]
    assert ST.detect_gpus("Linux", lambda *a: "") == [], "missing tools mean 'nothing detected', not a crash"


# ── Strata: sources + live server ────────────────────────────────────────

GiB = 2**30


def _hf(gguf_files, gguf_license="apache-2.0", base_license="other", gated=False):
    def handler(req: httpx.Request) -> httpx.Response:
        u = str(req.url)
        if "huggingface.co/api/models/" + ST.HF_GGUF in u:
            return httpx.Response(200, json={"siblings": [{"rfilename": n, "size": int(g * GiB)} for n, g in gguf_files],
                                             "cardData": {"license": gguf_license}, "gated": gated, "downloads": 5})
        if "huggingface.co/api/models/" + ST.HF_BASE in u:
            return httpx.Response(200, json={"siblings": [], "cardData": {"license": base_license}})
        if u == ST.ZIP_URL:
            return httpx.Response(200)
        return httpx.Response(404)
    return handler


def test_st2_confirms_real_downloads_and_surfaces_the_custom_licence():
    L = G.Ledger("st2")
    ST.source_check(L, client(_hf([("IQ2_XS/a-1.gguf", 36.5), ("IQ2_XS/a-2.gguf", 26.8), ("Q2_0/b.gguf", 12)])), G.call)
    assert verdicts(L)["ST2[-]"] == "CONFIRMED"
    ev = evidence(L, "ST2[-]")
    assert "IQ2_XS: 63" in ev and "'other'" in ev and "READ it" in ev


def test_st2_sizes_outside_the_claim_are_not_confirmed():
    L = G.Ledger("st2b")
    ST.source_check(L, client(_hf([("Tiny/a.gguf", 3.0)])), G.call)
    assert verdicts(L)["ST2[-]"] == "NOT_CONFIRMED"


def test_st2_gated_repo_is_not_confirmed_and_unreachable_hf_is_inconclusive():
    L = G.Ledger("st2c")
    ST.source_check(L, client(_hf([("IQ2_XS/a.gguf", 60)], gated=True)), G.call)
    assert verdicts(L)["ST2[-]"] == "NOT_CONFIRMED"
    L = G.Ledger("st2d")
    ST.source_check(L, client(lambda req: httpx.Response(403)), G.call)
    assert verdicts(L)["ST2[-]"] == "INCONCLUSIVE"


def _server(chat_status=200, usage=True):
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/status":
            return httpx.Response(200, json={"loaded": True, "structured_output": {"validation": "jsonschema"}})
        if req.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen3.8-flash-next"}]})
        body = {"choices": [{"message": {"content": "ok"}}], **({"usage": {"completion_tokens": 40}} if usage else {})}
        return httpx.Response(chat_status, json=body)
    return handler


@pytest.fixture
def stubbed_sources(monkeypatch):
    monkeypatch.setattr(ST, "hardware_report", lambda run=None: hw())
    monkeypatch.setattr(ST, "source_check", lambda L, c, call: None)


def test_strata_server_down_is_inconclusive_with_instructions_for_another_pc(stubbed_sources):
    def down(req):
        raise httpx.ConnectError("refused")
    L = G.Ledger("st3")
    assert ST.strata_checks(L, client(down), G.call, G.chat, "http://127.0.0.1:8080", None) is None
    assert verdicts(L)["ST3[server]"] == "INCONCLUSIVE"
    assert "--strata-url" in evidence(L, "ST3[server]") and "--api-key" in evidence(L, "ST3[server]")


def test_strata_server_up_is_tested_with_any_key_and_a_long_local_timeout(stubbed_sources):
    L = G.Ledger("st3b")
    ctx = ST.strata_checks(L, client(_server()), G.call, G.chat, "http://pc:8080", None)
    assert verdicts(L)["ST3[server]"] == "CONFIRMED"
    assert "tok/s" in evidence(L, "ST3[server]")
    assert ctx == {"base": "http://pc:8080/v1", "key": "any-api-key", "models": ["qwen3.8-flash-next"], "timeout_s": 600}


def test_the_any_api_key_claim_is_exercised_by_really_sending_a_key(stubbed_sources):
    """README: 'any API key and any model name work'. Sending NO key would not test that."""
    sent = []

    def handler(req):
        sent.append(req.headers.get("authorization"))
        return _server()(req)

    ST.strata_checks(G.Ledger("st3k"), client(handler), G.call, G.chat, "http://pc:8080", None)
    chat_auth = sent[-1]  # the last request is the chat call
    assert chat_auth == "Bearer any-api-key", f"chat must carry an arbitrary key, got {chat_auth!r}"


def test_strata_chat_failure_is_not_confirmed(stubbed_sources):
    L = G.Ledger("st3c")
    ST.strata_checks(L, client(_server(chat_status=502)), G.call, G.chat, "http://pc:8080", "secret-key-123")
    assert verdicts(L)["ST3[server]"] == "NOT_CONFIRMED"


# ── plumbing: timeout, preflight, claim coverage ─────────────────────────

def test_group_timeout_reaches_litellm_so_a_slow_local_model_is_not_cut_off(monkeypatch):
    import google.adk.models.lite_llm as lite
    seen = {}
    monkeypatch.setattr(lite, "LiteLlm", lambda **kw: seen.update(kw))
    S.build_model(S.Group(5, (), ("m",), bearer_base="http://x/v1", api_key="k", timeout_s=600), "m")
    assert seen["timeout"] == 600
    S.build_model(S.Group(5, (), ("m",), bearer_base="http://x/v1"), "m")
    assert seen["timeout"] == 120, "hosted-API default must be unchanged"


def _args(only):
    return argparse.Namespace(only=only, with_keys=False, repeats=1, keep=False, dry_run=False,
                              strata_url="http://127.0.0.1:8080", strata_key=None)


def test_modelsdev_alone_does_not_need_docker(monkeypatch):
    def boom():
        raise AssertionError("Docker must not be required for --only modelsdev")
    monkeypatch.setattr(G, "docker_ready", boom)
    monkeypatch.setattr(S, "host_reachable", lambda h: (False, "blocked"))
    assert asyncio.run(G.amain(_args("modelsdev"))) == 2
    report = next(iter(Path(G.REPORTS).glob("gateway_claims_*.md"))).read_text()
    assert "api.groq.com not reachable" in report and "VPN looks ON" in report


def test_strata_alone_needs_off_vpn_only_for_our_pipeline_not_for_docker(monkeypatch):
    monkeypatch.setattr(G, "docker_ready", lambda: (_ for _ in ()).throw(AssertionError("no Docker for strata")))
    monkeypatch.setattr(S, "host_reachable", lambda h: (False, "blocked"))
    assert asyncio.run(G.amain(_args("strata"))) == 2
    assert "query1.finance.yahoo.com not reachable" in next(iter(Path(G.REPORTS).glob("gateway_claims_*.md"))).read_text()


def test_every_claim_id_used_anywhere_in_the_script_family_is_defined():
    import re
    used = set()
    for f in _SCRIPTS.glob("gateway_claims_*.py"):
        if f.name.endswith("_data.py"):
            continue
        src = f.read_text()
        used |= set(re.findall(r'L\.add\("([A-Z]{2}\d)"', src)) | set(re.findall(r'"(FL9|OR6|ST3|ST4|ST5)"', src))
    defined = {c.id for c in G.CLAIMS}
    assert used <= defined, used - defined
    assert {"MD1", "MD2", "MD3", "ST1", "ST2", "ST3", "ST4", "ST5"} <= defined


def test_code_puppy_is_listed_as_not_tested_with_its_reason_written_down():
    entry = next((why for what, why in G.NOT_TESTED_REASONS if what.startswith("Code Puppy")), "")
    assert "PydanticAI" in entry and "judgement call" in entry
