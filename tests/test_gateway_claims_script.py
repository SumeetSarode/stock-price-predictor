"""Offline tests for scripts/gateway_claims_offvpn.py: a fake local gateway, no Docker, no providers.

What these prove: the verdict RULES (a missing-Chromium failure on the base image must be
INCONCLUSIVE, never a refutation), that secrets never reach docker's argv or the saved report,
and that the client logic parses the response shapes the two projects' docs describe. They
cannot prove the real projects behave that way: that is what the off-VPN run is for.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlparse

import httpx
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import gateway_claims_offvpn as G  # noqa: E402     (one shared module object; see test_model_survey_script.py)
import model_survey_offvpn as S  # noqa: E402

UNIFIED, TOKEN = "freellmapi-TESTKEY12345", "tok-SECRETTOKEN-123"


class _Fake(BaseHTTPRequestHandler):
    state: ClassVar[dict] = {}

    def log_message(self, *a):
        pass

    def _send(self, status: int, body, headers: dict | None = None) -> None:
        raw = (body if isinstance(body, str) else json.dumps(body)).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        u, st = urlparse(self.path), self.state
        if u.path in ("/api/ping", "/api/monitoring/health"):
            return self._send(200, {"ok": True})
        if u.path == "/api/settings/api-key":
            ok = self.headers.get("authorization") == f"Bearer {TOKEN}"
            return self._send(200 if ok else 401, {"apiKey": UNIFIED})
        if u.path == "/v1/models":
            rows = st["ready"] if "execution_status=ready" in u.query else st["models"]
            return self._send(200, {"object": "list", "data": rows})
        return self._send(404, {})

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/auth/login":
            return self._send(200, {"token": TOKEN, "email": "claims@local.test"})
        status, text, hdrs = self.state["chat"]
        body = {"choices": [{"message": {"content": text}}]} if status == 200 else text
        return self._send(status, body, hdrs)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    _Fake.state = {
        "models": [{"id": f"groq/m{i}", "owned_by": "groq"} for i in range(400)]
                  + [{"id": "qwen3.8-27b", "owned_by": "groq"}, {"id": "gemini-3.5-flash", "owned_by": "google"}],
        "ready": [{"id": "kilo/free-a", "owned_by": "kilo"}, {"id": "qwen3.8-27b", "owned_by": "groq"}],
        "chat": (200, "ok", {"X-Routed-Via": "kilo/free-a"}),
    }
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    for mod in (G, S):
        monkeypatch.setattr(mod, "REPORTS", tmp_path)
    monkeypatch.setattr(G, "FL_PORT", srv.server_port)
    monkeypatch.setattr(G, "OR_PORT", srv.server_port)
    monkeypatch.setattr(G, "start_container", lambda *a, **k: (True, "started", "img:test"))
    monkeypatch.setattr(G, "docker", lambda *a, **k: (0, ""))
    monkeypatch.setattr(G, "_SECRETS", [])
    yield _Fake.state
    srv.shutdown()


def _verdicts(L) -> dict[str, str]:
    return {f"{f.claim_id}[{f.phase}]": f.verdict for f in L.findings}


# ── secrets hygiene ──────────────────────────────────────────────────────

def test_clean_scrubs_remembered_secrets_and_key_shaped_strings(monkeypatch):
    monkeypatch.setattr(G, "_SECRETS", [])
    G.remember_secret("my-very-secret-value")
    out = G.clean("x my-very-secret-value y gsk_" + "a" * 30 + " Bearer abcdefghijklmnopqrstuv AIza" + "b" * 30)
    assert "my-very-secret-value" not in out
    assert "gsk_aaaa" not in out and "AIzabbbb" not in out and "abcdefghijklmnopqrstuv" not in out


def test_read_keys_maps_names_and_never_includes_cohere_or_dead_accounts(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(
        "GEMINI_API_KEY=g-key-111111\nGROQ_API_KEY=q-key-222222\nCOHERE_API_KEY=c-key-333333\n"
        "TOGETHER_API_KEY=t-key-444444\nSAMBANOVA_API_KEY=s-key-555555\nMISTRAL_API_KEY=\n")
    monkeypatch.setattr(G, "REPO", tmp_path)
    monkeypatch.setattr(G, "_SECRETS", [])
    keys = G.read_keys()
    assert keys == {"google": "g-key-111111", "groq": "q-key-222222"}
    assert "g-key-111111" not in G.clean("leak g-key-111111")


def test_start_container_keeps_secrets_out_of_argv_and_deletes_the_env_file(monkeypatch):
    seen = {}

    def fake_docker(*args, **kw):
        if args[0] == "run":
            path = args[args.index("--env-file") + 1]
            seen["argv"], seen["path"] = " ".join(args), path
            seen["content"], seen["mode"] = Path(path).read_text(), oct(os.stat(path).st_mode & 0o777)
        return 0, "container-id"

    monkeypatch.setattr(G, "docker", fake_docker)
    ok, _, img = G.start_container("n", ("img:a",), 1, 1, {"FREEAPI_CONFIG_JSON": "TOPSECRET"})
    assert ok and img == "img:a"
    assert "TOPSECRET" not in seen["argv"], "secret must not appear in `ps`-visible argv"
    assert "TOPSECRET" in seen["content"]
    assert seen["mode"] == "0o600"
    assert not Path(seen["path"]).exists(), "env-file must be deleted right after docker run"


def test_start_container_falls_back_to_the_next_image_tag(monkeypatch):
    calls = []

    def fake_docker(*args, **kw):
        calls.append(args)
        if args[0] == "pull" and args[1] == "img:latest-web":
            return 1, "manifest unknown"
        return 0, "ok"

    monkeypatch.setattr(G, "docker", fake_docker)
    ok, _, used = G.start_container("n", ("img:latest-web", "img:next-web"), 1, 1, {})
    assert ok and used == "img:next-web"
    assert next(c for c in calls if c[0] == "run")[-1] == "img:next-web"


def test_start_container_reports_every_pull_failure(monkeypatch):
    monkeypatch.setattr(G, "docker", lambda *a, **k: (1, "denied") if a[0] == "pull" else (0, ""))
    ok, detail, used = G.start_container("n", ("img:a", "img:b"), 1, 1, {})
    assert not ok and used == "" and "img:a" in detail and "img:b" in detail


# ── freellmapi flow ──────────────────────────────────────────────────────

def test_freellmapi_phase_confirms_each_claim_and_leaks_nothing(fake, tmp_path, monkeypatch):
    import pydantic

    import price_predictor.agents.news_impact.agent as agent_mod

    class Tiny(pydantic.BaseModel):
        sentiment: str

    fake["chat"] = (200, json.dumps({"sentiment": "bullish"}), {"X-Routed-Via": "kilo/free-a"})
    monkeypatch.setattr(agent_mod, "ImpactAssessment", Tiny)
    L = G.Ledger("t1")
    with httpx.Client(timeout=10) as c:
        ctx = G.freellmapi_phase(L, c, "keyless", {})
    v = _verdicts(L)
    assert v == {"FL1[keyless]": "CONFIRMED", "FL2[keyless]": "CONFIRMED", "FL3[keyless]": "CONFIRMED",
                 "FL4[keyless]": "CONFIRMED", "FL5[keyless]": "CONFIRMED", "FL6[keyless]": "CONFIRMED",
                 "FL8[keyless]": "CONFIRMED"}
    assert ctx["models"][0] == "auto" and "qwen3.8-27b" in ctx["models"]
    saved = L.jsonl.read_text()
    assert UNIFIED not in saved and TOKEN not in saved, "unified key / session token leaked into the report"


def test_freellmapi_schema_violations_are_reported_not_hidden(fake, monkeypatch):
    import pydantic

    import price_predictor.agents.news_impact.agent as agent_mod

    class Strict(pydantic.BaseModel):
        needs_this: str

    monkeypatch.setattr(agent_mod, "ImpactAssessment", Strict)
    fake["chat"] = (200, json.dumps({"sentiment": "bullish"}), {})
    L = G.Ledger("t2")
    with httpx.Client(timeout=10) as c:
        G.freellmapi_phase(L, c, "keyless", {})
    assert _verdicts(L)["FL8[keyless]"] == "NOT_CONFIRMED"


def test_freellmapi_keyed_phase_marks_failover_inconclusive_when_no_dead_hop_was_hit(fake, monkeypatch):
    L = G.Ledger("t3")
    with httpx.Client(timeout=10) as c:
        G.freellmapi_phase(L, c, "keyed", {"groq": "gk-123456"})
    assert _verdicts(L)["FL7[keyed]"] == "INCONCLUSIVE"


def test_freellmapi_keyed_phase_confirms_failover_when_a_fallback_header_appears(fake):
    fake["chat"] = (200, "ok", {"X-Fallback-Attempts": "2", "X-Routed-Via": "groq/x"})
    L = G.Ledger("t4")
    with httpx.Client(timeout=10) as c:
        G.freellmapi_phase(L, c, "keyed", {"groq": "gk-123456"})
    assert _verdicts(L)["FL7[keyed]"] == "CONFIRMED"


def test_freellmapi_that_never_starts_is_a_finding_not_a_crash(fake, monkeypatch):
    monkeypatch.setattr(G, "start_container", lambda *a, **k: (False, "pull failed: denied", ""))
    L = G.Ledger("t5")
    with httpx.Client(timeout=10) as c:
        assert G.freellmapi_phase(L, c, "keyless", {}) is None
    assert _verdicts(L) == {"FL1[keyless]": "INCONCLUSIVE"}


# ── OmniRoute flow: the rule that matters most ───────────────────────────

CHROMIUM_502 = "[502] Cloudflare Playground browser session failed: browserType.launch: Executable doesn't exist"


def test_missing_chromium_on_the_base_image_is_inconclusive_never_a_refutation(fake):
    fake["models"] = [{"id": "oc/free-a"}]
    fake["chat"] = (502, CHROMIUM_502, {})
    L = G.Ledger("t6")
    with httpx.Client(timeout=10) as c:
        G.omniroute_phase(L, c, "base", ("img:latest",))
    v = _verdicts(L)
    assert v["OR2[base]"] == "INCONCLUSIVE", "documented prerequisite, so the claim is NOT refuted"
    assert v["OR1[base]"] == "CONFIRMED"
    # the SAME fairness rule must cover every probe, not just `auto` (regression: OR5 was unfair)
    assert v["OR4[base]"] == v["OR5[base]"] == "INCONCLUSIVE"


def test_same_failure_on_the_web_image_does_count_against_the_claim(fake):
    fake["models"] = [{"id": "oc/free-a"}]
    fake["chat"] = (502, CHROMIUM_502, {})
    L = G.Ledger("t7")
    with httpx.Client(timeout=10) as c:
        G.omniroute_phase(L, c, "web", ("img:latest-web",))
    v = _verdicts(L)
    assert v["OR3[web]"] == "NOT_CONFIRMED"
    assert v["OR4[web]"] == v["OR5[web]"] == "NOT_CONFIRMED", "no excuse left on the web image"


def test_omniroute_zero_config_is_confirmed_when_auto_answers(fake):
    fake["models"] = [{"id": "oc/free-a"}, {"id": "oc/free-b"}]
    ctx_holder = {}
    L = G.Ledger("t8")
    with httpx.Client(timeout=10) as c:
        ctx_holder["ctx"] = G.omniroute_phase(L, c, "base", ("img:latest",))
    v = _verdicts(L)
    assert v["OR2[base]"] == v["OR4[base]"] == v["OR5[base]"] == "CONFIRMED"
    assert ctx_holder["ctx"]["models"][0] == "auto"
    assert ctx_holder["ctx"]["key"] is None


# ── pipeline verdicts + report + abort paths ─────────────────────────────

def _w(recs):
    w = S.Writer("unit")
    w.recs = recs
    return w


def test_pipeline_verdict_needs_BOTH_steps_in_every_run(fake):
    R = S.Rec
    full = [R("g", "m", 1, "news_impact", "PASS"), R("g", "m", 1, "synthesizer", "PASS")]
    half = [*full, R("g", "m", 2, "news_impact", "PASS"), R("g", "m", 2, "synthesizer", "FAIL")]
    none = [R("g", "m", 1, "news_impact", "FAIL"), R("g", "m", 1, "synthesizer", "FAIL")]
    out = {}
    for name, recs in (("full", full), ("half", half), ("none", none), ("empty", [])):
        L = G.Ledger(f"p-{name}")
        G.pipeline_finding(L, _w(recs), "FL9", "g", "keyless")
        out[name] = L.findings[0].verdict
    assert out == {"full": "CONFIRMED", "half": "INCONCLUSIVE", "none": "NOT_CONFIRMED", "empty": "INCONCLUSIVE"}


def test_report_contains_ledger_not_tested_section_and_scrubbed_logs(fake):
    G.remember_secret("SUPERSECRETKEY")
    L = G.Ledger("r1")
    L.add("FL1", "keyless", G.CONFIRMED, "started with SUPERSECRETKEY")
    md = G.render(L, _w([]), {"outcome": "completed"}, {"freellmapi [keyless]": G.clean("log SUPERSECRETKEY")})
    assert "| FL1 | freellmapi |" in md and "NOT TESTED, and why" in md
    assert "SUPERSECRETKEY" not in md
    assert "Nothing here is a verdict that a project is useless" in md


def test_aborts_with_a_report_file_when_docker_is_not_running(fake, monkeypatch, capsys):
    monkeypatch.setattr(G, "docker_ready", lambda: (False, "Cannot connect to the Docker daemon"))
    args = argparse.Namespace(only="freellmapi,omniroute", with_keys=False, repeats=1, keep=False, dry_run=False)
    assert asyncio.run(G.amain(args)) == 2
    reports = list(Path(G.REPORTS).glob("gateway_claims_*.md"))
    assert reports and "Docker daemon not reachable" in reports[0].read_text()


def test_aborts_when_the_vpn_is_on(fake, monkeypatch):
    monkeypatch.setattr(G, "docker_ready", lambda: (True, "28.4.0"))
    monkeypatch.setattr(S, "host_reachable", lambda h: (False, "DNS blocked"))
    args = argparse.Namespace(only="omniroute", with_keys=False, repeats=1, keep=False, dry_run=False)
    assert asyncio.run(G.amain(args)) == 2
    assert "VPN looks ON" in next(iter(Path(G.REPORTS).glob("gateway_claims_*.md"))).read_text()


def test_every_claim_id_used_by_the_flows_is_defined():
    import re
    src = (_SCRIPTS / "gateway_claims_offvpn.py").read_text()
    used = set(re.findall(r'L\.add\("([A-Z]{2}\d)"', src)) | set(re.findall(r'"(FL9|OR6)"', src))
    assert used <= {c.id for c in G.CLAIMS}, used - {c.id for c in G.CLAIMS}
