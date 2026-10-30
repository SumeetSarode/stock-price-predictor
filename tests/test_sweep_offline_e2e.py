"""End-to-end, offline: the whole off-VPN sweep (sweep_untested_offvpn.main) against fake providers.

Every awkward reply seen in the wild is served here: content as a list of parts, content null, garbage JSON,
a probe that raises, 404 for a renamed model, rate limits, a paid model, a non-chat name. The run must finish,
write its report, resolve the renamed model, and resume without re-asking settled questions. No network, no
Docker, no processes touched.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import free_model_sweep_offvpn as FM
import gateway_claims_offvpn as G
import model_survey_offvpn as S
import sweep_data as D
import sweep_env as SE
import sweep_untested_offvpn as U

ALPHA = "http://alpha.test/v1"
BETA = "http://beta.test/v1"
ALPHA_MODELS = ["good-1", "listy-1", "nullc-1", "garbage-1", "explode-1", "gone-v1", "busy-1", "paid-1",
                "text-embedding-1"]
CATALOG = {"alpha": {"models": {"gone-v2": {}, "unrelated-9": {}}}}


def _reply(content):
    return G.Resp(200, 0.1, json.dumps({"choices": [{"message": {"content": content}}]}))


def fake_call(c, method, url, **kw):
    if "models.dev" in url:
        return G.Resp(200, 0.1, json.dumps(CATALOG))
    if "raw.githubusercontent" in url or "127.0.0.1" in url:
        return G.Resp(0, 0.1, error="ConnectError: refused")
    if url == f"{ALPHA}/models":
        return G.Resp(200, 0.1, json.dumps({"data": [{"id": m} for m in ALPHA_MODELS]}))
    if url == f"{BETA}/models":
        return G.Resp(200, 0.1, json.dumps({"data": [{"id": "m:free"}, {"id": "paid-only"}]}))
    model = (kw.get("json") or {}).get("model", "")
    wants_json = "response_format" in (kw.get("json") or {})
    if model == "explode-1":
        raise RuntimeError("provider sent something unreadable")
    if model == "gone-v1":
        return G.Resp(404, 0.1, '{"error": "model not found"}')
    if model == "busy-1":
        return G.Resp(429, 0.1, '{"error": "rate limit"}')
    if model == "paid-1":
        return G.Resp(402, 0.1, '{"error": "payment required"}')
    if model == "text-embedding-1":
        return G.Resp(400, 0.1, '{"error": "not a chat model"}')
    if model == "garbage-1":
        return G.Resp(200, 0.1, "<html>oops</html>")
    if model == "nullc-1":
        return _reply(None)
    answer = '{"answer": "ok"}' if wants_json else "ok"
    if model == "listy-1":
        return _reply([{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": answer}])
    return _reply(answer)


async def fake_run_once(P, group, model_id, run, tv, neutral, w):
    ok = model_id.split("/")[-1] in ("good-1", "gone-v2", "m:free")
    for step in ("news_impact", "synthesizer"):
        w.add(S.Rec("g", model_id, run, step, "PASS" if ok else "FAIL", 0.1, "", "" if ok else "bad_output"))


@pytest.fixture
def offline(tmp_path, monkeypatch):
    saved = dict(D.SPECS)
    D.SPECS.clear()
    D.SPECS.update({
        "alpha": D.Spec(ALPHA, f"{ALPHA}/models", "bearer", "all", "fake keyed tier", 0, 400, ("ALPHA_KEY",)),
        "beta": D.Spec(BETA, f"{BETA}/models", None, "suffix", "fake anonymous tier", 0, 400, (), None, True),
        "deepseek": D.Spec("http://never.test/v1", "http://never.test/models", "bearer", "all", "x", 0, 1, ("X",)),
    })
    monkeypatch.setattr(D, "NON_CHAT", D.NON_CHAT)
    monkeypatch.setattr(S, "REPORTS", tmp_path)
    monkeypatch.setattr(FM, "REPORTS", tmp_path)
    monkeypatch.setattr(S, "GROUPS", dict(S.GROUPS))
    monkeypatch.setattr(G, "call", fake_call)
    monkeypatch.setattr(S, "run_once", fake_run_once)
    monkeypatch.setattr(S, "prepare", lambda: asyncio.sleep(0, result=(None, None, None)))
    monkeypatch.setattr(SE, "wait_for_off_vpn", lambda *a, **k: True)
    monkeypatch.setattr(SE, "tee_stdout", lambda path: None)
    monkeypatch.setattr(SE, "keep_awake", lambda: None)
    monkeypatch.setattr(FM, "ensure_docker", lambda: False)
    monkeypatch.setattr(U, "stop_earlier_sweeps", lambda: "")
    monkeypatch.setattr(U.os, "chdir", lambda path: None)
    monkeypatch.setattr("dotenv.dotenv_values", lambda path: {"ALPHA_KEY": "secret-alpha-key"})
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda s, result=None: real_sleep(0, result))
    yield tmp_path
    D.SPECS.clear()
    D.SPECS.update(saved)


def _verdicts(reports: Path) -> dict[str, dict]:
    return {r["uid"]: r for r in map(json.loads, (reports / FM.LATEST_VERDICTS).read_text().splitlines())}


def test_full_run_finishes_and_judges_every_model(offline):
    assert U.main() == 0
    v = _verdicts(offline)
    verdict = {uid.split("|", 2)[2]: r["verdict"] for uid, r in v.items() if uid.split("|")[1] in ("alpha", "beta")}
    assert verdict["good-1"] == D.USEFUL
    assert verdict["gone-v2"] == D.USEFUL                     # found because gone-v1 was "not found"
    assert verdict["m:free"] == D.USEFUL
    assert verdict["listy-1"] == D.NOT_USEFUL                 # answered chat (list content read), failed the pipeline
    assert verdict["paid-1"] == D.NOT_USEFUL and verdict["gone-v1"] == D.NOT_USEFUL
    assert verdict["text-embedding-1"] == D.NOT_USEFUL        # probed, not skipped by its name
    assert verdict["busy-1"] == D.INCONCLUSIVE
    assert {"nullc-1", "garbage-1", "explode-1"} <= set(verdict)
    assert "paid-only" not in verdict                         # beta only offers ':free' routes for free
    assert not any("|deepseek|" in uid for uid in v)          # never called
    report = (offline / FM.LATEST_MD).read_text()
    assert "completed" in report and "secret-alpha-key" not in report
    assert v["direct|alpha|gone-v1"]["note"].count("tested instead: gone-v2") == 1


def test_second_run_resumes_and_does_not_reask_passes(offline, monkeypatch):
    assert U.main() == 0
    asked = []

    def counting_call(c, method, url, **kw):
        if method == "POST":
            asked.append(kw["json"]["model"])
        return fake_call(c, method, url, **kw)
    monkeypatch.setattr(G, "call", counting_call)
    assert U.main() == 0
    assert "good-1" not in asked and "gone-v2" not in asked   # settled: kept from the first run
    assert "busy-1" in asked                                  # failed: asked again
    assert _verdicts(offline)["direct|alpha|good-1"]["verdict"] == D.USEFUL


def test_stops_before_spending_anything_if_the_pipeline_cannot_start(offline, monkeypatch):
    asked = []

    async def broken_prepare():
        raise RuntimeError("no prices (VPN on?)")
    monkeypatch.setattr(S, "prepare", broken_prepare)
    monkeypatch.setattr(G, "call", lambda c, m, url, **kw: asked.append(url) or fake_call(c, m, url, **kw))
    assert U.main() == 3
    assert not [u for u in asked if u.endswith("/chat/completions")]
    assert "ABORTED: the pipeline test could not start" in (offline / FM.LATEST_MD).read_text()
