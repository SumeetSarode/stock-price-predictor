"""Offline tests for scripts/model_survey_offvpn.py (localhost only, no providers).

The load-bearing one: anonymous endpoints must receive NO Authorization header.
A fake bearer key is treated as an invalid key and rejected (OVH: 'authentication
failed'), which would make a working anonymous tier look dead.
"""
from __future__ import annotations

import json
import sys
import threading
from datetime import UTC
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

# A plain import, NOT a private copy loaded from the file: a second copy in sys.modules meant other test files patched
# one module object while the code under test called another, so their "fake" network checks silently went live.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import model_survey_offvpn as S


@pytest.fixture
def fake_server():
    seen: list[tuple[str, dict, dict]] = []

    class _H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
            seen.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
            msg = {"role": "assistant", "content": json.dumps({"sentiment": "bullish"})}
            out = json.dumps({
                "id": "x", "object": "chat.completion", "created": 1, "model": body.get("model", "m"),
                "choices": [{"index": 0, "finish_reason": "stop", "message": msg}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1", seen
    srv.shutdown()


@pytest.mark.asyncio
async def test_anon_probe_sends_no_authorization_header(fake_server):
    base, seen = fake_server
    ok, _ = await S.anon_probe(S.Group(1, ("h",), ("m",), anon_base=base))
    assert ok
    assert "authorization" not in seen[-1][1]


@pytest.mark.asyncio
async def test_anon_model_sends_no_authorization_header_through_adk(fake_server):
    """The REAL ADK -> LiteLlm path, with a response_schema, still carries no auth."""
    from google.adk.models.llm_request import LlmRequest
    from google.genai import types

    from price_predictor.agents.news_impact.agent import ImpactAssessment

    base, seen = fake_server
    model = S.build_model(S.Group(1, ("h",), ("fake",), anon_base=base), "fake")
    req = LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
        config=types.GenerateContentConfig(response_schema=ImpactAssessment),
    )
    out = [r async for r in model.generate_content_async(req)]
    assert out, "no response came back through the anonymous client"
    path, headers, body = seen[-1]
    assert "authorization" not in headers
    assert path.endswith("/chat/completions")
    assert body["model"] == "fake"


@pytest.mark.parametrize(("text", "klass"), [
    ("Payment required to access this resource", "payment"),
    ("Forbidden: authentication failed", "auth"),
    ("GeminiException 503 high demand", "overloaded"),
    ("Connection timed out. Timeout passed=60.0", "timeout"),
    ("Rate limit exceeded", "rate_limited"),
    ("Generated JSON does not match the expected schema", "bad_output"),
    ("cohere_chat does not support parameters: ['response_format']", "unsupported_params"),
    ("something unexpected", "other"),
])
def test_classify_real_error_strings(text, klass):
    assert S.classify(text) == klass


def _fake_predictor(news_ok: bool):
    async def news(_t):
        if not news_ok:
            raise ValueError("news boom")
        return SimpleNamespace(sentiment="bullish", confidence=0.7)

    async def synth(_si):
        logger.warning("synthesizer: unparseable LLM output (demo)")
        raise RuntimeError("429 Rate limit exceeded")

    return SimpleNamespace(
        _resolve_ticker=lambda t: t, IST=UTC,
        _news_impact_agent=SimpleNamespace(model=None),
        _synthesizer_agent=SimpleNamespace(model=None),
        run_news_impact_agent=news, synthesize_with_guardrails=synth,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("news_ok", "expect_input"), [(True, "own"), (False, "neutral-baseline")])
async def test_run_once_records_each_step_separately(tmp_path, monkeypatch, news_ok, expect_input):
    import price_predictor.prediction.inputs as inputs

    monkeypatch.setattr(S, "REPORTS", tmp_path)
    monkeypatch.setattr(S, "build_model", lambda g, m: "PINNED")
    monkeypatch.setattr(inputs, "SynthesisInput", lambda **kw: kw)
    P = _fake_predictor(news_ok)
    group = S.GROUPS["groq"]
    w = S.Writer("test")

    await S.run_once(P, group, "groq/x", 1, tv=object(), neutral="NEUTRAL", w=w)

    news, synth = w.recs
    assert (news.step, news.status) == ("news_impact", "PASS" if news_ok else "FAIL")
    assert (synth.step, synth.status, synth.klass) == ("synthesizer", "FAIL", "rate_limited")
    assert synth.synth_input == expect_input
    assert any("unparseable LLM output" in ln for ln in synth.log), "loguru lines must be captured"
    assert P._news_impact_agent.model == P._synthesizer_agent.model == "PINNED"
    lines = w.jsonl.read_text().splitlines()
    assert len(lines) == 2, "each step must be flushed to the jsonl immediately"
    assert json.loads(lines[1])["klass"] == "rate_limited"


def test_markdown_verdicts_and_skips():
    R = S.Rec
    recs = [
        R("g", "all-pass", 1, "news_impact", "PASS", 5), R("g", "all-pass", 1, "synthesizer", "PASS", 6),
        R("g", "half", 1, "news_impact", "PASS", 5), R("g", "half", 1, "synthesizer", "FAIL", 6, "x", "bad_output"),
        R("g", "none", 1, "news_impact", "FAIL", 5, "x", "timeout"),
        R("ovh", "skipped", 0, "group", "SKIP", 0, "anonymous call rejected: HTTP 403", "auth"),
    ]
    md = S.render_markdown(recs, {"outcome": "completed", "preflight": {"h": "OK"}})
    assert "**CANDIDATE**" in md
    assert "**PARTIAL**" in md
    assert "**NO**" in md
    assert "| skipped |" in md or "skipped" in md
