"""Offline tests for the free-model sweep's core: verdict rules, probes, resumable store, budget,
scheduling and discovery. httpx.MockTransport plays every remote; nothing touches the network."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import gateway_claims_offvpn as G  # noqa: E402
import sweep_data as D  # noqa: E402
import sweep_probe as P  # noqa: E402
import sweep_sources as SRC  # noqa: E402


def resp(status=200, content="ok", reasoning=None, text=None):
    msg = {"content": content, **({"reasoning_content": reasoning} if reasoning else {})}
    body = text if text is not None else json.dumps({"choices": [{"message": msg}]})
    return G.Resp(status, 0.1, body)


def fake_chat(r):
    return lambda c, base, key, model, prompt="", **kw: (r, "")


CAND = D.Candidate("direct", "groq", "m1", base="http://x/v1", key="k")


# ── static triage ────────────────────────────────────────────────────────

@pytest.mark.parametrize("provider, model, status", [
    ("cohere", "command-r", "TERMS_BLOCKED"),
    ("alibaba-coding-plan", "qwen3-coder", "NOT_FREE"),
    ("kimi-code-plan-cn", "k2", "NOT_FREE"),
    ("xiaomi-token-plan-ams", "mimo", "NOT_FREE"),
    ("opencode", "big-pickle", "NOT_FREE"),
    ("sambanova", "llama", "NOT_FREE"),
    ("planetscale-ai", "m", "TESTABLE"),   # contains 'plan' but is not a plan provider
    ("minimax-token-plan-cn", "m2", "NOT_FREE"),
    ("google", "text-embedding-004", "NOT_CHAT"),
    ("groq", "whisper-large-v3", "NOT_CHAT"),
    ("groq", "llama-guard-4-12b", "NOT_CHAT"),
    ("google", "gemini-2.5-flash-image", "NOT_CHAT"),
    ("groq", "openai/gpt-oss-120b", "TESTABLE"),
    ("google", "gemini-2.5-flash", "TESTABLE"),
    ("nvidia", "deepseek-ai/deepseek-v4.1-flash", "TESTABLE"),
    ("groq", "qwen/qwen3.8-27b", "TESTABLE"),
])
def test_static_triage(provider, model, status):
    assert D.classify_candidate(provider, model)[0] == status


# ── the verdict rules ────────────────────────────────────────────────────

def run(full=True, klasses=()):
    return {"full": full, "klasses": list(klasses)}


@pytest.mark.parametrize("chat, js, runs, verdict", [
    (None, None, [run(), run()], "USEFUL"),
    (None, None, [run(), run(False, ["bad_output"])], "PARTIAL"),
    (None, None, [run(False, ["rate_limited"]), run(False, ["rate_limited"])], "INCONCLUSIVE"),
    (None, None, [run(False, ["bad_output"]), run(False, ["bad_output"])], "NOT_USEFUL"),
    (None, None, [run(False, ["bad_output"]), run(False, ["timeout"])], "INCONCLUSIVE"),
    ({"verdict": "FAIL", "klass": "payment"}, None, [], "NOT_USEFUL"),
    ({"verdict": "FAIL", "klass": "not_found"}, None, [], "NOT_USEFUL"),
    ({"verdict": "FAIL", "klass": "unsupported"}, None, [], "NOT_USEFUL"),
    ({"verdict": "FAIL", "klass": "auth"}, None, [], "INCONCLUSIVE"),
    ({"verdict": "FAIL", "klass": "rate_limited"}, None, [], "INCONCLUSIVE"),
    ({"verdict": "FAIL", "klass": "timeout"}, None, [], "INCONCLUSIVE"),
    ({"verdict": "PASS"}, {"verdict": "FAIL"}, [], "UNTESTED"),
    (None, None, [], "UNTESTED"),
])
def test_final_verdict(chat, js, runs, verdict):
    assert D.final_verdict(chat, js, runs)[0] == verdict


def test_a_rate_limit_is_never_reported_as_a_model_being_useless():
    for klass in ("rate_limited", "overloaded", "timeout", "network", "empty"):
        assert D.final_verdict({"verdict": "FAIL", "klass": klass}, None, [])[0] == "INCONCLUSIVE", klass


def test_usefulness_is_about_compatibility_not_accuracy_and_the_docs_say_so():
    assert "does NOT mean accurate" in D.__doc__


# ── probes ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("status, text, error, klass", [
    (200, "", "", ""), (402, "", "", "payment"), (401, "", "", "auth"), (403, "", "", "auth"),
    (404, "", "", "not_found"), (429, "", "", "rate_limited"), (500, "", "", "overloaded"),
    (503, "", "", "overloaded"), (400, "response_format unsupported", "", "unsupported"),
    (400, "Rate limit reached", "", "rate_limited"), (422, "daily quota exceeded", "", "rate_limited"),
    (0, "", "ReadTimeout: timeout", "timeout"), (0, "", "ConnectError: refused", "network"),
])
def test_classify_http(status, text, error, klass):
    assert P.classify_http(status, text, error) == klass


def test_chat_probe_passes_on_content_and_on_reasoning_only_replies():
    assert P.probe_chat(fake_chat(resp()), None, CAND).verdict == "PASS"
    only_reasoning = P.probe_chat(fake_chat(resp(content="", reasoning="hmm")), None, CAND)
    assert only_reasoning.verdict == "PASS" and "reasoning" in only_reasoning.detail


def test_chat_probe_failure_classes():
    assert P.probe_chat(fake_chat(resp(content="")), None, CAND).klass == "empty"
    assert P.probe_chat(fake_chat(resp(429, text="slow down")), None, CAND).klass == "rate_limited"
    assert P.probe_chat(fake_chat(resp(402, text="pay")), None, CAND).klass == "payment"
    down = G.Resp(0, 1.0, error="ConnectError: refused")
    assert P.probe_chat(fake_chat(down), None, CAND).klass == "network"


def test_json_probe_valid_fenced_prose_and_unsupported():
    assert P.probe_json(fake_chat(resp(content='{"answer": "ok"}')), None, CAND).verdict == "PASS"
    fenced = P.probe_json(fake_chat(resp(content='```json\n{"answer": "ok"}\n```')), None, CAND)
    assert fenced.verdict == "PASS" and "fences" in fenced.detail
    prose = P.probe_json(fake_chat(resp(content="Sure! here you go")), None, CAND)
    assert (prose.verdict, prose.klass) == ("FAIL", "bad_output")
    wrong = P.probe_json(fake_chat(resp(content='{"answer": 5}')), None, CAND)
    assert wrong.verdict == "FAIL", "a non-string answer violates the schema"
    assert P.probe_json(fake_chat(resp(400, text="bad param")), None, CAND).klass == "unsupported"


def test_probes_send_the_model_the_key_and_a_json_schema():
    seen = {}

    def chat(c, base, key, model, prompt="", **kw):
        seen.update(base=base, key=key, model=model, kw=kw)
        return resp(), ""
    P.probe_json(chat, None, CAND)
    assert (seen["base"], seen["key"], seen["model"]) == ("http://x/v1", "k", "m1")
    assert seen["kw"]["response_format"]["type"] == "json_schema"


# ── store, resume, budget, scheduling ────────────────────────────────────

def test_store_persists_every_result_and_resumes_from_a_file(tmp_path):
    s1 = P.Store(tmp_path / "a.jsonl")
    s1.add(P.Result("u1", "chat", "PASS"))
    s1.add(P.Result("u2", "chat", "FAIL", "not_found"))
    s1.add(P.Result("u3", "chat", "FAIL", "rate_limited"))
    assert len((tmp_path / "a.jsonl").read_text().splitlines()) == 3
    s2 = P.Store(tmp_path / "b.jsonl", (tmp_path / "a.jsonl",))
    assert not s2.needs("u1", "chat"), "a pass is never repeated"
    assert not s2.needs("u2", "chat"), "a permanent failure is never repeated"
    assert s2.needs("u3", "chat"), "a rate limit IS retried on resume"
    assert s2.needs("never-seen", "chat")


def test_a_pipeline_run_with_a_mix_of_failure_classes_is_retried_if_any_is_transient(tmp_path):
    s = P.Store(tmp_path / "s.jsonl")
    s.add(P.Result("u", "pipeline", "FAIL", "bad_output,timeout", run=1))
    s.add(P.Result("v", "pipeline", "FAIL", "bad_output,not_found", run=1))
    assert s.needs("u", "pipeline", 1), "one transient class among the failures makes the run worth retrying"
    assert not s.needs("v", "pipeline", 1), "all-permanent failures are final"


def test_transient_pending_picks_only_retryable_failures_and_never_budget_skips(tmp_path):
    s = P.Store(tmp_path / "s.jsonl")
    mk = lambda n, st="TESTABLE": D.Candidate("direct", "groq", n, st)  # noqa: E731
    slow, dead, skipped, fine, signup = (mk(n) for n in ("slow", "dead", "skipped", "fine", "x"))
    s.add(P.Result(slow.uid, "chat", "FAIL", "rate_limited"))
    s.add(P.Result(dead.uid, "chat", "FAIL", "not_found"))
    s.add(P.Result(skipped.uid, "chat", "SKIP", "budget"))
    s.add(P.Result(fine.uid, "chat", "PASS"))
    s.add(P.Result(fine.uid, "pipeline", "FAIL", "bad_output,overloaded", run=2))
    gated = mk("gated", "NEEDS_SIGNUP")
    s.add(P.Result(gated.uid, "chat", "FAIL", "timeout"))
    got = {c.model for c in P.transient_pending([slow, dead, skipped, fine, signup, gated], s, repeats=2)}
    assert got == {"slow", "fine"}


def test_budget_exhaustion_is_a_retryable_skip_not_a_failure(tmp_path):
    s = P.Store(tmp_path / "s.jsonl")
    s.add(P.Result("u", "chat", "SKIP", "budget"))
    assert s.needs("u", "chat")


def test_budget_counts_per_provider_and_has_a_default():
    b = P.Budget({"groq": 2}, default=1)
    assert [b.take("groq"), b.take("groq"), b.take("groq")] == [True, True, False]
    assert b.take("newprov") and not b.take("newprov")
    assert b.left("groq") == 0


def test_probe_provider_runs_chat_then_json_paced_and_skips_json_after_a_failed_chat(tmp_path):
    s, sleeps = P.Store(tmp_path / "s.jsonl"), []
    good, dead = D.Candidate("direct", "groq", "good", base="b"), D.Candidate("direct", "groq", "dead", base="b")

    def chat(c, base, key, model, prompt="", **kw):
        return (resp() if model == "good" else resp(404, text="gone")), ""
    P.probe_provider([good, dead], chat, None, s, P.Budget({}, 99), 5.0, sleeps.append, lambda *a: None)
    assert s.get(good.uid, "chat").verdict == "PASS" and s.get(good.uid, "json").verdict == "FAIL"
    assert s.get(dead.uid, "chat").klass == "not_found" and s.get(dead.uid, "json") is None
    assert sleeps == [5.0, 5.0, 5.0], "every call is paced"


def test_probe_provider_stops_at_the_budget_and_says_so(tmp_path):
    s = P.Store(tmp_path / "s.jsonl")
    cands = [D.Candidate("direct", "groq", f"m{i}", base="b") for i in range(3)]
    P.probe_provider(cands, lambda *a, **k: (resp(), ""), None, s, P.Budget({"groq": 2}, 99), 0, lambda x: None, lambda *a: None)
    skipped = [r for r in s.rows.values() if r.verdict == "SKIP"]
    assert skipped and all(r.klass == "budget" for r in skipped)
    assert sum(1 for r in s.rows.values() if r.verdict != "SKIP") == 2, "exactly the budget was spent"


def test_probe_provider_does_not_repeat_finished_probes_on_resume(tmp_path):
    s = P.Store(tmp_path / "s.jsonl")
    s.add(P.Result(CAND.uid, "chat", "PASS"))
    s.add(P.Result(CAND.uid, "json", "PASS"))
    calls = []
    P.probe_provider([CAND], lambda *a, **k: calls.append(1) or (resp(), ""), None, s, P.Budget({}, 9), 0, lambda x: None, lambda *a: None)
    assert calls == []


def test_scheduler_picks_the_provider_that_is_ready_soonest():
    q = {"a": [1], "b": [1], "c": []}
    assert P.pick_next(q, {"a": 50.0, "b": 10.0}, now=20.0) == ("b", 0.0)
    assert P.pick_next(q, {"a": 50.0, "b": 90.0}, now=20.0) == ("a", 30.0)
    assert P.pick_next({"a": [], "b": []}, {}, 0.0) == (None, 0.0)


def test_pipeline_order_puts_proven_json_models_first_and_excludes_none(tmp_path):
    s = P.Store(tmp_path / "s.jsonl")
    a, b, c = (D.Candidate("direct", "p", n, hint_structured=h) for n, h in (("a", None), ("b", True), ("c", None)))
    s.add(P.Result(c.uid, "json", "PASS"))
    assert [x.model for x in P.pipeline_order([a, b, c], s)] == ["c", "b", "a"]
    assert len(P.pipeline_order([a, b, c], s)) == 3


# ── discovery ────────────────────────────────────────────────────────────

def router(table):
    def handler(req: httpx.Request) -> httpx.Response:
        for prefix, (status, body) in table.items():
            if str(req.url).startswith(prefix):
                return httpx.Response(status, json=body)
        return httpx.Response(500)
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=5)


def test_env_keys_use_generic_names_and_aliases_and_never_return_cohere():
    env = {"GROQ_API_KEY": "g", "NVIDIA_NIM_API_KEY": "n", "COHERE_API_KEY": "c", "HF_TOKEN": "h",
           "MISTRAL_API_KEY": "  ", "NAVY_API_KEY": "v"}
    keys = SRC.read_env_keys(env, ["groq", "nvidia", "cohere", "huggingface", "mistral", "navy", "agnes"])
    assert keys == {"groq": "g", "nvidia": "n", "huggingface": "h", "navy": "v"}


def test_list_models_parses_openai_and_google_shapes_and_reports_failures():
    spec = D.SPECS["groq"]
    c = router({spec.models_url: (200, {"data": [{"id": "a"}, {"id": "b"}]})})
    assert [m["id"] for m in SRC.list_models(G.call, c, spec, "k")[0]] == ["a", "b"]
    g = D.SPECS["google"]
    c = router({"https://generativelanguage.googleapis.com/v1beta/models": (200, {"models": [
        {"name": "models/gemini-x", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/embed-1", "supportedGenerationMethods": ["embedContent"]}]})})
    rows = SRC.list_models(G.call, c, g, "k")[0]
    assert rows == [{"id": "gemini-x", "chat": True}, {"id": "embed-1", "chat": False}]
    rows, why = SRC.list_models(G.call, router({}), spec, "k")
    assert rows is None and "500" in why


def _direct_router():
    return router({
        D.SPECS["groq"].models_url: (200, {"data": [{"id": "openai/gpt-oss-120b"}, {"id": "whisper-large-v3"}]}),
        D.SPECS["openrouter"].models_url: (200, {"data": [{"id": "a/b:free"}, {"id": "c/d"}, {"id": "e/f"}]}),
        D.SPECS["ovh"].models_url: (200, {"data": [{"id": "Qwen3.8-27B"}]}),
    })


def test_direct_candidates_cover_keyed_and_anonymous_providers_only():
    catalog = {"groq": {"models": {"openai/gpt-oss-120b": {"structured_output": True}}}}
    cands, meta = SRC.direct_candidates(G.call, _direct_router(), catalog, {"groq": "gk", "openrouter": "ok"})
    by = {(c.provider, c.model): c for c in cands}
    g = by[("groq", "openai/gpt-oss-120b")]
    assert (g.route, g.key, g.litellm_id, g.hint_structured, g.status) == ("direct", "gk", "groq/openai/gpt-oss-120b", True, "TESTABLE")
    assert by[("groq", "whisper-large-v3")].status == "NOT_CHAT"
    assert ("openrouter", "a/b:free") in by and ("openrouter", "c/d") not in by, "only :free routes are candidates"
    assert meta["not_free"]["openrouter"] == 2
    ovh = by[("ovh", "Qwen3.8-27B")]
    assert ovh.route == "anon" and ovh.key is None and ovh.anonymous
    assert not any(p in ("mistral", "nvidia", "google", "cerebras") for p, _ in by), "no key, so not probed directly"
    cohere = by[("cohere", "*")]
    assert cohere.status == "TERMS_BLOCKED" and cohere.route == "ledger"


def test_direct_candidates_fall_back_to_known_models_or_mark_the_gap():
    cands, meta = SRC.direct_candidates(G.call, router({}), {}, {"groq": "gk"})
    assert meta["list_errors"].keys() >= {"groq", "ovh", "kilo"}
    kilo = [c.model for c in cands if c.provider == "kilo"]
    assert kilo == list(D.SPECS["kilo"].fallback), "a failed list must not silently drop a provider"
    groq = [c for c in cands if c.provider == "groq"]
    assert groq[0].status == "NO_ENDPOINT" and groq[0].model == "(model list unavailable)"


def test_gateway_candidates_triage_by_execution_status_and_skip_direct_providers():
    rows = [{"id": "m1", "owned_by": "kilo", "execution_status": "ready"},
            {"id": "m2", "owned_by": "navy", "execution_status": "needsKey"},
            {"id": "m3", "owned_by": "agnes", "execution_status": "exhausted"},
            {"id": "m4", "owned_by": "groq", "execution_status": "ready"},
            {"id": "m5", "owned_by": "cohere", "execution_status": "ready"}]
    cands = SRC.gateway_candidates(rows, "freellmapi", "http://gw/v1", "uk", G.provider_of, {"groq"})
    st = {c.model: c.status for c in cands}
    assert st == {"m1": "TESTABLE", "m2": "NEEDS_SIGNUP", "m3": "EXHAUSTED", "m5": "TERMS_BLOCKED"}
    assert "NAVY_API_KEY" in next(c for c in cands if c.model == "m2").note
    assert "m4" not in st, "groq is tested directly; never spend its quota twice"


def test_omniroute_candidates_name_the_provider_from_the_id_prefix():
    cands = SRC.omniroute_candidates([{"id": "oc/free-a"}, {"id": "auto"}], "omniroute-base", "http://or/v1")
    assert [(c.provider, c.key) for c in cands] == [("oc", None), ("omniroute", None)]


def test_ledger_lists_every_free_model_we_cannot_call_and_flags_the_traps():
    free = {"input": 0, "output": 0}
    catalog = {
        "groq": {"models": {"g1": {"cost": free}}},                       # covered: skipped
        "navy": {"models": {"n1": {"cost": free, "structured_output": True}, "n2": {"cost": {"input": 1, "output": 1}}}},
        "alibaba-coding-plan": {"models": {"q1": {"cost": free}}},
        "cohere": {"models": {"c1": {"cost": free}}},
    }
    snapshot = {"navy": {"models": {"n1": {"cost": free}, "stale": {"cost": free}}}}
    cands = SRC.ledger_candidates(catalog, snapshot, covered={"groq"})
    by = {(c.provider, c.model): c for c in cands}
    assert ("groq", "g1") not in by and ("navy", "n2") not in by
    assert by[("navy", "n1")].status == "NEEDS_SIGNUP" and by[("navy", "n1")].hint_structured is True
    assert by[("navy", "n1")].sources == ("models.dev", "code_puppy")
    assert by[("alibaba-coding-plan", "q1")].status == "NOT_FREE"
    assert by[("cohere", "c1")].status == "TERMS_BLOCKED"
    assert by[("navy", "stale")].status == "NO_ENDPOINT" and "stale" in by[("navy", "stale")].note


def test_ledger_knows_models_dev_calls_ovh_ovhcloud_so_a_tested_provider_is_not_listed_as_signup():
    free = {"input": 0, "output": 0}
    catalog = {"ovhcloud": {"models": {"m": {"cost": free}}}, "zhipuai": {"models": {"z": {"cost": free}}}}
    cands = SRC.ledger_candidates(catalog, {}, covered={"ovh"})
    assert [(c.provider, c.model) for c in cands] == [("zhipuai", "z")], "ovhcloud is OUR 'ovh': already covered"


def test_every_freellmapi_platform_we_may_send_a_key_to_excludes_terms_blocked_and_retired_ones():
    assert not set(D.FREELLMAPI_PLATFORMS) & (set(D.TERMS_BLOCKED_PROVIDERS) | {"sambanova", "opencode", "reka", "custom"})
    assert {"nvidia", "mistral", "cerebras", "google", "groq", "openrouter"} <= set(D.FREELLMAPI_PLATFORMS)


def test_unlock_list_ranks_accounts_by_how_many_structured_free_models_they_unlock():
    mk = lambda p, m, h: D.Candidate("ledger", p, m, "NEEDS_SIGNUP", hint_structured=h)  # noqa: E731
    cands = [mk("a", "1", False), mk("a", "2", False), mk("a", "3", False), mk("b", "1", True), mk("b", "2", True),
             D.Candidate("ledger", "z", "1", "TESTABLE")]
    ranked = SRC.unlock_list(cands)
    assert [r["provider"] for r in ranked] == ["b", "a"], "flagged-structured first, then raw count; TESTABLE ignored"
    assert ranked[0]["env"] == "B_API_KEY" and ranked[0]["models"] == 2


def test_gateway_routes_get_their_own_budget_so_one_cannot_drain_another():
    assert D.Candidate("direct", "groq", "m").budget_key == "groq"
    assert D.Candidate("anon", "ovh", "m").budget_key == "ovh"
    base = D.Candidate("omniroute-base", "oc", "m").budget_key
    web = D.Candidate("omniroute-web", "oc", "m").budget_key
    assert base != web and base == "omniroute-base:oc"


def test_candidate_row_never_serialises_the_key():
    row = D.Candidate("direct", "groq", "m", key="SECRET-KEY-123").row()
    assert "SECRET-KEY-123" not in json.dumps(row)
    assert D.Candidate("direct", "groq", "m", litellm_id="groq/m").test_id == "groq/m"
    assert D.Candidate("direct", "groq", "m").test_id == "m"


# ── probe order: a limited budget goes where a pass is likeliest ─────────

@pytest.mark.parametrize("model, tiny", [
    ("meta/llama-3.1-8b-instruct", True), ("qwen-1.5b-chat", True), ("phi-4-mini", True), ("gpt-nano", True),
    ("qwen3-30b-a3b", False),          # 30B MoE: 'a3b' is ACTIVE params, not a 3B model
    ("gpt-oss-120b", False), ("gemma-3-27b-it", False), ("qwen3-coder-480b-a35b", False),
    ("deepseek-ai/deepseek-v4-flash", False), ("llama-3.3-70b", False),
])
def test_tiny_detects_small_models_without_misreading_moe_active_params(model, tiny):
    assert bool(P.TINY.search(model)) is tiny


def test_deepseek_and_other_strong_families_are_probed_before_unknown_and_tiny_models():
    names = ["meta/llama-3.2-1b-instruct", "acme-chat-xl", "deepseek-ai/deepseek-v4-flash", "mistral-7b-instruct",
             "qwen3-30b-a3b", "nvidia/nemotron-ultra-253b"]
    ordered = [c.model for c in P.probe_order([D.Candidate("direct", "nvidia", n) for n in names])]
    assert set(ordered[:3]) == {"deepseek-ai/deepseek-v4-flash", "qwen3-30b-a3b", "nvidia/nemotron-ultra-253b"}
    assert ordered[3] == "acme-chat-xl", "unknown but not tiny comes before tiny"
    assert set(ordered[4:]) == {"meta/llama-3.2-1b-instruct", "mistral-7b-instruct"}


def test_a_catalog_flag_beats_a_name_and_nothing_is_ever_dropped():
    flagged = D.Candidate("direct", "x", "plain-model", hint_structured=True)
    famous = D.Candidate("direct", "x", "deepseek-v4")
    tiny = D.Candidate("direct", "x", "foo-1b")
    assert P.probe_order([tiny, famous, flagged]) == [flagged, famous, tiny]
    assert len(P.probe_order([tiny] * 5)) == 5


def test_probe_order_is_stable_so_equal_candidates_keep_the_providers_own_order():
    cands = [D.Candidate("direct", "x", f"model-{i}") for i in range(6)]
    assert P.probe_order(cands) == cands


def test_a_candidate_survives_a_round_trip_through_its_row_with_the_key_re_supplied_from_env():
    c = D.Candidate("direct", "groq", "m", base="https://b/v1", key="SECRET", litellm_id="groq/m", anonymous=False,
                    note="n", hint_structured=True, sources=("models.dev",))
    row = c.row()
    assert "SECRET" not in str(row) and "key" not in row
    back = D.Candidate.from_row(row, {"groq": "FROM-ENV"})
    assert (back.route, back.provider, back.model, back.base, back.litellm_id, back.hint_structured, back.sources) == \
        ("direct", "groq", "m", "https://b/v1", "groq/m", True, ("models.dev",))
    assert back.key == "FROM-ENV", "the key is re-read from .env, never taken from a saved file"
    assert D.Candidate.from_row(row).key is None
    assert D.Candidate.from_row(D.Candidate("anon", "ovh", "m", anonymous=True).row()).anonymous is True
