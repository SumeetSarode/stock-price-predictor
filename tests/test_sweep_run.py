"""End-to-end tests for the free-model sweep orchestrator, against a fake internet.

httpx.MockTransport plays every provider; the predictor's two LLM steps are replaced by a stub that
records PASS/FAIL. What this proves: the wiring (discover -> probe -> pipeline -> report), budgets,
resume, terms-blocking, secret hygiene and graceful degradation. What it cannot prove is how any real
provider behaves: that is what the off-VPN run measures.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import free_model_sweep_offvpn as FM  # noqa: E402
import gateway_claims_offvpn as G  # noqa: E402
import model_survey_offvpn as S  # noqa: E402
import sweep_data as D  # noqa: E402
import sweep_probe as PR  # noqa: E402
import sweep_report as REP  # noqa: E402
from gateway_claims_catalog import CATALOG_URL  # noqa: E402

GROQ_KEY = "gsk_FAKEKEYFAKEKEYFAKEKEY000000"
COHERE_KEY = "cohere-fake-key-must-never-be-used"
FREE = {"input": 0, "output": 0}
CATALOG = {
    "groq": {"models": {"good-model": {"structured_output": True, "last_updated": "2026-09-01"}}},
    "navy": {"models": {"n1": {"cost": FREE, "structured_output": True}}},
    "cohere": {"models": {"c1": {"cost": FREE}}},
    "alibaba-coding-plan": {"models": {"q1": {"cost": FREE}}},
}
SNAPSHOT = {"navy": {"models": {"n1": {"cost": FREE}, "old": {"cost": FREE}}}}


class Net:
    """The fake internet: records every request so tests can assert what was (not) called."""

    def __init__(self) -> None:
        self.hits: list[tuple[str, str]] = []
        self.groq_models = ["good-model", "bad-model", "whisper-large-v3"]
        self.chat_models: list[str] = []    # every model a chat call was made for, in order
        self.flaky: dict[str, int] = {}     # model -> how many chat calls get a 429 before it recovers

    def handler(self, req: httpx.Request) -> httpx.Response:
        host, path = req.url.host, req.url.path
        self.hits.append((host, path))
        if str(req.url) == CATALOG_URL:
            return httpx.Response(200, json=CATALOG)
        if "code_puppy" in str(req.url):
            return httpx.Response(200, json=SNAPSHOT)
        if host == "api.groq.com" and path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": m} for m in self.groq_models]})
        if host == "api.groq.com" and path.endswith("/chat/completions"):
            body = json.loads(req.content)
            self.chat_models.append(body["model"])
            if self.flaky.get(body["model"], 0) > 0:
                self.flaky[body["model"]] -= 1
                return httpx.Response(429, json={"error": "rate limited"})
            if body["model"] != "good-model":
                return httpx.Response(404, json={"error": "model not found"})
            text = '{"answer": "ok"}' if "response_format" in body else "ok"
            return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})
        if host == "gpu.local":       # a Strata server on the LAN
            if path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "qwen-local"}]})
            if path.endswith("/status"):
                return httpx.Response(200, json={"ok": True})
            body = json.loads(req.content)
            text = '{"answer": "ok"}' if "response_format" in body else "ok"
            return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})
        if host == "127.0.0.1":
            raise httpx.ConnectError("refused")
        return httpx.Response(500)

    def count(self, host: str, suffix: str = "") -> int:
        return sum(1 for h, p in self.hits if h == host and p.endswith(suffix))


@pytest.fixture
def world(monkeypatch, tmp_path):
    net, calls = Net(), {"run_once": 0}
    (tmp_path / ".env").write_text(f"GROQ_API_KEY={GROQ_KEY}\nCOHERE_API_KEY={COHERE_KEY}\nMISTRAL_API_KEY=\n")
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(net.handler), **kw))
    for mod in (FM, S):
        monkeypatch.setattr(mod, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(FM, "REPO", tmp_path)
    monkeypatch.setattr(FM, "pace_for", lambda provider: 0)
    monkeypatch.setattr(G, "docker", lambda *a, **k: (0, ""))
    monkeypatch.setattr(FM, "ensure_docker", lambda: False)   # the real one would launch a real app
    monkeypatch.setattr(S, "host_reachable", lambda host: (True, "ok"))

    async def prepare():
        return object(), "tv", "neutral"

    async def run_once(predictor, group, model_id, run, tv, neutral, w):
        calls["run_once"] += 1
        name = S.group_name(group)
        w.add(S.Rec(name, model_id, run, "news_impact", "PASS", 1.0))
        w.add(S.Rec(name, model_id, run, "synthesizer", "PASS", 2.0))

    monkeypatch.setattr(S, "prepare", prepare)
    monkeypatch.setattr(S, "run_once", run_once)
    monkeypatch.setattr(S, "GROUPS", dict(S.GROUPS))
    return net, calls, tmp_path / "reports"


def args(**over):
    base = dict(only=["direct", "strata"], with_keys=False, resume=None, budget=0, repeats=2, workers=2,
                max_pipeline=40, max_minutes=0, strata_url="http://127.0.0.1:8080", strata_key=None,
                discover_only=False, keep=False, dry_run=False, retry_passes=0, retry_wait=0, wait_vpn_minutes=0)
    return argparse.Namespace(**{**base, **over})


def verdicts(reports: Path) -> dict[tuple[str, str], dict]:
    f = max(reports.glob("*_verdicts.jsonl"))
    return {(r["provider"], r["model"]): r for r in map(json.loads, f.read_text().splitlines())}


def report_text(reports: Path) -> str:
    return max(f for f in reports.glob("free_model_sweep_*.md")).read_text()


def run(a):
    return asyncio.run(FM.amain(a))


# ── the whole flow ───────────────────────────────────────────────────────

def test_full_run_gives_every_model_a_verdict_or_a_written_reason(world):
    _net, calls, reports = world
    assert run(args()) == 0
    v = verdicts(reports)
    assert v[("groq", "good-model")]["verdict"] == "USEFUL"
    assert v[("groq", "bad-model")]["verdict"] == "NOT_USEFUL" and "not_found" in v[("groq", "bad-model")]["reason"]
    assert v[("groq", "whisper-large-v3")]["verdict"] == "UNTESTED" and "NOT_CHAT" in v[("groq", "whisper-large-v3")]["reason"]
    assert "TERMS_BLOCKED" in v[("cohere", "*")]["reason"]
    assert v[("cohere", "c1")]["verdict"] == "UNTESTED" and "TERMS_BLOCKED" in v[("cohere", "c1")]["reason"]
    assert v[("alibaba-coding-plan", "q1")]["verdict"] == "UNTESTED" and "NOT_FREE" in v[("alibaba-coding-plan", "q1")]["reason"]
    assert "NEEDS_SIGNUP" in v[("navy", "n1")]["reason"] and v[("navy", "n1")]["sources"] == ["models.dev", "code_puppy"]
    assert "stale" in v[("navy", "old")]["reason"]
    assert "NO_ENDPOINT" in v[("strata", "Qwen3.8-Flash-Next")]["reason"]
    assert calls["run_once"] == 2, "only the one model that answered chat is put through the pipeline, twice"


def test_the_report_says_what_to_do_next(world):
    _, _, reports = world
    run(args())
    md = report_text(reports)
    assert "USEFUL: candidates for the chain (1)" in md
    assert "`groq/good-model`" in md and "CHAIN_AGENTIC" in md, "a usable model must say HOW to add it"
    assert "Accounts to open" in md and "navy" in md and "NAVY_API_KEY" in md
    assert "compatible with our pipeline, not accurate" in md
    assert "coding-plan" in md and "LIST PRICE is $0" in md, "the models.dev blind spot must be disclosed"


def test_terms_blocked_providers_are_never_called_and_their_keys_never_read(world):
    net, _, reports = world
    run(args())
    assert not any("cohere" in h for h, _ in net.hits), "Cohere must not be contacted"
    assert COHERE_KEY not in "".join(f.read_text() for f in reports.glob("*"))


def test_no_key_ever_lands_in_any_output_file(world):
    _, _, reports = world
    run(args())
    for f in reports.glob("*"):
        assert GROQ_KEY not in f.read_text(), f.name


def test_discover_only_lists_everything_and_makes_no_llm_call(world):
    net, calls, reports = world
    assert run(args(discover_only=True)) == 0
    assert net.count("api.groq.com", "/chat/completions") == 0 and calls["run_once"] == 0
    v = verdicts(reports)
    assert v[("groq", "good-model")]["verdict"] == "UNTESTED"
    assert ("navy", "n1") in v, "discovery still builds the full ledger"


# ── budgets, resume, degradation ─────────────────────────────────────────

def test_the_call_budget_caps_spend_and_says_what_was_left(world):
    net, _, reports = world
    run(args(budget=2))
    assert net.count("api.groq.com", "/chat/completions") == 2, "exactly the budget, never more"
    v = verdicts(reports)
    assert v[("groq", "bad-model")]["verdict"] == "UNTESTED" and "BUDGET" in v[("groq", "bad-model")]["reason"]


def test_resume_repeats_nothing_that_is_finished(world):
    net, calls, reports = world
    run(args())
    chats, runs = net.count("api.groq.com", "/chat/completions"), calls["run_once"]
    time.sleep(1.1)  # a new stamp, so a new output file
    run(args(resume="latest"))
    assert net.count("api.groq.com", "/chat/completions") == chats, "no probe repeated"
    assert calls["run_once"] == runs, "no pipeline run repeated"
    assert verdicts(reports)[("groq", "good-model")]["verdict"] == "USEFUL", "earlier results still count"


def test_resume_latest_ignores_the_verdicts_file(world):
    _, _, reports = world
    run(args())
    time.sleep(1.1)
    run(args(resume="latest"))   # would crash on the verdicts schema if latest_results picked it
    assert PR.latest_results(reports).name.endswith(".jsonl") and "_verdicts" not in PR.latest_results(reports).name


def test_resume_retries_a_rate_limit_but_not_a_permanent_failure(world):
    net, _calls, reports = world
    run(args())
    first = max(f for f in reports.glob("free_model_sweep_*.jsonl") if "_verdicts" not in f.name)
    rows = [json.loads(ln) for ln in first.read_text().splitlines()]
    bad = next(r for r in rows if r["uid"].endswith("|bad-model") and r["level"] == "chat")
    assert bad["klass"] == "not_found"
    before = net.count("api.groq.com", "/chat/completions")
    time.sleep(1.1)
    run(args(resume="latest"))
    assert net.count("api.groq.com", "/chat/completions") == before, "a permanent not_found is not retried"


def test_vpn_on_aborts_before_any_llm_call_but_still_writes_a_report(world, monkeypatch):
    net, _calls, reports = world
    monkeypatch.setattr(S, "host_reachable", lambda host: (False, "blocked"))
    assert run(args()) == 2
    assert "VPN looks ON" in report_text(reports)
    assert net.count("api.groq.com", "/chat/completions") == 0


def test_docker_down_skips_gateways_but_the_direct_route_still_runs(world, monkeypatch):
    _net, _, reports = world
    monkeypatch.setattr(G, "docker_ready", lambda: (False, "daemon not running"))
    assert run(args(only=["direct", "freellmapi", "omniroute"])) == 0
    md = report_text(reports)
    assert "SKIPPED (freellmapi, omniroute): Docker is not running" in md
    assert verdicts(reports)[("groq", "good-model")]["verdict"] == "USEFUL"


def test_the_deadline_stops_new_pipeline_runs(world):
    _, calls, reports = world
    run(args(max_minutes=-1))   # a deadline already in the past
    assert calls["run_once"] == 0
    assert verdicts(reports)[("groq", "good-model")]["verdict"] == "UNTESTED"


# ── pipeline orchestration units ─────────────────────────────────────────

def rec(step, status="PASS", klass="", secs=1.0):
    return S.Rec("g", "m", 1, step, status, secs, klass=klass)


def test_a_run_is_a_pass_only_if_BOTH_steps_pass():
    cand = D.Candidate("direct", "groq", "m")
    ok = FM.pipeline_result([rec("news_impact"), rec("synthesizer")], cand, 1)
    assert (ok.verdict, ok.klass, ok.seconds) == ("PASS", "", 2.0)
    papered = FM.pipeline_result([rec("news_impact", "FAIL", "bad_output"), rec("synthesizer")], cand, 1)
    assert papered.verdict == "FAIL" and papered.klass == "bad_output", "a neutral-baseline cover-up is not a pass"
    both = FM.pipeline_result([rec("news_impact", "FAIL", "timeout"), rec("synthesizer", "FAIL", "rate_limited")], cand, 2)
    assert both.klass == "rate_limited,timeout" and both.run == 2
    assert FM.pipeline_result([], cand, 1).verdict == "FAIL"


def test_group_for_picks_the_production_path_the_anonymous_path_or_the_bearer_path(world):
    prod = D.Candidate("direct", "groq", "m", base="b", key="k", litellm_id="groq/m")
    anon = D.Candidate("anon", "ovh", "m", base="http://ovh/v1", anonymous=True)
    bearer = D.Candidate("direct", "nvidia", "m", base="http://nv/v1", key="nk")
    strata = D.Candidate("strata", "strata", "m", base="http://s/v1", key="any")
    assert FM.group_for(prod).models == ("groq/m",) and not FM.group_for(prod).bearer_base
    assert FM.group_for(anon).anon_base == "http://ovh/v1"
    g = FM.group_for(bearer)
    assert (g.bearer_base, g.api_key, g.timeout_s) == ("http://nv/v1", "nk", 120)
    assert FM.group_for(strata).timeout_s == 600, "a local model gets a long timeout"
    assert FM.group_for(prod) is FM.group_for(prod), "one group per route+provider"


def test_pipeline_pass_honours_the_cap_the_order_and_the_budget(world, tmp_path):
    _, _calls, _ = world
    store = PR.Store(tmp_path / "s.jsonl")
    W = S.Writer("unit")
    cands = [D.Candidate("direct", "groq", n, hint_structured=h) for n, h in (("a", None), ("b", True), ("c", None))]
    for c in cands:
        store.add(PR.Result(c.uid, "chat", "PASS"))
    store.add(PR.Result(cands[2].uid, "json", "PASS"))
    order = []
    orig = S.run_once

    async def spy(predictor, group, model_id, run, tv, neutral, w):
        order.append(model_id)
        await orig(predictor, group, model_id, run, tv, neutral, w)
    S.run_once = spy
    try:
        asyncio.run(FM.pipeline_pass(W, (object(), "tv", "n"), cands, store, PR.Budget({}, 99), repeats=1, cap=2,
                                     deadline=float("inf")))
    finally:
        S.run_once = orig
    assert order == ["c", "b"], "json-proven first, then catalog-flagged; the cap leaves 'a' out"


def test_pipeline_pass_records_a_budget_skip_and_a_crash_instead_of_dying(world, tmp_path, monkeypatch):
    store = PR.Store(tmp_path / "s.jsonl")
    a, b = D.Candidate("direct", "groq", "a"), D.Candidate("direct", "groq", "b")
    for c in (a, b):
        store.add(PR.Result(c.uid, "chat", "PASS"))

    async def boom(*args, **kw):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(S, "run_once", boom)
    asyncio.run(FM.pipeline_pass(S.Writer("unit2"), (object(), "tv", "n"), [a, b], store, PR.Budget({"groq": 2}, 99),
                                 repeats=1, cap=0, deadline=float("inf")))
    assert store.get(a.uid, "pipeline", 1).klass == "other" and "kaboom" in store.get(a.uid, "pipeline", 1).detail
    skipped = store.get(b.uid, "pipeline", 1)
    assert skipped.verdict == "SKIP" and skipped.klass == "budget"


# ── report units ─────────────────────────────────────────────────────────

def test_verdict_for_never_calls_an_untestable_candidate_useful(tmp_path):
    store = PR.Store(tmp_path / "s.jsonl")
    cand = D.Candidate("ledger", "navy", "n1", D.NEEDS_SIGNUP, note="needs a key")
    assert REP.verdict_for(cand, store)[0] == "UNTESTED"
    live = D.Candidate("direct", "groq", "m")
    store.add(PR.Result(live.uid, "chat", "SKIP", "budget"))
    verdict, why = REP.verdict_for(live, store)
    assert verdict == "UNTESTED" and "BUDGET" in why


@pytest.mark.parametrize("cand, needle", [
    (D.Candidate("direct", "groq", "m", litellm_id="groq/m"), "CHAIN_AGENTIC"),
    (D.Candidate("direct", "nvidia", "m"), "provider entry"),
    (D.Candidate("anon", "ovh", "m"), "provider entry"),
    (D.Candidate("freellmapi", "kilo", "m"), "freellmapi gateway"),
])
def test_how_to_add_is_honest_about_what_the_app_supports_today(cand, needle):
    assert needle in REP.how_to_add(cand)


# ── unattended behaviour: retries, checkpoints, crashes, defaults ────────

def test_a_rate_limit_is_retried_automatically_and_the_model_ends_up_useful(world):
    net, calls, reports = world
    net.flaky["good-model"] = 1
    run(args(retry_passes=2))
    assert verdicts(reports)[("groq", "good-model")]["verdict"] == "USEFUL"
    assert calls["run_once"] == 2


def test_without_retry_passes_the_same_rate_limit_stays_inconclusive(world):
    net, calls, reports = world
    net.flaky["good-model"] = 1
    run(args(retry_passes=0))
    v = verdicts(reports)[("groq", "good-model")]
    assert v["verdict"] == "INCONCLUSIVE" and "rate_limited" in v["reason"]
    assert calls["run_once"] == 0


def test_retry_passes_never_re_call_a_permanent_failure_or_a_budget_skip(world):
    net, _, _ = world
    run(args(retry_passes=3, budget=2))
    assert net.count("api.groq.com", "/chat/completions") == 2, "budget 2 is a hard ceiling even with 3 retry passes"


def test_a_permanent_failure_is_called_exactly_once_however_many_retry_passes(world):
    net, _, _ = world
    run(args(retry_passes=3))
    bad = [h for h in net.hits if h == ("api.groq.com", "/openai/v1/chat/completions")]
    assert len(bad) == 3, "good-model: chat + json (2), bad-model: ONE chat (1), nothing more across 3 retry passes"


def test_a_crash_midway_still_leaves_a_current_report_and_says_so(world, monkeypatch):
    _, _, reports = world

    async def boom(*a, **k):
        raise RuntimeError("strata exploded")
    monkeypatch.setattr(FM, "sweep_strata", boom)
    assert run(args()) == 1
    md = report_text(reports)
    assert "CRASHED: RuntimeError: strata exploded" in md and "resumes automatically" in md
    assert verdicts(reports)[("groq", "good-model")]["verdict"] == "USEFUL", "work done before the crash is kept"


def test_the_report_is_checkpointed_after_each_route_not_only_at_the_end(world, monkeypatch):
    _, _, reports = world
    seen = []

    async def spy(*a, **k):
        seen.append(report_text(reports))     # what is on disk when the NEXT route starts
    monkeypatch.setattr(FM, "sweep_strata", spy)
    run(args())
    assert "in progress: direct done" in seen[0], "the direct route's results were already on disk"


def test_a_stable_latest_copy_always_matches_the_newest_report(world):
    _, _, reports = world
    run(args())
    assert (reports / FM.LATEST_MD).read_text() == report_text(reports)
    assert (reports / FM.LATEST_VERDICTS).exists()


def test_resume_latest_ignores_the_stable_latest_copy_too(world):
    _, _, reports = world
    run(args())
    assert "latest" not in PR.latest_results(reports).name, "a copy must never be mistaken for the raw probe log"


def test_a_strata_server_on_your_lan_is_picked_up_from_dot_env_with_no_flag(world):
    net, _, reports = world
    env = reports.parent / ".env"
    env.write_text(env.read_text() + "STRATA_URL=http://gpu.local:8080\n")
    run(args(strata_url="http://127.0.0.1:8080"))
    assert verdicts(reports)[("strata", "qwen-local")]["verdict"] == "USEFUL"
    assert net.count("gpu.local") > 0


def test_the_defaults_mean_zero_manual_steps():
    a = FM.normalize(FM.build_parser().parse_args([]))
    assert (a.with_keys, a.resume, a.retry_passes, a.max_minutes, a.wait_vpn_minutes) == (True, "latest", 3, 480, 30)
    assert a.only == ["direct", "freellmapi", "omniroute", "strata"], "every source, by default"
    assert FM.normalize(FM.build_parser().parse_args(["--fresh"])).resume is None
    assert FM.normalize(FM.build_parser().parse_args(["--no-keys"])).with_keys is False


def test_with_a_budget_for_one_call_the_one_call_goes_to_deepseek_even_when_listed_last(world):
    net, _, reports = world
    net.groq_models = ["tiny-1b-chat", "acme-model", "deepseek-v4-flash"]     # the provider lists DeepSeek LAST
    run(args(budget=1))
    assert net.chat_models == ["deepseek-v4-flash"], "the single call was spent on the most promising model"
    v = verdicts(reports)
    assert "BUDGET" in v[("groq", "tiny-1b-chat")]["reason"], "and the rest are honestly reported as not reached"
