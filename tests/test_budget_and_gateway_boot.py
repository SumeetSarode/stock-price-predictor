"""Three failures seen in a real run, each of which made a whole run useless without saying so:

1. cheap chat/json probes spent every provider's whole call budget, so NO model reached the full pipeline test;
2. one platform name the gateway no longer knew (GitHub Models, retired upstream) crashed the entire gateway;
3. a failed pipeline step kept only its class, so "daily quota" and "request too big" looked identical.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import free_model_sweep_offvpn as FM
import gateway_claims_offvpn as G
import sweep_data as D
import sweep_probe as P


def passing(*a, **k):
    return SimpleNamespace(status=200, text='{"answer": "ok"}', error="", json=lambda: {}, evidence=lambda c="": ""), ""


# ── the budget reserve ──────────────────────────────────────────────────

def test_half_of_every_allowance_is_held_back_for_the_pipeline():
    b = P.Budget({"groq": 10}, 40)
    assert b.reserve("groq") == 5 and b.reserve("anything-else") == 20 and P.Budget({}, 1).reserve("x") == 0


def test_take_refuses_to_dip_into_what_is_kept_back_but_defaults_to_the_old_behavior():
    b = P.Budget({"g": 6}, 0)
    assert [b.take("g", 1, keep=3) for _ in range(4)] == [True, True, True, False]
    assert b.left("g") == 3
    assert b.take("g", 3) is True and b.take("g") is False, "no `keep`: the whole remainder is spendable, as before"


@pytest.mark.parametrize("limit", [2, 3, 10, 40, 60])
def test_cheap_probes_can_never_touch_the_pipeline_reserve(tmp_path, limit):
    s = P.Store(tmp_path / "s.jsonl")
    cands = [D.Candidate("direct", "groq", f"m{i}", base="b") for i in range(60)]
    b = P.Budget({"groq": limit}, 99)
    P.probe_provider(cands, passing, None, s, b, 0, lambda x: None, lambda *a: None)
    assert b.left("groq") >= b.reserve("groq"), "the reserve is intact however many models are waiting"
    assert b.used["groq"] <= limit - b.reserve("groq")


def test_the_pipeline_can_spend_the_reserved_calls_after_the_cheap_probes_stop(tmp_path):
    s = P.Store(tmp_path / "s.jsonl")
    cands = [D.Candidate("direct", "groq", f"m{i}", base="b") for i in range(20)]
    b = P.Budget({"groq": 10}, 99)
    P.probe_provider(cands, passing, None, s, b, 0, lambda x: None, lambda *a: None)
    assert b.used["groq"] == 5
    assert [b.take("groq", 2) for _ in range(3)] == [True, True, False], "the pipeline takes 2 per run: two runs fit in 5"


# ── gateway startup tolerates an unknown platform ──────────────────────

def make_boot(supported: set[str]):
    calls: list[set[str]] = []

    def boot(c, keys):
        calls.append(set(keys))
        for p in keys:
            if p not in supported:
                return False, f"[server] Failed to start:\n  unknown provider platform: {p}; keys ...", "img", {}
        return True, "HTTP 200", "img", {"base": "http://x"}
    return boot, calls


def test_a_clean_boot_is_returned_untouched():
    boot, calls = make_boot({"groq", "ollama"})
    ok, _, _, info, keys, dropped = FM.boot_freellmapi(None, {"groq": "k1", "ollama": "k2"}, boot)
    assert ok and dropped == [] and keys == {"groq": "k1", "ollama": "k2"} and len(calls) == 1 and info["base"]


def test_the_retired_platform_is_dropped_and_everything_else_still_starts():
    boot, calls = make_boot({"huggingface", "ollama", "cloudflare"})
    keys = {"huggingface": "h", "github": "g", "ollama": "o", "cloudflare": "c"}
    ok, _, _, _, used, dropped = FM.boot_freellmapi(None, keys, boot)
    assert ok and dropped == ["github"]
    assert set(used) == {"huggingface", "ollama", "cloudflare"}, "only the unsupported one was lost"
    assert len(calls) == 2 and "github" not in calls[1]


def test_several_unsupported_platforms_are_each_dropped_in_turn():
    boot, _ = make_boot({"ollama"})
    ok, _, _, _, used, dropped = FM.boot_freellmapi(None, {"a": "1", "ollama": "o", "b": "2"}, boot)
    assert ok and sorted(dropped) == ["a", "b"] and used == {"ollama": "o"}


def test_an_unrelated_failure_is_reported_as_is_and_not_retried():
    calls = []

    def boot(c, keys):
        calls.append(1)
        return False, "pull failed: no network", "", {}
    ok, detail, _, _, used, dropped = FM.boot_freellmapi(None, {"groq": "k"}, boot)
    assert not ok and "pull failed" in detail and dropped == [] and used == {"groq": "k"} and len(calls) == 1


def test_a_platform_name_we_never_sent_cannot_cause_a_loop():
    calls = []

    def boot(c, keys):
        calls.append(1)
        return False, "unknown provider platform: ghost", "", {}
    ok, _, _, _, _, dropped = FM.boot_freellmapi(None, {"groq": "k"}, boot)
    assert not ok and dropped == [] and len(calls) == 1


def test_the_callers_key_dict_is_not_mutated_and_no_secret_leaks_into_the_drop_list():
    boot, _ = make_boot({"ollama"})
    keys = {"github": "SUPERSECRETVALUE", "ollama": "o"}
    _, detail, _, _, _, dropped = FM.boot_freellmapi(None, keys, boot)
    assert keys == {"github": "SUPERSECRETVALUE", "ollama": "o"}
    assert "SUPERSECRETVALUE" not in repr(dropped) and "SUPERSECRETVALUE" not in detail


def test_github_models_is_no_longer_offered_to_the_gateway():
    assert "github" not in D.FREELLMAPI_PLATFORMS


# ── pipeline failures keep the provider's own words ────────────────────

def rec(step, status, klass="", detail="", seconds=1.0):
    return SimpleNamespace(step=step, status=status, klass=klass, detail=detail, seconds=seconds)


def test_a_failed_step_keeps_the_providers_message_so_the_cause_can_be_read():
    cand = D.Candidate("direct", "groq", "m", base="b")
    msg = "Rate limit reached for model on tokens per minute (TPM): Limit 8000, Requested 9500"
    r = FM.pipeline_result([rec("news_impact", "PASS"), rec("synthesizer", "FAIL", "rate_limited", msg)], cand, 1)
    assert r.verdict == "FAIL" and r.klass == "rate_limited"
    assert "Limit 8000, Requested 9500" in r.detail, "now 'request bigger than the limit' is distinguishable from 'quota used up'"
    assert "news_impact: PASS" in r.detail


def test_secrets_in_a_provider_message_are_scrubbed_before_saving():
    cand = D.Candidate("direct", "groq", "m", base="b")
    G.remember_secret("gsk_TOTALLYSECRET123456")
    r = FM.pipeline_result([rec("synthesizer", "FAIL", "auth", "bad key gsk_TOTALLYSECRET123456 rejected")], cand, 1)
    assert "gsk_TOTALLYSECRET123456" not in r.detail and "rejected" in r.detail


def test_a_passing_run_is_unchanged():
    cand = D.Candidate("direct", "groq", "m", base="b")
    r = FM.pipeline_result([rec("news_impact", "PASS", detail="sentiment=bullish"), rec("synthesizer", "PASS")], cand, 1)
    assert r.verdict == "PASS" and r.klass == ""
