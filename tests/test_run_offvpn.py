"""The single entry point: ordering, hand-offs, and what it leaves behind, with both phases faked."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import free_model_sweep_offvpn as FM
import gateway_claims_offvpn as G
import model_scorecard_offvpn as MS
import model_survey_offvpn as S
import run_offvpn as RUN
import sweep_env as SE


@pytest.fixture
def env(monkeypatch, tmp_path):
    reports = tmp_path / "reports"
    monkeypatch.setattr(FM, "REPORTS", reports)
    st = {"order": [], "claims_args": None, "sweep_args": None, "opened": [], "docker": True, "vpn_off": True,
          "keep_awake": 0, "docker_calls": 0, "claims_exc": None, "sweep_exc": None, "sweep_code": 0,
          "scorecard_args": None, "scorecard_exc": None, "scorecard_code": 0}

    async def claims(a):
        st["order"].append("claims")
        st["claims_args"] = a
        print("claims says hello")
        if st["claims_exc"]:
            raise st["claims_exc"]
        return 0

    async def sweep(a):
        st["order"].append("sweep")
        st["sweep_args"] = a
        if a.dry_run:               # like the real one: a dry run writes nothing
            return 0
        reports.mkdir(exist_ok=True)
        (reports / FM.LATEST_VERDICTS).write_text(
            json.dumps({"provider": "groq", "model": "good-model", "route": "direct", "verdict": "USEFUL"}) + "\n"
            + json.dumps({"provider": "navy", "model": "n1", "route": "ledger", "verdict": "UNTESTED"}) + "\n")
        if st["sweep_exc"]:
            raise st["sweep_exc"]
        return st["sweep_code"]

    async def scorecard(a):
        st["order"].append("scorecard")
        st["scorecard_args"] = a
        if a.dry_run:
            return 0
        reports.mkdir(exist_ok=True)
        (reports / MS.LATEST_MD).write_text("# Model scorecard\n\n## Headline\n\n4 models, 20 cases. BEATS_BASELINE: 1\n\n"
                                            "**1 model(s) beat the baseline:** `groq/g1`\n\n## Ranked\n\n| table |\n")
        if st["scorecard_exc"]:
            raise st["scorecard_exc"]
        return st["scorecard_code"]

    def docker():
        st["docker_calls"] += 1
        return st["docker"]

    monkeypatch.setattr(G, "amain", claims)
    monkeypatch.setattr(FM, "amain", sweep)
    monkeypatch.setattr(MS, "amain", scorecard)
    monkeypatch.setattr(FM, "ensure_docker", docker)
    monkeypatch.setattr(SE, "keep_awake", lambda: st.__setitem__("keep_awake", st["keep_awake"] + 1))
    monkeypatch.setattr(SE, "open_file", lambda p: st["opened"].append(p))
    monkeypatch.setattr(SE, "wait_for_off_vpn", lambda reachable, minutes, **k: st["vpn_off"])
    monkeypatch.setattr(S, "host_reachable", lambda h: (True, "ok"))
    monkeypatch.setattr(sys, "stdout", sys.stdout)       # pytest restores whatever the Tee installs
    return st, reports


def go(*argv):
    return RUN.run_all(FM.normalize(RUN.build_parser().parse_args(list(argv))))


def test_no_arguments_runs_the_claims_check_then_the_sweep_then_the_scorecard_then_opens_the_summary(env):
    st, reports = env
    assert go() == 0
    assert st["order"] == ["claims", "sweep", "scorecard"]
    assert st["opened"] == [reports / RUN.INDEX]
    index = (reports / RUN.INDEX).read_text()
    assert "claims check:** finished with exit code 0" in index and "sweep:** finished" in index
    assert "groq / `good-model` (direct)" in index, "the useful models are listed right in the summary"
    assert "UNTESTED: 1" in index
    assert "scorecard:** finished with exit code 0" in index
    assert "Scorecard headline" in index and "**1 model(s) beat the baseline:** `groq/g1`" in index, "the answer is IN the summary"
    assert "| table |" not in index, "only the headline, not the whole report"


def test_the_claims_check_is_configured_from_the_same_flags_and_runs_lighter(env):
    st, _ = env
    go("--no-keys", "--strata-url", "http://gpu.local:1")
    c = st["claims_args"]
    assert c.with_keys is False and c.strata_url == "http://gpu.local:1" and c.repeats == 1
    assert set(c.only.split(",")) == {"modelsdev", "strata", "freellmapi", "omniroute"}
    assert st["sweep_args"].with_keys is False, "one flag, both phases"


def test_with_no_docker_the_claims_check_drops_the_gateways_instead_of_aborting(env):
    st, reports = env
    st["docker"] = False
    assert go() == 0
    assert set(st["claims_args"].only.split(",")) == {"modelsdev", "strata"}
    assert st["order"] == ["claims", "sweep", "scorecard"], "everything else still runs"
    assert "docker:** could not be started" in (reports / RUN.INDEX).read_text()


def test_a_vpn_that_stays_on_tests_nothing_and_says_exactly_that(env):
    st, reports = env
    st["vpn_off"] = False
    assert go() == 2
    assert st["order"] == []
    assert "still ON" in (reports / RUN.INDEX).read_text()
    assert st["opened"] == [], "no pretend-success window"


def test_skip_claims_and_no_open(env):
    st, _ = env
    go("--skip-claims", "--no-open")
    assert st["order"] == ["sweep", "scorecard"] and st["opened"] == []


def test_a_dry_run_launches_nothing_locks_nothing_and_writes_nothing(env):
    st, reports = env
    assert go("--dry-run") == 0
    assert st["docker_calls"] == 0 and st["keep_awake"] == 0 and st["opened"] == []
    assert st["claims_args"].dry_run and st["sweep_args"].dry_run and st["scorecard_args"].dry_run
    assert not reports.exists(), "not even the reports folder"


def test_discover_only_does_not_wait_for_the_vpn_to_clear(env, monkeypatch):
    st, _ = env
    st["vpn_off"] = False          # discovery makes no LLM calls: the VPN does not matter
    go("--discover-only")
    assert st["order"] == ["claims", "sweep", "scorecard"]


def test_keep_awake_is_engaged_for_a_real_run(env):
    st, _ = env
    go()
    assert st["keep_awake"] == 1


def test_everything_printed_is_also_in_the_log_file(env):
    _, reports = env
    go()
    log = next(reports.glob("offvpn_run_*.log")).read_text()
    assert "claims says hello" in log


def test_a_crash_in_the_sweep_is_recorded_not_lost_and_stdout_is_restored(env):
    st, reports = env
    st["sweep_exc"] = RuntimeError("provider exploded")
    before = sys.stdout
    assert go() == 1
    index = (reports / RUN.INDEX).read_text()
    assert "provider exploded" in index and "claims check:** finished" in index, "earlier phase results survive"
    assert sys.stdout is before, "the tee is unwound"


def test_ctrl_c_leaves_a_resumable_note_and_exit_130(env):
    st, reports = env
    st["claims_exc"] = KeyboardInterrupt()
    assert go() == 130
    assert "re-run to continue" in (reports / RUN.INDEX).read_text()


def test_the_exit_code_is_the_worst_of_the_phases(env, monkeypatch):
    async def failing(a):
        return 7
    monkeypatch.setattr(G, "amain", failing)
    assert go() == 7


def test_the_index_handles_an_empty_sweep_and_caps_the_useful_list():
    empty = RUN.render_index("t", {"sweep": "x"}, {"sweep report": "/r.md", "claims report": ""}, [])
    assert "Sweep headline" not in empty and "/r.md" in empty and "claims report" not in empty
    rows = [{"provider": "p", "model": f"m{i}", "route": "direct", "verdict": "USEFUL"} for i in range(40)]
    index = RUN.render_index("t", {}, {}, rows)
    assert "**40 model(s) passed" in index and index.count("- p / ") == 30 and "- ..." in index


def test_the_script_declares_no_flag_that_is_needed():
    ns = RUN.build_parser().parse_args([])
    assert isinstance(ns, argparse.Namespace) and ns.skip_claims is False and ns.no_open is False


def test_no_scorecard_skips_only_the_scorecard(env):
    st, reports = env
    go("--no-scorecard")
    assert st["order"] == ["claims", "sweep"]
    index = (reports / RUN.INDEX).read_text()
    assert "scorecard:** finished" not in index and "Scorecard headline" not in index


def test_a_dry_run_with_no_scorecard_does_not_even_print_the_scorecard_plan(env):
    st, _ = env
    go("--dry-run", "--no-scorecard")
    assert st["scorecard_args"] is None


def test_the_scorecard_is_skipped_when_the_vpn_never_cleared_because_there_is_nothing_to_score(env):
    st, _reports = env
    st["sweep_code"] = 2
    assert go() == 2
    assert st["order"] == ["claims", "sweep"]


def test_the_scorecard_shares_the_fresh_flag_so_one_switch_restarts_everything(env):
    st, _ = env
    go("--fresh")
    assert st["scorecard_args"].fresh is True and st["scorecard_args"].resume is None
    st["order"].clear()
    go()
    assert st["scorecard_args"].fresh is False and st["scorecard_args"].resume == "latest"


def test_a_scorecard_crash_is_recorded_and_the_earlier_phases_survive(env):
    st, reports = env
    st["scorecard_exc"] = RuntimeError("grader exploded")
    assert go() == 1
    index = (reports / RUN.INDEX).read_text()
    assert "grader exploded" in index and "sweep:** finished" in index and "claims check:** finished" in index


def test_the_exit_code_includes_the_scorecard(env):
    st, _ = env
    st["scorecard_code"] = 9
    assert go() == 9


def test_the_headline_helper_returns_just_the_headline_or_nothing(env):
    _, reports = env
    assert RUN.scorecard_headline() == ""
    reports.mkdir(exist_ok=True)
    (reports / MS.LATEST_MD).write_text("# T\n\n## Headline\n\nONLY THIS\n\n## Ranked\n\nnot this\n")
    assert RUN.scorecard_headline() == "ONLY THIS"
    (reports / MS.LATEST_MD).write_text("# no headline section here\n")
    assert RUN.scorecard_headline() == ""
