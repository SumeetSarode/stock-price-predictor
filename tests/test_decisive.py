"""The decisive test's driver: the order, the skips, the stages, who leaves when, and the verdict, all offline."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import decisive_scorecard_offvpn as DC
import model_scorecard_offvpn as MS
import scorecard_core as SC
import scorecard_stats as ST

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorecard_fakes import cand


def prior(name, provider="google", sec=20.0, completion=0.9, attempted=30, ok=27, rl=0, k=7, n=13, baseline=0.6):
    return DC.Prior(name, provider, sec, completion, attempted, ok, rl, k, n, baseline)


def args(**kw):
    base = dict(cases=150, mover_share=0.73, tickers="nifty50", horizons="daily,weekly", window_days=70, lag_days=14,
                from_sweep="x.jsonl", include_skipped=False, probe=False, dry_run=False, probe_cases=DC.PROBE_CASES, fresh=False)
    return SimpleNamespace(**{**base, **kw})


# ── the order and the skips ─────────────────────────────────────────────

def test_models_are_ordered_by_what_a_valid_answer_costs_not_by_raw_speed():
    rows = DC.plan_models([prior("slow-reliable", sec=30, completion=0.95), prior("fast-flaky", sec=10, completion=0.2),
                           prior("middle", sec=20, completion=0.9)])
    assert [r.model for r in rows] == ["middle", "slow-reliable", "fast-flaky"], "10s at 20% completion costs 50s per answer"


def test_a_model_already_hopeless_on_real_moves_is_skipped_with_the_evidence():
    rows = DC.plan_models([prior("good"), prior("bad", k=0, n=7)])
    bad = next(r for r in rows if r.model == "bad")
    assert bad.action == DC.SKIP and "0/7" in bad.reason and "below the 60% baseline" in bad.reason
    assert next(r for r in rows if r.model == "good").action == DC.RUN


def test_a_model_that_is_merely_unproven_is_not_skipped():
    rows = DC.plan_models([prior("maybe", k=2, n=6)])           # 2/6: its best case is still above the baseline
    assert rows[0].action == DC.RUN


def test_a_model_with_too_little_evidence_is_never_ruled_out():
    assert DC.plan_models([prior("tiny", k=0, n=DC.PRIOR_MIN_JUDGED - 1)])[0].action == DC.RUN


def test_a_quota_bound_model_is_skipped_because_a_free_key_cannot_serve_it():
    rows = DC.plan_models([prior("capped", attempted=30, ok=3, rl=26, k=1, n=3)])
    assert rows[0].action == DC.SKIP and "quota-bound" in rows[0].reason and "3 of 30" in rows[0].reason


def test_skipped_models_can_be_forced_back_in():
    rows = DC.plan_models([prior("capped", attempted=30, ok=3, rl=26), prior("bad", k=0, n=7)], include_skipped=True)
    assert [r.action for r in rows] == [DC.RUN, DC.RUN]


def test_models_nobody_has_timed_still_get_a_place_in_the_order_they_came():
    rows = DC.plan_models([DC.Prior("a", "google"), DC.Prior("b", "mistral"), DC.Prior("c", "google")])
    assert [r.model for r in rows] == ["a", "b", "c"] and all(r.action == DC.RUN for r in rows)


# ── stages and time ─────────────────────────────────────────────────────

def test_stages_are_a_screen_a_middle_look_and_the_whole_grid():
    assert DC.stage_cuts(150) == [40, 100, 150]
    assert DC.stage_cuts(150, floor=60) == [60, 100, 150], "the screen is never smaller than the probe"
    assert DC.stage_cuts(3, floor=2) == [2, 3] and DC.stage_cuts(3) == [3], "never a duplicate, never past the grid"


def test_a_case_costs_the_slower_of_the_news_download_and_the_models_because_they_overlap():
    rows = DC.plan_models([prior("a", sec=10), prior("b", sec=20), prior("c", sec=99, k=0, n=7)])     # c is skipped
    assert DC.eta_minutes(rows, 100, overhead_s=90) == pytest.approx(100 * 90 / 60), "news-bound"
    assert DC.eta_minutes(rows, 100, overhead_s=10) == pytest.approx(100 * 30 / 60), "model-bound: only a and b count"


def test_the_plan_names_every_model_the_sweep_found_including_the_partial_ones():
    rows = DC.plan_models([prior("good"), prior("bad", k=0, n=7)])
    text = DC.format_plan(rows, args(), DC.stage_cuts(150), ["mistral  codestral-2508  passed only part of the runs"])
    for needle in ("good", "bad", "already hopeless", "codestral-2508", "110 real moves, 40 calm", "40 -> 100 -> 150"):
        assert needle in text, needle


# ── connections ─────────────────────────────────────────────────────────

def hosts(**state):
    def reachable(host):
        ok = state.get(host, True)
        return ok, "ok" if ok else "DNS blocked"
    return reachable


def test_yahoo_unreachable_means_the_vpn_is_on_and_it_is_retried_before_giving_up():
    slept = []
    problems, _down = DC.check_hosts({"google"}, say=lambda *_: None, reachable=hosts(**{DC.S.REQUIRED_HOST: False}),
                                     sleep=slept.append)
    assert len(slept) == 2, "tried three times"
    assert "VPN looks ON" in problems[0]


def test_news_unreachable_stops_everything_but_a_provider_down_only_removes_its_models():
    problems, down = DC.check_hosts({"google", "mistral"}, say=lambda *_: None, sleep=lambda s: None,
                                    reachable=hosts(**{"api.gdeltproject.org": False, "api.mistral.ai": False}))
    assert len(problems) == 1 and "gdelt" in problems[0]
    assert down == {"mistral"}


# ── the driver, with every outside thing faked ──────────────────────────

class Harness:
    """Fake amain/score/hosts that record what the driver asked for, and answer per a script."""

    def __init__(self, tmp_path, monkeypatch, priors, results):
        monkeypatch.setattr(MS, "REPORTS", tmp_path)
        self.said, self.amain_calls, self.priors, self.results, self.stage = [], [], priors, results, -1
        self.no_inputs, self.code, self.cumulative, self.fresh_calls = [], 0, {}, []

    async def amain(self, a):
        self.amain_calls.append((a.only.split(",") if a.only else [], a.first_cases))
        self.fresh_calls.append(a.fresh)
        if a.first_cases > DC.PROBE_CASES:
            self.stage += 1
        return self.code

    def score(self, a, names):
        self.cumulative.update(self.results[max(self.stage, 0)])      # saved results persist: every planned model is reported
        res = self.cumulative
        scores = [SimpleNamespace(cand=cand(m, "google"), k_moves=res[m][0], n_moves=res[m][1], k_calm=5, n_calm=10, ok=40,
                                  attempted=45, avg_seconds=20.0, fail_classes={}) for m in names if m in res]
        return DC.Snapshot(scores, 0.6, 100, self.no_inputs)

    def run(self, **kw):
        return asyncio.run(DC.run(args(**kw), amain=self.amain, priors_fn=lambda a: self.priors, score_fn=self.score,
                                  hosts_fn=kw.pop("hosts", None) or (lambda providers, say: ([], set())),
                                  partial_fn=lambda a: [], say=self.said.append))


def models(*names):
    return [prior(n, sec=10 + i) for i, n in enumerate(names)]


def test_a_dry_run_shows_the_plan_and_touches_nothing(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, models("a", "b"), [{}])
    assert h.run(dry_run=True) == 0
    assert h.amain_calls == [] and "DECISIVE TEST PLAN" in h.said[0]


def test_blocked_network_stops_before_any_model_is_called(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, models("a"), [{}])
    code = asyncio.run(DC.run(args(), amain=h.amain, priors_fn=lambda a: h.priors, score_fn=h.score,
                              hosts_fn=lambda providers, say: (["query1.finance.yahoo.com unreachable; the VPN looks ON"], set()),
                              partial_fn=lambda a: [], say=h.said.append))
    assert code == 2 and h.amain_calls == [] and any("STOPPED before spending anything" in s for s in h.said)


def test_a_provider_that_is_down_loses_its_models_for_this_run_only(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, [prior("g", "google"), prior("m", "mistral")], [{"g": (5, 10)}])
    asyncio.run(DC.run(args(probe=True), amain=h.amain, priors_fn=lambda a: h.priors, score_fn=h.score,
                       hosts_fn=lambda providers, say: ([], {"mistral"}), partial_fn=lambda a: [], say=h.said.append))
    assert h.amain_calls == [(["g"], DC.PROBE_CASES)]
    assert any("not reachable now" in s and "m" in s for s in h.said)


def test_the_probe_runs_two_cases_per_model_and_stops(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, models("a", "b"), [{"a": (5, 10), "b": (5, 10)}])
    assert h.run(probe=True) == 0
    assert h.amain_calls == [(["a", "b"], DC.PROBE_CASES)], "no stage was started"
    assert any("PROBE OK" in s for s in h.said)
    assert not list(tmp_path.glob("decisive_verdict_*")), "a probe writes no verdict"


def test_a_probe_whose_cases_never_got_their_inputs_fails_loudly(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, models("a"), [{"a": (5, 10)}])
    h.no_inputs = ["TCS.NS@2026-08-04"]
    assert h.run(probe=True) == 3
    assert any("PROBE FAILED" in s for s in h.said) and any("never arrived" in s for s in h.said)


def test_a_full_run_that_fails_its_probe_spends_nothing_more(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, models("a"), [{"a": (5, 10)}])
    h.no_inputs = ["TCS.NS@2026-08-04"]
    assert h.run() == 3
    assert h.amain_calls == [(["a"], DC.PROBE_CASES)]


def test_hopeless_models_leave_proven_models_stop_and_the_rest_go_on_to_the_end(tmp_path, monkeypatch):
    results = [{"a": (0, 14), "b": (14, 15), "c": (7, 13)},            # stage 1: a hopeless, b proven, c unclear
               {"c": (50, 100)},                                       # stage 2: c still unclear at 99%
               {"c": (50, 100)}]                                       # final: c has no edge
    h = Harness(tmp_path, monkeypatch, models("a", "b", "c"), results)
    assert h.run() == 0
    assert h.amain_calls == [(["a", "b", "c"], DC.PROBE_CASES), (["a", "b", "c"], 40), (["c"], 100), (["c"], 150)]
    report = (tmp_path / "decisive_verdict_latest.md").read_text()
    assert "| google/a | WORSE_THAN_BASELINE" in report and "| google/b | BEATS_BASELINE" in report
    assert "| google/c | NO_EDGE" in report
    assert "b beat" in report and "first in CHAIN_AGENTIC" in report


def test_when_every_model_is_decided_early_the_remaining_stages_are_not_run(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, models("a", "b"), [{"a": (0, 14), "b": (14, 15)}])
    assert h.run() == 0
    assert h.amain_calls == [(["a", "b"], DC.PROBE_CASES), (["a", "b"], 40)], "nothing left to test after stage 1"
    assert list(tmp_path.glob("decisive_verdict_2*.md"))


def test_an_interrupted_stage_stops_at_once_and_writes_no_verdict(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, models("a"), [{"a": (5, 10)}])
    h.code = 130
    assert h.run() == 130
    assert not list(tmp_path.glob("decisive_verdict_*"))


# ── the recommendation ──────────────────────────────────────────────────

def snap(**acc):
    scores = [SimpleNamespace(cand=cand(m), k_moves=k, n_moves=n) for m, (k, n) in acc.items()]
    return DC.Snapshot(scores, 0.6, 100, [])


def test_the_best_proven_model_goes_first_by_its_worst_plausible_accuracy():
    s = snap(a=(80, 100), b=(90, 100))
    text = DC.recommend({"a": ST.BEATS, "b": ST.BEATS}, s)
    assert text.index("b, a") >= 0 and "Put b first" in text


def test_no_proven_model_and_enough_data_says_the_pipeline_is_the_thing_to_work_on():
    text = DC.recommend({"a": ST.NO_EDGE, "b": ST.WORSE}, snap(a=(50, 100), b=(30, 100)))
    assert "No model shows skill" in text and "pipeline" in text and "60%" in text


def test_an_undecided_model_says_so_instead_of_guessing():
    text = DC.recommend({"a": ST.NO_EDGE, "b": ST.CANT_TELL}, snap(a=(50, 100), b=(30, 50)))
    assert "Not decided yet: b" in text and "resumes" in text


# ── where the earlier evidence comes from ───────────────────────────────

def old_file(path: Path, picker, rows=True):
    lines = [json.dumps({"type": "cases", "cases": ["TCS.NS@2026-07-20"], **({"picker": picker} if picker else {})})]
    if rows:
        lines.append(json.dumps({"model_uid": "direct|groq|a", "case": "TCS.NS@2026-07-20", "status": "FAIL", "klass": "other"}))
    path.write_text("\n".join(lines) + "\n")


def test_earlier_evidence_is_the_newest_scorecard_of_a_DIFFERENT_grid_so_the_plan_never_drifts(tmp_path, monkeypatch):
    old_file(tmp_path / "model_scorecard_20261019T000000Z.jsonl", "movers-v1")
    old_file(tmp_path / "model_scorecard_20261020T000000Z.jsonl", "movers-v1")
    old_file(tmp_path / "model_scorecard_20261026T000000Z.jsonl", SC.PICKER)          # this test's own grid: ignored
    monkeypatch.setattr(MS, "REPORTS", tmp_path)
    monkeypatch.setattr(MS, "shortlist_for", lambda a: ([cand("a")], []))
    seen = []

    def aggregate(models, store):
        seen.append((store.picker, [r.case for r in store.rows.values()]))
        return [], 0.6, 0
    monkeypatch.setattr(SC, "aggregate", aggregate)
    DC.prior_evidence(args())
    assert seen == [("movers-v1", ["TCS.NS@2026-07-20"])], "the v2 grid was skipped, the newest older one was read"


def test_with_no_earlier_evidence_every_shortlisted_model_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(MS, "REPORTS", tmp_path)
    monkeypatch.setattr(MS, "shortlist_for", lambda a: ([cand("a"), cand("b")], []))
    priors = DC.prior_evidence(args())
    assert [p.name for p in priors] == ["a", "b"] and all(not DC.skip_reason(p) for p in priors)


def test_models_that_only_partly_passed_the_sweep_are_listed_so_the_plan_covers_all_twelve(tmp_path, monkeypatch):
    monkeypatch.setattr(MS, "REPORTS", tmp_path)
    (tmp_path / "x.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"verdict": "USEFUL", "provider": "g", "model": "m1"}, {"verdict": "PARTIAL", "provider": "mistral", "model": "codestral-2508"}]))
    out = DC.partial_models(args(from_sweep="x.jsonl"))
    assert len(out) == 1 and "codestral-2508" in out[0]
    assert DC.partial_models(args(from_sweep="missing.jsonl")) == []


# ── the stage rules: stricter while looking repeatedly, the usual interval at the end ──

B = 0.6


def test_an_interim_look_needs_the_99_percent_interval_to_call_a_model_proven():
    assert ST.stage_verdict(37, 50, B, final=False) == ST.KEEP, "74%: proven at 95%, not at 99%: keep testing, do not crown it"
    assert ST.stage_verdict(37, 50, B, final=True) == ST.BEATS


def test_an_interim_look_needs_the_99_percent_interval_to_drop_a_model_as_hopeless():
    assert ST.stage_verdict(12, 30, B, final=False) == ST.KEEP, "40%: hopeless at 95%, not yet at 99%: do not drop it early"
    assert ST.stage_verdict(12, 30, B, final=True) == ST.WORSE


def test_a_hopeless_model_needs_the_baseline_to_be_uncertain_too():
    assert ST.stage_verdict(12, 30, B, baseline_n=0, final=True) == ST.WORSE
    assert ST.stage_verdict(12, 30, B, baseline_n=10, final=True) != ST.WORSE, "a baseline from 10 outcomes may really be lower"


def test_too_little_data_is_never_a_verdict():
    assert ST.stage_verdict(3, ST.MIN_JUDGED - 1, B, final=False) == ST.KEEP
    assert ST.stage_verdict(3, ST.MIN_JUDGED - 1, B, final=True) == ST.TOO_FEW


def test_the_last_stage_says_no_edge_only_with_enough_data_otherwise_cant_tell():
    enough = ST.judged_needed()
    assert ST.stage_verdict(round(0.55 * enough), enough, B, final=True) == ST.NO_EDGE
    assert ST.stage_verdict(round(0.55 * (enough - 20)), enough - 20, B, final=True) == ST.CANT_TELL


# ── the probe's findings ────────────────────────────────────────────────

def probe_snap(no_inputs=(), **models):
    scores = [SimpleNamespace(cand=cand(m, "google"), ok=ok, attempted=att, fail_classes=fc, avg_seconds=sec)
              for m, (ok, att, fc, sec) in models.items()]
    return DC.Snapshot(scores, 0.6, 100, list(no_inputs))


def test_a_probe_projects_how_many_graded_answers_each_model_will_yield():
    snap = probe_snap(good=(5, 6, {"overloaded": 1}, 20.0), refused=(2, 8, {"rejected": 6}, 30.0))
    ok, text = DC.probe_findings(snap, args(probe_cases=6), {})
    assert ok
    assert "projects ~110 of 97 needed" in text, "good: 5 answered, none refused: completion 100% of 110 real-move cases"
    assert "projects ~27 of 97 needed" in text, "refused: 2 answered, 6 refused: 25% of the 109.5 real-move cases"
    assert "world-failures 1" in text and "refused 6" in text


def test_a_probe_does_not_fail_because_a_model_is_often_refused_only_because_the_set_up_is_broken():
    assert DC.probe_findings(probe_snap(m=(1, 6, {"rejected": 5}, 20.0)), args(), {})[0] is True
    ok, text = DC.probe_findings(probe_snap(no_inputs=["TCS.NS@2026-08-04"], m=(3, 6, {}, 20.0)), args(), {})
    assert not ok and "never arrived" in text
    ok, text = DC.probe_findings(probe_snap(m=(0, 6, {"rejected": 6}, None)), args(), {})
    assert not ok and "no model produced an answer" in text


def test_the_probe_pace_uses_the_models_measured_seconds_and_falls_back_to_the_plans():
    snap = probe_snap(measured=(3, 6, {}, 50.0), unmeasured=(0, 6, {"overloaded": 6}, None))
    _ok, text = DC.probe_findings(snap, args(), {"unmeasured": 70.0})
    assert "about 120 s per case" in text, "50 + 70 > the 90 s news download"


def test_overloaded_and_timeouts_are_the_worlds_fault_not_the_models():
    snap = probe_snap(m=(4, 8, {"overloaded": 2, "timeout": 2}, 20.0))
    _ok, text = DC.probe_findings(snap, args(), {})
    assert "refused 0" in text and "world-failures 4" in text and "projects ~110 of 97" in text


def test_fresh_starts_a_new_result_file_once_and_every_later_stage_resumes_it(tmp_path, monkeypatch):
    """After the pipeline changed, old answers came from different code. But stage 2 must continue stage 1's file."""
    h = Harness(tmp_path, monkeypatch, models("a", "b"), [{"a": (3, 5), "b": (3, 5)}] * 4)
    h.run(fresh=True)
    assert h.fresh_calls[0] is True and not any(h.fresh_calls[1:]) and len(h.fresh_calls) > 1


def test_without_fresh_nothing_ever_starts_over(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, models("a"), [{}])
    h.run(probe=True)
    assert h.fresh_calls == [False]
