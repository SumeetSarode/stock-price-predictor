"""The scorecard's statistics, case handling, store, engine, scoring and report, all offline."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import scorecard_core as SC
import scorecard_report as REP
import scorecard_stats as ST
import sweep_probe as PR
from price_predictor.prediction.grading import grade_many

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorecard_fakes import (
    BULL,
    NEUTRAL,
    World,
    bars,
    cand,
    cases3,
    fetch,
    go,
    grader,
    oracle,
    prediction,
    run_models,
)


def test_wilson_matches_known_values_and_never_leaves_zero_to_one():
    low, high = ST.wilson(5, 10)
    assert (round(low, 3), round(high, 3)) == (0.237, 0.763)
    assert ST.wilson(0, 0) == (0.0, 1.0)
    for k, n in [(0, 5), (5, 5), (1, 1), (0, 1), (50, 100)]:
        lo, hi = ST.wilson(k, n)
        assert 0.0 <= lo <= hi <= 1.0
    assert ST.wilson(50, 100)[1] - ST.wilson(50, 100)[0] < ST.wilson(5, 10)[1] - ST.wilson(5, 10)[0], "more data, tighter"

def test_baseline_is_the_best_constant_strategy_not_a_coin_flip():
    t = 0.02    # the grader's neutral tolerance: a neutral call is right if |return| <= t
    assert ST.baseline_accuracy([(0.05, t), (0.03, t), (0.04, t), (-0.05, t)]) == 0.75, "rising: always-bullish scores 75%"
    assert ST.baseline_accuracy([(-0.05, t), (-0.03, t), (0.05, t)]) == pytest.approx(2 / 3), "falling: always-bearish"
    assert ST.baseline_accuracy([(0.001, t), (-0.001, t), (0.05, t)]) == pytest.approx(2 / 3), "calm: always-NEUTRAL wins"
    assert ST.baseline_accuracy([(0.05, t), (-0.05, t)]) == 0.5
    assert ST.baseline_accuracy([]) == 0.5
    assert ST.baseline_accuracy([(0.0, t), (0.0, t)]) == 1.0, "nothing moved: always-neutral was perfect"

def test_the_sample_size_needed_for_ten_points_is_about_a_hundred():
    assert ST.judged_needed() == 97
    assert ST.judged_needed(0.05) > 380

@pytest.mark.parametrize("k, n, baseline, verdict", [
    (9, 9, 0.5, ST.TOO_FEW),                 # perfect, but only 9 graded calls: no interval is drawn
    (12, 12, 0.5, ST.BEATS),
    (8, 12, 0.5, ST.CANT_TELL),              # 67% looks good; the interval still includes 50%
    (0, 12, 0.5, ST.WORSE),
    (12, 12, 0.95, ST.CANT_TELL),            # perfect, but the baseline is 95%: cannot prove it beats that
    (7, 10, 0.55, ST.CANT_TELL),             # 70% > 55% as a NUMBER, but the interval is what counts
])
def test_the_verdict_compares_the_whole_interval_to_the_baseline(k, n, baseline, verdict):
    assert ST.judge(k, n, baseline)[0] == verdict

def test_cases_are_spread_across_the_period_rotate_tickers_and_are_deterministic():
    days = [date(2026, 7, 1) + timedelta(days=i) for i in range(30)]
    cases = SC.pick_cases(["A", "B", "C"], days, 6)
    assert [c.as_of for c in cases] == [days[0], days[6], days[12], days[17], days[23], days[29]]
    assert [c.ticker for c in cases] == ["A", "B", "C", "A", "B", "C"]
    assert cases == SC.pick_cases(["A", "B", "C"], days, 6)
    assert len({c.as_of for c in cases}) == 6, "no date twice"

def test_pick_cases_edge_cases():
    days = [date(2026, 7, 1) + timedelta(days=i) for i in range(3)]
    assert len(SC.pick_cases(["A"], days, 50)) == 3, "cannot ask for more cases than days"
    assert SC.pick_cases(["A"], days, 1) == [SC.Case("A", days[0])]
    assert SC.pick_cases([], days, 3) == SC.pick_cases(["A"], [], 3) == SC.pick_cases(["A"], days, 0) == []

def test_case_key_round_trips():
    c = SC.Case("TCS.NS", date(2026, 7, 20))
    assert SC.Case.parse(c.key) == c and c.key == "TCS.NS@2026-07-20"

def test_interleave_alternates_providers_so_pacing_is_spent_on_someone_elses_model():
    ms = [cand("g1", "groq"), cand("g2", "groq"), cand("n1", "nvidia"), cand("m1", "mistral"), cand("g3", "groq")]
    assert [m.model for m in SC.interleave(ms)] == ["g1", "n1", "m1", "g2", "g3"]

def test_the_store_keeps_cases_and_history_across_three_generations(tmp_path):
    g1 = SC.CaseStore(tmp_path / "1.jsonl")
    g1.set_cases([SC.Case("A", date(2026, 7, 1))])
    g1.add(SC.CaseResult("u1", "A@2026-07-01", "OK", predictions=[prediction().model_dump_json()]))
    g2 = SC.CaseStore(tmp_path / "2.jsonl", (tmp_path / "1.jsonl",))
    g2.add(SC.CaseResult("u2", "A@2026-07-01", "OK"))
    g3 = SC.CaseStore(tmp_path / "3.jsonl", (tmp_path / "2.jsonl",))     # resumes from the NEWEST file only
    assert [c.key for c in g3.cases] == ["A@2026-07-01"], "the case grid is pinned, so every run scores the same cases"
    assert g3.get("u1", "A@2026-07-01") and g3.get("u2", "A@2026-07-01"), "generation-1 results survive"
    assert len(g3.predictions("u1")) == 1 and g3.predictions("u1")[0].ticker == "UP.NS"

@pytest.mark.parametrize("status, klass, retry", [
    ("OK", "", False), ("FAIL", "rate_limited", True), ("FAIL", "timeout", True), ("FAIL", "overloaded", True),
    ("FAIL", "bad_output", False), ("FAIL", "auth", False), ("FAIL", "bad_output,timeout", True),
    ("SKIP", "budget", True), ("FAIL", "inputs", True),
])
def test_only_failures_worth_retrying_are_retried(tmp_path, status, klass, retry):
    s = SC.CaseStore(tmp_path / "s.jsonl")
    s.add(SC.CaseResult("u", "c", status, klass))
    assert s.needs("u", "c") is retry
    assert s.needs("u", "never-seen") is True

def test_every_model_sees_the_same_inputs_and_they_are_gathered_once_per_case_not_per_model(tmp_path):
    w, store = World(), SC.CaseStore(tmp_path / "s.jsonl")
    go([cand("a"), cand("b", "nvidia"), cand("c", "mistral")], cases3(), store, w)
    assert w.shared_calls == [c.key for c in cases3()], "ONE gather per case, however many models"
    seen: dict[str, set] = {}
    for _model, case, tv in w.predict_calls:
        seen.setdefault(case, set()).add(tv)
    assert all(len(tvs) == 1 for tvs in seen.values()), "every model on a case got the identical technicals"
    assert len(w.predict_calls) == 9

def test_cases_are_the_outer_loop_so_a_cutoff_leaves_every_model_equally_far_along(tmp_path):
    w, store = World(), SC.CaseStore(tmp_path / "s.jsonl")
    go([cand("a"), cand("b", "nvidia")], cases3(), store, w)
    assert [c for _, c, _ in w.predict_calls][:2] == [cases3()[0].key] * 2, "both models finish case 1 before case 2"

def test_a_model_that_fails_is_a_result_not_a_crash_and_the_others_carry_on(tmp_path):
    w, store = World(), SC.CaseStore(tmp_path / "s.jsonl")
    w.fail[("a", cases3()[0].key)] = RuntimeError("429 rate limit hit")
    w.fail[("a", cases3()[1].key)] = TimeoutError()
    w.fail[("a", cases3()[2].key)] = ValueError("validation error: field required")
    go([cand("a"), cand("b", "nvidia")], cases3(), store, w)
    assert [store.get(cand("a").uid, c.key).klass for c in cases3()] == ["rate_limited", "timeout", "bad_output"]
    assert all(store.get(cand("b", "nvidia").uid, c.key).status == "OK" for c in cases3())

def test_inputs_that_cannot_be_gathered_blame_nobody_and_cost_no_budget(tmp_path):
    w, store, budget = World(), SC.CaseStore(tmp_path / "s.jsonl"), PR.Budget({}, 999)
    w.bad_inputs.add(cases3()[1].key)
    go([cand("a")], cases3(), store, w, budget)
    assert store.get("*", cases3()[1].key).klass == "inputs"
    assert store.get(cand("a").uid, cases3()[1].key) is None, "the model is not marked as failing"
    assert budget.used[cand("a").uid] == 12, "two cases ran (2 x 6 calls); the failed gather spent nothing"
    assert store.needs(cand("a").uid, cases3()[1].key), "and it is retried next run"

def test_the_call_budget_is_a_hard_ceiling_and_the_rest_are_marked_not_reached(tmp_path):
    w, store = World(), SC.CaseStore(tmp_path / "s.jsonl")
    go([cand("a")], cases3(), store, w, PR.Budget({cand("a").uid: 12}, 0), calls=6)
    assert len(w.predict_calls) == 2
    r = store.get(cand("a").uid, cases3()[2].key)
    assert r.status == "SKIP" and r.klass == "budget" and store.needs(cand("a").uid, cases3()[2].key)

def test_the_gather_is_skipped_entirely_when_nobody_has_budget_left(tmp_path):
    w, store = World(), SC.CaseStore(tmp_path / "s.jsonl")
    go([cand("a")], cases3(), store, w, PR.Budget({cand("a").uid: 0}, 0))
    assert w.shared_calls == [] and w.predict_calls == []

def test_a_provider_is_paced_but_interleaving_hides_the_wait(tmp_path):
    w, store = World(), SC.CaseStore(tmp_path / "s.jsonl")
    go([cand("g1"), cand("g2")], cases3()[:1], store, w, pace=30)         # same provider twice: must wait
    assert w.clock.sleeps == [30]
    w2 = World()
    go([cand("g1"), cand("n1", "nvidia")], cases3()[:1], SC.CaseStore(tmp_path / "t.jsonl"), w2, pace=30)
    assert w2.clock.sleeps == [], "two providers: each is idle while the other runs"

def test_the_deadline_stops_new_work(tmp_path):
    w, store = World(), SC.CaseStore(tmp_path / "s.jsonl")
    go([cand("a")], cases3(), store, w, deadline=-1)
    assert w.predict_calls == [] and w.shared_calls == []

def test_a_rerun_repeats_nothing_finished_but_retries_a_rate_limit(tmp_path):
    w, store = World(), SC.CaseStore(tmp_path / "s.jsonl")
    w.fail[("a", cases3()[0].key)] = RuntimeError("429 rate limit")
    go([cand("a")], cases3(), store, w)
    assert len(w.predict_calls) == 3
    del w.fail[("a", cases3()[0].key)]
    go([cand("a")], cases3(), store, w)
    assert [c for _, c, _ in w.predict_calls[3:]] == [cases3()[0].key], "only the rate-limited case was retried"

def test_the_scorecard_separates_a_model_that_knows_from_one_that_guesses_from_one_that_is_backwards(tmp_path):
    models = [cand("oracle"), cand("anti", "nvidia"), cand("always-bull", "mistral")]
    _, (scores, baseline, n_outcomes) = run_models(tmp_path, models, oracle)
    by = {s.cand.model: s for s in scores}
    assert baseline == 0.5 and n_outcomes == 12, "half the outcomes rose, half fell"
    assert by["oracle"].verdict == ST.BEATS and by["oracle"].accuracy == 1.0 and by["oracle"].low > 0.7
    assert by["anti"].verdict == ST.WORSE and by["anti"].accuracy == 0.0
    assert by["always-bull"].verdict == ST.CANT_TELL and by["always-bull"].accuracy == 0.5
    assert [s.cand.model for s in scores] == ["oracle", "always-bull", "anti"], "ranked best worst-case first"

def test_in_a_rising_market_always_bullish_is_exactly_as_good_as_the_baseline_not_better(tmp_path):
    cases = [SC.Case("UP.NS", date(2026, 7, 1) + timedelta(days=i)) for i in range(12)]
    w, store = World(direction=BULL), SC.CaseStore(tmp_path / "s.jsonl")
    go([cand("always-bull")], cases, store, w)
    (score,), baseline, _ = SC.aggregate([cand("always-bull")], store, grader)
    assert baseline == 1.0 and score.accuracy == 1.0
    assert score.verdict == ST.CANT_TELL, "100% in a sample where 100% was free proves nothing"

def test_few_cases_say_too_few_rather_than_a_verdict(tmp_path):
    _, (scores, _, _) = run_models(tmp_path, [cand("oracle")], oracle, n_cases=6)
    assert scores[0].verdict == ST.TOO_FEW and scores[0].n_judged == 6

def test_neutral_calls_are_graded_by_the_apps_tolerance_so_a_trending_market_punishes_them(tmp_path):
    _, (scores, _, _) = run_models(tmp_path, [cand("shy")], lambda c, case: NEUTRAL)
    s = scores[0]
    assert s.n_judged == 12 and s.accuracy == 0.0 and s.neutral_share == 1.0
    assert s.verdict == ST.WORSE, "prices trended well past the tolerance, so every neutral call was wrong"

def test_in_a_calm_market_hiding_behind_neutral_is_perfect_on_calm_days_but_earns_no_verdict(tmp_path):
    flat = lambda ticker, start, end: bars(0)  # noqa: E731
    cases = [SC.Case("UP.NS", date(2026, 7, 1) + timedelta(days=i)) for i in range(12)]
    w, store = World(direction=NEUTRAL), SC.CaseStore(tmp_path / "s.jsonl")
    go([cand("shy")], cases, store, w)
    (s,), _baseline, n_moves = SC.aggregate([cand("shy")], store, lambda p: grade_many(p, fetch_ohlcv=flat, today=date(2026, 9, 30)))
    assert s.accuracy == 1.0 and s.calm_accuracy == 1.0 and s.n_calm == 12, "free points on calm days, shown for information"
    assert s.n_moves == 0 and n_moves == 0, "nothing moved, so there is nothing to judge"
    assert s.verdict == ST.TOO_FEW, "hiding behind neutral cannot earn a verdict, good or bad"

def test_reliability_counts_failed_cases_against_the_model_but_accuracy_ignores_them(tmp_path):
    cases = [SC.Case("UP.NS", date(2026, 7, 1) + timedelta(days=i)) for i in range(12)]
    w, store = World(direction=BULL), SC.CaseStore(tmp_path / "s.jsonl")
    for c in cases[:4]:
        w.fail[("flaky", c.key)] = ValueError("validation error")
    go([cand("flaky")], cases, store, w)
    (s,), _, _ = SC.aggregate([cand("flaky")], store, grader)
    assert (s.ok, s.attempted, s.reliability) == (8, 12, pytest.approx(2 / 3))
    assert s.fail_classes == {"bad_output": 4} and s.n_judged == 8

def test_a_model_with_no_results_yet_scores_cleanly(tmp_path):
    (s,), baseline, n = SC.aggregate([cand("untried")], SC.CaseStore(tmp_path / "s.jsonl"), grader)
    assert (s.attempted, s.n_judged, s.accuracy, s.reliability, s.verdict) == (0, 0, None, None, ST.TOO_FEW)
    assert baseline == 0.5 and n == 0

def scores_for(tmp_path, direction=oracle):
    models = [cand("oracle", litellm_id="groq/oracle"), cand("anti", "nvidia"), cand("always-bull", "mistral")]
    _, (scores, baseline, n) = run_models(tmp_path, models, direction)
    cases = [SC.Case("UP.NS", date(2026, 7, 1))]
    return scores, baseline, n, cases

def test_the_report_explains_itself_ranks_models_and_gives_a_provisional_chain(tmp_path):
    scores, baseline, n, cases = scores_for(tmp_path)
    md = REP.render(scores, baseline, n, cases, {"outcome": "completed"})
    assert "Baseline: 50%" in md and "95% interval" in md and "CANT_TELL_YET" in md
    assert "1 model(s) beat the baseline:** `groq/oracle`" in md
    assert md.index("groq/oracle") < md.index("mistral/always-bull") < md.index("nvidia/anti")
    chain = md.split("## Suggested chain order")[1].split("## Why cases failed")[0]
    assert "groq/oracle" in chain and "always-bull" in chain and "anti" not in chain, "worse-than-baseline never suggested"
    assert "CHAIN_AGENTIC" in chain, "how to add the model, reusing the sweep's wording"
    assert "always neutral" in md, "the baseline explanation mentions the neutral strategy"
    for caveat in ("Luck", "Look-ahead", "Replay is not live", "Neutral calls.", "real moves", "not profit"):
        assert caveat in md

def test_the_chain_leaves_out_unreliable_undecided_models():
    ok = SimpleNamespace(verdict=ST.CANT_TELL, reliability=0.9, low=0.4)
    flaky = SimpleNamespace(verdict=ST.CANT_TELL, reliability=0.5, low=0.45)
    proven = SimpleNamespace(verdict=ST.BEATS, reliability=0.5, low=0.6)
    assert REP.chain_order([ok, flaky, proven]) == [proven, ok], "proven first even if flaky; undecided must be reliable"

def test_an_empty_scorecard_still_renders_and_says_what_to_do():
    md = REP.render([], 0.5, 0, [], {"outcome": "nothing to score yet"})
    assert "No model has beaten the baseline yet" in md and "_nothing qualifies yet_" in md and "_no failures_" in md


def test_outcomes_the_grader_could_not_measure_do_not_distort_the_baseline(tmp_path):
    """If the price feed is down for a ticker, the grade is INCONCLUSIVE (return 0.0). Counting those as 'nothing
    moved, so always-neutral was right' would corrupt the baseline."""
    def flaky_fetch(ticker, start, end):
        if ticker.startswith("GONE"):
            raise RuntimeError("price feed down")
        return fetch(ticker, start, end)

    cases = [SC.Case(t, date(2026, 7, 1) + timedelta(days=i)) for i, t in enumerate(["UP.NS", "DOWN.NS"] * 6 + ["GONE.NS"] * 6)]
    w, store = World(direction=oracle), SC.CaseStore(tmp_path / "s.jsonl")
    go([cand("oracle")], cases, store, w)
    (s,), baseline, n_outcomes = SC.aggregate([cand("oracle")], store, lambda p: grade_many(p, fetch_ohlcv=flaky_fetch, today=date(2026, 9, 30)))
    assert n_outcomes == 12, "only the 12 measurable outcomes feed the baseline"
    assert baseline == 0.5, "6 up, 6 down: unchanged by the 6 unmeasurable ones"
    assert s.n_pred == 18 and s.n_judged == 12, "unmeasurable predictions are not judged either"


def test_with_equal_accuracy_the_model_with_more_evidence_ranks_higher(tmp_path):
    cases = [SC.Case("UP.NS", date(2026, 7, 1) + timedelta(days=i)) for i in range(30)]
    w, store = World(direction=BULL), SC.CaseStore(tmp_path / "s.jsonl")
    small, big = cand("small", "groq"), cand("big", "nvidia")
    budget = PR.Budget({small.uid: 12 * 6}, 999)        # one model only gets 12 cases; the other gets all 30
    go([small, big], cases, store, w, budget)
    scores, _, _ = SC.aggregate([small, big], store, grader)
    by = {s.cand.model: s for s in scores}
    assert by["small"].accuracy == by["big"].accuracy == 1.0 and by["small"].n_judged == 12 and by["big"].n_judged == 30
    assert by["big"].low > by["small"].low
    assert [s.cand.model for s in scores] == ["big", "small"], "ranked by the worst plausible accuracy, not the headline"


def test_cases_not_reached_for_budget_are_not_counted_against_a_models_reliability(tmp_path):
    s = SC.CaseStore(tmp_path / "s.jsonl")
    for i in range(2):
        s.add(SC.CaseResult("direct|groq|m", f"c{i}", "OK", predictions=[prediction().model_dump_json()]))
    for i in range(2, 5):
        s.add(SC.CaseResult("direct|groq|m", f"c{i}", "SKIP", "budget"))
    (score,), _, _ = SC.aggregate([cand("m")], s, grader)
    assert (score.attempted, score.ok, score.skipped, score.reliability) == (2, 2, 3, 1.0)
