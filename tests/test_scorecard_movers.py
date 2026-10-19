"""The scorecard must judge models on days the price REALLY moved.

A real run showed why: with the app's 2% daily / 4% weekly neutral tolerance, calm days dominate a random sample,
"always neutral" scored 100% on all ten graded outcomes, and no model could ever beat that. These tests pin the fix.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import scorecard_core as SC
import scorecard_stats as ST
import sweep_probe as PR
from price_predictor.prediction.grading import grade_many

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorecard_fakes import (
    BULL,
    NEUTRAL,
    World,
    cand,
    cases3,
    go,
    oracle,
    run_models,
)

D0 = date(2026, 7, 1)
WINDOWS = [(1, 0.02), (5, 0.04)]          # (trading days, neutral tolerance): daily and weekly


def days(n):
    return [D0 + timedelta(days=i) for i in range(n)]


# ── what counts as a real move ──────────────────────────────────────────

def test_a_move_must_go_past_the_tolerance_so_exactly_at_it_is_still_calm():
    assert ST.is_move(0.0401, 0.04) and ST.is_move(-0.0401, 0.04)
    assert not ST.is_move(0.04, 0.04) and not ST.is_move(-0.01, 0.04) and not ST.is_move(0.0, 0.02)


def test_move_strength_is_measured_like_the_grader_and_the_strongest_horizon_counts():
    ds = days(12)
    closes = [(d, 100.0) for d in ds]
    closes[5] = (ds[5], 103.0)                       # a +3% blip on trading day 5, back to 100 on day 6
    out = SC.case_moves("T", closes, ds[:7], WINDOWS)
    assert out[("T", ds[0])] == pytest.approx(0.75), "5 days ahead: +3% against a 4% tolerance (1 day ahead: flat)"
    assert out[("T", ds[4])] == pytest.approx(1.5), "1 day ahead: +3% against a 2% tolerance"
    assert out[("T", ds[5])] == pytest.approx(abs(100 / 103 - 1) / 0.02), "a fall is a real move too: direction does not matter"
    assert out[("T", ds[6])] == 0.0, "flat ahead"


def test_days_whose_horizons_have_not_resolved_or_have_no_price_are_left_out_not_called_calm():
    ds = days(8)
    closes = [(d, 100.0) for d in ds]
    out = SC.case_moves("T", closes, [*ds, D0 + timedelta(days=99)], WINDOWS)
    assert set(out) == {("T", ds[0]), ("T", ds[1]), ("T", ds[2])}, "the last 5 days have no weekly outcome yet; day 99 has no price"


# ── picking the cases ───────────────────────────────────────────────────

def strength(ds, tickers, real):
    return {(t, d): (2.0 if (t, d) in real else 0.3) for d in ds for t in tickers}


def test_half_the_cases_are_real_moves_and_half_calm_in_date_order():
    ds, tickers = days(40), ["A", "B"]
    real = {(t, d) for t in tickers for i, d in enumerate(ds) if i % 3 == 0}
    cases = SC.pick_cases(tickers, ds, 10, strength(ds, tickers, real))
    assert len(cases) == 10 and len({c.key for c in cases}) == 10, "no case twice"
    assert sum((c.ticker, c.as_of) in real for c in cases) == 5
    assert [c.as_of for c in cases] == sorted(c.as_of for c in cases)
    assert cases == SC.pick_cases(tickers, ds, 10, strength(ds, tickers, real)), "deterministic: same cases every run"


def test_when_real_moves_are_rare_calm_days_fill_the_grid_instead_of_leaving_it_short():
    ds, tickers = days(30), ["A"]
    real = {("A", ds[5]), ("A", ds[20])}
    cases = SC.pick_cases(tickers, ds, 10, strength(ds, tickers, real))
    assert len(cases) == 10 and sum((c.ticker, c.as_of) in real for c in cases) == 2, "both movers used, rest calm"


def test_when_most_days_are_real_moves_they_fill_the_grid_but_calm_days_still_get_their_half():
    ds, tickers = days(30), ["A"]
    real = {("A", d) for d in ds[:27]}
    cases = SC.pick_cases(tickers, ds, 10, strength(ds, tickers, real))
    assert sum((c.ticker, c.as_of) in real for c in cases) == 7, "only 3 calm days exist; movers take the remaining slots"


def test_with_no_price_information_it_falls_back_to_an_even_spread():
    ds = days(20)
    plain = SC.pick_cases(["A", "B"], ds, 6)
    assert SC.pick_cases(["A", "B"], ds, 6, {}) == plain == SC.pick_cases(["A", "B"], ds, 6, None)
    assert SC.pick_cases(["A", "B"], ds, 6, {("Z", D0): 5.0}) == plain, "prices for other tickers/days are no information"


def test_nothing_to_pick_from_gives_nothing():
    assert SC.pick_cases([], days(5), 3, {("A", D0): 2.0}) == [] and SC.pick_cases(["A"], days(5), 0, {}) == []


# ── the verdict is about real moves ─────────────────────────────────────

def test_a_model_that_reads_direction_beats_the_baseline_on_real_moves(tmp_path):
    models = [cand("oracle"), cand("anti", "nvidia"), cand("always-bull", "mistral")]
    _, (scores, baseline, n_moves) = run_models(tmp_path, models, oracle, n_cases=30)
    by = {s.cand.model: s for s in scores}
    assert baseline == 0.5 and n_moves == 30, "half rose, half fell: always-bull is a coin flip"
    assert by["oracle"].verdict == ST.BEATS and by["oracle"].k_moves == by["oracle"].n_moves == 30
    assert by["always-bull"].verdict == ST.CANT_TELL and by["always-bull"].move_accuracy == 0.5
    assert by["anti"].verdict == ST.WORSE and by["anti"].k_moves == 0


def test_calm_days_are_reported_but_never_decide_the_verdict(tmp_path):
    flat = lambda ticker, start, end: __import__("scorecard_fakes").bars(0)  # noqa: E731

    cases = [SC.Case("UP.NS", D0 + timedelta(days=i)) for i in range(12)]
    w, store = World(direction=BULL), SC.CaseStore(tmp_path / "s.jsonl")
    go([cand("m")], cases, store, w)
    (s,), _, _ = SC.aggregate([cand("m")], store, lambda p: grade_many(p, fetch_ohlcv=flat, today=date(2026, 9, 30)))
    assert s.n_calm == 12 and s.n_moves == 0 and s.verdict == ST.TOO_FEW


def test_neutral_is_wrong_on_a_real_move_so_a_shy_model_cannot_hide(tmp_path):
    _, (scores, baseline, _) = run_models(tmp_path, [cand("shy")], lambda c, case: NEUTRAL, n_cases=30)
    s = scores[0]
    assert s.k_moves == 0 and s.n_moves == 30 and s.neutral_share == 1.0
    assert s.verdict == ST.WORSE and baseline == 0.5, "the best constant on real moves is bull-or-bear, never neutral"


def test_baseline_on_real_moves_ignores_calm_outcomes_entirely():
    assert ST.baseline_accuracy([(0.05, 0.04), (0.06, 0.04), (-0.05, 0.04)]) == pytest.approx(2 / 3)
    assert ST.baseline_accuracy([(0.05, 0.04), (-0.05, 0.04)]) == 0.5


# ── do not call a model WORSE on a shaky baseline ───────────────────────

def test_a_baseline_measured_on_few_outcomes_is_uncertain_so_worse_needs_clear_daylight():
    # 7 of 10 (interval about 40-89%) against a baseline of 100% measured on only 10 outcomes (really 72-100%)
    assert ST.judge(7, 10, 1.0, baseline_n=10)[0] == ST.CANT_TELL
    assert ST.judge(7, 10, 1.0)[0] == ST.WORSE, "treating the same baseline as exactly known is the overconfident old behavior"


def test_with_a_well_measured_baseline_a_clearly_worse_model_is_still_called_worse():
    assert ST.judge(3, 30, 0.8, baseline_n=200)[0] == ST.WORSE


def test_beats_still_compares_against_the_baseline_itself():
    assert ST.judge(29, 30, 0.5, baseline_n=30)[0] == ST.BEATS
    assert ST.judge(6, 10, 0.5, baseline_n=10)[0] == ST.CANT_TELL


@pytest.mark.parametrize("k,n", [(0, 10), (10, 10), (5, 9), (0, 0)])
def test_judging_never_crashes_at_the_edges(k, n):
    assert ST.judge(k, n, 0.5, baseline_n=0)[0] in {ST.BEATS, ST.WORSE, ST.CANT_TELL, ST.TOO_FEW}
    assert ST.judge(k, n, 0.5, baseline_n=7)[0] in {ST.BEATS, ST.WORSE, ST.CANT_TELL, ST.TOO_FEW}


# ── the budget is per model ─────────────────────────────────────────────

def test_two_models_of_one_provider_each_get_their_own_allowance(tmp_path):
    a, b = cand("a"), cand("b")                       # both groq
    store, w = SC.CaseStore(tmp_path / "s.jsonl"), World()
    go([a, b], cases3(), store, w, PR.Budget({a.uid: 12, b.uid: 12}, 0), calls=6)
    assert sum(1 for m, _, _ in w.predict_calls if m == "a") == 2 and sum(1 for m, _, _ in w.predict_calls if m == "b") == 2
