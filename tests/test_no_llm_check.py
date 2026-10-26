"""The no-LLM check: rules, scoring and plumbing, with no network."""
from __future__ import annotations

import asyncio
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import no_llm_check_offvpn as NC
import scorecard_core as SC
from price_predictor.prediction.schema import PredictionDirection as D
from price_predictor.prediction.schema import PredictionHorizon as H

TOL = {H.DAILY: 0.01, H.WEEKLY: 0.02}


def tolerance(h):
    return TOL[h]


def tv(trend="neutral", momentum="neutral", volatility="neutral", levels="neutral", close=100.0):
    sig = lambda s: SimpleNamespace(signal=s)           # noqa: E731
    return SimpleNamespace(trend=sig(trend), momentum=sig(momentum), volatility=sig(volatility), levels=sig(levels),
                           close_price=close, bars_used=400)


def outcome(case, horizon, ret, *, ungradable=False):
    """What a bullish/bearish/neutral call would have scored, by the grader's own definitions."""
    right = {D.BULLISH: ret > 0, D.BEARISH: ret < 0, D.NEUTRAL: abs(ret) <= TOL[horizon]}
    return NC.Outcome(case, horizon, ret, {d: None for d in D} if ungradable else right)


# ── the rules ──────────────────────────────────────────────────────────

def test_majority_goes_with_the_larger_side_and_a_tie_gets_the_tie_breaker():
    assert NC.majority(tv("bullish", "bullish", "neutral", "bearish")) is D.BULLISH
    assert NC.majority(tv("bearish", "bullish", "bearish", "neutral")) is D.BEARISH
    assert NC.majority(tv("bullish", "bearish")) is D.NEUTRAL
    assert NC.majority(tv("bullish", "bearish"), tie=D.BULLISH) is D.BULLISH
    assert NC.majority(tv()) is D.NEUTRAL


def test_a_single_cluster_rule_follows_that_cluster_only():
    assert NC.cluster("momentum")(tv(trend="bullish", momentum="bearish")) is D.BEARISH
    assert NC.cluster("trend")(tv(trend="neutral", momentum="bullish")) is D.NEUTRAL


def test_the_tie_follows_trend_variant_picks_a_side_only_when_trend_has_one():
    rule = NC.RULES["majority of the 4 clusters (tie follows trend)"]
    assert rule(tv("bullish", "bearish")) is D.BULLISH
    assert rule(tv("neutral", "bullish", "bearish")) is D.NEUTRAL


def test_the_inverse_flips_a_side_and_leaves_neutral_alone():
    inverse = NC.RULES["INVERSE of the majority (diagnostic only)"]
    assert inverse(tv("bullish", "bullish")) is D.BEARISH
    assert inverse(tv("bearish", "bearish")) is D.BULLISH
    assert inverse(tv()) is D.NEUTRAL


def test_every_rule_returns_a_direction_for_every_view():
    for name, rule in NC.RULES.items():
        assert rule(tv("bullish", "bearish", "neutral", "bullish")) in set(D), name


# ── scoring ────────────────────────────────────────────────────────────

def test_outcomes_fold_one_graded_call_per_direction_into_one_outcome():
    def graded(direction, correct):
        pred = SimpleNamespace(ticker="TCS.NS", as_of=SimpleNamespace(date=lambda: date(2026, 8, 3)), horizon=H.WEEKLY, direction=direction)
        return SimpleNamespace(prediction=pred, realized_return=0.04, direction_correct=correct)
    out = NC.outcomes_from_graded([graded(D.BULLISH, True), graded(D.BEARISH, False), graded(D.NEUTRAL, False)])
    assert len(out) == 1 and out[0].case == "TCS.NS@2026-08-03" and out[0].horizon is H.WEEKLY
    assert out[0].right == {D.BULLISH: True, D.BEARISH: False, D.NEUTRAL: False} and out[0].ret == 0.04


def test_the_baseline_counts_only_gradable_real_moves():
    outs = [outcome("a", H.DAILY, 0.03), outcome("b", H.DAILY, 0.04), outcome("c", H.DAILY, -0.02),
            outcome("d", H.DAILY, 0.001),                       # calm: ignored
            outcome("e", H.DAILY, 0.05, ungradable=True)]       # no prices yet: ignored
    baseline, n = NC.baseline_of(outs, tolerance)
    assert n == 3 and baseline == pytest.approx(2 / 3)


def test_a_rule_is_scored_on_real_moves_and_calm_days_separately():
    outs = [outcome("a", H.DAILY, 0.03), outcome("b", H.DAILY, -0.03), outcome("c", H.DAILY, 0.001)]
    picks = {("a", H.DAILY): D.BULLISH, ("b", H.DAILY): D.BULLISH, ("c", H.DAILY): D.NEUTRAL}
    row = NC.score_rule("r", picks, outs, tolerance, baseline=0.5, baseline_n=2)
    assert (row.k_moves, row.n_moves, row.k_calm, row.n_calm) == (1, 2, 1, 1)
    assert row.neutral_share == pytest.approx(1 / 3)


def test_a_neutral_call_on_a_real_move_is_a_miss():
    outs = [outcome("a", H.DAILY, 0.05)]
    row = NC.score_rule("r", {("a", H.DAILY): D.NEUTRAL}, outs, tolerance, 0.5, 1)
    assert (row.k_moves, row.n_moves) == (0, 1)


def test_calls_that_could_not_be_graded_or_were_never_made_are_skipped():
    outs = [outcome("a", H.DAILY, 0.05, ungradable=True), outcome("b", H.DAILY, 0.05)]
    row = NC.score_rule("r", {("a", H.DAILY): D.BULLISH}, outs, tolerance, 0.5, 1)      # no pick for b
    assert row.n_moves == 0 and row.n_calm == 0


def test_the_verdict_uses_the_same_statistics_as_the_models():
    outs = [outcome(f"c{i}", H.DAILY, 0.03 if i % 10 else -0.03) for i in range(100)]    # 90% up
    always_up = NC.score_rule("up", {(f"c{i}", H.DAILY): D.BULLISH for i in range(100)}, outs, tolerance, 0.9, 100)
    always_down = NC.score_rule("down", {(f"c{i}", H.DAILY): D.BEARISH for i in range(100)}, outs, tolerance, 0.9, 100)
    assert always_up.k_moves == 90 and always_down.k_moves == 10
    assert always_down.verdict == "WORSE_THAN_BASELINE"


# ── plumbing ───────────────────────────────────────────────────────────

def fake_grader(returns):
    """Grades like the app: bullish right if up, bearish if down, neutral if within the tolerance."""
    def grade(preds):
        out = []
        for p in preds:
            ret = returns[(p.ticker, p.as_of.date(), p.horizon)]
            right = {D.BULLISH: ret > 0, D.BEARISH: ret < 0, D.NEUTRAL: abs(ret) <= TOL[p.horizon]}[p.direction]
            out.append(SimpleNamespace(prediction=p, realized_return=ret, direction_correct=right))
        return out
    return grade


def test_the_placeholder_predictions_are_valid_for_every_direction_and_horizon():
    case = SC.Case("TCS.NS", date(2026, 8, 3))
    for h in H:
        for d in D:
            p = NC.placeholder("TCS.NS", case, h, d, tv(close=3500.0))
            assert p.direction is d and p.horizon is h and p.as_of.tzinfo is not None


def test_run_check_grades_three_calls_per_case_and_horizon_and_scores_every_rule():
    cases = [SC.Case("A.NS", date(2026, 8, 3)), SC.Case("B.NS", date(2026, 8, 4))]
    views = {"A.NS@2026-08-03": tv("bullish", "bullish"), "B.NS@2026-08-04": tv("bearish", "bearish")}
    returns = {("A.NS", date(2026, 8, 3), H.DAILY): 0.03, ("B.NS", date(2026, 8, 4), H.DAILY): -0.03}
    graded_calls = []
    grader = fake_grader(returns)
    rows, baseline, n, outs = NC.run_check(cases, views, [H.DAILY], lambda t: t, lambda ps: graded_calls.append(len(ps)) or grader(ps), tolerance)
    assert graded_calls == [6], "2 cases x 1 horizon x 3 directions"
    by = {r.name: r for r in rows}
    assert (by["majority of the 4 clusters (tie = neutral)"].k_moves, by["majority of the 4 clusters (tie = neutral)"].n_moves) == (2, 2)
    assert by["INVERSE of the majority (diagnostic only)"].k_moves == 0
    assert by["always bullish"].k_moves == 1 and by["always neutral"].k_moves == 0
    assert n == 2 and baseline == 0.5 and len(outs) == 2


def test_cases_without_technicals_are_left_out_not_guessed():
    cases = [SC.Case("A.NS", date(2026, 8, 3)), SC.Case("B.NS", date(2026, 8, 4))]
    returns = {("A.NS", date(2026, 8, 3), H.DAILY): 0.03}
    rows, _b, _n, outs = NC.run_check(cases, {"A.NS@2026-08-03": tv()}, [H.DAILY], lambda t: t, fake_grader(returns), tolerance)
    assert len(outs) == 1 and all(r.n_moves + r.n_calm <= 1 for r in rows)


def test_a_failed_download_is_reported_and_does_not_stop_the_others():
    async def compose(ticker, as_of):
        if ticker == "BAD.NS":
            raise RuntimeError("yahoo said no")
        return tv()
    cases = [SC.Case("OK.NS", date(2026, 8, 3)), SC.Case("BAD.NS", date(2026, 8, 3))]
    views, failed = asyncio.run(NC.collect_views(cases, compose, lambda t: t, say=lambda *_: None))
    assert list(views) == ["OK.NS@2026-08-03"] and len(failed) == 1 and "yahoo said no" in failed[0]


def test_restricting_to_a_model_keeps_only_the_cases_it_answered(tmp_path):
    store = SC.CaseStore(tmp_path / "s.jsonl")
    store.add(SC.CaseResult("direct|mistral|codestral-latest", "A.NS@2026-08-03", "OK"))
    store.add(SC.CaseResult("direct|mistral|codestral-latest", "B.NS@2026-08-04", "FAIL", "rejected"))
    store.add(SC.CaseResult("direct|google|gemini-x", "C.NS@2026-08-05", "OK"))
    assert NC.answered_cases(store, "codestral-latest") == {"A.NS@2026-08-03"}


def test_the_report_shows_every_rule_the_baseline_and_the_diagnostic_warning():
    rows = [NC.Row("always bullish", "TOO_FEW", 0, 0, 0.0, 1.0, 0, 0, 0.0), NC.Row("x", "KEEP_TESTING", 5, 10, 0.2, 0.8, 3, 4, 0.1)]
    text = NC.render(rows, 0.6, 100, {"run": "t"})
    assert "always bullish" in text and "50% (5/10)" in text and "60% on 100 real-move outcomes" in text
    assert "diagnostic" in text and "n/a" in text
