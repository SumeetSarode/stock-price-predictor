"""The scorecard's live wiring, shortlist, end-to-end flow and contract with the real app, all offline."""
from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import free_model_sweep_offvpn as FM
import model_scorecard_offvpn as MS
import model_survey_offvpn as S
import scorecard_core as SC
import sweep_data as D
from price_predictor.prediction.schema import (
    PredictionHorizon,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorecard_fakes import (
    GROQ_KEY,
    IST,
    RealPlumbing,
    World,
    cand,
    grader,
    oracle,
    prediction,
)


def fake_predictor(events):
    @contextlib.contextmanager
    def replay_context(as_of):
        events.append(("replay_on", as_of))
        yield
        events.append(("replay_off", as_of))

    async def news(canonical):
        events.append(("news", canonical))
        return "impact"

    async def synth(si):
        events.append(("synth", si.horizon))
        return prediction()

    return SimpleNamespace(
        _news_impact_agent=SimpleNamespace(model=None), _synthesizer_agent=SimpleNamespace(model=None), IST=IST,
        _resolve_ticker=lambda t: t.upper(), _ensure_news_snapshot_installed=lambda: events.append(("snapshot",)),
        replay_context=replay_context, run_news_impact_agent=news, synthesize_with_guardrails=synth,
        _finalize_prediction=lambda p, news_degraded: p.model_copy(update={"model_chain": (*p.model_chain, f"degraded={news_degraded}")}),
        _NEWS_BACKTEST_REPLAY_TAG="news_impact:agentic_replay")

def live(events, monkeypatch, horizons=(PredictionHorizon.DAILY, PredictionHorizon.WEEKLY), timeout=5):
    sentinel = object()
    monkeypatch.setattr(S, "build_model", lambda group, model_id: sentinel)
    inputs = []
    lv = MS.Live(fake_predictor(events), list(horizons), timeout, make_input=lambda **kw: inputs.append(kw) or SimpleNamespace(**kw))
    return lv, sentinel, inputs

def test_the_live_adapter_puts_the_model_on_BOTH_agents_and_runs_the_news_step_under_replay(monkeypatch):
    events = []
    lv, sentinel, _inputs = live(events, monkeypatch)
    case = SC.Case("tcs.ns", date(2026, 7, 20))
    preds = asyncio.run(lv.predict(cand("m", litellm_id="groq/m"), case, "TECHNICALS"))
    assert lv.P._news_impact_agent.model is sentinel and lv.P._synthesizer_agent.model is sentinel
    assert events[:3] == [("snapshot",), ("replay_on", date(2026, 7, 20)), ("news", "TCS.NS")]
    assert events.index(("replay_off", date(2026, 7, 20))) < events.index(("synth", "daily")), "replay covers news only"
    assert len(preds) == 2 and all(p.model_chain[-1] == "degraded=False" for p in preds)

def test_the_live_adapter_builds_each_synthesis_input_from_the_shared_inputs_at_the_case_date(monkeypatch):
    events = []
    lv, _, inputs = live(events, monkeypatch)
    asyncio.run(lv.predict(cand("m"), SC.Case("TCS.NS", date(2026, 7, 20)), "TECHNICALS"))
    assert [i["horizon"] for i in inputs] == ["daily", "weekly"]
    first = inputs[0]
    assert first["ticker"] == "TCS.NS" and first["technical_view"] == "TECHNICALS" and first["impact_assessment"] == "impact"
    assert first["as_of"] == datetime(2026, 7, 20, 15, 30, tzinfo=IST), "the prediction is stamped at that day's close"
    assert first["model_chain"] == ("news_impact:agentic_replay",)

def test_a_failing_synthesizer_propagates_so_the_engine_can_classify_it(monkeypatch):
    events = []
    lv, _, _ = live(events, monkeypatch)

    async def boom(si):
        raise RuntimeError("503 high demand")
    lv.P.synthesize_with_guardrails = boom
    with pytest.raises(RuntimeError, match="503"):
        asyncio.run(lv.predict(cand("m"), SC.Case("TCS.NS", date(2026, 7, 20)), "tv"))

def test_a_news_step_that_never_answers_times_out_instead_of_hanging_the_run(monkeypatch):
    events = []
    lv, _, _ = live(events, monkeypatch, timeout=0.05)

    async def slow(canonical):
        await asyncio.sleep(5)
    lv.P.run_news_impact_agent = slow
    with pytest.raises(TimeoutError):
        asyncio.run(lv.predict(cand("m"), SC.Case("TCS.NS", date(2026, 7, 20)), "tv"))

def test_shared_inputs_are_composed_for_the_canonical_ticker_at_the_case_date():
    seen = []

    async def compose(ticker, as_of):
        seen.append((ticker, as_of))
        return "tv"
    lv = MS.Live(fake_predictor([]), [], compose=compose)
    assert asyncio.run(lv.shared(SC.Case("tcs.ns", date(2026, 7, 20)))) == "tv"
    assert seen == [("TCS.NS", date(2026, 7, 20))]

def verdict_rows():
    def row(provider, model, route, verdict, **kw):
        return {**D.Candidate(route, provider, model, base="https://b/v1", **kw).row(), "verdict": verdict, "reason": ""}
    return [row("groq", "g1", "direct", "USEFUL", litellm_id="groq/g1"), row("groq", "g2", "direct", "USEFUL"),
            row("nvidia", "n1", "direct", "USEFUL"), row("kilo", "k1", "freellmapi", "USEFUL"),
            row("ovh", "o1", "anon", "USEFUL", anonymous=True), row("groq", "bad", "direct", "NOT_USEFUL"),
            row("mistral", "p1", "direct", "PARTIAL")]

def test_the_shortlist_takes_only_useful_models_the_app_can_call_and_says_what_it_left_out(tmp_path):
    f = tmp_path / "v.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in verdict_rows()))
    models, left_out = MS.load_shortlist(f, {"groq": "GK", "nvidia": "NK"}, 10)
    assert {m.model for m in models} == {"g1", "g2", "n1", "o1"}, "USEFUL and callable; PARTIAL/NOT_USEFUL/gateway excluded"
    assert [m.model for m in models][:3] == ["g1", "n1", "o1"], "spread across providers, not all of one first"
    assert next(m for m in models if m.model == "g1").key == "GK" and next(m for m in models if m.model == "g1").litellm_id == "groq/g1"
    assert next(m for m in models if m.model == "o1").anonymous
    assert left_out == ["kilo/k1 (freellmapi: reachable only through a gateway)"]
    assert len(MS.load_shortlist(f, {}, 2)[0]) == 2

@pytest.fixture
def flow(monkeypatch, tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "free_model_sweep_latest_verdicts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in verdict_rows()))
    (tmp_path / ".env").write_text(f"GROQ_API_KEY={GROQ_KEY}\n")
    for mod in (MS, FM, S):
        monkeypatch.setattr(mod, "REPORTS", reports)
    monkeypatch.setattr(MS, "REPO", tmp_path)
    monkeypatch.setattr(FM, "pace_for", lambda p: 0)
    days = [date(2026, 7, 1) + timedelta(days=i) for i in range(20)]
    monkeypatch.setattr(MS, "choose_days", lambda today, lag, window: days)
    world = World(direction=oracle)
    world.horizons = (PredictionHorizon.WEEKLY,)
    orig = SC.aggregate
    monkeypatch.setattr(SC, "aggregate", lambda m, s, grade_fn=None: orig(m, s, grader))
    monkeypatch.setattr(MS.SC, "Plumbing", lambda shared, predict, pace, classify: RealPlumbing(
        shared, predict, lambda p: 0, classify, world.clock.sleep, world.clock.now, lambda m: None))
    live_factory = lambda args: SimpleNamespace(shared=world.shared, predict=world.predict)  # noqa: E731
    return reports, world, live_factory

def args(**over):
    ns = MS.normalize(MS.build_parser().parse_args([]))
    ns.tickers, ns.cases, ns.calls_per_provider = "UP.NS,DOWN.NS", 12, 999
    for k, v in over.items():
        setattr(ns, k, v)
    return ns

def run(a, factory):
    return asyncio.run(MS.amain(a, live_factory=factory))

def test_end_to_end_it_scores_the_shortlist_and_writes_a_report_and_a_stable_copy(flow):
    reports, world, factory = flow
    assert run(args(), factory) == 0
    md = (reports / MS.LATEST_MD).read_text()
    assert "Model scorecard" in md and "models scored:** 4" in md
    assert "kilo/k1 (freellmapi: reachable only through a gateway)" in md, "gateway-only models are listed as not scored"
    assert {m for m, _, _ in world.predict_calls} == {"g1", "g2", "n1", "o1"}

def test_a_rerun_uses_the_same_cases_and_repeats_nothing(flow):
    _reports, world, factory = flow
    run(args(), factory)
    first = len(world.predict_calls)
    import time
    time.sleep(1.1)
    run(args(), factory)
    assert len(world.predict_calls) == first, "same pinned cases, all done: nothing re-run"

def test_a_third_run_still_remembers_the_first(flow):
    import time
    reports, world, factory = flow
    run(args(), factory)
    first = len(world.predict_calls)
    for _ in range(2):
        time.sleep(1.1)
        run(args(), factory)
    assert len(world.predict_calls) == first
    assert "models scored:** 4" in (reports / MS.LATEST_MD).read_text()

def test_fresh_starts_over(flow):
    import time
    _reports, world, factory = flow
    run(args(), factory)
    first = len(world.predict_calls)
    time.sleep(1.1)
    run(args(fresh=True, resume=None), factory)
    assert len(world.predict_calls) == 2 * first

def test_the_provider_budget_is_shared_by_its_models_and_a_later_run_finishes_the_job(flow):
    import time
    _reports, world, factory = flow
    run(args(calls_per_provider=12), factory)           # 12 calls = 2 cases PER PROVIDER, not per model
    per_provider = lambda calls: {  # noqa: E731
        p: sum(1 for m, _, _ in calls if m in ms) for p, ms in {"groq": {"g1", "g2"}, "nvidia": {"n1"}, "ovh": {"o1"}}.items()}
    assert per_provider(world.predict_calls) == {"groq": 2, "nvidia": 2, "ovh": 2}, "groq's TWO models split one allowance"
    time.sleep(1.1)
    run(args(calls_per_provider=12), factory)
    assert len(world.predict_calls) == 12, "the next run did the next slice of every provider's cases"

def test_no_sweep_results_is_a_clear_message_not_a_crash(flow, capsys):
    reports, _, factory = flow
    (reports / "free_model_sweep_latest_verdicts.jsonl").unlink()
    assert run(args(), factory) == 2
    assert "Run the sweep first" in capsys.readouterr().out

def test_a_dry_run_writes_nothing_and_needs_no_sweep_results(flow):
    reports, world, factory = flow
    (reports / "free_model_sweep_latest_verdicts.jsonl").unlink()
    before = set(reports.iterdir())
    assert run(args(dry_run=True), factory) == 0
    assert set(reports.iterdir()) == before and world.predict_calls == []

def test_a_crash_still_leaves_a_report_that_says_so(flow):
    reports, world, _factory = flow

    async def boom(c, case, tv):
        raise KeyboardInterrupt
    bad = lambda a: SimpleNamespace(shared=world.shared, predict=boom)  # noqa: E731
    assert run(args(), bad) == 130
    assert "INTERRUPTED" in (reports / MS.LATEST_MD).read_text()

def test_no_useful_models_is_reported_honestly(flow):
    reports, _, factory = flow
    (reports / "free_model_sweep_latest_verdicts.jsonl").write_text(json.dumps({**verdict_rows()[5]}) + "\n")
    assert run(args(), factory) == 0
    assert "nothing to score yet" in (reports / MS.LATEST_MD).read_text()

def test_no_key_is_ever_written_to_any_output(flow):
    reports, _, factory = flow
    run(args(), factory)
    for f in reports.iterdir():
        assert GROQ_KEY not in f.read_text(), f.name

def test_the_scorecard_never_calls_anything_when_there_is_nothing_to_score(flow):
    reports, world, factory = flow
    (reports / "free_model_sweep_latest_verdicts.jsonl").write_text("")
    run(args(), factory)
    assert world.shared_calls == [] and world.predict_calls == []




def test_the_cases_stay_pinned_even_when_the_calendar_moves_so_evidence_accumulates(flow, monkeypatch):
    """Real life: 'the last 70 days' is a different window tomorrow. If every run re-picked its cases, no model
    would ever accumulate more than one run of evidence."""
    import time
    _reports, world, factory = flow
    run(args(), factory)
    first_calls = len(world.predict_calls)
    first_cases = {c for _, c, _ in world.predict_calls}
    monkeypatch.setattr(MS, "choose_days", lambda today, lag, window: [date(2026, 9, 1) + timedelta(days=i) for i in range(20)])
    time.sleep(1.1)
    run(args(), factory)
    assert len(world.predict_calls) == first_calls, "the old cases were reused, so nothing new needed running"
    assert {c for _, c, _ in world.predict_calls} == first_cases, "no case from the shifted window ever appeared"

def test_every_app_name_the_live_adapter_uses_exists_in_the_real_predictor():
    """The fakes above would happily pass if the app renamed one of these. This reads the REAL modules (no network)."""
    import inspect

    from price_predictor.prediction import predictor as P
    from price_predictor.prediction.inputs import SynthesisInput, compose_technical_view
    for name in ("_news_impact_agent", "_synthesizer_agent", "_ensure_news_snapshot_installed", "replay_context",
                 "run_news_impact_agent", "synthesize_with_guardrails", "_finalize_prediction", "_resolve_ticker",
                 "_normalize_horizons", "_NEWS_BACKTEST_REPLAY_TAG", "IST"):
        assert hasattr(P, name), f"predictor.{name} is gone: the scorecard's live adapter would break"
    used = {"ticker", "horizon", "as_of", "technical_view", "impact_assessment", "model_chain"}
    assert used <= set(SynthesisInput.model_fields), f"SynthesisInput lost: {used - set(SynthesisInput.model_fields)}"
    assert "as_of" in inspect.signature(compose_technical_view).parameters
    assert "news_degraded" in inspect.signature(P._finalize_prediction).parameters
    assert {h.value for h in P._normalize_horizons(["daily", "weekly"])} == {"daily", "weekly"}, "horizon names parse"

def test_the_real_replay_context_and_agents_are_the_kind_the_adapter_assumes():
    from price_predictor.prediction import predictor as P
    assert hasattr(P._news_impact_agent, "model") and hasattr(P._synthesizer_agent, "model"), "agents carry a swappable .model"
    assert callable(P.replay_context(date(2026, 7, 20)).__enter__), "replay_context(as_of) is a context manager"


def test_the_scorecard_has_no_time_limit_unless_you_ask_for_one():
    assert MS.build_parser().parse_args([]).max_minutes == 0
    assert MS.build_parser().parse_args(["--max-minutes", "90"]).max_minutes == 90, "a limit is still available"
