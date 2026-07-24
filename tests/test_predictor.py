"""Unit tests for the predict() orchestrator (Step 3.4.2 commit 4).

SCOPE
=====
Tests the orchestration logic by MOCKING the two agent helpers
(run_news_impact_agent, synthesize_with_guardrails). No real LLM is invoked,
no Runner machinery exercised. End-to-end behavior with real LLMs is
deferred to commit 6 (marker-gated).

Tests for the runner.py singletons live alongside (TestRunnerSingletons).

WHY MOCK THE HELPERS, NOT THE RUNNER
====================================
The Runner is ADK's contract; if we mock IT, we couple our tests to
ADK internals. Mocking our own helpers (run_news_impact_agent,
synthesize_with_guardrails) keeps tests at the "predictor business logic"
boundary — exactly the level we own.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from price_predictor.agents.news_impact import Catalyst, ImpactAssessment
from price_predictor.data import _shared_cache
from price_predictor.prediction import (
    Prediction,
    PredictionError,
    SynthesisInput,
    TechnicalViewError,
    predict,
)
from price_predictor.prediction import runner as predictor_runner
from price_predictor.prediction.inputs import ClusterView, TechnicalView
from price_predictor.prediction.schema import (
    AnalysisBasis,
    PredictionDirection,
    PredictionHorizon,
    PriceLevel,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

# Single-horizon shortcut for legacy single-prediction tests. New
# multi-horizon tests should call predict() directly and inspect the
# returned dict. Keeps existing assertions terse without polluting
# production code with a deprecated 1-prediction return shape.
_LEGACY_TEST_HORIZON = PredictionHorizon.WEEKLY


def _run_predict_one(*args, **kwargs) -> Prediction:
    """Run predict() and unwrap the dict to a single Prediction.

    Defaults to a single-horizon call (`[WEEKLY]`) so mocks can use
    `assert_awaited_once()` semantics. Pass an explicit `horizons=`
    kwarg to override.
    """
    horizons = kwargs.pop("horizons", [_LEGACY_TEST_HORIZON])
    result_dict = asyncio.run(predict(*args, horizons=horizons, **kwargs))
    return result_dict[_LEGACY_TEST_HORIZON]

def _build_uptrend_df(n: int = 400) -> pd.DataFrame:
    closes = np.linspace(100, 200, n)
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "open": closes, "high": closes + 1, "low": closes - 1,
            "close": closes, "adj_close": closes,
            "volume": np.full(n, 1_000_000),
        },
        index=dates,
    )


class _FakeCache:
    def __init__(self, df, *, raise_exc=None):
        self.df = df
        self.raise_exc = raise_exc

    async def get(self, ticker, start, end, interval="1d"):
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.df.copy()


def _sample_impact() -> ImpactAssessment:
    """Stand-in ImpactAssessment that the news_impact agent would return."""
    return ImpactAssessment(
        ticker="RELIANCE.NS",
        sentiment="bullish",
        confidence=0.75,
        estimated_pct_move=2.0,
        reasoning="Q3 beat plus margin guidance lift on a healthy tape.",
        catalysts=[
            Catalyst(
                description="Q3 earnings beat consensus by 12% YoY",
                source="news",
                impact="positive",
            ),
        ],
    )


def _sample_prediction() -> Prediction:
    """Stand-in Prediction that the synthesizer agent would return.

    BULLISH so the level invariants are exercised. close ~200 (matches
    our fake-cache uptrend), entry just below close, target above,
    stop below. RR ~2.
    """
    return Prediction(
        ticker="RELIANCE.NS",
        as_of=datetime(2026, 4, 28, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
        horizon=PredictionHorizon.WEEKLY,
        # Synthesizer's prompt copies model_chain from input, so it'll
        # have the news tag here. Predictor appends synth tag after.
        model_chain=("news_impact:agentic",),
        direction=PredictionDirection.BULLISH,
        confidence=0.72,
        entry_zone=(199.0, 201.0),
        target=PriceLevel(value=210.0, rationale="20-day SMA + recent swing high"),
        stop_loss=PriceLevel(value=195.0, rationale="below recent swing low"),
        rationale="Trend bullish across SMAs, RSI healthy at 65, news tailwind from Q3 beat.",
        contributing_signals=("trend bullish", "RSI 65", "Q3 earnings beat"),
        conflicting_signals=("volatility neutral",),
        analysis_basis=AnalysisBasis(
            close_price_at_prediction=200.0,
            bars_used=400,
      technical_summary="Trend bullish, momentum bullish, vol neutral, levels bullish",
            news_sentiment_score=0.75,
            news_articles_considered=1,
            filings_considered=0,
        ),
    )


@pytest.fixture(autouse=True)
def _reset_state():
    """Hermetic isolation: reset cache + runner singletons each test."""
    _shared_cache.set_cache(None)
    predictor_runner.reset()
    yield
    _shared_cache.set_cache(None)
    predictor_runner.reset()


@pytest.fixture
def fake_cache():
    """Inject a 400-bar uptrend into the shared cache."""
    cache = _FakeCache(_build_uptrend_df())
    _shared_cache.set_cache(cache)
    return cache


# ─────────────────────────────────────────────────────────────
# 1. Runner singleton behavior
# ─────────────────────────────────────────────────────────────
class TestRunnerSingletons:
    def test_session_service_is_singleton(self):
        a = predictor_runner.get_session_service()
        b = predictor_runner.get_session_service()
        assert a is b

    def test_runner_cached_per_agent(self):
        from price_predictor.agents.synthesizer import root_agent
        r1 = predictor_runner.get_runner(root_agent)
        r2 = predictor_runner.get_runner(root_agent)
        assert r1 is r2

    def test_reset_clears_singletons(self):
        from price_predictor.agents.synthesizer import root_agent
        r1 = predictor_runner.get_runner(root_agent)
        predictor_runner.reset()
        r2 = predictor_runner.get_runner(root_agent)
        assert r1 is not r2  # fresh after reset


# ─────────────────────────────────────────────────────────────
# 2. predict() happy path
# ─────────────────────────────────────────────────────────────
class TestPredictHappyPath:
    @patch("price_predictor.prediction.predictor.synthesize_with_guardrails",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_returns_prediction(self, mock_news, mock_synth, fake_cache):
        mock_news.return_value = _sample_impact()
        mock_synth.return_value = _sample_prediction()

        result = _run_predict_one("RELIANCE.NS")
        assert isinstance(result, Prediction)
        assert result.ticker == "RELIANCE.NS"
        assert result.direction == PredictionDirection.BULLISH

    @patch("price_predictor.prediction.predictor.synthesize_with_guardrails",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_calls_both_agents(self, mock_news, mock_synth, fake_cache):
        mock_news.return_value = _sample_impact()
        mock_synth.return_value = _sample_prediction()

        _run_predict_one("RELIANCE.NS")
        mock_news.assert_awaited_once_with("RELIANCE.NS")
        mock_synth.assert_awaited_once()
        # Synthesizer received a SynthesisInput
        si = mock_synth.call_args.args[0]
        assert isinstance(si, SynthesisInput)
        assert si.ticker == "RELIANCE.NS"
        assert si.impact_assessment.sentiment == "bullish"

    @patch("price_predictor.prediction.predictor.synthesize_with_guardrails",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_appends_synth_to_model_chain(self, mock_news, mock_synth, fake_cache):
        """Predictor MUST append the synthesizer tag after the call,
        regardless of LLM compliance with copy-from-input."""
        mock_news.return_value = _sample_impact()
        mock_synth.return_value = _sample_prediction()

        result = _run_predict_one("RELIANCE.NS")
        assert "synthesizer:agentic" in result.model_chain
        assert result.model_chain[-1] == "synthesizer:agentic"

    @patch("price_predictor.prediction.predictor.synthesize_with_guardrails",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_resolves_ticker_via_kb(self, mock_news, mock_synth, fake_cache):
        """User can pass 'reliance' (any case); predictor canonicalizes."""
        mock_news.return_value = _sample_impact()
        mock_synth.return_value = _sample_prediction()

        _run_predict_one("reliance")
        # Both agents see the canonical form
        mock_news.assert_awaited_once_with("RELIANCE.NS")
        si = mock_synth.call_args.args[0]
        assert si.ticker == "RELIANCE.NS"


# ─────────────────────────────────────────────────────────────
# 3. predict() failure modes
# ─────────────────────────────────────────────────────────────
class TestPredictFailures:
    @patch("price_predictor.prediction.predictor.synthesize_with_guardrails",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_technical_failure_raises_prediction_error(
        self, mock_news, mock_synth,
    ):
        """Cluster tools failing -> TechnicalViewError -> wrapped as PredictionError."""
        from price_predictor.data.prices import PriceFetchError
        _shared_cache.set_cache(
            _FakeCache(_build_uptrend_df(),
                       raise_exc=PriceFetchError("yahoo down")),
        )
        mock_news.return_value = _sample_impact()
        mock_synth.return_value = _sample_prediction()

        with pytest.raises(PredictionError, match="Technical analysis failed"):
            _run_predict_one("RELIANCE.NS")

    @patch("price_predictor.prediction.predictor.synthesize_with_guardrails",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_news_failure_degrades_to_neutral(
        self, mock_news, mock_synth, fake_cache,
    ):
        """News failure should degrade gracefully (commit 6 contract).

        The synthesizer must still get called (with a degraded
        ImpactAssessment), and predict() must return a Prediction.
        """
        mock_news.side_effect = PredictionError("news_impact failed")
        mock_synth.return_value = _sample_prediction()

        result = _run_predict_one("RELIANCE.NS")
        assert isinstance(result, Prediction)

        # Synthesizer was called - confirm with degraded assessment.
        mock_synth.assert_awaited_once()
        si = mock_synth.call_args.args[0]
        assert si.impact_assessment.sentiment == "neutral"
        assert si.impact_assessment.confidence == 0.0
        assert si.impact_assessment.catalysts == []
        assert "News unavailable" in si.impact_assessment.reasoning

    @patch("price_predictor.prediction.predictor.synthesize_with_guardrails",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_news_failure_marks_model_chain(
        self, mock_news, mock_synth, fake_cache,
    ):
        """Degraded path tags model_chain so consumers can spot it."""
        mock_news.side_effect = RuntimeError("GDELT timeout")
        mock_synth.return_value = _sample_prediction()

        result = _run_predict_one("RELIANCE.NS")
        # Synthesizer's input had degraded marker; final still has
        # synthesizer tag appended after.
        assert "news_impact:degraded" in " ".join(
            mock_synth.call_args.args[0].model_chain
        )
        assert result.model_chain[-1] == "synthesizer:agentic"

    @patch("price_predictor.prediction.predictor.synthesize_with_guardrails",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_news_failure_zeros_analysis_basis(
        self, mock_news, mock_synth, fake_cache,
    ):
        """Final Prediction's analysis_basis must reflect 'no news' even
        if the LLM put non-zero values there."""
        mock_news.side_effect = ValueError("network blip")
        # Sample prediction has news_articles_considered=1 - we expect
        # this to be FORCED to 0 by the degradation override.
        mock_synth.return_value = _sample_prediction()

        result = _run_predict_one("RELIANCE.NS")
        assert result.analysis_basis.news_articles_considered == 0
        assert result.analysis_basis.filings_considered == 0
        assert result.analysis_basis.news_sentiment_score is None

    @patch("price_predictor.prediction.predictor.synthesize_with_guardrails",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_synthesizer_failure_propagates(
        self, mock_news, mock_synth, fake_cache,
    ):
        mock_news.return_value = _sample_impact()
        mock_synth.side_effect = PredictionError("synth bad json")

        with pytest.raises(PredictionError, match="synth bad json"):
            _run_predict_one("RELIANCE.NS")

    def test_empty_ticker_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _run_predict_one("")


# ─────────────────────────────────────────────────────────────
# 4. Multi-horizon fan-out (commit 3 of multi-horizon refactor)
# ─────────────────────────────────────────────────────────────
class TestNormalizeHorizons:
    """Pure unit tests for _normalize_horizons — no async, no mocks."""

    def test_none_returns_default_horizons(self):
        from price_predictor.prediction.predictor import _normalize_horizons
        from price_predictor.prediction.schema import DEFAULT_HORIZONS

        assert _normalize_horizons(None) == DEFAULT_HORIZONS

    def test_accepts_enum_values(self):
        from price_predictor.prediction.predictor import _normalize_horizons

        result = _normalize_horizons([PredictionHorizon.DAILY, PredictionHorizon.MONTHLY])
        assert result == (PredictionHorizon.DAILY, PredictionHorizon.MONTHLY)

    def test_accepts_string_values(self):
        from price_predictor.prediction.predictor import _normalize_horizons

        result = _normalize_horizons(["daily", "weekly"])
        assert result == (PredictionHorizon.DAILY, PredictionHorizon.WEEKLY)

    def test_accepts_mixed_enum_and_strings(self):
        from price_predictor.prediction.predictor import _normalize_horizons

        result = _normalize_horizons(["daily", PredictionHorizon.WEEKLY, "monthly"])
        assert result == (
            PredictionHorizon.DAILY,
            PredictionHorizon.WEEKLY,
            PredictionHorizon.MONTHLY,
        )

    def test_dedupes_preserving_first_occurrence(self):
        from price_predictor.prediction.predictor import _normalize_horizons

        result = _normalize_horizons(["weekly", "daily", "weekly", "daily"])
        assert result == (PredictionHorizon.WEEKLY, PredictionHorizon.DAILY)

    def test_empty_list_raises(self):
        from price_predictor.prediction.predictor import _normalize_horizons

        with pytest.raises(ValueError, match="non-empty"):
            _normalize_horizons([])

    def test_unknown_string_raises(self):
        from price_predictor.prediction.predictor import _normalize_horizons

        with pytest.raises(ValueError, match="Unknown horizon"):
            _normalize_horizons(["yearly"])


class TestPredictMultiHorizon:
    """End-to-end fan-out behavior with mocked agents."""

    @patch("price_predictor.prediction.predictor.synthesize_with_guardrails",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_default_returns_all_four_horizons(
        self, mock_news, mock_synth, fake_cache,
    ):
        """predict() with no horizons argument fans out to DEFAULT_HORIZONS."""
        from price_predictor.prediction.schema import DEFAULT_HORIZONS

        mock_news.return_value = _sample_impact()
        mock_synth.return_value = _sample_prediction()

        result = asyncio.run(predict("RELIANCE.NS"))

        assert isinstance(result, dict)
        assert set(result.keys()) == set(DEFAULT_HORIZONS)
        # Synth called once per horizon (4×); news called ONCE (shared).
        assert mock_synth.await_count == len(DEFAULT_HORIZONS)
        mock_news.assert_awaited_once_with("RELIANCE.NS")

    @patch("price_predictor.prediction.predictor.synthesize_with_guardrails",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_explicit_horizons_subset(
        self, mock_news, mock_synth, fake_cache,
    ):
        """Caller can request a subset of horizons."""
        mock_news.return_value = _sample_impact()
        mock_synth.return_value = _sample_prediction()

        result = asyncio.run(predict(
            "RELIANCE.NS",
            [PredictionHorizon.DAILY, PredictionHorizon.MONTHLY],
        ))

        assert set(result.keys()) == {PredictionHorizon.DAILY, PredictionHorizon.MONTHLY}
        assert mock_synth.await_count == 2

    @patch("price_predictor.prediction.predictor.synthesize_with_guardrails",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_each_synth_input_carries_correct_horizon(
        self, mock_news, mock_synth, fake_cache,
    ):
        """Each parallel synthesis call gets its OWN horizon in SynthesisInput."""
        mock_news.return_value = _sample_impact()
        mock_synth.return_value = _sample_prediction()

        horizons = [PredictionHorizon.DAILY, PredictionHorizon.WEEKLY, PredictionHorizon.MONTHLY]
        asyncio.run(predict("RELIANCE.NS", horizons))

        # Collect horizon values from each synth call's SynthesisInput.
        seen = {call.args[0].horizon for call in mock_synth.call_args_list}
        assert seen == {h.value for h in horizons}

    @patch("price_predictor.prediction.predictor.synthesize_with_guardrails",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_partial_synth_failure_aborts_whole_predict(
        self, mock_news, mock_synth, fake_cache,
    ):
        """If ANY horizon's synth fails, the entire predict() raises.

        Fail-fast contract: the daily+weekly UX promise breaks if we
        return partial dicts. Caller wants either all-or-none.
        """
        mock_news.return_value = _sample_impact()

        call_count = {"n": 0}
        async def _synth_side(*args, **kwargs):
            call_count["n"] += 1
            # Second call fails; first/third would succeed.
            if call_count["n"] == 2:
                raise PredictionError("synth bombed on horizon #2")
            return _sample_prediction()
        mock_synth.side_effect = _synth_side

        with pytest.raises(PredictionError, match="synth bombed"):
            asyncio.run(predict(
                "RELIANCE.NS",
                [PredictionHorizon.DAILY, PredictionHorizon.WEEKLY, PredictionHorizon.MONTHLY],
            ))

    @patch("price_predictor.prediction.predictor.synthesize_with_guardrails",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_gather_phase_runs_once_for_all_horizons(
        self, mock_news, mock_synth, fake_cache,
    ):
        """Technicals + news fetched ONCE even when N horizons requested.

        This is the whole point of the fan-out design: don't waste
        compute or rate limits refetching horizon-agnostic evidence.
        """
        mock_news.return_value = _sample_impact()
        mock_synth.return_value = _sample_prediction()

        asyncio.run(predict(
            "RELIANCE.NS",
            list(PredictionHorizon),  # all 4
        ))

        # News agent called exactly once (gather phase is shared).
        assert mock_news.await_count == 1

    @patch("price_predictor.prediction.predictor.synthesize_with_guardrails",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_string_horizons_work_too(
        self, mock_news, mock_synth, fake_cache,
    ):
        """Ergonomics: caller can pass raw strings; enum is internal."""
        mock_news.return_value = _sample_impact()
        mock_synth.return_value = _sample_prediction()

        result = asyncio.run(predict("RELIANCE.NS", ["daily", "weekly"]))

        assert set(result.keys()) == {PredictionHorizon.DAILY, PredictionHorizon.WEEKLY}

    def test_empty_horizons_list_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            asyncio.run(predict("RELIANCE.NS", []))

    def test_unknown_horizon_raises(self):
        with pytest.raises(ValueError, match="Unknown horizon"):
            asyncio.run(predict("RELIANCE.NS", ["yearly"]))


# ─────────────────────────────────────────────────────────────
# 9. as_of plumbing (Step 1 of backtest harness)
# ─────────────────────────────────────────────────────────────
# These tests pin the contract that as_of:
#   - rejects future dates loudly,
#   - skips news in backtest mode (until Step 1.5),
#   - tags the audit trail distinctly,
#   - flows down to the OHLCV fetch as the `end` cutoff.
#
# Mocking strategy mirrors the existing happy-path tests: we mock the
# two agent helpers, leaving the orchestration + tool plumbing real.
class _SpyCache:
    """Fake cache that records every (start, end) pair it's queried with.

    We need this to assert as_of actually reaches the price-fetch layer
    rather than being silently dropped somewhere mid-call.
    """

    def __init__(self, df):
        self.df = df
        self.calls: list[tuple[date, date]] = []

    async def get(self, ticker, start, end, interval="1d"):
        self.calls.append((start, end))
        return self.df.copy()


@pytest.fixture
def spy_cache():
    cache = _SpyCache(_build_uptrend_df())
    _shared_cache.set_cache(cache)
    return cache


class TestAsOfPlumbing:
    def test_news_tag_for_maps_all_states(self):
        from price_predictor.prediction.predictor import _news_tag_for
        assert _news_tag_for("live") == "news_impact:agentic"
        assert _news_tag_for("degraded") == "news_impact:degraded"
        assert _news_tag_for("agentic_replay") == "news_impact:agentic_replay"
        # Step-1 legacy tag still maps for back-compat.
        assert _news_tag_for("backtest_pending") == "news_impact:backtest_pending"

    def test_future_as_of_rejected(self, fake_cache):
        future = date.today() + pd.Timedelta(days=1).to_pytimedelta()
        with pytest.raises(ValueError, match="future"):
            asyncio.run(predict("RELIANCE.NS", as_of=future))

    @patch("price_predictor.prediction.predictor.synthesize_with_guardrails",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_past_as_of_runs_news_under_replay_context(
        self, mock_news, mock_synth, fake_cache,
    ):
        """Step 1.5: backtest mode now CALLS the news agent (no longer
        skipped). The agent runs under a replay context so its tools
        consult the snapshot store / pin date windows to as_of.

        We assert two contracts:
          1. The news agent IS awaited (live and backtest both call it).
          2. The model_chain carries the replay-distinct tag
             ``news_impact:agentic_replay`` so audits can tell live and
             replay predictions apart.
        """
        # Honest mock: real synthesizer copies model_chain from input.
        async def _synth_copy_chain(si, **kwargs):
            return _sample_prediction().model_copy(
                update={"model_chain": si.model_chain}
            )
        mock_synth.side_effect = _synth_copy_chain
        mock_news.return_value = _sample_impact()
        past = date(2024, 6, 14)

        result = _run_predict_one("RELIANCE.NS", as_of=past)

        mock_news.assert_awaited_once_with("RELIANCE.NS")
        # Audit trail: replay-mode predictions carry a distinct tag.
        assert "news_impact:agentic_replay" in result.model_chain
        assert "news_impact:agentic" not in result.model_chain  # live tag
        assert "news_impact:backtest_pending" not in result.model_chain  # Step-1 tag

    @patch("price_predictor.prediction.predictor.synthesize_with_guardrails",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_default_as_of_still_calls_news(
        self, mock_news, mock_synth, fake_cache,
    ):
        """Live behavior unchanged when as_of is omitted (back-compat)."""
        mock_news.return_value = _sample_impact()
        mock_synth.return_value = _sample_prediction()

        _run_predict_one("RELIANCE.NS")  # no as_of

        mock_news.assert_awaited_once_with("RELIANCE.NS")

    @patch("price_predictor.prediction.predictor.synthesize_with_guardrails",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_past_as_of_stamps_prediction_at_market_close(
        self, mock_news, mock_synth, fake_cache,
    ):
        """Backtest predictions get an as_of pinned to 15:30 IST so the
        timestamp reflects EoD on the requested trading date.
        """
        mock_news.return_value = _sample_impact()
        mock_synth.return_value = _sample_prediction()
        past = date(2024, 6, 14)

        _run_predict_one("RELIANCE.NS", as_of=past)

        si = mock_synth.call_args.args[0]
        assert si.as_of.date() == past
        assert (si.as_of.hour, si.as_of.minute) == (15, 30)
        assert si.as_of.tzinfo is not None  # tz-aware

    @patch("price_predictor.prediction.predictor.synthesize_with_guardrails",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_as_of_reaches_ohlcv_fetch(
        self, mock_news, mock_synth, spy_cache,
    ):
        """The whole point: as_of MUST flow down to the price fetch as
        `end`, otherwise the cluster tools silently see today's bars
        and the entire backtest is a lie.
        """
        mock_news.return_value = _sample_impact()
        mock_synth.return_value = _sample_prediction()
        past = date(2024, 6, 14)

        _run_predict_one("RELIANCE.NS", as_of=past)

        # 4 cluster tools + 1 close/bars-used fetch = 5 cache hits, all
        # ending exactly at as_of.
        assert len(spy_cache.calls) >= 4
        for start, end in spy_cache.calls:
            assert end == past, (
                f"cache fetched with end={end}, expected {past}: as_of leak!"
            )


# ── news-impact LLM gate (LLM only when there's evidence) ───────────
class TestNewsImpactGate:
    """run_news_impact_agent must skip the LLM when gather is empty."""

    def _inputs(self, **kw):
        from price_predictor.agents.news_impact import NewsImpactInputs
        base = dict(
            ticker="X.NS", company_name="X", sector=None,
            window_start="2026-01-01", window_end="2026-01-08",
        )
        base.update(kw)
        return NewsImpactInputs(**base)

    @patch("price_predictor.prediction.predictor._run_agent_for_text",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.gather_news_impact_inputs",
           new_callable=AsyncMock)
    def test_empty_evidence_skips_llm(self, mock_gather, mock_llm):
        from price_predictor.prediction.predictor import run_news_impact_agent
        mock_gather.return_value = self._inputs()  # no evidence
        out = asyncio.run(run_news_impact_agent("X.NS"))
        mock_llm.assert_not_called()             # LLM never invoked
        assert out.sentiment == "neutral"
        assert out.confidence == 0.0
        assert out.catalysts == []

    @patch("price_predictor.prediction.predictor._run_agent_for_text",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.gather_news_impact_inputs",
           new_callable=AsyncMock)
    def test_evidence_invokes_llm(self, mock_gather, mock_llm):
        from price_predictor.prediction.predictor import run_news_impact_agent
        mock_gather.return_value = self._inputs(
            company_news=[{"title": "Big news", "url": "u",
                           "published_at": "2026-01-05", "source": "ET"}],
        )
        mock_llm.return_value = _sample_impact().model_dump_json()
        out = asyncio.run(run_news_impact_agent("X.NS"))
        mock_llm.assert_called_once()            # LLM invoked exactly once
        assert isinstance(out, ImpactAssessment)

    @patch("price_predictor.prediction.predictor._run_agent_for_text",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.gather_news_impact_inputs",
           new_callable=AsyncMock)
    def test_unparseable_output_retries_and_falls_over(
        self, mock_gather, mock_llm, monkeypatch,
    ):
        """A 200-OK-but-unparseable response (reasoning prose) penalizes the
        model and retries down the chain, up to _PARSE_MAX_ATTEMPTS."""
        from price_predictor.llm.resilient import ResilientModel
        from price_predictor.prediction import predictor as P
        from price_predictor.prediction.predictor import (
            _PARSE_MAX_ATTEMPTS,
            PredictionError,
            run_news_impact_agent,
        )
        mock_gather.return_value = self._inputs(
            company_news=[{"title": "Big", "url": "u",
                           "published_at": "2026-01-05", "source": "ET"}],
        )
        mock_llm.return_value = "We need to synthesize... Let's craft."  # no JSON
        assert isinstance(P._news_impact_agent.model, ResilientModel)
        # Simulate a resilient chain that just served this (bad) response, so
        # the penalty has a culprit to cool -> retries fire.
        monkeypatch.setattr(
            P._news_impact_agent.model, "last_used_model", "gemini/x",
        )
        with pytest.raises(PredictionError, match="invalid ImpactAssessment"):
            asyncio.run(run_news_impact_agent("X.NS"))
        assert mock_llm.call_count == _PARSE_MAX_ATTEMPTS  # retried, not 1

    @patch("price_predictor.prediction.predictor._run_agent_for_text",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.gather_news_impact_inputs",
           new_callable=AsyncMock)
    def test_unparseable_output_without_culprit_fails_fast(
        self, mock_gather, mock_llm, monkeypatch,
    ):
        """No penalizable model recorded => single attempt, no spinning."""
        from price_predictor.llm.resilient import ResilientModel
        from price_predictor.prediction import predictor as P
        from price_predictor.prediction.predictor import (
            PredictionError,
            run_news_impact_agent,
        )
        mock_gather.return_value = self._inputs(
            company_news=[{"title": "Big", "url": "u",
                           "published_at": "2026-01-05", "source": "ET"}],
        )
        mock_llm.return_value = "garbage"
        if isinstance(P._news_impact_agent.model, ResilientModel):
            monkeypatch.setattr(
                P._news_impact_agent.model, "last_used_model", None,
            )
        with pytest.raises(PredictionError):
            asyncio.run(run_news_impact_agent("X.NS"))
        assert mock_llm.call_count == 1
