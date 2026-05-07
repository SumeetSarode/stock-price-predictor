"""Unit tests for the predict() orchestrator (Step 3.4.2 commit 4).

SCOPE
=====
Tests the orchestration logic by MOCKING the two agent helpers
(run_news_impact_agent, run_synthesizer_agent). No real LLM is invoked,
no Runner machinery exercised. End-to-end behavior with real LLMs is
deferred to commit 6 (marker-gated).

Tests for the runner.py singletons live alongside (TestRunnerSingletons).

WHY MOCK THE HELPERS, NOT THE RUNNER
====================================
The Runner is ADK's contract; if we mock IT, we couple our tests to
ADK internals. Mocking our own helpers (run_news_impact_agent,
run_synthesizer_agent) keeps tests at the "predictor business logic"
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
        horizon=PredictionHorizon.SHORT,
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
    @patch("price_predictor.prediction.predictor.run_synthesizer_agent",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_returns_prediction(self, mock_news, mock_synth, fake_cache):
        mock_news.return_value = _sample_impact()
        mock_synth.return_value = _sample_prediction()

        result = asyncio.run(predict("RELIANCE.NS"))
        assert isinstance(result, Prediction)
        assert result.ticker == "RELIANCE.NS"
        assert result.direction == PredictionDirection.BULLISH

    @patch("price_predictor.prediction.predictor.run_synthesizer_agent",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_calls_both_agents(self, mock_news, mock_synth, fake_cache):
        mock_news.return_value = _sample_impact()
        mock_synth.return_value = _sample_prediction()

        asyncio.run(predict("RELIANCE.NS"))
        mock_news.assert_awaited_once_with("RELIANCE.NS")
        mock_synth.assert_awaited_once()
        # Synthesizer received a SynthesisInput
        si = mock_synth.call_args.args[0]
        assert isinstance(si, SynthesisInput)
        assert si.ticker == "RELIANCE.NS"
        assert si.impact_assessment.sentiment == "bullish"

    @patch("price_predictor.prediction.predictor.run_synthesizer_agent",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_appends_synth_to_model_chain(self, mock_news, mock_synth, fake_cache):
        """Predictor MUST append the synthesizer tag after the call,
        regardless of LLM compliance with copy-from-input."""
        mock_news.return_value = _sample_impact()
        mock_synth.return_value = _sample_prediction()

        result = asyncio.run(predict("RELIANCE.NS"))
        assert "synthesizer:agentic" in result.model_chain
        assert result.model_chain[-1] == "synthesizer:agentic"

    @patch("price_predictor.prediction.predictor.run_synthesizer_agent",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_resolves_ticker_via_kb(self, mock_news, mock_synth, fake_cache):
        """User can pass 'reliance' (any case); predictor canonicalizes."""
        mock_news.return_value = _sample_impact()
        mock_synth.return_value = _sample_prediction()

        asyncio.run(predict("reliance"))
        # Both agents see the canonical form
        mock_news.assert_awaited_once_with("RELIANCE.NS")
        si = mock_synth.call_args.args[0]
        assert si.ticker == "RELIANCE.NS"


# ─────────────────────────────────────────────────────────────
# 3. predict() failure modes
# ─────────────────────────────────────────────────────────────
class TestPredictFailures:
    @patch("price_predictor.prediction.predictor.run_synthesizer_agent",
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
            asyncio.run(predict("RELIANCE.NS"))

    @patch("price_predictor.prediction.predictor.run_synthesizer_agent",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_news_failure_propagates(self, mock_news, mock_synth, fake_cache):
        """News-side PredictionError bubbles up (commit 5 will degrade this)."""
        mock_news.side_effect = PredictionError("news_impact failed")
        mock_synth.return_value = _sample_prediction()

        with pytest.raises(PredictionError, match="news_impact failed"):
            asyncio.run(predict("RELIANCE.NS"))

    @patch("price_predictor.prediction.predictor.run_synthesizer_agent",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.run_news_impact_agent",
           new_callable=AsyncMock)
    def test_synthesizer_failure_propagates(
        self, mock_news, mock_synth, fake_cache,
    ):
        mock_news.return_value = _sample_impact()
        mock_synth.side_effect = PredictionError("synth bad json")

        with pytest.raises(PredictionError, match="synth bad json"):
            asyncio.run(predict("RELIANCE.NS"))

    def test_empty_ticker_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            asyncio.run(predict(""))
