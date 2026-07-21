"""Golden / regression tests for the prediction pipeline.

WHY THIS FILE EXISTS
====================
The pipeline is half deterministic Python (technicals, gather, gates)
and half stochastic LLM (news impact, synthesis). You cannot golden-test
an LLM's words, but you CAN:

  1. Freeze the data inputs (a fixed synthetic OHLCV frame, no network).
  2. Stub the single LLM chokepoint (no model call, no quota, no VPN).

...and then assert the pipeline is DETERMINISTIC and structurally stable.
That catches the real regressions a refactor introduces: "my change
silently rerouted the pipeline / changed a technical number / dropped a
horizon / stopped finalizing predictions."

This whole file runs OFFLINE. No VPN, no LLM, no network. Every input is
synthetic and every model call is stubbed.

It also PRINTS a snapshot of the computed technical view + a sample
prediction (visible with `pytest -s`) so a human can eyeball the golden
baseline and diff it across runs.
"""
from __future__ import annotations

import asyncio
import json
from datetime import date, datetime
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from price_predictor.agents.news_impact import Catalyst, ImpactAssessment
from price_predictor.data import _shared_cache
from price_predictor.prediction import Prediction, predict
from price_predictor.prediction import runner as predictor_runner
from price_predictor.prediction.inputs import compose_technical_view
from price_predictor.prediction.schema import (
    AnalysisBasis,
    PredictionDirection,
    PredictionHorizon,
    PriceLevel,
)

IST = ZoneInfo("Asia/Kolkata")
_TICKER = "RELIANCE.NS"


# ═════════════════════════════════════════════════════════════
# Deterministic, network-free fixtures
# ═════════════════════════════════════════════════════════════
def _fixed_uptrend(n: int = 400) -> pd.DataFrame:
    """A perfectly reproducible 100→200 linear uptrend. No randomness."""
    closes = np.linspace(100.0, 200.0, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "open": closes, "high": closes + 1.0, "low": closes - 1.0,
            "close": closes, "adj_close": closes,
            "volume": np.full(n, 1_000_000),
        },
        index=idx,
    )


class _FixedCache:
    """Serves the SAME frame for any ticker/date — full determinism."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    async def get(self, ticker, start, end, interval="1d"):
        return self._df.copy()


@pytest.fixture(autouse=True)
def _hermetic():
    """Reset shared cache + runner singletons around every test."""
    _shared_cache.set_cache(None)
    predictor_runner.reset()
    yield
    _shared_cache.set_cache(None)
    predictor_runner.reset()


@pytest.fixture
def frozen_cache():
    _shared_cache.set_cache(_FixedCache(_fixed_uptrend()))


def _sample_impact() -> ImpactAssessment:
    return ImpactAssessment(
        ticker=_TICKER,
        sentiment="bullish",
        confidence=0.7,
        estimated_pct_move=1.5,
        reasoning="Q3 earnings beat with strong guidance; sector tailwind.",
        catalysts=[
            Catalyst(description="Q3 earnings beat consensus by 12% YoY",
                     source="news", impact="positive"),
        ],
    )


def _sample_prediction(horizon: PredictionHorizon) -> Prediction:
    """Deterministic stand-in for the synthesizer's LLM output."""
    return Prediction(
        ticker=_TICKER,
        as_of=datetime(2026, 4, 28, 15, 30, tzinfo=IST),
        horizon=horizon,
        model_chain=("news_impact:agentic",),
        direction=PredictionDirection.BULLISH,
        confidence=0.72,
        entry_zone=(199.0, 201.0),
        target=PriceLevel(value=210.0, rationale="20-day SMA + swing high"),
        stop_loss=PriceLevel(value=195.0, rationale="below recent swing low"),
        rationale="Trend bullish across SMAs, RSI healthy, Q3 news tailwind.",
        contributing_signals=("trend bullish", "RSI healthy", "Q3 beat"),
        conflicting_signals=("volatility neutral",),
        analysis_basis=AnalysisBasis(
            close_price_at_prediction=200.0,
            bars_used=400,
            technical_summary="Trend bullish, momentum bullish, vol neutral",
            news_sentiment_score=0.7,
            news_articles_considered=1,
            filings_considered=0,
        ),
    )


# ═════════════════════════════════════════════════════════════
# 1. Technical view — the deterministic heart (REAL code, no stubs)
# ═════════════════════════════════════════════════════════════
class TestTechnicalGolden:
    """Same OHLCV in → byte-identical technical view out, every time."""

    def test_deterministic_and_stable(self, frozen_cache, capsys):
        tv1 = asyncio.run(compose_technical_view(_TICKER))
        # rebuild cache (autouse reset wiped it) and run again
        _shared_cache.set_cache(_FixedCache(_fixed_uptrend()))
        tv2 = asyncio.run(compose_technical_view(_TICKER))

        d1 = tv1.model_dump(mode="json")
        d2 = tv2.model_dump(mode="json")

        # (a) DETERMINISM: identical output for identical input.
        assert d1 == d2, "technical view is NOT deterministic!"

        # (b) INVARIANTS: fixed frame → known anchors.
        assert tv1.close_price == pytest.approx(200.0)
        assert tv1.bars_used == 400
        for cluster in (tv1.trend, tv1.momentum, tv1.volatility, tv1.levels):
            assert cluster.signal in {
                "bullish", "bearish", "neutral", "mixed",
            }

        # (c) SNAPSHOT for human review (visible with `pytest -s`).
        print("\n===== GOLDEN: technical_view =====")
        print(json.dumps({
            "close_price": tv1.close_price,
            "bars_used": tv1.bars_used,
            "trend": tv1.trend.signal,
            "momentum": tv1.momentum.signal,
            "volatility": tv1.volatility.signal,
            "levels": tv1.levels.signal,
        }, indent=2))


# ═════════════════════════════════════════════════════════════
# 2. News gate — LLM only when there's evidence (stubbed chokepoint)
# ═════════════════════════════════════════════════════════════
class TestNewsGateGolden:
    def _inputs(self, **kw):
        from price_predictor.agents.news_impact import NewsImpactInputs
        base = dict(
            ticker=_TICKER, company_name="Reliance", sector=None,
            window_start="2026-01-01", window_end="2026-01-08",
        )
        base.update(kw)
        return NewsImpactInputs(**base)

    @patch("price_predictor.prediction.predictor._run_agent_for_text",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.gather_news_impact_inputs",
           new_callable=AsyncMock)
    def test_no_evidence_skips_llm(self, mock_gather, mock_llm):
        from price_predictor.prediction.predictor import run_news_impact_agent
        mock_gather.return_value = self._inputs()  # empty
        out = asyncio.run(run_news_impact_agent(_TICKER))
        mock_llm.assert_not_called()
        assert out.sentiment == "neutral"
        assert out.confidence == 0.0

    @patch("price_predictor.prediction.predictor._run_agent_for_text",
           new_callable=AsyncMock)
    @patch("price_predictor.prediction.predictor.gather_news_impact_inputs",
           new_callable=AsyncMock)
    def test_evidence_calls_llm_and_parses(self, mock_gather, mock_llm):
        from price_predictor.prediction.predictor import run_news_impact_agent
        mock_gather.return_value = self._inputs(
            company_news=[{"title": "Q3 beat", "url": "u",
                           "published_at": "2026-01-05", "source": "ET"}],
        )
        mock_llm.return_value = _sample_impact().model_dump_json()
        out = asyncio.run(run_news_impact_agent(_TICKER))
        mock_llm.assert_called_once()
        assert isinstance(out, ImpactAssessment)
        assert out.sentiment == "bullish"


# ═════════════════════════════════════════════════════════════
# 3. Full predict() orchestration — deterministic wiring
# ═════════════════════════════════════════════════════════════
class TestPredictGolden:
    """End-to-end predict() with real technicals + stubbed LLM helpers."""

    _ALL = (
        PredictionHorizon.DAILY,
        PredictionHorizon.WEEKLY,
        PredictionHorizon.MONTHLY,
    )

    def _run(self):
        with patch(
            "price_predictor.prediction.predictor.run_news_impact_agent",
            new_callable=AsyncMock,
        ) as mock_news, patch(
            "price_predictor.prediction.predictor.synthesize_with_guardrails",
            new_callable=AsyncMock,
        ) as mock_synth:
            mock_news.return_value = _sample_impact()
            mock_synth.side_effect = lambda si: _sample_prediction(
                PredictionHorizon(si.horizon)
            )
            return asyncio.run(
                predict(_TICKER, horizons=list(self._ALL),
                        as_of=date(2026, 4, 28))
            )

    def test_all_horizons_present(self, frozen_cache):
        out = self._run()
        assert set(out.keys()) == set(self._ALL)
        for h, pred in out.items():
            assert isinstance(pred, Prediction)
            assert pred.horizon == h
            assert pred.ticker == _TICKER
            # finalize ran → model_chain carries at least the news tag.
            assert len(pred.model_chain) >= 1

    def test_deterministic_across_runs(self, frozen_cache, capsys):
        _shared_cache.set_cache(_FixedCache(_fixed_uptrend()))
        out1 = self._run()
        _shared_cache.set_cache(_FixedCache(_fixed_uptrend()))
        out2 = self._run()

        dump1 = {h.value: p.model_dump(mode="json") for h, p in out1.items()}
        dump2 = {h.value: p.model_dump(mode="json") for h, p in out2.items()}
        assert dump1 == dump2, "predict() is NOT deterministic!"

        print("\n===== GOLDEN: predict() sample (weekly) =====")
        wk = out1[PredictionHorizon.WEEKLY]
        print(json.dumps({
            "direction": wk.direction.value,
            "confidence": wk.confidence,
            "entry_zone": list(wk.entry_zone),
            "target": wk.target.value,
            "stop_loss": wk.stop_loss.value,
            "model_chain": list(wk.model_chain),
            "horizons_returned": sorted(h.value for h in out1),
        }, indent=2))
