"""Unit tests for hallucination guardrails (Step 3.4.2 commit 5).

Three layers tested:
1. Each validator individually (grounding / citations / consistency)
2. validate_all composition (first-failure-wins ordering)
3. synthesize_with_guardrails retry loop (in test_predictor.py extension)

NO real LLM is invoked. Predictions and inputs are constructed by hand.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pytest

from price_predictor.agents.news_impact import Catalyst, ImpactAssessment
from price_predictor.prediction import (
    HallucinationError,
    Prediction,
    PredictionError,
    SynthesisInput,
    synthesize_with_guardrails,
    validate_all,
    validate_citations,
    validate_consistency,
    validate_grounding,
)
from price_predictor.prediction.guardrails import _build_input_vocabulary
from price_predictor.prediction.inputs import ClusterView, TechnicalView
from price_predictor.prediction.schema import (
    AnalysisBasis,
    PredictionDirection,
    PredictionHorizon,
    PriceLevel,
)


# ─────────────────────────────────────────────────────────────
# Fixtures — all values are realistic for a stock around ₹1455
# ─────────────────────────────────────────────────────────────
CLOSE = 1455.0
ATR = 30.0       # ~2% daily ATR — typical mid-cap
SWING_HIGH = 1500.0
SWING_LOW = 1400.0
R1 = 1480.0
S1 = 1420.0
HIGH_52W = 1620.0
LOW_52W = 1200.0


def _cluster(name, signal, *, indicators=None, derived=None,
             rationale=("ok",)):
    return ClusterView(
        name=name, signal=signal,
        indicators=indicators or {"close": CLOSE},
        derived=derived or {},
        rationale=rationale,
    )


def _make_si(
    *,
    trend="bullish", momentum="bullish",
    volatility="neutral", levels="bullish",
    news_sentiment="bullish",
    catalyst_desc="Q3 earnings beat consensus by 12% YoY",
):
    """Build a SynthesisInput with sensible bullish-by-default defaults."""
    tv = TechnicalView(
        ticker="RELIANCE.NS",
        as_of=date(2026, 4, 28),
        close_price=CLOSE,
        bars_used=400,
        sensitivity="standard",
        trend=_cluster("trend", trend, rationale=("sma_20 above sma_50",)),
        momentum=_cluster(
            "momentum", momentum,
            indicators={"rsi": 65.0, "macd": 1.2},
            rationale=("rsi 65 healthy",),
        ),
        volatility=_cluster(
            "volatility", volatility,
            indicators={"atr": ATR},
            rationale=("atr stable",),
        ),
        levels=_cluster(
            "levels", levels,
            indicators={
                "close": CLOSE, "swing_high": SWING_HIGH, "swing_low": SWING_LOW,
                "r1": R1, "s1": S1, "high_52w": HIGH_52W, "low_52w": LOW_52W,
            },
            derived={"atr": ATR, "breakout_state": "near_resistance"},
            rationale=("near swing_high",),
        ),
    )
    ia = ImpactAssessment(
        ticker="RELIANCE.NS",
        sentiment=news_sentiment,
        confidence=0.7,
        estimated_pct_move=2.5,
        reasoning=f"News flow is {news_sentiment}.",
        catalysts=[
            Catalyst(description=catalyst_desc, source="news", impact="positive"),
        ],
    )
    return SynthesisInput(
        ticker="RELIANCE.NS",
        horizon="short",
        as_of=datetime(2026, 4, 28, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
        technical_view=tv,
        impact_assessment=ia,
        model_chain=("news_impact:agentic",),
    )


def _make_pred(
    *,
    direction=PredictionDirection.BULLISH,
    target_value=SWING_HIGH,
    stop_value=CLOSE - ATR,
    entry=(CLOSE - 2.0, CLOSE + 2.0),
    contributing=("trend bullish across smas", "momentum rsi 65 healthy",
                  "Q3 earnings beat consensus"),
    conflicting=("volatility neutral atr stable",),
):
    return Prediction(
        ticker="RELIANCE.NS",
        as_of=datetime(2026, 4, 28, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
        horizon=PredictionHorizon.SHORT,
        model_chain=("news_impact:agentic",),
        direction=direction,
        confidence=0.72,
        entry_zone=entry,
        target=PriceLevel(value=target_value, rationale="swing_high resistance"),
        stop_loss=PriceLevel(value=stop_value, rationale="below recent low"),
        rationale="Bullish synthesis.",
        contributing_signals=contributing,
        conflicting_signals=conflicting,
        analysis_basis=AnalysisBasis(
            close_price_at_prediction=CLOSE,
            bars_used=400,
            technical_summary="trend bullish, momentum bullish, vol neutral, levels bullish",
            news_sentiment_score=0.7,
            news_articles_considered=1,
            filings_considered=0,
        ),
    )


# ─────────────────────────────────────────────────────────────
# 1. Tier 1 — grounding
# ─────────────────────────────────────────────────────────────
class TestValidateGrounding:
    def test_passes_for_valid_bullish_prediction(self):
        validate_grounding(_make_pred(), _make_si())  # no raise

    def test_fails_when_target_is_invented(self):
        # 1599 passes Pydantic (> entry_top) but isn't within 0.5% of
        # any anchor: swing_high=1500, r1=1480, high_52w=1620 (1.3% off),
        # close+k×ATR ∈ {1485..1545}. So grounding must reject.
        with pytest.raises(HallucinationError, match="grounding") as ex:
            validate_grounding(_make_pred(target_value=1599.0), _make_si())
        assert "target" in str(ex.value)

    def test_passes_when_target_is_atr_multiple(self):
        # close + 2×ATR = 1515 is a valid bullish anchor
        validate_grounding(
            _make_pred(target_value=CLOSE + 2 * ATR), _make_si(),
        )

    def test_fails_when_stop_too_close(self):
        # 0.1×ATR — way too tight
        with pytest.raises(HallucinationError, match="stop_loss"):
            validate_grounding(
                _make_pred(stop_value=CLOSE - 0.1 * ATR), _make_si(),
            )

    def test_fails_when_stop_too_far(self):
        # 3×ATR — way too wide
        with pytest.raises(HallucinationError, match="stop_loss"):
            validate_grounding(
                _make_pred(stop_value=CLOSE - 3.0 * ATR), _make_si(),
            )

    def test_fails_when_entry_drifted_from_close(self):
        # Entry at 1475-1477 (>1% above close=1455) but still satisfies
        # bullish invariants: target=swing_high=1500 > entry_top=1477,
        # stop=1425 < entry_top.
        with pytest.raises(HallucinationError, match="entry"):
            validate_grounding(
                _make_pred(entry=(1475.0, 1477.0)), _make_si(),
            )

    def test_bearish_anchors_to_swing_low(self):
        # Bearish prediction with target near swing_low
        validate_grounding(
            _make_pred(
                direction=PredictionDirection.BEARISH,
                target_value=SWING_LOW,
                stop_value=CLOSE + ATR,  # bearish stop is ABOVE close
            ),
            _make_si(trend="bearish", levels="bearish"),
        )

    def test_bearish_target_above_close_rejected(self):
        # Bearish but target is bullish-side — wrong anchors
        # (Pydantic will catch the cross-field invariant first; we ensure
        # grounding ALSO rejects if it somehow gets through.)
        with pytest.raises(Exception):
            # Either Pydantic invariant or grounding will reject
            pred = _make_pred(
                direction=PredictionDirection.BEARISH,
                target_value=SWING_HIGH,  # wrong direction
                stop_value=CLOSE + ATR,
            )
            validate_grounding(pred, _make_si())


# ─────────────────────────────────────────────────────────────
# 2. Tier 2 — citations
# ─────────────────────────────────────────────────────────────
class TestValidateCitations:
    def test_passes_when_signals_cite_real_evidence(self):
        validate_citations(_make_pred(), _make_si())

    def test_fails_when_signal_is_fabricated(self):
        # 'gamma squeeze' shares no tokens with our fixture vocab
        with pytest.raises(HallucinationError, match="citation"):
            validate_citations(
                _make_pred(contributing=("gamma squeeze imminent",)),
                _make_si(),
            )

    def test_fails_when_signal_is_only_stopwords(self):
        with pytest.raises(HallucinationError, match="informative"):
            validate_citations(
                _make_pred(contributing=("the and or",)),
                _make_si(),
            )

    def test_signal_referencing_catalyst_passes(self):
        validate_citations(
            _make_pred(contributing=("Q3 earnings beat tailwind",)),
            _make_si(catalyst_desc="Q3 earnings beat consensus"),
        )

    def test_signal_referencing_indicator_key_passes(self):
        # 'swing_high' is a key in levels.indicators
        validate_citations(
            _make_pred(contributing=("price near swing_high resistance",)),
            _make_si(),
        )

    def test_vocabulary_includes_cluster_names(self):
        vocab = _build_input_vocabulary(_make_si())
        assert {"trend", "momentum", "volatility", "levels"}.issubset(vocab)


# ─────────────────────────────────────────────────────────────
# 3. Tier 3 — consistency
# ─────────────────────────────────────────────────────────────
class TestValidateConsistency:
    def test_bullish_passes_with_2_bull_clusters(self):
        # default fixture: trend+momentum+levels bullish (3/4)
        validate_consistency(_make_pred(), _make_si())

    def test_bullish_fails_against_bearish_majority(self):
        si = _make_si(
            trend="bearish", momentum="bearish", levels="bearish",
            news_sentiment="neutral",
        )
        with pytest.raises(HallucinationError, match="BULLISH"):
            validate_consistency(_make_pred(), si)

    def test_bullish_passes_with_1_bull_cluster_plus_bullish_news(self):
        si = _make_si(
            trend="bullish", momentum="bearish",
            volatility="neutral", levels="neutral",
            news_sentiment="bullish",
        )
        validate_consistency(_make_pred(), si)  # 1 bull + bullish news = OK

    def test_bearish_passes_with_2_bear_clusters(self):
        si = _make_si(
            trend="bearish", momentum="bearish",
            volatility="neutral", levels="neutral",
            news_sentiment="neutral",
        )
        # Build bearish prediction using bearish anchors
        pred = _make_pred(
            direction=PredictionDirection.BEARISH,
            target_value=SWING_LOW,
            stop_value=CLOSE + ATR,
        )
        validate_consistency(pred, si)

    def test_neutral_always_passes(self):
        # Even with all bearish clusters, neutral is defensible
        si = _make_si(
            trend="bearish", momentum="bearish",
            volatility="bearish", levels="bearish",
        )
        pred = _make_pred(
            direction=PredictionDirection.NEUTRAL,
            target_value=CLOSE + 0.5 * ATR,  # near close
            stop_value=CLOSE - ATR,
        )
        validate_consistency(pred, si)


# ─────────────────────────────────────────────────────────────
# 4. validate_all composition
# ─────────────────────────────────────────────────────────────
class TestValidateAll:
    def test_passes_for_clean_input(self):
        validate_all(_make_pred(), _make_si())

    def test_grounding_runs_first(self):
        # Pred has BOTH bad target (invented 1599) AND bad citations
        # (gamma squeeze). Grounding error must win since it's first.
        with pytest.raises(HallucinationError) as ex:
            validate_all(
                _make_pred(
                    target_value=1599.0,                # invented
                    contributing=("gamma squeeze",),    # fabricated
                ),
                _make_si(),
            )
        assert ex.value.tier == "grounding"


# ─────────────────────────────────────────────────────────────
# 5. synthesize_with_guardrails retry loop
# ─────────────────────────────────────────────────────────────
class TestSynthesizeWithGuardrails:
    @patch("price_predictor.prediction.predictor.run_synthesizer_agent",
           new_callable=AsyncMock)
    def test_first_attempt_passes_no_retry(self, mock_synth):
        mock_synth.return_value = _make_pred()
        result = asyncio.run(synthesize_with_guardrails(_make_si()))
        assert result is mock_synth.return_value
        mock_synth.assert_awaited_once()  # no retry

    @patch("price_predictor.prediction.predictor.run_synthesizer_agent",
           new_callable=AsyncMock)
    def test_retry_succeeds_after_grounding_fix(self, mock_synth):
        bad = _make_pred(target_value=1599.0)  # grounding fail (invented)
        good = _make_pred()                     # grounding pass
        mock_synth.side_effect = [bad, good]

        result = asyncio.run(synthesize_with_guardrails(_make_si()))
        assert result is good
        assert mock_synth.await_count == 2
        # Second call must include feedback
        assert mock_synth.call_args_list[1].kwargs.get("feedback") is not None

    @patch("price_predictor.prediction.predictor.run_synthesizer_agent",
           new_callable=AsyncMock)
    def test_two_failures_raise_prediction_error(self, mock_synth):
        bad = _make_pred(target_value=1599.0)
        mock_synth.side_effect = [bad, bad]

        with pytest.raises(PredictionError, match="twice"):
            asyncio.run(synthesize_with_guardrails(_make_si()))
        assert mock_synth.await_count == 2  # tried twice, then gave up
