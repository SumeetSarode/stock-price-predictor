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
    SynthesisParseError,
    synthesize_with_guardrails,
    validate_all,
    validate_calibration,
    validate_citations,
    validate_consistency,
    validate_grounding,
)
from price_predictor.prediction.guardrails import _build_input_vocabulary, _tokenize
from price_predictor.prediction.horizon_constants import (
    confidence_cap,
    stop_atr_range,
)
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
    horizon="weekly",
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
        horizon=horizon,
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
    confidence=0.72,
    horizon=PredictionHorizon.WEEKLY,
    contributing=("trend bullish across smas", "momentum rsi 65 healthy",
                  "Q3 earnings beat consensus"),
    conflicting=("volatility neutral atr stable",),
):
    return Prediction(
        ticker="RELIANCE.NS",
        as_of=datetime(2026, 4, 28, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
        horizon=horizon,
        model_chain=("news_impact:agentic",),
        direction=direction,
        confidence=confidence,
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
# 2b. Tier 2 regression tests — the tokenizer/vocab "informative tokens" bug.
#
# The original tokenizer had three bugs that combined to reject
# perfectly-grounded LLM citations:
#   - alpha tokens filtered by `len > 3` dropped RSI, ATR, ADX, SMA, EMA
#   - regex stripped all numerics, dropping the actual indicator values
#   - "above" / "below" / "near" / "high" / "low" were stopwords
# Together: a citation like "RSI 39.6 is below 50" tokenized to set()
# → "no informative tokens" → retry → same failure → PredictionError.
# These tests pin the fix so the bug class cannot regress.
# ─────────────────────────────────────────────────────────────
class TestTokenizerRegression:
    """Pin the tokenizer/vocab fixes for short indicator names + numerics."""

    def test_rsi_three_char_token_survives_filter(self):
        """`rsi` is exactly 3 chars; old `len > 3` filter dropped it."""
        assert "rsi" in _tokenize("RSI is bullish")

    def test_atr_three_char_token_survives_filter(self):
        assert "atr" in _tokenize("ATR widening")

    def test_decimal_numeric_is_kept(self):
        """\"39.6\" is the strongest grounding signal; must tokenize."""
        tokens = _tokenize("RSI 39.6 is below 50")
        assert "39.6" in tokens

    def test_three_digit_numeric_is_kept(self):
        assert "200" in _tokenize("SMA 200 reclaimed")

    def test_short_generic_numeric_is_dropped(self):
        """"50" matches everywhere — too generic to count as evidence."""
        assert "50" not in _tokenize("price near 50")

    def test_below_is_no_longer_stopword(self):
        """\"below\" is core TA vocabulary, not filler."""
        assert "below" in _tokenize("close below sma_50")

    def test_above_is_no_longer_stopword(self):
        assert "above" in _tokenize("close above resistance")

    def test_pure_filler_still_dropped(self):
        assert _tokenize("the and a or of in on") == set()

    def test_rsi_citation_passes_validate_citations(self):
        """End-to-end: the exact LLM output that triggered prod failure."""
        si = _make_si()
        # Inject a rationale containing "rsi" so the LLM's citation is grounded
        # via the input vocabulary (matches how real cluster outputs look).
        pred = _make_pred(contributing=("RSI 39.6 is below 50: bearish bias",))
        validate_citations(pred, si)  # MUST NOT raise

    def test_atr_citation_passes_validate_citations(self):
        si = _make_si()
        pred = _make_pred(contributing=("ATR widening above 20-day average",))
        validate_citations(pred, si)  # MUST NOT raise

    def test_indicator_key_rsi_in_vocabulary(self):
        """Vocab builder must include short indicator names too."""
        vocab = _build_input_vocabulary(_make_si())
        assert "rsi" in vocab
        assert "atr" in vocab


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

    def test_reports_all_failures_at_once(self):
        # Pred has BOTH bad target (invented 1599) AND fabricated citation
        # (gamma squeeze). Collect-all: the raised error must name BOTH
        # tiers so the LLM can fix them together (no whack-a-mole).
        with pytest.raises(HallucinationError) as ex:
            validate_all(
                _make_pred(
                    target_value=1599.0,                # invented -> grounding
                    contributing=("gamma squeeze",),    # fabricated -> citation
                ),
                _make_si(),
            )
        # Combined tier field lists every failing tier.
        assert "grounding" in ex.value.tier
        assert "citation" in ex.value.tier
        # Message enumerates both violations.
        msg = str(ex.value)
        assert "grounding" in msg and "citation" in msg
        assert "simultaneously" in msg

    def test_single_failure_passes_through_unwrapped(self):
        # Only ONE tier fails -> the original single error is raised as-is
        # (no needless 'N checks failed' wrapper).
        with pytest.raises(HallucinationError) as ex:
            validate_all(_make_pred(target_value=1599.0), _make_si())
        assert ex.value.tier == "grounding"
        assert "simultaneously" not in str(ex.value)


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
    def test_third_attempt_can_still_succeed(self, mock_synth):
        """Budget is now 3 total attempts (1 initial + 2 retries).

        Pinned because the SBIN.NS prod failure was caused by the LLM's
        stochastic sampling: attempts 1+3 fail on boundary rounding,
        attempt 2 passes. With only 2 attempts, the diagnostic showed
        we were unlucky enough to hit two failures back-to-back.
        """
        bad = _make_pred(target_value=1599.0)
        good = _make_pred()
        mock_synth.side_effect = [bad, bad, good]

        result = asyncio.run(synthesize_with_guardrails(_make_si()))
        assert result is good
        assert mock_synth.await_count == 3
        # Last call must include feedback from the prior failure
        assert mock_synth.call_args_list[2].kwargs.get("feedback") is not None

    @patch("price_predictor.prediction.predictor.run_synthesizer_agent",
           new_callable=AsyncMock)
    def test_persistent_failures_raise_prediction_error(self, mock_synth):
        # Budget is sized to the model chain so a persistent guardrail trip
        # rotates through EVERY model (incl. the Ollama tail) before giving
        # up -- so exhaust exactly that many attempts, not a hardcoded 3.
        from price_predictor.prediction.predictor import (
            _guardrail_attempt_budget,
        )
        budget = _guardrail_attempt_budget()
        bad = _make_pred(target_value=1599.0)
        mock_synth.side_effect = [bad] * budget

        with pytest.raises(PredictionError, match="Synthesizer failed"):
            asyncio.run(synthesize_with_guardrails(_make_si()))
        assert mock_synth.await_count == budget  # tried whole chain, gave up

    @patch("price_predictor.prediction.predictor.run_synthesizer_agent",
           new_callable=AsyncMock)
    def test_feedback_is_cumulative_across_attempts(self, mock_synth):
        """The retry loop must REMEMBER earlier failures (anti whack-a-mole).

        Attempt 1 fails grounding, attempt 2 fails a DIFFERENT tier
        (citation). The feedback handed to attempt 3 must still mention
        the grounding problem from attempt 1 -- otherwise the LLM would
        happily re-break what it already fixed.
        """
        bad_grounding = _make_pred(target_value=1599.0)          # grounding
        bad_citation = _make_pred(contributing=("gamma squeeze",))  # citation
        good = _make_pred()
        mock_synth.side_effect = [bad_grounding, bad_citation, good]

        result = asyncio.run(synthesize_with_guardrails(_make_si()))
        assert result is good
        assert mock_synth.await_count == 3
        # Attempt 3's feedback remembers BOTH prior problems.
        feedback3 = mock_synth.call_args_list[2].kwargs.get("feedback")
        assert feedback3 is not None
        assert "grounding" in feedback3   # from attempt 1, NOT forgotten
        assert "citation" in feedback3    # from attempt 2

    # ─────────────────────────────────────────────────────────────
    # SynthesisParseError retry path — regression for TCS.NS daily prod
    # failure (2026-05-16). Symptom: Groq returned an empty / unparseable
    # response from the synthesizer, run_synthesizer_agent raised
    # SynthesisParseError, and the old retry loop (which only caught
    # HallucinationError) let it escape on attempt 1 with no recovery.
    # ─────────────────────────────────────────────────────────────
    @patch("price_predictor.prediction.predictor.run_synthesizer_agent",
           new_callable=AsyncMock)
    def test_retry_succeeds_after_parse_failure(self, mock_synth):
        """Empty/garbage LLM response on attempt 1 → retry → clean parse."""
        parse_err = SynthesisParseError(
            "synthesizer agent returned invalid Prediction JSON: "
            "Invalid JSON: expected value at line 1 column 1"
        )
        good = _make_pred()
        mock_synth.side_effect = [parse_err, good]

        result = asyncio.run(synthesize_with_guardrails(_make_si()))
        assert result is good
        assert mock_synth.await_count == 2
        # Retry MUST surface the parse error in the feedback so the LLM
        # knows to emit clean JSON next time.
        retry_feedback = mock_synth.call_args_list[1].kwargs.get("feedback")
        assert retry_feedback is not None
        assert "JSON" in retry_feedback

    @patch("price_predictor.prediction.predictor.run_synthesizer_agent",
           new_callable=AsyncMock)
    def test_retry_succeeds_after_mixed_parse_then_guardrail_failure(self, mock_synth):
        """Realistic cascade: parse-fail → guardrail-fail → clean."""
        parse_err = SynthesisParseError("unparseable")
        bad = _make_pred(target_value=1599.0)  # grounding fail
        good = _make_pred()
        mock_synth.side_effect = [parse_err, bad, good]

        result = asyncio.run(synthesize_with_guardrails(_make_si()))
        assert result is good
        assert mock_synth.await_count == 3

    @patch("price_predictor.prediction.predictor.run_synthesizer_agent",
           new_callable=AsyncMock)
    def test_persistent_parse_failures_raise_prediction_error(self, mock_synth):
        """If every attempt yields unparseable JSON, give up with context."""
        from price_predictor.prediction.predictor import (
            _guardrail_attempt_budget,
        )
        budget = _guardrail_attempt_budget()
        parse_err = SynthesisParseError("unparseable")
        mock_synth.side_effect = [parse_err] * budget

        with pytest.raises(PredictionError, match="Synthesizer failed"):
            asyncio.run(synthesize_with_guardrails(_make_si()))
        assert mock_synth.await_count == budget


# ────────────────────────────────────────────
# 6. Tier 4 — calibration (per-horizon confidence cap)
# ────────────────────────────────────────────
class TestValidateCalibration:
    """Tier 4 — the humility check.

    Long-horizon predictions are inherently more uncertain. The cap
    enforces "a monthly call cannot claim 0.95 confidence." Caps live
    in horizon_constants.CONFIDENCE_CAP_BY_HORIZON; tests reference the
    helper rather than hard-coding numbers so future tuning doesn't
    silently break here.
    """

    @pytest.mark.parametrize("horizon", list(PredictionHorizon))
    def test_passes_below_cap(self, horizon: PredictionHorizon):
        cap = confidence_cap(horizon)
        # confidence comfortably below cap
        pred = _make_pred(horizon=horizon, confidence=cap - 0.05)
        validate_calibration(pred, _make_si(horizon=horizon.value))

    @pytest.mark.parametrize("horizon", list(PredictionHorizon))
    def test_passes_at_cap(self, horizon: PredictionHorizon):
        # confidence == cap is the boundary; spec says "exceeds cap"
        # is rejected, so equal-to-cap must pass.
        cap = confidence_cap(horizon)
        pred = _make_pred(horizon=horizon, confidence=cap)
        validate_calibration(pred, _make_si(horizon=horizon.value))

    @pytest.mark.parametrize("horizon", list(PredictionHorizon))
    def test_rejects_above_cap(self, horizon: PredictionHorizon):
        cap = confidence_cap(horizon)
        # cap + 0.001 to test boundary; cap + 0.05 to test obvious case
        for over in (cap + 0.001, min(cap + 0.05, 0.99)):
            if over <= cap:
                continue  # cap was already 0.99+
            pred = _make_pred(horizon=horizon, confidence=over)
            with pytest.raises(HallucinationError, match="calibration") as ex:
                validate_calibration(pred, _make_si(horizon=horizon.value))
            assert ex.value.tier == "calibration"
            assert horizon.value in str(ex.value)

    def test_validate_all_chain_includes_calibration(self):
        """validate_all runs calibration after the substantive tiers.

        A prediction that passes grounding/citations/consistency but
        has confidence above the horizon cap must be caught by the
        calibration tier inside validate_all.
        """
        # Default fixture is WEEKLY (cap=0.85). Set confidence > cap.
        pred = _make_pred(confidence=0.95)  # weekly cap is 0.85
        with pytest.raises(HallucinationError) as ex:
            validate_all(pred, _make_si())
        assert ex.value.tier == "calibration"


# ────────────────────────────────────────────
# 7. Per-horizon stop bounds (Tier 1 grounding, parametrized)
# ────────────────────────────────────────
class TestPerHorizonStopBounds:
    """Confirm stop bounds are pulled per-horizon (commit B integration).

    Same prediction shape, varying horizon — the same stop ATR multiple
    can be valid for one horizon and invalid for another. This is the
    single most important behavioral change in commit B.
    """

    @pytest.mark.parametrize("horizon", list(PredictionHorizon))
    def test_stop_at_min_passes(self, horizon: PredictionHorizon):
        """Stop exactly at horizon's min ATR multiple should pass."""
        stop_min, _ = stop_atr_range(horizon)
        # Stop at exactly stop_min * ATR (boundary inclusive)
        pred = _make_pred(
            horizon=horizon,
            stop_value=CLOSE - stop_min * ATR,
            # confidence must respect the per-horizon cap too
            confidence=min(0.70, confidence_cap(horizon)),
        )
        validate_grounding(pred, _make_si(horizon=horizon.value))

    @pytest.mark.parametrize("horizon", list(PredictionHorizon))
    def test_stop_at_max_passes(self, horizon: PredictionHorizon):
        """Stop exactly at horizon's max ATR multiple should pass."""
        _, stop_max = stop_atr_range(horizon)
        pred = _make_pred(
            horizon=horizon,
            stop_value=CLOSE - stop_max * ATR,
            confidence=min(0.70, confidence_cap(horizon)),
        )
        validate_grounding(pred, _make_si(horizon=horizon.value))

    @pytest.mark.parametrize("horizon", list(PredictionHorizon))
    def test_stop_below_min_rejected(self, horizon: PredictionHorizon):
        """Stop tighter than horizon's minimum should be rejected."""
        stop_min, _ = stop_atr_range(horizon)
        # Cut min in half — unambiguously too tight
        pred = _make_pred(
            horizon=horizon,
            stop_value=CLOSE - (stop_min / 2.0) * ATR,
            confidence=min(0.70, confidence_cap(horizon)),
        )
        with pytest.raises(HallucinationError, match="stop_loss"):
            validate_grounding(pred, _make_si(horizon=horizon.value))

    @pytest.mark.parametrize("horizon", list(PredictionHorizon))
    def test_stop_above_max_rejected(self, horizon: PredictionHorizon):
        """Stop wider than horizon's maximum should be rejected."""
        _, stop_max = stop_atr_range(horizon)
        # 1.5× the max — unambiguously too wide
        pred = _make_pred(
            horizon=horizon,
            stop_value=CLOSE - (stop_max * 1.5) * ATR,
            confidence=min(0.70, confidence_cap(horizon)),
        )
        with pytest.raises(HallucinationError, match="stop_loss"):
            validate_grounding(pred, _make_si(horizon=horizon.value))

    def test_daily_stop_too_wide_for_daily_but_ok_for_monthly(self):
        """The headline test: same stop value, different verdicts.

        Stop at 1.8×ATR is OUTSIDE daily's (0.5, 1.0) but INSIDE
        monthly's (1.5, 2.5). Same prediction shape, the per-horizon
        rulebook gives opposite answers — exactly the bug commit B
        was built to fix.
        """
        stop_value = CLOSE - 1.8 * ATR

        # DAILY: 1.8×ATR > daily max (1.0) → reject
        daily_pred = _make_pred(
            horizon=PredictionHorizon.DAILY,
            stop_value=stop_value,
            confidence=0.70,  # under daily cap (0.90)
        )
        with pytest.raises(HallucinationError, match="stop_loss"):
            validate_grounding(daily_pred, _make_si(horizon="daily"))

        # MONTHLY: 1.8×ATR ∈ (1.5, 2.5) → accept
        monthly_pred = _make_pred(
            horizon=PredictionHorizon.MONTHLY,
            stop_value=stop_value,
            confidence=0.70,  # under monthly cap (0.75)
        )
        validate_grounding(monthly_pred, _make_si(horizon="monthly"))
