"""Tests for price_predictor.prediction.schema.

Coverage map (~28 tests across 7 sections):
    1. Happy-path construction (2)
    2. JSON round-trip + hashability (3)
    3. Immutability (frozen contract) (2)
    4. Field-level constraints (confidence, prices, sentiment range) (5)
    5. Cross-field validation (entry_zone, model_chain, tz-aware) (5)
    6. Direction-specific level ordering (5)
    7. risk_reward computed math — all three directions (5)
    8. PriceLevel + AnalysisBasis edge cases (4)
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from price_predictor.prediction import (
    AnalysisBasis,
    Prediction,
    PredictionDirection,
    PredictionHorizon,
    PriceLevel,
)


# ─────────────────────────────────────────────────────────────
# Helpers — known-valid kwargs for negative-test overrides
# ─────────────────────────────────────────────────────────────
def _valid_basis_kwargs() -> dict:
    return {
        "close_price_at_prediction": 1455.0,
        "bars_used": 400,
        "technical_summary": "Trend bullish (ADX 32), RSI 65",
        "news_sentiment_score": 0.1,
        "news_articles_considered": 12,
        "filings_considered": 2,
    }


def _valid_bullish_kwargs() -> dict:
    """Baseline kwargs for a valid bullish prediction.

    Override any single field to test a specific violation.
    """
    return {
        "ticker": "RELIANCE.NS",
        "as_of": datetime(2026, 4, 28, 15, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
        "horizon": PredictionHorizon.WEEKLY,
        "model_chain": ("gemini/gemini-2.5-flash",),
        "direction": PredictionDirection.BULLISH,
        "confidence": 0.7,
        "entry_zone": (1450.0, 1470.0),
        "target": PriceLevel(value=1600.0, rationale="Resistance from Q4 high"),
        "stop_loss": PriceLevel(value=1400.0, rationale="Below 50-day SMA"),
        "rationale": "Trend bullish, momentum positive, setup is clean.",
        "contributing_signals": ("ADX 32 (strong trend)",),
        "conflicting_signals": (),
        "analysis_basis": AnalysisBasis(**_valid_basis_kwargs()),
    }


def _valid_bearish_kwargs() -> dict:
    kw = _valid_bullish_kwargs()
    kw.update({
        "direction": PredictionDirection.BEARISH,
        # For bearish: target BELOW entry-bottom, stop ABOVE entry-bottom
        "target": PriceLevel(value=1300.0, rationale="Prior support break"),
        "stop_loss": PriceLevel(value=1500.0, rationale="Above swing high"),
    })
    return kw


# ─────────────────────────────────────────────────────────────
# 1. Happy-path construction
# ─────────────────────────────────────────────────────────────
class TestHappyPath:
    def test_bullish_prediction_constructs(self):
        """The minimal valid bullish setup constructs without error."""
        p = Prediction(**_valid_bullish_kwargs())
        assert p.ticker == "RELIANCE.NS"
        assert p.direction == PredictionDirection.BULLISH
        assert p.confidence == 0.7
        assert p.not_advice is True       # default
        assert p.is_educational is True   # default

    def test_bearish_prediction_constructs(self):
        """The mirror-image bearish setup also constructs cleanly."""
        p = Prediction(**_valid_bearish_kwargs())
        assert p.direction == PredictionDirection.BEARISH
        assert p.target.value == 1300.0
        assert p.stop_loss.value == 1500.0


# ─────────────────────────────────────────────────────────────
# 2. JSON round-trip + hashability
# ─────────────────────────────────────────────────────────────
class TestJSONAndHashing:
    def test_json_round_trip_preserves_fields(self):
        """Prediction -> JSON -> Prediction reconstructs an EQUAL object.

        This is the contract that consumers (logs, UIs, backtest replay)
        depend on. If this breaks, every downstream layer breaks.
        """
        original = Prediction(**_valid_bullish_kwargs())
        rebuilt = Prediction.model_validate_json(original.model_dump_json())
        assert rebuilt == original

    def test_json_uses_string_enum_values(self):
        """Enums serialize as strings ('bullish'), not 'PredictionDirection.BULLISH'.

        This matters because consumers may parse the JSON in non-Python
        contexts (logs, JS UIs, jq filters). String values are stable.
        """
        p = Prediction(**_valid_bullish_kwargs())
        js = p.model_dump_json()
        assert '"direction":"bullish"' in js
        assert '"horizon":"weekly"' in js

    def test_prediction_is_hashable(self):
        """frozen=True + tuple collections => the model is hashable.

        Required for batch dedup (Step 3.4.3) using sets / dict keys.
        """
        p = Prediction(**_valid_bullish_kwargs())
        # Must not raise
        h = hash(p)
        assert h is not None
        # And usable in sets
        assert {p, p} == {p}


# ─────────────────────────────────────────────────────────────
# 3. Immutability (frozen contract)
# ─────────────────────────────────────────────────────────────
class TestImmutability:
    def test_prediction_field_assignment_rejected(self):
        """Mutating any field on a frozen Prediction raises ValidationError."""
        p = Prediction(**_valid_bullish_kwargs())
        with pytest.raises(ValidationError):
            p.confidence = 0.9  # type: ignore[misc]

    def test_price_level_is_also_frozen(self):
        """Nested PriceLevel must be frozen too (else parent loses immutability guarantees)."""
        level = PriceLevel(value=1500.0, rationale="test")
        with pytest.raises(ValidationError):
            level.value = 9999.0  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────
# 4. Field-level constraints
# ─────────────────────────────────────────────────────────────
class TestFieldConstraints:
    @pytest.mark.parametrize("bad_value", [-0.1, 1.1, 2.0, -1.0])
    def test_confidence_outside_zero_one_rejected(self, bad_value: float):
        """confidence must be in [0, 1]."""
        kw = _valid_bullish_kwargs()
        kw["confidence"] = bad_value
        with pytest.raises(ValidationError):
            Prediction(**kw)

    @pytest.mark.parametrize("boundary", [0.0, 1.0])
    def test_confidence_boundaries_accepted(self, boundary: float):
        """Both 0.0 and 1.0 are valid (we never EMIT 1.0 by convention,
        but the schema doesn't enforce that — agent prompt does)."""
        kw = _valid_bullish_kwargs()
        kw["confidence"] = boundary
        p = Prediction(**kw)
        assert p.confidence == boundary

    def test_zero_price_level_rejected(self):
        """PriceLevel.value must be > 0 (gt=0)."""
        with pytest.raises(ValidationError):
            PriceLevel(value=0.0, rationale="zero is not a price")

    def test_empty_rationale_rejected(self):
        """PriceLevel.rationale must have content (min_length=1)."""
        with pytest.raises(ValidationError):
            PriceLevel(value=100.0, rationale="")

    def test_news_sentiment_outside_range_rejected(self):
        """AnalysisBasis.news_sentiment_score must be in [-1, +1]."""
        kw = _valid_basis_kwargs()
        kw["news_sentiment_score"] = 1.5
        with pytest.raises(ValidationError):
            AnalysisBasis(**kw)


# ─────────────────────────────────────────────────────────────
# 5. Cross-field validation
# ─────────────────────────────────────────────────────────────
class TestCrossFieldValidation:
    def test_naive_datetime_rejected(self):
        """as_of must be tz-aware. Naive datetime -> ValueError."""
        kw = _valid_bullish_kwargs()
        kw["as_of"] = datetime(2026, 4, 28, 15, 30)  # no tzinfo
        with pytest.raises(ValidationError, match="tz-aware"):
            Prediction(**kw)

    def test_entry_zone_inverted_rejected(self):
        """entry_zone must be (low, high) with low <= high."""
        kw = _valid_bullish_kwargs()
        kw["entry_zone"] = (1500.0, 1450.0)
        with pytest.raises(ValidationError, match="entry_zone"):
            Prediction(**kw)

    def test_entry_zone_zero_rejected(self):
        """entry_zone values must be > 0."""
        kw = _valid_bullish_kwargs()
        kw["entry_zone"] = (0.0, 1500.0)
        with pytest.raises(ValidationError, match="entry_zone"):
            Prediction(**kw)

    def test_entry_zone_equal_low_high_allowed(self):
        """A point entry_zone (low == high) is allowed — a 'snap' entry order.

        Edge case: this happens when the agent has high conviction on a
        specific level rather than a range. Adjust target/stop to stay
        valid for the bullish direction constraints.
        """
        kw = _valid_bullish_kwargs()
        kw["entry_zone"] = (1460.0, 1460.0)
        # Existing target=1600 > 1460 ✓ and stop=1400 < 1460 ✓
        p = Prediction(**kw)
        assert p.entry_zone == (1460.0, 1460.0)

    def test_empty_model_chain_rejected(self):
        """model_chain must list at least one model."""
        kw = _valid_bullish_kwargs()
        kw["model_chain"] = ()
        with pytest.raises(ValidationError, match="model_chain"):
            Prediction(**kw)

    def test_unknown_field_rejected_loudly(self):
        """extra='forbid' catches LLM-style field-name typos at construction.

        Real-world scenario: an LLM generating Prediction JSON might emit
        `confidence_score` instead of `confidence`. Without extra='forbid',
        the typo would be silently dropped and confidence would default-
        fail-required — a confusing error far from the actual cause.

        With extra='forbid', the unknown key surfaces directly: developer
        sees 'confidence_score is not permitted' and fixes the prompt.
        """
        kw = _valid_bullish_kwargs()
        kw["confidence_score"] = 0.7  # type: ignore[typeddict-unknown-key]
        with pytest.raises(ValidationError, match="confidence_score"):
            Prediction(**kw)


# ─────────────────────────────────────────────────────────────
# 6. Direction-specific level ordering
# ─────────────────────────────────────────────────────────────
class TestDirectionalInvariants:
    def test_bullish_target_below_entry_top_rejected(self):
        """A bullish prediction with target <= entry-top makes no sense."""
        kw = _valid_bullish_kwargs()
        kw["target"] = PriceLevel(value=1465.0, rationale="bad")  # < entry top 1470
        with pytest.raises(ValidationError, match="bullish"):
            Prediction(**kw)

    def test_bullish_stop_above_entry_top_rejected(self):
        """A bullish prediction with stop above entry-top is nonsense."""
        kw = _valid_bullish_kwargs()
        kw["stop_loss"] = PriceLevel(value=1480.0, rationale="bad")  # > entry top 1470
        with pytest.raises(ValidationError, match="bullish"):
            Prediction(**kw)

    def test_bearish_target_above_entry_bottom_rejected(self):
        """Bearish needs target < entry-bottom."""
        kw = _valid_bearish_kwargs()
        # entry bottom = 1450; target must be < 1450
        kw["target"] = PriceLevel(value=1455.0, rationale="bad")
        with pytest.raises(ValidationError, match="bearish"):
            Prediction(**kw)

    def test_bearish_stop_below_entry_bottom_rejected(self):
        """Bearish needs stop > entry-bottom."""
        kw = _valid_bearish_kwargs()
        # entry bottom = 1450; stop must be > 1450
        kw["stop_loss"] = PriceLevel(value=1440.0, rationale="bad")
        with pytest.raises(ValidationError, match="bearish"):
            Prediction(**kw)

    def test_neutral_skips_level_ordering(self):
        """Neutral predictions don't enforce target/stop direction.

        Range-bound predictions are symmetric — the schema must not
        force them into a directional shape. Use bullish kw as base
        but flip direction; levels that would violate bullish ordering
        are now legal.
        """
        kw = _valid_bullish_kwargs()
        kw["direction"] = PredictionDirection.NEUTRAL
        # Levels that would FAIL bullish ordering should now pass
        kw["target"] = PriceLevel(value=1465.0, rationale="range top-ish")
        kw["stop_loss"] = PriceLevel(value=1480.0, rationale="break-out invalidation")
        p = Prediction(**kw)
        assert p.direction == PredictionDirection.NEUTRAL


# ─────────────────────────────────────────────────────────────
# 7. risk_reward computed field — worst-case math
# ─────────────────────────────────────────────────────────────
class TestRiskRewardMath:
    def test_bullish_rr_uses_entry_top_worst_case(self):
        """Bullish RR = (target - entry_top) / (entry_top - stop).

        For (1450, 1470), target=1600, stop=1400:
            (1600 - 1470) / (1470 - 1400) = 130 / 70 = 1.857...
        """
        p = Prediction(**_valid_bullish_kwargs())
        assert p.risk_reward == pytest.approx(130.0 / 70.0)

    def test_bearish_rr_uses_entry_bottom_worst_case(self):
        """Bearish RR = (entry_bottom - target) / (stop - entry_bottom).

        For (1450, 1470), target=1300, stop=1500:
            (1450 - 1300) / (1500 - 1450) = 150 / 50 = 3.0
        """
        p = Prediction(**_valid_bearish_kwargs())
        assert p.risk_reward == pytest.approx(3.0)

    def test_neutral_rr_is_one(self):
        """Neutral predictions have no directional edge -> RR = 1.0 by convention."""
        kw = _valid_bullish_kwargs()
        kw["direction"] = PredictionDirection.NEUTRAL
        # Levels can be whatever for neutral; ordering not enforced
        p = Prediction(**kw)
        assert p.risk_reward == 1.0

    def test_risk_reward_is_in_serialized_json(self):
        """As a @computed_field, risk_reward must appear in JSON output.

        Consumers (UIs, logs) shouldn't have to reconstruct the model
        just to display RR.
        """
        p = Prediction(**_valid_bullish_kwargs())
        js = p.model_dump_json()
        assert '"risk_reward"' in js

    def test_risk_reward_cannot_be_set_at_construction(self):
        """@computed_field is the SINGLE source of truth for RR.

        Mechanism: the schema's `_strip_computed_fields` mode='before'
        validator strips `risk_reward` from any input dict (kwarg- or
        JSON-path) before extra='forbid' sees it. So whatever the caller
        passes for risk_reward is silently discarded, and the property
        recomputes from target/stop/entry.

        Net guarantee: NO drift between the levels and the RR number.
        Even a malicious / buggy caller cannot inject a wrong RR.
        """
        kw = _valid_bullish_kwargs()
        kw["risk_reward"] = 999.0  # type: ignore[typeddict-unknown-key]
        p = Prediction(**kw)
        # 999 was silently dropped; computed value wins
        assert p.risk_reward == pytest.approx(130.0 / 70.0)
        assert p.risk_reward != 999.0


# ─────────────────────────────────────────────────────────────
# 8. PriceLevel + AnalysisBasis edge cases
# ─────────────────────────────────────────────────────────────
class TestNestedModels:
    def test_analysis_basis_minimal_construction(self):
        """AnalysisBasis can omit news/filings (default 0/None)."""
        b = AnalysisBasis(
            close_price_at_prediction=1455.0,
            bars_used=400,
            technical_summary="Bullish trend",
        )
        assert b.news_sentiment_score is None
        assert b.news_articles_considered == 0
        assert b.filings_considered == 0

    def test_analysis_basis_bars_used_floor(self):
        """bars_used must be >= 20 (most indicators are noise below that)."""
        kw = _valid_basis_kwargs()
        kw["bars_used"] = 19
        with pytest.raises(ValidationError):
            AnalysisBasis(**kw)

    def test_analysis_basis_is_frozen(self):
        """Required-embedded sub-model must be immutable too."""
        b = AnalysisBasis(**_valid_basis_kwargs())
        with pytest.raises(ValidationError):
            b.bars_used = 500  # type: ignore[misc]

    def test_price_level_round_trip(self):
        """Standalone PriceLevel JSON round-trip works."""
        original = PriceLevel(value=1500.5, rationale="20-day SMA")
        rebuilt = PriceLevel.model_validate_json(original.model_dump_json())
        assert rebuilt == original


# ─────────────────────────────────────────────────────────────
# 9. target_datetime computed field (added in commit 2 of multi-horizon refactor)
# ─────────────────────────────────────────────────────────────
class TestTargetDatetime:
    """target_datetime is the @computed_field that delegates to the NSE
    trading-calendar. Schema-level tests verify the integration: the
    full math contract is covered in test_trading_calendar.py.
    """

    def _kwargs_with_horizon(self, horizon: PredictionHorizon, as_of: datetime) -> dict:
        kwargs = _valid_bullish_kwargs()
        kwargs["horizon"] = horizon
        kwargs["as_of"] = as_of
        return kwargs

    def test_daily_target_is_today_close_when_predicted_mid_session(self):
        """Predicted Wed 10am IST → target = Wed 15:30 IST."""
        from zoneinfo import ZoneInfo
        ist = ZoneInfo("Asia/Kolkata")
        as_of = datetime(2026, 4, 15, 10, 0, tzinfo=ist)
        p = Prediction(**self._kwargs_with_horizon(PredictionHorizon.DAILY, as_of))
        assert p.target_datetime == datetime(2026, 4, 15, 15, 30, tzinfo=ist)

    def test_daily_target_is_next_session_when_predicted_post_close(self):
        """Predicted Wed 5pm IST → target = Thu 15:30 IST."""
        from zoneinfo import ZoneInfo
        ist = ZoneInfo("Asia/Kolkata")
        as_of = datetime(2026, 4, 15, 17, 0, tzinfo=ist)
        p = Prediction(**self._kwargs_with_horizon(PredictionHorizon.DAILY, as_of))
        assert p.target_datetime == datetime(2026, 4, 16, 15, 30, tzinfo=ist)

    def test_weekly_target_is_seven_calendar_days_later(self):
        """Predicted Thu Apr 16 → target = Thu Apr 23 (next Thu, trading)."""
        from zoneinfo import ZoneInfo
        ist = ZoneInfo("Asia/Kolkata")
        as_of = datetime(2026, 4, 16, 10, 0, tzinfo=ist)
        p = Prediction(**self._kwargs_with_horizon(PredictionHorizon.WEEKLY, as_of))
        assert p.target_datetime == datetime(2026, 4, 23, 15, 30, tzinfo=ist)

    def test_monthly_target_handles_month_end_via_relativedelta(self):
        """Predicted Mon Aug 31 → target = Wed Sep 30 (NOT Sep 31, which doesn't exist)."""
        from zoneinfo import ZoneInfo
        ist = ZoneInfo("Asia/Kolkata")
        as_of = datetime(2026, 8, 31, 10, 0, tzinfo=ist)
        p = Prediction(**self._kwargs_with_horizon(PredictionHorizon.MONTHLY, as_of))
        assert p.target_datetime == datetime(2026, 9, 30, 15, 30, tzinfo=ist)

    def test_target_datetime_serializes_into_json(self):
        """JSON dump must include target_datetime (audit trail)."""
        kwargs = _valid_bullish_kwargs()
        p = Prediction(**kwargs)
        js = p.model_dump_json()
        assert "target_datetime" in js

    def test_target_datetime_round_trips_via_json(self):
        """Round-trip through JSON works despite extra='forbid' policy.

        The _strip_computed_fields validator drops target_datetime on parse;
        the @computed_field re-derives it on read. Round-trip is byte-equal
        for the underlying inputs (horizon + as_of), and target_datetime is
        identical because it's a pure function of those inputs.
        """
        original = Prediction(**_valid_bullish_kwargs())
        rebuilt = Prediction.model_validate_json(original.model_dump_json())
        assert rebuilt == original
        assert rebuilt.target_datetime == original.target_datetime

    def test_target_datetime_consistent_across_all_horizons(self):
        """Sanity: every horizon produces a tz-aware datetime at 15:30 IST."""
        from zoneinfo import ZoneInfo
        ist = ZoneInfo("Asia/Kolkata")
        as_of = datetime(2026, 4, 15, 10, 0, tzinfo=ist)
        for h in PredictionHorizon:
            p = Prediction(**self._kwargs_with_horizon(h, as_of))
            td = p.target_datetime
            assert td.tzinfo is not None
            assert (td.hour, td.minute) == (15, 30)
