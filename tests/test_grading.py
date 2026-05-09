"""Tests for prediction.grading.

Coverage map:
- Each GradeOutcome value has at least one happy-path test
- Edge cases:
  * empty future_bars (INCONCLUSIVE)
  * NEUTRAL prediction (NOT_APPLICABLE + direction tolerance)
  * same-bar tie (STOP_HIT_AMBIGUOUS)
  * exactly-touched levels (>= and <= boundary semantics)
  * window slicing: passing more bars than the horizon allows
  * realized_return arithmetic
  * direction_correct for all 3 directions
- Schema-level checks via missing columns
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from price_predictor.prediction.grading import (
    NEUTRAL_TOLERANCE,
    GradeOutcome,
    GradedPrediction,
    _direction_correct,
    grade_many,
    grade_one,
    horizon_window,
)
from price_predictor.prediction.schema import (
    AnalysisBasis,
    Prediction,
    PredictionDirection,
    PredictionHorizon,
    PriceLevel,
)


# ─────────────────────────────────────────────────────────────
# Factories
# ─────────────────────────────────────────────────────────────
def _make_pred(
    direction: PredictionDirection = PredictionDirection.BULLISH,
    horizon: PredictionHorizon = PredictionHorizon.WEEKLY,
    close: float = 100.0,
    target: float = 110.0,
    stop: float = 95.0,
) -> Prediction:
    """Default 100/110/95 levels - easy mental math in assertions."""
    return Prediction(
        ticker="TEST.NS",
        as_of=datetime(2026, 4, 28, 10, 30, tzinfo=ZoneInfo("Asia/Kolkata")),
        horizon=horizon,
        model_chain=("synthesizer:test",),
        direction=direction,
        confidence=0.7,
        entry_zone=(close - 1.0, close + 1.0),
        target=PriceLevel(value=target, rationale="test target"),
        stop_loss=PriceLevel(value=stop, rationale="test stop"),
        rationale="for testing",
        contributing_signals=("test signal",),
        conflicting_signals=(),
        analysis_basis=AnalysisBasis(
            close_price_at_prediction=close,
            bars_used=400,
            technical_summary="ok",
        ),
    )


def _make_bars(
    rows: list[tuple[float, float, float]],
    start_date: str = "2026-04-29",
) -> pd.DataFrame:
    """Build a DataFrame from (high, low, close) tuples.

    Trading-day index starts the day AFTER the default prediction's
    as_of (2026-04-28). We use plain dates - tz handling isn't part of
    grade_one's contract (it just walks the rows in order).
    """
    base = datetime.fromisoformat(start_date)
    return pd.DataFrame(
        rows,
        columns=["high", "low", "close"],
        index=[base + timedelta(days=i) for i in range(len(rows))],
    )


# ─────────────────────────────────────────────────────────────
# horizon_window lookup
# ─────────────────────────────────────────────────────────────
class TestHorizonWindow:
    def test_all_horizons_have_a_window(self):
        # Every enum value must map to a positive int. Catches bugs
        # where a new horizon is added but the lookup isn't extended.
        for h in PredictionHorizon:
            assert horizon_window(h) > 0

    def test_longer_horizon_means_more_bars(self):
        # Sanity: monotonically increasing.
        assert (
            horizon_window(PredictionHorizon.DAILY)
            < horizon_window(PredictionHorizon.WEEKLY)
            < horizon_window(PredictionHorizon.BIWEEKLY)
            < horizon_window(PredictionHorizon.MONTHLY)
        )


# ─────────────────────────────────────────────────────────────
# direction_correct (pure helper)
# ─────────────────────────────────────────────────────────────
class TestDirectionCorrect:
    @pytest.mark.parametrize("ret,expected", [(0.05, True), (-0.05, False), (0.0, False)])
    def test_bullish(self, ret, expected):
        assert _direction_correct(PredictionDirection.BULLISH, ret) is expected

    @pytest.mark.parametrize("ret,expected", [(-0.05, True), (0.05, False), (0.0, False)])
    def test_bearish(self, ret, expected):
        assert _direction_correct(PredictionDirection.BEARISH, ret) is expected

    def test_neutral_within_tolerance_is_correct(self):
        assert _direction_correct(PredictionDirection.NEUTRAL, NEUTRAL_TOLERANCE / 2) is True
        assert _direction_correct(PredictionDirection.NEUTRAL, -NEUTRAL_TOLERANCE / 2) is True

    def test_neutral_outside_tolerance_is_wrong(self):
        assert _direction_correct(PredictionDirection.NEUTRAL, NEUTRAL_TOLERANCE * 2) is False
        assert _direction_correct(PredictionDirection.NEUTRAL, -NEUTRAL_TOLERANCE * 2) is False


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────
class TestValidation:
    def test_missing_columns_raises_value_error(self):
        bars = pd.DataFrame({"open": [100.0], "close": [101.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            grade_one(_make_pred(), bars)


# ─────────────────────────────────────────────────────────────
# Bullish outcomes
# ─────────────────────────────────────────────────────────────
class TestBullishOutcomes:
    def test_target_hit_on_day_3(self):
        # close=100, target=110, stop=95
        bars = _make_bars([
            (105, 99, 102),   # day 1: neither
            (108, 100, 105),  # day 2: neither
            (112, 103, 110),  # day 3: target hit (high >= 110)
            (115, 105, 113),  # day 4: should be IGNORED (already resolved)
        ])
        result = grade_one(_make_pred(), bars)
        assert result.outcome == GradeOutcome.TARGET_HIT
        assert result.days_to_resolution == 3
        assert result.bars_examined == 4  # all bars in window were 'examined'
        assert result.realized_return == pytest.approx((113 - 100) / 100)
        assert result.direction_correct is True
        assert result.hit is True  # convenience property

    def test_stop_hit_on_day_2(self):
        bars = _make_bars([
            (105, 96, 100),   # day 1: neither
            (102, 94, 95),    # day 2: stop hit (low <= 95)
            (110, 100, 108),  # day 3: target after stop = ignored
        ])
        result = grade_one(_make_pred(), bars)
        assert result.outcome == GradeOutcome.STOP_HIT
        assert result.days_to_resolution == 2
        assert result.hit is False
        # NOTE: realized_return is computed from the WINDOW close, not
        # the stop level. So even though the trade got stopped out, the
        # underlying eventually rallied (close went 100 -> 108 = +8%).
        # direction_correct reports whether the LLM's directional call
        # was vindicated by the underlying, INDEPENDENT of trade
        # execution. This separation matters: 'is the LLM directionally
        # right?' vs 'would the trade have made money?' are two
        # different questions and both deserve their own metric.
        assert result.realized_return == pytest.approx(0.08)
        assert result.direction_correct is True  # bullish + price ended up

    def test_stop_hit_with_bearish_close(self):
        # Mirror: stopped out AND price closed lower in window =
        # both metrics agree (loss + wrong direction).
        bars = _make_bars([
            (102, 94, 95),    # day 1: stop hit
            (96, 90, 92),     # subsequent days irrelevant for outcome
            (95, 89, 91),
        ])
        result = grade_one(_make_pred(), bars)
        assert result.outcome == GradeOutcome.STOP_HIT
        assert result.realized_return < 0
        assert result.direction_correct is False

    def test_expired_neither_hit(self):
        bars = _make_bars([
            (105, 96, 102),
            (108, 97, 105),
            (109, 98, 107),
            (109, 96, 108),
            (109, 97, 108),  # 5 bars (full SHORT window), neither hit
        ])
        result = grade_one(_make_pred(), bars)
        assert result.outcome == GradeOutcome.EXPIRED
        assert result.days_to_resolution is None
        assert result.bars_examined == 5
        assert result.realized_return == pytest.approx(0.08)
        assert result.direction_correct is True  # bullish + return > 0

    def test_same_bar_ambiguous_called_as_stop(self):
        # Day 1: high=112 (target), low=94 (stop) - both touched same day
        bars = _make_bars([
            (112, 94, 100),
        ])
        result = grade_one(_make_pred(), bars)
        assert result.outcome == GradeOutcome.STOP_HIT_AMBIGUOUS
        assert result.days_to_resolution == 1
        assert result.hit is False  # ambiguous = not a clean win

    def test_exact_target_touch_counts_as_hit(self):
        # high == target exactly. Boundary semantics: '>=' means
        # touching the level counts.
        bars = _make_bars([(110.0, 105.0, 108.0)])
        result = grade_one(_make_pred(), bars)
        assert result.outcome == GradeOutcome.TARGET_HIT


# ─────────────────────────────────────────────────────────────
# Bearish outcomes (mirror)
# ─────────────────────────────────────────────────────────────
class TestBearishOutcomes:
    def _bearish(self):
        # close=100, target=90 (down), stop=105 (up)
        return _make_pred(
            direction=PredictionDirection.BEARISH,
            target=90.0, stop=105.0,
        )

    def test_target_hit(self):
        bars = _make_bars([
            (102, 96, 99),
            (101, 89, 92),  # low <= 90, target hit
        ])
        result = grade_one(self._bearish(), bars)
        assert result.outcome == GradeOutcome.TARGET_HIT
        assert result.days_to_resolution == 2

    def test_stop_hit(self):
        bars = _make_bars([
            (106, 99, 105),  # high >= 105, stop hit
        ])
        result = grade_one(self._bearish(), bars)
        assert result.outcome == GradeOutcome.STOP_HIT
        assert result.days_to_resolution == 1

    def test_bearish_expired_with_correct_direction(self):
        # No hit but price drifted down -> bearish call vindicated by close
        bars = _make_bars([(101, 96, 98)] * 5)
        result = grade_one(self._bearish(), bars)
        assert result.outcome == GradeOutcome.EXPIRED
        assert result.realized_return < 0
        assert result.direction_correct is True


# ─────────────────────────────────────────────────────────────
# Neutral predictions
# ─────────────────────────────────────────────────────────────
class TestNeutralOutcomes:
    def test_neutral_always_not_applicable(self):
        # Neutral predictions don't have target/stop semantics.
        pred = _make_pred(direction=PredictionDirection.NEUTRAL)
        bars = _make_bars([
            (115, 90, 100),  # huge intraday range - irrelevant for NEUTRAL
        ])
        result = grade_one(pred, bars)
        assert result.outcome == GradeOutcome.NOT_APPLICABLE
        assert result.days_to_resolution is None

    def test_neutral_correct_when_flat(self):
        pred = _make_pred(direction=PredictionDirection.NEUTRAL)
        bars = _make_bars([(101, 99, 100.5)])  # +0.5%, well within tolerance
        result = grade_one(pred, bars)
        assert result.direction_correct is True

    def test_neutral_wrong_when_trending(self):
        pred = _make_pred(direction=PredictionDirection.NEUTRAL)
        bars = _make_bars([(108, 100, 107)])  # +7%, way outside tolerance
        result = grade_one(pred, bars)
        assert result.direction_correct is False


# ─────────────────────────────────────────────────────────────
# Inconclusive
# ─────────────────────────────────────────────────────────────
class TestInconclusive:
    def test_empty_bars_is_inconclusive(self):
        empty = _make_bars([])
        result = grade_one(_make_pred(), empty)
        assert result.outcome == GradeOutcome.INCONCLUSIVE
        assert result.realized_return == 0.0
        assert result.direction_correct is None
        assert result.days_to_resolution is None
        assert result.bars_examined == 0
        assert result.close_at_window_end is None
        assert result.hit is False


# ─────────────────────────────────────────────────────────────
# Window slicing
# ─────────────────────────────────────────────────────────────
class TestWindowSlicing:
    def test_extra_bars_beyond_window_are_ignored(self):
        # SHORT horizon = 5 bars. We pass 10. Bar 7 hits target -
        # should NOT count because it's outside the window.
        rows = [(105, 96, 102)] * 5  # 5 bars: nothing happens
        rows.extend([(115, 96, 113), (115, 96, 113), (115, 96, 113),
                     (115, 96, 113), (115, 96, 113)])  # 5 more: target hit
        bars = _make_bars(rows)
        result = grade_one(_make_pred(horizon=PredictionHorizon.WEEKLY), bars)
        assert result.outcome == GradeOutcome.EXPIRED
        assert result.bars_examined == 5  # only the in-window bars

    def test_intraday_uses_only_first_bar(self):
        bars = _make_bars([
            (105, 96, 100),   # day 1: nothing
            (115, 90, 110),   # day 2: would hit but outside intraday
        ])
        result = grade_one(_make_pred(horizon=PredictionHorizon.DAILY), bars)
        assert result.bars_examined == 1
        assert result.outcome == GradeOutcome.EXPIRED


# ─────────────────────────────────────────────────────────────
# GradedPrediction model contract
# ─────────────────────────────────────────────────────────────
class TestGradedPredictionModel:
    def test_frozen(self):
        result = grade_one(_make_pred(), _make_bars([(105, 96, 102)]))
        with pytest.raises(Exception):  # pydantic's ValidationError or AttributeError
            result.realized_return = 999.0  # type: ignore[misc]

    def test_round_trips_through_json(self):
        # Critical: a graded prediction must persist+reload as itself.
        # Catches any forgotten serializer for nested Prediction.
        result = grade_one(_make_pred(), _make_bars([(112, 96, 110)]))
        blob = result.model_dump_json()
        restored = GradedPrediction.model_validate_json(blob)
        assert restored == result


# ─────────────────────────────────────────────────────────────
# grade_many (orchestration with injected fetcher)
# ─────────────────────────────────────────────────────────────
class TestGradeMany:
    """grade_many fetches OHLCV per prediction and runs grade_one.

    All tests use a stub fetcher (zero network) so they're fast and
    deterministic. The DEFAULT fetcher (data.prices.fetch_ohlcv) is
    only swapped in by production code.
    """

    def _stub_fetcher(self, bars_by_ticker: dict):
        """Build a fake fetch_ohlcv that returns canned bars.

        Records every call into a list so tests can assert on the
        date range / ticker arguments.
        """
        calls: list[tuple] = []

        def _f(ticker, start, end):
            calls.append((ticker, start, end))
            return bars_by_ticker.get(ticker, _make_bars([]))

        _f.calls = calls  # type: ignore[attr-defined]
        return _f

    def test_grades_each_prediction_in_order(self):
        preds = [_make_pred(), _make_pred()]
        # Both predictions get target-hit data
        bars = _make_bars([(112, 99, 110)])
        fetcher = self._stub_fetcher({"TEST.NS": bars})
        results = grade_many(preds, fetch_ohlcv=fetcher, today=date(2026, 6, 1))
        assert len(results) == 2
        assert all(r.outcome == GradeOutcome.TARGET_HIT for r in results)

    def test_fetch_window_starts_day_after_prediction(self):
        # Prediction on April 28 -> fetch starts April 29 (no lookahead
        # bias from the prediction's own bar).
        preds = [_make_pred()]
        fetcher = self._stub_fetcher({"TEST.NS": _make_bars([(112, 99, 110)])})
        grade_many(preds, fetch_ohlcv=fetcher, today=date(2026, 6, 1))
        ticker, start, end = fetcher.calls[0]  # type: ignore[attr-defined]
        assert ticker == "TEST.NS"
        assert start == date(2026, 4, 29)
        # End is generously buffered past the 5-day SHORT window
        assert end > start

    def test_fetch_window_capped_at_today(self):
        # Even if horizon would extend further, we can't fetch the future.
        preds = [_make_pred(horizon=PredictionHorizon.MONTHLY)]  # 60 trading days
        fetcher = self._stub_fetcher({"TEST.NS": _make_bars([(112, 99, 110)])})
        # 'today' is only 10 days after prediction
        today = date(2026, 5, 8)
        grade_many(preds, fetch_ohlcv=fetcher, today=today)
        _, start, end = fetcher.calls[0]  # type: ignore[attr-defined]
        assert end <= today

    def test_fetcher_failure_marks_inconclusive_keeps_position(self):
        # If fetch raises, that prediction gets an INCONCLUSIVE
        # GradedPrediction so the output list stays aligned with input.
        preds = [_make_pred(), _make_pred(), _make_pred()]

        def _f(ticker, start, end):
            # fail the middle one
            if len(_f.calls) == 1:  # type: ignore[attr-defined]
                _f.calls.append(None)  # type: ignore[attr-defined]
                raise RuntimeError("network down")
            _f.calls.append(None)  # type: ignore[attr-defined]
            return _make_bars([(112, 99, 110)])

        _f.calls = []  # type: ignore[attr-defined]
        results = grade_many(preds, fetch_ohlcv=_f, today=date(2026, 6, 1))
        assert len(results) == 3
        assert results[0].outcome == GradeOutcome.TARGET_HIT
        assert results[1].outcome == GradeOutcome.INCONCLUSIVE
        assert results[2].outcome == GradeOutcome.TARGET_HIT

    def test_no_elapsed_time_marks_inconclusive(self):
        # Prediction made today. fetch_end < fetch_start -> can't fetch.
        # Should NOT call the fetcher and should mark INCONCLUSIVE.
        preds = [_make_pred()]  # default as_of = 2026-04-28
        fetcher = self._stub_fetcher({})
        # 'today' BEFORE the prediction date - degenerate but defensive
        results = grade_many(preds, fetch_ohlcv=fetcher, today=date(2026, 4, 28))
        assert results[0].outcome == GradeOutcome.INCONCLUSIVE
        # fetcher should not have been called
        assert fetcher.calls == []  # type: ignore[attr-defined]

    def test_empty_input_returns_empty(self):
        fetcher = self._stub_fetcher({})
        assert grade_many([], fetch_ohlcv=fetcher) == []
