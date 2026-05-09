"""Unit tests for predict_many() batch predictor (Step 3.4.3 commit 1).

Tests verify:
  - Happy path returns Predictions
  - Partial failures return BatchError, don't kill the batch
  - Concurrency cap is respected (using a counting mock)
  - Input validation (empty list, bad concurrency)
  - Deduplication preserves first-occurrence order
  - Output order matches input order

NO real LLM calls — predict() is mocked at the boundary.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pytest

from price_predictor.prediction import (
    BatchError,
    Prediction,
    PredictionError,
    predict_many,
)
from price_predictor.prediction.schema import (
    AnalysisBasis,
    PredictionDirection,
    PredictionHorizon,
    PriceLevel,
)


# ─────────────────────────────────────────────────────────────
# Test factory — minimal valid Prediction. Local to avoid
# cross-test-file imports (each test file should be self-contained).
# ─────────────────────────────────────────────────────────────
def _make_pred(ticker: str = "RELIANCE.NS") -> Prediction:
    return Prediction(
        ticker=ticker,
        as_of=datetime(2026, 4, 28, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
        horizon=PredictionHorizon.WEEKLY,
        model_chain=("news_impact:agentic", "synthesizer:agentic"),
        direction=PredictionDirection.BULLISH,
        confidence=0.7,
        entry_zone=(1453.0, 1457.0),
        target=PriceLevel(value=1500.0, rationale="swing high"),
        stop_loss=PriceLevel(value=1425.0, rationale="below recent low"),
        rationale="Bullish setup across multiple timeframes.",
        contributing_signals=("trend bullish",),
        conflicting_signals=(),
        analysis_basis=AnalysisBasis(
            close_price_at_prediction=1455.0,
            bars_used=400,
            technical_summary="trend bullish, momentum bullish",
            news_sentiment_score=0.6,
            news_articles_considered=5,
            filings_considered=0,
        ),
    )


def _ok_dict(ticker: str, horizons: list, *_a, **_k) -> dict:
    """Mock side_effect: predict() now returns dict[Horizon, Prediction].

    batch.py wraps the caller's `horizon` arg in a single-element list
    before calling predict(), so mocks need to key the returned dict
    on horizons[0] to match what batch.py will look up.
    """
    return {horizons[0]: _make_pred(ticker)}


# ─────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────
class TestInputValidation:
    def test_empty_tickers_raises(self):
        with pytest.raises(ValueError, match="empty"):
            asyncio.run(predict_many([]))

    def test_zero_concurrency_raises(self):
        with pytest.raises(ValueError, match="concurrency"):
            asyncio.run(predict_many(["A"], concurrency=0))

    def test_negative_concurrency_raises(self):
        with pytest.raises(ValueError, match="concurrency"):
            asyncio.run(predict_many(["A"], concurrency=-1))


# ─────────────────────────────────────────────────────────────
# Happy path + result shape
# ─────────────────────────────────────────────────────────────
class TestHappyPath:
    @patch("price_predictor.prediction.batch.predict",
           new_callable=AsyncMock)
    def test_all_succeed_returns_predictions(self, mock_predict):
        # Each call returns a Prediction tagged with the ticker it
        # received, so we can assert order. Mock returns a dict because
        # batch.py now unwraps predict()'s dict[Horizon, Prediction].
        mock_predict.side_effect = _ok_dict

        results = asyncio.run(predict_many(["A", "B", "C"]))

        assert len(results) == 3
        assert all(isinstance(r, Prediction) for r in results)
        assert [r.ticker for r in results] == ["A", "B", "C"]

    @patch("price_predictor.prediction.batch.predict",
           new_callable=AsyncMock)
    def test_horizon_and_sensitivity_forwarded(self, mock_predict):
        mock_predict.return_value = {PredictionHorizon.BIWEEKLY: _make_pred()}
        asyncio.run(predict_many(
            ["A"], horizon="biweekly", sensitivity="sensitive",
        ))
        # Verify predict() got the right kwargs. batch.py now wraps the
        # caller's horizon string into a [PredictionHorizon enum] list
        # before forwarding.
        call = mock_predict.call_args_list[0]
        assert call.args[0] == "A"
        assert call.args[1] == [PredictionHorizon.BIWEEKLY]
        assert call.kwargs["sensitivity"] == "sensitive"


# ─────────────────────────────────────────────────────────────
# Failure tolerance
# ─────────────────────────────────────────────────────────────
class TestPartialFailure:
    @patch("price_predictor.prediction.batch.predict",
           new_callable=AsyncMock)
    def test_one_failure_other_succeed(self, mock_predict):
        # Middle ticker fails, others succeed.
        def _side(ticker, horizons, *a, **k):
            if ticker == "BAD":
                raise PredictionError(f"{ticker} blew up")
            return {horizons[0]: _make_pred(ticker)}
        mock_predict.side_effect = _side

        results = asyncio.run(predict_many(["A", "BAD", "C"]))

        assert len(results) == 3
        assert isinstance(results[0], Prediction) and results[0].ticker == "A"
        assert isinstance(results[1], BatchError)
        assert results[1].ticker == "BAD"
        assert results[1].error_type == "PredictionError"
        assert "blew up" in str(results[1].error)
        assert isinstance(results[2], Prediction) and results[2].ticker == "C"

    @patch("price_predictor.prediction.batch.predict",
           new_callable=AsyncMock)
    def test_all_fail_returns_all_batch_errors(self, mock_predict):
        mock_predict.side_effect = RuntimeError("everything broken")
        results = asyncio.run(predict_many(["A", "B"]))
        assert len(results) == 2
        assert all(isinstance(r, BatchError) for r in results)
        assert all(r.error_type == "RuntimeError" for r in results)

    @patch("price_predictor.prediction.batch.predict",
           new_callable=AsyncMock)
    def test_batch_error_preserves_original_exception(self, mock_predict):
        # Caller should be able to re-raise the original.
        original = ValueError("specific message")
        mock_predict.side_effect = original
        results = asyncio.run(predict_many(["A"]))
        assert isinstance(results[0], BatchError)
        # Same exception object — not a copy or wrapper
        assert results[0].error is original


# ─────────────────────────────────────────────────────────────
# Concurrency cap
# ─────────────────────────────────────────────────────────────
class TestConcurrency:
    @patch("price_predictor.prediction.batch.predict",
           new_callable=AsyncMock)
    def test_respects_concurrency_cap(self, mock_predict):
        """Use a shared counter to verify max in-flight never exceeds cap."""
        in_flight = 0
        max_seen = 0

        async def _slow_predict(ticker, horizons, *a, **k):
            nonlocal in_flight, max_seen
            in_flight += 1
            max_seen = max(max_seen, in_flight)
            # Yield to scheduler so other tasks can also enter
            await asyncio.sleep(0.01)
            in_flight -= 1
            return {horizons[0]: _make_pred(ticker)}
        mock_predict.side_effect = _slow_predict

        # 10 tickers with cap=2 - max in-flight should never exceed 2.
        asyncio.run(predict_many(
            [f"T{i}" for i in range(10)], concurrency=2,
        ))
        assert max_seen == 2, f"expected 2 concurrent, saw {max_seen}"

    @patch("price_predictor.prediction.batch.predict",
           new_callable=AsyncMock)
    def test_concurrency_higher_than_tickers_is_fine(self, mock_predict):
        # cap=10 with 3 tickers - just runs all in parallel
        mock_predict.side_effect = _ok_dict
        results = asyncio.run(predict_many(["A", "B", "C"], concurrency=10))
        assert len(results) == 3


# ─────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────
class TestDeduplication:
    @patch("price_predictor.prediction.batch.predict",
           new_callable=AsyncMock)
    def test_duplicates_called_once(self, mock_predict):
        mock_predict.side_effect = _ok_dict
        results = asyncio.run(predict_many(["A", "B", "A", "B", "A"]))
        assert len(results) == 2
        assert mock_predict.await_count == 2
        assert [r.ticker for r in results] == ["A", "B"]

    @patch("price_predictor.prediction.batch.predict",
           new_callable=AsyncMock)
    def test_dedup_preserves_first_occurrence_order(self, mock_predict):
        mock_predict.side_effect = _ok_dict
        # 'C' appears first at index 1, before 'A' at index 2.
        results = asyncio.run(predict_many(["B", "C", "A", "C", "B"]))
        assert [r.ticker for r in results] == ["B", "C", "A"]
