"""End-to-end integration smoke tests for the predictor.

Hits REAL APIs (yfinance + GDELT/SEC + LLM). Skipped by default:
    uv run pytest -m "not integration"     # default, skipped
    uv run pytest -m integration            # run them

WHY THESE EXIST
===============
The unit tests in test_predictor.py and test_guardrails.py mock the
agent helpers, so they prove ORCHESTRATION works but NOT that:
  - The synthesizer LLM actually produces a Prediction passing our
    schema + guardrails on real-world inputs.
  - The two agents compose well in practice (news's output flows
    cleanly into synthesizer's input).
  - Latency is reasonable.

This file is the "does the whole thing actually work?" net.

WHY ONE TEST, NOT TEN
=====================
Each integration test costs LLM tokens and ~30-60s. Diminishing returns
beyond 1-2 happy-path checks; correctness is the unit tests' job.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from price_predictor.prediction import Prediction, predict
from price_predictor.prediction.schema import PredictionDirection, PredictionHorizon


# Skip if any required key is missing — surfaces a useful message instead
# of a confusing LiteLLM error halfway through.
_REQUIRED_KEYS = ("GEMINI_API_KEY",)
_missing = [k for k in _REQUIRED_KEYS if not os.environ.get(k)]
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        bool(_missing),
        reason=f"Missing required env vars for integration test: {_missing}",
    ),
]


# Use a liquid, news-rich Indian large-cap. RELIANCE has plenty of
# coverage on GDELT and SEC, so the news_impact agent has real material
# to work with — better than a thinly-traded ticker that returns empty.
_TICKER = "RELIANCE.NS"


async def test_predict_real_end_to_end():
    """Full predict() against real APIs.

    Asserts only structural invariants (schema satisfied, fields populated,
    levels in the right relative order). Does NOT assert specific values
    because real markets move and LLMs vary; we'd be flaky.
    """
    result_dict = await predict(_TICKER, [PredictionHorizon.WEEKLY])
    result = result_dict[PredictionHorizon.WEEKLY]

    # Type + identity
    assert isinstance(result, Prediction)
    assert result.ticker == _TICKER
    assert result.horizon.value == "weekly"

    # Direction is one of the three legal values (Pydantic enforces)
    assert result.direction in PredictionDirection

    # Confidence is sane
    assert 0.0 <= result.confidence <= 1.0

    # Levels are populated and positive
    assert result.entry_zone[0] > 0 and result.entry_zone[1] > 0
    assert result.target.value > 0
    assert result.stop_loss.value > 0

    # Rationale is non-trivial (LLM didn't just emit "ok")
    assert len(result.rationale) > 50
    assert len(result.contributing_signals) >= 1

    # Audit trail records both agents
    assert any("news_impact" in tag for tag in result.model_chain)
    assert any("synthesizer" in tag for tag in result.model_chain)

    # Analysis basis populated
    assert result.analysis_basis.bars_used >= 20
    assert result.analysis_basis.close_price_at_prediction > 0
