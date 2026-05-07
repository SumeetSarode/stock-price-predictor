"""Prediction layer — output contract + per-stock predictor.

Public API:
- `predict(ticker, horizon)`           : one-stock async predictor (commit 4)
- `Prediction` + sub-models            : output schema (commit 0)
- `compose_technical_view`             : technical gathering (commit 1)
- `SynthesisInput`                     : gather/synth contract (commit 2)
- `PredictionError`, `TechnicalViewError`: error types

The `runner` module (Runner/SessionService singletons) is intentionally
NOT re-exported here - it's predictor-internal plumbing. Tests that
need to mock it should import from prediction.runner directly.
"""
from price_predictor.prediction.inputs import (
    ClusterView,
    SynthesisInput,
    TechnicalView,
    TechnicalViewError,
    compose_technical_view,
)
from price_predictor.prediction.predictor import (
    PredictionError,
    predict,
    run_news_impact_agent,
    run_synthesizer_agent,
)
from price_predictor.prediction.schema import (
    AnalysisBasis,
    Prediction,
    PredictionDirection,
    PredictionHorizon,
    PriceLevel,
)

__all__ = [
    # schema (3.4.1)
    "AnalysisBasis",
    "Prediction",
    "PredictionDirection",
    "PredictionHorizon",
    "PriceLevel",
    # inputs (3.4.2 commits 1-2)
    "ClusterView",
    "SynthesisInput",
    "TechnicalView",
    "TechnicalViewError",
    "compose_technical_view",
    # orchestrator (3.4.2 commit 4)
    "PredictionError",
    "predict",
    "run_news_impact_agent",
    "run_synthesizer_agent",
]
