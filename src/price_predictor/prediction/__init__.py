"""Prediction layer — output contract + per-stock predictor.

Currently exposes:
- The output schema (Step 3.4.1)
- TechnicalView + compose_technical_view (Step 3.4.2 commit 1)

Predictor orchestrator and synthesizer agent land in subsequent commits.
"""
from price_predictor.prediction.inputs import (
    ClusterView,
    TechnicalView,
    TechnicalViewError,
    compose_technical_view,
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
    # inputs (3.4.2 commit 1)
    "ClusterView",
    "TechnicalView",
    "TechnicalViewError",
    "compose_technical_view",
]
