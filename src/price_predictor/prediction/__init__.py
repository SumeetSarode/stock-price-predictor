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
from price_predictor.prediction.batch import (
    BatchError,
    BatchResult,
    predict_many,
)
from price_predictor.prediction.inputs import (
    ClusterView,
    SynthesisInput,
    TechnicalView,
    TechnicalViewError,
    compose_technical_view,
)
from price_predictor.prediction.guardrails import (
    HallucinationError,
    validate_all,
    validate_calibration,
    validate_citations,
    validate_consistency,
    validate_grounding,
)
from price_predictor.prediction.horizon_constants import (
    CONFIDENCE_CAP_BY_HORIZON,
    ENTRY_ZONE_PCT_BY_HORIZON,
    STOP_ATR_RANGE_BY_HORIZON,
    TARGET_ATR_RANGE_BY_HORIZON,
    confidence_cap,
    entry_zone_pct,
    stop_atr_range,
    target_atr_range,
)
from price_predictor.prediction.predictor import (
    PredictionError,
    predict,
    run_news_impact_agent,
    run_synthesizer_agent,
    synthesize_with_guardrails,
)
from price_predictor.prediction.store import (
    PredictionStore,
    PredictionStoreError,
)
from price_predictor.prediction.grading import (
    GradeOutcome,
    GradedPrediction,
    grade_many,
    grade_one,
    horizon_window,
)
from price_predictor.prediction.calibration import (
    CalibrationReport,
    compute_breakdown,
    compute_calibration,
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
    # orchestrator + guardrails (3.4.2 commits 4-5)
    "PredictionError",
    "HallucinationError",
    "predict",
    "run_news_impact_agent",
    "run_synthesizer_agent",
    "synthesize_with_guardrails",
    "validate_all",
    "validate_calibration",
    "validate_citations",
    "validate_consistency",
    "validate_grounding",
    # horizon constants (multi-horizon refactor commit A)
    "CONFIDENCE_CAP_BY_HORIZON",
    "ENTRY_ZONE_PCT_BY_HORIZON",
    "STOP_ATR_RANGE_BY_HORIZON",
    "TARGET_ATR_RANGE_BY_HORIZON",
    "confidence_cap",
    "entry_zone_pct",
    "stop_atr_range",
    "target_atr_range",
    # batch (3.4.3 commit 1)
    "BatchError",
    "BatchResult",
    "predict_many",
    # store (3.4.3 commit 2)
    "PredictionStore",
    "PredictionStoreError",
    # grading (3.5 commit 1)
    "GradeOutcome",
    "GradedPrediction",
    "grade_one",
    "grade_many",
    "horizon_window",
    # calibration (3.5 commit 2)
    "CalibrationReport",
    "compute_breakdown",
    "compute_calibration",
]
