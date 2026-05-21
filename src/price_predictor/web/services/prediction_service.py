"""Service adapter: bridges the web layer to the core prediction logic.

This is the ONLY place in the web layer that imports from
`price_predictor.prediction`. Keeps the FastAPI route handlers thin
and focused on HTTP concerns.

Responsibilities:
  - Validate + normalize user input (ticker case, horizon enum coercion)
  - Call into the async `predict()` orchestrator
  - Translate the Pydantic Prediction model into a flat dict the
    template can render directly (no Jinja knowing about Enum, datetime,
    or PriceLevel — separation of concerns)
  - Convert core exceptions into structured (message, hint) tuples
    that the error template can render
"""
from __future__ import annotations

from typing import Any

from price_predictor.prediction import (
    Prediction,
    PredictionDirection,
    PredictionError,
    PredictionHorizon,
    predict,
)
from price_predictor.web.services import prediction_cache


# ── Mapping helpers ──────────────────────────────────────────────────
# Keep these tiny + pure so they're trivially testable in isolation.

_DIRECTION_CLASS: dict[PredictionDirection, str] = {
    PredictionDirection.BULLISH: "bullish",
    PredictionDirection.BEARISH: "bearish",
    PredictionDirection.NEUTRAL: "neutral",
}

# When the target is bullish, the stop sits on the bearish side (and
# vice-versa). Used for color-coding the stop-loss stat card opposite
# to the direction.
_OPPOSITE_CLASS: dict[PredictionDirection, str] = {
    PredictionDirection.BULLISH: "bearish",
    PredictionDirection.BEARISH: "bullish",
    PredictionDirection.NEUTRAL: "neutral",
}


def _to_view_dict(p: Prediction) -> dict[str, Any]:
    """Flatten a Prediction into a dict the template understands.

    The template never imports from `prediction.schema` — it only sees
    primitives (str, float, list[str]) and a couple of derived fields.
    Keeps the template dumb and the contract explicit.
    """
    return {
        "ticker": p.ticker,
        "horizon": p.horizon.value,
        "direction": p.direction.value,
        "direction_class": _DIRECTION_CLASS[p.direction],
        "opposite_class": _OPPOSITE_CLASS[p.direction],
        "confidence_pct": round(p.confidence * 100),
        "as_of_str": p.as_of.strftime("%d %b %Y %H:%M IST"),
        # Levels
        "close_price": p.analysis_basis.close_price_at_prediction,
        "entry_low": p.entry_zone[0],
        "entry_high": p.entry_zone[1],
        "target_value": p.target.value,
        "target_rationale": p.target.rationale,
        "stop_value": p.stop_loss.value,
        "stop_rationale": p.stop_loss.rationale,
        "risk_reward": p.risk_reward,
        # Reasoning
        "rationale": p.rationale,
        "contributing": list(p.contributing_signals),
        "conflicting": list(p.conflicting_signals),
        # Provenance
        "model_chain": ", ".join(p.model_chain),
        "technical_summary": p.analysis_basis.technical_summary,
    }


# ── Public service API ───────────────────────────────────────────────


class PredictionServiceError(Exception):
    """Raised when the prediction service can't fulfil a request.

    Carries a user-facing ``message`` and an optional ``hint`` so the
    UI can render a friendly error without leaking stack traces.
    """

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


async def run_prediction(ticker: str, horizon: str) -> dict[str, Any]:
    """Run one prediction and return a render-ready view dict.

    Args:
        ticker: User input — any case, with or without ``.NS`` suffix.
            Normalization happens inside the core ``predict()`` via the KB.
        horizon: One of ``daily`` / ``weekly`` / ``biweekly`` / ``monthly``.

    Returns:
        Dict suitable for `prediction_card.html` template rendering.

    Raises:
        PredictionServiceError: with a user-facing message + hint on any
            failure (bad ticker, missing API key, LLM timeout, etc.).
    """
    # Input validation — fail fast, fail friendly.
    if not ticker or not ticker.strip():
        raise PredictionServiceError(
            "Ticker cannot be empty.",
            hint="Try entering something like 'RELIANCE' or 'TCS.NS'.",
        )

    try:
        horizon_enum = PredictionHorizon(horizon.lower().strip())
    except ValueError as exc:
        raise PredictionServiceError(
            f"Unknown horizon '{horizon}'.",
            hint="Pick one of: daily, weekly, biweekly, monthly.",
        ) from exc

    # Call into the core orchestrator. It's async, may take ~30s.
    try:
        results = await predict(ticker.strip(), horizons=[horizon_enum])
    except PredictionError as exc:
        raise PredictionServiceError(
            str(exc),
            hint=_hint_for_error(exc),
        ) from exc
    except Exception as exc:
        # Defensive — anything we didn't anticipate. Don't leak the
        # repr; just say something went wrong with a generic hint.
        raise PredictionServiceError(
            "An unexpected error occurred while running the prediction.",
            hint="Check the server logs for details.",
        ) from exc

    if horizon_enum not in results:
        raise PredictionServiceError(
            "Prediction completed but no result for the requested horizon was returned.",
            hint="This is unusual — please report on GitHub if it persists.",
        )

    view = _to_view_dict(results[horizon_enum])

    # Persist to the cache so the watchlist panel can render this
    # prediction instantly on subsequent loads. Failure to cache is
    # NOT fatal — we'd rather show the user their result than blow
    # up over a disk write.
    try:
        prediction_cache.save(view)
    except Exception:
        # Logged at DEBUG only; cache failures shouldn't pollute the
        # main log stream. The user still sees their prediction.
        from loguru import logger as _logger
        _logger.opt(exception=True).debug(
            "prediction_cache.save() failed for {} {} — ignoring",
            view.get("ticker"), view.get("horizon"),
        )

    return view


def _hint_for_error(exc: PredictionError) -> str | None:
    """Map common PredictionError messages to actionable user hints."""
    msg = str(exc).lower()
    if "api key" in msg or "credentials" in msg or "unauthorized" in msg:
        return "Make sure GEMINI_API_KEY and GROQ_API_KEY are set in your .env file."
    if "ticker" in msg or "kb" in msg or "unknown" in msg:
        return "Double-check the ticker symbol. Try the full form like 'RELIANCE.NS'."
    if "rate" in msg or "quota" in msg:
        return "You may have hit your LLM provider's daily quota. Wait a bit and try again."
    if "network" in msg or "timeout" in msg or "connection" in msg:
        return "Check your internet connection and try again."
    return None
