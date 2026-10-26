"""Price levels are the app's job, not the LLM's.

WHY THIS MODULE EXISTS
======================
The synthesizer used to emit the entry zone, stop and target itself, and the Tier-1 grounding guardrail then
re-checked that arithmetic (entry within a % of close, stop N x ATR away, target on an anchor). Measured on
170 refused predictions from four free models, 77% were refused for exactly that arithmetic, not for the call:
a model could read the market correctly and still be thrown away because its stop was 0.28 x ATR instead of
0.5. Each refusal also burned a retry (3 LLM calls per case) and free-tier quota.

The levels are a pure function of close, ATR and the per-horizon constants, so the app computes them. The model
keeps what only a model can do: direction, confidence, rationale, and the cited signals.

WHAT IS DERIVED (the middle of each per-horizon band the guardrail already accepts)
===================================================================================
  entry zone : close +/- the horizon's entry half-width (narrowed if the target is too close to leave room)
  stop       : close -/+ the midpoint of the horizon's stop ATR band   (below close if bullish or neutral)
  target     : close +/- the midpoint of the horizon's target ATR band (neutral: close itself)

Tier-1 grounding stays in `validate_all` as a safety net; derived levels pass it by construction.
"""
from __future__ import annotations

from price_predictor.prediction.guardrails import HallucinationError, atr_from_input
from price_predictor.prediction.horizon_constants import (
    entry_zone_pct,
    stop_atr_range,
    target_atr_range,
)
from price_predictor.prediction.inputs import SynthesisInput
from price_predictor.prediction.schema import Prediction, PredictionDirection

_PRICE_DECIMALS = 2


def derive_levels(pred: Prediction, si: SynthesisInput) -> Prediction:
    """Return `pred` with entry zone, stop and target recomputed from close and ATR; everything else is kept."""
    close, atr, horizon = si.technical_view.close_price, atr_from_input(si), pred.horizon
    sign = {PredictionDirection.BULLISH: 1, PredictionDirection.BEARISH: -1, PredictionDirection.NEUTRAL: 0}[pred.direction]

    stop_k = sum(stop_atr_range(horizon)) / 2
    target_k = sum(target_atr_range(horizon)) / 2
    stop_dist = stop_k * atr
    target_dist = target_k * atr * abs(sign)                       # neutral: the target is the close
    half = close * entry_zone_pct(horizon)
    if sign:
        half = min(half, target_dist / 2)                          # the schema needs the target beyond the zone

    side = sign or 1                                               # neutral stops sit below, like bullish
    stop, target = close - side * stop_dist, close + sign * target_dist
    target_why = (f"{horizon.value} target: close {'+' if sign > 0 else '-'} {target_k:g} x ATR({atr:.2f})"
                  if sign else "neutral: no directional target, the close itself")
    try:
        return Prediction.model_validate({
            **pred.model_dump(),
            "entry_zone": (round(close - half, _PRICE_DECIMALS), round(close + half, _PRICE_DECIMALS)),
            "stop_loss": {"value": round(stop, _PRICE_DECIMALS),
                          "rationale": f"{horizon.value} stop: close {'+' if side < 0 else '-'} {stop_k:g} x ATR({atr:.2f})"},
            "target": {"value": round(target, _PRICE_DECIMALS), "rationale": target_why},
        })
    except ValueError as e:                                        # e.g. a near-zero ATR collapses the levels onto the close
        raise HallucinationError("grounding", f"levels derived from close={close:.2f}, ATR={atr:.4f} are not a valid "
                                              f"{pred.direction.value} setup: {e}") from e
