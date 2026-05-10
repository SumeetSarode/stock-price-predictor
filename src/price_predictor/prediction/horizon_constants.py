"""Per-horizon tunable constants — single source of truth.

Why this module exists
======================
The synthesizer prompt and the guardrails BOTH need to know what
"sensible" looks like per horizon (e.g. "stops should be 0.5-1.0×ATR
for daily but 1.5-2.5×ATR for monthly"). Before this module those
numbers lived in two places (or, worse, in only one — the guardrails —
while the prompt waved its hands with "tighter for daily, wider for
monthly"). That guarantees drift the moment we tune one side.

This module is the ONE place these numbers live. Guardrails import them
(Commit B). The synthesizer prompt builder injects them (Commit C).
Tune in one place, both layers stay coherent.

What lives here
===============
Four families of per-horizon knobs:

  STOP_ATR_RANGE_BY_HORIZON       — (min, max) ATR multiples for stops
  TARGET_ATR_RANGE_BY_HORIZON     — (min, max) ATR multiples for targets
  ENTRY_ZONE_PCT_BY_HORIZON       — half-width of entry zone, as % of close
  CONFIDENCE_CAP_BY_HORIZON       — maximum allowed confidence per horizon

Plus four public lookup helpers (`stop_atr_range`, `target_atr_range`,
`entry_zone_pct`, `confidence_cap`) so callers don't dict-poke. If we
ever swap these dicts for interpolation (custom-day horizons), the
function bodies change but the call sites don't.

Status of the numbers
=====================
All values are 🔬 NEEDS BACKTEST per `docs/research/constants_dossier.md`
§12. They are *literature-bracketed* — the citations below show that
each value falls inside the published-author range — but the EXACT
per-horizon picks are pending empirical calibration on NIFTY 50
historical data.

Citations live next to each dict below.
"""
from __future__ import annotations

from price_predictor.prediction.schema import PredictionHorizon

# ─────────────────────────────────────────────────────────────
# Stop-loss ATR multipliers per horizon
# ─────────────────────────────────────────────────────────────
# Sources (see dossier §12.1):
#   - Wilder (1978), "New Concepts in Technical Trading Systems":
#     1×ATR is the canonical swing-trading stop. Sits in our WEEKLY band.
#   - Van Tharp, "Trade Your Way to Financial Freedom" (2007):
#     2-3×ATR for longer-horizon swing/positional. Sits in our MONTHLY band.
#   - Murphy, "Technical Analysis of the Financial Markets" (1999):
#     Tighter (sub-1×ATR) stops appropriate for short-horizon trades.
#
# Picks (literature-bracketed, exact per-horizon values pending backtest):
#   DAILY    — tight, intraday-noise tolerance band
#   WEEKLY   — Wilder's 1×ATR sits in middle of band
#   BIWEEKLY — interpolated
#   MONTHLY  — Van Tharp's longer-horizon swing standard
STOP_ATR_RANGE_BY_HORIZON: dict[PredictionHorizon, tuple[float, float]] = {
    PredictionHorizon.DAILY:    (0.5, 1.0),
    PredictionHorizon.WEEKLY:   (0.7, 1.5),
    PredictionHorizon.BIWEEKLY: (1.0, 2.0),
    PredictionHorizon.MONTHLY:  (1.5, 2.5),
}

# ─────────────────────────────────────────────────────────────
# Target ATR multipliers per horizon
# ─────────────────────────────────────────────────────────────
# Sources (see dossier §12.2):
#   - Murphy, Pring, Tharp consensus: target = 1.3-1.5× the stop
#     distance for a positive expectancy trade (R:R > 1).
#
# Picks: per-horizon ATR target band, scaled with stop band such that
# the midpoint of (target_min, target_max) ≈ 1.4× midpoint of
# (stop_min, stop_max). Pending backtest.
TARGET_ATR_RANGE_BY_HORIZON: dict[PredictionHorizon, tuple[float, float]] = {
    PredictionHorizon.DAILY:    (0.75, 1.5),
    PredictionHorizon.WEEKLY:   (1.0,  2.0),
    PredictionHorizon.BIWEEKLY: (1.5,  3.0),
    PredictionHorizon.MONTHLY:  (2.0,  4.0),
}

# ─────────────────────────────────────────────────────────────
# Entry zone half-width, as fraction of close price
# ─────────────────────────────────────────────────────────────
# Source: pure design (no literature). Wider for longer horizons because
# the LLM has more leeway in entry timing for a monthly call than a
# daily call. See dossier §12.3.
#
# Interpretation: entry_zone = [close * (1 - pct), close * (1 + pct)]
# i.e. a ±pct band around the close price.
ENTRY_ZONE_PCT_BY_HORIZON: dict[PredictionHorizon, float] = {
    PredictionHorizon.DAILY:    0.005,   # ±0.5%
    PredictionHorizon.WEEKLY:   0.010,   # ±1.0%
    PredictionHorizon.BIWEEKLY: 0.015,   # ±1.5%
    PredictionHorizon.MONTHLY:  0.020,   # ±2.0%
}

# ─────────────────────────────────────────────────────────────
# Confidence cap per horizon
# ─────────────────────────────────────────────────────────────
# Source: pure design (no literature). See dossier §12.4. Longer
# horizons are inherently more uncertain (more events between
# prediction and target date) — the LLM should not be allowed to
# claim 0.95 confidence on a monthly call.
#
# Recalibrate from realized hit-rate per horizon once we accumulate
# enough graded predictions (Phase 2).
CONFIDENCE_CAP_BY_HORIZON: dict[PredictionHorizon, float] = {
    PredictionHorizon.DAILY:    0.90,
    PredictionHorizon.WEEKLY:   0.85,
    PredictionHorizon.BIWEEKLY: 0.80,
    PredictionHorizon.MONTHLY:  0.75,
}


# ─────────────────────────────────────────────────────────────
# Public lookup helpers
# ─────────────────────────────────────────────────────────────
# Why wrappers around dict access?
#   1. Single import path: `from prediction import stop_atr_range` is
#      nicer than poking at module-level dicts.
#   2. If we ever swap a dict for interpolation (e.g. for arbitrary-day
#      horizons — currently REJECTED per schema.py docstring), only
#      the function body changes, not every call site.
#   3. Errors raised here are uniform and descriptive.
def stop_atr_range(horizon: PredictionHorizon) -> tuple[float, float]:
    """Return (min, max) ATR multiplier for stop placement at this horizon."""
    return STOP_ATR_RANGE_BY_HORIZON[horizon]


def target_atr_range(horizon: PredictionHorizon) -> tuple[float, float]:
    """Return (min, max) ATR multiplier for target placement at this horizon."""
    return TARGET_ATR_RANGE_BY_HORIZON[horizon]


def entry_zone_pct(horizon: PredictionHorizon) -> float:
    """Return half-width of entry zone (as fraction of close) at this horizon."""
    return ENTRY_ZONE_PCT_BY_HORIZON[horizon]


def confidence_cap(horizon: PredictionHorizon) -> float:
    """Return maximum allowed confidence at this horizon."""
    return CONFIDENCE_CAP_BY_HORIZON[horizon]
