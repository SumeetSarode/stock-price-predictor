"""Tests for prediction/horizon_constants.py.

The constants are 🔬 NEEDS BACKTEST per dossier §12 (we don't assert
specific numerical values — those will change with empirical
calibration). What we DO assert is structural:

  1. Every PredictionHorizon enum value has an entry in every dict
     (regression guard against forgetting to add a new horizon).
  2. The shape of "longer horizon = wider stops/targets, lower
     confidence" is preserved across all horizons (any tuning that
     breaks this is almost certainly a typo).
  3. Lookup helpers return what the dicts contain (no clever logic
     drift).
"""
from __future__ import annotations

import pytest

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
from price_predictor.prediction.schema import PredictionHorizon

# Canonical horizon order: shortest -> longest. Used to assert the
# "longer horizon = wider tolerances" monotonicity contract. If a new
# horizon is added between WEEKLY and MONTHLY, drop it in the right
# position here AND in the constants module.
_ORDERED_HORIZONS: tuple[PredictionHorizon, ...] = (
    PredictionHorizon.DAILY,
    PredictionHorizon.WEEKLY,
    PredictionHorizon.BIWEEKLY,
    PredictionHorizon.MONTHLY,
)


# ─────────────────────────────────────────────────────────────
# Coverage: every enum value is keyed in every dict
# ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("horizon", list(PredictionHorizon))
def test_stop_atr_range_covers_horizon(horizon: PredictionHorizon) -> None:
    assert horizon in STOP_ATR_RANGE_BY_HORIZON


@pytest.mark.parametrize("horizon", list(PredictionHorizon))
def test_target_atr_range_covers_horizon(horizon: PredictionHorizon) -> None:
    assert horizon in TARGET_ATR_RANGE_BY_HORIZON


@pytest.mark.parametrize("horizon", list(PredictionHorizon))
def test_entry_zone_pct_covers_horizon(horizon: PredictionHorizon) -> None:
    assert horizon in ENTRY_ZONE_PCT_BY_HORIZON


@pytest.mark.parametrize("horizon", list(PredictionHorizon))
def test_confidence_cap_covers_horizon(horizon: PredictionHorizon) -> None:
    assert horizon in CONFIDENCE_CAP_BY_HORIZON


def test_ordered_horizons_matches_enum() -> None:
    """If we add a new horizon to the enum, this test forces us to
    decide where it goes in the ordering (rather than silently leaving
    it out of monotonicity checks)."""
    assert set(_ORDERED_HORIZONS) == set(PredictionHorizon)


# ─────────────────────────────────────────────────────────────
# Tuple sanity: each (min, max) has min <= max
# ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("horizon", list(PredictionHorizon))
def test_stop_atr_range_min_le_max(horizon: PredictionHorizon) -> None:
    lo, hi = STOP_ATR_RANGE_BY_HORIZON[horizon]
    assert lo <= hi
    assert lo > 0  # zero-distance stop is meaningless


@pytest.mark.parametrize("horizon", list(PredictionHorizon))
def test_target_atr_range_min_le_max(horizon: PredictionHorizon) -> None:
    lo, hi = TARGET_ATR_RANGE_BY_HORIZON[horizon]
    assert lo <= hi
    assert lo > 0


# ─────────────────────────────────────────────────────────────
# Monotonicity: longer horizon = wider tolerances
# ─────────────────────────────────────────────────────────────
def test_stop_range_widens_with_horizon() -> None:
    """Both min AND max stop ATR multiples grow as horizon lengthens."""
    mins = [STOP_ATR_RANGE_BY_HORIZON[h][0] for h in _ORDERED_HORIZONS]
    maxs = [STOP_ATR_RANGE_BY_HORIZON[h][1] for h in _ORDERED_HORIZONS]
    assert mins == sorted(mins), f"stop min not monotonic: {mins}"
    assert maxs == sorted(maxs), f"stop max not monotonic: {maxs}"


def test_target_range_widens_with_horizon() -> None:
    """Both min AND max target ATR multiples grow as horizon lengthens."""
    mins = [TARGET_ATR_RANGE_BY_HORIZON[h][0] for h in _ORDERED_HORIZONS]
    maxs = [TARGET_ATR_RANGE_BY_HORIZON[h][1] for h in _ORDERED_HORIZONS]
    assert mins == sorted(mins), f"target min not monotonic: {mins}"
    assert maxs == sorted(maxs), f"target max not monotonic: {maxs}"


def test_entry_zone_widens_with_horizon() -> None:
    """Entry-zone half-width grows as horizon lengthens."""
    widths = [ENTRY_ZONE_PCT_BY_HORIZON[h] for h in _ORDERED_HORIZONS]
    assert widths == sorted(widths), f"entry zone not monotonic: {widths}"


def test_confidence_cap_shrinks_with_horizon() -> None:
    """Confidence cap shrinks as horizon lengthens (more uncertainty)."""
    caps = [CONFIDENCE_CAP_BY_HORIZON[h] for h in _ORDERED_HORIZONS]
    assert caps == sorted(caps, reverse=True), f"confidence cap not monotonic: {caps}"


# ─────────────────────────────────────────────────────────────
# Sanity bounds (catch obvious typos like 90.0 instead of 0.90)
# ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("horizon", list(PredictionHorizon))
def test_entry_zone_pct_in_unit_range(horizon: PredictionHorizon) -> None:
    pct = ENTRY_ZONE_PCT_BY_HORIZON[horizon]
    assert 0.0 < pct < 0.10  # entry zone above ±10% would be absurd


@pytest.mark.parametrize("horizon", list(PredictionHorizon))
def test_confidence_cap_in_probability_range(horizon: PredictionHorizon) -> None:
    cap = CONFIDENCE_CAP_BY_HORIZON[horizon]
    assert 0.0 < cap <= 1.0
    # Cap of 1.0 means "we'd accept impossible certainty" — nope.
    assert cap < 1.0


# ─────────────────────────────────────────────────────────────
# Lookup helpers match the underlying dicts
# ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("horizon", list(PredictionHorizon))
def test_stop_atr_range_helper_matches_dict(horizon: PredictionHorizon) -> None:
    assert stop_atr_range(horizon) == STOP_ATR_RANGE_BY_HORIZON[horizon]


@pytest.mark.parametrize("horizon", list(PredictionHorizon))
def test_target_atr_range_helper_matches_dict(horizon: PredictionHorizon) -> None:
    assert target_atr_range(horizon) == TARGET_ATR_RANGE_BY_HORIZON[horizon]


@pytest.mark.parametrize("horizon", list(PredictionHorizon))
def test_entry_zone_pct_helper_matches_dict(horizon: PredictionHorizon) -> None:
    assert entry_zone_pct(horizon) == ENTRY_ZONE_PCT_BY_HORIZON[horizon]


@pytest.mark.parametrize("horizon", list(PredictionHorizon))
def test_confidence_cap_helper_matches_dict(horizon: PredictionHorizon) -> None:
    assert confidence_cap(horizon) == CONFIDENCE_CAP_BY_HORIZON[horizon]
