"""Unit tests for backtest.insights -- rule-based exec summary generation.

WHAT WE TEST
============
- Each individual insight fires under the right conditions (or stays
  silent / returns None when it shouldn't fire).
- generate_insights() orders by severity (CRITICAL first, POSITIVE last).
- Edge cases: empty evaluation, single-ticker / single-horizon runs.
- Sample-size guard: under-sampled axes don't generate noisy claims.

WHAT WE DON'T TEST
==================
- Exact wording of every detail string (too brittle; we check headline
  patterns + the level).
- HTML rendering of insights (covered in test_backtest_html_report).
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pytest

from price_predictor.backtest.evaluation import BacktestEvaluation
from price_predictor.backtest.insights import (
    Insight,
    InsightLevel,
    generate_insights,
)
from price_predictor.backtest.runner import BacktestRun
from price_predictor.prediction.calibration import CalibrationReport
from price_predictor.prediction.schema import (
    PredictionDirection,
    PredictionHorizon,
)


# ─────────────────────────────────────────────────────────────
# Builders -- compose test fixtures cheaply
# ─────────────────────────────────────────────────────────────
def _report(
    *,
    n_predictions: int = 50,
    direction_accuracy: float = 0.6,
    n_judged: int = 50,
    brier_skill_score: float | None = 0.10,
    hit_rate_resolved: float = 0.55,
    n_target_hit: int = 10,
    n_stop_hit: int = 8,
    n_stop_hit_ambiguous: int = 0,
) -> CalibrationReport:
    """Quick CalibrationReport builder with sensible defaults."""
    return CalibrationReport(
        n_predictions=n_predictions,
        n_inconclusive=max(0, n_predictions - n_judged),
        n_target_hit=n_target_hit,
        n_stop_hit=n_stop_hit,
        n_stop_hit_ambiguous=n_stop_hit_ambiguous,
        n_expired=0,
        n_not_applicable=0,
        hit_rate_strict=hit_rate_resolved * 0.8,
        hit_rate_resolved=hit_rate_resolved,
        hit_rate_optimistic=hit_rate_resolved,
        n_with_direction_judgement=n_judged,
        direction_accuracy=direction_accuracy,
        brier_score=0.20,
        brier_skill_score=brier_skill_score,
        base_rate=0.52,
        mean_confidence=0.65,
        mean_return=0.01,
        median_return=0.005,
    )


def _eval(
    *,
    overall: CalibrationReport | None = None,
    by_horizon: dict | None = None,
    by_ticker: dict | None = None,
    by_direction: dict | None = None,
    by_month: dict | None = None,
) -> BacktestEvaluation:
    """Compose a BacktestEvaluation from individual reports.

    Run + graded fields use MagicMocks because insights only read
    the *.overall and breakdown dicts.
    """
    return BacktestEvaluation(
        run=MagicMock(spec=BacktestRun),
        graded=[],  # not consulted by insights
        overall=overall or _report(),
        by_horizon=by_horizon or {},
        by_ticker=by_ticker or {},
        by_direction=by_direction or {},
        by_month=by_month or {},
    )


# ─────────────────────────────────────────────────────────────
# Overall verdict (always fires)
# ─────────────────────────────────────────────────────────────
class TestOverallVerdict:
    def test_positive_when_da_and_bss_both_good(self):
        ev = _eval(overall=_report(
            direction_accuracy=0.62,
            brier_skill_score=0.10,
            n_judged=50,
        ))
        ins = generate_insights(ev)
        verdicts = [i for i in ins if "skill" in i.headline.lower()
                    or "anti-skill" in i.headline.lower()
                    or "marginal" in i.headline.lower()]
        assert any(i.level == InsightLevel.POSITIVE for i in verdicts)

    def test_critical_when_da_below_threshold(self):
        """DA <= 0.45 should trip CRITICAL even if BSS is borderline."""
        ev = _eval(overall=_report(
            direction_accuracy=0.40,
            brier_skill_score=-0.10,
            n_judged=50,
        ))
        ins = generate_insights(ev)
        assert any(i.level == InsightLevel.CRITICAL for i in ins)

    def test_neutral_when_in_noise_band(self):
        """0.45 < DA < 0.55 -> NEUTRAL verdict."""
        ev = _eval(overall=_report(
            direction_accuracy=0.50,
            brier_skill_score=0.0,
            n_judged=50,
        ))
        ins = generate_insights(ev)
        # Find the verdict insight (one with 'accuracy' in headline).
        verdicts = [i for i in ins if "accuracy" in i.headline.lower()]
        assert any(i.level == InsightLevel.NEUTRAL for i in verdicts)

    def test_warning_when_no_judged_predictions(self):
        """Guards against div-by-zero in HTML; ensures the user sees WHY."""
        ev = _eval(overall=_report(
            direction_accuracy=0.0,
            brier_skill_score=None,
            n_judged=0,
            n_predictions=10,
        ))
        ins = generate_insights(ev)
        assert any(i.level == InsightLevel.WARNING and "no judged" in i.headline.lower()
                   for i in ins)


# ─────────────────────────────────────────────────────────────
# Hit rate takeaway
# ─────────────────────────────────────────────────────────────
class TestHitRate:
    def test_positive_when_hit_rate_high(self):
        ev = _eval(overall=_report(
            hit_rate_resolved=0.70,
            n_target_hit=14, n_stop_hit=6,
        ))
        ins = generate_insights(ev)
        positives = [i for i in ins if "target" in i.headline.lower()]
        assert any(i.level == InsightLevel.POSITIVE for i in positives)

    def test_warning_when_hit_rate_low(self):
        ev = _eval(overall=_report(
            hit_rate_resolved=0.30,
            n_target_hit=6, n_stop_hit=14,
        ))
        ins = generate_insights(ev)
        # Should contain a warning about low hit rate.
        warnings = [i for i in ins if "only" in i.headline.lower()
                    and "target" in i.headline.lower()]
        assert any(i.level == InsightLevel.WARNING for i in warnings)

    def test_neutral_when_no_resolutions(self):
        """No T/S resolutions -> can't claim anything about hit rate."""
        ev = _eval(overall=_report(
            n_target_hit=0, n_stop_hit=0, n_stop_hit_ambiguous=0,
        ))
        ins = generate_insights(ev)
        assert any("no t/s" in i.headline.lower() for i in ins)


# ─────────────────────────────────────────────────────────────
# Best horizon (only fires with multiple horizons)
# ─────────────────────────────────────────────────────────────
class TestBestHorizon:
    def test_silent_with_single_horizon(self):
        """Single-horizon backtest -> nothing to compare -> no insight."""
        ev = _eval(by_horizon={
            PredictionHorizon.WEEKLY: _report(n_judged=40),
        })
        ins = generate_insights(ev)
        # No insight should mention "outperforms"
        assert not any("outperforms" in i.headline.lower() for i in ins)

    def test_fires_with_meaningful_spread(self):
        """When two horizons have different DA, the better one is named."""
        ev = _eval(by_horizon={
            PredictionHorizon.DAILY: _report(direction_accuracy=0.45, n_judged=40),
            PredictionHorizon.WEEKLY: _report(direction_accuracy=0.65, n_judged=40),
        })
        ins = generate_insights(ev)
        # Should call out weekly outperforming.
        outperform = [i for i in ins if "outperforms" in i.headline.lower()]
        assert len(outperform) >= 1
        assert "weekly" in outperform[0].headline.lower()

    def test_silent_with_undersampled_horizons(self):
        """Even with 2 horizons, if neither has enough samples -> silent."""
        ev = _eval(by_horizon={
            PredictionHorizon.DAILY: _report(n_judged=5),
            PredictionHorizon.WEEKLY: _report(n_judged=5),
        })
        ins = generate_insights(ev)
        assert not any("outperforms" in i.headline.lower() for i in ins)

    def test_silent_when_spread_too_small(self):
        """5%+ spread required; below that = noise."""
        ev = _eval(by_horizon={
            PredictionHorizon.DAILY: _report(direction_accuracy=0.55, n_judged=40),
            PredictionHorizon.WEEKLY: _report(direction_accuracy=0.56, n_judged=40),
        })
        ins = generate_insights(ev)
        assert not any("outperforms" in i.headline.lower() for i in ins)


# ─────────────────────────────────────────────────────────────
# Ticker outliers
# ─────────────────────────────────────────────────────────────
class TestTickerOutliers:
    def test_silent_with_single_ticker(self):
        ev = _eval(by_ticker={"RELIANCE.NS": _report(n_judged=40)})
        ins = generate_insights(ev)
        assert not any("above overall" in i.headline.lower()
                       or "vs overall" in i.headline.lower() for i in ins)

    def test_fires_for_ticker_significantly_above(self):
        # NOTE: keep deltas asymmetric so floating-point doesn't pick
        # the wrong winner via `max(...)` ties (e.g. 0.70-0.50=0.1999...
        # vs 0.30-0.50=0.2 exactly -- the negative would win).
        overall = _report(direction_accuracy=0.50, n_judged=80)
        ev = _eval(
            overall=overall,
            by_ticker={
                "RELIANCE.NS": _report(direction_accuracy=0.75, n_judged=40),
                "TCS.NS": _report(direction_accuracy=0.45, n_judged=40),
            },
        )
        ins = generate_insights(ev)
        # RELIANCE +25% should be the most extreme.
        flagged = [i for i in ins if "RELIANCE.NS" in i.headline]
        assert len(flagged) == 1
        assert flagged[0].level == InsightLevel.POSITIVE

    def test_fires_for_ticker_significantly_below(self):
        overall = _report(direction_accuracy=0.55, n_judged=80)
        ev = _eval(
            overall=overall,
            by_ticker={
                "RELIANCE.NS": _report(direction_accuracy=0.55, n_judged=40),
                "WORST.NS": _report(direction_accuracy=0.30, n_judged=40),
            },
        )
        ins = generate_insights(ev)
        flagged = [i for i in ins if "WORST.NS" in i.headline]
        assert len(flagged) == 1
        assert flagged[0].level == InsightLevel.WARNING


# ─────────────────────────────────────────────────────────────
# Sample size warning
# ─────────────────────────────────────────────────────────────
class TestSampleSize:
    def test_warns_on_small_n(self):
        ev = _eval(overall=_report(n_judged=10))
        ins = generate_insights(ev)
        assert any(i.level == InsightLevel.WARNING and "sample" in i.headline.lower()
                   for i in ins)

    def test_silent_on_large_n(self):
        ev = _eval(overall=_report(n_judged=100))
        ins = generate_insights(ev)
        assert not any("small sample" in i.headline.lower() for i in ins)


# ─────────────────────────────────────────────────────────────
# Orchestrator behavior
# ─────────────────────────────────────────────────────────────
class TestOrchestrator:
    def test_severity_order_critical_first(self):
        """generate_insights must sort CRITICAL -> WARNING -> NEUTRAL -> POSITIVE.

        Critical items are action items -- they MUST render at the top.
        """
        ev = _eval(overall=_report(
            direction_accuracy=0.40,  # critical
            brier_skill_score=-0.10,
            n_judged=10,  # also triggers warning
            hit_rate_resolved=0.30,  # warning
            n_target_hit=3, n_stop_hit=7,
        ))
        ins = generate_insights(ev)
        levels = [i.level for i in ins]
        # No POSITIVE before CRITICAL etc.
        order = {InsightLevel.CRITICAL: 0, InsightLevel.WARNING: 1,
                 InsightLevel.NEUTRAL: 2, InsightLevel.POSITIVE: 3}
        ranks = [order[lvl] for lvl in levels]
        assert ranks == sorted(ranks)

    def test_returns_list_of_insights(self):
        ev = _eval()
        ins = generate_insights(ev)
        assert isinstance(ins, list)
        for i in ins:
            assert isinstance(i, Insight)
            assert i.headline
            assert i.detail
