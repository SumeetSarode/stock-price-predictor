"""Tests for the background grading scheduler.

Covers three layers:
  - run_grading_pass(): the unit of work (empty history, happy path,
    error-swallowing).
  - grading_loop(): cadence + clean cancellation.
  - lifespan wiring: scheduler is opt-in (off by default) and gets
    cancelled on shutdown when on.
"""
from __future__ import annotations

import asyncio

import pytest

from price_predictor.web.services import scheduler
from price_predictor.web.services.grading_service import (
    GradedPrediction,
    Scorecard,
)
from price_predictor.web.services.history_service import HistoryRow


def _hist_row(rid: int, direction="bullish") -> HistoryRow:
    from datetime import datetime, timezone
    return HistoryRow(
        id=rid, ticker="RELIANCE.NS", horizon="weekly",
        created_at=datetime.now(timezone.utc),
        direction=direction, confidence_pct=70, close_price=100.0,
        entry_low=99.0, entry_high=101.0, target_value=110.0,
        stop_value=95.0, risk_reward=2.0, grade=None,
    )


def _graded(row: HistoryRow, outcome="hit", r=2.0) -> GradedPrediction:
    return GradedPrediction(
        row=row, outcome=outcome, r_multiple=r,
        resolved_at=None, bars_used=1, note=None,
    )


class TestRunGradingPass:
    async def test_empty_history_returns_empty_scorecard(self, monkeypatch):
        monkeypatch.setattr(scheduler, "list_history", lambda **kw: ([], 0))
        sc = await scheduler.run_grading_pass(limit=10)
        assert isinstance(sc, Scorecard)
        assert sc.total == 0

    async def test_happy_path_grades_and_aggregates(self, monkeypatch):
        rows = [_hist_row(1), _hist_row(2)]
        monkeypatch.setattr(scheduler, "list_history", lambda **kw: (rows, 2))

        async def _fake_grade_rows(rs):
            return [_graded(rs[0], "hit"), _graded(rs[1], "stopped", r=-1.0)]

        monkeypatch.setattr(scheduler, "grade_rows", _fake_grade_rows)

        sc = await scheduler.run_grading_pass(limit=10)
        assert sc.total == 2
        assert sc.hits == 1
        assert sc.stops == 1
        assert sc.hit_rate == 0.5

    async def test_grade_rows_error_is_swallowed(self, monkeypatch):
        rows = [_hist_row(1)]
        monkeypatch.setattr(scheduler, "list_history", lambda **kw: (rows, 1))

        async def _boom(rs):
            raise RuntimeError("NSE unreachable")

        monkeypatch.setattr(scheduler, "grade_rows", _boom)

        # Must NOT raise — a bad pass returns an empty scorecard.
        sc = await scheduler.run_grading_pass(limit=10)
        assert sc.total == 0

    async def test_list_history_error_is_swallowed(self, monkeypatch):
        def _boom(**kw):
            raise RuntimeError("db locked")

        monkeypatch.setattr(scheduler, "list_history", _boom)
        sc = await scheduler.run_grading_pass(limit=10)
        assert sc.total == 0


class TestGradingLoop:
    async def test_loop_runs_passes_then_cancels_cleanly(self, monkeypatch):
        calls = {"n": 0}

        async def _fake_pass(*, limit):
            calls["n"] += 1
            return Scorecard(0, 0, 0, 0, 0, 0, None, None)

        monkeypatch.setattr(scheduler, "run_grading_pass", _fake_pass)

        # Tiny interval so a couple of passes happen fast, no startup delay.
        task = asyncio.create_task(
            scheduler.grading_loop(
                interval_seconds=0.01, startup_delay_seconds=0.0, limit=5
            )
        )
        await asyncio.sleep(0.05)  # let a few passes run
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert calls["n"] >= 1  # at least one pass fired

    async def test_startup_delay_is_respected(self, monkeypatch):
        calls = {"n": 0}

        async def _fake_pass(*, limit):
            calls["n"] += 1
            return Scorecard(0, 0, 0, 0, 0, 0, None, None)

        monkeypatch.setattr(scheduler, "run_grading_pass", _fake_pass)

        task = asyncio.create_task(
            scheduler.grading_loop(
                interval_seconds=10, startup_delay_seconds=5, limit=5
            )
        )
        # Cancel before the (long) startup delay elapses → zero passes.
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls["n"] == 0


class TestLifespanWiring:
    async def test_scheduler_off_by_default(self):
        from price_predictor.web.settings import settings
        # Guards the invariant the whole test suite relies on: no
        # background task spawned unless explicitly enabled.
        assert settings.enable_scheduler is False

    async def test_lifespan_spawns_and_cancels_when_enabled(self, monkeypatch):
        from price_predictor.web import app as app_module

        spawned = {"started": False, "cancelled": False}

        async def _fake_loop(**kw):
            spawned["started"] = True
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                spawned["cancelled"] = True
                raise

        monkeypatch.setattr(app_module, "grading_loop", _fake_loop)
        monkeypatch.setattr(app_module.settings, "enable_scheduler", True)

        test_app = app_module.create_app()
        async with app_module._lifespan(test_app):
            await asyncio.sleep(0.02)
            assert spawned["started"] is True
        # Exiting the context must cancel the task.
        assert spawned["cancelled"] is True

    async def test_lifespan_noop_when_disabled(self, monkeypatch):
        from price_predictor.web import app as app_module

        started = {"v": False}

        async def _fake_loop(**kw):
            started["v"] = True

        monkeypatch.setattr(app_module, "grading_loop", _fake_loop)
        monkeypatch.setattr(app_module.settings, "enable_scheduler", False)

        test_app = app_module.create_app()
        async with app_module._lifespan(test_app):
            await asyncio.sleep(0.02)
        assert started["v"] is False
