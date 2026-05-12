"""Unit tests for the replay_context contextvar.

WHAT WE TEST
============
The contract that lets us flip backtest mode on for the news_impact
agent's tools without changing their signatures:
  - Default is None (live mode).
  - Inside a replay_context block, get_as_of() returns the date.
  - On exit (normal OR exception), the previous value is restored.
  - Two parallel asyncio tasks see independent values (asyncio-safe).
  - Nesting works: inner block shadows outer, restores on exit.
"""
from __future__ import annotations

import asyncio
from datetime import date

import pytest

from price_predictor.prediction.replay_context import (
    get_as_of,
    replay_context,
)


class TestReplayContext:
    def test_default_is_none(self):
        """Live mode is the default -- no opt-in required."""
        assert get_as_of() is None

    def test_inside_block_returns_set_value(self):
        with replay_context(date(2024, 6, 14)):
            assert get_as_of() == date(2024, 6, 14)

    def test_exit_restores_previous_value(self):
        assert get_as_of() is None
        with replay_context(date(2024, 6, 14)):
            assert get_as_of() == date(2024, 6, 14)
        assert get_as_of() is None  # restored

    def test_exit_on_exception_restores(self):
        """If a backtest call blows up mid-way, the contextvar must NOT
        leak into the next call -- otherwise live predictions after a
        failed backtest could quietly inherit a stale as_of.
        """
        with pytest.raises(RuntimeError):
            with replay_context(date(2024, 6, 14)):
                assert get_as_of() == date(2024, 6, 14)
                raise RuntimeError("simulated failure")
        assert get_as_of() is None  # cleaned up

    def test_nested_blocks_shadow_and_restore(self):
        with replay_context(date(2024, 1, 1)):
            assert get_as_of() == date(2024, 1, 1)
            with replay_context(date(2024, 6, 14)):
                assert get_as_of() == date(2024, 6, 14)  # inner wins
            assert get_as_of() == date(2024, 1, 1)  # outer restored
        assert get_as_of() is None  # all the way out

    def test_explicit_none_inside_block_is_live(self):
        """Edge case -- callers can force live mode inside a backtest."""
        with replay_context(date(2024, 6, 14)):
            with replay_context(None):
                assert get_as_of() is None
            assert get_as_of() == date(2024, 6, 14)

    def test_parallel_asyncio_tasks_are_isolated(self):
        """The whole reason we use contextvars (vs threading.local or a
        module global): two concurrent backtest calls with different
        as_of MUST see their own values, not race each other's.
        """
        observed: dict[str, list] = {"a": [], "b": []}

        async def _task(name: str, as_of: date, delay: float):
            with replay_context(as_of):
                # Yield control so the scheduler interleaves us with
                # the other task -- if isolation is broken we'll see
                # the other task's as_of here.
                await asyncio.sleep(delay)
                observed[name].append(get_as_of())
                await asyncio.sleep(delay)
                observed[name].append(get_as_of())

        async def _main():
            await asyncio.gather(
                _task("a", date(2024, 1, 1), 0.01),
                _task("b", date(2024, 6, 14), 0.005),
            )

        asyncio.run(_main())

        assert observed["a"] == [date(2024, 1, 1), date(2024, 1, 1)]
        assert observed["b"] == [date(2024, 6, 14), date(2024, 6, 14)]
