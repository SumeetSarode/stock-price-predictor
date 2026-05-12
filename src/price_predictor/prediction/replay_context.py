"""Replay context - process-local as_of for backtest mode.

WHY THIS EXISTS
===============
The news_impact agent calls a handful of tools (fetch_recent_news,
fetch_recent_filings, fetch_recent_prices, fetch_estimates) whose
signatures the LLM controls. We can't add an `as_of` parameter to
those tools -- the LLM has no idea what value to pass and would
either omit it (silent leak to today's data) or hallucinate a date.

A contextvar lets us set as_of *implicitly* before invoking the
agent: the tools, deep inside the call tree, read the contextvar to
discover whether they're in backtest mode and what date to pin to.

contextvars are the right tool here (vs threading.local or a module
global) because:
  - They're asyncio-aware: each task has its own copy, so two
    parallel predict() calls with different as_of values don't trample
    each other.
  - They're scoped: `with replay_context(as_of=X):` block restores
    the previous value on exit, even on exception.

LIVE MODE
=========
Default value is None. Tools that consult `get_as_of()` and see None
behave exactly as before: end = today. No-op for live callers.

NOT A KITCHEN SINK
==================
Just the as_of date. If we ever need more replay-mode metadata
(e.g., a "snapshot store handle" or "leak-mode strict|warn"), we'd
upgrade this to a small frozen dataclass. YAGNI for now -- one
date is the entire load-bearing contract.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date
from typing import Iterator


# Sentinel: None means "live mode" (no replay context active).
_as_of_var: ContextVar[date | None] = ContextVar(
    "price_predictor_replay_as_of", default=None,
)


def get_as_of() -> date | None:
    """Return the current task's replay as_of, or None if live mode.

    Tools should call this at the TOP of their fetch logic and pass
    the result down to the underlying date-window computation. None
    means "no replay active, use today".
    """
    return _as_of_var.get()


@contextmanager
def replay_context(as_of: date | None) -> Iterator[None]:
    """Set the replay as_of for the duration of the with-block.

    Usage:
        with replay_context(as_of=date(2024, 6, 14)):
            await run_news_impact_agent("RELIANCE.NS")

    Passing as_of=None inside the block is treated as "explicitly
    live" -- shadows any outer context. This is rarely useful but
    keeps the API symmetric with the contextvar's None sentinel.
    """
    token = _as_of_var.set(as_of)
    try:
        yield
    finally:
        _as_of_var.reset(token)
