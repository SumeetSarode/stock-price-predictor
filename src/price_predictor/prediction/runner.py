"""Process-wide ADK Runner + SessionService singletons for the predictor.

WHY THIS EXISTS
===============
Every call to `predict()` needs to invoke 2 agents (news_impact and
synthesizer). Each agent invocation requires a Runner wired to that
agent + a SessionService to record the conversation. Building those
per-call would be wasteful (~zero benefit, extra GC pressure).

This module owns the singletons. Same pattern as data/_shared_cache.py
(one PriceCache per process). Tests can override via setter for
deterministic mocking.

INTENTIONAL TRADE-OFFS
======================
- Module-level state is usually a code smell, but ADK's Runner is
  conceptually one-per-(agent, app, session_service) — building it
  once at module load is the canonical pattern in ADK best-practices.
- InMemorySessionService is fine for v1 (CLI use, single process). To
  go multi-process / persistent, swap the constructor to
  DatabaseSessionService("sqlite:///data/sessions.db") here. No other
  module needs to change.
"""
from __future__ import annotations

from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService, InMemorySessionService

# Same app_name across all runners so sessions are colocated. Treat it
# as the predictor's "namespace" inside ADK's session store.
APP_NAME = "price_predictor"

# One logical user — the predictor itself. ADK's session model assumes
# a multi-user world; we're a library, so we collapse to a single user.
USER_ID = "predictor"

# ─────────────────────────────────────────────────────────────
# Singleton state (lazily built)
# ─────────────────────────────────────────────────────────────
_session_service: Optional[BaseSessionService] = None
_runners: dict[str, Runner] = {}


def get_session_service() -> BaseSessionService:
    """Return the process-wide SessionService, building it lazily."""
    global _session_service
    if _session_service is None:
        _session_service = InMemorySessionService()
    return _session_service


def get_runner(agent: LlmAgent) -> Runner:
    """Return a cached Runner for `agent`, building it on first use.

    Cached by agent.name so each distinct agent gets exactly one Runner
    in the process. This matches the "one Runner per app" guidance in
    docs/best_practices.md while still supporting multiple agents (we
    have 2: news_impact and synthesizer).
    """
    if agent.name not in _runners:
        _runners[agent.name] = Runner(
            agent=agent,
            app_name=APP_NAME,
            session_service=get_session_service(),
        )
    return _runners[agent.name]


def reset() -> None:
    """Clear the singleton state. Used by tests for hermetic isolation.

    Without this, a test that overrides the session_service would leak
    that override into other tests via the cached Runners.
    """
    global _session_service, _runners
    _session_service = None
    _runners = {}
