"""predict() — the per-stock prediction orchestrator.

THE BIG PICTURE
===============
This module wires the gather and synthesis phases into one async function:

    predict(ticker, horizons=DEFAULT_HORIZONS)
        |
        +-- PHASE 1 (parallel via asyncio.gather, ONCE per call):
        |     compose_technical_view(ticker)      [Layer-1 cluster tools]
        |     run_news_impact_agent(ticker)       [Agent #1]
        |
        +-- PHASE 2 (parallel fan-out across N horizons):
        |     for each h in horizons:
        |         build SynthesisInput(horizon=h, ...)
        |         run_synthesizer_agent(si) + guardrails  [Agent #2 × N]
        |
        +-- return dict[PredictionHorizon, Prediction]

WHY GATHER ONCE, SYNTHESIZE N TIMES
===================================
Technical indicators (RSI, ATR, etc.) and news/filings are HORIZON-AGNOSTIC
— the same evidence applies whether we're predicting daily or monthly.
The horizon-specific reasoning happens at SYNTHESIS time, where the LLM
weighs the same evidence against different time windows. Refetching prices
or news per horizon would be pure waste.

This is the FIRST place we invoke ADK agents programmatically (vs via
`adk run` CLI). The agent-call helpers (`run_news_impact_agent`,
`run_synthesizer_agent`) hide the Runner / SessionService / Event-stream
plumbing behind a clean async function-call shape.

DEGRADATION POLICY
==================
- Technical failure   -> raise PredictionError. Technicals are core; a
                         prediction missing them is unreliable.
- News failure        -> degrade to neutral ImpactAssessment, mark
                         analysis_basis fields, append degraded tag to
                         model_chain. Predictions still produced.
- Synthesizer failure -> raise PredictionError. Fail-fast across ALL
                         horizons: a partial result (e.g. daily ok but
                         weekly failed) breaks the daily+weekly UX
                         contract and masks reliability problems.

WHY THE AGENT HELPERS LIVE HERE (not in their agent packages)
=============================================================
Each agent package owns the agent DEFINITION (factory + root_agent for
adk run). Programmatic invocation is a PREDICTOR concern - the predictor
is the one that needs Runners. Putting the helpers here keeps the agent
packages free of Runner imports and avoids a circular dependency
(news_impact would otherwise need to import prediction.runner).
"""
from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterable
from datetime import date, datetime, time
from typing import Literal

from google.adk.agents import LlmAgent
from google.genai import types
from loguru import logger

from price_predictor.agents.news_impact import (
    ImpactAssessment,
    build_news_impact_prompt,
    gather_news_impact_inputs,
    make_news_impact_agent,
    neutral_impact_assessment,
)
from price_predictor.agents.synthesizer import (
    build_synth_prompt,
    make_synthesizer_agent,
)
from price_predictor.config.settings import settings
from price_predictor.data.news_snapshot import (
    NewsSnapshot,
    get_news_snapshot,
    set_news_snapshot,
)
from price_predictor.prediction.guardrails import (
    HallucinationError,
    validate_all,
)
from price_predictor.prediction.inputs import (
    SynthesisInput,
    TechnicalView,
    TechnicalViewError,
    _resolve_ticker,
    compose_technical_view,
)
from price_predictor.prediction.replay_context import replay_context
from price_predictor.prediction.runner import USER_ID, get_runner, get_session_service
from price_predictor.prediction.schema import (
    DEFAULT_HORIZONS,
    Prediction,
    PredictionHorizon,
)

# Horizon literal mirrors PredictionHorizon enum values. We accept the
# raw string for ergonomic API; the synthesizer agent passes it through
# to Prediction unchanged.
Horizon = Literal["daily", "weekly", "biweekly", "monthly"]

# Caller-friendly alias: anywhere we accept a horizon, accept either the
# enum or the string (which gets normalized to the enum internally).
HorizonLike = PredictionHorizon | Horizon

# India Standard Time anchor for as_of. Same convention used elsewhere.
from datetime import timedelta, timezone  # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))

# Module-level agent instances. Built once at import time so each
# prediction reuses the same Runner (Runner is cached per agent inside
# prediction.runner). Same singleton philosophy as _shared_cache.
_news_impact_agent: LlmAgent = make_news_impact_agent()
_synthesizer_agent: LlmAgent = make_synthesizer_agent()

# Model-chain identifier for the news_impact / synthesizer agents. Used
# to populate Prediction.model_chain. Hardcoded to the profile name for
# v1; commit 6 may upgrade to read the resolved model name from the
# resilient chain after invocation.
_NEWS_MODEL_TAG = "news_impact:agentic"
_SYNTH_MODEL_TAG = "synthesizer:agentic"
# Marker appended to model_chain when news fell back to the degraded
# 'neutral' assessment. Lets downstream consumers tell at a glance.
_NEWS_DEGRADED_TAG = "news_impact:degraded"

# Distinct tag for the live-news-fetch-failed case versus the
# backtest-news-replay-pending case below.
_NEWS_BACKTEST_REPLAY_TAG = "news_impact:agentic_replay"

# Legacy tag retained for back-compat with any predictions written
# between Step 1 and Step 1.5 -- the test suite still imports it.
# New predictions made under as_of WITH replay infrastructure get
# the _NEWS_BACKTEST_REPLAY_TAG above.
_NEWS_BACKTEST_PENDING_TAG = "news_impact:backtest_pending"


# ──────────────────────────────────────────────────────────────
# News degradation
# ──────────────────────────────────────────────────────────────
def _degraded_impact(ticker: str, error_msg: str) -> ImpactAssessment:
    """Build a degenerate but valid 'no news' ImpactAssessment.

    Used when run_news_impact_agent raises. The synthesizer sees this
    as a confidence-0, neutral-sentiment, no-catalysts assessment and
    naturally weights technicals more heavily (the prompt teaches
    'tie-break to technical for short horizons; trust news for medium/
    long' — confidence=0 means news contributes nothing either way).

    Truncates the error message to keep the reasoning field readable.
    """
    return ImpactAssessment(
        ticker=ticker,
        sentiment="neutral",
        confidence=0.0,
        estimated_pct_move=0.0,
        reasoning=f"News unavailable (degraded): {error_msg[:200]}",
        catalysts=[],
    )


# Total attempts the synthesizer gets before we surface the guardrail
# failure to the caller. 1 initial + 2 retries with feedback. See
# `synthesize_with_guardrails` docstring for rationale.
_MAX_GUARDRAIL_ATTEMPTS = 3

# ─────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────
class PredictionError(RuntimeError):
    """Raised when the predictor cannot produce a Prediction.

    Wraps the underlying cause (TechnicalViewError, agent failure, etc.)
    so callers have one exception type to catch. The original cause is
    available via `.__cause__`.
    """


class SynthesisParseError(PredictionError):
    """Raised when the synthesizer LLM returned content that cannot be
    parsed as a Prediction.

    Distinct from the parent so that the guardrail retry loop can catch
    this (recoverable: re-sample the LLM) without also catching the
    loop's own exhaustion error (which IS a PredictionError but must
    flow through to the caller).

    Common causes:
      - Groq returned 200 OK with empty content (silent structured-output
        validation failure -- distinct from the BadRequestError-wrapped
        `json_validate_failed` path handled in llm/resilient.py).
      - Model emitted a markdown fence around the JSON.
      - Response was truncated by max_tokens.

    Recovery: re-invoke the synthesizer. Either the same model produces
    cleaner output on a fresh sample, or (if cooled-down) the resilient
    chain has fallen to a different model with stronger JSON discipline.
    """


# ─────────────────────────────────────────────────────────────
# Agent invocation helpers
# Each takes a typed input and returns the agent's typed output.
# The Runner / Session / Event-stream plumbing is hidden inside.
# ─────────────────────────────────────────────────────────────
async def _run_agent_for_text(
    agent: LlmAgent, prompt_text: str
) -> str:
    """Drive one agent invocation; return the final response text.

    This is the SINGLE place where ADK Runner mechanics live. Both
    helpers below delegate here and parse the JSON themselves.

    The flow:
        1. Wrap the prompt as a Content message
        2. Create a fresh session (one per call - no cross-contamination)
        3. Stream events from runner.run_async; grab the last
           is_final_response event's text
        4. Return the raw text (caller parses it via Pydantic)

    Raises:
        PredictionError: agent produced no final response (unusual; means
                         the LLM crashed mid-stream or the resilient chain
                         exhausted all fallbacks).
    """
    runner = get_runner(agent)
    session_id = str(uuid.uuid4())

    # Sessions are scoped per call. ADK requires the session to exist
    # before run_async, so we create it explicitly.
    await get_session_service().create_session(
        app_name=runner.app_name, user_id=USER_ID, session_id=session_id,
    )

    new_message = types.Content(
        role="user", parts=[types.Part(text=prompt_text)],
    )

    final_text: str | None = None
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=new_message,
    ):
        # is_final_response() returns True for the agent's terminal
        # output. With output_schema set, the text IS the JSON blob.
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text

    if final_text is None:
        raise PredictionError(
            f"Agent {agent.name!r} produced no final response. "
            "Likely cause: resilient chain exhausted all fallback models."
        )
    return final_text


async def run_news_impact_agent(ticker: str) -> ImpactAssessment:
    """Gather impact inputs (pure code) then synthesize with ONE LLM call.

    'Gather in code, reason once': all data (company news, sector news,
    filings, estimates, prices) is fetched deterministically and in
    parallel by gather_news_impact_inputs(), then handed to the tool-less
    synthesizer in a single prompt. No tool loop, no re-sent context.

    Look-ahead is automatic: gather reads the replay contextvar, so a
    backtest that wraps this call in replay_context(as_of=...) stays
    honest with no extra plumbing here.

    Args:
        ticker: Canonical yfinance ticker (caller resolves KB aliases).

    Returns:
        Validated ImpactAssessment.

    Raises:
        PredictionError: agent failed or returned unparseable output.
    """
    inputs = await gather_news_impact_inputs(ticker)

    # LLM only when there's something to reason about. If gather found no
    # news, filings, or covered estimates, there is no news-driven impact
    # to assess — return a deterministic neutral assessment and skip the
    # model entirely (saves a call on quiet / illiquid names).
    if not inputs.has_news_evidence:
        logger.info(
            f"news_impact: no evidence for {ticker}; "
            f"skipping LLM, returning neutral assessment"
        )
        return neutral_impact_assessment(ticker)

    prompt = build_news_impact_prompt(inputs)
    raw = await _run_agent_for_text(_news_impact_agent, prompt)
    try:
        return ImpactAssessment.model_validate_json(raw)
    except Exception as e:
        raise PredictionError(
            f"news_impact agent returned invalid ImpactAssessment JSON: {e}"
        ) from e


async def run_synthesizer_agent(
    si: SynthesisInput, *, feedback: str | None = None,
) -> Prediction:
    """Invoke the synthesizer agent and return the parsed Prediction.

    Args:
        si: Bundled gather-phase output. The synthesizer reads it as
            JSON and emits a Prediction.
        feedback: Optional error message from a prior failed attempt.
            Appended to the prompt so the LLM sees what went wrong.
            Used by the retry loop in synthesize_with_guardrails().

    Returns:
        Validated Prediction.

    Raises:
        PredictionError: agent failed or returned unparseable output.
    """
    prompt = build_synth_prompt(si)
    if feedback:
        # Surface the prior failure inline. The LLM tends to over-correct
        # if we shout - keep it factual + actionable.
        prompt = (
            f"{prompt}\n\n"
            f"NOTE: Your previous attempt FAILED guardrail validation:\n"
            f"{feedback}\n"
            f"Re-read the input carefully and produce a Prediction that "
            f"avoids this issue."
        )
    raw = await _run_agent_for_text(_synthesizer_agent, prompt)
    try:
        return Prediction.model_validate_json(raw)
    except Exception as e:
        raise SynthesisParseError(
            f"synthesizer agent returned invalid Prediction JSON: {e}"
        ) from e


async def synthesize_with_guardrails(si: SynthesisInput) -> Prediction:
    """Run the synthesizer + guardrails with up to 2 retries.

    Flow:
      1. Call synthesizer.
      2. validate_all(prediction, si).
      3. If HallucinationError OR SynthesisParseError, retry up to
         _MAX_GUARDRAIL_RETRIES more times, each time feeding the most
         recent error back into the prompt as actionable feedback.
      4. If all attempts fail, raise PredictionError wrapping the last
         error.

    Why catch BOTH error classes here:
      - HallucinationError: LLM produced valid JSON but violated a
        grounding/citation/consistency/calibration rule. Retry with
        the rule violation as feedback.
      - SynthesisParseError: LLM produced unparseable content (empty
        string from Groq's silent structured-output failure, markdown
        fence around JSON, truncated response). Retry: stochastic
        re-sampling almost always recovers, and if the underlying
        model is now cooled down, the resilient chain transparently
        falls back to a different provider.

    Why up to 3 total attempts (was 2): empirically, the synthesizer
    LLM's outputs vary stochastically across attempts. With a single
    retry, predictions that COULD pass guardrails on attempt 3 (because
    1 and 2 happened to hit boundary edge-cases on entry-zone rounding
    or LLM stubbornness on direction) were being thrown out. 3 attempts
    is the smallest budget that empirically covers the realistic LLM
    sampling variance without burning excessive tokens on genuinely
    ambiguous inputs.

    Raises:
        PredictionError: every retry failed (with last error as cause).
    """
    feedback: str | None = None
    last_error: Exception | None = None

    for attempt in range(1, _MAX_GUARDRAIL_ATTEMPTS + 1):
        try:
            prediction = await run_synthesizer_agent(si, feedback=feedback)
            validate_all(prediction, si)
            if attempt > 1:
                logger.info(
                    f"retry succeeded on attempt {attempt} after synthesis feedback"
                )
            return prediction
        except SynthesisParseError as e:
            # LLM returned garbage (empty / markdown-fenced / truncated).
            # No partial Prediction to inspect; just nudge the model.
            last_error = e
            feedback = (
                f"Your previous response could not be parsed as JSON: {e}. "
                f"Emit ONLY a single JSON object that exactly matches the "
                f"Prediction schema. No markdown fences, no commentary, "
                f"no leading or trailing text."
            )
            if attempt < _MAX_GUARDRAIL_ATTEMPTS:
                logger.warning(
                    f"synthesizer parse failure on attempt {attempt}: {e}. "
                    f"Retrying (attempt {attempt + 1}/{_MAX_GUARDRAIL_ATTEMPTS})."
                )
        except HallucinationError as e:
            last_error = e
            feedback = str(e)
            if attempt < _MAX_GUARDRAIL_ATTEMPTS:
                logger.warning(
                    f"guardrail tripped on attempt {attempt}: {e}. "
                    f"Retrying (attempt {attempt + 1}/{_MAX_GUARDRAIL_ATTEMPTS})."
                )

    # All attempts exhausted.
    assert last_error is not None  # loop body always sets it on the failure path
    raise PredictionError(
        f"Synthesizer failed {_MAX_GUARDRAIL_ATTEMPTS}×. "
        f"Last error: {last_error}"
    ) from last_error


# ─────────────────────────────────────────────────────────────
# The public API: predict()
# ─────────────────────────────────────────────────────────────
def _normalize_horizons(
    horizons: Iterable[HorizonLike] | None,
) -> tuple[PredictionHorizon, ...]:
    """Coerce the caller's horizons argument to a deduplicated enum tuple.

    Accepts:
      - None                           -> DEFAULT_HORIZONS
      - PredictionHorizon enum members
      - Raw strings ("daily" / "weekly" / "biweekly" / "monthly")
      - Mixed iterables of either

    Why dedupe: callers passing `["weekly", "weekly"]` (whether by
    accident or by glueing together CLI args) shouldn't pay 2x synth
    calls. We preserve first-occurrence order so output is stable.

    Raises:
        ValueError: empty input, or any element fails to coerce to enum.
    """
    if horizons is None:
        return DEFAULT_HORIZONS

    materialized = list(horizons)
    if not materialized:
        raise ValueError(
            "horizons must be non-empty (pass None for DEFAULT_HORIZONS)"
        )

    out: list[PredictionHorizon] = []
    seen: set[PredictionHorizon] = set()
    for h in materialized:
        try:
            enum_val = PredictionHorizon(h) if not isinstance(h, PredictionHorizon) else h
        except ValueError as e:
            raise ValueError(
                f"Unknown horizon {h!r}. "
                f"Valid: {[m.value for m in PredictionHorizon]}"
            ) from e
        if enum_val not in seen:
            seen.add(enum_val)
            out.append(enum_val)
    return tuple(out)


async def predict(
    ticker: str,
    horizons: Iterable[HorizonLike] | None = None,
    *,
    sensitivity: Literal["standard", "sensitive", "smooth"] = "standard",
    as_of: date | None = None,
) -> dict[PredictionHorizon, Prediction]:
    """Produce one Prediction per requested horizon for one stock.

    GATHER PHASE: technicals + news fetched ONCE (horizon-agnostic).
    SYNTHESIS PHASE: N synthesizer calls in parallel, one per horizon.

    The gather phase is shared across horizons because:
      - Technical indicators (RSI, ATR, etc.) are computed from the SAME
        OHLCV history regardless of prediction window.
      - News + filings are inputs to ALL horizons; refetching per horizon
        would be wasteful.
      - The horizon-specific reasoning happens at SYNTHESIS time, where
        the LLM weighs the same evidence against different time windows.

    Args:
        ticker: Stock symbol in any form ('reliance', 'RELIANCE.NS',
            'AAPL'). Resolved to canonical via the KB.
        horizons: Iterable of PredictionHorizon enum values OR strings
            ("daily"/"weekly"/etc). Defaults to DEFAULT_HORIZONS (all 4).
            Duplicates are removed; first-occurrence order preserved.
        sensitivity: Indicator-cluster preset. Default 'standard'.
        as_of: Keyword-only. The trading date the prediction should be
            anchored to. ``None`` (default) means "now" — the live
            behavior using ``datetime.now(IST)``. A past ``date`` is
            backtest mode: technicals are fetched up to and including
            this date, no future data leaks in. Future dates are
            rejected with ValueError.

            BACKTEST NEWS BEHAVIOR (Step 1.5+): the news_impact agent
            now runs in backtest mode under a replay context. Its
            tools (news/filings/prices) read from the point-in-time
            ``NewsSnapshot`` store on disk -- first call hits GDELT/
            NSE with a window ending at as_of and snapshots the
            response; subsequent runs serve from disk for full
            reproducibility. The estimates tool short-circuits to
            no-coverage in replay mode because yfinance has no
            historical estimates archive (returning today's consensus
            would leak forward-looking info).

            Backtest predictions therefore differ from live in two
            ways the audit trail flags via the ``news_impact:agentic_replay``
            model-chain tag:
              1. No analyst-estimates evidence (vs available in live).
              2. News evidence is point-in-time-honest but limited to
                 what GDELT had indexed by as_of + the snapshot window.

    Returns:
        dict[PredictionHorizon, Prediction] keyed by horizon enum, in
        the order requested. ALL requested horizons present on success.

    Raises:
        PredictionError: gather phase failed, OR ANY horizon's synthesis
            failed (fail-fast: partial results would mask reliability
            problems and break the daily+weekly UX contract). Wraps the
            underlying cause.
        ValueError: empty/unknown horizons argument, or as_of in the future.
    """
    horizon_tuple = _normalize_horizons(horizons)
    canonical = _resolve_ticker(ticker)

    # Resolve as_of to both (a) the date passed to gather (used by the
    # cluster tools' OHLCV cutoff) and (b) the tz-aware datetime stamped
    # on the Prediction for audit. Live mode uses now; backtest mode
    # pins to 15:30 IST (NSE close) on the requested date.
    if as_of is None:
        as_of_date: date | None = None
        as_of_dt = datetime.now(IST)
    else:
        if as_of > date.today():
            raise ValueError(
                f"as_of={as_of.isoformat()} is in the future; refusing to "
                f"predict using data we cannot have. Pass a past trading "
                f"date or omit as_of to use today."
            )
        as_of_date = as_of
        as_of_dt = datetime.combine(as_of, time(15, 30), tzinfo=IST)

    horizon_labels = [h.value for h in horizon_tuple]
    mode_tag = "live" if as_of_date is None else f"backtest@{as_of_date.isoformat()}"
    logger.info(
        f"predict() start: ticker={canonical} horizons={horizon_labels} "
        f"sensitivity={sensitivity} mode={mode_tag}"
    )

    # ── PHASE 1: GATHER (parallel, horizon-agnostic) ───────────
    technical_view, impact_assessment, news_status = await _gather_phase(
        canonical, sensitivity, as_of=as_of_date,
    )
    news_degraded = news_status != "live"

    # ── PHASE 2: SYNTHESIZE per horizon (parallel fan-out) ─────
    # All N synthesizer calls share the same gathered evidence; each
    # gets its own SynthesisInput with a horizon-specific prompt slot.
    initial_chain: tuple[str, ...] = (_news_tag_for(news_status),)

    async def _synth_one(h: PredictionHorizon) -> Prediction:
        si = SynthesisInput(
            ticker=canonical,
            horizon=h.value,
            as_of=as_of_dt,
            technical_view=technical_view,
            impact_assessment=impact_assessment,
            model_chain=initial_chain,
        )
        prediction = await synthesize_with_guardrails(si)
        # Append synth tag here (single source of truth for audit trail).
        return _finalize_prediction(prediction, news_degraded=news_degraded)

    # asyncio.gather without return_exceptions: ANY horizon failing
    # raises immediately. See docstring 'Raises' for rationale.
    results = await asyncio.gather(*(_synth_one(h) for h in horizon_tuple))

    out: dict[PredictionHorizon, Prediction] = dict(zip(horizon_tuple, results))

    summary = ", ".join(
        f"{h.value}={p.direction.value}@{p.confidence:.2f}"
        for h, p in out.items()
    )
    logger.info(f"predict() done: ticker={canonical} {summary}")
    return out


async def _gather_phase(
    canonical: str,
    sensitivity: Literal["standard", "sensitive", "smooth"],
    *,
    as_of: date | None = None,
) -> tuple[TechnicalView, ImpactAssessment, str]:
    """Run the gather phase (technicals + news) once for all horizons.

    Extracted from the old predict() body so the public function reads
    as a clean two-phase orchestration.

    Args:
        canonical: KB-resolved ticker.
        sensitivity: Indicator preset.
        as_of: When set to a past date, technicals are pinned to that
            date AND the news_impact agent runs under a replay context
            so its tools (news/filings/prices) read from the
            point-in-time snapshot store instead of live sources. The
            estimates tool short-circuits to no-coverage in replay
            mode (yfinance has no historical estimates archive).

    Returns:
        Tuple of ``(technical_view, impact_assessment, news_status)``
        where ``news_status`` is one of:
          - ``"live"``            — news_impact agent ran in live mode
          - ``"agentic_replay"``  — news_impact agent ran in replay mode
          - ``"degraded"``        — agent failed; neutral fallback
        The string shape (vs the old bool) lets the audit trail tell
        the three modes apart — important for calibration analysis
        because replay-mode predictions exclude estimates and may have
        different news coverage characteristics.

    Raises:
        PredictionError: technicals failed (core, non-degradable).
    """
    technical_task = compose_technical_view(
        canonical, sensitivity=sensitivity, as_of=as_of,
    )

    backtest_mode = as_of is not None and as_of != date.today()
    if backtest_mode:
        # Ensure a snapshot store is installed before invoking the
        # agent. Idempotent: if the caller (e.g. a backtest runner)
        # already wired up a custom store, leave it alone.
        _ensure_news_snapshot_installed()
        # `replay_context` flips the contextvar that the news_impact
        # tools consult to discover backtest mode. The technical task
        # runs in parallel and is unaffected by the contextvar (it
        # uses explicit as_of plumbing, not contextvars).
        with replay_context(as_of):
            news_task = run_news_impact_agent(canonical)
            technical_result, news_result = await asyncio.gather(
                technical_task, news_task, return_exceptions=True,
            )
    else:
        news_task = run_news_impact_agent(canonical)
        technical_result, news_result = await asyncio.gather(
            technical_task, news_task, return_exceptions=True,
        )

    # Technicals are CORE - failure aborts the prediction.
    if isinstance(technical_result, BaseException):
        if isinstance(technical_result, TechnicalViewError):
            raise PredictionError(
                f"Technical analysis failed for {canonical}: {technical_result}"
            ) from technical_result
        # Unexpected exception - propagate so we don't swallow bugs.
        raise technical_result
    technical_view = technical_result

    # News is OPTIONAL - failure degrades to a neutral assessment.
    news_status: str
    if isinstance(news_result, BaseException):
        logger.warning(
            f"news_impact failed for {canonical}; degrading to neutral. "
            f"Cause: {type(news_result).__name__}: {news_result}"
        )
        impact_assessment = _degraded_impact(canonical, str(news_result))
        news_status = "degraded"
    else:
        impact_assessment = news_result
        news_status = "agentic_replay" if backtest_mode else "live"

    logger.info(
        f"gather done: technical_view bars={technical_view.bars_used} "
        f"signals=trend:{technical_view.trend.signal}/"
        f"momentum:{technical_view.momentum.signal}/"
        f"volatility:{technical_view.volatility.signal}/"
        f"levels:{technical_view.levels.signal} | "
        f"impact sentiment={impact_assessment.sentiment} "
        f"confidence={impact_assessment.confidence:.2f} "
        f"news_status={news_status}"
    )
    return technical_view, impact_assessment, news_status


def _ensure_news_snapshot_installed() -> None:
    """Install the default NewsSnapshot singleton if none is set.

    Lazy install on first backtest call keeps live-mode startup
    free of FS side-effects (no directory created if you never run
    a backtest). Tests that need a custom root or want to disable
    the store can call set_news_snapshot() directly.
    """
    if get_news_snapshot() is not None:
        return
    snapshot = NewsSnapshot(settings.news_snapshots_dir)
    set_news_snapshot(snapshot)
    logger.info(
        f"installed default NewsSnapshot at {snapshot.root} "
        f"(first backtest call this process)"
    )


def _news_tag_for(news_status: str) -> str:
    """Map news_status -> initial model_chain tag.

    Single-source-of-truth helper so predict() doesn't grow a 3-arm
    if/elif and the audit trail stays consistent across modes.
    """
    return {
        "live": _NEWS_MODEL_TAG,
        "degraded": _NEWS_DEGRADED_TAG,
        "agentic_replay": _NEWS_BACKTEST_REPLAY_TAG,
        # Legacy mapping retained for back-compat with any caller
        # passing the Step-1 status string directly. New code paths
        # produce "agentic_replay", not this.
        "backtest_pending": _NEWS_BACKTEST_PENDING_TAG,
    }[news_status]


def _finalize_prediction(
    prediction: Prediction, *, news_degraded: bool
) -> Prediction:
    """Append synth tag to model_chain; null-out news fields if degraded.

    The synthesizer's prompt instructs it to copy model_chain verbatim
    from the input. We append the synthesizer tag AFTER the call by
    constructing a new Prediction (frozen models) - this guarantees the
    audit trail reflects reality regardless of LLM compliance.

    When news was degraded, force analysis_basis to reflect that:
    consumers should see news_articles_considered=0 and
    news_sentiment_score=None regardless of what the LLM put there.
    """
    final = prediction.model_copy(
        update={"model_chain": (*prediction.model_chain, _SYNTH_MODEL_TAG)}
    )
    if news_degraded:
        final = final.model_copy(
            update={
                "analysis_basis": final.analysis_basis.model_copy(
                    update={
                        "news_sentiment_score": None,
                        "news_articles_considered": 0,
                        "filings_considered": 0,
                    }
                )
            }
        )
    return final
