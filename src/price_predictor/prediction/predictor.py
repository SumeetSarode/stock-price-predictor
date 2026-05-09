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
from datetime import datetime
from typing import Literal

from google.adk.agents import LlmAgent
from google.genai import types
from loguru import logger

from price_predictor.agents.news_impact import (
    ImpactAssessment,
    make_news_impact_agent,
)
from price_predictor.agents.synthesizer import (
    build_synth_prompt,
    make_synthesizer_agent,
)
from price_predictor.prediction.inputs import (
    SynthesisInput,
    TechnicalView,
    TechnicalViewError,
    compose_technical_view,
    _resolve_ticker,
)
from price_predictor.prediction.guardrails import (
    HallucinationError,
    validate_all,
)
from price_predictor.prediction.runner import USER_ID, get_runner, get_session_service
from price_predictor.prediction.schema import (
    DEFAULT_HORIZONS,
    Prediction,
    PredictionHorizon,
)

from collections.abc import Iterable

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


# ─────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────
class PredictionError(RuntimeError):
    """Raised when the predictor cannot produce a Prediction.

    Wraps the underlying cause (TechnicalViewError, agent failure, etc.)
    so callers have one exception type to catch. The original cause is
    available via `.__cause__`.
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
    """Invoke the news_impact agent and return the parsed ImpactAssessment.

    Args:
        ticker: Canonical yfinance ticker (caller is responsible for KB
            resolution).

    Returns:
        Validated ImpactAssessment.

    Raises:
        PredictionError: agent failed or returned unparseable output.
    """
    # The news_impact agent has its own internal prompt logic for tools
    # discovery; its user message just needs to identify the target.
    prompt = (
        f"Analyze the news, filings, estimates, and recent price action "
        f"for {ticker}. Produce an ImpactAssessment."
    )
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
        raise PredictionError(
            f"synthesizer agent returned invalid Prediction JSON: {e}"
        ) from e


async def synthesize_with_guardrails(si: SynthesisInput) -> Prediction:
    """Run the synthesizer + guardrails with one retry on hallucination.

    Flow:
      1. Call synthesizer.
      2. validate_all(prediction, si).
      3. If HallucinationError, call synthesizer ONCE more with the
         error fed back into the prompt, then re-validate.
      4. If second attempt also fails, raise PredictionError wrapping
         the second HallucinationError.

    Why one-shot: two failures in a row almost always means the input
    is genuinely ambiguous. More retries waste tokens without improving
    outcomes.

    Raises:
        PredictionError: synth failed twice (with last guardrail msg as
            cause), OR synth raised PredictionError directly.
    """
    prediction = await run_synthesizer_agent(si)
    try:
        validate_all(prediction, si)
        return prediction
    except HallucinationError as e:
        # Capture for retry feedback - except-as is scoped to the
        # except block in Py3, so we re-bind explicitly.
        first_error = e
        logger.warning(
            f"guardrail tripped on first synth attempt: {e}. Retrying once."
        )

    # Retry with feedback. If THIS one fails grounding too, give up.
    prediction = await run_synthesizer_agent(si, feedback=str(first_error))
    try:
        validate_all(prediction, si)
        logger.info("retry succeeded after guardrail feedback")
        return prediction
    except HallucinationError as e2:
        raise PredictionError(
            f"Synthesizer failed guardrails twice. Last error: {e2}"
        ) from e2


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

    Returns:
        dict[PredictionHorizon, Prediction] keyed by horizon enum, in
        the order requested. ALL requested horizons present on success.

    Raises:
        PredictionError: gather phase failed, OR ANY horizon's synthesis
            failed (fail-fast: partial results would mask reliability
            problems and break the daily+weekly UX contract). Wraps the
            underlying cause.
        ValueError: empty/unknown horizons argument.
    """
    horizon_tuple = _normalize_horizons(horizons)
    canonical = _resolve_ticker(ticker)
    as_of = datetime.now(IST)
    horizon_labels = [h.value for h in horizon_tuple]
    logger.info(
        f"predict() start: ticker={canonical} horizons={horizon_labels} "
        f"sensitivity={sensitivity}"
    )

    # ── PHASE 1: GATHER (parallel, horizon-agnostic) ───────────
    technical_view, impact_assessment, news_degraded = await _gather_phase(
        canonical, sensitivity
    )

    # ── PHASE 2: SYNTHESIZE per horizon (parallel fan-out) ─────
    # All N synthesizer calls share the same gathered evidence; each
    # gets its own SynthesisInput with a horizon-specific prompt slot.
    initial_chain: tuple[str, ...] = (
        (_NEWS_DEGRADED_TAG,) if news_degraded else (_NEWS_MODEL_TAG,)
    )

    async def _synth_one(h: PredictionHorizon) -> Prediction:
        si = SynthesisInput(
            ticker=canonical,
            horizon=h.value,
            as_of=as_of,
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
) -> tuple[TechnicalView, ImpactAssessment, bool]:
    """Run the gather phase (technicals + news) once for all horizons.

    Extracted from the old predict() body so the public function reads
    as a clean two-phase orchestration. Returns:
      (technical_view, impact_assessment, news_degraded_flag)

    Raises:
        PredictionError: technicals failed (core, non-degradable).
    """
    technical_task = compose_technical_view(canonical, sensitivity=sensitivity)
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
    news_degraded = False
    if isinstance(news_result, BaseException):
        logger.warning(
            f"news_impact failed for {canonical}; degrading to neutral. "
            f"Cause: {type(news_result).__name__}: {news_result}"
        )
        impact_assessment = _degraded_impact(canonical, str(news_result))
        news_degraded = True
    else:
        impact_assessment = news_result

    logger.info(
        f"gather done: technical_view bars={technical_view.bars_used} "
        f"signals=trend:{technical_view.trend.signal}/"
        f"momentum:{technical_view.momentum.signal}/"
        f"volatility:{technical_view.volatility.signal}/"
        f"levels:{technical_view.levels.signal} | "
        f"impact sentiment={impact_assessment.sentiment} "
        f"confidence={impact_assessment.confidence:.2f}"
    )
    return technical_view, impact_assessment, news_degraded


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
