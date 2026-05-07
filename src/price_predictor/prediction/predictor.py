"""predict() — the per-stock prediction orchestrator (Step 3.4.2 commit 4).

THE BIG PICTURE
===============
This module wires the gather and synthesis phases into one async function:

    predict(ticker, horizon)
        |
        +-- PHASE 1 (parallel via asyncio.gather):
        |     compose_technical_view(ticker)      [Layer-1 cluster tools]
        |     run_news_impact_agent(ticker)       [Agent #1]
        |
        +-- Bundle into SynthesisInput
        |
        +-- PHASE 2 (sequential, depends on gather):
        |     run_synthesizer_agent(synthesis_input)  [Agent #2]
        |
        +-- return Prediction

This is the FIRST place we invoke ADK agents programmatically (vs via
`adk run` CLI). The agent-call helpers (`run_news_impact_agent`,
`run_synthesizer_agent`) hide the Runner / SessionService / Event-stream
plumbing behind a clean async function-call shape.

DEGRADATION POLICY (commit 4 baseline; revisited in commit 5)
=============================================================
- Technical failure  -> raise PredictionError. Technicals are core; a
                        prediction missing them is unreliable.
- News failure       -> raise PredictionError. (Commit 5 will replace
                        this with graceful degradation: log warning,
                        substitute a 'neutral' ImpactAssessment, mark
                        the Prediction's analysis_basis appropriately.)
- Synthesizer failure -> bubble up. ADK already retries on schema
                        violations; if it still fails, the caller
                        needs to know.

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
from price_predictor.prediction.schema import Prediction

# Horizon literal mirrors PredictionHorizon enum values. We accept the
# raw string for ergonomic API; the synthesizer agent passes it through
# to Prediction unchanged.
Horizon = Literal["intraday", "short", "medium", "long"]

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
async def predict(
    ticker: str,
    horizon: Horizon = "short",
    *,
    sensitivity: Literal["standard", "sensitive", "smooth"] = "standard",
) -> Prediction:
    """Produce one Prediction for one stock.

    GATHER PHASE: technicals + news in parallel.
    SYNTHESIS PHASE: hand both to the synthesizer agent.

    Args:
        ticker: Stock symbol in any form ('reliance', 'RELIANCE.NS',
            'AAPL'). Resolved to canonical via the KB.
        horizon: Prediction window. Default 'short' (1-5 trading days).
        sensitivity: Indicator-cluster preset. Default 'standard'.

    Returns:
        A validated Prediction.

    Raises:
        PredictionError: any phase failed in a way that makes the
            prediction unreliable. Wraps the underlying cause.
    """
    canonical = _resolve_ticker(ticker)
    as_of = datetime.now(IST)
    logger.info(
        f"predict() start: ticker={canonical} horizon={horizon} "
        f"sensitivity={sensitivity}"
    )

    # ── PHASE 1: GATHER (parallel) ──────────────────────────
    technical_task = compose_technical_view(canonical, sensitivity=sensitivity)
    news_task = run_news_impact_agent(canonical)

    try:
        technical_view, impact_assessment = await asyncio.gather(
            technical_task, news_task,
        )
    except TechnicalViewError as e:
        # Technicals are core. No degradation in commit 4 (commit 5
        # revisits news degradation; technicals stay fail-loud).
        raise PredictionError(
            f"Technical analysis failed for {canonical}: {e}"
        ) from e
    except PredictionError:
        # News-side failure - re-raise as-is (already wrapped).
        raise

    logger.info(
        f"gather done: technical_view bars={technical_view.bars_used} "
        f"signals=trend:{technical_view.trend.signal}/"
        f"momentum:{technical_view.momentum.signal}/"
        f"volatility:{technical_view.volatility.signal}/"
        f"levels:{technical_view.levels.signal} | "
        f"impact sentiment={impact_assessment.sentiment} "
        f"confidence={impact_assessment.confidence:.2f}"
    )

    # ── Bundle ─────────────────────────────────────────────
    synthesis_input = SynthesisInput(
        ticker=canonical,
        horizon=horizon,
        as_of=as_of,
        technical_view=technical_view,
        impact_assessment=impact_assessment,
        # Audit trail: news_impact ran first, synthesizer appends itself
        # logically. We pass the news tag here; the synthesizer's tag is
        # added below after Prediction comes back (see model_chain note
        # in Prediction schema).
        model_chain=(_NEWS_MODEL_TAG,),
    )

    # ── PHASE 2: SYNTHESIZE (with hallucination guardrails) ──
    prediction = await synthesize_with_guardrails(synthesis_input)

    # The synthesizer's prompt instructs it to copy model_chain verbatim
    # from the input. We append the synthesizer tag AFTER the call by
    # constructing a new Prediction (frozen models) - this guarantees
    # the audit trail reflects reality regardless of LLM compliance.
    final = prediction.model_copy(
        update={"model_chain": (*prediction.model_chain, _SYNTH_MODEL_TAG)}
    )

    logger.info(
        f"predict() done: ticker={canonical} direction={final.direction.value} "
        f"confidence={final.confidence:.2f} target={final.target.value:.2f} "
        f"stop={final.stop_loss.value:.2f}"
    )
    return final
