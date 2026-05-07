"""Synthesizer agent — combines TechnicalView + ImpactAssessment into a Prediction.

ARCHITECTURE
============
LlmAgent with NO tools and `output_schema=Prediction`. The agent is pure
reasoning over already-gathered evidence:

    SynthesisInput (JSON) ──> [LLM with Prediction schema] ──> Prediction

Why no tools: this agent's job is synthesis, not gathering. All evidence
arrives in the prompt as a SynthesisInput. Adding tools would tempt the
LLM to "go fetch more" and break the gather/synthesize separation that
makes the predictor cheap and parallelizable.

Why output_schema=Prediction: we need a typed object back, not prose.
ADK + the underlying provider's structured-output mode forces the LLM
to emit JSON matching the Prediction schema. Pydantic then validates
direction-specific level ordering, range constraints, and the rest.
ADK retries internally on schema-violation up to its default budget.

WHY SEPARATE PROMPT FILE
========================
prompt.py owns the system instruction + per-call prompt builder.
This file owns the agent's wiring (model, schema, name). Splitting
keeps prompt iteration clean and lets tests assert on prompt content
without importing the LLM-construction machinery.

CLI ENTRY POINT
===============
`root_agent` at module bottom is what `adk run synthesizer` /
`adk web` discovers. Keeping it here (not deferring to a sub-package
imports module) matches the news_impact / technical_agent shape.
"""
from __future__ import annotations

from google.adk.agents import LlmAgent

from price_predictor.agents.synthesizer.prompt import SYSTEM_INSTRUCTION
from price_predictor.llm.factory import make_resilient_model
from price_predictor.prediction.schema import Prediction


def make_synthesizer_agent() -> LlmAgent:
    """Build the synthesizer agent.

    Wiring choices:
        name='synthesizer'    — short identifier for ADK's session/
                                event tracking
        model=resilient chain — uses the 'agentic' profile (same as
                                news_impact); falls back across providers
                                if the primary fails
        instruction=SYSTEM_INSTRUCTION
                              — defines the agent's behavior; per-call
                                user prompts come from build_synth_prompt
        tools=[]              — synthesizer reasons; it does not gather.
                                Empty list is explicit (None would also
                                work, but [] documents the intent)
        output_schema=Prediction
                              — forces structured JSON output; ADK
                                validates with Pydantic and retries
                                on schema violation

    Returns:
        Fresh LlmAgent. Each call creates a new instance so tests can
        assert on the wiring without touching the module-level
        root_agent (which is built at import time for the ADK CLI).
    """
    return LlmAgent(
        name="synthesizer",
        description=(
            "Synthesizes technical and news evidence into a single "
            "calibrated price Prediction with entry/target/stop levels."
        ),
        model=make_resilient_model(profile="agentic"),
        instruction=SYSTEM_INSTRUCTION,
        tools=[],
        output_schema=Prediction,
    )


# ─────────────────────────────────────────────────────────────
# ADK CLI entry point
# Module-level instance required by `adk run synthesizer` etc.
# Built at import time — same pattern as news_impact.root_agent.
# ─────────────────────────────────────────────────────────────
root_agent = make_synthesizer_agent()
