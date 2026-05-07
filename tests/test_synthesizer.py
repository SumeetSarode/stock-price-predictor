"""Unit tests for the synthesizer agent.

SCOPE
=====
These are wiring + prompt-construction tests. NO real LLM is invoked.
End-to-end behavior (LLM actually produces a coherent Prediction) is
covered by the integration test in commit 6 (marker-gated).

Why this split: LLM calls cost money and need API keys / network.
Wiring tests should be fast, deterministic, and hermetic.
"""
from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from google.adk.agents import LlmAgent

from price_predictor.agents.news_impact import Catalyst, ImpactAssessment
from price_predictor.agents.synthesizer import (
    SYSTEM_INSTRUCTION,
    build_synth_prompt,
    make_synthesizer_agent,
    root_agent,
)
from price_predictor.prediction.inputs import (
    ClusterView,
    SynthesisInput,
    TechnicalView,
)
from price_predictor.prediction.schema import Prediction


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────
def _sample_synthesis_input() -> SynthesisInput:
    """Minimal valid SynthesisInput for prompt + wiring tests.

    Carries one catalyst per source so tests can assert that the prompt
    surfaces them. Mirrors the fixture pattern in test_prediction_inputs
    but kept local to keep this file self-contained.
    """
    cv = lambda name, signal: ClusterView(  # noqa: E731 (terse fixture builder)
        name=name, signal=signal, indicators={"close": 1455.0},
    )
    tv = TechnicalView(
        ticker="RELIANCE.NS",
        as_of=datetime(2026, 4, 28, tzinfo=ZoneInfo("Asia/Kolkata")).date(),
        close_price=1455.0,
        bars_used=400,
        sensitivity="standard",
        trend=cv("trend", "bullish"),
        momentum=cv("momentum", "bullish"),
        volatility=cv("volatility", "neutral"),
        levels=cv("levels", "bullish"),
    )
    ia = ImpactAssessment(
        ticker="RELIANCE.NS",
        sentiment="bullish",
        confidence=0.7,
        estimated_pct_move=2.5,
        reasoning="Q3 beat plus margin guidance lift.",
        catalysts=[
            Catalyst(
                description="Q3 earnings beat consensus by 12%",
                source="news",
                impact="positive",
            ),
            Catalyst(
                description="Board approved 1:1 stock split filing",
                source="filing",
                impact="positive",
            ),
        ],
    )
    return SynthesisInput(
        ticker="RELIANCE.NS",
        horizon="short",
        as_of=datetime(2026, 4, 28, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
        technical_view=tv,
        impact_assessment=ia,
        model_chain=("gemini-2.5-flash",),
    )


# ─────────────────────────────────────────────────────────────
# 1. Agent factory wiring
# ─────────────────────────────────────────────────────────────
class TestAgentFactory:
    def test_factory_returns_llm_agent(self):
        agent = make_synthesizer_agent()
        assert isinstance(agent, LlmAgent)

    def test_agent_name_is_synthesizer(self):
        agent = make_synthesizer_agent()
        assert agent.name == "synthesizer"

    def test_agent_has_no_tools(self):
        """Synthesizer reasons over given evidence; it MUST NOT gather more.

        Empty tools list (or None) is the load-bearing invariant. If a
        future change adds tools, this test will fire and force a
        deliberate decision.
        """
        agent = make_synthesizer_agent()
        # ADK normalizes empty/None tools — accept either as 'no tools'
        assert not agent.tools

    def test_output_schema_is_prediction(self):
        """structured-output enforcement is wired.

        Without this, the LLM returns prose and the predictor breaks.
        """
        agent = make_synthesizer_agent()
        assert agent.output_schema is Prediction

    def test_instruction_is_system_instruction(self):
        """The agent is wired to OUR prompt, not ADK's default.

        Catches accidental wipes of the instruction during refactors.
        """
        agent = make_synthesizer_agent()
        assert agent.instruction == SYSTEM_INSTRUCTION


# ─────────────────────────────────────────────────────────────
# 2. SYSTEM_INSTRUCTION content invariants
# ─────────────────────────────────────────────────────────────
class TestSystemInstruction:
    """Lightweight content checks — not testing prompt quality, just
    that key concepts the prompt MUST cover are present.

    These are a regression net: if someone deletes the calibration
    section by accident, this test fires.
    """

    def test_mentions_prediction_schema(self):
        assert "Prediction" in SYSTEM_INSTRUCTION

    def test_lists_all_four_clusters(self):
        for cluster in ("trend", "momentum", "volatility", "levels"):
            assert cluster in SYSTEM_INSTRUCTION, (
                f"system instruction must mention {cluster} cluster"
            )

    def test_explains_direction_invariants(self):
        """Direction-specific level rules MUST be in the prompt.

        Without these, the LLM emits invalid JSON, schema rejects, ADK
        retries — works but wastes calls. Better to teach upfront.
        """
        assert "BULLISH" in SYSTEM_INSTRUCTION
        assert "BEARISH" in SYSTEM_INSTRUCTION
        assert "NEUTRAL" in SYSTEM_INSTRUCTION

    def test_provides_confidence_calibration(self):
        assert "calibration" in SYSTEM_INSTRUCTION.lower()
        # Must include concrete anchors, not just hand-waving
        assert "0.85" in SYSTEM_INSTRUCTION

    def test_warns_about_anti_patterns(self):
        """We've called out specific failure modes by name."""
        assert "ANTI-PATTERNS" in SYSTEM_INSTRUCTION


# ─────────────────────────────────────────────────────────────
# 3. build_synth_prompt — per-call user prompt
# ─────────────────────────────────────────────────────────────
class TestBuildSynthPrompt:
    def test_returns_string(self):
        prompt = build_synth_prompt(_sample_synthesis_input())
        assert isinstance(prompt, str)
        assert len(prompt) > 100  # non-trivial content

    def test_embeds_ticker_and_horizon(self):
        prompt = build_synth_prompt(_sample_synthesis_input())
        assert "RELIANCE.NS" in prompt
        assert "short" in prompt

    def test_embeds_close_price_and_signals(self):
        """Critical evidence must reach the LLM verbatim."""
        prompt = build_synth_prompt(_sample_synthesis_input())
        assert "1455" in prompt           # close_price
        assert "bullish" in prompt         # at least one cluster signal
        assert "Q3 earnings beat" in prompt  # catalyst description

    def test_embedded_json_is_round_trippable(self):
        """The JSON blob in the prompt MUST be parseable.

        This is the contract the LLM relies on: malformed JSON in the
        prompt would confuse the LLM and silently degrade output.
        We extract the fenced JSON block and validate.
        """
        si = _sample_synthesis_input()
        prompt = build_synth_prompt(si)

        # Extract content between ```json and ```
        start = prompt.index("```json\n") + len("```json\n")
        end = prompt.index("\n```", start)
        json_blob = prompt[start:end]

        # Must be valid JSON
        parsed = json.loads(json_blob)
        # Must round-trip back to a SynthesisInput equal to the original
        rebuilt = SynthesisInput.model_validate(parsed)
        assert rebuilt == si

    def test_pretty_printed_for_log_readability(self):
        """We use indent=2 so prompt logs are eyeball-friendly."""
        prompt = build_synth_prompt(_sample_synthesis_input())
        # Pretty JSON has newlines between fields
        assert "\n  " in prompt


# ─────────────────────────────────────────────────────────────
# 4. CLI discoverability (root_agent is the ADK entry point)
# ─────────────────────────────────────────────────────────────
class TestRootAgent:
    def test_root_agent_is_llm_agent(self):
        assert isinstance(root_agent, LlmAgent)

    def test_root_agent_named_synthesizer(self):
        """`adk run synthesizer` matches on this name."""
        assert root_agent.name == "synthesizer"

    def test_root_agent_has_prediction_schema(self):
        assert root_agent.output_schema is Prediction
