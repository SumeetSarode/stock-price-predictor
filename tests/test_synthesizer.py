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
from price_predictor.agents.synthesizer.prompt import _render_per_horizon_table
from price_predictor.prediction.horizon_constants import (
    confidence_cap,
    entry_zone_pct,
    stop_atr_range,
    target_atr_range,
)
from price_predictor.prediction.inputs import (
    ClusterView,
    SynthesisInput,
    TechnicalView,
)
from price_predictor.prediction.schema import Prediction, PredictionHorizon


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
        horizon="weekly",
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

    # ──────────────────────────────────────────────────────────
    # MA crossover wiring — prompt must teach the LLM where to find
    # ma_crosses, how to cite it, and when NOT to.
    # ──────────────────────────────────────────────────────────
    def test_documents_ma_crosses_field_location(self):
        """Inputs section must explicitly point at trend.derived.ma_crosses."""
        assert "ma_crosses" in SYSTEM_INSTRUCTION
        assert "trend.derived" in SYSTEM_INSTRUCTION

    def test_documents_ma_cross_struct_shape(self):
        """L3 struct field names must be in the prompt so the LLM can
        unambiguously read them."""
        for field in ("current", "last_event", "bars_since_event",
                      "short_ma", "long_ma"):
            assert field in SYSTEM_INSTRUCTION, (
                f"prompt must teach LLM the {field!r} field of ma_crosses"
            )

    def test_teaches_golden_cross_naming_convention(self):
        """Only sma_50_200 + bullish gets the 'Golden Cross' name; the
        prompt must say so to prevent the LLM calling EMA-9/21 cross a
        Golden Cross (Murphy 1999 reserves the term)."""
        assert "Golden Cross" in SYSTEM_INSTRUCTION
        assert "Death Cross" in SYSTEM_INSTRUCTION
        # Must mention the canonical pair name
        assert "sma_50_200" in SYSTEM_INSTRUCTION

    def test_teaches_freshness_window(self):
        """The 5-bar freshness window must be in the prompt so the LLM
        only cites fresh crosses in contributing_signals."""
        # Either '5' bars or 'fresh' qualifier must appear with cross context
        ma_section = SYSTEM_INSTRUCTION.split("ma_crosses", 1)[1]
        assert "≤ 5" in ma_section or "<= 5" in ma_section or "5 bars" in ma_section

    def test_anti_pattern_against_inferring_cross_from_static_position(self):
        """The single most likely hallucination is 'close > SMA-200 so
        Golden Cross' — prompt must call this out explicitly."""
        # Anti-pattern about inferring a cross from above_sma_*
        assert "above_sma" in SYSTEM_INSTRUCTION
        assert "static" in SYSTEM_INSTRUCTION.lower() or (
            "NOT a cross" in SYSTEM_INSTRUCTION
        )

    def test_anti_pattern_against_citing_stale_crosses(self):
        """Stale crosses should appear in rationale prose only — NOT in
        contributing_signals. Prompt must say so."""
        assert "stale" in SYSTEM_INSTRUCTION.lower() or "STALE" in SYSTEM_INSTRUCTION


# ────────────────────────────────────────────
# 2b. PER-HORIZON RULES table (commit C of multi-horizon refactor)
# ────────────────────────────────────────────
class TestPerHorizonTable:
    """The prompt MUST teach the LLM the per-horizon rules — and those
    rules MUST be the same numbers the guardrails enforce. Anything
    else means the LLM is being told one thing and graded against
    another (the bug commit C was built to fix).

    These tests treat horizon_constants.py as the single source of
    truth: they read the helpers, then assert the rendered table and
    the SYSTEM_INSTRUCTION reflect those values. Tune in horizon_
    constants and these tests stay green.
    """

    def test_render_helper_returns_table_with_header(self):
        table = _render_per_horizon_table()
        assert "horizon" in table
        assert "stop ATR" in table
        assert "target ATR" in table
        assert "entry zone" in table
        assert "conf cap" in table

    def test_render_helper_includes_every_horizon(self):
        table = _render_per_horizon_table()
        for horizon in PredictionHorizon:
            assert horizon.value in table, (
                f"per-horizon table missing row for {horizon.value}"
            )

    @pytest.mark.parametrize("horizon", list(PredictionHorizon))
    def test_table_rows_match_horizon_constants(self, horizon: PredictionHorizon):
        """Every number in every row MUST match horizon_constants.

        This is the load-bearing single-source-of-truth check: if
        horizon_constants.py changes, the prompt updates automatically;
        if a developer ever hand-codes numbers into the prompt instead,
        this test will fire.
        """
        table = _render_per_horizon_table()

        s_lo, s_hi = stop_atr_range(horizon)
        t_lo, t_hi = target_atr_range(horizon)
        ez = entry_zone_pct(horizon)
        cap = confidence_cap(horizon)

        # Find the row for this horizon. Match by the column-padded
        # prefix to avoid "weekly" matching "biweekly" too.
        row_prefix = f"| {horizon.value:<8} |"
        rows = [line for line in table.splitlines() if line.startswith(row_prefix)]
        assert len(rows) == 1, f"expected one row for {horizon.value}, got {rows}"
        row = rows[0]

        # Each constant MUST appear in the row, formatted as the
        # renderer formats it. We check string presence (not parse the
        # row) so the test is robust to layout tweaks.
        assert f"{s_lo}–{s_hi}×ATR" in row, (
            f"stop ATR band {s_lo}–{s_hi} missing for {horizon.value}"
        )
        assert f"{t_lo}–{t_hi}×ATR" in row, (
            f"target ATR band {t_lo}–{t_hi} missing for {horizon.value}"
        )
        assert f"{ez*100:.1f}%" in row, (
            f"entry zone {ez*100:.1f}% missing for {horizon.value}"
        )
        assert f"{cap:.2f}" in row, (
            f"confidence cap {cap:.2f} missing for {horizon.value}"
        )

    def test_system_instruction_embeds_the_table(self):
        """The rendered table MUST appear inside SYSTEM_INSTRUCTION."""
        assert _render_per_horizon_table() in SYSTEM_INSTRUCTION

    def test_system_instruction_has_per_horizon_section(self):
        assert "PER-HORIZON RULES" in SYSTEM_INSTRUCTION

    @pytest.mark.parametrize("horizon", list(PredictionHorizon))
    def test_system_instruction_mentions_each_horizon_cap(
        self, horizon: PredictionHorizon,
    ):
        """The cap value for each horizon must literally appear in the
        prompt — belt-and-braces over the table-embedding test.
        """
        cap_str = f"{confidence_cap(horizon):.2f}"
        assert cap_str in SYSTEM_INSTRUCTION, (
            f"per-horizon cap {cap_str} for {horizon.value} not in prompt"
        )

    # ────────────────────────────────────────────
    # Regression nets: the dead hand-wavy phrasing MUST stay deleted.
    # If someone re-adds vague language about "tighter for daily,
    # wider for monthly," the per-horizon table can be silently
    # contradicted. These tests prevent that drift.
    # ────────────────────────────────────────────
    @pytest.mark.parametrize("dead_phrase", [
        "tighter for daily, wider for monthly",
        "±0.5%\n                            for daily/weekly",
        "close_price ∓ ~1*ATR is a sane",
    ])
    def test_dead_handwavy_phrasing_removed(self, dead_phrase: str):
        assert dead_phrase not in SYSTEM_INSTRUCTION, (
            f"vague phrasing {dead_phrase!r} re-introduced — use the "
            "PER-HORIZON RULES table instead."
        )


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
        assert "weekly" in prompt

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
