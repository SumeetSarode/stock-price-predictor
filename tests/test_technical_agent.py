"""Tests for price_predictor.agents.technical_agent.

Strategy:
    - We do NOT test LLM behavior here. Those checks happen manually via:
          uv run adk run price_predictor.agents.technical_agent
      and via the smoke script:
          uv run python scripts/smoke_test_tools.py
    - We DO test:
        * Factory returns an LlmAgent with the right name / tools / model
        * root_agent exists at module level (ADK CLI discoverability)
        * Instruction contains the key contracts the LLM needs to follow
          (so a prompt edit that drops a critical rule is caught in CI)

WHY ASSERT ON PROMPT CONTENT?
=============================
The instruction is the agent's API surface as much as the tools list is.
A casual edit can silently drop a critical rule (e.g., "no buy/sell
recommendations"). A regression test pinning the key phrases catches that
edit BEFORE we ship a recommendation-giving robo-advisor by accident.

We pin BEHAVIOR-CRITICAL substrings only -- not the whole prompt. That
way prompt copy-edits don't break the tests, but rule-removals do.
"""
from __future__ import annotations

import pytest

from price_predictor.agents.technical_agent import (
    TECHNICAL_AGENT_INSTRUCTION,
    make_technical_agent,
    root_agent,
)
from price_predictor.agents.technical_agent.tools.get_levels import get_levels
from price_predictor.agents.technical_agent.tools.get_momentum import get_momentum
from price_predictor.agents.technical_agent.tools.get_trend import get_trend
from price_predictor.agents.technical_agent.tools.get_volatility import get_volatility


# ─────────────────────────────────────────────────────────────
# Factory smoke
# ─────────────────────────────────────────────────────────────
class TestMakeTechnicalAgent:
    def test_returns_llm_agent_with_correct_name(self):
        agent = make_technical_agent()
        assert agent.name == "technical_agent"

    def test_description_mentions_purpose_and_disclaimer(self):
        agent = make_technical_agent()
        # Description is what other agents see when they look at us.
        # It MUST mention what we do AND what we don't do.
        assert "technical" in agent.description.lower()
        assert "NSE" in agent.description
        assert "buy/sell" in agent.description.lower() or \
               "recommend" in agent.description.lower()

    def test_has_all_four_cluster_tools(self):
        """Each cluster from Step C MUST be wired in."""
        agent = make_technical_agent()
        assert len(agent.tools) == 4
        # Tool identity check -- these are the actual functions, not
        # FunctionTool wrappers (ADK auto-wraps callables).
        assert get_trend in agent.tools
        assert get_momentum in agent.tools
        assert get_volatility in agent.tools
        assert get_levels in agent.tools

    def test_uses_resilient_model(self):
        """Agents MUST use the resilient fallback chain, not a single model.

        Quoting llm/factory.py:
            'Agents MUST use make_resilient_model(profile=...).
             Never call make_model() directly from an agent.'
        """
        from price_predictor.llm.resilient import ResilientModel
        agent = make_technical_agent()
        assert isinstance(agent.model, ResilientModel)

    def test_returns_fresh_instance_every_call(self):
        """Factory pattern: each call must produce a new agent.

        Important for tests that mutate agent state and expect isolation.
        """
        a1 = make_technical_agent()
        a2 = make_technical_agent()
        assert a1 is not a2


# ─────────────────────────────────────────────────────────────
# root_agent discovery (ADK CLI contract)
# ─────────────────────────────────────────────────────────────
class TestRootAgent:
    def test_root_agent_exists(self):
        assert root_agent is not None

    def test_root_agent_name_matches(self):
        """ADK uses agent.name in routing logs and the web UI title bar."""
        assert root_agent.name == "technical_agent"

    def test_root_agent_has_four_tools(self):
        assert len(root_agent.tools) == 4


# ─────────────────────────────────────────────────────────────
# Instruction content (pin behavior-critical substrings)
# ─────────────────────────────────────────────────────────────
class TestInstructionContent:
    """Each test pins ONE behavior-critical rule.

    These are NOT testing the LLM follows the rules (that's a manual
    eval). They're testing that an editor doesn't ACCIDENTALLY DELETE
    a rule that the project depends on.
    """

    def test_no_recommendations_rule_present(self):
        """Hard rule: this agent does NOT give buy/sell advice."""
        # Two phrasings -- prompt may use either; both should always exist.
        assert "buy/sell" in TECHNICAL_AGENT_INSTRUCTION.lower()
        assert "do not give" in TECHNICAL_AGENT_INSTRUCTION.lower() or \
               "not give buy/sell" in TECHNICAL_AGENT_INSTRUCTION.lower()

    def test_all_four_tools_documented(self):
        """The prompt must teach the LLM what each tool does."""
        for tool_name in ("get_trend", "get_momentum", "get_volatility", "get_levels"):
            assert tool_name in TECHNICAL_AGENT_INSTRUCTION, \
                f"prompt is missing tool documentation for {tool_name}"

    def test_default_rule_for_general_questions(self):
        """For ambiguous 'how does X look?' questions, call all four."""
        # Pin the DEFAULT phrase so a prompt edit can't quietly drop it
        assert "DEFAULT" in TECHNICAL_AGENT_INSTRUCTION
        assert "ALL FOUR" in TECHNICAL_AGENT_INSTRUCTION or \
               "all four" in TECHNICAL_AGENT_INSTRUCTION

    def test_sensitivity_default_is_standard(self):
        """If unclear, the LLM must pick 'standard', not guess."""
        assert "'standard'" in TECHNICAL_AGENT_INSTRUCTION
        assert "NEVER guess" in TECHNICAL_AGENT_INSTRUCTION or \
               "never guess" in TECHNICAL_AGENT_INSTRUCTION.lower()

    def test_warnings_are_surfaced(self):
        """Tools return warnings (e.g. pattern_signal_conflict) -- the
        agent must NEVER hide them from the user."""
        assert "warning" in TECHNICAL_AGENT_INSTRUCTION.lower()
        # Hide-warnings is a recurring LLM failure mode -- pin the rule.
        assert "Don't hide" in TECHNICAL_AGENT_INSTRUCTION or \
               "don't hide" in TECHNICAL_AGENT_INSTRUCTION.lower() or \
               "do not hide" in TECHNICAL_AGENT_INSTRUCTION.lower()

    def test_self_recovery_on_suggested_ticker(self):
        """When tools return suggested_ticker (e.g. HDFC -> HDFCBANK.NS),
        agent retries automatically WITHOUT bothering the user."""
        assert "suggested_ticker" in TECHNICAL_AGENT_INSTRUCTION
        assert "without asking" in TECHNICAL_AGENT_INSTRUCTION.lower() or \
               "without bothering" in TECHNICAL_AGENT_INSTRUCTION.lower() or \
               "self-recover" in TECHNICAL_AGENT_INSTRUCTION.lower()

    def test_no_fabrication_rule(self):
        """Hard rule: don't make up numbers for tools you didn't call."""
        assert "fabricate" in TECHNICAL_AGENT_INSTRUCTION.lower() or \
               "make up" in TECHNICAL_AGENT_INSTRUCTION.lower()

    def test_response_format_uses_rs_not_rupee_symbol(self):
        """Indian stocks: use 'Rs' (LLM-safe ASCII) not the Rupee symbol
        (which can render badly in some terminals)."""
        assert "Rs" in TECHNICAL_AGENT_INSTRUCTION
