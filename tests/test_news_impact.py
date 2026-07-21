"""Tests for price_predictor.agents.news_impact (post-refactor).

Strategy:
    - Verify the Pydantic schemas reject invalid inputs (unchanged).
    - Verify build_news_impact_prompt() renders gathered inputs faithfully.
    - Structural smoke for make_news_impact_agent() -- now TOOL-LESS.

Data-gathering behaviour (shaping, look-ahead, soft-fail) lives in
gather.py and is covered by test_gather.py. LLM behaviour is verified via
    uv run adk run price_predictor.agents.news_impact
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from price_predictor.agents.news_impact import (
    Catalyst,
    ImpactAssessment,
    NewsImpactInputs,
    build_news_impact_prompt,
    make_news_impact_agent,
    root_agent,
)


# ═════════════════════════════════════════════════════════════
# Catalyst schema
# ═════════════════════════════════════════════════════════════
class TestCatalystSchema:
    def test_happy(self):
        c = Catalyst(
            description="Q4 EPS beat consensus by 12%",
            source="news",
            impact="positive",
        )
        assert c.source == "news"

    def test_description_too_short(self):
        with pytest.raises(ValidationError):
            Catalyst(description="short", source="news", impact="positive")

    def test_invalid_source(self):
        with pytest.raises(ValidationError):
            Catalyst(
                description="A perfectly valid description here",
                source="twitter",  # not allowed
                impact="positive",
            )

    def test_invalid_impact(self):
        with pytest.raises(ValidationError):
            Catalyst(
                description="A perfectly valid description here",
                source="news",
                impact="meh",  # not allowed
            )

    @pytest.mark.parametrize("impact", ["positive", "negative", "neutral", "mixed"])
    def test_all_four_impact_values_accepted(self, impact):
        """Regression: 'neutral' was missing, causing real Gemini calls to
        fail mid-stream when the model assessed a catalyst as neutral
        (perfectly reasonable -- e.g. a CEO interview with no new info).
        Schema must include all four; the *prompt* tells the LLM when to
        use each. See agent.py docstring on Catalyst.impact.
        """
        c = Catalyst(
            description="A perfectly valid description here",
            source="news",
            impact=impact,
        )
        assert c.impact == impact


class TestImpactAssessmentSchema:
    def _valid_kwargs(self) -> dict:
        return {
            "ticker": "RELIANCE",
            "sentiment": "bullish",
            "confidence": 0.7,
            "estimated_pct_move": 3.5,
            "reasoning": (
                "Recent news coverage shows three positive articles about new "
                "retail expansion. Filings indicate an upcoming board meeting. "
                "Price action over the past month has been steadily positive."
            ),
            "catalysts": [
                Catalyst(
                    description="Retail expansion announcement",
                    source="news",
                    impact="positive",
                ),
            ],
        }

    def test_happy(self):
        a = ImpactAssessment(**self._valid_kwargs())
        assert a.sentiment == "bullish"
        assert len(a.catalysts) == 1

    def test_empty_catalysts_allowed(self):
        kwargs = self._valid_kwargs()
        kwargs["catalysts"] = []
        a = ImpactAssessment(**kwargs)
        assert a.catalysts == []

    def test_confidence_out_of_range(self):
        kwargs = self._valid_kwargs()
        kwargs["confidence"] = 1.5
        with pytest.raises(ValidationError):
            ImpactAssessment(**kwargs)

    def test_estimated_move_clamped(self):
        kwargs = self._valid_kwargs()
        kwargs["estimated_pct_move"] = 50.0  # too extreme
        with pytest.raises(ValidationError):
            ImpactAssessment(**kwargs)

    def test_reasoning_too_short(self):
        kwargs = self._valid_kwargs()
        kwargs["reasoning"] = "too short"  # 9 chars, below floor of 20
        with pytest.raises(ValidationError):
            ImpactAssessment(**kwargs)

    def test_reasoning_accepts_honest_short_no_data_answer(self):
        """Regression: when all sources fail, the agent's honest short
        reasoning ('All sources empty; no evidence to assess.') must be
        accepted. Schemas encode invariants, not preferences. Length
        quality is steered by the prompt; the floor only rejects garbage.
        """
        kwargs = self._valid_kwargs()
        kwargs["reasoning"] = "All sources empty; no evidence to assess."
        kwargs["catalysts"] = []  # honest: no data, no catalysts
        a = ImpactAssessment(**kwargs)
        assert len(a.reasoning) < 100, "This is the case the old schema rejected"
        assert len(a.reasoning) >= 20, "...but still above the new garbage floor"

    def test_too_many_catalysts(self):
        kwargs = self._valid_kwargs()
        kwargs["catalysts"] = [
            Catalyst(
                description=f"Catalyst number {i} description goes here",
                source="news", impact="positive",
            )
            for i in range(11)
        ]
        with pytest.raises(ValidationError):
            ImpactAssessment(**kwargs)


# ═════════════════════════════════════════════════════════════
# build_news_impact_prompt
# ═════════════════════════════════════════════════════════════
def _inputs(**over) -> NewsImpactInputs:
    base = dict(
        ticker="INFY.NS",
        company_name="Infosys",
        sector="Technology",
        window_start="2026-01-01",
        window_end="2026-01-08",
        company_news=[{"title": "Infosys wins big deal", "published_at": "2026-01-05",
                       "source": "ET", "url": "http://x/1"}],
        sector_news=[{"title": "Indian IT sector rallies", "published_at": "2026-01-04",
                      "source": "Mint", "url": "http://x/2"}],
        filings=[{"kind": "announcement", "announced_at": "2026-01-03",
                  "event_type": "results", "subject": "Q3 board meeting"}],
        estimates={"has_coverage": True, "next_quarter_eps_consensus": 12.5,
                   "current_price": 100.0, "price_target_mean": 120.0},
        prices={"status": "success", "last_close": 100.0, "pct_change_30d": 4.2},
        errors=[],
    )
    base.update(over)
    return NewsImpactInputs(**base)


class TestBuildPrompt:
    def test_includes_ticker_and_company(self):
        p = build_news_impact_prompt(_inputs())
        assert "INFY.NS" in p
        assert "Infosys" in p

    def test_includes_sector_line(self):
        assert "Technology" in build_news_impact_prompt(_inputs())

    def test_renders_company_and_sector_news(self):
        p = build_news_impact_prompt(_inputs())
        assert "Infosys wins big deal" in p
        assert "Indian IT sector rallies" in p

    def test_renders_filings_and_estimates_and_prices(self):
        p = build_news_impact_prompt(_inputs())
        assert "Q3 board meeting" in p
        assert "120.0" in p           # price target mean
        assert "last_close=100.0" in p

    def test_empty_sections_render_gracefully(self):
        p = build_news_impact_prompt(
            _inputs(company_news=[], sector_news=[], filings=[],
                    estimates=None, prices=None, sector=None)
        )
        assert "(none found)" in p or "(none)" in p
        assert "(unavailable" in p
        assert "(unknown)" in p

    def test_data_gaps_surfaced(self):
        p = build_news_impact_prompt(_inputs(errors=["filings unavailable: X"]))
        assert "Data gaps" in p
        assert "filings unavailable" in p

    def test_estimates_skipped_note_in_backtest(self):
        # None estimates (replay) → prompt must say unavailable/skipped.
        p = build_news_impact_prompt(_inputs(estimates=None))
        assert "unavailable" in p.lower() or "skipped" in p.lower()


# ═════════════════════════════════════════════════════════════
# Agent factory smoke (no LLM calls) — now TOOL-LESS
# ═════════════════════════════════════════════════════════════
class TestAgentFactory:
    def test_make_news_impact_agent_structure(self):
        agent = make_news_impact_agent()
        assert agent.name == "news_impact"
        assert "Indian stock" in agent.description
        # Refactor: the synthesizer has NO tools — data is pre-gathered.
        assert not agent.tools
        # Structured output still bound.
        assert agent.output_schema is ImpactAssessment

    def test_root_agent_module_level(self):
        """root_agent must exist at module level for ADK CLI discovery."""
        assert root_agent is not None
        assert root_agent.name == "news_impact"

    def test_instruction_forbids_fabrication(self):
        """The synthesis prompt must anchor the model to given evidence."""
        prompt = make_news_impact_agent().instruction
        assert "NEVER fabricate" in prompt
        assert "sector" in prompt.lower()  # sector-aware judgment
