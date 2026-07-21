"""News-impact analyzer ADK agent — public package interface.

ADK CLI (`adk run`, `adk web`) discovers `root_agent` from this module.
The predictor imports the gather layer, prompt builder, factory and
schemas directly.
"""
from price_predictor.agents.news_impact.agent import (
    Catalyst,
    ImpactAssessment,
    build_news_impact_prompt,
    make_news_impact_agent,
    neutral_impact_assessment,
    root_agent,
)
from price_predictor.agents.news_impact.gather import (
    NewsImpactInputs,
    gather_news_impact_inputs,
)

__all__ = [
    "Catalyst",
    "ImpactAssessment",
    "NewsImpactInputs",
    "build_news_impact_prompt",
    "gather_news_impact_inputs",
    "make_news_impact_agent",
    "neutral_impact_assessment",
    "root_agent",
]
