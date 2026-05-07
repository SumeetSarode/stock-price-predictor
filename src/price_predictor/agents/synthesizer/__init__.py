"""Synthesizer ADK agent — public package interface.

ADK CLI (`adk run synthesizer`, `adk web`) discovers `root_agent` from
this module. Tests + the predictor (commit 4) import the factory and
prompt builder directly.
"""
from price_predictor.agents.synthesizer.agent import (
    make_synthesizer_agent,
    root_agent,
)
from price_predictor.agents.synthesizer.prompt import (
    SYSTEM_INSTRUCTION,
    build_synth_prompt,
)

__all__ = [
    "SYSTEM_INSTRUCTION",
    "build_synth_prompt",
    "make_synthesizer_agent",
    "root_agent",
]
