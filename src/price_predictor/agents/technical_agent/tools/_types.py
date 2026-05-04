"""Shared types + tool-response schema for technical_agent tools.

Using TypedDict (not Pydantic) per the Step C design discussion -- matches
the existing tool convention while still giving editors useful type hints.
The TypedDict is structural only; ADK doesn't enforce it. We do.
"""
from __future__ import annotations

from typing import Literal, TypedDict

# ── Enums (just string literal types) ────────────────────────────
Signal    = Literal["bullish", "neutral", "bearish"]
Strength  = Literal["weak", "moderate", "strong"]
Status    = Literal["success", "error"]

# Valid sensitivity preset names. Mirrors VALID_PRESETS in analysis/__init__.
Sensitivity = Literal["standard", "sensitive", "smooth"]


class ToolErrorResponse(TypedDict):
    """Returned when a tool fails (bad ticker, fetch error, etc.)."""
    status: Literal["error"]
    error_message: str
    ticker: str


class ToolSuccessResponse(TypedDict, total=False):
    """Uniform success-shape across the four cluster tools.

    `total=False` because some fields (strength, warnings) are optional and
    cluster-specific. Every tool MUST populate the required fields below.

    REQUIRED: status, ticker, as_of, preset, signal, indicators, rationale
    OPTIONAL: strength, derived, warnings
    """
    status: Literal["success"]
    ticker: str
    as_of: str               # ISO date the analysis is "as of"
    preset: Sensitivity
    signal: Signal
    strength: Strength       # only emitted when meaningful (e.g. trend has ADX)
    indicators: dict         # raw indicator values (cluster-specific)
    derived: dict            # derived booleans / percentages
    rationale: list[str]     # human-readable bullets the LLM can weave in
    warnings: list[str]      # e.g. ["insufficient_history"]
