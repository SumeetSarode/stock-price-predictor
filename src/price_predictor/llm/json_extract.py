"""Extract a clean JSON object from a possibly-chatty LLM reply.

WHY THIS EXISTS
===============
Hosted models (Gemini, Groq) honour ADK's ``output_schema`` and return the
bare JSON object the parser expects. Small local / "thinking" models --
notably ``qwen3`` via Ollama, our last-resort offline tier -- do not. qwen3
narrates its reasoning as prose ("We need to produce Impact... neutral. Now
output.") and/or wraps it in ``<think>...</think>`` blocks, THEN (usually)
emits the JSON. Feeding that whole blob to ``model_validate_json`` blows up
on the first prose character.

This module strips the chatter so downstream Pydantic parsing sees only the
JSON object. It is deliberately model-agnostic: any provider that ever adds
a preamble, a markdown fence, or reasoning tags is handled the same way.

CLEANUP ORDER
=============
1. Remove ``<think>...</think>`` reasoning blocks (so a stray ``{`` inside
   the reasoning can't be mistaken for the real payload).
2. Unwrap a markdown code fence (```` ```json ... ``` ```` or bare ```` ``` ````).
3. Extract the FIRST balanced ``{...}`` object via brace matching that is
   aware of strings and escapes (so braces inside string values don't throw
   off the depth count).

If no JSON object can be located, the cleaned text is returned unchanged so
the caller's ``model_validate_json`` still raises an informative error
(rather than this helper swallowing the failure).
"""
from __future__ import annotations

import re

# Closed reasoning blocks: <think> ... </think> (case-insensitive, spans lines).
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# A markdown fence, optionally tagged 'json'. Capture the body.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json(raw: str | None) -> str:
    """Return the best-effort JSON object embedded in ``raw``.

    Never raises: on any ambiguity it returns the cleaned text so the
    caller's JSON parser produces the real, informative error.
    """
    if not raw:
        return ""
    text = _THINK_BLOCK.sub("", raw).strip()

    # If the model wrapped its answer in a markdown fence, prefer the body.
    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1).strip()

    obj = _first_json_object(text)
    return obj if obj is not None else text


def _first_json_object(text: str) -> str | None:
    """Return the first brace-balanced ``{...}`` substring, or None.

    String-aware: quotes toggle an 'inside string' state so that braces
    appearing inside string values (e.g. a rationale containing '{') do not
    corrupt the depth count. Backslash escapes inside strings are honoured.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    # Unbalanced (truncated response): let the caller's parser complain.
    return None
