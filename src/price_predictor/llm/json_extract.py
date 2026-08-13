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
# An UNCLOSED reasoning block: '<think>' with no matching '</think>'. Happens
# when a reasoning model is truncated mid-thought (hit max_tokens) or just
# omits the closing tag. We only drop the TAG ITSELF, never the text after
# it -- the answer usually follows the reasoning, so deleting to end-of-string
# would throw away the very object we're looking for. Removing just the tag
# leaves the reasoning as ordinary prose, which the largest-object scan in
# _first_json_object() already handles correctly.
_UNCLOSED_THINK = re.compile(r"</?think>", re.IGNORECASE)
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
    # Any <think>/</think> still present is unpaired (truncated or malformed).
    # Strip the stray tags only -- see _UNCLOSED_THINK for why we must NOT
    # delete the text after them.
    text = _UNCLOSED_THINK.sub("", text).strip()

    # If the model wrapped its answer in a markdown fence, prefer the body.
    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1).strip()

    obj = _first_json_object(text)
    return obj if obj is not None else text


def _first_json_object(text: str) -> str | None:
    """Return the best candidate brace-balanced ``{...}`` substring, or None.

    Scans EVERY balanced object in the text and returns the largest one,
    rather than blindly taking the first.

    WHY NOT JUST THE FIRST: reasoning models narrate before answering, and
    that narration frequently contains braces -- "We need {x} first. Final:
    {...real answer...}". Taking the first match returns ``{x}``, which then
    fails Pydantic validation and surfaces as "unparsable LLM output" even
    though a perfectly good object was sitting right there. The real payload
    is the substantial one, so size is the discriminator: a stray brace in
    prose is tiny next to a populated Prediction/ImpactAssessment object.

    Still string-aware: quotes toggle an 'inside string' state so braces
    inside string values (e.g. a rationale containing '{') don't corrupt the
    depth count. Backslash escapes inside strings are honoured.
    """
    candidates: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        obj = _balanced_from(text, i)
        if obj is None:
            # Unbalanced from here (truncated tail) -- no later start can
            # close either, so stop scanning.
            break
        candidates.append(obj)
        # Resume AFTER this object so we find siblings, not its own children.
        i += len(obj)
    if not candidates:
        return None
    return max(candidates, key=len)


def _balanced_from(text: str, start: int) -> str | None:
    """Return the balanced ``{...}`` beginning at ``start``, or None."""
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
