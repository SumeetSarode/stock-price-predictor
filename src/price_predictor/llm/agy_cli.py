"""Antigravity CLI (``agy``) as a chain model -- Google's own scripting route.

WHY THIS EXISTS
===============
Antigravity's free plan includes strong models (Gemini 3.x, ...) but exposes
NO API key and forbids reusing its OAuth from third-party tools (its terms
call that a breach). What Google DOES document is its own CLI in headless
mode (https://antigravity.google/docs/cli/headless/): ``agy -p "<prompt>"
--output-format json`` -- "script agent tasks, integrate with CI pipelines,
and capture machine-readable output". This module drives exactly that, and
nothing else: no token extraction, no cookie replay, no private endpoints.

WHAT IT RELIES ON (all from the headless docs)
==============================================
- ``-p <prompt> --output-format json`` prints ONE JSON envelope on stdout;
  diagnostics (auth prompts, progress) go to stderr.
- Envelope fields used: ``status``, ``response``, ``error``, ``usage``.
- Auth is the CLI's own cached sign-in (done once, interactively, by the
  user: run ``agy``). Unauthenticated runs exit with "authentication
  required" instead of hanging.

DESIGN DECISIONS
================
- It's an AGENT, not a raw completion endpoint, so we (a) run it in an
  EMPTY temp directory, (b) prefix the prompt telling it to use no tools.
  News headlines are untrusted text; this is defence in depth against a
  prompt-injected headline trying to make a tool-capable agent act.
  Whether the CLI can hard-disable tools is NOT documented, so we don't
  pretend we've done it.
- Prompt goes via argv with no shell (create_subprocess_exec): no shell
  injection, and a prompt can't be parsed as a flag because the preamble
  guarantees it starts with a letter.
- "Not signed in / quota gone / binary missing" raise AgyUnavailableError,
  which resilient.py files under MODEL_UNAVAILABLE (1h cooldown). It must
  NOT be AuthenticationError -- that is STRUCTURAL and would kill the whole
  chain over an optional provider.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import shutil
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from litellm.exceptions import Timeout

logger = logging.getLogger(__name__)

AGY_PROVIDER = "agy"
DEFAULT_AGY_TIMEOUT_S = 120.0
_FALLBACK_INSTALL_PATH = Path.home() / ".local" / "bin" / "agy"

# Prepended to every prompt. See DESIGN DECISIONS: untrusted text + agent.
_NO_TOOLS_PREAMBLE = (
    "You are acting as a plain text-generation endpoint. Do NOT use any "
    "tools, read or write files, or run commands. Reply with ONLY the "
    "output requested below.\n\n"
)

# stderr / error wording that means "this tier can't serve us for a while"
# (vs. a one-off failure). Conservative on purpose: unmatched failures fall
# to the catch-all in resilient.py, which still falls over -- just with the
# short cooldown.
_UNAVAILABLE_PATTERN = re.compile(
    r"authentication required|not authenticated|sign.?in|log.?in|"
    r"quota|rate.?limit|exhaust|usage limit",
    re.IGNORECASE,
)


class AgyCliError(RuntimeError):
    """A one-off failure running agy (bad status, unparseable output)."""


class AgyUnavailableError(Exception):
    """agy can't serve requests for a while: not installed, not signed in,
    or its quota is spent. Cooled down for an hour by ResilientModel."""


def find_agy_binary(explicit: str = "") -> str | None:
    """Locate the agy executable, or None if it isn't installed.

    Order: explicit path (AGY_PATH) -> PATH lookup -> the installer's
    default location (~/.local/bin/agy, which isn't always on PATH).
    """
    if explicit:
        return explicit if Path(explicit).is_file() else None
    on_path = shutil.which(AGY_PROVIDER)
    if on_path:
        return on_path
    return str(_FALLBACK_INSTALL_PATH) if _FALLBACK_INSTALL_PATH.is_file() else None


def _content_text(content: types.Content | None) -> str:
    return "\n".join(
        p.text for p in (getattr(content, "parts", None) or []) if getattr(p, "text", None)
    )


def build_prompt(llm_request: LlmRequest) -> str:
    """Flatten an ADK request (system instruction + turns) into one prompt."""
    system = getattr(llm_request.config, "system_instruction", None)
    system_text = system if isinstance(system, str) else _content_text(system)
    turns = [t for t in (_content_text(c) for c in llm_request.contents or []) if t]
    sections = [s for s in (system_text, *turns) if s]
    return _NO_TOOLS_PREAMBLE + "\n\n".join(sections)


def parse_envelope(stdout: str) -> dict:
    """Parse agy's one-line JSON envelope (tolerating stray leading lines)."""
    for line in reversed([ln for ln in stdout.splitlines() if ln.strip()]):
        with contextlib.suppress(json.JSONDecodeError):
            envelope = json.loads(line)
            if isinstance(envelope, dict):
                return envelope
    raise AgyCliError(f"agy printed no JSON envelope. stdout head: {stdout[:200]!r}")


class AgyCliLlm(BaseLlm):
    """ADK model that answers by running ``agy -p`` once per request."""

    binary: str
    timeout_s: float = DEFAULT_AGY_TIMEOUT_S

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False,
    ) -> AsyncGenerator[LlmResponse]:
        envelope = await self._run(build_prompt(llm_request))
        text = (envelope.get("response") or "").strip()
        if envelope.get("status") != "SUCCESS" or not text:
            self._raise_failure(
                f"status={envelope.get('status')!r} error={envelope.get('error')!r}"
            )
        usage = envelope.get("usage") or {}
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=text)]),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=usage.get("input_tokens", 0),
                candidates_token_count=usage.get("output_tokens", 0),
                total_token_count=usage.get("total_tokens", 0),
            ),
        )

    async def _run(self, prompt: str) -> dict:
        # Empty throwaway cwd: if the agent is ever talked into touching
        # files, there is nothing of ours to touch.
        with tempfile.TemporaryDirectory(prefix="agy-sandbox-") as cwd:
            try:
                proc = await asyncio.create_subprocess_exec(
                    self.binary, "-p", prompt, "--output-format", "json",
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                )
            except FileNotFoundError as e:
                raise AgyUnavailableError(f"agy binary not found at {self.binary!r}") from e
            try:
                out, err = await asyncio.wait_for(proc.communicate(), self.timeout_s)
            except TimeoutError as e:
                proc.kill()
                await proc.wait()
                raise Timeout(
                    message=f"agy exceeded {self.timeout_s:.0f}s",
                    model=self.model, llm_provider=AGY_PROVIDER,
                ) from e
        stderr = err.decode(errors="replace")
        if proc.returncode != 0:
            self._raise_failure(f"exit={proc.returncode} stderr={stderr[:300]!r}")
        return parse_envelope(out.decode(errors="replace"))

    @staticmethod
    def _raise_failure(detail: str) -> None:
        if _UNAVAILABLE_PATTERN.search(detail):
            raise AgyUnavailableError(f"agy unavailable: {detail}")
        raise AgyCliError(f"agy failed: {detail}")
