"""The claims under test, kept apart from the machinery that tests them.

Every row is a claim a project makes (or one we need for our own use case), with its source, so
a verdict can always be traced back to the exact sentence it was tested against.
"""
from __future__ import annotations

from dataclasses import dataclass

CONFIRMED, NOT_CONFIRMED, INCONCLUSIVE, NOT_TESTED = (
    "CONFIRMED", "NOT_CONFIRMED", "INCONCLUSIVE", "NOT_TESTED")


@dataclass(frozen=True)
class Claim:
    id: str
    repo: str
    text: str
    source: str


CLAIMS: tuple[Claim, ...] = (
    Claim("FL1", "freellmapi", "Official Docker image starts and serves the API on :3001",
          "docs/en/install: Docker Compose; image ghcr.io/tashfeenahmed/freellmapi"),
    Claim("FL2", "freellmapi", "Declarative config provisions the admin; login works and the unified key is retrievable",
          "docs/en/install: Declarative startup config; routes/auth.ts, routes/settings.ts"),
    Claim("FL3", "freellmapi", "Catalog of hundreds of free models (README: 635 endpoints, 34 providers; free installs get a monthly snapshot)",
          "README headline; 'free installs get the monthly snapshot'"),
    Claim("FL4", "freellmapi", "/v1/models?execution_status=ready reports which models can serve a request now",
          "docs/en/api: Model listing"),
    Claim("FL5", "freellmapi", "Keyless providers (kilo, ovh, aihorde, llm7, pollinations) work with no account",
          "docs/en/providers/01-supported-platforms.md (Keyless rows)"),
    Claim("FL6", "freellmapi", "model=auto routes to a working model and says which (X-Routed-Via)",
          "docs/en/api: Chat completions, Response headers"),
    Claim("FL7", "freellmapi", "Fails over past dead keys (429/402) and still serves",
          "README: 'on a 429/5xx it cools that key down and retries the next model'"),
    Claim("FL8", "freellmapi", "Structured output: response_format json_schema is honoured / format-ignoring models are skipped",
          "issues #516, #933, #983; lib/structured-output.ts"),
    Claim("FL9", "freellmapi", "(our use case) Both pipeline steps, news-impact and synthesizer, pass through it",
          "not a project claim: the test that matters for price_predictor"),
    Claim("MD1", "models.dev", "Catalog covers ~226 providers and ~8.4k models",
          "models.dev homepage / api.json"),
    Claim("MD2", "models.dev", "Each provider's catalog entries are models that provider actually serves (no stale/retired ones)",
          "implied by listing them as available; earlier we saw a retired Groq model listed active"),
    Claim("MD3", "models.dev", "structured_output=true means a json_schema request really returns valid JSON",
          "catalog field `structured_output`; earlier gpt-oss-120b was flagged true but failed live"),
    Claim("ST1", "Strata", "Runs Qwen3.8-Flash-Next 'on any consumer hardware' (README narrows it: NVIDIA/AMD GPU with 12 GB+ VRAM, 32 GB+ RAM, ~80 GB disk, Windows/Linux)",
          "repo description vs README 'What you need'; issue #518 (maintainer: Apple Silicon 'isn't planned')"),
    Claim("ST2", "Strata", "The model downloads it points at are real, open, and about 70-80 GB for one size",
          "README 'Install'; HuggingFace ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF; model licence"),
    Claim("ST3", "Strata", "Serves an OpenAI-compatible API on localhost:8080/v1 where 'any API key and any model name work'",
          "README 'Using it'"),
    Claim("ST4", "Strata", "json_schema responses are validated server-side (schema prompting + validation, not constrained decoding)",
          "docs/DETAILS.md 'JSON response formats'; open issues #1328, #1805"),
    Claim("ST5", "Strata", "(our use case) Both pipeline steps, news-impact and synthesizer, pass through it",
          "not a project claim: the test that matters for price_predictor"),
    Claim("OR1", "OmniRoute", "Official Docker image starts and reports healthy",
          "README: docker run ... diegosouzapw/omniroute:latest; /api/monitoring/health"),
    Claim("OR2", "OmniRoute", "'Works the second you install it - no keys, no config': model=auto answers",
          "README banner + curl example (base image)"),
    Claim("OR3", "OmniRoute", "Same zero-config claim on the -web image (bundles Chromium)",
          "docs/guides/TROUBLESHOOTING.md: Playwright Chromium"),
    Claim("OR4", "OmniRoute", "'OpenCode Free (no auth)': oc/... models answer without credentials",
          "README: Connect a FREE provider; freellmapi notes this roster was locked (403) in 2026-09"),
    Claim("OR5", "OmniRoute", "auto/coding:free routes to a working free model",
          "issue #15969 (reporter's model id)"),
    Claim("OR6", "OmniRoute", "(our use case) Both pipeline steps pass through whichever models answer",
          "not a project claim: the test that matters for price_predictor"),
)
NOT_TESTED_REASONS = (
    ("OmniRoute: connecting your own provider keys", "needs its management API/dashboard "
     "(separate credentials, undocumented payload); guessing it would give an unfair result"),
    ("OmniRoute: web-cookie / browser-session providers", "terms-of-service gray area; we only "
     "observe whether the documented zero-config path works, we do not enable cookie providers"),
    ("Code Puppy: round-robin / retry-backoff / model switching", "it is the DEVELOPMENT tool, built on "
     "PydanticAI; the product runtime uses Google ADK and must not depend on it, so there is no "
     "runtime behaviour to run here. Its one reusable piece, models.dev, is covered by MD1-MD3. "
     "This is a judgement call, not a test result: say so if you want it exercised anyway"),
    ("Strata: MCP server, multi-GPU, vision, Anthropic/Responses APIs", "irrelevant to a text-only JSON "
     "pipeline; nothing to evaluate there (YAGNI)"),
    ("Strata: installing it and downloading the model here", "this machine cannot run it (see ST1); ST3-ST5 "
     "only run if a Strata server is reachable, e.g. on a qualifying Windows/Linux PC"),
    ("freellmapi: premium live catalog ($19/yr)", "paid"),
    ("Both: reliability over days, desktop/Android apps", "out of scope for a one-shot check"),
)
