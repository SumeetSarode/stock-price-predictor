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
    ("freellmapi: premium live catalog ($19/yr)", "paid"),
    ("Both: reliability over days, desktop/Android apps", "out of scope for a one-shot check"),
)
