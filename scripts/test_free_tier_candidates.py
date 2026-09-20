"""Test every candidate free-tier LLM provider, one by one, off-VPN.

BOTTOM LINE AFTER 4 REAL RUNS (2026-09-18 .. 2026-09-20)
=======================================================
Of six advertised "free tiers", exactly ONE is genuinely free and works:

    USABLE   Cohere (cohere_chat/command-r-08-2024) -- clean schema-valid
             JSON, ~1-3s, passed on two separate days. Trial key: rate
             limited (~20/min) and non-production per Cohere's terms.

    NOT FREE Cerebras  -- "Payment required to access this resource"
    NOT FREE SambaNova -- "A payment method is required"
    NOT FREE Together  -- sign-up requires a DEPOSIT (confirmed by hand)
    BLOCKED  NVIDIA NIM -- every model 404s "Function not found for
             account"; NIM needs per-model provisioning on build.nvidia.com
             before the API works at all (5 models tried, incl. tiny ones)
    BLOCKED  Mistral   -- model ID confirmed valid in the live catalog,
             but every call 429s: account/workspace quota, not a bad model

The wider lesson, and why OmniRoute looked the way it did: the
free-API-key era has largely closed. OmniRoute's reliance on browser
session-cookie providers wasn't gratuitous -- it was routing around a
market that dried up. Assume any "free tier" claim needs verification
before it's worth a signup.

WHY THIS EXISTS
===============
"Full free access like OmniRoute" turned out to mean two very different
things: (1) stealing session cookies from your own paid ChatGPT/Claude/Felo
browser tabs -- fragile, ToS-risky, not something this project does -- and
(2) plain sign-up-and-get-a-key free tiers, same arrangement as the Groq/
Gemini keys already in CHAIN_AGENTIC. This script is (2), done properly:
one real schema-JSON test per candidate, run in isolation, so the decision
of "is this good enough to add to the chain" is based on real model output,
not on litellm metadata (which lied about Groq once already).

This deliberately does NOT reimplement the test protocol -- it imports
`check_model_isolated()` from verify_offvpn.py, the exact 4-step check
(raw schema call -> bare JSON? -> json_extract rescue? -> <think> leak?)
that caught Nemotron's prose problem. One protocol, every candidate.

HOW TO USE
==========
  1. Pick whichever candidates below you want to try. Sign up (free) at
     the URL listed, paste the key into .env under the named env var.
     Leave the rest blank -- blank candidates are SKIPPED, not failed.
  2. Many of these API hosts are blocked on this network's VPN (same
     pattern as openrouter.ai / api.mistral.ai / registry.npmjs.org
     earlier). Run this OFF-VPN:

       uv run python scripts/test_free_tier_candidates.py

  3. Read the Leaderboard at the bottom of the report -- ranked usable-
     first, then fastest. Send back the .md file.

Writes reports/free_tier_candidates_<timestamp>.md (+ .json), same
format as verify_offvpn.py's reports.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_offvpn import Check, Report, check_model_isolated  # noqa: E402  (DRY: reuse, don't duplicate)

# Providers whose live catalog we can ASK, instead of guessing model IDs.
# Hard-won lesson: litellm's static catalog lists ZERO nvidia_nim chat models
# (only rerankers), and two hand-picked IDs failed differently -- one 404, one
# HTTP 410 "reached end of life on 2026-08-26". A third guess would be a third
# wasted round-trip, so the script now discovers what each provider actually
# serves TODAY. OpenAI-compatible providers all expose GET /v1/models.
#
# Every base URL below was cross-checked against litellm's OWN default
# api_base per provider (litellm_core_utils/get_llm_provider_logic.py on
# GitHub main, since AI provider domains are DNS-blocked from dev but GitHub
# isn't) -- not typed from memory. Caught one real mismatch this way: Together
# AI's api.together.xyz (which I'd used) is an alias litellm's own endpoint
# matcher recognizes, but their actual DEFAULT is api.together.ai -- switched
# to match it exactly rather than lean on an alias domain.
# (display name -> (base url, .env key var))
DISCOVERY_ENDPOINTS: dict[str, tuple[str, str]] = {
    "Cerebras": ("https://api.cerebras.ai/v1/models", "CEREBRAS_API_KEY"),
    "NVIDIA NIM": ("https://integrate.api.nvidia.com/v1/models", "NVIDIA_NIM_API_KEY"),
    "Together AI": ("https://api.together.ai/v1/models", "TOGETHER_API_KEY"),
    "SambaNova": ("https://api.sambanova.ai/v1/models", "SAMBANOVA_API_KEY"),
    "Mistral": ("https://api.mistral.ai/v1/models", "MISTRAL_API_KEY"),
}


def discover_models(rep: Report) -> None:
    """Ask each provider what it ACTUALLY hosts for this specific key.

    Runs before the model tests so the report answers "was that model ID
    wrong, or is the account restricted?" directly -- the Cerebras 404 said
    'does not exist OR you do not have access', and only the live catalog
    can separate those two. Cheap, unauthenticated-ish, and non-fatal.
    """
    import httpx

    _section_banner("0. Model discovery -- what does each provider actually serve?")

    for name, (url, key_env) in DISCOVERY_ENDPOINTS.items():
        key = (os.environ.get(key_env) or "").strip()
        if not key:
            rep.add(Check(
                f"discover: {name}", f"discovery/{name}", "SKIP",
                f"no {key_env} set",
            ))
            continue

        def probe(url=url, key=key, name=name):
            r = httpx.get(
                url, timeout=30.0,
                headers={"Authorization": f"Bearer {key}"},
            )
            if r.status_code != 200:
                return (
                    "FAIL",
                    f"HTTP {r.status_code} listing models: {r.text[:300]}",
                    {"status_code": r.status_code},
                )
            payload = r.json()
            ids = sorted(
                m.get("id", "?") for m in payload.get("data", payload if isinstance(payload, list) else [])
            )
            return (
                "PASS",
                f"{len(ids)} models available to THIS key:\n"
                + "\n".join(f"  - {i}" for i in ids[:60])
                + ("\n  ..." if len(ids) > 60 else ""),
                {"model_ids": ids},
            )

        rep.run(f"discover: {name}", f"discovery/{name}", probe)


def _section_banner(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


# (display name, litellm model string, .env key var, free signup URL, note)
CANDIDATES: list[tuple[str, str, str, str, str]] = [
    (
        "Mistral", "mistral/mistral-small-latest", "MISTRAL_API_KEY",
        "https://console.mistral.ai/",
        "Live discovery (run 3) CONFIRMS mistral-small-latest is genuinely in "
        "this key's 46-model catalog -- the model ID has never been the "
        "problem. Every attempt (off-VPN and on) hits an immediate 429 "
        "RateLimitError instead, so this is a persistent account/workspace "
        "quota block, not a code or model issue.",
    ),
    (
        "Cerebras: gpt-oss-120b", "cerebras/gpt-oss-120b", "CEREBRAS_API_KEY",
        "https://cloud.cerebras.ai/",
        "DEAD END -- NOT FREE. Both of the only two models this key's catalog "
        "offers return 'Payment required to access this resource. Visit your "
        "billing tab.' Restricted catalog AND paywalled inference.",
    ),
    (
        "Cerebras: qwen-3.8-27b", "cerebras/qwen-3.8-27b", "CEREBRAS_API_KEY",
        "https://cloud.cerebras.ai/",
        "The other of the exactly-two models this key's live catalog contains.",
    ),
    (
        "SambaNova", "sambanova/Meta-Llama-3.3-70B-Instruct", "SAMBANOVA_API_KEY",
        "https://cloud.sambanova.ai/",
        "Live discovery (run 3) CONFIRMS this exact model is in the key's "
        "catalog -- so model choice was never the problem. The blocker is "
        "purely 'A payment method is required' at the billing layer, which no "
        "model swap fixes. Left in only so the report keeps documenting that.",
    ),
    (
        "NVIDIA NIM: mistral-7b", "nvidia_nim/mistralai/mistral-7b-instruct-v0.3",
        "NVIDIA_NIM_API_KEY", "https://build.nvidia.com/",
        "REPLACED after live discovery (run 3) showed the real lesson: NIM's "
        "/v1/models LISTS 82 models, but listed != deployed. The 253B Nemotron "
        "guess returned 'Function ... not found for account' -- catalog "
        "presence doesn't guarantee an inference endpoint is provisioned for "
        "this account. Switching to small, extremely common models that are "
        "far more likely to have a default deployment, to finally learn if the "
        "NIM pipe itself works before chasing exotic model IDs further.",
    ),
    (
        "NVIDIA NIM: gemma-3-4b", "nvidia_nim/google/gemma-3-4b-it",
        "NVIDIA_NIM_API_KEY", "https://build.nvidia.com/",
        "Second small/common model from the same live catalog, in case the "
        "first candidate above also turns out to be undeployed for this account.",
    ),
    (
        "Cohere", "cohere_chat/command-r-08-2024", "COHERE_API_KEY",
        "https://dashboard.cohere.com/",
        "CONFIRMED USABLE (run 3): passed raw call, bare-JSON, AND json_extract "
        "in 0.8s using the prompt-only path (prompt states the exact required "
        "keys since cohere_chat can't take response_format). First candidate "
        "in this whole shootout to actually pass. Good chain candidate.",
    ),
    (
        "DeepSeek", "deepseek/deepseek-chat", "DEEPSEEK_API_KEY",
        "https://platform.deepseek.com/",
        "Cheap even past any free allotment; strong model for the price.",
    ),
    (
        "Together AI", "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "TOGETHER_API_KEY", "https://api.together.ai/",
        "DEAD END -- NOT FREE. Two separate confirmations: the API returns HTTP "
        "401 on /v1/models with the stored key, AND signing up for a valid key "
        "requires a DEPOSIT (confirmed by hand, 2026-09-20). Their sign-up "
        "credit model no longer yields a usable no-payment key. Kept in the "
        "lineup only as a documented negative result -- do not re-attempt "
        "without a funded account.",
    ),
]


def _run_all(rep: Report) -> None:
    for name, model, key_env, url, note in CANDIDATES:
        key = (os.environ.get(key_env) or "").strip()
        if not key:
            rep.add(Check(
                f"{name} API key present", name, "SKIP",
                f"No key in {key_env}. Free signup: {url}\n{note}",
            ))
            continue
        # Pass the DISPLAY name as the label: several candidates share a
        # provider prefix (3x cerebras, 2x nvidia_nim), and the default
        # prefix-derived label would make them collide into one row,
        # silently hiding all but the last result.
        check_model_isolated(rep, model=model, api_key=key, label=name)


def _leaderboard(rep: Report) -> str:
    """Group the flat check list back into one row per candidate."""
    by_label: dict[str, dict[str, Check]] = {}
    for c in rep.checks:
        by_label.setdefault(c.section, {})[c.name.split(".", 1)[0].strip()] = c

    rows: list[tuple[str, str, str, str, str, str, float]] = []
    for label, checks in by_label.items():
        if label.startswith("discovery/"):
            continue  # catalog listing, not a candidate verdict
        a = checks.get("3a")
        if a is None:
            continue  # SKIP-only entry (no key provided)
        b = checks.get("3b")
        cext = checks.get("3c")
        usable = a.status == "PASS" and (
            (b is not None and b.status == "PASS")
            or (cext is not None and cext.status == "PASS")
        )
        # Was strict schema mode actually available, or did we fall back to
        # prompt-only? A model that only produces good JSON when BEGGED is a
        # weaker chain member than one the API can hard-constrain.
        used_schema = a.evidence.get("used_schema_mode")
        mode = "-" if used_schema is None else ("schema" if used_schema else "prompt-only")
        rows.append((
            label,
            "USABLE" if usable else "NOT USABLE",
            a.status,
            b.status if b else "-",
            cext.status if cext else "-",
            mode,
            a.elapsed_s,
        ))

    rows.sort(key=lambda r: (r[1] != "USABLE", r[6]))

    lines = [
        "## Leaderboard (usable first, then fastest)",
        "",
        "| Candidate | Verdict | 3a raw call | 3b bare-JSON | 3c json_extract | JSON mode | Latency |",
        "|---|---|---|---|---|---|---|",
    ]
    if not rows:
        lines.append("| *(no candidate had a key configured -- nothing to rank)* | | | | | | |")
    for label, verdict, a_s, b_s, c_s, mode, latency in rows:
        lines.append(
            f"| {label} | **{verdict}** | {a_s} | {b_s} | {c_s} | {mode} | {latency:.1f}s |"
        )
    return "\n".join(lines)


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env", override=False)
    configured = sum(1 for *_, key_env, _u, _n in CANDIDATES if os.environ.get(key_env, "").strip())
    print(
        f"Free-tier candidate shootout -- {configured}/{len(CANDIDATES)} candidates "
        "have a key configured.\nThis WILL hit the real network.\n"
    )
    if configured == 0:
        print(
            "No candidate keys found in .env. Sign up for at least one (they're all "
            "free) and paste the key in, e.g. CEREBRAS_API_KEY=... -- see the list "
            "in this script's CANDIDATES for signup URLs. Running anyway so the "
            "report documents exactly what's still missing.\n"
        )

    rep = Report()
    discover_models(rep)
    _run_all(rep)
    md, js = rep.write(prefix="free_tier_candidates")

    board = _leaderboard(rep)
    print("\n" + board)
    with md.open("a", encoding="utf-8") as f:
        f.write("\n" + board + "\n")

    passed = sum(c.status == "PASS" for c in rep.checks)
    failed = sum(c.status == "FAIL" for c in rep.checks)
    skipped = sum(c.status == "SKIP" for c in rep.checks)
    print(f"\n{'=' * 68}")
    print(f"RESULT: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"Report (markdown): {md}")
    print(f"Report (json):     {js}")
    print("=" * 68)
    return 0  # a candidate being unusable is a finding, not a script failure


if __name__ == "__main__":
    sys.exit(main())
