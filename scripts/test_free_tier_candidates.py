"""Test every candidate free-tier LLM provider, one by one, off-VPN.

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

# (display name, litellm model string, .env key var, free signup URL, note)
CANDIDATES: list[tuple[str, str, str, str, str]] = [
    (
        "Mistral", "mistral/mistral-small-latest", "MISTRAL_API_KEY",
        "https://console.mistral.ai/",
        "Already staged. Prior attempts: 429 rate-limit off-VPN (inconclusive), "
        "DNS-blocked on-VPN. Never got a clean pass or fail.",
    ),
    (
        "Cerebras", "cerebras/llama-3.3-70b", "CEREBRAS_API_KEY",
        "https://cloud.cerebras.ai/",
        "Same personal-key arrangement as Groq. Known for very fast inference.",
    ),
    (
        "SambaNova", "sambanova/Meta-Llama-3.3-70B-Instruct", "SAMBANOVA_API_KEY",
        "https://cloud.sambanova.ai/",
        "Free tier, fast inference, same personal-key arrangement as Groq.",
    ),
    (
        "NVIDIA NIM", "nvidia_nim/meta/llama-3.1-8b-instruct", "NVIDIA_NIM_API_KEY",
        "https://build.nvidia.com/",
        "Direct NIM catalog -- separate from the OpenRouter/Nemotron route "
        "already in the chain. Verified via litellm source: NvidiaNimConfig "
        "subclasses OpenAIGPTConfig, i.e. genuine OpenAI-compatible passthrough "
        "-- litellm ships NO static model whitelist for it, so this exact model "
        "ID is unverified until this test actually runs against build.nvidia.com's "
        "live catalog. That live call IS the verification.",
    ),
    (
        "Cohere", "cohere_chat/command-r", "COHERE_API_KEY",
        "https://dashboard.cohere.com/",
        "Free trial key. NOTE: litellm splits Cohere into two providers --\n"
        "'cohere/' is embeddings-only, 'cohere_chat/' is the real chat "
        "completions route. Verified against the installed litellm's own "
        "LlmProviders enum, not assumed.",
    ),
    (
        "DeepSeek", "deepseek/deepseek-chat", "DEEPSEEK_API_KEY",
        "https://platform.deepseek.com/",
        "Cheap even past any free allotment; strong model for the price.",
    ),
    (
        "Together AI", "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "TOGETHER_API_KEY", "https://api.together.ai/",
        "Sign-up credit is the general story there, NOT a recurring free tier -- "
        "but this specific '-Free' model is confirmed as the ONLY '-Free'-suffixed "
        "entry in litellm's entire 96-model together_ai catalog (checked directly), "
        "consistent with most of Together's old always-free endpoints having been "
        "discontinued and this one surviving. Verify it's still live before relying "
        "on it long-term.",
    ),
]


def _run_all(rep: Report) -> None:
    for name, model, key_env, url, note in CANDIDATES:
        key = (os.environ.get(key_env) or "").strip()
        if not key:
            rep.add(Check(
                f"{name} API key present", model.split("/", 1)[0], "SKIP",
                f"No key in {key_env}. Free signup: {url}\n{note}",
            ))
            continue
        check_model_isolated(rep, model=model, api_key=key)


def _leaderboard(rep: Report) -> str:
    """Group the flat check list back into one row per candidate."""
    by_label: dict[str, dict[str, Check]] = {}
    for c in rep.checks:
        by_label.setdefault(c.section, {})[c.name.split(".", 1)[0].strip()] = c

    rows: list[tuple[str, str, str, str, str, float]] = []
    for label, checks in by_label.items():
        a = checks.get("3a")
        if a is None:
            continue  # SKIP-only entry (no key provided)
        b = checks.get("3b")
        cext = checks.get("3c")
        usable = a.status == "PASS" and (
            (b is not None and b.status == "PASS")
            or (cext is not None and cext.status == "PASS")
        )
        rows.append((
            label,
            "USABLE" if usable else "NOT USABLE",
            a.status,
            b.status if b else "-",
            cext.status if cext else "-",
            a.elapsed_s,
        ))

    rows.sort(key=lambda r: (r[1] != "USABLE", r[5]))

    lines = [
        "## Leaderboard (usable first, then fastest)",
        "",
        "| Provider | Verdict | 3a raw call | 3b bare-JSON | 3c json_extract | Latency |",
        "|---|---|---|---|---|---|",
    ]
    if not rows:
        lines.append("| *(no candidate had a key configured -- nothing to rank)* | | | | | |")
    for label, verdict, a_s, b_s, c_s, latency in rows:
        lines.append(f"| {label} | **{verdict}** | {a_s} | {b_s} | {c_s} | {latency:.1f}s |")
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
