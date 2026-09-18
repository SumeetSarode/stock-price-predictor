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
        "Cerebras 70B", "cerebras/llama-3.3-70b", "CEREBRAS_API_KEY",
        "https://cloud.cerebras.ai/",
        "Model string CONFIRMED correct against litellm's upstream main branch "
        "catalog -- the earlier 404 said 'does not exist OR you do not have "
        "access to it', and since the ID is valid, it's the access half: 70B "
        "likely isn't enabled on a fresh free account. Smaller siblings below "
        "test that theory rather than writing Cerebras off.",
    ),
    (
        "Cerebras 8B", "cerebras/llama3.1-8b", "CEREBRAS_API_KEY",
        "https://cloud.cerebras.ai/",
        "Smallest Cerebras model -- most likely to be enabled on a free tier. "
        "If 70B 404s but this passes, it's an account-tier limit, not a dead "
        "provider.",
    ),
    (
        "Cerebras Qwen-32B", "cerebras/qwen-3-32b", "CEREBRAS_API_KEY",
        "https://cloud.cerebras.ai/",
        "Mid-size Cerebras option, also in upstream catalog. Brackets the "
        "access question between 8B and 70B.",
    ),
    (
        "SambaNova", "sambanova/Meta-Llama-3.3-70B-Instruct", "SAMBANOVA_API_KEY",
        "https://cloud.sambanova.ai/",
        "HEADS UP: last run returned 'A payment method is required' -- inference "
        "is gated behind a card on file, so this is NOT free-tier in the same "
        "sense as Groq/Gemini. Left in the lineup only so the report keeps "
        "documenting that fact; expect the same failure unless billing was added.",
    ),
    (
        "NVIDIA NIM (Llama-3.3-70B)", "nvidia_nim/meta/llama-3.3-70b-instruct",
        "NVIDIA_NIM_API_KEY", "https://build.nvidia.com/",
        "CORRECTED after reading litellm's nvidia_nim source on main: the "
        "previous pick (nvidia/nemotron-4-340b-instruct) is one of only FOUR "
        "hardcoded special-cased models whose supported-param list omits "
        "response_format -- which is exactly why it errored, and that was my "
        "model choice, NOT a NIM limitation. Every other NIM model hits the "
        "DEFAULT branch, which explicitly DOES support response_format, tools "
        "and tool_choice. This model uses that default path.",
    ),
    (
        "NVIDIA NIM (Nemotron-Ultra)",
        "nvidia_nim/nvidia/llama-3.1-nemotron-ultra-253b-v1",
        "NVIDIA_NIM_API_KEY", "https://build.nvidia.com/",
        "The genuinely interesting comparison: Nemotron family via NVIDIA's OWN "
        "API instead of the OpenRouter free proxy already in CHAIN_AGENTIC (the "
        "source of our prose-not-JSON grief). Not one of the four special-cased "
        "IDs, so it gets the default param set incl. response_format. If NIM "
        "doesn't host this exact ID the call will 404 cleanly -- which is itself "
        "a useful answer, and the Llama-3.3 candidate above still isolates "
        "whether the NIM pipe works at all.",
    ),
    (
        "Cohere", "cohere_chat/command-r-08-2024", "COHERE_API_KEY",
        "https://dashboard.cohere.com/",
        "litellm splits Cohere into 'cohere/' (embeddings only) and "
        "'cohere_chat/' (real chat route) -- verified via LlmProviders enum. "
        "Checked litellm main: cohere_chat STILL omits response_format from its "
        "supported params, so upgrading wouldn't help; the prompt-only retry in "
        "check_model_isolated is the correct path, and it's a fair test since "
        "our chain relies on json_extract anyway. Dated model ID used because "
        "it's a stable pin present in upstream's catalog.",
    ),
    (
        "DeepSeek", "deepseek/deepseek-chat", "DEEPSEEK_API_KEY",
        "https://platform.deepseek.com/",
        "Cheap even past any free allotment; strong model for the price.",
    ),
    (
        "Together AI", "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "TOGETHER_API_KEY", "https://api.together.ai/",
        "Confirmed the ONLY '-Free'-suffixed entry across together_ai's whole "
        "catalog. HEADS UP: last run returned 'Invalid API key provided' -- the "
        "stored key ('key_' prefix, ~21 chars) does not match Together's usual "
        "long-hex secret format, so it may be a project/org ID rather than the "
        "actual secret key. Re-copy from api.together.ai/settings/api-keys if "
        "this fails again the same way.",
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
