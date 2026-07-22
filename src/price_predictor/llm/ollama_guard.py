"""Startup guard: warn if a configured local (Ollama) fallback model isn't pulled.

WHY THIS EXISTS
===============
The agentic chain can end in a local Ollama model (e.g.
``ollama_chat/qwen3:8b``) as an OFFLINE last-resort tier. That tier only
fires when every hosted provider is rate-limited -- which might be days
after startup. If the model was never ``ollama pull``-ed, or the Ollama
server isn't running, you'd only discover it at the *worst* possible moment
(mid-outage, when the fallback is your last hope).

This guard checks at STARTUP instead: it reads the configured chain, finds
any Ollama entries, asks the Ollama server what's actually pulled, and logs
a clear, actionable warning if there's a mismatch.

DESIGN
======
- NON-FATAL. A missing fallback model must never crash the app -- it's a
  safety net, not a hard dependency. We only ever *log*.
- QUIET when it can't help. No Ollama entries in the chain -> instant no-op.
  Server unreachable -> DEBUG (not WARNING): we can't verify, but that's
  not necessarily wrong (maybe you start Ollama later).
- LOUD only when actionable: server is up AND a configured model is missing
  -> WARNING with the exact ``ollama pull`` command to fix it.
- No new dependencies: uses httpx (already a project dep).
"""
from __future__ import annotations

import httpx
from loguru import logger

from price_predictor.config.settings import settings

# Providers whose entries denote a local Ollama model (mirrors factory.py).
_LOCAL_PROVIDERS: frozenset[str] = frozenset({"ollama", "ollama_chat"})

# Startup checks must be snappy -- we won't block boot on a slow/absent server.
_PROBE_TIMEOUT_SECONDS = 1.5


def ollama_tags_in_chain(chain: list[str]) -> list[str]:
    """Extract the bare Ollama model tags from a chain.

    'ollama_chat/qwen3:8b' -> 'qwen3:8b'. Non-Ollama entries are ignored.
    Order-preserving and de-duplicated.
    """
    tags: list[str] = []
    for entry in chain:
        provider, _, tag = entry.partition("/")
        if provider in _LOCAL_PROVIDERS and tag and tag not in tags:
            tags.append(tag)
    return tags


def _pulled_models(base_url: str) -> set[str] | None:
    """Return the set of model names Ollama has pulled, or None if unreachable.

    Ollama's GET /api/tags returns {"models": [{"name": "qwen3:8b", ...}]}.
    """
    try:
        resp = httpx.get(
            f"{base_url.rstrip('/')}/api/tags",
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        return None
    return {m.get("name", "") for m in data.get("models", [])}


def _is_pulled(tag: str, pulled: set[str]) -> bool:
    """True if `tag` is satisfied by the pulled set.

    Forgiving on the implicit ':latest' tag: a configured 'qwen3' matches a
    pulled 'qwen3:latest', and a bare pull of 'qwen3' shows as 'qwen3:latest'.
    """
    if tag in pulled:
        return True
    return ":" not in tag and f"{tag}:latest" in pulled


def check_local_models(chain: list[str], base_url: str) -> list[str]:
    """Return a list of human-readable warning strings (empty = all good).

    Pure/inspectable core -- takes its inputs explicitly so it's trivial to
    unit-test without touching settings or the network mock.
    """
    tags = ollama_tags_in_chain(chain)
    if not tags:
        return []  # no local fallback configured -> nothing to check

    pulled = _pulled_models(base_url)
    if pulled is None:
        return [
            f"Could not reach Ollama at {base_url} to verify the offline "
            f"fallback model(s) {tags}. If you rely on the local fallback, "
            f"start it with `ollama serve`."
        ]

    warnings: list[str] = []
    for tag in tags:
        if not _is_pulled(tag, pulled):
            warnings.append(
                f"Offline fallback model '{tag}' is NOT pulled. The chain will "
                f"fail over to it only when hosted providers are exhausted -- "
                f"pull it now so it's ready: `ollama pull {tag}`"
            )
    return warnings


def warn_if_local_models_missing() -> None:
    """Startup convenience: check the agentic chain, log findings. Never raises.

    Call once at app startup (CLI callback / web lifespan). Reachability
    problems log at DEBUG (non-actionable); a missing-but-server-up model
    logs at WARNING (actionable, with the fix command).
    """
    try:
        chain = settings.effective_chain("agentic")
    except Exception:  # never let a startup check break boot
        return

    tags = ollama_tags_in_chain(chain)
    if not tags:
        return

    pulled = _pulled_models(settings.ollama_api_base)
    if pulled is None:
        logger.debug(
            "ollama guard: could not reach Ollama at {} to verify {} "
            "(fine if you start it later; it's only a fallback tier).",
            settings.ollama_api_base, tags,
        )
        return

    for tag in tags:
        if _is_pulled(tag, pulled):
            logger.info("ollama guard: offline fallback '{}' is ready.", tag)
        else:
            logger.warning(
                "ollama guard: offline fallback '{}' is NOT pulled. "
                "Run `ollama pull {}` so the fallback works when hosted "
                "providers are rate-limited.",
                tag, tag,
            )
