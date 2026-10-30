"""When a provider answers "model not found", look for the name it serves that model under now.

Providers rename and re-version models (kimi-k2.6 -> kimi-k2.6-instruct, gemini-2.5-pro -> a dated preview id)
while old ids linger in their own lists. For each not-found model, the closest current id of the SAME model
is found in (1) that provider's own model list and (2) the models.dev catalog for that provider: same name words,
only the version or a suffix may differ. Those not already being tested become new candidates and go through the
normal chat check and pipeline test. Each not-found model is resolved once; its note says what was tried.
Pure: no network, so it is unit-tested.
"""
from __future__ import annotations

import re
from dataclasses import replace
from difflib import SequenceMatcher

from sweep_data import LEDGER_ALIASES, SPECS, TESTABLE, Candidate

MAX_ALTERNATIVES = 3      # per not-found model: enough to find a renamed id, few enough to protect the quota
MIN_SIMILARITY = 0.6
_PREFIXES = re.compile(r"^(@cf/|@hf/|models/)")
_WORD = re.compile(r"[a-z]{2,}")
MARK = "not found, so"


def _norm(model_id: str) -> str:
    """'@cf/moonshotai/Kimi-K2.6:free' -> 'kimi-k2.6': compare names, not vendor paths or route suffixes."""
    return _PREFIXES.sub("", model_id.lower()).split("/")[-1].removesuffix(":free")


def _words(model_id: str) -> set[str]:
    """The name words ('gemini-2.5-pro' -> {'gemini', 'pro'}): versions and sizes are what may change."""
    return set(_WORD.findall(_norm(model_id)))


def alternatives(model_id: str, pool, limit: int = MAX_ALTERNATIVES) -> list[str]:
    """The ids in `pool` most likely to be `model_id` under its current name, best first. A candidate must keep
    every name word (so gemini-2.5-pro never maps to gemini-2.5-flash); only versions and suffixes may differ."""
    want, words = _norm(model_id), _words(model_id)
    scored = []
    for other in set(pool):
        if other == model_id or not words or not words <= _words(other):
            continue
        theirs = _norm(other)
        # the same name plus a suffix ('-instruct', '-0905', '-latest') is a rename, not a different version
        score = 1 + len(want) / len(theirs) if theirs.startswith(want) else SequenceMatcher(None, want, theirs).ratio()
        if score >= MIN_SIMILARITY:
            scored.append((-score, other))
    return [o for _, o in sorted(scored)[:limit]]


def catalog_ids(catalog: dict, provider: str) -> set[str]:
    """Model ids models.dev lists for this provider (its key can differ: 'ovhcloud', 'cloudflare-workers-ai')."""
    keys = {k for k in catalog if k == provider or LEDGER_ALIASES.get(k) == provider or k.startswith(f"{provider}-")}
    return {mid for k in keys for mid in ((catalog.get(k) or {}).get("models") or {})}


def replacements(cands: list[Candidate], store, catalog: dict) -> list[Candidate]:
    """New candidates for every model whose chat check came back not_found. Notes on the originals say what was
    tried; ids already being tested are only mentioned, never added twice."""
    known = {c.uid for c in cands}
    served: dict[str, set[str]] = {}
    for c in cands:
        served.setdefault(c.provider, set()).add(c.model)
    out: list[Candidate] = []
    for c in cands:
        r = store.get(c.uid, "chat")
        if c.status != TESTABLE or not r or r.verdict != "FAIL" or r.klass != "not_found" or c.route not in ("direct", "anon"):
            continue
        if MARK in c.note or "no current id" in c.note:      # already resolved in an earlier pass
            continue
        spec = SPECS.get(c.provider)
        pool = served.get(c.provider, set()) | catalog_ids(catalog, c.provider)
        if spec and spec.free_rule == "suffix":
            pool = {m for m in pool if m.endswith(":free")}
        alts = alternatives(c.model, pool)
        if not alts:
            c.note = f"{c.note}; not found, and no current id of the same model was found".lstrip("; ")
            continue
        c.note = f"{c.note}; {MARK} these ids were tested instead: {', '.join(alts)}".lstrip("; ")
        for alt in alts:
            new = replace(c, model=alt, note=f"tried because {c.model} was not found", sources=("not-found-resolver",),
                          litellm_id=c.litellm_id[: -len(c.model)] + alt if c.litellm_id else None)
            if new.uid not in known:
                known.add(new.uid)
                out.append(new)
    return out
