"""models.dev checks: does the catalog tell the truth about what providers actually serve?

Direct calls to YOUR OWN providers with YOUR OWN keys (the same calls the app already makes);
nothing here is handed to a third party. The catalog's own flags are a hint, so each is compared
with the provider's `/models` list and, for `structured_output`, with a live call.
"""
from __future__ import annotations

import json

from gateway_claims_data import CONFIRMED, INCONCLUSIVE, NOT_CONFIRMED

CATALOG_URL = "https://models.dev/api.json"
# models.dev provider id -> (provider's own model-list URL, how to authenticate, chat URL or None)
PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1/models", "bearer",
             "https://api.groq.com/openai/v1/chat/completions"),
    "google": ("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000", "x-goog-api-key",
               "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"),
    "mistral": ("https://api.mistral.ai/v1/models", "bearer", "https://api.mistral.ai/v1/chat/completions"),
    "cerebras": ("https://api.cerebras.ai/v1/models", "bearer", None),
    "nvidia": ("https://integrate.api.nvidia.com/v1/models", "bearer", None),
    "openrouter": ("https://openrouter.ai/api/v1/models", None, None),  # list is public
}
MAX_STALE_SHARE = 0.10   # more than this fraction of catalog entries not served => NOT_CONFIRMED
MIN_PROVIDERS = 200      # catalog claim: ~226 providers
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}


def _auth(kind: str | None, key: str | None) -> dict[str, str]:
    if not key or not kind:
        return {}
    return {"Authorization": f"Bearer {key}"} if kind == "bearer" else {kind: key}


def served_ids(call, c, provider: str, key: str | None):
    """-> (set of ids the provider itself serves, error text). Normalises Google's `models/` prefix."""
    url, kind, _ = PROVIDERS[provider]
    r = call(c, "GET", url, headers=_auth(kind, key))
    body = r.json() or {}
    rows = body.get("data") or body.get("models") or []
    ids = {str(m.get("id") or m.get("name", "")).removeprefix("models/") for m in rows if isinstance(m, dict)}
    return (ids - {""}, "") if r.status == 200 and ids else (set(), r.evidence()[:200])


def _flagged(models: dict, served: set[str], n: int = 2) -> list[str]:
    """Newest text-output models the catalog says support structured output AND the provider serves."""
    rows = [m for m in models.values()
            if m.get("structured_output") and m["id"] in served
            and (m.get("modalities") or {}).get("output") == ["text"]]
    rows.sort(key=lambda m: m.get("last_updated") or "", reverse=True)
    return [m["id"] for m in rows[:n]]


def _structured_live(call, c, provider: str, key: str, model: str):
    chat_url = PROVIDERS[provider][2]
    body = {"model": model, "max_tokens": 600, "messages": [
        {"role": "user", "content": 'Reply with JSON: {"answer": "ok"}'}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "t", "schema": SCHEMA}}}
    r = call(c, "POST", chat_url, json=body, headers={"Authorization": f"Bearer {key}"})
    try:
        text = r.json()["choices"][0]["message"]["content"] or ""
        ok = isinstance(json.loads(text).get("answer"), str)
    except (TypeError, KeyError, IndexError, ValueError, AttributeError):
        text, ok = "", False
    return ok, r.evidence(text)[:220]


def catalog_phase(L, c, keys: dict[str, str], call) -> None:
    r = call(c, "GET", CATALOG_URL)
    catalog = r.json() if r.status == 200 else None
    if not isinstance(catalog, dict):
        L.add("MD1", "-", INCONCLUSIVE, f"could not fetch the catalog: {r.evidence()[:200]}")
        return
    n_models = sum(len(p.get("models", {})) for p in catalog.values())
    L.add("MD1", "-", CONFIRMED if len(catalog) >= MIN_PROVIDERS else NOT_CONFIRMED,
          f"{len(catalog)} providers, {n_models} models (claim: ~226 / ~8.4k)")

    for provider, (_, kind, chat_url) in PROVIDERS.items():
        key = keys.get(provider)
        models = (catalog.get(provider) or {}).get("models", {})
        if kind and not key:
            L.add("MD2", provider, INCONCLUSIVE, "no key in .env, so the provider's own list was not fetched")
            continue
        served, err = served_ids(call, c, provider, key)
        if not served:
            L.add("MD2", provider, INCONCLUSIVE, f"provider list unavailable: {err}")
            continue
        stale, missing = sorted(set(models) - served), sorted(served - set(models))
        share = len(stale) / len(models) if models else 1.0
        L.add("MD2", provider, CONFIRMED if share <= MAX_STALE_SHARE else NOT_CONFIRMED,
              f"catalog {len(models)} vs served {len(served)}. In catalog but NOT served "
              f"({len(stale)}, {share:.0%}; includes models your account/tier cannot see): {stale[:8]}. "
              f"Served but absent from catalog ({len(missing)}): {missing[:8]}")

        if chat_url and key:
            for model in _flagged(models, served):
                ok, ev = _structured_live(call, c, provider, key, model)
                L.add("MD3", f"{provider}/{model}", CONFIRMED if ok else NOT_CONFIRMED,
                      f"catalog says structured_output=true; live json_schema call valid={ok}: {ev}")
