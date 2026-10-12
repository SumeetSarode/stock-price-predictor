"""Discovery for the free-model sweep: turn every source's model list into Candidates.

Sources: the providers' OWN /models lists (authoritative for what they serve), freellmapi's catalog,
OmniRoute's catalog, models.dev (and Code Puppy's bundled snapshot of it), and Strata. Network access
is injected (`call`), so all of this is testable with fakes.
"""
from __future__ import annotations

from collections import defaultdict

from sweep_data import (
    EXHAUSTED,
    KEY_ALIASES,
    LEDGER_ALIASES,
    NEEDS_SIGNUP,
    NO_ENDPOINT,
    NOT_CHAT,
    SPECS,
    TERMS_BLOCKED,
    TERMS_BLOCKED_PROVIDERS,
    TESTABLE,
    Candidate,
    Spec,
    classify_candidate,
)


def env_names(provider: str) -> tuple[str, ...]:
    spec = SPECS.get(provider)
    return spec.env if spec else (*KEY_ALIASES.get(provider, ()), f"{provider.upper().replace('-', '_')}_API_KEY")


def read_env_keys(env: dict, providers) -> dict[str, str]:
    """Keys found for `providers`. Providers whose terms forbid our use are never returned, even if present."""
    found = {}
    for p in providers:
        if p in TERMS_BLOCKED_PROVIDERS:
            continue
        val = next(((env.get(n) or "").strip() for n in env_names(p) if (env.get(n) or "").strip()), "")
        if val:
            found[p] = val
    return found


def list_models(call, c, spec: Spec, key: str | None) -> tuple[list[dict] | None, str]:
    """The provider's own list of what it serves: -> ([{'id', 'chat'}], '') or (None, why)."""
    headers = {}
    if key and spec.list_auth == "bearer":
        headers["Authorization"] = f"Bearer {key}"
    elif key and spec.list_auth:
        headers[spec.list_auth] = key
    r = call(c, "GET", spec.models_url, headers=headers)
    body = r.json() if r.status == 200 else None
    raw = (body.get("data") or body.get("models") or []) if isinstance(body, dict) else []
    rows = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        mid = str(m.get("id") or m.get("name") or "").removeprefix("models/")
        methods = m.get("supportedGenerationMethods")
        if mid:
            rows.append({"id": mid, "chat": "generateContent" in methods if isinstance(methods, list) else True})
    return (rows, "") if rows else (None, r.evidence()[:200])


def _hint(catalog: dict, provider: str, model: str) -> bool | None:
    m = ((catalog.get(provider) or {}).get("models") or {}).get(model)
    return bool(m.get("structured_output")) if m else None


def direct_candidates(call, c, catalog: dict, keys: dict[str, str]) -> tuple[list[Candidate], dict]:
    """Providers we can reach ourselves: those we hold a key for, plus the anonymous ones.

    meta['not_free'][provider] counts served models excluded because they are not free (e.g. OpenRouter paid)."""
    out: list[Candidate] = []
    meta: dict = {"not_free": {}, "list_errors": {}}
    for provider, spec in SPECS.items():
        key = None if spec.anonymous else keys.get(provider)
        if not spec.anonymous and not key:
            continue
        rows, err = list_models(call, c, spec, key)
        if rows is None:
            meta["list_errors"][provider] = err
            rows = [{"id": m, "chat": True} for m in spec.fallback]
            if not rows:
                out.append(Candidate("direct", provider, "(model list unavailable)", NO_ENDPOINT, note=err))
                continue
        for row in rows:
            if spec.free_rule == "suffix" and not row["id"].endswith(":free"):
                meta["not_free"][provider] = meta["not_free"].get(provider, 0) + 1
                continue
            status, note = classify_candidate(provider, row["id"])
            if not row["chat"]:
                status, note = NOT_CHAT, "provider says it does not support text generation"
            out.append(Candidate(
                "anon" if spec.anonymous else "direct", provider, row["id"], status, spec.base, key,
                f"{spec.litellm_prefix}/{row['id']}" if spec.litellm_prefix else None, spec.anonymous,
                note or spec.free_basis, _hint(catalog, provider, row["id"]), ("provider-list",)))
    for provider, why in TERMS_BLOCKED_PROVIDERS.items():
        out.append(Candidate("ledger", provider, "*", TERMS_BLOCKED, note=why, sources=("policy",)))
    return out, meta


def gateway_candidates(rows: list[dict], route: str, base: str, key: str | None, provider_of,
                       skip_providers: set[str]) -> list[Candidate]:
    """A gateway's catalog. Providers we already test directly are skipped (never spend one quota twice)."""
    out = []
    for m in rows:
        provider = provider_of(m)
        if provider in skip_providers:
            continue
        status, note = classify_candidate(provider, str(m["id"]))
        exec_status = str(m.get("execution_status") or "ready").lower()
        if status == TESTABLE and exec_status == "needskey":
            status, note = NEEDS_SIGNUP, f"free, but no key configured: add {env_names(provider)[-1]} to .env and re-run"
        elif status == TESTABLE and exec_status == "exhausted":
            status, note = EXHAUSTED, "free, but every key is rate-limited right now; retry later"
        out.append(Candidate(route, provider, str(m["id"]), status, base, key, None, False, note,
                             None, (route,)))
    return out


def omniroute_candidates(rows: list[dict], route: str, base: str) -> list[Candidate]:
    """OmniRoute lists models it may not be able to serve without credentials: the probe decides."""
    out = []
    for m in rows:
        mid = str(m["id"])
        provider = mid.split("/")[0] if "/" in mid else "omniroute"
        status, note = classify_candidate(provider, mid)
        out.append(Candidate(route, provider, mid, status, base, None, None, False, note, None, (route,)))
    return out


def ledger_candidates(catalog: dict, snapshot: dict, covered: set[str]) -> list[Candidate]:
    """Every free-priced model in models.dev (and Code Puppy's snapshot) for providers we cannot call.

    Nothing is probed: these rows exist so no free model is silently left out of the report."""
    def is_free(m: dict) -> bool:
        cost = m.get("cost") or {}
        return cost.get("input") == 0 and cost.get("output") == 0

    snap_ids = {(p, mid) for p, pd in snapshot.items() for mid, m in (pd.get("models") or {}).items() if is_free(m)}
    out, live_ids = [], set()
    for provider, pd in catalog.items():
        if LEDGER_ALIASES.get(provider, provider) in covered:
            continue
        for mid, m in (pd.get("models") or {}).items():
            if not is_free(m):
                continue
            live_ids.add((provider, mid))
            status, note = classify_candidate(provider, mid)
            if status == TESTABLE:
                status, note = NEEDS_SIGNUP, ("$0 list price per models.dev, needs an account/key we do not hold; "
                                              "confirm it is a genuine free tier, not bundled with a paid plan")
            sources = ("models.dev", "code_puppy") if (provider, mid) in snap_ids else ("models.dev",)
            out.append(Candidate("ledger", provider, mid, status, None, None, None, False, note,
                                 bool(m.get("structured_output")), sources))
    for provider, mid in sorted(snap_ids - live_ids):
        if LEDGER_ALIASES.get(provider, provider) not in covered:
            out.append(Candidate("ledger", provider, mid, NO_ENDPOINT, None, None, None, False,
                                 "only in Code Puppy's bundled snapshot; gone from the live models.dev catalog (stale)",
                                 None, ("code_puppy",)))
    return out


def unlock_list(cands: list[Candidate]) -> list[dict]:
    """Which accounts would unlock the most free models: the 'left on the table' list."""
    groups: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for cand in cands:
        if cand.status == NEEDS_SIGNUP:
            groups[(cand.route, cand.provider)].append(cand)
    rows = [{"route": r, "provider": p, "models": len(v), "flagged_structured": sum(1 for x in v if x.hint_structured),
             "env": env_names(p)[-1], "examples": [x.model for x in v[:3]]} for (r, p), v in groups.items()]
    return sorted(rows, key=lambda r: (-r["flagged_structured"], -r["models"], r["provider"]))
