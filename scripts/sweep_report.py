"""Verdicts and the report for the free-model sweep. Pure: takes candidates + the result store, returns text."""
from __future__ import annotations

from collections import Counter, defaultdict

from sweep_data import (
    INCONCLUSIVE,
    NOT_USEFUL,
    PARTIAL,
    SPECS,
    TESTABLE,
    UNTESTED,
    USEFUL,
    Candidate,
    final_verdict,
)
from sweep_probe import Store

ORDER = (USEFUL, PARTIAL, INCONCLUSIVE, NOT_USEFUL, UNTESTED)


def verdict_for(cand: Candidate, store: Store) -> tuple[str, str]:
    """One verdict per candidate. Anything we could not or would not call is UNTESTED, with the reason."""
    if cand.status != TESTABLE:
        return UNTESTED, f"{cand.status}: {cand.note}"
    chat, js = store.get(cand.uid, "chat"), store.get(cand.uid, "json")
    if chat and chat.verdict == "SKIP":
        return UNTESTED, "BUDGET: this provider's call budget ran out before the model was probed (re-run with --resume)"
    runs = [{"full": r.verdict == "PASS", "klasses": [k for k in r.klass.split(",") if k]}
            for r in store.runs(cand.uid) if r.verdict in ("PASS", "FAIL")]
    as_dict = lambda r: {"verdict": r.verdict, "klass": r.klass} if r else None  # noqa: E731
    return final_verdict(as_dict(chat), as_dict(js), runs)


def how_to_add(cand: Candidate) -> str:
    if cand.route == "direct" and cand.litellm_id:
        return f"works today: put `{cand.litellm_id}` in CHAIN_AGENTIC"
    if cand.route in ("direct", "anon"):
        return "needs a small provider entry in factory.py (OpenAI-compatible, base URL + key)"
    return f"only through the {cand.route} gateway (or a new provider entry)"


def avg_seconds(cand: Candidate, store: Store) -> str:
    secs = [r.seconds for r in store.runs(cand.uid) if r.verdict == "PASS"]
    return f"{sum(secs) / len(secs):.0f}s" if secs else "-"


def json_verdict(cand: Candidate, store: Store) -> str:
    r = store.get(cand.uid, "json")
    return r.verdict if r else "-"


def _table(head: tuple[str, ...], rows: list[tuple]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head), *("| " + " | ".join(map(str, r)) + " |" for r in rows)]


def render(rows: list[tuple[Candidate, str, str]], store: Store, meta: dict, unlock: list[dict]) -> str:
    by_verdict: dict[str, list[tuple[Candidate, str]]] = defaultdict(list)
    for cand, verdict, why in rows:
        by_verdict[verdict].append((cand, why))
    n = Counter({v: len(by_verdict[v]) for v in ORDER})
    out = ["# Free-model sweep (off-VPN)", "", *(f"- **{k}:** {v}" for k, v in meta.items()), "",
           "## How to read this", "",
           "`USEFUL` = both pipeline steps (news-impact and synthesizer) passed in every run. That means "
           "**compatible with our pipeline, not accurate**: judging accuracy needs the backtest. `PARTIAL` passed "
           "some runs. `INCONCLUSIVE` failed in a way that looks temporary (rate limit, timeout, 5xx): re-run with "
           "`--resume` and only those are retried. `NOT_USEFUL` will not work, or is not actually free. `UNTESTED` "
           "was never put through the pipeline test, and the reason is written next to it.", "",
           "## Executive summary", "",
           f"**{len(rows)} models/entries across every source.** " + ", ".join(f"{v}: {n[v]}" for v in ORDER), ""]
    coverage: dict[str, Counter] = defaultdict(Counter)
    for cand, verdict, _ in rows:
        coverage[cand.route][cand.status if cand.status != TESTABLE else verdict] += 1
    out += [*_table(("source / route", "breakdown"), [(r, ", ".join(f"{k} {v}" for k, v in c.most_common()))
                                                       for r, c in sorted(coverage.items())]), ""]
    out += [f"## {USEFUL}: candidates for the chain ({n[USEFUL]})", ""]
    useful = sorted(by_verdict[USEFUL], key=lambda cw: avg_seconds(cw[0], store))
    out += (_table(("provider", "model", "route", "json probe", "avg pipeline", "how to add"),
                   [(c.provider, c.model, c.route, json_verdict(c, store), avg_seconds(c, store),
                     how_to_add(c)) for c, _ in useful]) if useful else ["_none yet_"]) + [""]
    for verdict, title in ((PARTIAL, "PARTIAL: passed some runs"), (INCONCLUSIVE, "INCONCLUSIVE: retry with --resume")):
        out += [f"## {title} ({n[verdict]})", ""]
        out += ([f"- `{c.route}` **{c.provider}** `{c.model}`: {why}" for c, why in by_verdict[verdict]]
                or ["_none_"]) + [""]
    out += [f"## {NOT_USEFUL}: tested, will not work ({n[NOT_USEFUL]})", ""]
    reasons: dict[str, list[Candidate]] = defaultdict(list)
    for c, why in by_verdict[NOT_USEFUL]:
        reasons[why.split(";")[0][:70]].append(c)
    for why, cs in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
        out.append(f"- **{why}** ({len(cs)}): " + ", ".join(f"{c.provider}/{c.model}" for c in cs[:6])
                   + (" ..." if len(cs) > 6 else ""))
    out += ["", f"## {UNTESTED}: and why ({n[UNTESTED]})", ""]
    why_count = Counter(why.split(":")[0] for _, why in by_verdict[UNTESTED])
    out += [*_table(("reason", "count"), why_count.most_common()), ""]
    out += ["## Accounts to open: free models left on the table", "",
            "Ranked by how many models the catalog flags as structured-output capable (a hint, not proof). "
            "Add the key to `.env` as shown and re-run with `--with-keys --resume`: the sweep picks it up with no code "
            "change. (Cloudflare's key is `account_id:token`.) Terms differ per provider: read them before signing up.", ""]
    out += (_table(("provider", "via", "free models", "flagged structured", ".env name", "examples"),
                   [(u["provider"], u["route"], u["models"], u["flagged_structured"], u["env"], ", ".join(u["examples"]))
                    for u in unlock[:40]]) if unlock else ["_nothing outstanding_"]) + [""]
    used = sorted({c.provider for c, _, _ in rows if c.provider in SPECS and c.status == TESTABLE})
    out += ["## Terms and privacy of what was exercised", ""]
    out += [f"- **{p}**: {SPECS[p].free_basis}" for p in used]
    out += ["", "## Caveats", "",
            "- models.dev marks a model free only when its LIST PRICE is $0, so providers whose free tier is a quota on "
            "priced models (Google, Groq, Cloudflare Workers AI, AnyAPI...) are under-counted there; freellmapi's catalog "
            "covers those.",
            "- `coding-plan`/`token-plan` providers are paid subscriptions, listed as NOT_FREE.",
            "- Free tiers change often. Re-run to refresh; nothing here is permanent."]
    return "\n".join(out) + "\n"
