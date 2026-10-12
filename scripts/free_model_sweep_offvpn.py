"""Find every free model in the repos you shared, and check which ones can run OUR pipeline.

Run it OFF the VPN. It is read-mostly and resumable; it never edits the app.

    uv run python scripts/free_model_sweep_offvpn.py --dry-run          # the plan, no network
    uv run python scripts/free_model_sweep_offvpn.py --discover-only    # list every candidate, spend no LLM calls
    uv run python scripts/free_model_sweep_offvpn.py                    # probe + pipeline-test (see budgets in --dry-run)
    uv run python scripts/free_model_sweep_offvpn.py --resume latest    # continue; only transient failures are retried
    uv run python scripts/free_model_sweep_offvpn.py --with-keys        # let freellmapi use keys for EXTRA providers

Sources (every model gets a verdict, or a written reason it has none):
  direct      providers you hold a key for (Google, Groq, OpenRouter, Mistral, Cerebras, NVIDIA) and the anonymous
              ones (OVH, Kilo): their OWN /models list is the truth about what they serve
  freellmapi  its whole catalog, through its Docker image (keyless providers work with no account)
  omniroute   its catalog, on the base image AND the -web image that bundles Chromium
  strata      a running Strata server, if one answers (--strata-url)
  models.dev  every $0-priced model of providers we cannot call, plus Code Puppy's bundled snapshot of it
              (Code Puppy ships no models of its own), listed with what account would unlock each

Each reachable model gets: (1) a tiny chat call, (2) a tiny json_schema call, (3) OUR two real steps
(news-impact + synthesizer) repeated --repeats times. USEFUL means both steps passed every time: that is
compatibility with our pipeline, not accuracy.

SAFETY. Probes spend YOUR free quota, so each provider has a call budget (see --dry-run) and models are
tested most-promising first; whatever is left is reported UNTESTED:budget and --resume continues tomorrow.
Providers whose terms forbid our use (Cohere) are never called. Keys are only ever sent to the provider
that issued them, except with --with-keys, which hands keys for providers we do NOT test directly to the
freellmapi container (0600 env-file, deleted right after start; scrubbed from every saved string).

Output: reports/free_model_sweep_<UTC>.md (the report), _verdicts.jsonl (one line per model), and the raw
probe log .jsonl that --resume reads.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime

import httpx

import gateway_claims_offvpn as G
import model_survey_offvpn as S
import sweep_data as D
import sweep_probe as PR
import sweep_report as REP
import sweep_sources as SRC
from gateway_claims_catalog import CATALOG_URL

CP_SNAPSHOT_URL = "https://raw.githubusercontent.com/mpfaffenberger/code_puppy/main/code_puppy/models_dev_api.json"
ROUTES = ("direct", "freellmapi", "omniroute", "strata")
TIMEOUTS = {"strata": 600, "omniroute-base": 180, "omniroute-web": 180}   # per LLM call, seconds
GATEWAY_PACE_S, GATEWAY_BUDGET = 5, 30
REPO, REPORTS = S.REPO, S.REPORTS


def pace_for(provider: str) -> float:
    spec = D.SPECS.get(provider)
    return spec.pace_s if spec else GATEWAY_PACE_S


def fetch_json(c: httpx.Client, url: str) -> dict:
    r = G.call(c, "GET", url)
    body = r.json() if r.status == 200 else None
    return body if isinstance(body, dict) else {}


def group_for(cand: D.Candidate) -> S.Group:
    """The survey Group that makes run_once call this candidate the way production would, or as close as possible."""
    name = f"{cand.route}:{cand.provider}"
    if name not in S.GROUPS:
        pace = int(pace_for(cand.provider))
        if cand.anonymous:
            group = S.Group(pace, (), (cand.model,), anon_base=cand.base)
        elif cand.litellm_id:     # the app's own make_model() path: gemini/, groq/, openrouter/
            group = S.Group(pace, (), (cand.litellm_id,))
        else:
            group = S.Group(pace, (), (cand.model,), bearer_base=cand.base, api_key=cand.key,
                            timeout_s=TIMEOUTS.get(cand.route, 120))
        S.GROUPS[name] = group
    return S.GROUPS[name]


def pipeline_result(recs: list, cand: D.Candidate, run: int) -> PR.Result:
    """Fold one run's two step records into a Result. Both steps must PASS: a failed news-impact step that the
    synthesizer papers over with a neutral baseline is NOT a pass (production would silently degrade too)."""
    steps = {r.step: r for r in recs}
    ok = all(s in steps and steps[s].status == "PASS" for s in ("news_impact", "synthesizer"))
    klasses = sorted({r.klass for r in recs if r.status == "FAIL" and r.klass} | ({"other"} if not ok and not recs else set()))
    detail = " | ".join(f"{r.step}: {r.status} {r.klass or r.detail[:70]}" for r in recs)
    return PR.Result(cand.uid, "pipeline", "PASS" if ok else "FAIL", ",".join(klasses), detail[:400],
                     round(sum(r.seconds for r in recs), 1), run)


async def probe_route(cands: list[D.Candidate], chat, c, store: PR.Store, budget: PR.Budget, workers: int) -> None:
    """Cheap probes, one provider at a time per thread, `workers` providers in parallel."""
    by_provider: dict[str, list[D.Candidate]] = defaultdict(list)
    for cand in cands:
        if cand.status == D.TESTABLE:
            by_provider[cand.provider].append(cand)
    gate = asyncio.Semaphore(workers)

    async def one(provider: str, group: list[D.Candidate]) -> None:
        async with gate:
            await asyncio.to_thread(PR.probe_provider, group, chat, c, store, budget, pace_for(provider))
    await asyncio.gather(*(one(p, g) for p, g in by_provider.items()))


async def pipeline_pass(W: S.Writer, prepared, cands: list[D.Candidate], store: PR.Store, budget: PR.Budget,
                        repeats: int, cap: int, deadline: float) -> None:
    """Our two real steps for models that answered chat. Serialised on purpose: the predictor's agents are
    module singletons, so two runs at once would trample each other's model."""
    predictor, tv, neutral = prepared
    answered = [x for x in cands if x.status == D.TESTABLE and (r := store.get(x.uid, "chat")) and r.verdict == "PASS"]
    todo = [x for x in PR.pipeline_order(answered, store)
            if any(store.needs(x.uid, "pipeline", i) for i in range(1, repeats + 1))]
    queues: dict[str, list[D.Candidate]] = defaultdict(list)
    for cand in (todo[:cap] if cap else todo):
        queues[cand.budget_key].append(cand)
    ready_at: dict[str, float] = {}
    while time.monotonic() < deadline:
        key, wait = PR.pick_next(queues, ready_at, time.monotonic())
        if key is None:
            return
        if wait:
            await asyncio.sleep(wait)
        cand = queues[key].pop(0)
        for run in range(1, repeats + 1):
            if not store.needs(cand.uid, "pipeline", run):
                continue
            if not budget.take(cand.budget_key, 2):
                store.add(PR.Result(cand.uid, "pipeline", "SKIP", "budget", "call budget used up", run=run))
                break
            before = len(W.recs)
            try:
                await S.run_once(predictor, group_for(cand), cand.test_id, run, tv, neutral, W)
            except Exception as e:
                store.add(PR.Result(cand.uid, "pipeline", "FAIL", "other", f"{type(e).__name__}: {e}"[:300], run=run))
                continue
            store.add(pipeline_result(W.recs[before:], cand, run))
        ready_at[key] = time.monotonic() + pace_for(cand.provider) * 2


def plan(args: argparse.Namespace) -> str:
    lines = ["PLAN", f"  routes: {', '.join(args.only)}   repeats: {args.repeats}   workers: {args.workers}   "
                     f"max models through the pipeline: {args.max_pipeline or 'no cap'}"]
    for p, s in D.SPECS.items():
        who = "anonymous (no key)" if s.anonymous else "your key from .env, if present"
        lines.append(f"  direct {p:11s} {who:32s} budget {args.budget or s.budget:3d} calls, 1 call/{s.pace_s:g}s")
    lines += [f"  gateways: freellmapi (Docker), omniroute base + -web (Docker): budget {args.budget or GATEWAY_BUDGET} calls/provider",
              f"  strata: only if a server answers at {args.strata_url}",
              "  models.dev + Code Puppy snapshot: listed with what would unlock each, never probed",
              "  never called: " + "; ".join(f"{p} ({why})" for p, why in D.TERMS_BLOCKED_PROVIDERS.items()),
              ("  --with-keys: keys for providers NOT tested directly go to the freellmapi container"
               if args.with_keys else "  no key leaves your machine except to the provider that issued it"),
              "  --discover-only: no LLM calls at all" if args.discover_only else
              "  WARNING: probes and pipeline runs spend your free quota; the budgets above cap it per provider"]
    return "\n".join(lines)


async def amain(args: argparse.Namespace) -> int:
    print(plan(args))
    if args.dry_run:
        print("\n--dry-run: nothing started, no network calls.")
        return 0
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    REPORTS.mkdir(exist_ok=True)
    resume = []
    if args.resume:
        found = PR.latest_results(REPORTS) if args.resume == "latest" else REPORTS.parent / args.resume
        resume = [found] if found and found.exists() else []
    store = PR.Store(REPORTS / f"free_model_sweep_{stamp}.jsonl", tuple(resume))
    W = S.Writer(f"sweep_{stamp}")
    started = time.monotonic()
    deadline = started + args.max_minutes * 60 if args.max_minutes else float("inf")
    meta: dict = {"started_utc": stamp, "git_head": subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
        "routes": ", ".join(args.only), "resumed_from": str(resume[0].name) if resume else "-", "outcome": "in progress"}
    cands: list[D.Candidate] = []
    catalog: dict = {}
    snapshot: dict = {}

    def finish(outcome: str, code: int) -> int:
        meta["outcome"] = outcome
        meta["minutes"] = round((time.monotonic() - started) / 60, 1)
        ledger = SRC.ledger_candidates(catalog, snapshot, {x.provider for x in cands if x.route != "ledger"})
        every = cands + ledger
        rows = [(x, *REP.verdict_for(x, store)) for x in every]
        md = REPORTS / f"free_model_sweep_{stamp}.md"
        md.write_text(REP.render(rows, store, meta, SRC.unlock_list(every)), encoding="utf-8")
        (REPORTS / f"free_model_sweep_{stamp}_verdicts.jsonl").write_text(
            "".join(json.dumps({**x.row(), "verdict": v, "reason": why}, ensure_ascii=False) + "\n" for x, v, why in rows),
            encoding="utf-8")
        print(f"\n{outcome}\nreport:   {md}\nverdicts: {md.with_name(md.stem + '_verdicts.jsonl')}")
        return code

    gateways = [r for r in args.only if r in ("freellmapi", "omniroute")]
    if gateways and not G.docker_ready()[0]:
        meta["gateways"] = f"SKIPPED ({', '.join(gateways)}): Docker is not running. Start Rancher/Docker Desktop and re-run."
        args.only = [r for r in args.only if r not in gateways]
    if not args.discover_only and not S.host_reachable(S.REQUIRED_HOST)[0]:
        return finish("ABORTED: Yahoo is not reachable -- the VPN looks ON. Disconnect and re-run.", 2)
    prepared = None
    if not args.discover_only:
        try:
            prepared = await S.prepare()
        except Exception as e:
            meta["pipeline"] = f"unavailable: {type(e).__name__}: {e}"[:300]

    budget = PR.Budget({p: (args.budget or s.budget) for p, s in D.SPECS.items()}, args.budget or GATEWAY_BUDGET)
    from dotenv import dotenv_values
    env = dotenv_values(REPO / ".env")

    async def run_route(found: list[D.Candidate], c: httpx.Client) -> None:
        cands.extend(found)
        if args.discover_only:
            return
        await probe_route(found, G.chat, c, store, budget, args.workers)
        if prepared:
            await pipeline_pass(W, prepared, found, store, budget, args.repeats, args.max_pipeline, deadline)

    try:
        with httpx.Client(timeout=60) as c:
            catalog, snapshot = fetch_json(c, CATALOG_URL), fetch_json(c, CP_SNAPSHOT_URL)
            meta["catalog"] = f"models.dev {sum(len(p.get('models', {})) for p in catalog.values())} models, " \
                              f"Code Puppy snapshot {sum(len(p.get('models', {})) for p in snapshot.values())}"
            if "direct" in args.only:
                keys = SRC.read_env_keys(env, [p for p, s in D.SPECS.items() if not s.anonymous])
                for v in keys.values():
                    G.remember_secret(v)
                found, dmeta = SRC.direct_candidates(G.call, c, catalog, keys)
                meta["direct: keys found"] = ", ".join(sorted(keys)) or "none"
                meta["direct: served but not free (excluded)"] = dmeta["not_free"] or "-"
                meta["direct: model lists that failed"] = ", ".join(dmeta["list_errors"]) or "-"
                await run_route(found, c)
            tested = {x.provider for x in cands if x.route in ("direct", "anon") and x.status == D.TESTABLE}
            if "freellmapi" in args.only:
                await sweep_freellmapi(c, env, tested, args, meta, run_route)
            if "omniroute" in args.only:
                await sweep_omniroute(c, meta, run_route)
            if "strata" in args.only:
                await sweep_strata(c, args, run_route)
    except KeyboardInterrupt:
        return finish("INTERRUPTED (partial results; --resume continues)", 130)
    finally:
        if not args.keep:
            G.docker("rm", "-f", G.FL_NAME)
            G.docker("rm", "-f", G.OR_NAME)
    return finish("completed", 0)


async def sweep_freellmapi(c, env, tested: set[str], args, meta: dict, run_route) -> None:
    extra = SRC.read_env_keys(env, [p for p in D.FREELLMAPI_PLATFORMS if p not in tested]) if args.with_keys else {}
    for v in extra.values():
        G.remember_secret(v)
    ok, detail, image, boot = G.fl_boot(c, extra)
    meta["freellmapi"] = f"{image or G.FL_IMAGE}: {detail}; keys handed over for: {sorted(extra) or 'none'}"
    unified = G.fl_login(c, boot)[0] if ok else None
    if not unified:
        return
    rows = G.model_rows(G.call(c, "GET", f"{boot['base']}/models", headers={"Authorization": f"Bearer {unified}"}))
    await run_route(SRC.gateway_candidates(rows, "freellmapi", boot["base"], unified, G.provider_of, tested), c)
    G.docker("rm", "-f", G.FL_NAME)


async def sweep_omniroute(c, meta: dict, run_route) -> None:
    root = f"http://127.0.0.1:{G.OR_PORT}"
    for variant, images in G.OR_VARIANTS:
        ok, detail, image = G.start_container(G.OR_NAME, images, G.OR_PORT, G.OR_PORT, {})
        if ok:
            ok, detail = G.wait_ready(c, f"{root}/api/monitoring/health", G.OR_NAME, seconds=150)
        meta[f"omniroute {variant}"] = f"{image or images[0]}: {detail}"
        if ok:
            rows = G.model_rows(G.call(c, "GET", f"{root}/v1/models"))
            await run_route(SRC.omniroute_candidates(rows, f"omniroute-{variant}", f"{root}/v1"), c)
        G.docker("rm", "-f", G.OR_NAME)


async def sweep_strata(c, args, run_route) -> None:
    auth = {"Authorization": f"Bearer {args.strata_key}"} if args.strata_key else {}
    status = G.call(c, "GET", f"{args.strata_url}/v1/status", headers=auth)
    if status.status == 0:
        await run_route([D.Candidate("strata", "strata", "Qwen3.8-Flash-Next", D.NO_ENDPOINT, note=(
            f"no Strata server at {args.strata_url}: it needs an NVIDIA/AMD PC. Start it there with --host 0.0.0.0 "
            "--api-key <secret> and re-run with --strata-url/--strata-key"))], c)
        return
    rows = G.model_rows(G.call(c, "GET", f"{args.strata_url}/v1/models", headers=auth))
    model = str(rows[0]["id"]) if rows else "any-model-name"
    await run_route([D.Candidate("strata", "strata", model, D.TESTABLE, f"{args.strata_url}/v1",
                                 args.strata_key or "any-api-key", note="local engine; no quota")], c)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=",".join(ROUTES), help=f"comma list of: {','.join(ROUTES)}")
    ap.add_argument("--with-keys", action="store_true", help="hand keys for providers NOT tested directly to freellmapi")
    ap.add_argument("--resume", nargs="?", const="latest", default=None, help="'latest' or a path to an earlier probe .jsonl")
    ap.add_argument("--budget", type=int, default=0, help="override every provider's call budget (default: per provider)")
    ap.add_argument("--repeats", type=int, default=2, help="pipeline runs per model (default 2)")
    ap.add_argument("--workers", type=int, default=6, help="providers probed in parallel (default 6)")
    ap.add_argument("--max-pipeline", type=int, default=40, help="max models put through the pipeline per route (0 = no cap)")
    ap.add_argument("--max-minutes", type=int, default=0, help="stop starting new pipeline runs after this long (0 = no limit)")
    ap.add_argument("--strata-url", default="http://127.0.0.1:8080")
    ap.add_argument("--strata-key", default=None)
    ap.add_argument("--discover-only", action="store_true", help="list every candidate; make no LLM calls")
    ap.add_argument("--keep", action="store_true", help="leave gateway containers running")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit; no network")
    args = ap.parse_args()
    args.only = [r.strip() for r in args.only.split(",") if r.strip() in ROUTES]
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
