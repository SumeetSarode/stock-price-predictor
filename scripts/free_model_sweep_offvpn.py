"""Find every free model in the repos you shared, and check which ones can run OUR pipeline.

Just run it, no arguments: it does everything itself.

    uv run python scripts/free_model_sweep_offvpn.py

It starts Docker if needed, waits for you to leave the VPN, retries temporary failures on its own, resumes from
the previous run automatically, checkpoints the report as it goes, and never edits the app. Optional switches
(all unnecessary): --dry-run (print the plan), --discover-only (no LLM calls), --fresh (ignore earlier results),
--no-keys (never hand keys to the freellmapi container), --only direct,freellmapi,omniroute,strata.

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
import re
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import httpx

import gateway_claims_offvpn as G
import model_survey_offvpn as S
import sweep_data as D
import sweep_env as SE
import sweep_probe as PR
import sweep_report as REP
import sweep_sources as SRC
from gateway_claims_catalog import CATALOG_URL

CP_SNAPSHOT_URL = "https://raw.githubusercontent.com/mpfaffenberger/code_puppy/main/code_puppy/models_dev_api.json"
ROUTES = ("direct", "freellmapi", "omniroute", "strata")
TIMEOUTS = {"strata": 600, "omniroute-base": 180, "omniroute-web": 180}   # per LLM call, seconds
GATEWAY_PACE_S, GATEWAY_BUDGET = 5, 30
LATEST_MD, LATEST_VERDICTS = "free_model_sweep_latest.md", "free_model_sweep_latest_verdicts.jsonl"
REPO, REPORTS = S.REPO, S.REPORTS


def pace_for(provider: str) -> float:
    spec = D.SPECS.get(provider)
    return spec.pace_s if spec else GATEWAY_PACE_S


def fetch_json(c: httpx.Client, url: str) -> dict:
    r = G.call(c, "GET", url)
    body = r.json() if r.status == 200 else None
    return body if isinstance(body, dict) else {}


def ensure_docker() -> bool:
    """Docker answering, starting it first if it is not. (Tests replace this: it would launch a real app.)"""
    return SE.ensure_docker(lambda: G.docker_ready()[0], SE.launch_docker_app)


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
    # A failed step keeps the provider's own words (scrubbed): the class alone cannot tell "daily quota used up" from
    # "this request is bigger than the model's per-minute limit", and the fix for those is completely different.
    detail = " | ".join(f"{r.step}: {r.status} " + (f"{r.klass} ({G.clean(r.detail)[:150]})" if r.status == "FAIL" and r.klass
                                                     else r.klass or r.detail[:70]) for r in recs)
    return PR.Result(cand.uid, "pipeline", "PASS" if ok else "FAIL", ",".join(klasses), detail[:600],
                     round(sum(r.seconds for r in recs), 1), run)


async def probe_route(cands: list[D.Candidate], chat, c, store: PR.Store, budget: PR.Budget, workers: int) -> None:
    """Cheap probes, one provider at a time per thread, `workers` providers in parallel."""
    by_provider: dict[str, list[D.Candidate]] = defaultdict(list)
    for cand in cands:
        if cand.status == D.TESTABLE:
            by_provider[cand.provider].append(cand)
    by_provider = {p: PR.probe_order(group) for p, group in by_provider.items()}
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
              ("  keys for providers NOT tested directly are handed to the freellmapi container (--no-keys stops that)"
               if args.with_keys else "  no key leaves your machine except to the provider that issued it"),
              f"  temporary failures: up to {args.retry_passes} automatic retry passes; earlier results are resumed",
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
    from dotenv import dotenv_values
    env = dotenv_values(REPO / ".env")
    args.strata_url = env.get("STRATA_URL") or args.strata_url      # a GPU PC on your LAN: set it once in .env
    args.strata_key = env.get("STRATA_API_KEY") or args.strata_key
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

    def write_report(outcome: str) -> Path:
        """Rewritten after every route, so a crash or a closed lid still leaves a current report."""
        meta["outcome"] = outcome
        meta["minutes"] = round((time.monotonic() - started) / 60, 1)
        ledger = SRC.ledger_candidates(catalog, snapshot, {x.provider for x in cands if x.route != "ledger"})
        every = cands + ledger
        rows = [(x, *REP.verdict_for(x, store)) for x in every]
        md = REPORTS / f"free_model_sweep_{stamp}.md"
        verdicts = md.with_name(md.stem + "_verdicts.jsonl")
        md.write_text(REP.render(rows, store, meta, SRC.unlock_list(every)), encoding="utf-8")
        verdicts.write_text(
            "".join(json.dumps({**x.row(), "verdict": v, "reason": why}, ensure_ascii=False) + "\n" for x, v, why in rows),
            encoding="utf-8")
        shutil.copyfile(md, REPORTS / LATEST_MD)                         # a stable name to open or hand to someone
        shutil.copyfile(verdicts, REPORTS / LATEST_VERDICTS)
        return md

    def finish(outcome: str, code: int) -> int:
        md = write_report(outcome)
        print(f"\n{outcome}\nreport:   {md}\nlatest:   {REPORTS / LATEST_MD}")
        return code

    def checkpoint(route: str) -> None:
        write_report(f"in progress: {route} done")

    gateways = [r for r in args.only if r in ("freellmapi", "omniroute")]
    if gateways and not ensure_docker():
        meta["gateways"] = f"SKIPPED ({', '.join(gateways)}): Docker is not running and could not be started."
        args.only = [r for r in args.only if r not in gateways]
    if not args.discover_only and not SE.wait_for_off_vpn(
            lambda: S.host_reachable(S.REQUIRED_HOST)[0], args.wait_vpn_minutes):
        return finish(f"ABORTED: Yahoo is not reachable -- the VPN looks ON (waited {args.wait_vpn_minutes:g} min). "
                      "Disconnect and re-run.", 2)
    prepared = None
    if not args.discover_only:
        try:
            prepared = await S.prepare()
        except Exception as e:
            meta["pipeline"] = f"unavailable: {type(e).__name__}: {e}"[:300]

    budget = PR.Budget({p: (args.budget or s.budget) for p, s in D.SPECS.items()}, args.budget or GATEWAY_BUDGET)

    async def run_route(found: list[D.Candidate], c: httpx.Client) -> None:
        cands.extend(found)
        if args.discover_only:
            return
        todo = found
        for attempt in range(args.retry_passes + 1):
            if attempt:    # a rate limit or an overloaded server usually clears: wait, then redo only those
                todo = PR.transient_pending(found, store, args.repeats)
                if not todo or time.monotonic() >= deadline:
                    return
                print(f"retry pass {attempt}: {len(todo)} model(s) hit a temporary failure; waiting {args.retry_wait * attempt:g}s")
                await asyncio.sleep(args.retry_wait * attempt)
            await probe_route(todo, G.chat, c, store, budget, args.workers)
            if prepared:
                await pipeline_pass(W, prepared, todo, store, budget, args.repeats, args.max_pipeline, deadline)

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
                checkpoint("direct")
            tested = {x.provider for x in cands if x.route in ("direct", "anon") and x.status == D.TESTABLE}
            if "freellmapi" in args.only:
                await sweep_freellmapi(c, env, tested, args, meta, run_route)
                checkpoint("freellmapi")
            if "omniroute" in args.only:
                await sweep_omniroute(c, meta, run_route)
                checkpoint("omniroute")
            if "strata" in args.only:
                await sweep_strata(c, args, run_route)
    except KeyboardInterrupt:
        return finish("INTERRUPTED (partial results saved; the next run resumes automatically)", 130)
    except Exception as e:
        return finish(f"CRASHED: {type(e).__name__}: {e} (partial results saved; the next run resumes automatically)", 1)
    finally:
        if not args.keep:
            G.docker("rm", "-f", G.FL_NAME)
            G.docker("rm", "-f", G.OR_NAME)
    return finish("completed", 0)


UNKNOWN_PLATFORM = re.compile(r"unknown provider platform: ([\w.-]+)")


def boot_freellmapi(c, keys: dict[str, str], boot=None) -> tuple[bool, str, str, dict, dict[str, str], list[str]]:
    """Start the gateway; if it refuses a platform name it does not know, drop THAT key and try again.

    The gateway validates every key's platform at startup and one unknown name aborts the whole container, taking
    every other platform down with it (GitHub Models was retired upstream and crashed a whole run this way). Our
    list of names can always go stale, so tolerate it: -> (ok, detail, image, boot_info, keys_used, dropped).
    """
    boot = boot or G.fl_boot
    keys, dropped = dict(keys), []
    while True:
        ok, detail, image, info = boot(c, keys)
        bad = None if ok else UNKNOWN_PLATFORM.search(detail)
        if not bad or bad.group(1) not in keys:
            return ok, detail, image, info, keys, dropped
        dropped.append(bad.group(1))
        del keys[bad.group(1)]


async def sweep_freellmapi(c, env, tested: set[str], args, meta: dict, run_route) -> None:
    extra = SRC.read_env_keys(env, [p for p in D.FREELLMAPI_PLATFORMS if p not in tested]) if args.with_keys else {}
    for v in extra.values():
        G.remember_secret(v)
    ok, detail, image, boot, extra, dropped = boot_freellmapi(c, extra)
    meta["freellmapi"] = (f"{image or G.FL_IMAGE}: {detail}; keys handed over for: {sorted(extra) or 'none'}"
                          + (f"; NOT handed over, the gateway does not support these platforms: {sorted(dropped)}"
                             if dropped else ""))
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=",".join(ROUTES), help=f"comma list of: {','.join(ROUTES)}")
    ap.add_argument("--no-keys", dest="with_keys", action="store_false",
                    help="never hand keys (for providers we do not test directly) to the freellmapi container")
    ap.add_argument("--resume", default="latest", help="'latest' (default) or a path to an earlier probe .jsonl")
    ap.add_argument("--fresh", action="store_true", help="ignore earlier results and start over")
    ap.add_argument("--retry-passes", type=int, default=3, help="automatic retries of temporary failures (default 3)")
    ap.add_argument("--retry-wait", type=float, default=90, help="seconds before retry pass 1; grows each pass")
    ap.add_argument("--wait-vpn-minutes", type=float, default=30, help="wait this long for you to leave the VPN")
    ap.add_argument("--budget", type=int, default=0, help="override every provider's call budget (default: per provider)")
    ap.add_argument("--repeats", type=int, default=2, help="pipeline runs per model (default 2)")
    ap.add_argument("--workers", type=int, default=6, help="providers probed in parallel (default 6)")
    ap.add_argument("--max-pipeline", type=int, default=40, help="max models put through the pipeline per route (0 = no cap)")
    ap.add_argument("--max-minutes", type=int, default=0,
                    help="stop starting new pipeline runs after this many minutes (default 0 = no limit)")
    ap.add_argument("--strata-url", default="http://127.0.0.1:8080")
    ap.add_argument("--strata-key", default=None)
    ap.add_argument("--discover-only", action="store_true", help="list every candidate; make no LLM calls")
    ap.add_argument("--keep", action="store_true", help="leave gateway containers running")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit; no network")
    return ap


def normalize(args: argparse.Namespace) -> argparse.Namespace:
    args.only = [r.strip() for r in args.only.split(",") if r.strip() in ROUTES]
    args.resume = None if args.fresh else args.resume
    return args


def main() -> int:
    try:
        return asyncio.run(amain(normalize(build_parser().parse_args())))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
