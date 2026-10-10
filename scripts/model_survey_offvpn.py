"""Off-VPN survey: which free models really pass BOTH production LLM steps?

Run it OFF the VPN (AI-provider DNS is blocked on it):

    uv run python scripts/model_survey_offvpn.py --dry-run     # plan + call count, no calls
    uv run python scripts/model_survey_offvpn.py               # the real thing
    uv run python scripts/model_survey_offvpn.py --only ovh,kilo --repeats 3

Output (reports/ is git-ignored), written as it goes so Ctrl-C loses nothing:
    reports/model_survey_<UTC>.jsonl   one line per step, flushed immediately
    reports/model_survey_<UTC>.md      summary table + every failure reason

How it tests (and why it is not just predict()):
  * predict() SWALLOWS a news-impact failure ("news is OPTIONAL -> neutral"),
    so "predict() returned" would pass a model that produced garbage.
    Each step is therefore called and checked on its own.
  * Technicals are gathered ONCE, so every model sees identical inputs.
  * The model is pinned onto the module-level agent singletons, exactly like
    verify_offvpn.py --only-model (settings overrides are ignored at import).
  * Synthesizer input = the model's OWN news-impact result when that passed,
    else one shared neutral baseline (recorded in `synth_input`).

Keyless ("anonymous") endpoints are called with NO Authorization header at all:
sending a fake bearer key is treated as an invalid key and rejected.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import socket
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports"
TICKER = "RELIANCE.NS"
STEP_TIMEOUT_S = 240
REQUIRED_HOST = "query1.finance.yahoo.com"  # technicals need prices; also a VPN canary
MAX_LOG_LINES, MAX_LOG_CHARS = 12, 400


@dataclass(frozen=True)
class Group:
    pace_s: int                 # sleep between runs (anonymous tiers are ~2 req/min/model)
    hosts: tuple[str, ...]
    models: tuple[str, ...]     # litellm ids (keyed) or bare model ids (anonymous)
    anon_base: str | None = None


GROUPS: dict[str, Group] = {
    # Keyed -- your own keys from .env, built exactly like production (make_model).
    "gemini": Group(10, ("generativelanguage.googleapis.com",), (
        "gemini/gemini-2.5-flash", "gemini/gemini-2.5-flash-lite", "gemini/gemini-3.5-flash",
    )),
    "groq": Group(10, ("api.groq.com",), ("groq/qwen/qwen3.8-27b",)),
    # Anonymous -- no account, no key.
    "ovh": Group(65, ("oai.endpoints.kepler.ai.cloud.ovh.net",), (
        "Qwen3.8-27B", "Qwen3.5-397B-A17B", "gpt-oss-120b",
    ), anon_base="https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"),
    "kilo": Group(20, ("api.kilo.ai",), (
        "nvidia/nemotron-3-super-120b-a12b:free", "nvidia/nemotron-3-ultra-550b-a55b:free",
    ), anon_base="https://api.kilo.ai/api/gateway/v1"),
}


@dataclass
class Rec:
    group: str
    model: str
    run: int
    step: str                   # news_impact | synthesizer | group
    status: str                 # PASS | FAIL | SKIP
    seconds: float = 0.0
    detail: str = ""
    klass: str = ""
    synth_input: str = ""
    log: list[str] = field(default_factory=list)
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


def classify(text: str) -> str:
    """Coarse failure class so the summary says WHY, not just 'failed'."""
    t = text.lower()
    rules = (
        ("payment", ("402", "payment required", "billing")),
        ("auth", ("401", "403", "forbidden", "authentication", "unauthorized", "invalid api key")),
        ("rate_limited", ("429", "rate limit", "quota", "resource_exhausted", "too many requests")),
        ("overloaded", ("503", "unavailable", "high demand", "overloaded", "502", "504")),
        ("timeout", ("timeout", "timed out")),
        ("bad_output", ("validation error", "json_invalid", "literal_error", "field required",
                        "does not match the expected schema", "unparseable", "invalid json")),
        ("unsupported_params", ("unsupportedparams", "does not support parameters")),
    )
    for name, needles in rules:
        if any(n in t for n in needles):
            return name
    return "other"


# --- plumbing ---------------------------------------------------------------

@contextlib.contextmanager
def capture_logs():
    """Collect the predictor's (loguru) and chain's (stdlib) own log lines."""
    from loguru import logger

    lines: list[str] = []
    sink = logger.add(
        lambda m: lines.append(f"{m.record['level'].name:7s} {m.record['message']}"[:MAX_LOG_CHARS]),
        level="WARNING", format="{message}",
    )

    class _H(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(f"{record.levelname:7s} {record.getMessage()}"[:MAX_LOG_CHARS])

    handler, target = _H(level=logging.WARNING), logging.getLogger("price_predictor")
    target.addHandler(handler)
    try:
        yield lines
    finally:
        target.removeHandler(handler)
        logger.remove(sink)


def host_reachable(host: str) -> tuple[bool, str]:
    try:
        socket.gethostbyname(host)
    except OSError:
        return False, "DNS blocked"
    try:
        import ssl
        with socket.create_connection((host, 443), timeout=8) as s, \
                ssl.create_default_context().wrap_socket(s, server_hostname=host):
            return True, "ok"
    except OSError as e:
        return False, f"TLS/connect failed ({type(e).__name__})"


def anon_client(base: str):
    """OpenAI client that sends NO Authorization header (see module docstring)."""
    import httpx
    import openai

    async def _strip_auth(request: httpx.Request) -> None:
        request.headers.pop("authorization", None)

    http = httpx.AsyncClient(
        event_hooks={"request": [_strip_auth]}, timeout=httpx.Timeout(90, connect=15),
    )
    return openai.AsyncOpenAI(api_key="unused", base_url=base, http_client=http, max_retries=0)


def build_model(group: Group, model_id: str):
    if group.anon_base:
        from google.adk.models.lite_llm import LiteLlm
        return LiteLlm(model=f"openai/{model_id}", client=anon_client(group.anon_base), num_retries=0)
    from price_predictor.llm.factory import make_model
    return make_model(model_id)


async def anon_probe(group: Group) -> tuple[bool, str]:
    """One raw, no-auth chat call: does this endpoint serve anonymous users at all?"""
    import httpx

    body = {"model": group.models[0], "max_tokens": 8,
            "messages": [{"role": "user", "content": "say ok"}]}
    try:
        async with httpx.AsyncClient(timeout=45) as c:
            r = await c.post(f"{group.anon_base}/chat/completions", json=body)
    except httpx.HTTPError as e:
        return False, f"{type(e).__name__}: {e}"[:300]
    return r.status_code == 200, f"HTTP {r.status_code} {' '.join(r.text.split())[:240]}"


class Writer:
    def __init__(self, stamp: str) -> None:
        REPORTS.mkdir(exist_ok=True)
        self.jsonl = REPORTS / f"model_survey_{stamp}.jsonl"
        self.md = REPORTS / f"model_survey_{stamp}.md"
        self.recs: list[Rec] = []

    def add(self, rec: Rec) -> None:
        self.recs.append(rec)
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        mark = {"PASS": "ok  ", "FAIL": "FAIL", "SKIP": "skip"}[rec.status]
        print(f"  [{mark}] {rec.model:42s} run{rec.run} {rec.step:12s} {rec.seconds:5.1f}s "
              f"{rec.klass or ''} {rec.detail[:90]}", flush=True)

    def write_markdown(self, meta: dict) -> None:
        self.md.write_text(render_markdown(self.recs, meta), encoding="utf-8")


def render_markdown(recs: list[Rec], meta: dict) -> str:
    out = ["# Model survey (off-VPN)", "",
           *(f"- **{k}:** {v}" for k, v in meta.items() if k != "preflight"), "",
           "## Preflight", "", "| host | result |", "|---|---|",
           *(f"| {h} | {r} |" for h, r in meta.get("preflight", {}).items()), ""]
    by: dict[tuple[str, str], list[Rec]] = {}
    for r in recs:
        by.setdefault((r.group, r.model), []).append(r)

    def cell(rs: list[Rec], step: str) -> tuple[str, str]:
        rows = [r for r in rs if r.step == step and r.status != "SKIP"]
        ok = [r for r in rows if r.status == "PASS"]
        med = f"{statistics.median(r.seconds for r in ok):.0f}s" if ok else "-"
        return (f"{len(ok)}/{len(rows)}" if rows else "-"), med

    out += ["## Results (small n: treat as screening, not proof)", "",
            "| group | model | news-impact | synthesizer | median s (n/s) | verdict | failure classes |",
            "|---|---|---|---|---|---|---|"]
    for (g, m), rs in by.items():
        if all(r.status == "SKIP" for r in rs):
            out.append(f"| {g} | {m} | SKIP | SKIP | - | skipped | {rs[0].detail[:80]} |")
            continue
        (n, nm), (s, sm) = cell(rs, "news_impact"), cell(rs, "synthesizer")
        steps = [r for r in rs if r.status != "SKIP"]
        verdict = ("CANDIDATE" if all(r.status == "PASS" for r in steps)
                   else "NO" if not any(r.status == "PASS" for r in steps) else "PARTIAL")
        classes = ", ".join(sorted({r.klass for r in steps if r.status == "FAIL"})) or "-"
        out.append(f"| {g} | {m} | {n} | {s} | {nm}/{sm} | **{verdict}** | {classes} |")

    fails = [r for r in recs if r.status in ("FAIL", "SKIP")]
    if fails:
        out += ["", "## Every failure / skip, verbatim", ""]
        for r in fails:
            out += [f"### {r.status} {r.group} / {r.model} / run{r.run} / {r.step} [{r.klass or '-'}]",
                    "```", r.detail[:1200], *r.log, "```", ""]
    return "\n".join(out) + "\n"


# --- the actual test --------------------------------------------------------

async def run_once(P, group: Group, model_id: str, run: int, tv, neutral, w: Writer) -> None:
    from price_predictor.prediction.inputs import SynthesisInput

    canonical = P._resolve_ticker(TICKER)
    gname = group_name(group)
    try:
        model = build_model(group, model_id)
    except Exception as e:
        w.add(Rec(gname, model_id, run, "news_impact", "FAIL",
                  detail=f"cannot build model: {type(e).__name__}: {e}"[:600],
                  klass=classify(str(e))))
        return

    async def step(name: str, agent, coro_fn, synth_input: str = ""):
        agent.model = model
        t0 = time.time()
        with capture_logs() as lines:
            try:
                result = await asyncio.wait_for(coro_fn(), STEP_TIMEOUT_S)
                status, detail, klass = "PASS", describe(name, result), ""
            except Exception as e:
                result = None
                msg = " ".join(f"{type(e).__name__}: {e}".split())
                status, detail, klass = "FAIL", msg[:1200], classify(msg)
        w.add(Rec(gname, model_id, run, name, status, round(time.time() - t0, 1), detail, klass,
                  synth_input, lines[:MAX_LOG_LINES]))
        return result

    impact = await step("news_impact", P._news_impact_agent,
                        lambda: P.run_news_impact_agent(canonical))
    si = SynthesisInput(
        ticker=canonical, horizon="daily", as_of=datetime.now(P.IST), technical_view=tv,
        impact_assessment=impact or neutral, model_chain=("survey",),
    )
    await step("synthesizer", P._synthesizer_agent,
               lambda: P.synthesize_with_guardrails(si),
               synth_input="own" if impact else "neutral-baseline")


def group_name(group: Group) -> str:
    return next(n for n, g in GROUPS.items() if g is group)


def describe(step: str, result) -> str:
    if step == "news_impact":
        return f"sentiment={result.sentiment} confidence={result.confidence}"
    return f"direction={result.direction.value} confidence={result.confidence}"


def plan(selected: list[str], repeats: int) -> tuple[list[str], int]:
    lines, runs, pace = [], 0, 0
    for n in selected:
        g = GROUPS[n]
        kind = "ANONYMOUS (no key)" if g.anon_base else "keyed (.env)"
        lines.append(f"  {n:7s} {kind:20s} {len(g.models)} model(s) x {repeats} run(s), "
                     f"{g.pace_s}s pause")
        lines += [f"           - {m}" for m in g.models]
        runs += len(g.models) * repeats
        pace += len(g.models) * repeats * g.pace_s
    lines.append(f"  => {runs} runs; each = news-impact + synthesizer = at least 2 LLM calls "
                 f"(up to ~8 with retries/guardrails)")
    lines.append(f"  => roughly {(runs * 75 + pace) // 60} min. Gemini free quotas are per-model "
                 f"and small: keep --repeats low.")
    return lines, runs


async def amain(args: argparse.Namespace) -> int:
    selected = [s.strip() for s in args.only.split(",") if s.strip()]
    unknown = sorted(set(selected) - set(GROUPS))
    if unknown:
        print(f"unknown group(s): {unknown}; choose from {sorted(GROUPS)}")
        return 2
    plan_lines, _ = plan(selected, args.repeats)
    print("PLAN\n" + "\n".join(plan_lines))
    if args.dry_run:
        print("\n--dry-run: no network calls made.")
        return 0

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    w = Writer(stamp)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                          text=True).stdout.strip()
    meta: dict = {"started_utc": stamp, "git_head": head, "repeats": args.repeats,
                  "groups": ", ".join(selected), "outcome": "in progress", "preflight": {}}

    # Preflight: a VPN-blocked network must not turn into 20 misleading FAILs.
    hosts = [REQUIRED_HOST] + [h for n in selected for h in GROUPS[n].hosts]
    ok_by_host: dict[str, bool] = {}
    for h in hosts:
        ok, why = host_reachable(h)
        ok_by_host[h], meta["preflight"][h] = ok, ("OK" if ok else why)
        print(f"  preflight {h:44s} {'OK' if ok else why}")
    if not ok_by_host[REQUIRED_HOST]:
        meta["outcome"] = "ABORTED: yahoo not reachable -- the VPN looks ON. Disconnect and re-run."
        w.write_markdown(meta)
        print(f"\n{meta['outcome']}\nwrote {w.md}")
        return 2

    from loguru import logger
    logger.remove()  # our own sink captures per-step; keep the console readable

    from price_predictor.prediction import predictor as P
    from price_predictor.prediction.inputs import compose_technical_view

    print("\nGathering technicals ONCE (same inputs for every model)...", flush=True)
    try:
        tv = await compose_technical_view(P._resolve_ticker(TICKER))
    except Exception as e:
        meta["outcome"] = f"ABORTED: technicals failed: {type(e).__name__}: {e}"[:500]
        w.write_markdown(meta)
        print(meta["outcome"], f"\nwrote {w.md}")
        return 2
    neutral = P._degraded_impact(P._resolve_ticker(TICKER), "survey baseline")

    t_start = time.time()
    try:
        for n in selected:
            g = GROUPS[n]
            print(f"\n== {n} ==", flush=True)
            reason = None
            if not all(ok_by_host[h] for h in g.hosts):
                reason = "host unreachable from this network (see preflight)"
            elif g.anon_base:
                ok, reason_txt = await anon_probe(g)
                print(f"  anonymous probe ({g.models[0]}): {reason_txt}", flush=True)
                meta["preflight"][f"{g.hosts[0]} (anonymous chat)"] = reason_txt
                reason = None if ok else f"anonymous call rejected: {reason_txt}"
            if reason:
                for m in g.models:
                    w.add(Rec(n, m, 0, "group", "SKIP", detail=reason, klass=classify(reason)))
                continue
            for m in g.models:
                for i in range(1, args.repeats + 1):
                    await run_once(P, g, m, i, tv, neutral, w)
                    await asyncio.sleep(g.pace_s)
        meta["outcome"] = "completed"
    except (KeyboardInterrupt, asyncio.CancelledError):
        # asyncio.run turns Ctrl-C into a task cancel; record it, then let it propagate.
        meta["outcome"] = "INTERRUPTED (partial results below)"
        raise
    finally:
        meta["elapsed_min"] = round((time.time() - t_start) / 60, 1)
        w.write_markdown(meta)
        print(f"\nresults:\n  {w.md}\n  {w.jsonl}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=",".join(GROUPS), help=f"comma list of {sorted(GROUPS)}")
    ap.add_argument("--repeats", type=int, default=2, help="runs per model (default 2)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit; no network")
    args = ap.parse_args()
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
