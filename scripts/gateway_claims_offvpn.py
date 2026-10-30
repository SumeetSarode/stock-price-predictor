"""Fair, evidence-first test of the claims made by freellmapi and OmniRoute.

Run it OFF the VPN, with Docker running (Rancher/Docker Desktop):

    uv run python scripts/gateway_claims_offvpn.py --dry-run          # plan only, no network
    uv run python scripts/gateway_claims_offvpn.py                    # keyless: nothing of yours leaves
    uv run python scripts/gateway_claims_offvpn.py --with-keys        # ALSO hand your free-tier keys
                                                                      # to the gateway container

WHY THIS EXISTS. Both projects are popular, so "it did not work for us" is not a verdict.
Our one OmniRoute attempt (502 on `auto`) turned out to match its own tracker (#15969:
the free path needs a Chromium browser the base image does not ship). Every claim below is
therefore tested the way the project documents it, on the official image, and recorded as:

    CONFIRMED      worked as documented
    NOT_CONFIRMED  followed the docs, it did not work; evidence attached
    INCONCLUSIVE   could not test fairly (missing prerequisite, could not start, ...)
    NOT_TESTED     deliberately skipped; the reason is written down

Nothing is declared "useless". Containers are removed at the end (unless --keep); keys travel
via a 0600 temp --env-file that is deleted right after `docker run`; every saved string is
scrubbed of keys/tokens. Output (reports/ is git-ignored):
    reports/gateway_claims_<UTC>.md      ledger + evidence + our-pipeline results + container logs
    reports/gateway_claims_<UTC>.jsonl   one line per finding, flushed as it happens
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

import httpx

import model_survey_offvpn as S  # reuse: Writer/run_once/prepare/host_reachable/render_markdown
from gateway_claims_catalog import catalog_phase
from gateway_claims_data import (  # what we test and why: pure data, kept apart from behaviour
    CLAIMS,
    CONFIRMED,
    INCONCLUSIVE,
    NOT_CONFIRMED,
    NOT_TESTED_REASONS,
)
from gateway_claims_strata import strata_checks
from sweep_data import as_text
from sweep_sources import read_env_keys

REPO, REPORTS = S.REPO, S.REPORTS


FL_IMAGE, FL_NAME, FL_PORT = "ghcr.io/tashfeenahmed/freellmapi:latest", "claims-freellmapi", 3001
OR_NAME, OR_PORT = "claims-omniroute", 20128
OR_VARIANTS = (  # the base image has NO Chromium; the -web image bundles it (OmniRoute docs)
    ("base", ("diegosouzapw/omniroute:latest",)),
    ("web", ("diegosouzapw/omniroute:latest-web", "diegosouzapw/omniroute:next-web")),
)
KEYLESS = {"kilo", "ovh", "aihorde", "llm7", "pollinations"}
# Providers whose key may go to the freellmapi container (names resolved by sweep_sources.env_names).
# Cohere is excluded on purpose (its terms forbid personal use); Together/SambaNova need a deposit.
KEY_PLATFORMS = ("google", "groq", "mistral", "cerebras", "openrouter", "nvidia")
PREFERRED_PINS = (r"qwen3\.8", r"gemini-2\.5-flash$", r"gemini-3\.5-flash", r"gpt-oss-120b")


@dataclass
class Finding:
    claim_id: str
    phase: str
    verdict: str
    evidence: str
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


# --- secrets hygiene --------------------------------------------------------

_SECRETS: list[str] = []
_PATTERNS = (re.compile(r"gsk_[A-Za-z0-9]{20,}"), re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
             re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{16,}"), re.compile(r"freellmapi-[A-Za-z0-9]{8,}"))


def remember_secret(value: str | None) -> None:
    if value and len(value) >= 6 and value not in _SECRETS:
        _SECRETS.append(value)


def clean(text: str) -> str:
    for s in _SECRETS:
        text = text.replace(s, "***")
    for p in _PATTERNS:
        text = p.sub("***", text)
    return text


class Ledger:
    def __init__(self, stamp: str) -> None:
        REPORTS.mkdir(exist_ok=True)
        self.jsonl, self.md = REPORTS / f"gateway_claims_{stamp}.jsonl", REPORTS / f"gateway_claims_{stamp}.md"
        self.findings: list[Finding] = []

    def add(self, claim_id: str, phase: str, verdict: str, evidence: str) -> None:
        f = Finding(claim_id, phase, verdict, clean(evidence)[:2500])
        self.findings.append(f)
        with self.jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(f), ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print(f"  [{verdict:13s}] {claim_id:4s} {phase:9s} {f.evidence[:100]!r}", flush=True)


# --- http -------------------------------------------------------------------

@dataclass
class Resp:
    status: int
    seconds: float
    text: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    error: str = ""

    def json(self):
        try:
            return json.loads(self.text)
        except ValueError:
            return None

    def evidence(self, content: str = "") -> str:
        if self.error:
            return f"no response in {self.seconds:.1f}s: {self.error}"
        hdrs = {k: v for k, v in self.headers.items() if k.startswith("x-")}
        body = content or " ".join(self.text.split())
        return f"HTTP {self.status} in {self.seconds:.1f}s {hdrs or ''} :: {body[:500]}"


def call(c: httpx.Client, method: str, url: str, **kw) -> Resp:
    t0 = time.time()
    try:
        r = c.request(method, url, **kw)
    except httpx.HTTPError as e:
        return Resp(0, time.time() - t0, error=f"{type(e).__name__}: {e}")
    return Resp(r.status_code, time.time() - t0, r.text,
                {k.lower(): v for k, v in r.headers.items() if k.lower().startswith("x-")})


def chat(c: httpx.Client, base: str, key: str | None, model: str,
         prompt: str = "Reply with exactly the word: ok", **extra) -> tuple[Resp, str]:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 400, **extra}
    r = call(c, "POST", f"{base}/chat/completions", json=body, headers=headers)
    try:
        content = as_text(r.json()["choices"][0]["message"]["content"])
    except (TypeError, KeyError, IndexError, AttributeError):
        content = ""
    return r, content


def model_rows(r: Resp) -> list[dict]:
    j = r.json()
    rows = j.get("data") if isinstance(j, dict) else None
    return [m for m in rows if isinstance(m, dict) and "id" in m] if isinstance(rows, list) else []


def provider_of(m: dict) -> str:
    return str(m.get("owned_by") or m.get("platform") or str(m["id"]).split("/")[0])


def pick_pins(ids: list[str], n: int = 2) -> list[str]:
    out: list[str] = []
    for pat in PREFERRED_PINS:
        out += [i for i in ids if re.search(pat, i) and i not in out][:1]
    return out[:n]


# --- docker -----------------------------------------------------------------

def docker(*args: str, timeout: int = 120) -> tuple[int, str]:
    try:
        p = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, f"{type(e).__name__}: {e}"
    return p.returncode, (p.stdout + p.stderr).strip()


def docker_ready() -> tuple[bool, str]:
    rc, out = docker("info", "--format", "{{.ServerVersion}}", timeout=30)
    return rc == 0, out[:300]


def start_container(name: str, images: tuple[str, ...], host_port: int, port: int,
                    env: dict[str, str]) -> tuple[bool, str, str]:
    """-> (ok, detail, image_used). Tries each image tag in order until one pulls."""
    docker("rm", "-f", name)
    errors, used = [], ""
    for img in images:
        rc, out = docker("pull", img, timeout=900)
        if rc == 0:
            used = img
            break
        errors.append(f"{img}: {out[-200:]}")
    if not used:
        return False, "pull failed: " + " | ".join(errors), ""
    args = ["run", "-d", "--name", name, "-p", f"127.0.0.1:{host_port}:{port}"]
    env_path = None
    if env:  # --env-file, not -e: secrets must not appear in `ps` output
        fd, env_path = tempfile.mkstemp(suffix=".env")  # mkstemp creates it 0600
        with os.fdopen(fd, "w") as fh:
            fh.write("".join(f"{k}={v}\n" for k, v in env.items()))
        args += ["--env-file", env_path]
    try:
        rc, out = docker(*args, used, timeout=120)
    finally:
        if env_path:
            os.unlink(env_path)
    return rc == 0, out[-300:], used


def wait_ready(c: httpx.Client, url: str, name: str, seconds: int = 120) -> tuple[bool, str]:
    last = "never answered"
    for _ in range(seconds // 3):
        r = call(c, "GET", url)
        if r.status == 200:
            return True, f"HTTP 200 after readiness poll ({url})"
        last = r.evidence()
        rc, out = docker("inspect", "-f", "{{.State.Running}}", name)
        if rc == 0 and out.strip() == "false":
            return False, "container exited early: " + clean(docker("logs", "--tail", "40", name)[1])[-500:]
        time.sleep(3)
    return False, f"not ready after {seconds}s; last: {last}"


def container_logs(name: str, tail: int = 80) -> str:
    return clean(docker("logs", "--tail", str(tail), name, timeout=30)[1])


# --- freellmapi -------------------------------------------------------------

def read_keys() -> dict[str, str]:
    from dotenv import dotenv_values
    keys = read_env_keys(dotenv_values(REPO / ".env"), KEY_PLATFORMS)
    for val in keys.values():
        remember_secret(val)
    return keys


def fl_config(email: str, password: str, keys: dict[str, str]) -> str:
    return json.dumps({"admin": {"email": email, "password": password},
                       "keys": [{"platform": p, "key": k, "label": "claims-check"} for p, k in keys.items()]})


def fl_boot(c: httpx.Client, keys: dict[str, str]) -> tuple[bool, str, str, dict]:
    """Start the freellmapi container with a declarative admin + `keys`. -> (ok, detail, image, boot info)."""
    root = f"http://127.0.0.1:{FL_PORT}"
    email, password = "claims@local.test", secrets.token_hex(8) + "Aa1!"
    remember_secret(password)
    env = {"ENCRYPTION_KEY": secrets.token_hex(32), "PORT": str(FL_PORT), "NODE_ENV": "production",
           "FREEAPI_CONFIG_JSON": fl_config(email, password, keys)}
    ok, detail, image = start_container(FL_NAME, (FL_IMAGE,), FL_PORT, FL_PORT, env)
    if ok:
        ok, detail = wait_ready(c, f"{root}/api/ping", FL_NAME)
    return ok, detail, image, {"root": root, "base": f"{root}/v1", "email": email, "password": password}


def fl_login(c: httpx.Client, boot: dict) -> tuple[str | None, Resp, Resp | None]:
    """Log in as the declared admin and fetch the unified API key. -> (unified key | None, login, key response)."""
    login = call(c, "POST", f"{boot['root']}/api/auth/login", json={"email": boot["email"], "password": boot["password"]})
    token = (login.json() or {}).get("token") if login.status == 200 else None
    remember_secret(token)
    key_resp = (call(c, "GET", f"{boot['root']}/api/settings/api-key", headers={"Authorization": f"Bearer {token}"})
                if token else None)
    unified = (key_resp.json() or {}).get("apiKey") if key_resp and key_resp.status == 200 else None
    remember_secret(unified)
    return unified, login, key_resp


def freellmapi_phase(L: Ledger, c: httpx.Client, phase: str, keys: dict[str, str]) -> dict | None:
    """Returns {'base','key','models'} for the pipeline test, or None if it never came up."""
    ok, detail, image, boot = fl_boot(c, keys)
    base = boot["base"]
    L.add("FL1", phase, CONFIRMED if ok else (INCONCLUSIVE if "pull failed" in detail else NOT_CONFIRMED),
          f"image={image or FL_IMAGE}; {detail}")
    if not ok:
        return None

    unified, login, key_resp = fl_login(c, boot)
    L.add("FL2", phase, CONFIRMED if unified else NOT_CONFIRMED,
          f"login: {login.evidence()[:160]} | api-key: {key_resp.evidence()[:160] if key_resp else 'skipped (no token)'}")
    if not unified:
        return None

    models = call(c, "GET", f"{base}/models", headers={"Authorization": f"Bearer {unified}"})
    rows = model_rows(models)
    by_provider = Counter(provider_of(m) for m in rows)
    ids = [str(m["id"]) for m in rows]
    wanted = {k: any(re.search(k, i) for i in ids) for k in (r"qwen3\.8", r"gemini-3\.5-flash")}
    L.add("FL3", phase, CONFIRMED if len(rows) >= 300 else NOT_CONFIRMED,
          f"{len(rows)} models, {len(by_provider)} providers (claim: 635 / 34; free installs get a "
          f"30-day-late snapshot). newest-model presence {wanted}. top providers "
          f"{by_provider.most_common(8)}. sample entry keys: {sorted(rows[0]) if rows else '-'}")

    ready = call(c, "GET", f"{base}/models?execution_status=ready", headers={"Authorization": f"Bearer {unified}"})
    ready_rows = model_rows(ready)
    L.add("FL4", phase, CONFIRMED if ready.status == 200 and ready.json() is not None else NOT_CONFIRMED,
          f"{len(ready_rows)} ready of {len(rows)}; by provider {Counter(provider_of(m) for m in ready_rows).most_common(8)}; "
          f"{ready.evidence()[:120]}")
    ready_ids = [str(m["id"]) for m in ready_rows]

    keyless_ids = [str(m["id"]) for m in ready_rows if provider_of(m) in KEYLESS][:3]
    if phase == "keyless":
        got = [(i, *chat(c, base, unified, i)) for i in keyless_ids]
        good = [i for i, r, txt in got if r.status == 200 and txt.strip()]
        L.add("FL5", phase, CONFIRMED if good else NOT_CONFIRMED,
              (f"keyless ready models: {keyless_ids or 'NONE reported ready'}; answered: {good}; " +
               " | ".join(f"{i}: {r.evidence(txt)[:140]}" for i, r, txt in got)))

    autos = [chat(c, base, unified, "auto") for _ in range(3)]
    n_ok = sum(1 for r, txt in autos if r.status == 200 and txt.strip())
    L.add("FL6", phase, CONFIRMED if n_ok >= 2 else NOT_CONFIRMED,
          f"{n_ok}/3 auto calls answered. " + " | ".join(r.evidence(txt)[:170] for r, txt in autos))

    if phase == "keyed":
        trails = [{k: v for k, v in r.headers.items() if "fallback" in k} for r, _ in autos]
        hit = any(t for t in trails)
        L.add("FL7", phase, CONFIRMED if hit and n_ok else INCONCLUSIVE,
              f"fallback headers per call: {trails}. " + ("" if hit else
              "No dead hop was hit in these calls, so failover was not exercised."))

    structured_check(L, c, base, unified, phase)
    return {"base": base, "key": unified, "models": ["auto", *pick_pins(ready_ids)]}


def structured_check(L: Ledger, c: httpx.Client, base: str, key: str, phase: str,
                     claim_id: str = "FL8", model: str = "auto") -> None:
    try:
        from price_predictor.agents.news_impact.agent import ImpactAssessment
    except Exception as e:
        L.add(claim_id, phase, INCONCLUSIVE, f"could not import ImpactAssessment: {e}")
        return
    rf = {"type": "json_schema", "json_schema": {"name": "ImpactAssessment",
                                                 "schema": ImpactAssessment.model_json_schema()}}
    prompt = ("Assess the likely 1-week price impact on RELIANCE.NS of: record quarterly profit, "
              "Jio adds 8M subscribers, crude oil falls 3%. Answer ONLY with JSON matching the schema.")
    rows = []
    for _ in range(3):
        r, txt = chat(c, base, key, model, prompt, response_format=rf, max_tokens=1500)
        try:
            ImpactAssessment.model_validate_json(txt)
            valid = True
        except Exception:
            valid = False
        rows.append((valid, r.headers.get("x-routed-via", "?"), r.evidence(txt)[:150]))
    n = sum(v for v, _, _ in rows)
    L.add(claim_id, phase, CONFIRMED if n == 3 else NOT_CONFIRMED,
          f"{n}/3 answers validated against OUR ImpactAssessment schema. " +
          " | ".join(f"valid={v} via={via} {ev}" for v, via, ev in rows))


# --- OmniRoute --------------------------------------------------------------

def judged(variant: str, r: Resp, content: str) -> str:
    """One fairness rule for every OmniRoute probe.

    A missing-Chromium failure on the BASE image is a documented prerequisite, not a failure of
    the claim: INCONCLUSIVE there, and the -web run decides.
    """
    if r.status == 200 and content.strip():
        return CONFIRMED
    needs_browser = bool(re.search(r"(?i)chromium|playwright|browser", r.text + r.error))
    return INCONCLUSIVE if variant == "base" and needs_browser else NOT_CONFIRMED


def omniroute_phase(L: Ledger, c: httpx.Client, variant: str, images: tuple[str, ...]) -> dict | None:
    root = f"http://127.0.0.1:{OR_PORT}"
    base = f"{root}/v1"
    claim_zero = "OR2" if variant == "base" else "OR3"
    ok, detail, image = start_container(OR_NAME, images, OR_PORT, OR_PORT, {})
    if ok:
        ok, detail = wait_ready(c, f"{root}/api/monitoring/health", OR_NAME, seconds=150)
    if variant == "base":
        L.add("OR1", variant, CONFIRMED if ok else (INCONCLUSIVE if "pull failed" in detail else NOT_CONFIRMED),
              f"image={image or images[0]}; {detail}")
    if not ok:
        if variant != "base":
            L.add(claim_zero, variant, INCONCLUSIVE, f"could not start {images}: {detail}")
        return None

    models = call(c, "GET", f"{base}/models")
    rows = model_rows(models)
    ids = [str(m["id"]) for m in rows]
    prefixes = Counter(i.split("/")[0] for i in ids if "/" in i)
    zero, txt = chat(c, base, None, "auto")
    answered = zero.status == 200 and bool(txt.strip())
    verdict = judged(variant, zero, txt)
    note = " (missing-Chromium prerequisite: the -web run decides)" if verdict == INCONCLUSIVE else ""
    L.add(claim_zero, variant, verdict,
          f"{len(ids)} models listed, top prefixes {prefixes.most_common(6)}. auto: {zero.evidence(txt)[:260]}{note}")

    oc_ids = [i for i in ids if i.startswith("oc/")][:3]
    got = [(i, *chat(c, base, None, i)) for i in oc_ids]
    good = [i for i, r, t in got if r.status == 200 and t.strip()]
    oc_verdict = CONFIRMED if good else (judged(variant, got[0][1], got[0][2]) if got else INCONCLUSIVE)
    L.add("OR4", variant, oc_verdict,
          f"oc/ models listed: {oc_ids or 'none'}; answered {good}; " +
          " | ".join(f"{i}: {r.evidence(t)[:140]}" for i, r, t in got))
    r5, t5 = chat(c, base, None, "auto/coding:free")
    L.add("OR5", variant, judged(variant, r5, t5), r5.evidence(t5)[:300])
    workers = []
    if answered:
        workers.append("auto")
    for i in [i for i in ids if i.startswith("oc/")][:4]:
        r, t = chat(c, base, None, i)
        if r.status == 200 and t.strip():
            workers.append(i)
    return {"base": base, "key": None, "models": workers[:3]} if workers else None


# --- our pipeline through a gateway -----------------------------------------

async def pipeline(W: S.Writer, prepared, name: str, ctx: dict, repeats: int) -> None:
    if not ctx or not ctx["models"]:
        return
    P, tv, neutral = prepared
    S.GROUPS[name] = S.Group(5, (), tuple(ctx["models"]), bearer_base=ctx["base"], api_key=ctx["key"],
                             timeout_s=ctx.get("timeout_s", 120))
    for m in ctx["models"]:
        for i in range(1, repeats + 1):
            await S.run_once(P, S.GROUPS[name], m, i, tv, neutral, W)
            await asyncio.sleep(5)


def pipeline_finding(L: Ledger, W: S.Writer, claim_id: str, group: str, phase: str) -> None:
    recs = [r for r in W.recs if r.group == group]
    if not recs:
        L.add(claim_id, phase, INCONCLUSIVE, "no pipeline runs happened (gateway never became usable)")
        return
    passed = sum(r.status == "PASS" for r in recs)
    both = {(r.model, r.run): {x.step for x in recs if x.model == r.model and x.run == r.run and x.status == "PASS"}
            for r in recs}
    full = sum(1 for steps in both.values() if steps >= {"news_impact", "synthesizer"})
    L.add(claim_id, phase, CONFIRMED if full and full == len(both) else (NOT_CONFIRMED if not full else INCONCLUSIVE),
          f"{passed}/{len(recs)} steps passed; {full}/{len(both)} runs passed BOTH steps. Per-step table below.")


# --- report -----------------------------------------------------------------

def render(L: Ledger, W: S.Writer, meta: dict, logs: dict[str, str]) -> str:
    by_id = {c.id: c for c in CLAIMS}
    out = ["# Gateway claims check (off-VPN)", "", *(f"- **{k}:** {v}" for k, v in meta.items()), "",
           "## How to read this", "",
           "`CONFIRMED` worked as documented. `NOT_CONFIRMED` we followed the docs and it did not "
           "work (evidence below). `INCONCLUSIVE` we could not test fairly. `NOT_TESTED` skipped on "
           "purpose. **Nothing here is a verdict that a project is useless.**", "",
           "## Verdict ledger", "", "| id | project | claim | phase | verdict |", "|---|---|---|---|---|"]
    out += [f"| {f.claim_id} | {by_id[f.claim_id].repo} | {by_id[f.claim_id].text} | {f.phase} | **{f.verdict}** |"
            for f in L.findings]
    out += ["", "## Evidence, verbatim", ""]
    for f in L.findings:
        c = by_id[f.claim_id]
        out += [f"### {f.claim_id} [{f.phase}] {f.verdict}", f"*Claim:* {c.text}  ", f"*Source:* {c.source}", "",
                "```", f.evidence, "```", ""]
    out += ["## Our pipeline (news-impact + synthesizer) through each gateway", "",
            S.render_markdown(W.recs, {"outcome": meta.get("outcome", ""), "preflight": {}}), "",
            "## NOT TESTED, and why", ""]
    out += [f"- **{what}**: {why}" for what, why in NOT_TESTED_REASONS]
    out += ["", "## Container logs (last lines, secrets scrubbed)", ""]
    out += [f"### {n}\n```\n{t}\n```\n" for n, t in logs.items()]
    return "\n".join(out) + "\n"


def plan(only: list[str], with_keys: bool, repeats: int) -> str:
    lines = ["PLAN"]
    if "freellmapi" in only:
        lines.append(f"  freellmapi  image {FL_IMAGE} on 127.0.0.1:{FL_PORT}")
        lines.append("    phase keyless: no keys given to it (tests FL1-FL6, FL8, FL9)")
        lines.append("    phase keyed  : " + ("your free-tier keys -> the container: "
                     f"{', '.join(KEY_PLATFORMS)} (those present in .env)" if with_keys
                     else "SKIPPED (add --with-keys to also test failover with your keys)"))
    if "omniroute" in only:
        for v, imgs in OR_VARIANTS:
            lines.append(f"  omniroute   {v}: {' -> fallback '.join(imgs)} on 127.0.0.1:{OR_PORT}")
    if "strata" in only:
        lines.append("  strata      read-only: hardware check, HuggingFace/GitHub metadata, then the live API "
                     "ONLY if a Strata server answers (--strata-url); nothing installed or downloaded")
    if "modelsdev" in only:
        lines.append("  models.dev  catalog vs each provider's own /models list + live structured-output "
                     "calls, using YOUR keys against YOUR providers (never handed to a third party)")
    lines += [f"  pipeline    {repeats} run(s) per model through each gateway (>=2 LLM calls per run)",
              "  cleanup     containers removed at the end; temp env-files deleted; keys scrubbed from output",
              "  output      reports/gateway_claims_<UTC>.md + .jsonl"]
    return "\n".join(lines)


async def amain(args: argparse.Namespace) -> int:
    only = [s.strip() for s in args.only.split(",") if s.strip()]
    print(plan(only, args.with_keys, args.repeats))
    if args.dry_run:
        print("\n--dry-run: nothing started, no network calls.")
        return 0

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    L, W = Ledger(stamp), S.Writer(f"gateway_{stamp}")
    meta: dict = {"started_utc": stamp, "git_head": subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
        "with_keys": args.with_keys, "outcome": "in progress"}
    logs: dict[str, str] = {}

    def finish(outcome: str, code: int) -> int:
        meta["outcome"] = outcome
        L.md.write_text(render(L, W, meta, logs), encoding="utf-8")
        print(f"\n{outcome}\nresults:\n  {L.md}\n  {L.jsonl}")
        return code

    gateways = [g for g in only if g in ("freellmapi", "omniroute")]
    pipeline_needed = bool(gateways) or "strata" in only
    if gateways:
        ok, why = docker_ready()
        if not ok:
            return finish(f"ABORTED: Docker daemon not reachable ({why}). Start Rancher/Docker Desktop and re-run.", 2)
    for host, needed in (("query1.finance.yahoo.com", pipeline_needed), ("ghcr.io", gateways),
                         ("registry-1.docker.io", gateways), ("api.groq.com", "modelsdev" in only)):
        if not needed:
            continue
        reach, how = S.host_reachable(host)
        meta[f"preflight {host}"] = "OK" if reach else how
        if host in ("query1.finance.yahoo.com", "api.groq.com") and not reach:
            return finish(f"ABORTED: {host} not reachable -- the VPN looks ON. Disconnect and re-run.", 2)

    prepared = None
    if pipeline_needed:
        try:
            prepared = await S.prepare()
        except Exception as e:
            meta["pipeline"] = f"unavailable: {type(e).__name__}: {e}"[:300]

    all_keys = read_keys()  # for models.dev checks: direct to your own providers
    keys = all_keys if args.with_keys else {}  # for the gateways: only on explicit consent
    try:
        with httpx.Client(timeout=60) as c:
            if "freellmapi" in only:
                for phase, k in (("keyless", {}), *((("keyed", keys),) if args.with_keys else ())):
                    ctx = freellmapi_phase(L, c, phase, k)
                    logs[f"freellmapi [{phase}]"] = container_logs(FL_NAME)
                    group = f"freellmapi-{phase}"  # one group per phase: runs must not mix
                    if prepared and ctx:
                        await pipeline(W, prepared, group, ctx, args.repeats)
                    pipeline_finding(L, W, "FL9", group, phase)
                    docker("rm", "-f", FL_NAME)
            if "strata" in only:
                ctx = strata_checks(L, c, call, chat, args.strata_url, args.strata_key)
                if ctx:
                    structured_check(L, c, ctx["base"], ctx["key"], "server", "ST4", ctx["models"][0])
                    if prepared:
                        await pipeline(W, prepared, "strata", ctx, args.repeats)
                    pipeline_finding(L, W, "ST5", "strata", "server")
            if "modelsdev" in only:
                catalog_phase(L, c, all_keys, call)
            if "omniroute" in only:
                for variant, images in OR_VARIANTS:
                    ctx = omniroute_phase(L, c, variant, images)
                    logs[f"omniroute [{variant}]"] = container_logs(OR_NAME)
                    if prepared and ctx:
                        await pipeline(W, prepared, f"omniroute-{variant}", ctx, args.repeats)
                    if variant == "web" or ctx:
                        pipeline_finding(L, W, "OR6", f"omniroute-{variant}", variant)
                    docker("rm", "-f", OR_NAME)
    except KeyboardInterrupt:
        return finish("INTERRUPTED (partial results)", 130)
    finally:
        if not args.keep:
            docker("rm", "-f", FL_NAME)
            docker("rm", "-f", OR_NAME)
    return finish("completed", 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="freellmapi,omniroute,modelsdev,strata",
                    help="comma list: freellmapi,omniroute,modelsdev,strata")
    ap.add_argument("--strata-url", default="http://127.0.0.1:8080", help="a running Strata server (default: this PC)")
    ap.add_argument("--strata-key", default=None, help="that server's --api-key, if it has one")
    ap.add_argument("--with-keys", action="store_true", help="also give your free-tier keys to the freellmapi container")
    ap.add_argument("--repeats", type=int, default=2, help="pipeline runs per model (default 2)")
    ap.add_argument("--keep", action="store_true", help="leave containers running afterwards")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit; no network")
    args = ap.parse_args()
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
