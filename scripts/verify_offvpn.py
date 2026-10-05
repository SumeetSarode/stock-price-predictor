"""Off-VPN verification: the checks that CANNOT be run from a blocked network.

WHY THIS SCRIPT EXISTS
======================
Three things were changed/claimed recently that a corporate/VPN network
physically cannot verify, because the endpoints are allowlist-blocked:

  1. A newly-added hosted chain candidate (e.g. Mistral, OmniRoute) actually
     WORKS as a chain member -- i.e. it returns JSON our Pydantic models
     accept, not prose. litellm metadata flags like supports_function_calling
     are NOT proof of this (Groq 70B reports the same flag and still breaks
     in production) -- only a real call proves it. Run with --test-model.
  2. json_extract.py's reasoning-model fixes hold against REAL model
     output, not just the synthetic strings used in unit tests.
  3. The price chain survives Yahoo's "Invalid Crumb" error -- which only
     reproduces against live Yahoo.

Everything here hits the network on purpose. Nothing here is a substitute
for the unit suite (`uv run pytest`); this is the layer ON TOP that the
unit suite deliberately mocks away.

USAGE
=====
    uv run python scripts/verify_offvpn.py

    # skip the LLM sections entirely:
    uv run python scripts/verify_offvpn.py --skip-llm

    # isolate-test ONE candidate model before adding it to CHAIN_AGENTIC
    # (the exact protocol previously used to catch Nemotron's prose problem):
    uv run python scripts/verify_offvpn.py --test-model mistral/mistral-small-latest --test-key-env MISTRAL_API_KEY

Writes a full transcript to reports/offvpn_verification_<timestamp>.md
(markdown, so it pastes cleanly into a PR or chat) plus a machine-readable
reports/offvpn_verification_<timestamp>.json.

EXIT CODES
==========
    0 = every check passed (or was intentionally skipped)
    1 = at least one check FAILED -- read the report
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

# ── Repo root on sys.path so `uv run python scripts/...` works from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

REPORT_DIR = _REPO_ROOT / "reports"


# ─────────────────────────────────────────────────────────────
# Result plumbing
# ─────────────────────────────────────────────────────────────
@dataclass
class Check:
    """One verification step and its outcome."""

    name: str
    section: str
    status: str = "PENDING"  # PASS | FAIL | SKIP
    detail: str = ""
    elapsed_s: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)


class Report:
    """Collects checks, prints live progress, writes the transcript."""

    def __init__(self) -> None:
        self.checks: list[Check] = []
        self.started = datetime.now(UTC)

    def add(self, c: Check) -> Check:
        self.checks.append(c)
        icon = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}.get(c.status, "[????]")
        print(f"{icon} {c.name} ({c.elapsed_s:.1f}s)")
        if c.detail:
            for line in c.detail.splitlines():
                print(f"         {line}")
        return c

    # -- context-manager-ish helper so every check gets timed + crash-proofed
    def run(self, name: str, section: str, fn) -> Check:
        """Run `fn()`; it returns (status, detail, evidence)."""
        t0 = time.monotonic()
        try:
            status, detail, evidence = fn()
        except Exception as e:
            status = "FAIL"
            detail = f"{type(e).__name__}: {e}"
            evidence = {"traceback": traceback.format_exc()[-2000:]}
        return self.add(
            Check(
                name=name,
                section=section,
                status=status,
                detail=detail,
                elapsed_s=time.monotonic() - t0,
                evidence=evidence,
            )
        )

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == "FAIL"]

    def write(self, prefix: str = "offvpn_verification") -> tuple[Path, Path]:
        REPORT_DIR.mkdir(exist_ok=True)
        stamp = self.started.strftime("%Y%m%dT%H%M%SZ")
        md_path = REPORT_DIR / f"{prefix}_{stamp}.md"
        json_path = REPORT_DIR / f"{prefix}_{stamp}.json"

        passed = sum(c.status == "PASS" for c in self.checks)
        failed = sum(c.status == "FAIL" for c in self.checks)
        skipped = sum(c.status == "SKIP" for c in self.checks)

        lines: list[str] = [
            "# Off-VPN verification report" if prefix == "offvpn_verification" else f"# {prefix.replace('_', ' ').title()} report",
            "",
            f"- **When (UTC):** {self.started.isoformat()}",
            f"- **Host:** {platform.node()} ({platform.system()} {platform.release()})",
            f"- **Python:** {platform.python_version()}",
            f"- **Git HEAD:** {_git_head()}",
            f"- **Result:** {passed} passed, {failed} failed, {skipped} skipped",
            "",
            "## Summary",
            "",
            "| Section | Check | Status | Time |",
            "|---|---|---|---|",
        ]
        for c in self.checks:
            lines.append(
                f"| {c.section} | {c.name} | **{c.status}** | {c.elapsed_s:.1f}s |"
            )
        lines += ["", "## Detail", ""]
        for c in self.checks:
            lines += [f"### [{c.status}] {c.name}", "", f"*{c.section}*", ""]
            if c.detail:
                lines += ["```", c.detail.rstrip(), "```", ""]
            for k, v in c.evidence.items():
                text = v if isinstance(v, str) else json.dumps(v, indent=2, default=str)
                lines += [f"<details><summary>{k}</summary>", "", "```", str(text)[:6000], "```", "", "</details>", ""]

        md_path.write_text("\n".join(lines), encoding="utf-8")
        json_path.write_text(
            json.dumps(
                {
                    "started_utc": self.started.isoformat(),
                    "git_head": _git_head(),
                    "python": platform.python_version(),
                    "passed": passed, "failed": failed, "skipped": skipped,
                    "checks": [asdict(c) for c in self.checks],
                },
                indent=2, default=str,
            ),
            encoding="utf-8",
        )
        return md_path, json_path


def _git_head() -> str:
    import subprocess
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT, capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _section(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


@contextlib.contextmanager
def _capture_llm_log():
    """Collect the chain's own log lines (which hop was tried, WHY it failed).

    Without this the report only keeps the LAST error -- on an exhausted
    chain that is just 'Ollama isn't running', hiding why every earlier hop
    failed. Yields a list that fills as the run goes.
    """
    lines: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(f"{record.levelname:7s} {record.getMessage()}"[:500])

    handler = _Collect(level=logging.INFO)
    target = logging.getLogger("price_predictor.llm")
    old_level = target.level
    target.setLevel(logging.INFO)
    target.addHandler(handler)
    try:
        yield lines
    finally:
        target.removeHandler(handler)
        target.setLevel(old_level)


# ─────────────────────────────────────────────────────────────
# Section 1 — connectivity (cheap, tells you WHY later checks fail)
# ─────────────────────────────────────────────────────────────
def check_connectivity(rep: Report) -> dict[str, bool]:
    _section("1. Connectivity — which endpoints does this network allow?")
    import httpx

    targets = {
        "yahoo": "https://query1.finance.yahoo.com/v8/finance/chart/RELIANCE.NS?range=5d&interval=1d",
        "nse": "https://www.nseindia.com",
        "google_ai": "https://generativelanguage.googleapis.com",
        # The other LLM hosts in the chain. Without these a report can't say
        # whether a dead hop was the network (VPN DNS-blocks them) or the
        # provider -- which is the first thing you need to know.
        "groq": "https://api.groq.com",
        "cohere": "https://api.cohere.com",
        "openrouter": "https://openrouter.ai",
    }
    reachable: dict[str, bool] = {}

    for name, url in targets.items():
        def probe(name=name, url=url):
            try:
                r = httpx.get(
                    url, timeout=15.0,
                    follow_redirects=True,
                    headers={"User-Agent": "Mozilla/5.0 (verify-offvpn)"},
                )
                reachable[name] = r.status_code < 500
                # Blocked-by-network is a FACT about where you're running,
                # not a code defect -- SKIP keeps the report honest.
                return (
                    "PASS" if reachable[name] else "SKIP",
                    f"HTTP {r.status_code} from {url}",
                    {"status_code": r.status_code},
                )
            except Exception as e:
                reachable[name] = False
                return (
                    "SKIP",
                    f"BLOCKED/unreachable: {type(e).__name__}: {str(e)[:200]}\n"
                    "(expected on VPN; running off-VPN is what this script is for)",
                    {},
                )

        rep.run(f"reach {name}", "connectivity", probe)

    return reachable


# ─────────────────────────────────────────────────────────────
# Section 2 — the price chain + the "Invalid Crumb" question
# ─────────────────────────────────────────────────────────────
def check_prices(rep: Report, reachable: dict[str, bool]) -> None:
    _section("2. Price providers — incl. the Yahoo 'Invalid Crumb' path")
    from price_predictor.config.settings import settings
    from price_predictor.data.providers import PriceFetchError, build_provider

    end = date.today()
    start = end - timedelta(days=10)
    ticker = "RELIANCE.NS"

    configured = [p.strip() for p in settings.price_chain.split(",") if p.strip()]
    rep.run(
        "PRICE_CHAIN has a fallback",
        "prices",
        lambda: (
            ("PASS", f"chain = {configured} ({len(configured)} providers)", {"chain": configured})
            if len(configured) > 1
            else (
                "SKIP",
                f"ADVISORY: chain = {configured} -- SINGLE PROVIDER.\n"
                "A Yahoo 'Invalid Crumb' / rate-limit has nothing to fall back to,\n"
                "so any Yahoo hiccup becomes a hard failure rather than a silent\n"
                "fallback. Not a code defect -- a config choice worth revisiting.\n"
                "Consider: PRICE_CHAIN=jugaad,nse_bhavcopy,yfinance",
                {"chain": configured},
            )
        ),
    )

    # Probe each provider INDIVIDUALLY -- that's the only way to see which
    # ones actually work on this network vs which are silently carried by
    # a healthy neighbour in the chain.
    for name in ("yfinance", "jugaad", "nse_bhavcopy"):
        def probe(name=name):
            try:
                df = build_provider(name).fetch_ohlcv(ticker, start, end, "1d")
            except PriceFetchError as e:
                msg = str(e)
                # The specific thing we're hunting.
                if "crumb" in msg.lower():
                    return (
                        "FAIL",
                        f"INVALID CRUMB reproduced:\n{msg[:400]}",
                        {"error": msg[:2000], "crumb_error": True},
                    )
                return "FAIL", f"PriceFetchError: {msg[:300]}", {"error": msg[:2000]}
            if df is None or df.empty:
                return "FAIL", "returned an empty frame", {}
            return (
                "PASS",
                f"{len(df)} rows, last close={df['close'].iloc[-1]:.2f}",
                {"rows": len(df), "last_close": float(df["close"].iloc[-1])},
            )

        # A provider whose upstream host is firewalled tells us nothing about
        # the CODE -- mark it SKIP so a blocked run doesn't read as red.
        host = {"yfinance": "yahoo", "jugaad": "nse", "nse_bhavcopy": "nse"}[name]
        if not reachable.get(host, True):
            rep.add(Check(
                f"provider: {name}", "prices", "SKIP",
                f"{host} unreachable from this network -- cannot evaluate.",
            ))
            continue

        rep.run(f"provider: {name}", "prices", probe)

    # The real question: does the CHAIN survive whatever just failed?
    def chain_probe():
        from price_predictor.data.prices import fetch_ohlcv, reset_default_fetcher

        reset_default_fetcher()
        df = fetch_ohlcv(ticker, start, end)
        if df is None or df.empty:
            return "FAIL", "chain produced no data", {}
        return (
            "PASS",
            f"chain delivered {len(df)} rows despite any individual failures above",
            {"rows": len(df)},
        )

    if not any(reachable.get(h, True) for h in ("yahoo", "nse")):
        rep.add(Check(
            "full chain end-to-end", "prices", "SKIP",
            "Every price host is blocked on this network -- run this off-VPN.",
        ))
    else:
        rep.run("full chain end-to-end", "prices", chain_probe)


# ─────────────────────────────────────────────────────────────
# Section 3 -- isolate ONE candidate model, prove JSON not prose
# ─────────────────────────────────────────────────────────────
def check_model_isolated(
    rep: Report, model: str, api_key: str, api_base: str | None = None,
    label: str | None = None,
) -> None:
    """Prove a SINGLE candidate model returns schema-valid JSON, not prose.

    Generic replacement for the old Nemotron-only check. Point `model` at
    any 'provider/model-name' litellm string and its API key to run the
    exact protocol that caught Nemotron's "emits prose, not JSON" failure
    before it goes anywhere near CHAIN_AGENTIC. No pre-flight reachability
    probe -- the raw call itself surfaces connection failures with full
    detail, which is more honest than a synthetic ping.

    `label` groups the emitted checks in the report. It defaults to the
    provider prefix, but callers testing SEVERAL models from the SAME
    provider must pass distinct labels -- otherwise the rows collide and
    later results silently overwrite earlier ones in the summary.
    """
    label = label or model.split("/", 1)[0]
    _section(f"3. {model} -- does it return SCHEMA-VALID JSON, or prose?")

    # A LOCAL OpenAI-compatible gateway (OmniRoute, Foundry Local, LM Studio)
    # needs no real key -- litellm still wants a non-empty string, so a dummy
    # stands in. Only HOSTED providers genuinely require a key, so the
    # can't-test-without-a-key bail-out applies to them alone.
    if api_base and not api_key:
        api_key = "dummy-local-gateway-key"

    if not api_key:
        rep.add(Check(
            f"{label} API key present", label, "SKIP",
            f"No API key provided for {model} -- cannot test it.\n"
            "Pass --test-key-env <ENV_VAR_NAME> pointing at the key in .env.",
        ))
        return

    # ── 3a. Raw call. Isolate the model so it MUST answer -- if we went
    #        through the normal chain, an earlier model would serve every
    #        request and this one would never actually be exercised.
    # Bounds MUST mirror the REAL contract in agents/news_impact/agent.py
    # (confidence: float = Field(..., ge=0.0, le=1.0)). An unbounded "number"
    # here silently passes output the production Pydantic model would REJECT
    # -- which is exactly what happened: Nemotron returned confidence=85 on
    # one trial and 0.78 on another, and the loose schema scored both USABLE.
    schema = {
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "string"},
        },
        "required": ["direction", "confidence", "rationale"],
        "additionalProperties": False,
    }
    prompt = (
        "Stock RELIANCE.NS rose 2.1% on strong refining margins and heavy "
        "institutional buying. Classify the near-term outlook. Respond with "
        "ONLY a JSON object, no prose."
    )
    # Prompt-only fallback (providers that reject response_format) must still
    # state the REQUIRED SHAPE -- otherwise the model invents its own keys and
    # "fails" a contract it was never shown. First run of this hit exactly that:
    # Cohere returned perfectly valid JSON, just keyed 'near_term_outlook'.
    # Judging JSON-ability on an unstated schema is not a fair test.
    prompt_with_shape = (
        prompt
        + " Use EXACTLY these keys: "
        '"direction" (one of: "bullish", "bearish", "neutral"), '
        '"confidence" (a number between 0 and 1), '
        '"rationale" (a short string). '
        'Example: {"direction": "bullish", "confidence": 0.8, '
        '"rationale": "margin expansion"}'
    )
    raw_holder: dict[str, str] = {}

    def raw_call():
        import litellm

        def _call(with_schema: bool):
            kwargs: dict[str, Any] = dict(
                model=model,
                messages=[{
                    "role": "user",
                    "content": prompt if with_schema else prompt_with_shape,
                }],
                num_retries=0,
                timeout=90,
                api_key=api_key,
                **({"api_base": api_base} if api_base else {}),
            )
            if with_schema:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "outlook", "schema": schema, "strict": True},
                }
            return litellm.completion(**kwargs)

        # Some provider/litellm-integration combos reject response_format
        # outright (litellm.UnsupportedParamsError) BEFORE the model ever
        # sees the request -- that is a param-plumbing gap, not proof the
        # model can't do JSON. Retry prompt-only so 3b/3c still get a real
        # answer to judge instead of a false unusable verdict from a rejected kwarg.
        used_schema = True
        try:
            resp = _call(with_schema=True)
        except litellm.UnsupportedParamsError as e:
            if "response_format" not in str(e):
                raise
            used_schema = False
            resp = _call(with_schema=False)

        content = resp.choices[0].message.content or ""
        raw_holder["raw"] = content
        note = (
            "" if used_schema else
            " (provider/integration rejects response_format -- retried prompt-only)"
        )
        return (
            "PASS",
            f"responded with {len(content)} chars{note}",
            {"raw_response": content[:4000], "used_schema_mode": used_schema},
        )

    got_raw = rep.run("3a. raw completion (response_format=json_schema)", label, raw_call)
    if got_raw.status != "PASS":
        return

    raw = raw_holder.get("raw", "")

    # ── 3b. Is it USABLE as-is? (bare json.loads, no cleanup)
    def bare_parse():
        try:
            json.loads(raw)
            return "PASS", "raw output is already valid JSON -- no cleanup needed", {}
        except json.JSONDecodeError as e:
            return (
                "FAIL",
                f"raw output is NOT bare JSON ({e}).\n"
                "This is the 'unparsable LLM output' failure mode.\n"
                "Check 3c below decides whether json_extract.py rescues it.",
                {"raw_head": raw[:500]},
            )

    rep.run("3b. raw output parses as bare JSON", label, bare_parse)

    # ── 3c. THE decisive one: does OUR extractor rescue it?
    def extract_parse():
        from price_predictor.llm.json_extract import extract_json

        cleaned = extract_json(raw)
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError as e:
            return (
                "FAIL",
                f"json_extract.py could NOT rescue this output: {e}\n"
                f"=> {model} is genuinely unusable in the chain as-is.",
                {"raw": raw[:3000], "cleaned": cleaned[:1500]},
            )
        missing = [k for k in ("direction", "confidence", "rationale") if k not in obj]
        if missing:
            return (
                "FAIL",
                f"parsed, but missing required keys: {missing}",
                {"parsed": obj},
            )
        # Parsing is necessary but NOT sufficient -- the production model
        # (agents/news_impact/agent.py) enforces confidence ge=0.0 le=1.0 and
        # a direction enum. Output that parses but violates those still blows
        # up as a Pydantic ValidationError in the real pipeline, so grade it
        # here rather than reporting a false USABLE.
        conf = obj["confidence"]
        if not isinstance(conf, int | float) or isinstance(conf, bool):
            return (
                "FAIL",
                f"confidence is {type(conf).__name__}, not a number: {conf!r}\n"
                "=> production Pydantic (confidence: float) would reject this.",
                {"parsed": obj},
            )
        if not 0.0 <= conf <= 1.0:
            return (
                "FAIL",
                f"confidence={conf} is OUTSIDE the required 0.0-1.0 range.\n"
                "Parses fine, but agents/news_impact/agent.py declares\n"
                "confidence: float = Field(..., ge=0.0, le=1.0) -- so in the\n"
                "real pipeline this raises a Pydantic ValidationError. Looks\n"
                "like a 0-100 percentage where a 0-1 probability was asked for.",
                {"parsed": obj},
            )
        if obj["direction"] not in ("bullish", "bearish", "neutral"):
            return (
                "FAIL",
                f"direction={obj['direction']!r} is outside the allowed enum.",
                {"parsed": obj},
            )
        return (
            "PASS",
            f"json_extract rescued it -> direction={obj['direction']!r}, "
            f"confidence={obj['confidence']}",
            {"parsed": obj, "was_already_clean": cleaned.strip() == raw.strip()},
        )

    rep.run("3c. json_extract.py + parse (THE decisive check)", label, extract_parse)

    # ── 3d. Reasoning-model tell: does it emit <think> blocks? This is what
    #        the json_extract fix was built for; good to know for real.
    def think_probe():
        has_think = "<think>" in raw.lower()
        unclosed = has_think and "</think>" not in raw.lower()
        return (
            "PASS",
            f"<think> present: {has_think}; unclosed: {unclosed}\n"
            + ("(unclosed <think> is exactly the case the parser fix added)"
               if unclosed else "(no unclosed-reasoning problem in this sample)"),
            {"has_think": has_think, "unclosed": unclosed},
        )

    rep.run("3d. reasoning-tag inspection", label, think_probe)


# ─────────────────────────────────────────────────────────────
# Section 4 — a REAL prediction through the real agent
# ─────────────────────────────────────────────────────────────
def check_real_prediction(rep: Report, only_model: str | None = None) -> None:
    _section("4. End-to-end — a real ImpactAssessment through the real agent")

    # --only-model forces the end-to-end prediction through ONE hop.
    #
    # Getting this right took three attempts, so the reasoning is recorded:
    #   1. Setting os.environ["CHAIN_AGENTIC"] does NOT work -- Settings is
    #      declared with env_file=".env", and the dotenv value wins over the
    #      process environment. Verified: the agent still built all 6 hops.
    #   2. Mutating settings.chain_agentic afterwards does NOT work either --
    #      predictor.py builds its agent as a MODULE-LEVEL SINGLETON at import
    #      time, so anything applied post-import is ignored. Verified: chain
    #      reported 1 entry while the live agent still held all 6.
    # What actually works is rebuilding the singleton's model from the
    # narrowed chain, which is what make_resilient_model already does.
    if only_model:
        from price_predictor.config.settings import settings as _s
        from price_predictor.llm.factory import make_model
        from price_predictor.prediction import predictor as _predictor

        object.__setattr__(_s, "chain_agentic", only_model)
        _predictor._news_impact_agent.model = make_model(only_model)
        print(f"  [--only-model] agent forced to a single hop: {only_model}")

    def run_agent():

        from price_predictor.agents.news_impact.agent import make_news_impact_agent
        from price_predictor.config.settings import settings

        if only_model:
            return (
                "SKIP",
                f"--only-model={only_model} deliberately narrows the chain to "
                "one hop, so a full-chain comparison is meaningless here.",
                {"only_model": only_model},
            )

        chain = settings.effective_chain("agentic")
        agent = make_news_impact_agent()
        inner = getattr(agent.model, "inner_models", [])
        names = [getattr(m, "model", "?") for m in inner]

        if names == chain:
            return (
                "PASS",
                f"live agent chain matches CHAIN_AGENTIC exactly: {names}",
                {"chain": names},
            )
        return (
            "FAIL",
            f"live agent chain {names}\ndoes NOT match settings.effective_chain() {chain}.\n"
            "Check CHAIN_AGENTIC in .env and any optional-key skip logic in settings.py.",
            {"chain": names, "settings_chain": chain},
        )

    rep.run("4a. wired agent chain matches configured CHAIN_AGENTIC", "end-to-end", run_agent)

    def real_pred():
        # run_news_impact_agent() does its OWN gathering internally
        # ("gather in code, reason once") -- it takes a bare ticker, so
        # there is no separate gather step to call here.
        from price_predictor.llm.resilient import AllModelsExhaustedError
        from price_predictor.prediction import predictor as _predictor
        from price_predictor.prediction.predictor import run_news_impact_agent

        try:
            with _capture_llm_log() as hop_log:
                result = asyncio.run(run_news_impact_agent("RELIANCE.NS"))
        except AllModelsExhaustedError as e:
            # Every model refused. Usually quota, not a code defect -- but it
            # DOES mean no prediction happened, so don't dress it up as a pass.
            # hop_log says WHY each earlier hop failed; `str(e)` alone only
            # carries the last one.
            return (
                "FAIL",
                f"whole chain exhausted (usually rate-limit/quota, or the local\n"
                f"Ollama tail isn't running):\n{str(e)[:600]}\n\n"
                "WHY EACH HOP FAILED (from the chain's own log):\n"
                + "\n".join(ln for ln in hop_log if "trying model" not in ln)[:3000],
                {"error": str(e)[:3000], "hop_log": hop_log},
            )

        # WHICH HOP SERVED IT matters as much as the result: the chain is
        # ordered, so a bare PASS only proves the first model that answered
        # works -- it says nothing about the hop we actually care about.
        # resilient.py records this on the agent instance, and predictor.py
        # builds that agent ONCE at import time as a module-level singleton,
        # so read it straight off the module (a freshly-built agent would
        # report last_used_model=None and quietly tell us nothing).
        # Under --only-model the model is a plain LiteLlm with no fallback
        # wrapper, so there is nothing to record -- but we already know
        # exactly which hop ran, because we pinned it.
        served = only_model or getattr(
            getattr(_predictor._news_impact_agent, "model", None),
            "last_used_model",
            None,
        )

        # Validate against the REAL contract, not just "no exception":
        # ImpactAssessment enforces confidence 0.0-1.0, and the whole reason
        # Cohere was promoted ahead of Nemotron is that Nemotron returned 85
        # on one trial -- which parses fine but is a ValidationError here.
        conf_ok = isinstance(result.confidence, int | float) and (
            0.0 <= result.confidence <= 1.0
        )
        if not conf_ok:
            return (
                "FAIL",
                f"confidence={result.confidence} is outside the required "
                f"0.0-1.0 range (served by {served or 'unknown'}).",
                {"assessment": result.model_dump(), "served_by": served},
            )
        # Field names verified against ImpactAssessment.model_fields:
        # ticker, sentiment, confidence, estimated_pct_move, reasoning, catalysts
        return (
            "PASS",
            f"got a valid ImpactAssessment: sentiment={result.sentiment!r}, "
            f"confidence={result.confidence}, "
            f"est_move={result.estimated_pct_move}%\n"
            f"SERVED BY: {served or '(unknown)'}",
            {"assessment": result.model_dump(), "served_by": served, "hop_log": hop_log},
        )

    rep.run("4b. real news-impact prediction (hits live news + LLM)", "end-to-end", real_pred)


# ─────────────────────────────────────────────────────────────
# Section 5 — the integration suite (mirrors run_integration.sh, wider)
# ─────────────────────────────────────────────────────────────
def check_integration_suite(rep: Report) -> None:
    _section("5. Integration test suite (-m integration)")
    import subprocess

    def run_suite():
        proc = subprocess.run(
            ["uv", "run", "pytest", "-m", "integration", "-v", "--no-cov", "--tb=short"],
            cwd=_REPO_ROOT, capture_output=True, text=True, timeout=2400,
        )
        tail = proc.stdout[-4000:] if proc.stdout else proc.stderr[-4000:]
        last = [ln for ln in (proc.stdout or "").splitlines() if "passed" in ln or "failed" in ln]
        summary = last[-1] if last else "(no pytest summary line found)"
        return (
            "PASS" if proc.returncode == 0 else "FAIL",
            f"exit={proc.returncode}\n{summary}",
            {"pytest_tail": tail},
        )

    rep.run("pytest -m integration", "integration-suite", run_suite)


# ─────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--only-model",
        default=None,
        help=(
            "Force the end-to-end prediction to use ONLY this chain entry, "
            "e.g. 'cohere_chat/command-r-08-2024'. Without this, an ordered "
            "chain means hop 1 answers and later hops are never exercised."
        ),
    )
    ap.add_argument("--skip-llm", action="store_true", help="skip the LLM candidate + prediction sections")
    ap.add_argument("--skip-suite", action="store_true", help="skip the pytest integration suite (slowest)")
    ap.add_argument("--test-model", default=None, help="isolate-test ONE candidate model, e.g. mistral/mistral-small-latest")
    ap.add_argument("--test-key-env", default=None, help="env var (in .env) holding --test-model's API key, e.g. MISTRAL_API_KEY")
    ap.add_argument(
        "--test-api-base", default=None,
        help="OpenAI-compatible base URL for a LOCAL gateway, e.g. "
             "http://localhost:20128/v1 (OmniRoute). Makes --test-key-env optional.",
    )
    args = ap.parse_args()
    if args.test_model and not (args.test_key_env or args.test_api_base):
        ap.error("--test-model requires --test-key-env (hosted) or --test-api-base (local gateway)")

    print("Off-VPN verification -- this WILL hit the real network.\n")
    rep = Report()

    reachable = check_connectivity(rep)
    check_prices(rep, reachable)

    if args.skip_llm:
        rep.add(Check("LLM sections", "candidate", "SKIP", "--skip-llm passed"))
    else:
        if args.test_model:
            import os

            from dotenv import load_dotenv

            load_dotenv(_REPO_ROOT / ".env", override=False)
            key = (
                (os.environ.get(args.test_key_env) or "").strip()
                if args.test_key_env else ""
            )
            check_model_isolated(
                rep, model=args.test_model, api_key=key,
                api_base=args.test_api_base,
            )
        check_real_prediction(rep, only_model=args.only_model)

    if args.skip_suite:
        rep.add(Check("integration suite", "integration-suite", "SKIP", "--skip-suite passed"))
    else:
        check_integration_suite(rep)

    md, js = rep.write()

    passed = sum(c.status == "PASS" for c in rep.checks)
    failed = sum(c.status == "FAIL" for c in rep.checks)
    skipped = sum(c.status == "SKIP" for c in rep.checks)
    print(f"\n{'=' * 68}")
    print(f"RESULT: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"Report (markdown): {md}")
    print(f"Report (json):     {js}")
    print("=" * 68)
    if failed:
        print("\nFailed checks:")
        for c in rep.failed:
            print(f"  - [{c.section}] {c.name}")
        print("\nSend me the .md file and I'll work from it.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
