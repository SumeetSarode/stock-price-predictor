"""Off-VPN verification: the checks that CANNOT be run from a blocked network.

WHY THIS SCRIPT EXISTS
======================
Three things were changed/claimed recently that a corporate/VPN network
physically cannot verify, because the endpoints are allowlist-blocked:

  1. Nemotron (openrouter/...) actually WORKS as a chain member -- i.e. it
     returns JSON our Pydantic models accept, not prose. This was asserted
     from a litellm metadata flag that turned out to mean "unknown model",
     NOT "unsupported". It was never proven with a real call.
  2. json_extract.py's reasoning-model fixes hold against REAL Nemotron
     output, not just the synthetic strings used in unit tests.
  3. The price chain survives Yahoo's "Invalid Crumb" error -- which only
     reproduces against live Yahoo.

Everything here hits the network on purpose. Nothing here is a substitute
for the unit suite (`uv run pytest`); this is the layer ON TOP that the
unit suite deliberately mocks away.

USAGE
=====
    uv run python scripts/verify_offvpn.py

    # skip the LLM section (no OpenRouter key / don't want to burn quota):
    uv run python scripts/verify_offvpn.py --skip-llm

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
import json
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

# The model under investigation. Kept as a module constant (not inlined) so
# that if the slug is ever rotated, there is exactly ONE place to change it.
NEMOTRON = "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"


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

    def write(self) -> tuple[Path, Path]:
        REPORT_DIR.mkdir(exist_ok=True)
        stamp = self.started.strftime("%Y%m%dT%H%M%SZ")
        md_path = REPORT_DIR / f"offvpn_verification_{stamp}.md"
        json_path = REPORT_DIR / f"offvpn_verification_{stamp}.json"

        passed = sum(c.status == "PASS" for c in self.checks)
        failed = sum(c.status == "FAIL" for c in self.checks)
        skipped = sum(c.status == "SKIP" for c in self.checks)

        lines: list[str] = [
            "# Off-VPN verification report",
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


def _openrouter_key() -> str:
    """Return the configured OpenRouter key ('' if unset).

    WHY NOT os.environ.get(...): nothing in this repo calls load_dotenv().
    `.env` reaches os.environ only as a SIDE EFFECT of importing app modules
    (litellm loads it), so a bare os.environ read can return None purely
    because of import ORDER -- which would make this script wrongly SKIP the
    Nemotron section even though the key is sitting right there in .env.
    Reading through Settings uses the same resolution path the app itself
    uses (pydantic-settings reads .env directly), so what we test is what
    actually runs. Also tolerant of the `KEY = value` spaced style already
    present in this project's .env.
    """
    try:
        from price_predictor.config.settings import settings

        return settings.openrouter_api_key.get_secret_value().strip()
    except Exception:
        import os

        return (os.environ.get("OPENROUTER_API_KEY") or "").strip()


# ─────────────────────────────────────────────────────────────
# Section 1 — connectivity (cheap, tells you WHY later checks fail)
# ─────────────────────────────────────────────────────────────
def check_connectivity(rep: Report) -> dict[str, bool]:
    _section("1. Connectivity — which endpoints does this network allow?")
    import httpx

    targets = {
        "openrouter": "https://openrouter.ai/api/v1/models",
        "yahoo": "https://query1.finance.yahoo.com/v8/finance/chart/RELIANCE.NS?range=5d&interval=1d",
        "nse": "https://www.nseindia.com",
        "google_ai": "https://generativelanguage.googleapis.com",
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
# Section 3 — Nemotron: the check that was never actually done
# ─────────────────────────────────────────────────────────────
def check_nemotron(rep: Report, reachable: dict[str, bool]) -> None:
    _section("3. Nemotron — does it return SCHEMA-VALID JSON, or prose?")

    if not _openrouter_key():
        rep.add(Check(
            "OPENROUTER_API_KEY present", "nemotron", "SKIP",
            "No OPENROUTER_API_KEY configured -- cannot test Nemotron.\n"
            "Set it in .env to run this section.",
        ))
        return
    if not reachable.get("openrouter"):
        rep.add(Check(
            "openrouter reachable", "nemotron", "SKIP",
            "openrouter.ai unreachable from this network -- skipping.\n"
            "This is the exact thing that must be run OFF-VPN.",
        ))
        return

    # ── 3a. Raw call. Isolate Nemotron so it MUST answer -- if we went
    #        through the normal chain, Gemini would serve every request and
    #        Nemotron would never actually be exercised.
    schema = {
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
            "confidence": {"type": "number"},
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
    raw_holder: dict[str, str] = {}

    def raw_call():
        import litellm

        resp = litellm.completion(
            model=NEMOTRON,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "outlook", "schema": schema, "strict": True},
            },
            num_retries=0,
            timeout=90,
            # Pass the key EXPLICITLY rather than relying on os.environ:
            # nothing here calls load_dotenv(), so the env var may be unset
            # even when .env has the key. See _openrouter_key().
            api_key=_openrouter_key(),
        )
        content = resp.choices[0].message.content or ""
        raw_holder["raw"] = content
        return (
            "PASS",
            f"responded with {len(content)} chars",
            {"raw_response": content[:4000]},
        )

    got_raw = rep.run("3a. raw completion (response_format=json_schema)", "nemotron", raw_call)
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

    rep.run("3b. raw output parses as bare JSON", "nemotron", bare_parse)

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
                "=> Nemotron is genuinely unusable in the chain as-is.",
                {"raw": raw[:3000], "cleaned": cleaned[:1500]},
            )
        missing = [k for k in ("direction", "confidence", "rationale") if k not in obj]
        if missing:
            return (
                "FAIL",
                f"parsed, but missing required keys: {missing}",
                {"parsed": obj},
            )
        return (
            "PASS",
            f"json_extract rescued it -> direction={obj['direction']!r}, "
            f"confidence={obj['confidence']}",
            {"parsed": obj, "was_already_clean": cleaned.strip() == raw.strip()},
        )

    rep.run("3c. json_extract.py + parse (THE decisive check)", "nemotron", extract_parse)

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

    rep.run("3d. reasoning-tag inspection", "nemotron", think_probe)


# ─────────────────────────────────────────────────────────────
# Section 4 — a REAL prediction through the real agent
# ─────────────────────────────────────────────────────────────
def check_real_prediction(rep: Report) -> None:
    _section("4. End-to-end — a real ImpactAssessment through the real agent")

    def run_agent():

        from price_predictor.agents.news_impact.agent import make_news_impact_agent
        from price_predictor.config.settings import settings

        chain = settings.effective_chain("agentic")
        agent = make_news_impact_agent()
        inner = getattr(agent.model, "inner_models", [])
        names = [getattr(m, "model", "?") for m in inner]
        in_chain = any(NEMOTRON in str(n) for n in names)

        if in_chain:
            return (
                "PASS",
                f"resolved chain: {names}\nNemotron IS wired into the live agent chain.",
                {"chain": names},
            )
        # No key configured is EXPECTED, not a failure: settings
        # ._optional_key_missing() deliberately strips openrouter entries
        # when OPENROUTER_API_KEY is blank (an AuthenticationError is a
        # STRUCTURAL error that would otherwise hard-crash every
        # prediction). Report that as SKIP so it doesn't masquerade as a
        # code defect.
        if not _openrouter_key():
            return (
                "SKIP",
                f"resolved chain: {names}\n"
                "Nemotron absent because OPENROUTER_API_KEY is not set -- this is\n"
                "the intended safety behaviour, not a bug. Set the key to test it.",
                {"chain": names},
            )
        return (
            "FAIL",
            f"resolved chain: {names}\n"
            "OPENROUTER_API_KEY IS set, yet Nemotron is missing from the chain.\n"
            "Check CHAIN_AGENTIC in .env (and that launch.bat/sync_env.py ran).",
            {"chain": names, "settings_chain": chain},
        )

    rep.run("4a. Nemotron present in the wired agent chain", "end-to-end", run_agent)

    def real_pred():
        # run_news_impact_agent() does its OWN gathering internally
        # ("gather in code, reason once") -- it takes a bare ticker, so
        # there is no separate gather step to call here.
        from price_predictor.llm.resilient import AllModelsExhaustedError
        from price_predictor.prediction.predictor import run_news_impact_agent

        try:
            result = asyncio.run(run_news_impact_agent("RELIANCE.NS"))
        except AllModelsExhaustedError as e:
            # Every model refused. Usually quota, not a code defect -- but it
            # DOES mean no prediction happened, so don't dress it up as a pass.
            return (
                "FAIL",
                f"whole chain exhausted (usually rate-limit/quota, or the local\n"
                f"Ollama tail isn't running):\n{str(e)[:600]}",
                {"error": str(e)[:3000]},
            )
        # Field names verified against ImpactAssessment.model_fields:
        # ticker, sentiment, confidence, estimated_pct_move, reasoning, catalysts
        return (
            "PASS",
            f"got a valid ImpactAssessment: sentiment={result.sentiment!r}, "
            f"confidence={result.confidence}, "
            f"est_move={result.estimated_pct_move}%",
            {"assessment": result.model_dump()},
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
    ap.add_argument("--skip-llm", action="store_true", help="skip Nemotron + prediction sections")
    ap.add_argument("--skip-suite", action="store_true", help="skip the pytest integration suite (slowest)")
    args = ap.parse_args()

    print("Off-VPN verification -- this WILL hit the real network.\n")
    rep = Report()

    reachable = check_connectivity(rep)
    check_prices(rep, reachable)

    if args.skip_llm:
        rep.add(Check("LLM sections", "nemotron", "SKIP", "--skip-llm passed"))
    else:
        check_nemotron(rep, reachable)
        check_real_prediction(rep)

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
