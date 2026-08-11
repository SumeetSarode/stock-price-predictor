"""OFF-VPN diagnostic: prove (or disprove) free-tier access to two candidate
LLM providers with REAL API calls, and save the result so it can be copied
back into chat.

CONTEXT: this repo's fallback chain (see src/price_predictor/llm/factory.py
and config/settings.py) currently goes
    gemini -> groq (x2) -> ollama_chat/qwen3:8b (local)
We're evaluating whether to add NVIDIA Nemotron-3-Ultra:free (via OpenRouter)
and/or GLM-5.2 (via Zenmux or Zhipu's own api.z.ai) as extra tiers between
Groq and Ollama. Docs/ToS research suggested Nemotron has a real working free
tier and GLM-5.2 does not (Zenmux's "-free" slug redirects to the paid
model) -- this script proves it with live calls instead of relying on that.

Runs automatically, non-fatally, every launch (see windows_setup/launch.bat)
-- but costs nothing and does nothing beyond an instant skip unless you've
actually added one of the three optional keys below to your .env.

    uv run python scripts/test_free_tier_access.py

WHAT YOU NEED TO PROVIDE (I cannot get these myself -- they're tied to your
identity, not the app's):

  OPENROUTER_API_KEY  -- for Nemotron-3-Ultra:free
      Sign up (free, ~1 min, no card): https://openrouter.ai/keys

  ZENMUX_API_KEY      -- for GLM-5.2 via Zenmux (both the "-free" slug the
                          blog claimed, and the standard paid slug, for
                          comparison)
      Sign up (free, ~1 min): https://zenmux.ai -> Create API Key

  ZAI_API_KEY         -- for GLM-5.2 via Zhipu's own direct API
      Sign up: https://z.ai or https://open.bigmodel.cn -> API Keys
      NOTE: may require phone verification (Chinese platform) -- not a
      script bug if that takes longer than the other two.

Add whichever you have to .env (see .env.example) -- any missing key just
skips that one test, the rest still run. None of this touches the app's
actual chain; it's a pure read-only diagnostic, same category as
scripts/diagnose.py or scripts/ensure_ollama.py.

OUTPUT: diagnostics/free_tier_test_<UTC-timestamp>.txt AND
diagnostics/free_tier_test_latest.txt (always overwritten) -- copy either
one back into chat.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

TIMEOUT = 30
_OUT_DIR = Path("diagnostics")
_ENV_KEYS = ("OPENROUTER_API_KEY", "ZENMUX_API_KEY", "ZAI_API_KEY")

TINY_PROMPT = {
    "messages": [{"role": "user", "content": "Reply with just the word OK."}],
    "max_tokens": 5,
}


def _load_keys() -> dict[str, str | None]:
    """env var s; falls back to .env (same pattern as
    scripts/list_gemini_models.py). Placeholder values (your_..._here) are
    treated as unset, same convention as the rest of this repo's tooling."""
    found: dict[str, str | None] = dict.fromkeys(_ENV_KEYS)
    for name in _ENV_KEYS:
        val = os.environ.get(name)
        if val and not val.startswith("your_"):
            found[name] = val
    env_path = Path(".env")
    if env_path.exists():
        text = env_path.read_text(encoding="utf-8")
        for name in _ENV_KEYS:
            if found[name]:
                continue
            m = re.search(rf"^{name}=(.+)$", text, re.MULTILINE)
            if m and not m.group(1).strip().startswith("your_"):
                found[name] = m.group(1).strip()
    return found


def _post_json(url: str, headers: dict[str, str], payload: dict) -> tuple[int, str]:
    """POST payload as JSON. Returns (status_code, response_body_text).
    On total connection failure (DNS/timeout/refused), status is -1 and the
    body holds the exception string."""
    data = json.dumps(payload).encode("utf-8")
    headers = {**headers, "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return -1, str(e)


def _verdict(status: int, body: str) -> str:
    if status == -1:
        return f"CONNECTION FAILED ({body}) -- can't reach the endpoint from here"
    if status == 200:
        return "WORKS -- got a real 200 response. This tier is live and callable."
    if status == 401:
        return "AUTH FAILED -- bad/missing API key (not proof of paid-vs-free either way)"
    if status == 402:
        return "PAYMENT REQUIRED -- confirmed NOT free, needs a paid balance"
    if status == 404:
        return "MODEL NOT FOUND -- this model id/slug does not exist on this platform"
    if status == 429:
        return "RATE LIMITED -- key is VALID and tier EXISTS, just throttled right now (counts as proof the free tier is real)"
    return f"UNEXPECTED STATUS {status} -- inspect the raw body below"


def _run_test(log: list[str], label: str, key: str | None, url: str, model: str) -> None:
    log.append(f"\n=== {label} ===")
    if not key:
        log.append("  SKIPPED -- no API key configured for this one (see .env.example)")
        return
    payload = {"model": model, **TINY_PROMPT}
    status, body = _post_json(url, {"Authorization": f"Bearer {key}"}, payload)
    log.append(f"  URL:     {url}")
    log.append(f"  Model:   {model}")
    log.append(f"  Status:  {status}")
    log.append(f"  Verdict: {_verdict(status, body)}")
    snippet = body[:400] + ("..." if len(body) > 400 else "")
    log.append(f"  Raw body (first 400 chars): {snippet}")


def _write(log: list[str]) -> tuple[Path, Path]:
    _OUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    text = "\n".join(log) + "\n"
    timestamped = _OUT_DIR / f"free_tier_test_{stamp}.txt"
    latest = _OUT_DIR / "free_tier_test_latest.txt"
    timestamped.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")
    return timestamped, latest


def main() -> None:
    keys = _load_keys()
    log: list[str] = [
        "Empirical free-tier test -- real HTTP calls, real response codes.",
        f"Run (UTC): {datetime.now(UTC).isoformat()}",
        "Any test without its API key configured is skipped, not failed.",
    ]

    if not any(keys.values()):
        log.append(
            "\nNo optional keys found (OPENROUTER_API_KEY / ZENMUX_API_KEY / "
            "ZAI_API_KEY) -- all tests skipped. Add one to .env to actually "
            "test a provider. See the module docstring for signup links."
        )
    else:
        _run_test(
            log,
            "Nemotron-3-Ultra:free via OpenRouter",
            keys["OPENROUTER_API_KEY"],
            "https://openrouter.ai/api/v1/chat/completions",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
        )
        _run_test(
            log,
            "GLM-5.2-free via Zenmux (the exact slug the blog claimed)",
            keys["ZENMUX_API_KEY"],
            "https://zenmux.ai/api/v1/chat/completions",
            "z-ai/glm-5.2-free",
        )
        _run_test(
            log,
            "GLM-5.2 STANDARD (paid) via Zenmux -- comparison only",
            keys["ZENMUX_API_KEY"],
            "https://zenmux.ai/api/v1/chat/completions",
            "z-ai/glm-5.2",
        )
        _run_test(
            log,
            "GLM-5.2 via Zhipu's own direct API (api.z.ai)",
            keys["ZAI_API_KEY"],
            "https://api.z.ai/api/paas/v4/chat/completions",
            "glm-5.2",
        )

    timestamped, latest = _write(log)
    print("\n".join(log))
    print(f"\nSaved -> {timestamped}")
    print(f"Saved -> {latest}  (always overwritten, copy this one)")


if __name__ == "__main__":
    main()
