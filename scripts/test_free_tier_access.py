"""OFF-VPN diagnostic: prove (or disprove) free-tier access to NVIDIA
Nemotron-3-Ultra with a REAL API call, and save the result so it can be
copied back into chat.

CONTEXT: this repo's fallback chain (see src/price_predictor/llm/factory.py
and config/settings.py) currently goes
    gemini -> groq (x2) -> ollama_chat/qwen3:8b (local)
We're evaluating whether to add NVIDIA Nemotron-3-Ultra:free (via OpenRouter)
as an extra tier between Groq and Ollama. A live call confirmed it works
(HTTP 200, real completion) as of 2026-08-11 -- this script exists to keep
proving that stays true, since NVIDIA's own Trial Terms of Service say they
can end/paywall it at any time with zero notice (it's an explicitly
credit-metered trial, not a stable free tier like Groq/Gemini).

(GLM-5.2 was evaluated the same way and dropped entirely: confirmed, three
independent ways, to have no free API access anywhere -- Zenmux's "-free"
slug redirects to the paid model, Zenmux's own docs say their free tier is
Studio-Chat-only with no API access, and Zhipu's own direct api.z.ai
endpoint returned "Insufficient balance or no resource package" on a live
call. Not revisiting unless a genuinely new free path turns up.)

Runs automatically, non-fatally, every launch (see windows_setup/launch.bat)
-- but costs nothing and does nothing beyond an instant skip unless you've
added OPENROUTER_API_KEY to your .env.

    uv run python scripts/test_free_tier_access.py

WHAT YOU NEED TO PROVIDE (I cannot get this myself -- it's tied to your
identity, not the app's):

  OPENROUTER_API_KEY  -- for Nemotron-3-Ultra:free
      Sign up (free, ~1 min, no card): https://openrouter.ai/keys

Leave it unset to skip the test entirely (near-instant, no network call).
None of this touches the app's actual chain; it's a pure read-only
diagnostic, same category as scripts/diagnose.py or scripts/ensure_ollama.py.

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
_ENV_KEY = "OPENROUTER_API_KEY"

TINY_PROMPT = {
    "messages": [{"role": "user", "content": "Reply with just the word OK."}],
    "max_tokens": 5,
}


def _load_key() -> str | None:
    """Env var first; falls back to .env (same pattern as
    scripts/list_gemini_models.py). A placeholder value (empty or
    "your_..._here") is treated as unset, same convention as the rest of
    this repo's tooling."""
    val = os.environ.get(_ENV_KEY)
    if val and not val.startswith("your_"):
        return val
    env_path = Path(".env")
    if env_path.exists():
        text = env_path.read_text(encoding="utf-8")
        m = re.search(rf"^{_ENV_KEY}=(.+)$", text, re.MULTILINE)
        if m and not m.group(1).strip().startswith("your_"):
            return m.group(1).strip()
    return None


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
    """Translate an HTTP status into a plain-English verdict.

    NOTE: a status code alone can mislead -- e.g. some providers use 429 as
    a billing wall rather than a rate limit (seen with Zhipu's direct API:
    429 + "Insufficient balance"). Always read the raw body too, not just
    this line, before trusting a verdict.
    """
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
        return (
            "RATE LIMITED OR BILLING WALL -- key is valid but check the raw "
            "body below: a real quota throttle counts as proof the free "
            "tier exists, but some providers (e.g. Zhipu) return 429 to "
            "mean 'no balance', not 'too many requests'."
        )
    return f"UNEXPECTED STATUS {status} -- inspect the raw body below"


def _run_test(log: list[str], key: str | None) -> None:
    label = "Nemotron-3-Ultra:free via OpenRouter"
    log.append(f"\n=== {label} ===")
    if not key:
        log.append(f"  SKIPPED -- no {_ENV_KEY} configured (see .env.example)")
        return
    url = "https://openrouter.ai/api/v1/chat/completions"
    model = "nvidia/nemotron-3-ultra-550b-a55b:free"
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
    key = _load_key()
    log: list[str] = [
        "Empirical free-tier test -- real HTTP calls, real response codes.",
        f"Run (UTC): {datetime.now(UTC).isoformat()}",
        "Skipped (not failed) if OPENROUTER_API_KEY isn't configured.",
    ]
    _run_test(log, key)

    timestamped, latest = _write(log)
    print("\n".join(log))
    print(f"\nSaved -> {timestamped}")
    print(f"Saved -> {latest}  (always overwritten, copy this one)")


if __name__ == "__main__":
    main()
