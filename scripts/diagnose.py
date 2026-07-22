"""OFF-VPN diagnostic: test the REAL external calls and save results to a file.

Run this on a network that can reach the internet directly (e.g. off the
corporate VPN). It exercises the things that unit tests can't:

    * Which Gemini models your API key can actually call (live ListModels)
    * A real 1-token completion against EVERY model in your chain + a few
      candidate Gemini models -> tells us exactly which model to pin
    * A live GDELT news fetch
    * Ollama server + model availability
    * A real end-to-end prediction for one ticker (the ultimate test)

Everything is written to  diagnostics/diag_<UTC-timestamp>.{json,txt}  AND
mirrored to  diagnostics/latest.{json,txt}  so it's easy to find and share.

    uv run python scripts/diagnose.py            # full run
    uv run python scripts/diagnose.py --no-predict   # skip the heavy predict

Non-fatal: every check is isolated; one failure never aborts the rest, and
results are flushed to disk after each section so a hang still leaves a file.
"""
from __future__ import annotations

import asyncio
import json
import platform
import sys
import time
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

# Importing settings validates + loads .env (keys, chain, proxy).
from price_predictor.config.settings import settings

_OUT_DIR = Path("diagnostics")
_CANDIDATE_GEMINI = [
    "gemini/gemini-2.0-flash",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-flash-latest",
]


def _mask(secret: str) -> str:
    if not secret:
        return "(empty)"
    if len(secret) <= 8:
        return "***"
    return f"{secret[:4]}...{secret[-4:]} (len={len(secret)})"


def _write(results: dict[str, Any]) -> Path:
    """(Re)write both the timestamped and 'latest' result files."""
    _OUT_DIR.mkdir(exist_ok=True)
    stamp = results["_meta"]["started_utc"].replace(":", "").replace("-", "")
    stamp = stamp.split(".")[0]
    json_path = _OUT_DIR / f"diag_{stamp}.json"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (_OUT_DIR / "latest.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    txt = _render_txt(results)
    (_OUT_DIR / f"diag_{stamp}.txt").write_text(txt, encoding="utf-8")
    (_OUT_DIR / "latest.txt").write_text(txt, encoding="utf-8")
    return json_path


def _render_txt(r: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 66)
    lines.append("PRICE PREDICTOR - OFF-VPN DIAGNOSTIC RESULTS")
    lines.append(f"started (UTC): {r['_meta']['started_utc']}")
    lines.append("=" * 66)

    env = r.get("environment", {})
    lines.append("\n[ENVIRONMENT]")
    for k, v in env.items():
        lines.append(f"  {k}: {v}")

    gm = r.get("gemini_listmodels", {})
    lines.append("\n[GEMINI - models your key can call]")
    if gm.get("ok"):
        for m in gm.get("flash_models", []):
            lines.append(f"  {m}")
        if not gm.get("flash_models"):
            lines.append("  (no flash models returned)")
    else:
        lines.append(f"  FAILED: {gm.get('error')}")

    lines.append("\n[LIVE MODEL PROBES] (1-token completion per model)")
    for p in r.get("model_probes", []):
        status = "OK  " if p["ok"] else "FAIL"
        detail = f"{p['latency_s']}s" if p["ok"] else p["error"][:120]
        lines.append(f"  [{status}] {p['model']:45s} {detail}")

    news = r.get("gdelt_news", {})
    lines.append("\n[GDELT NEWS]")
    for p in news.get("probes", []):
        if p["ok"]:
            lines.append(f"  [OK  ] {p['query']:10s} {p['rows']} headline(s)  {p.get('sample','')}")
        else:
            lines.append(f"  [FAIL] {p['query']:10s} {p['error'][:90]}")
    if not news.get("probes"):
        lines.append(f"  FAILED: {news.get('error')}")

    floor = r.get("gdelt_floor", {})
    if floor.get("probes"):
        lines.append("\n[GDELT SHORT-NAME RECALL] (raw probes -- accepted? + article count)")
        for p in floor["probes"]:
            if p["accepted"] is True:
                mark = f"OK  {p.get('articles', 0):>3} art"
            elif p["accepted"] is False:
                mark = "REJECTED   "
            else:
                mark = "ERROR      "
            label = p.get("label", "")
            lines.append(f"  [{mark}] {label:38s} {p.get('note','')[:60]}")

    oll = r.get("ollama", {})
    lines.append("\n[OLLAMA]")
    for k, v in oll.items():
        lines.append(f"  {k}: {v}")

    pred = r.get("prediction", {})
    lines.append("\n[END-TO-END PREDICTION]")
    if pred.get("skipped"):
        lines.append("  (skipped via --no-predict)")
    elif pred.get("ok"):
        lines.append(f"  OK - {pred['ticker']} in {pred['latency_s']}s")
        for h in pred.get("horizons", []):
            lines.append(f"    {h}")
    else:
        lines.append(f"  FAILED: {pred.get('error')}")

    lines.append("\n" + "=" * 66)
    lines.append("VERDICT")
    lines.append("=" * 66)
    for v in r.get("verdict", []):
        lines.append(f"  - {v}")
    return "\n".join(lines) + "\n"


def check_environment() -> dict[str, Any]:
    import os

    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "cwd": str(Path.cwd()),
        "chain_agentic": ",".join(settings.effective_chain("agentic")),
        "ollama_api_base": settings.ollama_api_base,
        "gemini_key": _mask(settings.gemini_api_key.get_secret_value()),
        "groq_key": _mask(settings.groq_api_key.get_secret_value()),
        "HTTPS_PROXY": os.environ.get("HTTPS_PROXY", "(unset)"),
        "HTTP_PROXY": os.environ.get("HTTP_PROXY", "(unset)"),
    }


def check_gemini_listmodels() -> dict[str, Any]:
    key = settings.gemini_api_key.get_secret_value()
    if not key or key.startswith("your_"):
        return {"ok": False, "error": "no real GEMINI_API_KEY"}
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        data = json.load(urllib.request.urlopen(url, timeout=20))
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    usable = [
        m["name"].replace("models/", "")
        for m in data.get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]
    return {
        "ok": True,
        "all_usable": sorted(usable),
        "flash_models": sorted(f"gemini/{n}" for n in usable if "flash" in n),
    }


def _api_key_for(model: str) -> str | None:
    provider = model.split("/", 1)[0]
    if provider == "gemini":
        return settings.gemini_api_key.get_secret_value()
    if provider == "groq":
        return settings.groq_api_key.get_secret_value()
    return None  # ollama etc.


def probe_model(model: str) -> dict[str, Any]:
    """Do a real 1-token completion. Records ok/latency/error."""
    import litellm

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the word: ok"}],
        "max_tokens": 8,
        "timeout": 40,
    }
    key = _api_key_for(model)
    if key:
        kwargs["api_key"] = key
    if model.split("/", 1)[0] in ("ollama", "ollama_chat"):
        kwargs["api_base"] = settings.ollama_api_base

    t0 = time.monotonic()
    try:
        resp = litellm.completion(**kwargs)
        content = resp.choices[0].message.content or ""
        return {
            "model": model,
            "ok": True,
            "latency_s": round(time.monotonic() - t0, 2),
            "reply": content.strip()[:60],
        }
    except Exception as exc:
        return {
            "model": model,
            "ok": False,
            "latency_s": round(time.monotonic() - t0, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }


def check_chain_models(discovered_flash: list[str]) -> list[dict[str, Any]]:
    chain = settings.effective_chain("agentic")
    # chain models first, then any candidate/discovered gemini not already there
    candidates = list(dict.fromkeys([*chain, *_CANDIDATE_GEMINI, *discovered_flash]))
    return [probe_model(m) for m in candidates]


def check_gdelt_floor() -> dict[str, Any]:
    """Measure GDELT's ACTUAL behaviour for short-name queries -- both what
    it ACCEPTS and how much it RETURNS (recall), bypassing our own code.

    The real question isn't 'does it stop erroring' -- it's 'which query
    gets the most ITC news'. So for every candidate we record ACCEPTED/
    REJECTED *and* the article count, so we can pick the best variant from
    data instead of guessing.

    Includes the crucial untested case: UNQUOTED bare 'ITC' (a loose token,
    not a quoted phrase) -- if GDELT allows that, it's far better recall
    than any qualified-phrase workaround.

    NOTE: GDELT rate-limits bursts (HTTP 429), so we sleep between probes.
    Read-only.
    """
    end = date.today()
    start = end - timedelta(days=30)  # wider window = more meaningful counts
    sd = start.strftime("%Y%m%d") + "000000"
    ed = end.strftime("%Y%m%d") + "235959"
    probes: list[dict[str, Any]] = []
    # label -> raw GDELT query (exactly as sent, our code NOT involved)
    candidates = [
        ("quoted 2-char floor test", '"IT"'),
        ("quoted bare name", '"ITC"'),
        ("UNQUOTED bare name", "ITC"),
        ("unquoted + IN bias", "ITC"),  # same term, sourcecountry added below
        ("quoted OR (proven broken)", '("ITC" OR "ITC Limited")'),
        ("qualified-phrase group (current fix)",
         '("ITC Limited" OR "ITC Ltd" OR "ITC shares" OR "ITC stock")'),
    ]
    for i, (label, term) in enumerate(candidates):
        if i:
            time.sleep(6)  # avoid GDELT's burst rate-limit (429)
        q = f"{term} sourcelang:eng"
        if label == "unquoted + IN bias":
            q += " sourcecountry:IN"
        url = (
            "https://api.gdeltproject.org/api/v2/doc/doc?"
            f"query={urllib.parse.quote(q)}&mode=ArtList&format=json"
            f"&maxrecords=75&startdatetime={sd}&enddatetime={ed}"
        )
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
                body = r.read().decode("utf-8", "replace")
            try:
                payload = json.loads(body)
                n = len(payload.get("articles", []))
                probes.append({"term": q, "label": label, "accepted": True,
                               "articles": n, "note": f"{n} article(s)"})
            except ValueError:
                probes.append({"term": q, "label": label, "accepted": False,
                               "articles": 0, "note": body.strip()[:100] or "non-JSON"})
        except Exception as exc:
            probes.append({"term": q, "label": label, "accepted": None,
                           "articles": 0, "note": f"{type(exc).__name__}: {exc}"})
    return {"probes": probes}


async def check_gdelt() -> dict[str, Any]:
    """Probe GDELT with a normal name AND a short acronym (ITC) -- the latter
    proves the too-short-query padding fix works live."""
    from price_predictor.data.news import fetch_news

    end = date.today()
    start = end - timedelta(days=7)
    probes: list[dict[str, Any]] = []
    for q in ("Infosys", "ITC"):
        try:
            df = await fetch_news(q, start.isoformat(), end.isoformat(), max_records=3)
            sample = str(df.iloc[0]["title"])[:70] if len(df) else ""
            probes.append({"query": q, "ok": True, "rows": len(df), "sample": sample})
        except Exception as exc:
            probes.append({"query": q, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    # Top-level ok = every probe succeeded (short-name fix included).
    ok = all(p["ok"] for p in probes)
    first = probes[0]
    return {
        "ok": ok,
        "query": first["query"],
        "rows": first.get("rows", 0),
        "sample": first.get("sample", ""),
        "error": next((p["error"] for p in probes if not p["ok"]), None),
        "probes": probes,
    }


def check_ollama() -> dict[str, Any]:
    from price_predictor.llm.ollama_guard import (
        _pulled_models,
        ollama_tags_in_chain,
    )

    tags = ollama_tags_in_chain(settings.effective_chain("agentic"))
    if not tags:
        return {"configured": False, "note": "no ollama model in chain"}
    pulled = _pulled_models(settings.ollama_api_base)
    if pulled is None:
        return {"configured": True, "server_reachable": False, "wanted": tags}
    return {
        "configured": True,
        "server_reachable": True,
        "wanted": tags,
        "pulled": sorted(pulled),
    }


async def check_prediction() -> dict[str, Any]:
    from price_predictor.prediction.predictor import predict

    ticker = "RELIANCE.NS"
    t0 = time.monotonic()
    try:
        out = await predict(ticker, horizons=["daily"])
        horizons = [
            f"{h.value if hasattr(h, 'value') else h}: "
            f"{p.direction} @ conf {getattr(p, 'confidence', '?')}"
            for h, p in out.items()
        ]
        return {
            "ok": True,
            "ticker": ticker,
            "latency_s": round(time.monotonic() - t0, 2),
            "horizons": horizons,
        }
    except Exception as exc:
        return {
            "ok": False,
            "ticker": ticker,
            "latency_s": round(time.monotonic() - t0, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _build_verdict(r: dict[str, Any]) -> list[str]:
    out: list[str] = []
    probes = r.get("model_probes", [])
    working = [p["model"] for p in probes if p["ok"]]
    broken = [p["model"] for p in probes if not p["ok"]]
    gem_ok = [m for m in working if m.startswith("gemini/")]
    if gem_ok:
        out.append(f"WORKING Gemini model(s): {', '.join(gem_ok)}")
        out.append(f"-> recommend setting CHAIN_AGENTIC primary to: {gem_ok[0]}")
    else:
        out.append("NO Gemini model worked - check the key / account access.")
    if broken:
        out.append(f"BROKEN models (do not use): {', '.join(broken)}")
    news = r.get("gdelt_news", {})
    out.append("News (GDELT): " + ("WORKING" if news.get("ok") else f"FAILED - {news.get('error','')[:80]}"))
    oll = r.get("ollama", {})
    if oll.get("configured"):
        if oll.get("server_reachable"):
            missing = set(oll.get("wanted", [])) - set(oll.get("pulled", []))
            out.append("Ollama: reachable" + (f", MISSING {missing}" if missing else ", model present"))
        else:
            out.append("Ollama: server NOT reachable (start it / install model)")
    pred = r.get("prediction", {})
    if not pred.get("skipped"):
        out.append("End-to-end prediction: " + ("WORKING" if pred.get("ok") else f"FAILED - {pred.get('error','')[:80]}"))
    return out


def main() -> None:
    do_predict = "--no-predict" not in sys.argv
    results: dict[str, Any] = {
        "_meta": {"started_utc": datetime.now(UTC).isoformat()},
    }

    print("Running diagnostic (this makes real network + LLM calls)...\n")

    results["environment"] = check_environment()
    _write(results)
    print("  [1/6] environment captured")

    results["gemini_listmodels"] = check_gemini_listmodels()
    _write(results)
    print("  [2/6] gemini ListModels done")

    discovered = (
        results["gemini_listmodels"].get("flash_models", [])
        if results["gemini_listmodels"].get("ok")
        else []
    )
    results["model_probes"] = check_chain_models(discovered)
    _write(results)
    print("  [3/6] live model probes done")

    results["gdelt_news"] = asyncio.run(check_gdelt())
    results["gdelt_floor"] = check_gdelt_floor()
    _write(results)
    print("  [4/6] GDELT news + query-floor check done")

    results["ollama"] = check_ollama()
    _write(results)
    print("  [5/6] ollama check done")

    if do_predict:
        results["prediction"] = asyncio.run(check_prediction())
    else:
        results["prediction"] = {"skipped": True}
    _write(results)
    print("  [6/6] end-to-end prediction done")

    results["verdict"] = _build_verdict(results)
    path = _write(results)

    print("\n" + _render_txt(results))
    print(f"\nResults saved to: {path}")
    print(f"Share this file with me: {(_OUT_DIR / 'latest.txt').resolve()}")


if __name__ == "__main__":
    main()
