"""Probes, resumable result store, per-provider budget and scheduling for the free-model sweep.

Cheap probes first (one tiny chat call, one tiny json_schema call); the expensive pipeline test
(two real LLM steps per run) only for models that answer. Nothing here knows about Docker or the
network: `chat` and `c` are injected, so every rule is testable with a fake.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sweep_data import TESTABLE, TRANSIENT, Candidate

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
RETRYABLE = TRANSIENT | {"budget", "other"}   # on --resume these are tried again; permanent failures are not
FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


@dataclass
class Result:
    uid: str
    level: str                      # chat | json | pipeline
    verdict: str                    # PASS | FAIL | SKIP
    klass: str = ""
    detail: str = ""
    seconds: float = 0.0
    run: int = 0
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


def classify_http(status: int, text: str, error: str = "") -> str:
    """Why a call failed, from the HTTP outcome alone. '' means it did not fail."""
    if error:
        return "timeout" if "timeout" in error.lower() else "network"
    if status == 200:
        return ""
    low = text.lower()
    if status == 402:
        return "payment"
    if status in (401, 403):
        return "auth"
    if status == 404:
        return "not_found"
    if status == 429 or (status in (400, 422) and ("rate limit" in low or "quota" in low)):
        return "rate_limited"
    if status in (400, 422):
        return "unsupported"
    return "overloaded" if status >= 500 else "other"


def reply_text(resp) -> tuple[str, bool]:
    """(content, whether the model also returned reasoning). A reasoning-only reply still proves it is alive."""
    try:
        msg = resp.json()["choices"][0]["message"]
    except (TypeError, KeyError, IndexError):
        return "", False
    return msg.get("content") or "", bool(msg.get("reasoning_content") or msg.get("reasoning"))


def probe_chat(chat, c, cand: Candidate) -> Result:
    t0 = time.time()
    r, _ = chat(c, cand.base, cand.key, cand.model, "Reply with exactly the word: ok", max_tokens=300)
    content, reasoned = reply_text(r)
    secs = round(time.time() - t0, 1)
    if r.status == 200 and (content.strip() or reasoned):
        via = f" via={r.headers['x-routed-via']}" if r.headers.get("x-routed-via") else ""  # which platform really answered
        return Result(cand.uid, "chat", "PASS", detail=((content.strip() or "(reasoning only)")[:80] + via), seconds=secs)
    klass = classify_http(r.status, r.text, r.error) or "empty"
    return Result(cand.uid, "chat", "FAIL", klass, r.evidence()[:300], secs)


def probe_json(chat, c, cand: Candidate) -> Result:
    t0 = time.time()
    rf = {"type": "json_schema", "json_schema": {"name": "t", "schema": SCHEMA}}
    r, _ = chat(c, cand.base, cand.key, cand.model, 'Reply with JSON: {"answer": "ok"}',
                response_format=rf, max_tokens=600)
    content, _ = reply_text(r)
    secs = round(time.time() - t0, 1)
    stripped = FENCE.sub("", content.strip())
    try:
        ok = isinstance(json.loads(stripped).get("answer"), str)
    except (ValueError, AttributeError):
        ok = False
    if ok:
        return Result(cand.uid, "json", "PASS", detail="valid (needed code fences stripped)" if stripped != content.strip()
                      else "valid", seconds=secs)
    klass = "bad_output" if r.status == 200 else (classify_http(r.status, r.text, r.error) or "other")
    return Result(cand.uid, "json", "FAIL", klass, r.evidence(content)[:300], secs)


class Store:
    """Append-only JSONL of probe results. Loading earlier files makes the sweep resumable."""

    def __init__(self, path: Path, resume_from: tuple[Path, ...] = ()) -> None:
        self.path, self.rows, self._lock = path, {}, threading.Lock()
        for p in resume_from:
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = Result(**json.loads(line))
                    self.rows[(r.uid, r.level, r.run)] = r

    def add(self, r: Result) -> None:
        with self._lock:
            self.rows[(r.uid, r.level, r.run)] = r
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def get(self, uid: str, level: str, run: int = 0) -> Result | None:
        return self.rows.get((uid, level, run))

    def runs(self, uid: str) -> list[Result]:
        return sorted((r for (u, lv, _), r in self.rows.items() if u == uid and lv == "pipeline"), key=lambda r: r.run)

    def needs(self, uid: str, level: str, run: int = 0) -> bool:
        """True if never done, or done but failed in a way worth retrying."""
        r = self.get(uid, level, run)
        # a pipeline run stores its failure classes comma-joined ("bad_output,timeout"): any retryable one counts
        return r is None or (r.verdict != "PASS" and any(k in RETRYABLE for k in r.klass.split(",")))


def transient_pending(cands: list[Candidate], store: Store, repeats: int) -> list[Candidate]:
    """Candidates that failed in a way a short wait may fix. Budget skips are excluded: they only clear tomorrow."""
    out = []
    for cand in cands:
        rows = [store.get(cand.uid, "chat"), store.get(cand.uid, "json"),
                *(store.get(cand.uid, "pipeline", i) for i in range(1, repeats + 1))]
        if cand.status == TESTABLE and any(r and r.verdict == "FAIL" and set(r.klass.split(",")) & TRANSIENT for r in rows):
            out.append(cand)
    return out


def latest_results(reports: Path) -> Path | None:
    files = sorted(f for f in reports.glob("free_model_sweep_*.jsonl") if not f.name.endswith("_verdicts.jsonl"))
    return files[-1] if files else None


class Budget:
    """Upstream-call allowance per provider: the sweep must not eat the quota your real app needs."""

    def __init__(self, limits: dict[str, int], default: int) -> None:
        self.limits, self.default, self.used, self._lock = limits, default, Counter(), threading.Lock()

    def left(self, provider: str) -> int:
        return self.limits.get(provider, self.default) - self.used[provider]

    def take(self, provider: str, n: int = 1) -> bool:
        with self._lock:
            if self.left(provider) < n:
                return False
            self.used[provider] += n
            return True


def probe_provider(cands: list[Candidate], chat, c, store: Store, budget: Budget, pace_s: float,
                   sleep=time.sleep, say=print) -> None:
    """Chat probe, then json_schema probe, for one provider's candidates, one call at a time."""
    for cand in cands:
        for level, fn in (("chat", probe_chat), ("json", probe_json)):
            if level == "json" and (store.get(cand.uid, "chat") or Result("", "", "FAIL")).verdict != "PASS":
                break
            if not store.needs(cand.uid, level):
                continue
            if not budget.take(cand.budget_key):
                store.add(Result(cand.uid, level, "SKIP", "budget", f"{cand.budget_key} call budget used up"))
                break
            res = fn(chat, c, cand)
            store.add(res)
            say(f"  [{res.verdict:4s}] {cand.provider:11s} {cand.model[:48]:48s} {level:5s} {res.klass or res.detail[:30]}")
            sleep(pace_s)


def pick_next(queues: dict[str, list], ready_at: dict[str, float], now: float) -> tuple[str | None, float]:
    """Round-robin by readiness: the provider whose pace allows a call soonest, and how long to wait for it."""
    live = [p for p, q in queues.items() if q]
    if not live:
        return None, 0.0
    p = min(live, key=lambda k: ready_at.get(k, 0.0))
    return p, max(0.0, ready_at.get(p, 0.0) - now)


# Families that tend to do well at structured, reasoning-heavy output: probed first when a budget cannot cover all.
STRONG = re.compile(r"deepseek|qwen3|gpt-oss|gemini|kimi|glm|minimax|nemotron-(super|ultra)|llama-?(3\.3|4)|"
                    r"mistral-(large|medium)|gemma-?[34]", re.I)
TINY = re.compile(r"(?<![\w.])(\d|1[0-4])(\.\d)?b\b|\b(mini|nano|tiny)\b", re.I)   # <=14B or "mini": 'a3b' is not 3B


def probe_order(cands: list[Candidate]) -> list[Candidate]:
    """Most promising first, so a limited call budget is spent where a pass is likeliest. Nothing is excluded."""
    return sorted(cands, key=lambda c: (0 if c.hint_structured else 1, 0 if STRONG.search(c.model) else 1,
                                        1 if TINY.search(c.model) else 0))


def pipeline_order(cands: list[Candidate], store: Store) -> list[Candidate]:
    """Most promising first, so a limited budget is spent where a pass is likeliest. Nothing is excluded."""
    def key(cand: Candidate):
        j = store.get(cand.uid, "json")
        return (0 if j and j.verdict == "PASS" else 1, 0 if cand.hint_structured else 1, cand.provider, cand.model)
    return sorted(cands, key=key)
