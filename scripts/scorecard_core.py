"""The scorecard engine: run each shortlisted model over the SAME historical cases, grade, aggregate.

Everything that touches the network or the LLM is injected (`Plumbing`), so this module is tested end to end with
fakes and the live adapters in model_scorecard_offvpn.py stay thin.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import scorecard_stats as ST
from sweep_data import Candidate
from sweep_probe import RETRYABLE, Budget, read_jsonl

CASE_RETRYABLE = RETRYABLE | {"inputs"}


@dataclass(frozen=True)
class Case:
    ticker: str
    as_of: date

    @property
    def key(self) -> str:
        return f"{self.ticker}@{self.as_of.isoformat()}"

    @classmethod
    def parse(cls, key: str) -> Case:
        ticker, day = key.split("@")
        return cls(ticker, date.fromisoformat(day))


def pick_cases(tickers: list[str], days: list[date], n: int) -> list[Case]:
    """`n` cases spread evenly across `days`, tickers rotated, so every period and every ticker is represented."""
    if not tickers or not days or n <= 0:
        return []
    n = min(n, len(days))
    idx = [round(i * (len(days) - 1) / max(n - 1, 1)) for i in range(n)]
    return [Case(tickers[i % len(tickers)], days[j]) for i, j in enumerate(idx)]


def interleave(models: list[Candidate]) -> list[Candidate]:
    """Round-robin across providers, so one provider's pacing delay is spent running another provider's model."""
    queues: dict[str, list[Candidate]] = defaultdict(list)
    for m in models:
        queues[m.provider].append(m)
    out: list[Candidate] = []
    while any(queues.values()):
        out += [q.pop(0) for q in queues.values() if q]
    return out


@dataclass
class CaseResult:
    model_uid: str                     # "*" = the case's shared inputs could not be gathered (nobody's fault)
    case: str
    status: str                        # OK | FAIL | SKIP
    klass: str = ""
    detail: str = ""
    seconds: float = 0.0
    predictions: list[str] = field(default_factory=list)      # Prediction JSON, one per horizon
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


class CaseStore:
    """Append-only JSONL, one row per (model, case) attempt, plus a header row that pins the case grid.

    The grid is saved so a re-run uses the SAME cases (evidence accumulates instead of restarting), and each
    resume carries history forward into its own file, because --resume latest only ever reads the newest one.
    """

    def __init__(self, path: Path, resume_from: tuple[Path, ...] = ()) -> None:
        self.path, self.rows, self.cases, self._lock = path, {}, None, threading.Lock()
        for p in resume_from:
            for d in read_jsonl(p):
                if d.get("type") == "cases":
                    self.cases = [Case.parse(k) for k in d["cases"]]
                else:
                    r = CaseResult(**d)
                    self.rows[(r.model_uid, r.case)] = r
        if self.cases or self.rows:
            head = [self._header()] if self.cases else []
            self.path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n"
                                         for x in [*head, *(asdict(r) for r in self.rows.values())]), encoding="utf-8")

    def _header(self) -> dict:
        return {"type": "cases", "cases": [c.key for c in self.cases or []]}

    def _append(self, obj: dict) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def set_cases(self, cases: list[Case]) -> None:
        self.cases = cases
        self._append(self._header())

    def add(self, r: CaseResult) -> None:
        self.rows[(r.model_uid, r.case)] = r
        self._append(asdict(r))

    def get(self, uid: str, case_key: str) -> CaseResult | None:
        return self.rows.get((uid, case_key))

    def needs(self, uid: str, case_key: str) -> bool:
        """True if never attempted, or it failed in a way worth retrying. A model's own bad output is final."""
        r = self.get(uid, case_key)
        return r is None or (r.status != "OK" and any(k in CASE_RETRYABLE for k in r.klass.split(",")))

    def attempts(self, uid: str) -> list[CaseResult]:
        return [r for (u, _), r in self.rows.items() if u == uid]

    def predictions(self, uid: str) -> list:
        from price_predictor.prediction.schema import Prediction
        return [Prediction.model_validate_json(js) for r in self.attempts(uid) if r.status == "OK"
                for js in r.predictions]


@dataclass
class Plumbing:
    """The outside world, injected: live adapters in production, fakes in tests."""
    shared: Callable[[Case], Awaitable[object]]                                  # inputs every model shares
    predict: Callable[[Candidate, Case, object], Awaitable[list]]                # -> list[Prediction]
    pace_for: Callable[[str], float]
    classify: Callable[[str], str]
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    clock: Callable[[], float] = time.monotonic
    say: Callable[[str], None] = print


async def run_scorecard(models: list[Candidate], cases: list[Case], store: CaseStore, budget: Budget, plumb: Plumbing,
                        *, calls_per_case: int, deadline: float = float("inf")) -> None:
    """Cases outermost, models inner: if time or budget runs out, every model has seen the same earliest cases.

    Sequential on purpose: the predictor's agents are module singletons, so two runs at once would trample each
    other's model.
    """
    ordered, ready_at = interleave(models), {}
    for case in cases:
        if plumb.clock() >= deadline:
            plumb.say("scorecard: time limit reached; the next run continues")
            return
        runnable = []
        for m in (m for m in ordered if store.needs(m.uid, case.key)):
            if budget.left(m.budget_key) < calls_per_case:
                store.add(CaseResult(m.uid, case.key, "SKIP", "budget", f"{m.budget_key} call budget used up"))
            else:
                runnable.append(m)
        if not runnable:
            continue
        try:
            shared = await plumb.shared(case)
        except Exception as e:
            store.add(CaseResult("*", case.key, "FAIL", "inputs", f"{type(e).__name__}: {e}"[:300]))
            plumb.say(f"  [skip] {case.key}: shared inputs failed ({type(e).__name__}); retried next run")
            continue
        for m in runnable:
            if plumb.clock() >= deadline or not budget.take(m.budget_key, calls_per_case):
                continue
            wait = ready_at.get(m.provider, 0.0) - plumb.clock()
            if wait > 0:
                await plumb.sleep(wait)
            t0 = plumb.clock()
            try:
                preds = await plumb.predict(m, case, shared)
                res = CaseResult(m.uid, case.key, "OK", seconds=round(plumb.clock() - t0, 1),
                                 predictions=[p.model_dump_json() for p in preds])
            except TimeoutError:
                res = CaseResult(m.uid, case.key, "FAIL", "timeout", "step timed out", round(plumb.clock() - t0, 1))
            except Exception as e:
                msg = " ".join(f"{type(e).__name__}: {e}".split())
                res = CaseResult(m.uid, case.key, "FAIL", plumb.classify(msg), msg[:400], round(plumb.clock() - t0, 1))
            store.add(res)
            ready_at[m.provider] = plumb.clock() + plumb.pace_for(m.provider)
            plumb.say(f"  [{res.status:4s}] {m.provider:11s} {m.model[:40]:40s} {case.key:24s} {res.klass or ''}")


@dataclass(frozen=True)
class ModelScore:
    cand: Candidate
    attempted: int
    ok: int
    fail_classes: dict[str, int]
    skipped: int
    n_pred: int
    n_judged: int
    k_correct: int
    verdict: str
    low: float
    high: float
    baseline: float
    hit_rate_resolved: float | None
    bss: float | None
    neutral_share: float | None
    avg_seconds: float | None

    @property
    def accuracy(self) -> float | None:
        return self.k_correct / self.n_judged if self.n_judged else None

    @property
    def reliability(self) -> float | None:
        return self.ok / self.attempted if self.attempted else None


def aggregate(models: list[Candidate], store: CaseStore, grade_fn: Callable[[list], list] | None = None
              ) -> tuple[list[ModelScore], float, int]:
    """Grade every stored prediction with the APP's grader and score each model with the APP's calibration maths.

    -> (scores ranked best first, baseline accuracy, number of distinct graded outcomes behind the baseline)
    """
    from price_predictor.prediction.calibration import compute_calibration
    from price_predictor.prediction.grading import neutral_tolerance
    if grade_fn is None:
        from price_predictor.prediction.grading import grade_many as grade_fn
    preds = {m.uid: store.predictions(m.uid) for m in models}
    flat = [p for ps in preds.values() for p in ps]
    graded_iter = iter(grade_fn(flat) if flat else [])
    graded = {uid: [next(graded_iter) for _ in ps] for uid, ps in preds.items()}
    outcomes = {(g.prediction.ticker, g.prediction.as_of.date(), g.prediction.horizon):
                (g.realized_return, neutral_tolerance(g.prediction.horizon))
                for gs in graded.values() for g in gs if g.direction_correct is not None}    # judged outcomes only
    baseline = ST.baseline_accuracy(outcomes.values())
    scores = []
    for m in models:
        tries = [r for r in store.attempts(m.uid) if r.status in ("OK", "FAIL")]
        ok = [r for r in tries if r.status == "OK"]
        cal = compute_calibration(graded[m.uid]) if graded[m.uid] else None
        n = cal.n_with_direction_judgement if cal else 0
        k = round(cal.direction_accuracy * n) if cal else 0
        verdict, low, high = ST.judge(k, n, baseline)
        scores.append(ModelScore(
            m, len(tries), len(ok), dict(Counter(r.klass or "other" for r in tries if r.status == "FAIL")),
            sum(1 for r in store.attempts(m.uid) if r.status == "SKIP"), len(graded[m.uid]), n, k, verdict, low, high,
            baseline, cal.hit_rate_resolved if cal else None, cal.brier_skill_score if cal else None,
            cal.n_not_applicable / cal.n_predictions if cal else None,
            sum(r.seconds for r in ok) / len(ok) if ok else None))
    scores.sort(key=lambda s: (ST.VERDICT_RANK[s.verdict], -s.low, -(s.reliability or 0)))
    return scores, baseline, len(outcomes)
