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


PICKER = "movers-v1"      # how the saved case grid was chosen; a grid chosen any other way is not resumed


def case_moves(ticker: str, closes: list[tuple[date, float]], days: list[date],
               windows: list[tuple[int, float]]) -> dict[tuple[str, date], float]:
    """How hard the price moved after each day, in units of the neutral tolerance: above 1 is a REAL move.

    `windows` is (trading days, neutral tolerance) per horizon; the strongest horizon counts. The return is measured
    exactly as the grader does: from the as-of close to the close that many trading days later. A day whose
    horizons have not all resolved yet is left out, so it is never mistaken for a calm one.
    """
    index = {d: i for i, (d, _) in enumerate(closes)}
    out: dict[tuple[str, date], float] = {}
    for day in days:
        i = index.get(day)
        if i is None or i + max(w for w, _ in windows) >= len(closes):
            continue
        out[(ticker, day)] = max(abs(closes[i + w][1] / closes[i][1] - 1) / tol for w, tol in windows)
    return out


def _spread(items: list, k: int) -> list:
    """`k` items spread evenly across a (date-ordered) list."""
    if k <= 0:
        return []
    return [items[round(i * (len(items) - 1) / max(k - 1, 1))] for i in range(k)]


def pick_cases(tickers: list[str], days: list[date], n: int,
               moves: dict[tuple[str, date], float] | None = None) -> list[Case]:
    """`n` cases spread across `days`, tickers rotated, so every period and every ticker is represented.

    With `moves` (see `case_moves`), half the cases are days after which the price really moved and half are calm
    days. Without that, most of a random sample is calm and "always neutral" wins it, so no model could stand out.
    """
    if not tickers or not days or n <= 0:
        return []
    known = [(t, d) for d in days for t in tickers if (t, d) in (moves or {})]
    if not known:                                     # no price information: plain even spread
        n = min(n, len(days))
        idx = [round(i * (len(days) - 1) / max(n - 1, 1)) for i in range(n)]
        return [Case(tickers[i % len(tickers)], days[j]) for i, j in enumerate(idx)]
    movers = [c for c in known if moves[c] > 1.0]
    calm = [c for c in known if moves[c] <= 1.0]
    n_calm = min(len(calm), n // 2)
    n_move = min(len(movers), n - n_calm)
    n_calm = min(len(calm), n - n_move)               # too few movers: calm days fill the rest
    chosen = sorted(_spread(movers, n_move) + _spread(calm, n_calm), key=lambda c: (c[1], c[0]))
    return [Case(t, d) for t, d in chosen]


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
        self.picker: str | None = None            # how the grid was chosen (None = an older run that did not say)
        for p in resume_from:
            for d in read_jsonl(p):
                if d.get("type") == "cases":
                    self.cases = [Case.parse(k) for k in d["cases"]]
                    self.picker = d.get("picker")
                else:
                    r = CaseResult(**d)
                    self.rows[(r.model_uid, r.case)] = r
        if self.cases or self.rows:
            head = [self._header()] if self.cases else []
            self.path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n"
                                         for x in [*head, *(asdict(r) for r in self.rows.values())]), encoding="utf-8")

    def _header(self) -> dict:
        return {"type": "cases", "cases": [c.key for c in self.cases or []], "picker": self.picker}

    def _append(self, obj: dict) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def set_cases(self, cases: list[Case], picker: str = PICKER) -> None:
        self.cases, self.picker = cases, picker
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

    The budget is per MODEL, not per provider: models of one provider have their own quotas, and a shared pool let
    six Gemini models split one model's worth of cases, so none of them ever got enough to judge.

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
            if budget.left(m.uid) < calls_per_case:
                store.add(CaseResult(m.uid, case.key, "SKIP", "budget", f"{m.provider}/{m.model} call budget used up"))
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
            if plumb.clock() >= deadline or not budget.take(m.uid, calls_per_case):
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
    n_moves: int = 0           # graded calls on days the price REALLY moved: the verdict is based on these
    k_moves: int = 0
    n_calm: int = 0            # graded calls on calm days: shown for information, never decides the verdict
    k_calm: int = 0

    @property
    def accuracy(self) -> float | None:
        return self.k_correct / self.n_judged if self.n_judged else None

    @property
    def move_accuracy(self) -> float | None:
        return self.k_moves / self.n_moves if self.n_moves else None

    @property
    def calm_accuracy(self) -> float | None:
        return self.k_calm / self.n_calm if self.n_calm else None

    @property
    def reliability(self) -> float | None:
        return self.ok / self.attempted if self.attempted else None


def aggregate(models: list[Candidate], store: CaseStore, grade_fn: Callable[[list], list] | None = None
              ) -> tuple[list[ModelScore], float, int]:
    """Grade every stored prediction with the APP's grader and score each model with the APP's calibration maths.

    The verdict looks only at days the price REALLY moved (further than the neutral tolerance): on a calm day
    "always neutral" is a free win, which made the baseline 85-100% and the contest unwinnable.

    -> (scores ranked best first, baseline accuracy on real moves, number of distinct real-move outcomes behind it)
    """
    from price_predictor.prediction.calibration import compute_calibration
    from price_predictor.prediction.grading import neutral_tolerance
    if grade_fn is None:
        from price_predictor.prediction.grading import grade_many as grade_fn
    preds = {m.uid: store.predictions(m.uid) for m in models}
    flat = [p for ps in preds.values() for p in ps]
    graded_iter = iter(grade_fn(flat) if flat else [])
    graded = {uid: [next(graded_iter) for _ in ps] for uid, ps in preds.items()}
    def real_move(g) -> bool:
        return ST.is_move(g.realized_return, neutral_tolerance(g.prediction.horizon))

    outcomes = {(g.prediction.ticker, g.prediction.as_of.date(), g.prediction.horizon):
                (g.realized_return, neutral_tolerance(g.prediction.horizon))
                for gs in graded.values() for g in gs if g.direction_correct is not None and real_move(g)}
    baseline = ST.baseline_accuracy(outcomes.values())
    scores = []
    for m in models:
        tries = [r for r in store.attempts(m.uid) if r.status in ("OK", "FAIL")]
        ok = [r for r in tries if r.status == "OK"]
        cal = compute_calibration(graded[m.uid]) if graded[m.uid] else None
        n = cal.n_with_direction_judgement if cal else 0
        k = round(cal.direction_accuracy * n) if cal else 0
        judged = [g for g in graded[m.uid] if g.direction_correct is not None]
        moves, calm = [g for g in judged if real_move(g)], [g for g in judged if not real_move(g)]
        k_moves = sum(1 for g in moves if g.direction_correct)
        verdict, low, high = ST.judge(k_moves, len(moves), baseline, len(outcomes))
        scores.append(ModelScore(
            m, len(tries), len(ok), dict(Counter(r.klass or "other" for r in tries if r.status == "FAIL")),
            sum(1 for r in store.attempts(m.uid) if r.status == "SKIP"), len(graded[m.uid]), n, k, verdict, low, high,
            baseline, cal.hit_rate_resolved if cal else None, cal.brier_skill_score if cal else None,
            cal.n_not_applicable / cal.n_predictions if cal else None,
            sum(r.seconds for r in ok) / len(ok) if ok else None,
            len(moves), k_moves, len(calm), sum(1 for g in calm if g.direction_correct)))
    scores.sort(key=lambda s: (ST.VERDICT_RANK[s.verdict], -s.low, -(s.reliability or 0)))
    return scores, baseline, len(outcomes)
