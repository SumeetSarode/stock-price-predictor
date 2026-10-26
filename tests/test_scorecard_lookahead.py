"""The next case's news downloads while the current case's models answer, and nothing is left running or unread."""
from __future__ import annotations

import asyncio
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import scorecard_core as SC
import sweep_probe as PR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorecard_fakes import cand, prediction

A, B, C = SC.Case("A.NS", date(2026, 8, 3)), SC.Case("B.NS", date(2026, 8, 4)), SC.Case("C.NS", date(2026, 8, 5))


class Rig:
    """Fake plumbing that logs the order things start and finish, with real awaits so tasks can interleave."""

    def __init__(self, shared_fails=(), model_fails=(), slow=()):
        self.log, self.shared_fails, self.model_fails, self.slow = [], set(shared_fails), set(model_fails), set(slow)

    async def shared(self, case):
        self.log.append(f"shared_start:{case.ticker}")
        await asyncio.sleep(30 if case.ticker in self.slow else 0.01)
        if case.ticker in self.shared_fails:
            raise RuntimeError("news never arrived")
        self.log.append(f"shared_end:{case.ticker}")
        return "tv"

    async def predict(self, m, case, tv):
        self.log.append(f"predict_start:{m.model}:{case.ticker}")
        await asyncio.sleep(0.05)
        self.log.append(f"predict_end:{m.model}:{case.ticker}")
        if (m.model, case.ticker) in self.model_fails:
            raise RuntimeError("model fell over")
        return [prediction()]

    def plumbing(self):
        return SC.Plumbing(self.shared, self.predict, lambda p: 0, lambda msg: "other", say=lambda *_: None)


def run(rig, cases, models, store, budget=None, deadline=float("inf")):
    async def main():
        await SC.run_scorecard(models, cases, store, budget or PR.Budget({}, 999), rig.plumbing(), calls_per_case=6, deadline=deadline)
        return len(asyncio.all_tasks())
    return asyncio.run(main())


def fresh(tmp_path, cases):
    store = SC.CaseStore(tmp_path / "s.jsonl")
    store.set_cases(cases)
    return store


def test_the_next_cases_inputs_start_before_this_cases_models_finish(tmp_path):
    rig = Rig()
    run(rig, [A, B], [cand("m")], fresh(tmp_path, [A, B]))
    assert rig.log.index("shared_start:B.NS") < rig.log.index("predict_end:m:A.NS"), rig.log
    assert rig.log.index("shared_end:A.NS") < rig.log.index("shared_start:B.NS"), "but never two downloads at once"


def test_every_case_is_still_scored_in_order_by_every_model(tmp_path):
    rig, store = Rig(), fresh(tmp_path, [A, B, C])
    m1, m2 = cand("m1"), cand("m2")                      # one provider, so they run in the order given
    run(rig, [A, B, C], [m1, m2], store)
    predicted = [e.split(":", 1)[1] for e in rig.log if e.startswith("predict_start")]
    assert predicted == ["m1:A.NS", "m2:A.NS", "m1:B.NS", "m2:B.NS", "m1:C.NS", "m2:C.NS"]
    assert all(store.get(m.uid, c.key).status == "OK" for m in (m1, m2) for c in (A, B, C))


def test_a_download_that_fails_while_being_fetched_ahead_skips_only_that_case(tmp_path):
    rig, store = Rig(shared_fails={"B.NS"}), fresh(tmp_path, [A, B, C])
    leftover = run(rig, [A, B, C], [cand("m")], store)
    assert store.get("*", B.key).klass == "inputs" and store.get(cand("m").uid, B.key) is None
    assert store.get(cand("m").uid, A.key).status == "OK" and store.get(cand("m").uid, C.key).status == "OK"
    assert leftover == 1, "no download left running"


def test_a_model_failure_does_not_stop_the_next_cases_download_or_run(tmp_path):
    rig, store = Rig(model_fails={("m", "A.NS")}), fresh(tmp_path, [A, B])
    run(rig, [A, B], [cand("m")], store)
    assert store.get(cand("m").uid, A.key).status == "FAIL" and store.get(cand("m").uid, B.key).status == "OK"


def test_no_download_is_started_for_a_case_nobody_needs(tmp_path):
    rig, store = Rig(), fresh(tmp_path, [A, B])
    store.add(SC.CaseResult(cand("m").uid, B.key, "OK", predictions=[prediction().model_dump_json()]))
    run(rig, [A, B], [cand("m")], store)
    assert "shared_start:B.NS" not in rig.log, "B is already done: do not download its news again"


def test_stopping_early_leaves_nothing_running_and_nothing_unread(tmp_path):
    rig, store = Rig(shared_fails={"B.NS"}), fresh(tmp_path, [A, B, C])
    leftover = run(rig, [A, B, C], [cand("m")], store, deadline=-1.0)       # the deadline has already passed
    assert leftover == 1 and rig.log == []


def test_a_budget_that_runs_out_cancels_the_download_started_for_the_case_it_cannot_afford(tmp_path):
    rig, store = Rig(), fresh(tmp_path, [A, B])
    leftover = run(rig, [A, B], [cand("m")], store, budget=PR.Budget({}, 6))       # one case's worth
    assert store.get(cand("m").uid, A.key).status == "OK"
    assert store.get(cand("m").uid, B.key).status == "SKIP"
    assert leftover == 1, "B's download was cancelled, not left to run"


def test_a_slow_download_is_cancelled_at_once_not_waited_for_when_the_run_stops(tmp_path):
    rig, store = Rig(slow={"B.NS"}), fresh(tmp_path, [A, B])
    started = time.monotonic()
    leftover = run(rig, [A, B], [cand("m")], store, budget=PR.Budget({}, 6))          # B is unaffordable: its download is moot
    assert time.monotonic() - started < 5, "the 30 s download must be cancelled, not waited out"
    assert leftover == 1 and "shared_end:B.NS" not in rig.log
