"""A run stopped hard (closed terminal, sleep, kill) can leave a damaged last line. Resuming must survive it."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import scorecard_core as SC
import sweep_probe as PR


def result(model: str, detail: str = "ok") -> PR.Result:
    return PR.Result(f"direct|groq|{model}", "json", "PASS", detail=detail)


def write_good(path: Path, n: int = 3, detail: str = "ok") -> list[PR.Result]:
    rows = [result(f"m{i}", detail) for i in range(n)]
    path.write_text("".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return rows


def test_read_jsonl_skips_only_the_damaged_lines(tmp_path):
    f = tmp_path / "a.jsonl"
    f.write_text('{"a": 1}\n\n{"a": 2\nnot json at all\n[1, 2]\n"just a string"\n{"a": 3}\n', encoding="utf-8")
    assert list(PR.read_jsonl(f)) == [{"a": 1}, {"a": 3}], "good dict rows only; cut-off, junk and non-objects skipped"


def test_a_line_cut_off_mid_write_does_not_crash_the_sweep_resume(tmp_path):
    old = tmp_path / "old.jsonl"
    write_good(old, 3)
    old.write_bytes(old.read_bytes()[:-40])                       # the hard stop: the newest row is half-written
    store = PR.Store(tmp_path / "new.jsonl", (old,))
    assert len(store.rows) == 2, "the two complete rows survive; only the unfinished one is lost"


def test_a_cut_in_the_middle_of_a_multibyte_character_is_survived_too(tmp_path):
    old = tmp_path / "old.jsonl"
    good = json.dumps(asdict(result("m0")), ensure_ascii=False).encode("utf-8") + b"\n"
    cut = '{"uid": "direct|groq|m1", "detail": "日本'.encode()[:-1]      # ends INSIDE a 3-byte character
    old.write_bytes(good + cut)
    with pytest.raises(UnicodeDecodeError):
        (good + cut).decode("utf-8")                                          # proves strict decoding really breaks
    assert len(PR.Store(tmp_path / "new.jsonl", (old,)).rows) == 1, "the complete row survives"


def test_the_new_file_is_clean_so_damage_does_not_travel_to_the_next_run(tmp_path):
    old = tmp_path / "old.jsonl"
    write_good(old, 3)
    old.write_bytes(old.read_bytes()[:-40])
    new = tmp_path / "new.jsonl"
    PR.Store(new, (old,))
    lines = new.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and all(json.loads(x) for x in lines), "every carried-forward line parses"


def test_an_empty_or_blank_log_is_just_nothing_to_resume(tmp_path):
    old = tmp_path / "old.jsonl"
    old.write_text("\n\n", encoding="utf-8")
    assert PR.Store(tmp_path / "new.jsonl", (old,)).rows == {}


def test_a_damaged_middle_line_loses_only_that_row(tmp_path):
    old = tmp_path / "old.jsonl"
    rows = write_good(old, 3)
    lines = old.read_text(encoding="utf-8").splitlines()
    old.write_text("\n".join([lines[0], lines[1][:30], lines[2]]) + "\n", encoding="utf-8")
    assert {r.uid for r in PR.Store(tmp_path / "new.jsonl", (old,)).rows.values()} == {rows[0].uid, rows[2].uid}


def test_the_scorecard_store_survives_a_cut_off_line_and_keeps_its_case_grid(tmp_path):
    old = SC.CaseStore(tmp_path / "old.jsonl")
    old.set_cases([SC.Case("A", date(2026, 7, 1)), SC.Case("B", date(2026, 7, 2))])
    for i in range(3):
        old.add(SC.CaseResult(f"u{i}", "A@2026-07-01", "OK"))
    raw = (tmp_path / "old.jsonl").read_bytes()
    (tmp_path / "old.jsonl").write_bytes(raw[:-25])               # the newest result is half-written
    again = SC.CaseStore(tmp_path / "new.jsonl", (tmp_path / "old.jsonl",))
    assert [c.key for c in again.cases or []] == ["A@2026-07-01", "B@2026-07-02"], "the pinned case grid survives"
    assert again.get("u0", "A@2026-07-01") and again.get("u1", "A@2026-07-01"), "earlier results survive"
    assert again.get("u2", "A@2026-07-01") is None, "only the unfinished one is lost, so it simply runs again"


def test_a_damaged_header_row_does_not_stop_results_loading(tmp_path):
    old = tmp_path / "old.jsonl"
    good = json.dumps(asdict(SC.CaseResult("u0", "A@2026-07-01", "OK")))
    old.write_text('{"type": "cases", "cases": ["A@2026-07\n' + good + "\n", encoding="utf-8")
    store = SC.CaseStore(tmp_path / "new.jsonl", (old,))
    assert store.get("u0", "A@2026-07-01") is not None and store.cases is None
