"""Offline tests for scripts/sweep_untested_offvpn.py: no network, no processes touched."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sweep_untested_offvpn as U


def _line(uid: str, level: str, klass: str = "", verdict: str = "FAIL", run: int = 0) -> str:
    return json.dumps({"uid": uid, "level": level, "verdict": verdict, "klass": klass, "run": run})


def test_forget_rows_drops_only_not_found_probe_answers():
    lines = [_line("direct|nvidia|gone", "chat", "not_found"),
             _line("direct|nvidia|gone", "json", "not_found"),
             _line("direct|google|busy", "chat", "rate_limited"),
             _line("direct|mistral|ok", "chat", verdict="PASS"),
             _line("direct|mistral|ok", "pipeline", "timeout", run=1),
             '{"uid": "cut off mid-wri']
    kept = [json.loads(x) for x in U.forget_rows(lines)]
    assert [(r["uid"], r["level"]) for r in kept] == [
        ("direct|google|busy", "chat"), ("direct|mistral|ok", "chat"), ("direct|mistral|ok", "pipeline")]


def test_resume_file_is_never_picked_up_as_the_latest_sweep(tmp_path):
    (tmp_path / "free_model_sweep_20261026T000000Z.jsonl").write_text(_line("a|b|c", "chat", "not_found") + "\n")
    out = U.resume_file(tmp_path, "X")
    assert out is not None and out.read_text() == ""
    assert U.PR.latest_results(tmp_path).name == "free_model_sweep_20261026T000000Z.jsonl"


def test_stop_earlier_sweeps_only_kills_when_one_is_running():
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd[0])
        return type("R", (), {"stdout": "123 python free_model_sweep_offvpn.py" if cmd[0] == "pgrep" else ""})()
    assert U.stop_earlier_sweeps(fake_run).startswith("123")
    assert calls == ["pgrep", "pkill"]
    calls.clear()
    assert U.stop_earlier_sweeps(lambda cmd, **kw: calls.append(cmd[0]) or type("R", (), {"stdout": ""})()) == ""
    assert calls == ["pgrep"]
