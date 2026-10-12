"""The unattended-run babysitter, proven without launching an app, sleeping, or opening a window."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sweep_env as SE


class Clock:
    """A fake clock: sleeping just moves time forward."""

    def __init__(self) -> None:
        self.t, self.sleeps = 0.0, []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s
        self.sleeps.append(s)


def test_docker_already_running_launches_nothing():
    launched = []
    assert SE.ensure_docker(lambda: True, lambda: launched.append(1) or "x") is True
    assert launched == []


def test_docker_is_launched_once_then_waited_for_until_it_answers():
    clk, polls, launches = Clock(), iter([False, False, False, True]), []
    msgs = []
    ok = SE.ensure_docker(lambda: next(polls), lambda: launches.append(1) or "Rancher Desktop", clk.sleep, 5, msgs.append, clk.now)
    assert ok is True and launches == [1], "launched exactly once"
    assert "launched Rancher Desktop" in msgs[0]
    assert clk.sleeps == [5, 5, 5]


def test_docker_that_never_comes_up_gives_up_after_the_deadline_instead_of_hanging():
    clk = Clock()
    assert SE.ensure_docker(lambda: False, lambda: "", clk.sleep, 1, lambda m: None, clk.now) is False
    assert 60 <= clk.t < 70, "waited the one minute, then stopped"


def test_launch_prefers_rancher_then_docker_and_does_nothing_off_macos():
    calls = []
    run = lambda cmd, **kw: calls.append(cmd)  # noqa: E731
    both = lambda p: p in ("/Applications/Rancher Desktop.app", "/Applications/Docker.app")  # noqa: E731
    only_docker = lambda p: p == "/Applications/Docker.app"  # noqa: E731
    assert SE.launch_docker_app("Darwin", both, run) == "Rancher Desktop"
    assert SE.launch_docker_app("Darwin", only_docker, run) == "Docker"
    assert calls == [["open", "-a", "Rancher Desktop"], ["open", "-a", "Docker"]]
    assert SE.launch_docker_app("Darwin", lambda p: False, run) == ""
    assert SE.launch_docker_app("Linux", both, run) == "" and len(calls) == 2, "never guesses on other systems"


def test_vpn_wait_returns_at_once_when_already_off_and_never_waits_when_told_not_to():
    clk = Clock()
    assert SE.wait_for_off_vpn(lambda: True, 30, clk.sleep, lambda m: None, clk.now) is True and clk.sleeps == []
    assert SE.wait_for_off_vpn(lambda: False, 0, clk.sleep, lambda m: None, clk.now) is False and clk.sleeps == []


def test_vpn_wait_polls_until_you_disconnect_and_then_continues():
    clk, answers, msgs = Clock(), iter([False, False, False, True]), []
    assert SE.wait_for_off_vpn(lambda: next(answers), 30, clk.sleep, msgs.append, clk.now) is True
    assert clk.sleeps == [20, 20, 20]
    assert "VPN looks ON" in msgs[0] and "continuing" in msgs[-1]


def test_vpn_wait_gives_up_after_the_limit():
    clk = Clock()
    assert SE.wait_for_off_vpn(lambda: False, 1, clk.sleep, lambda m: None, clk.now) is False
    assert 60 <= clk.t < 90


def test_keep_awake_holds_the_machine_for_exactly_this_process_or_does_nothing():
    spawned = []
    SE.keep_awake(lambda name: "/usr/bin/caffeinate", lambda cmd: spawned.append(cmd) or "proc", lambda: 4242)
    assert spawned == [["/usr/bin/caffeinate", "-i", "-w", "4242"]], "-w ties it to our pid: it can never outlive us"
    assert SE.keep_awake(lambda name: None, lambda cmd: pytest.fail("must not spawn"), lambda: 1) is None


@pytest.mark.parametrize("system, expected", [
    ("Darwin", ["open", "/r.md"]),
    ("Windows", ["cmd", "/c", "start", "", "/r.md"]),
    ("Linux", ["xdg-open", "/r.md"]),
])
def test_open_file_uses_the_native_opener(system, expected):
    calls = []
    assert SE.open_file(Path("/r.md"), system, lambda cmd, **kw: calls.append(cmd)) is True
    assert calls == [expected]


def test_a_failing_opener_never_fails_the_run():
    def boom(cmd, **kw):
        raise FileNotFoundError("no opener")
    assert SE.open_file(Path("/r.md"), "Linux", boom) is False
    def timeout(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)
    assert SE.open_file(Path("/r.md"), "Darwin", timeout) is False


def test_tee_sends_everything_to_the_screen_and_the_log(tmp_path, capsys):
    log = tmp_path / "run.log"
    tee = SE.Tee(sys.stdout, log)
    tee.write("hello ")
    tee.write("log\n")
    tee.flush()
    tee.close()
    assert log.read_text() == "hello log\n"
    assert capsys.readouterr().out == "hello log\n"


def test_tee_stdout_replaces_stdout_so_plain_prints_are_logged(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stdout", sys.stdout)   # so pytest restores whatever the tee installs
    tee = SE.tee_stdout(tmp_path / "run.log")
    print("an unattended line")
    tee.close()
    assert "an unattended line" in (tmp_path / "run.log").read_text()
