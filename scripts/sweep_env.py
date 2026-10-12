"""Babysitting for an unattended run: Docker, the VPN, sleep, logging, and showing you the result.

Every function takes its side effects as arguments (launcher, clock, sleep, opener), so the tests can prove the
behaviour without launching an app, sleeping, or opening a window.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path


def launch_docker_app(system: str | None = None, exists=os.path.exists, run=subprocess.run) -> str:
    """Best-effort start of a container runtime. Returns what was launched ('' if nothing could be)."""
    system = system or platform.system()
    if system == "Darwin":
        for app in ("Rancher Desktop", "Docker"):
            if exists(f"/Applications/{app}.app"):
                run(["open", "-a", app], capture_output=True, timeout=30)
                return app
    return ""


def ensure_docker(ready: Callable[[], bool], launch: Callable[[], str], sleep=time.sleep, minutes: float = 5,
                  say=print, clock=time.monotonic) -> bool:
    """True once Docker answers. Starts it if it is not running, then waits up to `minutes` for the daemon."""
    if ready():
        return True
    launched = launch()
    say(f"Docker is not running; {'launched ' + launched if launched else 'could not launch anything'}; "
        f"waiting up to {minutes:g} min for the daemon...")
    deadline = clock() + minutes * 60
    while clock() < deadline:
        sleep(5)
        if ready():
            return True
    return False


def wait_for_off_vpn(reachable: Callable[[], bool], minutes: float, sleep=time.sleep, say=print,
                     clock=time.monotonic) -> bool:
    """True once the canary host answers (the VPN is off). Waits instead of aborting, up to `minutes`."""
    if reachable():
        return True
    if minutes <= 0:
        return False
    say(f"The VPN looks ON (the canary host is unreachable). Disconnect it; waiting up to {minutes:g} min...")
    deadline = clock() + minutes * 60
    while clock() < deadline:
        sleep(20)
        if reachable():
            say("Off the VPN: continuing.")
            return True
    return False


def keep_awake(which=shutil.which, popen=subprocess.Popen, pid=os.getpid):
    """On macOS, hold off sleep for as long as this process lives (a long run must not die with the lid)."""
    exe = which("caffeinate")
    return popen([exe, "-i", "-w", str(pid())]) if exe else None


def open_file(path: Path, system: str | None = None, run=subprocess.run) -> bool:
    """Show the finished report. Best effort: a failure here must never fail the run."""
    system = system or platform.system()
    cmd = {"Darwin": ["open"], "Windows": ["cmd", "/c", "start", ""]}.get(system, ["xdg-open"])
    try:
        run([*cmd, str(path)], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


class Tee:
    """Everything printed also goes to a log file, so a run nobody watched can still be read afterwards."""

    def __init__(self, stream, path: Path) -> None:
        self.stream, self.fh = stream, open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.fh.write(text)
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def tee_stdout(path: Path) -> Tee:
    tee = Tee(sys.stdout, path)
    sys.stdout = tee
    return tee
