"""Ensure the offline Ollama fallback is actually USABLE at launch.

WHY THIS EXISTS
===============
The chain can end in a local Ollama model as an offline last resort. But a
model that isn't pulled (or a server that isn't running) is dead weight -- the
fallback silently fails exactly when you need it. The in-app guard only
*warns*; this script *acts*:

    1. Is Ollama in the chain at all? No  -> do nothing.
    2. Is the `ollama` binary installed?  No -> try winget (Windows),
       else print install instructions. Non-fatal.
    3. Is the server reachable?           No -> start `ollama serve` detached
       and wait briefly for it to come up.
    4. Is the configured model pulled?     No -> `ollama pull <model>`
       (streams progress; one-time ~GB download).

NON-FATAL by design: any failure prints a message and returns -- a fallback
setup step must NEVER stop the app from starting on its primary providers.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import time

from price_predictor.config.settings import settings
from price_predictor.llm.ollama_guard import (
    _is_pulled,
    _pulled_models,
    ollama_tags_in_chain,
)


def ollama_installed() -> bool:
    """True if the `ollama` CLI is on PATH."""
    return shutil.which("ollama") is not None


def models_to_pull(tags: list[str], pulled: set[str]) -> list[str]:
    """Return the subset of `tags` not satisfied by the `pulled` set."""
    return [t for t in tags if not _is_pulled(t, pulled)]


def _try_winget_install() -> bool:
    """Best-effort silent install of Ollama on Windows via winget."""
    if platform.system() != "Windows" or shutil.which("winget") is None:
        return False
    print("[ollama] Installing Ollama via winget (one-time)...")
    try:
        subprocess.run(
            ["winget", "install", "--id", "Ollama.Ollama", "-e", "--silent",
             "--accept-package-agreements", "--accept-source-agreements"],
            check=False,
        )
    except Exception as exc:  # never fatal
        print(f"[ollama] winget install failed ({exc}).")
        return False
    return ollama_installed()


def _start_server() -> None:
    """Launch `ollama serve` detached so it outlives this script."""
    print("[ollama] Starting the Ollama server...")
    kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if platform.system() == "Windows":
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP -> survives console close.
        kwargs["creationflags"] = 0x00000008 | 0x00000200
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen(["ollama", "serve"], **kwargs)
    except Exception as exc:  # never fatal
        print(f"[ollama] Could not start server ({exc}).")


def _wait_for_server(base_url: str, timeout_s: float = 20.0) -> set[str] | None:
    """Poll the server until reachable or timeout. Returns pulled set or None."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        pulled = _pulled_models(base_url)
        if pulled is not None:
            return pulled
        time.sleep(1.0)
    return None


def _pull(tag: str) -> None:
    """Run `ollama pull <tag>`, streaming progress to the console."""
    print(f"[ollama] Pulling '{tag}' (one-time download, may take a while)...")
    try:
        subprocess.run(["ollama", "pull", tag], check=False)
    except Exception as exc:  # never fatal
        print(f"[ollama] Pull of '{tag}' failed ({exc}).")


def ensure() -> None:
    """Make the offline fallback usable. Non-fatal end to end."""
    try:
        tags = ollama_tags_in_chain(settings.effective_chain("agentic"))
    except Exception:
        return
    if not tags:
        return  # no local fallback configured -> nothing to ensure

    base_url = settings.ollama_api_base

    # 1. Installed?
    if not ollama_installed() and not _try_winget_install():
        print(
            "[ollama] Ollama isn't installed, so the offline fallback "
            f"{tags} can't run. Install it from https://ollama.com/download "
            "to enable predictions when the hosted AI is rate-limited."
        )
        return

    # 2. Server reachable? Start it if not.
    pulled = _pulled_models(base_url)
    if pulled is None:
        _start_server()
        pulled = _wait_for_server(base_url)
        if pulled is None:
            print(
                f"[ollama] Server not reachable at {base_url} after starting. "
                "The app will still run on its hosted providers."
            )
            return

    # 3. Pull any missing models.
    missing = models_to_pull(tags, pulled)
    if not missing:
        print(f"[ollama] Offline fallback ready: {tags}")
        return
    for tag in missing:
        _pull(tag)

    # 4. Verify.
    pulled = _pulled_models(base_url) or set()
    still_missing = models_to_pull(tags, pulled)
    if still_missing:
        print(f"[ollama] Still missing after pull: {still_missing}. "
              "The app runs on hosted providers; retry later for the fallback.")
    else:
        print(f"[ollama] Offline fallback ready: {tags}")


def main() -> None:
    ensure()


if __name__ == "__main__":
    main()
