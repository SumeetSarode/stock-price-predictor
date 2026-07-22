"""List the Gemini models THIS account/key can actually call.

Run when you hit a `NotFoundError: model ... is no longer available` -- it
tells you exactly which model strings your GEMINI_API_KEY is allowed to use,
so you can pick a current one for CHAIN_AGENTIC.

    python scripts/list_gemini_models.py

Reads GEMINI_API_KEY from .env (or the environment). Read-only; no writes.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path


def _load_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^GEMINI_API_KEY=(.+)$", line.strip())
            if m:
                return m.group(1).strip()
    return None


def main() -> None:
    key = _load_key()
    if not key or key.startswith("your_"):
        print("No real GEMINI_API_KEY found (checked env + .env).")
        return

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        data = json.load(urllib.request.urlopen(url, timeout=15))
    except Exception as exc:  # network/auth issues -> just report
        print(f"ListModels call failed: {exc}")
        return

    usable = [
        m["name"].replace("models/", "")
        for m in data.get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]
    flash = [n for n in usable if "flash" in n]

    print("Flash models your key can call:")
    for n in sorted(flash):
        print(f"  gemini/{n}")
    print("\nAll usable models:", len(usable))
    print("\nTip: prefer an alias like 'gemini/gemini-flash-latest' in CHAIN_AGENTIC")
    print("so a retired version number never 404s you again.")


if __name__ == "__main__":
    main()
