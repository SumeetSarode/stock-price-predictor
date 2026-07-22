"""Merge new settings from `.env.example` into the user's `.env` safely.

WHY THIS EXISTS
===============
The Windows launcher does `git reset --hard origin/release` every run, which
refreshes the tracked `.env.example`. But the user's real `.env` is gitignored
and never touched -- so when we add a NEW setting to the template (e.g.
`OLLAMA_API_BASE`), an already-deployed laptop would never learn about it.

This script bridges that gap with ONE hard rule:

    ADD-MISSING-ONLY. We only ever *append* keys that the user doesn't
    already have. We NEVER modify or remove an existing line. Their API
    keys, their `PRICE_CHAIN=yfinance` geo tweak, their `WEB_PORT` -- all
    left exactly as they are.

If you change the *default value* of an existing key in `.env.example`, an
existing user won't get it (by design -- we won't clobber their choice). The
escape hatch: they can delete that line from `.env` and the next launch will
re-add it with the template's fresh default.

FIRST RUN: if `.env` doesn't exist yet, we copy the template verbatim.

NON-FATAL: any error is swallowed with a message -- a config merge must never
stop the app from starting.
"""
from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

# An "active" assignment: KEY=value at column 0 (not commented, not indented).
_ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=")


def active_keys(text: str) -> set[str]:
    """Return the set of uncommented KEY names defined in `text`.

    Commented lines (`#KEY=...`) and indented lines are NOT active keys --
    they're documentation/examples, so we don't count or sync them.
    """
    keys: set[str] = set()
    for line in text.splitlines():
        m = _ASSIGN_RE.match(line)
        if m:
            keys.add(m.group(1))
    return keys


def missing_key_blocks(example_text: str, env_text: str) -> list[str]:
    """Build the text blocks to append for keys present in the template but
    NOT active in the user's env.

    Each block carries the contiguous comment/blank lines that immediately
    precede the key in the template, so the appended setting keeps its
    explanatory context.
    """
    have = active_keys(env_text)
    blocks: list[str] = []
    buffer: list[str] = []  # pending comment/blank lines before a key

    for line in example_text.splitlines():
        m = _ASSIGN_RE.match(line)
        if m is None:
            # comment or blank -> accumulate as context for the next key
            buffer.append(line)
            continue
        # It's an active assignment line.
        key = m.group(1)
        if key not in have:
            block_lines = [*buffer, line]
            # Trim leading blank lines in the captured context for tidiness.
            while block_lines and block_lines[0].strip() == "":
                block_lines.pop(0)
            blocks.append("\n".join(block_lines))
        buffer = []  # any assignment consumes the pending comment buffer

    return blocks


def merge(example_text: str, env_text: str) -> tuple[str, list[str]]:
    """Return (new_env_text, added_key_names).

    Pure function -- trivial to unit-test. If nothing is missing, returns the
    original env text unchanged and an empty added list.
    """
    blocks = missing_key_blocks(example_text, env_text)
    if not blocks:
        return env_text, []

    added = [
        m.group(1)
        for block in blocks
        for line in block.splitlines()
        if (m := _ASSIGN_RE.match(line))
    ]

    header = (
        "# ---------------------------------------------------------------\n"
        f"# Added by the launcher on {date.today().isoformat()} - new settings\n"
        "# from .env.example. Your existing settings above were left untouched.\n"
        "# Delete any line below to have the launcher restore its default.\n"
        "# ---------------------------------------------------------------"
    )

    base = env_text.rstrip("\n")
    new_text = base + "\n\n" + header + "\n" + "\n\n".join(blocks) + "\n"
    return new_text, added


def sync_env(env_path: Path, example_path: Path) -> list[str]:
    """Sync missing keys from `example_path` into `env_path`. Non-fatal.

    Returns the list of key names that were added (empty if none / on error).
    """
    try:
        if not example_path.exists():
            return []  # no template -> nothing to do

        example_text = example_path.read_text(encoding="utf-8")

        # First run: no .env yet -> seed it from the template verbatim.
        if not env_path.exists():
            env_path.write_text(example_text, encoding="utf-8")
            print(f"[env] Created {env_path.name} from template.")
            return sorted(active_keys(example_text))

        env_text = env_path.read_text(encoding="utf-8")
        new_text, added = merge(example_text, env_text)
        if added:
            env_path.write_text(new_text, encoding="utf-8")
            print(f"[env] Added {len(added)} new setting(s): {', '.join(added)}")
            print("[env] Your existing settings (API keys etc.) were left intact.")
        else:
            print("[env] Up to date - no new settings to add.")
        return added
    except Exception as exc:  # never let a config merge break startup
        print(f"[env] Skipped merge ({exc}). Using your .env as-is.")
        return []


def main() -> None:
    """CLI entry: `python scripts/sync_env.py [ENV_PATH] [EXAMPLE_PATH]`."""
    env_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".env")
    example_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".env.example")
    sync_env(env_path, example_path)


if __name__ == "__main__":
    main()
