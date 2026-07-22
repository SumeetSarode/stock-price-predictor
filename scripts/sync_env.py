"""Merge new settings from `.env.example` into the user's `.env` safely.

WHY THIS EXISTS
===============
The Windows launcher does `git reset --hard origin/release` every run, which
refreshes the tracked `.env.example`. But the user's real `.env` is gitignored
and never touched -- so when we add a NEW setting to the template (e.g.
`OLLAMA_API_BASE`), an already-deployed laptop would never learn about it.

This script bridges that gap with two rules:

    1. ADD-MISSING-ONLY (default). We only ever *append* keys that the user
    doesn't already have. We NEVER modify or remove an existing line. Their
    API keys, their `PRICE_CHAIN=yfinance` geo tweak, their `WEB_PORT` -- all
    left exactly as they are.

    2. APP-MANAGED KEYS (the exception). A tiny allowlist of keys the app
    *owns* -- the model chains (`CHAIN_AGENTIC`, `PAID_AGENTIC`). For these,
    if the template's value differs from the user's, we DO update the user's
    line to the template value. This is how a model/chain change (e.g. adding
    an Ollama fallback tail) reaches an already-deployed laptop. Secrets and
    user/geo tweaks are deliberately NOT on this list.

If you change the *default value* of a NON-managed existing key in
`.env.example`, an existing user won't get it (by design -- we won't clobber
their choice). The escape hatch: they can delete that line from `.env` and
the next launch re-adds it with the template's fresh default.

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

# Keys the APP owns: when present in BOTH files, the user's value is synced to
# the template's. Keep this list TINY -- never add secrets or user/geo tweaks
# (API keys, PRICE_CHAIN, ports); those stay user-owned and untouchable.
MANAGED_KEYS: frozenset[str] = frozenset({"CHAIN_AGENTIC", "PAID_AGENTIC"})


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


def template_values(example_text: str) -> dict[str, str]:
    """Map each active KEY to its value string in the template."""
    values: dict[str, str] = {}
    for line in example_text.splitlines():
        m = _ASSIGN_RE.match(line)
        if m:
            values[m.group(1)] = line[m.end():]
    return values


def apply_managed_updates(env_text: str, example_text: str) -> tuple[str, list[str]]:
    """Sync APP-MANAGED keys in `env_text` to the template's value.

    Only rewrites a line when BOTH: the key is in MANAGED_KEYS, AND its value
    differs from the template's. Everything else (secrets, PRICE_CHAIN, all
    non-managed keys) is passed through byte-for-byte. Returns
    (new_text, updated_key_names).
    """
    tmpl = template_values(example_text)
    updated: list[str] = []
    out_lines: list[str] = []
    for line in env_text.splitlines():
        m = _ASSIGN_RE.match(line)
        if m is not None:
            key = m.group(1)
            if key in MANAGED_KEYS and key in tmpl:
                new_line = f"{key}={tmpl[key]}"
                if new_line != line:
                    updated.append(key)
                    out_lines.append(new_line)
                    continue
        out_lines.append(line)

    new_text = "\n".join(out_lines)
    if env_text.endswith("\n"):
        new_text += "\n"
    return new_text, updated


def merge(example_text: str, env_text: str) -> tuple[str, list[str], list[str]]:
    """Return (new_env_text, added_key_names, updated_key_names).

    Two phases: (1) sync app-managed keys in place, (2) append missing keys.
    Pure function -- trivial to unit-test.
    """
    env_text, updated = apply_managed_updates(env_text, example_text)

    blocks = missing_key_blocks(example_text, env_text)
    if not blocks:
        return env_text, [], updated

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
    return new_text, added, updated


def sync_env(env_path: Path, example_path: Path) -> dict[str, list[str]]:
    """Sync `example_path` into `env_path`. Non-fatal.

    Returns {"added": [...], "updated": [...]} -- new keys appended and
    app-managed keys whose value was synced. Empty lists on no-op / error.
    """
    try:
        if not example_path.exists():
            return {"added": [], "updated": []}  # no template -> nothing to do

        example_text = example_path.read_text(encoding="utf-8")

        # First run: no .env yet -> seed it from the template verbatim.
        if not env_path.exists():
            env_path.write_text(example_text, encoding="utf-8")
            print(f"[env] Created {env_path.name} from template.")
            return {"added": sorted(active_keys(example_text)), "updated": []}

        env_text = env_path.read_text(encoding="utf-8")
        new_text, added, updated = merge(example_text, env_text)
        if added or updated:
            env_path.write_text(new_text, encoding="utf-8")
            if added:
                print(f"[env] Added {len(added)} new setting(s): {', '.join(added)}")
            if updated:
                print(f"[env] Updated app-managed setting(s): {', '.join(updated)}")
            print("[env] Your API keys and personal settings were left intact.")
        else:
            print("[env] Up to date - nothing to change.")
        return {"added": added, "updated": updated}
    except Exception as exc:  # never let a config merge break startup
        print(f"[env] Skipped merge ({exc}). Using your .env as-is.")
        return {"added": [], "updated": []}


def main() -> None:
    """CLI entry: `python scripts/sync_env.py [ENV_PATH] [EXAMPLE_PATH]`."""
    env_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".env")
    example_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".env.example")
    sync_env(env_path, example_path)


if __name__ == "__main__":
    main()
