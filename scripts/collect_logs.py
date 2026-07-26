"""Bundle recent logs into ONE redacted, shareable file.

Run this right after a prediction fails, then send me the single file
it prints -- no hunting through data/logs/, no leaking API keys.

    python scripts/collect_logs.py

What it does
    * Reads settings.logs_dir (respects your .env, cross-platform).
    * Collects: the FULL errors.log, the tail of predictor.log and
      web.log, plus the most recent rotated *.gz of each (so a fresh
      rotation doesn't hide the failure).
    * REDACTS anything that looks like a secret (Gemini AIza..., Groq
      gsk_..., "api_key=...", Bearer tokens) before writing.
    * Writes diagnostics/logs_share_<timestamp>.txt and prints the path.

Options
    --lines N   How many trailing lines to keep from the big/rolling
                logs (predictor.log, web.log). Default 600.
    --errors-lines N  Trailing lines from errors.log. Default: ALL.
"""
from __future__ import annotations

import argparse
import gzip
import re
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

# Make `price_predictor` importable when run as a bare script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from price_predictor.config.settings import settings  # noqa: E402

# ── Secret redaction ────────────────────────────────────────────────
# Order matters: specific provider-key shapes first, then generic
# key=value assignments, then bearer tokens.
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Gemini / Google API keys: literally start with "AIza".
    (re.compile(r"AIza[0-9A-Za-z_\-]{10,}"), "AIza***REDACTED***"),
    # Groq keys: start with "gsk_".
    (re.compile(r"gsk_[0-9A-Za-z]{10,}"), "gsk_***REDACTED***"),
    # OpenAI-style: "sk-...".
    (re.compile(r"sk-[0-9A-Za-z]{10,}"), "sk-***REDACTED***"),
    # Bearer tokens -- MUST run before the generic key=value rule below,
    # otherwise that rule redacts the word "Bearer" and leaves the token.
    (re.compile(r"(?i)(bearer\s+)([A-Za-z0-9._\-]{10,})"), r"\1***REDACTED***"),
    # Generic key=value / key: value (api_key, api-key, apikey, token, secret).
    (
        re.compile(
            r"(?i)((?:api[_-]?key|apikey|token|secret|authorization)"
            r"\s*[:=]\s*['\"]?)([^\s'\"]{6,})"
        ),
        r"\1***REDACTED***",
    ),
)


def _redact(text: str) -> str:
    """Strip anything that smells like a credential."""
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


def _tail(path: Path, n: int | None) -> str:
    """Return the last n lines of a (plain) log file, or all if n is None."""
    if not path.exists():
        return f"[not found: {path}]"
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        if n is None:
            return fh.read()
        return "".join(deque(fh, maxlen=n))


def _tail_gz(path: Path, n: int) -> str:
    """Return the last n lines of a gzipped rotated log."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        return "".join(deque(fh, maxlen=n))


def _latest_gz(logs_dir: Path, stem: str) -> Path | None:
    """Most recently modified rotated <stem>*.gz, if any."""
    candidates = sorted(
        logs_dir.glob(f"{stem}*.gz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _section(title: str, body: str) -> str:
    bar = "=" * 70
    return f"\n{bar}\n{title}\n{bar}\n{body.rstrip()}\n"


def collect(lines: int, errors_lines: int | None) -> Path:
    logs_dir: Path = settings.logs_dir
    out_dir = Path(__file__).resolve().parent.parent / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_path = out_dir / f"logs_share_{ts}.txt"

    parts: list[str] = []
    parts.append(
        _section(
            "COLLECTED DIAGNOSTICS",
            f"generated_utc : {datetime.now(timezone.utc).isoformat()}\n"
            f"logs_dir      : {logs_dir}\n"
            f"tail_lines    : {lines}\n"
            f"errors_lines  : {'ALL' if errors_lines is None else errors_lines}\n"
            f"note          : API keys/tokens are redacted before writing.",
        )
    )

    # errors.log — the money file: full backtraces of ERROR+ events.
    parts.append(
        _section(
            "errors.log (ERROR+ with backtraces)",
            _tail(logs_dir / "errors.log", errors_lines),
        )
    )
    gz = _latest_gz(logs_dir, "errors")
    if gz is not None:
        parts.append(
            _section(f"errors.log — latest rotated ({gz.name})", _tail_gz(gz, lines))
        )

    # predictor.log — full DEBUG stream, tailed.
    parts.append(
        _section(
            f"predictor.log (last {lines} lines)",
            _tail(logs_dir / "predictor.log", lines),
        )
    )

    # web.log — uvicorn/web stdout, if present.
    if (logs_dir / "web.log").exists():
        parts.append(
            _section(f"web.log (last {lines} lines)", _tail(logs_dir / "web.log", lines))
        )

    # A focused grep so the cause jumps out even in a big file.
    keys = re.compile(
        r"(?i)(structural|contextwindow|authentication|allmodelsexhausted|"
        r"token limit|api.?key|invalid|crumb|error|traceback)"
    )
    interesting: list[str] = []
    for name in ("errors.log", "predictor.log", "web.log"):
        fp = logs_dir / name
        if not fp.exists():
            continue
        with fp.open("r", encoding="utf-8", errors="replace") as fh:
            hits = [ln.rstrip() for ln in fh if keys.search(ln)]
        if hits:
            interesting.append(f"--- {name} ---")
            interesting.extend(hits[-lines:])
    parts.append(
        _section("QUICK GREP: error/auth/context/token lines", "\n".join(interesting))
    )

    redacted = _redact("".join(parts))
    out_path.write_text(redacted, encoding="utf-8")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Bundle recent logs into one redacted file.")
    ap.add_argument("--lines", type=int, default=600, help="tail size for rolling logs")
    ap.add_argument(
        "--errors-lines",
        type=int,
        default=None,
        help="tail size for errors.log (default: all)",
    )
    args = ap.parse_args()

    out = collect(args.lines, args.errors_lines)
    size_kb = out.stat().st_size / 1024
    print(f"\n Wrote {out}  ({size_kb:.1f} KB)")
    print("   Secrets redacted. Send me THIS file.\n")


if __name__ == "__main__":
    main()
