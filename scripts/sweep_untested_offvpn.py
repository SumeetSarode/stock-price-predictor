"""Test every free model that has not been settled yet. Run it OFF the VPN, then hand the report to Code Puppy.

    uv run python scripts/sweep_untested_offvpn.py

What it does, on top of the normal sweep (scripts/free_model_sweep_offvpn.py), resuming from its latest results:
  * calls EVERY model every provider lists: Google, Groq, OpenRouter, Mistral, Cerebras, NVIDIA, Cloudflare,
    Ollama Cloud, Hugging Face, OVH, Kilo, plus the freellmapi / OmniRoute gateway catalogs
  * no model is skipped by its name: embedding/speech/image-looking names get the chat call too, and fail it on
    their own if they cannot chat
  * forgets every earlier failed chat/json check (not found, paid, busy, rejected...), so each one is asked again;
    finished pipeline tests are kept, they are the expensive part
  * stops any earlier sweep still running, because this run replaces it (two at once would share one free quota)
  * 400 calls per provider and no cap on how many models reach the pipeline test
  * retries temporary failures (rate limits, overloads, timeouts) once, after the first full pass
  * never calls DeepSeek (a paid API, blocked by the company on purpose), Cohere (terms, agreed earlier) or
    SiliconFlow (no free chat model on siliconflow.com; removed by you)

Expect several hours. The laptop is kept awake, but keep it plugged in. Interrupted? Run it again: it resumes.

When it finishes, point Code Puppy at the files it prints at the end. They are always also copied to:
  reports/free_model_sweep_latest.md               (the report)
  reports/free_model_sweep_latest_verdicts.jsonl   (one line per model)
"""
from __future__ import annotations

import asyncio
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import free_model_sweep_offvpn as FM
import sweep_data as D
import sweep_env as SE
import sweep_probe as PR

SKIP_PROVIDERS = ("deepseek",)
NEVER_MATCHES = re.compile(r"(?!)")
SWEEP_ARGS = ["--budget", "400", "--max-pipeline", "0", "--retry-passes", "1"]


def stop_earlier_sweeps(run=subprocess.run) -> str:
    """Stop any sweep (and any queued one) still running. -> what was stopped, for the log."""
    found = run(["pgrep", "-fl", "free_model_sweep_offvpn.py"], capture_output=True, text=True).stdout.strip()
    if found:
        run(["pkill", "-f", "free_model_sweep_offvpn.py"], check=False)
    return found


def forget_rows(lines) -> list[str]:
    """Earlier result lines, minus the chat/json checks of every model whose chat check did not pass."""
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:      # a line cut off when the earlier sweep was stopped
            continue
    drop = {r["uid"] for r in rows if r.get("level") == "chat" and r.get("verdict") != "PASS"}
    return [json.dumps(r, ensure_ascii=False) + "\n" for r in rows
            if not (r.get("uid") in drop and r.get("level") in ("chat", "json"))]


def resume_file(reports: Path, stamp: str) -> Path | None:
    latest = PR.latest_results(reports)
    if not latest:
        return None
    out = reports / f"sweep_resume_{stamp}.jsonl"     # not named free_model_sweep_*: it must never become "latest"
    out.write_text("".join(forget_rows(latest.read_text(encoding="utf-8").splitlines())), encoding="utf-8")
    return out


def main() -> int:
    FM.REPORTS.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log = FM.REPORTS / f"sweep_untested_{stamp}.log"
    SE.tee_stdout(log)
    SE.keep_awake()
    stopped = stop_earlier_sweeps()
    print(f"stopped earlier sweep(s):\n{stopped}" if stopped else "no earlier sweep was running")
    for p in SKIP_PROVIDERS:
        D.SPECS.pop(p, None)
    D.NON_CHAT = NEVER_MATCHES      # classify_candidate reads it at call time: no model is skipped by its name
    resume = resume_file(FM.REPORTS, stamp)
    args = SWEEP_ARGS + (["--resume", str(resume.relative_to(FM.REPO))] if resume else ["--fresh"])
    print(f"resuming from: {resume or 'nothing (fresh start)'}\nlog: {log}\n")
    code = asyncio.run(FM.amain(FM.normalize(FM.build_parser().parse_args(args))))
    print(f"\nDONE (exit {code}). Point Code Puppy at:\n  {FM.REPORTS / FM.LATEST_MD}\n"
          f"  {FM.REPORTS / FM.LATEST_VERDICTS}\n  {log}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
