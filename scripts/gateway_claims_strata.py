"""Strata checks: can THIS machine run it, are its downloads real, and (if a server is up) does it work?

Strata is an inference engine, not a hosted gateway: it runs a 125B-parameter MoE model on the
user's own NVIDIA/AMD GPU. So the checks are different in kind:

  ST1  does THIS machine meet the README's requirements?   (hardware report, no network)
  ST2  are the model downloads real, open, and the claimed size?  (HuggingFace + GitHub metadata)
  ST3  if a Strata server is reachable (this PC, or another on the LAN), does its API work?

ST4/ST5 (structured output, our pipeline) need the HTTP helpers that live in the main script, so
`strata_checks` returns a context for the caller to run them: that keeps this module free of an
import cycle. Everything here is read-only; nothing is installed or downloaded.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from gateway_claims_data import CONFIRMED, INCONCLUSIVE, NOT_CONFIRMED

HF_GGUF, HF_BASE = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "Qwen/Qwen3.8-Flash-Next"
ZIP_URL = "https://github.com/Niko1221/Strata/archive/refs/heads/main.zip"
MIN_VRAM_GB, MIN_RAM_GB, MIN_DISK_GB = 12, 32, 80        # README "What you need"
SUPPORTED_OS, SUPPORTED_VENDORS = {"Windows", "Linux"}, {"nvidia", "amd"}
ONE_SIZE_GIB = (30, 95)                                   # README: "about 70-80 GB" for one size
MAC_NOTE = ("Strata maintainer on Apple Silicon, issue #518 (2026-10-03): \"isn't planned\"; the community "
            "Metal PR #1449 was closed UNMERGED (2026-10-10) and still demanded 64 GB+. As of 2026-10-11.")


def _run(*cmd: str, timeout: int = 15) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return p.stdout if p.returncode == 0 else ""


def vendor_of(name: str) -> str:
    low = name.lower()
    for needle, vendor in (("nvidia", "nvidia"), ("geforce", "nvidia"), ("rtx", "nvidia"), ("radeon", "amd"),
                           ("amd", "amd"), ("apple", "apple"), ("intel", "intel")):
        if needle in low:
            return vendor
    return "unknown"


def detect_gpus(system: str, run=_run) -> list[dict]:
    gpus = []
    for line in run("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits").splitlines():
        name, _, mib = line.rpartition(",")
        if name.strip() and mib.strip().isdigit():
            gpus.append({"name": name.strip(), "vram_gb": round(int(mib) / 1024, 1), "vendor": "nvidia"})
    if gpus:
        return gpus
    if system == "Darwin":
        names = re.findall(r"Chipset Model:\s*(.+)", run("system_profiler", "SPDisplaysDataType"))
    elif system == "Windows":
        out = run("powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name")
        names = [ln.strip() for ln in out.splitlines() if ln.strip()]
    else:
        names = [ln.split(": ", 1)[-1] for ln in run("lspci").splitlines() if re.search(r"VGA|3D|Display", ln)]
    return [{"name": n.strip(), "vram_gb": None, "vendor": vendor_of(n)} for n in names]


def ram_gb(system: str, run=_run) -> int | None:
    if system == "Windows":
        out = run("powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory").strip()
        return round(int(out) / 2**30) if out.isdigit() else None
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)
    except (ValueError, OSError, AttributeError):
        return None


def hardware_report(run=_run) -> dict:
    system = platform.system()
    return {"system": system, "machine": platform.machine(), "ram_gb": ram_gb(system, run),
            "free_disk_gb": round(shutil.disk_usage(Path.home()).free / 2**30),
            "gpus": detect_gpus(system, run)}


def evaluate(hw: dict) -> tuple[str, str]:
    """Verdict on ST1 for the machine described by `hw`. Unknowns are INCONCLUSIVE, never a fail."""
    problems, unknown = [], []
    if hw["system"] not in SUPPORTED_OS:
        problems.append(f"OS is {hw['system']} (needs Windows or Linux)")
    usable = [g for g in hw["gpus"] if g["vendor"] in SUPPORTED_VENDORS]
    if not usable:
        found = ", ".join(f"{g['name']} ({g['vendor']})" for g in hw["gpus"]) or "none detected"
        problems.append(f"no NVIDIA/AMD GPU (found: {found}); integrated GPUs are not supported")
    else:
        best = max(usable, key=lambda g: g["vram_gb"] or 0)
        if best["vram_gb"] is None:
            unknown.append(f"VRAM of {best['name']} not detected (needs >= {MIN_VRAM_GB} GB)")
        elif best["vram_gb"] < MIN_VRAM_GB:
            problems.append(f"{best['name']} has {best['vram_gb']} GB VRAM (needs >= {MIN_VRAM_GB}; 8 GB 'runs, slowly')")
    for label, value, need in (("RAM", hw["ram_gb"], MIN_RAM_GB), ("free disk", hw["free_disk_gb"], MIN_DISK_GB)):
        if value is None:
            unknown.append(f"{label} not detected")
        elif value < need:
            problems.append(f"{label} {value} GB (needs >= {need})")
    summary = f"{hw['system']}/{hw['machine']}, RAM {hw['ram_gb']} GB, free disk {hw['free_disk_gb']} GB, GPUs {hw['gpus']}"
    note = f" {MAC_NOTE}" if hw["system"] == "Darwin" else ""
    if problems:
        return NOT_CONFIRMED, f"THIS machine fails: {'; '.join(problems)}. [{summary}]{note}"
    if unknown:
        return INCONCLUSIVE, f"could not verify: {'; '.join(unknown)}. [{summary}]"
    return CONFIRMED, f"THIS machine meets the README requirements. [{summary}]"


def source_check(L, c, call) -> None:
    """ST2: metadata only. Reports the size of every quantisation and the licence of both repos."""
    info = {}
    for repo in (HF_GGUF, HF_BASE):
        r = call(c, "GET", f"https://huggingface.co/api/models/{repo}?blobs=true")
        j = r.json() if r.status == 200 else None
        info[repo] = j if isinstance(j, dict) and "siblings" in j else None
    gguf = info[HF_GGUF]
    if not gguf:
        L.add("ST2", "-", INCONCLUSIVE, "HuggingFace metadata unreachable (VPN? proxy?); sizes and licences unchecked")
        return
    quants: dict[str, float] = {}
    for s in gguf["siblings"]:
        if "/" in s["rfilename"] and s.get("size"):
            quants[s["rfilename"].split("/")[0]] = quants.get(s["rfilename"].split("/")[0], 0) + s["size"] / 2**30
    lic = {repo: ((j or {}).get("cardData") or {}).get("license") for repo, j in info.items()}
    zip_r = call(c, "HEAD", ZIP_URL, follow_redirects=True)
    one_size = [q for q, g in quants.items() if ONE_SIZE_GIB[0] <= g <= ONE_SIZE_GIB[1]]
    ok = bool(one_size) and zip_r.status == 200 and not gguf.get("gated")
    L.add("ST2", "-", CONFIRMED if ok else NOT_CONFIRMED,
          f"quant sizes GiB {{{', '.join(f'{q}: {g:.0f}' for q, g in sorted(quants.items(), key=lambda kv: kv[1]))}}}; "
          f"sizes in {ONE_SIZE_GIB} GiB: {one_size}; gated={gguf.get('gated')}; licences {lic} "
          f"('other' = a custom licence: READ it before shipping anything built on it); "
          f"installer zip HEAD -> HTTP {zip_r.status}; downloads={gguf.get('downloads')}")


def strata_checks(L, c, call, chat, url: str, key: str | None) -> dict | None:
    """ST1-ST3. Returns a pipeline context only if a Strata server answered."""
    L.add("ST1", "this-machine", *evaluate(hardware_report()))
    source_check(L, c, call)

    auth = {"Authorization": f"Bearer {key}"} if key else {}
    status = call(c, "GET", f"{url}/v1/status", headers=auth)
    if status.status == 0:
        L.add("ST3", "server", INCONCLUSIVE,
              f"no Strata server at {url} ({status.error[:120]}). To test one on a qualifying Windows/Linux PC: "
              "start it with `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`, then re-run with "
              "`--strata-url http://<pc-ip>:8080 --strata-key <secret>`")
        return None
    rows = (call(c, "GET", f"{url}/v1/models", headers=auth).json() or {}).get("data") or []
    model = str(rows[0]["id"]) if rows and isinstance(rows[0], dict) and "id" in rows[0] else "any-model-name"
    r, text = chat(c, f"{url}/v1", key or "any-api-key", model, "Reply with exactly the word: ok")
    tokens = ((r.json() or {}).get("usage") or {}).get("completion_tokens") if r.status == 200 else None
    speed = f"~{tokens / r.seconds:.0f} tok/s including prompt processing" if tokens and r.seconds else "speed n/a"
    L.add("ST3", "server", CONFIRMED if r.status == 200 and text.strip() else NOT_CONFIRMED,
          f"model={model}; /v1/status: {status.evidence()[:260]}; chat: {r.evidence(text)[:200]}; {speed}")
    return {"base": f"{url}/v1", "key": key or "any-api-key", "models": [model], "timeout_s": 600}
