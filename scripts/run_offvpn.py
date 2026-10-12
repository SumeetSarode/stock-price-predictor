"""THE off-VPN script. Run it with no arguments and walk away:

    uv run python scripts/run_offvpn.py

It does everything, in order, with nothing for you to do:
  1. keeps the Mac awake and logs everything to reports/offvpn_run_<time>.log
  2. starts Docker if it is not running
  3. waits for you to leave the VPN (it does not abort)
  4. CLAIMS CHECK: tests what freellmapi / OmniRoute / models.dev / Strata say they do
  5. SWEEP: finds every free model in every shared repo and tests which ones run our pipeline,
     retrying temporary failures by itself and resuming from the previous run automatically
  6. writes reports/OFFVPN_RESULTS.md (start here), refreshed after each phase, and opens it

Re-running continues where it stopped (call budgets protect your real quota, so a big catalog can take
more than one day). Optional switches are the sweep's: --dry-run, --discover-only, --fresh, --no-keys,
--only ... plus --skip-claims and --no-open. Nothing here edits the app.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from collections import Counter
from datetime import UTC, datetime

import free_model_sweep_offvpn as FM
import gateway_claims_offvpn as G
import model_survey_offvpn as S
import sweep_env as SE

INDEX = "OFFVPN_RESULTS.md"
CANARIES = (S.REQUIRED_HOST, "api.groq.com")      # both must answer: the VPN blocks them
GATEWAY_ROUTES = ("freellmapi", "omniroute")


def build_parser() -> argparse.ArgumentParser:
    ap = FM.build_parser()
    ap.description = __doc__
    ap.add_argument("--skip-claims", action="store_true", help="skip the claims check, run only the sweep")
    ap.add_argument("--no-open", action="store_true", help="do not open the report when finished")
    return ap


def claims_args(args: argparse.Namespace, docker_ok: bool) -> argparse.Namespace:
    """The claims check, configured from the SAME flags (one source of truth). Without Docker it still runs the
    catalog and Strata checks instead of aborting."""
    routes = ["modelsdev", "strata", *(GATEWAY_ROUTES if docker_ok else ())]
    return argparse.Namespace(only=",".join(routes), with_keys=args.with_keys, repeats=1, keep=args.keep,
                              strata_url=args.strata_url, strata_key=args.strata_key, dry_run=args.dry_run)


def newest(reports, pattern: str):
    found = sorted(reports.glob(pattern))
    return found[-1] if found else None


def render_index(stamp: str, outcomes: dict[str, str], files: dict[str, str], rows: list[dict]) -> str:
    counts = Counter(r["verdict"] for r in rows)
    useful = [r for r in rows if r["verdict"] == "USEFUL"]
    out = [f"# Off-VPN results ({stamp})", "", "Start here. Every file below is in `reports/`.", "",
           "## Outcome", "", *(f"- **{phase}:** {text}" for phase, text in outcomes.items()), "", "## Files", "",
           *(f"- {name}: `{path}`" for name, path in files.items() if path), ""]
    if rows:
        out += ["## Sweep headline", "", ", ".join(f"{v}: {n}" for v, n in counts.most_common()), "",
                f"**{len(useful)} model(s) passed both pipeline steps in every run** (compatible with our pipeline, "
                "not proven accurate):", "", *(f"- {r['provider']} / `{r['model']}` ({r['route']})" for r in useful[:30]),
                *(["- ..."] if len(useful) > 30 else []), ""]
    out += ["Re-run the same command to continue: results so far are resumed automatically.", ""]
    return "\n".join(out)


def sweep_rows() -> list[dict]:
    f = FM.REPORTS / FM.LATEST_VERDICTS
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()] if f.exists() else []


def write_index(stamp: str, outcomes: dict[str, str], log) -> None:
    files = {"sweep report": str(FM.REPORTS / FM.LATEST_MD), "sweep verdicts (one line per model)":
             str(FM.REPORTS / FM.LATEST_VERDICTS), "claims report": str(newest(FM.REPORTS, "gateway_claims_*.md") or ""),
             "full log of this run": str(log)}
    (FM.REPORTS / INDEX).write_text(render_index(stamp, outcomes, files, sweep_rows()), encoding="utf-8")


def dry_run(args: argparse.Namespace) -> int:
    """Print both plans. Starts nothing (no Docker, no sleep-lock), contacts nothing, writes nothing."""
    asyncio.run(G.amain(claims_args(args, docker_ok=True)))
    asyncio.run(FM.amain(args))
    return 0


def run_all(args: argparse.Namespace) -> int:
    if args.dry_run:
        return dry_run(args)
    FM.REPORTS.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log = FM.REPORTS / f"offvpn_run_{stamp}.log"
    tee = SE.tee_stdout(log)
    SE.keep_awake()
    outcomes: dict[str, str] = {}
    codes: list[int] = []
    try:
        docker_ok = True
        if any(r in args.only for r in GATEWAY_ROUTES):
            docker_ok = FM.ensure_docker()
            if not docker_ok:
                outcomes["docker"] = "could not be started: gateway routes were skipped (everything else still ran)"
        if not args.discover_only and not SE.wait_for_off_vpn(
                lambda: all(S.host_reachable(h)[0] for h in CANARIES), args.wait_vpn_minutes):
            outcomes["vpn"] = f"still ON after {args.wait_vpn_minutes:g} min: nothing was tested. Disconnect and re-run."
            write_index(stamp, outcomes, log)
            return 2
        if not args.skip_claims:
            codes.append(asyncio.run(G.amain(claims_args(args, docker_ok))))
            outcomes["claims check"] = f"finished with exit code {codes[-1]}"
            write_index(stamp, outcomes, log)
        codes.append(asyncio.run(FM.amain(args)))
        outcomes["sweep"] = f"finished with exit code {codes[-1]}: see the sweep report"
        write_index(stamp, outcomes, log)
        if not args.no_open:
            SE.open_file(FM.REPORTS / INDEX)
    except KeyboardInterrupt:
        outcomes["interrupted"] = "stopped by you; re-run to continue where it left off"
        write_index(stamp, outcomes, log)
        codes.append(130)
    except Exception:
        outcomes["crashed"] = traceback.format_exc()[-600:]
        write_index(stamp, outcomes, log)
        print(traceback.format_exc())
        codes.append(1)
    finally:
        sys.stdout = tee.stream
        tee.close()
    return max(codes, default=0)


def main() -> int:
    return run_all(FM.normalize(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
