"""
Regenerates requirements.txt from pyproject.toml + uv.lock.

Usage:
    python scripts/freeze.py

Run this whenever you `uv add` a new dependency, so requirements.txt
(used by pip-only installers) stays in sync with pyproject.toml + uv.lock.

Cross-platform — works on Mac, Linux, and Windows.
"""
import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    cmd = [
        "uv", "export",
        "--format", "requirements-txt",
        "--no-hashes",
        "--quiet",
        "-o", "requirements.txt",
    ]
    print(f"📦 Regenerating requirements.txt in {project_root}...")
    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("❌ Failed to export requirements.txt")
        print(result.stderr)
        sys.exit(result.returncode)
    print("✅ requirements.txt regenerated")


if __name__ == "__main__":
    main()
