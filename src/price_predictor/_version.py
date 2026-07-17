"""Single source of truth for the application version.

The version is declared exactly once, in ``pyproject.toml``. At runtime we
read it back from the installed package metadata so the CLI banner, the
FastAPI app, and the UI footer can never drift apart. A literal fallback
keeps things working when running from a source tree that was never
installed (e.g. some CI sandboxes).
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("price-predictor")
except PackageNotFoundError:  # pragma: no cover - source-tree fallback
    __version__ = "1.0.0"

__all__ = ["__version__"]
