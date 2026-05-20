"""Web-app settings — env-driven, isolated from the core prediction config.

Why a separate Settings class instead of reusing the existing
`price_predictor.config.settings`:

  - Single responsibility: this owns ONLY the web layer (port, host,
    static paths, etc.). The core settings owns LLM keys, rate limits,
    cache dirs.
  - Easier to test (no LLM creds required to spin up a dev server).
  - Cleaner failure mode: missing GEMINI_API_KEY won't stop the UI from
    booting; it surfaces as a friendly error when the user clicks Predict.

All paths resolved at import time so the app fails loudly at startup if
the frontend/ directory is missing (e.g. user cloned without LFS).
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Repo root = parent of src/  → ../../../  from this file.
# Resolved once at import time. Used for locating frontend/ assets.
_REPO_ROOT = Path(__file__).resolve().parents[3]


class WebSettings(BaseSettings):
    """Configuration for the local-first web app.

    All fields can be overridden via environment variables (with prefix
    ``WEB_``) or a ``.env`` file in the repo root.

    Example .env entries::

        WEB_HOST=127.0.0.1
        WEB_PORT=8000
        WEB_AUTO_OPEN_BROWSER=true
    """

    model_config = SettingsConfigDict(
        env_prefix="WEB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(
        default="127.0.0.1",
        description="Interface to bind to. 127.0.0.1 = localhost only (default, safest).",
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="TCP port. Change if 8000 is taken.",
    )
    auto_open_browser: bool = Field(
        default=True,
        description="Pop open the default browser when the server starts.",
    )
    reload: bool = Field(
        default=False,
        description="Uvicorn auto-reload (developer-only — don't enable for normal use).",
    )

    # ── Asset paths (resolved relative to repo root, not CWD) ──────────
    frontend_dir: Path = Field(
        default=_REPO_ROOT / "frontend",
        description="Root of the frontend/ directory (templates + static assets).",
    )

    @property
    def templates_dir(self) -> Path:
        return self.frontend_dir / "templates"

    @property
    def static_dir(self) -> Path:
        # We serve the whole frontend/ as /static so styles/, scripts/,
        # vendor/, and assets/ are all reachable via predictable URLs.
        return self.frontend_dir


# Module-level singleton — import this everywhere instead of
# instantiating per-request. Pydantic-cheap but not free.
settings = WebSettings()
