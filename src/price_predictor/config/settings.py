"""Centralized configuration for the price predictor app.

Settings are loaded from environment variables and `.env` (in dev). All API
keys are typed as `SecretStr` and must be unmasked via `.get_secret_value()`
when handed to external libraries.

Public surface:
    settings           — singleton instance, import this everywhere
    setup_directories  — call once at app startup to create data/log dirs
    setup_network      — called automatically on import; applies proxy env vars
"""
import os
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-wide settings sourced from env vars + .env."""

    model_config = SettingsConfigDict(env_file=".env")

    # ── Secrets ────────────────────────────────────────────────
    groq_api_key: SecretStr = Field(validation_alias="GROQ_API_KEY")
    gemini_api_key: SecretStr = Field(validation_alias="GEMINI_API_KEY")

    # ── Validators ────────────────────────────────────────────
    @field_validator("groq_api_key", "gemini_api_key")
    @classmethod
    def reject_placeholder_keys(cls, value: SecretStr) -> SecretStr:
        """Catch unset/placeholder keys at startup, not at first API call."""
        raw = value.get_secret_value()
        if not raw or raw.startswith("your_"):
            raise ValueError("Placeholder API key detected — set a real key in .env")
        return value

    # ── Model selection (LiteLLM 'provider/model' format) ─────────────
    primary_model: str = Field(validation_alias="PRIMARY_MODEL")
    secondary_model: str = Field(validation_alias="SECONDARY_MODEL")

    @field_validator("primary_model", "secondary_model")
    @classmethod
    def validate_model_format(cls, value: str) -> str:
        """Catch obvious typos like 'llama-3.3' (missing 'groq/' prefix)."""
        if "/" not in value:
            raise ValueError(
                f"Model {value!r} must be in LiteLLM 'provider/model' format "
                f"(e.g. 'groq/llama-3.3-70b-versatile')."
            )
        return value

    # ── Runtime config ─────────────────────────────
    log_level: str = Field(default="INFO", validation_alias="PREDICTOR_LOG_LEVEL")
    data_dir: Path = Field(default=Path("./data"), validation_alias="PREDICTOR_DATA_DIR")

    @field_validator("data_dir")
    @classmethod
    def resolve_data_dir(cls, value: Path) -> Path:
        """Resolve to absolute path. Directory creation happens in setup_directories()."""
        return value.resolve()

    # ── Network / proxy (optional — only set when behind a corporate proxy) ────
    # When running on Walmart network: api.groq.com etc. don't resolve directly.
    # Setting these makes HTTP libraries (httpx, aiohttp via LiteLLM) tunnel
    # through the proxy. Empty defaults = no-op on home/personal networks.
    https_proxy: str = Field(default="", validation_alias="HTTPS_PROXY")
    http_proxy: str = Field(default="", validation_alias="HTTP_PROXY")
    no_proxy: str = Field(default="", validation_alias="NO_PROXY")

    # ── Derived paths ─────────────────────────────────────────
    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def kb_dir(self) -> Path:
        return self.data_dir / "kb"


# Singleton — import this everywhere
settings = Settings()


def setup_network() -> None:
    """Push proxy settings into os.environ so HTTP clients see them.

    Why: LiteLLM uses aiohttp internally. aiohttp + httpx both read proxy
    config from os.environ at client construction time. Pydantic-settings
    loads `.env` into the Settings object, NOT into os.environ — so we
    copy them across explicitly.

    Also force LiteLLM to use httpx transport (which respects proxies
    consistently across all provider code paths). Without this, some
    LiteLLM providers bypass the proxy and DNS-fail on corporate networks.

    Idempotent. Called automatically at module import.
    """
    if settings.https_proxy:
        os.environ.setdefault("HTTPS_PROXY", settings.https_proxy)
    if settings.http_proxy:
        os.environ.setdefault("HTTP_PROXY", settings.http_proxy)
    if settings.no_proxy:
        os.environ.setdefault("NO_PROXY", settings.no_proxy)

    # Force LiteLLM to use httpx (which honors HTTPS_PROXY) instead of
    # aiohttp's transport, which has inconsistent proxy support.
    if settings.https_proxy or settings.http_proxy:
        os.environ.setdefault("DISABLE_AIOHTTP_TRANSPORT", "True")


# Apply network config eagerly so any subsequent HTTP client picks it up.
setup_network()


def setup_directories() -> None:
    """Create all required project directories. Idempotent.

    Call once at app startup BEFORE setup_logging() (logging needs logs_dir to exist).
    Side-effecting on purpose; kept out of Pydantic validators which should be pure.
    """
    for directory in (
        settings.data_dir,
        settings.outputs_dir,
        settings.logs_dir,
        settings.cache_dir,
        settings.kb_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
