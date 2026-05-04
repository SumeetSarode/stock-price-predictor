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

    # ── Model selection: profile-based fallback chains ─────────────────
    #
    # Each PROFILE is a comma-separated, ordered fallback chain of LiteLLM
    # 'provider/model' identifiers. Agents declare which profile they need
    # (via make_resilient_model(profile="agentic")) and the resilient layer
    # tries each model in order, skipping any that are rate-limited.
    #
    # When USE_PAID=true, the profile's PAID_<NAME> single-model override
    # is used instead of the chain (no fallback needed when paying).
    #
    # CONVENTION: add a new profile by adding CHAIN_<NAME> + PAID_<NAME> here
    # AND extending the PROFILES set below. Don't hardcode chains in agents.
    chain_agentic: str = Field(validation_alias="CHAIN_AGENTIC")
    paid_agentic: str = Field(validation_alias="PAID_AGENTIC")
    use_paid: bool = Field(default=False, validation_alias="USE_PAID")

    @field_validator("chain_agentic")
    @classmethod
    def validate_chain_format(cls, value: str) -> str:
        """Each model in the chain must be in 'provider/model' format."""
        if not value.strip():
            raise ValueError("Model chain cannot be empty.")
        models = [m.strip() for m in value.split(",") if m.strip()]
        for m in models:
            if "/" not in m:
                raise ValueError(
                    f"Model {m!r} in chain must be in LiteLLM 'provider/model' "
                    f"format (e.g. 'groq/openai/gpt-oss-120b')."
                )
        return value

    @field_validator("paid_agentic")
    @classmethod
    def validate_paid_format(cls, value: str) -> str:
        """Paid model must be in 'provider/model' format."""
        if "/" not in value:
            raise ValueError(
                f"Paid model {value!r} must be in LiteLLM 'provider/model' format."
            )
        return value

    # ── Profile resolution ────────────────────────────────────
    # Add new profiles by appending here. The factory consults this map
    # to translate profile names → ordered chains / paid overrides.
    @property
    def _profile_map(self) -> dict[str, tuple[str, str]]:
        """profile_name → (chain_csv, paid_model). Add new profiles here."""
        return {
            "agentic": (self.chain_agentic, self.paid_agentic),
            # "fast": (self.chain_fast, self.paid_fast),       # iter B
            # "deep": (self.chain_deep, self.paid_deep),       # future
        }

    def effective_chain(self, profile: str) -> list[str]:
        """Return the ordered model chain for `profile`.

        When use_paid=True, returns a single-element list with the profile's
        paid model override (no fallback needed when paying). Otherwise
        returns the full free-tier fallback chain.
        """
        if profile not in self._profile_map:
            raise ValueError(
                f"Unknown profile {profile!r}. Available: {sorted(self._profile_map)}."
            )
        chain_csv, paid_model = self._profile_map[profile]
        if self.use_paid:
            return [paid_model]
        return [m.strip() for m in chain_csv.split(",") if m.strip()]

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
