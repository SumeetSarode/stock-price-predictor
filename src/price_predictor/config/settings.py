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

from loguru import logger
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    """Project-wide settings sourced from env vars + .env."""

    # extra="ignore": .env may contain infra vars (SSL_CERT_FILE, HTTPS_PROXY,
    # REQUESTS_CA_BUNDLE) consumed by other libs (Python ssl, requests). We
    # don't want to model those here just to keep pydantic quiet.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Override default source priority: prefer .env over OS env.

        WHY: Tooling in the developer's shell (gcloud, other CLIs, etc.)
        sometimes sets vars like GEMINI_API_KEY to its own internal JWT.
        That pollutes os.environ and would shadow our project's .env values.
        For *this* project, .env is the canonical source -- shell pollution
        must not silently break us.

        Default order (highest precedence first):
            init_kwargs > env > dotenv > secrets_file
        Our order:
            init_kwargs > dotenv > env > secrets_file
        """
        return (init_settings, dotenv_settings, env_settings, file_secret_settings)

    # ── Secrets ────────────────────────────────────────────────
    groq_api_key: SecretStr = Field(validation_alias="GROQ_API_KEY")
    gemini_api_key: SecretStr = Field(validation_alias="GEMINI_API_KEY")

    # Optional -- only required if 'alpha_vantage' appears in PRICE_CHAIN
    # or PRICE_PAID. We default to empty SecretStr so users who don't use AV
    # don't need to set anything. The AlphaVantageProvider validates at
    # init time if it's actually being used in the active chain.
    alpha_vantage_api_key: SecretStr = Field(
        default=SecretStr(""), validation_alias="ALPHA_VANTAGE_API_KEY"
    )

    # Optional -- required if 'stooq' is in PRICE_CHAIN. Stooq added an
    # apikey requirement to their CSV endpoint in 2024; the key itself is
    # free to obtain via a one-time captcha (no signup, no email).
    # See data/providers/stooq_provider.py module docstring for the URL.
    stooq_api_key: SecretStr = Field(
        default=SecretStr(""), validation_alias="STOOQ_API_KEY"
    )

    # OPTIONAL: OpenRouter, used for candidate-model evaluation (e.g.
    # Nemotron-3-Ultra:free). Empty is fine if you don't use an
    # 'openrouter/*' entry in CHAIN_AGENTIC -- see effective_chain()'s
    # optional-key skip logic below for why an unset key here is safe
    # rather than a hard crash.
    openrouter_api_key: SecretStr = Field(
        default=SecretStr(""), validation_alias="OPENROUTER_API_KEY"
    )

    # ── Validators ────────────────────────────────────────────
    @field_validator("groq_api_key", "gemini_api_key")
    @classmethod
    def reject_placeholder_keys(cls, value: SecretStr) -> SecretStr:
        """Catch unset/placeholder keys at startup, not at first API call."""
        raw = value.get_secret_value()
        if not raw or raw.startswith("your_"):
            raise ValueError("Placeholder API key detected — set a real key in .env")
        return value

    @field_validator("alpha_vantage_api_key", "stooq_api_key", "openrouter_api_key")
    @classmethod
    def reject_placeholder_provider_keys(cls, value: SecretStr) -> SecretStr:
        """Allow EMPTY (provider not used) but reject obvious placeholders.

        Why empty is OK: a user who never uses a given provider shouldn't
        have to set its key. Price providers raise a clear PriceFetchError
        at fetch time if invoked without a key; LLM providers (openrouter)
        are silently skipped in effective_chain() if listed without a key
        -- see that method's optional-key logic.
        """
        raw = value.get_secret_value()
        if raw and raw.startswith("your_"):
            raise ValueError(
                "Placeholder provider API key detected. Either set a real "
                "key, remove the provider from its chain (PRICE_CHAIN / "
                "PRICE_PAID / CHAIN_AGENTIC), or leave the var unset."
            )
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

    # ── Local model runtime (Ollama) ───────────────────────────────────
    # Base URL of the local Ollama server. Only consulted when an
    # 'ollama_chat/*' or 'ollama/*' entry appears in a model chain -- e.g.
    # as the final OFFLINE fallback after Groq + Gemini are rate-limited.
    # A local model has no quota and no rate limit (provider_rate_limits
    # returns (0,0) for 'ollama'/'ollama_chat', so the limiter is a no-op).
    # Default matches Ollama's out-of-the-box listen address.
    ollama_api_base: str = Field(
        default="http://localhost:11434", validation_alias="OLLAMA_API_BASE"
    )

    # Ollama context window (num_ctx), in tokens. CRITICAL: Ollama defaults
    # num_ctx to just 2048 REGARDLESS of the model's real capacity, so a
    # large prediction prompt (technical view + news + filings + retry
    # feedback + reasoning) overflows it and litellm raises
    # ContextWindowExceededError -- surfaced to users as an "LLM token
    # limit" error even though Ollama has no quota/rate limit. qwen3:8b
    # Sizing: a bigger num_ctx costs proportionally more KV-cache RAM/VRAM
    # AND slows prompt processing, so we do NOT just max it out. A real
    # prediction prompt (view + news + filings + retry feedback) plus qwen3
    # reasoning lands well under 16k tokens, so 16384 gives comfortable
    # headroom without the memory/latency tax of qwen3 32768 ceiling.
    # Raise via OLLAMA_NUM_CTX only if you actually see a token-limit error;
    # lower it if RAM-constrained.
    ollama_num_ctx: int = Field(
        default=16384, validation_alias="OLLAMA_NUM_CTX"
    )

    # Ollama reasoning effort -> litellm maps this to Ollama's `think` flag.
    # CRITICAL nuance (verified against litellm ollama/chat/transformation.py):
    # for NON-gpt-oss models (i.e. qwen3), litellm does
    #     think = reasoning_effort in {"low", "medium", "high"}
    # so "low"/"medium"/"high" are IDENTICAL (all -> think=True) and ANY other
    # value (e.g. "none") -> think=False (thinking OFF). qwen3 has no middle
    # gear: it's think-on or think-off. Default "high" keeps reasoning ON for
    # prediction quality. Set OLLAMA_REASONING_EFFORT=none to benchmark the
    # faster think-off mode (validate accuracy on the backtest first).
    ollama_reasoning_effort: str = Field(
        default="high", validation_alias="OLLAMA_REASONING_EFFORT"
    )

    # ── Per-provider request rate limits (used by llm.rate_limiter) ──────
    # Defaults are slightly under each provider's free-tier ceiling so we
    # leave headroom for clock drift / other tooling sharing the API key.
    # Set RPM or RPD to 0 to disable that limit. Both 0 = limiter no-op.
    # When USE_PAID=true, the factory calls provider_rate_limits() with
    # the override path — caller can still tune via env if their paid plan
    # has a different ceiling.
    #
    # Free-tier reference (subject to change by providers):
    #   Gemini 2.5 Flash:  10 RPM, 250 RPD
    #   Groq (most free):  30 RPM, 14400 RPD
    gemini_rpm: int = Field(default=9,    validation_alias="GEMINI_RPM")
    gemini_rpd: int = Field(default=240,  validation_alias="GEMINI_RPD")
    groq_rpm:   int = Field(default=28,   validation_alias="GROQ_RPM")
    groq_rpd:   int = Field(default=14000, validation_alias="GROQ_RPD")

    # Per-provider overrides used when USE_PAID=true. Defaults are 0/0
    # (unlimited) which is the right behavior for most paid plans.
    gemini_rpm_paid: int = Field(default=0, validation_alias="GEMINI_RPM_PAID")
    gemini_rpd_paid: int = Field(default=0, validation_alias="GEMINI_RPD_PAID")
    groq_rpm_paid:   int = Field(default=0, validation_alias="GROQ_RPM_PAID")
    groq_rpd_paid:   int = Field(default=0, validation_alias="GROQ_RPD_PAID")

    # ── Pacing sleep cap (used by llm.rate_limiter) ─────────────────────
    # The per-minute limiter normally SLEEPS until a slot frees up. That's
    # right for big batch runs (keep load on fast hosted models), but for a
    # single interactive prediction a long sleep looks like a hang -- the
    # user would rather fall through to the next model in the chain.
    #
    # This caps how long the limiter is willing to sleep. If the required
    # wait exceeds the cap, the limiter RAISES a (per-minute) RateLimitError
    # instead of sleeping, which the ResilientModel handles by cooling the
    # provider down for 60s and falling over to the next model (e.g. Groq
    # saturated -> Gemini or the local Ollama tail). 0 = never cap (always
    # sleep; the original behavior).
    #
    # Default 10s: absorbs normal steady-state pacing jitter (Groq ~2s,
    # Gemini ~7s between calls at free-tier RPM) but bails fast when a
    # provider is genuinely saturated for a full window.
    pacing_max_sleep_s: float = Field(
        default=10.0, validation_alias="PACING_MAX_SLEEP_S"
    )

    # ── News RSS live fallback (used by data.news_snapshot) ──────────────
    # GDELT is keyless + IP-rate-limited (~1 req/5s) so bursts return 429 and
    # a run silently loses its news. When GDELT fails, fall back to Google
    # News RSS (live-only). CRITICAL: RSS returns RECENT news only, so it is
    # used ONLY when the requested window ends within news_rss_freshness_days
    # of now -- never for backtests (that would fabricate 'current' news for
    # a past date == look-ahead). Set enabled=false to keep GDELT-only.
    news_rss_fallback_enabled: bool = Field(
        default=True, validation_alias="NEWS_RSS_FALLBACK_ENABLED"
    )
    news_rss_freshness_days: int = Field(
        default=30, validation_alias="NEWS_RSS_FRESHNESS_DAYS"
    )

    def provider_rate_limits(self, provider: str) -> tuple[int, int]:
        """Return (rpm, rpd) for `provider`, honoring USE_PAID.

        Unknown providers return (0, 0) — i.e. no limiter applied. This is
        the right default: a new provider whose ceilings we haven't researched
        yet should not silently be hobbled to 9 RPM.
        """
        free_map = {
            "gemini": (self.gemini_rpm,      self.gemini_rpd),
            "groq":   (self.groq_rpm,        self.groq_rpd),
        }
        paid_map = {
            "gemini": (self.gemini_rpm_paid, self.gemini_rpd_paid),
            "groq":   (self.groq_rpm_paid,   self.groq_rpd_paid),
        }
        source = paid_map if self.use_paid else free_map
        return source.get(provider, (0, 0))

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
        chain = [m.strip() for m in chain_csv.split(",") if m.strip()]
        return [m for m in chain if not self._optional_key_missing(m)]

    # Providers with an OPTIONAL key -- unlike groq/gemini, which are
    # REQUIRED and validated at startup (reject_placeholder_keys), so they
    # can never silently reach here without a real key.
    #
    # WHY THIS EXISTS: llm.resilient.ResilientModel treats AuthenticationError
    # as a STRUCTURAL error, which does NOT fall through to the next model
    # in the chain -- it raises immediately (by design: an auth failure is a
    # config bug, not something retrying elsewhere fixes). So if an optional
    # provider is listed in CHAIN_AGENTIC without its key configured, EVERY
    # prediction would hard-crash the instant the chain reached that tier.
    # Skipping it here (once, at chain-resolution time) means an unconfigured
    # optional provider just quietly isn't part of your chain -- exactly how
    # PRICE_CHAIN's optional providers already behave.
    def _optional_key_missing(self, model_name: str) -> bool:
        """True if `model_name`'s provider needs a key we don't have."""
        provider = model_name.split("/", 1)[0]
        optional_keys = {"openrouter": self.openrouter_api_key}
        key = optional_keys.get(provider)
        if key is None:
            return False  # not an optional-key provider (groq/gemini/ollama)
        missing = not key.get_secret_value()
        if missing:
            logger.warning(
                "[settings] {!r} in chain has no configured key -- skipping "
                "it (set OPENROUTER_API_KEY in .env to enable it).",
                model_name,
            )
        return missing

    # ── Price-provider chain (parallels the LLM chain pattern above) ─────
    #
    # PRICE_CHAIN is the ordered free-tier fallback chain. Each entry is a
    # short provider name registered in data/providers/__init__.py's
    # PROVIDER_REGISTRY (currently: 'yfinance', 'jugaad', 'nse_bhavcopy',
    # 'stooq', 'alpha_vantage').
    #
    # PRICE_PAID is the single paid override used when USE_PAID_PRICES=true.
    # Same logic as LLM's USE_PAID: paying = no rate limits worth handling,
    # so chain collapses to a single best provider.
    #
    # Adding a new provider = register it in PROVIDER_REGISTRY and add its
    # short name here. No changes to settings.py needed.
    #
    # DEFAULT CHAIN: jugaad -> nse_bhavcopy -> yfinance (C1 decision).
    #   - jugaad        : NSE-native via the jugaad-data library (primary)
    #   - nse_bhavcopy  : NSE archives bhavcopy CSV (exchange-of-record EOD)
    #   - yfinance      : Yahoo's NSE mirror (community, breaks ~2x/yr)
    # Stooq is INTENTIONALLY OMITTED — verified empirically (2026-04-28) to
    # have ZERO NSE coverage. Class stays registered for non-Indian use cases.
    # Alpha Vantage is also OMITTED — its NSE feed is unreliable and free tier
    # is 25 req/day (C1 production scorecard).
    # Both remain in PROVIDER_REGISTRY for users who want to wire them in
    # explicitly, but they're not in the default fallback path.
    price_chain: str = Field(
        default="jugaad,nse_bhavcopy,yfinance", validation_alias="PRICE_CHAIN"
    )
    # PRICE_PAID = "the single most reliable NSE provider". jugaad-data is
    # free + NSE-native; we use it here because there's no paid NSE provider
    # in the registry yet (Kite/Upstox integration is H1-tier, deferred).
    # When that ships, switching this default to 'kite' is a one-line change.
    price_paid: str = Field(
        default="jugaad", validation_alias="PRICE_PAID"
    )
    use_paid_prices: bool = Field(
        default=False, validation_alias="USE_PAID_PRICES"
    )

    @field_validator("price_chain")
    @classmethod
    def validate_price_chain_format(cls, value: str) -> str:
        """Comma-separated, non-empty list of short provider names.

        We don't validate against the actual registry here -- pydantic-settings
        loads BEFORE the providers package is imported, so a registry check
        would create a circular import. The factory in data/prices.py does
        the registry lookup and surfaces 'unknown provider X' errors there.
        """
        if not value.strip():
            raise ValueError("PRICE_CHAIN cannot be empty.")
        names = [n.strip() for n in value.split(",") if n.strip()]
        if not names:
            raise ValueError("PRICE_CHAIN must list at least one provider name.")
        return value

    @field_validator("price_paid")
    @classmethod
    def validate_price_paid_format(cls, value: str) -> str:
        """Single non-empty provider name."""
        if not value.strip():
            raise ValueError("PRICE_PAID cannot be empty.")
        if "," in value:
            raise ValueError(
                f"PRICE_PAID must be a single provider name, got {value!r}. "
                "For multiple providers use PRICE_CHAIN instead."
            )
        return value.strip()

    def effective_price_chain(self) -> list[str]:
        """Return the ordered list of provider short-names actually in use.

        When USE_PAID_PRICES=true, returns just [PRICE_PAID] (no fallback;
        paying = solve the rate limit, not retry around it). Otherwise
        returns the full free-tier PRICE_CHAIN.
        """
        if self.use_paid_prices:
            return [self.price_paid]
        return [n.strip() for n in self.price_chain.split(",") if n.strip()]

    # ── Runtime config ─────────────────────────────
    log_level: str = Field(default="INFO", validation_alias="PREDICTOR_LOG_LEVEL")
    data_dir: Path = Field(default=Path("./data"), validation_alias="PREDICTOR_DATA_DIR")

    @field_validator("data_dir")
    @classmethod
    def resolve_data_dir(cls, value: Path) -> Path:
        """Resolve to absolute path. Directory creation happens in setup_directories()."""
        return value.resolve()

    # ── Network / proxy (optional — only set when behind a proxy) ────
    # Some networks require an HTTP proxy to reach api.groq.com etc.
    # Setting these makes HTTP libraries (httpx, aiohttp via LiteLLM) tunnel
    # through the proxy. Empty defaults = no-op on home/personal networks.
    https_proxy: str = Field(default="", validation_alias="HTTPS_PROXY")
    http_proxy: str = Field(default="", validation_alias="HTTP_PROXY")
    no_proxy: str = Field(default="", validation_alias="NO_PROXY")

    # ── SSL trust store (optional — only behind a TLS-inspecting proxy) ──
    # A TLS-inspecting proxy re-signs HTTPS traffic with a custom root CA.
    # Without these pointing at a combined certifi+custom bundle, EVERY
    # https request from Python fails with CERTIFICATE_VERIFY_FAILED.
    # Both names exist for compatibility:
    #   SSL_CERT_FILE      — Python stdlib + httpx (via our _http helper)
    #   REQUESTS_CA_BUNDLE — the requests library convention (yfinance uses requests)
    ssl_cert_file: str = Field(default="", validation_alias="SSL_CERT_FILE")
    requests_ca_bundle: str = Field(default="", validation_alias="REQUESTS_CA_BUNDLE")

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

    @property
    def predictions_dir(self) -> Path:
        """Default root for PredictionStore (CLI --save flag)."""
        return self.data_dir / "predictions"

    @property
    def news_snapshots_dir(self) -> Path:
        """Default root for NewsSnapshot (backtest news cache)."""
        return self.cache_dir / "news_snapshots"


# Singleton — import this everywhere
settings = Settings()


def setup_network() -> None:
    """Push proxy + TLS settings into os.environ so HTTP clients see them.

    Why: LiteLLM / httpx / aiohttp / requests all read proxy and CA-bundle
    config from os.environ at client construction time. Pydantic-settings
    loads `.env` into the Settings object, NOT into os.environ — so we
    copy them across explicitly.

    Also force LiteLLM to use httpx transport (which respects proxies
    consistently across all provider code paths). Without this, some
    LiteLLM providers bypass the proxy and DNS-fail on proxied networks.

    Idempotent. Called automatically at module import.
    """
    if settings.https_proxy:
        os.environ.setdefault("HTTPS_PROXY", settings.https_proxy)
    if settings.http_proxy:
        os.environ.setdefault("HTTP_PROXY", settings.http_proxy)
    if settings.no_proxy:
        os.environ.setdefault("NO_PROXY", settings.no_proxy)

    # CA bundle: needed for httpx-based providers (Stooq, AlphaVantage) on
    # any TLS-inspecting proxy network. yfinance uses requests, which reads
    # REQUESTS_CA_BUNDLE; httpx reads SSL_CERT_FILE (via our _http helper).
    # Set BOTH from whichever the user provided, so both code paths work.
    bundle = settings.ssl_cert_file or settings.requests_ca_bundle
    if bundle:
        os.environ.setdefault("SSL_CERT_FILE", bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)

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
