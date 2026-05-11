"""Tests for price_predictor.config.settings — profile resolution + validation.

Strategy:
    - Build Settings instances via env-var injection (monkeypatch).
    - Verify chain parsing, profile lookup, paid toggle, format validation.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from price_predictor.config.settings import Settings

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
_VALID_KEYS = {
    "GROQ_API_KEY": "gsk_test_realish_key_value",
    "GEMINI_API_KEY": "AIza_test_realish_key_value",
    "CHAIN_AGENTIC": "groq/openai/gpt-oss-120b,gemini/gemini-2.5-flash",
    "PAID_AGENTIC": "groq/openai/gpt-oss-120b",
}


def _build_settings(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    """Build a Settings with valid baseline + overrides, ignoring .env file."""
    env = {**_VALID_KEYS, **overrides}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    # _env_file=None tells pydantic-settings to skip .env entirely
    return Settings(_env_file=None)  # type: ignore[call-arg]


# ─────────────────────────────────────────────────────────────
# Chain parsing
# ─────────────────────────────────────────────────────────────
class TestChainParsing:
    def test_simple_two_model_chain(self, monkeypatch):
        s = _build_settings(monkeypatch)
        assert s.effective_chain("agentic") == [
            "groq/openai/gpt-oss-120b",
            "gemini/gemini-2.5-flash",
        ]

    def test_chain_strips_whitespace(self, monkeypatch):
        s = _build_settings(
            monkeypatch,
            CHAIN_AGENTIC="  groq/a , gemini/b ,  groq/c  ",
        )
        assert s.effective_chain("agentic") == ["groq/a", "gemini/b", "groq/c"]

    def test_chain_drops_empty_segments(self, monkeypatch):
        s = _build_settings(
            monkeypatch,
            CHAIN_AGENTIC="groq/a,,gemini/b,",
        )
        assert s.effective_chain("agentic") == ["groq/a", "gemini/b"]

    def test_single_model_chain(self, monkeypatch):
        s = _build_settings(monkeypatch, CHAIN_AGENTIC="groq/openai/gpt-oss-120b")
        assert s.effective_chain("agentic") == ["groq/openai/gpt-oss-120b"]


# ─────────────────────────────────────────────────────────────
# Profile resolution
# ─────────────────────────────────────────────────────────────
class TestProfileResolution:
    def test_unknown_profile_raises(self, monkeypatch):
        s = _build_settings(monkeypatch)
        with pytest.raises(ValueError, match="Unknown profile"):
            s.effective_chain("nonexistent")

    def test_unknown_profile_lists_available(self, monkeypatch):
        s = _build_settings(monkeypatch)
        with pytest.raises(ValueError, match="agentic"):
            s.effective_chain("nonexistent")


# ─────────────────────────────────────────────────────────────
# Paid toggle
# ─────────────────────────────────────────────────────────────
class TestPaidToggle:
    def test_use_paid_false_returns_full_chain(self, monkeypatch):
        s = _build_settings(monkeypatch, USE_PAID="false")
        assert s.use_paid is False
        assert len(s.effective_chain("agentic")) == 2

    def test_use_paid_true_collapses_to_single_paid_model(self, monkeypatch):
        s = _build_settings(
            monkeypatch,
            USE_PAID="true",
            PAID_AGENTIC="groq/openai/gpt-oss-120b",
        )
        assert s.use_paid is True
        assert s.effective_chain("agentic") == ["groq/openai/gpt-oss-120b"]

    def test_default_use_paid_is_false(self, monkeypatch):
        # Don't set USE_PAID at all
        env = {k: v for k, v in _VALID_KEYS.items()}
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        monkeypatch.delenv("USE_PAID", raising=False)
        s = Settings(_env_file=None)  # type: ignore[call-arg]
        assert s.use_paid is False


# ─────────────────────────────────────────────────────────────
# Format validation
# ─────────────────────────────────────────────────────────────
class TestFormatValidation:
    def test_chain_with_bad_format_rejected(self, monkeypatch):
        with pytest.raises(ValidationError, match="provider/model"):
            _build_settings(
                monkeypatch,
                CHAIN_AGENTIC="groq/openai/gpt-oss-120b,bad-no-slash",
            )

    def test_empty_chain_rejected(self, monkeypatch):
        with pytest.raises(ValidationError, match="empty"):
            _build_settings(monkeypatch, CHAIN_AGENTIC="")

    def test_paid_with_bad_format_rejected(self, monkeypatch):
        with pytest.raises(ValidationError, match="provider/model"):
            _build_settings(monkeypatch, PAID_AGENTIC="bad-no-slash")

    def test_placeholder_api_key_rejected(self, monkeypatch):
        with pytest.raises(ValidationError, match="Placeholder"):
            _build_settings(monkeypatch, GROQ_API_KEY="your_groq_key_here")


# ──────────────────────────────────────────────────────────────
# Source priority — .env must beat shell pollution
# ──────────────────────────────────────────────────────────────
class TestSourcePriority:
    """Real-world bug: Code Puppy sets GEMINI_API_KEY=<JWT> in the dev shell.
    ADK inherits + reapplies it over .env, breaking Gemini calls. Our project's
    .env must always win for project-scoped values.
    """

    def test_dotenv_overrides_polluted_os_environ(self, tmp_path, monkeypatch):
        # Write a project .env with the REAL key
        env_file = tmp_path / ".env"
        env_file.write_text(
            "GROQ_API_KEY=gsk_real_value_from_dotenv\n"
            "GEMINI_API_KEY=AIza_real_value_from_dotenv\n"
            "CHAIN_AGENTIC=groq/x/y\n"
            "PAID_AGENTIC=groq/x/y\n"
        )
        # Pollute os.environ with a Code-Puppy-style JWT (mimics the actual bug)
        monkeypatch.setenv("GEMINI_API_KEY", "eyJhbGc.fake.JWT")
        monkeypatch.setenv("GROQ_API_KEY", "gsk_should_be_ignored")

        s = Settings(_env_file=str(env_file))  # type: ignore[call-arg]

        assert s.gemini_api_key.get_secret_value() == "AIza_real_value_from_dotenv", (
            ".env value MUST override polluted os.environ (the JWT shell pollution bug)"
        )
        assert s.groq_api_key.get_secret_value() == "gsk_real_value_from_dotenv"


# ──────────────────────────────────────────────────────────────
# Price-provider chain (mirrors LLM chain pattern; same test shape)
# ──────────────────────────────────────────────────────────────
class TestPriceChain:
    """PRICE_CHAIN parses like CHAIN_AGENTIC; USE_PAID_PRICES toggles to
    PRICE_PAID just like USE_PAID toggles to PAID_AGENTIC."""

    def test_default_chain_is_nse_native_tier(self, monkeypatch):
        """No-config baseline: jugaad → nse_bhavcopy → yfinance.

        Per pred_logic_solutions.md C1: NSE-native primary, exchange-of-record
        secondary, Yahoo-mirror tertiary. Stooq + Alpha Vantage stay registered
        for non-NSE / explicit-opt-in callers but are NOT in the default chain.

        WHY EXPLICIT delenv: LiteLLM auto-loads .env into os.environ at
        import time (a quirk of that library). Other tests in the suite
        import LiteLLM, which leaks PRICE_CHAIN from our project's .env
        into os.environ. Without delenv, this test is order-dependent.
        """
        monkeypatch.delenv("PRICE_CHAIN", raising=False)
        monkeypatch.delenv("USE_PAID_PRICES", raising=False)
        s = _build_settings(monkeypatch)  # no PRICE_CHAIN override
        assert s.effective_price_chain() == ["jugaad", "nse_bhavcopy", "yfinance"]

    def test_full_chain_parses_in_order(self, monkeypatch):
        s = _build_settings(
            monkeypatch,
            PRICE_CHAIN="yfinance,stooq,alpha_vantage",
            ALPHA_VANTAGE_API_KEY="real_key_xyz",
        )
        # Order MUST be preserved -- it determines fallback priority
        assert s.effective_price_chain() == ["yfinance", "stooq", "alpha_vantage"]

    def test_chain_strips_whitespace_and_skips_empty(self, monkeypatch):
        """Same lenient parsing as the LLM chain (people add spaces / trailing
        commas)."""
        s = _build_settings(
            monkeypatch,
            PRICE_CHAIN="  yfinance , stooq , , alpha_vantage  ",
            ALPHA_VANTAGE_API_KEY="real_key_xyz",
        )
        assert s.effective_price_chain() == ["yfinance", "stooq", "alpha_vantage"]

    def test_use_paid_prices_collapses_to_single_provider(self, monkeypatch):
        """USE_PAID_PRICES=true bypasses the chain and uses PRICE_PAID alone.
        Mirrors USE_PAID's behavior for LLMs."""
        s = _build_settings(
            monkeypatch,
            PRICE_CHAIN="yfinance,stooq,alpha_vantage",
            PRICE_PAID="alpha_vantage",
            USE_PAID_PRICES="true",
            ALPHA_VANTAGE_API_KEY="real_key_xyz",
        )
        assert s.effective_price_chain() == ["alpha_vantage"]

    def test_empty_chain_rejected(self, monkeypatch):
        with pytest.raises(ValidationError, match="PRICE_CHAIN"):
            _build_settings(monkeypatch, PRICE_CHAIN="")

    def test_paid_must_be_single_provider(self, monkeypatch):
        """PRICE_PAID is a single name -- comma in it is a copy-paste mistake."""
        with pytest.raises(ValidationError, match="single provider name"):
            _build_settings(monkeypatch, PRICE_PAID="alpha_vantage,stooq")

    def test_use_paid_prices_defaults_false(self, monkeypatch):
        """If user doesn't set USE_PAID_PRICES, free chain is used.

        delenv USE_PAID_PRICES because LiteLLM-loaded .env may have set it.
        """
        monkeypatch.delenv("USE_PAID_PRICES", raising=False)
        s = _build_settings(monkeypatch, PRICE_CHAIN="yfinance,stooq",
                            STOOQ_API_KEY="real_key_xyz")
        assert s.use_paid_prices is False
        assert s.effective_price_chain() == ["yfinance", "stooq"]


class TestAlphaVantageKey:
    """Validator allows EMPTY (provider opt-out) but rejects placeholders."""

    def test_empty_key_allowed(self, monkeypatch):
        """A user who doesn't use AlphaVantage should NOT have to set anything.

        delenv ALPHA_VANTAGE_API_KEY because LiteLLM-loaded .env may have
        set it (see test_default_chain_is_yfinance_only for the same quirk).
        """
        monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
        s = _build_settings(monkeypatch)  # no ALPHA_VANTAGE_API_KEY
        assert s.alpha_vantage_api_key.get_secret_value() == ""

    def test_real_key_accepted(self, monkeypatch):
        s = _build_settings(monkeypatch, ALPHA_VANTAGE_API_KEY="real_key_value_xyz")
        assert s.alpha_vantage_api_key.get_secret_value() == "real_key_value_xyz"

    def test_placeholder_rejected(self, monkeypatch):
        """The string 'your_alpha_vantage_key_here' from .env.example is a
        red flag -- catch it at startup, not at first AlphaVantage call."""
        with pytest.raises(ValidationError, match="Placeholder"):
            _build_settings(monkeypatch, ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key")
