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
