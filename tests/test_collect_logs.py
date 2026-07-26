"""Tests for scripts/collect_logs.py -- specifically the secret redaction,
since that file is meant to be shared with third parties (me!) and MUST
never leak an API key or token.

The ordering bug where the generic 'Authorization:' rule redacted the
word 'Bearer' but left the real token exposed is locked below.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# collect_logs lives in scripts/, not on the package path.
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from collect_logs import _redact  # noqa: E402


@pytest.mark.parametrize(
    "secret",
    [
        "AIzaSyD1234567890abcdefghijABCDEFGHIJ",   # Gemini
        "gsk_ABCDEFGHIJ1234567890klmnopqrstuv",    # Groq
        "sk-super-secret-value-here-123456",       # OpenAI-style
        "eyJhbGciOiJIUzI1NiJ.token.value123",      # bearer JWT-ish
    ],
)
def test_known_secret_shapes_are_scrubbed(secret):
    line = f"some log context key={secret} more text"
    out = _redact(line)
    assert secret not in out
    assert "REDACTED" in out


def test_bearer_token_not_left_exposed_by_authorization_rule():
    # Regression: the generic 'authorization' rule used to run first and
    # redact only the word 'Bearer', leaving the token in the clear.
    token = "eyJhbGciOiJIUzI1NiJ.token.value123"
    out = _redact(f"Authorization: Bearer {token}")
    assert token not in out


def test_non_secret_text_is_preserved():
    line = "2026-07-23 10:43:35 | INFO | resilient success provider=yfinance"
    assert _redact(line) == line
