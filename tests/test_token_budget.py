"""Tests for ModelTokenBudget -- per-model tokens/min + tokens/day pacing.

Why this exists: free tiers run out of TOKENS long before requests. Groq's
gpt-oss-120b allows 8,000 tokens/min and one prediction is ~7,000, so the
request-only limiter said "plenty left" while Groq was already refusing.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from litellm.exceptions import RateLimitError

from price_predictor.llm import rate_limiter
from price_predictor.llm.rate_limiter import (
    TOKEN_BUDGETS,
    ModelTokenBudget,
    get_token_budget,
    reset_for_tests,
)
from price_predictor.llm.resilient import (
    SHORT_COOLDOWN,
    _classify_cooldown,
    _next_midnight_utc,
)


@pytest.fixture(autouse=True)
def _wipe_registry():
    reset_for_tests()
    yield
    reset_for_tests()


@pytest.fixture
def fake_clock(monkeypatch):
    """Controllable monotonic clock; fake sleep advances it."""
    now = {"t": 1_000.0}
    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        now["t"] += seconds

    monkeypatch.setattr(rate_limiter.time, "monotonic", lambda: now["t"])
    monkeypatch.setattr(rate_limiter.asyncio, "sleep", fake_sleep)
    return now, sleeps


def test_negative_limits_rejected():
    with pytest.raises(ValueError):
        ModelTokenBudget("m", tpm=-1)


@pytest.mark.asyncio
async def test_disabled_budget_is_a_no_op():
    b = ModelTokenBudget("m")
    assert b.disabled
    assert await b.reserve(10**9) is None


@pytest.mark.asyncio
async def test_second_prediction_waits_instead_of_hitting_a_429(fake_clock):
    """THE bug: 2 predictions in a row on an 8K TPM model.

    Before, both went straight out and the second one 429'd. Now the second
    waits for the first one's tokens to age out of the minute window.
    """
    _, sleeps = fake_clock
    b = ModelTokenBudget("groq/openai/gpt-oss-120b", tpm=8_000)
    await b.reserve(5_000)
    await b.reserve(5_000)  # 10K > 8K: must wait ~60s, not fire blind
    assert len(sleeps) == 1
    assert 59 < sleeps[0] <= 61


@pytest.mark.asyncio
async def test_long_wait_falls_over_instead_of_hanging(fake_clock):
    """With a sleep cap, a long wait raises so the chain tries the next model."""
    b = ModelTokenBudget("m", tpm=8_000, max_sleep_s=10.0)
    await b.reserve(5_000)
    with pytest.raises(RateLimitError) as exc:
        await b.reserve(5_000)
    # Must read as per-minute -> 60s cooldown, NOT benched until midnight.
    assert _classify_cooldown(exc.value) == SHORT_COOLDOWN


@pytest.mark.asyncio
async def test_request_bigger_than_whole_minute_budget_falls_over():
    b = ModelTokenBudget("m", tpm=8_000)
    with pytest.raises(RateLimitError):
        await b.reserve(9_000)  # can never fit, don't sleep forever


@pytest.mark.asyncio
async def test_settle_replaces_estimate_with_real_usage(fake_clock):
    """Overestimates must not waste budget once the real count is known."""
    _, sleeps = fake_clock
    b = ModelTokenBudget("m", tpm=8_000)
    r = await b.reserve(7_000)
    ModelTokenBudget.settle(r, 2_000)  # provider says it only used 2K
    await b.reserve(5_000)  # 2K + 5K fits -> no wait
    assert sleeps == []


@pytest.mark.asyncio
async def test_refused_call_gives_tokens_back(fake_clock):
    _, sleeps = fake_clock
    b = ModelTokenBudget("m", tpm=8_000)
    r = await b.reserve(7_000)
    ModelTokenBudget.settle(r, 0)
    await b.reserve(7_000)
    assert sleeps == []


@pytest.mark.asyncio
async def test_daily_budget_exhaustion_waits_for_midnight(fake_clock):
    b = ModelTokenBudget("m", tpd=10_000)
    await b.reserve(8_000)
    with pytest.raises(RateLimitError) as exc:
        await b.reserve(5_000)
    assert _classify_cooldown(exc.value) == _next_midnight_utc()


@pytest.mark.asyncio
async def test_daily_budget_resets_after_midnight(fake_clock, monkeypatch):
    b = ModelTokenBudget("m", tpd=10_000)
    await b.reserve(8_000)
    tomorrow = datetime.now(UTC) + timedelta(days=1)
    monkeypatch.setattr(
        rate_limiter, "_today_utc_midnight",
        lambda: datetime.combine(tomorrow.date(), datetime.min.time(), tzinfo=UTC),
    )
    await b.reserve(8_000)  # new day, fresh budget


def test_registry_uses_real_groq_limits_from_settings():
    b = get_token_budget("groq/openai/gpt-oss-120b")
    assert (b.tpm, b.tpd) == (8_000, 200_000)
    assert get_token_budget("groq/openai/gpt-oss-120b") is b
    assert "groq/openai/gpt-oss-120b" in TOKEN_BUDGETS


def test_unlisted_model_is_unmetered():
    assert get_token_budget("gemini/gemini-flash-latest").disabled
