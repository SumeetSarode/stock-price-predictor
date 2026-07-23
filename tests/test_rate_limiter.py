"""Tests for price_predictor.llm.rate_limiter.

Strategy:
    - Construct ProviderRateLimiter directly with tiny limits.
    - Use asyncio.sleep monkey-patching where time-sensitive (sliding window).
    - Verify: disabled mode, RPM pacing, RPD raising, daily rollover, registry
      caching, FIFO-ish queueing under concurrent acquire().

We deliberately do NOT test the integration with ResilientModel here;
that's covered in test_resilient.py via a separate fake-limiter injection.
"""
from __future__ import annotations

import asyncio

import pytest
from litellm.exceptions import RateLimitError

from price_predictor.llm.rate_limiter import (
    LIMITERS,
    ProviderRateLimiter,
    get_limiter,
    provider_of,
    reset_for_tests,
)


@pytest.fixture(autouse=True)
def _wipe_registry():
    """Each test starts with an empty singleton registry."""
    reset_for_tests()
    yield
    reset_for_tests()


# ──────────────────────────────────────────────────────────────
# Construction + disabled mode
# ──────────────────────────────────────────────────────────────
def test_negative_limits_rejected():
    with pytest.raises(ValueError):
        ProviderRateLimiter("x", rpm=-1, rpd=0)
    with pytest.raises(ValueError):
        ProviderRateLimiter("x", rpm=0, rpd=-1)


def test_both_zero_is_disabled():
    lim = ProviderRateLimiter("x", rpm=0, rpd=0)
    assert lim.disabled is True


def test_one_nonzero_is_enabled():
    assert ProviderRateLimiter("x", rpm=1, rpd=0).disabled is False
    assert ProviderRateLimiter("x", rpm=0, rpd=1).disabled is False


@pytest.mark.asyncio
async def test_disabled_acquire_is_noop_and_doesnt_count():
    lim = ProviderRateLimiter("x", rpm=0, rpd=0)
    for _ in range(50):
        await lim.acquire()
    # Disabled limiter shouldn't bump observability counters either —
    # callers reading total_acquired shouldn't see noise from no-op mode.
    assert lim.total_acquired == 0


# ──────────────────────────────────────────────────────────────
# RPM pacing — sliding window
# ──────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_rpm_under_limit_no_sleep(monkeypatch):
    """Calls under the RPM ceiling should not trigger any sleep."""
    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("price_predictor.llm.rate_limiter.asyncio.sleep", fake_sleep)
    lim = ProviderRateLimiter("x", rpm=5, rpd=0)
    for _ in range(5):
        await lim.acquire()
    assert sleeps == []  # all 5 fit in the window
    assert lim.total_acquired == 5
    assert lim.total_paced_sleeps == 0


@pytest.mark.asyncio
async def test_rpm_over_limit_sleeps(monkeypatch):
    """The 4th call (RPM=3) should sleep long enough for the oldest to age out."""
    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("price_predictor.llm.rate_limiter.asyncio.sleep", fake_sleep)
    lim = ProviderRateLimiter("x", rpm=3, rpd=0)
    for _ in range(4):
        await lim.acquire()
    # First 3 immediate, 4th paces.
    assert len(sleeps) == 1
    # We slept just enough to bring the oldest below the 60s cutoff —
    # should be ~60s ± epsilon (real wall clock barely advanced here).
    assert 55.0 < sleeps[0] <= 60.5
    assert lim.total_paced_sleeps == 1


# ────────────────────────────────────────────────────
# Pacing sleep cap -- fall over instead of hanging on a long sleep
# ────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_pacing_wait_over_cap_raises_instead_of_sleeping(monkeypatch):
    """When the pacing wait exceeds max_sleep_s, RAISE (don't sleep).

    Regression for 'chain gets stuck on groq rate limit': the limiter used
    to sleep up to ~60s, so the resilient chain never fell over to the next
    model. With a cap it raises a per-minute RateLimitError immediately.
    """
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr("price_predictor.llm.rate_limiter.asyncio.sleep", fake_sleep)
    # rpm=2, tiny cap: the 3rd call would need to wait ~60s >> 5s cap.
    lim = ProviderRateLimiter("groq", rpm=2, rpd=0, max_sleep_s=5.0)
    await lim.acquire()
    await lim.acquire()
    with pytest.raises(RateLimitError) as ei:
        await lim.acquire()
    # Never slept -- it bailed instead.
    assert slept == []
    assert lim.total_paced_sleeps == 0
    assert lim.total_pacing_fallthroughs == 1
    # Message must be PER-MINUTE flavoured: no 'daily'/'quota' substrings,
    # so ResilientModel picks the 60s short cooldown (not until-midnight).
    msg = str(ei.value).lower()
    assert "daily" not in msg and "quota" not in msg
    assert "falling over" in msg


@pytest.mark.asyncio
async def test_pacing_wait_under_cap_still_sleeps(monkeypatch):
    """A short wait (<= cap) still sleeps -- pacing behavior preserved."""
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr("price_predictor.llm.rate_limiter.asyncio.sleep", fake_sleep)
    # Generous cap so the ~60s wait is under it -> sleeps as before.
    lim = ProviderRateLimiter("groq", rpm=2, rpd=0, max_sleep_s=120.0)
    await lim.acquire()
    await lim.acquire()
    await lim.acquire()
    assert len(slept) == 1
    assert lim.total_paced_sleeps == 1
    assert lim.total_pacing_fallthroughs == 0


@pytest.mark.asyncio
async def test_cap_zero_never_caps(monkeypatch):
    """max_sleep_s=0 preserves the original 'always sleep' behavior."""
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr("price_predictor.llm.rate_limiter.asyncio.sleep", fake_sleep)
    lim = ProviderRateLimiter("groq", rpm=1, rpd=0, max_sleep_s=0.0)
    await lim.acquire()
    await lim.acquire()  # would raise if capping were active
    assert len(slept) == 1
    assert lim.total_pacing_fallthroughs == 0


def test_negative_max_sleep_rejected():
    with pytest.raises(ValueError):
        ProviderRateLimiter("x", rpm=1, rpd=0, max_sleep_s=-1.0)
@pytest.mark.asyncio
async def test_rpd_exhaustion_raises_ratelimit():
    """Hitting RPD must raise litellm.RateLimitError so caller can fall over."""
    lim = ProviderRateLimiter("x", rpm=0, rpd=2)
    await lim.acquire()
    await lim.acquire()
    with pytest.raises(RateLimitError) as ei:
        await lim.acquire()
    # The error message must contain 'daily' or 'quota' so the upstream
    # ResilientModel's _classify_cooldown picks the midnight-UTC cooldown
    # rather than the 60s short cooldown. Failing this assertion would
    # silently break the multi-provider fallback strategy.
    msg = str(ei.value).lower()
    assert "daily" in msg or "quota" in msg, f"bad error msg: {msg!r}"
    assert lim.total_daily_rejections == 1
    assert lim.total_acquired == 2  # the rejected one didn't count


@pytest.mark.asyncio
async def test_rpd_rollover_at_midnight_resets_counter(monkeypatch):
    """Crossing a UTC midnight boundary should reset the daily counter."""
    from datetime import UTC, datetime, timedelta

    lim = ProviderRateLimiter("x", rpm=0, rpd=2)
    await lim.acquire()
    await lim.acquire()
    # Simulate that midnight has passed by rewinding the anchor by 1 day.
    lim._day_anchor = datetime.now(UTC).replace(
        hour=0, minute=0, second=0, microsecond=0
    ) - timedelta(days=1)
    # Now the limiter should roll over on the next acquire instead of raising.
    await lim.acquire()
    assert lim._day_count == 1  # counter reset, then this call took 1 slot


# ──────────────────────────────────────────────────────────────
# Registry / singleton behavior
# ──────────────────────────────────────────────────────────────
def test_provider_of_extracts_first_segment():
    assert provider_of("gemini/gemini-2.5-flash") == "gemini"
    assert provider_of("groq/openai/gpt-oss-120b") == "groq"
    assert provider_of("openrouter/mistral-large") == "openrouter"
    # No slash → treat whole string as the bucket key (don't crash)
    assert provider_of("bareprovider") == "bareprovider"


@pytest.mark.asyncio
async def test_get_limiter_returns_singleton():
    """Two get_limiter calls for the same provider return the SAME object."""
    a = await get_limiter("gemini")
    b = await get_limiter("gemini")
    assert a is b
    assert "gemini" in LIMITERS


@pytest.mark.asyncio
async def test_get_limiter_distinct_providers_distinct_limiters():
    g = await get_limiter("gemini")
    q = await get_limiter("groq")
    assert g is not q
    assert g.name == "gemini"
    assert q.name == "groq"


@pytest.mark.asyncio
async def test_unknown_provider_gets_disabled_limiter():
    """A provider we have no limits for should get a no-op limiter, not crash."""
    lim = await get_limiter("acme_new_provider")
    assert lim.disabled is True


# ──────────────────────────────────────────────────────────────
# Concurrency: callers queue, don't lose pacing
# ──────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_concurrent_acquires_serialize_correctly(monkeypatch):
    """10 concurrent acquires against RPM=3 should produce exactly 7 pacing
    sleeps (3 free + 7 paced). Confirms the lock is held across the sleep."""
    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("price_predictor.llm.rate_limiter.asyncio.sleep", fake_sleep)
    lim = ProviderRateLimiter("x", rpm=3, rpd=0)
    await asyncio.gather(*(lim.acquire() for _ in range(10)))
    assert lim.total_acquired == 10
    # The 4th through 10th calls each had to wait for an older entry to
    # age out — that's 7 sleeps. If the lock weren't held across sleep
    # we'd see races and a smaller count.
    assert lim.total_paced_sleeps == 7
    assert len(sleeps) == 7
