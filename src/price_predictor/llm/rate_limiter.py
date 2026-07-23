"""Per-provider request-rate limiter for LLM calls.

WHY THIS EXISTS
===============
The ResilientModel layer (llm/resilient.py) is REACTIVE — it catches 429s
and falls through to the next model. That works for occasional bursts, but
breaks down for batch workloads:

  predict_many(50_tickers, concurrency=3)

Each predict() fans out to ~4 synthesizer calls (one per horizon). With
concurrency=3 that's ~12 LLM calls hitting one provider in a 1-2 second
window. Gemini free-tier is 10 RPM — every batch run starts with a fresh
flurry of 429s before the cooldown logic even gets a chance to spread the
load. By then the next 11 in-flight calls have already 429'd too.

The right fix is PROACTIVE pacing: never let more than N requests per
minute reach a given provider in the first place. Excess calls queue up
and proceed in order — slower wall-clock, but zero 429 storms.

DESIGN
======
- One ProviderRateLimiter per provider (gemini, groq, ...) — keyed by the
  string before the first '/' in the LiteLLM model name.
- Single process-wide registry (LIMITERS) so every ResilientModel instance
  shares the same buckets. A new agent / wrapper does NOT reset the count.
- Sliding-window log for per-minute (RPM): keeps timestamps of the last
  60s of requests; sleeps just long enough for the oldest to expire.
- Counter + midnight-UTC anchor for per-day (RPD): on daily exhaustion we
  RAISE a litellm RateLimitError instead of sleeping — so the existing
  ResilientModel fallback chain skips this provider until midnight and
  routes to the next one (Groq instead of waiting on Gemini, etc.).
- Locking: one asyncio.Lock per provider. The lock IS held across the
  in-pacing-sleep — that's intentional. Concurrent callers queue up and
  the pacing math stays accurate. The lock is released BEFORE the actual
  LLM call so we don't serialize the slow HTTP roundtrip.

KILL-SWITCH
===========
Each provider's RPM and RPD can be set to 0 via env (e.g. GEMINI_RPM=0)
to mean "unlimited — skip pacing entirely". Paid-tier users typically
want this; the factory auto-disables pacing when USE_PAID=true.

NON-GOALS
=========
- No cross-process state. Two `python -m price_predictor ...` processes
  running concurrently each have their own bucket → combined they may
  exceed the provider's true limit. If/when we need multi-process, swap
  the in-memory deque for a Redis sorted-set behind the same interface.
- No token-budget tracking (TPM/TPD). Only request-count limits. Token
  budgets would need to inspect each response's usage metadata, which
  ResilientModel doesn't currently surface. Future iteration if needed.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import UTC, datetime
from datetime import time as dtime

from litellm.exceptions import RateLimitError

logger = logging.getLogger(__name__)


def _today_utc_midnight() -> datetime:
    """Most recent UTC midnight (the start of today's RPD window)."""
    now = datetime.now(UTC)
    return datetime.combine(now.date(), dtime.min, tzinfo=UTC)


class ProviderRateLimiter:
    """Async sliding-window + daily-counter limiter for ONE provider.

    Two limits, applied independently:
        rpm: max requests per rolling 60-second window. 0 = unlimited.
        rpd: max requests per UTC day.                 0 = unlimited.

    On rpm exhaustion: SLEEP until oldest in-window request ages out --
        UNLESS the wait would exceed ``max_sleep_s`` (>0), in which case
        RAISE a per-minute RateLimitError so the caller's ResilientModel
        cools this provider down for 60s and falls over to the next model
        instead of hanging on a long pacing sleep.
    On rpd exhaustion: RAISE litellm.RateLimitError so the caller's
        ResilientModel marks this provider cooled-down until midnight UTC
        and falls through to the next provider in the chain.
    """

    def __init__(
        self, name: str, *, rpm: int = 0, rpd: int = 0, max_sleep_s: float = 0.0,
    ) -> None:
        if rpm < 0 or rpd < 0:
            raise ValueError(f"rpm/rpd must be >= 0 (got rpm={rpm}, rpd={rpd})")
        if max_sleep_s < 0:
            raise ValueError(f"max_sleep_s must be >= 0 (got {max_sleep_s})")
        self.name = name
        self.rpm = rpm
        self.rpd = rpd
        self.max_sleep_s = max_sleep_s
        self._lock = asyncio.Lock()
        # Sliding window: monotonic timestamps of recent requests
        self._minute_log: deque[float] = deque()
        # Daily counter + the midnight it's anchored to
        self._day_count: int = 0
        self._day_anchor: datetime = _today_utc_midnight()
        # Soft observability counters (useful for tests + future metrics)
        self.total_acquired: int = 0
        self.total_paced_sleeps: int = 0
        self.total_daily_rejections: int = 0
        self.total_pacing_fallthroughs: int = 0

    @property
    def disabled(self) -> bool:
        """True if both limits are 0 (i.e. this limiter is a no-op)."""
        return self.rpm == 0 and self.rpd == 0

    def _maybe_roll_day(self) -> None:
        """Reset the daily counter if we've crossed a UTC midnight boundary."""
        today_midnight = _today_utc_midnight()
        if today_midnight > self._day_anchor:
            logger.info(
                "[rate_limit] %s daily window rollover: %d -> 0",
                self.name, self._day_count,
            )
            self._day_count = 0
            self._day_anchor = today_midnight

    async def acquire(self) -> None:
        """Block until a request slot is available, then record the request.

        Raises:
            litellm.RateLimitError: when the DAILY quota is exhausted. The
                caller's ResilientModel will treat this as a daily cooldown
                (until next midnight UTC) and fall over to the next model.
        """
        if self.disabled:
            return

        async with self._lock:
            self._maybe_roll_day()

            # ── Daily check (raise; do NOT sleep until midnight) ──
            if self.rpd > 0 and self._day_count >= self.rpd:
                self.total_daily_rejections += 1
                logger.warning(
                    "[rate_limit] %s daily quota exhausted: %d/%d. "
                    "Raising RateLimitError so caller can fall over.",
                    self.name, self._day_count, self.rpd,
                )
                # Wording matters: ResilientModel._classify_cooldown looks for
                # 'daily' / 'quota' substrings to pick the midnight-UTC cooldown
                # instead of the 60-second short cooldown.
                raise RateLimitError(
                    message=(
                        f"{self.name} daily request quota exhausted "
                        f"({self._day_count}/{self.rpd}). Will reset at next "
                        f"midnight UTC."
                    ),
                    model=self.name,
                    llm_provider=self.name,
                )

            # ── Per-minute sliding-window pace ──
            if self.rpm > 0:
                # Drop entries older than 60s
                mono_now = time.monotonic()
                cutoff = mono_now - 60.0
                while self._minute_log and self._minute_log[0] < cutoff:
                    self._minute_log.popleft()

                if len(self._minute_log) >= self.rpm:
                    # Oldest entry's age determines how long to wait so that
                    # popping it brings us under rpm.
                    wait = self._minute_log[0] + 60.0 - mono_now + 0.05
                    if wait > 0:
                        # Cap: if the pacing sleep would be longer than we're
                        # willing to block, fall over to the next model in the
                        # chain instead of hanging. Raise a PER-MINUTE
                        # RateLimitError (wording deliberately avoids 'daily'/
                        # 'quota' so ResilientModel picks the 60s short cooldown,
                        # not the until-midnight one).
                        if 0 < self.max_sleep_s < wait:
                            self.total_pacing_fallthroughs += 1
                            logger.info(
                                "[rate_limit] %s pacing wait %.2fs exceeds cap "
                                "%.2fs -- raising RateLimitError to fall over "
                                "(%d/%d in last 60s)",
                                self.name, wait, self.max_sleep_s,
                                len(self._minute_log), self.rpm,
                            )
                            raise RateLimitError(
                                message=(
                                    f"{self.name} per-minute rate pacing would "
                                    f"block {wait:.1f}s (> cap "
                                    f"{self.max_sleep_s:.1f}s); falling over to "
                                    f"the next model."
                                ),
                                model=self.name,
                                llm_provider=self.name,
                            )
                        self.total_paced_sleeps += 1
                        logger.info(
                            "[rate_limit] %s pacing: sleeping %.2fs "
                            "(%d/%d in last 60s)",
                            self.name, wait, len(self._minute_log), self.rpm,
                        )
                        await asyncio.sleep(wait)
                    # Re-prune after the sleep so the log is fresh.
                    mono_now = time.monotonic()
                    cutoff = mono_now - 60.0
                    while self._minute_log and self._minute_log[0] < cutoff:
                        self._minute_log.popleft()

                self._minute_log.append(time.monotonic())

            self._day_count += 1
            self.total_acquired += 1


# ──────────────────────────────────────────────────────────────
# Process-wide registry: one limiter per provider, shared across all
# ResilientModel instances. Lazily constructed on first acquire() call
# so that pulling settings at import time isn't required.
# ──────────────────────────────────────────────────────────────
LIMITERS: dict[str, ProviderRateLimiter] = {}
_REGISTRY_LOCK = asyncio.Lock()


def provider_of(model_name: str) -> str:
    """Extract the provider prefix from a LiteLLM model string.

    Examples:
        'gemini/gemini-2.5-flash'    -> 'gemini'
        'groq/openai/gpt-oss-120b'   -> 'groq'
        'openrouter/mistral-large'   -> 'openrouter'
    """
    if "/" not in model_name:
        # Fallback: treat the whole name as the provider key. Better to
        # bucket it than crash on a malformed model name.
        return model_name
    return model_name.split("/", 1)[0]


async def get_limiter(provider: str) -> ProviderRateLimiter:
    """Return the singleton ProviderRateLimiter for `provider`.

    Constructs it on first access using per-provider limits pulled from
    settings. Subsequent calls return the same instance so the sliding-
    window state is shared across all ResilientModel instances.
    """
    if provider in LIMITERS:
        return LIMITERS[provider]
    async with _REGISTRY_LOCK:
        # Re-check inside the lock (another task may have created it)
        if provider in LIMITERS:
            return LIMITERS[provider]
        # Lazy import to avoid circular settings <-> llm import at module load
        from price_predictor.config.settings import settings
        rpm, rpd = settings.provider_rate_limits(provider)
        limiter = ProviderRateLimiter(
            provider, rpm=rpm, rpd=rpd, max_sleep_s=settings.pacing_max_sleep_s,
        )
        LIMITERS[provider] = limiter
        if not limiter.disabled:
            logger.info(
                "[rate_limit] registered provider=%s rpm=%d rpd=%d",
                provider, rpm, rpd,
            )
        return limiter


def reset_for_tests() -> None:
    """Wipe the singleton registry. Tests only — don't call from app code."""
    LIMITERS.clear()
