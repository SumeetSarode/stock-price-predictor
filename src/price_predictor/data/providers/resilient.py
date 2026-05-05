"""Resilient price fetcher -- ordered fallback over multiple providers.

PATTERN
=======
Mirrors `ResilientModel` for LLMs: takes an ordered list of providers,
tries them one at a time, falls back on transient failures, raises on
structural failures (caller bugs).

ERROR CLASSIFICATION
====================
- ValueError (caller bug: empty ticker, bad date range)
    -> RAISE immediately. Falling back to the next provider would just
       hit the same ValueError. No point.

- PriceFetchError (upstream issue: rate limit, network, empty result)
    -> FALL BACK to the next provider. Apply a cooldown so we don't keep
       hammering a provider that's clearly unhappy.

- Anything else (unexpected)
    -> FALL BACK with a warning log. Better to try the next provider than
       crash the whole request on something we didn't anticipate.

COOLDOWNS
=========
When a provider fails transiently, we mark it cooled-down for a short
window (default 60 seconds). Within that window, the resilient fetcher
skips that provider entirely. This stops us from re-trying a rate-limited
provider on every single request for the next minute.

If ALL providers are cooled down, we ignore cooldowns and try them anyway
(better to get rate-limited than to fail the user with no answer).
"""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
from loguru import logger

from price_predictor.data.providers.base import PriceFetchError, PriceProvider

# Default cooldown after a provider fails transiently. Short enough that
# transient blips clear quickly; long enough to avoid hammering a rate-
# limited API. Tunable per fetcher instance.
DEFAULT_COOLDOWN = timedelta(seconds=60)


class AllProvidersExhaustedError(PriceFetchError):
    """Raised when every provider in the chain has failed.

    Wraps the last failure for debugging; the chain itself is preserved
    in the message so logs explain what was tried. Ticker is included so
    users can scan logs and immediately see which symbol triggered the
    cascade.
    """

    def __init__(
        self,
        chain: list[str],
        last_error: Exception | None,
        ticker: str | None = None,
    ) -> None:
        ticker_part = f" for ticker={ticker!r}" if ticker else ""
        msg = (
            f"All price providers failed{ticker_part} "
            f"(tried in order: {chain}). Last error: {last_error}"
        )
        super().__init__(msg)
        self.chain = chain
        self.last_error = last_error
        self.ticker = ticker


class ResilientPriceFetcher:
    """Tries an ordered list of providers, falling back on transient failures.

    Stateful only in the cooldown map -- safe to share one instance across
    a process.
    """

    def __init__(
        self,
        providers: list[PriceProvider],
        cooldown: timedelta = DEFAULT_COOLDOWN,
    ) -> None:
        if not providers:
            raise ValueError("ResilientPriceFetcher needs at least one provider")
        self._providers = providers
        self._cooldown = cooldown
        self._cooled_until: dict[str, datetime] = {}

    @property
    def chain(self) -> list[str]:
        """Provider names in fallback order -- handy for logging."""
        return [p.name for p in self._providers]

    def _is_available(self, provider: PriceProvider) -> bool:
        """A provider is available if it isn't currently in cooldown."""
        until = self._cooled_until.get(provider.name)
        if until is None:
            return True
        return datetime.now(UTC) >= until

    def _set_cooldown(self, provider: PriceProvider) -> None:
        self._cooled_until[provider.name] = (
            datetime.now(UTC) + self._cooldown
        )
        logger.warning(
            f"[resilient-prices] cooling down provider={provider.name} "
            f"for {self._cooldown.total_seconds():.0f}s"
        )

    def _available_providers(self) -> list[PriceProvider]:
        """Providers not currently cooled down. If ALL are cooled, return all
        anyway -- better to try a (probably rate-limited) provider than to
        give up entirely."""
        available = [p for p in self._providers if self._is_available(p)]
        if available:
            return available
        logger.warning(
            "[resilient-prices] every provider is in cooldown; "
            "trying all anyway as last resort"
        )
        return list(self._providers)

    def fetch_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Try each available provider in order until one returns data.

        Raises:
            ValueError: caller-side bug (empty ticker, bad range). Fail fast,
                no fallback (would just hit the same ValueError again).
            AllProvidersExhaustedError: every provider failed transiently.
        """
        last_error: Exception | None = None

        for provider in self._available_providers():
            try:
                logger.info(
                    f"[resilient-prices] trying provider={provider.name} "
                    f"ticker={ticker} start={start} end={end}"
                )
                df = provider.fetch_ohlcv(ticker, start, end, interval)
                logger.info(
                    f"[resilient-prices] success provider={provider.name} "
                    f"rows={len(df)}"
                )
                return df

            except ValueError:
                # Caller's fault -- no point falling back, every provider
                # would reject the same input.
                raise

            except PriceFetchError as e:
                last_error = e
                self._set_cooldown(provider)
                logger.warning(
                    f"[resilient-prices] provider={provider.name} failed "
                    f"transiently: {e} -- falling back"
                )
                continue

            except Exception as e:
                # Unexpected error class -- treat as transient and try next.
                # Better than crashing the user's request on something we
                # didn't anticipate.
                last_error = e
                self._set_cooldown(provider)
                logger.error(
                    f"[resilient-prices] provider={provider.name} raised "
                    f"unexpected {type(e).__name__}: {e} -- falling back"
                )
                continue

        raise AllProvidersExhaustedError(self.chain, last_error, ticker=ticker)
