"""Batch predictor — run predict() over many tickers with concurrency cap.

WHY THIS EXISTS
===============
A nightly portfolio scan or dashboard refresh needs to predict 20-100
stocks. Calling `predict()` 100 times sequentially is slow (each call is
~30-60s of LLM + network latency); calling them all in parallel hammers
both the LLM rate limits and yfinance.

`predict_many()` solves both:
  - Caps concurrency via asyncio.Semaphore (default 3 — gentle on Gemini's
    free-tier rate limits, plenty fast for nightly batches).
  - Uses asyncio.gather(..., return_exceptions=True) so ONE bad ticker
    doesn't tank the entire batch. Each result is a discriminated union:
    Prediction (success) or BatchError (failure with original exception).

DESIGN CHOICES
==============
1. Why a discriminated union, not just dropping failures?
   The CALLER decides what to do with errors (retry, log, alert, ignore).
   Silently dropping failures hides operational problems. The shape:

       results = await predict_many(['RELIANCE.NS', 'BADTICKER'])
       successes = [r for r in results if isinstance(r, Prediction)]
       failures = [r for r in results if isinstance(r, BatchError)]

2. Why deduplicate input tickers?
   Two reasons. (a) Cost — predicting the same ticker twice in one batch
   is wasteful. (b) Determinism — the result list maps 1:1 to canonical
   tickers, easy to look up. Order of FIRST occurrence is preserved.

3. Why is concurrency=3 the default?
   Gemini's free-tier RPM is generous but not infinite, and each predict()
   makes ~3-5 LLM calls (news_impact + synthesizer + possible retry).
   3 concurrent batches = ~15 in-flight calls peak. Empirically safe.
   Easy to bump for paid tiers via the param.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal, Union

from loguru import logger

from price_predictor.prediction.predictor import predict
from price_predictor.prediction.schema import Prediction

# Re-export for callers (avoids them needing two imports).
Horizon = Literal["intraday", "short", "medium", "long"]
Sensitivity = Literal["standard", "sensitive", "smooth"]


@dataclass(frozen=True)
class BatchError:
    """One ticker's prediction failed within a batch.

    Frozen dataclass (not a Pydantic model) because:
      - It's pure orchestration metadata, not part of the persisted
        contract (Prediction is what gets saved).
      - We want to carry the ORIGINAL exception object (not just str)
        so callers can re-raise it or pattern-match on type.
    """
    ticker: str
    error: BaseException
    error_type: str  # convenience: type(error).__name__

    @classmethod
    def from_exception(cls, ticker: str, exc: BaseException) -> "BatchError":
        return cls(ticker=ticker, error=exc, error_type=type(exc).__name__)


# Discriminated union — runtime check via isinstance(result, Prediction)
# vs isinstance(result, BatchError). Type checkers see both shapes.
BatchResult = Union[Prediction, BatchError]


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    """Drop duplicates, keep first-occurrence order.

    Built-in `set()` loses order; `dict.fromkeys()` preserves it (Py 3.7+
    insertion-ordered dicts). Cleaner than a manual loop.
    """
    return list(dict.fromkeys(items))


async def predict_many(
    tickers: list[str],
    horizon: Horizon = "short",
    *,
    sensitivity: Sensitivity = "standard",
    concurrency: int = 3,
) -> list[BatchResult]:
    """Predict for many tickers, gracefully handling partial failures.

    Args:
        tickers: List of stock symbols (any form — same as predict()).
            Duplicates are removed, preserving first-occurrence order.
        horizon: Prediction window applied to all tickers. Default 'short'.
        sensitivity: Indicator-cluster preset applied to all tickers.
        concurrency: Maximum number of predict() calls in flight at once.
            Default 3 — gentle on rate limits.

    Returns:
        List of BatchResult, one per UNIQUE input ticker, in the order
        they first appeared. Each element is either a successful
        Prediction or a BatchError carrying the original exception.

    Raises:
        ValueError: tickers is empty or concurrency < 1. These are CALLER
            bugs, not data problems — fail loud, don't return an empty
            list (which a caller might silently iterate).
    """
    if not tickers:
        raise ValueError("predict_many: tickers list is empty")
    if concurrency < 1:
        raise ValueError(
            f"predict_many: concurrency must be >= 1 (got {concurrency})"
        )

    unique = _dedupe_preserving_order(tickers)
    if len(unique) < len(tickers):
        logger.info(
            f"predict_many: deduplicated {len(tickers)} -> {len(unique)} tickers"
        )

    sem = asyncio.Semaphore(concurrency)

    async def _one(ticker: str) -> BatchResult:
        """Wrap predict() with semaphore + exception capture.

        We catch BaseException (not Exception) so even KeyboardInterrupt
        in one task doesn't propagate up and cancel sibling tasks. The
        caller can still find the interrupt in the returned BatchError
        and re-raise it explicitly if they want.
        """
        async with sem:
            try:
                return await predict(
                    ticker, horizon, sensitivity=sensitivity,
                )
            except BaseException as exc:  # noqa: BLE001 - intentional broad catch
                logger.warning(
                    f"predict_many: {ticker} failed - "
                    f"{type(exc).__name__}: {exc}"
                )
                return BatchError.from_exception(ticker, exc)

    logger.info(
        f"predict_many: starting {len(unique)} predictions "
        f"(concurrency={concurrency}, horizon={horizon})"
    )
    results = await asyncio.gather(*(_one(t) for t in unique))

    n_ok = sum(1 for r in results if isinstance(r, Prediction))
    n_err = len(results) - n_ok
    logger.info(
        f"predict_many: done - {n_ok} succeeded, {n_err} failed"
    )
    return results
