"""Honest statistics for the model scorecard.

A handful of backtest cases cannot prove a model is good, and a model that "got 6 of 8" may just be lucky. So every
accuracy comes with a 95% confidence interval, and the verdict compares the WHOLE interval to a baseline instead of
comparing a single number. Small samples therefore say CANT_TELL_YET rather than crowning a winner.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

Z95 = 1.96
MIN_JUDGED = 10              # below this many graded directional calls we do not even draw an interval
TARGET_HALF_WIDTH = 0.10     # "tell me within +/-10 percentage points"

BEATS, WORSE, CANT_TELL, TOO_FEW = "BEATS_BASELINE", "WORSE_THAN_BASELINE", "CANT_TELL_YET", "TOO_FEW"
VERDICT_RANK = {BEATS: 0, CANT_TELL: 1, TOO_FEW: 2, WORSE: 3}


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials. Behaves at the edges (k=0, k=n, small n)."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def baseline_accuracy(outcomes: Iterable[tuple[float, float]]) -> float:
    """What the best CONSTANT strategy scores: always bullish, always bearish, or always neutral.

    `outcomes` is (realized_return, neutral_tolerance) once per distinct judged (ticker, date, horizon): outcomes do
    not depend on the model. The app's grader marks bullish right if the price rose, bearish if it fell, and NEUTRAL
    right if it barely moved (within the horizon's tolerance), so "always neutral" is a real free strategy in a calm
    market, and "always bullish" is one in a rising market. A model has to beat the best of them to show skill.
    """
    n = bull = bear = flat = 0
    for ret, tol in outcomes:
        n += 1
        bull += ret > 0
        bear += ret < 0
        flat += abs(ret) <= tol
    return 0.5 if n == 0 else max(bull, bear, flat) / n


def judged_needed(half_width: float = TARGET_HALF_WIDTH) -> int:
    """Graded directional calls needed before the interval is about +/-half_width (worst case, accuracy ~ 50%)."""
    return math.ceil(Z95 * Z95 * 0.25 / (half_width * half_width))


def judge(k: int, n: int, baseline: float) -> tuple[str, float, float]:
    """-> (verdict, low, high). BEATS only if the interval's LOWER end clears the baseline."""
    if n < MIN_JUDGED:
        return TOO_FEW, 0.0, 1.0
    low, high = wilson(k, n)
    if low > baseline:
        return BEATS, low, high
    if high < baseline:
        return WORSE, low, high
    return CANT_TELL, low, high
