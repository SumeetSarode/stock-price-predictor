"""Honest statistics for the model scorecard.

A handful of backtest cases cannot prove a model is good, and a model that "got 6 of 8" may just be lucky. So every
accuracy comes with a 95% confidence interval, and the verdict compares the WHOLE interval to a baseline instead of
comparing a single number. Small samples therefore say CANT_TELL_YET rather than crowning a winner.

Models are judged on the cases where the price REALLY MOVED (further than the app's neutral tolerance). On a calm
day "always neutral" is a free win, so a judgement over all days mostly measures how calm the market was: with the
app's 2% daily / 4% weekly tolerances, large Indian stocks make that baseline 85-100% and nothing could ever beat it.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

Z95 = 1.96
Z99 = 2.576                  # interim looks: stricter, because stopping the moment a model looks good crowns lucky ones
MIN_JUDGED = 10              # below this many graded directional calls we do not even draw an interval
TARGET_HALF_WIDTH = 0.10     # "tell me within +/-10 percentage points"

BEATS, WORSE, CANT_TELL, TOO_FEW = "BEATS_BASELINE", "WORSE_THAN_BASELINE", "CANT_TELL_YET", "TOO_FEW"
NO_EDGE, KEEP = "NO_EDGE", "KEEP_TESTING"          # the staged test's extra outcomes (see `stage_verdict`)
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


def is_move(ret: float, tolerance: float) -> bool:
    """A REAL move: the price went further than the horizon's neutral tolerance, so "neutral" is wrong by definition."""
    return abs(ret) > tolerance


def baseline_accuracy(outcomes: Iterable[tuple[float, float]]) -> float:
    """What the best CONSTANT strategy scores: always bullish, always bearish, or always neutral.

    `outcomes` is (realized_return, neutral_tolerance) once per distinct judged (ticker, date, horizon): outcomes do
    not depend on the model. The app's grader marks bullish right if the price rose, bearish if it fell, and NEUTRAL
    right if it barely moved (within the horizon's tolerance), so "always neutral" is a real free strategy in a calm
    market, and "always bullish" is one in a rising market. A model has to beat the best of them to show skill.
    Fed only real moves (see `is_move`), "always neutral" scores zero and the contest is bull versus bear.
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


def judge(k: int, n: int, baseline: float, baseline_n: int = 0) -> tuple[str, float, float]:
    """-> (verdict, low, high). BEATS only if the interval's LOWER end clears the baseline.

    The baseline is itself measured on `baseline_n` outcomes, so it is uncertain too: calling a model WORSE because
    it sits under a baseline that may really be lower is overconfident. WORSE needs the model's HIGH end to be
    below the baseline's LOW end. (`baseline_n` 0 means the baseline is exactly known.)
    """
    if n < MIN_JUDGED:
        return TOO_FEW, 0.0, 1.0
    low, high = wilson(k, n)
    if low > baseline:
        return BEATS, low, high
    baseline_low = wilson(round(baseline * baseline_n), baseline_n)[0] if baseline_n else baseline
    if high < baseline_low:
        return WORSE, low, high
    return CANT_TELL, low, high


def stage_verdict(k: int, n: int, baseline: float, baseline_n: int = 0, *, final: bool) -> str:
    """What to do with a model after a stage of the staged test: BEATS (proven: stop), WORSE (hopeless: stop spending
    on it), or keep going; and at the final stage, a definite answer instead of "keep going".

    Interim looks use the 99% interval (see Z99), the final look the usual 95%. The baseline is itself measured, so
    WORSE needs the model's high end below the baseline's LOW end, exactly as in `judge`. At the final stage a model
    that does not beat the baseline is NO_EDGE (nothing shows it is better than a free constant strategy) only if
    there is enough data to say so; with less it is CANT_TELL, never a guess.
    """
    z = Z95 if final else Z99
    if n < MIN_JUDGED:
        return TOO_FEW if final else KEEP
    low, high = wilson(k, n, z)
    if low > baseline:
        return BEATS
    baseline_low = wilson(round(baseline * baseline_n), baseline_n, z)[0] if baseline_n else baseline
    if high < baseline_low:
        return WORSE
    if not final:
        return KEEP
    return NO_EDGE if n >= judged_needed() else CANT_TELL
