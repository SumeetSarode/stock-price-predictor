"""Trend signal classifier -- pure function, no I/O.

Input: trend_snapshot output (close, sma dict, ema, adx dict, derived,
ma_crosses dict).
Output: (signal, strength, rationale_bullets).

Kept separate from get_trend.py so we can unit-test the classification
rules without mocking fetch/cache.

VOTE INTEGRATION
================
The classifier folds three discrete signals:
  1. SMA stack alignment (existing)
  2. ADX-gated DI direction  (existing)
  3. Fresh MA crossovers     (NEW)

For (3), only crosses with `bars_since_event <= MA_CROSS_FRESH_BARS`
contribute to the verdict. Stale crosses appear in the rationale text
("regime has been bullish for 47 bars") but do not vote.

Vote weights — deliberately conservative because empirical literature
(Sullivan-Timmermann-White 1999, Zakamulin 2014) shows MA-crossover
alpha on liquid large-caps is marginal post-1990:
  - SMA-50/200 fresh cross: ±0.5
  - EMA-9/21 fresh cross:   ±0.3   (faster pair = more whipsaws)

🔬 NEEDS BACKTEST against NSE-specific data.
"""
from __future__ import annotations

from price_predictor.agents.technical_agent.tools._types import Signal, Strength

# ADX thresholds for trend strength (TA convention)
ADX_STRONG = 40
ADX_MODERATE = 25
ADX_WEAK_FLOOR = 20

# MA-cross vote weights (see module docstring for sourcing)
MA_CROSS_FRESH_BARS = 5         # cross is "fresh" within ~1 trading week
MA_CROSS_VOTE_SMA_50_200 = 0.5  # canonical Golden / Death Cross
MA_CROSS_VOTE_EMA_9_21 = 0.3    # faster swing-trader pair

# Pretty-print label for the canonical Golden Cross pair only.
# Other pairs get generic "{kind}-{short}/{long}" framing.
_GOLDEN_CROSS_KEY = "sma_50_200"


def _stack_score(above_sma: dict[int, bool | None]) -> int:
    """Count how many SMAs the close is above. None == 0 (uncountable)."""
    return sum(1 for v in above_sma.values() if v is True)


def _adx_strength(adx: float | None) -> Strength:
    """Map ADX value to strength label.

    >=40 strong, >=25 moderate, otherwise weak (no real trend).
    """
    if adx is None:
        return "weak"
    if adx >= ADX_STRONG:
        return "strong"
    if adx >= ADX_MODERATE:
        return "moderate"
    return "weak"


def _cross_label(key: str, event: str) -> str:
    """Render a human-readable cross label.

    `sma_50_200` + `bullish` -> 'Golden Cross'
    `sma_50_200` + `bearish` -> 'Death Cross'
    Other pairs -> 'bullish EMA-9/21 cross' / 'bearish SMA-20/50 cross' etc.
    """
    if key == _GOLDEN_CROSS_KEY:
        return "Golden Cross" if event == "bullish" else "Death Cross"
    # Generic: 'sma_50_200' -> 'SMA-50/200'
    parts = key.split("_")
    if len(parts) == 3:
        kind, short, long = parts
        return f"{event} {kind.upper()}-{short}/{long} cross"
    return f"{event} {key} cross"


def _cross_vote_weight(key: str) -> float:
    """Look up the vote weight for an MA-cross pair.

    Unknown pairs return 0 (cited in rationale, do not vote).
    """
    if key == "sma_50_200":
        return MA_CROSS_VOTE_SMA_50_200
    if key == "ema_9_21":
        return MA_CROSS_VOTE_EMA_9_21
    return 0.0


def _ma_cross_vote(
    ma_crosses: dict[str, dict[str, object]],
) -> tuple[float, list[str]]:
    """Compute net cross vote and rationale bullets.

    Returns:
        (net_vote, rationale_bullets) where net_vote is positive for
        bullish, negative for bearish, zero if no fresh cross fired or
        crosses cancelled out.

    Stale crosses (bars_since_event > MA_CROSS_FRESH_BARS) contribute a
    rationale bullet ("regime has been bullish for N bars") but do NOT
    vote -- empirically the EVENT-edge decays in days while the regime
    info is already captured by the SMA stack score.
    """
    rationale: list[str] = []
    net_vote = 0.0

    for key, struct in ma_crosses.items():
        event = struct.get("last_event")
        bars_since = struct.get("bars_since_event")
        current = struct.get("current")

        # No cross ever observed in the available history.
        if event is None or bars_since is None:
            if current is not None:
                rationale.append(
                    f"No {key.replace('_', '-')} cross in available history "
                    f"(currently {current})"
                )
            continue

        label = _cross_label(key, str(event))
        if isinstance(bars_since, int) and bars_since <= MA_CROSS_FRESH_BARS:
            # Fresh -> votes
            weight = _cross_vote_weight(key)
            sign = 1.0 if event == "bullish" else -1.0
            net_vote += sign * weight
            bars_label = "today" if bars_since == 0 else (
                "1 bar ago" if bars_since == 1 else f"{bars_since} bars ago"
            )
            rationale.append(f"{label} fired {bars_label}")
        else:
            # Stale -> rationale only
            rationale.append(
                f"{label} regime since {bars_since} bars ago "
                f"(stale; not voting)"
            )

    return net_vote, rationale


def classify_trend(snapshot: dict) -> tuple[Signal, Strength, list[str]]:
    """Turn a trend_snapshot into (signal, strength, rationale).

    RULES
    =====
    Signal:
      bullish  -- close above >=2 of the 3 SMAs AND +DI > -DI
      bearish  -- close below >=2 of the 3 SMAs AND -DI > +DI
      neutral  -- everything else (mixed signals or insufficient data)

    Fresh MA crossovers can NUDGE a neutral verdict bullish / bearish
    when the cross-vote magnitude is >= 0.5 and DI direction does not
    contradict (see _ma_cross_vote for weights and freshness window).
    They do NOT override an existing bullish / bearish verdict; if SMA
    stack and DI already agree, the verdict is locked.

    Strength: derived from ADX value alone (independent of direction).
      A strong-but-mixed setup is still 'neutral' for signal but reports
      its ADX strength. A clear direction with low ADX is 'bullish'/'bearish'
      but with 'weak' strength (the trend hasn't asserted itself yet).

    Rationale: human-readable bullets explaining each component. The LLM
    quotes these to avoid inventing reasoning.
    """
    rationale: list[str] = []

    close = snapshot.get("close")
    sma = snapshot.get("sma", {})
    above = snapshot.get("above_sma", {})
    pct_above = snapshot.get("pct_above_sma", {})
    adx_dict = snapshot.get("adx", {})
    adx_val = adx_dict.get("adx")
    di_plus = adx_dict.get("di_plus")
    di_minus = adx_dict.get("di_minus")
    ma_crosses = snapshot.get("ma_crosses", {})

    # ── Insufficient-data short circuit ─────────────────────────
    if close is None or all(v is None for v in sma.values()):
        return "neutral", "weak", ["Insufficient price history for trend analysis"]

    # ── SMA stack ───────────────────────────────────────────────
    score = _stack_score(above)
    total_smas = sum(1 for v in above.values() if v is not None)
    sma_lengths_sorted = sorted(sma.keys())

    above_list = [n for n in sma_lengths_sorted if above.get(n) is True]
    below_list = [n for n in sma_lengths_sorted if above.get(n) is False]
    if above_list:
        rationale.append(
            f"Close above SMA-{', SMA-'.join(str(n) for n in above_list)}"
        )
    if below_list:
        rationale.append(
            f"Close below SMA-{', SMA-'.join(str(n) for n in below_list)}"
        )

    # Add the most informative pct distance (longest SMA we have)
    if pct_above:
        longest = max(pct_above.keys())
        if pct_above[longest] is not None:
            sign = "+" if pct_above[longest] >= 0 else ""
            rationale.append(
                f"Close is {sign}{pct_above[longest]:.2f}% relative to SMA-{longest}"
            )

    # ── ADX direction & strength ────────────────────────────────
    di_direction: str | None = None
    if di_plus is not None and di_minus is not None:
        if di_plus > di_minus:
            di_direction = "bullish"
            rationale.append(
                f"+DI ({di_plus:.1f}) > -DI ({di_minus:.1f}): bullish directional bias"
            )
        elif di_minus > di_plus:
            di_direction = "bearish"
            rationale.append(
                f"-DI ({di_minus:.1f}) > +DI ({di_plus:.1f}): bearish directional bias"
            )

    strength = _adx_strength(adx_val)
    if adx_val is not None:
        if strength == "strong":
            rationale.append(f"ADX {adx_val:.1f} indicates a strong trend")
        elif strength == "moderate":
            rationale.append(f"ADX {adx_val:.1f} indicates a developing trend")
        else:
            rationale.append(f"ADX {adx_val:.1f} indicates choppy/range-bound action")

    # ── MA crossovers (fresh -> vote, stale -> rationale only) ──
    cross_vote, cross_rationale = _ma_cross_vote(ma_crosses)
    rationale.extend(cross_rationale)

    # ── Combine into final signal ──────────────────────────────
    # Need majority of SMAs aligned AND DI direction agreeing.
    if total_smas == 0:
        signal: Signal = "neutral"
    elif score >= max(2, total_smas - 1) and di_direction == "bullish":
        signal = "bullish"
    elif score <= min(1, total_smas - 2) and di_direction == "bearish":
        signal = "bearish"
    elif score == total_smas and di_direction != "bearish":
        # All SMAs above, even if DI is unavailable -> still bullish
        signal = "bullish"
    elif score == 0 and di_direction != "bullish":
        signal = "bearish"
    else:
        signal = "neutral"

    # Cross-vote nudges neutral verdicts only -- never overrides a
    # locked-in bullish/bearish from SMA stack + DI agreement.
    if signal == "neutral" and abs(cross_vote) >= 0.5:
        if cross_vote >= 0.5 and di_direction != "bearish":
            signal = "bullish"
            rationale.append(
                f"Verdict nudged bullish by fresh MA cross vote ({cross_vote:+.1f})"
            )
        elif cross_vote <= -0.5 and di_direction != "bullish":
            signal = "bearish"
            rationale.append(
                f"Verdict nudged bearish by fresh MA cross vote ({cross_vote:+.1f})"
            )

    return signal, strength, rationale
