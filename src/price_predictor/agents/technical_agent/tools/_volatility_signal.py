"""Volatility signal classifier -- pure function, no I/O.

WHY VOLATILITY HAS A 'SIGNAL'
=============================
Volatility is direction-agnostic in classical TA. But for the LLM's
benefit we still emit a bullish/neutral/bearish signal based on
position WITHIN the Bollinger Band:
  - %B > 0.5: price in upper half of band (bullish location)
  - %B < 0.5: price in lower half of band (bearish location)

Strength is the more important field here:
  - 'strong': TTM Squeeze active OR fired this bar (Carter 2009 trigger).
  - 'moderate': normal volatility regime
  - 'weak': dead-quiet OR violently chaotic (neither tradeable)

WHY TTM, NOT BOLLINGER, FOR THE STRENGTH BUMP
=============================================
The Bollinger bandwidth-percentile squeeze (Bollinger 2001) and the TTM
Squeeze (Carter 2009) measure different things. TTM requires BOTH low
BB width AND low ATR-implied channel width — a stricter, less false-
positive-prone trigger. We use TTM as the strength bump and surface the
Bollinger flag for diagnostics.
"""
from __future__ import annotations

from price_predictor.agents.technical_agent.tools._types import Signal, Strength

# %B thresholds for direction classification.
# 🔬 NEEDS BACKTEST (M7). Our own design choice — a 5-percentage-point
# dead zone around %B = 0.5 to avoid flipping direction on every bar
# that grazes the median. Bollinger (2001) only labels 0.0 / 1.0
# (band touches); the 0.55 / 0.45 inner pair is editorial.
# See pred_logic.md §4.3 for the full attribution.
PCT_B_BULLISH = 0.55
PCT_B_BEARISH = 0.45

# ATR-as-%-of-price thresholds for regime classification.
# 🔬 NEEDS BACKTEST (M7). Our own design choice — anchored to the
# empirical observation that NSE large-caps spend most days in the
# 1–3% ATR-% range. No published source for the specific cutoffs;
# Phase 2 backtest should test 0.8/3.5/5.0 (tighter), 1.5/5.0/7.0
# (looser), and per-stock percentile-rank variants.
# See pred_logic.md §4.3 for the full attribution.
ATR_PCT_DEAD_QUIET = 1.0   # below = dead quiet (untradeable)
ATR_PCT_NORMAL_LOW = 1.0
ATR_PCT_NORMAL_HIGH = 4.0
ATR_PCT_MANIC = 6.0        # above = chaotic (avoid)


def classify_volatility(
    snapshot: dict,
) -> tuple[Signal, Strength, list[str], list[str]]:
    """Turn a volatility_snapshot into (signal, strength, rationale, warnings).

    Args:
        snapshot: output of analysis.volatility.volatility_snapshot()
                  shape: {atr, atr_pct_of_price, bbands: {...},
                          bollinger_squeeze: bool|None,
                          ttm_squeeze: {on, fire, bars_in_squeeze}}
    """
    rationale: list[str] = []
    warnings: list[str] = []

    atr = snapshot.get("atr")
    atr_pct = snapshot.get("atr_pct_of_price")
    bbands = snapshot.get("bbands", {})
    boll_squeeze = snapshot.get("bollinger_squeeze")
    ttm = snapshot.get("ttm_squeeze") or {}
    ttm_on = ttm.get("on")
    ttm_fire = ttm.get("fire")
    ttm_bars = ttm.get("bars_in_squeeze")

    pct_b = bbands.get("percent_b")
    bandwidth = bbands.get("bandwidth")
    upper = bbands.get("upper")
    lower = bbands.get("lower")
    middle = bbands.get("middle")

    # ── Insufficient-data short circuit ─────────────────────────
    if atr is None and pct_b is None:
        return (
            "neutral",
            "weak",
            ["Insufficient history for volatility analysis"],
            ["insufficient_history"],
        )

    # ── Direction signal from %B position ───────────────────────
    if pct_b is None:
        signal: Signal = "neutral"
    elif pct_b > PCT_B_BULLISH:
        signal = "bullish"
        rationale.append(
            f"BB %B = {pct_b:.2f} (price in upper half of band: bullish location)"
        )
    elif pct_b < PCT_B_BEARISH:
        signal = "bearish"
        rationale.append(
            f"BB %B = {pct_b:.2f} (price in lower half of band: bearish location)"
        )
    else:
        signal = "neutral"
        rationale.append(
            f"BB %B = {pct_b:.2f} (price near middle band: no directional bias)"
        )

    # ── %B extreme warnings (touching/exceeding bands) ──────────
    if pct_b is not None:
        if pct_b > 1.0:
            warnings.append("price_above_upper_band")
            rationale.append(f"⚠ Price above upper band (%B = {pct_b:.2f}): stretched")
        elif pct_b < 0.0:
            warnings.append("price_below_lower_band")
            rationale.append(f"⚠ Price below lower band (%B = {pct_b:.2f}): stretched")

    # ── ATR rationale ───────────────────────────────────────────
    if atr is not None and atr_pct is not None:
        rationale.append(
            f"ATR-14 = {atr:.2f} ({atr_pct:.2f}% of price)"
        )

    # ── Strength: TTM squeeze fire/on > regime ───────────────────
    # The TTM Squeeze (BB inside Keltner) is the highest-priority signal:
    # FIRE = squeeze just released (Carter's trade trigger), ON = coiling.
    # Both bump strength to "strong" regardless of current direction.
    if ttm_fire is True:
        strength: Strength = "strong"
        rationale.append(
            f"⚡ TTM SQUEEZE FIRED this bar after {ttm_bars or 0} bars of "
            "compression — Carter (2009) breakout trigger"
        )
    elif ttm_on is True:
        strength = "strong"
        rationale.append(
            f"⚡ TTM SQUEEZE active ({ttm_bars or 0} bars): BB inside Keltner "
            "channels — breakout likely incoming"
        )
    elif atr_pct is None:
        strength = "weak"
    elif atr_pct < ATR_PCT_DEAD_QUIET:
        strength = "weak"
        rationale.append(
            f"Volatility dead-quiet (ATR < {ATR_PCT_DEAD_QUIET}% of price): "
            "low conviction, hard to trade"
        )
    elif atr_pct > ATR_PCT_MANIC:
        strength = "weak"
        warnings.append("high_volatility")
        rationale.append(
            f"Volatility manic (ATR > {ATR_PCT_MANIC}% of price): "
            "wide stops needed, choppy"
        )
    elif ATR_PCT_NORMAL_LOW <= atr_pct <= ATR_PCT_NORMAL_HIGH:
        strength = "moderate"
    else:
        strength = "weak"  # 4-6% range: elevated but not manic

    # ── Bandwidth rationale + Bollinger-squeeze diagnostic ─────────
    if bandwidth is not None and ttm_on is not True and ttm_fire is not True:
        rationale.append(f"BB bandwidth = {bandwidth:.2f}% of middle band")
    if boll_squeeze is True and ttm_on is not True and ttm_fire is not True:
        # The Bollinger percentile flag is firing but TTM isn't — trending
        # market with calm BB width. Surface it as a soft signal only.
        rationale.append(
            "Bollinger bandwidth in lowest 20% historically (no TTM squeeze)"
        )

    return signal, strength, rationale, warnings


def classify_volatility_regime(atr_pct: float | None) -> str:
    """Classify the volatility regime as 'low' | 'normal' | 'high' | 'unknown'.

    Surfaced separately in the tool's `derived` block so the LLM can quote
    the regime explicitly without re-deriving it.
    """
    if atr_pct is None:
        return "unknown"
    if atr_pct < ATR_PCT_DEAD_QUIET:
        return "low"
    if atr_pct > ATR_PCT_MANIC:
        return "high"
    if atr_pct > ATR_PCT_NORMAL_HIGH:
        return "high"
    return "normal"
