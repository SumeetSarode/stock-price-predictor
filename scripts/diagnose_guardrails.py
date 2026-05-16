"""Diagnose WHY guardrails keep tripping for a ticker.

Runs the gather phase once, then calls the synthesizer N times (independent
LLM samples). For each attempt, prints the direction/confidence/level math
plus which guardrail tier (if any) it trips. Lets us see whether failures
are systematic (always same tier, always same direction) or stochastic
(different reasons each time).

Usage:
    uv run python scripts/diagnose_guardrails.py RELIANCE.NS daily 3
"""
from __future__ import annotations

import asyncio
import sys
import warnings

warnings.filterwarnings("ignore")

from loguru import logger
logger.remove()  # silence the noisy default sink

from price_predictor.prediction.inputs import (
    SynthesisInput,
    compose_technical_view,
)
from price_predictor.prediction.predictor import (
    _gather_phase,
    run_synthesizer_agent,
)
from price_predictor.prediction.schema import PredictionHorizon
from price_predictor.prediction.guardrails import (
    HallucinationError,
    validate_grounding,
    validate_citations,
    validate_consistency,
    validate_calibration,
)
from datetime import datetime, timezone


async def main(ticker: str, horizon_str: str, n_attempts: int) -> None:
    horizon = PredictionHorizon(horizon_str)

    print(f"\n=== Gather phase for {ticker} (one-time) ===")
    tv, ia, news_status = await _gather_phase(ticker, "standard")
    signals = [tv.trend.signal, tv.momentum.signal, tv.volatility.signal, tv.levels.signal]
    bull = signals.count("bullish")
    bear = signals.count("bearish")
    neutral = signals.count("neutral")
    print(f"  Close:        ₹{tv.close_price:,.2f}")
    print(f"  Tally:        bullish={bull}  bearish={bear}  neutral={neutral}")
    print(f"  Per cluster:  trend={tv.trend.signal} | mom={tv.momentum.signal} | "
          f"vol={tv.volatility.signal} | lvl={tv.levels.signal}")
    print(f"  News:         sentiment={ia.sentiment}  conf={ia.confidence:.2f}  "
          f"status={news_status}")
    print(f"  Catalysts:    {len(ia.catalysts)}")
    print()
    print(f"  Guardrail-legal directions:")
    bullish_ok = bull >= 2 or (bull >= 1 and ia.sentiment == "bullish")
    bearish_ok = bear >= 2 or (bear >= 1 and ia.sentiment == "bearish")
    print(f"    BULLISH: {'✅' if bullish_ok else '❌'}")
    print(f"    BEARISH: {'✅' if bearish_ok else '❌'}")
    print(f"    NEUTRAL: ✅ (always)")
    print()

    si = SynthesisInput(
        ticker=ticker,
        horizon=horizon.value,
        as_of=datetime.now(timezone.utc),
        technical_view=tv,
        impact_assessment=ia,
        model_chain=("test",),
    )

    print(f"=== Running synthesizer {n_attempts}× (horizon={horizon_str}) ===\n")
    for i in range(1, n_attempts + 1):
        print(f"--- Attempt {i} ---")
        try:
            pred = await run_synthesizer_agent(si)
        except Exception as e:
            print(f"  ❌ synth call itself failed: {type(e).__name__}: {e}")
            print()
            continue

        print(f"  Direction:    {pred.direction.value}")
        print(f"  Confidence:   {pred.confidence:.2f}")
        print(f"  Entry zone:   ₹{pred.entry_zone[0]:,.2f} – ₹{pred.entry_zone[1]:,.2f}")
        print(f"  Target:       ₹{pred.target.value:,.2f}")
        print(f"  Stop:         ₹{pred.stop_loss.value:,.2f}")

        # Run each tier independently so we know which one fires (or all pass).
        tiers = [
            ("1-grounding", validate_grounding),
            ("2-citations", validate_citations),
            ("3-consistency", validate_consistency),
            ("4-calibration", validate_calibration),
        ]
        all_pass = True
        for name, fn in tiers:
            try:
                fn(pred, si)
                print(f"  Tier {name}: ✅")
            except HallucinationError as e:
                print(f"  Tier {name}: ❌ {e}")
                all_pass = False
        if all_pass:
            print(f"  ✅ ALL TIERS PASS — this prediction would be ACCEPTED")
        print()


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    horizon = sys.argv[2] if len(sys.argv) > 2 else "daily"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    asyncio.run(main(ticker, horizon, n))
