"""Network-free reproduction of the 'everything is neutral' bug.

ROOT CAUSE
==========
yfinance returns a row for TODAY's in-progress bar with close=NaN.
Every technical cluster reads df['close'].iloc[-1], so it grabs that
NaN, computes close=None + all SMAs=None, and classify_trend()
short-circuits to ('neutral', 'weak', ['Insufficient price history']).

All 4 clusters go neutral -> the consistency guardrail (needs 2-of-4
agreeing for a directional call) forbids bullish/bearish -> every
prediction is forced NEUTRAL -> neutral is the ONLY direction where the
schema allows target-inside-entry (your 'target 2242 inside 2225-2260').

This script needs NO network. It builds a synthetic, unambiguous
uptrend, appends the poisoned trailing NaN bar, and shows:
    BEFORE fix -> neutral   (bug reproduced)
    AFTER  fix -> bullish   (trailing NaN row dropped)

Run:  .venv/bin/python scripts/repro_neutral_bug.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from price_predictor.analysis import TREND_PRESETS
from price_predictor.analysis.trend import trend_snapshot
from price_predictor.agents.technical_agent.tools._trend_signal import classify_trend


def _preset_kwargs(preset: dict) -> dict:
    """Map TREND_PRESETS entry -> trend_snapshot kwargs."""
    return {
        "sma_lengths": preset["sma"],
        "ema_length": preset["ema"],
        "adx_length": preset["adx"],
    }


def make_uptrend(n: int = 300) -> pd.DataFrame:
    """A clean, monotonic uptrend. Any honest classifier must call this bullish."""
    idx = pd.date_range("2025-01-01", periods=n, freq="B", tz="Asia/Kolkata")
    close = np.linspace(1000.0, 2000.0, n)          # steady climb
    high = close * 1.01
    low = close * 0.99
    open_ = close * 0.999
    vol = np.full(n, 1_000_000)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low,
         "close": close, "adj_close": close, "volume": vol},
        index=pd.Index(idx, name="Date"),
    )


def poison_with_todays_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Append a trailing in-progress bar with close=NaN, exactly like yfinance."""
    last = df.index[-1]
    today = last + pd.Timedelta(days=1)
    row = pd.DataFrame(
        {"open": [2000.0], "high": [2020.0], "low": [1990.0],
         "close": [np.nan], "adj_close": [np.nan], "volume": [0]},
        index=pd.Index([today], name="Date"),
    )
    return pd.concat([df, row])


def drop_incomplete_trailing_bars(df: pd.DataFrame) -> pd.DataFrame:
    """THE FIX: strip trailing rows whose close is NaN (in-progress bars)."""
    if df.empty:
        return df
    # Drop from the tail inward while close is NaN.
    mask = df["close"].notna()
    if mask.all():
        return df
    last_valid = mask[::-1].idxmax()  # most recent row with a real close
    return df.loc[:last_valid]


def classify(df: pd.DataFrame) -> tuple:
    snap = trend_snapshot(df, **_preset_kwargs(TREND_PRESETS["standard"]))
    return classify_trend(snap)


def main() -> None:
    clean = make_uptrend()
    poisoned = poison_with_todays_bar(clean)
    fixed = drop_incomplete_trailing_bars(poisoned)

    print("=" * 62)
    print("REPRO: 'everything is neutral' bug (no network required)")
    print("=" * 62)
    print(f"clean bars={len(clean)}  poisoned bars={len(poisoned)}  "
          f"fixed bars={len(fixed)}")
    print(f"last close (poisoned iloc[-1]): {poisoned['close'].iloc[-1]}  "
          f"<- the NaN yfinance today-bar\n")

    sig_clean, str_clean, rat_clean = classify(clean)
    sig_bug, str_bug, rat_bug = classify(poisoned)
    sig_fix, str_fix, rat_fix = classify(fixed)

    print(f"[BASELINE  clean uptrend ] signal={sig_clean:8s} strength={str_clean}")
    print(f"[BUG  poisoned w/ NaN bar] signal={sig_bug:8s} strength={str_bug}")
    print(f"     rationale: {rat_bug}")
    print(f"[FIX  trailing NaN dropped] signal={sig_fix:8s} strength={str_fix}")
    print(f"     rationale[:2]: {rat_fix[:2]}\n")

    ok = (
        sig_clean == "bullish"
        and sig_bug == "neutral"
        and sig_fix == "bullish"
    )
    if ok:
        print("RESULT:  BUG CONFIRMED and FIX VALIDATED.")
        print("  - clean uptrend            -> bullish (sanity)")
        print("  - trailing NaN today-bar   -> NEUTRAL  (this is the bug)")
        print("  - drop trailing NaN bar    -> bullish  (this is the fix)")
        raise SystemExit(0)
    print("RESULT:  did not reproduce as expected — inspect output above.")
    print(f"  clean={sig_clean} bug={sig_bug} fix={sig_fix}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
