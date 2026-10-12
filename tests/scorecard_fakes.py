"""Shared fakes for the scorecard tests: a fake market, fake predictions, a fake world that records calls."""
from __future__ import annotations

import asyncio
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import model_survey_offvpn as S
import scorecard_core as SC
import sweep_data as D
import sweep_probe as PR
from price_predictor.prediction.grading import grade_many
from price_predictor.prediction.schema import (
    AnalysisBasis,
    Prediction,
    PredictionDirection,
    PredictionHorizon,
    PriceLevel,
)

IST = ZoneInfo("Asia/Kolkata")

BULL, BEAR, NEUTRAL = PredictionDirection.BULLISH, PredictionDirection.BEARISH, PredictionDirection.NEUTRAL

GROQ_KEY = "gsk_SCORECARDFAKEKEY0000000000"

RealPlumbing = SC.Plumbing      # captured before any test patches it

def prediction(ticker="UP.NS", day=date(2026, 7, 20), horizon=PredictionHorizon.WEEKLY, direction=BULL, close=200.0):
    target, stop = {BULL: (1.05, 0.97), BEAR: (0.95, 1.03)}.get(direction, (1.02, 0.98))
    return Prediction(
        ticker=ticker, as_of=datetime(day.year, day.month, day.day, 15, 30, tzinfo=IST), horizon=horizon,
        model_chain=("synthesizer:agentic",), direction=direction, confidence=0.7,
        entry_zone=(close * 0.995, close * 1.005),
        target=PriceLevel(value=close * target, rationale="x" * 20), stop_loss=PriceLevel(value=close * stop, rationale="x" * 20),
        rationale="x" * 60, contributing_signals=("trend",), conflicting_signals=(),
        analysis_basis=AnalysisBasis(close_price_at_prediction=close, bars_used=400, technical_summary="t",
                                     news_sentiment_score=0.5, news_articles_considered=3, filings_considered=0))

def bars(sign: int, close=200.0, n=30):
    closes = [close * (1.0 + 0.01 * sign * (i + 1)) for i in range(n)]
    return pd.DataFrame({"open": closes, "high": [c * 1.005 for c in closes], "low": [c * 0.995 for c in closes],
                         "close": closes, "volume": [1_000_000] * n})

def fetch(ticker, start, end):
    """UP.NS always rises, DOWN.NS always falls: a market where direction is knowable."""
    return bars(+1 if ticker.startswith("UP") else -1)

def grader(preds):
    return grade_many(preds, fetch_ohlcv=fetch, today=date(2026, 9, 30))

def cand(model="m", provider="groq", route="direct", **kw):
    return D.Candidate(route, provider, model, **kw)

class Clock:
    def __init__(self):
        self.t, self.sleeps = 0.0, []

    def now(self):
        return self.t

    async def sleep(self, s):
        self.t += s
        self.sleeps.append(s)

class World:
    """Fake collaborators that record what was asked of them."""

    def __init__(self, direction=BULL, horizons=(PredictionHorizon.WEEKLY,)):
        self.shared_calls, self.predict_calls, self.fail, self.bad_inputs = [], [], {}, set()
        self.direction, self.horizons, self.clock = direction, horizons, Clock()

    async def shared(self, case):
        self.shared_calls.append(case.key)
        if case.key in self.bad_inputs:
            raise RuntimeError("yfinance is down")
        return f"tv:{case.key}"

    async def predict(self, c, case, tv):
        self.predict_calls.append((c.model, case.key, tv))
        if (c.model, case.key) in self.fail:
            raise self.fail[(c.model, case.key)]
        d = self.direction(c, case) if callable(self.direction) else self.direction
        return [prediction(case.ticker, case.as_of, h, d) for h in self.horizons]

    def plumbing(self, pace=0.0):
        return RealPlumbing(self.shared, self.predict, lambda p: pace, S.classify, self.clock.sleep, self.clock.now,
                            lambda m: None)

def cases3():
    return [SC.Case("UP.NS", date(2026, 7, 1)), SC.Case("DOWN.NS", date(2026, 7, 2)), SC.Case("UP.NS", date(2026, 7, 3))]

def go(models, cases, store, world, budget=None, calls=6, deadline=float("inf"), pace=0.0):
    asyncio.run(SC.run_scorecard(models, cases, store, budget or PR.Budget({}, 999), world.plumbing(pace),
                                 calls_per_case=calls, deadline=deadline))

def run_models(tmp_path, models, direction, n_cases=12):
    cases = [SC.Case("UP.NS" if i % 2 == 0 else "DOWN.NS", date(2026, 7, 1) + timedelta(days=i)) for i in range(n_cases)]
    w, store = World(direction=direction), SC.CaseStore(tmp_path / "s.jsonl")
    go(models, cases, store, w)
    return store, SC.aggregate(models, store, grader)

def oracle(c, case):
    right = BULL if case.ticker.startswith("UP") else BEAR
    wrong = BEAR if right is BULL else BULL
    return {"oracle": right, "anti": wrong}.get(c.model, BULL)        # anything else: always bullish
