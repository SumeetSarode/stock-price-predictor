# Next Steps — Stock Predictor v1

> **Scope**: Detailed plan for **upcoming work**. For "what's done already"
> see `implementation_flow.md`. For the higher-level project roadmap see
> `implementation_plan.md`.
>
> **Last updated**: 2026-04-28 — post Step 3.5 (grading + calibration shipped).
> Previous update: post-C Provider Expansion.

---

## 🟢 Where we are right now

**The v1 prediction loop is closed end-to-end:**

  prediction → persistence → grading → calibration

| Surface | Shipped? |
|---|---|
| `predict TICKER` | ✅ |
| `predict-many T1 T2 ...` | ✅ |
| `history TICKER` | ✅ |
| `grade` | ✅ |
| `calibration [--by horizon\|ticker\|direction\|month]` | ✅ |

- **854 unit tests passing** (+ 7 integration tests deselected, run off-corp)
- **5 ADK agents** live (`hello`, `price`, `news_impact`, `technical`, `synthesizer`)
- **All v1 data layers** shipped (prices chain, news, estimates, filings, KB)
- **All v1 analysis primitives** shipped (trend / momentum / volatility / levels / patterns)

What's NOT yet shipped for v1:
- Backtest replay + runner + evaluator (3.5.5 → 3.5.7)
- Concurrency + rate-limit-aware LLM router (3.6)
- LightRAG knowledge layer (Phase 2)

---

## 🚦 Four reasonable next moves

The v1 loop closes nicely; nothing forces our hand. Pick based on what
provides the most signal vs effort right now.

### Option A — Step 3.5.5 → 3.5.7: backtest replay + runner + evaluator 🎯 (recommended)

**Why**: Calibration today only works on real-elapsed-time predictions. We
have no way to answer the most interesting question: *"would this system
have made money over the last 2 years?"* Backtest closes that loop.

**Components needed**:
- `backtest/replay.py` — as-of-date data shim. "Give me prices/news/filings
  AS THEY WOULD HAVE LOOKED on date X." **Critical** for honest backtest;
  any leak of future info inflates results.
- `backtest/runner.py` — historical loop over dates, calling `predict()`
  with the replay shim active.
- `backtest/evaluator.py` — composes calibration metrics across runs
  (per-month, per-regime). Reuses `compute_breakdown()` from 3.5.2.

**Open design questions to resolve first** (worth 30 min before writing code):
- How do we honestly replay GDELT? News articles published AFTER our
  as-of-date must NOT be visible. Cache snapshots? Filter by `published_at`?
- How do we handle survivorship bias in the Nifty50 list? (Today's 50
  ≠ the 50 from 5 years ago.) Use historical NIFTY50 constituent data?
- Do we replay at end-of-day cadence or hourly? (EoD is simpler; hourly
  needs intraday data which we mostly don't have.)
- Do we re-run the LLM for every historical date, or cache synthesizer
  outputs? (Re-run is honest but expensive; caching is cheap but masks
  prompt-version effects.)

**Estimated commits**: 5-7
1. `backtest/replay.py` design + tests with synthetic data
2. Replay-aware fetcher integration
3. `backtest/runner.py` core loop
4. Survivorship-bias handling
5. `backtest/evaluator.py` (composes existing CalibrationReport)
6. CLI surface (`price-predictor backtest START_DATE END_DATE [--tickers ...]`)
7. End-to-end smoke test on 6 months of one ticker

**Acceptance criteria**:
- [ ] Replay shim never leaks future data (test with `--as-of` cutoffs)
- [ ] Backtest of 30 days × 3 tickers completes in <5min on a laptop
- [ ] `backtest` CLI command produces a CalibrationReport identical in shape
      to live `calibration` (so the same renderer works)
- [ ] Survivorship-bias handling documented in flow doc
- [ ] 30-50 unit tests + 1 integration test

---

### Option B — Step 3.6: concurrency & rate-limit-aware router

**Why**: `predict_many` works but the LLM rate-limit handling is naive.
Production-scale runs (Nifty50 in ~5 min target) need a smart router that
falls back Groq → Gemini on rate limit and respects per-provider caps.

**Components needed**:
- `concurrency/runner.py` — semaphore-based bounded concurrency at the
  HTTP + LLM layer (already exists in spec)
- Rate-limit-aware LLM router — wraps `llm/factory.py`. On 429, fall back
  to next provider; on persistent 429, exponential backoff.
- Token budgeting — track tokens-per-minute across all in-flight calls.

**Estimated commits**: 3-5

**Why NOT first**: `predict_many` already works for 5-10 tickers. We don't
have backtest yet, so we can't actually run the system at full Nifty50
scale meaningfully. **Do this AFTER backtest** — backtest will surface the
real rate-limit pain.

---

### Option C — Step 4.1: LightRAG knowledge layer (Phase 2)

**Why**: Persistent retrieval over the news/filings corpus so agents can
cite historical context: *"this looks like the Q3 2024 announcement that
moved the stock 8%."* Big quality bump for predictions, not a v1 must-have.

**Components needed**:
- `kb/interface.py` — abstract KnowledgeBase ABC (currently we have only
  the structured stocks.json)
- `kb/lightrag_layer.py` — LightRAG-backed implementation
- Bootstrap script to ingest historical news + filings
- Synthesizer prompt update to use retrieval

**Estimated commits**: 5-7

**Why NOT first**: Phase 2 work. v1 should be backtest-validated before
we add a Phase 2 quality layer; otherwise we can't tell if LightRAG
actually helped or just changed numbers.

---

### Option D — Code-tour learning detour (no new code)

**Why**: With 854 tests and 5 agents shipped, we now have a substantial
ADK + agentic codebase. A focused walk through the synthesizer + predictor
+ runner trio (~30 min, zero LOC) builds the mental model needed to design
the backtest replay layer well. Backtest will need to interact with the
predictor in non-trivial ways; understanding it deeply pays off.

**Why NOT permanent**: It's preparation, not progress. Time-box to 30 min,
then move to A.

---

## 🎯 Recommendation

**Do A (backtest), but lead with a 30-min Option D walk-through** of the
predict pipeline first to refresh the mental model. Backtest will need the
predictor to be replay-shim-aware, and the cleanest insertion point only
becomes obvious once you've re-traced predict() end-to-end.

**Skip B until backtest reveals the real rate-limit pain.**
**Skip C until v1 is backtest-validated.**

---

## ✅ Recently completed (reverse chronological)

For full detail see `implementation_flow.md`.

| When | What | Test delta |
|---|---|---|
| 2026-04-28 | **Step 3.5.3** — `grade` + `calibration` CLI commands (+ `--by` breakdown axes) | 871 → 854 (net*) |
| 2026-04-28 | **Step 3.5.2** — `grade_many()` + `CalibrationReport` (Brier, 3 hit-rate variants, breakdowns) | 845 → 871 (+26) |
| 2026-04-28 | **Step 3.5.1** — `grade_one()` + `GradedPrediction` + 6-outcome enum (incl. STOP_HIT_AMBIGUOUS) | 817 → 845 (+28) |
| 2026-04-28 | **Step D.9 / 3.4.4** — typer + rich CLI surface (`predict` / `predict-many` / `history`) | ~850 → ~864 (+14) |
| 2026-04-28 | **Step D.8 / 3.4.5** — `PredictionStore` JSON-on-disk persistence | ~820 → ~850 (+30) |
| 2026-04-28 | **Step D.7 / 3.4.3** — `predict_many()` batch with bounded concurrency + `BatchError` | ~795 → ~820 (+25) |
| 2026-04-28 | **Step D.6** — News degradation handling + integration smoke test | ~780 → ~795 (+15) |
| 2026-04-28 | **Step D.5** — Hallucination guardrails Tiers 1-3 + retry-with-feedback | ~735 → ~780 (+45) |
| 2026-04-28 | **Step D.4** — `predict()` orchestrator + ADK `Runner` singletons | ~705 → ~735 (+30) |
| 2026-04-28 | **Step D.3** — `prediction/inputs.py` prompt assembly | ~685 → ~705 (+20) |
| 2026-04-28 | **Step D.2** — `synthesizer` agent with `output_schema=Prediction` | ~650 → ~685 (+35) |
| 2026-04-28 | **Step D.1 / 3.4.1** — `prediction/schema.py` (frozen Pydantic Prediction) | 625 → ~650 (+25) |
| 2026-04-28 | **Provider Expansion** — Stooq + AlphaVantage providers, `USE_PAID_PRICES` toggle | 554 → 625 (+71) |
| 2026-04-28 | Step C.6 — Manual smoke in `adk web` + LLM-chain bug fix | env-only |
| 2026-04-28 | Step C.5 — `technical_agent` wiring | 538 → 554 (+16) |
| 2026-04-27 | Step C.4 — `get_levels` tool + chart pattern integration | 509 → 538 (+29) |
| 2026-04-27 | Step C.3 — `get_volatility` tool + position-sizing | 474 → 509 (+35) |
| 2026-04-26 | Step C.2 — `get_momentum` tool + candlestick context-gating | 425 → 474 (+49) |
| 2026-04-26 | Step C.1 — `get_trend` tool + signal classifier + cache singleton | 394 → 425 (+31) |
| Earlier | Steps A, B.1–B.4 | 289 → 394 (+105) |

\*Step D + 3.5 per-substep test counts are reconstructed from commit log;
the 854 figure is the actual current `pytest --collect-only` count.

---

## 🅿️ Parking lot — design questions deferred

| Item | Why deferred | Revisit when |
|---|---|---|
| Disk persistence for price cache | Avoids cache-invalidation bugs | Sessions get long enough that re-fetching is painful |
| AV intraday support (`TIME_SERIES_INTRADAY`) | v1 daily is enough; chain falls back for non-daily | Specific intraday use case shows up |
| Stooq weekly/monthly intervals | Daily-only provider; chain falls back to yfinance | Same |
| Chart patterns: cup & handle, flags, wedges | Too noisy / hard to detect reliably | Multimodal LLM + chart image becomes the answer instead |
| Indicator parameters as raw integers | Adds tool-call surface area; LLMs can't pick wisely | Backtesting (Python-driven) needs sweeps |
| Volume profile / spike detection | Beyond OBV is v2 territory | Specific user requests it |
| Web UI / dashboard | Out of v1 scope | After CLI / `adk web` flow is solid |
| Pydantic models for tool returns | TypedDict is enough for v1 | Schema drift becomes a real problem |
| ADK ToolContext for cache injection | Module singleton works fine | We need per-request cache scoping |
| News/filings deduplication (`data/dedupe.py`) | Synthesizer prompt assembly didn't reveal duplicate-event noise (Step D.3); guardrails caught the worst hallucinations without it | Backtest reveals duplicate-event noise materially affecting outcomes |
| Persistence migration to SQLite | JSON-on-disk wins below ~1M predictions | Above ~36k files/yr × 3 years = ~100k files |
| Auto-adjust raw confidence based on calibration curve | v2+ idea from project desc §12; need more graded predictions before this means anything | After 100+ graded predictions accumulate |
| Hourly intraday backtest cadence | Free data sources don't reliably provide intraday history | Paid intraday source becomes available |
| Survivorship-bias handling | Becomes an actual concern only when backtest exists | Step 3.5.5 design |

### Removed from parking (now done)

- ~~Stooq / NSE direct provider~~ — landed in Provider Expansion (post-C)
- ~~yfinance fallback chain~~ — same
- ~~Backtesting framework~~ — partially: grading + calibration shipped; replay/runner/evaluator are Option A above
- ~~Schema for predictions~~ — shipped as Step D.1 (`prediction/schema.py`)
- ~~Per-stock predictor orchestration~~ — shipped as Steps D.2-D.6
- ~~Batch pipeline~~ — shipped as Step D.7 (`predict_many`)
- ~~CLI surface~~ — shipped as Steps D.9 + 3.5.3
