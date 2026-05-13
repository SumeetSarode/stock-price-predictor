# Implementation Plan

This document captures the implementation plan we discussed so far in chat, aligned with the current `README.md` roadmap.

## Status legend
- **done** — completed and verified
- **wip** — started, but not fully complete / reliable / production-ready
- **not started** — planned, but no meaningful implementation yet

---

## 1) Foundation and learning spike

These steps were about proving the local ADK setup, tool calling, model routing, and Walmart-network execution path before starting the real stock-predictor feature work.

| Step | Description | Status | Notes |
|---|---|---|---|
| 1.1 | Create project scaffold and basic repo structure | done | Project exists with `src/`, `tests/`, `docs/`, `data/`, config, and uv-based setup |
| 1.2 | Add centralized settings and logging foundation | done | `settings.py` and logging config are in place |
| 1.3 | Build a simple ADK hello agent | done | `hello_agent` works via `adk run` |
| 1.4 | Add a tool to the hello agent (`get_current_time`) | done | Tool calling works end-to-end |
| 1.5 | Add tests for the hello-agent spike | done | Unit tests are passing |
| 1.6 | Make the app work on Walmart network / corp proxy | done | Proxy auto-config added via `.env` + `settings.py` |
| 1.7 | Verify model execution through LiteLLM + ADK | done | Confirmed working through proxy |
| 1.8 | Make Groq the primary model | done | `.env` updated to Groq primary, Gemini secondary |
| 1.9 | Improve Groq tool-call reliability with prompt tuning | wip | Reliability improved, but Groq/Llama tool calling is still not perfectly deterministic |

### Notes
- The hello-agent work is a **learning spike / infrastructure proof**, not the final product feature set.
- We explicitly decided to keep **Groq as primary**, even though Gemini is more reliable for tool calling.

---

## 2) Current implementation direction for the real project (v1)

We discussed that the next real feature should be the **data layer**, starting with price fetching, because everything else depends on it.

| Step | Description | Status | Notes |
|---|---|---|---|
| 2.1 | Implement `data/prices.py` for OHLCV fetching | done | `fetch_ohlcv()` returns tz-aware DataFrame; `auto_adjust=False` keeps both `close` and `adj_close`; end-date inclusive |
| 2.2 | Add typed schema / validation for price bars | done | `OHLCVBar` Pydantic model in `data/schema.py` — for boundaries (JSON, fixtures); engine still uses DataFrames |
| 2.3 | Add unit tests for price fetching | done | 7 tests passing (5 named + 3 parametrized); mocks at `yfinance.download` |
| 2.4 | Add one real integration test using yfinance | done | `test_fetch_ohlcv_real_reliance` against fixed Jan 2024 range; `@pytest.mark.integration` |
| 2.5 | Optionally expose the price fetcher as an ADK tool | done | `agents/price_agent/` with `fetch_prices_tool` (string args, hybrid summary + opt-in `include_bars`); 13 mocked tests passing. Manual `adk run` play pending (rate-limited by yfinance public API) |

### Why this is next
- Technical analysis depends on price history
- Backtesting depends on price history
- Prediction generation depends on technical analysis and price context
- This is the smallest useful slice of the real app

---

## 3) v1 roadmap for the stock price predictor

This section reflects the implementation path implied by the README and our discussion so far.

### 3.1 Data layer

| Step | Description | Status | Notes |
|---|---|---|---|
| 3.1.1 | Price fetcher (`data/prices.py`) | done | Iteration 2.1 above; expanded post-C with multi-provider resilient chain. `PriceProvider` ABC + `YFinanceProvider`, `StooqProvider`, `AlphaVantageProvider`. `PRICE_CHAIN` ordered fallback + `USE_PAID_PRICES` toggle (parallels LLM `USE_PAID`). Stooq + AV both behind free api keys (lazy validation). |
| 3.1.2 | News fetcher (`data/news.py`) using GDELT | done | Async-first; `fetch_news` + `fetch_news_batch` (concurrency=5 default) for discovery; `fetch_article_body` for separate body extraction via trafilatura. `NewsArticle` (metadata) + `ArticleBody` (status-tagged result) schemas. 40 unit tests + 2 integration tests (GDELT skipped on Walmart corp network — DNS-blocked; works off-corp). New deps: `trafilatura`, `respx` (dev) |
| 3.1.2.5 | Estimates fetcher (`data/estimates.py`) via yfinance | done | Async-first wrapper around yfinance analyst-data properties. **Coverage spike PASSED off-VPN: 20/20 (100%) across large/mid/small cap.** Schemas: `Estimates`/`QuarterlyEstimate`/`RecommendationDistribution`/`PriceTargets` with `has_coverage` gating. 35 unit tests with mocked yfinance. Spike script at `scripts/coverage_spike_estimates.py`. Known nuance: quarterly consensus thin for some Indian large-caps (1-2 analysts) but annual consensus robust (26-32 analysts) — downstream analyzer should weight by `num_analysts`. |
| 3.1.3 | NSE corporate filings (`data/filings.py`) | done + spike-verified | Async fan-out across NSE corporate-events endpoints (announcement / board_meeting / corporate_action). Cookie warmup via NSE homepage; browser-like headers. Unified `Filing` schema with separate `announced_at` (filing date) and `event_at` (when split/dividend/meeting actually happens) — enables both backward ("what was filed?") and forward ("what's coming in 60 days?") queries. Per-endpoint parsers handle quirky JSON shapes; endpoint-specific extras preserved in `metadata: dict`. Partial-failure tolerant (one bad endpoint doesn't kill the batch). 58 unit tests + 1 integration test. **Off-VPN spike (`filings_coverage_20260502T142439Z`) confirmed**: 3 endpoints fully working (60/60 board_meetings, 2/2 corp_actions, 199/202 announcements after parser fix). 4th endpoint `financial_result` returns empty body — excluded from `DEFAULT_KINDS` (data still captured via `announcement` desc='Financial Results'). Spike-driven fixes: (a) `_safe_url()` helper degrades blank/relative/dash URL fields to None instead of killing rows; (b) brotli dropped from Accept-Encoding; (c) warmup soft-fails on 4xx. No new deps. |
| 3.1.4 | News / filings deduplication (`data/dedupe.py`) | not started | EventID + fuzzy-title fallback |
| 3.1.5 | Cache layer (`data/cache.py`) | not started | Deferred until fetchers exist and caching is actually needed |

### 3.2 Knowledge base

| Step | Description | Status | Notes |
|---|---|---|---|
| 3.2.1 | Define KB interface (`kb/interface.py`) | not started | Keep implementation swappable |
| 3.2.2 | Add stock registry (`kb/stocks.py`) | not started | Nifty50 / supported ticker registry |
| 3.2.3 | Implement structured KB (`kb/structured.py`) | not started | SQLite-backed phase-1 KB |
| 3.2.4 | Build KB bootstrapper (`kb/builder.py`) | not started | Wiki + NSE + LLM-assisted population |
| 3.2.5 | Add LightRAG integration layer (`kb/lightrag_layer.py`) | not started | Explicitly phase 2, not v1-critical |

### 3.3 Analysis layer

> **Re-scoped during Step C** (see `implementation_flow.md`): the analysis
> primitives live under `agents/technical_agent/tools/` rather than a
> separate `analysis/` package. Same scope, cleaner agent boundary.

| Step | Description | Status | Notes |
|---|---|---|---|
| 3.3.1 | Technical indicators (now in `tools/_trend_signal.py`, `_momentum_signal.py`, `_volatility_signal.py`) | ✅ done | Built across C.1–C.3 |
| 3.3.2 | Levels analysis (now in `tools/_levels_signal.py`) | ✅ done | Built in C.4 |
| 3.3.3 | Pattern detection (chart patterns inside `_levels_signal.py`; candlesticks in `_candlestick_gating.py`) | ✅ done | Candlestick gating in C.2; chart patterns in C.4 |
| 3.3.4 | News impact analysis (`agents/news_impact/`) | ✅ done | LLM-assisted article scoring |
| 3.3.5 | Grounded reasoning (now lives in `technical_agent`'s instruction prompt) | ✅ done | Synthesis is the LlmAgent's job (C.5) |
| 3.3.6 | Pluggable analyzers (`analysis/pluggable.py`) | not started | Nice extension hook, but lower priority than core v1 |

### 3.4 Prediction pipeline

| Step | Description | Status | Notes |
|---|---|---|---|
| 3.4.1 | Define output schema (`prediction/schema.py`) | ✅ done | Frozen Pydantic v2 model. `Prediction` + `PredictionDirection` + `PredictionHorizon` + `PriceLevel` + `AnalysisBasis`. JSON round-trips, computed `risk_reward`. ~25 schema tests. |
| 3.4.2 | Per-stock predictor (`prediction/predictor.py`) | ✅ done (initially shipped single-horizon; **truly multi-horizon as of Step 3.4.6**) | Shipped in 6 commits (synthesizer agent, predictor orchestrator, Runner singletons, hallucination guardrails Tiers 1-3 with retry, news degradation, integration smoke test). `inputs.py` builds the synthesizer prompt from technical/news/price snapshots. `guardrails.py` enforces ticker/level/direction sanity post-LLM. **Caveat (now closed)**: as originally shipped, the predictor took a `horizon` argument but used the same hard-coded thresholds for all horizons — "fake multi-horizon." Step 3.4.6 closes that gap honestly. |
| 3.4.3 | Batch pipeline (`prediction/batch.py`) | ✅ done | `predict_many()` over an arbitrary ticker list. Concurrent with bounded parallelism. Per-ticker errors don't kill the batch (`BatchError` accumulates failures). |
| 3.4.4 | CLI commands for prediction | ✅ done | `price-predictor predict TICKER` + `predict-many T1 T2 ...` + `history TICKER`. Typer + Rich rendering. |
| 3.4.5 | Persist self-contained JSON outputs (`prediction/store.py`) | ✅ done | `PredictionStore` writes per-prediction JSON to `predictions_dir`. Indexed by ticker + date. `list_for_ticker()`, `list_in_date_range()`. Fully round-trippable. |
| 3.4.6 | **Multi-horizon hardening** (NEW — not in original plan) | ✅ done | 8-commit refactor that closed a latent bug in 3.4.2: predictor accepted `horizon` but used horizon-blind constants. Built across: (a) `0d5cdec` NSE trading-calendar helper for honest horizon math, (b) `505cb4d` rename horizon enum to DAILY/WEEKLY/BIWEEKLY/MONTHLY, (c) `ff037fc` `predict()` fans out across all 4 horizons in parallel, (d) `3253d89` per-horizon NEUTRAL grading tolerance (sqrt-t scaled), (e) `9efa283` research-grounded `docs/research/constants_dossier.md` + LMW chart-pattern alignment, (f) `34bb240` `prediction/horizon_constants.py` as single source of truth (Commit A), (g) `c66388e` guardrails wired to per-horizon bands + new Tier 4 calibration cap (Commit B), (h) `eb3c84f` synthesizer prompt embeds the per-horizon rules table read straight from `horizon_constants` (Commit C). End state: one tunable surface (`horizon_constants.py`); guardrails + LLM prompt cannot drift apart (regression tests prove it). |

### 3.5 Tracking and backtesting

> **Note**: Originally planned as SQLite + a separate backtest framework. We
> shipped the tracking + calibration halves first as plain JSON (simpler, no
> migration cost) and added a NEW sub-step (3.5.3 grading core) that the
> original plan didn't anticipate but turned out to be the unit of work that
> made calibration meaningful. Backtest replay/runner/evaluator are still
> ahead of us.

| Step | Description | Status | Notes |
|---|---|---|---|
| 3.5.1 | Prediction store (`prediction/store.py`) | ✅ done | JSON-on-disk (not SQLite — no migration cost, easier to inspect). Lives in `prediction/` not `tracking/` since it's just persistence of the prediction artifact. |
| 3.5.2 | Calibration logic (`prediction/calibration.py`) | ✅ done | `CalibrationReport` (frozen Pydantic) + `compute_calibration()` + `compute_breakdown()`. Three hit-rate variants (strict / resolved / optimistic), Brier score for confidence calibration, direction accuracy, mean+median return. 20 tests. |
| 3.5.3 | Grading core (`prediction/grading.py`) **(NEW — not in original plan)** | ✅ done | `GradedPrediction` + `grade_one()` (per-prediction, pure function on OHLCV) + `grade_many()` (orchestration with injected fetcher). 6 outcome enum: target_hit / stop_hit / stop_hit_ambiguous / expired / not_applicable / inconclusive. Same-bar T+S ambiguity surfaced as a first-class outcome. 34 tests. |
| 3.5.4 | CLI surface for grading + calibration | ✅ done | `price-predictor grade` + `calibration` (with `--by horizon|ticker|direction|month` breakdowns). Dispatch-dict for breakdown axes (open/closed). |
| 3.5.5 | Replay layer (`backtest/replay.py` — actually shipped as point-in-time news replay via contextvar + NewsSnapshot store, Step 1.5) | ✅ done | News articles published AFTER the as-of date are filtered out at fetch time. Predict() against 2018 prices does NOT see 2024 news. |
| 3.5.6 | Backtest runner (`backtest/runner.py`) | ✅ done | `run_backtest(tickers, dates, ...)` (cartesian) + `run_backtest_grid(pairs, ...)` (sparse, for survivorship-aware index runs). Concurrency-bounded, eager-save to PredictionStore, per-pair error capture, Rich progress callback. |
| 3.5.7 | Evaluator (`backtest/evaluator.py`) | ✅ done | `evaluate_backtest(run)` composes existing grading + calibration into a `BacktestEvaluation` artifact. HTML report + rule-based insights renderer (`backtest/html_report.py` + `backtest/insights.py`). CLI: `price-predictor backtest --start ... --end ... [--tickers ... | --index NIFTY50]`. End-to-end integration test gates wall-clock <5min. |
| 3.5.8 | Survivorship-bias defense (NEW — not in original plan) | ✅ done | `kb/membership.py` walks the Wikipedia event log backwards from today's NIFTY 50 to reconstruct historical constituents on any as-of date. `backtest --index NIFTY50` consumes it via `_expand_index_to_pairs` → `run_backtest_grid`. A 2018 backtest predicts the 2018 NIFTY 50, not today's. |

### 3.6 Concurrency and scale

| Step | Description | Status | Notes |
|---|---|---|---|
| 3.6.1 | Async batching / semaphores (`concurrency/runner.py`) | not started | Important once multi-stock runs exist |
| 3.6.2 | Rate-limit-aware LLM routing | not started | Groq primary, Gemini fallback |

---

## 4) Post-v1 roadmap

| Step | Description | Status | Notes |
|---|---|---|---|
| 4.1 | LightRAG-backed knowledge layer | not started | Explicitly phase 2 in README |
| 4.2 | Scheduled runs / automation | not started | Phase 2+ |
| 4.3 | Web UI for predictions and history | not started | Phase 3 |
| 4.4 | Additional analyzers (fundamentals, alt-data, sentiment) | not started | Phase 3 extensibility |

---

## 5) Immediate next step

### Current state (as of 2026-05-12, **v1 DONE**)

**Option A (backtest replay + runner + evaluator + survivorship-bias
+ integration test) shipped.** v1 is feature-complete and gated
end-to-end. The full loop is now:

  prediction (×4 horizons) → persistence → grading
  → calibration → historical replay (point-in-time) → backtest
  runner (cartesian OR sparse via --index) → evaluator → HTML report

**What's working today:**
- 5 ADK agents shipped: `hello_agent`, `price_agent`, `news_impact`,
  `technical_agent`, `synthesizer`
- 7 CLI commands: `predict`, `predict-many`, `history`, `grade`,
  `calibration`, `backtest --tickers`, `backtest --index NIFTY50`
- **1576 unit tests passing**, 8 integration tests deselected by
  default (run off-corp via `pytest -m integration`). Up from 1021
  pre-Option-A.
- All v1 data layers (prices, news, estimates, filings, KB-membership) shipped + verified
- All v1 analysis primitives (trend / momentum / volatility / levels / patterns) shipped
- All v1 prediction infrastructure (predict, predict_many, store, grade, calibration) shipped
- **Honest historical replay**: news + filings filtered by
  `published_at` / `announced_at` against the as-of date via the
  Step 1.5 contextvar. Predict() against 2018 prices won't see
  2024 news.
- **Honest survivorship-bias defense**: `--index NIFTY50` walks
  the Wikipedia event log backwards from today's 50 to reconstruct
  the historical constituents on each as-of date. A 2018 backtest
  predicts the 2018 NIFTY 50, not today's.
- **Honest multi-horizon**: `predict()` fans out across DAILY /
  WEEKLY / BIWEEKLY / MONTHLY in parallel; each horizon has its
  own ATR bands, entry zone, and confidence cap; guardrails
  enforce them; LLM prompt is taught the same numbers; tests prove
  no drift.

**What's NOT yet shipped (post-v1):**
- 3.6: concurrency + rate-limit-aware LLM router (the integration
  test surfaced this as the next real pain point; the naive
  Groq/Gemini chain can't sustain NIFTY 50 × months runs)
- 4.1: LightRAG knowledge layer (Phase 2)

### Three reasonable next moves — user picks (post-v1)

**Option B: Step 3.6 — concurrency & rate-limit-aware router (recommended)**
Backtest revealed the real rate-limit pain (integration test on 12
pairs blew through the Gemini daily quota). Wraps `llm/factory.py`
with provider fallback on 429, exponential backoff, token budgeting.
~3-5 commits.

**Option C: Step 4.1 — LightRAG knowledge layer (Phase 2)**
Big quality bump for predictions, not v1-critical. ~5-7 commits.

**Option D: Run a real backtest and write up findings**
With v1 done, run `backtest --index NIFTY50 --start 2024-01-01
--end 2024-12-31` (off-corp, with full Gemini quota), then write
up the calibration findings. **No new code**; pure validation.

---

## 6) Open implementation notes from our discussion

| Topic | Current decision |
|---|---|
| Primary LLM | Groq |
| Secondary / fallback LLM | Gemini |
| Groq reliability | Acceptable for now, but still a known weak spot for tool-calling |
| Learning spike vs product work | Hello agent was a spike; real app work starts with the data layer |
| Build order | Data layer first, then analysis, then prediction, then tracking/backtest |
| Design principles | Interfaces over implementations, self-contained outputs, as-of-date correctness, async-first |
| v1 surface (post-3.5 + post-3.4.6) | `predict`, `predict-many`, `history`, `grade`, `calibration` — full prediction loop is closed end-to-end AND truly multi-horizon (4 horizons fanned out per ticker) |
| Hit-rate reporting (3.5) | Three flavours surfaced: strict / resolved / optimistic. We REPORT all three rather than picking one because each answers a different honest question and same-bar T+S ambiguity makes any single number lossy. |
| Confidence calibration (3.5) | Brier score over log-loss — bounded [0,1], no log(0) edge case at confidence=1.0, quadratic penalty matches user intuition. |
| Per-horizon tunables (3.4.6) | Single source of truth: `prediction/horizon_constants.py`. Helpers `stop_atr_range()`, `target_atr_range()`, `entry_zone_pct()`, `confidence_cap()`, `neutral_tolerance_pct()` consulted by guardrails AND the LLM prompt. Tune one place; both layers update; regression tests prove no drift. |
| NEUTRAL grading tolerance (3.4.6) | Per-horizon and sqrt-t scaled (longer horizons → wider tolerance) so a 0.5% move at daily isn't graded the same as a 0.5% move at monthly. |
| Persistence for predictions/grades | JSON-on-disk via `prediction/store.py` (NOT SQLite). Easier to inspect, no migration cost. Will revisit if scale demands it. |

---

## 7) Suggested update rule for this doc

When work progresses, update each step to one of:
- **done**
- **wip**
- **not started**

If a step is split, add child rows rather than rewriting history. Keep this file as the high-level implementation tracker.
