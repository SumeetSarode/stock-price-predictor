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
| 3.1.1 | Price fetcher (`data/prices.py`) | done | Iteration 2.1 — see above |
| 3.1.2 | News fetcher (`data/news.py`) using GDELT | done | Async-first; `fetch_news` + `fetch_news_batch` (concurrency=5 default) for discovery; `fetch_article_body` for separate body extraction via trafilatura. `NewsArticle` (metadata) + `ArticleBody` (status-tagged result) schemas. 40 unit tests + 2 integration tests (GDELT skipped on Walmart corp network — DNS-blocked; works off-corp). New deps: `trafilatura`, `respx` (dev) |
| 3.1.2.5 | Estimates fetcher (`data/estimates.py`) via yfinance | done | Async-first wrapper around yfinance analyst-data properties. **Coverage spike PASSED off-VPN: 20/20 (100%) across large/mid/small cap.** Schemas: `Estimates`/`QuarterlyEstimate`/`RecommendationDistribution`/`PriceTargets` with `has_coverage` gating. 35 unit tests with mocked yfinance. Spike script at `scripts/coverage_spike_estimates.py`. Known nuance: quarterly consensus thin for some Indian large-caps (1-2 analysts) but annual consensus robust (26-32 analysts) — downstream analyzer should weight by `num_analysts`. |
| 3.1.3 | NSE corporate filings (`data/filings.py`) | done + spike-verified | Async fan-out across NSE corporate-events endpoints (announcement / board_meeting / corporate_action). Cookie warmup via NSE homepage; browser-like headers. Unified `Filing` schema with separate `announced_at` (filing date) and `event_at` (when split/dividend/meeting actually happens) — enables both backward ("what was filed?") and forward ("what's coming in 60 days?") queries. Per-endpoint parsers handle quirky JSON shapes; endpoint-specific extras preserved in `metadata: dict`. Partial-failure tolerant (one bad endpoint doesn't kill the batch). 58 unit tests + 1 integration test. **Off-VPN spike (`filings_coverage_20260502T142439Z`) confirmed**: 3 endpoints fully working (60/60 board_meetings, 2/2 corp_actions, 199/202 announcements after parser fix). 4th endpoint `financial_result` returns empty body — excluded from `DEFAULT_KINDS` (data still captured via `announcement` desc='Financial Results'). Spike-driven fixes: (a) `_safe_url()` helper degrades blank/relative/dash URL fields to None instead of killing rows; (b) brotli dropped from Accept-Encoding; (c) warmup soft-fails on 4xx. No new deps. |
| 3.1.3 | Filings fetcher (`data/filings.py`) for NSE announcements | not started | 30-day corporate events window |
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

| Step | Description | Status | Notes |
|---|---|---|---|
| 3.3.1 | Technical indicators (`analysis/technical.py`) | not started | pandas / pandas-ta based |
| 3.3.2 | Levels analysis (`analysis/levels.py`) | not started | Support / resistance / pivots / swings |
| 3.3.3 | Pattern detection (`analysis/patterns.py`) | not started | Candlestick / chart patterns |
| 3.3.4 | News impact analysis (`analysis/news_impact.py`) | not started | LLM-assisted article scoring |
| 3.3.5 | Grounded reasoning (`analysis/reasoning.py`) | not started | Combine evidence into prediction-ready context |
| 3.3.6 | Pluggable analyzers (`analysis/pluggable.py`) | not started | Nice extension hook, but lower priority than core v1 |

### 3.4 Prediction pipeline

| Step | Description | Status | Notes |
|---|---|---|---|
| 3.4.1 | Define output schema (`prediction/schema.py`) | not started | Must match README output contract |
| 3.4.2 | Implement per-stock predictor (`prediction/predictor.py`) | not started | Orchestrates one full stock analysis |
| 3.4.3 | Implement batch pipeline (`prediction/pipeline.py`) | not started | Run over a stock universe |
| 3.4.4 | Add CLI commands for analysis | not started | `analyze` entry point |
| 3.4.5 | Persist self-contained JSON outputs | not started | Required for auditability and later UI |

### 3.5 Tracking and backtesting

| Step | Description | Status | Notes |
|---|---|---|---|
| 3.5.1 | Tracking store (`tracking/store.py`) | not started | SQLite storage of predictions |
| 3.5.2 | Calibration logic (`tracking/calibration.py`) | not started | Historical confidence vs reality |
| 3.5.3 | Replay layer (`backtest/replay.py`) | not started | Honest as-of-date simulation |
| 3.5.4 | Backtest runner (`backtest/runner.py`) | not started | Historical loop over dates |
| 3.5.5 | Evaluator (`backtest/evaluator.py`) | not started | Hit rate, stop rate, P&L, calibration |

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

### Current state
**Iteration 3.1.3 (NSE filings) DONE.** All four data-layer modules for v1 are
shipped: prices, news, estimates, filings. Schema additions for v1 data are
complete. 163 unit tests passing.

### Three reasonable next moves — user picks

**Option A: Off-VPN integration verification of NSE filings**
Same pattern as the estimates spike. Add a `scripts/coverage_spike_filings.py`
that hits real NSE for ~10 stocks across the 4 endpoints, captures actual JSON
shapes (vs our inferred shapes), writes a report. Confirms whether our parsers
match reality OR surfaces shape mismatches we need to fix. ~15 min to write,
user runs off-VPN and brings back report. RECOMMENDED before depending on
filings in any analyzer.

**Option B: Move to iteration 3.2 — first analyzer (`analyzers/news_impact`)**
With all data fetchers in place, build the first ADK analyzer agent that
consumes news + filings + estimates + prices and produces an impact score.
This is the FUN part — actual agent / LLM work begins here.

**Option C: Quick polish iteration — ADK tool wraps for news/filings/estimates**
We have `agents/price_agent` from iteration 2. Wrap the other 3 modules as
ADK tools (`news_agent`, `filings_agent`, `estimates_agent`) so an LLM can
call them directly via ADK. Mirrors price_agent pattern. Quick (~15 min each).
Learning value: more reps with the ADK tool-wrap pattern.

### What's done so far in iterations 2 + 3.1 (data layer foundation, COMPLETE)
- `data/prices.py` — OHLCV fetcher (yfinance, sync)
- `data/news.py` — GDELT discovery + trafilatura body extraction (async)
- `data/estimates.py` — yfinance analyst-data wrapper (async, coverage 100%)
- `data/filings.py` — NSE 4-endpoint fan-out (async, integration verification pending)
- `data/schema.py` — OHLCVBar, NewsArticle, ArticleBody, QuarterlyEstimate,
  RecommendationDistribution, PriceTargets, Estimates, FilingKind, Filing
- `agents/price_agent/` — ADK tool wrap of price fetcher (learning artifact)
- `scripts/coverage_spike_estimates.py` — off-VPN coverage runner
- All test suites passing: **163 unit tests**; integration tests pass off-corp

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

---

## 7) Suggested update rule for this doc

When work progresses, update each step to one of:
- **done**
- **wip**
- **not started**

If a step is split, add child rows rather than rewriting history. Keep this file as the high-level implementation tracker.
