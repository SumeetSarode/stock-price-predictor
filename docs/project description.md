# 📈 Predictor — Project Description

> **Status:** 🚧 v1 in progress — `predict` + `grade` + `calibration`
> shipped end-to-end (854 unit tests passing). Backtest replay + concurrency
> still ahead.
> **Owner:** Sumeet
> **Codename:** *Predictor* (placeholder — final name TBD)
> **Last updated:** 2026-04-28

> This is the **canonical full spec** — problem, design, output schema,
> decisions, risks. For practical "how do I run it" content see
> [`../README.md`](../README.md). For per-step build status see
> `../implementation_plan.md` and `../implementation_flow.md`.

---

## 0. 🟢 Current state (added 2026-04-28)

| Layer | Status |
|---|---|
| Data layer (prices / news / estimates / filings) | ✅ shipped |
| KB (Nifty50 registry) | ✅ shipped |
| Analysis primitives (trend / momentum / volatility / levels / patterns) | ✅ shipped |
| ADK agents (price / news / technical / synthesizer) | ✅ shipped |
| Prediction pipeline (predict / predict-many / store) | ✅ shipped |
| Grading + Calibration | ✅ shipped |
| Backtest replay / runner / evaluator | ⏸️ not started |
| Concurrency / scale (rate-limit-aware router) | ⏸️ not started |
| LightRAG knowledge layer (Phase 2) | ⏸️ not started |

---

## 1. 🎯 One-liner

A **free, locally-runnable backend** that analyzes Nifty50 stocks and produces **daily + weekly trading predictions** (entry, target, stop loss — all with explainable reasoning) by combining **comprehensive technical analysis** with **news impact assessment**, designed for serious personal trading use with first-class backtesting and prediction tracking.

---

## 2. 🤔 Problem Statement

Retail traders covering Nifty50 stocks face three persistent problems:

1. **Information overload** — dozens of news sources, broker reports, and indicator readings per stock per day. No human can systematically synthesize all of this for 50 stocks.
2. **Lack of explainability** in existing tools — most signal services give a "BUY/SELL" without showing *which* news, *which* indicators, or *why* a particular target/stop was chosen.
3. **No personal track record** — without systematically tracking predictions vs outcomes, traders cannot calibrate trust in any signal source over time.

This system addresses all three by being **systematic, explainable, and self-tracking** — and being free + local means no vendor lock-in or recurring cost.

---

## 3. 🎯 Goals & Non-Goals

### In scope (v1)
- Analyze all 50 Nifty50 stocks on demand
- Generate **daily** and **weekly** predictions per stock
- Each prediction includes: direction, confidence, suggested entry, target price + range, stop loss, risk/reward, key drivers, and **full explainability** (which news, which indicators)
- Auto-built **knowledge base** of stock metadata + macro sensitivities
- **Backtesting** against historical data (hit-rate, stop-rate, P&L, calibration)
- **Prediction tracking** with historical accuracy fed back into new predictions
- Modular architecture supporting future analyses (fundamentals, sentiment, etc.)
- Free, local, runs on any laptop (no GPU)

### Explicitly out of scope (v1)
- ❌ Real-time / streaming updates (on-demand only for v1)
- ❌ Web UI (CLI + JSON output only for v1; UI is v3)
- ❌ Order execution / broker integration
- ❌ Portfolio optimization / position sizing
- ❌ Hourly / intraday predictions (too noisy with free data sources)
- ❌ Stocks outside Nifty50
- ❌ Fundamental analysis (extensible later via plugin)
- ❌ Sentiment scoring beyond what news LLM analysis already provides

### Phased roadmap
| Phase  | Scope                                                                                                |
| ------ | ---------------------------------------------------------------------------------------------------- |
| **v1** | On-demand CLI; structured KB; daily+weekly predictions; backtest + tracking; explainability complete |
| **v2** | LightRAG knowledge layer over annual reports + accumulated news; scheduled runs                      |
| **v3** | Web UI; pluggable additional analyzers (fundamentals, alt-data, sentiment)                           |

---

## 4. 🚧 Constraints

| Constraint                         | Implication                                                  |
| ---------------------------------- | ------------------------------------------------------------ |
| **Free everything**                | No paid APIs, no paid LLMs, no paid news sources             |
| **Runs on any laptop**             | No GPU, no heavyweight local LLM, minimal install            |
| **Python 3.13 + uv**               | Modern Python, fast dependency management                    |
| **Walmart artifactory PyPI index** | All `uv pip install` calls use Walmart's mirror              |
| **No system-level dependencies**   | Pure-Python libs preferred (e.g., `pandas-ta` over `TA-Lib`) |

---

## 5. 🏗️ High-Level Architecture

```
                              ┌─────────────────────┐
                              │   CLI / Scripts     │
                              │  analyze, backtest, │
                              │  bootstrap-kb       │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │     Pipeline        │
                              │  (orchestration)    │
                              └──────────┬──────────┘
                                         │
            ┌────────────────────────────┼────────────────────────────┐
            │                            │                            │
   ┌────────▼────────┐         ┌─────────▼────────┐         ┌────────▼────────┐
   │  Data Layer     │         │  Analysis Layer  │         │ Knowledge Base  │
   │  - prices       │         │  - technicals    │         │  - structured   │
   │  - news (GDELT) │         │  - news impact   │◀────────│  - LightRAG (P2)│
   │  - filings      │         │  - reasoning     │         │  - builder      │
   │  - cache        │         │  - pluggable     │         └─────────────────┘
   └─────────────────┘         └─────────┬────────┘
                                         │
                              ┌──────────▼──────────┐
                              │    LLM Router       │
                              │  Groq → Gemini      │
                              │  (rate-limit aware) │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  Prediction Output  │
                              │  (full JSON snapshot)│
                              └──────────┬──────────┘
                                         │
                       ┌─────────────────┼─────────────────┐
                       │                 │                 │
              ┌────────▼─────┐  ┌────────▼──────┐  ┌──────▼────────┐
              │ Tracking DB  │  │  JSON Outputs │  │ Backtest      │
              │ (SQLite)     │  │  (per run)    │  │ Evaluator     │
              └──────────────┘  └───────────────┘  └───────────────┘
```

**Design principles:**
- **Interfaces over implementations** — `LLMClient` and `KnowledgeBase` are abstract; swap providers without touching anything else
- **Self-contained outputs** — every prediction JSON has everything needed for UI/audit (no hidden lookups)
- **As-of-date everywhere** — every data fetch supports "as of date X" so backtest is honest
- **Async-first** — concurrency baked in for batching the 50-stock universe

---

## 6. 🗂️ Project Structure

> **Note (2026-04-28):** Below shows what's actually shipped today, with
> phase-2 / not-yet-built modules marked `(planned)`. The original spec
> proposed `tracking/` as a separate package and SQLite-backed storage —
> we shipped both calibration and predictions inside `prediction/` as
> JSON-on-disk for inspectability and zero-migration cost. See
> `../implementation_flow.md` for the rationale.

```
price_predictor/
├── pyproject.toml                # uv-managed
├── README.md
├── .env.example                  # API keys template
├── .gitignore
│
├── src/price_predictor/
│   ├── __init__.py
│   │
│   ├── cli/                       # ── typer + rich entry points ──
│   │   └── main.py                #    predict / predict-many / history / grade / calibration
│   │
│   ├── config/
│   │   └── settings.py            #    central config (env, paths, defaults)
│   │
│   ├── llm/                       # ── LLM abstraction ──
│   │   ├── factory.py             #    LiteLLM factory (Groq / Gemini)
│   │   └── resilient.py           #    retry/backoff wrapper
│   │
│   ├── data/                      # ── External data fetchers ──
│   │   ├── prices.py              #    Thin shim over providers
│   │   ├── cache.py               #    Range-aware in-memory cache
│   │   ├── _shared_cache.py       #    Singleton cache instance
│   │   ├── providers/             #    yfinance → Stooq → Alpha Vantage chain
│   │   │   ├── base.py / resilient.py
│   │   │   ├── yfinance_provider.py
│   │   │   ├── stooq_provider.py
│   │   │   └── alpha_vantage_provider.py
│   │   ├── news.py                #    GDELT discovery + trafilatura body extraction
│   │   ├── filings.py             #    NSE corporate-events fan-out (30d)
│   │   ├── estimates.py           #    yfinance analyst data wrapper
│   │   └── schema.py              #    OHLCVBar, NewsArticle, Filing, Estimates
│   │
│   ├── kb/                        # ── Knowledge base ──
│   │   └── stocks.py              #    Nifty50 ticker registry (Wikipedia-sourced)
│   │   # (planned) interface.py / lightrag_layer.py / builder.py for Phase 2
│   │
│   ├── analysis/                  # ── Pure-function analyzers ──
│   │   ├── trend.py               #    SMA/EMA/MACD/ADX/Ichimoku
│   │   ├── momentum.py            #    RSI/Stoch/CCI/Williams%R/ROC
│   │   ├── volatility.py          #    Bollinger/ATR/Keltner
│   │   ├── levels.py              #    pivots, swings, S/R clusters
│   │   ├── candlestick_patterns.py
│   │   └── chart_patterns.py
│   │
│   ├── agents/                    # ── ADK agents ──
│   │   ├── hello_agent/           #    Learning spike
│   │   ├── price_agent/           #    Tool-wrapped price fetcher
│   │   ├── news_impact/           #    LLM scoring per article
│   │   ├── technical_agent/       #    4 thematic tools (trend/momentum/volatility/levels)
│   │   └── synthesizer/           #    LlmAgent with output_schema=Prediction
│   │
│   └── prediction/                # ── Prediction generation, persistence, grading ──
│       ├── schema.py              #    Frozen Pydantic Prediction model (§10)
│       ├── inputs.py              #    Prompt assembly from technical/news/price snapshots
│       ├── predictor.py           #    predict() per-stock orchestrator
│       ├── runner.py              #    ADK Runner singletons (one per Agent)
│       ├── guardrails.py          #    Hallucination guardrails Tiers 1-3 + retry
│       ├── batch.py               #    predict_many() with bounded concurrency
│       ├── store.py               #    PredictionStore (JSON-on-disk)
│       ├── grading.py             #    grade_one + grade_many + 6-outcome enum
│       └── calibration.py         #    CalibrationReport + Brier + 3 hit-rate variants
│
├── # (planned) src/price_predictor/backtest/
│ #            ├── replay.py        #    as-of-date data shim
│ #            ├── runner.py        #    historical loop
│ #            └── evaluator.py     #    composes calibration across runs
│
├── # (planned) src/price_predictor/concurrency/
│ #            └── runner.py        #    asyncio semaphores + rate-limit-aware router
│
├── data/                          # ── Runtime data (gitignored) ──
│   ├── kb/                        #    stocks.json + indices.json (committed)
│   └── predictions/               #    per-prediction JSON files (gitignored)
│
├── scripts/
│   └── bootstrap_indices.py       #    one-time index registry build
│
├── tests/                         # 854 unit tests + 7 integration
│
└── docs/
    └── project description.md     ← this file
```

---

## 7. 🔌 Data Sources

| Source                           | What                                       | Why                                                                              | Cost   |
| -------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------- | ------ |
| **yfinance**                     | OHLCV price data, intraday + daily         | Free, reliable, supports `.NS` (NSE) tickers, years of history                   | ✅ Free |
| **GDELT**                        | News articles, entities, tone, source URL  | Free, indexed for India, supports historical queries (back to 2015) for backtest | ✅ Free |
| **NSE corporate announcements**  | Official filings, earnings, board meetings | Authoritative, free, 30-day window                                               | ✅ Free |
| **Wikipedia**                    | Company background, segments, history      | Free, used in KB bootstrap                                                       | ✅ Free |
| **NSE sectoral classifications** | Sector / sub-sector mapping                | Free, official                                                                   | ✅ Free |

### News windows
- **General news:** last 7 days, recency-weighted by `hours_ago` in LLM prompt
- **Corporate filings / earnings:** last 30 days (longer-tail relevance)

### Deduplication strategy
1. Primary: GDELT `EventID` (groups related coverage natively)
2. Fallback: fuzzy title similarity (~85% threshold)
3. Quality preference: Reuters > major Indian outlets > smaller blogs

---

## 8. 📊 Technical Analysis Catalog

Library: **`pandas-ta`** (pure Python, no system deps).

| Category       | Indicators                                                                        |
| -------------- | --------------------------------------------------------------------------------- |
| **Trend**      | SMA, EMA (20/50/200), MACD, ADX, Ichimoku                                         |
| **Momentum**   | RSI(14), Stochastic, CCI, Williams %R, ROC                                        |
| **Volatility** | Bollinger Bands (20,2), ATR(14), Keltner Channels                                 |
| **Volume**     | OBV, VWAP, MFI, Volume SMA ratio                                                  |
| **Levels**     | Daily + weekly pivot points, recent swing highs/lows, support/resistance clusters |
| **Patterns**   | Candlestick (doji, hammer, engulfing, etc.), basic chart patterns where feasible  |

All computed deterministically from price data and exposed in the prediction output's `technical_snapshot` field.

---

## 9. 🧠 Knowledge Base

### Phase 1: Structured KB (SQLite)

Auto-built per stock by `scripts/bootstrap_kb.py`:

```python
{
  ticker: "RELIANCE",
  name: "Reliance Industries Ltd",
  sector: "Energy",                      # from NSE
  sub_sector: "Refineries",
  business_segments: [...],              # from Wikipedia + LLM
  revenue_split: { ... },                # if available
  top_customers: [ ... ],
  top_suppliers: [ ... ],
  competitors: [ ... ],
  macro_sensitivities: [
    "crude_oil", "usd_inr", "us_fed_rate", ...
  ],
  geographic_exposure: { india: 60, us: 30, other: 10 },
  last_refreshed: "2026-04-24"
}
```

**Build process:**
1. Fetch NSE official sector + sub-sector
2. Pull Wikipedia "Business segments", "Operations", "Subsidiaries" sections
3. Optional: scrape company "About" page if accessible
4. Single LLM pass to extract structured JSON
5. Persist to `data/kb.sqlite`

Refresh: monthly cron / manual.

### Phase 2: LightRAG layer (deferred)

Plugged into `KnowledgeBase` interface so existing code is unaware of the change. Use cases:
- Mining annual report PDFs for nuanced relationships the structured KB misses
- Building emergent graphs from accumulated news over time
- Pattern-matching new news against historical similar events

**Deferred because:**
- Adds significant complexity
- Hallucination risk on free-tier LLM extraction
- We need the system working end-to-end first to know what LightRAG actually adds

---

## 10. 📤 Output Schema (per stock, per run)

The output JSON is **fully self-contained** — UI / audit tooling needs no extra lookups.

```jsonc
{
  "ticker": "RELIANCE",
  "company_name": "Reliance Industries Ltd",
  "as_of": "2026-04-24T18:30:00+05:30",
  "current_price": 2455.50,

  "knowledge_base_snapshot": {
    "sector": "...", "sub_sector": "...",
    "business_segments": [...],
    "macro_sensitivities": [...],
    // ... (snapshot of KB at time of prediction)
  },

  "technical_snapshot": {
    "trend":      { "sma_20": ..., "sma_50": ..., "sma_200": ..., "ema_20": ..., "macd": {...}, "adx": ... },
    "momentum":   { "rsi_14": 62, "stoch": {...}, "cci": ..., "williams_r": ..., "roc": ... },
    "volatility": { "bollinger": {"upper":..., "lower":...}, "atr_14": ..., "keltner": {...} },
    "volume":     { "obv": ..., "vwap": ..., "mfi": ..., "volume_sma_ratio": ... },
    "levels":     { "support": [2425, 2400, 2380], "resistance": [2480, 2510, 2545],
                    "pivots": { "daily": {...}, "weekly": {...} } },
    "patterns":   ["bullish_engulfing", "ascending_triangle"],
    "summary":    "Above all key MAs, RSI 62 neutral-bullish, ..."
  },

  "news_articles": [
    {
      "id": "news-id-1",
      "title": "RIL beats Q3 estimates, profit up 15% YoY",
      "source": "Moneycontrol",
      "published_at": "2026-04-22T14:30:00+05:30",
      "hours_ago": 52,
      "url": "https://...",
      "summary": "...",
      "relevance_score": 0.95,         // how related to this stock
      "impact": "POSITIVE",            // POS / NEG / NEUTRAL
      "impact_magnitude": "HIGH",      // HIGH / MED / LOW
      "reasoning": "Q3 EPS beat sets up bullish narrative for next 1-2 weeks",
      "cited_in_prediction": true      // did the LLM actually use it?
    }
    // ... all articles considered, with cited_in_prediction flag
  ],

  "predictions": {
    "daily": {
      "direction": "UP",
      "confidence": 72,
      "suggested_entry": {
        "price": 2455,
        "basis": ["current_price", "above_ema_20"],
        "rationale": "Enter at current level — already above key 20-EMA support."
      },
      "target_price": 2480,
      "target_range": [2470, 2495],
      "target_basis": ["resistance_50d", "bollinger_upper", "fibonacci_618"],
      "target_rationale": "Cluster of resistance near ₹2480 — coincides with prior swing high (Mar 12), upper Bollinger band (20,2), and 0.618 Fib retracement of the Jan-Feb decline.",
      "stop_loss": 2420,
      "stop_basis": ["ema_50_support", "atr_1.5x_below_entry", "swing_low_recent"],
      "stop_rationale": "Below 50-EMA (₹2425) which has held twice in last 3 weeks, and 1.5× ATR (₹30) below current price. Invalidates the bullish thesis if breached.",
      "risk_reward": 0.71,
      "reasoning": "Positive Q3 results + sector tailwind from RBI rate pause; technicals confirm uptrend with healthy pullback to support.",
      "key_drivers": [
        { "driver": "Q3 EPS beat by 8%",       "source_type": "news",      "source_ids": ["news-id-1"] },
        { "driver": "RSI bullish divergence",  "source_type": "technical", "source_ids": ["rsi_14"] },
        { "driver": "Volume confirmation",     "source_type": "technical", "source_ids": ["volume_sma_ratio"] },
        { "driver": "RBI dovish tone",         "source_type": "news",      "source_ids": ["news-id-3"] }
      ],
      "historical_accuracy": {
        "predictions_evaluated": 30,
        "hit_rate": 0.62,
        "stop_rate": 0.20,
        "calibration_note": "When system says 70-80% confident, actual hit-rate is 65%"
      }
    },
    "weekly": { /* same shape, weekly horizon */ }
  },

  "meta": {
    "llm_model":               "groq:llama-3.3-70b",
    "articles_fetched":        47,
    "articles_after_dedupe":   23,
    "articles_cited":          8,
    "generation_time_seconds": 8.4,
    "tokens_used":             12500
  }
}
```

### Why this schema is the way it is
- **`news_articles[]` with `cited_in_prediction`** — full audit trail; UI can show "all considered" + highlight "actually used"
- **`technical_snapshot`** — every indicator value the LLM saw, exposed for the UI / audit
- **`*_basis` + `*_rationale` for entry/target/stop** — forces the LLM to ground every number in actual signals; can't pull numbers out of thin air
- **`key_drivers` with `source_ids`** — every driver traceable back to either an article or a specific indicator
- **`historical_accuracy`** — the system's own track record for this stock × timeframe, displayed alongside the new prediction

---

## 11. 🧪 Backtest & Evaluation

### Per-prediction outcomes
For each historical prediction, evaluated within its timeframe window:

| Outcome              | Logic                                                     |
| -------------------- | --------------------------------------------------------- |
| ✅ **TARGET_HIT**     | Price entered `target_range` before `stop_loss` triggered |
| ❌ **STOP_HIT**       | Price hit `stop_loss` before `target_range`               |
| ⚪ **NEITHER**        | Window ended, neither happened                            |
| 🟡 **DIRECTION_ONLY** | Soft credit — direction correct but target not reached    |

### Aggregated metrics
- **Hit-rate** = `TARGET_HIT / total predictions`
- **Stop-rate** = `STOP_HIT / total`
- **Avg R:R when right** vs **avg loss when wrong**
- **Hypothetical P&L** if every signal were traded at suggested entry/stop/target (most honest metric)
- **Direction accuracy** — broader sanity check
- **Calibration curve** — "when system says 80% confident, is it right ~80% of the time?"

### Slice-and-dice analytics
The `*_basis` tags let us answer:
- "Do predictions with `stop_basis = ema_support` outperform `atr_only`?"
- "Which `target_basis` clusters work best for energy stocks?"
- "Does `news-driven` outperform `technical-driven` predictions?"

### Backtest data
- **Historical prices:** yfinance (years of daily, multi-month of intraday)
- **Historical news:** GDELT (back to 2015, structured event data for India)

Note: Backtest uses GDELT title + entity data, not full article body. This is a deliberate trade-off (free, stable, sufficient signal). Live runs use the same GDELT pipeline → backtest predictions are comparable to live predictions.

---

## 12. 📓 Prediction Tracking & Calibration

Every live prediction is persisted to `data/predictions.sqlite`:

**Tables:**
- `predictions` — full JSON blob + indexed columns (`ticker`, `as_of`, `timeframe`, `direction`, `confidence`)
- `articles` — deduped article store (same article may inform multiple stocks)
- `prediction_articles` — join table (which articles influenced which predictions)
- `outcomes` — populated retroactively (background job): TARGET_HIT / STOP_HIT / NEITHER / DIRECTION_ONLY

**Calibration loop:**
1. Generate new prediction
2. Query tracking DB: "what's our hit-rate for this stock × timeframe over last N predictions?"
3. Attach `historical_accuracy` block to the new prediction
4. Optional v2+: auto-adjust raw confidence based on calibration curve

---

## 13. ⚙️ Concurrency & Rate-Limit Strategy

**Async pipeline using `asyncio` with semaphores:**

```python
HTTP_SEMAPHORE = asyncio.Semaphore(20)   # price + news fetches (no rate limit)
LLM_SEMAPHORE  = asyncio.Semaphore(5)    # LLM calls (rate-limited)
```

**Rate-limit handling:**
- Primary: Groq free tier (~30 req/min, fast Llama 3.3 70B)
- Fallback: Google Gemini free tier (15 req/min, 1M tokens/day, Gemini 2.0 Flash)
- Router auto-fails over on rate-limit / error
- Per-call retry with exponential backoff

**Expected runtime (full Nifty50, on-demand):** ~5 minutes
- ~2-3 LLM calls per stock × 50 stocks = 100-150 LLM calls
- Heavy caching: news fetched once per run (shared across stocks where applicable), KB read once

---

## 14. 🤖 LLM Strategy

| Use case                                            | LLM call type                                                                                                    |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **KB bootstrap** (one-time per stock)               | Single call, structured JSON extraction from Wikipedia + NSE data                                                |
| **News impact scoring** (per article, per stock)    | Cheap call: relevance + impact + magnitude + reasoning                                                           |
| **Prediction reasoning** (per stock, per timeframe) | Main call: takes KB + technicals + scored news → grounded prediction with all `*_basis` and `*_rationale` fields |

**Prompt principles:**
- Always provide the LLM with a **structured menu of available signals** (RSI level, BB bands, support/resistance prices, EMAs, swing points, ATR) and require it to **cite signal names** when filling `*_basis` fields
- Always provide news articles with stable IDs and require LLM to **cite article IDs** in `key_drivers`
- Always require the LLM to set `cited_in_prediction` per article — forces honest accounting of what was actually used vs ignored

---

## 15. 🧰 Tech Stack Summary

| Layer       | Choice                   | Why                             |
| ----------- | ------------------------ | ------------------------------- |
| Language    | Python 3.13              | Modern, async-first             |
| Package mgr | `uv`                     | Fast, deterministic             |
| Validation  | Pydantic                 | Schema-as-code for predictions  |
| Async       | `asyncio`                | Built-in, no extra deps         |
| LLM clients | Groq SDK + Google AI SDK | Free tiers, fast                |
| Prices      | `yfinance`               | Free, reliable, NSE support     |
| News        | GDELT (raw HTTP)         | Free, historical, India-indexed |
| Technicals  | `pandas-ta`              | Pure Python, comprehensive      |
| Storage     | SQLite                   | Zero-setup, file-based          |
| HTTP        | `httpx` (async)          | Async-native                    |
| Caching     | SQLite-backed            | Same store, simple              |
| CLI         | `typer` or `click`       | TBD during impl                 |

---

## 16. 🚧 Risks & Mitigations

| Risk                                                     | Mitigation                                                                                                        |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Free LLM rate limits hit during full-Nifty50 runs**    | Groq + Gemini fallback router; aggressive caching; batch LLM calls where possible                                 |
| **GDELT news quality / coverage gaps**                   | Augment with NSE corporate announcements; deduplicate aggressively; weight by source quality                      |
| **LLM hallucinated relationships in KB**                 | Phase 1 KB structured + auto-built from authoritative sources; LightRAG deferred until structured layer is proven |
| **Backtest fidelity (GDELT title-only vs full article)** | Use same GDELT pipeline for live runs → backtest and live are comparable; full-article scraping deferred          |
| **Predictions used for real money before validation**    | Backtest + tracking + calibration all first-class; user is responsible for validation before any real capital     |
| **Web scraping fragility**                               | Avoided in v1 — only structured/free APIs                                                                         |
| **Free service deprecation (Groq, Gemini)**              | LLMClient interface allows swap to new provider with no other code changes                                        |

---

## 17. 📜 Decision Log

| Decision                                                  | Why                                                                  | Alternatives considered                                  |
| --------------------------------------------------------- | -------------------------------------------------------------------- | -------------------------------------------------------- |
| Daily + weekly only (no hourly)                           | Hourly predictions on free data are too noisy                        | Hourly was original ask, dropped after honest discussion |
| GDELT for news (not scraping)                             | Free, stable, historical, India-indexed                              | Scraping Moneycontrol/ET — too fragile                   |
| Structured KB Phase 1, LightRAG Phase 2                   | Validate end-to-end first; structured is deterministic + inspectable | LightRAG from day 1 — too risky / complex up front       |
| `pandas-ta` over `TA-Lib`                                 | Pure Python, no C compilation needed                                 | TA-Lib has more patterns but install is painful          |
| Groq primary, Gemini fallback                             | Best free-tier speed + quality combo as of 2026-04                   | OpenAI / Anthropic (paid); Ollama (needs local hardware) |
| SQLite over Postgres                                      | Zero setup, file-based, fits "any laptop" constraint                 | Postgres — overkill for v1                               |
| On-demand v1, scheduled v2                                | Simpler to build & validate; scheduling is wrapper around same code  | Scheduled from day 1 — premature                         |
| API-based LLM (no local)                                  | "Runs on any laptop, nothing installed" rules out Ollama             | Ollama — requires 16GB+ RAM and model download           |
| 7-day general news window, 30-day filings                 | Captures slow-burn stories; filings have longer tail                 | Single window — too restrictive                          |
| Self-contained JSON output                                | UI / audit need no extra lookups                                     | Normalized DB-only — UI gets complicated                 |
| Explainable entry/target/stop (`*_basis` + `*_rationale`) | Forces grounded numbers; aids debugging + backtest analytics         | Plain numbers — no audit trail                           |
| Trading style irrelevant to design                        | System emits both daily + weekly; user picks                         | Optimize for one — limits flexibility                    |
| **JSON-on-disk for predictions/grades** (not SQLite)      | Inspectable; no migration cost; backup is `cp -r`; YAGNI at our scale (~36k files/yr) | SQLite — wins above ~1M rows; we're 30x below |
| **Three hit-rate variants reported** (strict/resolved/optimistic) | Same-bar T+S ambiguity makes any single number lossy; reporting all three is honest, picking one is cherry-picking | Pick one — dishonest |
| **Brier score over log-loss** for confidence calibration  | Bounded [0,1]; no log(0) edge case at confidence=1.0; quadratic penalty matches user intuition; 0.25 baseline (always p=0.5) is memorable | Log-loss — unbounded, edge cases |
| **Six grading outcomes**, not pass/fail                   | Pass/fail is lossy; ambiguous/expired/N/A/inconclusive carry signal | Binary outcome — throws away signal |
| **`output_schema=Prediction` on synthesizer**             | Forces LLM to emit valid Pydantic JSON; zero parse logic our side; pairs perfectly with frozen models | Free-form output + parse — brittle |
| **Hallucination guardrails Tiers 1-3 + retry-with-feedback** | Without them synthesizer invented target prices on wrong side of entry; retry fixed ~80% of failures | Trust the LLM — no |

---

## 18. 📚 Reference Links

- **Anthropic — Building Effective Agents:** https://www.anthropic.com/research/building-effective-agents
- **GDELT Project:** https://www.gdeltproject.org/
- **yfinance docs:** https://github.com/ranaroussi/yfinance
- **pandas-ta docs:** https://github.com/twopirllc/pandas-ta
- **Groq API:** https://console.groq.com/docs
- **Google Gemini API:** https://ai.google.dev/
- **LightRAG (HKU):** https://github.com/HKUDS/LightRAG
- **NSE corporate announcements:** https://www.nseindia.com/companies-listing/corporate-filings-announcements
- **Hamel Husain — AI evals:** https://hamel.dev/blog/posts/evals/

---

## 19. ❓ Open Questions (post-spec)

These come up during implementation, not design:

- Final project name (currently *Predictor* placeholder)
- Exact set of candlestick / chart patterns to detect (will iterate based on `patterns` field utility)
- Initial set of `macro_sensitivities` taxonomy (start small, extend as observed in news)
- LLM prompt templates (will iterate during build)
- Outcome evaluation cron cadence (daily? hourly?)
- Initial Nifty50 ticker list source (NSE official; periodic check for index reconstitution)

---

## 20. 📝 Notes

- This spec is a living document. Update sections + add to the decision log as we build.
- Implementation discussion happens **after** this doc is reviewed.
- Any scope change → update this doc first, then code.
