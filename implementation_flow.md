# Implementation Flow — Stock Predictor v1

> **Scope**: This doc tracks the **multi-step build of the v1 Stock Predictor**
> (Steps A, B, C). It complements `implementation_plan.md` (which is the higher-
> level project roadmap) and `next_steps.md` (which details what's coming next).
>
> **Last updated**: 2026-04-28 — after Step B.1 (provider refactor) landed.

---

## 🎯 The multi-step build

We agreed to build the v1 predictor in three sequential layers, each
self-contained and testable:

```
┌─ Step A ─────────────────────────────────────────────────────────┐
│  Knowledge Base — universal stock registry (Wikipedia-sourced)    │
│  STATUS: ✅ DONE                                                   │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ Step B ─────────────────────────────────────────────────────────┐
│  technical_agent — 4 thematic ADK tools for indicator analysis    │
│  STATUS: 🟡 IN PROGRESS (B.1 of 10 sub-steps complete)            │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ Step C ─────────────────────────────────────────────────────────┐
│  prediction_agent — orchestrator that combines KB + technicals    │
│  + news + filings into a structured prediction                    │
│  STATUS: ⏸️ NOT STARTED                                            │
└───────────────────────────────────────────────────────────────────┘
```

---

## ✅ Step A — Knowledge Base (DONE)

**Goal**: Replace the hardcoded `TICKER_ALIASES` dict with a Wikipedia-sourced
universal stock registry that supports multi-index membership from day one.

### What landed

| Artifact | Purpose |
|---|---|
| `scripts/bootstrap_indices.py` | Fetches Wikipedia, merges per-index data into universal JSON |
| `data/kb/stocks.json` | Universal registry, 50 stocks, multi-index ready, **committed** |
| `data/kb/indices.json` | Per-index metadata (source URL, last_updated), **committed** |
| `src/price_predictor/kb/stocks.py` | Runtime API: `Stock`, `lookup()`, `by_index()`, `all_stocks()` |
| `tests/test_kb_stocks.py` | 36 behavior tests pinning lookup tiers + edge cases |

### What was deleted (replaced)

- `src/price_predictor/data/nse_tickers.py` — hardcoded ticker→name dict
- `tests/test_nse_tickers.py` — tested the deleted module

### What was refactored (same external contract, new internals)

- `agents/price_agent/agent.py` — uses `kb.stocks.lookup()` instead of
  `suggest_alternative()`; same JSON shape on the wire
- `agents/news_impact/agent.py` — prompt drops the hardcoded GOTCHAS list
  (HDFC→HDFCBANK, L&T→LT, etc); now relies on `suggested_ticker` field

### Key design decisions

1. **Bootstrap from live Wikipedia** (not hardcoded) — index reconstitutes
   quarterly; hardcoded data goes stale silently.
2. **JSON committed to repo** — fresh clone works without setup.
3. **NOT exposed as an ADK tool** — KB lookups are knowledge, not actions.
   Tools cost an LLM round-trip; lookups should be Python imports.
4. **Fuzzy matching, not alias dict** — substring/RapidFuzz matching against
   live names self-heals on mergers (e.g. 2023 HDFC merger).
5. **Lookup tier order**: exact ticker > exact name > substring (shortest
   wins) > fuzzy on name > fuzzy on ticker.
6. **Multi-index from day one** — `Stock.indices: list[str]` lets the same
   stock belong to NIFTY50, Bank Nifty, etc. without duplication.

### Edge cases caught by tests

- `lookup('!!!!')` originally returned ITC (the shortest-named stock) because
  the normalizer stripped non-alphanumerics to empty string, and `''` is a
  substring of every name. **Fixed**: bail early on empty normalized query.

### Test count delta: 289 → 308 (+19 net, after test churn)

---

## 🟡 Step B — Technical Agent (IN PROGRESS — 1 of 10 done)

**Goal**: Build a `technical_agent` that exposes 4 thematic ADK tools
(`get_trend`, `get_momentum`, `get_volatility`, `get_levels`), each
returning indicators + a `signal` enum, backed by a cached price layer.

### Locked design (from discussion)

- **Cluster shape**: 4 thematic tools (option C) — not 1 mega-tool, not 20 micro-tools
- **Output**: numbers + signal enum (option Y) — both raw and interpreted
- **Data flow**: self-contained tools + range-aware cache (option R)
- **Tool knobs**: 1 `sensitivity` preset per tool (`standard | sensitive | smooth`) — semantic, not raw integers
- **Pattern home** (option β): candlestick → `get_momentum`, chart → `get_levels`
- **History window**: 1 year of daily bars (~252 trading days)
- **Cache**: range-aware, in-memory, per-ticker async lock, no persistence
- **Library**: `pandas-ta` for everything except 3 hand-rolled chart patterns
- **Provider strategy**: yfinance only for v1; Stooq/NSE direct slot in later via the resilient layer

### Sub-step progress

| # | Sub-step | Status | Test count |
|---|---|---|---|
| B.1 | Provider pattern refactor (`PriceProvider` ABC + `YFinanceProvider` + `ResilientPriceFetcher`) | ✅ DONE | 308 → 322 (+14) |
| B.2 | Cache layer (range-aware in-memory, per-ticker `asyncio.Lock`) | ⏸️ NEXT | — |
| B.3 | Indicator primitives (pure functions: trend math, momentum math, volatility math, levels math) | ⏸️ NOT STARTED | — |
| B.4 | Chart pattern detectors (3 hand-rolled: double top/bottom, head & shoulders, triangles) | ⏸️ NOT STARTED | — |
| B.5 | `get_trend` tool wiring (first cluster end-to-end) | ⏸️ NOT STARTED | — |
| B.6 | `get_momentum` tool wiring (+ candlestick patterns folded in) | ⏸️ NOT STARTED | — |
| B.7 | `get_volatility` tool wiring | ⏸️ NOT STARTED | — |
| B.8 | `get_levels` tool wiring (+ chart patterns folded in) | ⏸️ NOT STARTED | — |
| B.9 | `technical_agent` wiring (4 tools → `LlmAgent` with prompt + structured output) | ⏸️ NOT STARTED | — |
| B.10 | Manual smoke test in `adk web` | ⏸️ NOT STARTED | — |

### What B.1 (provider refactor) delivered

| Artifact | Purpose |
|---|---|
| `data/providers/__init__.py` | Public exports |
| `data/providers/base.py` | `PriceProvider` ABC + `PriceFetchError` (the contract) |
| `data/providers/yfinance_provider.py` | yfinance implementation (column normalize, tz, error wrap) |
| `data/providers/resilient.py` | `ResilientPriceFetcher` (cooldowns, fallback, error classification) |
| `data/prices.py` | Refactored to thin shim; public API unchanged |
| `tests/test_resilient_price_fetcher.py` | 14 new tests (construction, happy path, fallback, ValueError propagation, cooldown behavior) |
| `tests/test_prices.py` | Mock paths updated to new module location |

### Key design decisions in Step B (so far)

1. **ABC over Protocol** for the provider interface — explicit subclassing
   is clearer for a learning codebase; type-checked at definition time.
2. **Two close columns documented on the interface** — `close` (raw,
   what traded, target/SL math) vs `adj_close` (split-adjusted, indicators).
3. **Cooldown is per-provider, not per-(provider,ticker)** — rate limits
   hit the API, not specific symbols.
4. **In-memory cooldown only** — no persistence; process restart resets.
5. **"Last resort" branch** — if all providers are cooled, try them anyway
   (better to get rate-limited than to fail with no answer).
6. **Public API unchanged** — every existing import still works:
   `from price_predictor.data.prices import fetch_ohlcv, PriceFetchError`

### Test count delta so far in Step B: 308 → 322 (+14)

---

## ⏸️ Step C — Prediction Agent (NOT STARTED)

**Goal**: An orchestrator agent that combines outputs from `kb`, `price_agent`,
`technical_agent`, `news_impact`, and `filings` into a structured prediction
(direction, confidence, entry, target, stop, risk-reward).

To be designed in detail after Step B is complete. Open questions parked in
`next_steps.md`.

---

## 📁 Current file structure (relevant portions)

```
src/price_predictor/
├── agents/
│   ├── hello_agent/                    # Learning spike (DONE)
│   ├── price_agent/                    # Refactored to use KB (Step A)
│   └── news_impact/                    # Prompt simplified by KB (Step A)
├── data/
│   ├── prices.py                       # Thin shim (Step B.1)
│   ├── providers/                      # NEW (Step B.1)
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── yfinance_provider.py
│   │   └── resilient.py
│   ├── estimates.py
│   ├── filings.py
│   ├── news.py
│   └── schema.py
├── kb/                                 # NEW (Step A)
│   ├── __init__.py
│   └── stocks.py
├── llm/
│   ├── factory.py
│   └── resilient.py                    # Pattern reused for Step B.1
└── config/
    └── settings.py

data/kb/
├── stocks.json                         # NEW, COMMITTED (Step A)
└── indices.json                        # NEW, COMMITTED (Step A)

scripts/
└── bootstrap_indices.py                # NEW (Step A)

tests/
├── test_kb_stocks.py                   # NEW, 36 tests (Step A)
├── test_prices.py                      # Updated mock paths (Step B.1)
└── test_resilient_price_fetcher.py     # NEW, 14 tests (Step B.1)
```

---

## 📊 Test history

| After step | Total tests | Delta | Notes |
|---|---|---|---|
| Pre-Step-A baseline | 289 | — | |
| Step A complete | 308 | +19 | KB tests added, deleted `test_nse_tickers.py` |
| Step B.1 complete | **322** | **+14** | Resilient layer tests added; existing tests preserved |

---

## 🎓 Lessons accumulated

### From Step A
1. ADK tools = actions; modules = knowledge. Don't tool-ify lookups.
2. Bootstrap-and-cache > hardcode > fetch-every-call.
3. Fuzzy matching > alias dictionaries (self-healing on data changes).
4. Open/Closed: small extra design today saves a refactor in 2 weeks.
5. Wikipedia 403s default Python UA — always use a browser UA.

### From Step B.1
1. Resilience patterns generalize — LLM fallback chain and price-source
   fallback chain are the SAME shape.
2. Error classification > error catching. `ValueError` (caller bug)
   propagates immediately; `PriceFetchError` (upstream issue) triggers fallback.
3. The interface contract belongs on the ABC, not in each implementation.
4. Open/Closed pays off the SECOND time you swap a provider — first time
   it just looks like extra structure; second time it's free.
5. (Meta-lesson) When we agree on a build plan together, sticking to it
   is the contract. Don't unilaterally YAGNI parts of an agreed plan.
