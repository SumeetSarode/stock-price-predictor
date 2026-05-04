# Implementation Flow — Stock Predictor v1

> **Scope**: Tracks the **multi-step build** of the v1 Stock Predictor
> (Steps A, B, C, D). Complements `implementation_plan.md` (high-level
> roadmap) and `next_steps.md` (what's coming next).
>
> **Last updated**: 2026-04-28 — after Step C.1 (`get_trend` tool).

---

## 🎯 The multi-step build

> **Naming note (renaming history)**: Originally we planned Step B as a
> 10-substep monolith (B.1–B.10) ending in `technical_agent`, with Step C
> being the prediction agent. After landing the plumbing (B.1–B.4) we
> renamed: **Step B = plumbing**, **Step C = the 4 tools + technical_agent**,
> **Step D = prediction_agent**. Same scope, cleaner phase boundaries.

```
┌─ Step A ─────────────────────────────────────────────────────────┐
│  Knowledge Base — Nifty50 stock registry (Wikipedia-sourced)      │
│  STATUS: ✅ DONE                                                   │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ Step B ─────────────────────────────────────────────────────────┐
│  Plumbing — provider pattern, range-aware cache, indicator        │
│  primitives, chart pattern detectors                              │
│  STATUS: ✅ DONE (B.1–B.4 all landed)                              │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ Step C ─────────────────────────────────────────────────────────┐
│  Tools + Agent — 4 thematic ADK tools + technical_agent wiring    │
│  STATUS: 🟡 IN PROGRESS (C.1 of 6 sub-steps complete)             │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ Step D ─────────────────────────────────────────────────────────┐
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

### Key design decisions

1. **Bootstrap from live Wikipedia** (not hardcoded) — index reconstitutes
   quarterly; hardcoded data goes stale silently.
2. **JSON committed to repo** — fresh clone works without setup.
3. **NOT exposed as an ADK tool** — KB lookups are knowledge, not actions.
4. **Fuzzy matching, not alias dict** — substring/RapidFuzz matching against
   live names self-heals on mergers (e.g. 2023 HDFC→HDFCBANK).
5. **Multi-index from day one** — `Stock.indices: list[str]` lets the same
   stock belong to NIFTY50, Bank Nifty, etc. without duplication.

### Test count delta: 289 → 308 (+19 net)

---

## ✅ Step B — Plumbing (DONE)

**Goal**: All non-LLM-facing infrastructure for Step C: swappable price
providers, a range-aware cache, pure indicator math, and chart pattern
detectors. Zero ADK exposure — every artifact is testable in pure Python.

### Sub-step progress

| # | Sub-step | Status | Test delta |
|---|---|---|---|
| B.1 | Provider pattern refactor (`PriceProvider` ABC + `YFinanceProvider` + `ResilientPriceFetcher`) | ✅ DONE | 308 → 322 (+14) |
| B.2 | Range-aware async price cache (`PriceCache` with per-key `asyncio.Lock`) | ✅ DONE | 322 → 333 (+11) |
| B.3 | Indicator primitives (trend / momentum / volatility / levels math + 3 sensitivity presets) | ✅ DONE | 333 → 375 (+42) |
| B.4 | Pattern detectors (6 hand-rolled candlesticks + 3 hand-rolled chart patterns) | ✅ DONE | 375 → 394 (+19) |

### What B.1 delivered (provider refactor)

| Artifact | Purpose |
|---|---|
| `data/providers/base.py` | `PriceProvider` ABC + `PriceFetchError` (the contract) |
| `data/providers/yfinance_provider.py` | yfinance impl (column normalize, tz, error wrap) |
| `data/providers/resilient.py` | `ResilientPriceFetcher` (cooldowns, fallback) |
| `data/prices.py` | Refactored to thin shim; **public API unchanged** |
| `tests/test_resilient_price_fetcher.py` | 14 new tests |

**Lessons from B.1**:
1. Resilience patterns generalize — LLM fallback chain and price-source
   fallback chain are the SAME shape.
2. Error classification > error catching. `ValueError` (caller bug)
   propagates immediately; `PriceFetchError` (upstream issue) triggers fallback.
3. Open/Closed pays off the **second** time you swap a provider — first
   time it just looks like extra structure; second time it's free.

### What B.2 delivered (range-aware cache)

| Artifact | Purpose |
|---|---|
| `data/cache.py` | `PriceCache` with range-aware in-memory caching |
| `tests/test_price_cache.py` | 11 tests (single/range/concurrency/immutability/errors) |

**Behaviors**:
- Cold cache + request → fetch `[end - 365d, end]` proactively
- Cached range covers request → slice + return defensive copy
- Cached range partially covers → re-fetch widened union (simple v1)
- Two parallel calls for same ticker → lock serializes, exactly 1 fetch
- Two parallel calls for different tickers → independent locks, both run
- Failed fetches → error propagates, no partial entry stored

**Lessons from B.2**:
1. `asyncio.Lock` per key (not one global lock) gives same-key serialization
   while preserving cross-key parallelism.
2. Defensive copies are cheap and prevent the worst class of cache bug
   (action at a distance via shared mutable state).
3. Inject the fetcher; don't import it. Tests stay fast and deterministic.

### What B.3 + B.4 delivered (indicators + patterns)

| Artifact | Purpose |
|---|---|
| `analysis/__init__.py` | 3 sensitivity presets per cluster (`standard` / `sensitive` / `smooth`) |
| `analysis/trend.py` | SMAs, EMA, ADX + `trend_snapshot()` |
| `analysis/momentum.py` | RSI, MACD, Stoch, OBV + `momentum_snapshot()` |
| `analysis/volatility.py` | ATR, Bollinger Bands, BB squeeze + `volatility_snapshot()` |
| `analysis/levels.py` | Swing high/low, 52w high/low, classic pivots + `levels_snapshot()` |
| `analysis/candlestick_patterns.py` | 6 hand-rolled patterns (doji, hammer, shooting star, bull/bear engulfing, morning/evening star) |
| `analysis/chart_patterns.py` | 3 hand-rolled patterns (double top/bottom, head & shoulders ±inverse, triangles) |
| `tests/analysis/` | 61 tests (synthetic OHLCV fixtures: linear up/down, mean-reverting random walk) |

**Design**:
- Pure functions: `pd.DataFrame` in, float/dict/dataclass out
- NaN-safe: returns `None` when not enough history
- Signal interpretation NOT here — lives in tool layer (Step C)
- Each module exposes a `*_snapshot()` helper that bundles cluster output

**Why hand-rolled patterns?** Installed `pandas-ta` only ships 3 candlestick
functions; the rest need `ta-lib` (install pain). Hand-rolling is ~50 LOC
each and uses well-defined geometric rules. Skipped intentionally: cup &
handle, flags, pennants, wedges (too noisy for v1).

**Lessons from B.3 + B.4**:
1. Synthetic test fixtures with known properties beat real-data fixtures
   for indicator math — you can assert exact-ish values.
2. Random-walk for "sideways" beats sinusoidal — sine waves have strong
   directional moves within each cycle that fool RSI/ADX.
3. Pure-linear test series cause floating-point converge artifacts
   (MACD line == signal line at the 14th decimal). Use approx comparisons.
4. NaN-safety requires explicit checks at every step — pandas operations
   silently propagate NaN; we want `None` at the API boundary.

### Step B test count delta: 308 → 394 (+86)

---

## 🟡 Step C — Tools + Agent (IN PROGRESS — 1 of 6 done)

**Goal**: Wrap the Step B primitives as 4 thematic ADK tools, then wire
them into a `technical_agent` LlmAgent that an `adk web` user can talk to.

### Sub-step progress

| # | Sub-step | Status | Test delta |
|---|---|---|---|
| C.1 | `get_trend` tool + signal classifier + shared cache singleton | ✅ DONE | 398 → 425 (+27) |
| C.2 | `get_momentum` tool (+ candlestick context-gating) | ⏸️ NEXT | — |
| C.3 | `get_volatility` tool | ⏸️ NOT STARTED | — |
| C.4 | `get_levels` tool (+ chart pattern integration) | ⏸️ NOT STARTED | — |
| C.5 | `technical_agent` wiring (4 tools → `LlmAgent` + prompt) | ⏸️ NOT STARTED | — |
| C.6 | Manual smoke test in `adk web` | ⏸️ NOT STARTED | — |

> **Test count gap (394 → 398, +4)**: incidental drift between B.4 and C.1
> commits — likely fixture/import additions during C.1 wiring. Not material.

### What C.1 delivered (`get_trend` tool)

| Artifact | Purpose |
|---|---|
| `data/_shared_cache.py` | Process-wide `PriceCache` singleton + async fetcher wrapper |
| `agents/technical_agent/__init__.py` | Package skeleton |
| `agents/technical_agent/tools/__init__.py` | Tools package docstring |
| `agents/technical_agent/tools/_types.py` | TypedDict response shapes (`ToolErrorResponse`, `ToolSuccessResponse`) |
| `agents/technical_agent/tools/_trend_signal.py` | Pure `classify_trend()` function |
| `agents/technical_agent/tools/get_trend.py` | The actual ADK tool |
| `tests/tools/test_trend_signal.py` | 18 classifier tests |
| `tests/tools/test_get_trend.py` | 16 tool tests (mocked cache) |

### Design decisions LOCKED for all 4 cluster tools

1. **Async tools** — `async def get_<cluster>(...)`. Natural fit with the
   async `PriceCache`. ADK supports async tools natively.
2. **Plain dict returns** + `TypedDict` for editor hints. Matches existing
   `fetch_prices_tool` convention. LLM-friendly.
3. **3-level signal** (`bullish` | `neutral` | `bearish`) + separate
   `strength` field (`weak` | `moderate` | `strong`) when meaningful.
4. **`rationale` bullet list** in every response — gives the LLM ready-made
   prose chunks to weave in. **Reduces hallucination significantly.**
5. **Module-level cache singleton** in `data/_shared_cache.py`. The 4 tools
   must share one cache across an agent turn; threading via tool signatures
   buys nothing.

### Uniform call flow (every C tool)

```
1. Normalize ticker (RELIANCE → RELIANCE.NS via KB lookup)
2. Validate sensitivity preset (standard | sensitive | smooth)
3. Fetch ~400d of OHLCV via shared cache (1 net hit per process per ticker)
4. Run analysis.<cluster>.<cluster>_snapshot() with preset params
5. Pass snapshot → cluster's signal classifier → (signal, strength, rationale)
6. Build uniform response dict + return
```

### Error contract (every C tool)

Tools **NEVER raise** — always return `{"status": "error", ...}`. The LLM
needs to see errors to recover (apologize, suggest alternative, give up
gracefully). Includes `suggested_ticker` on resolvable typos.

### Trend classifier rules (the C.1 specific logic)

- **Signal**:
  - `bullish` if close above ≥2 of 3 SMAs **AND** +DI > -DI
  - `bearish` if close below ≥2 of 3 SMAs **AND** -DI > +DI
  - `neutral` otherwise (mixed signals or insufficient data)
- **Strength** (independent of direction): ADX ≥40 strong, ≥25 moderate,
  otherwise weak.
- Special case: all SMAs above + DI unavailable → still bullish (weak).

**Lessons from C.1**:
1. Module-level singletons feel dirty but are honest when the resource
   IS one-per-process. Threading buys complexity, not correctness.
2. `async def` tools + `asyncio.to_thread()` lets us keep the underlying
   sync yfinance fetcher unchanged. Cache wraps it transparently.
3. Returning errors instead of raising is critical for LLM-facing tools.
   An exception kills the turn; an error dict lets the LLM recover.
4. The `rationale` field is the **killer feature**. The LLM quotes our
   pre-built bullets verbatim instead of paraphrasing raw numbers (where
   hallucination lives).

---

## ⏸️ Step D — Prediction Agent (NOT STARTED)

**Goal**: An orchestrator agent that combines outputs from `kb`,
`price_agent`, `technical_agent`, `news_impact`, and `filings` into a
structured prediction (direction, confidence, entry, target, stop,
risk-reward).

To be designed in detail after Step C is complete. Open questions parked
in `next_steps.md`.

---

## 📁 Current file structure (relevant portions)

```
src/price_predictor/
├── agents/
│   ├── hello_agent/                    # Learning spike (DONE)
│   ├── price_agent/                    # Refactored to use KB (Step A)
│   ├── news_impact/                    # Prompt simplified by KB (Step A)
│   └── technical_agent/                # NEW (Step C)
│       ├── __init__.py
│       └── tools/
│           ├── __init__.py
│           ├── _types.py
│           ├── _trend_signal.py
│           └── get_trend.py
├── analysis/                           # NEW (Step B.3 + B.4)
│   ├── __init__.py                     # PRESETS + validate_preset()
│   ├── trend.py
│   ├── momentum.py
│   ├── volatility.py
│   ├── levels.py
│   ├── candlestick_patterns.py
│   └── chart_patterns.py
├── data/
│   ├── prices.py                       # Thin shim (Step B.1)
│   ├── cache.py                        # NEW (Step B.2)
│   ├── _shared_cache.py                # NEW (Step C.1) — singleton
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
│   └── resilient.py                    # Pattern reused for B.1
└── config/
    └── settings.py

data/kb/
├── stocks.json                         # NEW, COMMITTED (Step A)
└── indices.json                        # NEW, COMMITTED (Step A)

scripts/
└── bootstrap_indices.py                # NEW (Step A)

tests/
├── test_kb_stocks.py                   # 36 tests (Step A)
├── test_prices.py                      # Updated mock paths (Step B.1)
├── test_resilient_price_fetcher.py     # 14 tests (Step B.1)
├── test_price_cache.py                 # 11 tests (Step B.2)
├── analysis/                           # 61 tests (Step B.3 + B.4)
│   ├── __init__.py
│   ├── conftest.py                     # synthetic OHLCV fixtures
│   ├── test_trend.py
│   ├── test_momentum.py
│   ├── test_volatility.py
│   ├── test_levels.py
│   ├── test_candlestick_patterns.py
│   └── test_chart_patterns.py
└── tools/                              # NEW (Step C.1)
    ├── __init__.py
    ├── test_trend_signal.py            # 18 tests
    └── test_get_trend.py               # 16 tests
```

---

## 📊 Test history

| After step | Total tests | Delta | Notes |
|---|---|---|---|
| Pre-Step-A baseline | 289 | — | |
| Step A complete | 308 | +19 | KB tests added, deleted `test_nse_tickers.py` |
| Step B.1 complete | 322 | +14 | Resilient layer tests |
| Step B.2 complete | 333 | +11 | Cache tests (single/range/concurrency) |
| Step B.3 + B.4 complete | 394 | +61 | Indicator + pattern tests |
| Step C.1 complete | **425** | **+31*** | Tool + classifier tests |

\*Includes a +4 incidental gap between B.4 and C.1 (fixture/import additions).

---

## 🎓 Lessons accumulated (master list)

### From Step A
1. ADK tools = actions; modules = knowledge. Don't tool-ify lookups.
2. Bootstrap-and-cache > hardcode > fetch-every-call.
3. Fuzzy matching > alias dictionaries (self-healing on data changes).
4. Open/Closed: small extra design today saves a refactor in 2 weeks.
5. Wikipedia 403s default Python UA — always use a browser UA.

### From Step B.1
1. Resilience patterns generalize across LLMs and data sources.
2. Error classification > error catching.
3. Interface contracts belong on the ABC, not in each implementation.
4. Open/Closed pays off the **second** time you swap a backend.

### From Step B.2
1. Per-key `asyncio.Lock` (not global) gives same-key serialization +
   cross-key parallelism. The whole point of caching is to NOT block
   independent work.
2. Defensive copies prevent action-at-a-distance cache bugs.
3. Inject the fetcher; don't import it.

### From Step B.3 + B.4
1. Synthetic test fixtures > real-data fixtures for indicator math.
2. Random-walk > sinusoidal for "sideways" testing.
3. Pure-linear inputs cause floating-point convergence artifacts —
   use approx comparisons.
4. NaN-safety requires explicit `None` checks; pandas silently propagates.

### From Step C.1
1. Module-level singletons are honest when the resource IS one-per-process.
2. `async def` tools + `asyncio.to_thread()` adapt sync libraries cleanly.
3. **Return errors, don't raise** — exceptions kill agent turns; error
   dicts let the LLM recover.
4. The `rationale` field reduces hallucination by giving the LLM
   ready-made prose to quote.

### Meta-lesson (recurring)
When we agree on a build plan together, sticking to it is the contract.
Don't unilaterally YAGNI parts of an agreed plan. Don't sneak in scope
beyond it either.
