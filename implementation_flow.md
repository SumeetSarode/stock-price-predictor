# Implementation Flow — Stock Predictor v1

> **Scope**: Tracks the **multi-step build** of the v1 Stock Predictor
> (Steps A, B, C, D). Complements `implementation_plan.md` (high-level
> roadmap) and `next_steps.md` (what's coming next).
>
> **Last updated**: 2026-04-28 — post Step 3.4.6 (multi-horizon
> hardening; 8 commits closing the "fake multi-horizon" gap from
> Step 3.4.2). Previous update: post Step 3.5 (grading + calibration
> shipped).

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
│  STATUS: ✅ DONE (6 of 6 sub-steps complete)                       │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ Step D ────────────────────────────────────────────────────────┐
│  prediction_agent — synthesizer + predict + batch + store + CLI   │
│  (predict / predict-many / history)                               │
│  STATUS: ✅ DONE                                                   │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ Step 3.5 ────────────────────────────────────────────────────┐
│  Grading + Calibration — grade_one + grade_many +                 │
│  CalibrationReport + CLI (grade / calibration)                    │
│  STATUS: ✅ DONE (3 of 3 commits)                                  │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ Step 3.4.6 ──────────────────────────────────────────────────┐
│  Multi-horizon hardening — trading-calendar helper, horizon enum  │
│  rename, predict() fan-out, per-horizon NEUTRAL tolerance,        │
│  research dossier, horizon_constants.py (single SoT), guardrails  │
│  per-horizon + Tier 4, synthesizer prompt embeds the rules        │
│  STATUS: ✅ DONE (8 of 8 commits; 854 → 1021 tests)                │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─ Step 3.5.5+ ─────────────────────────────────────────────────┐
│  Backtest — replay + runner + evaluator (historical calibration)  │
│  STATUS: ⏸️ NOT STARTED                                            │
└───────────────────────────────────────────────────────────────┘
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
| `analysis/volatility.py` | ATR, Bollinger Bands, `bollinger_squeeze` (Bollinger 2001) + `ttm_squeeze` (Carter 2009) + `volatility_snapshot()` |
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

## ✅ Step C — Tools + Agent (COMPLETE — 6 of 6 done)

**Goal**: Wrap the Step B primitives as 4 thematic ADK tools, then wire
them into a `technical_agent` LlmAgent that an `adk web` user can talk to.

### Sub-step progress

| # | Sub-step | Status | Test delta |
|---|---|---|---|
| C.1 | `get_trend` tool + signal classifier + shared cache singleton | ✅ DONE | 398 → 425 (+27) |
| C.2 | `get_momentum` tool (+ candlestick context-gating) | ✅ DONE | 425 → 474 (+49) |
| C.3 | `get_volatility` tool (+ position-sizing helpers) | ✅ DONE | 474 → 509 (+35) |
| C.4 | `get_levels` tool (+ chart pattern integration) | ✅ DONE | 509 → 538 (+29) |
| C.5 | `technical_agent` wiring (4 tools → `LlmAgent` + prompt) | ✅ DONE | 538 → 554 (+16) |
| C.6 | Manual smoke test in `adk web` (+ LLM-chain bug fix) | ✅ DONE | 554 → 554 (env-only) |

**Final test count: 554** (vs. 515 originally projected — +39 over plan,
driven by tighter coverage on cluster signal classifiers and chart-pattern
edge cases).

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

## 🔧 Provider Expansion (post-C addendum) — DONE

**Goal**: Resolve the C.6-deferred yfinance rate-limit issue by filling
out the resilient provider chain that B.1 had already scaffolded.

**Why now (not at B.1)**: At B.1 the architecture was speculative — we
hadn't yet hit the throttling problem in anger. C.6 smoke testing exposed
it (4 tools × N questions → Yahoo cools down within minutes). Building
the second + third providers AFTER feeling the pain confirmed B.1's
Open/Closed design pays off the second time you swap a backend.

### What landed

| Artifact | Purpose |
|---|---|
| `data/providers/stooq_provider.py` | Free CSV download endpoint; captcha-only key (no signup) |
| `data/providers/alpha_vantage_provider.py` | Free 25 req/day; paid tier toggleable |
| `data/providers/_http.py` | Shared `get_verify_setting()` helper for httpx-based providers (corp-MITM TLS support, defensive certifi fallback) |
| `data/providers/__init__.py` | `PROVIDER_REGISTRY` factory pattern; adding provider #4 = 1 line |
| `config/settings.py` | New fields: `price_chain`, `price_paid`, `use_paid_prices`, `alpha_vantage_api_key`, `stooq_api_key`, `ssl_cert_file`, `requests_ca_bundle`. `effective_price_chain()` mirrors `effective_chain()` for LLMs. |
| `tests/test_stooq_provider.py` | 25 tests — mocked httpx, parametrized ticker translation |
| `tests/test_alpha_vantage_provider.py` | 29 tests — mocked httpx, all error envelopes covered |
| `tests/test_settings.py` | +10 tests — chain parsing, paid toggle, lazy key validation |

### Design decisions

1. **Mirror the LLM `USE_PAID` pattern**. `PRICE_CHAIN` (free chain) +
   `PRICE_PAID` (single paid provider) + `USE_PAID_PRICES` (toggle).
   Same mental model, learned once, applied twice.
2. **Lazy api-key validation**. Empty key OK at construction (so the
   factory builds every registered provider unconditionally), fails at
   fetch time with `PriceFetchError` (so resilient layer falls back
   cleanly). User only configures keys for providers they actually use.
3. **Registry-based factory**. `PROVIDER_REGISTRY: dict[str, Callable]`
   maps short-name → zero-arg factory closure. Closures inject per-provider
   config (api keys) without leaking into the `Settings` constructor.
4. **Tell-Don't-Ask ticker translation**. Caller still passes
   `RELIANCE.NS`. Each provider's `_to_X_ticker()` translates internally.
   Zero churn anywhere upstream.
5. **Shared HTTP helper for httpx providers**. Centralizes corp-MITM TLS
   support + httpx 0.28 deprecation handling. Adding the next HTTP-based
   provider gets it free.

### Real-world findings worth keeping

1. **Stooq added an apikey requirement in 2024.** Still 100% free —
   captcha-only, no signup, no email, no expiry. Empty-key error message
   includes the captcha URL so users self-serve.
2. **httpx ≥ 0.28 deprecated `verify=<str>`.** Switched to
   `ssl.create_default_context()`. One line, one place.
3. **LiteLLM auto-loads `.env` into `os.environ` at import time.** Caused
   an order-dependent test flake. Fix: `monkeypatch.delenv()` in
   affected tests + inline comment so future-self doesn't lose hours.
4. **Off-corp default matters.** Initial `.env` had Walmart
   `HTTPS_PROXY` + corp `SSL_CERT_FILE` uncommented. Off the corp
   network: DNS fail + TLS fail. Now off-corp is the default; Walmart
   bits are clearly-marked opt-in. `_http.py` falls back to certifi
   gracefully if the configured CA bundle path doesn't exist.

### Test count delta: 554 → 625 (+71)

| Source | Tests added |
|---|---|
| `test_stooq_provider.py` | +25 |
| `test_alpha_vantage_provider.py` | +29 |
| `test_settings.py` (chain + key handling) | +10 |
| Misc adjustments (resilient error message, lazy fetcher) | +7 |

### Lessons from Provider Expansion

1. **Open/Closed pays off the SECOND time** — B.1's architecture cost
   ~14 tests. Adding the next two providers cost ~54 tests of pure
   feature code with zero refactoring upstream. Zen of Python: "namespaces
   are one honking great idea".
2. **Real-world APIs drift.** Stooq changed terms between B.1 design and
   provider implementation. Defensive lazy validation (vs. hard fail at
   construction) made the change a 3-line patch instead of a redesign.
3. **Fail open on non-security config.** Missing CA bundle path? Log a
   warning and use certifi (same trust roots browsers use). Hard fail
   only on actual secrets/auth missing.
4. **`.env` is documentation too.** Commenting out + adding instructional
   headers ("UNCOMMENT only when on Walmart corp network") gives both
   Walmart and off-corp users a working starting point in the SAME file.

---

## ✅ Step D — Prediction Agent (DONE)

**Goal**: Orchestrator that combines outputs from `kb`, `price_agent`,
`technical_agent`, `news_impact`, and `filings` into a structured
`Prediction` (direction, confidence, entry, target, stop, risk-reward) +
user-facing CLI.

### Sub-step progress

| # | Sub-step | Status |
|---|---|---|
| D.1 | Schema (`prediction/schema.py`) — frozen Pydantic v2 `Prediction` | ✅ |
| D.2 | Synthesizer agent (`agents/synthesizer/`) with `output_schema=Prediction` | ✅ |
| D.3 | Inputs builder (`prediction/inputs.py`) — prompt assembly | ✅ |
| D.4 | `predict()` orchestrator + Runner singletons (`prediction/predictor.py`, `runner.py`) | ✅ |
| D.5 | Hallucination guardrails Tiers 1–3 + retry loop (`prediction/guardrails.py`) | ✅ |
| D.6 | News degradation + integration smoke test | ✅ |
| D.7 | Batch (`prediction/batch.py`) — `predict_many()` with bounded concurrency | ✅ |
| D.8 | Persistence (`prediction/store.py`) — JSON-on-disk | ✅ |
| D.9 | CLI surface (`cli/main.py`) — typer + rich (predict / predict-many / history) | ✅ |

### What D.1 delivered (output schema)

| Artifact | Purpose |
|---|---|
| `prediction/schema.py` | Frozen Pydantic v2 model. `Prediction` + `PredictionDirection` + `PredictionHorizon` + `PriceLevel` + `AnalysisBasis` |

**Design decisions**:
1. **Frozen models** (`model_config = ConfigDict(frozen=True)`) — predictions
   are facts about a moment in time, not mutable state. Hashing is free.
2. **`risk_reward` is computed** in a `model_validator` from entry/target/stop
   — single source of truth, can't drift.
3. **`model_chain: tuple[str, ...]`** as audit trail — captures which LLMs
   actually participated in producing this prediction.
4. **`AnalysisBasis` sub-model** — captures the synthesizer's view of the
   world at decision time. Makes grading possible later: "the LLM thought
   close was X; was it?"

### What D.2–D.6 delivered (predict pipeline)

| Artifact | Purpose |
|---|---|
| `agents/synthesizer/agent.py` | LlmAgent with `output_schema=Prediction` (forces structured JSON; no parse logic in our code) |
| `prediction/inputs.py` | Pure prompt assembly: technical snapshot + news + price → string prompt |
| `prediction/predictor.py` | `predict(ticker, horizon)` orchestrator. Calls technical_agent + news_impact + price_agent + synthesizer in sequence |
| `prediction/runner.py` | Singleton ADK `Runner` instances per agent (one Runner per Agent is the ADK contract) |
| `prediction/guardrails.py` | 3 tiers: ticker match, level sanity (target on right side of entry), direction-vs-levels coherence. Retry-with-feedback loop on failure |

**Critical lessons from D.2–D.6**:
1. **`output_schema=Prediction` is the killer ADK feature for synthesis.**
   Forces the LLM to emit valid Pydantic JSON; zero parsing logic our side.
2. **News degradation is a real production concern.** GDELT goes down,
   articles 404, body extraction times out. Graceful degradation
   (`news_impact` returns empty list with explanatory note) > hard fail.
3. **Hallucination guardrails are NOT optional.** Without Tiers 1-3 + retry,
   the synthesizer would happily invent target prices on the wrong side
   of entry, or claim BULLISH while putting target below stop.
   Retry-with-feedback fixed ~80% of these.
4. **Run cheapest validation first.** Tier 1 (regex/range) > Tier 2 (cross-
   field) > Tier 3 (LLM self-critique). Never invoke Tier 3 if 1-2 fail.

### What D.7 delivered (batch)

| Artifact | Purpose |
|---|---|
| `prediction/batch.py` | `predict_many(tickers, horizon)` with `asyncio.gather` + bounded semaphore (5) |
| `BatchError` | Accumulates per-ticker failures without killing the batch |

**Lessons from D.7**:
1. **Per-ticker failure must NOT kill the batch.** A single 404 from GDELT
   should produce 1 error in the result, not 49 lost predictions.
2. **Bounded concurrency (semaphore=5) > unbounded.** LLM rate limits bite
   at ~10 concurrent; 5 leaves headroom.

### What D.8 delivered (persistence)

| Artifact | Purpose |
|---|---|
| `prediction/store.py` | `PredictionStore` writes per-prediction JSON files; reads via `list_for_ticker()` / `list_in_date_range()` |
| Storage layout | `predictions_dir/<TICKER>/<as_of_iso>.json` — trivially inspectable, no DB |

**Why JSON-on-disk over SQLite**:
1. **Inspectable.** `cat predictions/RELIANCE.NS/*.json` works.
2. **No migration cost.** Schema change → add a field with a default;
   old files still load.
3. **Backup is `cp -r`.** No `pg_dump`-equivalent needed.
4. **YAGNI.** ~36k files/year at 100 predictions/day. SQLite wins above
   ~1M rows; we're 30x below that.

### What D.9 delivered (CLI)

| Artifact | Purpose |
|---|---|
| `cli/main.py` | typer app with `predict` / `predict-many` / `history` commands |
| Rich rendering | Color-coded direction, formatted tables, helpful empty states |

**Lessons from D.9**:
1. **Typer + Rich = the right CLI stack for Python in 2026.**
2. **Render functions return Tables, not strings.** Lets tests assert on
   cell contents directly via Rich's API; no string-grep brittleness.

### Step D test count delta: 625 → ~817 (+~192)

---

## ✅ Step 3.5 — Grading + Calibration (DONE)

**Goal**: The prediction loop is incomplete without measuring the LLM's
actual skill. Step 3.5 added the math (per-prediction grading), the
aggregation (calibration metrics), and the user-facing surface (CLI).

### Sub-step progress

| # | Sub-step | Status |
|---|---|---|
| 3.5.1 | Grading core (`prediction/grading.py`) — `grade_one()` + `GradedPrediction` + 6-outcome enum | ✅ |
| 3.5.2 | Aggregation (`prediction/calibration.py`) + orchestration (`grade_many()`) — hit-rate variants, Brier score | ✅ |
| 3.5.3 | CLI surface (`cli/main.py`) — `grade` + `calibration` commands with `--by` breakdown axes | ✅ |

### What 3.5.1 delivered (per-prediction grading)

| Artifact | Purpose |
|---|---|
| `grade_one(pred, bars)` | Pure function on a Prediction + post-prediction OHLCV DataFrame |
| `GradeOutcome` (enum) | TARGET_HIT / STOP_HIT / STOP_HIT_AMBIGUOUS / EXPIRED / NOT_APPLICABLE / INCONCLUSIVE |
| `GradedPrediction` (frozen Pydantic) | Wraps prediction + outcome + realized_return + direction_correct + days_to_resolution |

**The same-bar ambiguity problem (THE central insight of 3.5.1)**:
If a bar's high ≥ target AND low ≤ stop, we genuinely don't know which
was hit first without intraday data. We surface this as `STOP_HIT_AMBIGUOUS`
rather than silently picking one. Honest > convenient.

**Lessons from 3.5.1**:
1. **Pure functions are the right shape for math-heavy logic.** Inject the
   OHLCV DataFrame; don't fetch inside grade_one. Tests run with synthetic
   bars; production injects real fetches.
2. **6 outcomes, not 2.** Pass/fail is lossy. The middle four
   (ambiguous/expired/N/A/inconclusive) carry signal that pure pass/fail
   would discard.

### What 3.5.2 delivered (orchestration + aggregation)

| Artifact | Purpose |
|---|---|
| `grade_many()` | Loops grade_one over a list, fetches OHLCV per prediction (lazy yfinance import) |
| `CalibrationReport` | Frozen Pydantic. Hit-rate variants, Brier score, direction accuracy, mean+median return |
| `compute_calibration(graded)` | Pure aggregation: list → single CalibrationReport |
| `compute_breakdown(graded, key_fn)` | Generic group-by: returns `dict[K, CalibrationReport]` |

**Three hit-rate variants (the 3.5.2 design debate)**:
Same-bar ambiguity bubbles up. We REPORT all three:
  - `hit_rate_strict`     = wins / (wins + losses + ambig + expired + na)
  - `hit_rate_resolved`   = wins / (wins + losses + ambig)  ← industry standard
  - `hit_rate_optimistic` = wins / (wins + clean losses)

Picking one would be cherry-picking. Tests assert
`strict ≤ resolved ≤ optimistic` as a STRUCTURAL INVARIANT — mathematically
true by definition; the test catches any future refactor that breaks it.

**Brier score over log-loss (the other 3.5.2 design debate)**:
  - Brier = mean((confidence − actual)²). Bounded [0, 1].
  - Log-loss is unbounded; has log(0) edge case at confidence=1.0.
  - Brier's quadratic penalty matches user intuition: 90%-wrong is 3x worse
    than 60%-wrong (0.81 vs 0.36).
  - 0.25 baseline (always p=0.5, half right) is a useful comparison point
    any user can hold in their head.

### What 3.5.3 delivered (CLI surface)

| Artifact | Purpose |
|---|---|
| `cli/main.py::grade` | Loads predictions from store, runs grade_many, renders per-prediction outcome table |
| `cli/main.py::calibration` | Same loader, then compute_calibration (or compute_breakdown if `--by`) |
| `_BREAKDOWN_KEYS` (dispatch dict) | `{horizon, ticker, direction, month}` — adding a new axis = 1 dict entry |
| `_load_predictions()` | Shared loader (DRY between grade + calibration) |

**Lessons from 3.5.3**:
1. **Dispatch dict for `--by` axes is open/closed in 5 lines.** Help text
   auto-syncs from `sorted(_BREAKDOWN_KEYS)`.
2. **Mock at the boundary.** CLI tests mock `grade_many`; the contract of
   grade_many is exhaustively tested elsewhere with synthetic OHLCV.

---

## ✅ Step 3.4.6 — Multi-horizon hardening (DONE)

**Goal**: Close a latent bug in Step 3.4.2 — the predictor accepted a
`horizon` argument but used a single hard-coded set of constants (ATR
bands, entry zones, confidence cap, NEUTRAL grading tolerance) for ALL
horizons. *Fake multi-horizon.* This phase makes it real.

**Trigger**: While building Step 3.5.3 (CLI grade + calibration), the
`--by horizon` breakdown surfaced that DAILY and MONTHLY were being
graded against the same 2% NEUTRAL tolerance — mathematically wrong
(longer horizons have larger expected moves; tolerance must scale).
Pulling that thread revealed the same horizon-blindness in guardrails
AND the synthesizer prompt.

### Sub-step progress

| # | Commit | What | Status |
|---|---|---|---|
| 3.4.6.1 | `0d5cdec` | NSE trading-calendar helper for honest horizon math (handles weekends + Indian market holidays via `holidays` lib) | ✅ |
| 3.4.6.2 | `505cb4d` | Rename horizon enum from short/medium/long to **DAILY / WEEKLY / BIWEEKLY / MONTHLY** (concrete > abstract; matches how traders actually talk) | ✅ |
| 3.4.6.3 | `ff037fc` | `predict()` fans out across all 4 horizons in parallel (`asyncio.gather`) | ✅ |
| 3.4.6.4 | `3253d89` | Per-horizon NEUTRAL grading tolerance (sqrt-t scaled: longer horizon → wider band) | ✅ |
| 3.4.6.5 | `9efa283` | Research-grounded `docs/research/constants_dossier.md` (36 KB) replacing vibes-based numbers + LMW (Lower-Moving-Window) chart-pattern alignment fix | ✅ |
| 3.4.6.6 | `34bb240` | **Commit A**: `prediction/horizon_constants.py` as single source of truth (helpers `stop_atr_range`, `target_atr_range`, `entry_zone_pct`, `confidence_cap`, `neutral_tolerance_pct`); 100% covered | ✅ |
| 3.4.6.7 | `c66388e` | **Commit B**: guardrails wired to per-horizon ATR bands + entry zones; new **Tier 4 calibration cap** rejects predictions whose confidence exceeds the per-horizon ceiling | ✅ |
| 3.4.6.8 | `eb3c84f` | **Commit C**: synthesizer prompt embeds the per-horizon rules table rendered at module import from `horizon_constants` (LLM cannot drift from guardrails) | ✅ |

### What landed (the SoT chain)

```
              prediction/horizon_constants.py
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
      prediction/             agents/synthesizer/
      guardrails.py           prompt.py
      (enforces)              (teaches the LLM)
            │                     │
            └──────────┬──────────┘
                       ▼
              same numbers, always
              (regression tests prove it)
```

| Artifact | Purpose |
|---|---|
| `prediction/horizon_constants.py` | Single source of truth. 5 helpers, one per tunable axis. Frozen `MappingProxyType` tables under the hood so consumers can't mutate. |
| `prediction/trading_calendar.py` | NSE-aware `add_trading_days()`, `horizon_to_trading_days()`, `expiry_for_horizon()`. Backed by `holidays` library; handles weekends + Indian market holidays. |
| `prediction/predictor.py::predict()` | Now fans out across ALL 4 horizons in parallel via `asyncio.gather`; per-horizon errors don't kill the batch. |
| `prediction/guardrails.py` | Tier 1-3 unchanged. **NEW Tier 4** (`_validate_calibration`) checks `confidence ≤ confidence_cap(horizon)`. ATR band checks read from `stop_atr_range(horizon)` / `target_atr_range(horizon)` / `entry_zone_pct(horizon)`. |
| `prediction/grading.py` | `_neutral_tolerance_pct()` now reads from `horizon_constants.neutral_tolerance_pct(horizon)` instead of a hard-coded 2.0. |
| `agents/synthesizer/prompt.py` | New `_render_per_horizon_table()` helper builds a markdown table from `horizon_constants` at module import; spliced into `SYSTEM_INSTRUCTION` via f-string. Removed all dead hand-wavy phrasing ("tighter for daily, wider for monthly", "±0.5% for daily/weekly", "close_price ∓ ~1*ATR is a sane default"). |
| `docs/research/constants_dossier.md` | 36 KB research grounding for every per-horizon number (sqrt-t scaling derivation, ATR-band rationale, citations). Reviewable, citable, replaces vibes. |

### Critical lessons from Step 3.4.6

1. **"Accepts a parameter" ≠ "is parameterized."** Step 3.4.2 took a
   `horizon` argument and threaded it through the call stack — but every
   constant downstream was horizon-blind. *Fake multi-horizon* is worse
   than single-horizon because it looks honest while silently lying. The
   `--by horizon` breakdown was the canary that exposed it.

2. **Single source of truth is a refactor target, not a starting point.**
   We had constants scattered across 4 files (guardrails, grading, prompt
   text, magic numbers in inputs.py). The dossier work first cataloged
   them; only then did `horizon_constants.py` get written. Catalog →
   centralize → wire.

3. **Tests prove the SoT chain.** The killer test:
   `test_table_rows_match_horizon_constants` reads each helper and
   asserts the rendered prompt table contains those exact values for
   every horizon. Tune `horizon_constants.py` → prompt updates AND
   guardrails update AND tests still pass. Drift becomes mechanically
   impossible.

4. **Regression nets for deletions, not just additions.** When we
   removed the dead phrasing from the prompt, we added
   `test_dead_handwavy_phrasing_removed` parametrized over the exact
   strings we deleted. If a future contributor re-introduces "tighter
   for daily, wider for monthly", the test fires. Tests prevent both
   bug introduction AND regression-by-rewording.

5. **Sqrt-t scaling for time-uncertainty bands.** Standard finance: a
   2% NEUTRAL band over 1 day implies ~`2% × √(N/1)` over N days. The
   dossier walks through the math; `neutral_tolerance_pct()` implements
   it; grading is now horizon-honest.

6. **Research dossier > argument from authority.** Once we wrote
   `constants_dossier.md` with citations, every later number became
   defensible ("per dossier §3.2, daily entry zone is ±0.5% based on
   median Nifty ATR/price"). Future tuning becomes "update the dossier
   first, then the code," which is the right order.

7. **The trading-calendar helper is load-bearing.** Without it, "7 days
   from today" silently included weekends + Indian market holidays —
   horizon math was off by ~30%. Real markets only trade on real
   trading days. `holidays` lib + `pandas.tseries.offsets.CustomBusinessDay`
   makes this a few lines but the bug is invisible until you grade it.

### Step 3.4.6 test count delta: 854 → 1021 (+167)

---

## ⏸️ Step 3.5.5+ — Backtest replay/runner/evaluator (NOT STARTED)

**Goal**: Today, calibration only works on real-elapsed-time predictions.
Backtest would let us run the whole pipeline against historical data and
answer "would this system have made money?"

**Components needed**:
- `backtest/replay.py` — as-of-date data shim: "give me prices/news/filings
  AS THEY WOULD HAVE LOOKED on date X." Critical for honest backtest;
  any leak of future info inflates results.
- `backtest/runner.py` — historical loop over dates, calling predict()
  with the replay shim active.
- `backtest/evaluator.py` — composes calibration metrics across backtest
  runs (e.g., per-month, per-regime). Reuses `compute_breakdown()`.

**Open design questions parked for Step 3.5.5 design**:
- How do we honestly replay GDELT? News articles published AFTER our
  as-of-date must NOT be visible.
- How do we handle survivorship bias in the Nifty50 list?
- Do we replay at end-of-day cadence or hourly?

---

## 📁 Current file structure (relevant portions)

```
src/price_predictor/
├── agents/
│   ├── hello_agent/                    # Learning spike (DONE)
│   ├── price_agent/                    # Refactored to use KB (Step A)
│   ├── news_impact/                    # Prompt simplified by KB (Step A)
│   ├── technical_agent/                # NEW (Step C)
│   │   ├── __init__.py
│   │   └── tools/
│   │       ├── _types.py / _trend_signal.py
│   │       └── get_trend.py / get_momentum.py / get_volatility.py / get_levels.py
│   └── synthesizer/                    # NEW (Step D.2)
│       ├── __init__.py
│       └── agent.py                    # LlmAgent with output_schema=Prediction
├── analysis/                           # NEW (Step B.3 + B.4)
│   ├── __init__.py                     # PRESETS + validate_preset()
│   ├── trend.py / momentum.py / volatility.py / levels.py
│   └── candlestick_patterns.py / chart_patterns.py
├── prediction/                         # NEW (Step D + 3.5)
│   ├── __init__.py
│   ├── schema.py                       # D.1: Prediction model
│   ├── inputs.py                       # D.3: prompt assembly
│   ├── predictor.py                    # D.4: predict() orchestrator
│   ├── runner.py                       # D.4: ADK Runner singletons
│   ├── guardrails.py                   # D.5: hallucination guardrails Tiers 1-3
│   ├── batch.py                        # D.7: predict_many()
│   ├── store.py                        # D.8: PredictionStore (JSON-on-disk)
│   ├── grading.py                      # 3.5.1+3.5.2: grade_one + grade_many
│   └── calibration.py                  # 3.5.2: CalibrationReport + compute_*
├── cli/                                # NEW (Step D.9 + 3.5.3)
│   ├── __init__.py
│   └── main.py                         # typer + rich: predict / predict-many / history / grade / calibration
├── data/
│   ├── prices.py                       # Thin shim (Step B.1)
│   ├── cache.py                        # NEW (Step B.2)
│   ├── _shared_cache.py                # NEW (Step C.1) — singleton
│   ├── providers/                      # NEW (Step B.1)
│   │   ├── base.py / yfinance_provider.py / resilient.py
│   │   └── stooq_provider.py / alpha_vantage_provider.py / _http.py  # Provider Expansion
│   ├── estimates.py / filings.py / news.py / schema.py
├── kb/                                 # NEW (Step A)
│   └── stocks.py
├── llm/
│   ├── factory.py
│   └── resilient.py
└── config/
    └── settings.py

data/kb/
├── stocks.json                         # NEW, COMMITTED (Step A)
└── indices.json                        # NEW, COMMITTED (Step A)

scripts/
└── bootstrap_indices.py                # NEW (Step A)

tests/                                  # 1021 unit tests + 7 integration
├── test_kb_stocks.py                   # 36 tests (Step A)
├── test_prices.py / test_resilient_price_fetcher.py / test_price_cache.py
├── analysis/                           # 61 tests (Step B.3 + B.4)
├── tools/                              # Step C tool tests
├── prediction/                         # Step D + 3.5 tests
│   ├── test_schema.py
│   ├── test_inputs.py / test_predictor.py / test_runner.py
│   ├── test_guardrails.py
│   ├── test_batch.py / test_store.py
│   ├── test_grading.py / test_grade_many.py
│   └── test_calibration.py
└── cli/
    └── test_main.py                    # CLI integration tests
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
| Step C.1 complete | 425 | +31* | Tool + classifier tests |
| Step C.2 complete | 474 | +49 | Momentum tool + candlestick context-gating |
| Step C.3 complete | 509 | +35 | Volatility tool + position-sizing helpers |
| Step C.4 complete | 538 | +29 | Levels tool + chart pattern integration |
| Step C.5 complete | 554 | +16 | `technical_agent` wiring |
| Step C.6 complete | 554 | +0 | Manual smoke + LLM-chain bug fix (env-only) |
| Provider Expansion complete | 625 | +71 | Stooq + AlphaVantage providers, paid toggle |
| Step D.1 (schema) | ~650 | +25 | Frozen Pydantic Prediction + round-trip |
| Step D.2 (synthesizer) | ~685 | +35 | LlmAgent with `output_schema=Prediction` |
| Step D.3 (inputs) | ~705 | +20 | Prompt assembly |
| Step D.4 (predictor + runner) | ~735 | +30 | predict() orchestrator + Runner singletons |
| Step D.5 (guardrails) | ~780 | +45 | Tier 1-3 + retry-with-feedback |
| Step D.6 (news degradation) | ~795 | +15 | + integration smoke test |
| Step D.7 (batch) | ~820 | +25 | predict_many + BatchError |
| Step D.8 (store) | ~850 | +30 | PredictionStore JSON-on-disk |
| Step D.9 (CLI) | ~864 | +14 | typer + rich: predict / predict-many / history |
| Step 3.5.1 (grading) | ~898 | +34 | grade_one + GradedPrediction + 6-outcome enum |
| Step 3.5.2 (calibration) | ~924 | +26 | grade_many + CalibrationReport + Brier |
| Step 3.5.3 (CLI grade+calibration) | 854 | net | Net delta after test cleanup; gross +37 |
| Step 3.4.6.1 (trading-calendar) | ~880 | +26 | NSE-aware day math (`0d5cdec`) |
| Step 3.4.6.2 (horizon enum rename) | ~890 | +10 | DAILY/WEEKLY/BIWEEKLY/MONTHLY (`505cb4d`) |
| Step 3.4.6.3 (predict() fan-out) | ~925 | +35 | parallel ×4 horizons (`ff037fc`) |
| Step 3.4.6.4 (sqrt-t NEUTRAL grading) | ~945 | +20 | per-horizon tolerance (`3253d89`) |
| Step 3.4.6.5 (constants dossier + LMW fix) | ~960 | +15 | dossier + chart-pattern alignment (`9efa283`) |
| Step 3.4.6.6 (horizon_constants — Commit A) | ~975 | +15 | single SoT module (`34bb240`) |
| Step 3.4.6.7 (guardrails per-horizon — Commit B) | ~1006 | +31 | Tier 4 calibration cap + per-horizon bands (`c66388e`) |
| Step 3.4.6.8 (synthesizer prompt — Commit C) | **1021** | +15 | per-horizon rules table embedded in prompt (`eb3c84f`) |

\*Includes a +4 incidental gap between B.4 and C.1 (fixture/import additions).

Note: Step D + 3.5 totals are approximate per-substep snapshots reconstructed
from commit log; the **1021 figure** is the actual current `pytest --collect-only`
count (with 7 integration tests deselected, 1 skipped). Step 3.4.6 per-substep
rows are reconstructed from commit-by-commit deltas — the 854 → 1021 net is
the rock-solid number; intermediate snapshots are best-effort.

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

### From Provider Expansion
1. Open/Closed pays off the SECOND time you swap a backend — B.1's
   architecture cost ~14 tests; the next two providers cost ~54 tests
   of pure feature code with zero upstream refactor.
2. Real-world APIs drift (Stooq apikey, httpx 0.28). Defensive lazy
   validation makes drift a 3-line patch, not a redesign.
3. Fail open on non-security config (missing CA bundle → fall back to
   certifi + warn). Fail closed on auth/secrets.
4. `.env` is documentation too. Commented-out + instructional headers
   give both Walmart and off-corp users a working start in the same file.
5. LiteLLM auto-loads `.env` into `os.environ` at import time — worth
   knowing because it bites tests that expect default fallback behavior.

### From Step D (Prediction Agent)
1. **`output_schema=Prediction` is the killer ADK feature for synthesis.**
   Forces the LLM to emit valid Pydantic JSON; zero parse logic our side.
2. **Hallucination guardrails are NOT optional.** Without Tiers 1-3 +
   retry-with-feedback, the synthesizer would invent target prices on the
   wrong side of entry, or claim BULLISH while putting target below stop.
3. **Run cheapest validation first.** Tier 1 (regex/range) > Tier 2 (cross-
   field coherence) > Tier 3 (LLM self-critique). Never invoke Tier 3 if
   1-2 fail.
4. **News degradation is a real production concern.** Graceful degradation
   (return empty list with explanatory note) > hard fail.
5. **Per-ticker failure must NOT kill the batch.** A single 404 = 1 error,
   not 49 lost predictions. Use a `BatchError` accumulator.
6. **JSON-on-disk beats SQLite at our scale.** Inspectable, no migration
   cost, backup is `cp -r`. Will revisit above ~1M predictions.
7. **Render functions return Tables (not strings).** Lets tests assert on
   cell contents directly via Rich's API.

### From Step 3.5 (Grading + Calibration)
1. **Same-bar T+S ambiguity is a first-class outcome, not a bug.** If a
   bar's high ≥ target AND low ≤ stop, we genuinely don't know which was
   hit first. Surfacing it as `STOP_HIT_AMBIGUOUS` is honest > silently
   picking one.
2. **Six outcomes, not two.** Pass/fail is lossy. The middle four
   (ambiguous/expired/N/A/inconclusive) carry signal pure pass/fail throws away.
3. **Three hit-rate variants reported, not one.** strict / resolved /
   optimistic. Picking one would be cherry-picking; reporting all three is
   honest. Tests assert `strict ≤ resolved ≤ optimistic` as a structural
   invariant.
4. **Brier score over log-loss.** Bounded [0,1], no log(0) edge case at
   confidence=1.0, quadratic penalty matches user intuition (90%-wrong is
   3x worse than 60%-wrong).
5. **Pure functions are right for math-heavy logic.** Inject the OHLCV
   DataFrame into `grade_one`; don't fetch inside it. Tests use synthetic
   bars; production injects real fetches.
6. **Lazy yfinance import** in `grade_many` keeps `from prediction import
   grade_one` fast (yfinance load is ~1s).
7. **Dispatch dict for `--by` axes is open/closed in 5 lines.** Adding
   `--by week` would be one more entry; help text auto-syncs from
   `sorted(_BREAKDOWN_KEYS)`.
8. **Mock at the boundary.** CLI tests mock `grade_many`; the contract of
   grade_many is exhaustively tested elsewhere with synthetic OHLCV.
   No need to repeat.

### From Step 3.4.6 (Multi-horizon hardening)
1. **"Accepts a parameter" ≠ "is parameterized."** A function can take
   `horizon` and silently ignore it downstream. *Fake multi-horizon* is
   worse than single-horizon because it looks honest while lying. Audit
   the call stack, not the signature.
2. **`--by horizon` was the canary.** Calibration breakdowns surface
   horizon-blindness instantly: if every horizon has identical numbers,
   you're not actually multi-horizon. Build the breakdown surfaces FIRST
   so the bug shows up second.
3. **Single source of truth is a refactor target, not a starting point.**
   Catalog scattered constants → centralize them → wire consumers to the
   single module. Skipping the catalog step means you miss usages.
4. **Tests prove the SoT chain mechanically.** The killer test reads each
   helper from `horizon_constants` and asserts the rendered prompt
   contains those exact values per horizon. Tune the module → prompt AND
   guardrails update; tests still pass. Drift becomes impossible.
5. **Regression nets for deletions, not just additions.** When you delete
   dead phrasing, parametrize a test over the exact strings deleted so
   a future contributor can't reintroduce them.
6. **Sqrt-t scaling for time-uncertainty bands** (standard finance). A 2%
   NEUTRAL tolerance at 1 day implies ~`2% × √(N)` over N days. Hard-coded
   constants across horizons miss this completely.
7. **Research dossier > argument from authority.** Citations make every
   tunable defensible. Future tuning order: dossier first, then code.
8. **NSE trading-calendar is load-bearing.** Calendar days ≠ trading
   days; "+7 days" naively includes weekends + Indian market holidays,
   so horizon math is off by ~30%. `holidays` lib + `CustomBusinessDay`
   fixes it in a few lines, but the bug is invisible without grading.
9. **`MappingProxyType` for frozen lookup tables.** Lets consumers read
   but not mutate; better than convention-only "please don't mutate."
10. **Render-from-source-of-truth at module import.** The synthesizer's
    per-horizon rules table is `_render_per_horizon_table()` called once
    at import; embedded into `SYSTEM_INSTRUCTION` via f-string. No
    runtime cost, zero possibility of stale text.

### Meta-lesson (recurring)
When we agree on a build plan together, sticking to it is the contract.
Don't unilaterally YAGNI parts of an agreed plan. Don't sneak in scope
beyond it either.
