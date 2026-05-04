# Next Steps — Stock Predictor v1

> **Scope**: This doc captures the **detailed plan for upcoming work** in the
> Stock Predictor v1 build. For "what's done already" see
> `implementation_flow.md`. For the higher-level project roadmap see
> `implementation_plan.md`.
>
> **Last updated**: 2026-04-28 — after Step B.1 (provider refactor).

---

## 🚦 Immediate next: Step B.2 — Cache layer

### Goal

Add a **range-aware in-memory cache** between the resilient price fetcher
and the technical-analysis tools, so multiple tool calls within a single
user question hit yfinance **at most once per ticker**.

### Why this is next

Step B.5–B.8 will fan out 4 tool calls (one per cluster) for a single
"analyze HDFCBANK" question. Without a cache, that's 4 yfinance hits for
the same data. yfinance rate-limits eventually. The cache turns that into
1 hit.

### Locked design (from prior discussion)

- **Range-aware**: cache stores the widest date range fetched per ticker.
  If a later call asks for a sub-range, slice from cache (no fetch). If it
  asks for a wider range, fetch the missing chunk and merge.
- **Key**: `(ticker, interval)` — a single entry per ticker per bar size.
- **Storage**: in-memory only. No disk persistence (avoids cache-invalidation
  hell for v1).
- **Concurrency**: one `asyncio.Lock` per `(ticker, interval)` so two parallel
  tool calls for the same ticker don't both hit the network.
- **Lifetime**: lives one Python process. Restart = fresh cache.
- **Eviction**: none for v1. 50 stocks × ~50KB each = trivial memory.

### Acceptance criteria

- [ ] `PriceCache` class with at minimum: `async get(ticker, start, end, interval)` method
- [ ] First call for a ticker fetches from the resilient fetcher; stores result
- [ ] Second call for a sub-range slices from cache without re-fetching
- [ ] Second call for a wider range fetches missing portion + merges
- [ ] Two concurrent calls for same ticker → only one network fetch (lock test)
- [ ] Two concurrent calls for **different** tickers → both fetch in parallel
      (no cross-ticker blocking)
- [ ] Slicing returns a **copy** of the cached DataFrame (callers shouldn't
      be able to mutate the cache by accident)
- [ ] Public API of `data/prices.py` (`fetch_ohlcv`) unchanged — caching is
      orthogonal; tools use the cache directly when they need to

### Open design questions for B.2

1. **Where does the cache live?**
   - Option (a): inside `data/prices.py` — `fetch_ohlcv` becomes cached
     transparently. All callers benefit.
   - Option (b): `data/cache.py` — separate module, tools opt in by calling
     `cache.get(...)` instead of `fetch_ohlcv(...)`.
   - **Lean: (b)**. Caching is a tool-layer concern (multi-call fan-out is
     unique to thematic clusters). Keeping `data/prices.py` cache-free means
     other callers (price_agent, news_impact) get fresh data each call,
     which is what they want.

2. **Should the cache "round up" the fetch window?**
   - When asked for 60 days, fetch 365 days proactively (so subsequent calls
     for narrower windows are slices)?
   - **Lean: yes**, default to fetching 1y on first miss. The marginal cost
     is small; the savings on subsequent calls are large.

3. **Does the cache know about "today"?**
   - The bar for "today" is wrong tomorrow. Two options:
     - Naive: cache lives one process; "today" doesn't change within a session.
     - Smart: invalidate ticker if last fetch was on a different trading day.
   - **Lean: naive for v1**. Sessions are short; the day rarely flips mid-session.

### Test plan for B.2

A `test_price_cache.py` with these test classes:

- `TestSingleFetchCaches` — first call fetches, second returns cached
- `TestRangeSlicing` — narrower request returns cache slice, no re-fetch
- `TestRangeExpansion` — wider request fetches missing chunk, merges
- `TestConcurrencyLocking` — N parallel calls for same ticker → 1 fetch
- `TestConcurrencyParallelism` — N calls for different tickers → N fetches in parallel
- `TestImmutability` — mutating returned DataFrame doesn't affect cache
- `TestErrorPropagation` — fetch errors propagate, don't poison the cache

### Estimated commits

1. `PriceCache` class + basic single-fetch caching + tests
2. Range-aware slicing/merging + tests
3. Per-ticker async lock + concurrency tests
4. Wire cache into `data/prices.py` if option (a) chosen, otherwise leave
   for tools to use directly

---

## 📍 Step B.3 — Indicator primitives

### Goal

Pure functions for the math behind every cluster, completely decoupled from
ADK tooling. These are the building blocks B.5–B.8 will compose.

### Scope

Four files of pure functions, each consuming a `pd.DataFrame` of OHLCV bars
and returning floats / small dicts:

```
src/price_predictor/analysis/
├── trend.py        # SMAs, EMA-20, ADX
├── momentum.py     # RSI, MACD, Stochastic, OBV
├── volatility.py   # ATR, Bollinger Bands, BB %B, BB squeeze
└── levels.py       # swing high/low, 52w high/low, pivot points
```

### Library choice

`pandas-ta` for all of these. Pure Python, easy install, deterministic output,
already battle-tested. (We considered `ta-lib` but skipped due to install pain.)

### Output contract

Each function returns either:
- A `float` (single value: e.g. latest RSI)
- A `dict[str, float]` (multiple related values: e.g. MACD line + signal + histogram)
- A small `dataclass` if the shape gets complex

No "signal" interpretation at this layer — that happens in the tool layer
(B.5–B.8). These primitives are purely numerical.

### Sensitivity presets

Each `analysis/*.py` module exposes a `PRESETS` dict mapping preset name to
parameter bundle:

```python
# Conceptual — exact shape TBD
PRESETS = {
    "standard":  {"rsi_period": 14, "macd": (12, 26, 9), "stoch": (14, 3, 3)},
    "sensitive": {"rsi_period": 9,  "macd": (8, 17, 9),  "stoch": (9, 3, 3)},
    "smooth":    {"rsi_period": 21, "macd": (19, 39, 9), "stoch": (21, 5, 5)},
}
```

The tool layer picks a preset by name; primitives consume the bundle.

### Test plan

Each module gets `tests/analysis/test_<name>.py`:
- Synthetic OHLCV (linear, sinusoidal, etc.) → known indicator values
- Edge cases: too few bars, all-zeros, NaN handling
- Preset coverage: each preset produces sensibly different output

---

## 📍 Step B.4 — Chart pattern detectors

### Goal

Hand-rolled detectors for the 3 most reliable chart patterns. Lives in:

```
src/price_predictor/analysis/chart_patterns.py
```

### Scope (locked)

- **Double top / double bottom** — peak detection + symmetry check
- **Head & shoulders (regular + inverse)** — 3-peak relationship + neckline
- **Triangles (ascending / descending / symmetric)** — trendline fitting

Skipped (too noisy for v1): cup & handle, flags, pennants, wedges.

### Output shape

Each detector returns `list[Pattern]` where `Pattern` is a small dataclass:

```python
# Conceptual
@dataclass
class ChartPattern:
    name: str                    # "double_top", "head_and_shoulders", etc.
    confidence: float            # 0-1, based on geometric quality
    key_levels: dict[str, float] # neckline, target, stop, etc.
    bars_involved: tuple[date, date]  # range the pattern spans
```

### Filtering rule

Only patterns with `confidence >= 0.7` get surfaced to the tool layer.
Below that, they're noise that hurts more than it helps the LLM.

### Implementation notes

- Use `scipy.signal.find_peaks` for swing point detection
- Geometric validation: e.g. for H&S, check that head is significantly
  taller than shoulders, shoulders are roughly equal height, neckline
  is roughly horizontal
- Synthetic test fixtures: hand-craft OHLCV that should trigger each pattern
- Real-world fixtures: 1-2 known historical examples per pattern (committed
  as small CSVs under `tests/fixtures/`)

### Test plan

`tests/analysis/test_chart_patterns.py`:
- Each pattern: synthetic positive case (clean shape → detected)
- Each pattern: synthetic negative case (similar but invalid → not detected)
- Confidence filtering: low-confidence pattern returned by detector but
  filtered out at the tool layer
- Edge cases: insufficient bars, flat-line input

---

## 📍 Steps B.5–B.8 — The four cluster tools

Each cluster tool follows the same shape. Building one validates the
pattern; the others are mechanical.

### Build order

1. **B.5: `get_trend`** — first cluster end-to-end (validates the whole flow)
2. **B.6: `get_momentum`** (+ candlestick patterns folded in)
3. **B.7: `get_volatility`**
4. **B.8: `get_levels`** (+ chart patterns folded in)

### Tool shape (uniform)

```python
# Conceptual
def get_trend(ticker: str, sensitivity: str = "standard") -> dict:
    """ADK tool: trend analysis for a ticker.
    
    Returns:
      {
        "ticker": "HDFCBANK",
        "as_of": "2026-04-28",
        "indicators": {
          "sma_20": 1450.3,
          "sma_50": 1420.1,
          "sma_200": 1380.7,
          "ema_20": 1455.8,
          "adx_14": 28.4,
        },
        "derived": {
          "above_sma_200": True,
          "pct_above_sma_50": 4.2,
        },
        "signal": "uptrend",  # enum
        "preset": "standard",
        "warnings": [],  # e.g. ["insufficient_history"] for new IPOs
      }
    """
```

### Per-tool checklist

- [ ] Tool function with string-typed args (ADK requirement)
- [ ] Calls into `analysis/<cluster>.py` for primitives
- [ ] Translates raw indicator values → `signal` enum via small ruleset
- [ ] Graceful degradation for short-history tickers (no crash, return
      `None` for indicators that can't compute, populate `warnings`)
- [ ] Comprehensive unit tests: happy path, each signal value, each preset,
      edge cases
- [ ] Mock the cache + provider layer; never hit real yfinance in unit tests
- [ ] At least 1 integration test marked `@pytest.mark.integration` that
      hits real data

### Where patterns fold in

- `get_momentum` calls `analysis/candlestick_patterns.py` (using `pandas-ta`)
  and adds curated, **context-gated** results to the output (only flag a
  pattern if it occurs near a level — within 1×ATR of a swing high/low)
- `get_levels` calls `analysis/chart_patterns.py` (B.4) and adds high-
  confidence patterns to the output

### Estimated commits per tool

1. Tool function + signal classifier + happy-path tests
2. Edge cases + warnings + preset coverage
3. (For B.6 and B.8 only) pattern integration + pattern tests

---

## 📍 Step B.9 — `technical_agent` wiring

### Goal

Combine the 4 cluster tools into a single `LlmAgent` that an orchestrator
(or `adk web` user) can talk to.

### Shape (mirrors `news_impact` / `price_agent`)

```
src/price_predictor/agents/technical_agent/
├── __init__.py        # exports root_agent
├── agent.py           # LlmAgent factory + prompt + tools list + output schema
└── prompt.md          # the agent's system prompt (separate file for readability)
```

### Open design questions for B.9

1. **Output schema** — does `technical_agent` return raw cluster outputs
   verbatim, or a synthesized "technical view" object?
   - **Lean: synthesized**. Mirror `news_impact`'s `NewsImpactReport` —
     a structured Pydantic model with overall direction, confidence, and
     supporting indicator highlights.

2. **Should it always call all 4 tools, or let the LLM pick?**
   - **Lean: let the LLM pick.** That's the whole point of the thematic
     cluster design. The prompt nudges toward "for a comprehensive view,
     use all 4; for a focused question, use the relevant ones."

3. **Sensitivity preset selection** — does the agent always use "standard",
   or pick based on the question?
   - **Lean: prompt instructs it to default to "standard" unless the user
     asks for "swing" / "intraday" hints (→ "sensitive") or "long-term" /
     "position" hints (→ "smooth").**

### Test plan

`tests/test_technical_agent.py`:
- Agent factory structure (uses correct model, has all 4 tools)
- Prompt covers key behaviors (preset selection, error recovery, ticker resolution)
- Mocked tool-call flow (agent picks correct tools for sample questions)
- Output schema validates

---

## 📍 Step B.10 — Manual smoke test

Run `adk web`, point at `technical_agent`, ask 3-5 sample questions:
- "Analyze the technical setup for TCS"
- "Is HDFCBANK overbought?"
- "What are the key levels for INFY this week?"
- "Show me momentum and trend on RELIANCE"
- (Edge case) "Analyze ZOMATO" — newer listing, less history

Verify: tools are picked correctly, output is sensible, no crashes,
graceful degradation on the edge case.

---

## ⏸️ Step C — Prediction Agent (NOT STARTED — preview only)

After Step B is complete, design Step C. Preview of the open questions:

### What goes in?

- Outputs from `technical_agent` (direction, confidence, indicators)
- Outputs from `news_impact` (sentiment, key events)
- Outputs from `price_agent` (current price, recent action)
- Outputs from `kb` (sector, market cap, index membership)
- Optionally: filings analysis, peer comparison

### What comes out?

```python
# Conceptual
class Prediction:
    ticker: str
    horizon: str          # "intraday" | "short" | "medium" | "long"
    direction: str        # "bullish" | "bearish" | "neutral"
    confidence: float     # 0-1
    entry_zone: tuple[float, float]
    target: float
    stop_loss: float
    risk_reward: float
    rationale: str        # multi-paragraph explanation
    contributing_signals: list[str]  # what drove the call
```

### Architecture options

- (α) Single `LlmAgent` with all sub-agents as tools — simple, may struggle
  to coordinate
- (β) `SequentialAgent` with explicit pipeline: KB → technicals → news →
  synthesizer — predictable, less flexible
- (γ) Custom `BaseAgent` that programmatically calls sub-agents and feeds
  results to a final LLM synthesis step — most control, most code

### Open Step-C questions parked for later

- How do we handle conflicting signals (bullish technicals + bearish news)?
- How do we prevent the LLM from inventing numbers (entry / target / stop
  must come from real ATR / level math, not LLM imagination)?
- Do we need backtesting before we trust the predictions?
- How do we present prediction history (was the agent right last week)?

---

## 🅿️ Parking lot — design questions deferred

Things we explicitly chose NOT to tackle in v1, with a note on when to
revisit:

| Item | Why deferred | Revisit when |
|---|---|---|
| Disk persistence for cache | Avoids cache-invalidation bugs | Sessions get long enough that re-fetching is painful |
| Stooq / NSE direct provider | YAGNI; yfinance works for v1 | yfinance breaks production for >10 minutes |
| Chart pattern: cup & handle, flags, wedges | Too noisy / hard to detect reliably | Multimodal LLM + chart image becomes the answer instead |
| Indicator parameters as raw integers | Adds tool-call surface area; LLMs can't pick wisely | Backtesting (Python-driven, not LLM-driven) needs sweeps |
| Volume profile / spike detection | Beyond OBV is v2 territory | Specific user requests it |
| Backtesting framework | Step D, after predictions exist | After Step C produces predictions worth backtesting |
| Web UI / dashboard | Out of v1 scope | After CLI / `adk web` flow is solid |
