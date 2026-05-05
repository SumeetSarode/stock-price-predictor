# Next Steps — Stock Predictor v1

> **Scope**: Detailed plan for **upcoming work**. For "what's done already"
> see `implementation_flow.md`. For the higher-level project roadmap see
> `implementation_plan.md`.
>
> **Last updated**: 2026-04-28 — after Step C.1 (`get_trend` tool).

---

## 🚦 Immediate next: Step C.2 — `get_momentum` tool

### Goal

Mechanical application of the C.1 pattern to the **momentum cluster**, with
one new wrinkle: **candlestick context-gating**.

### Why this is next

C.1 validated the entire tool pattern (async + plain-dict + 3-level signal +
rationale + shared cache singleton). C.2 is the second tool — same shape, no
new design questions. Building it should be ~30% faster than C.1 because the
plumbing exists.

### Locked design (carries from C.1)

- Async tool: `async def get_momentum(ticker: str, sensitivity: str = "standard") -> dict`
- Returns: `{status, ticker, as_of, preset, signal, indicators, derived, rationale, warnings}`
- Calls into `analysis/momentum.py::momentum_snapshot()` with preset params
- Uses the same shared cache singleton from `data/_shared_cache.py`
- Errors return `{status: "error", ...}` — never raise

### NEW: Candlestick context-gating (the wrinkle)

Per the original design discussion, **candlestick patterns must be
context-gated**: a hammer is only meaningful **near support**. Without
gating, hammers fire on random bars and pollute the LLM's reasoning.

**Implementation approach**:
1. Call `analysis/candlestick_patterns.py::detect_recent_patterns(df, lookback=5)`
2. For each detected pattern, check whether the bar's low (or high) is
   within **1 × ATR-14** of a swing high or swing low (from `analysis/levels.py`)
3. Only surface patterns that pass the gating check
4. Add surviving patterns to the response under `derived.candlestick_patterns`

**Cross-cluster import alert**: this means `get_momentum.py` imports from
both `analysis/levels.py` (swing high/low) and `analysis/volatility.py`
(ATR). That's fine — these are pure-function imports, no architectural
violation. The tool composes primitives across clusters; primitives don't
know about each other.

### Momentum signal classifier rules (proposal — open for tweak)

- **Signal**:
  - `bullish` if RSI > 50 **AND** MACD histogram > 0 **AND** Stoch %K > %D
  - `bearish` if RSI < 50 **AND** MACD histogram < 0 **AND** Stoch %K < %D
  - `neutral` if mixed or insufficient data
- **Strength** (RSI-based extremes):
  - `strong` if RSI > 70 (overbought) or RSI < 30 (oversold)
  - `moderate` if RSI ∈ (40, 60)
  - `weak` otherwise (transition zones)
- **OBV cross-check**: if OBV slope_20 contradicts the signal direction,
  add a warning `"obv_divergence"` to the response.

### Acceptance criteria

- [ ] `_momentum_signal.py` with pure `classify_momentum()` function
- [ ] `get_momentum.py` ADK tool with full uniform call flow
- [ ] Candlestick patterns context-gated by ATR proximity to swing levels
- [ ] Tests: classifier (≥15 tests), tool (≥10 tests with mocked cache)
- [ ] All synthesis: bullish/bearish/neutral cases, edge cases (no patterns,
      patterns far from levels), insufficient history warnings
- [ ] OBV divergence detection works (synthetic test case)

### Estimated commits

1. `_momentum_signal.py` + classifier tests
2. `get_momentum.py` + tool tests + context-gating logic + tests

---

## 📍 Step C.3 — `get_volatility` tool

### Goal

Third cluster tool. The boring-but-critical one — ATR drives **stop-loss
sizing** in Step D. Get the math right.

### Locked design

- Same async + plain-dict + uniform shape as C.1/C.2
- Calls `analysis/volatility.py::volatility_snapshot()`
- No cross-cluster imports needed (no pattern integration)

### Volatility signal classifier rules (proposal)

- **Signal**:
  - `bullish` if BB %B > 0.5 (price in upper half of band)
  - `bearish` if BB %B < 0.5
  - `neutral` if %B is near 0.5 ±0.1
- **Strength**:
  - `strong` if BB squeeze is True (likely breakout incoming)
  - `moderate` if ATR-pct-of-price is in normal range (1–4%)
  - `weak` if ATR-pct < 1% (deeply quiet) or > 6% (manic)

### Position-sizing helpers in the response

Add to `derived`:
- `suggested_stop_loss_distance`: `2 × ATR` (the "2 ATR rule")
- `suggested_position_size_for_1pct_risk`: `account_size * 0.01 / (2 * ATR)`
  — but we don't know account size; emit a **per-share risk** instead
- `volatility_regime`: `"low" | "normal" | "high"` based on ATR percentile
  vs the last 60 bars

### Acceptance criteria

- [ ] `_volatility_signal.py` with pure `classify_volatility()`
- [ ] `get_volatility.py` tool with position-sizing helpers in `derived`
- [ ] Tests: ≥12 classifier tests, ≥8 tool tests
- [ ] Squeeze detection working end-to-end

### Estimated commits

1. `_volatility_signal.py` + classifier tests
2. `get_volatility.py` + tool tests + position-sizing helpers

---

## 📍 Step C.4 — `get_levels` tool

### Goal

Fourth cluster tool + **chart pattern integration** (the second wrinkle
beyond the basic pattern from C.1).

### Locked design

- Same async + plain-dict + uniform shape as C.1/C.2/C.3
- Calls `analysis/levels.py::levels_snapshot()` for swing/52w/pivots
- Calls `analysis/chart_patterns.py::detect_all_patterns()` with default
  confidence threshold 0.7
- Surfaces high-confidence patterns under `derived.chart_patterns`

### Levels signal classifier rules (proposal)

- **Signal**:
  - `bullish` if price near swing-low (within 1 ATR) — potential bounce
  - `bearish` if price near swing-high (within 1 ATR) — potential rejection
  - `bullish` if price broke above swing-high in last 3 bars (breakout)
  - `bearish` if price broke below swing-low in last 3 bars (breakdown)
  - `neutral` otherwise
- **Strength**:
  - `strong` if 52-week high/low is the relevant level (psychological weight)
  - `moderate` if swing high/low is the relevant level
  - `weak` if pivot points are the only relevant level

### Acceptance criteria

- [ ] `_levels_signal.py` with pure `classify_levels()`
- [ ] `get_levels.py` tool with chart pattern integration
- [ ] Tests: ≥15 classifier tests, ≥10 tool tests, ≥5 pattern integration tests
- [ ] Pattern confidence threshold respected (low-conf patterns NOT surfaced)
- [ ] Breakout/breakdown detection working

### Estimated commits

1. `_levels_signal.py` + classifier tests
2. `get_levels.py` + tool tests + chart pattern integration + tests

---

## ✅ Step C.5 — `technical_agent` wiring (DONE)

Delivered the LlmAgent that ties the 4 cluster tools together.

### What landed

```
src/price_predictor/agents/technical_agent/
├── __init__.py        # re-exports root_agent + factory + INSTRUCTION constant
├── agent.py           # LlmAgent factory + ~150-line instruction prompt + root_agent
└── tools/             # already from C.1–C.4
    ├── get_trend.py
    ├── get_momentum.py
    ├── get_volatility.py
    └── get_levels.py
```

### Design decisions taken

1. **No separate `prompt.md`** — instruction kept in `agent.py` as a
   module-level constant (`TECHNICAL_AGENT_INSTRUCTION`) so it can be
   pinned by tests directly without file IO.

2. **No `TechnicalView` Pydantic schema for v1** — the agent returns a
   conversational narrative (the way `price_agent` and `news_impact`
   already do). Structured `TechnicalView` will live in **Step 3.4.1
   (output schema)** so that all agents speak a common contract — not
   redundantly per-agent.

3. **LLM picks tools** (lean confirmed) — prompt teaches:
   - General "how does X look?" → ALL FOUR tools in parallel
   - Specific question → just the relevant cluster
   - "Should I buy?" → ALL FOUR + explicit no-advice disclaimer

4. **Sensitivity defaults to `'standard'`** with explicit "NEVER guess"
   fallback (lean confirmed).

5. **No ticker-format rules in the prompt** — each tool already
   normalizes via `_normalize_ticker`. Adding rules to the prompt too
   would duplicate the truth in two places (DRY).

### Tests added (+16, total 538 → 554)

- Factory smoke: name, description, 4 tools wired, ResilientModel used,
  fresh-instance-per-call
- root_agent module-level discovery (ADK CLI contract)
- Pinned 8 behavior-critical prompt substrings (no buy/sell, all 4 tools
  documented, DEFAULT rule, sensitivity ‘standard’, warnings surfaced,
  self-recovery on `suggested_ticker`, no fabrication, `Rs` format)

---

## ✅ Step C.6 — Manual smoke test (DONE)

Launched `adk web src/price_predictor/agents`, ran the 6-prompt checklist
in `docs/c6_smoke_test_results.md`.

### Bug found and fixed

**Symptom**: every chat turn produced noisy `[resilient] model
incompatibility` warnings; conversations took 5 events instead of 3.

**Root cause**: `.env` had `groq/llama-3.3-70b-versatile` as primary in
`CHAIN_AGENTIC`. Groq's API rejects the assistant message shape ADK
builds for multi-turn tool conversations (assistant turn containing
`thought` part + `function_call` part). Resilient layer fell back to
Gemini — functionally fine, but every turn wasted a roundtrip and spammed
logs. `groq/openai/gpt-oss-120b` was tested as alternative — same issue
on follow-up turns.

**Fix**: reordered `.env` and `.env.example` `CHAIN_AGENTIC` so Gemini
is primary (it's what ADK was designed against; function calling JUST
WORKS), Groq retained as last-resort fallback.

```
OLD: groq/llama-3.3-70b-versatile, gemini/gemini-2.5-flash, gemini/gemini-2.5-flash-lite
NEW: gemini/gemini-2.5-flash, gemini/gemini-2.5-flash-lite, groq/openai/gpt-oss-120b
```

### Known caveat (deferred, not v1-critical for the agent layer)

yfinance is rate-limiting aggressively. Tools return clean errors and
the agent apologizes correctly without fabricating numbers — i.e. the
error-path is verified. Real fix is on the data layer: add a 2nd
price provider (e.g. Stooq) to `data/providers/`. Captured in
`docs/c6_smoke_test_results.md`.

---

## ⏸️ Step D — Prediction Agent (PREVIEW ONLY)

To be designed in detail. Preview of open questions:

### What goes in?

- Outputs from `technical_agent` (`TechnicalView`)
- Outputs from `news_impact` (sentiment, key events)
- Outputs from `price_agent` (current price, recent action)
- Outputs from `kb` (sector, market cap, index membership)
- Optionally: filings analysis, peer comparison

### What comes out?

```python
class Prediction(BaseModel):
    ticker: str
    horizon: Literal["intraday", "short", "medium", "long"]
    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float                         # 0-1
    entry_zone: tuple[float, float]
    target: float
    stop_loss: float
    risk_reward: float
    rationale: str                            # multi-paragraph explanation
    contributing_signals: list[str]
```

### Architecture options

- **(α)** Single `LlmAgent` with all sub-agents as tools — simple, may
  struggle to coordinate
- **(β)** `SequentialAgent` with explicit pipeline: KB → technicals → news
  → synthesizer — predictable, less flexible
- **(γ)** Custom `BaseAgent` programmatically calling sub-agents and feeding
  results to a final LLM synthesis step — most control, most code

**Tentative lean**: (γ). The synthesis step needs to handle conflicting
signals deterministically; we want code-driven coordination, not LLM-
driven coordination.

### Open Step D questions parked for later

- How do we handle conflicting signals (bullish technicals + bearish news)?
- How do we prevent the LLM from inventing numbers (entry/target/stop must
  come from real ATR / level math, not LLM imagination)?
- Do we need backtesting before we trust the predictions?
- How do we present prediction history (was the agent right last week)?

---

## 🅿️ Parking lot — design questions deferred

| Item | Why deferred | Revisit when |
|---|---|---|
| Disk persistence for cache | Avoids cache-invalidation bugs | Sessions get long enough that re-fetching is painful |
| Stooq / NSE direct provider | YAGNI; yfinance works for v1 | yfinance breaks production for >10 min |
| Chart patterns: cup & handle, flags, wedges | Too noisy / hard to detect reliably | Multimodal LLM + chart image becomes the answer instead |
| Indicator parameters as raw integers | Adds tool-call surface area; LLMs can't pick wisely | Backtesting (Python-driven) needs sweeps |
| Volume profile / spike detection | Beyond OBV is v2 territory | Specific user requests it |
| Backtesting framework | Step E, after predictions exist | After Step D produces predictions worth backtesting |
| Web UI / dashboard | Out of v1 scope | After CLI / `adk web` flow is solid |
| Pydantic models for tool returns | TypedDict is enough for v1 | Schema drift becomes a real problem |
| ADK ToolContext for cache injection | Module singleton works fine | We need per-request cache scoping |

---

## 📊 Step C scoreboard — actuals vs estimates

| Sub-step | Est. commits | Actual commits | Est. test delta | Actual test delta |
|---|---|---|---|---|
| C.2 `get_momentum` (+ candlestick gating) | 2 | 2 | +25 | **+49** |
| C.3 `get_volatility` (+ position-sizing) | 2 | 2 | +20 | **+35** |
| C.4 `get_levels` (+ chart patterns) | 2 | 2 | +30 | **+29** |
| C.5 `technical_agent` wiring | 2 | 1 | +15 | **+16** |
| C.6 Manual smoke test (+ LLM-chain bug fix) | 0 | 1 | 0 | **0** |
| **Total Step C** | **~8** | **8** | **+~90** | **+129** |

Final test count at Step C complete: **554** (vs ~515 projected; +39 over plan).
The overage is concentrated in C.2 (cross-cluster pattern gating turned
out to need more edge-case coverage than expected) and C.3 (position-
sizing math has more boundary conditions than first scoped). Both are
worth-the-coverage areas.
