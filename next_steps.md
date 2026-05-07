# Next Steps — Stock Predictor v1

> **Scope**: Detailed plan for **upcoming work**. For "what's done already"
> see `implementation_flow.md`. For the higher-level project roadmap see
> `implementation_plan.md`.
>
> **Last updated**: 2026-04-28 — after post-C Provider Expansion
> (Stooq + AlphaVantage + `USE_PAID_PRICES` toggle).

---

## 🚦 Immediate next: Step 3.4.1 — Prediction output schema

### Goal

Define `prediction/schema.py` — the **output contract** every layer of
Step D will produce, consume, log, and (eventually) backtest against.

### Why this is next

Step D is the orchestrator that combines `technical_agent` + `news_impact`
+ `kb` + `price_agent` outputs into a single actionable prediction. Every
downstream concern — persistence, backtesting, calibration, UI rendering
— hangs off whatever shape we lock in here. **Define the contract first;
build the producers and consumers against it.**

This also forces us to answer the hard product questions BEFORE writing
orchestration code: what's a "horizon"? what does "confidence" mean?
how do we represent conflicting signals?

### Locked-in conventions (carry from Step C)

- **Pydantic v2 `BaseModel`** for the schema (matches `data/schema.py`,
  `data/news.py:NewsArticle`, etc.)
- **Frozen / immutable** (`model_config = ConfigDict(frozen=True)`) —
  predictions are facts about a moment in time, not mutable state
- **JSON-serializable** end-to-end — must round-trip via `.model_dump_json()`
  / `.model_validate_json()` for persistence and adk-web display
- **Self-contained** — every field needed to render or audit the prediction
  is in the model. No "dangling references" to other objects.
- **Disclaimer-aware** — explicit `not_advice` / `is_educational` fields
  so consumers can render appropriate UI

### Design proposal (open for tweak)

```python
class PredictionDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

class PredictionHorizon(str, Enum):
    INTRADAY = "intraday"     # same day
    SHORT    = "short"        # 1-5 trading days
    MEDIUM   = "medium"       # 1-4 weeks
    LONG     = "long"         # 1-3 months

class PriceLevel(BaseModel):
    """Reusable typed level — used for entry / target / stop."""
    value: float
    rationale: str            # e.g. "20-day SMA + nearest swing high"

class Prediction(BaseModel):
    # Identity / provenance
    ticker: str               # canonical RELIANCE.NS form
    as_of: datetime           # tz-aware, IST
    horizon: PredictionHorizon
    model_chain: list[str]    # which LLMs participated (audit trail)

    # Core prediction
    direction: PredictionDirection
    confidence: float = Field(ge=0.0, le=1.0)
    entry_zone: tuple[float, float]
    target: PriceLevel
    stop_loss: PriceLevel
    risk_reward: float        # computed: |target - entry| / |entry - stop|

    # Reasoning (for human audit + LLM-quotability)
    rationale: str            # multi-paragraph synthesis
    contributing_signals: list[str]  # bullet list, mirrors C-tool pattern
    conflicting_signals: list[str]   # surface contradictions explicitly

    # Compliance
    not_advice: bool = True
    is_educational: bool = True
```

### Acceptance criteria

- [ ] All field types align with what `technical_agent` / `news_impact`
      / `price_agent` already produce (no last-mile coercion needed)
- [ ] `risk_reward` is computed in a `model_validator` from
      entry/target/stop (single source of truth)
- [ ] `confidence` ∈ [0, 1] enforced at construction (Pydantic `Field`
      constraints)
- [ ] `Prediction` is hashable (frozen + tuple field for entry_zone)
- [ ] `model_dump_json(indent=2)` produces a human-readable artifact
- [ ] Round-trip via JSON preserves all field values (test it)
- [ ] **15-25 unit tests** covering: happy path, validation rules
      (confidence range, RR computation), JSON round-trip, immutability

### Estimated commits

1. Schema definition + tests (~25 tests)

---

## 📍 Step 3.4.2 — Per-stock predictor (after schema lands)

Once the schema exists, `prediction/predictor.py` orchestrates ONE full
analysis cycle per ticker:

1. Resolve ticker via KB
2. Call `technical_agent` for the 4-cluster `TechnicalView`
3. Call `news_impact` for sentiment + key events
4. Call `price_agent` for current price snapshot
5. Synthesize via LLM (with a deterministic template) into a `Prediction`
6. Return + log

**Open architecture question**: should this be a `BaseAgent` (custom code
calling sub-agents) or a `SequentialAgent` (declarative pipeline)? Lean
toward `BaseAgent` for the deterministic orchestration loop; the LLM only
gets called for the synthesis step where it has a tightly-bounded job.

---

## ⏸️ Step D — Prediction Agent (PREVIEW, post-3.4.x)

Builds on 3.4.1 + 3.4.2 to expose the predictor as an `adk web` agent
the user can converse with naturally.

### What goes in?

- Outputs from `technical_agent` (`TechnicalView`)
- Outputs from `news_impact` (sentiment, key events)
- Outputs from `price_agent` (current price, recent action)
- Outputs from `kb` (sector, market cap, index membership)
- Optionally: filings analysis, peer comparison

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

## ✅ Recently completed (reverse chronological)

For full detail see `implementation_flow.md`.

| When | What | Test delta |
|---|---|---|
| 2026-04-28 | **Provider Expansion** — Stooq + AlphaVantage providers, `USE_PAID_PRICES` toggle, off-corp default fix | 554 → 625 (+71) |
| 2026-04-28 | Step C.6 — Manual smoke test in `adk web` + LLM-chain bug fix | env-only |
| 2026-04-28 | Step C.5 — `technical_agent` wiring | 538 → 554 (+16) |
| 2026-04-27 | Step C.4 — `get_levels` tool + chart pattern integration | 509 → 538 (+29) |
| 2026-04-27 | Step C.3 — `get_volatility` tool + position-sizing | 474 → 509 (+35) |
| 2026-04-26 | Step C.2 — `get_momentum` tool + candlestick context-gating | 425 → 474 (+49) |
| 2026-04-26 | Step C.1 — `get_trend` tool + signal classifier + cache singleton | 394 → 425 (+31) |
| Earlier | Steps A, B.1–B.4 | 289 → 394 (+105) |

---

## 🅿️ Parking lot — design questions deferred

| Item | Why deferred | Revisit when |
|---|---|---|
| Disk persistence for cache | Avoids cache-invalidation bugs | Sessions get long enough that re-fetching is painful |
| AV intraday support (`TIME_SERIES_INTRADAY`) | v1 daily is enough; chain falls back for non-daily | Specific intraday use case shows up |
| Stooq weekly/monthly intervals | Daily-only provider; chain falls back to yfinance | Same |
| Chart patterns: cup & handle, flags, wedges | Too noisy / hard to detect reliably | Multimodal LLM + chart image becomes the answer instead |
| Indicator parameters as raw integers | Adds tool-call surface area; LLMs can't pick wisely | Backtesting (Python-driven) needs sweeps |
| Volume profile / spike detection | Beyond OBV is v2 territory | Specific user requests it |
| Backtesting framework | Step E, after predictions exist | After Step D produces predictions worth backtesting |
| Web UI / dashboard | Out of v1 scope | After CLI / `adk web` flow is solid |
| Pydantic models for tool returns | TypedDict is enough for v1 | Schema drift becomes a real problem |
| ADK ToolContext for cache injection | Module singleton works fine | We need per-request cache scoping |
| News/filings deduplication (`data/dedupe.py`) | Predictor may need it for non-double-counting | Step 3.4.2 reveals duplicate-event noise in synthesis |

### Removed from parking (now done)

- ~~Stooq / NSE direct provider~~ — landed in Provider Expansion (post-C)
- ~~yfinance fallback chain~~ — same
