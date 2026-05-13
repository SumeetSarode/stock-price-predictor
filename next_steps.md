# Next Steps — Stock Predictor v1

> **Scope**: Detailed plan for **upcoming work**. For "what's done already"
> see `implementation_flow.md`. For the higher-level project roadmap see
> `implementation_plan.md`.
>
> **Last updated**: 2026-05-12 — post Option A (backtest replay + runner +
> evaluator + survivorship-bias + integration test). **v1 is now feature-
> complete and gated end-to-end.**

---

## 🟢 Where we are right now — v1 DONE

**v1 closes today.** The full prediction → backtest loop is shipped:

  prediction (×4 horizons) → persistence → grading → calibration
  → historical replay (as-of date shim) → backtest runner
  (cartesian OR sparse via --index) → evaluator → HTML report

| Surface | Shipped? |
|---|---|
| `predict TICKER` (DAILY / WEEKLY / BIWEEKLY / MONTHLY fan-out) | ✅ |
| `predict-many T1 T2 ...` | ✅ |
| `history TICKER` | ✅ |
| `grade` (per-horizon NEUTRAL tolerance) | ✅ |
| `calibration [--by horizon\|ticker\|direction\|month]` | ✅ |
| `backtest --start ... --end ... --tickers ...` (cartesian) | ✅ |
| `backtest --start ... --end ... --index NIFTY50` (sparse, survivorship-aware) | ✅ |
| HTML report with insights + calibration breakdowns | ✅ |
| End-to-end integration test (real APIs, <5min gate) | ✅ |

- **1576 unit tests passing**, 8 integration tests deselected by
  default (run off-corp via `pytest -m integration`). Up from 1021
  pre-Option-A.
- **5 ADK agents** live (`hello`, `price`, `news_impact`, `technical`,
  `synthesizer`)
- **All v1 data layers** shipped (prices chain, news, estimates,
  filings, KB)
- **Honest historical replay**: news + filings filtered by
  `published_at`/`announced_at` against the as-of date via the Step 1.5
  contextvar. Predict() against 2018 prices won't see 2024 news.
- **Honest survivorship-bias defense**: `--index NIFTY50` walks the
  Wikipedia event log backwards from the current 50 to reconstruct
  the historical constituents on each as-of date. A backtest of 2018
  predicts the 2018 NIFTY 50, not today's.

## 🎯 Acceptance criteria for Option A — all met

| Criterion | Status |
|---|---|
| Replay shim never leaks future data | ✅ Step 1.5 (contextvar + NewsSnapshot store) |
| Backtest of 30 days × 3 tickers in <5 min | ✅ Step 2.7 (asserted in integration test) |
| `backtest` CLI produces a CalibrationReport identical in shape to live `calibration` | ✅ Step 2.4; Step 2.7 spot-checks the renderer accepts it |
| Survivorship-bias handling documented | ✅ Step 2.5 + 2.6; explained in `kb/membership.py` module docstring |
| 30–50 unit tests + 1 integration test | ✅ ~120 unit tests across 6 backtest test files + 1 integration test |

---

## 🚦 Three reasonable next moves (post-v1)

v1 is closed. Pick based on what provides the most signal vs effort
from here.

### Option B — Step 3.6: concurrency & rate-limit-aware router 🎯 (recommended)

**Why now**: The integration test (Step 2.7) just hit Gemini quota
exhaustion on a 12-pair run. Real backtest runs against NIFTY 50 ×
months of data will hit this every time. The naive Groq/Gemini chain
can't sustain production-scale runs.

**Components needed**:
- Rate-limit-aware LLM router — wraps `llm/factory.py`. On 429, fall
  back to next provider; on persistent 429, exponential backoff with
  jitter. (Partial scaffolding exists in `llm/resilient.py` but the
  cooldown windows are too long for backtest cadence.)
- Token budgeting — track tokens-per-minute across all in-flight
  calls; queue when near the cap.
- Per-provider concurrency caps (separate from the runner's
  `concurrency=` knob).

**Estimated commits**: 3–5.

### Option C — Step 4.1: LightRAG knowledge layer (Phase 2)

Persistent retrieval over the news/filings corpus so agents can cite
historical context: *"this looks like the Q3 2024 announcement that
moved the stock 8%."* Big quality bump, not v1-critical.
**Estimated commits**: 5–7.

### Option D — Run a real backtest and write up findings

With v1 done, the most interesting question becomes: *would this
system have made money?* Run `backtest --index NIFTY50 --start
2024-01-01 --end 2024-12-31` (off-corp, with full Gemini quota),
then write up the calibration findings. **No new code**; pure
validation of what we built.

---

## ✅ Recently completed (reverse chronological)

For full detail see `implementation_flow.md`.

| When | What | Test delta |
|---|---|---|
| 2026-05-12 | **Step 2.7** — end-to-end backtest integration smoke test (3 tickers × 30 days, <5min gate, rate-limit-aware skip) | 1574 → 1576 (+2) |
| 2026-05-12 | **Step 2.6B** — wire `--index NIFTY50` into the backtest CLI (mutex with `--tickers`; calls `_expand_index_to_pairs` → `run_backtest_grid`) | 1568 → 1574 (+6) |
| 2026-05-12 | **Step 2.6A** — `run_backtest_grid(pairs, ...)` for sparse (ticker, as_of) work; `run_backtest` becomes a thin wrapper | 1561 → 1568 (+7) |
| 2026-05-12 | **Step 2.5** doc move — survivorship-bias removed from parking lot |  |
| 2026-05-12 | **Step 2.5C** — golden-path probes against real NIFTY 50 membership data |  |
| 2026-05-12 | **Step 2.5B** — bootstrap real NIFTY 50 membership history from Wikipedia (script + JSON) |  |
| 2026-05-12 | **Step 2.5A** — `kb/membership.py` (members_on / changes_in_range / was_member with backwards event-walk) | (~1487 → 1561) |
| 2026-05-?? | **Step 2.4** — `price-predictor backtest` CLI command (Typer + Rich progress) |  |
| 2026-05-?? | **Step 2.3** — HTML report + rule-based insights |  |
| 2026-05-?? | **Step 2.2** — `evaluate_backtest()` wires grading + calibration into a single artifact |  |
| 2026-05-?? | **Step 2.1** — date sampling (`backtest/dates.py` NSE-aware) + `run_backtest` orchestrator |  |
| 2026-05-?? | **Step 1.5** — point-in-time news replay via contextvar + NewsSnapshot store |  |
| 2026-05-?? | Plumb `as_of` through technicals + skip news in backtest mode |  |
|  | *Option A net: 1021 → 1576 (+555)* |  |
| 2026-04-28 | Step 3.4.6 / Commits A–C — single-source-of-truth horizon constants (see prior log) | 854 → 1021 (+167) |

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

### Removed from parking (now done)

- ~~Survivorship-bias handling~~ — shipped end-to-end as Step 2.5 (kb.membership data layer) + Step 2.6A (`run_backtest_grid` for sparse pairs) + Step 2.6B (`backtest --index NIFTY50` CLI wiring).
- ~~Backtesting framework~~ — fully shipped as Option A: replay (Step 1.5) + runner (Step 2.1) + evaluator (Step 2.2) + report (Step 2.3) + CLI (Step 2.4) + integration test (Step 2.7).
- ~~Stooq / NSE direct provider~~ — landed in Provider Expansion (post-C)
- ~~yfinance fallback chain~~ — same
- ~~Schema for predictions~~ — shipped as Step D.1 (`prediction/schema.py`)
- ~~Per-stock predictor orchestration~~ — shipped as Steps D.2-D.6
- ~~Batch pipeline~~ — shipped as Step D.7 (`predict_many`)
- ~~CLI surface~~ — shipped as Steps D.9 + 3.5.3
