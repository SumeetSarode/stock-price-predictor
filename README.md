# 📈 price-predictor

> A free, locally-runnable backend that analyzes Nifty50 stocks and produces
> daily/weekly trading predictions with **explainable reasoning** —
> combining technical analysis, news impact, and self-tracking calibration.

> **Status**: ✅ **v1 DONE** · full predict → backtest loop shipped
> end-to-end · multi-horizon predictions (daily / weekly / biweekly /
> monthly) hardened with research-grounded per-horizon rules ·
> survivorship-bias-aware backtest via `--index NIFTY50` ·
> point-in-time honest historical replay (news + filings filtered by
> as-of date).
> **Owner**: Sumeet · **Last updated**: 2026-05-12 (post Option A — backtest)

---

## 🟢 Current state

| Layer | Status | Notes |
|---|---|---|
| Data fetchers (prices / news / estimates / filings) | ✅ shipped | yfinance + Stooq + AlphaVantage chain; GDELT for news (with point-in-time `published_at` filter); NSE for filings |
| Knowledge base (Nifty50 registry + historical membership) | ✅ shipped | Wikipedia-sourced; `kb/membership.py` reconstructs historical NIFTY 50 constituents on any as-of date via backwards event-walk |
| Analysis primitives (trend / momentum / volatility / levels / patterns) | ✅ shipped | pandas-ta backed; pure functions |
| ADK agents (price / news / technical / synthesizer) | ✅ shipped | LiteLLM router (Groq → Gemini fallback) |
| Prediction pipeline (predict / predict-many / store) | ✅ shipped | JSON-on-disk persistence; fans out across all 4 horizons in parallel; `as_of` plumbed for point-in-time honest replay |
| Multi-horizon rules (per-horizon ATR bands, entry zones, confidence caps) | ✅ shipped | Single source of truth in `prediction/horizon_constants.py`; guardrails + LLM prompt both consult it |
| Grading + Calibration (grade / calibration with breakdowns) | ✅ shipped | 6-outcome enum; 3 hit-rate variants; Brier score; sqrt-t scaled NEUTRAL band |
| **Backtest (replay / runner / evaluator + HTML report)** | ✅ **shipped (v1)** | Cartesian via `--tickers` OR sparse via `--index NIFTY50` (survivorship-bias-aware). End-to-end integration test gates wall-clock <5min. |
| Concurrency / scale (rate-limit-aware router) | ⏸️ post-v1 | See `next_steps.md` Option B — surfaced as the next pain point by the Step 2.7 integration test |
| LightRAG knowledge layer | ⏸️ Phase 2 | See `next_steps.md` Option C |

**Test count**: 1576 unit tests passing (+ 8 integration tests deselected by default; run off-corp via `pytest -m integration`).

---

## 🚀 Quick start

### Prereqs
- Python 3.13
- [`uv`](https://docs.astral.sh/uv/) for package management
- API keys for Groq (primary LLM) and Gemini (fallback) — both have free tiers

### Setup

```bash
# Clone + enter
cd price_predictor

# Create venv + install
uv venv
uv sync

# Configure API keys
cp .env.example .env  # then edit GROQ_API_KEY + GOOGLE_API_KEY
```

### Run

```bash
# Single prediction
uv run price-predictor predict RELIANCE.NS

# Batch over multiple tickers
uv run price-predictor predict-many RELIANCE.NS HDFCBANK.NS TCS.NS

# View prediction history
uv run price-predictor history RELIANCE.NS

# Grade past predictions against actual price action
uv run price-predictor grade

# Calibration report (overall, or broken down)
uv run price-predictor calibration
uv run price-predictor calibration --by horizon
uv run price-predictor calibration --by ticker
uv run price-predictor calibration --by direction
uv run price-predictor calibration --by month

# Backtest — explicit ticker basket (cartesian product of tickers x dates)
uv run price-predictor backtest \
    --start 2024-06-01 --end 2024-06-30 \
    --tickers RELIANCE.NS,TCS.NS,INFY.NS \
    --horizons weekly

# Backtest — survivorship-bias-aware NIFTY 50 (sparse: each as-of date
# uses ONLY that date's historical constituents, not today's 50)
uv run price-predictor backtest \
    --start 2024-01-01 --end 2024-12-31 \
    --index NIFTY50 \
    --horizons weekly --stride 5

# Both produce a self-contained Tailwind HTML report (auto-opens in
# browser by default; pass --no-open to suppress).
```

### Run the test suite

```bash
uv run pytest                            # all unit tests (1576 today)
uv run pytest -m integration             # integration tests (need network + off-corp)

# Convenience wrapper for the off-corp integration run —
# preflights you're actually off corp wifi + writes a timestamped log:
bash scripts/run_integration_offcorp.sh
```

---

## 🏗️ Architecture at a glance

```
                       ┌───────────────────────────────────┐
                       │     CLI (typer + rich)            │
                       │  predict / predict-many / history │
                       │  grade / calibration / backtest   │
                       └─────────────────┬─────────────────┘
                                         │
                       ┌─────────────────▼─────────────────┐
                       │     prediction/predictor.py       │
                       │  (orchestrator + Runner singletons)│
                       │  point-in-time honest via as_of   │
                       └─────────────────┬─────────────────┘
                                         │
        ┌────────────────┬───────────────┼───────────────┬───────────────┐
        │                │               │               │               │
   ┌────▼────┐     ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼──────┐  ┌─────▼──────┐
   │ price_  │     │ technical_│   │ news_     │   │ synthesizer│  │ guardrails │
   │ agent   │     │ agent     │   │ impact    │   │ (LlmAgent  │  │ Tiers 1-4  │
   │         │     │ (4 tools) │   │           │   │  output_   │  │ + retry    │
   │         │     │           │   │           │   │  schema=   │  │ feedback   │
   │         │     │           │   │           │   │  Prediction)│  │            │
   └─────────┘     └───────────┘   └───────────┘   └────────────┘  └────────────┘
        │                │               │               │
        ▼                ▼               ▼               ▼
   ┌──────────────────────────────────────────────────────────┐
   │  Data layer: prices (yfinance / Stooq / AV chain) ·          │
   │  news (GDELT, point-in-time) · filings (NSE) · estimates ·   │
   │  KB stocks + KB membership (NIFTY 50 historical constituents)│
   └──────────────────────────────────────────────────────────┘

                                 ▼ produces ▼

   ┌──────────────────────────────────────────────────────────┐
   │  prediction/store.py  →  JSON-on-disk (per-prediction file)  │
   └──────────────────────────────────────────────────────────┘

                                 ▼ later ▼

   ┌──────────────────────────────────────────────────────────┐
   │  grade_one + grade_many  →  CalibrationReport                │
   │  (6-outcome enum · 3 hit-rate variants · Brier score)        │
   └──────────────────────────────────────────────────────────┘

                  ▼ OR (historical, all-at-once) ▼

   ┌──────────────────────────────────────────────────────────┐
   │  backtest — sweeps predict() across (ticker x as_of) grids:  │
   │   run_backtest(...)        — cartesian (--tickers)           │
   │   run_backtest_grid(pairs) — sparse (--index NIFTY50)        │
   │     → evaluate_backtest → grade_many + calibration           │
   │     → write_html_report  → self-contained Tailwind HTML +    │
   │                            rule-based insights               │
   └──────────────────────────────────────────────────────────┘
```

**Design principles** (the ones we revisit every commit):
- **Interfaces over implementations** — providers, agents, stores all sit
  behind small ABCs. Swap one without touching the rest.
- **Self-contained outputs** — every prediction JSON has everything needed
  for later audit / grading / UI. No hidden lookups.
- **As-of-date everywhere** — every data fetch supports "as of date X" so
  backtest is honest (no future-info leakage). News articles published
  AFTER the as-of date are filtered out at fetch time via a contextvar.
- **Survivorship-bias defense by construction** — `--index NIFTY50`
  reconstructs the historical NIFTY 50 on each as-of date via backwards
  walk through the Wikipedia event log. A 2018 backtest predicts the
  2018 NIFTY 50, not today's.
- **Async-first** — concurrency baked into batches.
- **Pure functions for math, agents for synthesis** — grading, indicators,
  guardrails are pure functions; only the synthesis step calls an LLM.
- **Honest > convenient** — same-bar T+S ambiguity surfaces as a
  first-class outcome; we report 3 hit-rate variants, not the prettiest one.
- **Single source of truth for tunables** — per-horizon constants live in
  exactly one module (`prediction/horizon_constants.py`) and are consulted
  by both the runtime guardrails AND the LLM prompt. No drift possible.

---

## 📚 Documentation

| Doc | What lives there |
|---|---|
| **`README.md`** (this file) | Quick start, current state, architecture overview |
| **`docs/project description.md`** | Canonical full spec — problem, goals, output schema, decisions, risks |
| **`docs/pred_logic.md`** | Self-contained algorithm spec — every classifier, threshold, formula in plain English |
| **`implementation_plan.md`** | High-level v1 roadmap with per-step status |
| **`implementation_flow.md`** | Detailed per-step record of what shipped + lessons learned |
| **`next_steps.md`** | What's coming next (the actual TODO list) |

**Where to look first depending on what you want:**
- *I want to use it* → README quick start (above)
- *I want to understand the design* → `docs/project description.md`
- *I want to see what's done vs todo* → `implementation_plan.md`
- *I want the full backstory of a sub-step* → `implementation_flow.md`
- *I want to know what Sumeet is working on next* → `next_steps.md`

---

## 🧱 Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.13 | Modern, async-first |
| Package manager | `uv` | Fast, deterministic |
| Validation | Pydantic v2 (frozen models) | Schema-as-code; predictions are facts |
| LLM | Groq primary, Gemini fallback | Best free-tier speed + quality combo |
| LLM client | LiteLLM | One interface across providers |
| Agent framework | Google ADK | Tool-calling + `output_schema` support |
| Prices | yfinance / Stooq / Alpha Vantage | Resilient chain (free + paid toggle) |
| News | GDELT | Free, historical, India-indexed |
| Filings | NSE corporate-events endpoints | Free, authoritative, 30-day window |
| Technicals | `pandas-ta` | Pure Python, comprehensive |
| Storage | JSON-on-disk (predictions + grades) | Inspectable, no migration cost |
| CLI | `typer` + `rich` | Modern Python CLI standard |
| Tests | `pytest` + `pytest-asyncio` | 1576 unit tests today |

---

## 🤝 Contributing

This is a personal project — Sumeet calls the shots on direction. If
you're reading this and want to contribute:

1. Read `docs/project description.md` for the design contract
2. Read `next_steps.md` to see what's queued
3. Open a discussion before writing code — the docs evolve together with
   the code, never after

---

## ⚠️ Disclaimer

This system produces **educational predictions** to help build trading
intuition. **It is not financial advice.** Use at your own risk. The
calibration loop is your friend — trust the system only as much as the
hit-rate evidence supports.

---

## 📜 License

Personal project, all rights reserved. Ask before reusing.
