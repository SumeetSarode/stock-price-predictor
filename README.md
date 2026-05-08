# 📈 price-predictor

> A free, locally-runnable backend that analyzes Nifty50 stocks and produces
> daily/weekly trading predictions with **explainable reasoning** —
> combining technical analysis, news impact, and self-tracking calibration.

> **Status**: 🚧 v1 in progress · `predict` + `grade` + `calibration`
> shipped end-to-end · backtest replay + concurrency are next.
> **Owner**: Sumeet · **Last updated**: 2026-04-28

---

## 🟢 Current state

| Layer | Status | Notes |
|---|---|---|
| Data fetchers (prices / news / estimates / filings) | ✅ shipped | yfinance + Stooq + AlphaVantage chain; GDELT for news; NSE for filings |
| Knowledge base (Nifty50 registry) | ✅ shipped | Wikipedia-sourced, fuzzy ticker resolution |
| Analysis primitives (trend / momentum / volatility / levels / patterns) | ✅ shipped | pandas-ta backed; pure functions |
| ADK agents (price / news / technical / synthesizer) | ✅ shipped | LiteLLM router (Groq → Gemini fallback) |
| Prediction pipeline (predict / predict-many / store) | ✅ shipped | JSON-on-disk persistence |
| Grading + Calibration (grade / calibration with breakdowns) | ✅ shipped | 6-outcome enum; 3 hit-rate variants; Brier score |
| Backtest (replay / runner / evaluator) | ⏸️ not started | See `next_steps.md` |
| Concurrency / scale (rate-limit-aware router) | ⏸️ not started | See `next_steps.md` |

**Test count**: 854 unit tests passing (+ 7 integration tests run off-corp).

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
```

### Run the test suite

```bash
uv run pytest                            # all unit tests
uv run pytest -m integration             # integration tests (need network + off-corp)
```

---

## 🏗️ Architecture at a glance

```
                       ┌───────────────────────────────────┐
                       │     CLI (typer + rich)            │
                       │  predict / predict-many / history │
                       │  grade / calibration              │
                       └─────────────────┬─────────────────┘
                                         │
                       ┌─────────────────▼─────────────────┐
                       │     prediction/predictor.py       │
                       │  (orchestrator + Runner singletons)│
                       └─────────────────┬─────────────────┘
                                         │
        ┌────────────────┬───────────────┼───────────────┬───────────────┐
        │                │               │               │               │
   ┌────▼────┐     ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼──────┐  ┌─────▼──────┐
   │ price_  │     │ technical_│   │ news_     │   │ synthesizer│  │ guardrails │
   │ agent   │     │ agent     │   │ impact    │   │ (LlmAgent  │  │ Tiers 1-3  │
   │         │     │ (4 tools) │   │           │   │  output_   │  │ + retry    │
   │         │     │           │   │           │   │  schema=   │  │ feedback   │
   │         │     │           │   │           │   │  Prediction)│  │            │
   └─────────┘     └───────────┘   └───────────┘   └────────────┘  └────────────┘
        │                │               │               │
        ▼                ▼               ▼               ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  Data layer: prices (yfinance / Stooq / AV chain) ·          │
   │  news (GDELT) · filings (NSE) · estimates (yfinance) · KB    │
   └──────────────────────────────────────────────────────────────┘

                                 ▼ produces ▼

   ┌──────────────────────────────────────────────────────────────┐
   │  prediction/store.py  →  JSON-on-disk (per-prediction file)  │
   └──────────────────────────────────────────────────────────────┘

                                 ▼ later ▼

   ┌──────────────────────────────────────────────────────────────┐
   │  grade_one + grade_many  →  CalibrationReport                │
   │  (6-outcome enum · 3 hit-rate variants · Brier score)        │
   └──────────────────────────────────────────────────────────────┘
```

**Design principles** (the ones we revisit every commit):
- **Interfaces over implementations** — providers, agents, stores all sit
  behind small ABCs. Swap one without touching the rest.
- **Self-contained outputs** — every prediction JSON has everything needed
  for later audit / grading / UI. No hidden lookups.
- **As-of-date everywhere** — every data fetch supports "as of date X" so
  backtest is honest (no future-info leakage).
- **Async-first** — concurrency baked into batches.
- **Pure functions for math, agents for synthesis** — grading, indicators,
  guardrails are pure functions; only the synthesis step calls an LLM.
- **Honest > convenient** — same-bar T+S ambiguity surfaces as a
  first-class outcome; we report 3 hit-rate variants, not the prettiest one.

---

## 📚 Documentation

| Doc | What lives there |
|---|---|
| **`README.md`** (this file) | Quick start, current state, architecture overview |
| **`docs/project description.md`** | Canonical full spec — problem, goals, output schema, decisions, risks |
| **`implementation_plan.md`** | High-level v1 roadmap with per-step status |
| **`implementation_flow.md`** | Detailed per-step record of what shipped + lessons learned |
| **`next_steps.md`** | What's coming next (the actual TODO list) |
| **`agents.md`** | Notes on individual ADK agents |

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
| Tests | `pytest` + `pytest-asyncio` | 854 unit tests today |

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
