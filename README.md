# Stock Price Predictor

> A free, locally-runnable system that analyzes NSE stocks and produces
> daily/weekly/biweekly/monthly trading predictions with **explainable
> reasoning** — combining technical analysis, news impact, and
> self-tracking calibration. Ships with a CLI **and** a local web app.

> **Status**: **v1 - shipped.** Full predict -> grade -> backtest loop
> end-to-end - multi-horizon predictions hardened with research-grounded
> per-horizon rules - survivorship-bias-aware backtest via `--index NIFTY50`
> - point-in-time honest historical replay (news + filings + articles
> filtered by as-of date) - FastAPI + HTMX web app with panels, history,
> sparklines, live grading and a nightly scheduler.
>
> **Owner**: Sumeet Sarode  ·  **Version**: 1.0.0  ·  **Last updated**: 2026-07-10

---

## Current state

| Layer | Status | Notes |
|---|---|---|
| Data fetchers (prices / news / estimates / filings) |  shipped | yfinance + Stooq + AlphaVantage chain; GDELT 2.0 for news (point-in-time `published_at` filter); NSE + BSE for filings |
| Point-in-time article fetcher (Wayback) |  shipped | `data/wayback.py` — honest historical article bodies via the Wayback CDX API; never returns a snapshot after `as_of`, never falls back to live |
| Knowledge base (NIFTY registry + historical membership) |  shipped | `kb/membership.py` reconstructs historical NIFTY 50 constituents on any as-of date via backwards event-walk |
| Analysis primitives (trend / momentum / volatility / levels / patterns) |  shipped | pandas-ta + TA-Lib backed; pure functions |
| Indicators: Ichimoku cloud (H9b), India VIX regime gate (H9d) |  shipped | `analysis/ichimoku.py`, `analysis/vix.py` (+ `data/vix.py` fetcher) |
| ADK agents (price / news / technical / synthesizer) |  shipped | LiteLLM router (Groq → Gemini fallback) |
| Prediction pipeline (predict / predict-many / store) |  shipped | JSON-on-disk persistence; fans out across all 4 horizons in parallel; `as_of` plumbed for point-in-time honest replay |
| Multi-horizon rules (per-horizon ATR bands, entry zones, confidence caps) |  shipped | Single source of truth in `prediction/horizon_constants.py` |
| Grading + Calibration |  shipped | 6-outcome enum; 3 hit-rate variants; Brier score + **Brier skill score**; sqrt-t scaled NEUTRAL band |
| Backtest (replay / runner / evaluator + HTML report) |  shipped (v1) | Cartesian via `--tickers` OR sparse via `--index NIFTY50` (survivorship-bias-aware); end-to-end integration test gates wall-clock <5min |
| **Web app (FastAPI + HTMX)** |  shipped (v1) | `uv run price-predictor-web` — dashboard, predict form, watchlist panels, history, sparklines, live grade pills |
| **Nightly grading scheduler** |  shipped | `web/services/scheduler.py` — warms price cache + auto-resolves PENDING predictions; opt-in via `enable_scheduler` |

**Test count**: **1744 unit tests passing** (+ 8 integration tests
deselected by default; run via `pytest -m integration` on an open network).

---

## Quick start

### Prereqs
- Python 3.13
- [`uv`](https://docs.astral.sh/uv/) for package management
- API keys for Groq (primary LLM) and Gemini (fallback) — both have free tiers

### Setup

**Deploying to a non-technical user's Windows laptop?** See
[`windows_setup/SETUP.md`](windows_setup/SETUP.md) for the full step-by-step
guide — a free, self-updating, click-to-open deployment (clone once, they
double-click a desktop icon that pulls the latest `release` and opens the app).

**Developer setup (macOS / Linux / Windows):**

```bash
cd price_predictor
uv venv
uv sync
cp .env.example .env  # then edit GROQ_API_KEY + GEMINI_API_KEY
```

> **API keys:** a fresh clone has no `.env` (it's gitignored). You need
> your own free-tier keys: Groq (https://console.groq.com/keys) and
> Gemini (https://aistudio.google.com/app/apikey). Paste them into `.env`.

### Run — CLI

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
uv run price-predictor calibration --by horizon   # or ticker / direction / month

# Backtest — explicit ticker basket (cartesian: tickers x dates)
uv run price-predictor backtest \
    --start 2024-06-01 --end 2024-06-30 \
    --tickers RELIANCE.NS,TCS.NS,INFY.NS --horizons weekly

# Backtest — survivorship-bias-aware NIFTY 50 (sparse: each as-of date
# uses ONLY that date's historical constituents, not today's 50)
uv run price-predictor backtest \
    --start 2024-01-01 --end 2024-12-31 \
    --index NIFTY50 --horizons weekly --stride 5
# Both produce a self-contained HTML report (auto-opens; --no-open to suppress).
```

### Run — Web app

```bash
# Boots uvicorn against the FastAPI app and pops your browser open.
uv run price-predictor-web

# Nightly grading scheduler is opt-in (off by default so tests + local
# boots stay side-effect-free). Enable via env:
WEB_ENABLE_SCHEDULER=true uv run price-predictor-web
```

### Run the test suite

```bash
uv run pytest                 # all unit tests (1744 today)
uv run pytest -m integration  # integration tests (hit real APIs — need open network)

# Convenience wrapper for the integration run:
bash scripts/run_integration.sh
```

---

## Architecture at a glance

```
   ┌───────────────────────────┐        ┌───────────────────────────┐
   │  CLI (typer + rich)       │        │  Web app (FastAPI + HTMX) │
   │  predict / predict-many   │        │  dashboard / panels /     │
   │  history / grade /        │        │  history / sparklines /   │
   │  calibration / backtest   │        │  grade pills · scheduler  │
   └─────────────┬─────────────┘        └─────────────┬─────────────┘
                 └───────────────┬────────────────────┘
                   ┌─────────────▼─────────────┐
                   │   prediction/predictor.py │
                   │  (orchestrator + Runners) │
                   │  point-in-time via as_of  │
                   └─────────────┬─────────────┘
         ┌───────────┬───────────┼───────────┬───────────┐
    ┌────▼────┐ ┌────▼─────┐ ┌───▼─────┐ ┌───▼──────┐ ┌──▼───────┐
    │ price_  │ │technical_│ │ news_   │ │synthesizer│ │guardrails│
    │ agent   │ │agent     │ │ impact  │ │(output_   │ │Tiers 1-4 │
    │         │ │(4 tools) │ │         │ │ schema)   │ │+ retry   │
    └─────────┘ └──────────┘ └─────────┘ └───────────┘ └──────────┘
                             │
   ┌─────────────────────────▼───────────────────────────────────┐
   │ Data: prices (yfinance/Stooq/AV) · news (GDELT, PIT) ·        │
   │ filings (NSE/BSE) · estimates · India VIX · Wayback PIT       │
   │ articles · KB stocks + membership (NIFTY historical)          │
   └──────────────────────────────────────────────────────────────┘
                             ▼ produces ▼
   prediction/store.py → JSON-on-disk (per-prediction file)
                             ▼ later ▼
   grade_one + grade_many → CalibrationReport (6-outcome · 3 hit-rate
                             variants · Brier + Brier skill score)
                             ▼ OR (historical, all-at-once) ▼
   backtest — sweeps predict() across (ticker x as_of) grids →
     evaluate_backtest → grade_many + calibration → write_html_report
```

**Design principles** (revisited every commit):
- **Interfaces over implementations** — providers, agents, stores sit behind small ABCs.
- **Self-contained outputs** — every prediction JSON carries everything for later audit / grading / UI.
- **As-of-date everywhere** — every data fetch supports "as of date X"; news/filings/articles published AFTER the as-of date are filtered at fetch time via a contextvar. The Wayback fetcher enforces the same guarantee for article bodies.
- **Survivorship-bias defense by construction** — `--index NIFTY50` reconstructs the historical index on each as-of date.
- **Pure functions for math, agents for synthesis** — indicators, grading, guardrails are pure; only synthesis calls an LLM.
- **Analysis package is I/O-free** — pure math lives in `analysis/`; fetchers live in `data/` (e.g. `analysis/vix.py` regime gate vs `data/vix.py` fetcher).
- **Honest > convenient** — same-bar T+S ambiguity surfaces as a first-class outcome; we report 3 hit-rate variants.
- **Single source of truth for tunables** — per-horizon constants live in exactly one module, consulted by both guardrails AND the LLM prompt.
- **No HTML in Python** — the web layer keeps templates/CSS/JS in `frontend/`; a pre-commit gate (`scripts/check_no_html_in_python.sh`) enforces it.

---

## Documentation

| Doc | What lives there |
|---|---|
| **`README.md`** (this file) | Quick start, current state, architecture overview |
| **`docs/project description.md`** | Canonical full spec — problem, goals, output schema, decisions, risks |
| **`docs/best_practices.md`** | Living gotchas + patterns discovered while building |
| **`docs/constants_dossier.md`** | Every numeric constant traced to a cited source |
| **`docs/how_it_works.html`** | Visual plain-English walkthrough of the whole system (open in a browser) |
| **`docs/report/`** | Chapter pages behind the walkthrough + a single-file `all_in_one.html` edition |

**Where to look first:**
- *I want to use it* → README quick start (above)
- *I want the big picture, visually* → open `docs/how_it_works.html`
- *I want to understand the design* → `docs/project description.md`
- *I want to vet a numeric constant* → `docs/constants_dossier.md`

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.13 | Modern, async-first |
| Package manager | `uv` | Fast, deterministic |
| Validation | Pydantic v2 (frozen models) | Schema-as-code; predictions are facts |
| LLM | Groq primary, Gemini fallback | Best free-tier speed + quality combo |
| LLM client | LiteLLM | One interface across providers |
| Agent framework | Google ADK | Tool-calling + `output_schema` support |
| Prices | yfinance / Stooq / Alpha Vantage | Resilient chain (free + paid toggle) |
| News | GDELT 2.0 | Free, historical, India-indexed |
| Filings | NSE / BSE corporate-events | Free, authoritative |
| PIT articles | Wayback CDX API + trafilatura | Honest historical article bodies |
| Technicals | `pandas-ta` + TA-Lib | Comprehensive indicator + candlestick coverage |
| Web | FastAPI + HTMX + uvicorn | No build step, no Node, server-rendered |
| Storage | JSON-on-disk (predictions/grades) + SQLite (web/caches) | Inspectable, low migration cost |
| CLI | `typer` + `rich` | Modern Python CLI standard |
| Tests | `pytest` + `pytest-asyncio` | 1744 unit tests today |

---

## Disclaimer

This system produces **educational predictions** to help build trading
intuition. **It is not financial advice.** Use at your own risk. The
calibration loop is your friend — trust the system only as much as the
hit-rate evidence supports.

---

## License

Personal project, all rights reserved. Ask before reusing.
