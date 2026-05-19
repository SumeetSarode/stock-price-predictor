# User UI Design — Living Document

> **Purpose:** Single source of truth for what the web-app UI is, why we
> chose what we chose, and what's still undecided. Update as decisions
> land. Anyone landing here cold should be able to read top-to-bottom and
> understand the design intent without trawling chat history.

**Last updated:** 2026-05-18
**Status:** Pre-build (vision + architecture phase). No frontend code written yet.

---

## Status legend

| Mark | Meaning |
|---|---|
| ✅ **Decided** | Locked. Change requires explicit revisit. |
| ⏸ **Pending** | Discussed, recommendation on record, awaiting user sign-off. |
| ❓ **Open** | Raised but not yet discussed in depth. |
| 🚫 **Out of scope (v1)** | Explicitly deferred. |

---

## 1. Product vision

### The mental model shift

The app is **not** "type a ticker, get a prediction." It's a **market
dashboard** — Nifty 50 laid out at a glance — that happens to have our
LLM-driven predictions baked into it. Prediction is something you drill
*into* from the dashboard, not the front door.

This puts us in the mental category of **Zerodha Kite / TradingView watchlist
/ Sensibull**, with the LLM prediction overlay as our differentiator.
Other tools show prices. **We show prices + an opinion.**

The dashboard is the heart. You open it every morning with coffee.

### Shipping model

✅ **Decided: Local-first, self-hosted, BYO API keys.**

Each user clones the repo, drops their Gemini + Groq keys into `.env`, runs
`uv run price-predictor-web` (or `docker compose up`), and a browser opens
to `localhost:8000`. No accounts, no hosting, no cloud, no auth,
no rate-limit pooling, no GDPR, no SEBI hosting-liability headaches.

The architecture is the Obsidian / Ollama / Plex model. The fact that the
user can read AND run the code on their own hardware is the actual feature.

### Target user (v1)

Just the author. Single-tenant. After 2–4 weeks of dogfooding, open to
a few friends. Public release decided later based on real usage.

---

## 2. Architectural decisions (frontend / backend split)

### ✅ Strict separation rule

> **Python never produces HTML. The `frontend/` directory contains every
> visual atom of the app. The FastAPI backend only emits data (JSON for
> API endpoints, context dicts for template rendering).**

This means the entire UI could be rewritten in React / Svelte / vanilla
HTML without touching one Python file, and vice versa.

### ✅ Proposed directory layout

```
price_predictor/
├── frontend/                        ← 100% pure frontend, zero Python
│   ├── templates/
│   │   ├── layouts/
│   │   │   └── base.html            ← page shell: nav, footer, asset links
│   │   ├── pages/
│   │   │   ├── home.html
│   │   │   ├── predict.html
│   │   │   ├── watchlist.html
│   │   │   ├── history.html
│   │   │   └── settings.html
│   │   └── components/              ← reusable partials, HTMX-swappable
│   │       ├── ticker_search.html
│   │       ├── prediction_card.html
│   │       ├── price_chart.html
│   │       ├── horizon_picker.html
│   │       ├── confidence_badge.html
│   │       ├── nav.html
│   │       └── disclaimer.html
│   ├── styles/
│   │   ├── input.css                ← Tailwind source (@import, @apply)
│   │   └── components.css           ← custom styles for things Tailwind can't do
│   ├── scripts/
│   │   ├── app.js                   ← app-level JS (minimal — HTMX does most)
│   │   └── charts.js                ← Lightweight Charts wrappers / config
│   ├── vendor/                      ← pinned, vendored, no CDN at runtime
│   │   ├── htmx-1.9.10.min.js
│   │   ├── lightweight-charts-4.x.js
│   │   └── tailwind-3.4.x.css       ← pre-compiled, committed
│   └── assets/
│       └── icons/                   ← SVG icons, favicons
│
├── src/price_predictor/web/         ← BACKEND only — no HTML, no CSS
│   ├── app.py                       ← FastAPI, mounts frontend/ as static + templates
│   ├── routes/
│   │   ├── pages.py                 ← GET /, /predict, /watchlist (renders templates)
│   │   └── api.py                   ← POST /api/predict (JSON in, JSON or HTML partial out)
│   ├── models.py                    ← SQLAlchemy models
│   ├── services/                    ← adapters between FastAPI and core predict logic
│   │   ├── prediction.py
│   │   ├── watchlist.py
│   │   └── history.py
│   └── settings.py                  ← web-app-specific config
```

### ✅ Enforcement

A script `scripts/check_no_html_in_python.sh` runs `rg '<[a-zA-Z]'` over
`src/price_predictor/web/`. Any match fails the check. Wire into pre-commit
hook later.

### ✅ Vendored frontend dependencies (no CDN at runtime)

Pinned versions of HTMX, Lightweight Charts, and pre-compiled Tailwind CSS
live in `frontend/vendor/` and ship with the repo. The running app makes
**zero third-party HTTP calls** at runtime except to the explicitly-needed
APIs (Gemini, Groq, NSE, GDELT, yfinance). System fonts only — no
Google Fonts call.

### ✅ Walmart-purge as final pre-ship gate

Development can continue using Walmart artifactory / proxies for speed.
**Before tagging `v0.1.0`**, run a mandatory purge step:

- Regenerate `uv.lock` against public PyPI
- Delete or rewrite `scripts/build_ca_bundle.sh`
- Scrub cosmetic refs in `pyproject.toml` and `docs/best_practices.md`
- Add MIT `LICENSE`
- Run `scripts/check_no_walmart_traces.sh` — must exit 0
- Tag `walmart-free-v1`

Discipline during dev: don't add **new** files referencing Walmart,
don't hardcode Walmart paths, don't bake Walmart blue into design tokens
(make all colors env-configurable from day 1).

---

## 3. Design system

### ✅ Design philosophy: C + B blend

- **C (Linear / Vercel minimalism)** as the foundation — premium, lots of
  whitespace, Inter font, one accent color, crisp typography, micro
  interactions.
- **B (Modern fintech / Robinhood / Zerodha Kite)** for the data viz —
  colorful charts, friendly numeric displays, clear bullish/bearish
  semantics.

Result: looks like a premium developer tool that happens to be about
trading. Not Bloomberg (too dense), not Robinhood (too consumer).

### ✅ Light mode only

No dark mode toggle in v1. Simpler, more accessible, less code.
Revisit in v2 if user feedback demands it.

### ✅ Color tokens (env-configurable from day 1)

```css
/* defaults — change in .env to retheme without touching code */
--color-primary:   indigo-600    /* #4f46e5 — accents, primary buttons */
--color-bullish:   green-600     /* #16a34a — bullish signal */
--color-bearish:   red-600       /* #dc2626 — bearish signal */
--color-neutral:   slate-500     /* #64748b — neutral / no signal */
--color-bg:        white
--color-fg:        slate-900
--color-muted:     slate-500
```

**Hard no on Walmart blue (#0053e2)** — that's a Walmart trace.

### ✅ Typography

- **UI text:** Inter (system-installed on modern Mac/Windows, fallback to
  `-apple-system, BlinkMacSystemFont`).
- **Numbers / prices:** **JetBrains Mono** or IBM Plex Mono — monospace
  for tabular figures so ₹267.58 aligns cleanly in tables.
- All fonts loaded from `frontend/vendor/fonts/` — no Google Fonts call
  at runtime.

### ✅ Charts: Lightweight Charts (TradingView OSS)

Chosen over Chart.js because:

- Native OHLC candlesticks + volume bars (no plugin needed)
- TradingView quality out of the box — makes the app feel a class above
- Smaller bundle (~45 KB vs Chart.js ~75 KB)
- MIT-licensed, free

### ✅ Reasoning panel: expandable

Result cards show the bullish/bearish signal + key numbers by default.
LLM's news/technical/synthesizer chain of thought hidden in an
expandable `<details>` block — clean by default, available for power
users who want to verify *why* the model said what it said.

### ⏸ Mobile

🚫 **Out of scope (v1).** Local-first means people use it on their
laptop. Make it readable on a phone via Tailwind's `md:` breakpoints,
but don't optimize the UX for it.

---

## 4. Home page — the dashboard

### ✅ Vision summary

A **Nifty 50 market dashboard** with key OHLCV metrics, change %, 52W
high/low, our prediction signal per stock, and a search bar to find
non-Nifty-50 names. This is the front door.

### ⏸ Default column set (Nifty 50 table)

User-requested:
- Open, Close, High, Low, 52W High, 52W Low, Change%

Recommended additions:
- **Volume + 20-day average ratio** — volume spikes are signal
- **Distance from 52W high/low (%)** — actionable framing
- **🎯 Our prediction signal** — badge: `🟢 Bullish 78%` / `🔴 Bearish 65%`
  / `⚪ Neutral 50%` / `—` (no recent prediction). **This is our moat.**
  Sortable. Filterable.
- **Last predicted at** (relative: "2h ago" / "yesterday") — trust signal
- **Sector** (short label or icon) — enables filtering

Recommended skips for v1 (info overload):
- ❌ P/E, P/B, dividend yield — fundamental ratios, niche audience
- ❌ Beta, RSI, MACD as columns — belongs on detail page
- ❌ Sparklines per row — pretty but doubles render cost; v2 candidate

### ⏸ Above-table elements (context strip)

Menu, ranked by recommended priority:

| Element | Why | Effort |
|---|---|---|
| **Index summary bar** — Nifty 50, Nifty Bank, Sensex values + change% | Sets day's mood at a glance. Every trader app has this. | Small |
| **Top 5 gainers / Top 5 losers** | Quick "what moved today" without scanning 50 rows | Small |
| **🏆 Our top conviction calls** — top 3 bullish + 3 bearish predictions by confidence | **THE differentiator.** Other tools show prices; we show opinions. Surface them. | Medium (needs nightly batch) |
| **Watchlist row** (if user has saved stocks) | Personalization — their stuff before the universe | Small |
| **Market status pill** — `🟢 Open` / `🔴 Closed` / `⏸ Holiday: Diwali` with countdown to next open | NSE trading-hours awareness, builds trust | Tiny |

### ⏸ Below-table or side-panel elements

| Element | Why | Effort |
|---|---|---|
| **Sector heatmap** — colored grid showing sector performance (Finviz-style) | Visual, fast, popular | Medium |
| **Recent predictions you've run** — last 5 with grade (✅ on track / ❌ stopped / ⏳ pending) | Engagement loop. Reminds you what you predicted. | Small (uses history DB) |
| **Top news headlines** — top 5 market news from GDELT | Context for *why* things are moving | Medium (needs scheduled fetch) |
| **Calendar strip** — upcoming earnings / events for Nifty 50 in next 7 days | Forward-looking, not just rear-view | Medium |

### ⏸ Search bar

User vision: search by company name, autocomplete suggestions with
Nifty 50 first, then others.

Sub-decisions on the table:

| Question | Recommendation |
|---|---|
| **Universe for "others"** | Nifty 500 (90% useful coverage, manageable autocomplete index). Not all ~2,000 NSE-listed. |
| **What happens when you pick a result** | (a) scroll to it in the table if it's Nifty 50; (b) open side-panel detail if it's outside Nifty 50 |
| **Location** | Sticky in the top nav (always visible). `/` keyboard shortcut to focus. |
| **Search by what** | Company name + ticker, fuzzy-matched |
| **How to cache** | Pre-load the Nifty 500 ticker→name map into JS on page load — instant client-side autocomplete, zero backend round-trip per keystroke |

### ⏸ Row click behavior

Options on the table:

- **A.** Inline expand — shows mini chart + summary inside the row
- **B.** Side panel — slides in from right with full details + prediction (**recommended**)
- **C.** Full detail page `/stock/RELIANCE.NS`
- **D.** Modal

Recommendation: **B (side panel)**. Dashboard stays as reference; details
slide in/out without losing context. Linear / Notion pattern. Modal feels
heavier, full page loses dashboard context, inline expansion makes rows
reflow ugly.

Each row should also expose hover-revealed quick actions:
`[ ⭐ Watch ]  [ 🔮 Predict now ]  [ 📊 Details → ]`

---

## 5. Other pages (planned)

### ✅ v0 scope (must ship)

- **Home / Dashboard** — Nifty 50 table + above/below context elements
- **Watchlist** — saved tickers, latest predictions across horizons, sortable
- **History** — past predictions with auto-grading (hit / miss / pending)

### ⏸ v0.2 candidates (after dogfooding)

- **Single result detail page** — bigger chart, full reasoning, horizon switcher
  (may collapse into the side-panel from the dashboard for v1)
- **Settings** — API keys (read-only, edited via `.env`), rate-limit
  overrides, theme accent color, batch schedule on/off
- **About / Disclaimer** — SEBI disclaimer, license, how-to-get-API-keys,
  GitHub link

---

## 6. Wireframe sketch (rough)

```
┌─────────────────────────────────────────────────────────────────────┐
│  📈 Price Predictor   [Watchlist] [History] [⚙]   Search [____🔍] │  ← sticky nav
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   Nifty 50: 24,832 ▲ +0.42%   Bank Nifty: 53,210 ▼ -0.18%   🟢 Open │  ← index bar
│                                                                       │
│   🏆 Top conviction calls today                                       │
│   🟢 RELIANCE +78%   🟢 INFY +72%   🔴 ITC -65%   🔴 HINDUNILVR -61% │
│                                                                       │
│   📈 Top gainers              📉 Top losers                          │
│   TATAMOTORS +3.4%            ITC -2.1%                              │
│   ...                          ...                                    │
│                                                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ Ticker     Open  Close   High    Low    Chg%   52WH   52WL │   │  ← Nifty 50 table
│   │           Pred         Last Pred                             │   │
│   ├─────────────────────────────────────────────────────────────┤   │
│   │ RELIANCE  2845  2871  2880   2832  +1.4%  3024  2440        │   │
│   │           🟢 78% Wkly      2h ago      [⭐][🔮][📊]          │   │
│   │ TCS       3920  3895  3935   3884  -0.6%  4120  3640        │   │
│   │           🔴 62% Wkly      Yesterday   [⭐][🔮][📊]          │   │
│   │ ...                                                          │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
│   📰 Top headlines                  🗓 Upcoming earnings              │
│   ...                                ...                              │
│                                                                       │
│  ⚠ For educational purposes only. Not investment advice.   v0.1.0    │  ← footer
└─────────────────────────────────────────────────────────────────────┘
```

When a row is clicked → side panel slides in from the right with:
- Larger Lightweight Charts candlestick with entry/target/stop bands overlaid
- Full prediction details (all 4 horizons)
- Expandable LLM reasoning panel
- News headlines specific to that stock
- "Add to watchlist" / "Run fresh prediction" buttons

---

## 7. Open questions (chronological)

Numbered for easy back-reference in chat.

### Q1. Data freshness: EOD-only or attempt intraday?

| Option | Pros | Cons |
|---|---|---|
| **EOD only** *(recommended)* | Reliable. Honest. Simple code. Most retail-analysis happens EOD. | "Last updated 6 PM IST" feels stale during market hours. |
| **Intraday via yfinance** | "Live-ish" prices during market hours | Flaky on free APIs, ~15–20 min delay, rate-limited. Have to display delay clearly to avoid mis-leading users. |
| **Real-time via broker API (Zerodha Kite / Upstox)** | Actual real-time tick data | Requires user broker account + API subscription. Out of local-first scope. |

**Decision needed.** My pick: EOD with a clear "Market is open — prices
update after 6 PM IST" banner during 9:15–18:00 IST window.

### Q2. Prediction signal column from day 1?

If yes, requires a nightly batch script running Nifty 50 predictions so
the home table has data on first page load. Without it, the column is
empty on first load and users see "Run prediction" buttons per row.

**Decision needed.** My pick: nightly batch runs on first launch + every
night thereafter, so the user opening the dashboard at 7 AM IST always
sees fresh signals.

### Q3. Universe scope

| Option | Pros | Cons |
|---|---|---|
| **Nifty 50 only** | Simple, fast, covers most retail interest | Excludes mid/small caps users may track |
| **Nifty 50 home + Nifty 500 searchable** *(recommended)* | Home stays focused, search covers 90% of useful universe | Slightly bigger autocomplete index |
| **Nifty 50 home + all NSE-listed searchable** | Maximum coverage | ~2000 names, autocomplete UX gets messier |

### Q4. Row-click interaction

Recommended: side panel (B). See section 4 for full option list.

### Q5. Which optional home-page sections survive v1?

From the brainstorm menu in section 4, which to ship in v0 vs defer:

Above-table candidates:
- Index summary bar
- Top gainers / losers
- 🏆 Top conviction calls
- Watchlist row
- Market status pill

Below-table candidates:
- Sector heatmap
- Recent predictions you've run
- Top news headlines
- Calendar strip

**Decision needed per item.** My pick for v0: index summary bar + top
conviction calls + market status pill above; recent predictions below.
Defer the rest to v0.2.

### Q6. Reference UIs the user has in mind

❓ Open. User mentioned having a vision; specific reference apps /
screenshots not yet shared. Worth collecting if any.

---

## 8. Explicitly out of scope (v1)

- 🚫 Mobile-optimized UX (readable on phone, not optimized)
- 🚫 Dark mode (light only)
- 🚫 User accounts / signup / login
- 🚫 Multi-tenancy
- 🚫 Cloud hosting / deployment
- 🚫 Email infrastructure (no SMTP, no magic links)
- 🚫 Real-time tick data (EOD only)
- 🚫 Trading execution / broker integration
- 🚫 Portfolio P&L tracking (just predictions + watchlist)
- 🚫 Analytics / telemetry on usage
- 🚫 Internationalization (English only; ₹ INR only)

---

## 9. Build roadmap (recap)

| Step | Time | Output |
|---|---|---|
| **1. Web scaffold** | 3–4 hrs | FastAPI app, vendored static deps, single working page: ticker form → result. Tag `web-v0`. |
| **2. Real pages** | 1 day | Watchlist (SQLite), history with grading, Chart.js price chart on result page. |
| **3. Local-first ergonomics** | 2–3 hrs | Auto-open browser, friendly missing-key error, configurable port, optional APScheduler nightly batch (off by default). |
| **4. Public-OSS README + screenshots** | 2 hrs | Quick-start, how to get free Gemini/Groq keys, contributing guide. |
| **5. 🚨 MANDATORY: Walmart purge (pre-ship gate)** | 1 hr | Regenerate `uv.lock` against public PyPI, drop `build_ca_bundle.sh`, scrub cosmetic refs, add MIT LICENSE. Tag `walmart-free-v1`. |
| **6. Ship** | 30 min | Push to public GitHub, tag `v0.1.0`. |

---

## 10. Change log

| Date | Change | Why |
|---|---|---|
| 2026-05-18 | Initial draft | Captures vision discussion through Q5 of the open questions. |

---

## How to update this doc

- **When a decision lands:** flip the ⏸ to ✅ and add the rationale inline.
- **When a question is answered:** move it from "Open questions" into the
  relevant section and mark ✅.
- **When new questions surface:** add to "Open questions" with the next
  available `Q#`.
- **Always** add a row to the change log.
