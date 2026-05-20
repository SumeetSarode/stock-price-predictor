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
- **Sector** (short label or icon) — enables filtering

No prediction signal column — per Q2, predictions are pure on-demand.
User clicks a row → side panel opens → user clicks Predict there.

Table is **sortable by any column**, default sort by ticker A–Z. All 50
rows always visible (no pagination, no "top movers only" default —
simple, predictable, no hidden state).

Recommended skips for v1 (info overload):
- ❌ P/E, P/B, dividend yield — fundamental ratios, niche audience
- ❌ Beta, RSI, MACD as columns — belongs on detail page
- ❌ Sparklines per row — pretty but doubles render cost; v2 candidate

### ⏸ Above-table elements (context strip)

Menu, ranked by recommended priority. **🏆 Top conviction calls moved
to v0.2** per Q2 decision (no precomputed predictions in v1).

| Element | Why | Effort | v1 status |
|---|---|---|---|
| **Index summary bar** — Nifty 50, Nifty Bank, Sensex values + change% | Sets day's mood at a glance. Every trader app has this. | Small | ✅ keep |
| **Market status pill** — `🟢 Open` / `🔴 Closed` / `⏸ Holiday` with countdown | NSE trading-hours awareness | Tiny | ✅ keep |
| **Top 5 gainers / Top 5 losers** | Quick "what moved today" without scanning 50 rows | Small | ⏸ pending (Q5) |
| **Watchlist row** (if user has saved stocks) | Personalization | Small | ⏸ pending (Q5) |
| **🏆 Top conviction calls** | Needs precomputed predictions | Medium | 🚫 v0.2 (per Q2) |

### ⏸ Below-table or side-panel elements

| Element | Why | Effort | v1 status |
|---|---|---|---|
| **Recent predictions you've run** — last 5 with grade (✅ on track / ❌ stopped / ⏳ pending) | Engagement loop. Reminds you what you predicted. Still feasible since history DB has user's own runs (not global). | Small | ⏸ pending (Q5) |
| **Sector heatmap** — colored grid showing sector performance | Visual, fast, popular | Medium | 🚫 v0.2 |
| **Top news headlines** — top 5 market news from GDELT | Context for *why* things are moving | Medium | 🚫 v0.2 |
| **Calendar strip** — upcoming earnings / events for Nifty 50 in next 7 days | Forward-looking | Medium | 🚫 v0.2 |

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

✅ **Decided per Q4: side panel for Nifty 50, full page for non-Nifty 50.**

- Click row in Nifty 50 table → side panel slides in from right
- Search bar pick (stock IS Nifty 50) → scroll to row + open side panel
- Search bar pick (stock is OUTSIDE Nifty 50) → full page nav to `/stock/<TICKER>`

Each row exposes hover-revealed quick actions:
`[ ⭐ Watch ]  [ 🔮 Predict now ]  [ 📊 Details → ]`. Clicking anywhere
else on the row triggers the default action (open side panel).

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

### Home page

```
┌───────────────────────────────────────────────────────────────────┐
│  📈 Price Predictor   Search [____________🔍]   [History] [⚙]   │  ← sticky nav
├───────────────────────────────────────────────────────────────────┤
│                                                                       │
│   Nifty 50: 24,832 ▲ +0.42%   Bank Nifty: 53,210 ▼ -0.18%   🟢 Open  │  ← index bar
│   [ 🔄 Refresh prices  ·  Last updated 11:42 AM ]                        │  ← manual refresh
│                                                                       │
│   ┌─────────────────────────────────────────────────────────┐   │  ← Nifty 50 table
│   │ Ticker      Open    Close   High    Low    Chg%   52WH    52WL   │   │     (sortable)
│   ├─────────────────────────────────────────────────────────┤   │
│   │ RELIANCE    2845    2871🟢  2880    2832   +0.92%  3024   2440    │   │  ← 🟢 = LIVE badge
│   │ TCS         3920    3895🔵  3935    3884   -0.64%  4120   3640    │   │  ← 🔵 = EOD badge
│   │ INFY        1502    1518🟢  1525    1498   +1.06%  1620   1280    │   │
│   │ ...                                                              │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                       │
│   On row hover, reveal:  [ ⭐ Watch ]  [ 🔮 Predict ]  [ 📊 Details → ]    │
│                                                                       │
│  ⚠ For educational purposes only. Not investment advice.   v0.1.0     │  ← footer
└───────────────────────────────────────────────────────────────────┘
```

### Side panel (opens when Nifty 50 row clicked)

```
                                  ┌──────────────────────────────────┐
                                  │ RELIANCE INDUSTRIES        [ ✕ ] │
                                  ├──────────────────────────────────┤
                                  │ ₹2,871  🟢 LIVE ~18m            │
                                  │ +₹26  (+0.92%)  vs yesterday      │
                                  │                                   │
                                  │ [ chart — candlestick + volume ] │
                                  │                                   │
                                  │ Open  2845   High  2880           │
                                  │ Low   2832   Vol   8.2M           │
                                  │ 52WH  3024   52WL  2440           │
                                  │                                   │
                                  │ ──────────────────────────────  │
                                  │ Horizon                            │
                                  │ [Daily][Weekly✓][Biweekly][Monthly]│
                                  │                                   │
                                  │        [ 🔮 Run prediction ]       │
                                  │                                   │
                                  │ (after prediction runs:)          │
                                  │ 🔴 BEARISH · 65% confidence        │
                                  │ Predicted from ₹2,845 (yest close)│
                                  │ Entry  ₹2,840–2,850               │
                                  │ Target ₹2,920    Stop ₹2,810      │
                                  │                                   │
                                  │ ▾ View reasoning                   │
                                  │ ▾ Related news (3)                 │
                                  │                                   │
                                  │ [ ⭐ Add to watchlist ]            │
                                  └──────────────────────────────────┘
```

### Full detail page (search for stock outside Nifty 50, e.g. `/stock/IRCTC.NS`)

Same components as the side panel, but laid out as a full-width page
with bigger chart and side-by-side panels for prediction + reasoning +
news. URL is shareable (well, locally — each user's `localhost:8000`).

---

## 7. Open questions (chronological)

Numbered for easy back-reference in chat.

### Q1. Data freshness: EOD-only or attempt intraday?

✅ **Decided: EOD + delayed intraday via yfinance, with manual refresh.**

Two-source blend:

| Field | During market hours (9:15–15:30 IST) | Outside market hours |
|---|---|---|
| Open | yfinance (locked once market opens) | EOD bhavcopy / jugaad-data |
| **Close / current** | **yfinance LTP** (≈15–20 min delayed) | EOD close |
| High (running) | yfinance | EOD |
| Low (running) | yfinance | EOD |
| 52W High/Low | EOD historical | EOD historical |
| Change % | Computed from whichever "Close" is showing | EOD math |
| Volume | yfinance (running cumulative) | EOD |

**Per-row status badges** show data provenance honestly:

- 🟢 `LIVE ~18m` — yfinance current, fresh
- 🔵 `TODAY EOD` — after-market, today's settled close
- ⚪ `PREV CLOSE` — weekend / holiday / pre-open
- 🟡 `⚠ STALE 2h` — yfinance failed; fell back to last known value

**Sub-decisions locked:**

| | Decision | Note |
|---|---|---|
| Q1A | **Manual refresh only** — button in nav: `[ 🔄 Refresh prices  ·  Last updated 11:42 AM ]` | No background polling. User controls cadence. Simplest, most polite to yfinance. |
| Q1B | **Bulk fetch** — 1 batched yfinance call per refresh (`yf.download([...50 tickers])`) | No-brainer. ~1 HTTP call vs 50. |
| Q1C | **"Predicted from" price only on side panel, NOT home page** | Home table stays lean: one price per row + one badge + one signal. Side panel shows both `Current` and `Predicted from` for clarity when user drills into a specific stock. |
| Q1D | **Fallback chain** when yfinance fails: `yfinance current → cached yfinance from <5 min ago → today's EOD close → yesterday's EOD close` | Defensive engineering, no real choice. |
| Q1E | **Env toggle** `SHOW_INTRADAY=true` (default). Set to `false` for pure EOD mode (no yfinance calls at all). | Users who prefer privacy / lighter network can opt out. |

### Q2. When are predictions computed?

✅ **Decided: Pure on-demand. No batch, no scheduler, no background LLM calls.**

Flow:

```
1. User opens localhost:8000
   ↓
2. Home: Nifty 50 dashboard — prices only, no prediction column
   Two ways to engage:
     a) Search bar at top → pick any stock (Nifty 50 or wider universe)
     b) Click any row in the Nifty 50 table
   ↓
3. Detail surface opens for the chosen stock
     • Side panel (slides in from right) if stock is in Nifty 50
     • Full page navigation to /stock/<TICKER> if stock is outside Nifty 50
   ↓
4. Detail surface shows: chart, current price, key stats, horizon picker, Predict button
   ↓
5. User picks horizon (default: weekly) and clicks Predict
   → ~30s wait → result renders inline with signal, confidence,
     entry/target/stop, expandable reasoning. Saved to history.
```

**Why on-demand for v1:** zero scheduler complexity, zero risk of
burning a user's free-tier quota in the background, home loads
instantly, prediction is a *deliberate* action (matches the
"thinking tool, not a feed" vibe), cost stays under user's full
control.

**What we deferred (not closed, just v0.2+):**

- 🏆 "Top conviction calls" strip on home — needs precomputed
  predictions, which we don't have without a batch. Add later as
  opt-in nightly batch for power users.
- Prediction signal column on the Nifty 50 table — same reason.
- Auto-refresh / scheduled re-runs — same reason.

**Horizon picker on the detail surface:**

A horizon picker (4-tab control: Daily / Weekly / Biweekly / Monthly)
sits above the Predict button. Default = weekly (configurable via
`DEFAULT_HORIZON=weekly` in `.env`). User can switch before clicking.
Power users can click all 4 sequentially if they want full coverage.
Each run = ~8 LLM calls per horizon.

### Q3. Universe scope

| Option | Pros | Cons |
|---|---|---|
| **Nifty 50 only** | Simple, fast, covers most retail interest | Excludes mid/small caps users may track |
| **Nifty 50 home + Nifty 500 searchable** *(recommended)* | Home stays focused, search covers 90% of useful universe | Slightly bigger autocomplete index |
| **Nifty 50 home + all NSE-listed searchable** | Maximum coverage | ~2000 names, autocomplete UX gets messier |

### Q4. Row-click interaction

✅ **Decided: Side panel for Nifty 50 stocks; full page for searched stocks outside Nifty 50.**

| Source of selection | Surface | Why |
|---|---|---|
| Click row in Nifty 50 table | Side panel slides in from the right | Dashboard stays as context. Easy to close and pick another stock. Linear / Notion pattern. |
| Search bar pick — stock IS in Nifty 50 | Scroll to that row in the table + open side panel | Same in-context flow. |
| Search bar pick — stock is OUTSIDE Nifty 50 | Full page navigation to `/stock/<TICKER>` | Stock isn't on the dashboard, so there's no "context" to preserve. Bigger canvas. URL shareable. |

Each row in the Nifty 50 table also exposes hover-revealed quick actions:
`[ ⭐ Watch ]  [ 🔮 Predict now ]  [ 📊 Details → ]`. Clicking the row
anywhere except those actions opens the side panel (default action).

### Q5. Which optional home-page sections survive v1?

**Affected by Q2 decision** (no precomputed predictions). Updated recommendations:

Above-table candidates:
- ✅ Index summary bar (Nifty 50 / Bank Nifty / Sensex) — keep
- ✅ Market status pill (🟢 Open / 🔴 Closed / ⏸ Holiday) — keep
- ⏸ Top gainers / losers strip — still possible from EOD/intraday, **pending**
- 🚫 Top conviction calls — deferred to v0.2 (needs precomputed predictions)
- ⏸ Watchlist row — **pending** (depends on watchlist being built in same phase)

Below-table candidates:
- ⏸ Recent predictions you've run (from history) — **pending**, still feasible since history DB has user's own runs
- 🚫 Sector heatmap — defer to v0.2 (lots of work for v1 polish)
- 🚫 Top news headlines — defer to v0.2 (needs scheduled fetch)
- 🚫 Calendar strip — defer to v0.2

**Decision still needed on:** top gainers/losers strip, watchlist row,
recent predictions panel.

### Q6. Reference UIs the user has in mind

✅ **Decided: No additional references.** User confirmed the C+B design
philosophy locked in section 3 captures the vision. No specific app /
screenshot to match.

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
| 2026-05-18 | Q1 decided: EOD + delayed intraday via yfinance, manual refresh, badges per row, side-panel shows both "current" and "predicted from" prices, env toggle to disable intraday. Sub-decisions Q1A–Q1E all locked. Q6 closed (no additional references; locked design philosophy stands). | User chose Option B with manual-refresh sub-pick. "Predicted from" intentionally kept off home table to preserve dashboard density. |
| 2026-05-18 | Q2 + Q4 decided: pure on-demand predictions (no batch, no scheduler), side panel for Nifty 50 row clicks, full page for non-Nifty 50 search results. Horizon picker (4 tabs) with weekly default on the detail surface. Q5 partially closed: top conviction calls deferred to v0.2 (depends on Q2); index bar + market status pill confirmed; gainers/losers/watchlist/recent-predictions still pending. Home wireframe updated to reflect no prediction column. | User shifted to deliberate, intent-driven prediction model. Simpler v1, zero LLM cost without user action, prediction becomes a focused activity rather than background hum. |

---

## How to update this doc

- **When a decision lands:** flip the ⏸ to ✅ and add the rationale inline.
- **When a question is answered:** move it from "Open questions" into the
  relevant section and mark ✅.
- **When new questions surface:** add to "Open questions" with the next
  available `Q#`.
- **Always** add a row to the change log.
