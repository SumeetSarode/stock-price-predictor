# User UI Design — Living Document

> **Purpose:** Single source of truth for what the web-app UI is, why we
> chose what we chose, and what's still undecided. Update as decisions
> land. Anyone landing here cold should be able to read top-to-bottom and
> understand the design intent without trawling chat history.

**Last updated:** 2026-07-10
**Status:**  **Web app v1.1 in progress.** Beyond the Step-1 scaffold:
NIFTY 50 dashboard, watchlist row, side panel, search autocomplete,
prediction history, sparklines, live grade pills, and a nightly grading
scheduler have all shipped. Web service layer now has test coverage
(off 0%). Remaining before honest web-v1: one real end-to-end data-loop
run (predict -> grade -> sparkline with live bars) + deploy.

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

### ✅ Above-table elements (context strip)

Final list for v1, in vertical order from top of page:

| Element | Why | Effort |
|---|---|---|
| **Index summary bar** — Nifty 50, Nifty Bank, Sensex values + change% | Sets day's mood at a glance | Small |
| **Market status pill** — `🟢 Open` / `🔴 Closed` / `⏸ Holiday` | NSE trading-hours awareness | Tiny |
| **Manual refresh button** — `[ 🔄 Refresh prices · Last updated 11:42 AM ]` | Triggers yfinance batch fetch | Small |
| **⭐ Watchlist row** — saved tickers shown as compact horizontal strip | Personalization. Hidden if user has zero saved tickers. | Small |
| **📈 Top gainers / 📉 Top losers strip** — top 5 each, derived from Nifty 50 by sorting on Change% | Quick "what moved today" | Tiny |

### ✅ Below-table elements

Final list for v1:

| Element | Why | Effort |
|---|---|---|
| **🕘 Recent predictions panel** — last 5 predictions user has run, with grading (✅ on track / ❌ stopped / ⏳ in progress / ⏰ expired) | Engagement loop. Reminds user what they predicted. Builds accountability. Empty state shows friendly "Run your first prediction" prompt. | Small |

Deferred to v0.2:
- 🚫 Sector heatmap
- 🚫 Top news headlines
- 🚫 Calendar strip

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
│   [ 🔄 Refresh prices  ·  Last updated 11:42 AM ]                       │  ← manual refresh
│                                                                       │
│   ⭐ Your watchlist                                                    │  ← watchlist row
│   INFY 1518 +1.06%  ·  RELIANCE 2871 +0.92%  ·  IRCTC 892 -0.34%       │     (hidden if empty)
│                                                                       │
│   📈 Top gainers                  📉 Top losers                       │  ← gainers/losers strip
│   TATAMOTORS    +3.4%             ITC          -2.1%                  │
│   ADANIENT      +2.8%             HINDUNILVR   -1.6%                  │
│   ONGC          +2.1%             NESTLEIND    -1.4%                  │
│   ...                              ...                                │
│                                                                       │
│   ┌──────────────────────────────────────────────────────────┐   │  ← Nifty 50 table
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
│   🕘 Your recent predictions                                          │  ← recent predictions
│   RELIANCE   Weekly bullish 78%   2h ago      ⏳ in progress           │     panel
│   TCS        Weekly bearish 62%   yesterday   ✅ on track              │
│   INFY       Daily  neutral  50%  2 days      ⏰ expired               │
│   IRCTC      Monthly bullish 71%  5 days      ❌ stopped out (-3.2%)   │
│   (empty state: "Run your first prediction to see it here.")          │
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

✅ **Decided:**

| Layer | Scope |
|---|---|
| Home dashboard | Nifty 50 only (50 rows) |
| Search bar universe | Nifty 500 |
| Source of ticker list | Bundled `frontend/data/nifty500.csv` committed to the repo (in-memory at startup, no runtime fetch). Maintainer updates annually. |
| Primary search input | **Company name** — humans think "Reliance", not "RELIANCE.NS" |
| Also matches | Ticker symbol as fallback (power users / traders who think in tickers) |
| Display in autocomplete results | Company name (big) · Ticker (small gray) · Sector (smaller gray) |
| Match style | Case-insensitive. Prefix matches rank above substring matches. |
| Result ordering | Nifty 50 stocks always rank above non-Nifty-50 when both match |

Example CSV row:

```csv
ticker,company_name,sector
RELIANCE.NS,Reliance Industries Ltd,Energy
```

Example autocomplete behavior — user types `rel`:

```
🔍 Reliance Industries        RELIANCE.NS  ·  Energy        ← Nifty 50, ranks first
🔍 Reliance Infrastructure    RELINFRA.NS  ·  Infrastructure
🔍 Reliance Power             RPOWER.NS    ·  Energy
🔍 Relaxo Footwears           RELAXO.NS    ·  Consumer Goods
```

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

✅ **Decided: keep all three remaining items for v1.**

| Item | v1 status | Why |
|---|---|---|
| **Index summary bar** (Nifty 50 / Bank Nifty / Sensex) | ✅ keep | Sets day's mood at a glance |
| **Market status pill** (🟢 Open / 🔴 Closed / ⏸ Holiday) | ✅ keep | NSE trading-hours awareness |
| **Q5A: Top gainers / losers strip** | ✅ keep | Free given table data we already have; gives home page life beyond a plain grid |
| **Q5B: Watchlist row above Nifty 50 table** | ✅ keep | Personalization. Especially valuable for non-Nifty-50 watched tickers. Empty state = invisible row (acceptable) |
| **Q5C: Recent predictions panel below table** | ✅ keep | Engagement loop. Turns app from "prediction calculator" into "habit." High ROI for small effort. |
| 🏆 Top conviction calls | 🚫 v0.2 | Needs precomputed predictions (no batch in v1 per Q2) |
| Sector heatmap | 🚫 v0.2 | Lots of polish work; defer |
| Top news headlines | 🚫 v0.2 | Needs scheduled fetch |
| Calendar strip | 🚫 v0.2 | Defer |

**Empty-state design** for Q5B and Q5C:

- Watchlist row: if user has zero saved tickers, the entire row is
  hidden (no "add to watchlist" prompts cluttering home for first-timers)
- Recent predictions panel: if user has zero history, show a friendly
  empty card: "Run your first prediction to see it here." with a
  subtle arrow pointing toward the search bar / Nifty 50 table.

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
| 2026-05-18 | Q3 decided: Nifty 50 home + Nifty 500 searchable. Bundled CSV (`frontend/data/nifty500.csv`) committed to repo, no runtime fetch. Search matches company name (primary) + ticker (fallback), case-insensitive, prefix-first. Autocomplete displays name big, ticker + sector as metadata. Nifty 50 stocks ranked above non-Nifty-50 in results. | User confirmed primary input should be company name, not ticker. Both match for power-user convenience. Bundled CSV chosen for predictability + offline-capable + zero runtime deps. |
| 2026-05-18 | Q5 fully closed: keep all three remaining items for v1 (top gainers/losers strip, watchlist row above table, recent predictions panel below table). All deferred-to-v0.2 items remain deferred (sector heatmap, news, calendar). Wireframe updated to reflect final home layout. **ALL OPEN QUESTIONS NOW CLOSED.** Design phase complete; ready to build. | All three items are small effort with high ROI. Watchlist row and recent predictions panel both have well-defined empty states (hidden / friendly prompt). |
| 2026-05-18 | Step 1 (web scaffold) shipped. Directory layout per the strict frontend/backend split: `frontend/{templates,styles,scripts,vendor,assets}` + `src/price_predictor/web/{app,cli,settings,routes,services}`. Design system implemented with hand-crafted CSS (tokens.css + base.css + components.css) — no Tailwind for v1 (no build step, no Node, no vendor blob). HTMX 1.9.12 vendored. End-to-end working: `uv run price-predictor-web` boots, browser auto-opens, predict form POSTs to `/api/predict`, friendly errors render. Both discipline scripts in place: `check_no_html_in_python.sh` (passing) and `check_no_walmart_traces.sh` (gated for pre-ship). | Step 1 deliberately shipped ONE polished end-to-end flow before scaling out to the full Nifty 50 dashboard. Proves the plumbing AND the visual quality bar in one step. Nifty 50 dashboard, watchlist, side panel, search autocomplete all come in Step 2. |
| 2026-06 | Step 2+ shipped: NIFTY 50 dashboard table, watchlist row, market-summary strip, side panel (`panel_service`), search autocomplete (`search_service`, bundled `nifty500.csv`), prediction history (`history_service`), sparklines (`sparkline_service`), and live grade pills (`grading_service`, read-only replay). | Full home surface + detail flows landed per the Step-2 plan. |
| 2026-06-14 | Two prediction-error UX bugs fixed (`0e9c7b0`): Content-Length mismatch on toast injection; `AllModelsExhaustedError` now renders a friendly rate-limit message instead of a generic 500. Regression tests added (`5b1d0b5`). | Surfaced during v1 verification; locked with tests. |
| 2026-07-10 | Web service layer taken off 0% coverage (`0a0f25c`): smoke tests for sparkline / watchlist / market_summary / prediction_cache / search. Nightly grading scheduler shipped (`2c4ea9d`, `web/services/scheduler.py`) — opt-in via `enable_scheduler`, wired into the app lifespan; warms the price cache + auto-resolves PENDING predictions so the loop closes on its own. | Closes the "grading is manual" gap. Off-by-default keeps `create_app()` and the test suite side-effect-free. |
| 2026-07-10 | Docs cleanup: retired stale planning/journal docs; refreshed README + this doc + `project description.md` + `pred_logic.md` + `pred_logic_solutions.md` to current status. | Doc sprawl trim; git history is the build journal now. |

---

## How to update this doc

- **When a decision lands:** flip the ⏸ to ✅ and add the rationale inline.
- **When a question is answered:** move it from "Open questions" into the
  relevant section and mark ✅.
- **When new questions surface:** add to "Open questions" with the next
  available `Q#`.
- **Always** add a row to the change log.
