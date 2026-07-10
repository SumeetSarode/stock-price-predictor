# Prediction Logic — A Vettable Walkthrough

**Audience.** Anyone (you, a domain expert, an LLM) who wants to verify
that what we built is grounded in actual technical-analysis literature
and not vibes. This doc is **self-contained**: every algorithm,
threshold, prompt instruction, calendar rule, retry policy, and
storage convention is described in plain English. You should not need
to read any source code to vet what's here.

**Conventions used below**

- 🔬 **NEEDS BACKTEST** = chosen inside a literature-bracketed range
  but exact value pending empirical calibration on NIFTY 50 history.
- All "lengths" are in **trading-day bars** unless stated otherwise.
- All times are **Indian Standard Time (IST, UTC+05:30)** unless stated.
- "Close", "high", "low", "open", "volume" mean values from the most
  recent completed daily bar from the price-data layer.

**Reading order.** §1 gives the big picture. §2 covers raw data
sources. §3 covers everything we compute deterministically (no LLM).
§4 covers the deterministic cluster classifiers (the bullish/bearish
verdicts). §5 covers the news-impact LLM agent. §6 covers the
synthesizer LLM agent that produces the final prediction. §7 covers
the four guardrail tiers. §8 covers persistence and grading. §9
covers what we deliberately do NOT do. §10 lists open questions a
reviewer should pay particular attention to.

---

## 1. Objective and high-level architecture

### 1.1 What a "prediction" is

For a single NSE-listed stock, at four time horizons, we predict
**five things**:

| Field          | What it is                                              |
|----------------|---------------------------------------------------------|
| Direction      | BULLISH, BEARISH, or NEUTRAL                            |
| Entry zone     | A narrow `[low, high]` price band around current close  |
| Profit target  | A single price level (one number, not a ladder)         |
| Stop-loss      | A single price level                                    |
| Confidence     | A number in `[0, 1]`                                    |

**Risk-reward terminology (M4).** Every BULLISH/BEARISH prediction
surfaces **two** RR fields, both computed (never user-supplied), so
the entire schema stays consistent with the underlying levels.

- **`risk_reward`** — the **worst-fill anchor** of the entry zone
  (zone-high for longs, zone-low for shorts). This is the single-trade
  RR a trader would book if they got the worst possible fill within
  the entry band. We use it as the **conservative sizing filter**:
  rules like `risk_reward ≥ 1.5` then read as "even at the worst
  fill, RR is still acceptable". Renamed from "worst-case RR" — it's
  worst-fill within the band, not worst-case over all adverse paths.
- **`midpoint_rr`** — the **midpoint anchor** of the entry zone
  (entry assumed at `(zone_low + zone_high) / 2`). This matches the
  convention used in published edge studies (Bulkowski's pattern
  catalogues, the Edwards & Magee tradition, broker calculators that
  quote single-entry RR). Surfaced for **literature-comparison** only;
  not used for sizing.
- For any zone of nonzero width, `midpoint_rr ≥ risk_reward` (the
  midpoint fill is unambiguously better than the worst-end fill). They
  coincide when the zone collapses to a single price, which also
  matches the textbook single-entry formula
  `RR = (target − entry) / (entry − stop)`.

The four horizons:

| Horizon  | Trading days | What it means                                     |
|----------|-------------:|---------------------------------------------------|
| DAILY    | 1            | Next-session close (today's close if pre-15:30)   |
| WEEKLY   | 5            | +7 calendar days, snapped to last trading day     |
| BIWEEKLY | 10           | +14 calendar days, snapped to last trading day    |
| MONTHLY  | 21           | +1 calendar month, snapped to last trading day    |

**Side note on "trading days vs calendar days."** The horizon **window
for grading** uses trading-day counts (1, 5, 10, 21 — based on NIFTY's
~21 trading days/month). The **target evaluation moment**
(`target_datetime` on each prediction) uses **calendar windows
snapped backward to the most recent NSE trading day**. Why both? The
window count matches user mental model ("monthly = ~1 month of
trading"), while the snap-back guarantees we never silently extend a
prediction past a long weekend or Diwali holiday.

### 1.2 The two-phase pipeline

A single `predict(ticker, horizons=...)` call runs in two phases.

**Phase 1: GATHER (parallel, horizon-agnostic).** Two things happen in
parallel:

1. The **technical view** is composed: for one ticker we fetch ~1 year
   of daily OHLCV bars and compute four indicator clusters (trend,
   momentum, volatility, levels) plus candlestick and chart pattern
   detection.
2. The **news-impact assessment** is produced by a sub-agent (an
   LLM) that calls four tools (news, filings, estimates, price-action)
   and returns a structured `ImpactAssessment`.

Both outputs are horizon-agnostic. The same RSI value, the same
recent earnings filing, the same ATR — applies to a daily *and* a
monthly call. Re-fetching per horizon would be pure waste.

**Phase 2: SYNTHESIZE (parallel fan-out across N horizons).** For each
requested horizon (default = all four), we build a `SynthesisInput`
bundle (technical view + impact assessment + horizon label) and hand
it to the **synthesizer agent** (a second LLM). Its output is a
`Prediction` object, which is then run through **four guardrail
tiers** (grounding, citation, consistency, calibration). If any
guardrail fails, the synthesizer is retried **once** with the failure
message appended to the prompt.

All N synthesizer calls run in parallel (`asyncio.gather`) and **fail
fast**: if any horizon's synthesis errors, the whole `predict()` call
raises. We deliberately don't return partial results — partial
results mask reliability problems and break the daily+weekly UX
contract that "you always get all four horizons."

### 1.3 Degradation policy

- **Technical-view failure** → abort the whole `predict()` call. The
  technical view is the load-bearing input; without it, predictions
  would be news-only guesses with no anchoring.
- **News-impact failure** → degrade gracefully. We substitute a
  "neutral, confidence=0, no catalysts" `ImpactAssessment` and append
  a `news_impact:degraded` tag to the prediction's `model_chain` so
  consumers can see at a glance that news was missing. The prediction
  is still produced; the synthesizer naturally weights technicals
  more heavily because news contributes nothing either way.
- **Synthesizer failure** (after one retry) → abort. Same
  fail-fast reasoning as above.

### 1.4 Out of scope for v1

So a vetting reviewer doesn't waste time looking for them:

- **Implied volatility / options data.** Per-stock IV is not used —
  no free per-stock IV source for NSE single-name options. ATR is our
  per-stock volatility unit. **India VIX (the index-level volatility
  gauge published by NSE) IS used as a regime-detection input** in
  later chunks of the accuracy overhaul (it's free, daily, and one
  number). VIX is for regime gating only, not directional forecasting.
- **Macro indicators** (interest rates, INR/USD). Not used in v1.
  Acknowledged as a Phase-2 candidate. India VIX is *not* in this
  bucket — see the IV bullet above; it's a single-number daily index
  that costs nothing to integrate.
- **Sector or index relative strength.** Not used. Single-ticker only.
- **Fundamentals** (P/E, ROE, debt ratios). Not used directly. We
  assume the news/filings layer surfaces *material* fundamental
  events via earnings disclosures.
- **Multi-target ladders / partial exits.** Single target only in v1.
- **Position sizing.** Out of scope. We output a level, not a size.
- **Intraday tick or minute data.** Daily bars only.
- **Same-bar target/stop disambiguation.** Counted as a loss, not
  resolved (see §8.4).
- **Six-month / yearly horizons.** Parked for future work; calibration
  becomes meaningless when every prediction sits in its own bucket.
- **Custom-duration horizons** (e.g. "9 days"). Rejected at the schema
  level for the same calibration reason.

---

## 2. What we ingest

### 2.1 Price data (OHLCV daily bars)

**Sources, in fallback order:**

1. **yfinance** (primary)
2. **Stooq** (first fallback — different upstream than yfinance)
3. **Alpha Vantage** (final fallback — requires an API key)

**Resilient fetcher behavior.** A `ResilientPriceFetcher` wraps the
list and tries them in order. Three failure classes:

- **`ValueError`** (caller bug — empty ticker, bad date range): raise
  immediately. Falling back would just hit the same error.
- **`PriceFetchError`** (upstream issue — rate limit, network error,
  empty result): mark the failing provider in cooldown and try the
  next one.
- **Anything else** (unexpected): log a warning, mark the provider
  in cooldown, try the next one. Better than crashing the request on
  something we didn't anticipate.

**Cooldown rule.** When a provider fails transiently, it is marked
"cooled-down" for **60 seconds** (default; configurable per fetcher).
Within that window, the resilient fetcher skips it entirely. This
stops us from hammering a rate-limited API on every request for the
next minute.

**All-cooled-down fallback.** If every provider is cooled down at the
same time, we ignore cooldowns and try them all anyway — better to
risk one more rate-limited call than to fail the user with no answer.

**On total exhaustion** (every provider failed): raise
`AllProvidersExhaustedError`, which carries the chain tried, the last
failure, and the ticker. This bubbles up to the caller as a
`TechnicalViewError` and ultimately a `PredictionError`.

### 2.2 Price-cache layer

To avoid hitting yfinance four times for one user question
("technicals for HDFCBANK" calls all four cluster tools), there is a
process-wide in-memory cache.

- **Key:** `(ticker, interval)` — one entry per ticker per bar size.
- **Storage:** in-memory only. Process restart = fresh cache.
- **Range awareness:** the cache stores the widest date range fetched
  per ticker. A later request for a sub-range slices from cache without
  re-fetching. A wider request fetches the missing chunk and merges.
- **Concurrency:** one `asyncio.Lock` per `(ticker, interval)` key. Two
  parallel calls for the same ticker share one fetch; calls for
  different tickers fetch in parallel.
- **Proactive fetch window:** when a fetch happens, we proactively
  grab **750 calendar days** (≈ 520 NSE trading bars, ~2 calendar
  years). Bumped from 365 in the H7 indicator-math accuracy fix —
  see §3.2 ADX and §3.3 RSI: after the Wilder warmup discard
  (10×ADX-length = 140 bars + 200 SMA cushion + 60-bar squeeze
  lookback) we still want ≥ 180 usable bars. Subsequent calls for
  narrower windows are pure cache slices.
- **Eviction:** none. 50 stocks × ~50 KB ≈ 2.5 MB; trivial.
- **Copy-on-return:** slicing returns a defensive copy, so callers
  can mutate freely without poisoning the cache.

### 2.3 News data (GDELT)

- **Source:** **GDELT Doc API 2.0** at
  `api.gdeltproject.org/api/v2/doc/doc`. Free, no auth required.
- **Index window:** GDELT's rolling 7-day index. We typically request
  1–7 days back; the LLM may request up to 90 days for special cases.
- **Date format:** input as ISO `YYYY-MM-DD`; converted internally to
  GDELT's `YYYYMMDDHHMMSS` UTC. Start gets `000000`, end gets `235959`
  (end-inclusive, mirrors the prices convention).
- **Language filter:** English-only by default (`sourcelang:eng`).
- **Per-article fields kept:** title, URL, published-at (tz-aware UTC,
  later converted to IST for display), source domain, language.
- **What we deliberately DO NOT use:** GDELT's `tone` score and
  `themes` taxonomy. The LLM does its own reading; we don't trust
  GDELT's pre-computed sentiment.
- **HTTP client:** async `httpx.AsyncClient` with a **15-second
  default timeout** and a custom `User-Agent`.
- **Empty-result handling:** an empty result is a *successful zero-row
  DataFrame*, not an error. (Distinguishing the two matters: empty
  results are common; HTTP/JSON errors are not.)
- **Discovery vs extraction split.** Two separate functions:
  - `fetch_news` (and batch variant): returns metadata only. Raises
    `NewsFetchError` on HTTP/JSON/timeout failure.
  - `fetch_article_body`: fetches one URL, runs `trafilatura` for
    clean body text, returns an `ArticleBody` object with status =
    `"success"` (with body) or `"error"` (with `error_message`). It
    **never raises** — extraction failure is normal at scale (paywalls,
    JS-only sites, bot blocks), so we surface it as data instead of
    exceptions.
- **Polite parallelism.** When the news agent asks for many articles
  in one batch, we use an `asyncio.Semaphore` to cap concurrent
  fetches.

### 2.4 NSE corporate filings

- **Source: undocumented internal NSE endpoint**
  (`https://www.nseindia.com/api/corporate-announcements?index=equities&symbol=...`),
  the same one NSE's web UI uses for "Corporate Information →
  Announcements". This is **NOT a public API** — prior versions of
  this doc said "public corporate-announcements feed" which understated
  the operational fragility.
- **What it actually requires:**
  1. **Session-cookie priming.** The endpoint rejects naked GETs (HTTP
     401/403). We first hit `https://www.nseindia.com/` to obtain the
     `nsit`/`nseappid` cookies and a Cloudflare clearance cookie, then
     reuse the same `httpx.Client` for the JSON call.
  2. **Browser-like headers.** A realistic `User-Agent`,
     `Accept-Language`, `Referer`, and `Sec-Fetch-*` set are needed;
     the server fingerprints requests and silently 403s on missing
     headers.
  3. **Cloudflare-aware retry.** When Cloudflare rotates challenge
     tokens (typically Friday afternoon IST during their maintenance
     window), the priming request must be retried.
- **Operational risk.** Because there is no public contract, NSE may
  change the endpoint shape or tighten rate limits at any time without
  notice. This layer is the most fragile thing in the pipeline.
  Mitigation: cross-validate with BSE's equivalent endpoint when an
  important catalyst is at stake.
- **Filter:** by NSE bare-symbol (e.g. `RELIANCE`) and date range.
- **Common types surfaced:** insider trades, board-meeting notices
  (often the imminent earnings catalyst), earnings-result publications,
  regulatory disclosures.
- **Per-filing fields:** type, subject, date, attachment URL, raw
  text snippet.

### 2.5 Analyst estimates

- **Source:** yfinance analyst-estimates snapshot (no historical
  series — only "what analysts currently expect").
- **Per-ticker fields kept:** EPS / revenue forecasts (current quarter,
  next quarter, current year, next year), recommendation distribution
  (count of strong-buy / buy / hold / sell / strong-sell), price-target
  distribution (low / median / mean / high).

---

## 3. What we compute deterministically

All indicators are computed with no LLM in the loop. The functions
are pure: same OHLCV input → same numeric output. The cluster
classifiers in §4 turn these numbers into bullish/bearish verdicts.

### 3.1 Sensitivity presets

The whole indicator layer is parameterized by a **sensitivity preset**
chosen by the caller. Three presets exist; the LLM defaults to
`standard` and only switches if the user explicitly asks for
"short-term / day-trading view" (`sensitive`) or "long-term /
positional view" (`smooth`).

The full preset table:

| Cluster     | Parameter         | `standard`     | `sensitive`    | `smooth`       |
|-------------|-------------------|----------------|----------------|----------------|
| Trend       | SMA lengths       | 20, 50, 200    | 10, 30, 100    | 30, 70, 200    |
| Trend       | EMA length        | 20             | 10             | 30             |
| Trend       | ADX length        | 14             | 9              | 21             |
| Trend       | MA cross pairs    | SMA-50/200 + EMA-9/21 | SMA-50/200 + EMA-9/21 | SMA-50/200 + EMA-9/21 |
| Trend       | MA cross fresh window (bars) | 5    | 5              | 5              |
| Momentum    | RSI length        | 14             | 9              | 21             |
| Momentum    | MACD (fast/slow/signal) | 12/26/9 | 8/17/9         | 19/39/9        |
| Momentum    | Stoch (k/d/smooth)| 14/3/3         | 9/3/3          | 21/5/5         |
| Volatility  | ATR length        | 14             | 9              | 21             |
| Volatility  | Bollinger (length, k σ) | 20, 2.0  | 10, 2.0        | 30, 2.0        |
| Levels      | Swing lookback bars | 30           | 15             | 60             |

All numeric values shown in §3.2–§3.6 below assume `standard` unless
otherwise noted.

### 3.2 Trend cluster

#### Simple Moving Average (SMA)

- **Definition.** Mean of the last `N` close prices.
- **Lengths used:** 20, 50, 200.
- **Source.** Murphy (1999), industry-standard "stack" — short /
  medium / long-term gauges.
- **Derived per SMA:** boolean "is close above this SMA?", and signed
  "% distance to this SMA" (= `(close − SMA) / SMA × 100`).
- **Stack score:** count of SMAs the close is above (0 to 3).

#### Exponential Moving Average (EMA)

- **Definition.** Weighted moving average where each new bar gets
  weight `2 / (N + 1)` and the previous EMA gets `1 − 2 / (N + 1)`.
- **Length used:** 20.
- **Source.** Pring (2002).

#### MA Crossover (Golden Cross / Death Cross + EMA-9/21)

- **Definition.** A *moving-average crossover* is a discrete event:
  the SHORT MA's value crosses through the LONG MA's value on the
  current bar (was on one side yesterday, is on the other side
  today). The classic **Golden Cross** is SMA-50 crossing above
  SMA-200; the **Death Cross** is the inverse. We also ship the
  EMA-9/21 pair (faster swing-trader signal, Pring 2002).
- **Pairs computed by default:** `("sma", 50, 200)` and
  `("ema", 9, 21)`. Constant across all sensitivity presets — these
  are canonical pairs from the literature, not preset-tunable
  parameters. Users can query custom pairs via
  `analysis.trend.detect_ma_cross(df, short, long, kind)` directly.
- **Output contract — the L3 "regime + last event" struct.**
  `detect_ma_cross()` returns:
  ```
  {
    "current":           "above" | "below" | None,
    "last_event":        "bullish" | "bearish" | None,
    "bars_since_event":  int | None,
    "short_ma":          float | None,
    "long_ma":           float | None,
  }
  ```
  L3 was chosen over simpler L1 ("did a cross fire on the latest bar?
  yes/no") and L2 ("... in the last N bars?") because L1/L2 return
  `None` 99% of the time on real data — uninformative output that
  invites the LLM to hallucinate a cross from the static SMA stack.
  L3 always has something true to say: either "fresh cross today",
  "in-regime since N bars ago", "no cross in available history", or
  "insufficient data". Cost is ~3 extra lines of code; the gain is
  agent honesty. See `pred_logic_solutions.md` §H8.
- **Naming convention — code vs. prose.**
  - **Inside the data field**, `last_event` is `"bullish"` /
    `"bearish"` — generic, matches the vocabulary `momentum.py`
    uses for the MACD cross, and works for any pair.
  - **In agent prose and `_trend_signal.py` rationale strings**,
    only the canonical `sma_50_200` pair gets the marketing name
    "Golden Cross" / "Death Cross". EMA-9/21 and any custom pair
    get generic phrasing like `"bullish EMA-9/21 cross"`.
  - This split avoids both the trap of calling an EMA-9/21 cross a
    "Golden Cross" (technically wrong, Murphy 1999 reserves the
    term for SMA-50/200) AND the trap of stripping the term from
    user-facing output (users expect to hear it).
- **Source — definition.** Murphy, *Technical Analysis of the
  Financial Markets* (1999), ch. 9. Pring (2002) for the EMA
  cousin.
- **Source — empirical caveat.** The literature on MA-crossover
  alpha is decisive: it has weakened over time on liquid large-caps.
  - **Brock, Lakonishok & LeBaron (1992)**, *Journal of Finance*
    47(5), found statistically significant edge from the 1/50 and
    1/200 SMA crossover on the Dow 1897–1986 (~0.045% buy-day
    excess return). **Ignored transaction costs.**
  - **Sullivan, Timmermann & White (1999)**, *Journal of Finance*
    54(5), re-tested BLL using White's Reality Check bootstrap to
    correct for data-snooping across 7,846 trading rules. The
    50/200 crossover edge **does not replicate** out-of-sample
    (1987–1996) once data-snooping is corrected.
  - **Zakamulin (2014)**, *Journal of Asset Management*. The
    50/200 SMA Sharpe drops from ~0.6 pre-1970 to ~0.1 post-1990
    on US equity indices.
  - **Han, Yang & Zhou (2013)**, *J. of Financial & Quantitative
    Analysis* 48(5), found MA-timing remains profitable in
    low-volume small-caps during high-vol regimes — but vanishes
    on liquid large-caps after costs.
  - We ship the cross signal primarily as a **truth-telling /
    user-expectation feature**, not as alpha. Users ask about
    Golden Cross; the agent should answer truthfully ("yes, fired
    12 bars ago" or "no, no cross in the last 750 bars") rather
    than inferring one from `above_sma`. The vote weight in §4.1
    reflects this conservative reading.
- **NSE-specific calibration.** 🔬 **NEEDS BACKTEST** — the studies
  cited above are all US/global equity. Indian large-cap equity
  data (NIFTY 50 constituents, 2010–present) has not been
  backtested for these specific weights. The `±0.5 / ±0.3` weights
  in §4.1 are placeholders informed by the US literature; revisit
  once we have NSE backtest data.

#### ADX + Directional Indicators (+DI / −DI)

- **Definition.** Wilder's directional movement system. ADX measures
  *how strongly* price is trending (regardless of direction); +DI and
  −DI measure the *direction* of that trend.
- **Length used:** 14.
- **Source.** Wilder (1978), original definition.
- **Strength threshold attribution.** The trend-signal classifier
  (§4.1) gates on **ADX < 20 → neutral**. Two sourced anchors and one
  unsourced choice are involved here, and they should not be confused:
  - **Wilder (1978), *New Concepts in Technical Trading Systems*,
    Ch. VII**, proposed **25** as the threshold for "a strong trend is
    present". Wilder did not publish a 20-floor.
  - **The 20 floor is a modern convention**, popularised by StockCharts
    ChartSchool: "Wilder suggests that a strong trend is present when
    ADX is above 25 and no trend is present when ADX is below 20.
    There appears to be a gray zone between 20 and 25."
  - **The 0.5 / 0.7 / 0.85 confidence anchors at ADX 20 / 30 / 40
    (§4.1) are our own design choice**, not Wilder's and not
    StockCharts'. Deliberately conservative — we never claim
    near-certainty from a single indicator. 🔬 **NEEDS BACKTEST**
    against a 25-floor variant before being treated as final.
  Earlier prose in this doc said "Wilder's threshold for trending
  market is 20–25" — that conflated Wilder's 25 with the modern 20
  and is now retracted (see `pred_logic_solutions.md` §H1).
- **Convergence guard.** ADX is published only when at least
  `10 × length = 140` bars are available; otherwise null. The reason
  is mathematical: ADX is **doubly** Wilder-smoothed (first the True
  Range and Directional Movement get RMA(N) applied; then DX gets
  RMA(N) applied again to give ADX). Wilder smoothing is an EWMA with
  α = 1/N; the seed-value bias decays as `(1 − 1/N)^k` and only falls
  below 1% by ~5N bars. Because the smoothing is applied twice, ADX
  needs ~5N bars *past* its first-valid bar (which is itself at 2N per
  Wilder 1978), i.e. ~10N total. The previous `2 × length = 28`-bar
  guard left ~36% seed bias on the second smoothing pass — the largest
  single accuracy gap that existed in the trend cluster. Source for
  the convergence math: Skoglund (2017); Kirkpatrick & Dahlquist,
  *Technical Analysis* 3e (FT Press 2016). See also §3.3 RSI for the
  one-pass case (5N) and §3.3 MACD for EMA-of-EMA (5×slow).

### 3.3 Momentum cluster

#### Relative Strength Index (RSI)

- **Definition.** Wilder's RSI. Ratio of average up-moves to average
  down-moves over `N` bars, normalized to `[0, 100]`.
- **Length used:** 14.
- **Convention.** `> 70` overbought, `< 30` oversold (Wilder 1978,
  *New Concepts in Technical Trading Systems*, ch. 6).
- **Trend-RSI 60/40 thresholds.** When we use 60/40 as bull/bear vote
  thresholds elsewhere in this doc (§4.2 momentum signal), that is
  the **Andrew Cardwell / Constance Brown** trend-RSI school, NOT
  Wilder's. Cardwell observed that in uptrends RSI oscillates 40–80
  (40 acts as support); in downtrends 20–60 (60 acts as resistance).
  See Brown, *Technical Analysis for the Trading Professional* 2e
  (McGraw-Hill 2011) which credits Cardwell directly.
- **Convergence guard.** Require `5 × length = 70` bars. RSI uses
  Wilder's RMA, an EWMA with α = 1/N; seed bias decays as
  `(1 − 1/N)^k` and only falls below 1% by ~5N bars. The previous
  `2 × length = 28` minimum left ~36% seed bias — enough to flip
  RSI from 58 to 65 (false bullish vote per the Cardwell 60/40
  thresholds). See §3.2 ADX for the doubly-smoothed case (10N).

#### MACD

- **Definition.** Difference between fast and slow EMAs of close, with
  a signal line that is itself an EMA of the difference. Histogram is
  `MACD line − signal line`.
- **Parameters used:** fast = 12, slow = 26, signal = 9.
- **Source.** **Gerald Appel, late 1970s** (Signalert newsletters);
  later compiled in Appel, *Technical Analysis: Power Tools for Active
  Investors* (FT Press 2005). There is no single canonical 1979
  publication — prior versions of this doc said "Appel 1979" which is
  unsourced; the late-1970s Signalert origin is the verifiable claim.
- **"Cross" detection.** We check the histogram on the latest two
  bars. If it changed sign from negative-or-zero to positive, we
  report `cross = "bullish"`. From positive-or-zero to negative →
  `cross = "bearish"`. Otherwise `cross = None`.
- **Convergence guard.** Require `5 × slow = 130` bars. MACD is
  EMA-of-EMA (the signal line is an EMA of the MACD line, which is
  itself a difference of two EMAs); seed bias compounds across both
  passes. The previous `slow + signal = 35`-bar guard left ~30%
  seed bias in the signal line, which made the discrete `cross`
  field essentially noise on small histories. Source: pred_logic
  review/solutions §H8 derivation following the same EWMA-bias
  argument as ADX/RSI.

#### Stochastic Oscillator

- **Definition.** "Full" stochastic. `%K` is where today's close sits
  in the high/low range of the last `k` bars (`0` = at the low,
  `100` = at the high), smoothed over `smooth_k` bars. `%D` is a
  `d`-bar SMA of `%K`.
- **Parameters used:** `k = 14`, `d = 3`, `smooth_k = 3`.
- **Source.** Lane (1950s), full-stoch standard.

#### On-Balance Volume (OBV)

- **Definition.** Granville's running cumulative sum: add today's
  volume if close went up, subtract if it went down, ignore if flat.
- **Derived: 20-bar slope.**
  `(OBV_today − OBV_20_bars_ago) / |OBV_20_bars_ago| × 100`. A
  rate-of-change-style measure of whether volume has been building or
  fading over the past month.
- **Source.** Granville, *Granville's New Key to Stock Market Profits*
  (Prentice-Hall 1963), building on earlier "continuous volume" work
  by **Woods and Vignola** in the 1940s–50s. Granville named and
  popularized the indicator; he did not invent the cumulative-signed-
  volume concept itself.

### 3.4 Volatility cluster

#### Average True Range (ATR)

- **Definition.** Wilder's smoothed average of "true range" over `N`
  bars. True range for one bar =
  `max(high − low, |high − prev_close|, |low − prev_close|)`. So ATR
  captures gap risk too, not just intra-bar range.
- **Length used:** 14.
- **Source.** Wilder (1978).
- **Convergence guard.** Require `5 × length = 70` bars. ATR uses
  Wilder's RMA (same EWMA derivation as RSI — see §3.3). The
  previous `2 × length = 28`-bar guard left ~36% seed bias, which
  fed directly into Step D's stop-loss sizing (`2 × ATR`). Stops
  could be ~25% off purely from warmup noise. Now: ATR is `null`
  until 70 bars exist, and the synthesizer refuses to size off `null`.
- **Why it matters more than anything else.** ATR is the **single
  most important number** in the whole pipeline. It is the *unit* in
  which stop-loss distance and target distance are expressed (see §6).

#### Bollinger Bands (BB)

- **Definition.** middle = SMA of close over `N`; upper = middle +
  `k × stdev(close, N)`; lower = middle − `k × stdev(...)`.
- **Parameters used:** `length = 20`, `k = 2.0`.
- **Source.** John Bollinger — developed in the **early 1980s**
  while he was active in markets full-time (1980 onward), and **named
  on Financial News Network c. 1983** when an on-air host asked
  what he called them. Formally documented in his book *Bollinger
  on Bollinger Bands* (McGraw-Hill 2001). Source: Bollinger's own
  narrative at https://www.bollingerbands.com/bollinger-bands. The
  earlier wording "Bollinger (1980s)" was vague; the verifiable
  attribution is "early 1980s, named c. 1983 on FNN, formalized
  in book 2001" (M1).
- **Derived fields:**
  - **`%B`** = `(close − lower) / (upper − lower)`. `0` = at lower
    band, `1` = at upper band, `> 1` = above upper, `< 0` = below
    lower. The 0.1 / 0.9 "oversold/overbought" thresholds we use
    downstream (§4.4 volatility signal) come from Bollinger's own
    book ch. 8 — he calls these "the bands as relative high/low"
    territory.
  - **Bandwidth** = `(upper − lower) / middle × 100` (a percent).

#### Two distinct "squeeze" indicators (DO NOT CONFLATE)

The trading literature uses the word "squeeze" for two genuinely
different constructs that we now expose under separate names. Earlier
versions of this doc had a single ambiguous `bb_squeeze` field that
conflated them; the H2 fix split them.

**1. Bollinger bandwidth-percentile squeeze — `bollinger_squeeze`**

- **Definition.** Boolean. Current Bollinger bandwidth is in the
  **lowest 20%** of its values over the past **60 bars**.
- **Source.** John Bollinger himself, *Bollinger on Bollinger Bands*
  (Wiley 2001), Ch. 11 "The Squeeze", p. 121–127. Bollinger's own
  text says "a six-month low in bandwidth is a Squeeze" (~125 trading
  days); we use a 60-bar / 20%-quantile relaxation that shows up in
  nearly every modern broker platform.
- **What it tells you.** "Volatility is currently compressed *relative
  to its own past*." Wide net, prone to false positives in steady
  trending markets where bandwidth is calm because the trend is
  steady (not because a breakout is coiled).
- **Role downstream.** Diagnostic only. Surfaced in the rationale
  string so the synthesizer can see when it disagrees with TTM
  ("Bollinger bandwidth in lowest 20% historically (no TTM squeeze)"),
  but does **not** drive the strength bump.

**2. TTM Squeeze — `ttm_squeeze` (the actionable trigger)**

- **Definition.** Three-field dict: `{on, fire, bars_in_squeeze}`.
  - `on` is **True** when the upper Bollinger Band sits **inside** the
    upper Keltner Channel **AND** the lower BB sits **inside** the
    lower KC. I.e. normal-distribution-implied vol (BB) is *lower
    than* ATR-implied vol (Keltner) — the spring is coiled.
  - `fire` is **True** on the bar where BBs pop back **outside** KCs
    (squeeze just released). This is Carter's recommended trade
    trigger.
  - `bars_in_squeeze` is the consecutive-on count (longer compression
    → more violent breakout, per Carter's empirical observation).
- **Parameters used.** BB(20, 2.0) and Keltner(20, 1.5×ATR). Carter's
  defaults.
- **Source.** John Carter, *Mastering the Trade* (McGraw-Hill 2009),
  Ch. 11. Cross-reference: StockCharts ChartSchool, "TTM Squeeze":
  https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/ttm-squeeze
- **What it tells you.** "Both volatility measures (price-stdev *and*
  ATR-channel) are compressed." Stricter AND-gate than Bollinger
  alone, far fewer false positives in trending markets.
- **Role downstream.** Drives the strength bump in the volatility
  cluster classifier: `ttm_squeeze.on=True` or `ttm_squeeze.fire=True`
  forces strength to `"strong"` regardless of current direction.

**Why two and not one.** They answer different questions and disagree
in distinguishable cases (steady calm uptrend: Bollinger ON, TTM OFF).
Keeping them separate preserves both signals for the synthesizer
instead of collapsing the disagreement into a single ambiguous flag.
### 3.5 Levels cluster

#### Swing high / swing low

- **Definition.** The highest high and lowest low over the **last 30
  bars** (configurable per preset; `standard` = 30).
- **Used as.** The most recent actionable resistance / support.

#### 52-week high / low

- **Definition.** Highest high and lowest low over the last **252
  trading days** (NSE convention: ~52 weeks of trading sessions).
- **Used as.** Psychological levels every retail trader watches.

#### Classic floor-trader pivots

Computed from the most recent completed daily bar:

- `Pivot Point (PP) = (high + low + close) / 3`
- `R1 = 2 × PP − low` (first resistance)
- `S1 = 2 × PP − high` (first support)
- `R2 = PP + (high − low)` (second resistance)
- `S2 = PP − (high − low)` (second support)

Standard floor-trader formulas; appear in any TA textbook.

#### "Distance to level" derived field

For each level (swing_high, swing_low, high_52w, low_52w, R1, R2, S1,
S2), we also compute `(level − close) / close × 100` so the cluster
classifier can ask "is price *near* X?" without re-deriving.

### 3.6 Candlestick patterns

We hand-rolled detectors for **seven** classic single-, two-, and
three-bar patterns. We don't use TA-Lib for installation reasons; the
math is small enough to write directly.

**Lookback.** The most recent **5 bars** are scanned each call. So at
most 5 single-bar patterns + 4 two-bar patterns + 3 three-bar patterns
*could* fire per request — but most are filtered out by the gating
step described at the end of this section.

**Source for the pattern definitions.** Nison, *Japanese Candlestick
Charting Techniques* (1991) — the canonical English-language reference.

**Bar-shape vocabulary used in the rules below.**

- `body = |close − open|`
- `range = high − low`
- `upper_shadow = high − max(open, close)`
- `lower_shadow = min(open, close) − low`
- "Bullish bar" = `close > open`. "Bearish bar" = `close < open`.
- "Small body" = `body ≤ 0.30 × range` (one-third rule).
- "Long body" = `body ≥ 0.60 × range`.

#### Single-bar patterns

**1. Doji.** Open ≈ close. Specifically: `body ≤ 0.10 × range`
(stricter than the generic small-body rule). Meaning: indecision.

**2. Hammer.** All four conditions must hold:
- Small body (here `body ≤ 0.35 × range`, slightly looser than the
  generic 0.30 rule),
- Body is non-zero (avoids divide-by-zero),
- `lower_shadow ≥ 2 × body`,
- `upper_shadow ≤ 0.3 × body`.

Meaning: bullish reversal at support.

**3. Shooting star.** Mirror of hammer. All four:
- `body ≤ 0.35 × range`,
- Body is non-zero,
- `upper_shadow ≥ 2 × body`,
- `lower_shadow ≤ 0.3 × body`.

Meaning: bearish reversal at resistance.

#### Two-bar patterns

**4. Bullish engulfing.**
- Yesterday is a bearish bar.
- Today is a bullish bar.
- Today's body **fully engulfs** yesterday's: `today.open ≤ yesterday.close`
  AND `today.close ≥ yesterday.open`.
- **Real-body guard (M5).** Yesterday's body must be ≥ **10%** of
  yesterday's range. Source: Nison 1991, ch. 4 — "the second day's
  real body must engulf the first day's REAL BODY". Without this
  guard, a (near-)doji prior bar trivially satisfies the open/close
  inequalities (it has no body to fail on). The 10% floor mirrors
  our doji cutoff (`body / range < 0.1` is a doji), so anything
  ≥ 0.1 is by definition "not a doji".

**5. Bearish engulfing.** Mirror. Yesterday bullish, today bearish,
today's body engulfs yesterday's. **Same real-body guard applies.**

#### Three-bar patterns

**6. Morning star.**
- Bar 1: bearish AND long body.
- Bar 2: small body (any color, often gapping down).
- Bar 3: bullish AND long body, AND `bar3.close > midpoint of bar1`,
  where midpoint = `(bar1.open + bar1.close) / 2`.

Meaning: bullish reversal.

**7. Evening star.** Mirror. Bar 1: bullish + long. Bar 2: small body.
Bar 3: bearish + long, closing **below** midpoint of bar 1.

#### Context gating (the noise filter)

Raw candlestick patterns fire **constantly** on random bars. The
trading wisdom is: a hammer only matters near support, a shooting
star only matters near resistance. So before surfacing any pattern to
the LLM, we apply this filter using the swing high/low and current
ATR:

- **Bullish patterns** (hammer, bullish_engulfing, morning_star):
  surface **only if** the bar's `low` is within **`1 × ATR`** of the
  current swing-low.
- **Bearish patterns** (shooting_star, bearish_engulfing,
  evening_star): surface **only if** the bar's `high` is within
  **`1 × ATR`** of the current swing-high.
- **Doji** (neutral): surface if it's near **either** level
  (indecision at any pivot is meaningful). If both, pick the closer
  one for the `context` annotation.
- Otherwise: **dropped silently.** The LLM never sees them.

**Pre-conditions for gating to run at all.** If ATR is missing or ≤ 0,
OR both swing levels are missing, the gate returns an empty list (no
patterns surfaced). Without those anchors we have no notion of
"near".

**Each surviving pattern is enriched** with:
- `context`: one of `near_support`, `near_resistance`, or
  `near_either` (the doji case).
- `level_price`: the price of the level it's near.
- `distance_pct`: `|bar_extreme − level| / level × 100`.

🔬 The 1×ATR proximity threshold and the 5-bar lookback are **NEEDS
BACKTEST.** Nison's book describes the patterns and the
"context matters" principle but does not quantify the threshold.

### 3.7 Chart patterns

Hand-rolled detectors for **five** classic chart patterns. These are
distinct from candlestick patterns: they span many bars and reflect
*structural* price action.

**Source for tolerances.** Lo, Mamaysky, Wang (2000), "Foundations of
Technical Analysis", *Journal of Finance* 55(4) — academic-standard
geometric definitions. Where LMW's tolerance is referenced below, the
exact numbers (1.5%, 0.75%, 22-bar separation) come straight from
that paper.

**Pivot detection.** We use SciPy's `find_peaks` with a
minimum-distance parameter of 5 bars to identify swing highs/lows.
Anything closer than 5 bars apart isn't treated as a real pivot.

**M6 disclaimer (LMW deviation).** Lo, Mamaysky & Wang (2000) do
NOT use raw `find_peaks` on observed prices. They first apply a
**Nadaraya–Watson kernel-regression smoother** to the price series,
then identify extrema in the smoothed series, then enforce the
geometric definitions on those smoothed pivots. Our implementation
uses LMW's *geometric tolerances* (1.5%, 0.75%, 22-bar separation)
but applies them to `find_peaks`-detected pivots on raw OHLC data
— a faster, simpler approximation. Trade-offs: cheaper compute and
no bandwidth-tuning knob, but more sensitivity to single-bar wicks
that the kernel smoother would have washed out. See
`analysis/chart_patterns.py` module header for the same disclaimer
at call-site. 🔬 Phase 2 backtest should compare both pivot-detection
strategies; the kernel smoother is ~20 lines of additional code if
the gap is material.

**Confidence floor.** Any detected pattern with confidence `< 0.7` is
**dropped** before reaching the LLM. Below this floor the
noise-to-signal ratio is too high to help.

#### Double top

- **Geometry.** Two consecutive swing highs at similar prices,
  separated by a trough.
- **Tolerances.**
  - The two peaks must be within **1.5% of their average** (LMW
    Definition 5).
  - The peaks must be at least **22 trading days apart** — LMW (2000)
    Definition 5's own operationalization ("...the two tops occur at
    least a month, or 22 trading days, apart") of Edwards & Magee's
    qualitative "~one month / several weeks" guidance. The 22-day
    figure does NOT appear in E&M directly.
  - There must be a trough strictly between them (`between.empty`
    rejects).
- **Confidence formula:** `peak_similarity × depth_score`, where:
  - `peak_similarity` is `1.0` if the two peaks are identical, falling
    linearly to `0.0` at the 1.5% tolerance edge.
  - `depth_score = min(trough_drop_pct / 0.05, 1.0)` — i.e. a trough
    that drops at least **5%** below the average peak scores `1.0`.
- **Internal floor.** Patterns scoring `< 0.3` are dropped at this
  stage; the additional `≥ 0.7` filter is applied later by the
  caller.
- **Key levels emitted:** `resistance` (avg of peaks), `neckline`
  (the trough), `target = neckline − (resistance − neckline)`
  (classic measured-move projection).

#### Double bottom

- Mirror of double top. Two swing lows within 1.5% of average, ≥ 22
  bars apart, with a peak between.
- **Target** = `neckline + (neckline − support)`.

#### Head and shoulders

- **Geometry.** Three consecutive swing highs where the middle (the
  "head") is taller than the two on either side (the "shoulders").
- **Tolerances** (all from LMW Definition 1):
  - Two shoulder prices within **1.5%** of their average.
  - Two neckline troughs (between shoulder1↔head and head↔shoulder2)
    within **1.5%** of their average.
  - The head must be **≥ 3% above** the higher shoulder (prominence
    requirement).
- **Confidence formula:** `shoulder_similarity × prominence_score
  × neckline_similarity`, where `prominence_score = min(head_prominence
  / 0.03, 1.0)`. Internal floor `< 0.3` drops the result.
- **Key levels emitted:** `head`, both shoulders, `neckline` (avg of
  the two troughs), `target = neckline − (head − neckline)`.

#### Inverse head and shoulders

- Mirror. Three swing **lows** with the middle lowest. Same
  tolerances.
- **Target** = `neckline + (neckline − head)`.

#### Triangles (ascending / descending / symmetric)

- **Geometry.** Fit a least-squares line through the last 5 swing
  highs (the upper trendline) and 5 swing lows (the lower trendline).
  Then classify the resulting two lines:
  - **Ascending:** upper line is **flat**, lower line is **rising**
    (positive slope).
  - **Descending:** upper line is **falling** (negative slope), lower
    line is **flat**.
  - **Symmetric:** upper falling AND lower rising (the lines
    converge).
- **"Flat line" definition** (LMW Definition 4 for rectangles, reused
  here): the spread between the highest and lowest pivot price on
  that trendline is **≤ 0.75% of their mean.**
- **Confidence:**
  - `0.7` baseline for any matching geometry.
  - For symmetric triangles, the confidence rises toward `1.0` if the
    two slopes are mirror-image-equal in magnitude (perfect symmetry
    bonus). The exact formula uses a symmetry ratio
    `min(|upper_slope|, |lower_slope|) / max(|upper_slope|,
    |lower_slope|)` (after normalizing both slopes by recent mean
    price), and confidence becomes `0.5 + 0.5 × ratio`, then clamped
    to `[0.5, 1.0]`.
- **Key levels emitted:** `upper_trendline_now` and
  `lower_trendline_now` (each line's value at the latest bar index).


---

## 4. Deterministic cluster classifiers

The four indicator clusters in §3 emit numbers. The four **classifier
functions** here turn those numbers into one of three verdicts:
**bullish**, **bearish**, or **neutral**, plus a short list of
`signals` (terse strings the synthesizer LLM may quote verbatim).

These classifiers are **pure functions** — no LLM, no randomness.
They exist so the synthesizer LLM gets pre-digested verdicts (less
hallucination surface area, more grounding) instead of being asked to
interpret raw numbers itself.

Each classifier returns a `ClusterAssessment` with three fields:

- `verdict`: bullish | bearish | neutral
- `confidence`: float in `[0, 1]` measuring how strong the verdict is
- `signals`: a tuple of short human-readable strings

### 4.1 Trend classifier

**Inputs:** trend cluster (SMA stack, EMA, ADX, +DI, −DI, MA crosses).

**Verdict logic.**

1. **SMA-stack majority.** Count how many of `{SMA20, SMA50, SMA200}`
   the close is above. 2 or 3 above → bullish lean. 0 or 1 above →
   bearish lean.
2. **ADX strength gate.** If ADX is missing or < **20**, the verdict
   is **neutral** regardless of the SMA stack. **Wilder (1978, Ch.
   VII) used 25** as the strong-trend threshold; **20 is the modern
   StockCharts-popularised practical floor**, not Wilder's. The
   0.5/0.7/0.85 confidence anchors at ADX 20/30/40 below are our
   own design picks — deliberately conservative to encourage neutral
   verdicts in marginal trends. 🔬 **NEEDS BACKTEST against a 25-floor
   variant.** See §3.2 ADX for the full attribution.
3. **DI confirmation.** When ADX ≥ 20:
   - If `+DI > −DI` AND SMA stack is bullish → final verdict
     **bullish**.
   - If `−DI > +DI` AND SMA stack is bearish → final verdict
     **bearish**.
   - Otherwise (DI disagrees with SMA stack) → **neutral**.
4. **MA-crossover nudge.** Steps 1–3 produce a base verdict. The MA
   cross signal (§3.2 MA Crossover) can then **nudge a `neutral`
   verdict** up to `bullish` or down to `bearish`. It cannot override
   a `bullish` or `bearish` verdict already locked in by stack + DI.
   - Only **fresh** crosses (`bars_since_event ≤ 5`) contribute to
     the vote. Stale crosses appear in the rationale text only
     ("Golden Cross regime since 47 bars ago, stale; not voting")
     because the EVENT edge decays in days while the regime info is
     already captured by the SMA stack score.
   - Vote weights (deliberately conservative per §3.2 empirical
     caveat):
     - `sma_50_200` fresh cross: **±0.5**
     - `ema_9_21` fresh cross: **±0.3** (faster pair = more whipsaws)
     - Custom pairs registered later: default **0.0** (rationale
       only) until weights are explicitly added
   - Net vote magnitude must be **≥ 0.5** to nudge. The DI direction
     also acts as a veto: a bullish cross will not nudge to
     `bullish` if `−DI > +DI`.
   - **Conflict by design.** When the two pairs disagree (e.g.
     SMA-50/200 bullish + EMA-9/21 bearish on the same bar), the
     net vote shrinks toward zero (`+0.5 − 0.3 = +0.2`), failing
     the `≥ 0.5` threshold. This correctly captures "pullback
     within an uptrend" — the agent reports both events but the
     verdict stays neutral.
   - 🔬 **NEEDS BACKTEST** for NSE-specific weight calibration; see
     §3.2 MA Crossover for the US literature this is anchored to.

**Confidence formula.** Anchored to ADX strength because that's the
direct measure of trend conviction:
- ADX 20 → confidence 0.5
- ADX 30 → confidence 0.7
- ADX 40+ → confidence 0.85
- Linear interpolation between, capped at 0.85 (we never claim
  "near-certainty" from indicators alone).

**Signals emitted (examples):**
- `"close above SMA20/50/200 (full stack bullish)"`
- `"ADX 32 – strong trend"`
- `"+DI 28 > −DI 16 (buyers in control)"`
- `"Golden Cross fired 3 bars ago"` (fresh — contributes +0.5 vote)
- `"Death Cross regime since 47 bars ago (stale; not voting)"`
- `"bullish EMA-9/21 cross fired today"` (fresh — contributes +0.3 vote)
- `"No sma-50-200 cross in available history (currently above)"`
- `"Verdict nudged bullish by fresh MA cross vote (+0.8)"`

### 4.2 Momentum classifier

**Inputs:** momentum cluster (RSI, MACD line/signal/histogram + cross,
Stoch %K/%D, OBV slope).

**Verdict logic.** A weighted vote across:

1. **RSI level.** RSI > 60 → +1 bull, RSI < 40 → +1 bear, else 0.
   (We deliberately use 60/40 for the *vote*, not 70/30 — 70/30 is
   the overbought/oversold *flag*, which is a different concern.)
2. **MACD cross + position.** A new `bullish` cross → +1 bull;
   `bearish` cross → +1 bear. Additionally, `MACD line > signal line`
   without a fresh cross still counts as +0.5 in the matching
   direction. Histogram sign acts as a tiebreaker.
3. **Stochastic.** %K > %D and both above 50 → +0.5 bull. %K < %D and
   both below 50 → +0.5 bear. Overbought (`>80`) or oversold (`<20`)
   regions don't auto-flip the vote — they're just noted in `signals`.
4. **OBV slope.** Positive 20-bar slope → +0.5 bull. Negative → +0.5
   bear. Magnitude of the slope influences confidence, not direction.

Sum the votes. Net positive → **bullish**. Net negative → **bearish**.
Net zero → **neutral**.

**Confidence formula.** `min(|net_vote| / 3.0, 0.85)`. So a maximally
aligned momentum picture (RSI + MACD cross + stoch + OBV all pointing
the same way) gives ~0.83, capped at 0.85.

**Signals emitted (examples):**
- `"RSI 68 – strong, not yet overbought"`
- `"MACD bullish cross today"`
- `"OBV rising (+12% over 20 bars) – volume confirming"`

### 4.3 Volatility classifier

**Inputs:** volatility cluster (ATR + %ATR-of-price, BB %B, BB
bandwidth, **`bollinger_squeeze`** flag, **`ttm_squeeze`** dict
(`{on, fire, bars_in_squeeze}`)). See §3.4 for why two squeeze
definitions are surfaced separately.

Unlike trend / momentum / levels, the volatility classifier emits
**three orthogonal fields** rather than one verdict, because
volatility carries multiple independent pieces of information:

#### (a) Direction signal: `bullish` / `neutral` / `bearish`

From the location of price *within* the Bollinger band, via %B:
- `%B > 0.55` → **bullish** (price in upper half of band).
- `%B < 0.45` → **bearish** (price in lower half of band).
- Else → **neutral** (price near middle band).

🔬 **NEEDS BACKTEST.** The 0.55 / 0.45 cutoffs are our own design
choice. They form a *5-percentage-point dead zone* around the median
to avoid flipping direction on every bar that grazes 0.50. Bollinger
(2001) himself uses 0.0 / 1.0 (band touches) as the only threshold
pair he labels; the 0.55 / 0.45 inner pair is an editorialization
for a softer "is the bar leaning bullish or bearish?" classification.

#### (b) Strength: `weak` / `moderate` / `strong`

- **`strong`** — **TTM Squeeze on or fired** (Carter 2009 trigger).
  See §3.4. This is the only "strong" path; nothing else escalates
  to strong because TTM is the only volatility signal with documented
  predictive edge in the literature.
- **`weak`** — ATR-percent of price is in the dead-quiet (< 1.0%) or
  manic (> 6.0%) tail. Neither is tradeable: dead-quiet means no
  movement to capture; manic means stops will be eaten by noise.
- **`moderate`** — normal regime (default).

#### (c) Regime label: `low` / `normal` / `high` / `unknown`

A separate, ATR-percent-only categorical (computed by
`classify_volatility_regime`) surfaced for the LLM to quote without
re-deriving:
- `atr_pct_of_price < 1.0%` → **`low`**
- `atr_pct_of_price > 4.0%` → **`high`**
- otherwise → **`normal`**
- `atr_pct_of_price is None` → **`unknown`**

🔬 **NEEDS BACKTEST.** The 1.0% / 4.0% / 6.0% ATR-percent thresholds
are our own design choice. Anchored loosely to the empirical
observation that NSE large-caps spend most days in the 1–3% ATR‑%
range; sub-1% is statistically unusual ("dead") and over-4% indicates
an event-day or systemic stress. **No published source** for the
specific cutoffs; reasonable variants to test in Phase 2 backtest:
0.8 / 3.5 / 5.0 (tighter), 1.5 / 5.0 / 7.0 (looser), or per-stock
percentile-rank (e.g. "low = bottom 10% of own 252-bar history").

**Why this matters to the synthesizer.** Direction (a) feeds the
cross-cluster verdict aggregator. Strength (b) drives stop-loss
sizing weight (`strong` says "breakout incoming — set wider
target"; `weak` says "don't trade"). Regime (c) is purely a label
the LLM can quote in its rationale.

**Note on prior doc wording.** Earlier versions of this doc described
§4.3 as emitting `expanding` / `contracting` / `normal` with
`%B` near `<0.1` or `>0.9` as the trigger. That was the *original
design intent* but was never implemented; the actual code returns
the three orthogonal fields above. The 0.1 / 0.9 numbers had no
source. This section now describes the implemented behavior
(verifiable via `_volatility_signal.py` + `get_volatility.py`).

**Signals emitted (examples):**
- `"⚡ TTM SQUEEZE FIRED this bar after 12 bars of compression"`
- `"⚡ TTM SQUEEZE active (8 bars): BB inside Keltner channels"`
- `"Bollinger bandwidth in lowest 20% historically (no TTM squeeze)"`
- `"%B 0.94 – touching upper band"`
- `"ATR 2.3% of price – elevated"`

### 4.4 Levels classifier

**Inputs:** levels cluster (swing high/low, 52w high/low, pivots
PP/R1/R2/S1/S2, ATR for proximity).

**Verdict logic.** Find the **nearest** level to current close (in
ATR-units, not percent — proximity is regime-relative). Then:

- Close is **within 0.5 ATR above** the nearest support level →
  **bullish** lean (price is bouncing off / holding support).
- Close is **within 0.5 ATR below** the nearest resistance level →
  **bearish** lean (price is being capped at resistance).
- Close is **between** levels (no level within 0.5 ATR) → **neutral**.

**Why ATR-units.** A 1% distance is a lot in a low-vol stock and
nothing in a high-vol stock. Using ATR normalizes across stocks.

**Confidence formula.** Linear in proximity:
- Distance = 0 (sitting on level) → confidence 0.85
- Distance = 0.5 ATR (at the threshold) → confidence 0.5
- No level nearby → confidence 0.5 (neutral verdict)

**Signals emitted (examples):**
- `"close 0.3 ATR above swing-low support 1240"`
- `"R1 pivot 1280 within 0.4 ATR overhead"`
- `"approaching 52-week-high resistance 1520"`

### 4.5 Aggregation: how clusters become a "technical view"

The four classifiers are run independently and their outputs are
bundled into a **`TechnicalView`** dataclass that the synthesizer
sees. The synthesizer is **NOT** told a single aggregate verdict; it
sees all four cluster verdicts side-by-side and is instructed to
weigh them itself, because the right weighting depends on horizon.

That's the whole point of the two-phase architecture: aggregation is
horizon-dependent and lives in the synthesizer prompt (§6), not in
the indicator layer.

The `TechnicalView` bundle additionally includes:
- `ticker`, `as_of`, `current_close`, `bars_used`,
- the raw indicator clusters (not just the classifier outputs — so the
  synthesizer can look up exact RSI value if it wants to quote it),
- the surviving candlestick patterns (post-gating),
- the surviving chart patterns (post-confidence-floor).

---

## 5. The news-impact LLM agent

### 5.1 Why a sub-agent (and not just "fetch news, jam it in")

Headlines without context are a footgun. "Reliance Q4 EPS beats"
sounds bullish, but if the *guidance* in the same release is weaker
than expected, the stock often drops. To get this right we need an
LLM that can:
- Decide which news is **material** (vs noise),
- Read article bodies (not just titles),
- Compare expectations vs reality (analyst estimates as the
  expectations baseline),
- Cross-check with corporate filings and recent price action,
- Output a **structured, bounded** assessment — not a free-form essay.

Hence: a Pydantic-AI sub-agent with **four tools** and a structured
output schema.

### 5.2 The four tools the news agent can call

The agent decides which to call and in what order. It's not a fixed
sequence.

1. **`fetch_news(ticker, days_back=7)`** → list of `NewsArticle`
   (title, url, published_at, source, *no body yet*).
2. **`fetch_article_body(url)`** → `ArticleBody` (status + body text
   if successful, else error reason). Called per-URL after the agent
   filters the headline list down to the ones it deems material.
3. **`fetch_filings(ticker, days_back=30)`** → list of NSE corporate
   announcements (insider trades, board meetings, results).
4. **`fetch_estimates(ticker)`** → analyst-estimates snapshot
   (EPS/revenue forecasts, recommendation distribution, price-target
   distribution).
5. (Internal) **`get_recent_price_action(ticker, days=10)`** → simple
   summary of the last 10 trading days: `{open, close,
   pct_change_total, max_drawdown_intra, days_up, days_down}`. Used by
   the agent to check whether a "surprise" headline has actually
   already moved the price (in which case its forward impact is
   discounted).

### 5.3 What the news agent emits — `ImpactAssessment` schema

Frozen Pydantic v2 model. Required fields:

| Field             | Type                                   | Meaning |
|-------------------|----------------------------------------|---------|
| `direction`       | `bullish` / `bearish` / `neutral`      | Net directional read |
| `magnitude`       | `low` / `medium` / `high`              | How much it should *move the synthesizer's confidence*, not raw stock-price magnitude |
| `confidence`      | float `[0, 1]`                         | How sure the agent is of its own read |
| `summary`         | string                                 | One-paragraph plain-English digest |
| `catalysts`       | tuple of `Catalyst` objects            | Each: title, source URL, published_at, importance, why_it_matters |
| `risks`           | tuple of `Risk` objects                | Same shape — things that *cut against* the direction call |
| `articles_considered` | int                                | How many article bodies the agent actually read |
| `filings_considered`  | int                                | How many filings the agent considered |

`Catalyst` and `Risk` carry the source URL **explicitly**. This is
load-bearing for the citation guardrail in §7.

### 5.4 Behavioural rules the news agent's system prompt enforces

Lifted verbatim from the prompt, paraphrased here for readability:

**Materiality filter.**
- Earnings, M&A, regulatory action, management change, large insider
  trades, guidance changes → **material**.
- Sector op-eds, generic market commentary, "stock down 2% on no
  news" stories → **non-material**, drop them.

**Already-priced-in adjustment.**
If the agent sees a headline whose price-impact has already played
out in the last 10 days (per the price-action tool), it must **down-
grade** that headline's importance and explicitly note "already
absorbed" in `why_it_matters`.

**Expectation framing.**
For earnings, the agent must compare reported figures to
analyst-estimates consensus. "Beat" / "miss" is computed against
estimates, not against the prior period. A 5% earnings *growth* is
**bearish** if estimates expected 12%.

**Catalyst quota and floor.**
- Hard cap: at most **5 catalysts** AND at most **5 risks**. Forces
  prioritization.
- Required minimum: if the agent calls `bullish` or `bearish`, at
  least **one** catalyst must be present; otherwise the call must be
  `neutral`. (Enforced by both the prompt AND the schema validator.)

**Confidence calibration guidance.**
- 0.0–0.3: weak signal, mostly noise
- 0.4–0.6: moderate signal, multiple sources agree
- 0.7–0.85: strong signal, with clear catalyst
- 0.85+: avoid — reserved for "smoking gun" cases

**Hallucination guardrails (in the prompt itself).**
- Every catalyst / risk MUST quote a specific URL the agent actually
  fetched. Inventing sources is forbidden.
- If `fetch_news` returns zero articles, the agent must return
  `direction=neutral, magnitude=low, confidence=0.0` with empty
  catalyst/risk tuples and a summary that says so explicitly.
- If `fetch_article_body` returns `status=error` for all URLs, same
  zero-result behaviour applies to that bucket.

### 5.5 Resilient LLM chain

The news agent runs on a **resilient LLM router** — not a single
model. The router takes an ordered list of `(model_name, api_key)`
pairs and tries them in order. Same fallback semantics as the
resilient price fetcher:

- **Caller bugs** (bad prompt, schema mismatch) → raise immediately.
- **Provider failures** (rate limit, network, model overload) → mark
  the model in a 60-second cooldown, fall back to the next.
- **Schema-validation failures** on the LLM output → same as a
  provider failure; fall back. (The next model might produce valid
  JSON.)

The chain used by default (highest to lowest preference):
1. `gemini/gemini-2.5-flash`
2. `groq/llama-3.3-70b-versatile`
3. `openrouter/mistral-large` (or whatever's currently configured as
   tertiary)

The actual model that produced the assessment is recorded in the
prediction's `model_chain` field for audit.

### 5.6 What happens if the news agent fails entirely

Per §1.3 (degradation policy): we substitute a sentinel
`ImpactAssessment` with `direction=neutral`, `magnitude=low`,
`confidence=0.0`, `summary="news pipeline unavailable"`, and tag the
prediction's `model_chain` with `news_impact:degraded`. The prediction
still ships; consumers can filter on the tag.


---

## 6. The synthesizer LLM agent

The synthesizer is the agent that **actually emits the
`Prediction`**. It runs once per requested horizon (parallel
fan-out). Its inputs are:

- The horizon-agnostic **`TechnicalView`** (§4.5).
- The horizon-agnostic **`ImpactAssessment`** (§5.3).
- The **horizon label** (DAILY / WEEKLY / BIWEEKLY / MONTHLY).
- The current `as_of` timestamp (tz-aware IST).
- The current `close` price (anchors all the math).
- The current ATR value (the unit for stop / target distances).

Its output is a `Prediction` (§1.1). That output is then run through
four guardrail tiers (§7) and possibly retried once.

### 6.1 The per-horizon constants table — single source of truth

The canonical numbers live in code as
`prediction/horizon_constants.py` so every layer (synthesizer prompt,
guardrails, grading) reads the same values. The two tables below are
rendered FROM that file — if they ever drift, the code is authoritative.

#### Synthesizer-facing constants (per `horizon_constants.py`)

These are the bands the synthesizer prompt receives and the guardrails
enforce. **Source of truth: `horizon_constants.py`.**

| Horizon  | Stop ATR range | Target ATR range | Entry zone (±% close) | Confidence cap |
|----------|----------------|------------------|------------------------|----------------|
| DAILY    | 0.5–1.0 × ATR  | 0.75–1.5 × ATR   | ±0.5%                  | 0.90           |
| WEEKLY   | 0.7–1.5 × ATR  | 1.0–2.0 × ATR    | ±1.0%                  | 0.85           |
| BIWEEKLY | 1.0–2.0 × ATR  | 1.5–3.0 × ATR    | ±1.5%                  | 0.80           |
| MONTHLY  | 1.5–2.5 × ATR  | 2.0–4.0 × ATR    | ±2.0%                  | 0.75           |

#### Grading-facing constants (per grader / Mode-A thresholds)

These are the thresholds the grader uses when scoring a completed
prediction. They are SEPARATE from the synthesizer bands above — the
synthesizer is told a *hypothetical move band* via ATR, while the
grader cares about *actual realized return* in % terms.

| Horizon  | Bars | Mode-A win | Mode-A loss | Neutral cone | Target-tol |
|----------|-----:|------------|-------------|--------------|------------|
| DAILY    | 1    | +0.5%      | −0.5%       | ±0.5%        | 0.5%       |
| WEEKLY   | 5    | +1.5%      | −1.5%       | ±1.5%        | 0.5%       |
| BIWEEKLY | 10   | +2.5%      | −2.5%       | ±2.5%        | 0.5%       |
| MONTHLY  | 21   | +5.0%      | −5.0%       | ±5.0%        | 0.5%       |

**What each column means** (definitions referenced repeatedly below):

- **Bars** — count of trading-day bars in the grading window. Used by
  the backtest replay layer to step forward `Bars` bars from `as_of`
  and ask "what happened?".
- **Mode-A win / loss thresholds** — the *minimum directional move*
  required for a prediction to count as correct under the strict
  ("Mode A") grading rule. A bullish DAILY prediction needs the close
  to rise at least **+0.5%** between `as_of` and `target_datetime` to
  count as a hit. Below that, it's a miss.
- **Stop ATR range / Target ATR range** — the *(min, max) multiples
  of current ATR* that the synthesizer prompt allows for stop-loss
  distance and profit-target distance. Wider for longer horizons
  because longer horizons absorb more noise. The guardrail layer
  enforces these bands at validation time.
- **Entry zone (±% close)** — half-width of the entry band the
  synthesizer is allowed to construct around the current close.
  Tighter for daily, wider for monthly because a monthly call has
  more leeway in entry timing.
- **Target-tol** — tolerance (% of close) within which a
  near-miss target counts as "tagged". Used in grading variants only;
  not in the synthesizer's own math. Held constant across horizons in
  v1 (🔬 NEEDS BACKTEST whether it should scale).
- **Neutral cone** — the price-change band that defines "the stock
  effectively didn't move" for grading neutral predictions. A NEUTRAL
  WEEKLY prediction is correct iff the close moved by no more than
  ±1.5% over the 5 bars.
- **Confidence cap** — the synthesizer is told its emitted
  confidence MUST NOT exceed this value for that horizon. Longer
  horizons = more uncertainty = lower cap. This is **independent**
  of any cluster-classifier confidence cap in §4.

**Sources for the numbers above.**

- The 0.5 / 1.5 / 2.5 / 5.0 % win/loss thresholds and matching
  neutral cones are derived from typical NIFTY 50 daily volatility
  (ATR ≈ 1.0–1.5% of price). The threshold for a horizon equals
  approximately `√(bars) × 0.5%`, which mirrors the random-walk
  scaling of price moves with √time. The exact values are 🔬 NEEDS
  BACKTEST against the empirical NIFTY 50 distribution.
- The stop and target ATR ranges (`STOP_ATR_RANGE_BY_HORIZON`,
  `TARGET_ATR_RANGE_BY_HORIZON`) are literature-bracketed: Wilder
  (1978) 1×ATR canonical swing stop sits in WEEKLY band; Van Tharp
  (2007) 2–3×ATR positional sits in MONTHLY band; target midpoint
  ≈ 1.4 × stop midpoint per Murphy/Pring/Tharp consensus on positive
  expectancy (R:R > 1). Exact picks are 🔬 NEEDS BACKTEST.
  - **M10 explicit backtest variants.** The stop-ATR band is the
    single most load-bearing parameter for the system's win rate;
    Phase 2 backtest **must** test at minimum: (a) the current band,
    (b) a doubled band (1.0–2.0 daily … 3.0–5.0 monthly) — does
    win rate improve enough to justify the wider risk?, (c) a halved
    band (0.25–0.5 daily … 0.75–1.25 monthly) — does noise destroy
    win rate?, and (d) a flat ATR-multiplier across all horizons
    (e.g. constant 1.0×ATR) — does the per-horizon ladder structure
    actually matter, or is a single number good enough? If (d) is
    statistically indistinguishable from the ladder, simplifying
    removes a tunable knob.
- The confidence ceiling (0.90 → 0.75) reflects that further-out
  predictions are inherently harder to justify with high conviction.
  This is a design pick, not a literature value — to be recalibrated
  from realized hit-rate per horizon once enough graded predictions
  accumulate (Phase 2).

### 6.2 The synthesizer's system instruction — paraphrased

The full system prompt is long. Here is its operative content,
organized by what it tells the LLM to do.

**Role.** "You are a veteran NSE-focused trader writing a single,
actionable prediction for one stock at one time horizon. Your output
is a strict JSON object matching the `Prediction` schema."

**Inputs.** "You will be given (a) a `TechnicalView` summarizing
trend, momentum, volatility, and levels; (b) an `ImpactAssessment`
summarizing material news, filings, and estimates; (c) a horizon
label; (d) the current close price; (e) the current ATR. Use ALL of
these. Do not invent numbers."

**Direction call.** "Combine the technical and news views with the
following weighting:"

| Horizon  | Technical weight | News weight |
|----------|------------------|-------------|
| DAILY    | 35%              | 65%         |
| WEEKLY   | 50%              | 50%         |
| BIWEEKLY | 60%              | 40%         |
| MONTHLY  | 70%              | 30%         |

**Why this weighting.** Short-horizon (intraday/daily) price moves are
overwhelmingly news-driven. Longer-horizon moves are dominated by
trend regimes that show up in technicals. The exact weights are 🔬
NEEDS BACKTEST. They are explicit in the prompt so the LLM can't
silently pick its own.

**M8: the directional asymmetry is a hypothesis, not a fact.** The
shape of the table above (technicals dominate as horizon lengthens)
is itself a testable claim, not just the magnitudes. Counter-evidence
worth comparing against in Phase 2 backtest:
- **Tetlock (2007)**, *"Giving Content to Investor Sentiment: The
  Role of Media in the Stock Market"*, *J. Finance* 62(3) — finds
  newspaper sentiment predicts daily-horizon returns with subsequent
  mean-reversion. Implication: news may matter MORE at daily horizon
  than the table assumes, not less.
- **Da, Engelberg & Gao (2011)**, *"In Search of Attention"*,
  *J. Finance* 66(5) — Google search-attention measures predict
  next-2-week returns specifically. Implication: WEEKLY/BIWEEKLY
  news weight may be too low.

The asymmetric direction is defensible but not unique. Plausible
backtest variants: flat 50/50 across all horizons; or even
news-dominant at short horizons and tapering (e.g. 70/30 daily
news, 50/50 weekly, 30/70 monthly). Surface as a tunable knob,
not a buried constant.

**Disagreement protocol.** "If technical and news disagree:
- For DAILY/WEEKLY: news wins, but cap confidence at 0.55.
- For BIWEEKLY/MONTHLY: technical wins, but cap confidence at 0.60.
- If the disagreement is sharp (e.g. trend bullish, news strongly
  bearish), prefer NEUTRAL with confidence ≤ 0.50 rather than picking
  a side."

**Entry zone.** "Place the entry zone narrowly around current close,
as a symmetric ±% band of width `entry_zone_pct(horizon)` from
`horizon_constants.py` (§6.1, synthesizer-facing table). For BULLISH
you may bias the band slightly upward (close at the lower edge); for
BEARISH bias downward (close at the upper edge). For NEUTRAL keep the
band symmetric around close. The chosen `entry_zone[0]` (BEARISH) or
`entry_zone[1]` (BULLISH) is the *worst-fill* anchor for risk math
(see §1.1 "worst-fill RR" semantics)."

**Target placement.** "Set the target as follows, in this priority
order:
1. The **nearest level beyond the entry zone in the predicted
   direction** (from the levels cluster — swing high/low, R1/R2,
   S1/S2, 52w extreme). Use that level as the *anchor*.
2. If no such level exists within `4 × ATR`, use a pure ATR-based
   target inside that horizon's `TARGET_ATR_RANGE_BY_HORIZON` band
   (§6.1, synthesizer-facing table). The synthesizer picks any value
   in `[min, max] × ATR` and explains the choice.
3. Whichever method you used, **state the rationale** explicitly
   (e.g. 'target=1605 — anchored to swing-high resistance', or
   'target=1623 — 1.5×ATR projection inside WEEKLY band, no clear
   level above')."

**Stop placement.** "Set the stop as follows:
1. The **nearest level on the OPPOSITE side of the entry zone**, then
   verify the resulting stop distance lies inside that horizon's
   `STOP_ATR_RANGE_BY_HORIZON` band (§6.1). If the natural level is
   too tight, push the stop out to `stop_min × ATR`; if too wide,
   the prediction must downgrade to NEUTRAL (the structure can't
   accommodate a sane stop).
2. If no such level exists, use a pure ATR stop at the midpoint of
   the horizon's stop band.
3. State the rationale explicitly."

**Risk-reward floor.** "After computing target and stop, verify the
worst-fill risk-reward (per §1.1) is **≥ 1.5**. If it isn't:
- First, try moving the stop tighter (without leaving the horizon's
  `STOP_ATR_RANGE_BY_HORIZON` band). If that fixes RR, accept it.
- Otherwise, **switch the prediction to NEUTRAL** with a rationale
  explaining 'no asymmetric setup at current price'. Do NOT inflate
  the target to force RR."

**Confidence rules.** "Compute your confidence as follows:
1. Start with the *base* confidence = weighted average of (technical
   classifier confidences, news assessment confidence) using the
   horizon weights above.
2. Apply the disagreement caps if technical and news disagree.
3. Apply the **horizon ceiling** from §6.1 (DAILY/WEEKLY 0.85,
   BIWEEKLY 0.80, MONTHLY 0.75).
4. If you used the 'no level, ATR target' fallback, multiply by 0.85
   (less anchored = less confident).
5. Round to two decimal places."

**NEUTRAL semantics.** "Emit NEUTRAL when ANY of these are true:
- Technical and news sharply disagree.
- Risk-reward floor cannot be met.
- The price is mid-range (no nearby support OR resistance within
  `2 × ATR`).
- The volatility regime is `expanding` AND no clear directional
  catalyst exists.

For NEUTRAL predictions, the entry/target/stop fields are still
required, but they describe the *expected range*, not a directional
trade."

**Citation discipline.** "Every claim in the `rationale`,
`contributing_signals`, and `conflicting_signals` fields MUST be
traceable to either a specific signal in the `TechnicalView` (you may
quote the signal string verbatim) or a specific catalyst/risk in the
`ImpactAssessment` (you may quote it verbatim). Do NOT invent a
'recent management commentary' if it isn't in the news payload."

**Forbidden vocabulary.** "Do not use the words: `guarantee`,
`certain`, `definitely`, `risk-free`, `easy money`, `multibagger`. The
output is educational, not advice — `not_advice=True` and
`is_educational=True` are always set."

**JSON output discipline.** "Emit ONLY the JSON object that conforms
to the `Prediction` schema. No prose preamble, no markdown fences. The
schema is enforced by the framework — extra fields will cause hard
failure."

### 6.3 What the synthesizer does NOT see

For grounding (and to avoid the LLM second-guessing itself):
- The synthesizer is **not** told the previous prediction for the same
  ticker. Each prediction is independent.
- The synthesizer is **not** given a list of "stocks similar to this
  one." Single-ticker only.
- The synthesizer is **not** given any backtest outcome from prior
  predictions. We do not feedback-loop predictions into themselves.
- The synthesizer is **not** given the date of `target_datetime` —
  only the horizon label. The trading-calendar math happens after,
  in the schema's computed field.


---

## 7. Guardrails — the four-tier verification before a prediction ships

The synthesizer is an LLM and will occasionally produce predictions
that are wrong in mechanical ways: levels in the wrong order,
quoted-but-non-existent signals, RR violations, miscalibrated
confidence. The guardrail layer catches these deterministically
**after** synthesis but **before** persistence.

**Pipeline shape.** Each tier is a pure function:
`(Prediction, TechnicalView, ImpactAssessment) → GuardrailResult`,
where `GuardrailResult.passed` is bool and `.failures` is a tuple of
strings. Tiers run **in order**; the first failing tier short-circuits
the rest (no point checking calibration if the prediction's stops are
inverted).

**Retry policy.** If any tier fails, the synthesizer is invoked **once
more** with the failure messages appended to the user prompt as a
"Your previous attempt failed these checks: [...]. Fix them." The
retry uses the same `TechnicalView` and `ImpactAssessment` (we don't
re-run gathering — that would be wasteful and the inputs haven't
changed). If the retry **also** fails, the whole `predict()` call for
that horizon raises a `PredictionError` and the parallel fan-out
fails fast for the caller.

### 7.1 Tier 1 — Grounding

**Question:** "Are the prediction's basic numbers internally
consistent with their inputs?"

**Checks performed:**

1. **`close_price` echo.** `prediction.analysis_basis.close_price_at_prediction`
   must equal `technical_view.current_close` to within
   **0.01% of price** (allows for trivial floating-point
   differences). Catches the LLM hallucinating a different "current
   price" than what it was actually given.

2. **`bars_used` echo.** `analysis_basis.bars_used` must equal
   `technical_view.bars_used`. Catches the LLM under-reporting how
   much data it considered.

3. **Entry-zone width sanity.** The entry zone width
   (`zone_high − zone_low`) must be `≤ 0.5 × ATR`. The synthesizer
   prompt asks for `0.25 ATR` widths; we allow up to `0.5 ATR` as
   slack. Catches the LLM emitting a "zone" that's actually a 5%
   range (which would make worst-fill RR meaningless).

4. **Entry zone touches close.** The current close must lie within
   `[zone_low − 0.10 ATR, zone_high + 0.10 ATR]`. Catches the LLM
   placing the entry zone far from current price (which would mean
   the trade isn't actually entered now — a hallucinated setup).

5. **Direction-level sanity** (the schema validator already enforces
   the strict version; this re-checks defensively in case the schema
   was bypassed):
   - BULLISH: `target > zone_high` AND `stop < zone_high`.
   - BEARISH: `target < zone_low` AND `stop > zone_low`.
   - NEUTRAL: target/stop need not be ordered relative to entry zone.

6. **ATR-distance sanity for stop.** For BULLISH, the stop distance
   `(zone_high − stop)` must be `≥ 0.5 ATR` and `≤ 3.0 ATR`. Mirror
   for BEARISH. Catches both "stop too tight" (will get stopped by
   noise) and "stop too far" (RR can't possibly be met).

7. **NEUTRAL claims neutrality.** If `direction == NEUTRAL`, then the
   computed `risk_reward` must equal exactly `1.0` (the schema's
   convention). Catches schema bypass attempts.

### 7.2 Tier 2 — Citation

**Question:** "Did the LLM cite signals it actually saw — or did it
make them up?"

**Vocabulary construction.** Before checking, we build the **allowed
vocabulary** for this prediction by concatenating:

- Every `signal` string from each cluster classifier (trend, momentum,
  volatility, levels) — these are the verbatim strings the LLM was
  told it could quote.
- Every candlestick pattern's `name` field (e.g. `"hammer"`,
  `"bullish_engulfing"`).
- Every chart pattern's `pattern` field (e.g. `"head_and_shoulders"`,
  `"ascending_triangle"`).
- Every catalyst's `title` and `source` fields from the news
  assessment.
- Every risk's `title` and `source` fields.
- Indicator names that appear in cluster signals (`"RSI"`, `"MACD"`,
  `"ADX"`, `"ATR"`, `"BB"`, `"%B"`, `"OBV"`, `"SMA"`, `"EMA"`,
  `"+DI"`, `"−DI"`, `"%K"`, `"%D"`).

After lower-casing and stop-word removal, this becomes a `set[str]`
of "tokens the LLM is allowed to invoke."

**Stop-word list** (filtered out before vocabulary check):
`the, a, an, of, in, on, at, to, for, with, and, or, but, is, are,
was, were, be, been, being, has, have, had, this, that, these, those,
near, above, below, over, under, by, from, as, it, its.`

**Checks performed:**

1. **Rationale token grounding.** Tokenize `prediction.rationale`
   into lowercase words. For tokens that look like *meaningful*
   technical terms (length ≥ 3, not in stop-word list), at least
   **80%** must appear in the allowed vocabulary. Catches the LLM
   citing "Fibonacci retracement" when no Fibonacci tool was run, or
   "moving average crossover" if no such cross was reported.

2. **`contributing_signals` strict match.** Each entry must be either
   (a) **substring-equal** to some allowed-vocabulary phrase, OR
   (b) a slight rewording where ≥ **70%** of its non-stop-word tokens
   are in the vocabulary. Catches the LLM inventing supporting
   signals like `"strong dividend history"` (a fundamental, not in
   any of our cluster outputs).

3. **`conflicting_signals` same rule.** Symmetric; same threshold.

4. **Catalyst URL anchoring.** Any URL appearing in the rationale or
   signals must match a URL in `impact_assessment.catalysts +
   impact_assessment.risks`. Catches the LLM hallucinating a URL.

5. **Forbidden vocabulary.** The rationale must contain none of:
   `guarantee, certain, definitely, risk-free, easy money,
   multibagger`. Case-insensitive substring match.

🔬 The 80% / 70% thresholds are starting points; **NEEDS BACKTEST**
once we have a corpus of human-graded predictions.

**M9: specific concerns to test against.**

1. **80% rationale-token grounding is very strict.** A 5-token noun
   phrase needs ~4 tokens to come from vocabulary. That's
   "no-creativity" territory — the LLM cannot rephrase "ADX 32" as
   "strong ADX reading" because "reading" isn't in the vocab. Suggest
   starting at **60%** and tightening with data, rather than starting
   strict and loosening (false negatives are silent; false positives
   are loud).
2. **The OR-of-two-checks (substring-equal OR ≥70% tokens) is hard
   to reason about.** When `contributing_signals` includes
   `"strong ADX trend with bullish DI cross"`, did it pass because
   of substring match on "ADX" or because 4/5 tokens overlap? The
   logs don't say. Suggest collapsing both into a single token-overlap
   ratio (e.g. ≥60%) for cleaner explainability; the substring
   special-case is a strict subset of the token-overlap condition
   anyway.

### 7.3 Tier 3 — Consistency

**Question:** "Does the prediction internally hang together?"

**Checks performed:**

1. **Direction ↔ rationale agreement.** The rationale must contain
   *some* keyword consistent with the declared direction:
   - BULLISH must mention at least one of: `up, rise, rally, breakout,
     buy, long, support holding, momentum positive, trend up`.
   - BEARISH must mention at least one of: `down, fall, decline,
     breakdown, sell, short, resistance holding, momentum negative,
     trend down`.
   - NEUTRAL must mention at least one of: `range, sideways,
     consolidating, no setup, wait, neutral, choppy`.

   Catches the LLM emitting `direction=BULLISH` and a rationale that
   exclusively talks about bearish signals.

2. **Confidence ↔ disagreement consistency.** If
   `len(conflicting_signals) > len(contributing_signals)` (i.e. more
   things point against than for), confidence must be `≤ 0.55`.
   Catches "high-conviction" calls that explicitly enumerate more
   conflicts than supports.

3. **Confidence ↔ news magnitude consistency.** If
   `impact_assessment.magnitude == "high"` AND
   `impact_assessment.confidence ≥ 0.7` AND the synthesizer's
   `direction` matches `impact_assessment.direction`, the
   synthesizer's confidence must be `≥ 0.55` (it can't claim it
   ignored a clear strong-news signal that aligned with its call).

4. **NEUTRAL when sharply split.** If technical classifier verdicts
   are mixed (≥ 2 bullish AND ≥ 2 bearish across the four clusters),
   AND news direction conflicts with the synthesizer's call, the
   synthesizer must emit NEUTRAL. Catches "I'll just pick a side"
   behaviour.

5. **NEUTRAL doesn't claim a directional edge.** If
   `direction == NEUTRAL`, neither `contributing_signals` nor the
   rationale may claim "buy" / "sell" / "long" / "short" anywhere.

### 7.4 Tier 4 — Calibration

**Question:** "Is the confidence number reasonable given the
prediction's structure?"

**Checks performed:**

1. **Horizon ceiling enforcement.** `confidence ≤ horizon_ceiling`
   from §6.1 (DAILY/WEEKLY 0.85, BIWEEKLY 0.80, MONTHLY 0.75).
   Hard fail if violated.

2. **Risk-reward floor.** `risk_reward ≥ 1.5` for BULLISH/BEARISH.
   NEUTRAL bypasses this (RR = 1.0 by convention).

3. **NEUTRAL confidence ceiling.** `direction == NEUTRAL` ⟹
   `confidence ≤ 0.65`. Neutral is a "no edge" call; claiming
   high confidence in "no edge" is a contradiction.

4. **Fallback-target confidence reduction.** If the rationale or
   target's `rationale` field contains the substring `"projection"`
   or `"ATR projection"` (the sentinel for "no level was available, I
   used the ATR fallback per §6.2"), confidence must be `≤ 0.70`.
   Mirrors the synthesizer's "multiply by 0.85" rule, but as a
   ceiling.

5. **Confidence bounded by news confidence on news-driven horizons.**
   For DAILY (news weight 65%), the synthesizer's confidence may not
   exceed `news_assessment.confidence + 0.10`. For WEEKLY (50/50),
   `+0.15`. Catches the LLM emitting `confidence=0.85` on a DAILY
   call when the news layer reported `confidence=0.30`.

6. **Degraded news caps confidence.** If the prediction's
   `model_chain` contains `news_impact:degraded`, confidence must be
   `≤ 0.65` for any horizon. Catches "the news system was down but
   I'm still 80% sure."

### 7.5 What happens if a guardrail fails on the retry

If after the single retry **any tier** still fails, we raise a
`PredictionError` whose message lists the failing tier(s) and their
reasons. The error propagates through `asyncio.gather` and aborts the
whole `predict()` call (per §1.3 fail-fast policy).

**Why one retry, not many.** Empirically, when the first attempt
violates a guardrail, the second attempt with the failure message in
the prompt usually succeeds. Beyond one retry we've seen
diminishing returns and a tendency for the LLM to "play whack-a-mole"
with the rules rather than re-think the prediction. One retry is the
cost-vs-quality sweet spot.

🔬 The "one retry" policy itself is **NEEDS BACKTEST**: we may revisit
it with retry-twice or retry-zero variants once we have data.


---

## 8. Persistence and grading

### 8.1 Why we persist every prediction

Three reasons:

1. **Audit.** "Why did the agent think this on April 12?" should be
   answerable from disk, not from re-running the model (which would
   give a different answer because markets moved).
2. **Calibration.** "Across the last 90 days of bullish DAILY calls
   on RELIANCE, what was the actual hit rate?" requires a corpus.
3. **Backtesting.** Step 3.5.5 (replay) walks history forward day-by-
   day; without persistence we can't reconstruct what the agent would
   have said on a given day.

### 8.2 Storage layout — `PredictionStore`

- **One file per prediction.** Predictions are immutable (frozen
  Pydantic models), so a write-only append-store fits naturally.
  Benefits: atomic writes, easy to grep/cat/diff, easy to gzip old
  days, no DB to install or break, trivially parallel-safe (different
  files = no contention).
- **Layout:** `{root}/{YYYY-MM-DD}/{TICKER}_{HHMMSS}_{horizon}.json`.
  - Day directory enables easy archival
    (`tar czf 2026-04.tar.gz 2026-04-*`).
  - HHMMSS preserves intra-day ordering and supports multiple
    predictions per ticker per day (different horizons or re-runs).
  - Horizon in the filename allows quick filtering without opening
    files.
- **Filename sanitization.** Ticker may contain `.` (RELIANCE.NS) but
  must not contain `/` or `..`. The store uppercases and strips any
  character outside `[A-Z0-9.-]`. Empty result raises (caller passed
  garbage).
- **Atomic writes.** Save writes to a `.tmp` file in the same
  directory then `os.replace()` onto the final path. POSIX guarantees
  rename atomicity, so we never see a half-written file even on crash.
- **Round-trip contract.** Every prediction must satisfy
  `p == Prediction.model_validate_json(p.model_dump_json())`. Enforced
  by tests; any change that breaks round-trip is a breaking change to
  consumers (logs, UIs, backtest replay).
- **`@computed_field` handling.** The schema's computed fields
  (`risk_reward`, `midpoint_rr`, `target_datetime`) are emitted in
  the JSON dump (audit trail) but stripped before re-validation (so
  `extra=forbid` doesn't reject them on round-trip).

### 8.3 The grading question

For one stored prediction, given the **actual OHLCV bars from `as_of`
to `target_datetime + 1 day`**, the grader answers two questions:

1. **Did the prediction hit?** Bool, with caveats (see grading modes
   below).
2. **What's the realized return?** A single percent number describing
   what the trade would have made.

Both depend on (a) the direction the prediction took, (b) the
horizon's win/loss/neutral-cone thresholds from §6.1, and (c) the
intra-window price path (high, low, close per bar).

### 8.4 Grading modes

We compute **three** parallel verdicts per prediction. Different
downstream consumers care about different definitions of "hit", and
shipping all three lets us measure them all without losing
information.

#### Mode A: strict directional move

The prediction is correct iff the realized return between `as_of`
close and `target_datetime` close meets the **horizon's
win/loss threshold** from §6.1:

- BULLISH: `(target_close − as_of_close) / as_of_close × 100 ≥ +threshold`
- BEARISH: `(target_close − as_of_close) / as_of_close × 100 ≤ −threshold`
- NEUTRAL: `|(target_close − as_of_close) / as_of_close × 100| ≤ neutral_cone`

This is the **headline** metric. It rewards predictions that called
the right direction *and* magnitude, and penalises trades that "kind
of went up a little" when the prediction said BULLISH on a monthly
horizon.

#### Mode B: target/stop resolution (intra-window)

Walk the OHLCV bars from `as_of+1` to `target_datetime`. Track which
of `target` or `stop` is touched first by the *high* (for bullish)
or *low* (for bearish):

- **`TARGET_HIT`** — target was touched at least once by intra-bar
  high (bullish) or low (bearish). Realized = `(target − entry) /
  entry × 100`, signed for direction.
- **`STOP_HIT`** — stop was touched first. Realized = `(stop − entry)
  / entry × 100`, signed.
- **`STOP_HIT_AMBIGUOUS`** — on the same bar, both target and stop
  were touched. We CANNOT know from daily bars whether stop was hit
  before target; **we conservatively count this as a stop hit**
  (worst-case-for-trader assumption — same spirit as worst-fill RR).
- **`OPEN`** — neither touched by `target_datetime`. Realized =
  `(target_close − entry) / entry × 100`. The trade is "open" at
  evaluation; we mark-to-market.

The grading object stores the **bar index** at which the resolution
happened (or `None` for `OPEN`), so consumers can compute
"average bars to target" type metrics.

#### Mode C: neutral-cone for NEUTRAL predictions

NEUTRAL predictions don't have a meaningful target/stop, so Mode B
returns `None`. Mode C is the canonical evaluation:

- The prediction is correct iff `|realized_return| ≤ neutral_cone`
  for the horizon (§6.1).
- The grader emits both the bool verdict and the realized return.

### 8.5 The metrics the calibration layer computes

Given a corpus of graded predictions (filterable by ticker, horizon,
direction, date range), the calibration layer computes:

- **Hit rate (Mode A).** Fraction of predictions correct under Mode
  A's strict thresholds.
- **Hit rate (Mode B).** Among bullish/bearish predictions, the
  fraction where Mode B's resolution is `TARGET_HIT` (excludes
  `OPEN`). Reported alongside a separate "open rate" so the reader
  knows how much of the corpus is unresolved.
- **Hit rate (Mode C).** Among neutral predictions, the fraction
  inside the cone.
- **Mean realized return.** Average of Mode-B realized returns,
  weighted equally per prediction. (A naïve sum, not a compounded one
  — predictions are not assumed to be sequenced.)
- **Mean confidence.** Average of `prediction.confidence`. Useful as
  a sanity check vs hit rate (a well-calibrated agent's mean
  confidence should track its hit rate within ~10 percentage
  points).
- **Brier score.** `mean((confidence − outcome)²)` where outcome is
  1.0 if Mode-A correct, else 0.0. Lower is better; 0.25 is the
  random baseline (always-predict-0.5). Below 0.20 is "decent",
  below 0.15 is "good".
  - **M11 caveat.** 0.25 is the worst-case random baseline (highest
    variance for a 50/50 outcome prior). The "always predict the
    base rate" baseline is `p × (1 − p)` where `p` is the empirical
    Mode-A hit rate. So if our hit rate ends up at 0.40 the naive
    baseline is 0.24, not 0.25; if it ends up at 0.30 the baseline
    drops to 0.21. Comparing raw Brier to a fixed 0.25 will look
    falsely impressive at base rates far from 0.50.
  - **Use Brier-skill score (BSS) for the apples-to-apples
    comparison:** `BSS = 1 − BS / BS_baseline` where `BS_baseline`
    uses the empirical base rate `p × (1 − p)`. `BSS > 0` → real
    skill above the base rate; `BSS = 0` → no skill (equivalent to
    just guessing the base rate every time); `BSS < 0` → worse than
    guessing the base rate. Report both raw BS and BSS in the
    calibration dashboard so future readers don't anchor on the
    misleading 0.25 number.
- **Calibration slices / breakdowns.** All metrics above are also
  computed per breakdown dimension: per-ticker, per-horizon,
  per-direction, per-confidence-bucket (deciles 0.0–0.1, 0.1–0.2,
  …). The per-confidence-bucket breakdown is the most diagnostic:
  predictions emitted at confidence 0.8 should hit ~80% of the time
  in a well-calibrated system.

### 8.6 Backtest harness — Steps 3.5.5–3.5.7 (not yet built)

Three layers planned, in order:

1. **`backtest/replay.py`** — an "as-of-date data shim". Given
   `as_of`, returns OHLCV slices that contain only data available on
   or before that date. Wraps the existing data layer so the
   prediction pipeline runs unmodified. Critical for avoiding
   look-ahead bias.
2. **`backtest/runner.py`** — a historical loop. For each `as_of` in
   a date range and each ticker in a watchlist, runs `predict()`
   against the as-of-date shim, persists the predictions to a
   parallel store directory (so backtest predictions don't co-mingle
   with live ones), and tags them with `model_chain += ["backtest"]`.
3. **`backtest/evaluator.py`** — composes the existing grading and
   calibration layers over the stored backtest predictions. Outputs
   the same metrics as §8.5, partitioned by experiment configuration
   (which model chain, which sensitivity preset, etc.) for
   apples-to-apples comparison.

Until all three exist, the constants in §6.1 and the thresholds in §7
are 🔬 **NEEDS BACKTEST.** They are defensible from literature (see
"Sources for the numbers" in §6.1) but not yet empirically validated
on the actual NIFTY 50 distribution.


---

## 9. What we deliberately do NOT do (and why)

A vetting reviewer should **not** flag the following as gaps; they are
explicit design decisions, not omissions.

| Not done | Why |
|----------|-----|
| Implied volatility / options chains | Adds a whole new data source + pipeline. ATR is a usable proxy for v1. Re-evaluate in Phase 2. |
| Macro indicators (VIX, USD/INR, repo rate) | Single-stock predictions over short horizons are dominated by stock-specific catalysts. Macro adds noise more often than signal at this horizon scale. |
| Sector / index relative strength | Same reason. Ranking dimension that adds complexity without proportional value at v1. |
| Direct fundamentals (P/E, ROE, debt) | We assume the news/filings layer captures *material* changes in fundamentals via earnings results and guidance. Pure level-of-fundamentals ratios don't help short-horizon prediction. |
| Multi-target ladders / partial-exit logic | Adds significant schema complexity. Single target is a defensible v1 contract; ladder upgrade is non-breaking on a frozen Pydantic model. |
| Position sizing | Outside scope. We output a *level*, not a *size*. Sizing is the user's risk-budget call. |
| Intraday tick / minute data | Daily bars only. Minute data triples our storage and adds little for ≥1-day horizons. |
| Same-bar target/stop disambiguation | Daily bars don't tell us which level was hit first. We conservatively count it as a stop hit (worst-case). Resolving this would require minute data. |
| Six-month / yearly horizons | Calibration becomes meaningless when every prediction sits in its own bucket; would need years of history before any metric stabilises. Parked. |
| Custom-duration horizons (e.g. "9 days") | Same calibration reason — schema enforces the four canonical horizons only. |
| Sentiment from GDELT's `tone` field | We do our own LLM reading; not trusting GDELT's pre-computed sentiment. |
| Historical analyst-estimates time series | yfinance only gives a current snapshot. Historical estimates would let us detect "estimates raised this week" — a real catalyst type. Open issue, but not blocking v1. |
| Cross-prediction feedback / RL | The synthesizer never sees previous predictions or their outcomes. We do not feedback-loop. This keeps each prediction independently auditable. |
| Aggregate "consensus across horizons" | Each horizon is independent. We do not enforce that DAILY and MONTHLY agree; if they disagree, that's information for the user, not a bug. |

---

## 10. Open questions a reviewer should weigh in on

Specific, actionable items where a domain expert's pushback would
most improve v1. Not "general feedback welcome" — concrete decisions
where we want a second opinion.

### 10.1 Are the per-horizon thresholds defensible?

Reference §6.1. We chose:
- DAILY ±0.5%, WEEKLY ±1.5%, BIWEEKLY ±2.5%, MONTHLY ±5%.

These follow approximately `√(bars) × 0.5%` (random-walk scaling) and
are anchored at NIFTY 50's typical 1–1.5% daily ATR.

**Question for the reviewer:** does this match your intuition for
what a "real" weekly/monthly directional move looks like on NSE
large-caps? Should the monthly threshold be tighter (3%) or wider
(7%)? Should it be **stock-specific** (anchored to each stock's own
trailing ATR percentile) instead of one-size-fits-all?

### 10.2 Are the news weights right for short horizons?

Reference §6.2 weighting table:
- DAILY: 35% technical / 65% news.
- MONTHLY: 70% technical / 30% news.

**Question for the reviewer:** is news really 65% of a daily move, or
are we over-weighting it because it's "what's interesting today"?
Some traders argue daily moves are 80%+ news-driven for retail-heavy
stocks and 50/50 for institutional names. Are sector-dependent weights
worth the complexity?

### 10.3 Is the 1.5 RR floor too generous or too strict?

Reference §6.2 risk-reward floor.

**Question for the reviewer:** in your trading practice, what RR
floor causes a setup to be "worth taking" vs "skip it"? Pro literature
ranges from 1.0 (high-frequency) to 3.0 (low-frequency, swing). 1.5
is a defensible middle for daily/weekly retail timeframes; for
monthly positional swings, should we raise it to 2.0?

### 10.4 Is the candlestick gating threshold (1×ATR proximity) right?

Reference §3.6 context gating.

**Question for the reviewer:** does "within 1 ATR of swing extreme"
match your definition of "near support/resistance"? Some practitioners
use 0.5 ATR (stricter, fewer patterns surface) or 2 ATR (more
patterns, more noise). Without a backtest we have no way to choose
empirically. Your gut?

### 10.5 Are we missing a critical pattern or indicator?

Reference §3 + §3.7. What we have:
- Indicators: SMA, EMA, ADX/DI, RSI, MACD, Stoch, OBV, ATR, BB.
- Levels: swing high/low, 52w high/low, classic floor pivots.
- Candlesticks: 7 patterns (doji, hammer, shooting star, both
  engulfings, both stars).
- Chart patterns: 5 (double top/bottom, H&S + inverse, triangles).

**Question for the reviewer:** is there a classical TA tool you'd
consider table-stakes that's missing? Common candidates we
deliberately skipped: Ichimoku Cloud, Fibonacci retracements, Volume
Profile, VWAP. (We skipped them for v1 to avoid scope blow-up; happy
to be talked into adding one.)

### 10.6 Is the citation guardrail's 80% threshold too strict?

Reference §7.2.

**Question for the reviewer:** the 80% rationale-token grounding
threshold means a meaningful 5-token noun phrase needs ~4 of its
tokens in our vocabulary. That's strict — borderline rejecting any
LLM creativity. Is that the right tradeoff? Or should we relax to
60% and rely on Tier 3 (consistency) to catch genuinely off-topic
predictions?

### 10.7 Is "fail fast on any horizon failure" correct?

Reference §1.2 + §7.5.

**Question for the reviewer:** if MONTHLY synthesis fails but DAILY
and WEEKLY succeed, we currently abort the whole call. Alternative
designs:
- Return partial results with a `failed_horizons` field so users see
  what we got.
- Retry the failed horizon while serving partial results in the
  meantime.

The current "all-or-nothing" stance is conservative (no half-truths
to consumers) but maybe annoying in production. Worth changing?

### 10.8 Should grading downgrade `STOP_HIT_AMBIGUOUS` to a half-loss?

Reference §8.4 Mode B.

**Question for the reviewer:** when daily bars show both target and
stop touched on the same day, we count it as a full stop loss. A
softer alternative: count it as a half-loss (`realized = (stop −
entry)/entry / 2`). Or split the metric into "ambiguous rate" so it's
visible separately. Your call?

---

## Appendix A — File-and-module map (for reviewers who want to drill into code)

If a reviewer wants to verify a specific claim against code, this is
where each component lives:

```
src/price_predictor/
├── data/
│   ├── providers/
│   │   ├── base.py            # PriceProvider protocol, PriceFetchError
│   │   ├── yfinance.py        # primary
│   │   ├── stooq.py           # fallback 1
│   │   └── alpha_vantage.py   # fallback 2
│   ├── providers/resilient.py # ResilientPriceFetcher (§2.1)
│   ├── cache.py               # PriceCache (§2.2)
│   ├── news.py                # GDELT + trafilatura (§2.3)
│   ├── filings.py             # NSE corporate announcements (§2.4)
│   └── estimates.py           # yfinance analyst estimates (§2.5)
├── analysis/
│   ├── indicators/
│   │   ├── trend.py           # SMA, EMA, ADX (§3.2)
│   │   ├── momentum.py        # RSI, MACD, Stoch, OBV (§3.3)
│   │   ├── volatility.py      # ATR, BB, bollinger_squeeze + ttm_squeeze (§3.4)
│   │   └── levels.py          # swings, 52w, pivots (§3.5)
│   ├── candlestick.py         # 7 patterns + gating (§3.6)
│   ├── chart.py               # 5 chart patterns (§3.7)
│   └── classifiers.py         # 4 cluster classifiers (§4)
├── news_impact/
│   ├── agent.py               # Pydantic-AI sub-agent (§5)
│   ├── tools.py               # 4 tools the agent can call
│   └── schema.py              # ImpactAssessment, Catalyst, Risk
├── synthesis/
│   ├── agent.py               # the synthesizer (§6)
│   ├── prompt.py              # system instruction (§6.2)
│   └── horizon_constants.py   # the §6.1 table — single source of truth
├── prediction/
│   ├── schema.py              # Prediction Pydantic model (§1.1)
│   ├── store.py               # PredictionStore (§8.2)
│   └── trading_calendar.py    # NSE calendar + target_datetime (§1.1)
├── guardrails/
│   ├── grounding.py           # Tier 1 (§7.1)
│   ├── citation.py            # Tier 2 (§7.2)
│   ├── consistency.py         # Tier 3 (§7.3)
│   └── calibration.py         # Tier 4 (§7.4)
├── grading/
│   ├── grade_one.py           # Mode A/B/C for one prediction (§8.4)
│   └── calibration.py         # corpus-level metrics (§8.5)
└── orchestrator.py            # predict() — gather + fan-out (§1.2)
```

---

## Appendix B — How to vet a specific prediction by hand

A reviewer who wants to spot-check a single prediction should:

1. **Open the prediction JSON** at
   `predictions/YYYY-MM-DD/TICKER_HHMMSS_horizon.json`.
2. **Check `analysis_basis`** — does `close_price_at_prediction`
   match what the stock actually closed at on `as_of`? (Tier 1, §7.1.)
3. **Check direction-level math.** For BULLISH: `target.value >
   entry_zone[1]` AND `stop_loss.value < entry_zone[1]`. (Schema +
   Tier 1.)
4. **Check `risk_reward`** — recompute by hand using the worst-fill
   formula in §1.1. Should be ≥ 1.5.
5. **Check `confidence`** — should not exceed the horizon ceiling
   (§6.1), should not exceed `news_assessment.confidence + 0.10/0.15`
   for daily/weekly.
6. **Check `contributing_signals`** — do the strings *look like*
   they could have come from one of the cluster classifiers in §4?
   (No "Elliott wave", no "Fibonacci" — we don't ship those.
   "Golden Cross" / "Death Cross" / "bullish EMA-9/21 cross" ARE
   valid as of §3.2 MA Crossover, but only if the `ma_crosses` field
   in the trend tool output reports them. **Ichimoku** cloud signals
   ARE now valid — see the Addendum below — but only if the
   `ichimoku` block in the trend tool's `derived` output reports
   them.)
7. **Check `catalysts`** — every URL should be visit-able and the
   article should plausibly say what `why_it_matters` claims it
   says.
8. **For graded predictions, check the verdict against the bars** —
   pull the OHLCV from `as_of` to `target_datetime`, compute the
   close-to-close return, and compare against the §6.1 thresholds.

If any of those by-hand checks disagrees with the stored values, file
a bug — that's exactly the kind of grounding error this whole
architecture is designed to prevent.

---

## Addendum — indicators added after the original review (2026-07)

The following were speced in `pred_logic_solutions.md` and shipped after
this walkthrough was first written. Full literature, formulas and
citations live in that companion doc; this is the short vetting summary.

### A1. Ichimoku Kinko Hyo cloud (solutions §H9b)

- **Module:** `analysis/ichimoku.py`. **Params:** Hosoda canonical
  (tenkan 9, kijun 26, senkou-B 52, displacement 26).
- **What the snapshot reports:** the five lines plus three regime
  signals — `price_vs_cloud` (above / below / inside), `tk_signal`
  (tenkan vs kijun), and `kumo_twist_ahead` (future Senkou A/B sign
  change).
- **Wiring:** surfaced additively under the `get_trend` tool's
  `derived.ichimoku`. It is exposed to the LLM as *data*; the
  deterministic `classify_trend()` scoring is intentionally NOT changed
  (that stays a tracked follow-up), so an Ichimoku signal only appears
  in `contributing_signals` if the LLM chose to cite the reported block.
- **Vet:** any "above the cloud" / "tenkan-kijun cross" claim must match
  the `derived.ichimoku` values in the tool output for that run.

### A2. India VIX regime gate (solutions §H9d)

- **Modules:** `analysis/vix.py` (pure regime math) + `data/vix.py`
  (fetches `^INDIAVIX` via yfinance — NOT the GPL `nsepython` the
  solutions doc originally suggested).
- **What it reports:** `low_vol` / `normal` / `high_vol` / `unknown`,
  self-calibrated against the 60-day rolling median (±15% bands) rather
  than magic absolute levels.
- **Role:** a **regime gate**, not a directional signal — VIX describes
  the weather (position-sizing / stop-width context), never the
  direction. Treat any "VIX says go long" claim as a bug.

### A3. Point-in-time article fetcher (solutions § PIT / look-ahead)

- **Module:** `data/wayback.py` (Wayback CDX Server API via `httpx` +
  `trafilatura` — NOT `waybackpy`).
- **Hard guarantee:** `article_body_pit(url, asof)` never returns a
  snapshot captured after `asof`, and never silently falls back to the
  live URL (that would reintroduce look-ahead bias). No PIT snapshot =>
  `None`, and the caller drops the observation.
- **Vet:** in a backtest, every catalyst body used for an `as_of` date
  must trace to a Wayback capture timestamp ≤ that date.

