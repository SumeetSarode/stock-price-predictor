# Critical Review of `pred_logic.md`

**Reviewer:** Code Puppy 🐶 (acting as a strict TA/quant/eng instructor)
**Date:** 2026-04-30
**Document under review:** `pred_logic.md` (1,730 lines, 76.1 KB) — "Prediction Logic — A Vettable Walkthrough"
**Style of review:** Pedantic. Every numeric claim, attribution, year, formula, and threshold has been independently verified against primary literature, exchange documentation, vendor docs, and code-of-record.

---

## 0. Executive summary

The document is **well-written, well-structured, and intellectually honest** about its uncertainties (the 🔬 NEEDS BACKTEST tags are exactly the right discipline). The two-phase architecture is clean, the guardrail tiers are sensible, and the attempt to ground every claim in literature is laudable.

**However**, on careful, source-by-source verification, I found:

| Severity | Count | Examples |
|----------|------:|----------|
| 🔴 **Critical** (factual error, broken pipeline, or breaks user contract) | 7 | Stooq doesn't cover NSE; GDELT "7-day index" is wrong (it's 9 years); LMW only 6 of 10 patterns implemented; NSE filings API is *not* public; 252 trading days is wrong for India; 22-bar rule mis-attributed; Edwards & Magee mis-cited |
| 🟠 **High** (defensible but materially incomplete) | 9 | Only 7 of ~25 Nison patterns (no continuation patterns at all); BB Squeeze definition non-standard; Bulkowski says 5% trough too shallow; missing R3/S3 from "classic" pivots; missing major indicators (Ichimoku, VWAP, Volume Profile); hammer ratios are not Nison's; ADX threshold mis-stated; RSI 60/40 not original; MACD warmup not specified |
| 🟡 **Medium** (imprecision, terminology, or loose attribution) | 11 | "Bollinger 1980s" vague; "Appel 1979" loose; OBV credits Granville without Woods/Vignola predecessors; "worst-case RR" non-standard term; Wilder smoothing convergence guard insufficient; engulfing definition has equality-edge ambiguity; LMW uses kernel-regression not least-squares for pivots; ATR-percentile threshold for "expanding" undocumented; news weights have no source; calibration thresholds 80%/70% arbitrary; stop-pad ladder unsourced |
| 🟢 **Nits** | many | Spelling/style/citation pedantry — not enumerated except where they matter |

**Net call:** The document needs **one major revision pass before it can claim to be "vettable against literature"** as the title promises. The data-layer pipeline (§2.1) has at least one fallback that **silently does nothing** (Stooq for NSE), which is worse than not having a fallback at all. Until §2.1, §2.3, §3.6, §3.7, and §6.1 are corrected, downstream sections cannot be trusted.

---

## 1. Critical issues — must fix before shipping

### 🔴 C1. Stooq has no coverage of NSE Indian equities (§2.1)

**Doc says:** Sources in fallback order are (1) yfinance, (2) **Stooq** (first fallback — different upstream than yfinance), (3) Alpha Vantage.

**Reality:** Stooq does not list Indian (NSE/BSE) equities. I tested directly — both `RELIANCE.IN` and `RELIANCE.BO` return *"Symbol does not exist in database"*. Stooq's coverage is US, European (LSE / Xetra / Euronext), Japanese, HKEx, Polish (GPW), funds, indices, commodities, FX, macro — **no India equities**.

**Impact:** When yfinance fails, the resilient fetcher will hit Stooq, get an empty / not-found response for every NSE ticker, mark Stooq cooled-down, and *then* try Alpha Vantage. Net effect: Stooq is dead weight that adds latency on every primary failure and gives a false sense of "we have a fallback."

**Recommended fix:** Replace Stooq with something that actually covers NSE. Realistic options:
- **`jugaad-data`** (https://github.com/jugaad-py/jugaad-data) — actively maintained Python lib for NSE/BSE; the de-facto successor to the deprecated `nsepy`.
- **Zerodha Kite Connect** (https://kite.trade/) — official paid API (₹2,000/mo per app), best for production.
- **Upstox API** (https://upstox.com/developer/api/) — free with a Upstox account.
- **Direct from NSE bhavcopy** as cold-storage fallback (slower, but free and exchange-of-record).

**Source:** Stooq category list confirmed at https://stooq.com/db/d/

---

### 🔴 C2. GDELT Doc API 2.0 is **not** a 7-day rolling index (§2.3)

**Doc says:**
> "GDELT's rolling 7-day index. We typically request 1–7 days back; the LLM may request up to 90 days for special cases."

**Reality:** GDELT Doc 2.0 indexes news **back to 2017-01-01**. The "7-day window" claim is flat wrong. The default if no `STARTDATETIME` is supplied is the last 3 months. You can request any window in `[2017-01-01, now-15min]`.

**Impact:** Either (a) the doc is misinforming reviewers about a real limitation that doesn't exist, costing capability — the news agent could pull historical baselines, longer-window catalysts, or cross-reference patterns; or (b) the code is artificially clamping to 7 days and silently dropping older articles. Both are bad.

**Recommended fix:**
> "GDELT Doc 2.0 indexes from 2017-01-01 onward, updated every ~15 minutes. We request 1–7 days back by default and allow the agent to request up to 90 days for special cases (earnings season, M&A windows)."

**Source:** GDELT Blog announcement of Doc 2.0 — https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/

---

### 🔴 C3. NSE corporate-announcements is **not** a "public feed" (§2.4)

**Doc says:**
> "NSE's public corporate-announcements feed (the same endpoint NSE's website uses for 'Corporate Information → Announcements')."

**Reality:**
- The endpoint (`https://www.nseindia.com/api/corporate-announcements?index=equities`) is an **internal, undocumented** JSON endpoint backing the website. There is no published terms-of-use or developer portal entry.
- It is **Cloudflare-protected** — bare `requests.get(...)` with the default Python User-Agent gets blocked.
- It requires a **prior session-priming GET** to `https://www.nseindia.com/` to obtain cookies before the API call works.
- It rate-limits aggressively and has known blocking issues; production use needs exponential backoff, IP rotation, or a paid licensed feed.
- The endpoint has **changed with NSE site redesigns** without warning, breaking community libraries (NSEpy, nsetools) repeatedly.

**Impact:** Calling this a "public feed" understates operational fragility. In production this layer will be the first thing to break on a Friday afternoon.

**Recommended fix:**
> "We scrape NSE's *internal* announcements JSON endpoint at `https://www.nseindia.com/api/corporate-announcements?index=equities`. This is **not** an officially supported API; it requires session-cookie priming, a browser-like User-Agent, and aggressive rate-limiting. Production deployments should add a paid licensed feed (Zerodha Kite, Upstox, GlobalDatafeeds, or NSE Data Services) as a primary source and use the scrape only as a free-tier fallback."

**Source:** Endpoint behaviour documented in `jugaad-data`'s session-cookie helper code (https://github.com/jugaad-py/jugaad-data); blocking behaviour in multiple StackOverflow / Reddit reports (e.g. r/algotrading "NSE blocks requests" threads).

---

### 🔴 C4. The Lo–Mamaysky–Wang (2000) paper defines **10** patterns / **5 pairs** — the doc only implements **6 / 3 pairs** (§3.7)

**Doc says:**
> "Hand-rolled detectors for **five** classic chart patterns... Source for tolerances. Lo, Mamaysky, Wang (2000), 'Foundations of Technical Analysis'..."

then implements: double top, double bottom, head-and-shoulders, inverse H&S, triangles (asc/desc/sym).

**Reality:** Section II.A of the LMW paper opens with:
> "We focus on **five pairs** of technical patterns... head-and-shoulders (HS) and inverse head-and-shoulders (IHS), broadening tops (BT) and bottoms (BB), triangle tops (TT) and bottoms (TB), rectangle tops (RT) and bottoms (RB), and double tops (DT) and bottoms (DB)."

The five formal Definitions in the paper are:

| # | Pattern family | Patterns | Implemented? |
|---|----------------|----------|:-:|
| 1 | Head-and-Shoulders | HS, IHS | ✅ |
| 2 | Broadening | BTOP, BBOT (megaphone) | ❌ **MISSING** |
| 3 | Triangle | TTOP, TBOT | ✅ (treated as one bidirectional family) |
| 4 | Rectangle | RTOP, RBOT (range/consolidation) | ❌ **MISSING** |
| 5 | Double Top/Bottom | DTOP, DBOT | ✅ |

So the doc says "five chart patterns" but the *paper it cites* defines **ten**. Missing:
- **Broadening tops/bottoms** (megaphone — peaks rising AND troughs falling; classic late-stage-bull-market exhaustion or bottoming volatility-blowout pattern).
- **Rectangle tops/bottoms** (consolidation channels — flats both up and down; the LMW formal definition is "tops within 0.75% of their average AND bottoms within 0.75% of their average AND lowest top > highest bottom"). Rectangles are very common in NSE large-caps.

**Additional methodological pushback:** LMW use a **Nadaraya–Watson kernel-regression smoother** to identify the local extrema E1…E5 before applying the geometric definitions. The doc uses **SciPy `find_peaks` with min-distance=5** on raw prices instead. That's a defensible engineering simplification, but it is **not LMW's algorithm**. Either:
- Acknowledge the deviation explicitly in §3.7, or
- Implement the kernel-regression step (not hard — `scipy.stats.gaussian_kde` or `statsmodels`).

**Source:** Lo, Mamaysky, Wang (2000), *Journal of Finance* 55(4), 1705–1765, NBER WP 7613. PDF: https://www.nber.org/system/files/working_papers/w7613/w7613.pdf

---

### 🔴 C5. The "22 trading days apart" rule is **mis-attributed to Edwards & Magee (1966)** (§3.7 double top, also re-used in chart patterns)

**Doc says:**
> "The peaks must be at least **22 trading days apart** (Edwards & Magee, 1966 — cited by LMW)."

**Reality:**
- **Edwards & Magee** (*Technical Analysis of Stock Trends*, 1966 / 9th ed.) say the peaks of a double top are *"usually several weeks apart"* and the valley between them typically spans **5–7 weeks, rarely less than 3 weeks**. They **give no precise trading-day minimum**. It's narrative guidance, not a numeric threshold.
- The **22-day figure is LMW's own operational discretization** of E&M's qualitative "~one month / several weeks" guidance. From the LMW paper directly: *"…the two tops occur at least a month, or 22 trading days, apart. Therefore, we have the following Definition 5..."*

**Impact:** Anyone going to E&M to verify "22 days" will not find it and will (correctly) flag the doc as inaccurate.

**Recommended fix:**
> "The peaks must be at least **22 trading days apart** — Lo, Mamaysky & Wang's (2000) operationalization of Edwards & Magee's qualitative '~one month / several weeks' guidance."

**Sources:** LMW (2000) Definition 5; E&M (1966) Chapters VII–X (mirror PDF: https://vdthangmeomeo.wordpress.com/wp-content/uploads/2014/08/technical-analysis-of-stock-trends-9th-edition.pdf).

---

### 🔴 C6. NSE has ~**246–250** trading days/year, not 252 (§3.5, §1.1, indirectly throughout)

**Doc says:**
> "Highest high and lowest low over the last **252 trading days** (NSE convention: ~52 weeks of trading sessions)."

**Reality:**
- 252 is the **US/NYSE convention**, not NSE.
- NSE FY 2024–25 had **249 trading days** (per NSE's own Business Growth report).
- 2024 calendar year: ~**251 trading days**.
- 2026 calendar year: ~**246 trading days** (more weekday holidays).
- NSE's annual trading days range from ~246 to 252 depending on how holidays land.

**Impact:**
- Using 252 as a fixed constant for "52-week extreme" silently includes one extra session of "lookback" on most years and excludes a few on heavy-holiday years. For most signals this is in the noise.
- More importantly, "21 trading days/month" used as the MONTHLY horizon in §1.1 is fine (NSE averages ~20.5–20.8/month over a year), but the *justification* is "NIFTY's ~21 trading days/month" which is approximately correct only as a coarse round number. Worth a footnote.

**Recommended fix:**
> "Highest high and lowest low over the last **252 trading days** (US convention; NSE actually trades ~246–252 sessions/year, varying by holiday placement). Using 252 is harmless for a rolling extreme; for annualized vol math, see §6 footnote on NSE-vs-US trading-day count."

**Source:** NSE official holiday list — https://www.nseindia.com/resources/exchange-communication-holidays ; NSE 2024 calendar circular — https://nsearchives.nseindia.com/content/circulars/CMTR59722.pdf

---

### 🔴 C7. The "five chart patterns" / "seven candlestick patterns" framing under-counts the literature catastrophically (§3.6, §3.7)

This is part-spec-error, part-coverage-gap, and worth a separate critical bullet because the doc's own §10.5 explicitly invites this pushback (*"Are we missing a critical pattern or indicator?"*).

#### Candlesticks (§3.6) — only 7 of ~25 standard Nison patterns

The doc implements: doji, hammer, shooting star, bullish engulfing, bearish engulfing, morning star, evening star.

Steve Nison's *Japanese Candlestick Charting Techniques* (1991) — the cited source — covers, at minimum, the following **major reversal** patterns the doc is **missing**:

| Missing pattern | Why it matters |
|-----------------|----------------|
| **Hanging man** | Mirror of hammer; bearish reversal at resistance. Identical bar shape; only context differs. Free to implement. |
| **Inverted hammer** | Mirror of shooting star; bullish reversal at support. Free to implement. |
| **Harami / Harami cross** | "Inside bar" reversal — second bar's body fully *inside* the first's. Very common; both directions. |
| **Dark cloud cover** | Bearish 2-bar reversal; bullish bar then bearish bar opening above prior high but closing below midpoint. |
| **Piercing pattern** | Bullish mirror of dark cloud cover. |
| **Three white soldiers** | Strong bullish 3-bar continuation/reversal — three consecutive long bullish bars. |
| **Three black crows** | Bearish mirror — three consecutive long bearish bars. |
| **Tweezer tops/bottoms** | Two bars sharing the same high (top) or low (bottom). Often missed by reversal detectors. |
| **Abandoned baby (bullish/bearish)** | Three-bar gap reversal — most reliable reversal in Nison's table per Bulkowski. |

And the doc has **zero continuation patterns**, even though Nison covers an entire family:

| Missing continuation pattern | Why it matters |
|------------------------------|----------------|
| **Windows (gaps)** + **Tasuki gaps (up/down)** | Gap analysis is foundational. The doc currently has no gap detection at all. |
| **Rising / falling three methods** | 5-bar continuation pattern. Very common in trending NSE large-caps. |
| **Three-line strike** | High-reliability continuation per Bulkowski's stats. |
| **Mat hold** | Bullish continuation. |
| **Separating lines** | Continuation after gap. |

**Additionally:** the doc's quantitative thresholds for hammer (`body ≤ 0.35×range`, `lower_shadow ≥ 2×body`, `upper_shadow ≤ 0.3×body`) are **NOT from Nison**. Nison gives only one numeric rule: *"the lower shadow should be at least twice the height of the real body."* The other numeric ratios are modern algorithmic operationalizations (Bulkowski, TA-Lib). The doc should re-attribute:

> "Body/shadow numeric ratios are operationalizations following TA-Lib / Bulkowski conventions; Nison's original definitions are qualitative."

**Sources:**
- Nison, S. (1991), table of contents and pattern lists: http://www.r-5.org/files/books/trading/options-futures/charts-and-patterns/Steve_Nison-Japanese_Candlestick_Charting_Techniques-EN.pdf
- StockCharts ChartSchool — Candlestick Pattern Dictionary: https://chartschool.stockcharts.com/table-of-contents/chart-analysis/candlestick-charts/candlestick-pattern-dictionary
- Bulkowski's *Encyclopedia of Candlestick Charts* — pattern statistics: https://thepatternsite.com/

#### Chart patterns (§3.7) — see C4 above for LMW omissions

In addition to the LMW omissions (broadening, rectangles), several **non-LMW but industry-standard** patterns are absent:

| Missing | Notes |
|---------|-------|
| **Flag / Pennant** | Continuation patterns after sharp moves; very common, easy to detect. |
| **Wedge (rising/falling)** | Reversal patterns; both lines slope same direction. |
| **Cup & Handle** | O'Neil's signature pattern — popular in Indian retail community. |
| **Channel (parallel trendlines)** | Adds context for "buy near lower channel". |

These are not "in v1 because not LMW", which is fine — but the doc's §10.5 should explicitly enumerate them as deliberate omissions (right now §10.5 lists only Ichimoku/Fib/Volume Profile/VWAP).

---

## 2. High-importance issues

### 🟠 H1. Wilder's ADX threshold mis-stated (§3.2, §4.1)

**Doc says:**
> "ADX strength gate. If ADX is missing or < 20, the verdict is neutral... Wilder's threshold for 'this is a trending market' is 20–25."

**Reality:** Wilder's *New Concepts in Technical Trading Systems* (1978) gave **25** as the strong-trend threshold. The "20" lower-bound is a **modern convention** popularized by chartists/StockCharts, not Wilder. From StockCharts:

> "Wilder suggests that a strong trend is present when ADX is above 25 and no trend is present when ADX is below 20. There appears to be a gray zone between 20 and 25."

Wikipedia (citing Wilder 1978) gives a different rendering: *"ADX readings below 20 indicate trend weakness, and readings above 40 indicate trend strength. An extremely strong trend is indicated by readings above 50."*

**Recommended fix:** Drop the "Wilder's threshold is 20–25" framing. Use:
> "We gate on ADX < 20 → neutral. Wilder (1978) used **25** as the strong-trend threshold; **20 is the modern practical floor**. The 0.5/0.7/0.85 confidence anchors at ADX 20/30/40 are our own; defensibly low to encourage neutral verdicts in chop."

**Source:** https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx

---

### 🟠 H2. Bollinger Squeeze definition is non-standard and conflates two well-known squeezes (§3.4)

**Doc says:**
> "Bollinger-Band Squeeze. Definition. Boolean. The current bandwidth is in the **lowest 20%** of its values over the past **60 bars**. Source. Bollinger's own definition for 'compression precedes breakout'."

**Reality:** Two industry-standard definitions exist; **neither matches the doc**:

**(a) John Bollinger's original "Squeeze"** (*Bollinger on Bollinger Bands*, McGraw-Hill, 2001) uses **BandWidth = (Upper − Lower) / Middle** and signals Squeeze when BandWidth reaches a **6-month low (~125 trading days)** — not "lowest 20% over 60 bars".

**(b) John Carter's "TTM Squeeze"** (the more widely-traded modern variant; lives on TradingView, ThinkorSwim, Simpler Trading) uses **Bollinger Bands vs. Keltner Channels**:

> Squeeze ON ⟺ Upper BB < Upper KC AND Lower BB > Lower KC
> (default: 20-period BB at 2σ; 20-period KC at 1.5×ATR)

The doc's "20th percentile over 60 bars" is **a percentile-rank operationalization** that is not part of either Bollinger's or Carter's published definition. It is a perfectly reasonable empirical rule but should not be attributed to Bollinger.

**Recommended fix:**
> "We approximate Bollinger's Squeeze using a **rolling 20th-percentile of BandWidth over 60 bars**. Bollinger's original definition (Bollinger, 2001) uses a **125-bar BandWidth low**; the related TTM Squeeze (Carter) uses BB ⊂ Keltner Channels. Our percentile-rank approximation responds faster than Bollinger's 125-bar floor at the cost of more frequent firing — 🔬 NEEDS BACKTEST against both."

**Sources:**
- Bollinger BandWidth (StockCharts): https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/bollinger-bandwidth
- TTM Squeeze (StockCharts): https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/ttm-squeeze
- John Bollinger's own narrative: https://www.bollingerbands.com/bollinger-bands

---

### 🟠 H3. The 5% trough-drop saturation for double-top "depth" is too permissive (§3.7)

**Doc says:**
> "depth_score = min(trough_drop_pct / 0.05, 1.0) — i.e. a trough that drops at least **5%** below the average peak scores 1.0."

**Reality:** Bulkowski (the empirical reference for chart-pattern statistics; his *Encyclopedia of Chart Patterns* aggregates outcomes across thousands of patterns) explicitly says:

> *"Valley: The valley drop between the tops should measure at least **10%**, but allow exceptions."*
> *"Top price: The variation between price peaks is small, usually less than 3%."*

Bulkowski's "Big M" page cites **10%–20%** as typical trough depth.

**Impact:** Saturating the score at 5% means a shallow noise dip that wouldn't even register as a double-top under standard criteria gets a perfect depth-score of 1.0. The pattern detector will fire on a lot of garbage.

**Recommended fix:** Either
- Raise the saturation to **0.10** (10%) to align with Bulkowski, or
- Keep 5% as a "weak candidate" floor and add a tiered score: `0.5` at 5%, `1.0` at 10%.

Combined with the existing `confidence < 0.7` filter, the saturation point matters less, but the documentation is still misleading.

**Source:** https://thepatternsite.com/aadt.html and https://thepatternsite.com/bigm.html

---

### 🟠 H4. RSI 60/40 framing is sound but should credit Cardwell, not appear as "we deliberately use" (§4.2)

**Doc says:**
> "We deliberately use 60/40 for the *vote*, not 70/30 — 70/30 is the overbought/oversold *flag*, which is a different concern."

**Reality:** The 60/40 (and the broader 40-80/20-60) framing is the **Andrew Cardwell** / **Constance Brown** trend-RSI school:

- **Cardwell** (Brown's mentor): in uptrends, RSI oscillates **40–80** (40 acts as support); in downtrends, **20–60** (60 acts as resistance).
- **Brown** (*Technical Analysis for the Trading Professional*, 2nd ed., McGraw-Hill, 2011) uses a wider **40–90 / 10–60** range.

**Recommended fix:**
> "We use **60/40 as the trend-vote thresholds** (following Cardwell's RSI-range theory; cf. Brown 2011). The classical Wilder 70/30 is reserved for the overbought/oversold *flag*."

That makes it grounded rather than appearing arbitrary.

**Source:** https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi (cites Brown directly)

---

### 🟠 H5. "Classic floor-trader pivots" is missing R3 and S3 (§3.5)

**Doc says:**
> "Classic floor-trader pivots: PP, R1, R2, S1, S2."

**Reality:** Per Wikipedia and ActionForex, the classic set extends to **R3/S3** (and sometimes R4/S4):

```
R3 = H + 2×(PP − L)
S3 = L − 2×(H − PP)
```

R3/S3 are exactly the levels traders use to identify breakouts beyond the "normal" intraday range. By stopping at R2/S2, the levels classifier (§4.4) literally cannot tag a stock that's broken out of its R2/S2 envelope.

**Recommended fix:** Either add R3/S3 to the levels cluster, or explicitly state in §10 that R3/S3 are deliberately excluded (and explain why — e.g., "R3/S3 fire too rarely on daily bars to be useful for our horizons").

**Source:** https://en.wikipedia.org/wiki/Pivot_point_(technical_analysis)

---

### 🟠 H6. Alpha Vantage free-tier limits in §2.1 are not stated and are very tight (§2.1)

**Doc says:** "Alpha Vantage (final fallback — requires an API key)."

**Reality (verified 2026 on alphavantage.co/support):** the free tier is **25 requests/day total** — a hard cap. The old "5/min, 500/day" tier no longer exists. NSE coverage exists nominally (`RELIANCE.BSE` style) but multiple AmiBroker/Reddit/GitHub threads confirm persistent issues with NSE-specific feeds (data not updating, missing days, occasional outright stoppage).

**Impact:** As a "final fallback" the assumption is "uses key, unlimited-ish". Reality: 25 calls a day means with one watchlist of 50 NIFTY stocks, you exhaust Alpha Vantage in *two stocks*. It's effectively non-functional as an actual fallback.

**Recommended fix:** Document the 25/day limit prominently. Consider promoting one of the NSE-native sources (jugaad-data) to fallback-1 instead.

**Source:** https://www.alphavantage.co/support/#api-key

---

### 🟠 H7. Wilder-smoothing convergence guard (`2 × length = 28`) is mathematically insufficient (§3.2 ADX, §3.3 RSI)

**Doc says:**
> "ADX is published only when at least `2 × length = 28` bars are available; otherwise null."
> "[RSI] Convergence guard. Require `2 × length = 28` bars."

**Reality:** Wilder's smoothing is an EWMA with `α = 1/N`. It is **never fully converged**; it asymptotically approaches the true value. Common practice in quant literature is **at least 10× the length** (so ~140 bars for an N=14 indicator) before the value is "trustworthy"; some risk systems use 250 bars / one full year.

For comparison: a simple SMA of length 14 needs 14 bars to be exactly correct. An EWMA of length 14 with a "naive" 14-bar warmup carries ~30–40% bias from the seed value. A 28-bar warmup still carries ~10–15% bias.

**Impact:** ADX and RSI values for the first ~50 bars of any new instrument will be biased toward whatever seed value the implementation uses. If you're calling these "converged" at 28 bars and gating verdicts on them, you have systematic bias for any newly-listed stock or after a long data gap.

**Recommended fix:**
- Either bump the warmup to `10 × length` (140 bars), accepting that first-six-months-of-history predictions get NA, **or**
- Document this as a known approximation and add a `data_quality_warning` field to the prediction when `bars_used < 10 × max_indicator_length`.

**Source:** Standard quantitative finance reference; see e.g. discussion in Hudson & Thames' "Wilder smoothing convergence" technical notes (https://hudsonthames.org/) or Aronson, *Evidence-Based Technical Analysis* (Wiley 2007), Ch. 4 on indicator transformations.

---

### 🟠 H8. MACD has no convergence guard at all (§3.3)

**Doc says:** "Parameters used: fast = 12, slow = 26, signal = 9." — no warmup minimum stated.

**Reality:** MACD is built on EMAs. The signal line is an EMA-of-an-EMA, which compounds the convergence issue from H7. A reasonable warmup is `slow + signal × 2 = 44 bars` minimum, ideally `slow × 5 = 130 bars`. With only ~26 bars of history MACD values are essentially seed-noise.

**Recommended fix:** Add convergence guard for MACD analogous to the ones for ADX/RSI.

---

### 🟠 H9. The doc lists no major missing indicators / market regimes that are arguably essential

The doc's §10.5 acknowledges this and explicitly invites pushback. Here are concrete recommendations:

| Indicator/concept | Why arguably essential for NSE |
|-------------------|-------------------------------|
| **VWAP (Volume-Weighted Average Price)** | Institutional benchmark on NSE. Daily VWAP is computable from OHLCV; rolling/anchored variants are standard. The fact that you have OBV but no VWAP is odd. |
| **Ichimoku Kinko Hyo** | Hugely popular in Asia (and NIFTY 50 retail community). Provides cloud-as-support-resistance + multi-period momentum in one chart. |
| **Volume Profile / Market Profile (TPO)** | Shows where actual trading volume occurred, not just where prices went. Daily bars only support a degraded "volume profile" but it's still informative. |
| **Heikin-Ashi candles** | Smoothed-price candle variant used heavily in trend confirmation. |
| **Sector relative-strength** | The doc explicitly excludes this in §1.4 / §9 — but *single-stock predictions* on NSE are dominated by sector flows (e.g. IT, banking). At minimum acknowledge that a 5-day NIFTYIT relative move would change a TCS BIWEEKLY call. |
| **India VIX** | The doc calls macro out-of-scope but then uses ATR for volatility. India VIX is a forward-looking implied vol; it's free, daily, and a single number. Free to ingest. |

These are not bugs per se but are real omissions in any honest "is the indicator coverage complete?" review.

---

## 3. Medium-importance issues

### 🟡 M1. "Bollinger 1980s" is vague (§3.4)

Bollinger himself narrates: he "became active in markets full-time in 1980" and developed/named the bands on **Financial News Network c. 1983** when an on-air host asked what he called them. Use **"early 1980s, formalized c. 1983"**.

Source: https://www.bollingerbands.com/bollinger-bands

---

### 🟡 M2. "Appel 1979" for MACD is loose (§3.3)

Wikipedia and other secondary sources say MACD was created "in the late 1970s" via Gerald Appel's Signalert newsletters. There is no single canonical 1979 publication. Use **"Appel, late 1970s (Signalert newsletters)"** or **"Appel, c. 1979"**.

Source: https://en.wikipedia.org/wiki/MACD

---

### 🟡 M3. OBV credits Granville without acknowledging predecessors (§3.3)

OBV was named and popularized by Granville (1963), but the underlying concept (cumulative signed volume, called *"continuous volume"*) is earlier work by **Woods and Vignola**. If the doc is going to be pedantic about citations, this should read **"Granville (1963), building on Woods & Vignola"**.

Source: https://en.wikipedia.org/wiki/On-balance_volume

---

### 🟡 M4. "Worst-case RR" terminology is non-standard (§1.1, §6.2)

The math is right (using `zone_high` as the assumed entry for a long correctly captures the worst-case fill — most expensive entry → smallest reward and largest risk distance to a stop below the zone). But:

- The label "worst-case RR" is **non-standard** in mainstream trading literature. Most published RR formulae use a **single entry price**: `RR = (target − entry) / (entry − stop)`.
- Where entry zones exist (Wyckoff, supply/demand, ICT communities), people typically quote **midpoint RR** or **near-edge RR**.
- The "worst-fill" framing is more common in **execution / backtesting** literature than in pattern-trading.

**Recommended fix:** Rename to **"worst-fill RR"** and consider also reporting **midpoint RR** alongside. The current single number is conservative for sizing, but published edge studies you might compare against will use midpoint.

---

### 🟡 M5. Engulfing definition has equality-edge ambiguity (§3.6)

**Doc says (bullish engulfing):**
> "today.open ≤ yesterday.close AND today.close ≥ yesterday.open"

Nison's strict definition uses **strict inequality** for both — *"today's body must completely engulf yesterday's body"*. Allowing equality means a doji-like prior bar (open ≈ close) trivially satisfies the condition — and Nison explicitly says engulfing requires the **prior bar to have a real body**.

**Recommended fix:** Either tighten to strict inequality, or add a guard: `prior_body ≥ 0.10 × range` (so the engulfee actually has a body to engulf).

---

### 🟡 M6. LMW use kernel-regression smoothing for pivot detection — the doc uses raw `find_peaks` (§3.7)

Already mentioned in C4, but worth its own bullet for visibility. LMW use a **Nadaraya–Watson kernel regression** to smooth prices, then identify extrema in the smoothed series. The doc uses **SciPy `find_peaks` with min-distance=5** on raw prices — a faster, simpler approach but **not LMW's algorithm**. Either implement the kernel smoother (one extra dependency, ~20 lines of code) or explicitly disclaim "we use LMW's *geometric definitions* on `find_peaks`-detected pivots; the original paper uses kernel-smoothed extrema."

---

### 🟡 M7. Volatility-classifier "expanding" threshold is undocumented (§4.3)

**Doc says:**
> "**`expanding`** — bandwidth above its 60-bar median AND %B near either band edge (`<0.1` or `>0.9`)."

The "above its 60-bar median" is documented; the `%B` thresholds (0.1 / 0.9) are stated but **without source**. These are reasonable but appear to be the doc author's choice, not from any cited literature.

**Recommended fix:** Tag with 🔬 NEEDS BACKTEST or cite source.

---

### 🟡 M8. News weights have no source citation (§6.2)

The 35/65, 50/50, 60/40, 70/30 horizon × technical/news split is acknowledged as 🔬 NEEDS BACKTEST, which is honest. But the **directional choice** (technicals dominate as horizon lengthens) is itself a hypothesis. Possible counter-evidence to consider:

- Tetlock (2007), *"Giving Content to Investor Sentiment"*, *J. Finance* — finds news sentiment predicts short-horizon returns but with mean-reversion afterward (so news may matter MORE at daily horizon than the doc thinks).
- Da, Engelberg & Gao (2011), *"In Search of Attention"*, *J. Finance* — Google search-based attention measures predict next-2-week returns specifically.

So the asymmetric weighting is defensible but not unique; could equally be flat 50/50 across horizons or even more news-heavy at short horizons. Worth flagging as a hypothesis to test, not just a tuning parameter.

---

### 🟡 M9. Tier-2 citation thresholds (80% / 70%) are arbitrary (§7.2)

Acknowledged as 🔬 NEEDS BACKTEST in the doc. Two specific concerns:

1. **80% rationale-token grounding** is very strict. A 5-token noun phrase needs ~4 tokens in vocabulary. This is "no-creativity" territory. Suggest starting at **60%** and tightening with data.
2. **`contributing_signals` strict-substring-OR-70%-tokens** is really two checks chained with OR. It will be hard to reason about when it fires. Suggest collapsing to a single token-overlap ratio for explainability.

---

### 🟡 M10. Stop-pad ladder (0.10 / 0.15 / 0.20 / 0.25 ATR) unsourced (§6.1)

Hand-tuned by the author, acknowledged. But this is **the** load-bearing parameter for the win rate of the system. Any backtest should test at minimum:

- Doubling the ladder (0.20 / 0.30 / 0.40 / 0.50) — does it materially improve win rate at the cost of RR?
- Halving it — does it kill win rate by getting noise-stopped?
- A *flat* 0.20 ATR pad for all horizons — does the ladder structure even matter?

If the backtest finds the ladder structure doesn't matter, simplifying to a flat pad reduces a tunable knob.

---

### 🟡 M11. Brier-score baseline of 0.25 is correct only for a 50/50 base rate (§8.5)

**Doc says:**
> "Brier score. `mean((confidence − outcome)²)` where outcome is 1.0 if Mode-A correct, else 0.0. Lower is better; **0.25 is the random baseline (always-predict-0.5)**."

True for binary outcomes. But Brier-score interpretation depends on the **base rate**. If your Mode-A hit rate ends up at 0.40 (a meaningful tilt), the "always predict the base rate" reference is `p(1−p) = 0.24`. So 0.25 is the *worst-case* random baseline (highest variance for a 50/50 prior); a lower base rate has a lower naive baseline.

**Recommended fix:** Document Brier-skill score (BSS) as well: `BSS = 1 − BS / BS_baseline` where `BS_baseline` uses the empirical base rate. BSS > 0 means real skill; BSS = 0 means no skill vs. the base rate; BSS < 0 means worse than guessing the base rate.

---

## 4. Section-by-section detailed review

### §1. Objective and high-level architecture

**Strong points:**
- Separation of "horizon-agnostic gather" from "per-horizon synthesis" is good engineering. The reasoning ("same RSI, same earnings, same ATR — applies daily *and* monthly") is sound.
- The fail-fast policy is conservative and defensible — but see §10.7's open question, and consider: in a real product, one user-facing horizon failing while three succeed will frustrate users more than partial-results UI ambiguity.
- The degradation policy (technical fail = abort, news fail = degrade, synth fail = abort) is well-thought-out.

**Issues:**
- §1.1 "snapped to last trading day" — what calendar? The doc later uses NSE calendar (good) but here the contract is implicit. State that target_datetime uses the NSE trading calendar, and what library/source provides it (`pandas_market_calendars`? Custom?).
- §1.2 "asyncio.gather" — note this propagates the **first** exception and cancels siblings. That's deliberate fail-fast, but document it: the user gets the first horizon's error, not all four.
- §1.4 "Implied volatility / options data. Not used. ATR is our only volatility unit." — technically India VIX *is* free, daily, and an implied-vol figure. Inconsistent to claim "no IV" and then later (§9) say "macro indicators (VIX, ...) not used because macro adds noise." Pick one rationale.

---

### §2. Data ingestion

Already covered in C1, C2, C3, H6 above. Additional items:

- **§2.2 cache** — "process-wide in-memory cache" + "no eviction" is fine for the stated 50-stock workload but pathological for any growth. State an upper-bound ticker count or add LRU eviction (cheap insurance).
- **§2.2 "365 calendar days proactive fetch"** — for NSE that's ~250 trading days. Adequate for SMA200 (200 bars) but **insufficient for proper Wilder-smoothing convergence** (see H7). Consider 500 trading days proactive (~2 calendar years).
- **§2.3 "fetch_article_body never raises"** — good design choice. But the failure mode bag (paywall, JS-only, bot block) is varied; consider tagging the failure type so observability can distinguish "all failures are paywalls" (consistent issue) from random network blips.
- **§2.5 "yfinance analyst-estimates snapshot (no historical series)"** — true and a real limitation. Document that analyst-revisions detection (often the actual catalyst) is therefore impossible in v1. The §9 omissions list captures this implicitly; make it explicit.

---

### §3. Deterministic indicators

#### §3.1 sensitivity presets

The presets are reasonable bracketed ranges but **completely unsourced**. Where does (10, 30, 100) come from for `sensitive` SMA, vs. the more common (10, 20, 50)? The MACD `sensitive` (8, 17, 9) is a known short-period variant but `smooth` (19, 39, 9) isn't a textbook value. Either cite or tag 🔬.

#### §3.2 trend cluster

- SMA "Murphy (1999), industry-standard 'stack' — short / medium / long-term gauges" — fine but Murphy's *Technical Analysis of the Financial Markets* uses (10, 20, 50) more than (20, 50, 200). The (20, 50, 200) stack is **O'Neil's** *CAN SLIM* / Investor's Business Daily convention. Re-cite.
- EMA formula `2/(N+1)` is correct (smoothing factor α). Pring (2002) is a defensible cite but EMA pre-dates Pring by decades; cite him as a *modern textbook reference*, not the originator.
- ADX — see H1 above.

#### §3.3 momentum cluster

- RSI "Wilder (1978)" — correct.
- MACD — see M2.
- Stochastic — Lane is correct (late 1950s); a footnote on "Fast vs. Slow vs. Full" would help readers understand `k=14, d=3, smooth_k=3` is the **Full Stochastic** with standard parameters. Lane's original "Fast" is just `%K` raw + `%D = SMA(%K, 3)`. The doc's `smooth_k=3` makes it Full / "Slow with parameters exposed."
- OBV — see M3.

#### §3.4 volatility cluster

- ATR — Wilder (1978) correct.
- BB Squeeze — see H2.
- "Bandwidth = (upper - lower) / middle × 100" — this is normalized to a percent. Confirm whether the % is /100 normalized (common in charting libs) or unnormalized. Trivial but matters for cross-library comparison.

#### §3.5 levels cluster

- Pivot points — see H5 (missing R3/S3); also consider documenting that there are **multiple** pivot conventions (Camarilla, Woodie's, Fib, DeMark) and you've picked the "classic floor-trader" one.
- 52-week — see C6 (252 vs ~250).
- Swing high/low default 30 bars — fine; a footnote that "swing" definitions vary widely (some use fractal pivots à la Bill Williams) would strengthen the section.

#### §3.6 candlesticks

See C7 + M5.

Additionally:
- "Lookback. The most recent 5 bars" — sensible but means a 5-day-old hammer that just completed gating won't be picked up next day. Document whether re-detection on subsequent days is desired (probably yes) and if so confirm the gating still passes.
- "Bullish bar = close > open" — what about `close == open` (true doji)? The doc's "Bullish/Bearish/Doji" trichotomy needs an explicit `close == open` rule; otherwise some patterns silently miss-classify on a perfectly flat bar.

#### §3.7 chart patterns

See C4 + C5 + H3 + M6.

---

### §4. Cluster classifiers

#### §4.1 Trend classifier

- "ADX < 20 → neutral" — see H1.
- The DI confirmation rule is sensible.
- Confidence formula anchored to ADX is good. But the cap at 0.85 is hand-picked; consider what evidence would justify > 0.85.

#### §4.2 Momentum classifier

- 60/40 vote thresholds — see H4.
- "MACD line > signal without a fresh cross still counts as +0.5" — this is *position* signal, sensible. But the half-credit is arbitrary — why not 0.3 or 0.7? Document or 🔬.
- "Stochastic %K > %D and both above 50 → +0.5 bull" — fine, but **stochastic crosses are notoriously noisy**. Consider gating by minimum %K-%D separation (e.g. ≥ 5 points) to avoid double-counting noise.
- "OBV slope" — direction-and-magnitude, fine.
- Final cap `min(|net_vote| / 3.0, 0.85)` — note `|net_vote|` can theoretically reach 4.0 (1+1+0.5+0.5+0.5+0.5 = 4); you'd hit cap at 2.55. Worth confirming the algebra in code.

#### §4.3 Volatility classifier

- See M7 (undocumented thresholds).
- "Volatility is direction-agnostic" — true, and this is one of the cleaner decisions in the doc.
- The synthesizer prompt says volatility regime affects "target distance and stop distance" but the §6.2 prompt rules don't actually consume the `expanding/contracting/normal` label explicitly — they only consume ATR and stop-pad. So how does the synthesizer actually use this signal? Is it just narrative context for the LLM? Document.

#### §4.4 Levels classifier

- "Within 0.5 ATR" thresholds — sensible.
- Confidence formula linear in proximity — fine.
- One concern: "find the nearest level to current close" — what if the nearest level is the 52-week high which is **not** a real S/R level today (price has been miles below it for 6 months)? You'll get a spurious "near 52-week-high resistance" signal. Consider weighing levels by **recency** as well as proximity.

#### §4.5 Aggregation

- Good design. The decision to NOT aggregate to a single verdict and let the synthesizer do it horizon-dependent is correct.

---

### §5. News-impact LLM agent

- Strong section overall. The materiality filter, expectation framing ("5% growth is bearish if estimates expected 12%"), and already-priced-in adjustment are sophisticated.
- "Hard cap: at most 5 catalysts AND at most 5 risks" — sensible. But how is the cap enforced? At schema level (rejected if > 5) or at prompt level (begged for ≤ 5)? Schema-level is hard; prompt-level is soft. Which?
- "Confidence calibration guidance: 0.85+: avoid — reserved for 'smoking gun' cases" — good; but is this enforced or hoped? Tier-4 §7.4 has a `news.confidence + 0.10/0.15` ceiling on the synthesizer but **no explicit ceiling on the news agent itself**. Consider adding a tier-4 check on the impact assessment too.
- "Resilient LLM chain" ordering — `gemini/gemini-2.5-flash` first, `groq/llama-3.3-70b` second, `openrouter/mistral-large` third. Two real concerns:
  - **Cost vs. quality ordering**: this looks like cheapest-first, which optimizes cost but lets the weakest model decide *most* assessments. Consider a quality-first ordering for high-stakes calls (long-horizon predictions, big-cap stocks).
  - **Schema-validation failure → fall back** behaviour means a model that produces *almost-valid* JSON gets entirely skipped. Consider a one-attempt JSON-repair pass before fallback (saves money + time).
- Trafilatura for article body — confirmed real (M-version verified at https://trafilatura.readthedocs.io/), good choice.

---

### §6. Synthesizer

#### §6.1 Per-horizon constants table

- Mode-A thresholds (0.5/1.5/2.5/5.0%) acknowledged as 🔬. Good; **the open question in §10.1 is the right one**.
- The √T justification: random-walk scaling of **returns** does follow `σ × √T`. NIFTY 50 daily realized vol is ~16–18% annualized, which translates to ~1.0–1.1% daily. The 0.5%/0.75 ATR-pad style threshold is *roughly* `0.5 × σ_daily`, which is not the random-walk one-sigma move but rather a half-sigma signal threshold. The doc's "approximately √(bars) × 0.5%" works out to:
  - DAILY: 0.5% (matches ✓)
  - WEEKLY: √5 × 0.5% = 1.12% (doc says 1.5% — wider than √T would suggest)
  - BIWEEKLY: √10 × 0.5% = 1.58% (doc says 2.5% — wider)
  - MONTHLY: √21 × 0.5% = 2.29% (doc says 5.0% — much wider)

So the doc's thresholds **diverge from a clean √T schedule** for longer horizons — they're more conservative (require bigger moves to count as a hit). That's defensible (mean-reversion / fatter tails) but the doc shouldn't claim "approximately √T" when it's clearly not.

**Recommended fix:** Either restore the √T schedule (0.5 / 1.12 / 1.58 / 2.29) or rewrite the justification as: "Empirically chosen to require meaningful directional moves at each horizon; intentionally more conservative than pure √T scaling for longer horizons because (a) longer horizons have fatter tails and (b) we want hits to mean 'directional thesis worked', not 'noise drifted right way'."

- Stop-pad ladder — see M10.
- Horizon-confidence cap (0.85/0.85/0.80/0.75) — the *direction* (caps decrease with horizon) is sound. The numeric values are hand-picked. Test: a backtest can directly evaluate whether the per-horizon mean-confidence-vs-hit-rate slope is consistent with the cap.

#### §6.2 Synthesizer system prompt

- Direction-call weights — see M8.
- "Disagreement protocol" — the asymmetric resolution (news wins short, technical wins long) is consistent with the weights. Good.
- "Entry zone" — the BULLISH `[close, close + 0.25 ATR]` choice is defensible (you're entering on a strength setup, accepting a small premium to confirm direction). But the "worst-case fill at zone_high" makes RR conservative. Consider whether the entry should instead bracket close (e.g. `[close - 0.10 ATR, close + 0.15 ATR]`) for more typical fills.
- Target placement priority "1. nearest level beyond entry; 2. ATR fallback" — this is sensible. The 4×ATR look-distance in step 1 is a good guard against attaching to far-off levels.
- "Risk-reward floor 1.5" — see §10.3 in the doc; reasonable starting value.
- Forbidden vocabulary — "guarantee, certain, definitely, risk-free, easy money, multibagger" — good list; consider also "guaranteed", "sure-shot" (Indian retail-trader vernacular), "cannot fail", "no risk".

#### §6.3 What synthesizer does NOT see

- "Not given previous predictions" — sound for independence.
- "Not given backtest outcomes" — sound for avoiding overfitting/Goodhart.
- "Not given target_datetime" — odd. The horizon label is enough? Maybe — but if the synthesizer were told "target = 2026-05-21" it could note "target lands in earnings week" and adjust. Worth considering.

---

### §7. Guardrails

#### §7.1 Tier 1 — Grounding

- All checks well-motivated. The 0.01% close-price echo tolerance is appropriate for IEEE-754 floats.
- Stop-distance ATR sanity (0.5–3.0 ATR) is good; consider also a **target-distance** ATR sanity (≥ 1× ATR for non-NEUTRAL) so a "target right at the close" doesn't slip through.

#### §7.2 Tier 2 — Citation

- See M9 (thresholds arbitrary).
- The vocabulary construction is sensible. Concern: the indicator names list (`RSI, MACD, ADX, ATR, BB, %B, OBV, SMA, EMA, +DI, −DI, %K, %D`) is **hardcoded**. If you add an indicator (e.g. VWAP per §H9), you must remember to update this list. Consider deriving it from the indicator-cluster module.
- "Substring-equal OR 70%-tokens" is two checks with OR — see M9.

#### §7.3 Tier 3 — Consistency

- Direction-keyword check is fine but **language-fragile**. A rationale that says "the stock should appreciate" passes none of the BULLISH keywords (`up, rise, rally, breakout, buy, long, support holding, momentum positive, trend up`). Consider expanding the keyword set or using lemma-matching.
- The "≥ 2 bullish AND ≥ 2 bearish ⟹ NEUTRAL" rule is a strong opinion baked in. Document the trade-off: it forces non-commitment in mixed conditions, at the cost of always-NEUTRAL on choppy market days.

#### §7.4 Tier 4 — Calibration

- "Confidence ≤ news.confidence + 0.10 (DAILY) / +0.15 (WEEKLY)" — the +Δ acknowledges technicals can add a little independent signal. The numeric Δ is hand-picked; document.
- "Degraded news → confidence ≤ 0.65" — sensible.
- "Substring `'projection' or 'ATR projection'` → confidence ≤ 0.70" — this is **fragile**. The synthesizer can paraphrase ("ATR-based extension") and slip through. Better: have the schema include an explicit `target.method ∈ {level_anchored, atr_projection}` enum and gate on that.

#### §7.5 Retry policy

- One retry, fail-fast — defensible. The "diminishing returns / whack-a-mole" argument is real.

---

### §8. Persistence and grading

- Atomic writes via `os.replace` — POSIX-correct.
- Round-trip contract — good test discipline.
- File-per-prediction — fine for current scale; trivial to migrate later.
- Mode A/B/C — see §10.8 in the doc; the `STOP_HIT_AMBIGUOUS = stop_hit` worst-case-trader assumption is defensible. The `realized = stop_hit / 2` half-loss alternative is also defensible. **The most-honest** answer is to surface the ambiguity rate as a separate metric — predictions that resolve ambiguously are inherently less informative; a high ambiguity rate signals "your stops and targets are too close together for daily-bar resolution."
- Brier score — see M11.

---

### §9. Out of scope

The list is sensible. One inconsistency:
- §1.4 says "Implied volatility / options data. Not used. ATR is our only volatility unit."
- §9 says "Macro indicators (VIX, USD/INR, repo rate) — single-stock predictions over short horizons are dominated by stock-specific catalysts."

But India VIX *is* implied vol *is* macro. So is it excluded as "IV" or as "macro"? The reasoning differs (the IV exclusion is "another data source"; the macro exclusion is "adds noise"). Pick one rationale. India VIX is free, daily, and one number — getting it isn't a "whole new data source", which weakens the §1.4 rationale.

---

### §10. Open questions

These are well-framed. My specific take on each:

- **§10.1 thresholds:** consider stock-specific (each stock's own ATR percentile) — yes, this is the right move once you have backtest infrastructure. One-size-fits-all means a 0.5% DAILY hit threshold over-rewards low-vol names (TCS) and under-rewards high-vol ones (PVRINOX).
- **§10.2 news weights:** see M8 above. I'd argue 80/20 news for retail-heavy daily moves is plausible.
- **§10.3 RR floor:** 1.5 for daily/weekly, 2.0 for monthly. Concur.
- **§10.4 candlestick gating:** 1×ATR is reasonable but **untested**. Strongly suggest 0.5/1.0/2.0 sweep in backtest. Likely 0.75–1.0 ATR is the sweet spot.
- **§10.5 missing patterns:** see C7 + H9 above for my list.
- **§10.6 80% citation threshold:** start at 60%, tighten with data (see M9).
- **§10.7 fail fast:** I'd argue **partial-results with `failed_horizons` field** for production. Fail-fast is a debug-mode contract; it shouldn't be the user-facing one.
- **§10.8 STOP_HIT_AMBIGUOUS:** half-loss split + surface ambiguity rate as separate metric. Best-of-both.

---

### Appendix A (file map) and B (by-hand vetting)

Both useful. Appendix B is unusually well-thought-out — most TA systems are unauditable from disk. The 8-step checklist is the right one.

One addition for Appendix B: step 9 — *"Cross-check the news catalysts against your own news memory of that day. If the agent's catalysts include 'Q4 earnings beat' but you remember the day as 'sector-wide selloff on Fed news', that's a materiality miss."*

---

## 5. Items the doc author didn't ask about that I want to push back on

### 5a. Look-ahead bias risk in the trafilatura article-body fetch

When the news agent fetches an article body for a 2-day-old article, **`trafilatura` returns the article as it currently is on the source URL**, which may have been updated. Source publishers correct/update articles. Backtests using replayed news will get *current* article text, not historical text. This is a subtle look-ahead leak.

**Recommendation:** The backtest replay layer (§8.6) needs a **frozen snapshot of article bodies as-of `as_of`**, not live fetches. Either Wayback-Machine integration or a one-time snapshot store keyed on `(url, fetch_timestamp)`.

### 5b. yfinance EOD prices have known revisions

Yahoo silently revises historical prices for splits, dividend adjustments, and occasionally for late-correction trade reports. Backtests run today vs. last week may produce different hit rates for the **same** historical prediction. This is invisible without a frozen price snapshot.

**Recommendation:** The replay layer should snapshot prices once per (ticker, as_of) tuple and refuse to re-fetch.

### 5c. The 60-second cooldown is per-process

If you scale out to multiple workers, each worker has its own cooldown timer, so a hammered API gets hammered N× harder. Migrate to a Redis-backed shared cooldown if/when you scale.

### 5d. No discussion of timezone correctness for `published_at`

GDELT returns UTC; NSE filings are IST; analyst-estimates from yfinance are usually US-Eastern; price bars are exchange-local. The doc says "tz-aware UTC, later converted to IST for display." Confirm:
- **All datetime arithmetic** in the codebase uses tz-aware datetimes (no naive `datetime.now()`).
- The `as_of` timestamp is unambiguously IST.
- The 15:30 IST cutoff in §1.1 ("today's close if pre-15:30") is correctly applied — what happens for a `predict()` call at 15:30:00.5? At 15:29:59?

### 5e. The synthesizer's JSON output via `extra=forbid` — model-fragility

`extra=forbid` Pydantic schemas are great for validation but **brittle to LLM output drift** — a model that adds a helpful "explanation" field will hard-fail and trigger retry. With every model update (Gemini 2.5 → 2.6 → 3.0) this becomes a maintenance burden. Consider:
- `extra='ignore'` for non-critical fields,
- A pre-validation cleanup step that strips known-extra keys.

### 5f. `model_chain` field as audit trail — what's the schema?

The doc references `model_chain` as a field that gets tagged (`news_impact:degraded`, `backtest`) but never defines its schema. Is it a list of strings? Tuple? Stringified comma list? For audit grep-ability it should be a typed list.

### 5g. Calibration over what window?

§8.5 lists the metrics but doesn't say over what corpus window the calibration runs. "Last 90 days"? "Last N predictions"? "All-time"? Each has very different implications for production monitoring (a 90-day rolling Brier is a different metric than all-time).

### 5h. No mention of survivorship bias

If your watchlist is "current NIFTY 50", backtesting on it is **survivorship-biased** — you're testing on companies that were good enough to stay in the index. A proper backtest needs the **historical NIFTY 50 membership** (which changes ~quarterly) or at least an explicit acknowledgment of the bias.

---

## 6. Items the doc tagged 🔬 NEEDS BACKTEST — my take on priority

| Item | Doc § | Priority | Why |
|------|-------|---------:|-----|
| Per-horizon win/loss thresholds | §6.1, §10.1 | 🔴 Highest | Drives every Mode-A metric. Wrong thresholds = wrong hit rate = wrong everything downstream. |
| News vs technical horizon weights | §6.2, §10.2 | 🔴 Highest | The largest swing in synthesizer behavior. |
| Citation thresholds (80%/70%) | §7.2, §10.6 | 🟠 High | Affects retry rate and prediction throughput. |
| Stop-pad ATR ladder | §6.1 | 🟠 High | Drives Mode-B win rate directly. |
| Candlestick-gating ATR proximity | §3.6, §10.4 | 🟡 Medium | Affects how often patterns surface, not the hit rate of those that do. |
| 1.5 RR floor | §6.2, §10.3 | 🟡 Medium | Affects synthesizer's NEUTRAL-fallback rate; less direct on accuracy. |
| One-retry policy | §7.5 | 🟢 Low | Cost/throughput metric, not accuracy. |
| Same-bar T/S disambiguation | §8.4, §10.8 | 🟢 Low | Edge case; impact is bounded. |

---

## 7. Bibliography (for re-grounding the doc)

Primary literature the doc cites or should cite:

- **Wilder, J. W.** (1978). *New Concepts in Technical Trading Systems.* Trend Research. ISBN 978-0894590276. — RSI, ADX, ATR, parabolic SAR.
- **Murphy, J. J.** (1999). *Technical Analysis of the Financial Markets.* New York Institute of Finance. ISBN 978-0735200661. — SMA, MACD overview, classical patterns.
- **O'Neil, W. J.** (2009 4e). *How to Make Money in Stocks.* McGraw-Hill. ISBN 978-0071614139. — Origin of the (20, 50, 200) SMA stack via CAN SLIM / IBD.
- **Pring, M.** (2002). *Technical Analysis Explained* (4e). McGraw-Hill. ISBN 978-0071381932. — Modern textbook coverage of EMA and momentum.
- **Appel, G.** (late 1970s). Signalert newsletters, later compiled in Appel, G. (2005), *Technical Analysis: Power Tools for Active Investors.* FT Press. ISBN 978-0131479029. — MACD origin.
- **Lane, G.** (late 1950s). Investment Educators training materials. — Stochastic oscillator. (No primary publication; popularized via courses.)
- **Granville, J.** (1963). *Granville's New Key to Stock Market Profits.* Prentice-Hall. ISBN 978-1614277361. — On-Balance Volume.
- **Bollinger, J.** (2001). *Bollinger on Bollinger Bands.* McGraw-Hill. ISBN 978-0071373685. — Definitive Bollinger Bands + Squeeze definition.
- **Carter, J.** (2005). *Mastering the Trade.* McGraw-Hill. ISBN 978-0071459587. — TTM Squeeze (BB-vs-Keltner).
- **Brown, C.** (2011 2e). *Technical Analysis for the Trading Professional.* McGraw-Hill. ISBN 978-0071759144. — RSI ranges 40-90 / 10-60.
- **Cardwell, A.** — RSI range theory; primary sources are Cardwell's training courses, not a book. Best secondary reference is Brown (2011) which credits him.
- **Nison, S.** (1991). *Japanese Candlestick Charting Techniques.* New York Institute of Finance. ISBN 978-0139316500. — All candlestick patterns; doc currently uses ~28% of the book's pattern catalog.
- **Bulkowski, T.** (2005 2e). *Encyclopedia of Chart Patterns.* Wiley. ISBN 978-0471668268. — Empirical pattern statistics; minimum trough depth, separations, etc.
- **Bulkowski, T.** (2008). *Encyclopedia of Candlestick Charts.* Wiley. ISBN 978-0470181010. — Empirical candlestick statistics.
- **Edwards, R. & Magee, J.** (1966 5e, also 9e 2007). *Technical Analysis of Stock Trends.* Magee. ISBN 978-0814408643. — Foundational chart patterns (qualitative).
- **Lo, A. W., Mamaysky, H., & Wang, J.** (2000). "Foundations of Technical Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation." *Journal of Finance* 55(4), 1705–1765. — 5 pairs / 10 patterns; kernel-regression smoother. NBER WP 7613. PDF: https://www.nber.org/system/files/working_papers/w7613/w7613.pdf
- **Tetlock, P. C.** (2007). "Giving Content to Investor Sentiment: The Role of Media in the Stock Market." *Journal of Finance* 62(3), 1139–1168. — Daily-horizon news-sentiment evidence.
- **Da, Z., Engelberg, J., & Gao, P.** (2011). "In Search of Attention." *Journal of Finance* 66(5), 1461–1499. — Search-attention measures.
- **Aronson, D. R.** (2007). *Evidence-Based Technical Analysis.* Wiley. ISBN 978-0470008744. — Statistical hygiene for TA testing.

Online references verified during this review:

- StockCharts ChartSchool — https://chartschool.stockcharts.com/ (multiple pages cited inline)
- John Bollinger's site — https://www.bollingerbands.com/bollinger-bands
- Bulkowski's Pattern Site — https://thepatternsite.com/
- GDELT Doc 2.0 announcement — https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
- NSE official holiday list — https://www.nseindia.com/resources/exchange-communication-holidays
- NYU V-Lab NIFTY GARCH — https://vlab.stern.nyu.edu/volatility/VOL.NIFTY%3AIND-R.GARCH
- Pydantic-AI docs — https://ai.pydantic.dev/
- Trafilatura docs — https://trafilatura.readthedocs.io/
- jugaad-data (NSE successor library) — https://github.com/jugaad-py/jugaad-data
- Alpha Vantage rate limits — https://www.alphavantage.co/support/#api-key

---

## 8. Final scorecard

| Dimension | Score / 10 | Comments |
|-----------|:----------:|----------|
| Architecture | **9** | Two-phase split is excellent; degradation policy thoughtful. |
| Indicator coverage | **5** | Standard indicators present but missing VWAP/Ichimoku/Volume Profile; only 7 of ~25 candlestick patterns; 6 of 10 LMW chart patterns. |
| Citation rigor | **5** | Ambitious aspiration ("vettable") but multiple mis-attributions (Edwards & Magee, Bollinger Squeeze, hammer ratios). |
| Data-source soundness | **3** | Stooq doesn't cover NSE; GDELT 7-day claim wrong; NSE filings called "public API"; Alpha Vantage limits unstated. **The single weakest section.** |
| Synthesizer / prompt design | **8** | Disagreement protocols, RR floor, NEUTRAL semantics all well-thought-out. Numbers acknowledged as 🔬. |
| Guardrail design | **8** | Four-tier model is correct and well-motivated. Some thresholds arbitrary (acknowledged). |
| Persistence + grading | **7** | Sound mechanics. Brier-baseline footnote needed; survivorship bias unmentioned. |
| Documentation honesty | **9** | The 🔬 tags, §9 explicit non-goals, and §10 open-questions list all show the right intellectual posture. |
| **Overall** | **6.5 / 10** | A solid v0.9. The data layer (§2) and the chart-pattern coverage (§3.6/§3.7) need a real fix-up pass before v1 ships. |

---

## 9. Recommended next steps for the doc author

In priority order:

1. **Replace Stooq** with a real NSE data source (jugaad-data + Zerodha/Upstox tier). (C1)
2. **Fix the GDELT 7-day claim** — it's 9 years of history, not 7 days. (C2)
3. **Re-attribute the 22-bar rule** to LMW, not Edwards & Magee. (C5)
4. **Re-state NSE trading days** (~246–250, not 252). (C6)
5. **Add the missing LMW patterns** (broadening, rectangles) — 10 patterns total, not 5. (C4)
6. **Add at minimum these candlestick patterns**: hanging man, inverted hammer, harami, dark cloud cover, piercing pattern, three white soldiers, three black crows. (C7)
7. **Re-attribute hammer ratios** — they are not Nison's. (C7 / H1 prelude)
8. **Document NSE filings reality**: not a public API, requires session priming, Cloudflare-protected. (C3)
9. **Fix BB Squeeze attribution** — your definition is not Bollinger's. (H2)
10. **Bump Wilder-smoothing warmup** from 28 to ≥140 bars (or document the bias). (H7, H8)
11. **State Alpha Vantage's 25/day cap** prominently. (H6)
12. **Re-credit the 60/40 RSI** vote to Cardwell. (H4)
13. **Add R3/S3** to the "classic" floor-trader set. (H5)
14. **Resolve the IV / India VIX inconsistency** between §1.4 and §9.

After those: re-read the doc end-to-end and remove every claim that begins "X is the standard"/"per Y's original definition" unless a verifiable primary source is in hand.

---

*End of review. Reviewed by Code Puppy 🐶, instructor mode, no sugar-coating. The doc author shows good instincts and intellectual honesty (the 🔬 tags!), but the gap between the document's ambition ("vettable against literature") and its actual sourcing is wider than it should be. Fix the critical items and this is publishable.*
