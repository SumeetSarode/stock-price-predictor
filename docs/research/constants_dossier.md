# Constants Dossier — Research Sources & Justifications

> **Purpose**: Every numerical constant in the production code should trace back
> to a source documented here. If you can't find a constant in this file, it's a
> bug — open an issue, do the research, add the entry, then update the code with
> a reference like `# see docs/research/constants_dossier.md §X.Y`.
>
> **Methodology**: Web-research pass on 2026-04-28. Sources are cited inline
> with retrieval URLs. Where literature is silent or this is a novel use case
> (e.g. LLM-driven prediction), the entry is marked `🔬 NEEDS BACKTEST` and
> deferred to Phase 2.

## Legend

- 🟢 **CANONICAL**: Value is the original author's published number. No deviation.
- 🟡 **DERIVED**: Literature gave a value; we modified it (or picked a point in a
  range the literature offered). Deviation explained below.
- 🔴 **WAS VIBES → NOW RESEARCHED**: We had no source; this entry now grounds it.
- 🔬 **NEEDS BACKTEST**: Literature genuinely silent. Use literature analog as
  v1; revisit in Phase 2 calibration project.

---

## 1. Indicator Periods

### 1.1 RSI period = 14 🟢
- **Source**: Wilder, J. Welles. *New Concepts in Technical Trading Systems*
  (Trend Research, 1978). ISBN 978-0894590276.
- **Confirmed via**: Wikipedia, *Relative strength index*. Retrieved 2026-04-28
  from https://en.wikipedia.org/wiki/Relative_strength_index — direct quote:
  *"The RSI is most typically used on a 14-day timeframe ... Wilder recommended
  a smoothing period of 14."*
- **Our value**: 14 (`analysis/momentum.py:14`, `_momentum_signal.py` defaults).
- **Verdict**: Match. No change needed.

### 1.2 MACD = (12, 26, 9) 🟢
- **Source**: Appel, Gerald. Original MACD specification, late 1970s.
- **Confirmed via**: Wikipedia, *MACD*. Retrieved 2026-04-28 from
  https://en.wikipedia.org/wiki/MACD — direct quote: *"The formula for the MACD
  line is based on two exponential moving averages of the close prices, usually
  with the periods of 12 and 26 ... The signal line is then built as the
  exponential moving average of the MACD line ... EMA₉(MACD line)."*
- **Our value**: (12, 26, 9) (`analysis/momentum.py:26-28`).
- **Verdict**: Match. No change needed.

### 1.3 ATR period = 14 🟢
- **Source**: Wilder, *New Concepts* (1978).
- **Confirmed via**: Wikipedia, *Average true range*. Retrieved 2026-04-28 from
  https://en.wikipedia.org/wiki/Average_true_range — Wilder's α=1/14 SMMA
  formula confirmed.
- **Our value**: 14 (`analysis/volatility.py:15`).
- **Verdict**: Match.

### 1.4 ADX period = 14 🟢
- **Source**: Wilder, *New Concepts* (1978).
- **Confirmed via**: Wikipedia, *Average directional movement index*. Retrieved
  2026-04-28 from https://en.wikipedia.org/wiki/Average_directional_movement_index
  — *"Wilder used 14 days originally."*
- **Our value**: 14 (`analysis/trend.py:53`, `_trend_signal.py`).
- **Verdict**: Match.

### 1.5 Bollinger Bands = (20, 2.0) 🟢
- **Source**: Bollinger, John. Original specification, 1980s. Re-affirmed by
  the author on his official site:
  > *"The defaults today are the same as they were 35 years ago, **20 periods**
  > for the moving average with the bands set at plus and minus **two standard
  > deviations** of the price."*
  Retrieved 2026-04-28 from https://www.bollingerbands.com/ ("What Are
  Bollinger Bands?" article by John Bollinger, CFA, CMT).
- **Confirmed via**: StockCharts ChartSchool, *Bollinger Bands*. Retrieved
  2026-04-28 from https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/bollinger-bands
  — *"the middle band is a simple moving average that is usually set at
  20 periods... outer bands are usually set 2 standard deviations above and
  below the middle band."*
- **Note on TA-Lib**: TA-Lib's `BBANDS` function defaults to `timeperiod=5`
  which is unusual / non-canonical. We follow Bollinger himself, not TA-Lib.
- **Our value**: (20, 2.0) (`analysis/volatility.py:27-28`).
- **Verdict**: Match (Bollinger himself).

### 1.6 Stochastic = (14, 3, 3) 🟢
- **Source**: Lane, George C. Stochastic oscillator, 1950s. Per *Lane's
  Stochastics*, *Technical Analysis of Stocks and Commodities*, May/June 1984.
- **Confirmed via**: StockCharts ChartSchool, *Stochastic Oscillator*.
  Retrieved 2026-04-28 from https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/stochastic-oscillator-fast-slow-and-full
  — direct quote: *"The default setting for the Stochastic Oscillator is
  **14 periods**, which can be days, weeks, months or an intraday timeframe.
  A 14-period %K would use the most recent close, the highest high over the
  last 14 periods and the lowest low over the last 14 periods."*
- **Also confirmed via**: Wikipedia *Stochastic oscillator* — *"Typical values
  for N are 5, 9, or 14 periods. Smoothing the indicator over 3 periods is
  standard."* (14 is the most common.)
- **Note on TA-Lib**: TA-Lib's `STOCH` defaults to `fastk_period=5` which is
  the "Fast Stochastic" variant. Our 14 corresponds to the canonical "Full
  Stochastic" defaults that practitioners use by default on every charting
  platform (StockCharts, TradingView, etc.).
- **Our value**: %K=14, %D=3, smooth=3 (`analysis/momentum.py:71-73`).
- **Verdict**: Match (StockCharts canonical, Lane original).

### 1.7 Trading days per year = 252 🟢
- **Source**: NYSE annual trading-day count (~252 after weekends and ~9 US
  holidays). Standard in quant finance for annualization.
- **Note for NSE**: Indian NSE has ~248–250 trading days/year (more national
  holidays). Difference is minor (~1%) and doesn't affect any of our annualized
  calculations meaningfully. Keep 252 as the universal constant.
- **Our value**: 252 (`analysis/levels.py:13`).
- **Verdict**: Match (with NSE caveat documented).

### 1.8 SMA periods = (20, 50, 200) "standard" 🟢
- **Source**: Murphy, John J. *Technical Analysis of the Financial Markets*
  (NYIF, 1999). Classic short/medium/long combo.
- **Confirmed via**: Universal practitioner convention. 20-day = ~1 trading
  month, 50-day = ~10 weeks, 200-day = ~10 months. Used by most major financial
  publications, charting platforms, and academic studies.
- **Our value**: [20, 50, 200] (`analysis/__init__.py:31`).
- **Verdict**: Match.

### 1.9 EMA period = 20 🟢
- **Source**: Murphy, *Technical Analysis* (1999). Standard short-term EMA.
- **Our value**: 20 (`analysis/__init__.py:31`).
- **Verdict**: Match.

---

## 2. Indicator Thresholds

### 2.1 RSI overbought / oversold = 70 / 30 🟢
- **Source**: Wilder (1978).
- **Confirmed via**: Wikipedia *RSI* — *"Wilder believed that tops and bottoms
  are indicated when RSI goes above 70 or drops below 30."*
- **Our value**: `RSI_OVERBOUGHT=70`, `RSI_OVERSOLD=30` (`_momentum_signal.py:14-15`).
- **Verdict**: Match.

### 2.2 RSI neutral band = 40 / 60 🟡 (was 🔴, now grounded in Cardwell)
- **Original audit classification**: 🔴 vibes.
- **Actual finding**: NOT vibes. Andrew Cardwell's RSI extension established
  that *"uptrends generally traded between RSI 40 and 80, while downtrends
  usually traded between RSI 60 and 20"* — Wikipedia *RSI*, citing Cardwell's
  work in technical-analysis literature.
- **Interpretation**: RSI 40 acts as the floor in uptrends; RSI 60 acts as the
  ceiling in downtrends. Therefore [40, 60] is the **transition zone** where
  trend regime is unclear → call it "neutral".
- **Our value**: `RSI_NEUTRAL_LOW=40`, `RSI_NEUTRAL_HIGH=60`
  (`_momentum_signal.py:16-17`).
- **Verdict**: Match Cardwell's empirical zone bounds. **Add citation comment.**

### 2.3 ADX strong / moderate / weak = 40 / 25 / 20 🟡
- **Source**: Wikipedia *ADX*, retrieved 2026-04-28 — *"ADX readings below 20
  indicate trend weakness, and readings above 40 indicate trend strength. An
  extremely strong trend is indicated by readings above 50."*
- **Wilder original (1978)** used >25 as the trending threshold (per practitioner
  consensus and StockCharts ChartSchool).
- **Our values**:
  - `ADX_STRONG=40` ✅ matches Wikipedia "trend strength"
  - `ADX_WEAK_FLOOR=20` ✅ matches Wikipedia "trend weakness"
  - `ADX_MODERATE=25` ✅ matches Wilder's original trending threshold
- **Verdict**: All three values are literature-grounded. **Add citation comment.**

### 2.4 RSI 80 / 20 (extreme variant) — not currently used
- **Source**: Wikipedia *RSI* — *"High and low levels — 80 and 20, or 90 and
  10 — occur less frequently but indicate stronger momentum."*
- **Note**: We use 70/30, not 80/20. Choice is correct for swing-trading
  horizons (daily/weekly/biweekly/monthly). 80/20 is for "extreme/strong
  trending" markets where 70/30 fires too often.
- **Verdict**: No change. Document the choice.

---

## 3. Bollinger %B Bands 🔬 NEEDS BACKTEST

### 3.1 %B bullish / bearish = 0.55 / 0.45 🟡 (anti-flap dead-band)
- **Bollinger himself** defined the metric — from bollingerbands.com ("What
  Are Bollinger Bands?"): *"I created %b, an indicator that depicted where
  price was in relation to the bands."* Bollinger published the formula but
  did NOT specify a "bullish above X / bearish below Y" interpretation rule.
- **StockCharts ChartSchool** canonicalized the 6 reference levels of %B,
  retrieved 2026-04-28 from https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/b-indicator:
  > *"There are six basic relationship levels:*
  > * *%B is below 0 when price is below the lower band*
  > * *%B equals 0 when price is at the lower band*
  > * *%B is between 0 and .50 when price is between the lower and middle band (20-day SMA)*
  > * *%B is between .50 and 1 when price is between the middle and upper band*
  > * *%B equals 1 when price is at the upper band*
  > * *%B is above 1 when price is above the upper band"*
- **What this means**: 0.50 is the canonical midline (price = 20-SMA). Above
  it = upper half of the band envelope (bullish bias); below it = lower half
  (bearish bias). NEITHER Bollinger nor StockCharts defines a "dead-band"
  tolerance around 0.50 — that's our addition.
- **Practitioner convention**: %B > 0.5 is bullish bias, %B < 0.5 is bearish
  bias.
- **Our values**: 0.55 / 0.45 (`_volatility_signal.py:21-22`). The ±0.05
  buffer around 0.50 is OUR engineering choice, designed to prevent "flapping"
  when %B oscillates around exactly 0.50 across consecutive bars.
- **Verdict**: ⚠️ Partially literature-grounded (the 0.5 midpoint is canonical
  per Bollinger himself + StockCharts); the ±0.05 dead-band is a defensible
  practitioner anti-flap pick.
- **Action**: Document; 🔬 backtest in Phase 2 to confirm the dead-band width.

### 3.2 ATR% dead/normal/manic = 1.0 / 1.0–4.0 / 6.0 🔴 → 🔬
- **Literature on ATR-as-percentage-of-price**: Wilder (1978) defined ATR as
  absolute price units. The ratio ATR/Close (often called ATRP or "ATR
  percent") is a derived practitioner metric without a canonical band.
- **Empirical context**: For NIFTY 50 large-caps, daily ATR% typically sits in
  the 1.5–3.5% range; small/mid-caps can run 4–8%. Earnings/event days produce
  spikes to 6–10%. Source: practitioner consensus, no canonical paper.
- **Our values**: dead<1.0%, normal 1.0–4.0%, manic>6.0%.
- **Verdict**: ⚠️ Reasonable for large-cap Indian equities but no published
  source confirms exact bounds.
- **Action**: Document as v1 heuristic; backtest in Phase 2 against actual
  NIFTY 50 ATR% distribution to derive empirical percentile bands (e.g.
  `dead = p10`, `normal = p10–p90`, `manic = p90+`).

---

## 4. OBV Divergence Threshold = 2.0% 🔬 NEEDS BACKTEST
- **Source**: Wikipedia *On-balance volume* — describes OBV calculation
  (Granville, 1963) but does NOT specify a divergence quantification
  threshold. OBV divergence is qualitatively defined as "OBV makes lower high
  while price makes higher high" with no canonical magnitude rule.
- **Original source**: Granville, Joseph. *Granville's New Key to Stock Market
  Profits* (1963). Granville's original method was visual/discretionary, not
  quantitative.
- **Our value**: 2.0% magnitude (`_momentum_signal.py:20`).
- **Verdict**: ⚠️ No literature backing for the specific 2% threshold.
- **Action**: Document as v1 heuristic. Phase 2 backtest: derive divergence
  threshold from realized predictive value across NIFTY 50 names.

---

## 5. Sensitivity Presets 🟡

### 5.1 "Standard" preset bundle
All "standard" values in `analysis/__init__.py` match the canonical literature
periods listed in §1. Verdict: 🟢 across the board.

### 5.2 "Sensitive" preset bundle 🟡
Faster periods designed to surface signals earlier (more noise, more
responsiveness):

| Indicator | Standard | Sensitive | Justification |
|-----------|---------:|----------:|---------------|
| RSI       | 14       | 9         | Wikipedia *Stochastic* lists "5, 9, or 14" as common — 9 is the published "fast" variant |
| MACD      | 12/26/9  | 8/17/9    | Faster EMAs with 8/17 ratio preserved (~0.65 vs canonical 0.46); signal line unchanged |
| ATR       | 14       | 9         | Symmetry with RSI/Stochastic 9-period fast variant |
| Stoch     | 14/3/3   | 9/3/3     | Wikipedia *Stochastic* explicitly lists 9 |
| Bollinger | 20/2.0   | 10/2.0    | Half the lookback = ~2× as responsive; std unchanged per Bollinger's recommendation |
| Swing LB  | 30 bars  | 15 bars   | ⚠️ no literature anchor — heuristic halving |

- **Verdict**: Mostly literature-derived (faster variants of canonical periods).
  Swing lookback 15 is a heuristic; document as such.
- **Action**: Add citation comment to the PRESETS dict explaining the 9-period
  lineage from the Stochastic Wikipedia variant set.

### 5.3 "Smooth" preset bundle 🟡
Slower periods to filter out noise (fewer signals, more reliable):

| Indicator | Standard | Smooth   | Justification |
|-----------|---------:|---------:|---------------|
| RSI       | 14       | 21       | StockCharts canonical: *"raise it to decrease sensitivity"*; 21 follows the Fibonacci sequence (Murphy convention) |
| MACD      | 12/26/9  | 19/39/9  | Roughly 1.5× the standard fast/slow; signal unchanged |
| ATR       | 14       | 21       | Symmetric with RSI smooth variant |
| Stoch     | 14/3/3   | 21/5/5   | ⚠️ 21/5 not in Wikipedia's canonical set; Murphy/Fibonacci convention |
| Bollinger | 20/2.0   | 30/2.0   | 1.5× lookback, std unchanged |
| Swing LB  | 30       | 60       | ⚠️ heuristic doubling, no literature anchor |

- **Source for the "raise period to decrease sensitivity" principle**:
  StockCharts ChartSchool, *RSI*, retrieved 2026-04-28 — *"The default
  look-back period for RSI is 14, but you can lower it to increase sensitivity
  or raise it to decrease sensitivity. A 10-day RSI is more likely to reach
  overbought or oversold levels than a 20-day RSI."*
- **Verdict**: The PRINCIPLE of "raise period to smooth" is canonical. The
  exact 21 (Fibonacci) is a Murphy / practitioner convention — less canonical
  than the 9-period "sensitive" variants but defensible.
- **Action**: Document Murphy + Fibonacci attribution in the PRESETS dict.

---

## 6. Candlestick Pattern Ratios

### 6.1 Hammer pattern: lower shadow ≥ 2× body, upper shadow ≤ 0.3× body 🟡
- **Source**: Bulkowski, Thomas N. *Encyclopedia of Candlestick Charts*. Wiley,
  2008. Retrieved via thepatternsite.com on 2026-04-28
  (https://thepatternsite.com/Hammer.html) — direct quote from the
  "Identification Guidelines" table:
  > *"Look for the hammer to appear in a downward price trend and have a long
  > lower shadow at least two or three times the height of the body with
  > **little or no upper shadow**."*
- **Empirical performance** (Bulkowski): Hammer signals reversal 60% of the
  time; price meets the height-projected target 88% of the time on bull-market
  upward breakouts. Best performance: hammers within 1/3 of the yearly low.
- **Our values** (`candlestick_patterns.py:73,84`):
  - Lower shadow ≥ 2 × body ✅ matches Bulkowski's "two or three times"
  - Upper shadow ≤ 0.3 × body 🟡 — Bulkowski says "little or no upper shadow"
    (qualitative); we picked 0.3× body as a quantitative interpretation. A
    shadow ≤ 30% of body length matches the visual notion of "little or no"
    upper wick on a daily candle. **Defensible quantification of a qualitative
    rule; documented choice.**

### 6.2 Body size ratios: small ≤ 0.30, long ≥ 0.60, doji ≤ 0.10 🟡
- **Source**: Nison, Steve. *Japanese Candlestick Charting Techniques*
  (Prentice Hall, 1991/2001). The "rule of thirds" for candle body
  classification: small body ≤ 1/3 of range, large body ≥ 2/3 of range.
- **Our values** (`candlestick_patterns.py:48,52,59`):
  - Small body ratio = 0.30 ✅ matches Nison's 1/3 rule (rounded down from 0.333)
  - Long body ratio = 0.60 ✅ matches Nison's 2/3 rule (rounded down from 0.667)
  - Doji body ratio = 0.10 🟡 — Nison defines doji as "open ≈ close" without
    quantifying. The 5–15% range is practitioner consensus; we picked the high
    end (10%) for inclusivity.
- **Verdict**: Two of three are Nison-canonical (just rounded). Doji 10% is a
  practitioner pick within the accepted range.
- **Action**: Add Nison citation comment.

### 6.3 Hammer body-size threshold (`_is_small_body(m, ratio=0.35)`) 🟡
- **Note**: 0.35 is slightly looser than the 0.30 used elsewhere — gives
  hammers a bit more body-size tolerance.
- **Justification**: Bulkowski's spec doesn't quantify body size; he focuses on
  the shadow:body ratio. The 0.35 vs 0.30 delta lets a hammer with a slightly
  larger body still qualify if the shadow ratio is right.
- **Verdict**: Practitioner heuristic (intentionally permissive).
- **Action**: Document the choice.

---

## 7. Chart Pattern Heuristics

> **Academic anchor for this entire section**: Lo, A. W., Mamaysky, H., & Wang,
> J. (2000). "Foundations of Technical Analysis: Computational Algorithms,
> Statistical Inference, and Empirical Implementation." *Journal of Finance*
> 55(4):1705–1770. NBER Working Paper #7613, retrievable from
> https://www.nber.org/papers/w7613. Section II.A (Definitions 1–5) gives the
> precise geometric tolerances used by every detector below.

### 7.1 Triangle minimum pivots = 4 🟢
- **Source**: Murphy, *Technical Analysis* — a triangle requires at least 2
  pivots on each trendline to define both lines, so minimum 4 total.
- **Confirmed via**: StockCharts ChartSchool, *Symmetrical Triangle*. Retrieved
  2026-04-28 from https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-patterns/symmetrical-triangle
  — *"At least two points are required to form a trend line, and two trend
  lines are required to form a symmetrical triangle. There needs to be a
  minimum of four points to consider a formation as a symmetrical triangle."*
- **Our value**: 4 (`chart_patterns.py:detect_triangle min_pivots=4`).
- **Verdict**: Match.

### 7.2 Head & Shoulders shoulder + neckline tolerance = 1.5% 🟢 (LMW Def 1)
- **Source**: LMW (2000), Section II.A, Definition 1. Head-and-shoulders top
  is defined as 5 consecutive extrema E1…E5 satisfying:
  > E3 > E1, E3 > E5,  E1 and E5 within **1.5% of their average**,  AND
  > E2 and E4 within **1.5% of their average**.
- **What changed**: Pre-research the code used 5% shoulder tolerance and did
  NOT check the neckline-trough symmetry at all. Both fixed in this round
  to align with LMW.
- **Our values** (`chart_patterns.py`):
  - `_HS_SHOULDER_TOLERANCE = 0.015` (1.5%, LMW Def 1)
  - `_HS_NECKLINE_TOLERANCE = 0.015` (1.5%, LMW Def 1) ← NEW
- **Bulkowski empirical confirmation** (https://thepatternsite.com/HSTExplained.html):
  asymmetric H&S underperform symmetric — *"H&S tops with lower right shoulder
  see declines that average 25% compared to declines of 20% for higher right
  shoulders and 19% for even shoulders"*. Justifies tight symmetry tolerance.
- **Verdict**: Match LMW academic standard.

### 7.3 Double-top/bottom peak similarity = 1.5% + min separation = 22 bars 🟢 (LMW Def 5 + Edwards & Magee)
- **Source**: LMW (2000), Section II.A, Definition 5. Double top is defined
  as two tops Ea, Eb satisfying:
  > Ea and Eb within **1.5% of their average**, AND
  > separated by at least **22 trading days** (~1 calendar month) per
  > Edwards & Magee (1966), *Technical Analysis of Stock Trends*, 5th Ed.
- **What changed**: Pre-research the code used 3% peak tolerance and had no
  minimum-separation requirement. Both fixed in this round.
- **Our values** (`chart_patterns.py`):
  - `_DOUBLE_TOP_PEAK_TOLERANCE = 0.015` (1.5%, LMW Def 5)
  - `_DOUBLE_TOP_MIN_SEPARATION_BARS = 22` (Edwards & Magee 1966) ← NEW
- **Verdict**: Match LMW + Edwards & Magee academic standard.

### 7.4 Trough depth threshold = 5% 🟡
- **Context**: For a double-bottom (or double-top trough between peaks), the
  trough must be deep enough to count as a real correction (not noise).
- **Source**: LMW silent on this; practitioner heuristic (3–7% range typical).
- **Our value**: 5% (`chart_patterns.py:detect_double_top depth_score`).
- **Verdict**: Practitioner pick within accepted range. Defensible.
- **Action**: Document; 🔬 backtest in Phase 2.

### 7.5 Swing find distance = 5 bars 🟡
- **Context**: Used by `scipy.signal.find_peaks` to require pivots be at least
  N bars apart — prevents detecting micro-fluctuations as swings.
- **Practitioner convention**: 3–10 bars depending on timeframe. 5 is a
  reasonable middle ground for daily bars.
- **Our value**: 5 (`chart_patterns.py:_find_swing_highs/lows`).
- **Verdict**: Defensible heuristic. **Should ideally be ATR-scaled** (more
  volatile stocks deserve looser pivot detection) but constant is OK for v1.
- **Action**: Document; mark for ATR-scaling in Phase 2.

### 7.6 `DEFAULT_CONFIDENCE_THRESHOLD = 0.7` 🟡
- **Context**: Minimum confidence for surfacing a chart pattern to the LLM.
- **Justification**: Below 0.7, false-positive rate dominates (per Bulkowski's
  observation that pattern reliability rarely exceeds 80% even for the best
  patterns). 0.7 = "good enough that the LLM should consider it".
- **Verdict**: Reasonable cut-point. Document the rationale.

### 7.7 Triangle "flat line" tolerance = 0.75% (LMW Def 4 alignment) 🟢
- **Source**: LMW (2000), Section II.A, Definition 4 (Rectangle):
  > *Rectangle tops are characterized by a sequence of five consecutive local
  > extrema where E1, E3, E5 lie within **0.75% of their average** (the
  > horizontal resistance line) and E2, E4 lie within 0.75% of their average
  > (the horizontal support line).*
- **Why this applies to triangles**: An ascending triangle has a horizontal
  resistance line (same geometric construct as the rectangle's top) and a
  rising support line. A descending triangle is the mirror. We reuse LMW's
  0.75% spread tolerance for the "horizontal" side detection.
- **What changed**: Pre-research the code used `flat_threshold = 1e-4` and
  `rising_threshold = 5e-4` on normalized slopes — both invented constants.
  Replaced with `_FLAT_LINE_SPREAD_TOLERANCE = 0.0075` (LMW Def 4) plus a
  simple slope-sign check for rising/falling.
- **Our value**: `_FLAT_LINE_SPREAD_TOLERANCE = 0.0075` in
  `chart_patterns.py:_is_flat_line`.
- **Verdict**: Match LMW academic standard.

---

## 8. Levels & Lookbacks

### 8.1 Default swing lookback = 30 bars 🟡
- **Context**: How far back to look for swing high/low when computing
  support/resistance levels.
- **Practitioner consensus**: 20–60 bars depending on timeframe. 30 (~6 weeks
  of trading) is a common default for daily charts.
- **Our value**: 30 (`analysis/levels.py:16,73`).
- **Verdict**: Defensible. Document the rationale.

### 8.2 Data fetch lookback = 400 bars 🟡
- **Context**: How many trading days of historical data to pull.
- **Justification**: 400 trading days ≈ 1.6 calendar years — enough for:
  - 200-SMA calculation (the longest indicator we use) ✓
  - Swing levels with 60-bar lookback (smooth preset) ✓
  - 1-year (252-day) high/low computation ✓
  - ~150 bars of buffer for unusual indicator settings
- **Our value**: 400 (`inputs.py:63` and tool defaults).
- **Verdict**: Defensible. Could be reduced to 252+max_indicator_period (≈260)
  to save bandwidth, but 400 leaves headroom.

### 8.3 `BREAKOUT_EXCLUDE_BARS = 3` 🔴 → 🔬
- **Context**: When detecting breakout from a consolidation, exclude the most
  recent N bars from the consolidation range to avoid look-ahead.
- **Source**: None — practitioner heuristic.
- **Action**: 🔬 NEEDS BACKTEST. Document as v1.

### 8.4 `PATTERN_LOOKBACK_BARS = 5` (candlestick recent-pattern detection) 🟡
- **Context**: Only consider candlestick patterns formed in the last 5 bars —
  older signals are stale.
- **Practitioner consensus**: Candlestick signals lose predictive value within
  ~3–7 bars (Nison, Bulkowski).
- **Our value**: 5 (`get_momentum.py:52`).
- **Verdict**: Within accepted range.

---

## 9. Guardrail Tolerances 🔴 → 🔬 (POLICY DECISIONS)

These constants police LLM output (rejecting "hallucinated" predictions). They
are **policy decisions**, not derived from technical-analysis literature.
Justification must come from our own backtesting (Phase 2).

| Constant | Value | Justification (v1) |
|----------|------:|---------------------|
| `TARGET_ANCHOR_TOLERANCE_PCT` | 0.005 (0.5%) | Target must round to a recognizable level (ATR multiple or S/R within 0.5%) |
| `STOP_MIN_ATR` | 0.7 | Below 0.7×ATR, stops get hit by intraday noise |
| `STOP_MAX_ATR` | 1.8 | Above 1.8×ATR, R:R typically dips below 1:1 |
| `ENTRY_ZONE_TOLERANCE_PCT` | 0.01 (1%) | Entry zone must be within 1% of close (not "buy at $X" when close is $1.5X) |
| `ATR_MULTIPLIERS` | (1.0, 1.5, 2.0, 2.5, 3.0) | Allowed ATR-multiple targets — anything else is rounded |

- **Verdict**: All five are 🔬 NEEDS BACKTEST. The values are reasonable v1
  heuristics — but for a "research-grounded" project they MUST be derived from
  the realized hit-rate distribution on backtested predictions.
- **Action**: **Reconcile with the per-horizon multipliers we'll add for the
  synthesizer prompt.** It's incoherent to say "stop must be 0.7–1.8×ATR" at
  the guardrail layer while also saying "monthly horizon uses 2.0×ATR stop"
  in the prompt. The guardrails need to either:
  - (a) Become per-horizon (0.5–1.0×ATR for daily, 1.5–2.5×ATR for monthly)
  - (b) Widen to a global envelope that covers all horizons (0.5–2.5×ATR)
  - (c) Be removed and replaced with horizon-specific guardrails
- **Recommendation**: (a) per-horizon guardrails. Cleanest match to the
  per-horizon prompt logic.

---

## 10. Grading Tolerances

### 10.1 `NEUTRAL_TOLERANCE_BY_HORIZON` (per-horizon: 2/4/6/8%) 🟡 ✅ FIXED 2026-04-28
- **Was**: Single constant `NEUTRAL_TOLERANCE = 0.02` applied to all horizons.
- **Bug**: A 2% band that's tight for daily becomes effectively zero for
  monthly. Stocks drift roughly with sqrt(t); a NIFTY 50 large-cap with ~1.5-2%
  daily volatility will move >2% over a month from random walk alone, even with
  zero news/momentum. Monthly NEUTRAL predictions were systematically graded as
  wrong even when the prediction was actually correct.
- **Fix**: Per-horizon dict, sqrt(t) scaling rounded to even integers:
  - DAILY (1d):    2%   (baseline ~ 1 std daily move)
  - WEEKLY (5d):   4%   (2% * sqrt(5)  ~ 4.5%, rounded down)
  - BIWEEKLY (10d): 6%  (2% * sqrt(10) ~ 6.3%, rounded down)
  - MONTHLY (21d): 8%   (2% * sqrt(21) ~ 9.2%, rounded down for caution)
- **Public API**: `neutral_tolerance(horizon)` lookup function added so tests
  and calibration reports stay in sync with the grader.
- **Test coverage**: Parametrized over all PredictionHorizon enum values; new
  `test_tolerance_grows_with_horizon` and `test_tolerance_table_covers_all_horizons`
  guard against regressions.
- **Status**: 🟡 v1 values are sqrt(t)-anchored; final values pending Phase 2
  backtest against actual NIFTY 50 daily-return distributions per horizon.

### 10.2 `_FETCH_BUFFER_MULT = 1.7`, `_FETCH_BUFFER_FLOOR = 3` 🟢 (calendar math)
- **Context**: When grading, fetch enough calendar days to cover N trading
  days (accounting for weekends + holidays).
- **Math**: 5 trading days = 7 calendar days = ratio 1.4. Adding buffer for
  Indian-market holidays (~12/year ≈ 1 per month) → 1.7 is the right
  multiplier. Floor of 3 days handles edge cases (single-day predictions
  immediately preceding long weekends).
- **Verdict**: Calendar fact. No change needed.

---

## 11. News Impact Agent

### 11.1 Default lookback windows 🟡
- News default: 7 days. **Justification**: Standard event-study window for
  short-term news price impact. References: MacKinlay (1997), *"Event Studies
  in Economics and Finance"*, JEL 35(1):13-39. Most published equity news
  studies use 5–10 trading days for short-window impact.
- Filings default: 30 days. **Justification**: ≈1 month, captures regulatory
  filing cycle.
- Prices default: 30 days. **Justification**: ≈1 month context for the LLM
  agent to assess recent volatility / trend.
- `_MIN_DAYS=1, _MAX_DAYS=90`: 90-day cap matches event-study long-window
  convention (MacKinlay).
- **Verdict**: All defensible per event-study literature. Document.

### 11.2 `estimated_pct_move` cap = ±30% 🟡
- **Context**: Bounds the LLM-estimated single-event price impact to prevent
  hallucinated 1000% moves.
- **NSE context**: Indian individual securities have daily price bands of 2%,
  5%, 10%, or 20% depending on category. F&O-eligible stocks typically have
  no daily band (or very wide bands), and news events can cause >20% moves
  on these.
- **Verdict**: 30% is a reasonable upper bound for F&O-eligible large caps
  (e.g. major earnings surprises, M&A announcements can move stocks 15–25%
  in a single session). Document the rationale.

---

## 12. PROPOSED Per-Horizon Synthesizer Constants 🔬 NEEDS BACKTEST

These were the constants I was about to invent for commit 4 of the
multi-horizon refactor. After this research pass, here's what I now know:

### 12.1 Stop multipliers per horizon
- **Literature anchor**: Wilder (1978), Van Tharp (*Trade Your Way to Financial
  Freedom*), Murphy (*Technical Analysis*) — all use 1×ATR as the canonical
  swing-trading stop. Wider multiples (2–3×ATR) for longer horizons; tighter
  (~0.5×ATR) for intraday.
- **Theory**: Optimal stop scales as √t for Brownian motion (1, 2.2, 3.2, 4.6
  for 1/5/10/21 trading days), but practitioner experience shows linear-ish
  scaling matches realized risk tolerance better.
- **Proposed v1 values** (linear interpolation between literature endpoints):
  - Daily (1d): 0.75×ATR — tighter than Wilder's 1× (matches intraday practice)
  - Weekly (5d): 1.0×ATR — Wilder's standard
  - Biweekly (10d): 1.5×ATR — interpolated
  - Monthly (21d): 2.0×ATR — Van Tharp's longer-horizon swing standard
- **Verdict**: 🔬 v1 is literature-bracketed; exact numbers per horizon need
  backtest validation in Phase 2.

### 12.2 Target multipliers per horizon
- **Literature**: R:R ratio of 1.5–2.0 is the swing-trading consensus (Murphy,
  Tharp, Pring) for positive expectancy at 50% hit rate.
- **Proposed**: target = 1.3–1.5 × stop multiplier per horizon.
- **Verdict**: 🔬 v1 is literature-bracketed.

### 12.3 Entry zone widths per horizon
- **Literature**: No canonical source for "entry tolerance window" in
  practitioner books — most assume single-price entry.
- **Justification**: Pure design choice — wider for longer horizons because
  exact intraday timing matters less.
- **Verdict**: 🔴 → 🔬 NEEDS BACKTEST. Pure heuristic. Mark explicitly as v1.

### 12.4 Confidence caps per horizon
- **Literature**: NONE. LLM-driven prediction is a novel use case; no
  practitioner book or academic paper specifies confidence ceilings for
  multi-horizon LLM predictions.
- **Verdict**: 🔬 NEEDS BACKTEST. Cannot be literature-justified. Mark
  explicitly as "v1, recalibrate from realized hit-rate per horizon".

### 12.5 Tie-break (technicals vs news) per horizon
- **Literature**: Event-study literature (MacKinlay 1997, Fama 1991) supports
  the principle that news impact has a finite half-life — short-window (1–5
  days) news effect is real but quickly absorbed; long-window (>1 month) news
  effect dominates over technical patterns.
- **Verdict**: The PRINCIPLE (technicals win short-term, news wins long-term)
  is literature-backed. The specific tie-break thresholds are heuristics.

---

## 13. Operational Constants ⚪ (low priority, not analytical)

| Constant | Value | Note |
|----------|------:|------|
| HTTP timeouts | 10–15s | Standard web-API conservative defaults |
| Cache TTL | 365 days | Historical OHLCV doesn't change |
| Concurrency | 3 | Gentle on Gemini free-tier rate limits |
| Cooldown windows | 60s, 1h | Standard rate-limit backoff |

No literature; document as engineering defaults.

---

## Summary of Verdicts

| Category | Total | Already canonical | Now grounded | Still needs backtest |
|----------|------:|-----------------:|-------------:|--------------------:|
| Indicator periods (§1) | 9 | 9 | 0 | 0 |
| Indicator thresholds (§2) | 4 | 3 | 1 | 0 |
| %B / ATR%  (§3) | 4 | 1 | 1 | 2 |
| OBV (§4) | 1 | 0 | 0 | 1 |
| Sensitivity presets (§5) | 11 | 0 | 9 | 2 |
| Candlesticks (§6) | 5 | 2 | 3 | 0 |
| Chart patterns (§7) | 7 | **5** | 1 | 1 |
| Levels/lookbacks (§8) | 4 | 0 | 3 | 1 |
| **Guardrails (§9)** | 5 | 0 | 0 | **5 (policy)** |
| Grading (§10) | 3 | 1 | 1 | 1 |
| News impact (§11) | 5 | 0 | 5 | 0 |
| **Pending synth (§12)** | ~12 | 0 | 6 | **~6** |
| **TOTAL** | **~70** | **21** | **30** | **~19** |

**Net change after 3 research rounds**: We went from "37 🔴 vibes" to **19 🔬
backtest items** + **30 newly-grounded constants** + **21 canonical**. The
remaining 19 🔬 split as:

- 5 guardrails (policy decisions about LLM output validation)
- ~6 pending synthesizer per-horizon constants (literature-bracketed; need
  empirical fine-tuning)
- ~8 trading heuristics (%B dead-band, OBV magnitude, hammer body, swing find
  distance, etc.) where literature is qualitative and practitioner consensus
  exists but exact numbers are derived

### Round-3 promotions (2026-04-28)

Three items promoted from 🟡 to 🟢 after research-round-3:

1. **Bollinger Bands 20/2.0** — found exact citation on John Bollinger's own
   site ("defaults today are the same as they were 35 years ago, 20 periods...
   plus and minus two standard deviations"). TA-Lib's default of 5 was the
   outlier; we follow Bollinger himself. (§1.5)
2. **Stochastic %K = 14** — confirmed canonical via StockCharts ("The default
   setting for the Stochastic Oscillator is 14 periods"). TA-Lib's default of
   5 is the "Fast Stochastic" variant, not the practitioner default. (§1.6)
3. **Triangle "flat line" detection** — replaced invented `1e-4` /  `5e-4`
   normalized-slope thresholds with LMW (2000) Definition 4's 0.75% pivot-
   spread tolerance for horizontal-line detection (same construct as LMW's
   Rectangle definition). (§7.7)

Plus two earlier-round chart pattern fixes (§7.2, §7.3): tightened H&S and
double-top tolerances from 5%/3% to LMW's 1.5%, added the missing H&S neckline
symmetry check, and added Edwards & Magee's 22-bar minimum separation for
double tops/bottoms.

---

## Phase 2 Backtest Project (deferred)

The 20 🔬 NEEDS BACKTEST items will be tackled in a separate project after the
current multi-horizon refactor lands. Approach:

1. Build a "rewind-clock" harness in `backtest/` (currently empty) that runs
   `predict()` with `as_of=<historical date>` instead of `now()`.
2. Enforce point-in-time data discipline (no look-ahead).
3. Run the harness on N×M sample (N = NIFTY 50 tickers, M = sampled historical
   dates over 2–5 years).
4. Aggregate realized outcomes via existing `grading.py`.
5. For each 🔬 item, compute the empirical optimum from the outcome
   distribution.
6. Update this dossier with empirical findings + update the code.

---

## How to use this dossier when changing code

1. Touching a numerical constant? **Find it in this dossier first.** If it's
   not here, that's a bug — add it before changing anything.
2. Adding a new constant? **Add a section here first**, with research and
   citations. Then add the code with a comment like:
   ```python
   RSI_OVERBOUGHT = 70  # Wilder (1978); see docs/research/constants_dossier.md §2.1
   ```
3. Changing a literature-canonical value? **You shouldn't.** If you must, add
   a section here explaining the deviation.
4. A 🔬 item gets backtested? Update the entry to 🟢 or 🟡 with the empirical
   evidence; update the code citation comment.
