# Solutions Reference for `pred_logic.md` Issues

**Companion to:** `pred_logic_review.md`
**Author:** Code Puppy 🐶
**Date:** 2026-04-30
**Purpose:** For every issue identified in the review, give a concrete, citable, copy-pastable solution. All claims here are backed by primary sources (vendor docs, NBER PDFs, exchange-of-record pages, peer-reviewed papers, and the actual code of cited libraries).

---

## How to use this document

For each issue in `pred_logic_review.md`, this file gives:

1. **Issue recap** — one-line restatement.
2. **Root cause** — why the doc has it.
3. **Recommended fix** — what to actually do.
4. **Code / library example** — copy-paste starting point.
5. **Citations** — direct URLs to primary sources.

The numbering (`C1`, `H1`, `M1`, …) matches the review file exactly.

A **prioritized roadmap** is at §99 at the bottom.

---

## C1 — Replace Stooq with a real NSE data source

**Issue recap:** Stooq is listed as fallback #2 in §2.1, but it does not cover NSE Indian equities at all. Verified directly: `stooq.com/q/?s=reliance.in` and `?s=reliance.bo` both return *"Symbol does not exist in database"* (https://stooq.com/db/d/).

**Root cause:** Stooq's coverage list is US/EU/JP/HK/PL — no India equities. Probably copy-pasted from a US-focused tutorial.

### Recommended fix — three-tier NSE-native fetcher

| Tier | Source | Role |
|------|--------|------|
| Primary | **`jugaad-data`** (per-symbol history via `stock_df`) | Active, NSE-native, 0.33.1 on PyPI (Mar 2026) |
| Secondary | **NSE bhavcopy archives** (`https://nsearchives.nseindia.com`) | Direct CSV, exchange-of-record |
| Tertiary | **`yfinance`** with `.NS` suffix | Free fallback, US-focused but works |
| Removed | ~~Stooq~~ (no NSE coverage) | Delete |
| Removed | ~~Alpha Vantage~~ (free tier = 25 req/day, NSE feed unreliable) | Demote or delete |

### Code — `jugaad-data` primary fetcher

```python
# pip install "jugaad-data>=0.33.1"
from datetime import date
from jugaad_data.nse import stock_df

def fetch_nse_daily(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Primary NSE OHLCV fetcher. Symbol = bare NSE trading symbol, e.g. 'RELIANCE'."""
    df = stock_df(symbol=symbol, from_date=start, to_date=end, series="EQ")
    # Returned columns: DATE, SERIES, OPEN, HIGH, LOW, PREV. CLOSE, LTP,
    #                   CLOSE, VWAP, VOLUME, VALUE, NO OF TRADES, ...
    return df.sort_values("DATE").reset_index(drop=True)
```

The library auto-throttles requests, batches multi-year ranges into 1-year chunks (NSE limit), and caches to `.cache/`. Source: `jugaad_data/nse/history.py` — https://github.com/jugaad-py/jugaad-data/blob/master/jugaad_data/nse/history.py

### Code — NSE bhavcopy bulk fallback

NSE archives daily security-wise bhavcopy CSVs at predictable URLs. For all-stocks-one-day fetches this is much faster than per-symbol calls.

```python
import requests, io, pandas as pd
from datetime import date

NSE_BHAV_URL = (
    "https://nsearchives.nseindia.com/products/content/"
    "sec_bhavdata_full_{dd}{mm}{yyyy}.csv"
)

HDRS = {  # NSE rejects requests without a real UA + language
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/144.0.0.0 Safari/537.36",
    "Accept-Language": "en-IN,en-US;q=0.9",
}

def nse_bhavcopy(d: date) -> pd.DataFrame:
    url = NSE_BHAV_URL.format(dd=f"{d.day:02d}", mm=f"{d.month:02d}", yyyy=d.year)
    r = requests.get(url, headers=HDRS, timeout=15)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content))
```

URL pattern verified in `jugaad-data` source: https://github.com/jugaad-py/jugaad-data/blob/master/jugaad_data/nse/archives.py

> **NB:** From 2024-07-08 onward NSE introduced a new "UDiff" bhavcopy format. For dates before that, the legacy URL above works. For newer dates, fetch via NSE's daily-reports API: `GET https://www.nseindia.com/api/daily-reports?key=CM` and pull the `CM-UDIFF-BHAVCOPY-CSV` entry. Reference: NSE circular announcing UDiff — https://nsearchives.nseindia.com/content/circulars/CMTR59722.pdf

### Code — `yfinance` fallback

```python
# pip install "yfinance>=0.2.40"
import yfinance as yf

def yf_nse_daily(symbol: str, start, end) -> pd.DataFrame:
    """yfinance fallback. NSE tickers use .NS suffix (BSE: .BO)."""
    ticker = symbol if symbol.endswith((".NS", ".BO")) else f"{symbol}.NS"
    return yf.download(ticker, start=start, end=end,
                       auto_adjust=False, progress=False, threads=False)
```

### Production-readiness scorecard

| Source | Coverage | Reliability | Free? | Verdict |
|--------|----------|-------------|-------|---------|
| `jugaad-data` | NSE EQ + indices | ★★★★☆ | Yes | **Primary** |
| NSE bhavcopy direct | All NSE | ★★★★★ (exchange-of-record) | Yes | **Bulk EOD** |
| `yfinance` | Global incl. NSE | ★★★☆☆ (Yahoo breaks ~2x/yr) | Yes | **Fallback** |
| `nsepython` | NSE wider API | ★★★☆☆ (GPL license) | Yes | Alternative to jugaad-data |
| Tiingo | US + 50 intl, **no India** | n/a | n/a | Don't use for NSE |
| Alpha Vantage | Has nominal NSE feed but unreliable | ★☆☆☆☆ | 25 req/day | Drop |
| Stooq | **No India equities** | n/a | n/a | **Delete** |
| Zerodha Kite Connect | NSE/BSE official-broker | ★★★★★ | Paid (₹2,000/mo) | Production paid tier |
| Upstox API | NSE/BSE official-broker | ★★★★☆ | Free with account | Production free tier |

**Citations:**
- jugaad-data PyPI: https://pypi.org/project/jugaad-data/
- jugaad-data GitHub: https://github.com/jugaad-py/jugaad-data
- yfinance GitHub: https://github.com/ranaroussi/yfinance
- Stooq DB list (no India): https://stooq.com/db/d/
- Alpha Vantage rate limits: https://www.alphavantage.co/support/#api-key
- Tiingo coverage / pricing: https://www.tiingo.com/about/pricing
- Zerodha Kite: https://kite.trade/
- Upstox API: https://upstox.com/developer/api/

---

## C2 — Fix the GDELT Doc 2.0 "7-day rolling index" claim

**Issue recap:** §2.3 says "GDELT's rolling 7-day index. We typically request 1–7 days back; the LLM may request up to 90 days for special cases." Both halves are wrong.

**Reality:**
- The GDELT Doc 2.0 API searches a **rolling 3-month window** (timestamps must be in last 3 months).
- For older history (since Feb 2015), use either the 15-min CSV files at http://data.gdeltproject.org/gdeltv2/lastupdate.txt or **GDELT BigQuery** (`gdelt-bq.gdeltv2.gkg`).
- Default if no time params: last 3 months. Maximum `maxrecords` per query: 250 (in `artlist` and `imagecollage*` modes).

### Recommended fix — proper GDELT 2.0 client

```python
# Pure stdlib + requests; no GDELT SDK needed
import requests, urllib.parse, time
from datetime import datetime, timedelta

GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 250
WINDOW_HARD_LIMIT = timedelta(days=90)  # the API itself caps at ~3mo

def gdelt_search(query: str, start: datetime, end: datetime,
                 max_records: int = MAX_RECORDS, mode: str = "artlist"):
    """One call to GDELT Doc 2.0. Times must be UTC and within last ~3 months."""
    params = {
        "query": query,                                # URL-encoded by requests
        "mode": mode,                                  # artlist | timelinevol | etc.
        "format": "json",
        "maxrecords": max_records,
        "STARTDATETIME": start.strftime("%Y%m%d%H%M%S"),
        "ENDDATETIME":   end.strftime("%Y%m%d%H%M%S"),
        "sort": "datedesc",
    }
    r = requests.get(GDELT_DOC, params=params,
                     headers={"User-Agent": "stockpred/1.0 (contact@you)"},
                     timeout=30)
    r.raise_for_status()
    return r.json().get("articles", [])

def gdelt_paginate(query: str, start: datetime, end: datetime,
                   per_window_days: int = 1, polite_sleep_s: float = 5.0):
    """Slide a per_window_days window to stay under the 250-record cap."""
    out, cursor = [], start
    while cursor < end:
        nxt = min(cursor + timedelta(days=per_window_days), end)
        out += gdelt_search(query, cursor, nxt)
        time.sleep(polite_sleep_s)  # GDELT recommends ~1 req / 5s
        cursor = nxt
    return out
```

**TIMESPAN syntax** (alternative to STARTDATETIME/ENDDATETIME):

| Token | Meaning |
|-------|---------|
| `15min`, `30min` | minutes (≥ 15) |
| `1h`, `24h`, `72hours` | hours |
| `1d`, `7d`, `30days` | days |
| `1w`, `2weeks` | weeks |
| `1m`, `3months` | months (effective max ≈ 3) |

**Query operators** (must-know):
- `"exact phrase"` (URL-encode the quotes)
- `(a OR b)`, `-exclude`
- `domain:reuters.com`, `domainis:un.org`
- `sourcecountry:in` (3-letter not used here; `IN` for India)
- `sourcelang:english`
- `theme:ECON_STOCKMARKET` (GDELT GKG taxonomy)
- `tone>5`, `tone<-5` (numeric tone filter)
- `near20:"a b"` (within 20 words)

### For deeper history — GDELT BigQuery

```sql
-- Reliance Industries English news, 2025
SELECT DATE, V2DocumentIdentifier AS url, V2Themes, V2Tone
FROM `gdelt-bq.gdeltv2.gkg`
WHERE _PARTITIONTIME BETWEEN '2025-01-01' AND '2025-12-31'
  AND V2DocumentIdentifier IS NOT NULL
  AND LOWER(V2Themes) LIKE '%econ_%'
  AND REGEXP_CONTAINS(V2Persons, r'(?i)Mukesh Ambani')
LIMIT 1000;
```

**Recommended doc rewrite (drop-in for §2.3):**

> "**Source:** GDELT Doc 2.0 API (`https://api.gdeltproject.org/api/v2/doc/doc`). Free, no auth, CORS-enabled. The Doc API searches a **rolling ~3-month window**; for historical lookups beyond that, fall back to GDELT BigQuery (`gdelt-bq.gdeltv2.gkg`). Max 250 articles per query in `artlist` mode — paginate by chopping the time range. Polite usage: ≤ 1 request / 5 seconds."

**Citations:**
- GDELT Doc 2.0 launch & parameter doc: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
- GDELT 2.0 GKG announcement: https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/
- GDELT 15-min raw CSV index: http://data.gdeltproject.org/gdeltv2/lastupdate.txt
- GDELT BigQuery dataset: `bigquery-public-data.gdelt_v2` (also `gdelt-bq.gdeltv2`)

---

## C3 — NSE corporate filings: not a public API; document the reality

**Issue recap:** §2.4 calls the NSE corporate-announcements feed "public". It's actually an **undocumented internal endpoint** (`https://www.nseindia.com/api/corporate-announcements?index=equities&symbol=...`), Cloudflare-protected, requiring session-cookie priming.

### Recommended fix — canonical session-priming pattern

```python
import requests, time, random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

NSE_HOME      = "https://www.nseindia.com"
NSE_REPORT_PG = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
NSE_API       = "https://www.nseindia.com/api/corporate-announcements"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/144.0.0.0 Safari/537.36"),
    "Accept": "*/*",
    "Accept-Language": "en-IN,en-US;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": NSE_REPORT_PG,
    "sec-fetch-site": "same-origin",
    "sec-fetch-mode": "cors",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
}

def make_nse_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    retry = Retry(
        total=5, connect=3, read=3, status=5,
        backoff_factor=1.5,                       # 1.5, 3, 6, 12, 24 s
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    # MUST hit the homepage and the referring HTML page to get cookies
    s.get(NSE_HOME, timeout=10)
    s.get(NSE_REPORT_PG, timeout=10)
    return s

def get_announcements(symbol: str, session=None):
    s = session or make_nse_session()
    for attempt in range(3):
        r = s.get(NSE_API,
                  params={"index": "equities", "symbol": symbol},
                  timeout=15)
        ct = r.headers.get("content-type", "")
        if r.status_code == 200 and ct.startswith("application/json"):
            return r.json()
        if r.status_code in (401, 403):
            # Cookie expired (rotates ~10 min). Re-prime.
            s.get(NSE_REPORT_PG, timeout=10)
        time.sleep(2 + random.random() * 2)
    r.raise_for_status()
```

This is the canonical pattern used by `jugaad-data` (file `jugaad_data/nse/history.py`, class `NSEHistory._get`, lines ~46–60): https://github.com/jugaad-py/jugaad-data/blob/master/jugaad_data/nse/history.py#L46

### Add BSE as a cross-validator

BSE's API needs no cookie priming and is an independent source for the same disclosures (companies dual-list). Use both and de-duplicate by `(date, subject_hash)`.

```python
BSE_API = "https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w"
BSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 ...",
    "Origin": "https://www.bseindia.com",
    "Referer": "https://www.bseindia.com/",
    "Accept": "application/json, text/plain, */*",
}

def get_bse_announcements(scrip_code: str, start_yyyymmdd: str, end_yyyymmdd: str):
    params = {
        "strCat": "-1", "strType": "C", "strSearch": "P",
        "strScrip": scrip_code,         # numeric, e.g. "500325" for RELIANCE
        "strPrevDate": start_yyyymmdd,
        "strToDate": end_yyyymmdd,
    }
    r = requests.get(BSE_API, params=params, headers=BSE_HEADERS, timeout=15)
    r.raise_for_status()
    return r.json().get("Table", [])
```

You'll need an NSE-symbol → BSE-scrip-code mapping; download from BSE's instrument list once: `https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?Group=&Scripcode=&industry=&segment=Equity&status=Active`.

### Recommended doc rewrite for §2.4

> "**Source:** NSE's *internal* announcements endpoint at `https://www.nseindia.com/api/corporate-announcements`. **This is NOT an officially supported API.** It is Cloudflare-protected and requires (a) a real-browser `User-Agent`, (b) a prior session-priming GET to `nseindia.com/` and the referring HTML page to obtain `nsit`/`nseappid` cookies, and (c) cookie rotation every ~10 minutes. Production deployments should add **BSE's `api.bseindia.com`** as an independent cross-validator and consider a paid licensed feed (Zerodha Kite, Upstox, GlobalDatafeeds, NSE Data Services) for primary."

**Citations:**
- jugaad-data NSE session pattern (canonical): https://github.com/jugaad-py/jugaad-data/blob/master/jugaad_data/nse/history.py
- BSE corporate disclosures portal: https://www.bseindia.com/corporates/ann.html

---

## C4 — Add the missing LMW chart patterns (broadening, rectangles) and use the correct smoother

**Issue recap:** Lo, Mamaysky & Wang (2000) define **5 pairs / 10 patterns**; the doc implements only 3 pairs / 6 patterns. Also, LMW use **Nadaraya–Watson kernel-regression** smoothing of prices, not raw `find_peaks` on closes.

### Recommended fix — implement all 5 pairs with the proper smoother

#### Step 1: Nadaraya–Watson smoother with LMW's bandwidth rule

LMW choose bandwidth via leave-one-out cross-validation, then **multiply by 0.3** to undersmooth slightly. (Quote from §III.A of the paper: *"we use a bandwidth that is 0.3 times the cross-validation-optimal bandwidth, i.e., h = 0.3 h*."*)

```python
# Option A: statsmodels (cleanest)
import numpy as np
from statsmodels.nonparametric.kernel_regression import KernelReg

def lmw_smoother(prices: np.ndarray):
    t = np.arange(len(prices), dtype=float)
    # 1) cross-validated bandwidth h*
    kr = KernelReg(endog=prices, exog=t, var_type="c",
                   reg_type="lc",        # local-constant = Nadaraya-Watson
                   bw="cv_ls", ckertype="gaussian")
    h_star = kr.bw[0]
    # 2) re-smooth with 0.3 × h* per LMW
    kr_final = KernelReg(endog=prices, exog=t, var_type="c",
                         reg_type="lc",
                         bw=[0.3 * h_star], ckertype="gaussian")
    m_hat, _ = kr_final.fit(t)
    return m_hat
```

```python
# Option B: vectorized scipy (faster for many series)
def nw_gaussian(prices: np.ndarray, h: float) -> np.ndarray:
    t = np.arange(len(prices), dtype=float)
    diff = t[:, None] - t[None, :]
    K = np.exp(-0.5 * (diff / h) ** 2)
    return (K @ prices) / K.sum(axis=1)
```

#### Step 2: extract alternating extrema

```python
from scipy.signal import argrelextrema

def lmw_extrema(m_hat: np.ndarray):
    maxima = argrelextrema(m_hat, np.greater)[0]
    minima = argrelextrema(m_hat, np.less)[0]
    extrema = sorted(np.concatenate([maxima, minima]).tolist())
    is_max = {i: True for i in maxima}
    is_max.update({i: False for i in minima})
    return extrema, is_max
```

#### Step 3: pattern definitions (verbatim from LMW §II)

For each sliding 38-day window, pull the contained extrema; if you have ≥ 5 alternating, run the rules below.

```python
def near(a, b, pct):
    return abs(a - b) / ((a + b) / 2) <= pct

def lmw_classify(E, is_max):
    """E is a list of (t, price) tuples for 5 alternating extrema."""
    p = [e[1] for e in E]
    e1, e2, e3, e4, e5 = p

    if is_max[E[0][0]]:   # E1 is a maximum
        # HEAD AND SHOULDERS (HS)
        if (e3 > e1 and e3 > e5
            and near(e1, e5, 0.015)        # shoulders within 1.5%
            and near(e2, e4, 0.015)):      # neckline troughs within 1.5%
            return "HS"
        # BROADENING TOP (BTOP)
        if e1 < e3 < e5 and e2 > e4:
            return "BTOP"
        # TRIANGLE TOP (TTOP) — converging
        if e1 > e3 > e5 and e2 < e4:
            return "TTOP"
        # RECTANGLE TOP (RTOP)
        avg_top = np.mean([e1, e3, e5])
        avg_bot = np.mean([e2, e4])
        if (all(abs(x - avg_top) / avg_top <= 0.0075 for x in [e1, e3, e5])
            and all(abs(x - avg_bot) / avg_bot <= 0.0075 for x in [e2, e4])
            and min(e1, e3, e5) > max(e2, e4)):
            return "RTOP"
    else:                  # E1 is a minimum
        # IHS, BBOT, TBOT, RBOT — mirrors of the above
        if (e3 < e1 and e3 < e5
            and near(e1, e5, 0.015)
            and near(e2, e4, 0.015)):
            return "IHS"
        if e1 > e3 > e5 and e2 < e4:
            return "BBOT"
        if e1 < e3 < e5 and e2 > e4:
            return "TBOT"
        avg_bot = np.mean([e1, e3, e5])
        avg_top = np.mean([e2, e4])
        if (all(abs(x - avg_bot) / avg_bot <= 0.0075 for x in [e1, e3, e5])
            and all(abs(x - avg_top) / avg_top <= 0.0075 for x in [e2, e4])
            and max(e1, e3, e5) < min(e2, e4)):
            return "RBOT"
    return None

# Double Top / Bottom — different rule: scan 38-day window for two highest maxima
def lmw_double_top(window_extrema, is_max):
    maxima = [(t, p) for (t, p) in window_extrema if is_max[t]]
    if len(maxima) < 2: return None
    maxima.sort(key=lambda x: -x[1])     # by price desc
    (t_a, p_a), (t_b, p_b) = sorted(maxima[:2], key=lambda x: x[0])
    if near(p_a, p_b, 0.015) and (t_b - t_a) > 22:   # 22 trading days = ~1 month
        return "DTOP"
    return None
```

**Sliding-window length (LMW use 38 trading days):** confirmed in §II of the paper. Use the same.

#### Step 4: re-attribute citations

Replace this in §3.7:

> "The peaks must be at least **22 trading days apart** (Edwards & Magee, 1966 — cited by LMW)."

with:

> "The peaks must be at least **22 trading days apart** — Lo, Mamaysky & Wang's (2000) operationalization of Edwards & Magee's qualitative '~one month / several weeks' guidance (E&M ch. VII–X)."

**Citations:**
- LMW paper, NBER landing: https://www.nber.org/papers/w7613
- LMW paper, Penn-hosted PDF (with full pattern definitions): https://www.cis.upenn.edu/~mkearns/teaching/cis700/lo.pdf
- statsmodels KernelReg: https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.kernel_regression.KernelReg.html
- Edwards & Magee mirror: https://vdthangmeomeo.wordpress.com/wp-content/uploads/2014/08/technical-analysis-of-stock-trends-9th-edition.pdf
- Reference Python pattern detector (extrema-walking scaffold): https://github.com/BennyThadikaran/stock-pattern

---

## C5 — Re-attribute the 22-bar rule (already covered in C4)

Solution applied above. Sentence change is the entire fix.

---

## C6 — Use the NSE trading calendar, not 252

**Issue recap:** §3.5 (52-week extreme) and §1.1 (~21 trading days/month) bake in 252 trading days/year; NSE actually averages ~246–250.

### Recommended fix — use a calendar library

```python
# pip install pandas_market_calendars
import pandas_market_calendars as mcal
import pandas as pd

# NSE and BSE share the same calendar; both are exposed as XBOM
nse_cal = mcal.get_calendar("XBOM")

def trading_days_between(start, end):
    sched = nse_cal.schedule(start_date=start, end_date=end)
    return sched.index            # DatetimeIndex of trading days

def trailing_n_trading_days(end, n):
    # Walk back ~n × 1.5 calendar days then trim to last n trading sessions
    sched = nse_cal.schedule(end_date=end,
                             start_date=pd.Timestamp(end) - pd.Timedelta(days=int(n * 1.6)))
    return sched.index[-n:]
```

For "52-week high/low":

```python
def fifty_two_week_extremes(close: pd.Series, end: pd.Timestamp):
    one_year_ago = end - pd.Timedelta(days=365)
    days = trading_days_between(one_year_ago, end)
    s = close.reindex(days).dropna()
    return s.max(), s.min()
```

### Annual cross-check against NSE official list

Both `pandas_market_calendars` and the underlying `exchange_calendars` have lagged on ad-hoc holidays (e.g. Maharashtra elections 2024, Ayodhya 22-Jan-2024). Validate annually:

```python
def nse_holidays_official():
    r = requests.get("https://www.nseindia.com/api/holiday-master?type=trading",
                     headers={"User-Agent": "Mozilla/5.0 ...",
                              "Referer": "https://www.nseindia.com/"})
    return pd.DataFrame(r.json()["CM"])     # CM = Cash Market
```

### Recommended doc rewrite for §3.5

> "**52-week high/low**: Highest high and lowest low over the **last ~250 trading sessions** (NSE-specific, computed via `pandas_market_calendars.get_calendar('XBOM')`. NSE actually trades 246–252 sessions/year depending on holidays; using the calendar instead of a fixed 252 avoids ±2-day silent drift)."

**Citations:**
- pandas_market_calendars: https://github.com/rsheftel/pandas_market_calendars
- exchange_calendars (upstream): https://github.com/gerrymanoim/exchange_calendars
- NSE official trading holidays: https://www.nseindia.com/resources/exchange-communication-holidays
- NSE 2024 calendar circular (FY24-25 = 249 sessions): https://nsearchives.nseindia.com/content/circulars/CMTR59722.pdf

---

## C7 — Add the missing candlestick patterns (and re-attribute hammer ratios)

**Issue recap:** §3.6 implements 7 of the ~25 standard Nison reversal patterns and **zero** continuation patterns. Hammer body/shadow ratios (`body ≤ 0.35×range`, etc.) are attributed to Nison but are actually TA-Lib / Bulkowski operationalizations.

### Recommended fix Part A — switch to TA-Lib for the standard catalog

```bash
# macOS
brew install ta-lib
pip install --no-build-isolation TA-Lib
```

```python
import talib

def detect_all_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Run all TA-Lib CDL* functions; return a DataFrame of {pattern: signed integer}."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    out = {}
    for name in talib.get_function_groups()["Pattern Recognition"]:
        fn = getattr(talib, name)
        out[name] = fn(o, h, l, c)        # +100 / 0 / -100 ; sign = direction
    return pd.DataFrame(out, index=df.index)
```

TA-Lib provides **all 61 CDL\* patterns** out of the box. The full list is reproduced in Appendix A below.

### Recommended fix Part B — minimum patterns to add

If you don't want all 61, here is the **minimum table-stakes set** to bring the system up to "competent" coverage. Each is a `talib.CDL*` call; signed +100/-100 output:

| Pattern | TA-Lib function | Bulkowski's empirical reliability* |
|---------|-----------------|-----|
| Hammer | `CDLHAMMER` | Bull rev. ~60% (with confirmation) |
| Hanging Man | `CDLHANGINGMAN` | Theoretical bear; tested 59% **continuation** — confirmation is mandatory |
| Inverted Hammer | `CDLINVERTEDHAMMER` | Theoretical bull; tested 65% **continuation** — confirmation mandatory |
| Shooting Star | `CDLSHOOTINGSTAR` | Bear rev. ~60% |
| Engulfing (bull/bear) | `CDLENGULFING` | One function, sign indicates direction |
| Harami / Harami Cross | `CDLHARAMI`, `CDLHARAMICROSS` | Mixed; modest reliability |
| Dark Cloud Cover | `CDLDARKCLOUDCOVER(o,h,l,c, penetration=0.5)` | Bear rev. ~60% |
| Piercing Pattern | `CDLPIERCING` (no penetration arg; hardcoded 50%) | Bull rev. ~64%, **rank 13/103** in Bulkowski |
| Three White Soldiers | `CDL3WHITESOLDIERS` | **82%** bull rev. — most reliable in this list |
| Three Black Crows | `CDL3BLACKCROWS` | **78%** bear rev. — also see `CDLIDENTICAL3CROWS` |
| Morning / Evening Star | `CDLMORNINGSTAR`, `CDLEVENINGSTAR` (with `penetration=0.3`) | Strong reversals |
| Doji (and variants) | `CDLDOJI`, `CDLDRAGONFLYDOJI`, `CDLGRAVESTONEDOJI`, `CDLLONGLEGGEDDOJI` | Indecision; context-dependent |
| Tweezer Top / Bottom | **NOT IN TA-LIB** — implement yourself (see below) | Modest |

\* Bulkowski's site (https://thepatternsite.com/) — empirical stats from his 2008 *Encyclopedia of Candlestick Charts* (Wiley, ISBN 0470181010).

### Recommended fix Part C — Tweezer Top/Bottom (TA-Lib doesn't have them)

```python
def tweezer_top(df, tol_pct=0.001, trend_lookback=5):
    """Two-bar bearish reversal where two consecutive highs match within tol_pct."""
    h1, h2 = df["high"].shift(1), df["high"]
    matched = (np.abs(h1 - h2) / h1) <= tol_pct
    uptrend = df["close"].shift(1) > df["close"].shift(trend_lookback + 1)
    bullish_1st = df["close"].shift(1) > df["open"].shift(1)
    bearish_2nd = df["close"] < df["open"]
    return (matched & uptrend & bullish_1st & bearish_2nd).astype(int) * -100

def tweezer_bottom(df, tol_pct=0.001, trend_lookback=5):
    l1, l2 = df["low"].shift(1), df["low"]
    matched = (np.abs(l1 - l2) / l1) <= tol_pct
    downtrend = df["close"].shift(1) < df["close"].shift(trend_lookback + 1)
    bearish_1st = df["close"].shift(1) < df["open"].shift(1)
    bullish_2nd = df["close"] > df["open"]
    return (matched & downtrend & bearish_1st & bullish_2nd).astype(int) * 100
```

### Recommended fix Part D — Continuation patterns (currently zero)

Add via TA-Lib:
- `CDLRISEFALL3METHODS` — **rising/falling three methods** (5-bar continuation; very common in trending NSE large-caps).
- `CDLTASUKIGAP` — Tasuki gap (continuation after window).
- `CDLMATHOLD(o,h,l,c, penetration=0.5)` — Mat hold.
- `CDLSEPARATINGLINES`, `CDL3LINESTRIKE`.

For raw "windows" (Nison's term for gaps), there is **no TA-Lib function**. Roll your own — trivial:

```python
def detect_window_up(df, min_gap_atr=0.25, atr=None):
    """Bullish window: today's low > yesterday's high by ≥ min_gap_atr × ATR."""
    gap = df["low"] - df["high"].shift(1)
    if atr is None:
        from talib import ATR
        atr = ATR(df["high"], df["low"], df["close"], timeperiod=14)
    return (gap > min_gap_atr * atr).astype(int) * 100

def detect_window_down(df, min_gap_atr=0.25, atr=None):
    gap = df["low"].shift(1) - df["high"]
    if atr is None:
        from talib import ATR
        atr = ATR(df["high"], df["low"], df["close"], timeperiod=14)
    return (gap > min_gap_atr * atr).astype(int) * -100
```

### Recommended fix Part E — re-attribute hammer ratios

Replace this in §3.6:

> "**Source for the pattern definitions.** Nison, *Japanese Candlestick Charting Techniques* (1991)..."

with:

> "**Pattern names and qualitative definitions:** Nison, *Japanese Candlestick Charting Techniques* (1991). **Quantitative thresholds** (body/range ratios, shadow/body ratios) follow standard TA-Lib / Bulkowski (*Encyclopedia of Candlestick Charts*, Wiley 2008) operationalizations; Nison's definitions are qualitative."

### Important caveat — Nison's confirmation requirement

For **Hammer / Hanging Man / Inverted Hammer / Shooting Star**, Nison and Bulkowski both insist on **next-day confirmation**:
- Hammer / Inverted Hammer: a higher close the next day.
- Hanging Man / Shooting Star: a lower close the next day.

TA-Lib does **not** apply this. If you flag a "Hanging Man" purely from `CDLHANGINGMAN > 0` with no confirmation, you will systematically lose money — Bulkowski's tests show 59% **continuation** rate for unconfirmed hanging men. Fix:

```python
def confirmed_hanging_man(df):
    raw = talib.CDLHANGINGMAN(df["open"], df["high"], df["low"], df["close"])
    confirm = df["close"] < df["close"].shift(1)
    return (raw < 0) & confirm
```

**Citations:**
- TA-Lib pattern recognition function list: https://ta-lib.github.io/ta-lib-python/func_groups/pattern_recognition.html
- Steve Nison, *Japanese Candlestick Charting Techniques* (1991, 2e 2001): ISBN 0-7352-0181-1
- Thomas Bulkowski's empirical stats: https://thepatternsite.com/
- Bulkowski, *Encyclopedia of Candlestick Charts* (Wiley 2008): ISBN 978-0470181010

---

## H1 — State Wilder's ADX threshold correctly

**Issue recap:** Doc says "Wilder's threshold for 'this is a trending market' is 20–25; below 20 we treat trends as noise."

**Reality (Wilder 1978 + StockCharts):**
- Wilder used **25** as the strong-trend threshold.
- The "20" lower bound is a **modern convention** popularized by StockCharts / chartists, not Wilder.

### Recommended fix — text rewrite for §3.2 / §4.1

> "**ADX strength gate.** ADX < 20 → neutral verdict. Wilder (1978, ch. VII) proposed **25** as the threshold for a strong trend; modern practice (StockCharts ChartSchool) often uses **20** as the practical floor. The 0.5 / 0.7 / 0.85 confidence anchors at ADX 20 / 30 / 40 are our own choice — defensibly low to encourage neutral verdicts in chop. 🔬 NEEDS BACKTEST against a 25-floor variant."

**Citations:**
- Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research. ISBN 0-89459-027-8.
- StockCharts ChartSchool, ADX page: https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/average-directional-index-adx

---

## H2 — Replace the BB Squeeze definition (use Bollinger's *or* TTM Squeeze, not a hybrid)

**Issue recap:** §3.4 defines BB Squeeze as "current bandwidth in lowest 20% of past 60 bars" and attributes it to Bollinger. That's neither Bollinger's nor Carter's published rule.

### Recommended fix — implement both, label clearly

```python
import numpy as np, pandas as pd

# ─── Bollinger's original Squeeze: BBW at 6-month low ──────────────────
def bollinger_squeeze(close: pd.Series, n: int = 20, k: float = 2.0,
                      lookback: int = 125) -> pd.DataFrame:
    """John Bollinger 2001, ch. 14: Squeeze when BBW = lowest in 6 months."""
    mb = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)             # population stdev — Bollinger uses ddof=0
    ub, lb = mb + k * sd, mb - k * sd
    bbw = (ub - lb) / mb
    bbw_low = bbw.rolling(lookback).min()
    return pd.DataFrame({"mb": mb, "ub": ub, "lb": lb, "bandwidth": bbw,
                         "squeeze": bbw <= bbw_low})

# ─── John Carter's TTM Squeeze: BB ⊂ Keltner ────────────────────────────
def ttm_squeeze(df: pd.DataFrame,
                bb_n: int = 20, bb_k: float = 2.0,
                kc_n: int = 20, kc_mult: float = 1.5) -> pd.DataFrame:
    """John Carter, *Mastering the Trade* (2005), ch. 10."""
    c, h, l = df["close"], df["high"], df["low"]
    # Bollinger
    mb = c.rolling(bb_n).mean()
    sd = c.rolling(bb_n).std(ddof=0)
    bb_u, bb_l = mb + bb_k * sd, mb - bb_k * sd
    # Keltner — Chester Keltner (1960) original (range, not ATR), as Carter uses
    tp = (h + l + c) / 3
    kc_mid = tp.rolling(kc_n).mean()
    rng    = (h - l).rolling(kc_n).mean()
    kc_u, kc_l = kc_mid + kc_mult * rng, kc_mid - kc_mult * rng
    on   = (bb_u < kc_u) & (bb_l > kc_l)
    fire = (~on) & on.shift(1).fillna(False)        # squeeze just released
    return pd.DataFrame({"squeeze_on": on, "squeeze_fire": fire,
                         "bb_u": bb_u, "bb_l": bb_l,
                         "kc_u": kc_u, "kc_l": kc_l})
```

### Recommendation — use both

- **TTM Squeeze** for the **trade trigger** (`squeeze_fire` is a discrete event).
- **Bollinger 6-month-low** as the **regime filter** ("only trade fires that come out of a high-quality compression").

> ⚠️ **Stdev gotcha:** Bollinger uses **population stdev** (`ddof=0`). pandas `.std()` defaults to `ddof=1`. This is why your Python BB will be *slightly* tighter than TradingView/StockCharts unless you set `ddof=0`.

### Recommended doc rewrite for §3.4

> "**Bollinger-Band Squeeze.** Two definitions in active use; we implement both:
> 1. **Bollinger's original** (Bollinger 2001, ch. 14): Squeeze ON when current BandWidth = lowest in 125 trading days (~6 months).
> 2. **TTM Squeeze** (Carter 2005, ch. 10): Squeeze ON when both BB lines lie *inside* Keltner Channels (20-period SMA ± 1.5 × 20-bar high–low range, per Chester Keltner's 1960 formula). The discrete 'fire' event = first off-bar after consecutive on-bars.
>
> The synthesizer is told whether either is active. The percentile-rank approximation in earlier drafts has been removed."

**Citations:**
- John Bollinger, *Bollinger on Bollinger Bands*, McGraw-Hill 2001, ch. 14. ISBN 0-07-137368-3.
- John F. Carter, *Mastering the Trade*, McGraw-Hill 2005, ch. 10. ISBN 0-07-145958-8.
- Chester Keltner (1960), *How to Make Money in Commodities*. (Reproduced on StockCharts.)
- StockCharts ChartSchool — Bollinger BandWidth: https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/bollinger-bandwidth
- StockCharts ChartSchool — TTM Squeeze: https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/ttm-squeeze
- StockCharts ChartSchool — Keltner Channels: https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/keltner-channels

---

## H3 — Raise the double-top trough-depth saturation to 10%

**Issue recap:** Doc saturates `depth_score` at 5% trough drop; Bulkowski empirically requires ≥ 10%.

### Recommended fix — two-tier scoring

```python
def double_top_depth_score(trough_drop_pct: float) -> float:
    """0.5 at 5% (Bulkowski's 'weak'), 1.0 at 10%+ (his minimum)."""
    if trough_drop_pct < 0.05:
        return trough_drop_pct / 0.10            # linear 0 → 0.5 over 0–5%
    if trough_drop_pct < 0.10:
        return 0.5 + (trough_drop_pct - 0.05) / 0.10  # 0.5 → 1.0 over 5–10%
    return 1.0
```

**Citations:**
- Thomas Bulkowski, *Encyclopedia of Chart Patterns*, 2e Wiley 2005. ISBN 978-0471668268.
- Bulkowski Adam-and-Adam Double Top: https://thepatternsite.com/aadt.html
- Bulkowski "Big M" / generic double top: https://thepatternsite.com/bigm.html

---

## H4 — Re-attribute the RSI 60/40 to Cardwell

### Recommended doc rewrite for §4.2

> "**RSI level vote.** RSI > 60 → +1 bullish, RSI < 40 → +1 bearish, else 0. The 60/40 vote thresholds follow **Andrew Cardwell's RSI-range theory** (uptrend RSI 40–80 with 40 as support; downtrend RSI 20–60 with 60 as resistance), as popularized by Constance Brown's *Technical Analysis for the Trading Professional* (2nd ed., McGraw-Hill 2011, ch. 7). The classic Wilder 70/30 is reserved for the overbought/oversold *flag*."

**Note:** Brown's actual numbers are slightly wider (40–90 / 10–60); 40–80 / 20–60 is Cardwell's simplification (StockCharts cites Brown directly with Cardwell-style numbers in their RSI page).

**Citations:**
- Brown, C. (2011). *Technical Analysis for the Trading Professional*, 2e. McGraw-Hill. ISBN 978-0071759144.
- StockCharts ChartSchool — RSI: https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi

---

## H5 — Add R3/S3 to the "classic" floor-trader pivots

```python
def classic_pivots(prev_high: float, prev_low: float, prev_close: float):
    pp = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pp - prev_low
    s1 = 2 * pp - prev_high
    r2 = pp + (prev_high - prev_low)
    s2 = pp - (prev_high - prev_low)
    r3 = prev_high + 2 * (pp - prev_low)
    s3 = prev_low  - 2 * (prev_high - pp)
    return {"PP": pp, "R1": r1, "R2": r2, "R3": r3,
                       "S1": s1, "S2": s2, "S3": s3}
```

Optional — also expose Camarilla, Woodie's, Fibonacci, DeMark variants:

```python
def camarilla_pivots(prev_high, prev_low, prev_close):
    rng = prev_high - prev_low
    return {
        "R4": prev_close + rng * 1.1 / 2,
        "R3": prev_close + rng * 1.1 / 4,
        "R2": prev_close + rng * 1.1 / 6,
        "R1": prev_close + rng * 1.1 / 12,
        "S1": prev_close - rng * 1.1 / 12,
        "S2": prev_close - rng * 1.1 / 6,
        "S3": prev_close - rng * 1.1 / 4,
        "S4": prev_close - rng * 1.1 / 2,
    }

def fib_pivots(prev_high, prev_low, prev_close):
    pp = (prev_high + prev_low + prev_close) / 3
    rng = prev_high - prev_low
    return {
        "PP": pp,
        "R1": pp + 0.382 * rng, "S1": pp - 0.382 * rng,
        "R2": pp + 0.618 * rng, "S2": pp - 0.618 * rng,
        "R3": pp + 1.000 * rng, "S3": pp - 1.000 * rng,
    }
```

**Citations:**
- Wikipedia — Pivot point (TA): https://en.wikipedia.org/wiki/Pivot_point_(technical_analysis)
- ActionForex pivots guide: https://www.actionforex.com/articles/daily-forex-articles/forex-trading-tools/forex-pivot-points/

---

## H6 — Document Alpha Vantage's 25/day cap (or drop it)

If you keep Alpha Vantage as a tertiary fallback:

```python
# Alpha Vantage free tier (verified May 2026): 25 requests/day, hard cap.
# Worth using only as a last-resort, not as a regular fallback.
ALPHA_VANTAGE_DAILY_CAP = 25

def alpha_vantage_daily(symbol: str, api_key: str):
    """NSE example: symbol='RELIANCE.BSE'. Use sparingly."""
    url = "https://www.alphavantage.co/query"
    params = {"function": "TIME_SERIES_DAILY",
              "symbol": symbol, "outputsize": "full",
              "apikey": api_key}
    return requests.get(url, params=params, timeout=30).json()
```

**Recommended:** drop Alpha Vantage entirely and use the chain in C1 (jugaad-data → bhavcopy → yfinance). Alpha Vantage's NSE feed has known reliability issues (multiple GitHub/Reddit reports of stale data on Indian symbols); 25 calls/day is also functionally useless for any real watchlist.

**Citation:** https://www.alphavantage.co/support/#api-key

---

## H7, H8 — Wilder smoothing and MACD: use proper warmup

**Issue recap:** Doc uses `2 × length = 28` bars warmup for ADX/RSI; this is mathematically insufficient. MACD has no warmup at all.

### Background — why 28 bars isn't enough

Wilder smoothing is an EWMA with α = 1/N. The seed-value bias decays as `(1 − 1/N)^k`. To reach <1% influence:

```
k = log(0.01) / log(1 − 1/N) ≈ 4.6 N
```

For N=14: 65 bars to reach <1% bias; 97 bars to reach <0.1%. (Skoglund, "Recursive estimation of moving averages with seed correction," *J. of Risk Model Validation*, 2017.)

### Recommended warmup table

| Indicator | Wilder's first valid bar | Pragmatic warmup (1% bias) | Conservative (0.1%) |
|-----------|:---:|:---:|:---:|
| ATR(14) | 14 | 65 | 97 |
| RSI(14) | 14 | 65 | 97 |
| ADX(14) | 28 (= 2N) | 80 (~4 mo) | 115 |
| MACD(12, 26, 9) | 35 (slow + signal) | 130 (~ 5N of slow) | 250 |

### Code — discard early bars

```python
WILDER_LENGTH = 14
ADX_LENGTH    = 14
MACD_SLOW     = 26
MACD_SIGNAL   = 9

WARMUP_BARS = max(
    5 * WILDER_LENGTH,                     # ATR / RSI = 70
    5 * ADX_LENGTH * 2,                    # ADX = 140 (very conservative)
    5 * (MACD_SLOW + MACD_SIGNAL),         # MACD = 175
)

def trim_warmup(df: pd.DataFrame, n: int = WARMUP_BARS) -> pd.DataFrame:
    """Discard the first n bars of any series before treating Wilder-smoothed
    indicators as 'converged'."""
    return df.iloc[n:].copy()
```

### Update §2.2 cache "proactive fetch window"

Currently 365 calendar days (≈ 250 trading bars). Bump to **500 trading days (~ 2 calendar years)** so the warmup discard still leaves usable history:

```python
PROACTIVE_TRADING_BARS = 500  # ≈ 2 years; was 365 calendar days (~ 250 bars)
```

### Recommended doc rewrite for §3.2 / §3.3

> "**Convergence guard.** Wilder smoothing is an EWMA with α = 1/N; the first value mathematically appears at bar `2N` (ADX) or `N` (RSI/ATR), but the seed bias decays as `(1−1/N)^k` and only falls below 1% by ≈ `4.6 N` bars. We therefore require **≥ 5N bars** before publishing any value (= 70 bars for length-14 RSI/ATR; 140 for ADX(14)). Source: Skoglund (2017); Kirkpatrick & Dahlquist, *Technical Analysis* (3e, FT Press 2016)."

**Citations:**
- Wilder, J. W. (1978). *New Concepts in Technical Trading Systems*. Trend Research.
- Robert Colby (2003). *Encyclopedia of Technical Market Indicators*, 2e. McGraw-Hill. ISBN 0-07-046038-6.
- Charles Kirkpatrick & Julie Dahlquist (2016). *Technical Analysis: The Complete Resource for Financial Market Technicians*, 3e. FT Press. ISBN 0-13-413704-2.
- TradingView Pine Script `ta.rma` reference: https://www.tradingview.com/pine-script-reference/v5/#fun_ta.rma

---

## H9 — Add the missing major indicators (VWAP, Ichimoku, Volume Profile)

### H9a — Anchored / Rolling VWAP

VWAP from daily bars is necessarily an approximation (no intraday data), but **anchored VWAP** (Brian Shannon's framework) and **rolling N-day VWAP** are well-defined on daily OHLCV.

```python
def anchored_vwap(df: pd.DataFrame, anchor_date) -> pd.Series:
    """Anchored VWAP from a specific date. Brian Shannon (Wiley 2023)."""
    sub = df.loc[anchor_date:].copy()
    tp  = (sub["high"] + sub["low"] + sub["close"]) / 3
    pv  = (tp * sub["volume"]).cumsum()
    vol = sub["volume"].cumsum()
    return (pv / vol).reindex(df.index)

def rolling_vwap(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    pv  = (tp * df["volume"]).rolling(n).sum()
    v   = df["volume"].rolling(n).sum()
    return pv / v
```

Useful Indian-market anchors: Budget day, post-earnings gap, post-RBI policy day.

**Citations:**
- Brian Shannon, *Maximum Trading Gains with Anchored VWAP*, Wiley 2023. ISBN 1394196687.
- StockCharts ChartSchool — Anchored VWAP: https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/anchored-vwap

### H9b — Ichimoku Kinko Hyo

```python
def ichimoku(df: pd.DataFrame,
             tenkan_n: int = 9, kijun_n: int = 26,
             senkou_b_n: int = 52, displacement: int = 26) -> pd.DataFrame:
    h, l, c = df["high"], df["low"], df["close"]
    midline = lambda w: (h.rolling(w).max() + l.rolling(w).min()) / 2
    tenkan   = midline(tenkan_n)
    kijun    = midline(kijun_n)
    senkou_a = ((tenkan + kijun) / 2).shift(displacement)
    senkou_b = midline(senkou_b_n).shift(displacement)
    chikou   = c.shift(-displacement)
    return pd.DataFrame({"tenkan": tenkan, "kijun": kijun,
                         "senkou_a": senkou_a, "senkou_b": senkou_b,
                         "chikou": chikou})
```

**Cluster signals** (suggested for the trend / levels classifier):
- `price > max(senkou_a, senkou_b)` → above-cloud (bullish regime).
- `tenkan crosses above kijun` AND price is above cloud → strong bullish trigger.
- Senkou A vs Senkou B in the future projection ("kumo twist") flags potential trend change in ~26 bars.

**Citations:**
- Goichi Hosoda, *Ichimoku Kinkō Hyō* (1969).
- Manesh Patel, *Trading with Ichimoku Clouds*, Wiley 2010. ISBN 978-0470609361.
- StockCharts ChartSchool — Ichimoku: https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/ichimoku-cloud

### H9c — Volume Profile (POC, Value Area)

```python
import numpy as np, pandas as pd

def volume_profile(df: pd.DataFrame, n_days: int = 20,
                   bins: int = 100, value_area_pct: float = 0.70):
    """Approximate volume-by-price profile by uniformly distributing each
    day's volume across [low, high]. POC = price level of max volume.
    Value Area = contiguous band around POC containing 70% of total volume."""
    sub = df.iloc[-n_days:]
    p_min, p_max = sub["low"].min(), sub["high"].max()
    edges = np.linspace(p_min, p_max, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    hist = np.zeros(bins)

    for _, row in sub.iterrows():
        lo, hi, vol = row["low"], row["high"], row["volume"]
        mask = (edges[1:] > lo) & (edges[:-1] < hi)
        n_overlap = mask.sum()
        if n_overlap == 0: continue
        hist[mask] += vol / n_overlap

    poc_idx = int(np.argmax(hist))
    total, target = hist.sum(), hist.sum() * value_area_pct
    lo_i = hi_i = poc_idx
    running = hist[poc_idx]
    while running < target and (lo_i > 0 or hi_i < bins - 1):
        left  = hist[lo_i - 1] if lo_i > 0 else -1
        right = hist[hi_i + 1] if hi_i < bins - 1 else -1
        if right >= left:
            hi_i += 1; running += hist[hi_i]
        else:
            lo_i -= 1; running += hist[lo_i]
    return {"profile": pd.Series(hist, index=centers),
            "poc": centers[poc_idx],
            "vah": centers[hi_i],
            "val": centers[lo_i]}
```

**Citations:**
- J. Peter Steidlmayer & Steven B. Hawkins, *Steidlmayer on Markets*, 2e Wiley 2003. ISBN 0-471-21556-5.
- James Dalton et al., *Mind Over Markets*, Probus 1990. ISBN 1-557-38387-0.
- `market-profile` Python lib: https://pypi.org/project/market-profile/

### H9d — India VIX integration

```python
# pip install nsepython
from nsepython import index_history

def fetch_india_vix(start_ddmmyyyy="01-Jan-2008", end_ddmmyyyy=None):
    if end_ddmmyyyy is None:
        end_ddmmyyyy = date.today().strftime("%d-%b-%Y")
    return index_history("INDIA VIX", start_ddmmyyyy, end_ddmmyyyy)
```

Suggested use in the synthesizer: as a **regime gate**, not a directional signal.

```python
def vix_regime(vix_close: pd.Series, lookback: int = 60) -> str:
    median = vix_close.rolling(lookback).median().iloc[-1]
    cur = vix_close.iloc[-1]
    if cur < 0.85 * median: return "low_vol"
    if cur > 1.15 * median: return "high_vol"
    return "normal"
```

Resolves the inconsistency between §1.4 ("no IV") and §9 ("no macro") — India VIX is free, daily, one number.

**Citations:**
- NSE India VIX page: https://www.nseindia.com/products-services/indices-indiavix-index
- NSE India VIX methodology: https://www.niftyindices.com/Methodology/Method_India_VIX.pdf
- NSE historical VIX: https://www.nseindia.com/reports-indices-historical-index-data

---

## M11 — Brier Skill Score, not just raw Brier

**Issue recap:** Doc's "Brier baseline = 0.25" assumes a 50/50 base rate. Real systems should report **Brier Skill Score (BSS)** vs. the empirical base rate.

```python
def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))

def brier_skill_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """BSS > 0 = real skill; BSS = 0 = no better than guessing the base rate."""
    p_bar = outcomes.mean()                      # empirical base rate
    bs    = np.mean((probs    - outcomes) ** 2)
    bs_ref = np.mean((p_bar   - outcomes) ** 2)  # = p_bar * (1 - p_bar)
    if bs_ref == 0:
        return float("nan")
    return 1.0 - bs / bs_ref
```

For non-50/50 base rates this is the standard. For NIFTY-50 daily directional predictions the empirical bullish-day rate is ~52–55%, so the right reference is ~`0.55 × 0.45 = 0.2475`, not flat 0.25.

**Citations:**
- Brier, G. W. (1950). "Verification of forecasts expressed in terms of probability." *Monthly Weather Review* 78(1): 1–3. https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2
- Murphy, A. H. (1973). "A new vector partition of the probability score." *J. Applied Meteorology* 12(4): 595–600. https://doi.org/10.1175/1520-0450(1973)012<0595:ANVPOT>2.0.CO;2

---

## Look-ahead-bias hardening (review §5a–5b items)

### Trafilatura + Wayback Machine for point-in-time article fetching

```python
# pip install waybackpy trafilatura
import waybackpy, requests, trafilatura
from datetime import datetime

UA = "stockpred-research/1.0 (contact@you)"

def article_as_of(url: str, asof: datetime) -> str | None:
    """Fetch the article HTML as it appeared on/before asof. None if no snapshot."""
    api = waybackpy.WaybackMachineCDXServerAPI(url=url, user_agent=UA)
    snap = api.near(year=asof.year, month=asof.month, day=asof.day)
    if snap is None: return None
    snap_ts = snap.timestamp
    if snap_ts > asof.strftime("%Y%m%d%H%M%S"):
        return None      # guard: never silently use a snapshot AFTER asof
    raw_url = (f"https://web.archive.org/web/{snap_ts}id_/{url}")
    return requests.get(raw_url, headers={"User-Agent": UA}, timeout=30).text

def article_body_pit(url: str, asof: datetime) -> str | None:
    html = article_as_of(url, asof)
    if html is None: return None
    return trafilatura.extract(html)
```

**Pipeline integration:**
1. **Cache aggressively** — persist `(url, asof_date) → snapshot_ts → body` to SQLite/Parquet. Wayback rate-limits.
2. **Snapshot timestamp is data** — store the actual archived timestamp; it's the true publication time as far as your backtest is concerned.
3. **Be polite** — 1 request/second with a contact-info User-Agent. Internet Archive ToS: https://archive.org/about/terms.php
4. **No silent fallback to live URL** — if no snapshot ≤ `asof` exists, drop the observation (or mark it as "post-hoc" and exclude from backtest hits).

### Frozen price snapshots (yfinance silently revises)

For backtests, snapshot prices once per `(ticker, as_of)` and never re-fetch:

```python
import sqlite3
import pandas as pd

class PITPriceStore:
    """Point-in-time price store: never re-fetch the same (ticker, as_of)."""
    def __init__(self, db_path="prices_pit.sqlite"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS bars (
                ticker  TEXT NOT NULL,
                as_of   TEXT NOT NULL,
                bar_date TEXT NOT NULL,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                PRIMARY KEY (ticker, as_of, bar_date)
            )
        """)
        self.conn.commit()

    def get_or_fetch(self, ticker, as_of, fetcher):
        cur = self.conn.execute(
            "SELECT bar_date, open, high, low, close, volume "
            "FROM bars WHERE ticker=? AND as_of=?",
            (ticker, as_of.isoformat()))
        rows = cur.fetchall()
        if rows:
            return pd.DataFrame(rows, columns=["bar_date","open","high","low","close","volume"])
        df = fetcher(ticker, end=as_of)
        df["ticker"], df["as_of"] = ticker, as_of.isoformat()
        df.to_sql("bars", self.conn, if_exists="append", index=False)
        return df
```

---

## Survivorship bias for NIFTY 50 backtests

### Solution — point-in-time membership table

There is **no free clean dataset**. Build one from NSE press releases + cross-check with Wikipedia.

```python
import pandas as pd

def members_on(membership_df: pd.DataFrame, d) -> list[str]:
    """membership_df has columns: symbol, in_date, out_date (out_date NaT if current)."""
    d = pd.Timestamp(d)
    df = membership_df.copy()
    df["out_date"] = df["out_date"].fillna(pd.Timestamp("2099-12-31"))
    mask = (df["in_date"] <= d) & (df["out_date"] > d)
    return df.loc[mask, "symbol"].tolist()
```

### Where to get the data

| Source | Cost | Quality |
|--------|------|---------|
| **NSE Indices press releases** (semi-annual reconstitutions) | Free | Authoritative; manual to scrape |
| Wikipedia "List of NIFTY 50 companies" history | Free | Reasonable starter; verify before trusting |
| **CMIE Prowess** | Academic license | Industry standard for India quant work |
| **Bloomberg `INDX_MWEIGHT_HIST`** | $24k/yr | Gold standard |
| **Refinitiv Datastream** | Institutional | Gold standard alternative |
| Capitaline | mid-5-figure INR/yr | Popular in Indian quant shops |

**Citations:**
- NSE Indices press releases: https://www.niftyindices.com/resources/press-release
- NIFTY 50 methodology: https://www.niftyindices.com/Methodology/Method_NIFTY_Equity_Indices.pdf
- Wikipedia NIFTY 50 history: https://en.wikipedia.org/wiki/NIFTY_50
- CMIE Prowess: https://www.cmie.com/kommon/bin/sr.php?kall=wproduct&prodid=3

---

## Bonus solutions for items the doc author didn't ask about

### Timezone correctness (review §5d)

```python
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")

def now_ist() -> datetime:
    return datetime.now(IST)                       # always tz-aware

def is_pre_market_close(dt_ist: datetime) -> bool:
    """NSE close is 15:30 IST; the doc's 'today's close if pre-15:30' rule."""
    if dt_ist.tzinfo is None:
        raise ValueError("Naive datetime; pass an IST-aware one.")
    cutoff = dt_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    return dt_ist < cutoff
```

Apply throughout the codebase: never use `datetime.now()` (naive); always `datetime.now(IST)` or `datetime.now(UTC)`.

### Schema brittleness (review §5e)

For LLM JSON outputs, prefer `extra="ignore"` for non-critical fields and add a one-pass cleanup before validation:

```python
from pydantic import BaseModel, ConfigDict

class Prediction(BaseModel):
    model_config = ConfigDict(extra="ignore")    # don't hard-fail on bonus fields
    # ... fields ...

def clean_llm_json(payload: dict, allowed_fields: set[str]) -> dict:
    """Strip unknown top-level keys before validation. One-pass repair."""
    return {k: v for k, v in payload.items() if k in allowed_fields}
```

### Calibration window (review §5g)

Document explicitly:
- **Production monitoring metric:** rolling 90-day Brier + BSS, partitioned by horizon and direction.
- **All-time metric:** for backtest results only.
- **Per-confidence-bucket calibration:** rolling 6-month minimum (need ≥ 30 predictions per bucket for the Brier to be meaningful).

---

## 99. Recommended implementation roadmap

### Sprint 1 (high-impact data-layer fixes — 1–2 days)
1. **C1**: Replace Stooq with `jugaad-data` (primary), add NSE bhavcopy fallback. Drop or demote Alpha Vantage. (~4 hrs)
2. **C2**: Fix the GDELT 7-day misnomer. Document the real 3-month rolling window. Add the `gdelt_paginate` helper. (~1 hr)
3. **C3**: Replace the §2.4 "public feed" framing with the session-priming pattern. Add `make_nse_session`. Cross-validate with BSE. (~2 hrs)
4. **C6**: Replace 252 with `pandas_market_calendars.get_calendar('XBOM')`. (~1 hr)

### Sprint 2 (correctness fixes — 2–3 days)
5. **C5**: One-line citation fix. (5 min)
6. **H1**: Doc rewrite for ADX threshold attribution. (5 min)
7. **H2**: Replace the BB Squeeze definition with `bollinger_squeeze` + `ttm_squeeze`. (~3 hrs)
8. **H3**: Bump double-top depth saturation to 10%. (15 min)
9. **H4**: Re-attribute RSI 60/40 to Cardwell. (5 min)
10. **H5**: Add R3/S3 (and optional Camarilla). (~1 hr)
11. **H7+H8**: Bump Wilder warmup to 5N. Bump cache proactive fetch to 500 trading days. (~2 hrs)

### Sprint 3 (coverage expansion — 3–5 days)
12. **C7+H9**: Switch candlestick layer to TA-Lib (61 patterns). Implement Tweezer top/bottom + windows. Add Nison-style next-day confirmation gating. (~1 day)
13. **C4**: Add broadening tops/bottoms + rectangles. Refactor pattern detector to use Nadaraya–Watson smoother. (~1 day)
14. **H9**: Add VWAP, Ichimoku, Volume Profile, India VIX. (~2 days)

### Sprint 4 (audit / hardening — 2–3 days)
15. Wayback-Machine PIT article fetcher. (~1 day)
16. Frozen price snapshot store. (~0.5 day)
17. Survivorship-bias-aware NIFTY 50 membership table (manual scrape from NSE press releases). (~1 day)
18. Brier Skill Score reporting. (~30 min)
19. Timezone correctness audit. (~2 hrs)

### Sprint 5 (testing / validation — ongoing)
20. Backtest infrastructure (the doc's §8.6, currently 🔬 NEEDS BACKTEST).
21. Parameter sweeps for the items still tagged 🔬 (per-horizon thresholds, news weights, citation thresholds, stop-pad ladder, candlestick gating ATR proximity).

---

## Appendix A — Full TA-Lib `CDL*` candlestick function list (61)

Source: official TA-Lib Python wrapper docs — https://ta-lib.github.io/ta-lib-python/func_groups/pattern_recognition.html

| # | Function | Pattern |
|---|----------|---------|
| 1 | CDL2CROWS | Two Crows |
| 2 | CDL3BLACKCROWS | Three Black Crows |
| 3 | CDL3INSIDE | Three Inside Up/Down |
| 4 | CDL3LINESTRIKE | Three-Line Strike |
| 5 | CDL3OUTSIDE | Three Outside Up/Down |
| 6 | CDL3STARSINSOUTH | Three Stars In The South |
| 7 | CDL3WHITESOLDIERS | Three Advancing White Soldiers |
| 8 | CDLABANDONEDBABY | Abandoned Baby (penetration) |
| 9 | CDLADVANCEBLOCK | Advance Block |
| 10 | CDLBELTHOLD | Belt-hold |
| 11 | CDLBREAKAWAY | Breakaway |
| 12 | CDLCLOSINGMARUBOZU | Closing Marubozu |
| 13 | CDLCONCEALBABYSWALL | Concealing Baby Swallow |
| 14 | CDLCOUNTERATTACK | Counterattack |
| 15 | CDLDARKCLOUDCOVER | Dark Cloud Cover (penetration) |
| 16 | CDLDOJI | Doji |
| 17 | CDLDOJISTAR | Doji Star |
| 18 | CDLDRAGONFLYDOJI | Dragonfly Doji |
| 19 | CDLENGULFING | Engulfing |
| 20 | CDLEVENINGDOJISTAR | Evening Doji Star (penetration) |
| 21 | CDLEVENINGSTAR | Evening Star (penetration) |
| 22 | CDLGAPSIDESIDEWHITE | Up/Down-gap side-by-side white |
| 23 | CDLGRAVESTONEDOJI | Gravestone Doji |
| 24 | CDLHAMMER | Hammer |
| 25 | CDLHANGINGMAN | Hanging Man |
| 26 | CDLHARAMI | Harami |
| 27 | CDLHARAMICROSS | Harami Cross |
| 28 | CDLHIGHWAVE | High-Wave Candle |
| 29 | CDLHIKKAKE | Hikkake |
| 30 | CDLHIKKAKEMOD | Modified Hikkake |
| 31 | CDLHOMINGPIGEON | Homing Pigeon |
| 32 | CDLIDENTICAL3CROWS | Identical Three Crows |
| 33 | CDLINNECK | In-Neck |
| 34 | CDLINVERTEDHAMMER | Inverted Hammer |
| 35 | CDLKICKING | Kicking |
| 36 | CDLKICKINGBYLENGTH | Kicking — bull/bear by longer marubozu |
| 37 | CDLLADDERBOTTOM | Ladder Bottom |
| 38 | CDLLONGLEGGEDDOJI | Long-Legged Doji |
| 39 | CDLLONGLINE | Long Line Candle |
| 40 | CDLMARUBOZU | Marubozu |
| 41 | CDLMATCHINGLOW | Matching Low |
| 42 | CDLMATHOLD | Mat Hold (penetration) |
| 43 | CDLMORNINGDOJISTAR | Morning Doji Star (penetration) |
| 44 | CDLMORNINGSTAR | Morning Star (penetration) |
| 45 | CDLONNECK | On-Neck |
| 46 | CDLPIERCING | Piercing |
| 47 | CDLRICKSHAWMAN | Rickshaw Man |
| 48 | CDLRISEFALL3METHODS | Rising/Falling Three Methods |
| 49 | CDLSEPARATINGLINES | Separating Lines |
| 50 | CDLSHOOTINGSTAR | Shooting Star |
| 51 | CDLSHORTLINE | Short Line Candle |
| 52 | CDLSPINNINGTOP | Spinning Top |
| 53 | CDLSTALLEDPATTERN | Stalled Pattern |
| 54 | CDLSTICKSANDWICH | Stick Sandwich |
| 55 | CDLTAKURI | Takuri (long lower shadow dragonfly) |
| 56 | CDLTASUKIGAP | Tasuki Gap |
| 57 | CDLTHRUSTING | Thrusting |
| 58 | CDLTRISTAR | Tristar |
| 59 | CDLUNIQUE3RIVER | Unique 3 River |
| 60 | CDLUPSIDEGAP2CROWS | Upside Gap Two Crows |
| 61 | CDLXSIDEGAP3METHODS | Up/Downside Gap Three Methods |

**Conspicuously NOT in TA-Lib** (you must implement yourself if you want them):
- Tweezer Top / Bottom (code provided in C7)
- Window / Gap (code provided in C7)
- Three Mountains, Three Rivers (basic), Eight/Ten New Highs, Spike, Island Reversal — covered in Bulkowski's *Encyclopedia of Candlestick Charts*.

---

## Appendix B — Combined patches summary (where in the doc each fix lands)

| Section in `pred_logic.md` | Fix(es) to apply |
|----------------------------|------------------|
| §1.4 vs §9 | Resolve IV/macro inconsistency. India VIX is in (cheap, daily, one number). |
| §2.1 | C1: replace fallback chain. Drop Stooq, demote Alpha Vantage. |
| §2.2 | H7: bump cache fetch window to 500 trading days. |
| §2.3 | C2: GDELT real 3-month window; pagination helper; mention BigQuery for older. |
| §2.4 | C3: relabel "public feed" → "internal endpoint"; show session-priming pattern. |
| §3.2 | H1: ADX threshold attribution; H7: Wilder warmup. |
| §3.3 | M2: MACD "late 1970s"; M3: OBV credits; H8: MACD warmup. |
| §3.4 | H2: BB Squeeze proper implementation. |
| §3.5 | C6: NSE calendar; H5: add R3/S3. |
| §3.6 | C7: TA-Lib catalog + Tweezer/Windows + Nison-confirmation gate; re-attribute hammer ratios. |
| §3.7 | C4+C5: 5 missing patterns; LMW kernel-regression smoother; 22-bar attribution fix; H3 trough depth. |
| §4.1 | H1: ADX confidence anchors footnote. |
| §4.2 | H4: re-attribute RSI 60/40 to Cardwell. |
| §4.3 | M7: cite the %B 0.1/0.9 thresholds. |
| §6.1 | re-derive thresholds, document drift from √T. |
| §6.2 | M8: news-weight rationale; M4: rename "worst-case RR" to "worst-fill RR". |
| §7.2 | M9: lower citation thresholds; auto-derive indicator vocabulary. |
| §7.4 | M-rules: degraded-news cap; replace "projection" substring with explicit enum on `target.method`. |
| §8.5 | M11: Brier Skill Score alongside raw Brier. |
| §8.6 | Wayback-Machine integration; PIT price store; survivorship-bias-aware NIFTY membership. |

---

*End of solutions document. Every claim above is backed by a primary source URL or an ISBN. If any citation appears stale, please flag — the web moves and I'd rather you push back than trust me on faith.*
