"""Build the bundled search index CSV from authoritative sources.

Why a script (instead of hand-editing a CSV):
  - The Nifty 50 portion stays in lock-step with data/kb/stocks.json
    (single source of truth — no chance of drift).
  - The non-N50 'popular extras' are organized in Python code, grouped
    by sector with comments, so future-us can audit + extend cleanly.
  - Reproducible: anyone can `python scripts/build_search_index.py`
    and get the same CSV byte-for-byte.

Run:
    python scripts/build_search_index.py             # curated ~200 names
    python scripts/build_search_index.py --fetch-nse  # + every NSE stock
    python scripts/build_search_index.py --backfill-sectors  # real sectors

The --fetch-nse flag downloads NSE's official EQUITY_L.csv (the full
listed-equity master, ~2000 symbols) and MERGES it in: Nifty 50 and the
curated extras keep their hand-tuned sectors, and every remaining NSE
symbol is added with sector 'NSE Listed'. Must be run OFF the corporate
VPN — www.nseindia.com is blocked on many corporate networks.

Output:
    frontend/data/nifty500.csv      (ticker,name,sector,nifty50)

Naming note: the CSV is called nifty500.csv even though the curated build
has ~200 entries — the name signals the eventual target. It's a forward-
compatible filename, not a current claim.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

# Make src/ importable so we can reuse the tested sector-backfill module
# rather than re-implementing yfinance plumbing here (DRY).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from price_predictor.data.sector_lookup import (
    backfill_sectors,
)

# Repo root = parent of this script's parent (scripts/).
REPO_ROOT = Path(__file__).resolve().parent.parent
KB_STOCKS_JSON = REPO_ROOT / "data" / "kb" / "stocks.json"
OUTPUT_CSV = REPO_ROOT / "frontend" / "data" / "nifty500.csv"
# Persistent, resumable ticker→sector cache. Survives rate-limit
# interruptions so reruns only retry the stragglers.
SECTOR_CACHE = REPO_ROOT / "data" / "cache" / "yf_sectors.json"

# ── NSE full equity master (used only with --fetch-nse) ─────────────
# EQUITY_L.csv is NSE's authoritative list of every listed equity. It is
# served from the archives host; the www host is a fallback. NSE blocks
# plain Python User-Agents, so we prime a session (homepage GET for
# cookies) with browser-like headers first — same trick as data/filings.py.
NSE_HOMEPAGE = "https://www.nseindia.com/"
NSE_EQUITY_L_URLS = (
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    "https://www.nseindia.com/content/equities/EQUITY_L.csv",
)
_BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/csv,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Referer": NSE_HOMEPAGE,
    "Connection": "keep-alive",
}
# Which NSE series count as tradeable equity for our purposes. EQ = normal
# rolling settlement; BE = trade-to-trade (still real equity). Everything
# else (illiquid / non-equity series) is skipped.
_KEEP_SERIES: frozenset[str] = frozenset({"EQ", "BE"})
# Sanity floor: NSE lists ~2000 equities. If a fetch returns far fewer,
# it's almost certainly a partial/blocked/garbage response (e.g. an HTML
# error page). Refuse to overwrite the bundled index in that case so an
# unattended startup refresh can never clobber a good file with junk.
_MIN_NSE_ROWS: int = 1000


# ────────────────────────────────────────────────────────────────────
# Curated non-Nifty-50 extras — popular Indian stocks retail traders
# actually look for. Grouped by sector for easy maintenance.
#
# Conventions:
#   - Ticker WITHOUT .NS suffix (added at write time, single source of truth)
#   - Company name as you'd see on Moneycontrol / NSE website
#   - Sector matches the bucket names used in data/kb/stocks.json
# ────────────────────────────────────────────────────────────────────
# Name overrides — tickers whose official name lost a popular alias
# during a rebrand. We rewrite the name in the bundled CSV so users
# searching by the old / colloquial name find them.
#
# Pattern: SEARCHABLE_KEYWORD prefixed in parens so the name reads as
# 'NewName (formerly OldName)' — keeps the canonical name visible
# while ensuring substring search hits the popular alias.
# ────────────────────────────────────────────────────────────────────
NAME_OVERRIDES: dict[str, str] = {
    "ETERNAL":   "Eternal (formerly Zomato)",
    "TMPV":      "Tata Motors Passenger Vehicles (TaMo)",
    "BAJAJ-AUTO": "Bajaj Auto",  # keep canonical, just here as placeholder pattern
}


POPULAR_EXTRAS: list[tuple[str, str, str]] = [
    # ── Information Technology ──────────────────────────────────────
    ("LTIM",       "LTIMindtree",                  "Information Technology"),
    ("LTTS",       "L&T Technology Services",      "Information Technology"),
    ("PERSISTENT", "Persistent Systems",           "Information Technology"),
    ("MPHASIS",    "Mphasis",                      "Information Technology"),
    ("COFORGE",    "Coforge",                      "Information Technology"),
    ("OFSS",       "Oracle Financial Services",    "Information Technology"),
    ("KPITTECH",   "KPIT Technologies",            "Information Technology"),
    ("TATAELXSI",  "Tata Elxsi",                   "Information Technology"),

    # ── Financial Services — Banks ─────────────────────────────────
    ("INDUSINDBK", "IndusInd Bank",                "Financial Services"),
    ("IDFCFIRSTB", "IDFC First Bank",              "Financial Services"),
    ("FEDERALBNK", "Federal Bank",                 "Financial Services"),
    ("AUBANK",     "AU Small Finance Bank",        "Financial Services"),
    ("BANDHANBNK", "Bandhan Bank",                 "Financial Services"),
    ("YESBANK",    "Yes Bank",                     "Financial Services"),
    ("PNB",        "Punjab National Bank",         "Financial Services"),
    ("BANKBARODA", "Bank of Baroda",               "Financial Services"),
    ("CANBK",      "Canara Bank",                  "Financial Services"),
    ("UNIONBANK",  "Union Bank of India",          "Financial Services"),
    ("IOB",        "Indian Overseas Bank",         "Financial Services"),
    ("INDIANB",    "Indian Bank",                  "Financial Services"),

    # ── Financial Services — NBFCs / Insurance / Other ─────────────
    ("LICI",       "Life Insurance Corporation",   "Financial Services"),
    ("HDFCAMC",    "HDFC Asset Management",        "Financial Services"),
    ("NIPPONLIFE", "Nippon Life India AMC",        "Financial Services"),
    ("ICICIGI",    "ICICI Lombard General Insurance", "Financial Services"),
    ("ICICIPRULI", "ICICI Prudential Life",        "Financial Services"),
    ("MUTHOOTFIN", "Muthoot Finance",              "Financial Services"),
    ("MFSL",       "Max Financial Services",       "Financial Services"),
    ("CHOLAFIN",   "Cholamandalam Investment",     "Financial Services"),
    ("PFC",        "Power Finance Corporation",    "Financial Services"),
    ("RECLTD",     "REC Limited",                  "Financial Services"),
    ("IRFC",       "Indian Railway Finance Corp",  "Financial Services"),
    ("ANGELONE",   "Angel One",                    "Financial Services"),
    ("BSE",        "BSE Limited",                  "Financial Services"),
    ("MCX",        "Multi Commodity Exchange",     "Financial Services"),
    ("CDSL",       "Central Depository Services",  "Financial Services"),
    ("POLICYBZR",  "PB Fintech (Policybazaar)",    "Financial Services"),
    ("PAYTM",      "One 97 Communications (Paytm)", "Financial Services"),

    # ── Healthcare / Pharma ────────────────────────────────────────
    ("DIVISLAB",   "Divi's Laboratories",          "Healthcare"),
    ("LUPIN",      "Lupin",                        "Healthcare"),
    ("AUROPHARMA", "Aurobindo Pharma",             "Healthcare"),
    ("TORNTPHARM", "Torrent Pharmaceuticals",      "Healthcare"),
    ("ALKEM",      "Alkem Laboratories",           "Healthcare"),
    ("BIOCON",     "Biocon",                       "Healthcare"),
    ("GLENMARK",   "Glenmark Pharmaceuticals",     "Healthcare"),
    ("ZYDUSLIFE",  "Zydus Lifesciences",           "Healthcare"),
    ("IPCALAB",    "IPCA Laboratories",            "Healthcare"),
    ("MANKIND",    "Mankind Pharma",               "Healthcare"),
    ("FORTIS",     "Fortis Healthcare",            "Healthcare"),
    ("ABBOTINDIA", "Abbott India",                 "Healthcare"),

    # ── Automobiles / Auto Components ──────────────────────────────
    ("TVSMOTOR",   "TVS Motor Company",            "Automobile and Auto Components"),
    ("HEROMOTOCO", "Hero MotoCorp",                "Automobile and Auto Components"),
    ("ASHOKLEY",   "Ashok Leyland",                "Automobile and Auto Components"),
    ("MOTHERSON",  "Samvardhana Motherson Intl",   "Automobile and Auto Components"),
    ("BALKRISIND", "Balkrishna Industries",        "Automobile and Auto Components"),
    ("MRF",        "MRF",                          "Automobile and Auto Components"),
    ("APOLLOTYRE", "Apollo Tyres",                 "Automobile and Auto Components"),
    ("EXIDEIND",   "Exide Industries",             "Automobile and Auto Components"),
    ("BHARATFORG", "Bharat Forge",                 "Automobile and Auto Components"),
    ("ESCORTS",    "Escorts Kubota",               "Automobile and Auto Components"),
    ("BOSCHLTD",   "Bosch",                        "Automobile and Auto Components"),

    # ── Energy / Power / Oil & Gas ─────────────────────────────────
    ("TATAPOWER",  "Tata Power",                   "Power"),
    ("ADANIPOWER", "Adani Power",                  "Power"),
    ("ADANIGREEN", "Adani Green Energy",           "Power"),
    ("ADANIENSOL", "Adani Energy Solutions",       "Power"),
    ("NHPC",       "NHPC",                         "Power"),
    ("SJVN",       "SJVN",                         "Power"),
    ("CESC",       "CESC",                         "Power"),
    ("TORNTPOWER", "Torrent Power",                "Power"),
    ("IOC",        "Indian Oil Corporation",       "Oil, Gas & Consumable Fuels"),
    ("BPCL",       "Bharat Petroleum",             "Oil, Gas & Consumable Fuels"),
    ("HINDPETRO",  "Hindustan Petroleum",          "Oil, Gas & Consumable Fuels"),
    ("GAIL",       "GAIL (India)",                 "Oil, Gas & Consumable Fuels"),
    ("ATGL",       "Adani Total Gas",              "Oil, Gas & Consumable Fuels"),
    ("PETRONET",   "Petronet LNG",                 "Oil, Gas & Consumable Fuels"),
    ("OIL",        "Oil India",                    "Oil, Gas & Consumable Fuels"),

    # ── Metals & Mining ────────────────────────────────────────────
    ("VEDL",       "Vedanta",                      "Metals & Mining"),
    ("SAIL",       "Steel Authority of India",     "Metals & Mining"),
    ("JINDALSTEL", "Jindal Steel & Power",         "Metals & Mining"),
    ("NMDC",       "NMDC",                         "Metals & Mining"),
    ("NALCO",      "National Aluminium",           "Metals & Mining"),
    ("HINDCOPPER", "Hindustan Copper",             "Metals & Mining"),
    ("MOIL",       "MOIL",                         "Metals & Mining"),
    ("RATNAMANI",  "Ratnamani Metals & Tubes",     "Metals & Mining"),

    # ── Consumer / FMCG ────────────────────────────────────────────
    ("DMART",      "Avenue Supermarts (DMart)",    "Consumer Services"),
    ("DABUR",      "Dabur India",                  "Fast Moving Consumer Goods"),
    ("MARICO",     "Marico",                       "Fast Moving Consumer Goods"),
    ("GODREJCP",   "Godrej Consumer Products",     "Fast Moving Consumer Goods"),
    ("COLPAL",     "Colgate-Palmolive (India)",    "Fast Moving Consumer Goods"),
    ("BRITANNIA",  "Britannia Industries",         "Fast Moving Consumer Goods"),
    ("VBL",        "Varun Beverages",              "Fast Moving Consumer Goods"),
    ("UNITDSPR",   "United Spirits",               "Fast Moving Consumer Goods"),
    ("RADICO",     "Radico Khaitan",               "Fast Moving Consumer Goods"),
    ("NYKAA",      "FSN E-Commerce (Nykaa)",       "Consumer Services"),
    ("IRCTC",      "IRCTC",                        "Consumer Services"),
    ("JUBLFOOD",   "Jubilant FoodWorks",           "Consumer Services"),
    ("DEVYANI",    "Devyani International",        "Consumer Services"),
    ("ABFRL",      "Aditya Birla Fashion & Retail", "Consumer Services"),
    ("PAGEIND",    "Page Industries",              "Consumer Services"),
    ("TRENT",      "Trent",                        "Consumer Services"),  # duplicate of N50 but kept for safety

    # ── Capital Goods / Industrial ─────────────────────────────────
    ("SIEMENS",    "Siemens",                      "Capital Goods"),
    ("ABB",        "ABB India",                    "Capital Goods"),
    ("HAVELLS",    "Havells India",                "Consumer Durables"),
    ("VOLTAS",     "Voltas",                       "Consumer Durables"),
    ("WHIRLPOOL",  "Whirlpool of India",           "Consumer Durables"),
    ("CROMPTON",   "Crompton Greaves Consumer",    "Consumer Durables"),
    ("DIXON",      "Dixon Technologies",           "Consumer Durables"),
    ("HAL",        "Hindustan Aeronautics",        "Capital Goods"),
    ("BHEL",       "BHEL",                         "Capital Goods"),
    ("BDL",        "Bharat Dynamics",              "Capital Goods"),
    ("MAZDOCK",    "Mazagon Dock Shipbuilders",    "Capital Goods"),
    ("COCHINSHIP", "Cochin Shipyard",              "Capital Goods"),
    ("GRSE",       "Garden Reach Shipbuilders",    "Capital Goods"),
    ("CUMMINSIND", "Cummins India",                "Capital Goods"),
    ("THERMAX",    "Thermax",                      "Capital Goods"),
    ("ABFRL",      "Aditya Birla Fashion",         "Consumer Services"),  # dup safety

    # ── Cables / Wires / Electrical components ──────────────────────
    ("POLYCAB",    "Polycab India",                "Capital Goods"),
    ("KEI",        "KEI Industries",               "Capital Goods"),
    ("FINOLEXIND", "Finolex Industries",           "Capital Goods"),

    # ── Construction / Realty ──────────────────────────────────────
    ("DLF",        "DLF",                          "Realty"),
    ("GODREJPROP", "Godrej Properties",            "Realty"),
    ("LODHA",      "Macrotech Developers (Lodha)", "Realty"),
    ("PRESTIGE",   "Prestige Estates",             "Realty"),
    ("OBEROIRLTY", "Oberoi Realty",                "Realty"),
    ("BRIGADE",    "Brigade Enterprises",          "Realty"),
    ("PHOENIXLTD", "Phoenix Mills",                "Realty"),

    # ── Construction Materials / Cement ────────────────────────────
    ("AMBUJACEM",  "Ambuja Cements",               "Construction Materials"),
    ("ACC",        "ACC",                          "Construction Materials"),
    ("DALBHARAT",  "Dalmia Bharat",                "Construction Materials"),
    ("RAMCOCEM",   "Ramco Cements",                "Construction Materials"),
    ("JKCEMENT",   "JK Cement",                    "Construction Materials"),

    # ── Telecom ────────────────────────────────────────────────────
    ("IDEA",       "Vodafone Idea",                "Telecommunication"),
    ("INDUSTOWER", "Indus Towers",                 "Telecommunication"),
    ("TATACOMM",   "Tata Communications",          "Telecommunication"),

    # ── Chemicals / Specialty ──────────────────────────────────────
    ("PIDILITIND", "Pidilite Industries",          "Chemicals"),
    ("SRF",        "SRF",                          "Chemicals"),
    ("UPL",        "UPL",                          "Chemicals"),
    ("PIIND",      "PI Industries",                "Chemicals"),
    ("DEEPAKNTR",  "Deepak Nitrite",               "Chemicals"),
    ("AARTIIND",   "Aarti Industries",             "Chemicals"),
    ("NAVINFLUOR", "Navin Fluorine",               "Chemicals"),

    # ── Railways / Infra PSUs ──────────────────────────────────────
    ("RVNL",       "Rail Vikas Nigam",             "Construction"),
    ("IRCON",      "Ircon International",          "Construction"),
    ("NBCC",       "NBCC (India)",                 "Construction"),
    ("KEC",        "KEC International",            "Construction"),
    ("NCC",        "NCC Limited",                  "Construction"),
    ("ENGINERSIN", "Engineers India",              "Construction"),

    # ── Paints ─────────────────────────────────────────────────────
    ("BERGEPAINT", "Berger Paints",                "Consumer Durables"),
    ("KANSAINER",  "Kansai Nerolac Paints",        "Consumer Durables"),

    # ── Misc consumer / e-commerce ─────────────────────────────────
    ("IREDA",      "Indian Renewable Energy Dev",  "Financial Services"),
]


def fetch_nse_equity_list() -> list[tuple[str, str]]:
    """Download NSE's full listed-equity master (EQUITY_L.csv).

    Returns a list of (symbol, company_name) for every EQ/BE series
    listing. Requires network access to www.nseindia.com (run OFF the
    corporate VPN). Raises RuntimeError with an actionable message on
    any failure so the caller can fall back to the curated build.
    """
    try:
        import httpx  # local import: only needed on the --fetch-nse path
    except ImportError as exc:  # pragma: no cover - httpx is a project dep
        raise RuntimeError(
            "httpx is required for --fetch-nse (it's already a project dep; "
            "run `uv sync`)."
        ) from exc

    last_err: Exception | None = None
    # Tight timeouts: this runs on every app launch (see windows_setup/
    # launch.bat). A truly-blocked/offline laptop should fail in a few
    # seconds and fall back to the shipped index -- not hang for a minute.
    # connect=5s catches network-level blocks fast; read=12s tolerates a
    # slow-but-alive CDN.
    timeout = httpx.Timeout(12.0, connect=5.0)
    with httpx.Client(
        headers=_BROWSER_HEADERS, timeout=timeout, follow_redirects=True
    ) as client:
        # Prime cookies. NSE 403s the CSV without a warmed session.
        try:
            client.get(NSE_HOMEPAGE)
        except Exception as exc:
            last_err = exc

        for url in NSE_EQUITY_L_URLS:
            try:
                resp = client.get(url)
                resp.raise_for_status()
                return _parse_equity_l(resp.text)
            except Exception as exc:
                last_err = exc
                continue

    raise RuntimeError(
        "Couldn't download NSE EQUITY_L.csv. Are you OFF the corporate VPN? "
        f"(www.nseindia.com is blocked on many corp networks.) Last error: {last_err}"
    )


def _parse_equity_l(text: str) -> list[tuple[str, str]]:
    """Parse EQUITY_L.csv text -> [(symbol, name), ...] for EQ/BE series.

    NSE's header has inconsistent leading spaces (e.g. ' SERIES'), so we
    normalize keys by stripping whitespace before reading.
    """
    reader = csv.DictReader(io.StringIO(text))
    out: list[tuple[str, str]] = []
    for raw in reader:
        row = {(k or "").strip().upper(): (v or "").strip() for k, v in raw.items()}
        symbol = row.get("SYMBOL", "")
        name = row.get("NAME OF COMPANY", "")
        series = row.get("SERIES", "")
        if not symbol or series not in _KEEP_SERIES:
            continue
        out.append((symbol, name or symbol))
    if not out:
        raise RuntimeError(
            "EQUITY_L.csv parsed to 0 rows — the format may have changed."
        )
    return out


def build(fetch_nse: bool = False) -> None:
    """Build the search-index CSV and write to disk.

    When ``fetch_nse`` is True, the full NSE equity master is merged in on
    top of the Nifty 50 + curated extras (which keep their nicer sectors).
    """

    # Load Nifty 50 from the KB (single source of truth).
    with KB_STOCKS_JSON.open(encoding="utf-8") as fp:
        kb = json.load(fp)

    n50_rows: list[dict[str, str]] = []
    n50_tickers: set[str] = set()
    for stock in kb["stocks"]:
        ticker = stock["ticker"]
        n50_tickers.add(ticker)
        # Apply name override if one exists — lets users search by
        # popular aliases (e.g. 'zomato' → ETERNAL).
        name = NAME_OVERRIDES.get(ticker, stock["company_name"])
        n50_rows.append({
            "ticker": f"{ticker}.NS",
            "name": name,
            "sector": stock["sector"],
            "nifty50": "true",
        })

    # Dedup extras: skip anything already in N50 (defensive — POPULAR_EXTRAS
    # may contain duplicates of N50 names for safety during edits).
    seen: set[str] = set(n50_tickers)
    extra_rows: list[dict[str, str]] = []
    for ticker, name, sector in POPULAR_EXTRAS:
        if ticker in seen:
            continue
        seen.add(ticker)
        extra_rows.append({
            "ticker": f"{ticker}.NS",
            "name": name,
            "sector": sector,
            "nifty50": "false",
        })

    # Optionally merge the full NSE equity master. Curated rows win on
    # dedup, so popular names keep their hand-tuned sectors; the long tail
    # gets a neutral 'NSE Listed' sector.
    nse_rows: list[dict[str, str]] = []
    if fetch_nse:
        print("Downloading NSE EQUITY_L.csv (must be off-VPN)…")
        listings = fetch_nse_equity_list()
        if len(listings) < _MIN_NSE_ROWS:
            raise RuntimeError(
                f"NSE fetch returned only {len(listings)} rows "
                f"(expected ≥{_MIN_NSE_ROWS}); refusing to overwrite the "
                "bundled index with a suspiciously small list."
            )
        for symbol, name in listings:
            if symbol in seen:
                continue
            seen.add(symbol)
            nse_rows.append({
                "ticker": f"{symbol}.NS",
                "name": name,
                "sector": "NSE Listed",
                "nifty50": "false",
            })

    # Sort: Nifty 50 first (alphabetic within group), then curated extras,
    # then the full-NSE tail — each group alphabetic. This means simple
    # slicing can still do 'show N50 matches first'.
    n50_rows.sort(key=lambda r: r["ticker"])
    extra_rows.sort(key=lambda r: r["ticker"])
    nse_rows.sort(key=lambda r: r["ticker"])
    all_rows = n50_rows + extra_rows + nse_rows

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["ticker", "name", "sector", "nifty50"])
        writer.writeheader()
        writer.writerows(all_rows)

    n50_count = len(n50_rows)
    extra_count = len(extra_rows)
    nse_count = len(nse_rows)
    total = n50_count + extra_count + nse_count
    print(f"Wrote {OUTPUT_CSV.relative_to(REPO_ROOT)}")
    if fetch_nse:
        print(
            f"   {n50_count} Nifty 50 + {extra_count} curated extras + "
            f"{nse_count} full-NSE = {total} total tickers"
        )
    else:
        print(
            f"   {n50_count} Nifty 50 + {extra_count} popular extras = "
            f"{total} total tickers"
        )


def _rel(path: Path) -> Path:
    """Path relative to the repo root for display, or the path itself if it
    lives elsewhere (e.g. a tmp dir under test). Never raises."""
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def backfill_csv_sectors() -> None:
    """Overwrite every row's sector in the existing CSV with its real
    yfinance sector, resuming from the on-disk cache.

    Reads OUTPUT_CSV, resolves a real sector for each ticker (yfinance),
    and rewrites the file in place - rows, names and nifty50 flags are
    left untouched; only the sector column changes. Per the project
    decision, yfinance labels are used for ALL stocks (curated included)
    so the whole index shares one consistent taxonomy.

    Safe to re-run: cached tickers are skipped, and tickers that hit a
    transient rate-limit are retried on the next run. Rows yfinance has
    no sector for keep the neutral 'NSE Listed' label.
    """
    if not OUTPUT_CSV.exists():
        raise RuntimeError(
            f"{OUTPUT_CSV} not found - build the base index first "
            "(python scripts/build_search_index.py [--fetch-nse])."
        )

    with OUTPUT_CSV.open(encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    tickers = [r["ticker"] for r in rows]
    print(f"Backfilling sectors for {len(tickers)} tickers via yfinance...")
    print(f"   cache: {_rel(SECTOR_CACHE)} (resumable)")

    def _progress(done: int, total: int, failed: int) -> None:
        print(f"\r   resolved {done}/{total}  (failed/retry: {failed})",
              end="", flush=True)

    cache = backfill_sectors(
        tickers, cache_path=SECTOR_CACHE, progress=_progress,
    )
    print()  # newline after the progress line

    resolved = 0
    still_unknown = 0
    for r in rows:
        sector = cache.get(r["ticker"])
        if sector is None:
            # Never resolved (all attempts rate-limited). Keep existing.
            still_unknown += 1
            continue
        if sector == "NSE Listed":
            still_unknown += 1
        else:
            resolved += 1
        r["sector"] = sector

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=["ticker", "name", "sector", "nifty50"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {_rel(OUTPUT_CSV)}")
    print(f"   {resolved} real sectors, {still_unknown} still unresolved.")
    if still_unknown:
        print("   Re-run --backfill-sectors to retry the stragglers.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch-nse",
        action="store_true",
        help="Download NSE's full equity master and merge in every listed "
             "stock (run OFF the corporate VPN).",
    )
    parser.add_argument(
        "--backfill-sectors",
        action="store_true",
        help="Enrich the existing CSV with real yfinance sectors "
             "(resumable; run OFF the corporate VPN).",
    )
    args = parser.parse_args()
    try:
        if args.backfill_sectors:
            backfill_csv_sectors()
        else:
            build(fetch_nse=args.fetch_nse)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
