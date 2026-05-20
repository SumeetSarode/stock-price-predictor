"""Build the bundled search index CSV from authoritative sources.

Why a script (instead of hand-editing a CSV):
  - The Nifty 50 portion stays in lock-step with data/kb/stocks.json
    (single source of truth — no chance of drift).
  - The non-N50 'popular extras' are organized in Python code, grouped
    by sector with comments, so future-us can audit + extend cleanly.
  - Reproducible: anyone can `python scripts/build_search_index.py`
    and get the same CSV byte-for-byte.

Run:
    python scripts/build_search_index.py

Output:
    frontend/data/nifty500.csv      (ticker,name,sector,nifty50)

Future enhancement: add a `--fetch-nifty500` flag that downloads the
official NSE Nifty 500 CSV to replace the hand-curated extras list.
For v1, we ship with ~150 popular tickers — enough that search feels
useful for >95% of retail-investor queries.

Naming note: the CSV is called nifty500.csv even though v1 has ~150
entries — the name signals the eventual target. It's a forward-
compatible filename, not a current claim.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

# Repo root = parent of this script's parent (scripts/).
REPO_ROOT = Path(__file__).resolve().parent.parent
KB_STOCKS_JSON = REPO_ROOT / "data" / "kb" / "stocks.json"
OUTPUT_CSV = REPO_ROOT / "frontend" / "data" / "nifty500.csv"


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


def build() -> None:
    """Build the search-index CSV and write to disk."""

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

    # Sort: Nifty 50 first (alphabetic within group), then extras
    # (alphabetic). This means binary-search-style autocomplete can do
    # smart things like "show N50 matches first" with simple slicing.
    n50_rows.sort(key=lambda r: r["ticker"])
    extra_rows.sort(key=lambda r: r["ticker"])
    all_rows = n50_rows + extra_rows

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["ticker", "name", "sector", "nifty50"])
        writer.writeheader()
        writer.writerows(all_rows)

    n50_count = len(n50_rows)
    extra_count = len(extra_rows)
    print(f"✅ Wrote {OUTPUT_CSV.relative_to(REPO_ROOT)}")
    print(f"   {n50_count} Nifty 50 + {extra_count} popular extras = {n50_count + extra_count} total tickers")


if __name__ == "__main__":
    build()
