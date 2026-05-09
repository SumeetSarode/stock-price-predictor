#!/usr/bin/env bash
# ============================================================================
# Off-VPN research fetcher for price_predictor constants dossier
# ============================================================================
#
# WHY THIS EXISTS
# ---------------
# Walmart VPN blocks several domains that contain authoritative quantitative
# definitions for technical-analysis constants we use in this codebase:
#
#   - school.stockcharts.com  (SSL handshake fails on VPN)
#   - investopedia.com        (allowlist-blocked on VPN)
#   - ssrn.com                (academic papers — blocked)
#   - nseindia.com            (Indian market price-band data — blocked)
#   - sebi.gov.in             (Indian regulator — blocked)
#   - jstor.org               (academic papers — blocked)
#
# The on-VPN session already grabbed everything from Wikipedia and
# thepatternsite.com (Bulkowski) successfully. This script grabs the rest.
#
# HOW TO RUN
# ----------
#   1. Disconnect from Walmart VPN (use personal wifi / hotspot)
#   2. cd <repo root>
#   3. bash scripts/fetch_research_offvpn.sh
#   4. Reconnect to VPN
#   5. Tell Thor: "off-VPN fetch done, refresh the dossier"
#
# OUTPUT
# ------
# All pages land in /tmp/research_offvpn/ (HTML) and /tmp/research_offvpn/txt/
# (plain text via pandoc). Thor reads from /tmp/research_offvpn/txt/ to update
# the dossier with citations.
#
# SAFE TO RE-RUN
# --------------
# Idempotent. Each curl writes to a fixed filename, overwriting prior runs.
# ============================================================================

set -uo pipefail   # NOT -e: we want to continue on individual fetch failures

OUT="/tmp/research_offvpn"
mkdir -p "${OUT}" "${OUT}/txt"
cd "${OUT}"

UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# ----------------------------------------------------------------------------
# Helper: fetch a URL, save as HTML, log status
# ----------------------------------------------------------------------------
fetch() {
    local url="$1"
    local fname="$2"
    local desc="$3"
    printf "  [%-40s] " "${fname}"
    code=$(curl -sS --max-time 30 -L -A "${UA}" \
                "${url}" -o "${fname}" -w "%{http_code}" 2>/dev/null || echo "ERR")
    size=$(wc -c < "${fname}" 2>/dev/null || echo 0)
    printf "HTTP %s, %s bytes — %s\n" "${code}" "${size}" "${desc}"
}

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " StockCharts ChartSchool (canonical indicator parameters)"
echo "════════════════════════════════════════════════════════════════════"
SC="https://school.stockcharts.com/doku.php?id=technical_indicators"
fetch "${SC}:relative_strength_index_rsi"        "sc_rsi.html"          "RSI definitions + thresholds"
fetch "${SC}:average_directional_index_adx"      "sc_adx.html"          "ADX bands (Wilder original)"
fetch "${SC}:on_balance_volume_obv"              "sc_obv.html"          "OBV divergence quantification"
fetch "${SC}:bollinger_bands"                    "sc_bbands.html"       "Bollinger %B interpretation"
fetch "${SC}:moving_average_convergence_divergence_macd" "sc_macd.html" "MACD parameters"
fetch "${SC}:stochastic_oscillator_fast_slow_and_full"  "sc_stoch.html" "Stochastic settings"
fetch "${SC}:average_true_range_atr"             "sc_atr.html"          "ATR Wilder smoothing"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " StockCharts ChartSchool — chart patterns"
echo "════════════════════════════════════════════════════════════════════"
SCP="https://school.stockcharts.com/doku.php?id=chart_analysis:chart_patterns"
fetch "${SCP}:head_and_shoulders_top_reversal"   "sc_hst.html"          "H&S top tolerance specifics"
fetch "${SCP}:double_top_reversal"               "sc_doubletop.html"    "Double top peak tolerance"
fetch "${SCP}:double_bottom_reversal"            "sc_doublebot.html"    "Double bottom trough depth"
fetch "${SCP}:symmetrical_triangle_continuation" "sc_symtri.html"       "Triangle pivot count"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " StockCharts ChartSchool — candlestick patterns"
echo "════════════════════════════════════════════════════════════════════"
SCC="https://school.stockcharts.com/doku.php?id=chart_analysis:candlestick_pattern_dictionary"
fetch "${SCC}"                                   "sc_candles_index.html" "Candlestick pattern catalog"
fetch "https://school.stockcharts.com/doku.php?id=chart_analysis:introduction_to_candlesticks" \
      "sc_candles_intro.html" "Candlestick body/shadow definitions (Nison summary)"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " Investopedia (industry-standard practitioner explanations)"
echo "════════════════════════════════════════════════════════════════════"
INV="https://www.investopedia.com/terms"
fetch "${INV}/r/rsi.asp"                "inv_rsi.asp.html"          "RSI thresholds"
fetch "${INV}/a/adx.asp"                "inv_adx.asp.html"          "ADX strength bands"
fetch "${INV}/d/doji.asp"               "inv_doji.asp.html"         "Doji body-size threshold"
fetch "${INV}/h/hammer.asp"             "inv_hammer.asp.html"       "Hammer shadow:body ratio"
fetch "${INV}/h/headandshoulders.asp"   "inv_hs.asp.html"           "H&S symmetry guidelines"
fetch "${INV}/d/doubletop.asp"          "inv_doubletop.asp.html"    "Double-top peak tolerance"
fetch "${INV}/o/onbalancevolume.asp"    "inv_obv.asp.html"          "OBV divergence quantification"
fetch "${INV}/p/percent-b.asp"          "inv_percent_b.asp.html"    "Bollinger %B interpretation"
fetch "${INV}/a/atr.asp"                "inv_atr.asp.html"          "ATR percentile bands"
fetch "${INV}/t/trading-curb.asp"       "inv_curb.asp.html"         "Circuit breakers / price bands"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " NSE India (price bands for individual securities)"
echo "════════════════════════════════════════════════════════════════════"
fetch "https://www.nseindia.com/regulations/circular/regulatory-archives" \
      "nse_archives.html" "Price band circulars index"
fetch "https://www.nseindia.com/products-services/equity-market-trading" \
      "nse_equity.html" "Trading mechanism overview"
fetch "https://www.nseindia.com/market-data/price-band-meeting" \
      "nse_priceband.html" "Current price band rules"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " SEBI (regulator on price bands & circuit filters)"
echo "════════════════════════════════════════════════════════════════════"
fetch "https://www.sebi.gov.in/sebi_data/faqfiles/aug-2023/1693204748946.pdf" \
      "sebi_faq.pdf" "SEBI FAQ on circuit filters (PDF)"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " SSRN (academic — Indian-equity event-study half-lives)"
echo "════════════════════════════════════════════════════════════════════"
DDG="https://html.duckduckgo.com/html"
fetch "${DDG}/?q=site%3Assrn.com+nifty+event+study+price+impact+half+life" \
      "ssrn_nifty_search.html" "SSRN search: NIFTY event-study half-life"
fetch "${DDG}/?q=site%3Assrn.com+indian+stock+market+news+impact+abnormal+return" \
      "ssrn_news_search.html" "SSRN search: Indian news impact"
fetch "${DDG}/?q=bulkowski+head+and+shoulders+symmetry+failure+rate" \
      "ddg_hs_symmetry.html" "Search: H&S symmetry quantification"
fetch "${DDG}/?q=bulkowski+double+top+peak+variance+tolerance" \
      "ddg_doubletop_tol.html" "Search: double-top peak tolerance"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " Convert all HTML to plain text for grep-ability"
echo "════════════════════════════════════════════════════════════════════"
if ! command -v pandoc &>/dev/null; then
    echo "  ⚠️  pandoc not installed. Install: brew install pandoc"
    echo "     (text conversion skipped; HTML files still saved)"
else
    for f in *.html; do
        [ -f "$f" ] || continue
        base="${f%.html}"
        pandoc -f html -t plain --wrap=none "$f" -o "txt/${base}.txt" 2>/dev/null \
            && printf "  ✓ %s\n" "${base}.txt" \
            || printf "  ✗ %s (pandoc failed)\n" "${base}.txt"
    done
fi

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " DONE"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "  Output dir:     ${OUT}/"
echo "  Plain text:     ${OUT}/txt/"
echo "  HTML files:     $(ls ${OUT}/*.html 2>/dev/null | wc -l | tr -d ' ') saved"
echo "  Text files:     $(ls ${OUT}/txt/*.txt 2>/dev/null | wc -l | tr -d ' ') saved"
echo ""
echo "  Now reconnect to Walmart VPN and tell Thor:"
echo "    \"off-VPN fetch done, refresh the dossier\""
echo ""
echo "  Thor will read from ${OUT}/txt/ and update:"
echo "    - docs/research/constants_dossier.md"
echo "    - inline citations in source files"
echo ""
