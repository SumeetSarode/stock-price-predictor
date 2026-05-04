#!/usr/bin/env bash
# Build a combined CA bundle = certifi public roots + Walmart proxy roots.
#
# WHY: On Walmart corp network, HTTPS traffic goes through proxy-intlho which
# MITM-injects its own self-signed root cert. Python's default ssl trust store
# (certifi) doesn't know about it -> CERTIFICATE_VERIFY_FAILED on any HTTPS call.
# Conversely, if we point at JUST the Walmart bundle (4 certs), nothing public
# verifies when off-VPN. So: concatenate both, get the best of both worlds.
#
# USAGE:
#   ./scripts/build_ca_bundle.sh
# Then .env points SSL_CERT_FILE / REQUESTS_CA_BUNDLE at the output file.
#
# Re-run this whenever:
#   - Code Puppy updates its Walmart cert bundle
#   - certifi updates (rare; generally only matters on cert revocations)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/certs/ca-bundle.pem"
mkdir -p "$ROOT/certs"

# 1. Find certifi's public CA bundle (shipped in our project venv)
CERTIFI_BUNDLE="$(uv run python -c 'import certifi; print(certifi.where())')"
if [ ! -f "$CERTIFI_BUNDLE" ]; then
  echo "ERROR: certifi bundle not found at $CERTIFI_BUNDLE" >&2
  exit 1
fi

# 2. Find Code Puppy's Walmart cert bundle (4 corp roots)
WALMART_BUNDLE="/Users/s0s0jww/.code-puppy-venv/lib/python3.13/site-packages/code_puppy/plugins/walmart_specific/certs/walmart-bundle.pem"
if [ ! -f "$WALMART_BUNDLE" ]; then
  echo "ERROR: Walmart bundle not found at $WALMART_BUNDLE" >&2
  echo "       Is Code Puppy installed? (https://puppy.walmart.com)" >&2
  exit 1
fi

# 3. Concatenate -- order doesn't matter; OpenSSL walks the list looking for any match
{
  echo "# === certifi public CAs ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="
  cat "$CERTIFI_BUNDLE"
  echo ""
  echo "# === Walmart corp proxy CAs ==="
  cat "$WALMART_BUNDLE"
} > "$OUT"

CERT_COUNT=$(grep -c "BEGIN CERTIFICATE" "$OUT")
echo "Wrote combined bundle: $OUT"
echo "  certs: $CERT_COUNT"
echo "  bytes: $(wc -c < "$OUT" | tr -d ' ')"
echo ""
echo "Make sure .env contains:"
echo "  SSL_CERT_FILE=$OUT"
echo "  REQUESTS_CA_BUNDLE=$OUT"
