#!/usr/bin/env bash
# check_no_html_in_python.sh
#
# Discipline gate: NO HTML in Python files inside src/price_predictor/web/.
# All HTML must live in frontend/templates/. The backend emits ONLY data.
#
# This protects the strict frontend/backend separation declared in
# docs/user_ui_design.md §2 ("Strict separation rule"). A future-us
# tempted to f-string a <div> for "just this one place" will trip this
# script and be forced to add a template instead.
#
# Exit 0  -> clean
# Exit 1  -> HTML found in Python
#
# Run from the repo root:  ./scripts/check_no_html_in_python.sh

set -euo pipefail

cd "$(dirname "$0")/.."

# We look for opening tags like <div, <span, <button, <h1, <p,
# <a href, <input, <form etc. The regex matches "<" followed by a
# letter — broad enough to catch any HTML tag, narrow enough to skip
# Python generics like dict[str, int] or comparisons like a < b.
#
# Excludes:
#   --glob=!*.md   docs may contain HTML examples
#   --glob=!*.sh   shell scripts may grep for HTML
#   The check itself (this file)
PATTERN='<[a-zA-Z]'
TARGET='src/price_predictor/web/'

if rg -n "$PATTERN" "$TARGET" 2>/dev/null; then
    echo ""
    echo "🚨 HTML found in Python files inside $TARGET."
    echo "   All HTML belongs in frontend/templates/."
    echo "   See docs/user_ui_design.md §2 for the rationale."
    exit 1
fi

echo "✅ No HTML in $TARGET — separation intact."
