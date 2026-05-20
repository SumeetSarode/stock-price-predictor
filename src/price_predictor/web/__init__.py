"""Web layer — FastAPI app serving the local-first UI.

Strict separation: this package emits ONLY data (JSON via API routes,
context dicts via page routes). HTML lives in ../../../frontend/templates.

A linter check (scripts/check_no_html_in_python.sh) enforces that no
Python file in this package contains raw HTML strings.
"""
