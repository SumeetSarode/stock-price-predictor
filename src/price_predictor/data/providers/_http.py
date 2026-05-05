"""Tiny shared helper for HTTP-based providers (Stooq, AlphaVantage).

WHY THIS EXISTS
===============
httpx defaults to certifi for TLS verification. On corporate networks
that MITM HTTPS (Walmart proxy, Zscaler, etc.) we need to point httpx
at a combined CA bundle that includes the corp roots.

The standard env vars for this -- SSL_CERT_FILE and REQUESTS_CA_BUNDLE
-- are NOT auto-honored by httpx the way they are by `requests`. We have
to read them explicitly and pass them via verify=... as an SSL context.

NOTE: settings.setup_network() is responsible for copying these vars
from `.env` into os.environ. We only read os.environ here.
"""
from __future__ import annotations

import os
import ssl


def get_verify_setting() -> ssl.SSLContext | bool:
    """Return the value to pass as httpx's `verify=` argument.

    Returns:
        - SSLContext built from the configured CA bundle if SSL_CERT_FILE
          or REQUESTS_CA_BUNDLE is set in the environment (corp proxy case).
        - True (use httpx's bundled certifi) otherwise.

    Why two env vars: SSL_CERT_FILE is the Python-stdlib name; REQUESTS_CA_BUNDLE
    is the requests-library convention. Both are commonly set in dev shells.
    We check SSL_CERT_FILE first because it's the more "official" one.

    Why an SSLContext (not a path string): httpx >= 0.28 deprecated
    verify=<str> in favor of pre-built SSL contexts. Building it here
    keeps the deprecation pain in one place.
    """
    ca_bundle = (
        os.environ.get("SSL_CERT_FILE")
        or os.environ.get("REQUESTS_CA_BUNDLE")
    )
    if ca_bundle:
        return ssl.create_default_context(cafile=ca_bundle)
    return True
