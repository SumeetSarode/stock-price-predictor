"""Tiny shared helper for HTTP-based providers (Stooq, AlphaVantage).

WHY THIS EXISTS
===============
httpx defaults to certifi for TLS verification. On networks that inspect
HTTPS via a TLS-intercepting proxy we need to point httpx at a combined
CA bundle that includes the proxy's roots.

The standard env vars for this -- SSL_CERT_FILE and REQUESTS_CA_BUNDLE
-- are NOT auto-honored by httpx the way they are by `requests`. We have
to read them explicitly and pass them via verify=... as an SSL context.

NOTE: settings.setup_network() is responsible for copying these vars
from `.env` into os.environ. We only read os.environ here.

DEFENSIVE BEHAVIOR
==================
If the configured CA bundle file is missing or unreadable (e.g. someone
cloned the repo with a stale .env pointing at a path that doesn't exist
on their machine), we LOG and fall back to certifi rather than crashing.
WHY: a missing custom-CA bundle isn't a security risk — falling back to
certifi just means we trust the same public CAs every browser does. The
only case it'd break is on a network where a proxy inspects HTTPS;
there the user gets a clear TLS error and knows to fix their .env.
"""
from __future__ import annotations

import os
import ssl
from pathlib import Path

from loguru import logger


def get_verify_setting() -> ssl.SSLContext | bool:
    """Return the value to pass as httpx's `verify=` argument.

    Returns:
        - SSLContext built from the configured CA bundle if SSL_CERT_FILE
          or REQUESTS_CA_BUNDLE points to an existing readable file (corp
          proxy case).
        - True (use httpx's bundled certifi) if no CA bundle is configured
          OR the configured path doesn't exist (defensive fallback).

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
    if not ca_bundle:
        return True

    # Defensive: a stale .env path on a fresh clone shouldn't blow up
    # everything. Fall back to certifi with a one-line warning.
    if not Path(ca_bundle).is_file():
        logger.warning(
            f"[providers._http] Configured CA bundle does not exist: "
            f"{ca_bundle!r}. Falling back to certifi (system default). "
            "This is fine on normal networks; behind a TLS-inspecting proxy "
            "you'll need to point SSL_CERT_FILE at a real bundle."
        )
        return True

    return ssl.create_default_context(cafile=ca_bundle)
