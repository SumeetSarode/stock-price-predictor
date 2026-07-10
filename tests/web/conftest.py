"""Shared fixtures for web-layer tests.

The web app is built via the `create_app()` factory (never the module
singleton) so each test gets an isolated FastAPI instance. A Starlette
TestClient wraps it for synchronous request/response assertions.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from price_predictor.web.app import create_app


@pytest.fixture
def app():
    """A fresh FastAPI app per test."""
    return create_app()


@pytest.fixture
def client(app):
    """Synchronous TestClient over the app.

    NOTE: TestClient uses httpx's ASGI transport, which does not run
    uvicorn's httptools writer. That means it will NOT itself raise the
    'Response content longer than Content-Length' RuntimeError that bit
    us in production. So the Content-Length regression test asserts the
    *invariant* directly (header == len(body)) rather than relying on the
    transport to blow up. See test_prediction_endpoint_regressions.py.
    """
    return TestClient(app)
