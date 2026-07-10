"""Shared fixtures for web-layer tests.

The web app is built via the `create_app()` factory (never the module
singleton) so each test gets an isolated FastAPI instance. A Starlette
TestClient wraps it for synchronous request/response assertions.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from price_predictor.web.app import create_app
from price_predictor.web.services import db as db_module
from price_predictor.web.settings import settings


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


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point the web SQLite DB at an isolated temp file per test.

    Patches settings.db_path (read at call-time by db._resolve_db_path)
    and resets the module's one-shot init flag so the schema is created
    fresh in the temp location. Restores the flag after the test so we
    don't leak a stale 'initialized' state into the next test.
    """
    test_db = tmp_path / "test_app.db"
    monkeypatch.setattr(settings, "db_path", test_db)
    db_module.reset_for_tests()
    yield test_db
    db_module.reset_for_tests()
