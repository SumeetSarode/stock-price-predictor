"""FastAPI application factory.

`create_app()` builds the app, mounts static assets, registers routers.
Kept as a factory (not a module-level singleton) so tests can spin up
isolated instances with patched dependencies.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from price_predictor.web.routes import api, pages
from price_predictor.web.settings import settings


def create_app() -> FastAPI:
    """Build a configured FastAPI instance."""
    app = FastAPI(
        title="Price Predictor",
        description="Local-first AI-driven price predictions for NSE stocks.",
        version="0.1.0-dev",
        docs_url="/api/docs",   # Swagger UI lives under /api/docs so / stays clean.
        redoc_url=None,         # ReDoc is fine but redundant.
    )

    # Static assets — everything under frontend/ is browseable at /static/...
    # (e.g. /static/styles/tokens.css, /static/vendor/htmx-1.9.12.min.js)
    app.mount(
        "/static",
        StaticFiles(directory=str(settings.static_dir)),
        name="static",
    )

    # Routers — keep page routes and API routes cleanly separated.
    app.include_router(pages.router)
    app.include_router(api.router)

    return app


# Module-level singleton for uvicorn's import string.
# `uvicorn price_predictor.web.app:app` will find this.
app = create_app()
