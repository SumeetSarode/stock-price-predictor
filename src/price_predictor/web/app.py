"""FastAPI application factory.

`create_app()` builds the app, mounts static assets, registers routers.
Kept as a factory (not a module-level singleton) so tests can spin up
isolated instances with patched dependencies.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from loguru import logger

from price_predictor._version import __version__
from price_predictor.web.routes import api, pages
from price_predictor.web.services.scheduler import grading_loop
from price_predictor.web.settings import settings


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Start/stop background tasks around the app's lifetime.

    The grading scheduler is opt-in (settings.enable_scheduler). When
    off, this is a no-op and the app has zero background side effects
    — which is exactly what the test suite relies on.
    """
    task: asyncio.Task | None = None
    if settings.enable_scheduler:
        task = asyncio.create_task(
            grading_loop(
                interval_seconds=settings.grading_interval_hours * 3600,
                startup_delay_seconds=settings.grading_startup_delay_seconds,
                limit=settings.grading_pass_limit,
            ),
            name="grading-scheduler",
        )
        logger.info("lifespan: grading scheduler task spawned")
    else:
        logger.debug("lifespan: grading scheduler disabled (enable_scheduler=False)")

    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info("lifespan: grading scheduler task stopped")


def create_app() -> FastAPI:
    """Build a configured FastAPI instance."""
    app = FastAPI(
        title="Stock Price Predictor",
        description="Local-first AI-driven price predictions for NSE stocks.",
        version=__version__,
        docs_url="/api/docs",   # Swagger UI lives under /api/docs so / stays clean.
        redoc_url=None,         # ReDoc is fine but redundant.
        lifespan=_lifespan,
    )

    # Static assets — everything under frontend/ is browseable at /static/...
    # (e.g. /static/styles/tokens.css, /static/vendor/htmx-1.9.12.min.js)
    app.mount(
        "/static",
        StaticFiles(directory=str(settings.static_dir)),
        name="static",
    )

    # The 'How it works' walkthrough (docs/how_it_works.html + report/
    # chapters + assets) served read-only at /docs, linked from the nav.
    # Guarded: docs are OPTIONAL, so a deploy without docs/ (e.g. a slim
    # checkout) still boots -- unlike frontend/, whose absence is fatal.
    # html=True lets extensionless paths resolve and 404s render cleanly.
    if settings.docs_dir.is_dir():
        app.mount(
            "/docs",
            StaticFiles(directory=str(settings.docs_dir), html=True),
            name="docs",
        )
    else:
        logger.warning(
            f"docs/ not found at {settings.docs_dir} -- the 'How it works' "
            f"nav link will 404. This is non-fatal."
        )

    # Routers — keep page routes and API routes cleanly separated.
    app.include_router(pages.router)
    app.include_router(api.router)

    return app


# Module-level singleton for uvicorn's import string.
# `uvicorn price_predictor.web.app:app` will find this.
app = create_app()
