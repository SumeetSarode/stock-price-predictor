"""Page routes — render full HTML pages via Jinja2 templates.

These handlers ONLY pass context dicts to templates. No HTML strings,
no business logic. Pure routing + context assembly.
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from price_predictor.web.services.search_service import get_by_ticker
from price_predictor.web.settings import settings

router = APIRouter()
templates = Jinja2Templates(directory=str(settings.templates_dir))


# Single source of truth for the version string we render in the footer.
# Pulled from pyproject.toml at runtime would be nicer, but importlib.metadata
# in a uv-build pre-installed env is fine — keep it stub for v1.
APP_VERSION = "0.1.0-dev"


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """Landing page — the predict form."""
    return templates.TemplateResponse(
        request=request,
        name="pages/home.html",
        context={"app_version": APP_VERSION},
    )


@router.get("/history", response_class=HTMLResponse)
async def history(request: Request) -> HTMLResponse:
    """Prediction history — placeholder for v1, real impl in Step 2F."""
    # Step 1 placeholder so the nav link works. Step 2F will render the
    # actual history table from SQLite.
    return templates.TemplateResponse(
        request=request,
        name="pages/home.html",  # reuses home for now; gets its own template in Step 2F
        context={"app_version": APP_VERSION},
    )


@router.get("/stock/{ticker}", response_class=HTMLResponse)
async def stock_detail(request: Request, ticker: str) -> HTMLResponse:
    """Stock detail page — reached by clicking a search suggestion.

    For Step 2A this is a placeholder: shows the matched stock's name
    + sector and embeds the existing predict form pre-filled with the
    ticker. The full detail page (price chart, recent predictions,
    fundamentals card) lands in Step 2G.
    """
    stock = get_by_ticker(ticker)
    return templates.TemplateResponse(
        request=request,
        name="pages/stock_detail.html",
        context={
            "app_version": APP_VERSION,
            "ticker": ticker.upper(),
            "stock": stock,  # may be None if ticker is unknown
        },
    )
