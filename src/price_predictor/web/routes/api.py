"""API routes — JSON or HTML-partial responses for HTMX swaps.

Convention:
  - If the request has `HX-Request: true` header, return an HTML
    partial (rendered template fragment) for direct swap.
  - Otherwise, return JSON for programmatic consumers.

Step 1 ships only POST /api/predict. More endpoints in Step 2 (watchlist,
history, search autocomplete, intraday prices).
"""
from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from price_predictor.web.services.dashboard_service import (
    get_dashboard,
    snapshot_with_watchlist,
)
from price_predictor.web.services.prediction_service import (
    PredictionServiceError,
    run_prediction,
)
from price_predictor.web.services.search_service import search as search_stocks
from price_predictor.web.services.watchlist_service import is_watched, toggle as toggle_watchlist
from price_predictor.web.settings import settings

router = APIRouter(prefix="/api")
templates = Jinja2Templates(directory=str(settings.templates_dir))


def _is_htmx(request: Request) -> bool:
    """True iff the request came from an HTMX swap."""
    return request.headers.get("HX-Request", "").lower() == "true"


@router.post("/predict", response_model=None)
async def predict_endpoint(
    request: Request,
    ticker: str = Form(...),
    horizon: str = Form("weekly"),
) -> HTMLResponse | JSONResponse:
    """Run a prediction and return either an HTML partial or JSON.

    `response_model=None` tells FastAPI to skip Pydantic response-model
    inference (the union return type isn't a valid Pydantic field).
    The actual response shape is handled by the concrete Response
    objects we return below.
    """
    try:
        view = await run_prediction(ticker=ticker, horizon=horizon)
    except PredictionServiceError as exc:
        if _is_htmx(request):
            return templates.TemplateResponse(
                request=request,
                name="components/error.html",
                context={"message": exc.message, "hint": exc.hint},
                status_code=400,
            )
        return JSONResponse(
            status_code=400,
            content={"error": exc.message, "hint": exc.hint},
        )

    if _is_htmx(request):
        return templates.TemplateResponse(
            request=request,
            name="components/prediction_card.html",
            context={"result": view},
        )
    return JSONResponse(content=view)


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness check. Useful for Docker, scripts, and panic-debugging."""
    return {"status": "ok"}


# ───────────────────────────────────────────────────────────────────────
# Search / autocomplete
# ───────────────────────────────────────────────────────────────────────


@router.get("/search", response_model=None)
async def search_endpoint(
    request: Request,
    q: str = "",
    limit: int = 8,
) -> HTMLResponse | JSONResponse:
    """Autocomplete search across the bundled ticker index.

    Returns:
        - HTML partial (suggestions dropdown) when called from HTMX
        - JSON list of matches otherwise

    Empty query returns an empty result — we don't show 'all tickers'
    on focus; that's the dropdown closing, not opening.
    """
    # Clamp limit to a sane range to prevent abuse / accidents.
    limit = max(1, min(limit, 20))

    matches = search_stocks(q, limit=limit)

    if _is_htmx(request):
        return templates.TemplateResponse(
            request=request,
            name="components/search_suggestions.html",
            context={"matches": matches, "query": q},
        )
    return JSONResponse(content={
        "query": q,
        "matches": [m.to_dict() for m in matches],
    })


# ───────────────────────────────────────────────────────────────────────
# Nifty 50 dashboard
# ───────────────────────────────────────────────────────────────────────


@router.get("/dashboard", response_model=None)
async def dashboard_endpoint(
    request: Request,
    refresh: bool = False,
) -> HTMLResponse | JSONResponse:
    """Return the Nifty 50 dashboard (HTML partial for HTMX, JSON otherwise).

    Cache-aware: first call of the day takes ~5-10s, later calls are
    instant until tomorrow.
    """
    snapshot = await get_dashboard(force_refresh=refresh)
    snapshot = snapshot_with_watchlist(snapshot)

    if _is_htmx(request):
        return templates.TemplateResponse(
            request=request,
            name="components/dashboard_table.html",
            context={"snapshot": snapshot},
        )
    return JSONResponse(content={
        "fetched_at": snapshot.fetched_at.isoformat(),
        "trading_day": snapshot.trading_day.isoformat() if snapshot.trading_day else None,
        "rows": [r.to_dict() for r in snapshot.rows],
    })


# ───────────────────────────────────────────────────────────────────────
# Watchlist
# ───────────────────────────────────────────────────────────────────────


@router.post("/watchlist/toggle", response_model=None)
async def toggle_watchlist_endpoint(
    request: Request,
    ticker: str = Form(...),
) -> HTMLResponse | JSONResponse:
    """Star or unstar a ticker. Returns the swapped star button HTML for HTMX.

    Response also includes an HX-Trigger header firing 'watchlist-changed'
    so any listening components (the side panel in Phase 3) can refresh.
    """
    now_watched, was_full = toggle_watchlist(ticker)

    if _is_htmx(request):
        response = templates.TemplateResponse(
            request=request,
            name="components/star_button.html",
            context={
                "ticker": ticker,
                "is_watched": now_watched,
                "toast": "Watchlist is full (10 max)" if was_full else None,
            },
        )
        # Custom HTMX event so future side-panel listens for changes.
        response.headers["HX-Trigger"] = "watchlist-changed"
        return response

    return JSONResponse(content={
        "ticker": ticker,
        "is_watched": now_watched,
        "watchlist_full": was_full,
    })
