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

from price_predictor.web.services.prediction_service import (
    PredictionServiceError,
    run_prediction,
)
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
