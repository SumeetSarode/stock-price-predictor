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

from price_predictor.web.services.analysis_service import (
    AnalysisServiceError,
    compute_live_analysis,
)
from price_predictor.web.services.dashboard_service import (
    get_dashboard,
    snapshot_with_watchlist,
)
from price_predictor.web.services.chart_service import get_chart_series
from price_predictor.web.services.detail_service import get_stock_detail
from price_predictor.web.services.grading_service import grade_ticker
from price_predictor.web.services.news_service import fetch_recent_headlines
from price_predictor.web.services.panel_service import get_one_card, get_panel_cards
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

    # Derive the top-of-page widgets from the same snapshot — no extra
    # fetches. Keeps the home page atomic: one refresh updates all
    # three widgets together (summary bar, movers strips, table).
    from price_predictor.web.services.market_summary_service import (
        get_movers,
        summarize_market,
    )
    summary = summarize_market(snapshot)
    movers = get_movers(snapshot, top_n=5)

    if _is_htmx(request):
        return templates.TemplateResponse(
            request=request,
            name="components/dashboard_table.html",
            context={
                "snapshot": snapshot,
                "summary": summary,
                "movers": movers,
            },
        )
    return JSONResponse(content={
        "fetched_at": snapshot.fetched_at.isoformat(),
        "trading_day": snapshot.trading_day.isoformat() if snapshot.trading_day else None,
        "summary": {
            "avg_change_pct": summary.avg_change_pct,
            "n_advancing": summary.n_advancing,
            "n_declining": summary.n_declining,
            "n_unchanged": summary.n_unchanged,
            "n_total": summary.n_total,
        },
        "movers": {
            "gainers": [r.to_dict() for r in movers.gainers],
            "losers":  [r.to_dict() for r in movers.losers],
        },
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


# ───────────────────────────────────────────────────────────────────────
# Predictions panel (left sidebar)
# ───────────────────────────────────────────────────────────────────────


_VALID_HORIZONS = {"daily", "weekly", "biweekly", "monthly"}


@router.get("/predictions/panel", response_model=None)
async def predictions_panel_endpoint(
    request: Request,
    horizon: str = "weekly",
) -> HTMLResponse | JSONResponse:
    """Return the watchlist predictions panel body for the given horizon.

    Phase 2 returns price/context cards. Phase 3 will add cached
    prediction data per card.
    """
    horizon = horizon.lower().strip()
    if horizon not in _VALID_HORIZONS:
        horizon = "weekly"

    cards = await get_panel_cards(horizon=horizon)

    if _is_htmx(request):
        return templates.TemplateResponse(
            request=request,
            name="components/panel_body.html",
            context={"cards": cards, "horizon": horizon},
        )
    return JSONResponse(content={
        "horizon": horizon,
        "cards": [
            {
                "ticker": c.ticker, "name": c.name, "sector": c.sector,
                "is_nifty50": c.is_nifty50, "close": c.close,
                "change_pct": c.change_pct, "direction": c.price_direction,
                "has_prediction": c.prediction is not None,
            }
            for c in cards
        ],
    })


@router.post("/predictions/run", response_model=None)
async def run_prediction_endpoint(
    request: Request,
    ticker: str = Form(...),
    horizon: str = Form("weekly"),
    context: str = Form("panel"),
) -> HTMLResponse | JSONResponse:
    """Run a prediction for `ticker` at `horizon`, save to cache, return
    the updated UI partial.

    `context` decides which partial we return on HTMX requests:
      - "panel" (default) → components/panel_card.html   (single watchlist card swap)
      - "detail"          → components/detail_body.html  (stock detail page body swap)

    Long-running: ~30-60s. The caller's hx-disabled-elt + hx-indicator
    handle the pending UI state on the client.
    """
    horizon_norm = horizon.lower().strip()
    if horizon_norm not in _VALID_HORIZONS:
        horizon_norm = "weekly"

    error_message: str | None = None
    try:
        await run_prediction(ticker, horizon_norm)
    except PredictionServiceError as exc:
        error_message = exc.message

    if _is_htmx(request):
        if context == "detail":
            detail = await get_stock_detail(ticker, horizon_norm)
            response = templates.TemplateResponse(
                request=request,
                name="components/detail_body.html",
                context={"detail": detail},
            )
        else:
            card = await get_one_card(ticker, horizon_norm)
            response = templates.TemplateResponse(
                request=request,
                name="components/panel_card.html",
                context={"card": card, "horizon": horizon_norm},
            )

        if error_message:
            # Inject a data-toast attribute into the first element of
            # the response so toast.js picks it up after the swap.
            #
            # WARNING: Starlette's Response sets Content-Length when .body
            # is first assigned. Mutating .body afterwards leaves the header
            # stale; uvicorn then raises
            #   RuntimeError: Response content longer than Content-Length
            # because our injected `data-toast="..."` makes the body bigger.
            # Fix: recompute Content-Length after the mutation. The
            # transfer-encoding header (if any) stays as-is.
            response.body = response.body.replace(
                b'<li class="panel__card',
                f'<li data-toast="{error_message}" class="panel__card'.encode(),
                1,
            ).replace(
                b'<article class="card',
                f'<article data-toast="{error_message}" class="card'.encode(),
                1,
            )
            response.headers["content-length"] = str(len(response.body))
        return response

    # JSON fallback — same payload regardless of context.
    if context == "detail":
        detail = await get_stock_detail(ticker, horizon_norm)
        return JSONResponse(content={
            "ticker": detail.ticker,
            "horizon": horizon_norm,
            "has_prediction": detail.prediction is not None,
            "error": error_message,
        })
    card = await get_one_card(ticker, horizon_norm)
    return JSONResponse(content={
        "ticker": card.ticker,
        "horizon": horizon_norm,
        "has_prediction": card.prediction is not None,
        "error": error_message,
    })


@router.get("/predictions/detail", response_model=None)
async def predictions_detail_endpoint(
    request: Request,
    ticker: str,
    horizon: str = "weekly",
) -> HTMLResponse | JSONResponse:
    """Return the detail-body partial for a stock at a horizon.

    Swapped into #detail-body on the stock detail page when the user
    clicks a horizon tab. Renders the cached prediction in full if
    present, or a "Run prediction" CTA if not.
    """
    horizon_norm = horizon.lower().strip()
    if horizon_norm not in _VALID_HORIZONS:
        horizon_norm = "weekly"

    detail = await get_stock_detail(ticker, horizon_norm)

    if _is_htmx(request):
        return templates.TemplateResponse(
            request=request,
            name="components/detail_body.html",
            context={"detail": detail},
        )
    return JSONResponse(content={
        "ticker": detail.ticker,
        "horizon": detail.horizon,
        "has_prediction": detail.prediction is not None,
        "close": detail.close,
        "change_pct": detail.change_pct,
    })


@router.get("/chart", response_model=None)
async def chart_endpoint(
    request: Request,
    ticker: str,
    days: int = 90,
) -> JSONResponse:
    """Return historical closes for chart rendering.

    JSON-only — the chart is rendered client-side by Chart.js. No HTMX
    swap variant needed.
    """
    days = max(7, min(365, days))  # clamp to a sane range
    series = await get_chart_series(ticker, window_days=days)
    return JSONResponse(content={
        "ticker": series.ticker,
        "dates": series.dates,
        "closes": series.closes,
        "is_empty": series.is_empty,
    })


# ───────────────────────────────────────────────────────────────────────
# Live analysis tabs (stock detail page)
# ───────────────────────────────────────────────────────────────────────


async def _render_analysis_tab(
    request: Request,
    ticker: str,
    template_name: str,
) -> HTMLResponse | JSONResponse:
    """Shared helper for the indicators + patterns tabs.

    Both tabs need the same LiveAnalysis bundle; only the template
    differs. Keeping this DRY also means error-handling is identical
    across both tabs.
    """
    try:
        analysis = await compute_live_analysis(ticker)
        error: str | None = None
    except AnalysisServiceError as exc:
        analysis = None
        error = exc.message

    if _is_htmx(request):
        return templates.TemplateResponse(
            request=request,
            name=template_name,
            context={"analysis": analysis, "error": error},
        )
    if analysis is None:
        return JSONResponse(status_code=503, content={"error": error})
    # The JSON API stays schema-stable: strip the presentation-only `chart`
    # geometry (CandleChart dataclass) that the HTML template consumes.
    def _no_chart(items: list[dict]) -> list[dict]:
        return [{k: v for k, v in it.items() if k != "chart"} for it in items]
    return JSONResponse(content={
        "ticker": analysis.ticker,
        "as_of": analysis.as_of.isoformat(),
        "bars_used": analysis.bars_used,
        "trend": analysis.trend,
        "momentum": analysis.momentum,
        "volatility": analysis.volatility,
        "levels": analysis.levels,
        "candlesticks": _no_chart(analysis.candlesticks),
        "chart_patterns": _no_chart(analysis.chart_patterns),
    })


@router.get("/analysis/indicators", response_model=None)
async def analysis_indicators_endpoint(
    request: Request,
    ticker: str,
) -> HTMLResponse | JSONResponse:
    """Live trend / momentum / volatility / levels for `ticker`."""
    return await _render_analysis_tab(
        request, ticker, "components/detail_indicators.html",
    )


@router.get("/analysis/patterns", response_model=None)
async def analysis_patterns_endpoint(
    request: Request,
    ticker: str,
) -> HTMLResponse | JSONResponse:
    """Live candlestick + chart pattern hits for `ticker`."""
    return await _render_analysis_tab(
        request, ticker, "components/detail_patterns.html",
    )


@router.get("/analysis/news", response_model=None)
async def analysis_news_endpoint(
    request: Request,
    ticker: str,
    days: int = 7,
) -> HTMLResponse | JSONResponse:
    """Last `days` of GDELT headlines for `ticker`. Soft-fails on GDELT errors."""
    news = await fetch_recent_headlines(ticker, days=days)

    if _is_htmx(request):
        return templates.TemplateResponse(
            request=request,
            name="components/detail_news.html",
            context={"news": news},
        )
    return JSONResponse(content={
        "ticker": news.ticker,
        "query": news.query,
        "days": news.days,
        "error": news.error,
        "headlines": [
            {
                "title": h.title,
                "url": h.url,
                "source": h.source,
                "published_at": h.published_at.isoformat(),
                "age_label": h.age_label,
            }
            for h in news.headlines
        ],
    })


# ───────────────────────────────────────────────────────────────────────
# Per-ticker grading sidebar (stock detail page)
# ───────────────────────────────────────────────────────────────────────


@router.get("/grading/ticker", response_model=None)
async def grading_ticker_endpoint(
    request: Request,
    ticker: str,
    horizon: str | None = None,
    limit: int = 25,
) -> HTMLResponse | JSONResponse:
    """Grade the recent predictions for `ticker`. Optional horizon filter.

    HTMX returns the right-rail partial; plain GET returns JSON for
    programmatic use (e.g. CLI quickcheck of model calibration).
    """
    # Clamp limit to a sane window — the right rail isn't infinite scroll.
    limit = max(1, min(limit, 100))
    if horizon:
        horizon = horizon.lower().strip()
        if horizon not in _VALID_HORIZONS:
            horizon = None  # treat unknown filter as "all horizons"

    grading = await grade_ticker(ticker, horizon=horizon, limit=limit)

    if _is_htmx(request):
        # Build the R-multiple sparkline from resolved predictions only.
        # `graded` is newest-first; we reverse to chronological so the line
        # reads left=old → right=new like every other trend chart in the app.
        from price_predictor.web.utils.sparkline import build_sparkline
        resolved_r = [
            g.r_multiple for g in reversed(grading.graded)
            if g.r_multiple is not None
        ]
        sparkline = build_sparkline(resolved_r)

        return templates.TemplateResponse(
            request=request,
            name="components/detail_grading.html",
            context={
                "grading": grading,
                "selected_horizon": horizon or "all",
                "sparkline": sparkline,
                "resolved_count": len(resolved_r),
            },
        )
    return JSONResponse(content={
        "ticker": grading.ticker,
        "computed_at": grading.computed_at.isoformat(),
        "scorecard": {
            "total":    grading.scorecard.total,
            "hits":     grading.scorecard.hits,
            "stops":    grading.scorecard.stops,
            "expired":  grading.scorecard.expired,
            "pending":  grading.scorecard.pending,
            "skipped":  grading.scorecard.skipped,
            "hit_rate": grading.scorecard.hit_rate,
            "avg_r":    grading.scorecard.avg_r,
        },
        "graded": [
            {
                "id": g.row.id,
                "horizon": g.row.horizon,
                "created_at": g.row.created_at.isoformat(),
                "direction": g.row.direction,
                "entry_low": g.row.entry_low,
                "entry_high": g.row.entry_high,
                "target": g.row.target_value,
                "stop": g.row.stop_value,
                "outcome": g.outcome,
                "r_multiple": g.r_multiple,
                "resolved_at": g.resolved_at.isoformat() if g.resolved_at else None,
                "bars_used": g.bars_used,
                "note": g.note,
            }
            for g in grading.graded
        ],
    })
