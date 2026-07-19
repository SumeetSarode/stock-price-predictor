"""Page routes — render full HTML pages via Jinja2 templates.

These handlers ONLY pass context dicts to templates. No HTML strings,
no business logic. Pure routing + context assembly.
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from price_predictor._version import __version__
from price_predictor.web.services.search_service import get_by_ticker
from price_predictor.web.templating import templates

router = APIRouter()


# Single source of truth for the version string we render in the footer:
# read from the installed package metadata (declared once in pyproject.toml).
APP_VERSION = __version__


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """Landing page — the predict form."""
    return templates.TemplateResponse(
        request=request,
        name="pages/home.html",
        context={"app_version": APP_VERSION},
    )


@router.get("/history", response_class=HTMLResponse)
async def history(
    request: Request,
    ticker: str | None = None,
    horizon: str | None = None,
    page: int = 1,
) -> HTMLResponse:
    """Prediction history — reverse-chronological list of every cached
    prediction across all tickers and horizons.

    Supports ?ticker= and ?horizon= filters and ?page= pagination
    (50 per page).
    """
    from price_predictor.web.services.grading_service import (
        Scorecard, build_scorecard, grade_rows,
    )
    from price_predictor.web.services.history_service import list_history

    page = max(1, page)
    per_page = 50
    rows, total = list_history(
        ticker=ticker,
        horizon=horizon,
        limit=per_page,
        offset=(page - 1) * per_page,
    )
    total_pages = max(1, (total + per_page - 1) // per_page)

    # Grade what's on this page so the table can show real outcomes.
    # Keyed by row.id so the template can do {{ grades[r.id] }} lookups
    # without us having to clone the frozen HistoryRow dataclasses.
    graded = await grade_rows(rows) if rows else []
    grades_by_id: dict[int, str] = {gp.row.id: gp.outcome for gp in graded}

    # Page-local scorecard — just the rows visible right now. Honest:
    # if you're on page 2, it scores page 2's slice, not the whole DB.
    # A global scorecard across ALL predictions would force grading every
    # row in the DB on every page load — too expensive without caching.
    page_scorecard: Scorecard | None = build_scorecard(graded) if graded else None

    return templates.TemplateResponse(
        request=request,
        name="pages/history.html",
        context={
            "app_version": APP_VERSION,
            "rows": rows,
            "grades_by_id": grades_by_id,
            "page_scorecard": page_scorecard,
            "total": total,
            "page": page,
            "total_pages": total_pages,
            "per_page": per_page,
            "filter_ticker": ticker or "",
            "filter_horizon": horizon or "",
        },
    )


@router.get("/stock/{ticker}", response_class=HTMLResponse)
async def stock_detail(
    request: Request,
    ticker: str,
    horizon: str = "weekly",
) -> HTMLResponse:
    """Stock detail page — reached by clicking a watchlist card or
    dashboard row.

    Renders the chosen horizon's cached prediction in full when present
    (rich card with rationale + signals + technical summary), or a
    "Run prediction" CTA when not. Tabs let the user switch horizons
    without reloading the page (HTMX swaps the body).
    """
    from price_predictor.web.services.detail_service import get_stock_detail
    from price_predictor.web.services.history_service import list_history

    detail = await get_stock_detail(ticker, horizon)
    # Last 10 predictions for this ticker (any horizon) — shown below
    # the active prediction card. Cheap query.
    ticker_history, _ = list_history(ticker=detail.ticker, limit=10)

    return templates.TemplateResponse(
        request=request,
        name="pages/stock_detail.html",
        context={
            "app_version": APP_VERSION,
            "detail": detail,
            "ticker_history": ticker_history,
        },
    )
