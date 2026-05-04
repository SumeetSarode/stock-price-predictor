"""News-impact analyzer ADK agent — implementation module.

ARCHITECTURE
============
LLM-driven analyzer with 4 tools + structured output:

- 4 tools, all sharing the (ticker, days_back) signature for LLM consistency:
    fetch_recent_news_tool       (GDELT news + body extraction)
    fetch_recent_filings_tool    (NSE corporate filings, 3 endpoints)
    fetch_estimates_tool         (yfinance analyst estimates -- snapshot)
    fetch_recent_prices_tool     (yfinance OHLCV -- thin wrapper of the price_agent tool)

- Structured Pydantic output via `output_schema=ImpactAssessment`. The LLM
  MUST return valid JSON matching the schema. ADK + LiteLLM enforce this
  via the underlying provider's structured-output mode.

- Default model: profile="agentic" → fallback chain via make_resilient_model().

WHY THIS DESIGN
===============
- (ticker, days_back) makes the LLM's life easy: no date math, no "what's
  today" guessing. Each tool computes its own date window from `now()` in IST.
- Each tool returns a {"status": ..., ...} dict (matches price_agent
  convention). On success: structured payload. On error: {"status":"error",
  "error_message":...} so LLM can recover or report.
- Tools are async where the underlying fetcher is async (news, filings,
  estimates). The price one stays sync (yfinance is sync). ADK handles both.

LEARNING POINTS (this is iteration 3.2, the first real ADK agent)
==================================================================
1. Tool docstrings are the LLM's API spec. Be precise about ticker formats:
   each data source uses a DIFFERENT ticker convention (NSE bare, yfinance
   .NS suffix, GDELT free-text query). The instruction prompt names the
   exact format per tool.
2. output_schema constrains the LLM to a Pydantic model -- structured-output
   mode kicks in. Caller gets validated objects, not free text.
3. LLM-driven (not pipeline) -- the model decides which tools to call based
   on the question. For simple "is XYZ bullish?", it might only call news.
   For "is the bullish thesis priced in?", it might call estimates + prices.
"""
from datetime import datetime, timedelta, timezone
from typing import Literal

from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field

from price_predictor.agents.price_agent.agent import fetch_prices_tool
from price_predictor.data.estimates import (
    EstimatesFetchError,
    fetch_estimates,
)
from price_predictor.data.filings import FilingsFetchError, fetch_filings
from price_predictor.data.news import NewsFetchError, fetch_news
from price_predictor.llm.factory import make_resilient_model

# India Standard Time -- all date math anchored here so 'today' matches NSE
IST = timezone(timedelta(hours=5, minutes=30))

# Sane bounds for the LLM-controlled days_back parameter
_MIN_DAYS = 1
_MAX_DAYS = 90


# ─────────────────────────────────────────────────────────────
# Output schema (structured response)
# ─────────────────────────────────────────────────────────────
class Catalyst(BaseModel):
    """One specific event or factor driving the impact assessment."""

    description: str = Field(
        ...,
        min_length=10,
        max_length=300,
        description="Concrete event/factor (e.g., 'Q4 earnings beat by 12%')",
    )
    source: Literal["news", "filing", "estimate", "price_action"] = Field(
        ...,
        description="Which data source surfaced this catalyst",
    )
    impact: Literal["positive", "negative", "neutral", "mixed"] = Field(
        ...,
        description=(
            "Direction of expected price impact: "
            "'positive' (clearly bullish), 'negative' (clearly bearish), "
            "'neutral' (no expected price impact), "
            "'mixed' (multiple competing effects in different directions)"
        ),
    )


class ImpactAssessment(BaseModel):
    """Structured news/event impact assessment for a single ticker.

    Returned by the news_impact agent as its final output. Validated
    automatically by ADK's output_schema mechanism -- caller receives
    a parsed object, not raw text.
    """

    ticker: str = Field(..., description="Ticker analyzed (any format the user gave)")
    sentiment: Literal["bullish", "bearish", "neutral"] = Field(
        ...,
        description="Overall directional view based on the evidence gathered",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model's confidence in the assessment (0=uncertain, 1=high)",
    )
    estimated_pct_move: float = Field(
        ...,
        ge=-30.0,
        le=30.0,
        description="Expected % price move over next ~5 trading days, signed",
    )
    reasoning: str = Field(
        ...,
        min_length=100,
        max_length=2000,
        description="2-3 paragraph synthesis citing specific evidence from tools",
    )
    catalysts: list[Catalyst] = Field(
        default_factory=list,
        max_length=10,
        description="Specific events/factors driving the assessment",
    )


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _validate_days_back(days_back: int) -> int | None:
    """Return clamped int if valid, else None (signals error to caller).

    Bounded so an LLM hallucinating `days_back=10000` doesn't DOS our APIs.
    """
    if not isinstance(days_back, int) or isinstance(days_back, bool):
        return None
    if days_back < _MIN_DAYS or days_back > _MAX_DAYS:
        return None
    return days_back


def _date_window(days_back: int) -> tuple[str, str]:
    """Compute (start_iso, end_iso) anchored at today in IST."""
    end = datetime.now(IST).date()
    start = end - timedelta(days=days_back)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _err(msg: str) -> dict:
    return {"status": "error", "error_message": msg}


# ─────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────
async def fetch_recent_news_tool(query: str, days_back: int = 7) -> dict:
    """Fetch recent news articles about a company via GDELT.

    Args:
        query: Free-text company name or search query (e.g., 'Reliance
               Industries', 'TCS Q4 results'). NOT a ticker symbol --
               GDELT searches news text, so use the company's common name.
        days_back: How many days of news to look back. 1-90. Default 7
                   for 'recent news' questions; use 30 for 'past month'.

    Returns:
        On success:
            {
                "status": "success",
                "query": str, "start": str, "end": str,
                "article_count": int,
                "articles": [
                    {"title": str, "url": str, "published_at": str,
                     "domain": str, "snippet": str},
                    ...  # capped at 25 to keep LLM context tight
                ],
            }
        On error:
            {"status": "error", "error_message": str}
    """
    if not isinstance(query, str) or not query.strip():
        return _err("query must be a non-empty string (use the company name).")
    valid_days = _validate_days_back(days_back)
    if valid_days is None:
        return _err(
            f"days_back must be an int between {_MIN_DAYS} and {_MAX_DAYS}, "
            f"got {days_back!r}."
        )

    start, end = _date_window(valid_days)
    try:
        df = await fetch_news(query.strip(), start, end)
    except (ValueError, NewsFetchError) as e:
        return _err(str(e))

    # Cap rows to keep the LLM context manageable
    rows = df.head(25)
    articles = [
        {
            "title": str(r["title"])[:250],
            "url": str(r["url"]),
            "published_at": str(r["published_at"]),
            "domain": str(r.get("domain", "")),
            "snippet": str(r.get("snippet", ""))[:300],
        }
        for _, r in rows.iterrows()
    ]
    return {
        "status": "success",
        "query": query,
        "start": start,
        "end": end,
        "article_count": len(df),
        "articles": articles,
    }


async def fetch_recent_filings_tool(nse_symbol: str, days_back: int = 30) -> dict:
    """Fetch recent NSE corporate filings (announcements, board meetings, corporate actions).

    Args:
        nse_symbol: NSE bare ticker, NO suffix (e.g., 'RELIANCE', 'TCS', 'INFY').
                    The .NS suffix used by yfinance is WRONG here.
        days_back: How many days back to fetch. 1-90. Default 30 for
                   'recent filings' questions.

    Returns:
        On success:
            {
                "status": "success",
                "nse_symbol": str, "start": str, "end": str,
                "filing_count": int,
                "by_kind": {"announcement": int, "board_meeting": int,
                            "corporate_action": int},
                "filings": [
                    {"kind": str, "announced_at": str, "event_at": str|None,
                     "event_type": str|None, "subject": str},
                    ...  # capped at 20 most recent
                ],
            }
        On error:
            {"status": "error", "error_message": str}
    """
    if not isinstance(nse_symbol, str) or not nse_symbol.strip():
        return _err("nse_symbol must be a non-empty NSE bare ticker (no .NS suffix).")
    if "." in nse_symbol:
        return _err(
            f"nse_symbol must be the NSE bare ticker (no suffix), got "
            f"{nse_symbol!r}. Use 'RELIANCE' not 'RELIANCE.NS'."
        )
    valid_days = _validate_days_back(days_back)
    if valid_days is None:
        return _err(
            f"days_back must be an int between {_MIN_DAYS} and {_MAX_DAYS}, "
            f"got {days_back!r}."
        )

    start, end = _date_window(valid_days)
    try:
        df = await fetch_filings(nse_symbol.strip().upper(), start, end)
    except (ValueError, FilingsFetchError) as e:
        return _err(str(e))

    # Roll up per-kind counts (helps LLM see distribution at a glance)
    by_kind: dict[str, int] = {}
    if not df.empty:
        by_kind = df["kind"].value_counts().to_dict()

    rows = df.head(20)
    filings = [
        {
            "kind": str(r["kind"]),
            "announced_at": str(r["announced_at"]),
            "event_at": str(r["event_at"]) if r["event_at"] is not None else None,
            "event_type": r["event_type"],
            "subject": str(r["subject"])[:300],
        }
        for _, r in rows.iterrows()
    ]
    return {
        "status": "success",
        "nse_symbol": nse_symbol.upper(),
        "start": start,
        "end": end,
        "filing_count": len(df),
        "by_kind": by_kind,
        "filings": filings,
    }


async def fetch_estimates_tool(yfinance_ticker: str) -> dict:
    """Fetch analyst earnings/revenue estimates and price targets (yfinance snapshot).

    Args:
        yfinance_ticker: yfinance ticker symbol. NSE stocks REQUIRE the
                         '.NS' suffix (e.g., 'RELIANCE.NS', 'TCS.NS').
                         For US stocks, use the bare ticker.

    Returns:
        On success:
            {
                "status": "success",
                "yfinance_ticker": str,
                "has_coverage": bool,    # False = analyst data not available
                "summary": {
                    "next_quarter_eps_consensus": float|None,
                    "next_quarter_eps_num_analysts": int|None,
                    "next_quarter_revenue_consensus": float|None,
                    "current_price": float|None,
                    "price_target_mean": float|None,
                    "price_target_high": float|None,
                    "price_target_low": float|None,
                    "recommendations_current": {
                        "strong_buy": int, "buy": int, "hold": int,
                        "sell": int, "strong_sell": int, "total": int,
                    } | None,
                },
            }
        On error:
            {"status": "error", "error_message": str}

        If has_coverage=False, summary fields will be all None -- yfinance
        doesn't have analyst data for this ticker. This is COMMON for Indian
        small-caps; not an error, just a data gap.
    """
    if not isinstance(yfinance_ticker, str) or not yfinance_ticker.strip():
        return _err("yfinance_ticker must be a non-empty string.")

    try:
        est = await fetch_estimates(yfinance_ticker.strip())
    except (ValueError, EstimatesFetchError) as e:
        return _err(str(e))

    # Pick the next-quarter EPS / revenue consensus (period '0q' or '+1q')
    next_eps = next(
        (e for e in est.earnings_estimates if e.period in ("0q", "+1q")),
        None,
    )
    next_rev = next(
        (r for r in est.revenue_estimates if r.period in ("0q", "+1q")),
        None,
    )
    # Most recent recommendations snapshot (period '0m')
    current_rec = next(
        (r for r in est.recommendations if r.period == "0m"),
        None,
    )

    summary: dict = {
        "next_quarter_eps_consensus": next_eps.avg if next_eps else None,
        "next_quarter_eps_num_analysts": next_eps.num_analysts if next_eps else None,
        "next_quarter_revenue_consensus": next_rev.avg if next_rev else None,
        "current_price": est.price_targets.current if est.price_targets else None,
        "price_target_mean": est.price_targets.mean if est.price_targets else None,
        "price_target_high": est.price_targets.high if est.price_targets else None,
        "price_target_low": est.price_targets.low if est.price_targets else None,
        "recommendations_current": (
            {
                "strong_buy": current_rec.strong_buy,
                "buy": current_rec.buy,
                "hold": current_rec.hold,
                "sell": current_rec.sell,
                "strong_sell": current_rec.strong_sell,
                "total": current_rec.total,
            }
            if current_rec else None
        ),
    }
    return {
        "status": "success",
        "yfinance_ticker": yfinance_ticker,
        "has_coverage": est.has_coverage,
        "summary": summary,
    }


def fetch_recent_prices_tool(yfinance_ticker: str, days_back: int = 30) -> dict:
    """Fetch recent OHLCV price summary for a ticker.

    Thin relative-date wrapper around the price_agent's tool. Use this
    instead of computing absolute dates yourself.

    Args:
        yfinance_ticker: yfinance ticker symbol. NSE REQUIRES '.NS' suffix
                         (e.g., 'RELIANCE.NS', 'TCS.NS').
        days_back: How many days of history. 1-90. Default 30 for typical
                   'recent price action' questions.

    Returns:
        Same shape as fetch_prices_tool (see its docstring), with the
        date window pre-computed. Use include_bars=False; daily detail
        is rarely needed for an impact assessment.
    """
    valid_days = _validate_days_back(days_back)
    if valid_days is None:
        return _err(
            f"days_back must be an int between {_MIN_DAYS} and {_MAX_DAYS}, "
            f"got {days_back!r}."
        )
    start, end = _date_window(valid_days)
    return fetch_prices_tool(yfinance_ticker, start, end, include_bars=False)


# ─────────────────────────────────────────────────────────────
# Agent factory
# ─────────────────────────────────────────────────────────────
_SYSTEM_INSTRUCTION = """\
You are a financial-impact analyst for Indian (NSE-listed) stocks. Given a
ticker or company name, you assess the LIKELY DIRECTIONAL IMPACT on the stock
price over the next ~5 trading days based on news, filings, analyst estimates,
and recent price action.

YOU MUST RETURN A STRUCTURED ImpactAssessment OBJECT. Do not output free text.

WORKFLOW
========
1. Identify the ticker. If the user gives a company name (e.g., 'Reliance'),
   resolve it to the relevant identifiers BEFORE calling tools:
   - GDELT news query: full company name (e.g., 'Reliance Industries')
   - NSE filings symbol: bare ticker (e.g., 'RELIANCE')
   - yfinance ticker: bare + '.NS' suffix (e.g., 'RELIANCE.NS')

2. Decide which tools to call based on the question. RULES OF THUMB:
   - 'Is X bullish?' / 'How is X doing?' -> news (7d) + prices (30d) minimum.
   - 'Are upcoming earnings priced in?' -> + filings (board meetings) + estimates.
   - 'What just happened with X?' -> news (3-7d) + filings (7d).
   - For ANY assessment, fetch news at minimum -- it surfaces the freshest catalysts.
   - DON'T call all 4 tools unless the question is broad ('full assessment').

3. Synthesize. Weigh evidence:
   - Recent news with multiple sources covering same event = stronger signal
   - Filings of board meetings imminent = upcoming catalyst (uncertainty)
   - Analyst price target above current = bullish bias (but check confidence)
   - Recent price action up + bullish news = momentum; up + bearish news = caution

4. Build the ImpactAssessment:
   - sentiment: bullish/bearish/neutral based on the WEIGHT of evidence
   - confidence: 0.0-1.0. Lower if data is thin or contradictory
   - estimated_pct_move: signed % over next ~5 trading days. Be conservative:
     typical large-cap moves are -5 to +5%. Reserve >10% for major catalysts.
   - reasoning: 2-3 paragraphs CITING specific articles/filings/numbers.
     Don't make up data -- only cite what tools actually returned.
   - catalysts: list each concrete driver with its source and impact direction.

CRITICAL RULES
==============
- NEVER fabricate news, filings, or numbers. Only cite what tools returned.
- If a tool returns an error, note it in reasoning and lower confidence.
- If you have NO data (all tools returned empty), set sentiment=neutral,
  confidence<=0.2, and explain that you couldn't gather evidence.
- When unsure between two sentiments, prefer 'neutral' with lower confidence.
"""


def make_news_impact_agent() -> LlmAgent:
    """Build the news-impact analyzer agent."""
    return LlmAgent(
        name="news_impact",
        description=(
            "Analyzes news, filings, analyst estimates, and price action "
            "to produce a structured impact assessment for an Indian stock."
        ),
        model=make_resilient_model(profile="agentic"),
        instruction=_SYSTEM_INSTRUCTION,
        tools=[
            fetch_recent_news_tool,
            fetch_recent_filings_tool,
            fetch_estimates_tool,
            fetch_recent_prices_tool,
        ],
        output_schema=ImpactAssessment,
    )


# ─────────────────────────────────────────────────────────────
# ADK CLI entry point
# Module-level instance required by `adk run` / `adk web` / `adk api_server`.
# ─────────────────────────────────────────────────────────────
root_agent = make_news_impact_agent()
