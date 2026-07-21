"""News-impact analyzer — "gather in code, reason once".

ARCHITECTURE
============
Two clean halves:

1. GATHER (pure code, no LLM) — `gather.py`
   Fetches company news, sector news, filings, estimates and prices in
   parallel, honoring backtest look-ahead. Returns a `NewsImpactInputs`.

2. REASON (one LLM call) — this module
   `build_news_impact_prompt()` renders the gathered inputs into a single
   prompt; a tool-less `LlmAgent` with `output_schema=ImpactAssessment`
   turns it into a validated assessment. No tool loop, no re-sent context.

WHY THE CHANGE (from the old 4-tool agent)
==========================================
Fetching data needs no intelligence — it's deterministic code. The old
design paid LLM tokens for the model to *decide* which pure-code fetcher
to call, and re-sent the growing tool transcript every turn (2-4x token
cost). Only the final judgment (bullish? how much? why?) needs the model,
so that's the only thing we ask it to do now.

The output contract (`ImpactAssessment`) is UNCHANGED — everything
downstream in the predictor keeps working as-is.
"""
from typing import Literal

from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field

from price_predictor.agents.news_impact.gather import NewsImpactInputs
from price_predictor.llm.factory import make_resilient_model


# ─────────────────────────────────────────────────────────────
# Output schema (structured response) — UNCHANGED
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

    Returned by the news_impact synthesizer as its final output.
    Validated automatically by ADK's output_schema mechanism -- caller
    receives a parsed object, not raw text.
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
        # Floor of 20 chars: rejects empty/garbage but allows honest short
        # answers like "All sources empty; no evidence to assess."
        min_length=20,
        max_length=2000,
        description="Synthesis citing the gathered evidence (2-3 paragraphs ideal)",
    )
    catalysts: list[Catalyst] = Field(
        default_factory=list,
        max_length=10,
        description="Specific events/factors driving the assessment",
    )


# ─────────────────────────────────────────────────────────────
# Deterministic short-circuit (no LLM)
# ─────────────────────────────────────────────────────────────
def neutral_impact_assessment(ticker: str) -> ImpactAssessment:
    """Build a neutral assessment for the 'no evidence' case — no LLM.

    When gather finds no news, filings, or covered estimates there is
    literally nothing to reason about, so we skip the model and return
    this deterministically. Shape matches the degraded path: neutral,
    confidence 0, no catalysts — which makes the downstream synthesizer
    lean entirely on technicals (news contributes nothing either way).
    """
    return ImpactAssessment(
        ticker=ticker,
        sentiment="neutral",
        confidence=0.0,
        estimated_pct_move=0.0,
        reasoning=(
            "No company news, sector news, filings, or analyst estimates "
            "were found in the lookback window; no news-driven impact to "
            "assess. Prediction relies on technical evidence."
        ),
        catalysts=[],
    )


# ─────────────────────────────────────────────────────────────
# Prompt builder (renders gathered inputs → one synthesis prompt)
# ─────────────────────────────────────────────────────────────
def _fmt_news(rows: list[dict], label: str) -> str:
    if not rows:
        return f"{label}: (none found)\n"
    lines = [f"{label} ({len(rows)}):"]
    for r in rows:
        lines.append(
            f"  - [{r.get('published_at', '?')}] {r.get('title', '')} "
            f"({r.get('source', '')})"
        )
    return "\n".join(lines) + "\n"


def _fmt_filings(rows: list[dict]) -> str:
    if not rows:
        return "Corporate filings: (none)\n"
    lines = [f"Corporate filings ({len(rows)}):"]
    for r in rows:
        et = r.get("event_type") or r.get("kind", "")
        lines.append(
            f"  - [{r.get('announced_at', '?')}] {et}: {r.get('subject', '')}"
        )
    return "\n".join(lines) + "\n"


def _fmt_estimates(est: dict | None) -> str:
    if not est:
        return "Analyst estimates: (unavailable / skipped in backtest)\n"
    if not est.get("has_coverage"):
        return "Analyst estimates: no analyst coverage\n"
    return (
        "Analyst estimates:\n"
        f"  - next-quarter EPS consensus: {est.get('next_quarter_eps_consensus')}\n"
        f"  - current price: {est.get('current_price')}\n"
        f"  - mean price target: {est.get('price_target_mean')}\n"
    )


def _fmt_prices(prices: dict | None) -> str:
    if not prices or prices.get("status") != "success":
        return "Recent price action: (unavailable)\n"
    # The price tool returns a rich dict; surface the human-readable bits
    # without assuming an exact schema (stay robust to shape drift).
    keep = {k: v for k, v in prices.items()
            if k not in ("status",) and not isinstance(v, (list, dict))}
    body = ", ".join(f"{k}={v}" for k, v in keep.items())
    return f"Recent price action: {body}\n"


def build_news_impact_prompt(inputs: NewsImpactInputs) -> str:
    """Render gathered inputs into a single synthesis prompt.

    Deterministic and side-effect free — trivial to unit-test and cheap
    to reason about. The LLM sees ALL gathered evidence at once and
    returns one ImpactAssessment.
    """
    sector_line = (
        f"Sector: {inputs.sector}\n" if inputs.sector else "Sector: (unknown)\n"
    )
    errors_line = (
        "Data gaps: " + "; ".join(inputs.errors) + "\n" if inputs.errors else ""
    )
    return (
        f"Assess the likely directional price impact for the next ~5 trading "
        f"days for this Indian (NSE) stock.\n\n"
        f"Ticker: {inputs.ticker}\n"
        f"Company: {inputs.company_name}\n"
        f"{sector_line}"
        f"Window: {inputs.window_start} .. {inputs.window_end}\n"
        f"{errors_line}\n"
        f"{_fmt_news(inputs.company_news, 'Company news')}\n"
        f"{_fmt_news(inputs.sector_news, 'Sector news')}\n"
        f"{_fmt_filings(inputs.filings)}\n"
        f"{_fmt_estimates(inputs.estimates)}\n"
        f"{_fmt_prices(inputs.prices)}\n"
        f"Return an ImpactAssessment. Set ticker={inputs.ticker!r}."
    )


# ─────────────────────────────────────────────────────────────
# Synthesizer agent (tool-less, structured output)
# ─────────────────────────────────────────────────────────────
_SYNTHESIS_INSTRUCTION = """\
You are a financial-impact analyst for Indian (NSE-listed) stocks. You are
GIVEN a bundle of already-gathered evidence: company news, sector news,
corporate filings, analyst estimates, and recent price action. Your ONLY
job is to synthesize it into a structured ImpactAssessment for the next
~5 trading days. You have NO tools — reason purely over the evidence given.

YOU MUST RETURN A STRUCTURED ImpactAssessment OBJECT. Do not output free text.

HOW TO JUDGE
============
- sentiment: bullish/bearish/neutral based on the WEIGHT of evidence.
- Weigh the company AGAINST its sector: a strong company in a sinking
  sector is still at risk, and vice-versa. Use the sector news for this.
- confidence: 0.0-1.0. Lower it when evidence is thin, stale, or conflicting.
- estimated_pct_move: signed % over ~5 trading days. Be conservative:
  typical large-cap moves are -5 to +5%. Reserve >10% for major catalysts.
- reasoning: 2-3 paragraphs CITING specific headlines/filings/numbers from
  the evidence. Reference the sector view explicitly when it matters.
- catalysts: list each concrete driver with its source and impact direction.

CRITICAL RULES
==============
- NEVER fabricate news, filings, or numbers. Only cite what you were given.
- If a section says "(none)" / "(unavailable)", treat it as no signal and
  say so — do not invent it. Note any 'Data gaps' in your reasoning.
- If you have effectively NO evidence, set sentiment=neutral, confidence<=0.2,
  and explain that you couldn't gather enough to assess.
- When torn between two sentiments, prefer 'neutral' with lower confidence.
"""


def make_news_impact_agent() -> LlmAgent:
    """Build the tool-less news-impact synthesizer agent.

    Same name/output contract as before, but with NO tools — all data is
    gathered up front (see gather.py) and handed to it via the prompt.
    """
    return LlmAgent(
        name="news_impact",
        description=(
            "Synthesizes pre-gathered news, sector news, filings, estimates "
            "and price action into a structured impact assessment for an "
            "Indian stock."
        ),
        model=make_resilient_model(profile="agentic"),
        instruction=_SYNTHESIS_INSTRUCTION,
        output_schema=ImpactAssessment,
    )


# ─────────────────────────────────────────────────────────────
# ADK CLI entry point (adk run / adk web / adk api_server)
# ─────────────────────────────────────────────────────────────
root_agent = make_news_impact_agent()
