"""TechnicalView assembly — calls the 4 cluster tools in parallel.

PURPOSE (Step 3.4.2, commit 1)
==============================
This module is the predictor's structured-input gatherer for the TECHNICAL
side of the picture. It calls the 4 existing cluster tools (get_trend,
get_momentum, get_volatility, get_levels) in PARALLEL and packs their
already-structured outputs into a `TechnicalView` Pydantic model that the
synthesizer agent will consume as JSON.

WHY BYPASS technical_agent (the LlmAgent)?
==========================================
`technical_agent` exists for INTERACTIVE use: a human asks "how's RSI on
RELIANCE?" and the LLM picks the right subset of tools to call.

The PREDICTOR always wants the full picture (all 4 clusters), every time.
There's no decision for an LLM to make. Routing through the LlmAgent would:
    - add 1 LLM round-trip per prediction (~3-5s, $0.001)
    - convert structured tool outputs to prose, which the synthesizer
      would then have to re-parse back into structure (lossy round-trip)
    - serialize cluster execution (LLM calls tools sequentially)
    - hide warnings if the LLM forgets to weave them in

By calling the 4 tools directly from Python we:
    - run them in parallel (asyncio.gather)
    - keep all data structured end-to-end (no prose parsing)
    - guarantee 100% cluster coverage (no LLM might-skip)
    - share OHLCV across tools via the existing process-wide cache
      (4 tool calls -> 1 actual yfinance fetch)

`technical_agent` stays in the codebase, untouched, for `adk run` /
`adk web` / future Q&A use cases.

FAILURE POLICY
==============
Technicals are CORE to a prediction. If ANY cluster fails, we raise
`TechnicalViewError`. The predictor (commit 4) will catch this and decide
what to surface to the caller. Rationale: a prediction missing one cluster
isn't a degraded prediction, it's an unreliable one — better to fail loudly
than ship something silently wrong.

(Compare with news_impact: failures THERE are degradable because predictions
without news context are still meaningful.)
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from price_predictor.agents.news_impact import ImpactAssessment
from price_predictor.agents.technical_agent.tools.get_levels import get_levels
from price_predictor.agents.technical_agent.tools.get_momentum import get_momentum
from price_predictor.agents.technical_agent.tools.get_trend import get_trend
from price_predictor.agents.technical_agent.tools.get_volatility import get_volatility
from price_predictor.data._shared_cache import get_cache
from price_predictor.kb.stocks import lookup as resolve_stock

# Same lookback the 4 cluster tools use. Centralized here so we can fetch
# the OHLCV bar-count for `bars_used` without round-tripping a tool.
LOOKBACK_DAYS = 400

# String literals reused at the schema level. Match the tool _types.py.
Sensitivity = Literal["standard", "sensitive", "smooth"]
Signal = Literal["bullish", "neutral", "bearish"]
Strength = Literal["weak", "moderate", "strong"]
ClusterName = Literal["trend", "momentum", "volatility", "levels"]


# ─────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────
class TechnicalViewError(RuntimeError):
    """Raised when one or more cluster tools fail during composition.

    Carries `cluster_errors`: a {cluster_name -> error_message} dict so
    callers can log / surface specifics without parsing the message string.
    """

    def __init__(self, message: str, cluster_errors: dict[str, str]):
        super().__init__(message)
        self.cluster_errors = cluster_errors


# ─────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────
class ClusterView(BaseModel):
    """Frozen view of one cluster tool's output.

    Mirrors the ToolSuccessResponse TypedDict from
    agents/technical_agent/tools/_types.py but adds frozen+hashable
    semantics for use inside the larger Prediction pipeline.

    `indicators` and `derived` are intentionally `dict[str, Any]` because
    each cluster returns a different schema there (trend has SMAs+ADX,
    momentum has RSI+MACD+candlesticks, etc.). Forcing them into rigid
    sub-models would either duplicate the cluster code or constrain it.
    Synthesizer reads them as JSON; the LLM handles heterogeneity fine.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: ClusterName = Field(
        ..., description="Which cluster this is (trend / momentum / volatility / levels).",
    )
    signal: Signal = Field(
        ..., description="The cluster's directional verdict.",
    )
    strength: Strength | None = Field(
        default=None,
        description=(
            "Strength qualifier when the cluster exposes one (trend has ADX-based "
            "strength, levels has pattern-confidence-based, etc.). None if N/A."
        ),
    )
    indicators: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Raw indicator values produced by the cluster. Schema is "
            "cluster-specific (trend: SMA/EMA/ADX, momentum: RSI/MACD, etc.)."
        ),
    )
    derived: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Derived booleans / percentages from the cluster (e.g. "
            "above_sma_50, breakout_state, atr_pct_of_close)."
        ),
    )
    rationale: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Bullet-list of human-readable findings the LLM can quote.",
    )
    warnings: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Cluster warnings (e.g. 'insufficient_history', "
            "'pattern_signal_conflict'). Synthesizer SHOULD reflect these."
        ),
    )


class TechnicalView(BaseModel):
    """Bundled output of all 4 technical clusters for one ticker.

    The OUTPUT contract of `compose_technical_view()` and an INPUT to the
    synthesizer agent. Together with `ImpactAssessment` (from news_impact)
    it forms the synthesizer's full information set.

    DESIGN NOTES
    ------------
    - `as_of` is a date (not datetime) because cluster tools report at
      bar granularity; intra-bar timing isn't meaningful.
    - `close_price` is the latest bar's close; reused by the predictor
      to populate `Prediction.analysis_basis.close_price_at_prediction`.
    - `bars_used` is the actual bar count (post-fetch len(df)), not
      LOOKBACK_DAYS — calendar days != trading days.
    - `sensitivity` records which preset was used so consumers (logs,
      backtest replay) can reason about indicator parameter choice.
    - All 4 clusters are REQUIRED. Partial views aren't a thing here;
      see TechnicalViewError docstring for the rationale.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(
        ..., min_length=1,
        description="Canonical yfinance symbol (e.g. 'RELIANCE.NS').",
    )
    as_of: date = Field(
        ..., description="Date of the latest bar used in the analysis.",
    )
    close_price: float = Field(
        ..., gt=0,
        description="Latest bar's close. Source of truth for prediction price anchoring.",
    )
    bars_used: int = Field(
        ..., ge=20,
        description=(
            "Actual count of OHLCV bars analyzed. "
            "Floor of 20 because most indicators are noise below that."
        ),
    )
    sensitivity: Sensitivity = Field(
        ...,
        description="Preset used by all 4 clusters (standard / sensitive / smooth).",
    )
    trend: ClusterView = Field(..., description="Trend cluster output.")
    momentum: ClusterView = Field(..., description="Momentum cluster output.")
    volatility: ClusterView = Field(..., description="Volatility cluster output.")
    levels: ClusterView = Field(..., description="Levels & patterns cluster output.")


# ─────────────────────────────────────────────────────────────
# Composition
# ─────────────────────────────────────────────────────────────
def _resolve_ticker(ticker: str) -> str:
    """Return canonical form (yfinance symbol) for a user-supplied ticker.

    Strategy:
      1. KB lookup (handles 'reliance' / 'HDFC' / 'L&T' fuzzy resolution).
      2. If unknown to KB, return upper-stripped form so US tickers like
         'AAPL' still flow through unchanged.

    Cluster tools each re-normalize defensively — duplication is fine
    because normalization is idempotent. Doing it here means the predictor
    layer owns identity decisions explicitly.
    """
    if not ticker or not ticker.strip():
        raise ValueError("ticker must be non-empty")
    stock = resolve_stock(ticker)
    if stock is not None:
        return stock.yfinance_symbol
    return ticker.strip().upper()


def _cluster_view_from_response(
    name: ClusterName, response: dict
) -> ClusterView:
    """Convert a cluster tool's success-response dict into a ClusterView.

    Tools return TypedDicts with `total=False` semantics, so some fields
    may be absent. We use .get() with sensible defaults to be tolerant.
    """
    return ClusterView(
        name=name,
        signal=response["signal"],
        strength=response.get("strength"),
        indicators=response.get("indicators", {}),
        derived=response.get("derived", {}),
        rationale=tuple(response.get("rationale", [])),
        warnings=tuple(response.get("warnings", [])),
    )


async def _fetch_close_and_bar_count(
    ticker: str,
) -> tuple[float, int, date]:
    """Pull latest close, bar count, and as_of date from the shared cache.

    Cache hit (the 4 cluster tools have already populated it with the same
    400-day window for the same ticker). Independent fetch keeps this
    function decoupled from cluster internals — if a cluster tool's cache-
    interaction changes, we don't break.
    """
    end = date.today()
    start = end - timedelta(days=LOOKBACK_DAYS)
    df = await get_cache().get(
        ticker=ticker, start=start, end=end, interval="1d"
    )
    if df.empty:
        raise TechnicalViewError(
            f"No price data available for {ticker}",
            cluster_errors={"_fetch": "empty dataframe"},
        )
    latest = df.iloc[-1]
    close_price = float(latest["close"])
    bars_used = len(df)
    # Index is a tz-aware DatetimeIndex; .date() gives a plain date.
    last_index = df.index[-1]
    if isinstance(last_index, datetime):
        as_of = last_index.date()
    else:  # pragma: no cover (defensive — should always be Timestamp)
        as_of = end
    return close_price, bars_used, as_of


async def compose_technical_view(
    ticker: str, sensitivity: Sensitivity = "standard"
) -> TechnicalView:
    """Run all 4 cluster tools in parallel and pack into a TechnicalView.

    Args:
        ticker: User-supplied ticker (any form: 'reliance', 'RELIANCE.NS',
            'AAPL'). Resolved to canonical yfinance symbol via the KB.
        sensitivity: Preset for all 4 clusters. Default 'standard'.

    Returns:
        Fully-populated, frozen TechnicalView.

    Raises:
        ValueError: empty/blank ticker.
        TechnicalViewError: one or more cluster tools failed (carries
            per-cluster error messages in `cluster_errors`).
    """
    canonical = _resolve_ticker(ticker)

    # Parallel cluster execution. asyncio.gather + return_exceptions=False
    # would cancel siblings on first failure; we want ALL errors collected
    # so the user sees the full picture, hence return_exceptions=True with
    # explicit failure aggregation below.
    trend_task = get_trend(canonical, sensitivity)
    momentum_task = get_momentum(canonical, sensitivity)
    volatility_task = get_volatility(canonical, sensitivity)
    levels_task = get_levels(canonical, sensitivity)

    results = await asyncio.gather(
        trend_task, momentum_task, volatility_task, levels_task,
        return_exceptions=True,
    )
    cluster_names: tuple[ClusterName, ...] = (
        "trend", "momentum", "volatility", "levels",
    )

    # Aggregate failures: tool exceptions (shouldn't happen — tools return
    # error dicts) AND status='error' responses both count.
    cluster_errors: dict[str, str] = {}
    successful: dict[str, dict] = {}
    for name, result in zip(cluster_names, results, strict=True):
        if isinstance(result, BaseException):
            cluster_errors[name] = f"{type(result).__name__}: {result}"
        elif result.get("status") == "error":
            cluster_errors[name] = result.get("error_message", "unknown error")
        else:
            successful[name] = result

    if cluster_errors:
        raise TechnicalViewError(
            f"Technical composition failed for {canonical}: "
            f"{len(cluster_errors)}/4 cluster(s) failed: "
            f"{list(cluster_errors.keys())}",
            cluster_errors=cluster_errors,
        )

    # All 4 succeeded — pull the price-anchoring metadata and pack.
    close_price, bars_used, as_of = await _fetch_close_and_bar_count(canonical)

    return TechnicalView(
        ticker=canonical,
        as_of=as_of,
        close_price=close_price,
        bars_used=bars_used,
        sensitivity=sensitivity,
        trend=_cluster_view_from_response("trend", successful["trend"]),
        momentum=_cluster_view_from_response("momentum", successful["momentum"]),
        volatility=_cluster_view_from_response("volatility", successful["volatility"]),
        levels=_cluster_view_from_response("levels", successful["levels"]),
    )


# ──────────────────────────────────────────────────────────────
# Synthesis input (the gather → synthesizer contract)
# ──────────────────────────────────────────────────────────────
class SynthesisInput(BaseModel):
    """Complete typed input for the synthesizer agent.

    PURPOSE
    -------
    Sole envelope flowing from the predictor's gather phase into the
    synthesizer agent. Everything the synthesizer needs to produce a
    Prediction lives here — in ONE typed object, not scattered across
    prompt-string concatenations.

    DESIGN INVARIANTS
    -----------------
    - Frozen + extra='forbid' (matches the project-wide schema discipline)
    - All sub-models are themselves frozen (TechnicalView, ImpactAssessment),
      so the parent stays hashable
    - tz-aware as_of REQUIRED (same rule as Prediction.as_of) — anchors
      the prediction's identity to a specific moment
    - Non-empty model_chain REQUIRED (same rule as Prediction.model_chain)
      — audit trail must record at least the news_impact model that ran
      during gather; synthesizer adds itself in commit 3
    - Both technical_view and impact_assessment are NON-OPTIONAL. If
      gather couldn't produce one, the predictor raises before ever
      constructing this object. Synthesizer never sees half-data.
      (News degradation, when added in commit 5, will produce a
      degenerate-but-valid ImpactAssessment, not a None.)

    WHY NOT JUST PASS A DICT?
    -------------------------
    A dict would: (a) skip validation, (b) drift silently as fields are
    added, (c) make plete useless, (d) lose the typed
    relationship between this contract and what the synthesizer is
    documented to consume. Pydantic gives us all four for ~30 lines of
    schema. Cheap.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(
        ..., min_length=1,
        description=(
            "Canonical yfinance ticker (post-KB resolution). "
            "Should equal technical_view.ticker."
        ),
    )
    horizon: Literal["intraday", "short", "medium", "long"] = Field(
        ...,
        description=(
            "Time window the prediction targets. Drives the synthesizer's "
            "reasoning depth (intraday=tactical; long=positional). Mirrors "
            "PredictionHorizon enum values."
        ),
    )
    as_of: datetime = Field(
        ...,
        description=(
            "Moment the prediction cycle was anchored at. MUST be tz-aware. "
            "Convention: Asia/Kolkata for India-market predictions. "
            "Inherited by Prediction.as_of so the audit timeline is consistent."
        ),
    )
    technical_view: TechnicalView = Field(
        ...,
        description=(
            "Output of compose_technical_view(). All 4 cluster signals + "
            "close_price + bars_used. Synthesizer reads as nested JSON."
        ),
    )
    impact_assessment: ImpactAssessment = Field(
        ...,
        description=(
            "Output of news_impact agent. Sentiment, confidence, estimated "
            "% move, catalysts, reasoning. Synthesizer reads as nested JSON."
        ),
    )
    model_chain: tuple[str, ...] = Field(
        ...,
        description=(
            "LLMs that participated in GATHER (so far: just the news_impact "
            "model). Synthesizer appends its own model name before constructing "
            "the final Prediction. Audit trail — must be non-empty."
        ),
    )

    def model_post_init(self, __context: object) -> None:
        """Cross-field invariants.

        Pydantic v2 prefers @model_validator(mode='after'), but for two
        simple checks model_post_init keeps the file flat. Same end result:
        invariants enforced before any caller sees the object.
        """
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be tz-aware (got naive datetime)")
        if len(self.model_chain) == 0:
            raise ValueError("model_chain must contain at least one model name")
