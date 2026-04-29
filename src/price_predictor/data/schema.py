"""Pydantic schemas for data-layer payloads.

Use these at SYSTEM BOUNDARIES (JSON serialization, API responses, fixtures).
DO NOT use inside the engine — fetch_ohlcv() returns DataFrames for performance,
and pandas-ta + indicator math want DataFrames, not Pydantic instances.

Conversion at boundaries:
    df.reset_index().to_dict("records")  →  [OHLCVBar(**r) for r in records]
    [b.model_dump() for b in bars]       →  pd.DataFrame(...)

DESIGN ASSUMPTION — REGULAR-SESSION ONLY (prices):
    `close` represents the official regular-session close ONLY. Post-market /
    after-hours / pre-market trades are NOT represented in this model.

    - For NSE (our primary market): structurally impossible — the official
      close is locked at 15:30 IST as the VWAP of the 15:00-15:30 window.
      The 15:40-16:00 post-close session executes orders at this fixed price
      but creates no new price discovery.
    - For US markets (if ever fetched): we do NOT pass `prepost=True` to
      yfinance, so daily `close` = 4:00 PM ET regular-session close.

    If we ever need to represent pre/post sessions, add a
    `session: Literal["regular", "premarket", "afterhours"]` field.
"""
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


class OHLCVBar(BaseModel):
    """One row of OHLCV price data for a single ticker on a single timestamp.

    All price fields must be strictly positive (> 0); volume must be non-negative
    (>= 0, since 0-volume bars are valid on trading-halt days). High/low
    consistency is enforced by the model_validator below.
    """

    timestamp: datetime = Field(
        ...,
        description="Bar timestamp, tz-aware (Asia/Kolkata in this project)",
    )
    open: float = Field(..., gt=0, description="Opening price (unadjusted)")
    high: float = Field(..., gt=0, description="Highest price during the bar")
    low: float = Field(..., gt=0, description="Lowest price during the bar")
    close: float = Field(
        ...,
        gt=0,
        description=(
            "Closing price (unadjusted; what brokers display). "
            "Regular-session close ONLY — see module docstring."
        ),
    )
    adj_close: float = Field(
        ...,
        gt=0,
        description="Closing price adjusted for splits/dividends",
    )
    volume: int = Field(..., ge=0, description="Shares traded during the bar")

    @model_validator(mode="after")
    def check_high_low_consistency(self) -> "OHLCVBar":
        """High must be the max, low must be the min — basic sanity."""
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) < low ({self.low})")
        if self.high < max(self.open, self.close):
            raise ValueError(
                f"high ({self.high}) < max(open={self.open}, close={self.close})"
            )
        if self.low > min(self.open, self.close):
            raise ValueError(
                f"low ({self.low}) > min(open={self.open}, close={self.close})"
            )
        return self


class NewsArticle(BaseModel):
    """One news article — metadata only (body is fetched separately).

    Body is intentionally NOT part of this model. GDELT's Doc API only returns
    metadata; body extraction is a separate concern with its own failure modes
    (paywalls, JS-rendered pages, bot blocks). Use `ArticleBody` for that.
    """

    title: str = Field(..., min_length=1, description="Article headline")
    url: HttpUrl = Field(..., description="Canonical article URL")
    published_at: datetime = Field(
        ...,
        description="Publication timestamp, tz-aware UTC (parsed from GDELT seendate)",
    )
    source: str = Field(..., min_length=1, description="Domain (e.g. 'reuters.com')")
    language: str = Field(
        default="eng",
        min_length=2,
        description="ISO-639-2/B language code (GDELT uses 3-letter codes like 'eng')",
    )


class ArticleBody(BaseModel):
    """Result of an article body extraction attempt.

    Status-discriminated to keep success-with-empty-body distinguishable
    from extraction failure (paywall / bot block / network error).

    On success: `body` contains the extracted text.
    On error:   `body` is empty, `error_message` explains why.
    """

    status: Literal["success", "error"]
    body: str = Field(default="", description="Extracted article text (success only)")
    error_message: str | None = Field(
        default=None,
        description="Failure reason (error only)",
    )

    @model_validator(mode="after")
    def check_status_consistency(self) -> "ArticleBody":
        """Enforce status ↔ fields contract."""
        if self.status == "success" and self.error_message is not None:
            raise ValueError("success result must not have error_message")
        if self.status == "error" and not self.error_message:
            raise ValueError("error result must have a non-empty error_message")
        return self


# ─────────────────────────────────────────────────────────────
# Analyst estimates schemas (iteration 3.1.2.5)
# ─────────────────────────────────────────────────────────────
class QuarterlyEstimate(BaseModel):
    """Forward-looking analyst consensus for one upcoming period (EPS or revenue).

    yfinance returns 4 rows per stock with mixed period granularity:
    - '0q'  = current quarter (often THIN — Indian large-caps frequently have only 1-2 analysts)
    - '+1q' = next quarter (also typically thin)
    - '0y'  = current fiscal year (annual consensus — usually 20-32 analysts, ROBUST)
    - '+1y' = next fiscal year (annual consensus — also robust)

    Each row = aggregated stats across N analysts. Downstream analyzers
    SHOULD weight signals by num_analysts (≥5 = real consensus, 1-2 = weak).
    """

    period: str = Field(
        ...,
        description="yfinance period label: '0q'/'+1q' (quarters) or '0y'/'+1y' (fiscal years)",
    )
    num_analysts: int | None = Field(
        default=None, ge=0,
        description="How many analysts contributed to the consensus",
    )
    avg: float | None = Field(default=None, description="Mean estimate")
    low: float | None = Field(default=None, description="Lowest analyst estimate")
    high: float | None = Field(default=None, description="Highest analyst estimate")
    year_ago: float | None = Field(
        default=None,
        description="Same metric one year ago (for YoY comparison)",
    )
    growth: float | None = Field(
        default=None,
        description="Implied YoY growth (decimal, e.g., 0.12 = +12%)",
    )


class RecommendationDistribution(BaseModel):
    """Snapshot of analyst buy/hold/sell counts for a given period.

    yfinance returns 4 historical snapshots (current month, -1m, -2m, -3m)
    so callers can spot recommendation drift.
    """

    period: str = Field(..., description="yfinance period label: '0m', '-1m', '-2m', '-3m'")
    strong_buy: int = Field(default=0, ge=0)
    buy: int = Field(default=0, ge=0)
    hold: int = Field(default=0, ge=0)
    sell: int = Field(default=0, ge=0)
    strong_sell: int = Field(default=0, ge=0)

    @property
    def total(self) -> int:
        return self.strong_buy + self.buy + self.hold + self.sell + self.strong_sell


class PriceTargets(BaseModel):
    """Consensus price target across covering analysts."""

    current: float | None = Field(default=None, gt=0, description="Current price (snapshot)")
    low: float | None = Field(default=None, gt=0)
    mean: float | None = Field(default=None, gt=0)
    median: float | None = Field(default=None, gt=0)
    high: float | None = Field(default=None, gt=0)


class Estimates(BaseModel):
    """Full analyst consensus snapshot for one ticker.

    Designed to absorb yfinance's coverage gaps gracefully:
    - All sub-fields are Optional/empty-list-able
    - A stock with NO analyst coverage → all fields empty/None, no error
    - Caller checks `has_coverage` to decide whether to use the data

    `fetched_at` lets us reason about staleness (estimates rarely change intraday
    but matter for week-old caches).
    """

    symbol: str = Field(..., min_length=1, description="Ticker (e.g. 'RELIANCE.NS')")
    fetched_at: datetime = Field(
        ...,
        description="When this snapshot was retrieved, tz-aware UTC",
    )
    earnings_estimates: list[QuarterlyEstimate] = Field(
        default_factory=list,
        description="Forward EPS consensus per upcoming quarter",
    )
    revenue_estimates: list[QuarterlyEstimate] = Field(
        default_factory=list,
        description="Forward revenue consensus per upcoming quarter",
    )
    recommendations: list[RecommendationDistribution] = Field(
        default_factory=list,
        description="Buy/hold/sell counts (current + 3 months back)",
    )
    price_targets: PriceTargets | None = Field(
        default=None,
        description="Consensus price target (None if no coverage)",
    )

    @property
    def has_coverage(self) -> bool:
        """True iff yfinance returned any analyst data for this ticker."""
        return bool(
            self.earnings_estimates
            or self.revenue_estimates
            or self.recommendations
            or self.price_targets
        )


# ─────────────────────────────────────────────────────────────
# Corporate filings schemas (iteration 3.1.3)
# ─────────────────────────────────────────────────────────────
FilingKind = Literal[
    "announcement",       # Regulatory disclosures (M&A, fundraising, lawsuits, etc.)
    "board_meeting",      # Scheduled board meetings (forward-looking earnings windows)
    "corporate_action",   # Splits, dividends, bonuses, rights (with ex-dates)
    "financial_result",   # Quarterly/annual results filings (for surprise calc)
]


class Filing(BaseModel):
    """One corporate filing event from NSE.

    Unified shape across 4 NSE endpoints (announcement / board_meeting /
    corporate_action / financial_result). Each endpoint maps its quirky
    JSON shape into this common model, with endpoint-specific fields
    captured in `metadata` to avoid bloating the top-level schema.

    KEY DESIGN: TWO dates, not one
    --------------------------------
    - `announced_at`: when NSE received the filing (always past, always present)
    - `event_at`: when the actual event happens (split date, ex-div, meeting)
                  Often null; can be past, present, or future.

    This separation enables BOTH backward-looking ("what was filed recently?")
    AND forward-looking ("what known events are coming?") queries against the
    same dataset. Critical for catching e.g. a stock split announced today but
    effective in 60 days — a backward-only window misses it.
    """

    symbol: str = Field(..., min_length=1, description="NSE bare symbol (no .NS suffix)")
    kind: FilingKind = Field(..., description="Which endpoint this came from")
    announced_at: datetime = Field(
        ...,
        description="When the filing was made with NSE (tz-aware IST)",
    )
    event_at: datetime | None = Field(
        default=None,
        description="When the actual event occurs (split/ex-div/meeting date), tz-aware IST",
    )
    event_type: str | None = Field(
        default=None,
        description="NSE's free-text categorization (e.g. 'Dividend', 'Quarterly Results')",
    )
    subject: str = Field(..., min_length=1, description="Short headline / subject line")
    description: str = Field(
        default="",
        description="Longer text if NSE provides it (often empty — full content in PDF)",
    )
    attachment_url: HttpUrl | None = Field(
        default=None,
        description="PDF attachment link (often the only place with full content)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Endpoint-specific extras (split ratio, dividend amount, broadcast time, etc.)",
    )
