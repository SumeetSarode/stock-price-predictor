"""Pydantic schema for OHLCV price bars.

Use this at SYSTEM BOUNDARIES (JSON serialization, API responses, fixtures).
DO NOT use inside the engine — fetch_ohlcv() returns DataFrames for performance,
and pandas-ta + indicator math want DataFrames, not Pydantic instances.

Conversion at boundaries:
    df.reset_index().to_dict("records")  →  [OHLCVBar(**r) for r in records]
    [b.model_dump() for b in bars]       →  pd.DataFrame(...)

DESIGN ASSUMPTION — REGULAR-SESSION ONLY:
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

from pydantic import BaseModel, Field, model_validator


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
