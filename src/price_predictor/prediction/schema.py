"""Pydantic schema for prediction outputs (Step 3.4.1).

PURPOSE
=======
The output contract every layer of Step D (Prediction Agent) will produce,
consume, log, and (eventually) backtest against. Defined FIRST so producers
and consumers can be built against a single source of truth.

DESIGN INVARIANTS (locked with user before building)
====================================================
1. **Single target** (NOT a list / ladder). YAGNI for v1 — extending to
   a target ladder later is a non-breaking change on a frozen model.
2. **Worst-case risk_reward** when entry is a range. For bullish trades
   we anchor RR math at entry_zone[1] (the top of the zone — the price
   you'd be unlucky enough to fill at); for bearish, entry_zone[0]
   (the bottom — worst short fill). This is the conservative "risk
   filter" semantic that pro traders prefer.
3. **Embedded analysis_basis** — every prediction is self-contained. A
   week from now you can answer "why did the agent think this?" from
   ONE JSON file, no DB lookup required. Backtesting / calibration
   layer (Step 3.5) will consume this directly.

PYDANTIC CONVENTIONS (carry from data/schema.py)
================================================
- Pydantic v2 `BaseModel` everywhere
- All fields have explicit `Field(..., description=...)` for self-doc
- `model_config = ConfigDict(frozen=True)` on EVERY model — predictions
  are facts about a moment in time, not mutable state
- All nested models are frozen too, so the parent stays hashable
- Collections are `tuple[...]` not `list[...]` so the frozen model is
  hashable. JSON round-trip handles tuples as arrays cleanly.
- tz-aware datetime everywhere (Asia/Kolkata for IST market data)
- `model_validator(mode="after")` for cross-field consistency
- `@computed_field` for derived values (risk_reward) so there's exactly
  ONE source of truth, with no chance of input/computed drift

JSON ROUND-TRIP CONTRACT
========================
Every Prediction MUST satisfy:
    p == Prediction.model_validate_json(p.model_dump_json())

This is enforced in tests. Any change that breaks round-trip is a
breaking change to consumers (logs, UIs, backtest replay).
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)

from price_predictor.prediction.trading_calendar import (
    target_datetime_for_horizon,
)


# ─────────────────────────────────────────────────────────────
# Enums (stable controlled vocabularies)
# ─────────────────────────────────────────────────────────────
class PredictionDirection(str, Enum):
    """Direction the agent expects price to move over the horizon.

    Why `str, Enum`: lets Pydantic serialize as plain strings ("bullish"
    not "PredictionDirection.BULLISH") without custom encoders. Same
    pattern used everywhere in the codebase.

    Why include NEUTRAL: range-bound predictions are real and honest;
    forcing the agent to pick bullish-or-bearish when it sees neither
    leads to overconfident hallucinations. Neutral lets the agent say
    "I don't see edge here" cleanly.
    """

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class PredictionHorizon(str, Enum):
    """Time window over which the prediction is intended to play out.

    Concrete time labels matching user vocabulary and the project
    description's daily/weekly contract. Maps to calendar windows via
    `trading_calendar.target_datetime_for_horizon`:

      DAILY    -> end of next applicable NSE session (today's close if
                  before 15:30 IST on a trading day, else next session)
      WEEKLY   -> as_of + 7  calendar days, snapped to last trading day
      BIWEEKLY -> as_of + 14 calendar days, snapped to last trading day
      MONTHLY  -> as_of + 1  calendar month  (relativedelta), snapped

    Future horizons (parked in next_steps): SIX_MONTHS, YEARLY. Custom
    durations are NOT supported — calibration becomes meaningless when
    every prediction is its own bucket.
    """

    DAILY    = "daily"
    WEEKLY   = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY  = "monthly"


# ─────────────────────────────────────────────────────────────
# Reusable typed level (entry / target / stop)
# ─────────────────────────────────────────────────────────────
class PriceLevel(BaseModel):
    """A price + the WHY behind it.

    Pairing the number with its rationale makes the prediction
    self-explaining. Without rationale, "target=1600" is just a magic
    number; with it, "target=1600 (next major resistance from 2024 high)"
    is reviewable. The LLM is strongly encouraged to populate the
    rationale field with terse, specific reasoning.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: float = Field(
        ..., gt=0,
        description="The price level itself. Must be strictly positive.",
    )
    rationale: str = Field(
        ..., min_length=1,
        description=(
            "One-line WHY this level was chosen. "
            "E.g. '20-day SMA + nearest swing high'."
        ),
    )


# ─────────────────────────────────────────────────────────────
# Audit trail (embedded sub-model — required, not optional)
# ─────────────────────────────────────────────────────────────
class AnalysisBasis(BaseModel):
    """Snapshot of what the agent KNEW at prediction time.

    SEMANTICS: these values are FROZEN at prediction time. Re-fetching
    later gives different numbers (markets move). This is the WHOLE POINT
    — backtesting / calibration needs to know what the agent saw, not
    what's true now.

    REQUIRED: not Optional. Every prediction must populate at least the
    bare minimum (close_price_at_prediction, bars_used, technical_summary).
    News/filings counts are optional (agent may run without those layers).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    close_price_at_prediction: float = Field(
        ..., gt=0,
        description=(
            "The most recent close price the agent saw. "
            "Anchors all entry/target/stop sanity checks."
        ),
    )
    bars_used: int = Field(
        ..., ge=20,
        description=(
            "How many OHLCV bars the technical analysis ran over. "
            "Floor of 20 because most indicators are noise below that."
        ),
    )
    technical_summary: str = Field(
        ..., min_length=1,
        description=(
            "One-line distillation of the technical view. "
            "E.g. 'Trend bullish (ADX 32), RSI 68, BB squeeze breaking up'."
        ),
    )
    news_sentiment_score: float | None = Field(
        default=None, ge=-1.0, le=1.0,
        description=(
            "Aggregate news sentiment over the news window, [-1, +1]. "
            "None if no news layer ran."
        ),
    )
    news_articles_considered: int = Field(
        default=0, ge=0,
        description="How many articles the news layer scored. 0 if news skipped.",
    )
    filings_considered: int = Field(
        default=0, ge=0,
        description="How many corporate filings considered. 0 if filings skipped.",
    )


# ─────────────────────────────────────────────────────────────
# The main prediction — what the agent emits
# ─────────────────────────────────────────────────────────────
class Prediction(BaseModel):
    """One actionable (or 'I-don't-see-it') prediction for one ticker.

    This is the OUTPUT contract for Step D. Every field is here for a
    reason — see field-level docstrings.

    HASHABILITY
    ===========
    `frozen=True` + tuple-typed collections (model_chain, signals) +
    tuple-typed entry_zone make the whole model hashable. This lets us
    use Predictions as dict keys / set members during de-duplication
    in batch pipelines (Step 3.4.3).

    DIRECTION-SPECIFIC INVARIANTS
    =============================
    - BULLISH: target.value > entry_zone[1] AND stop_loss.value < entry_zone[1]
    - BEARISH: target.value < entry_zone[0] AND stop_loss.value > entry_zone[0]
    - NEUTRAL: no level-ordering enforced (range-bound predictions don't
      have a clean "right side"); RR is forced to 1.0 because there's no
      directional edge to measure.

    These are checked in the model_validator below.

    EXTRA-FIELD POLICY
    ==================
    `extra='forbid'` — unknown fields raise ValidationError. This is a
    deliberate safety net for LLM-generated JSON: if the agent emits
    `confidence_score` instead of `confidence`, we want to FAIL LOUDLY
    rather than silently drop the value. Same applies to attempts to
    override computed fields like `risk_reward` from input.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ── Identity / provenance ────────────────────────────────────────
    ticker: str = Field(
        ..., min_length=1,
        description=(
            "Canonical yfinance ticker (e.g. 'RELIANCE.NS'). "
            "Caller is responsible for KB resolution before construction."
        ),
    )
    as_of: datetime = Field(
        ...,
        description=(
            "When this prediction was generated. MUST be tz-aware. "
            "Convention: Asia/Kolkata for India-market predictions."
        ),
    )
    horizon: PredictionHorizon = Field(
        ..., description="Time window over which this prediction applies.",
    )
    model_chain: tuple[str, ...] = Field(
        ...,
        description=(
            "Which LLMs participated, in order they were tried. "
            "Audit trail for resilience-chain debugging. "
            "E.g. ('gemini/gemini-2.5-flash',) or "
            "('groq/llama-3.3-70b', 'gemini/gemini-2.5-flash') if first failed."
        ),
    )

    # ── Core prediction ──────────────────────────────────────────────
    direction: PredictionDirection = Field(
        ..., description="Bullish / bearish / neutral over the horizon.",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "Calibrated confidence in [0, 1]. "
            "0.5 = coin flip; 0.8 = high conviction; 1.0 reserved for "
            "impossible-to-reach 'certainty' (so we never emit it)."
        ),
    )
    entry_zone: tuple[float, float] = Field(
        ...,
        description=(
            "Suggested entry range (low, high). Both must be > 0 and "
            "low <= high. For neutral predictions, this is informational only."
        ),
    )
    target: PriceLevel = Field(
        ..., description="Single profit target. v1 uses one target; ladders are deferred.",
    )
    stop_loss: PriceLevel = Field(
        ...,
        description=(
            "Stop-loss level. Position is exited if price hits this "
            "(intraday breach is what counts in real trading)."
        ),
    )

    # ── Reasoning (human-readable + LLM-quotable) ───────────────────
    rationale: str = Field(
        ..., min_length=1,
        description=(
            "Multi-paragraph synthesis. The 'show your work' section. "
            "Should weave technical + news + filings into a coherent story."
        ),
    )
    contributing_signals: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Bullet-list of signals that SUPPORT the direction call. "
            "Mirrors the C-tool 'rationale' pattern that reduces LLM "
            "hallucination — gives consumers ready-made prose."
        ),
    )
    conflicting_signals: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Bullet-list of signals that POINT THE OTHER WAY. "
            "Surface contradictions explicitly so consumers don't have to "
            "infer them from the absence of mention."
        ),
    )

    # ── Audit trail (required) ──────────────────────────────────────
    analysis_basis: AnalysisBasis = Field(
        ...,
        description=(
            "Snapshot of what the agent KNEW at prediction time. "
            "Required so each Prediction is a self-contained audit unit."
        ),
    )

    # ── Compliance / disclaimer ─────────────────────────────────────
    not_advice: bool = Field(
        default=True,
        description=(
            "Always True. Present as a field so UIs / consumers can render "
            "an explicit disclaimer banner without out-of-band knowledge."
        ),
    )
    is_educational: bool = Field(
        default=True,
        description="Same spirit as not_advice — explicit field for UI rendering.",
    )

    # ── Input normalization (drop computed fields on parse) ─────────
    @model_validator(mode="before")
    @classmethod
    def _strip_computed_fields(cls, data: object) -> object:
        """Drop @computed_field values from input dict before validation.

        WHY: model_dump_json() includes computed fields (risk_reward,
        target_datetime), but extra='forbid' would reject them on
        model_validate_json(). Stripping them here makes JSON round-trip
        work while still catching genuine LLM typos like 'confidence_score'
        instead of 'confidence'.

        Only acts on dict inputs (the JSON-parsing path); leaves model
        instances and other types untouched.
        """
        _COMPUTED_KEYS = {"risk_reward", "target_datetime"}
        if isinstance(data, dict):
            # Copy to avoid mutating caller's dict
            data = {k: v for k, v in data.items() if k not in _COMPUTED_KEYS}
        return data

    # ── Cross-field validation ──────────────────────────────────────
    @model_validator(mode="after")
    def _validate_invariants(self) -> Prediction:
        """Enforce invariants the per-field constraints can't catch.

        Ordered by failure-frequency (cheap fail-fast checks first).
        """
        # 1. tz-aware as_of
        if self.as_of.tzinfo is None:
            raise ValueError("as_of must be tz-aware (got naive datetime)")

        # 2. entry_zone ordering + positivity
        lo, hi = self.entry_zone
        if lo <= 0 or hi <= 0:
            raise ValueError(
                f"entry_zone values must be > 0; got ({lo}, {hi})"
            )
        if lo > hi:
            raise ValueError(
                f"entry_zone must be (low, high); got ({lo}, {hi}) where low > high"
            )

        # 3. model_chain non-empty
        if len(self.model_chain) == 0:
            raise ValueError("model_chain must contain at least one model name")

        # 4. Direction-specific level ordering (skipped for neutral)
        if self.direction == PredictionDirection.BULLISH:
            self._check_bullish_levels()
        elif self.direction == PredictionDirection.BEARISH:
            self._check_bearish_levels()
        # NEUTRAL: no ordering enforced — range-bound is symmetric

        return self

    def _check_bullish_levels(self) -> None:
        """For bullish: target above entry-top, stop below entry-top."""
        entry_top = self.entry_zone[1]
        if self.target.value <= entry_top:
            raise ValueError(
                f"bullish prediction requires target ({self.target.value}) "
                f"> entry_zone top ({entry_top})"
            )
        if self.stop_loss.value >= entry_top:
            raise ValueError(
                f"bullish prediction requires stop_loss ({self.stop_loss.value}) "
                f"< entry_zone top ({entry_top})"
            )

    def _check_bearish_levels(self) -> None:
        """For bearish: target below entry-bottom, stop above entry-bottom."""
        entry_bot = self.entry_zone[0]
        if self.target.value >= entry_bot:
            raise ValueError(
                f"bearish prediction requires target ({self.target.value}) "
                f"< entry_zone bottom ({entry_bot})"
            )
        if self.stop_loss.value <= entry_bot:
            raise ValueError(
                f"bearish prediction requires stop_loss ({self.stop_loss.value}) "
                f"> entry_zone bottom ({entry_bot})"
            )

    # ── Computed (single source of truth, no input/derived drift) ───
    @computed_field  # type: ignore[prop-decorator]
    @property
    def risk_reward(self) -> float:
        """Worst-case risk-to-reward ratio.

        WHY WORST-CASE (not midpoint):
        ============================
        User-locked design choice. For bullish trades we anchor at the TOP
        of the entry zone — the price you'd be unlucky enough to fill at.
        For bearish, the bottom (worst short fill). Result is conservative;
        risk filters like `if rr >= 2.0` then become "even in the worst
        execution case, RR is still good".

        Bullish:  RR = (target - entry_top)    / (entry_top - stop)
        Bearish:  RR = (entry_bot - target)    / (stop - entry_bot)
        Neutral:  RR = 1.0                     (no directional edge)

        Returns:
            Float >= 0. 1.0 for neutral. Computed (never user-supplied)
            so the value is ALWAYS consistent with the underlying levels.
        """
        if self.direction == PredictionDirection.NEUTRAL:
            # No directional edge -> RR = 1 by convention.
            # Neutral predictions exist to say "no trade", not to be sized.
            return 1.0

        if self.direction == PredictionDirection.BULLISH:
            entry = self.entry_zone[1]   # worst (highest) fill
            reward = self.target.value - entry
            risk   = entry - self.stop_loss.value
        else:  # BEARISH
            entry = self.entry_zone[0]   # worst (lowest) short fill
            reward = entry - self.target.value
            risk   = self.stop_loss.value - entry

        # Both reward and risk are guaranteed > 0 by the model_validator
        # (direction-specific level ordering). Defensive guard for the
        # impossible case anyway.
        if risk <= 0:  # pragma: no cover (validator catches this)
            return 0.0
        return reward / risk

    @computed_field  # type: ignore[prop-decorator]
    @property
    def target_datetime(self) -> datetime:
        """The moment this prediction will be evaluated against actuals.

        Derived from (horizon, as_of) via the NSE trading-calendar.
        Always lands at 15:30 IST on a real NSE trading day.

        WHY @computed_field (not stored):
        - Single source of truth: the trading_calendar module.
        - No validator-populated field to maintain.
        - Still serializes into JSON (audit trail preserved).
        - The agent uses the calendar at predict-time — that's the
          moment that matters. We deliberately don't try to insulate
          historical predictions against retroactive holiday changes;
          the rare cost isn't worth the per-prediction storage bloat.

        Example mapping (predicted on Wed Apr 15 2026 at 10:00 IST):
          DAILY    -> Wed Apr 15 2026 15:30 IST  (today's close)
          WEEKLY   -> Wed Apr 22 2026 15:30 IST  (+7 cal days)
          BIWEEKLY -> Wed Apr 29 2026 15:30 IST  (+14 cal days)
          MONTHLY  -> Fri May 15 2026 15:30 IST  (+1 cal month)

        Returns:
            Tz-aware datetime in IST.
        """
        return target_datetime_for_horizon(self.horizon.value, self.as_of)
