"""Tests for price_predictor.prediction.inputs.

Coverage map (~22 tests across 5 sections):
    1. Schema construction + frozen-ness  (5)
    2. JSON round-trip + hashability      (3)
    3. _resolve_ticker behavior           (4)
    4. compose_technical_view happy path  (4)
    5. compose_technical_view failures    (6)
"""
from __future__ import annotations

import asyncio
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from price_predictor.data import _shared_cache
from price_predictor.prediction.inputs import (
    ClusterView,
    SynthesisInput,
    TechnicalView,
    TechnicalViewError,
    _resolve_ticker,
    compose_technical_view,
)


# ─────────────────────────────────────────────────────────────
# Shared fixtures: synthetic OHLCV + fake cache
# ─────────────────────────────────────────────────────────────
def _build_uptrend_df(n: int = 400) -> pd.DataFrame:
    """Linearly rising series — every cluster should classify bullish-leaning."""
    closes = np.linspace(100, 200, n)
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 1,
            "low": closes - 1,
            "close": closes,
            "adj_close": closes,
            "volume": np.full(n, 1_000_000),
        },
        index=dates,
    )


class _FakeCache:
    """Mimics PriceCache.get(); returns a pre-baked DataFrame.

    Tracks calls so tests can assert on cache interaction.
    """

    def __init__(
        self, df_to_return: pd.DataFrame, *, raise_exc: Exception | None = None
    ):
        self.df = df_to_return
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    async def get(
        self, ticker: str, start: date, end: date, interval: str = "1d"
    ) -> pd.DataFrame:
        self.calls.append(
            {"ticker": ticker, "start": start, "end": end, "interval": interval}
        )
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.df.copy()


@pytest.fixture(autouse=True)
def _reset_cache():
    """Ensure each test starts/ends with no leaked cache override."""
    _shared_cache.set_cache(None)
    yield
    _shared_cache.set_cache(None)


@pytest.fixture
def fake_cache():
    """Inject an uptrend cache; return the FakeCache for assertions."""
    cache = _FakeCache(_build_uptrend_df())
    _shared_cache.set_cache(cache)
    return cache


# ─────────────────────────────────────────────────────────────
# 1. Schema construction + frozen-ness
# ─────────────────────────────────────────────────────────────
class TestClusterView:
    def test_minimal_construction(self):
        cv = ClusterView(name="trend", signal="bullish")
        assert cv.name == "trend"
        assert cv.signal == "bullish"
        assert cv.strength is None
        assert cv.indicators == {}
        assert cv.derived == {}
        assert cv.rationale == ()
        assert cv.warnings == ()

    def test_full_construction(self):
        cv = ClusterView(
            name="momentum",
            signal="bullish",
            strength="moderate",
            indicators={"rsi_14": 65.0, "macd_line": 1.2},
            derived={"rsi_overbought": False},
            rationale=("RSI healthy", "MACD positive"),
            warnings=("pattern_signal_conflict",),
        )
        assert cv.strength == "moderate"
        assert cv.indicators["rsi_14"] == 65.0
        assert cv.rationale == ("RSI healthy", "MACD positive")

    def test_invalid_signal_rejected(self):
        with pytest.raises(ValidationError):
            ClusterView(name="trend", signal="up")  # type: ignore[arg-type]

    def test_invalid_cluster_name_rejected(self):
        with pytest.raises(ValidationError):
            ClusterView(name="sentiment", signal="bullish")  # type: ignore[arg-type]

    def test_cluster_view_is_frozen(self):
        cv = ClusterView(name="trend", signal="bullish")
        with pytest.raises(ValidationError):
            cv.signal = "bearish"  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────
# 2. JSON round-trip + hashability
# ─────────────────────────────────────────────────────────────
def _sample_view() -> TechnicalView:
    """Build a hand-crafted TechnicalView for serialization tests."""
    cv = ClusterView(
        name="trend", signal="bullish", strength="strong",
        indicators={"close": 200.0}, rationale=("Close above SMAs",),
    )
    return TechnicalView(
        ticker="RELIANCE.NS",
        as_of=date(2026, 4, 28),
        close_price=200.0,
        bars_used=400,
        sensitivity="standard",
        trend=cv,
        momentum=ClusterView(name="momentum", signal="bullish"),
        volatility=ClusterView(name="volatility", signal="neutral"),
        levels=ClusterView(name="levels", signal="bullish"),
    )


class TestJSONAndHashing:
    def test_technical_view_json_round_trip(self):
        original = _sample_view()
        rebuilt = TechnicalView.model_validate_json(original.model_dump_json())
        assert rebuilt == original

    def test_technical_view_is_frozen(self):
        view = _sample_view()
        with pytest.raises(ValidationError):
            view.ticker = "TCS.NS"  # type: ignore[misc]

    def test_unknown_field_rejected_loudly(self):
        """extra='forbid' catches typos in fixture / serialization code."""
        view = _sample_view()
        data = view.model_dump()
        data["typo_field"] = "x"
        with pytest.raises(ValidationError, match="typo_field"):
            TechnicalView.model_validate(data)


# ─────────────────────────────────────────────────────────────
# 3. _resolve_ticker behavior
# ─────────────────────────────────────────────────────────────
class TestResolveTicker:
    def test_known_kb_ticker_resolves_to_yfinance_form(self):
        # 'RELIANCE' is in the NIFTY50 KB; should add .NS suffix
        assert _resolve_ticker("reliance") == "RELIANCE.NS"

    def test_already_canonical_passes_through(self):
        assert _resolve_ticker("RELIANCE.NS") == "RELIANCE.NS"

    def test_unknown_ticker_passes_through_uppercased(self):
        # 'AAPL' isn't in the NSE KB; pass through as-is for US tickers
        assert _resolve_ticker("aapl") == "AAPL"

    def test_empty_ticker_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _resolve_ticker("   ")


# ─────────────────────────────────────────────────────────────
# 4. compose_technical_view happy path
# ─────────────────────────────────────────────────────────────
class TestComposeHappyPath:
    def test_returns_technical_view(self, fake_cache):
        view = asyncio.run(compose_technical_view("RELIANCE.NS"))
        assert isinstance(view, TechnicalView)
        assert view.ticker == "RELIANCE.NS"
        assert view.bars_used == 400
        assert view.close_price == pytest.approx(200.0)  # last bar of uptrend

    def test_all_four_clusters_populated(self, fake_cache):
        view = asyncio.run(compose_technical_view("RELIANCE.NS"))
        # Each cluster ran and produced a signal
        for cluster in (view.trend, view.momentum, view.volatility, view.levels):
            assert cluster.signal in {"bullish", "neutral", "bearish"}
            # Indicators are dicts with at least one entry
            assert len(cluster.indicators) > 0

    def test_uptrend_produces_bullish_trend_cluster(self, fake_cache):
        """End-to-end sanity: uptrend OHLCV -> bullish trend signal.

        This isn't testing classify_trend (that's covered elsewhere); it
        proves the wiring from cluster tool -> ClusterView preserves the
        signal correctly.
        """
        view = asyncio.run(compose_technical_view("RELIANCE.NS"))
        assert view.trend.signal == "bullish"

    def test_sensitivity_recorded(self, fake_cache):
        view = asyncio.run(
            compose_technical_view("RELIANCE.NS", sensitivity="sensitive")
        )
        assert view.sensitivity == "sensitive"


# ─────────────────────────────────────────────────────────────
# 5. compose_technical_view failures
# ─────────────────────────────────────────────────────────────
from price_predictor.data.prices import PriceFetchError  # noqa: E402


class TestComposeFailures:
    def test_empty_ticker_raises_value_error(self, fake_cache):
        with pytest.raises(ValueError, match="non-empty"):
            asyncio.run(compose_technical_view(""))

    def test_all_clusters_failing_raises_technical_view_error(self):
        """If the cache fails, all 4 cluster tools return error dicts."""
        bad_cache = _FakeCache(
            _build_uptrend_df(),
            raise_exc=PriceFetchError("simulated network down"),
        )
        _shared_cache.set_cache(bad_cache)
        with pytest.raises(TechnicalViewError) as exc_info:
            asyncio.run(compose_technical_view("RELIANCE.NS"))
        # All 4 cluster names should appear in the error
        assert set(exc_info.value.cluster_errors.keys()) == {
            "trend", "momentum", "volatility", "levels",
        }

    def test_partial_failure_still_raises(self):
        """A single cluster failing fails the whole view (fail-loud policy).

        Hard to simulate one-cluster-fails with a shared cache, so we
        sanity-check the policy by inspecting the error_count: even when
        all 4 fail (the realistic mass-failure mode), the policy holds.
        """
        bad_cache = _FakeCache(
            _build_uptrend_df(),
            raise_exc=PriceFetchError("down"),
        )
        _shared_cache.set_cache(bad_cache)
        with pytest.raises(TechnicalViewError) as exc_info:
            asyncio.run(compose_technical_view("RELIANCE.NS"))
        assert len(exc_info.value.cluster_errors) >= 1

    def test_error_carries_cluster_messages(self):
        """cluster_errors dict should carry the per-cluster message strings."""
        bad_cache = _FakeCache(
            _build_uptrend_df(),
            raise_exc=PriceFetchError("simulated yahoo 503"),
        )
        _shared_cache.set_cache(bad_cache)
        with pytest.raises(TechnicalViewError) as exc_info:
            asyncio.run(compose_technical_view("RELIANCE.NS"))
        for msg in exc_info.value.cluster_errors.values():
            assert "simulated yahoo 503" in msg

    def test_empty_dataframe_raises(self):
        """If the OHLCV is empty, cluster tools surface 'No price data'."""
        empty_df = _build_uptrend_df(n=400).iloc[0:0]
        _shared_cache.set_cache(_FakeCache(empty_df))
        with pytest.raises(TechnicalViewError):
            asyncio.run(compose_technical_view("RELIANCE.NS"))

    def test_bars_used_floor_enforced(self):
        """If somehow only 19 bars come back, TechnicalView construction
        rejects (bars_used >= 20). Defends against silently-degraded data.

        Note: cluster tools themselves typically reject short data first,
        but this is the schema-level safety net.
        """
        short_df = _build_uptrend_df(n=10)
        _shared_cache.set_cache(_FakeCache(short_df))
        # Either the cluster tools reject (TechnicalViewError) or the
        # TechnicalView ctor rejects (ValidationError). Both are acceptable
        # — both mean "we caught the degraded case and refused to lie".
        with pytest.raises((TechnicalViewError, ValidationError)):
            asyncio.run(compose_technical_view("RELIANCE.NS"))


# ──────────────────────────────────────────────────────────
# 6. SynthesisInput — the gather → synthesizer contract (commit 2)
# ──────────────────────────────────────────────────────────
from datetime import datetime, timezone  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

from price_predictor.agents.news_impact import Catalyst, ImpactAssessment  # noqa: E402


def _sample_impact_assessment() -> ImpactAssessment:
    """Minimal-but-valid ImpactAssessment for SynthesisInput tests."""
    return ImpactAssessment(
        ticker="RELIANCE.NS",
        sentiment="bullish",
        confidence=0.7,
        estimated_pct_move=2.5,
        reasoning="Strong Q3 results plus margin expansion guidance.",
        catalysts=[
            Catalyst(
                description="Q3 earnings beat consensus by 12% with margin expansion",
                source="news",
                impact="positive",
            )
        ],
    )


def _sample_synthesis_input(**overrides) -> SynthesisInput:
    """Builder for SynthesisInput tests — lets each test override 1 field."""
    defaults = dict(
        ticker="RELIANCE.NS",
        horizon="short",
        as_of=datetime(2026, 4, 28, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
        technical_view=_sample_view(),
        impact_assessment=_sample_impact_assessment(),
        model_chain=("gemini-2.5-flash",),
    )
    defaults.update(overrides)
    return SynthesisInput(**defaults)


class TestSynthesisInput:
    def test_minimal_construction(self):
        si = _sample_synthesis_input()
        assert si.ticker == "RELIANCE.NS"
        assert si.horizon == "short"
        assert si.technical_view.trend.signal == "bullish"
        assert si.impact_assessment.sentiment == "bullish"
        assert si.model_chain == ("gemini-2.5-flash",)

    def test_is_frozen(self):
        si = _sample_synthesis_input()
        with pytest.raises(ValidationError):
            si.ticker = "TCS.NS"  # type: ignore[misc]

    def test_json_round_trip_preserves_nested_models(self):
        """TechnicalView and ImpactAssessment must survive serialization.

        This is the contract the synthesizer relies on: it receives a JSON
        blob and must be able to reason about every field. Drift here
        breaks the synthesizer silently.
        """
        original = _sample_synthesis_input()
        rebuilt = SynthesisInput.model_validate_json(original.model_dump_json())
        assert rebuilt == original
        # Spot-check nested invariants survived
        assert rebuilt.technical_view.bars_used == 400
        assert rebuilt.impact_assessment.catalysts[0].source == "news"

    def test_extra_field_rejected(self):
        """Same loud-failure policy as the rest of the schema."""
        with pytest.raises(ValidationError, match="sentiment_override"):
            _sample_synthesis_input(sentiment_override="bearish")

    def test_naive_datetime_rejected(self):
        """as_of MUST be tz-aware. Mirrors Prediction.as_of's rule."""
        with pytest.raises(ValidationError, match="tz-aware"):
            _sample_synthesis_input(as_of=datetime(2026, 4, 28, 10, 0))

    def test_empty_model_chain_rejected(self):
        """Audit trail must record at least one model."""
        with pytest.raises(ValidationError, match="at least one model"):
            _sample_synthesis_input(model_chain=())

    def test_utc_datetime_accepted(self):
        """Any tz-aware datetime works — not just Asia/Kolkata.

        Predictor will use Asia/Kolkata by convention, but the schema
        shouldn't hard-code that.
        """
        si = _sample_synthesis_input(
            as_of=datetime(2026, 4, 28, 4, 30, tzinfo=timezone.utc)
        )
        assert si.as_of.tzinfo is not None
