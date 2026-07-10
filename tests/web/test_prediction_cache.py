"""Smoke tests for prediction_cache — append-only SQLite prediction store."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from price_predictor.web.services import prediction_cache as pc


def _view(ticker="RELIANCE.NS", horizon="weekly", direction="bullish", **over):
    """A minimal render-ready view dict shaped like _to_view_dict()."""
    base = {
        "ticker": ticker,
        "horizon": horizon,
        "direction": direction,
        "confidence_pct": 72,
        "close_price": 1400.0,
        "entry_low": 1390.0,
        "entry_high": 1410.0,
        "target_value": 1500.0,
        "stop_value": 1350.0,
        "risk_reward": 2.2,
    }
    base.update(over)
    return base


class TestSaveAndFetch:
    def test_empty_returns_none(self, tmp_db):
        assert pc.get_latest("RELIANCE", "weekly") is None

    def test_save_then_get_latest(self, tmp_db):
        pc.save(_view())
        got = pc.get_latest("reliance", "weekly")  # case + suffix insensitive
        assert got is not None
        assert got.ticker == "RELIANCE.NS"
        assert got.direction == "bullish"
        assert got.confidence_pct == 72
        assert got.risk_reward == 2.2
        # Full view blob round-trips.
        assert got.view["target_value"] == 1500.0

    def test_append_only_latest_wins(self, tmp_db):
        pc.save(_view(direction="bullish", confidence_pct=60))
        pc.save(_view(direction="bearish", confidence_pct=80))
        got = pc.get_latest("RELIANCE", "weekly")
        # Latest insert wins as "current".
        assert got.direction == "bearish"
        assert got.confidence_pct == 80

    def test_horizon_isolation(self, tmp_db):
        pc.save(_view(horizon="weekly"))
        assert pc.get_latest("RELIANCE", "daily") is None
        assert pc.get_latest("RELIANCE", "weekly") is not None

    def test_null_risk_reward(self, tmp_db):
        pc.save(_view(risk_reward=None))
        got = pc.get_latest("RELIANCE", "weekly")
        assert got.risk_reward is None

    def test_get_latest_many_bulk(self, tmp_db):
        pc.save(_view(ticker="RELIANCE.NS"))
        pc.save(_view(ticker="TCS.NS", direction="neutral"))
        result = pc.get_latest_many(["RELIANCE", "TCS", "INFY"], "weekly")
        assert set(result.keys()) == {"RELIANCE.NS", "TCS.NS"}  # INFY absent
        assert result["TCS.NS"].direction == "neutral"

    def test_get_latest_many_empty_input(self, tmp_db):
        assert pc.get_latest_many([], "weekly") == {}


class TestCachedPredictionProperties:
    def test_fresh_not_stale(self, tmp_db):
        pc.save(_view(horizon="weekly"))
        got = pc.get_latest("RELIANCE", "weekly")
        assert got.is_stale is False
        assert got.age_label == "just now"

    def test_stale_detection(self):
        # Build a CachedPrediction directly with an old timestamp — no DB.
        old = pc.CachedPrediction(
            ticker="RELIANCE.NS",
            horizon="daily",
            created_at=datetime.now(timezone.utc) - timedelta(days=3),
            direction="bullish",
            confidence_pct=50,
            close_price=1.0,
            entry_low=1.0,
            entry_high=1.0,
            target_value=1.0,
            stop_value=1.0,
            risk_reward=None,
            view={},
        )
        # Daily freshness is 24h → 3 days old is stale.
        assert old.is_stale is True
        assert "day" in old.age_label

    def test_age_label_hours(self):
        cp = pc.CachedPrediction(
            ticker="X.NS", horizon="weekly",
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
            direction="bullish", confidence_pct=1, close_price=1.0,
            entry_low=1.0, entry_high=1.0, target_value=1.0, stop_value=1.0,
            risk_reward=None, view={},
        )
        assert cp.age_label == "2 hrs ago"
