"""Unit tests for PredictionStore (Step 3.4.3 commit 2).

Coverage:
  - Round-trip save+load (the basics)
  - Filename layout matches spec
  - Atomic write (no .tmp leftovers, no half-written files)
  - Idempotent save (overwrites cleanly)
  - Path traversal safety (ticker sanitization)
  - list_for_ticker / list_in_date_range / count
  - Foreign content in root (other dirs/files) ignored gracefully
  - Corrupted JSON load raises PredictionStoreError
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from price_predictor.prediction import (
    Prediction,
    PredictionStore,
    PredictionStoreError,
)
from price_predictor.prediction.schema import (
    AnalysisBasis,
    PredictionDirection,
    PredictionHorizon,
    PriceLevel,
)


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────
def _make_pred(
    ticker: str = "RELIANCE.NS",
    *,
    as_of: datetime | None = None,
    horizon: PredictionHorizon = PredictionHorizon.WEEKLY,
) -> Prediction:
    if as_of is None:
        as_of = datetime(2026, 4, 28, 10, 30, 45, tzinfo=ZoneInfo("Asia/Kolkata"))
    return Prediction(
        ticker=ticker,
        as_of=as_of,
        horizon=horizon,
        model_chain=("news_impact:agentic", "synthesizer:agentic"),
        direction=PredictionDirection.BULLISH,
        confidence=0.7,
        entry_zone=(1453.0, 1457.0),
        target=PriceLevel(value=1500.0, rationale="swing high"),
        stop_loss=PriceLevel(value=1425.0, rationale="below swing low"),
        rationale="Bullish across multiple timeframes.",
        contributing_signals=("trend bullish",),
        conflicting_signals=(),
        analysis_basis=AnalysisBasis(
            close_price_at_prediction=1455.0,
            bars_used=400,
            technical_summary="trend bullish, momentum bullish",
            news_sentiment_score=0.6,
            news_articles_considered=5,
            filings_considered=0,
        ),
    )


# ─────────────────────────────────────────────────────────────
# Construction + path computation
# ─────────────────────────────────────────────────────────────
class TestStoreInit:
    def test_creates_root_dir(self, tmp_path: Path):
        root = tmp_path / "predictions"
        assert not root.exists()
        PredictionStore(root)
        assert root.exists() and root.is_dir()

    def test_existing_dir_is_fine(self, tmp_path: Path):
        # Should not raise on second construction
        PredictionStore(tmp_path)
        PredictionStore(tmp_path)

    def test_accepts_str_or_path(self, tmp_path: Path):
        store = PredictionStore(str(tmp_path))
        assert store.root == tmp_path.resolve()


class TestPathComputation:
    def test_path_layout_matches_spec(self, tmp_path: Path):
        store = PredictionStore(tmp_path)
        pred = _make_pred()
        path = store.path_for(pred)
        # Expected: {root}/2026-04-28/RELIANCE.NS_103045_weekly.json
        assert path.parent.name == "2026-04-28"
        assert path.name == "RELIANCE.NS_103045_weekly.json"

    def test_horizon_in_filename(self, tmp_path: Path):
        store = PredictionStore(tmp_path)
        for h in PredictionHorizon:
            pred = _make_pred(horizon=h)
            assert h.value in store.path_for(pred).name


# ─────────────────────────────────────────────────────────────
# Save + load round-trip
# ─────────────────────────────────────────────────────────────
class TestSaveLoad:
    def test_round_trip(self, tmp_path: Path):
        store = PredictionStore(tmp_path)
        original = _make_pred()
        path = store.save(original)
        loaded = store.load(path)
        # Pydantic equality compares all fields
        assert loaded == original

    def test_save_returns_correct_path(self, tmp_path: Path):
        store = PredictionStore(tmp_path)
        pred = _make_pred()
        path = store.save(pred)
        assert path == store.path_for(pred)
        assert path.exists()

    def test_save_creates_day_dir(self, tmp_path: Path):
        store = PredictionStore(tmp_path)
        pred = _make_pred()
        store.save(pred)
        assert (tmp_path / "2026-04-28").is_dir()

    def test_save_is_idempotent(self, tmp_path: Path):
        """Saving the same prediction twice should be a clean overwrite."""
        store = PredictionStore(tmp_path)
        pred = _make_pred()
        store.save(pred)
        store.save(pred)  # no exception
        assert store.count() == 1  # not duplicated

    def test_no_tmp_leftovers(self, tmp_path: Path):
        """Atomic write should leave no .tmp files behind on success."""
        store = PredictionStore(tmp_path)
        store.save(_make_pred())
        leftovers = list(tmp_path.rglob("*.tmp"))
        assert leftovers == []


# ─────────────────────────────────────────────────────────────
# Filename safety
# ─────────────────────────────────────────────────────────────
class TestFilenameSafety:
    def test_path_traversal_chars_stripped(self, tmp_path: Path):
        """Saved path must NEVER escape the store root, even with funky
        ticker chars. The actual security property: resolved path is
        a descendant of root. (Substrings like '..' inside a filename
        are harmless - only '..' as a path COMPONENT is dangerous.)
        """
        store = PredictionStore(tmp_path)
        pred = _make_pred(ticker="EVIL..NS")
        path = store.save(pred).resolve()
        # Real security check: path stays inside the store root
        assert str(path).startswith(str(store.root))
        # No path component is literally '..'
        assert ".." not in path.parts

    def test_empty_after_sanitization_raises(self, tmp_path: Path):
        """A ticker that sanitizes to empty must error out clearly."""
        from price_predictor.prediction.store import _safe_ticker_for_filename
        with pytest.raises(PredictionStoreError, match="empty"):
            _safe_ticker_for_filename("@@@@")

    def test_lowercase_ticker_uppercased(self, tmp_path: Path):
        store = PredictionStore(tmp_path)
        pred = _make_pred(ticker="reliance.ns")
        # Schema may or may not uppercase; store always does for filename
        path = store.save(pred)
        assert "RELIANCE.NS" in path.name


# ─────────────────────────────────────────────────────────────
# Listing
# ─────────────────────────────────────────────────────────────
class TestListing:
    def test_list_for_ticker_chronological(self, tmp_path: Path):
        store = PredictionStore(tmp_path)
        # Three predictions on different days
        ist = ZoneInfo("Asia/Kolkata")
        store.save(_make_pred(as_of=datetime(2026, 4, 26, 10, 0, 0, tzinfo=ist)))
        store.save(_make_pred(as_of=datetime(2026, 4, 28, 10, 0, 0, tzinfo=ist)))
        store.save(_make_pred(as_of=datetime(2026, 4, 27, 10, 0, 0, tzinfo=ist)))

        results = store.list_for_ticker("RELIANCE.NS")
        assert len(results) == 3
        assert [p.as_of.date() for p in results] == [
            date(2026, 4, 26),
            date(2026, 4, 27),
            date(2026, 4, 28),
        ]

    def test_list_for_ticker_filters_by_ticker(self, tmp_path: Path):
        store = PredictionStore(tmp_path)
        store.save(_make_pred(ticker="RELIANCE.NS"))
        store.save(_make_pred(ticker="INFY.NS"))
        store.save(_make_pred(ticker="TCS.NS"))

        results = store.list_for_ticker("INFY.NS")
        assert len(results) == 1
        assert results[0].ticker == "INFY.NS"

    def test_list_for_unknown_ticker_returns_empty(self, tmp_path: Path):
        store = PredictionStore(tmp_path)
        store.save(_make_pred())
        assert store.list_for_ticker("NOPE.NS") == []

    def test_list_in_date_range_inclusive(self, tmp_path: Path):
        store = PredictionStore(tmp_path)
        ist = ZoneInfo("Asia/Kolkata")
        store.save(_make_pred(as_of=datetime(2026, 4, 25, 10, 0, 0, tzinfo=ist)))
        store.save(_make_pred(as_of=datetime(2026, 4, 27, 10, 0, 0, tzinfo=ist)))
        store.save(_make_pred(as_of=datetime(2026, 4, 30, 10, 0, 0, tzinfo=ist)))

        in_range = store.list_in_date_range(date(2026, 4, 26), date(2026, 4, 28))
        assert len(in_range) == 1
        assert in_range[0].as_of.date() == date(2026, 4, 27)

    def test_list_in_date_range_inverted_raises(self, tmp_path: Path):
        store = PredictionStore(tmp_path)
        with pytest.raises(ValueError, match="must be <="):
            store.list_in_date_range(date(2026, 4, 30), date(2026, 4, 1))

    def test_count_across_days(self, tmp_path: Path):
        store = PredictionStore(tmp_path)
        ist = ZoneInfo("Asia/Kolkata")
        store.save(_make_pred(as_of=datetime(2026, 4, 25, 10, 0, 0, tzinfo=ist)))
        store.save(_make_pred(
            ticker="INFY.NS",
            as_of=datetime(2026, 4, 25, 11, 0, 0, tzinfo=ist),
        ))
        store.save(_make_pred(as_of=datetime(2026, 4, 28, 12, 0, 0, tzinfo=ist)))
        assert store.count() == 3


# ─────────────────────────────────────────────────────────────
# Robustness — foreign content + corruption
# ─────────────────────────────────────────────────────────────
class TestRobustness:
    def test_foreign_dirs_in_root_ignored(self, tmp_path: Path):
        """User might drop other stuff into the root - we must not crash."""
        store = PredictionStore(tmp_path)
        store.save(_make_pred())
        # Create a non-date directory
        (tmp_path / "scratch").mkdir()
        (tmp_path / "scratch" / "notes.txt").write_text("hi")
        # Create a stray file at root
        (tmp_path / "README").write_text("hi")
        # Create a non-conforming file inside a real day dir
        (tmp_path / "2026-04-28" / "garbage.json").write_text("{}")

        # All operations should still work - foreign content is ignored
        assert store.count() == 1
        assert len(store.list_for_ticker("RELIANCE.NS")) == 1
        in_range = store.list_in_date_range(date(2026, 1, 1), date(2026, 12, 31))
        assert len(in_range) == 1

    def test_corrupted_file_raises_clean_error(self, tmp_path: Path):
        store = PredictionStore(tmp_path)
        pred = _make_pred()
        path = store.save(pred)
        # Corrupt the file
        path.write_text("{ not valid json")

        with pytest.raises(PredictionStoreError, match="validation"):
            store.load(path)

    def test_missing_file_raises_clean_error(self, tmp_path: Path):
        store = PredictionStore(tmp_path)
        with pytest.raises(PredictionStoreError, match="Cannot read"):
            store.load(tmp_path / "nope.json")
