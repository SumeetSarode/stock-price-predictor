"""Unit tests for kb.membership (Step 2.5, Commit A).

Strategy
========
Tests build a small synthetic JSON fixture in a tmp_path and load it
through the public API. No real Wikipedia data here -- those golden-
path probes (JETAIRWAYS, HDFC) land in Commit C against the real
data/kb/index_membership.json.

Why a synthetic fixture (not a tiny real one)
=============================================
- Keeps tests deterministic across re-bootstraps of the real file.
- Lets us construct edge cases (out-of-order events, missing fields,
  bad symbols) that real data never has.
- The fixture has 3 fake tickers and 4 events spanning ~5 years,
  enough to exercise every code path in members_on / changes_in_range.

Test file is imported as ``tests.kb.test_membership``; pytest picks it
up via the standard recursive discovery.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from price_predictor.kb.membership import (
    IndexEvent,
    MembershipDataError,
    MembershipHistory,
    NIFTY50_EXPECTED_COUNT,
    _build_from_payload,
    _clear_cache,
    _parse_event,
    _parse_iso_date,
    changes_in_range,
    load_membership_history,
    members_on,
    was_member,
)


# ─────────────────────────────────────────────────────────────────
# Fixtures + factories
# ─────────────────────────────────────────────────────────────────
def _ns_fillers(start_index: int, count: int) -> list[str]:
    """Generate count fake .NS tickers starting at FAKE{start_index}.NS.

    Used to pad current_members up to NIFTY50_EXPECTED_COUNT (50) for
    the schema invariant -- tests focus on a small subset of "real"
    tickers but the count check is non-negotiable.
    """
    return [f"FAKE{i:03d}.NS" for i in range(start_index, start_index + count)]


# Three "real" tickers we'll move in and out of the index across events.
# Suffixed .NS to satisfy the symbol regex.
_RELIANCE = "RELIANCE.NS"
_INFY = "INFY.NS"
_OLDCO = "OLDCO.NS"  # was a member, removed in 2019
_NEWCO = "NEWCO.NS"  # added in 2019 (replaces OLDCO)
_RECENT = "RECENT.NS"  # added in 2023 (replaces some filler)


def _build_fixture_payload() -> dict:
    """Synthetic payload for one index. Designed to exercise every path.

    Timeline (all dates inclusive in [2015-01-01, 2024-12-31]):
        2015-01-01: history starts
        2018-09-15: (no events) -- pure backwards-walk to anchor test
        2019-04-08: OLDCO removed, NEWCO added
        2023-06-01: FAKE049 removed, RECENT added
        2024-12-31: snapshot date

    Current members (50 total): RELIANCE, INFY, NEWCO, RECENT,
    + FAKE000..FAKE045 (= 46 fillers). 4 + 46 = 50. OLDCO and FAKE049
    are NOT current.
    """
    current = [_RELIANCE, _INFY, _NEWCO, _RECENT] + _ns_fillers(0, 46)
    assert len(current) == NIFTY50_EXPECTED_COUNT
    # Events sorted DESCENDING by date (the storage convention).
    events = [
        {"symbol": _RECENT, "action": "added",
         "date": "2023-06-01", "reason": "test"},
        {"symbol": "FAKE049.NS", "action": "removed",
         "date": "2023-06-01", "reason": "test"},
        {"symbol": _NEWCO, "action": "added",
         "date": "2019-04-08", "reason": "test"},
        {"symbol": _OLDCO, "action": "removed",
         "date": "2019-04-08", "reason": "Inadequate market capitalization"},
    ]
    return {
        "display_name": "Fake Nifty 50",
        "source_url": "https://example.com/fixture",
        "history_starts": "2015-01-01",
        "current_snapshot_date": "2024-12-31",
        "current_members": current,
        "events": events,
    }


@pytest.fixture
def fixture_path(tmp_path: Path) -> Path:
    """Write a fresh fixture JSON for one test, return its path."""
    payload = {"NIFTY50": _build_fixture_payload()}
    out = tmp_path / "index_membership.json"
    out.write_text(json.dumps(payload))
    return out


@pytest.fixture
def history(fixture_path: Path) -> MembershipHistory:
    """Loaded MembershipHistory from the synthetic fixture."""
    _clear_cache()  # don't share cached state across tests
    return load_membership_history("NIFTY50", path=fixture_path)


@pytest.fixture(autouse=True)
def _isolate_module_cache():
    """Module-level shortcut tests use the real default path -- clear
    before AND after every test so cross-test cache pollution can't
    yield false positives or false negatives.
    """
    _clear_cache()
    yield
    _clear_cache()


# ─────────────────────────────────────────────────────────────────
# Schema validation -- fail-loud paths at load time
# ─────────────────────────────────────────────────────────────────
class TestSchemaValidation:
    def test_valid_fixture_loads(self, fixture_path: Path):
        h = load_membership_history("NIFTY50", path=fixture_path)
        assert h.index == "NIFTY50"
        assert h.history_starts == date(2015, 1, 1)
        assert h.current_snapshot_date == date(2024, 12, 31)
        assert len(h.current_members) == 50
        assert len(h.events) == 4

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(MembershipDataError, match="not found"):
            load_membership_history("NIFTY50", path=tmp_path / "nope.json")

    def test_invalid_json_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("{this is not json}")
        with pytest.raises(MembershipDataError, match="not valid JSON"):
            load_membership_history("NIFTY50", path=bad)

    def test_root_must_be_dict(self, tmp_path: Path):
        bad = tmp_path / "list.json"
        bad.write_text("[1, 2, 3]")
        with pytest.raises(MembershipDataError, match="root is list"):
            load_membership_history("NIFTY50", path=bad)

    def test_unknown_index_raises(self, fixture_path: Path):
        with pytest.raises(MembershipDataError, match="not in"):
            load_membership_history("BANKNIFTY", path=fixture_path)

    def test_missing_required_field(self, tmp_path: Path):
        payload = {"NIFTY50": _build_fixture_payload()}
        del payload["NIFTY50"]["current_members"]
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(payload))
        with pytest.raises(MembershipDataError,
                           match="missing required field 'current_members'"):
            load_membership_history("NIFTY50", path=bad)

    def test_history_starts_after_snapshot(self, tmp_path: Path):
        payload = {"NIFTY50": _build_fixture_payload()}
        payload["NIFTY50"]["history_starts"] = "2099-01-01"
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(payload))
        with pytest.raises(MembershipDataError,
                           match="history_starts.*after.*current_snapshot_date"):
            load_membership_history("NIFTY50", path=bad)

    def test_wrong_member_count_raises(self, tmp_path: Path):
        payload = {"NIFTY50": _build_fixture_payload()}
        payload["NIFTY50"]["current_members"] = ["RELIANCE.NS"]  # only 1
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(payload))
        with pytest.raises(MembershipDataError, match="expected 50"):
            load_membership_history("NIFTY50", path=bad)

    def test_duplicate_current_members_raises(self, tmp_path: Path):
        payload = {"NIFTY50": _build_fixture_payload()}
        # Replace the last filler with a dupe of RELIANCE.
        payload["NIFTY50"]["current_members"][-1] = "RELIANCE.NS"
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(payload))
        with pytest.raises(MembershipDataError, match="duplicates"):
            load_membership_history("NIFTY50", path=bad)

    def test_non_nse_symbol_in_current_members(self, tmp_path: Path):
        payload = {"NIFTY50": _build_fixture_payload()}
        payload["NIFTY50"]["current_members"][0] = "AAPL"  # missing .NS
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(payload))
        with pytest.raises(MembershipDataError, match="non-NSE symbols"):
            load_membership_history("NIFTY50", path=bad)

    def test_non_nse_symbol_in_events(self, tmp_path: Path):
        payload = {"NIFTY50": _build_fixture_payload()}
        payload["NIFTY50"]["events"][0]["symbol"] = "AAPL"
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(payload))
        with pytest.raises(MembershipDataError, match="non-NSE symbols"):
            load_membership_history("NIFTY50", path=bad)

    def test_events_not_sorted_descending(self, tmp_path: Path):
        payload = {"NIFTY50": _build_fixture_payload()}
        # Reverse to ascending order.
        payload["NIFTY50"]["events"] = list(
            reversed(payload["NIFTY50"]["events"])
        )
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(payload))
        with pytest.raises(MembershipDataError,
                           match="not sorted descending"):
            load_membership_history("NIFTY50", path=bad)

    def test_event_before_history_starts(self, tmp_path: Path):
        payload = {"NIFTY50": _build_fixture_payload()}
        payload["NIFTY50"]["events"].append({
            "symbol": "OLDCO.NS", "action": "removed", "date": "2010-01-01",
        })
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(payload))
        with pytest.raises(MembershipDataError,
                           match="before history_starts"):
            load_membership_history("NIFTY50", path=bad)

    def test_event_after_snapshot(self, tmp_path: Path):
        payload = {"NIFTY50": _build_fixture_payload()}
        # Insert at the FRONT to keep events sorted descending.
        payload["NIFTY50"]["events"].insert(0, {
            "symbol": "RELIANCE.NS", "action": "added", "date": "2099-01-01",
        })
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(payload))
        with pytest.raises(MembershipDataError,
                           match="after current_snapshot_date"):
            load_membership_history("NIFTY50", path=bad)

    def test_invalid_event_action(self, tmp_path: Path):
        payload = {"NIFTY50": _build_fixture_payload()}
        payload["NIFTY50"]["events"][0]["action"] = "frobnicated"
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(payload))
        with pytest.raises(MembershipDataError, match="action must be"):
            load_membership_history("NIFTY50", path=bad)

    def test_event_must_be_dict(self, tmp_path: Path):
        payload = {"NIFTY50": _build_fixture_payload()}
        payload["NIFTY50"]["events"][0] = "not a dict"
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(payload))
        with pytest.raises(MembershipDataError, match="event entries must be dicts"):
            load_membership_history("NIFTY50", path=bad)

    def test_event_missing_required_key(self, tmp_path: Path):
        payload = {"NIFTY50": _build_fixture_payload()}
        del payload["NIFTY50"]["events"][0]["date"]
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(payload))
        with pytest.raises(MembershipDataError,
                           match="event missing required field 'date'"):
            load_membership_history("NIFTY50", path=bad)

    def test_current_members_must_be_list(self, tmp_path: Path):
        payload = {"NIFTY50": _build_fixture_payload()}
        payload["NIFTY50"]["current_members"] = {"oops": "not a list"}
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(payload))
        with pytest.raises(MembershipDataError,
                           match="current_members must be a list"):
            load_membership_history("NIFTY50", path=bad)

    def test_events_must_be_list(self, tmp_path: Path):
        payload = {"NIFTY50": _build_fixture_payload()}
        payload["NIFTY50"]["events"] = {"oops": "not a list"}
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(payload))
        with pytest.raises(MembershipDataError,
                           match="events must be a list"):
            load_membership_history("NIFTY50", path=bad)

    def test_invalid_iso_date(self, tmp_path: Path):
        payload = {"NIFTY50": _build_fixture_payload()}
        payload["NIFTY50"]["history_starts"] = "01/01/2015"
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(payload))
        with pytest.raises(MembershipDataError, match="invalid date"):
            load_membership_history("NIFTY50", path=bad)


# ─────────────────────────────────────────────────────────────────
# members_on() correctness
# ─────────────────────────────────────────────────────────────────
class TestMembersOn:
    def test_today_returns_current_members(
        self, history: MembershipHistory,
    ):
        result = history.members_on(history.current_snapshot_date)
        assert sorted(history.current_members) == result

    def test_pre_2019_includes_oldco_excludes_newco(
        self, history: MembershipHistory,
    ):
        d = date(2018, 9, 15)
        members = history.members_on(d)
        assert _OLDCO in members
        assert _NEWCO not in members

    def test_one_day_before_swap_pre_state(
        self, history: MembershipHistory,
    ):
        # 2019-04-07 = day before the OLDCO/NEWCO swap -> pre-state.
        members = history.members_on(date(2019, 4, 7))
        assert _OLDCO in members
        assert _NEWCO not in members

    def test_swap_day_inclusive_post_state(
        self, history: MembershipHistory,
    ):
        # The swap event date itself counts as POST: members_on(d)
        # returns membership "as of close of day d", and the event
        # took effect on that day. Backwards-walk's `evt.date <= d`
        # break clause encodes this.
        members = history.members_on(date(2019, 4, 8))
        assert _OLDCO not in members
        assert _NEWCO in members

    def test_returns_sorted_unique_50(
        self, history: MembershipHistory,
    ):
        for d in [date(2016, 1, 1), date(2020, 6, 15), date(2023, 12, 31)]:
            members = history.members_on(d)
            assert len(members) == NIFTY50_EXPECTED_COUNT
            assert len(set(members)) == NIFTY50_EXPECTED_COUNT
            assert members == sorted(members)

    def test_idempotent(self, history: MembershipHistory):
        d = date(2020, 6, 15)
        assert history.members_on(d) == history.members_on(d)

    def test_history_starts_inclusive(
        self, history: MembershipHistory,
    ):
        # Should NOT raise -- inclusive lower bound.
        members = history.members_on(history.history_starts)
        assert len(members) == NIFTY50_EXPECTED_COUNT

    def test_snapshot_date_inclusive(
        self, history: MembershipHistory,
    ):
        # Should NOT raise -- inclusive upper bound.
        members = history.members_on(history.current_snapshot_date)
        assert len(members) == NIFTY50_EXPECTED_COUNT

    def test_before_history_starts_raises(
        self, history: MembershipHistory,
    ):
        with pytest.raises(MembershipDataError, match="before history_starts"):
            history.members_on(date(2014, 12, 31))

    def test_after_snapshot_raises(
        self, history: MembershipHistory,
    ):
        with pytest.raises(MembershipDataError,
                           match="after current_snapshot_date"):
            history.members_on(date(2099, 1, 1))


# ─────────────────────────────────────────────────────────────────
# changes_in_range()
# ─────────────────────────────────────────────────────────────────
class TestChangesInRange:
    def test_full_range_returns_all_events(
        self, history: MembershipHistory,
    ):
        events = history.changes_in_range(
            history.history_starts, history.current_snapshot_date,
        )
        assert len(events) == 4

    def test_returns_chronological_order(
        self, history: MembershipHistory,
    ):
        events = history.changes_in_range(
            history.history_starts, history.current_snapshot_date,
        )
        dates = [e.date for e in events]
        assert dates == sorted(dates)  # ASCENDING (storage is desc)

    def test_filters_by_range(self, history: MembershipHistory):
        # Only the 2019 swap, not the 2023 swap.
        events = history.changes_in_range(
            date(2019, 1, 1), date(2019, 12, 31),
        )
        assert len(events) == 2  # added + removed for the same swap day
        assert all(e.date == date(2019, 4, 8) for e in events)

    def test_empty_window_returns_empty_list(
        self, history: MembershipHistory,
    ):
        events = history.changes_in_range(
            date(2020, 1, 1), date(2022, 12, 31),
        )
        assert events == []

    def test_inclusive_bounds(self, history: MembershipHistory):
        # Both endpoints exactly on event dates -> both included.
        events = history.changes_in_range(
            date(2019, 4, 8), date(2023, 6, 1),
        )
        assert len(events) == 4

    def test_start_after_end_raises(self, history: MembershipHistory):
        with pytest.raises(MembershipDataError, match=r"start.*>.*end"):
            history.changes_in_range(date(2020, 12, 31), date(2020, 1, 1))


# ─────────────────────────────────────────────────────────────────
# was_member()
# ─────────────────────────────────────────────────────────────────
class TestWasMember:
    def test_true_for_current(self, history: MembershipHistory):
        assert history.was_member(_RELIANCE, history.current_snapshot_date)

    def test_false_for_unknown_symbol(
        self, history: MembershipHistory,
    ):
        assert not history.was_member(
            "NEVERWAS.NS", history.current_snapshot_date,
        )

    def test_true_for_oldco_pre_2019(
        self, history: MembershipHistory,
    ):
        assert history.was_member(_OLDCO, date(2018, 6, 1))

    def test_false_for_oldco_post_2019(
        self, history: MembershipHistory,
    ):
        assert not history.was_member(_OLDCO, date(2020, 1, 1))

    def test_out_of_range_still_raises(
        self, history: MembershipHistory,
    ):
        with pytest.raises(MembershipDataError):
            history.was_member(_RELIANCE, date(2099, 1, 1))


# ─────────────────────────────────────────────────────────────────
# Module-level shortcuts (test that they hit the right cache key)
# ─────────────────────────────────────────────────────────────────
class TestModuleShortcuts:
    def test_members_on_shortcut(
        self, fixture_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        # Point MEMBERSHIP_FILE at our fixture so the shortcut's
        # default path resolves correctly.
        monkeypatch.setattr(
            "price_predictor.kb.membership.MEMBERSHIP_FILE", fixture_path,
        )
        result = members_on(date(2024, 12, 31))
        assert len(result) == NIFTY50_EXPECTED_COUNT

    def test_changes_in_range_shortcut(
        self, fixture_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(
            "price_predictor.kb.membership.MEMBERSHIP_FILE", fixture_path,
        )
        events = changes_in_range(date(2019, 1, 1), date(2019, 12, 31))
        assert len(events) == 2

    def test_was_member_shortcut(
        self, fixture_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(
            "price_predictor.kb.membership.MEMBERSHIP_FILE", fixture_path,
        )
        assert was_member(_RELIANCE, date(2024, 12, 31))
        assert not was_member(_OLDCO, date(2024, 12, 31))


# ─────────────────────────────────────────────────────────────────
# Caching behavior
# ─────────────────────────────────────────────────────────────────
class TestCaching:
    def test_load_caches_by_index_and_path(
        self, fixture_path: Path,
    ):
        h1 = load_membership_history("NIFTY50", path=fixture_path)
        h2 = load_membership_history("NIFTY50", path=fixture_path)
        # Same object -- the lru_cache returned the cached instance.
        assert h1 is h2

    def test_clear_cache_forces_reload(
        self, fixture_path: Path,
    ):
        h1 = load_membership_history("NIFTY50", path=fixture_path)
        _clear_cache()
        h2 = load_membership_history("NIFTY50", path=fixture_path)
        # Different object -- cache was cleared.
        assert h1 is not h2
        # But identical content.
        assert h1.current_members == h2.current_members


# ─────────────────────────────────────────────────────────────────
# Internal helpers (worth testing because parse errors funnel through them)
# ─────────────────────────────────────────────────────────────────
class TestInternalHelpers:
    def test_parse_iso_date_happy(self):
        assert _parse_iso_date("2024-06-14", ctx="x") == date(2024, 6, 14)

    def test_parse_iso_date_rejects_non_string(self):
        with pytest.raises(MembershipDataError, match="expected ISO date string"):
            _parse_iso_date(20240614, ctx="x")

    def test_parse_event_minimal(self):
        evt = _parse_event(
            {"symbol": "RELIANCE.NS", "action": "added", "date": "2020-01-01"},
            index="NIFTY50",
        )
        assert evt.symbol == "RELIANCE.NS"
        assert evt.action == "added"
        assert evt.date == date(2020, 1, 1)
        assert evt.reason is None

    def test_parse_event_with_reason(self):
        evt = _parse_event(
            {"symbol": "X.NS", "action": "removed", "date": "2020-01-01",
             "reason": "marketcap"},
            index="NIFTY50",
        )
        assert evt.reason == "marketcap"

    def test_build_from_payload_round_trip(self):
        payload = _build_fixture_payload()
        h = _build_from_payload("NIFTY50", payload)
        assert h.source_url == "https://example.com/fixture"
        # Tuples not lists -- frozen dataclass guarantee.
        assert isinstance(h.current_members, tuple)
        assert isinstance(h.events, tuple)
