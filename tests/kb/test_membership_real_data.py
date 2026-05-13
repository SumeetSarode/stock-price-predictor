"""Golden-path tests pinned to the REAL data/kb/index_membership.json.

Why these tests are separate from test_membership.py
====================================================
test_membership.py uses a synthetic fixture so its assertions are
independent of whatever Wikipedia happens to say today. THIS file
exercises the actual committed JSON, so it catches:

- Future bootstrap re-runs that quietly mangle the data (e.g. a
  symbol-resolution bug in HISTORICAL_NAME_TO_TICKER that flips two
  tickers).
- Membership-module changes that break the loader against real shapes
  (extra/missing fields, encoding quirks).
- Wikipedia layout changes that we recover from incorrectly.

Each probe corresponds to a real, well-known event with a verifiable
date. Don't add probes that are only meaningful relative to your
mental model -- if a future maintainer can't tell whether a probe is
correct without a financial-history textbook, it's not a useful test.

What we deliberately do NOT test
================================
- The exact symbol list on a specific date (would change every quarter
  after re-running the bootstrap script).
- Symbols added in 2024+ (recent enough that the 'historical' overrides
  may shift as Wikipedia updates).
- The reason field text (Wikipedia editorializes; brittle).
"""
from __future__ import annotations

from datetime import date

import pytest

from price_predictor.kb.membership import (
    NIFTY50_EXPECTED_COUNT,
    MembershipDataError,
    _clear_cache,
    load_membership_history,
    members_on,
    was_member,
)


@pytest.fixture(autouse=True)
def _isolate_cache():
    """Drop the cached singleton before/after every test so a
    leftover synthetic fixture from another file can't leak in."""
    _clear_cache()
    yield
    _clear_cache()


# ─────────────────────────────────────────────────────────────────
# Schema-level smoke (real file loads at all)
# ─────────────────────────────────────────────────────────────────
class TestRealFileSmoke:
    def test_loads_without_error(self):
        h = load_membership_history()
        assert h.index == "NIFTY50"

    def test_history_starts_2015(self):
        # We deliberately chose 2015 as our cutoff; if a future bootstrap
        # silently changes it, this test will scream and we'll know.
        h = load_membership_history()
        assert h.history_starts == date(2015, 1, 1)

    def test_current_members_count(self):
        h = load_membership_history()
        assert len(h.current_members) == NIFTY50_EXPECTED_COUNT

    def test_has_meaningful_event_count(self):
        # ~4-8 swaps/year * 10 years * 2 events/swap = 80-160 events.
        # Lower bound 30 (very conservative) catches catastrophic
        # filtering bugs without being date-fragile.
        h = load_membership_history()
        assert len(h.events) >= 30, (
            f"only {len(h.events)} events -- bootstrap likely dropped "
            f"events silently. Re-run scripts/bootstrap_membership_history.py "
            f"and inspect."
        )


# ─────────────────────────────────────────────────────────────────
# Specific historical events -- each one corresponds to a verifiable
# real-world swap. If any of these fail, either the bootstrap is
# broken OR Wikipedia revised the historical record.
# ─────────────────────────────────────────────────────────────────
class TestKnownHistoricalEvents:
    def test_hdfc_merger_into_hdfc_bank_2023(self):
        """HDFC Ltd merged into HDFC Bank on 2023-07-13.

        The standalone HDFC.NS ticker was removed from NIFTY 50 around
        the merger. Pre-merger HDFC Ltd was a NIFTY 50 staple for
        decades; post-merger HDFC.NS is no longer in the index.
        """
        assert was_member("HDFC.NS", date(2023, 6, 1)) is True
        assert was_member("HDFC.NS", date(2023, 8, 1)) is False

    def test_indiabulls_housing_dropped_2019(self):
        """IBULHSGFIN was removed 2019-09-27 amid the NBFC crisis.

        Stock collapsed ~80% over the year; index reconstitution
        catches up and drops it. Was a member earlier in the year.
        """
        assert was_member("IBULHSGFIN.NS", date(2019, 8, 1)) is True
        assert was_member("IBULHSGFIN.NS", date(2019, 12, 1)) is False

    def test_britannia_added_2019(self):
        """BRITANNIA.NS was added on 2019-03-29 (replaced HINDPETRO)."""
        assert was_member("BRITANNIA.NS", date(2019, 1, 15)) is False
        assert was_member("BRITANNIA.NS", date(2019, 4, 15)) is True

    def test_yes_bank_in_index_pre_collapse(self):
        """YESBANK was in NIFTY 50 in early 2015 (added 2015-03-27).

        If history starts at 2015-01-01, the very first event in our
        log should be the 2015-03-27 swap that added YESBANK.NS and
        IDEA.NS. Probing pre- and post-March-27 catches off-by-one
        errors in the boundary handling.
        """
        # Just before 2015-03-27 -- not yet a member.
        assert was_member("YESBANK.NS", date(2015, 3, 1)) is False
        # Just after -- now a member.
        assert was_member("YESBANK.NS", date(2015, 4, 1)) is True


# ─────────────────────────────────────────────────────────────────
# Property: the count is always exactly 50 across ALL of history
# ─────────────────────────────────────────────────────────────────
class TestCountInvariant:
    @pytest.mark.parametrize("d", [
        # Year-by-year sweep across the whole window. If the events
        # log has a paired-add/remove bug at any swap, one of these
        # will land on a date where the count drifts off 50.
        date(2015, 6, 1),
        date(2016, 6, 1),
        date(2017, 6, 1),
        date(2018, 6, 1),
        date(2019, 6, 1),
        date(2020, 6, 1),
        date(2021, 6, 1),
        date(2022, 6, 1),
        date(2023, 6, 1),
        date(2024, 6, 1),
        date(2025, 6, 1),
    ])
    def test_exactly_50_members(self, d: date):
        h = load_membership_history()
        # Skip dates outside the window if today is mid-2026 etc.
        if d > h.current_snapshot_date:
            pytest.skip(f"{d} > snapshot_date {h.current_snapshot_date}")
        members = members_on(d)
        assert len(members) == NIFTY50_EXPECTED_COUNT, (
            f"on {d} got {len(members)} members; "
            f"events log likely has an unpaired add/remove"
        )

    def test_returns_unique_symbols(self):
        """Sets and sorted lists of the same data should match -- no
        duplicates can sneak through the backwards-walk."""
        h = load_membership_history()
        for d in [
            date(2015, 6, 1), date(2018, 9, 15),
            date(2021, 1, 1), h.current_snapshot_date,
        ]:
            m = members_on(d)
            assert len(m) == len(set(m))


# ─────────────────────────────────────────────────────────────────
# Range queries on real data
# ─────────────────────────────────────────────────────────────────
class TestRealRangeQueries:
    def test_full_range_returns_all_events(self):
        h = load_membership_history()
        events = h.changes_in_range(
            h.history_starts, h.current_snapshot_date,
        )
        assert len(events) == len(h.events)

    def test_2019_window_finds_known_swap(self):
        """2019 had the HINDPETRO->BRITANNIA and ANDHRABANK->NESTLE
        swaps. At MINIMUM we should see >= 1 event in March 2019."""
        h = load_membership_history()
        events = h.changes_in_range(date(2019, 3, 1), date(2019, 4, 30))
        assert len(events) >= 1
        # Both BRITANNIA add and HINDPETRO remove are on 2019-03-29.
        symbols = {e.symbol for e in events}
        assert "BRITANNIA.NS" in symbols

    def test_pre_history_query_raises(self):
        with pytest.raises(MembershipDataError, match="before history_starts"):
            members_on(date(2010, 1, 1))


# ─────────────────────────────────────────────────────────────────
# Cross-check: every event symbol either currently exists or is a
# known delisted/merged historical entry. Catches silent data drift
# where the bootstrap resolves a name to a wrong/typo'd ticker.
# ─────────────────────────────────────────────────────────────────
class TestSymbolHygiene:
    def test_event_symbols_match_ticker_format(self):
        h = load_membership_history()
        for evt in h.events:
            assert evt.symbol.endswith(".NS"), evt
            assert evt.symbol.replace(".NS", "").isupper() or any(
                c in evt.symbol for c in "&-"
            ), f"unexpected symbol shape: {evt.symbol}"

    def test_no_event_symbol_is_empty(self):
        h = load_membership_history()
        for evt in h.events:
            assert evt.symbol.strip() != ".NS"
            assert len(evt.symbol) > 3  # at minimum X.NS

    def test_each_swap_day_is_balanced(self):
        """For every date that has events, count of 'added' should
        equal count of 'removed'. Catches the case where a name failed
        to resolve and got dropped from one side of the swap.
        """
        h = load_membership_history()
        by_date: dict[date, dict[str, int]] = {}
        for evt in h.events:
            counts = by_date.setdefault(evt.date, {"added": 0, "removed": 0})
            counts[evt.action] += 1
        for d, counts in by_date.items():
            assert counts["added"] == counts["removed"], (
                f"unbalanced events on {d}: "
                f"{counts['added']} added, {counts['removed']} removed"
            )
