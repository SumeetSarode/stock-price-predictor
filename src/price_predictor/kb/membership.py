"""NIFTY 50 (and friends) membership history -- survivorship-bias defense.

Why this exists
===============
Today's NIFTY 50 is not the NIFTY 50 from 2018. About 4-8 swaps happen
each year via NSE's semi-annual reconstitution. A backtest run against
"current NIFTY 50" silently excludes every company that got DROPPED --
usually because it underperformed -- so results are systematically
inflated by survivorship bias.

This module exposes the history needed to ask: "which 50 tickers were
in NIFTY 50 on date X?" without leaking any future information.

How it works
============
``data/kb/index_membership.json`` stores:

1. The current 50 members (the "anchor" -- known exactly from
   Wikipedia's constituents table).
2. An event log (Wikipedia's "List of replacements since 2005" table,
   filtered to events on/after ``history_starts``).

To answer ``members_on(d)``, we walk the events backwards in time
starting from the anchor, undoing each event that happened AFTER d:
- An ``added`` event after d -> the symbol wasn't in the index at d
- A ``removed`` event after d -> the symbol WAS in the index at d

This is O(events_after_d) per call and is bounded by the gap between
``d`` and ``current_snapshot_date`` (typically <30 events for a recent
backtest date).

Why backwards-walk (not period table)
=====================================
- We *know* today's 50 exactly. Walking backwards, every error is
  bounded by the gap between now and d.
- A pre-computed (symbol, in_date, out_date) table would force us to
  bootstrap a "what was in the index when history starts" set, which
  introduces another error source.
- Refresh story is simple: re-run ``scripts/bootstrap_membership_history``
  quarterly to keep the anchor current; nothing else changes.

Data discipline
===============
Schema invariants are enforced at LOAD time -- malformed JSON fails
loud with ``MembershipDataError``. Lookups outside the supported date
range fail loud too. There is NO silent guessing for dates we don't
have data for; survivorship-bias defense that quietly returns wrong
answers is worse than no defense at all.
"""
from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Final, Literal

from price_predictor.config.settings import settings

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
MEMBERSHIP_FILE: Final[Path] = settings.kb_dir / "index_membership.json"

# NSE tickers we accept: caps + digits + & + - then mandatory .NS suffix.
# & shows up in real symbols (M&M.NS = Mahindra & Mahindra). - is rare
# but appears in some merged-entity symbols. No lower-case -- NSE's
# convention.
_TICKER_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Z0-9&\-]+\.NS$")

# Default index code. Hard-coded as a constant (not a magic string at
# call sites) so renames are one-line changes.
DEFAULT_INDEX: Final[str] = "NIFTY50"

# Expected member count for NIFTY 50. If the JSON ever drifts from
# this, something is wrong and we want a loud failure.
NIFTY50_EXPECTED_COUNT: Final[int] = 50

EventAction = Literal["added", "removed"]


# ─────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────
class MembershipDataError(ValueError):
    """Raised when the membership JSON is malformed or queried out of range.

    Subclass of ValueError so callers can broadly ``except ValueError`` --
    a malformed kb file is a programming/data problem, not a retryable
    condition.
    """


# ─────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class IndexEvent:
    """One add or remove event in an index's history.

    Frozen + hashable so events can live in tuples / sets and be
    cheaply compared in tests.

    ``reason`` is optional because future indices may not provide one,
    and tests that build synthetic events shouldn't have to invent
    plausible strings.
    """
    symbol: str
    action: EventAction
    date: date
    reason: str | None = None


@dataclass(frozen=True)
class MembershipHistory:
    """Loaded + validated membership history for ONE index.

    Constructed only via ``load_membership_history`` -- direct
    instantiation is allowed for tests but skips the file I/O. All
    invariants are checked in ``__post_init__`` so a hand-built
    instance is held to the same standard as a loaded one.

    Attributes are tuples (not lists) so this dataclass is hashable
    and a frozen guarantee actually means something at runtime.
    """
    index: str
    history_starts: date
    current_snapshot_date: date
    current_members: tuple[str, ...]
    events: tuple[IndexEvent, ...]  # sorted DESCENDING by date
    expected_count: int = NIFTY50_EXPECTED_COUNT
    source_url: str | None = None

    # ── Invariants enforced once, at construction ───────────────
    def __post_init__(self) -> None:
        # 1. Date range sanity.
        if self.history_starts > self.current_snapshot_date:
            raise MembershipDataError(
                f"{self.index}: history_starts ({self.history_starts}) "
                f"is after current_snapshot_date ({self.current_snapshot_date})"
            )

        # 2. Current-member count.
        if len(self.current_members) != self.expected_count:
            raise MembershipDataError(
                f"{self.index}: current_members has "
                f"{len(self.current_members)} entries; "
                f"expected {self.expected_count}"
            )

        # 3. No duplicate current members.
        if len(set(self.current_members)) != len(self.current_members):
            dupes = sorted(
                {s for s in self.current_members
                 if self.current_members.count(s) > 1}
            )
            raise MembershipDataError(
                f"{self.index}: current_members has duplicates: {dupes}"
            )

        # 4. Symbol format.
        bad_now = [s for s in self.current_members if not _TICKER_RE.match(s)]
        if bad_now:
            raise MembershipDataError(
                f"{self.index}: current_members contains non-NSE symbols: "
                f"{bad_now[:5]}"
            )
        bad_evt = [
            (e.symbol, e.date) for e in self.events
            if not _TICKER_RE.match(e.symbol)
        ]
        if bad_evt:
            raise MembershipDataError(
                f"{self.index}: events contain non-NSE symbols: {bad_evt[:5]}"
            )

        # 5. Events sorted descending by date.
        for prev, curr in zip(self.events, self.events[1:]):
            if prev.date < curr.date:
                raise MembershipDataError(
                    f"{self.index}: events not sorted descending by date "
                    f"(saw {prev.date} before {curr.date})"
                )

        # 6. All events fall in [history_starts, current_snapshot_date].
        for e in self.events:
            if e.date < self.history_starts:
                raise MembershipDataError(
                    f"{self.index}: event for {e.symbol} on {e.date} "
                    f"is before history_starts ({self.history_starts})"
                )
            if e.date > self.current_snapshot_date:
                raise MembershipDataError(
                    f"{self.index}: event for {e.symbol} on {e.date} "
                    f"is after current_snapshot_date "
                    f"({self.current_snapshot_date})"
                )

    # ── Lookups ─────────────────────────────────────────────────
    def members_on(self, d: date) -> list[str]:
        """Return symbols that were in the index on date ``d``.

        Returns a list sorted alphabetically (not insertion order) so
        call-site comparisons are stable.

        Raises:
            MembershipDataError: ``d`` is outside the supported range.
                Range is inclusive on both ends.
        """
        self._assert_in_range(d)

        # Start from the known anchor and undo every event that
        # happened AFTER d. Events are sorted desc, so we can break
        # as soon as we see one whose date <= d.
        members: set[str] = set(self.current_members)
        for evt in self.events:
            if evt.date <= d:
                break
            # evt.date > d -- event happened AFTER d, undo it.
            if evt.action == "added":
                # It got added after d, so it WASN'T in the index at d.
                members.discard(evt.symbol)
            elif evt.action == "removed":
                # It got removed after d, so it WAS in the index at d.
                members.add(evt.symbol)
            # else: the EventAction Literal makes this unreachable; if
            # someone hand-builds an IndexEvent with a bad action, the
            # JSON loader will catch it.
        return sorted(members)

    def changes_in_range(
        self, start: date, end: date,
    ) -> list[IndexEvent]:
        """All events with ``start <= event.date <= end`` (inclusive).

        Useful for detecting "the index changed mid-backtest, slice
        carefully" before kicking off a long run.

        Returned in chronological order (oldest first) for readability;
        the storage order is descending but a consumer of "what
        happened in this window" expects forward time.

        Raises:
            MembershipDataError: ``start > end``.
        """
        if start > end:
            raise MembershipDataError(
                f"{self.index}: changes_in_range start ({start}) > end ({end})"
            )
        in_range = [e for e in self.events if start <= e.date <= end]
        return sorted(in_range, key=lambda e: (e.date, e.symbol, e.action))

    def was_member(self, symbol: str, d: date) -> bool:
        """Convenience predicate: was ``symbol`` in the index on ``d``?

        Returns False (not raises) for unknown symbols -- this is a
        boolean predicate, not a lookup. Out-of-range ``d`` still
        raises via ``members_on``.
        """
        return symbol in self.members_on(d)

    # ── Internals ───────────────────────────────────────────────
    def _assert_in_range(self, d: date) -> None:
        if d < self.history_starts:
            raise MembershipDataError(
                f"{self.index}: date {d} is before history_starts "
                f"({self.history_starts}); we don't have data for that date"
            )
        if d > self.current_snapshot_date:
            raise MembershipDataError(
                f"{self.index}: date {d} is after current_snapshot_date "
                f"({self.current_snapshot_date}); refresh "
                f"data/kb/index_membership.json by re-running "
                f"scripts/bootstrap_membership_history.py"
            )


# ─────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=8)
def load_membership_history(
    index: str = DEFAULT_INDEX,
    *,
    path: Path | None = None,
) -> MembershipHistory:
    """Load + validate the membership JSON for one index. Cached.

    Args:
        index: Index code (key in the JSON top-level dict). Default
            NIFTY50; future indices reuse the same loader.
        path: Override the JSON file path. Used by tests; production
            callers should leave None so the global cache hits.

    Returns:
        Validated ``MembershipHistory``. Subsequent calls with the
        same args return the cached instance.

    Raises:
        MembershipDataError: file missing, malformed, or any schema
            invariant violated.
    """
    file_path = path or MEMBERSHIP_FILE
    if not file_path.exists():
        raise MembershipDataError(
            f"membership file not found: {file_path} -- run "
            f"scripts/bootstrap_membership_history.py to create it"
        )

    try:
        raw = json.loads(file_path.read_text())
    except json.JSONDecodeError as exc:
        raise MembershipDataError(
            f"membership file is not valid JSON ({file_path}): {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise MembershipDataError(
            f"membership file root is {type(raw).__name__}; expected dict"
        )
    if index not in raw:
        raise MembershipDataError(
            f"index '{index}' not in {file_path}; "
            f"available: {sorted(raw.keys())}"
        )

    return _build_from_payload(index, raw[index])


def _clear_cache() -> None:
    """Drop the module-level cache. Used by tests that swap fixtures."""
    load_membership_history.cache_clear()


def _build_from_payload(index: str, payload: dict) -> MembershipHistory:
    """Convert the raw JSON dict for one index -> validated MembershipHistory.

    Centralizing the parse here means the dataclass __post_init__ only
    deals with already-typed inputs (date, tuple) -- the JSON-specific
    error messages live in one place.
    """
    # Required string fields.
    for key in ("history_starts", "current_snapshot_date",
                "current_members", "events"):
        if key not in payload:
            raise MembershipDataError(
                f"{index}: missing required field '{key}'"
            )

    history_starts = _parse_iso_date(
        payload["history_starts"],
        ctx=f"{index}.history_starts",
    )
    snapshot_date = _parse_iso_date(
        payload["current_snapshot_date"],
        ctx=f"{index}.current_snapshot_date",
    )

    members_raw = payload["current_members"]
    if not isinstance(members_raw, list):
        raise MembershipDataError(
            f"{index}: current_members must be a list, got "
            f"{type(members_raw).__name__}"
        )
    current_members = tuple(str(m) for m in members_raw)

    events_raw = payload["events"]
    if not isinstance(events_raw, list):
        raise MembershipDataError(
            f"{index}: events must be a list, got {type(events_raw).__name__}"
        )
    events = tuple(_parse_event(e, index=index) for e in events_raw)

    return MembershipHistory(
        index=index,
        history_starts=history_starts,
        current_snapshot_date=snapshot_date,
        current_members=current_members,
        events=events,
        source_url=payload.get("source_url"),
    )


def _parse_iso_date(s: object, *, ctx: str) -> date:
    """Parse 'YYYY-MM-DD' or raise MembershipDataError with a context tag."""
    if not isinstance(s, str):
        raise MembershipDataError(
            f"{ctx}: expected ISO date string, got {type(s).__name__}"
        )
    try:
        return date.fromisoformat(s)
    except ValueError as exc:
        raise MembershipDataError(
            f"{ctx}: invalid date '{s}' (expected YYYY-MM-DD): {exc}"
        ) from exc


_VALID_ACTIONS: Final[frozenset[str]] = frozenset({"added", "removed"})


def _parse_event(raw: object, *, index: str) -> IndexEvent:
    """Convert one event dict from JSON -> IndexEvent. Strict validation."""
    if not isinstance(raw, dict):
        raise MembershipDataError(
            f"{index}: event entries must be dicts, got "
            f"{type(raw).__name__}: {raw!r}"
        )
    for key in ("symbol", "action", "date"):
        if key not in raw:
            raise MembershipDataError(
                f"{index}: event missing required field '{key}': {raw!r}"
            )
    action = raw["action"]
    if action not in _VALID_ACTIONS:
        raise MembershipDataError(
            f"{index}: event action must be 'added' or 'removed', "
            f"got {action!r}"
        )
    return IndexEvent(
        symbol=str(raw["symbol"]),
        action=action,  # type: ignore[arg-type]  -- validated above
        date=_parse_iso_date(raw["date"], ctx=f"{index}.event.date"),
        reason=raw.get("reason"),
    )


# ─────────────────────────────────────────────────────────────────
# Module-level shortcuts
# ─────────────────────────────────────────────────────────────────
def members_on(d: date, *, index: str = DEFAULT_INDEX) -> list[str]:
    """Top-level shortcut: ``load_membership_history(index).members_on(d)``."""
    return load_membership_history(index).members_on(d)


def changes_in_range(
    start: date, end: date, *, index: str = DEFAULT_INDEX,
) -> list[IndexEvent]:
    """Top-level shortcut: ``load_membership_history(index).changes_in_range(start, end)``."""
    return load_membership_history(index).changes_in_range(start, end)


def was_member(
    symbol: str, d: date, *, index: str = DEFAULT_INDEX,
) -> bool:
    """Top-level shortcut: ``load_membership_history(index).was_member(symbol, d)``."""
    return load_membership_history(index).was_member(symbol, d)
