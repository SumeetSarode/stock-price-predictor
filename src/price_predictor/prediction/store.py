"""PredictionStore - JSON file persistence for predictions.

WHY THIS EXISTS
===============
Predictions are evidence about a moment in time. To answer:
  - "How accurate were last quarter's bullish calls on RELIANCE?"
  - "What's the hit rate on short-horizon predictions for NIFTY50?"
  - "Show me everything we predicted yesterday."

...we need to STORE predictions and read them back. This is the
foundation for Step 3.5 (calibration metrics).

DESIGN
======
1. ONE FILE per prediction, not a SQL table or single big JSON.
   Why? Predictions are immutable (frozen Pydantic models), so a
   write-only append-store fits naturally. One file per prediction:
     - Atomic writes (rename trick)
     - Easy to grep / cat / diff via shell
     - Easy to gzip old days
     - No DB to install or break
     - Trivially parallel-safe (different files = no contention)

2. Layout: {root}/{YYYY-MM-DD}/{TICKER}_{HHMMSS}_{horizon}.json
   - Day directory = easy archival ('tar czf 2026-04.tar.gz 2026-04-*')
   - HHMMSS = preserves intra-day ordering, supports multiple
     predictions per ticker per day (different horizons or re-runs)
   - Horizon in filename = quick filter without opening files

3. Filename sanitization: ticker may contain '.' (RELIANCE.NS), but
   never '/' or '..' (path traversal). We strip dangerous chars and
   uppercase for consistency.

4. Save is atomic: write to .tmp, then os.replace() onto the final
   path. POSIX guarantees rename atomicity, so we never see a
   half-written file even on crash.
"""
from __future__ import annotations

import os
import re
import tempfile
from datetime import date, datetime
from pathlib import Path

from loguru import logger

from price_predictor.prediction.schema import Prediction


# Filename = "{TICKER}_{HHMMSS}_{horizon}.json"
# Examples:
#   RELIANCE.NS_103045_short.json
#   AAPL_141200_medium.json
_FILENAME_PATTERN = re.compile(
    r"^(?P<ticker>[A-Z0-9.\-]+)_(?P<time>\d{6})_(?P<horizon>\w+)\.json$"
)

# Strip anything that's not alphanum, dot, or hyphen. No '/', '\', '..'.
# Pre-compiled for speed; called on every save.
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Z0-9.\-]")


class PredictionStoreError(Exception):
    """Raised when persistence operations fail in unrecoverable ways.

    Examples: corrupted JSON file on disk, unwriteable root directory,
    schema mismatch on load.
    """


def _safe_ticker_for_filename(ticker: str) -> str:
    """Sanitize ticker for safe use in a filename.

    Uppercases, then strips anything that's not [A-Z0-9.-]. Empty
    result raises - the caller passed garbage.
    """
    cleaned = _UNSAFE_FILENAME_CHARS.sub("", ticker.upper())
    if not cleaned:
        raise PredictionStoreError(
            f"Ticker {ticker!r} sanitizes to empty - cannot persist."
        )
    return cleaned


class PredictionStore:
    """Filesystem-backed store for Predictions.

    Thread-safe for different predictions; not safe for concurrent
    writes of the SAME prediction (last-writer-wins). In practice this
    is fine - predictions are timestamped to the second, so a true
    collision means you ran predict() twice on the same ticker within
    one second. Don't do that.
    """

    def __init__(self, root: Path | str):
        """Args:
            root: Directory to store predictions under. Created if it
                doesn't exist (mkdir -p semantics). Must be writable.
        """
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        logger.debug(f"PredictionStore rooted at {self.root}")

    # ─────────────────────────────────────────────────────────
    # Path computation
    # ─────────────────────────────────────────────────────────
    def _day_dir(self, as_of: datetime) -> Path:
        """Daily subdirectory for an as_of timestamp (date-only)."""
        return self.root / as_of.date().isoformat()

    def _filename(self, prediction: Prediction) -> str:
        """Build filename for a prediction."""
        ticker = _safe_ticker_for_filename(prediction.ticker)
        time_str = prediction.as_of.strftime("%H%M%S")
        horizon = prediction.horizon.value
        return f"{ticker}_{time_str}_{horizon}.json"

    def path_for(self, prediction: Prediction) -> Path:
        """Full path where this prediction would be stored.

        Useful for callers who want to check existence or print the
        location before/after save.
        """
        return self._day_dir(prediction.as_of) / self._filename(prediction)

    # ─────────────────────────────────────────────────────────
    # Write
    # ─────────────────────────────────────────────────────────
    def save(self, prediction: Prediction) -> Path:
        """Persist a prediction. Atomic write.

        Returns:
            Path the prediction was saved to.

        Raises:
            PredictionStoreError: write failed (permissions, full disk).
        """
        target = self.path_for(prediction)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = prediction.model_dump_json(indent=2)

        # Atomic write: tempfile in same dir (so os.replace is on same
        # filesystem), then rename onto target. POSIX guarantees
        # rename atomicity. Same-dir is critical - cross-FS rename
        # falls back to copy+delete which is NOT atomic.
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.stem}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp.write(payload)
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, target)
        except OSError as e:
            raise PredictionStoreError(
                f"Failed to write prediction to {target}: {e}"
            ) from e

        logger.debug(f"saved prediction: {target}")
        return target

    # ─────────────────────────────────────────────────────────
    # Read
    # ─────────────────────────────────────────────────────────
    def load(self, path: Path) -> Prediction:
        """Load a single prediction from a file path.

        Raises:
            PredictionStoreError: file missing, unreadable, or fails
                schema validation (corruption, schema drift).
        """
        try:
            payload = path.read_text(encoding="utf-8")
        except OSError as e:
            raise PredictionStoreError(
                f"Cannot read prediction file {path}: {e}"
            ) from e
        try:
            return Prediction.model_validate_json(payload)
        except Exception as e:  # pydantic ValidationError or JSONDecodeError
            raise PredictionStoreError(
                f"Prediction at {path} failed validation: {e}"
            ) from e

    def list_for_ticker(self, ticker: str) -> list[Prediction]:
        """All stored predictions for one ticker, oldest -> newest.

        Walks the entire root - O(N) in total stored predictions.
        Fine for v1 (we'll have hundreds, not millions); switch to an
        index file when we hit 100k.
        """
        wanted = _safe_ticker_for_filename(ticker)
        paths: list[Path] = []
        # Day dirs sort lexically = chronologically (ISO format).
        for day_dir in sorted(self.root.iterdir()):
            if not day_dir.is_dir():
                continue
            for f in sorted(day_dir.iterdir()):
                m = _FILENAME_PATTERN.match(f.name)
                if m and m.group("ticker") == wanted:
                    paths.append(f)
        return [self.load(p) for p in paths]

    def list_in_date_range(
        self, start: date, end: date,
    ) -> list[Prediction]:
        """All predictions saved on dates in [start, end] inclusive.

        Walks day-dirs whose ISO names fall in range. Sorted by day,
        then by filename (= time of day).
        """
        if start > end:
            raise ValueError(f"start ({start}) must be <= end ({end})")
        out: list[Prediction] = []
        for day_dir in sorted(self.root.iterdir()):
            if not day_dir.is_dir():
                continue
            try:
                day = date.fromisoformat(day_dir.name)
            except ValueError:
                # Not a YYYY-MM-DD dir (junk/foreign content) - skip
                continue
            if not (start <= day <= end):
                continue
            for f in sorted(day_dir.iterdir()):
                if _FILENAME_PATTERN.match(f.name):
                    out.append(self.load(f))
        return out

    def count(self) -> int:
        """Total number of stored predictions across all days.

        Cheap-ish: walks dirs but doesn't open files.
        """
        n = 0
        for day_dir in self.root.iterdir():
            if not day_dir.is_dir():
                continue
            for f in day_dir.iterdir():
                if _FILENAME_PATTERN.match(f.name):
                    n += 1
        return n
