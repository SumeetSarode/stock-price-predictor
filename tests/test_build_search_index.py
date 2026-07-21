"""Tests for the --backfill-sectors CSV wrapper in build_search_index.py.

The script isn't a package, so we import it by path. yfinance is fully
mocked via monkeypatching the backfill_sectors call.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "build_search_index.py"


@pytest.fixture
def bsi():
    """Import the build_search_index script as a module."""
    spec = importlib.util.spec_from_file_location("build_search_index", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_csv(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    lines = ["ticker,name,sector,nifty50"]
    lines += [",".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestBackfillCsvSectors:
    def test_overwrites_sectors_from_cache(self, bsi, tmp_path, monkeypatch):
        csv_path = tmp_path / "nifty500.csv"
        _write_csv(csv_path, [
            ("RELIANCE.NS", "Reliance", "NSE Listed", "false"),
            ("INFY.NS", "Infosys", "Information Technology", "true"),
        ])
        monkeypatch.setattr(bsi, "OUTPUT_CSV", csv_path)
        monkeypatch.setattr(bsi, "SECTOR_CACHE", tmp_path / "cache.json")
        # Fake the resolver: return real yfinance-style sectors.
        monkeypatch.setattr(bsi, "backfill_sectors", lambda tickers, **kw: {
            "RELIANCE.NS": "Energy", "INFY.NS": "Technology",
        })

        bsi.backfill_csv_sectors()

        out = csv_path.read_text(encoding="utf-8").splitlines()
        assert out[0] == "ticker,name,sector,nifty50"
        assert "RELIANCE.NS,Reliance,Energy,false" in out
        assert "INFY.NS,Infosys,Technology,true" in out

    def test_missing_ticker_in_cache_keeps_existing(self, bsi, tmp_path, monkeypatch):
        csv_path = tmp_path / "nifty500.csv"
        _write_csv(csv_path, [
            ("A.NS", "A Ltd", "Information Technology", "false"),
        ])
        monkeypatch.setattr(bsi, "OUTPUT_CSV", csv_path)
        monkeypatch.setattr(bsi, "SECTOR_CACHE", tmp_path / "c.json")
        # Cache came back empty (all rate-limited) -> keep existing sector.
        monkeypatch.setattr(bsi, "backfill_sectors", lambda tickers, **kw: {})

        bsi.backfill_csv_sectors()

        out = csv_path.read_text(encoding="utf-8")
        assert "A.NS,A Ltd,Information Technology,false" in out

    def test_preserves_row_count_and_names(self, bsi, tmp_path, monkeypatch):
        csv_path = tmp_path / "nifty500.csv"
        rows = [(f"T{i}.NS", f"Name {i}", "NSE Listed", "false") for i in range(50)]
        _write_csv(csv_path, rows)
        monkeypatch.setattr(bsi, "OUTPUT_CSV", csv_path)
        monkeypatch.setattr(bsi, "SECTOR_CACHE", tmp_path / "c.json")
        monkeypatch.setattr(bsi, "backfill_sectors",
                            lambda tickers, **kw: {t: "Energy" for t in tickers})

        bsi.backfill_csv_sectors()

        out = csv_path.read_text(encoding="utf-8").splitlines()
        assert len(out) == 51  # header + 50 rows
        assert "T7.NS,Name 7,Energy,false" in out

    def test_raises_if_csv_missing(self, bsi, tmp_path, monkeypatch):
        monkeypatch.setattr(bsi, "OUTPUT_CSV", tmp_path / "nope.csv")
        with pytest.raises(RuntimeError, match="not found"):
            bsi.backfill_csv_sectors()
