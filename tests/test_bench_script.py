"""Tests for scripts/bench_ollama_configs.py -- the pure, non-network bits.

The orchestration (subprocess spawn, ollama pull, run_backtest) is network-
bound and covered by the perf_bench unit tests + manual runs. Here we lock
the two pure helpers that are easy to get subtly wrong: litellm usage
extraction and the default date window.
"""
from __future__ import annotations

import importlib.util
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from price_predictor.backtest.perf_bench import CallCollector

_SPEC = importlib.util.spec_from_file_location(
    "bench_ollama_configs",
    Path(__file__).resolve().parents[1] / "scripts" / "bench_ollama_configs.py",
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)  # type: ignore[union-attr]


class TestFeed:
    def test_extracts_usage_and_latency(self):
        c = CallCollector()
        resp = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=1234, completion_tokens=88)
        )
        t0 = datetime(2026, 1, 1, 0, 0, 0)
        t1 = datetime(2026, 1, 1, 0, 0, 12)
        mod._feed(c, resp, t0, t1)
        m = c.snapshot()[0]
        assert m.prompt_tokens == 1234
        assert m.completion_tokens == 88
        assert m.latency_s == 12.0

    def test_missing_usage_is_zeroed_not_crash(self):
        c = CallCollector()
        resp = SimpleNamespace(usage=None)
        t0 = datetime(2026, 1, 1)
        mod._feed(c, resp, t0, t0)
        m = c.snapshot()[0]
        assert (m.prompt_tokens, m.completion_tokens, m.latency_s) == (0, 0, 0.0)

    def test_bad_timestamps_dont_crash(self):
        c = CallCollector()
        resp = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        )
        mod._feed(c, resp, None, None)  # subtraction will TypeError -> guarded
        assert c.snapshot()[0].latency_s == 0.0


class TestDefaults:
    def test_default_dates_are_past_and_ordered(self):
        start, end = mod._default_dates()
        s, e = date.fromisoformat(start), date.fromisoformat(end)
        assert s < e
        # end should be well in the past so horizons have resolved
        assert e <= date.today() - timedelta(days=60)

    def test_parser_worker_flag_defaults_false(self):
        args = mod._build_parser().parse_args([])
        assert args._worker is False
        assert args.models == ["qwen3:8b", "qwen3:4b"]
        assert args.num_ctx == 16384  # held fixed across the sweep
