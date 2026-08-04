"""Tests for backtest.perf_bench -- pure benchmark math + rendering."""
from __future__ import annotations

from price_predictor.backtest.perf_bench import (
    BenchConfig,
    CallCollector,
    CallMetric,
    ConfigResult,
    SpeedStats,
    _percentile,
    aggregate_speed,
    default_configs,
    format_comparison_table,
    recommend_num_ctx,
)


class TestCallCollector:
    def test_add_and_snapshot(self):
        c = CallCollector()
        c.add(100, 50, 2.0)
        c.add(200, 80, 4.0)
        snap = c.snapshot()
        assert len(snap) == 2
        assert snap[0] == CallMetric(100, 50, 2.0)
        # snapshot is a copy -- mutating the collector doesn't change it
        c.reset()
        assert len(snap) == 2
        assert c.snapshot() == []

    def test_add_coerces_none_to_zero(self):
        c = CallCollector()
        c.add(None, None, None)  # type: ignore[arg-type]
        assert c.snapshot()[0] == CallMetric(0, 0, 0.0)


class TestPercentile:
    def test_empty(self):
        assert _percentile([], 0.95) == 0.0

    def test_single(self):
        assert _percentile([42.0], 0.95) == 42.0

    def test_median_even(self):
        assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5

    def test_p95_interpolates(self):
        vals = [float(i) for i in range(1, 101)]  # 1..100
        # (100-1)*0.95 = 94.05 -> between index 94 (95.0) and 95 (96.0)
        assert abs(_percentile(vals, 0.95) - 95.05) < 1e-9


class TestAggregateSpeed:
    def test_empty_is_zeroed(self):
        s = aggregate_speed([])
        assert s == SpeedStats(0, 0, 0.0, 0.0, 0.0, 0, 0)

    def test_tokens_per_second_is_ratio_of_sums(self):
        calls = [CallMetric(10, 100, 10.0), CallMetric(10, 50, 5.0)]
        s = aggregate_speed(calls)
        # (100+50)/(10+5) = 10 tok/s
        assert s.tokens_per_second == 10.0
        assert s.total_completion_tokens == 150
        assert s.n_calls == 2

    def test_prompt_token_extremes(self):
        calls = [CallMetric(p, 1, 1.0) for p in (1000, 5000, 3000)]
        s = aggregate_speed(calls)
        assert s.max_prompt_tokens == 5000

    def test_zero_latency_guard(self):
        s = aggregate_speed([CallMetric(10, 10, 0.0)])
        assert s.tokens_per_second == 0.0  # no divide-by-zero


class TestRecommendNumCtx:
    def test_rounds_up_with_headroom(self):
        # 5000 * 1.5 = 7500 -> round up to next 2048 multiple = 8192
        assert recommend_num_ctx(5000) == 8192

    def test_respects_floor(self):
        # tiny prompt -> still at least the floor (4096)
        assert recommend_num_ctx(100) == 4096

    def test_large_prompt(self):
        # 12000 * 1.5 = 18000 -> round up to 18432 (9 * 2048)
        assert recommend_num_ctx(12000) == 18432

    def test_custom_params(self):
        assert recommend_num_ctx(1000, headroom=2.0, floor=1024,
                                 round_to=1000) == 2000


class TestDefaultConfigs:
    def test_2x2_matrix(self):
        cfgs = default_configs(("qwen3:8b", "qwen3:4b"))
        assert len(cfgs) == 4
        labels = [c.label for c in cfgs]
        assert labels == [
            "qwen3:8b think-on", "qwen3:8b think-off",
            "qwen3:4b think-on", "qwen3:4b think-off",
        ]
        # think-on -> high, think-off -> none
        assert cfgs[0].reasoning_effort == "high"
        assert cfgs[1].reasoning_effort == "none"
        # model id gets provider prefix
        assert cfgs[0].model == "ollama_chat/qwen3:8b"

    def test_model_tag_strips_provider(self):
        c = BenchConfig("x", "ollama_chat/qwen3:4b", "high")
        assert c.model_tag == "qwen3:4b"


def _mk_result(label, da, tps, bss=0.1) -> ConfigResult:
    return ConfigResult(
        config=BenchConfig(label, "ollama_chat/qwen3:8b", "high"),
        n_predictions=20, n_judged=18,
        direction_accuracy=da, hit_rate_resolved=0.5,
        brier_skill_score=bss,
        speed=SpeedStats(30, 3000, 300.0, tps, 12.0, 5200, 4800),
        wall_clock_s=600.0,
    )


class TestConfigResultRoundtrip:
    def test_from_dict_inverts_as_dict_for_table_fields(self):
        r = _mk_result("qwen3:8b think-on", 0.56, 8.3)
        r2 = ConfigResult.from_dict(r.as_dict())
        assert r2.config.label == r.config.label
        assert r2.direction_accuracy == r.direction_accuracy
        assert r2.speed.tokens_per_second == r.speed.tokens_per_second
        assert r2.speed.max_prompt_tokens == r.speed.max_prompt_tokens
        assert r2.n_judged == r.n_judged

    def test_from_dict_handles_none_bss(self):
        r = _mk_result("x", 0.5, 5.0, bss=None)
        r2 = ConfigResult.from_dict(r.as_dict())
        assert r2.brier_skill_score is None


class TestComparisonTable:
    def test_empty(self):
        assert format_comparison_table([]) == "(no results)"

    def test_renders_rows_and_recommendation(self):
        results = [
            _mk_result("qwen3:8b think-on", 0.56, 3.1),
            _mk_result("qwen3:8b think-off", 0.54, 9.8),
        ]
        out = format_comparison_table(results)
        assert "qwen3:8b think-on" in out
        assert "qwen3:8b think-off" in out
        # max prompt seen 5200 -> 5200*1.5=7800 -> round up 8192
        assert "recommended OLLAMA_NUM_CTX=8192" in out
        assert "56.0%" in out  # direction accuracy formatting

    def test_handles_none_bss_row(self):
        out = format_comparison_table([_mk_result("x", 0.5, 5.0, bss=None)])
        assert "n/a" in out
