"""Pure, testable helpers for the Ollama config benchmark.

WHY THIS EXISTS
===============
`scripts/bench_ollama_configs.py` answers the two questions we could NOT
answer from published benchmarks (no external benchmark measures directional
equity prediction) and could NOT declare "sure-shot":

    1. num_ctx  -- what context window do real prediction prompts actually
       need? (Lowering it is a free speedup ONLY if we don't truncate.)
    2. think=off / qwen3:4b -- do the faster configs keep prediction skill?
       (Accuracy tradeoff -> must be measured on OUR backtest.)

The script drives the network (pull models, run predict(), grade). This
module holds the DATA + MATH so it can be unit-tested with zero network:
call aggregation, tok/s, prompt-token percentiles, the num_ctx
recommendation, the config matrix, and the comparison table rendering.

DESIGN
======
- CallCollector is a dumb sink: the script's litellm callback feeds it one
  .add() per LLM call. Pure list append -> GIL-safe under asyncio.
- Everything else is frozen dataclasses + free functions. No I/O, no
  global state, no litellm import -> trivially testable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────
# Per-call metric capture
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CallMetric:
    """One LLM call's cost. Populated from litellm's success callback.

    latency_s is end-to-end wall time for the call (the number the USER
    feels), NOT just model compute -- that's the honest speed metric for
    "is this config faster in practice."
    """
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class CallCollector:
    """Accumulates CallMetrics across a backtest run.

    The benchmark script registers a litellm callback that calls .add()
    once per completed LLM call. reset() between configs so each config's
    numbers are isolated. Plain list append is atomic under the GIL, which
    is all we need for asyncio concurrency.
    """

    def __init__(self) -> None:
        self.calls: list[CallMetric] = []

    def add(
        self, prompt_tokens: int, completion_tokens: int, latency_s: float
    ) -> None:
        self.calls.append(
            CallMetric(
                prompt_tokens=int(prompt_tokens or 0),
                completion_tokens=int(completion_tokens or 0),
                latency_s=float(latency_s or 0.0),
            )
        )

    def reset(self) -> None:
        self.calls.clear()

    def snapshot(self) -> list[CallMetric]:
        """Copy of the calls so far (so aggregation can't race a live run)."""
        return list(self.calls)


# ─────────────────────────────────────────────────────────────
# Speed aggregation
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SpeedStats:
    """Aggregate speed picture for one config."""
    n_calls: int
    total_completion_tokens: int
    total_latency_s: float
    # sum(completion)/sum(latency) -- the throughput that matters for a
    # generation-bound CPU workload. Mean-of-ratios would over-weight tiny
    # calls, so we use ratio-of-sums.
    tokens_per_second: float
    median_latency_s: float
    max_prompt_tokens: int
    p95_prompt_tokens: int


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolated percentile. pct in [0,1]. Empty -> 0."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * pct
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(sorted_vals[int(k)])
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


def aggregate_speed(calls: list[CallMetric]) -> SpeedStats:
    """Roll a list of per-call metrics into one SpeedStats."""
    if not calls:
        return SpeedStats(0, 0, 0.0, 0.0, 0.0, 0, 0)
    total_completion = sum(c.completion_tokens for c in calls)
    total_latency = sum(c.latency_s for c in calls)
    tps = (total_completion / total_latency) if total_latency > 0 else 0.0
    latencies = sorted(c.latency_s for c in calls)
    prompts = sorted(c.prompt_tokens for c in calls)
    return SpeedStats(
        n_calls=len(calls),
        total_completion_tokens=total_completion,
        total_latency_s=total_latency,
        tokens_per_second=tps,
        median_latency_s=_percentile(latencies, 0.5),
        max_prompt_tokens=int(prompts[-1]),
        p95_prompt_tokens=int(round(_percentile(prompts, 0.95))),
    )


# ─────────────────────────────────────────────────────────────
# num_ctx recommendation
# ─────────────────────────────────────────────────────────────
def recommend_num_ctx(
    max_prompt_tokens: int,
    *,
    headroom: float = 1.5,
    floor: int = 4096,
    round_to: int = 2048,
) -> int:
    """Smallest safe num_ctx for the observed prompts.

    Lowering num_ctx from 16384 is a free speedup ONLY if prompts never
    overflow it (otherwise -> ContextWindowExceededError). So: take the
    LARGEST prompt actually seen, add headroom for prompts we didn't
    sample (retry feedback, longer news days), round UP to a clean
    boundary, and never go below `floor`.

    Args:
        max_prompt_tokens: largest prompt_tokens observed in the sweep.
        headroom: multiplier for unseen worst-cases (1.5 = +50%).
        floor: never recommend below this (small prompts still want slack).
        round_to: round the result UP to a multiple of this.

    Returns:
        Recommended num_ctx (tokens).
    """
    target = max(int(max_prompt_tokens * headroom), floor)
    # round UP to the next multiple of round_to
    blocks = math.ceil(target / round_to)
    return blocks * round_to


# ─────────────────────────────────────────────────────────────
# Config matrix
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class BenchConfig:
    """One point in the sweep: a model + a thinking setting."""
    label: str
    model: str            # litellm id, e.g. "ollama_chat/qwen3:8b"
    reasoning_effort: str  # "high" -> think on; "none" -> think off

    @property
    def model_tag(self) -> str:
        """Bare Ollama tag for `ollama pull`, e.g. 'qwen3:8b'."""
        return self.model.split("/", 1)[1] if "/" in self.model else self.model


def default_configs(
    models: tuple[str, ...] = ("qwen3:8b", "qwen3:4b"),
    *,
    provider: str = "ollama_chat",
) -> list[BenchConfig]:
    """The 2x2 matrix: each model x {think-on, think-off}.

    Order is (per model) think-on first so the baseline (8b think-on)
    is row 1 -- the current production config.
    """
    configs: list[BenchConfig] = []
    for tag in models:
        model_id = f"{provider}/{tag}"
        configs.append(BenchConfig(f"{tag} think-on", model_id, "high"))
        configs.append(BenchConfig(f"{tag} think-off", model_id, "none"))
    return configs


# ─────────────────────────────────────────────────────────────
# Result + rendering
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ConfigResult:
    """Everything measured for one config: accuracy + speed."""
    config: BenchConfig
    n_predictions: int
    n_judged: int
    direction_accuracy: float
    hit_rate_resolved: float
    brier_skill_score: float | None
    speed: SpeedStats
    wall_clock_s: float
    n_errors: int = 0

    def as_dict(self) -> dict:
        """JSON-friendly flat dict for the diagnostics file."""
        return {
            "label": self.config.label,
            "model": self.config.model,
            "reasoning_effort": self.config.reasoning_effort,
            "n_predictions": self.n_predictions,
            "n_judged": self.n_judged,
            "n_errors": self.n_errors,
            "direction_accuracy": round(self.direction_accuracy, 4),
            "hit_rate_resolved": round(self.hit_rate_resolved, 4),
            "brier_skill_score": (
                round(self.brier_skill_score, 4)
                if self.brier_skill_score is not None else None
            ),
            "tokens_per_second": round(self.speed.tokens_per_second, 2),
            "median_latency_s": round(self.speed.median_latency_s, 2),
            "max_prompt_tokens": self.speed.max_prompt_tokens,
            "p95_prompt_tokens": self.speed.p95_prompt_tokens,
            "n_llm_calls": self.speed.n_calls,
            "wall_clock_s": round(self.wall_clock_s, 1),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConfigResult":
        """Rebuild a ConfigResult from as_dict() output (for rendering
        worker-subprocess JSON through the one table path). Fields not
        needed by the table (token/latency SUMS) are reconstructed as 0.
        """
        speed = SpeedStats(
            n_calls=int(d.get("n_llm_calls", 0)),
            total_completion_tokens=0,
            total_latency_s=0.0,
            tokens_per_second=float(d.get("tokens_per_second", 0.0)),
            median_latency_s=float(d.get("median_latency_s", 0.0)),
            max_prompt_tokens=int(d.get("max_prompt_tokens", 0)),
            p95_prompt_tokens=int(d.get("p95_prompt_tokens", 0)),
        )
        bss = d.get("brier_skill_score")
        return cls(
            config=BenchConfig(
                label=d["label"],
                model=d["model"],
                reasoning_effort=d["reasoning_effort"],
            ),
            n_predictions=int(d.get("n_predictions", 0)),
            n_judged=int(d.get("n_judged", 0)),
            direction_accuracy=float(d.get("direction_accuracy", 0.0)),
            hit_rate_resolved=float(d.get("hit_rate_resolved", 0.0)),
            brier_skill_score=(None if bss is None else float(bss)),
            speed=speed,
            wall_clock_s=float(d.get("wall_clock_s", 0.0)),
            n_errors=int(d.get("n_errors", 0)),
        )


def _fmt_pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{x * 100:5.1f}%"


def format_comparison_table(results: list[ConfigResult]) -> str:
    """Human-readable comparison table, baseline (row 0) first.

    Columns: config | n | dir-acc | BSS | hit-rate | tok/s | med-lat | wall.
    Speed and accuracy side by side so the Pareto tradeoff is obvious.
    """
    if not results:
        return "(no results)"
    header = (
        f"{'config':<18} {'n':>4} {'dir-acc':>7} {'BSS':>7} "
        f"{'hit':>6} {'tok/s':>7} {'med-lat':>8} {'wall(s)':>8}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for r in results:
        bss = "  n/a" if r.brier_skill_score is None else f"{r.brier_skill_score:6.3f}"
        lines.append(
            f"{r.config.label:<18} {r.n_judged:>4} "
            f"{_fmt_pct(r.direction_accuracy):>7} {bss:>7} "
            f"{_fmt_pct(r.hit_rate_resolved):>6} "
            f"{r.speed.tokens_per_second:>7.1f} "
            f"{r.speed.median_latency_s:>7.1f}s {r.wall_clock_s:>7.0f}s"
        )
    # num_ctx recommendation from the LARGEST prompt seen across ALL configs
    max_prompt = max((r.speed.max_prompt_tokens for r in results), default=0)
    rec = recommend_num_ctx(max_prompt)
    lines.append(sep)
    lines.append(
        f"Largest prompt seen: {max_prompt} tokens  ->  "
        f"recommended OLLAMA_NUM_CTX={rec} (current default 16384)"
    )
    return "\n".join(lines)
