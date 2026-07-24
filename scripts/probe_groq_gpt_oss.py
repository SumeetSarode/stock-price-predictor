"""LIVE probe: does gpt-oss-120b on Groq return parseable JSON, and do
reasoning_effort / max_completion_tokens fix the "prose instead of JSON"
failure?

WHY: news_impact/synthesizer sometimes got a 200-OK response whose content
was truncated REASONING PROSE ("We need to synthesize... Let's craft.")
instead of the JSON object. Theory:
  * root cause = truncation mid-reasoning (finish_reason == "length")
  * fix candidates = bump max_completion_tokens (Option 2) and/or lower
    reasoning_effort (Option 1)

This hits Groq FOR REAL (needs VPN + GROQ_API_KEY in .env). It does NOT
touch the app; it's a throwaway diagnostic. Delete when done.

RUN:  .venv/bin/python scripts/probe_groq_gpt_oss.py
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import litellm

from price_predictor.agents.news_impact.agent import ImpactAssessment
from price_predictor.config.settings import settings

litellm.suppress_debug_info = True  # silence the 'Give Feedback' banner

MODEL = "groq/openai/gpt-oss-120b"
API_KEY = settings.groq_api_key.get_secret_value()

# Tee every line to the terminal AND a timestamped results file so the run
# is captured for later review / pasting.
_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_FILE = Path(__file__).with_name(f"probe_groq_gpt_oss_{_STAMP}.txt")
_LINES: list[str] = []


def out(line: str = "") -> None:
    """Print to stdout and buffer for the results file."""
    print(line)
    _LINES.append(line)


# A realistic news_impact-style ask: enough evidence to make the model
# actually reason, then a hard demand for the exact JSON object.
SYSTEM = (
    "You are a financial news impact analyst. Return ONLY a JSON object "
    "matching this schema, no prose, no markdown fence:\n"
    f"{json.dumps(ImpactAssessment.model_json_schema())}"
)
USER = (
    "Ticker: ITC.NS (ITC Ltd). Recent evidence:\n"
    "- ITC Q3 net profit up 8% YoY, cigarette volumes steady.\n"
    "- Board approves hotels business demerger, record date announced.\n"
    "- FMCG margins expand 60bps on softer input costs.\n"
    "- Brokerage upgrades to BUY, target implies ~12% upside.\n"
    "Assess the ~5-trading-day impact. Output the ImpactAssessment JSON."
)

JSON_OBJECT = {"type": "json_object"}


def _classify(content: str | None) -> str:
    if not content:
        return "EMPTY"
    try:
        ImpactAssessment.model_validate_json(content)
        return "PARSES "
    except Exception:
        # maybe it's JSON but wrong shape, or pure prose
        try:
            json.loads(content)
            return "JSON but WRONG SHAPE"
        except Exception:
            return "PROSE / NOT JSON "


async def _run(label: str, **kw) -> None:
    try:
        resp = await litellm.acompletion(
            model=MODEL,
            api_key=API_KEY,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER},
            ],
            **kw,
        )
    except Exception as e:  # we WANT to see 400s / connection errors
        out(f"  {label:38} -> ERROR {type(e).__name__}: {str(e)[:160]}")
        return

    choice = resp.choices[0]
    content = choice.message.content
    finish = choice.finish_reason
    reasoning = getattr(choice.message, "reasoning_content", None)
    usage = resp.usage
    verdict = _classify(content)
    out(
        f"  {label:38} -> {verdict:22} finish={finish!s:10} "
        f"tok(c/t)={getattr(usage,'completion_tokens','?')}/"
        f"{getattr(usage,'total_tokens','?')} "
        f"reasoning_content={'yes' if reasoning else 'no'}"
    )
    if verdict.startswith("PROSE") or finish == "length":
        head = (content or "")[:120].replace("\n", " ")
        out(f'       content head: "{head}"')


async def main() -> None:
    out(f"\nProbing {MODEL} live via Groq...\n")

    out("A) BASELINE (what the app does today: nothing set):")
    await _run("no knobs, no response_format")
    await _run("response_format=json_object", response_format=JSON_OBJECT)

    out("\nB) OPTION 2 -- bump max_completion_tokens (json_object):")
    for mx in (1024, 4096, 8192):
        await _run(
            f"max_completion_tokens={mx}",
            response_format=JSON_OBJECT,
            max_completion_tokens=mx,
        )

    out("\nC) OPTION 1 -- reasoning_effort (json_object, max=8192):")
    for eff in ("none", "minimal", "low", "medium"):
        await _run(
            f"reasoning_effort={eff!r}",
            response_format=JSON_OBJECT,
            reasoning_effort=eff,
            max_completion_tokens=8192,
        )

    out("\nD) COMBO -- reasoning_effort='low' + modest budget:")
    await _run(
        "low + max=2048",
        response_format=JSON_OBJECT,
        reasoning_effort="low",
        max_completion_tokens=2048,
    )

    out("\nDone. Look for: finish=length == truncation; PROSE == the bug.")

    OUT_FILE.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"\nResults saved to: {OUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
