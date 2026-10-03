"""Tests for the Antigravity CLI chain model (llm/agy_cli.py).

The real `agy` can't be installed on every network (its download bucket is
policy-blocked on some), so these run against a FAKE `agy` shell script that
emits the envelope shape documented at antigravity.google/docs/cli/headless.
They prove OUR side: prompt building, parsing, error classification, and
chain fallover. They cannot prove the real binary's behaviour -- that's what
`scripts/verify_offvpn.py --only-model agy/default` is for.
"""
from __future__ import annotations

import json
import stat
from datetime import UTC, datetime, timedelta

import pytest
from google.adk.models.llm_request import LlmRequest
from google.genai import types
from litellm.exceptions import Timeout

from price_predictor.llm.agy_cli import (
    AgyCliError,
    AgyCliLlm,
    AgyUnavailableError,
    build_prompt,
    find_agy_binary,
    parse_envelope,
)
from price_predictor.llm.rate_limiter import reset_for_tests
from price_predictor.llm.resilient import (
    INCOMPAT_COOLDOWN,
    ResilientModel,
    _classify_cooldown,
)

# Envelope copied from Google's headless docs (field names + shapes).
GOOD_ENVELOPE = {
    "conversation_id": "055a398f-db14-4c5f-abbb-1bf03f8120a7",
    "status": "SUCCESS",
    "response": '{"direction":"bullish","confidence":0.7}\n',
    "duration_seconds": 7.16,
    "num_turns": 1,
    "usage": {"input_tokens": 10415, "output_tokens": 657, "total_tokens": 11072},
}


@pytest.fixture(autouse=True)
def _wipe_registries():
    reset_for_tests()
    yield
    reset_for_tests()


def _fake_agy(tmp_path, *, stdout="", stderr="", exit_code=0, sleep=0, log=None):
    """Write an executable fake `agy`. It logs argv + cwd listing if asked."""
    script = tmp_path / "agy"
    log_line = (
        f'printf "%s\\n" "$@" > "{log}"; ls -A . > "{log}.cwd"\n' if log else ""
    )
    script.write_text(
        "#!/bin/sh\n"
        f"{log_line}"
        f"sleep {sleep}\n"
        f"printf '%s' '{stderr}' 1>&2\n"
        f"printf '%s\\n' '{stdout}'\n"
        f"exit {exit_code}\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def _request(user="Predict RELIANCE", system="You are a stock analyst."):
    return LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text=user)])],
        config=types.GenerateContentConfig(system_instruction=system),
    )


async def _collect(model: AgyCliLlm, req=None):
    return [r async for r in model.generate_content_async(req or _request())]


# ── prompt building ───────────────────────────────────────────
def test_prompt_has_system_user_and_no_tools_preamble():
    p = build_prompt(_request())
    assert p.startswith("You are acting as a plain text-generation endpoint")
    assert "Do NOT use any tools" in p
    assert "You are a stock analyst." in p
    assert "Predict RELIANCE" in p


def test_prompt_never_starts_with_a_dash_so_it_cant_parse_as_a_flag():
    p = build_prompt(_request(user="--output-format text", system="-p evil"))
    assert not p.startswith("-")


# ── envelope parsing ──────────────────────────────────────────
def test_parse_envelope_ignores_stray_leading_lines():
    out = "warming up...\n" + json.dumps(GOOD_ENVELOPE) + "\n"
    assert parse_envelope(out)["status"] == "SUCCESS"


def test_parse_envelope_rejects_non_json():
    with pytest.raises(AgyCliError):
        parse_envelope("just prose, no json")


# ── running the (fake) CLI ────────────────────────────────────
@pytest.mark.asyncio
async def test_success_returns_text_and_real_usage(tmp_path):
    binary = _fake_agy(tmp_path, stdout=json.dumps(GOOD_ENVELOPE))
    [resp] = await _collect(AgyCliLlm(model="agy/default", binary=binary))
    assert resp.content.parts[0].text == '{"direction":"bullish","confidence":0.7}'
    assert resp.usage_metadata.total_token_count == 11072


@pytest.mark.asyncio
async def test_runs_in_empty_dir_with_prompt_as_single_argv_item(tmp_path):
    log = tmp_path / "call.log"
    binary = _fake_agy(tmp_path, stdout=json.dumps(GOOD_ENVELOPE), log=log)
    await _collect(AgyCliLlm(model="agy/default", binary=binary))
    argv = log.read_text().splitlines()
    assert argv[0] == "-p"
    assert argv[-2:] == ["--output-format", "json"]
    assert (tmp_path / "call.log.cwd").read_text() == "", "cwd must be empty"


@pytest.mark.asyncio
async def test_non_success_status_is_a_plain_failure(tmp_path):
    bad = {**GOOD_ENVELOPE, "status": "ERROR", "response": "", "error": "model glitch"}
    binary = _fake_agy(tmp_path, stdout=json.dumps(bad))
    with pytest.raises(AgyCliError):
        await _collect(AgyCliLlm(model="agy/default", binary=binary))


@pytest.mark.asyncio
async def test_not_signed_in_is_unavailable_not_authentication_error(tmp_path):
    """Must NOT be litellm.AuthenticationError: that's STRUCTURAL and would
    kill the whole chain over an optional provider."""
    binary = _fake_agy(tmp_path, stderr="authentication required", exit_code=1)
    with pytest.raises(AgyUnavailableError):
        await _collect(AgyCliLlm(model="agy/default", binary=binary))


@pytest.mark.asyncio
async def test_missing_binary_is_unavailable(tmp_path):
    m = AgyCliLlm(model="agy/default", binary=str(tmp_path / "nope"))
    with pytest.raises(AgyUnavailableError):
        await _collect(m)


@pytest.mark.asyncio
async def test_timeout_kills_process_and_raises_transient(tmp_path):
    binary = _fake_agy(tmp_path, stdout="{}", sleep=5)
    m = AgyCliLlm(model="agy/default", binary=binary, timeout_s=0.3)
    with pytest.raises(Timeout):
        await _collect(m)


# ── taxonomy + chain integration ──────────────────────────────
def test_unavailable_gets_the_long_cooldown():
    assert _classify_cooldown(AgyUnavailableError("x")) == INCOMPAT_COOLDOWN


@pytest.mark.asyncio
async def test_signed_out_agy_falls_over_to_next_model(tmp_path):
    """The whole point: a signed-out agy must degrade, not break predictions."""
    from tests.test_resilient import _make_fake, _make_request

    agy = AgyCliLlm(
        model="agy/default",
        binary=_fake_agy(tmp_path, stderr="authentication required", exit_code=1),
    )
    backup = _make_fake("cohere_chat/command-r-08-2024")
    m = ResilientModel(inner_models=[agy, backup])

    responses = [r async for r in m.generate_content_async(_make_request())]

    assert len(responses) == 1 and backup.call_count == 1
    expiry = m.cooldowns["agy/default"]
    assert timedelta(minutes=59) < expiry - datetime.now(UTC) <= timedelta(hours=1)


# ── discovery / settings / factory ────────────────────────────
def test_find_binary_honours_explicit_path(tmp_path):
    real = _fake_agy(tmp_path, stdout="{}")
    assert find_agy_binary(real) == real
    assert find_agy_binary(str(tmp_path / "missing")) is None


def test_chain_skips_agy_when_binary_missing(monkeypatch):
    from price_predictor.config.settings import settings

    monkeypatch.setattr("price_predictor.llm.agy_cli.shutil.which", lambda _: None)
    monkeypatch.setattr(
        "price_predictor.llm.agy_cli._FALLBACK_INSTALL_PATH",
        __import__("pathlib").Path("/nonexistent/agy"),
    )
    monkeypatch.setattr(settings, "agy_path", "")
    assert settings._optional_key_missing("agy/default") is True


def test_chain_keeps_agy_when_binary_present(monkeypatch, tmp_path):
    from price_predictor.config.settings import settings

    monkeypatch.setattr(settings, "agy_path", _fake_agy(tmp_path, stdout="{}"))
    assert settings._optional_key_missing("agy/default") is False


def test_make_model_builds_agy_llm(monkeypatch, tmp_path):
    from price_predictor.config.settings import settings
    from price_predictor.llm.factory import make_model

    monkeypatch.setattr(settings, "agy_path", _fake_agy(tmp_path, stdout="{}"))
    m = make_model("agy/default")
    assert isinstance(m, AgyCliLlm) and m.model == "agy/default"


def test_make_model_agy_without_binary_is_a_clear_error(monkeypatch):
    from price_predictor.config.settings import settings
    from price_predictor.llm.factory import make_model

    monkeypatch.setattr("price_predictor.llm.agy_cli.shutil.which", lambda _: None)
    monkeypatch.setattr(
        "price_predictor.llm.agy_cli._FALLBACK_INSTALL_PATH",
        __import__("pathlib").Path("/nonexistent/agy"),
    )
    monkeypatch.setattr(settings, "agy_path", "")
    with pytest.raises(ValueError, match="Antigravity CLI"):
        make_model("agy/default")
