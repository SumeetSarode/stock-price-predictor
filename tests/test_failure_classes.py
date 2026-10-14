"""Failure classes decide what is retried and what is branded NOT_USEFUL, so a wrong label has real cost.

Found in a real run: Groq answered chat and JSON fine, then its heavier step hit a RATE LIMIT whose message carries
an "upgrade at .../billing" link. The classifier saw "billing", called it PAYMENT, and payment is never retried,
so working Groq models were on their way to a permanent NOT_USEFUL verdict.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import model_survey_offvpn as S
import sweep_data as D
import sweep_probe as PR

# Cerebras, verbatim from the run log.
CEREBRAS_402 = "HTTP 402 in 0.1s :: Payment required to access this resource. Visit your billing tab."
# Shaped like Groq's reply (the log kept only a truncated start), including the billing link and token counts.
GROQ_429 = ('litellm.RateLimitError: RateLimitError: GroqException - {"error":{"message":"Rate limit reached for model '
            '`openai/gpt-oss-20b` on tokens per day (TPD): Limit 200000, Used 199403, Requested 1200. Please try again '
            'in 5m. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing",'
            '"code":"rate_limit_exceeded"}}')


@pytest.mark.parametrize(("text", "klass"), [
    (CEREBRAS_402, "payment"),
    (GROQ_429, "rate_limited"),                                       # the bug: was "payment"
    ("429 Too Many Requests. See your billing settings.", "rate_limited"),
    ("insufficient_quota: check your plan and billing details", "payment"),
    ("Please add credits on the billing page", "payment"),            # a bare billing mention is still payment
    ("HTTP 403 Forbidden", "auth"),
    ("HTTP 401 Unauthorized", "auth"),
    ("status 503 service unavailable", "overloaded"),
    ("Synthesis exceeded the 90s wall-clock budget; request timed out", "timeout"),
    ("request used 199403 tokens", "other"),                          # 403 inside a longer number is not HTTP 403
    ("request id req_402abc", "other"),
    ("error code 1502", "other"),                                     # nor 502 inside 1502
])
def test_classify_uses_the_right_signal(text, klass):
    assert S.classify(text) == klass


def test_a_rate_limit_with_a_billing_link_is_retryable_not_permanent():
    klass = S.classify(GROQ_429)
    assert klass in D.TRANSIENT and klass not in D.PERMANENT


def store_with(tmp_path, level, klass, verdict="FAIL") -> PR.Store:
    s = PR.Store(tmp_path / "s.jsonl")
    s.add(PR.Result("direct|groq|m", level, verdict, klass))
    return s


@pytest.mark.parametrize(("level", "klass", "verdict", "retry"), [
    ("pipeline", "payment", "FAIL", True),            # cheap probes passed, so this is a mislabel or a one-off
    ("pipeline", "auth", "FAIL", True),
    ("pipeline", "bad_output,payment", "FAIL", True),
    ("pipeline", "rate_limited", "FAIL", True),
    ("chat", "payment", "FAIL", False),               # a genuine 402 at the first probe: the model really is not free
    ("chat", "auth", "FAIL", False),
    ("pipeline", "bad_output", "FAIL", False),        # the model really produced bad output: permanent
    ("pipeline", "not_found", "FAIL", False),
    ("pipeline", "", "PASS", False),
])
def test_what_gets_retried_on_resume(tmp_path, level, klass, verdict, retry):
    assert store_with(tmp_path, level, klass, verdict).needs("direct|groq|m", level) is retry


def test_a_model_never_tried_is_always_needed(tmp_path):
    assert PR.Store(tmp_path / "s.jsonl").needs("direct|groq|never", "chat") is True


def test_the_mislabeled_run_from_the_real_log_is_retried_after_the_fix(tmp_path):
    """The exact shape found: pipeline FAIL with klass 'bad_output,payment' for a model whose chat passed."""
    s = PR.Store(tmp_path / "s.jsonl")
    s.add(PR.Result("direct|groq|openai/gpt-oss-120b", "chat", "PASS"))
    s.add(PR.Result("direct|groq|openai/gpt-oss-120b", "pipeline", "FAIL", "bad_output,payment", run=0))
    s.add(PR.Result("direct|groq|openai/gpt-oss-120b", "pipeline", "FAIL", "bad_output,payment", run=1))
    assert not s.needs("direct|groq|openai/gpt-oss-120b", "chat"), "the passed chat probe is not repeated"
    assert s.needs("direct|groq|openai/gpt-oss-120b", "pipeline", 0) and s.needs("direct|groq|openai/gpt-oss-120b", "pipeline", 1)
