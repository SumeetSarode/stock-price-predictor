"""Regression tests for the two prediction-error UX bugs fixed in
commit 0e9c7b0 (fix(web): two prediction-error UX bugs surfaced by
LLM rate limits).

Both bugs lived on the error path of POST /api/predictions/run and
were only exposed when all LLM providers were simultaneously
rate-limited. They are locked here so they can never silently return.

BUG 1 — Content-Length mismatch on toast injection
    The endpoint mutates response.body to inject a data-toast="..."
    attribute when a prediction fails, but did not update the
    Content-Length header. The injected string is LONGER than what it
    replaced, so uvicorn's writer raised:
        RuntimeError: Response content longer than Content-Length
    leaving the browser request hanging forever.

BUG 2 — AllModelsExhaustedError fell into the generic catch-all
    AllModelsExhaustedError is a RuntimeError (not a PredictionError),
    so it hit the generic `except Exception` arm and surfaced the
    useless message 'An unexpected error occurred'. Users had no idea
    they were rate-limited or when to retry.
"""
from __future__ import annotations

import pytest

from price_predictor.llm.resilient import AllModelsExhaustedError
from price_predictor.web.routes import api as api_module
from price_predictor.web.services.panel_service import PanelCard
from price_predictor.web.services.prediction_service import (
    PredictionServiceError,
    run_prediction,
)


# ─────────────────────────────────────────────────────────────
# BUG 1 — Content-Length invariant on the error path
# ─────────────────────────────────────────────────────────────
class TestContentLengthOnToastInjection:
    """The toast-injection mutation must keep Content-Length in sync
    with the (now longer) body."""

    def _fake_card(self) -> PanelCard:
        # A real PanelCard in its 'no prediction yet' state — renders
        # panel_card.html without touching the network or the LLM.
        return PanelCard(
            ticker="AXISBANK.NS",
            name="Axis Bank",
            sector="Financial Services",
            is_nifty50=True,
            close=1100.0,
            change_pct=0.5,
            price_direction="bullish",
            prediction=None,
        )

    def test_error_path_keeps_content_length_in_sync(self, client, monkeypatch):
        """When a prediction errors, the injected data-toast makes the
        body longer; Content-Length must be recomputed to match.

        This is the exact invariant that broke: stale header < body len.
        """
        async def _boom(ticker, horizon):
            raise PredictionServiceError(
                "All LLM providers are currently unavailable (rate-limited "
                "or errored). Try again later.",
            )

        async def _fake_get_one_card(ticker, horizon="weekly"):
            return self._fake_card()

        monkeypatch.setattr(api_module, "run_prediction", _boom)
        monkeypatch.setattr(api_module, "get_one_card", _fake_get_one_card)

        resp = client.post(
            "/api/predictions/run",
            data={"ticker": "AXISBANK.NS", "horizon": "weekly", "context": "panel"},
            headers={"HX-Request": "true"},
        )

        assert resp.status_code == 200
        # The toast was actually injected (proves we exercised the buggy path).
        assert b'data-toast="All LLM providers' in resp.content
        # THE REGRESSION ASSERTION: header must equal real body length.
        # Pre-fix this was strictly less than len(resp.content).
        assert int(resp.headers["content-length"]) == len(resp.content)

    def test_success_path_content_length_also_correct(self, client, monkeypatch):
        """Sanity: the happy path (no toast injection) must also keep a
        correct Content-Length — guards against a fix that only patched
        the error branch."""
        async def _ok(ticker, horizon):
            return {"ticker": "AXISBANK.NS", "horizon": "weekly"}

        async def _fake_get_one_card(ticker, horizon="weekly"):
            return self._fake_card()

        monkeypatch.setattr(api_module, "run_prediction", _ok)
        monkeypatch.setattr(api_module, "get_one_card", _fake_get_one_card)

        resp = client.post(
            "/api/predictions/run",
            data={"ticker": "AXISBANK.NS", "horizon": "weekly", "context": "panel"},
            headers={"HX-Request": "true"},
        )

        assert resp.status_code == 200
        assert b"data-toast=" not in resp.content
        assert int(resp.headers["content-length"]) == len(resp.content)


# ─────────────────────────────────────────────────────────────
# BUG 2 — AllModelsExhaustedError → friendly, specific message
# ─────────────────────────────────────────────────────────────
class TestAllModelsExhaustedHandling:
    """When every LLM provider is cooled-down, the service must surface
    a clear rate-limit message, NOT the generic 'unexpected error'."""

    async def test_exhausted_maps_to_specific_message(self, monkeypatch):
        async def _exhausted(ticker, horizons):
            raise AllModelsExhaustedError(
                ["gemini/gemini-2.5-flash", "groq/llama-3.3-70b-versatile"],
                last_error=RuntimeError("rate limit"),
            )

        monkeypatch.setattr(
            "price_predictor.web.services.prediction_service.predict",
            _exhausted,
        )

        with pytest.raises(PredictionServiceError) as excinfo:
            await run_prediction("AXISBANK.NS", "weekly")

        err = excinfo.value
        # Specific, actionable message — NOT the generic catch-all.
        assert "LLM providers" in err.message
        assert "unavailable" in err.message.lower()
        assert err.message != "An unexpected error occurred while running the prediction."
        # Hint tells the user WHEN it'll recover.
        assert err.hint is not None
        assert "quota" in err.hint.lower() or "rate limit" in err.hint.lower()

    async def test_generic_exception_still_generic(self, monkeypatch):
        """Guard: a truly-unexpected error must still hit the generic
        arm — we only special-cased AllModelsExhaustedError, nothing else."""
        async def _weird(ticker, horizons):
            raise KeyError("some internal invariant broke")

        monkeypatch.setattr(
            "price_predictor.web.services.prediction_service.predict",
            _weird,
        )

        with pytest.raises(PredictionServiceError) as excinfo:
            await run_prediction("AXISBANK.NS", "weekly")

        assert excinfo.value.message == (
            "An unexpected error occurred while running the prediction."
        )


# ─────────────────────────────────────────────────────────────
# Structural LLM errors → precise, self-reporting messages
# (so users never have to read server logs to learn the cause)
# ─────────────────────────────────────────────────────────────
class TestStructuralLLMErrorsSurfacePrecisely:
    """AuthenticationError / ContextWindowExceededError are STRUCTURAL:
    the resilient chain does not fall back on them, so they propagate
    raw. They must NOT hit the generic 'unexpected error' arm."""

    async def test_auth_error_names_the_cause(self, monkeypatch):
        from litellm import AuthenticationError

        async def _auth(ticker, horizons):
            raise AuthenticationError(
                message="invalid api key",
                llm_provider="gemini",
                model="gemini/gemini-flash-latest",
            )

        monkeypatch.setattr(
            "price_predictor.web.services.prediction_service.predict", _auth
        )
        with pytest.raises(PredictionServiceError) as excinfo:
            await run_prediction("RELIANCE.NS", "weekly")

        err = excinfo.value
        assert "unauthenticated" in err.message.lower()
        assert err.message != "An unexpected error occurred while running the prediction."
        assert err.hint is not None
        assert "api key" in err.hint.lower()
        # Must be explicit that this is NOT a token/quota problem.
        assert "token" in err.hint.lower() or "quota" in err.hint.lower()

    async def test_context_window_error_points_at_num_ctx(self, monkeypatch):
        from litellm import ContextWindowExceededError

        async def _ctx(ticker, horizons):
            raise ContextWindowExceededError(
                message="prompt is too long",
                model="ollama_chat/qwen3:8b",
                llm_provider="ollama_chat",
            )

        monkeypatch.setattr(
            "price_predictor.web.services.prediction_service.predict", _ctx
        )
        with pytest.raises(PredictionServiceError) as excinfo:
            await run_prediction("RELIANCE.NS", "weekly")

        err = excinfo.value
        assert "context window" in err.message.lower()
        assert err.message != "An unexpected error occurred while running the prediction."
        assert err.hint is not None
        assert "ollama_num_ctx" in err.hint.lower()
