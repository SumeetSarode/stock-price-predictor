"""Tests for price_predictor.agents.news_impact.

Strategy:
    - Mock each underlying data fetcher at the boundary (fetch_news,
      fetch_filings, fetch_estimates, fetch_ohlcv).
    - Verify each tool's response shape, error handling, input validation.
    - Verify the Pydantic schemas reject invalid inputs.
    - Structural smoke for make_news_impact_agent() -- no LLM behavior tests.

We do NOT test LLM responses here. Those are verified via:
    uv run adk run price_predictor.agents.news_impact
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from pydantic import ValidationError

from price_predictor.agents.news_impact import (
    Catalyst,
    ImpactAssessment,
    fetch_estimates_tool,
    fetch_recent_filings_tool,
    fetch_recent_news_tool,
    fetch_recent_prices_tool,
    make_news_impact_agent,
    root_agent,
)
from price_predictor.data.estimates import EstimatesFetchError
from price_predictor.data.filings import FilingsFetchError
from price_predictor.data.news import NewsFetchError
from price_predictor.data.schema import (
    Estimates,
    PriceTargets,
    QuarterlyEstimate,
    RecommendationDistribution,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _news_df(rows: int = 3) -> pd.DataFrame:
    """Build a DataFrame matching fetch_news()'s normalized output."""
    return pd.DataFrame([
        {
            "title": f"Big news headline #{i}",
            "url": f"https://example.com/article{i}",
            "published_at": f"2026-04-2{i} 10:00:00",
            "domain": "example.com",
            "snippet": f"Body snippet for article {i}, explaining what happened.",
        }
        for i in range(1, rows + 1)
    ])


def _filings_df(rows: int = 3) -> pd.DataFrame:
    """Build a DataFrame matching fetch_filings()'s normalized output."""
    return pd.DataFrame([
        {
            "symbol": "RELIANCE",
            "kind": ["announcement", "board_meeting", "corporate_action"][i % 3],
            "announced_at": f"2026-04-2{i}T10:00:00+05:30",
            "event_at": None if i % 2 else f"2026-05-0{i}T00:00:00+05:30",
            "event_type": "Test Event",
            "subject": f"Filing subject #{i}",
            "description": "",
            "attachment_url": None,
            "metadata": {},
        }
        for i in range(1, rows + 1)
    ])


def _estimates_with_coverage() -> Estimates:
    return Estimates(
        symbol="RELIANCE.NS",
        fetched_at=datetime.now(UTC),
        earnings_estimates=[
            QuarterlyEstimate(
                period="0q", num_analysts=18,
                avg=15.5, low=14.0, high=17.0, year_ago=13.0,
            ),
            QuarterlyEstimate(
                period="+1q", num_analysts=16,
                avg=16.8, low=15.0, high=18.5, year_ago=14.0,
            ),
        ],
        revenue_estimates=[
            QuarterlyEstimate(
                period="0q", num_analysts=18,
                avg=2_500_000_000_000.0,
                low=2_400_000_000_000.0,
                high=2_600_000_000_000.0,
                year_ago=2_300_000_000_000.0,
            ),
        ],
        recommendations=[
            RecommendationDistribution(
                period="0m",
                strong_buy=10, buy=12, hold=8, sell=2, strong_sell=0,
            ),
            RecommendationDistribution(
                period="-1m",
                strong_buy=9, buy=13, hold=8, sell=2, strong_sell=0,
            ),
        ],
        price_targets=PriceTargets(
            current=2900.0, mean=3200.0, high=3500.0, low=2800.0, median=3150.0,
        ),
    )


def _prices_df() -> pd.DataFrame:
    """Mimics fetch_ohlcv's post-normalization shape."""
    idx = pd.DatetimeIndex([
        datetime(2026, 4, d, tzinfo=ZoneInfo("Asia/Kolkata"))
        for d in range(1, 6)
    ])
    return pd.DataFrame({
        "open": [2400.0, 2410.0, 2420.0, 2430.0, 2440.0],
        "high": [2450.0, 2460.0, 2470.0, 2480.0, 2490.0],
        "low": [2380.0, 2390.0, 2400.0, 2410.0, 2420.0],
        "close": [2420.0, 2430.0, 2440.0, 2450.0, 2460.0],
        "adj_close": [2400.0, 2410.0, 2420.0, 2430.0, 2440.0],
        "volume": [1_000_000] * 5,
    }, index=idx)


# ═════════════════════════════════════════════════════════════
# fetch_recent_news_tool
# ═════════════════════════════════════════════════════════════
class TestFetchRecentNewsTool:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        with patch(
            "price_predictor.agents.news_impact.agent.fetch_news",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = _news_df(rows=3)
            result = await fetch_recent_news_tool("Reliance Industries", days_back=7)

        assert result["status"] == "success"
        assert result["query"] == "Reliance Industries"
        assert result["article_count"] == 3
        assert len(result["articles"]) == 3
        assert result["articles"][0]["title"] == "Big news headline #1"
        assert "start" in result and "end" in result

    @pytest.mark.asyncio
    async def test_caps_articles_at_25(self):
        with patch(
            "price_predictor.agents.news_impact.agent.fetch_news",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = _news_df(rows=50)
            result = await fetch_recent_news_tool("Reliance", days_back=30)

        assert result["article_count"] == 50  # raw count preserved
        assert len(result["articles"]) == 25  # but list capped

    @pytest.mark.asyncio
    async def test_empty_query_rejected(self):
        result = await fetch_recent_news_tool("", days_back=7)
        assert result["status"] == "error"
        assert "non-empty" in result["error_message"]

    @pytest.mark.asyncio
    async def test_whitespace_query_rejected(self):
        result = await fetch_recent_news_tool("   ", days_back=7)
        assert result["status"] == "error"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_days", [0, -1, 91, 1000, "7", 7.5, True])
    async def test_invalid_days_back(self, bad_days):
        result = await fetch_recent_news_tool("Reliance", days_back=bad_days)
        assert result["status"] == "error"
        assert "days_back" in result["error_message"]

    @pytest.mark.asyncio
    async def test_news_fetch_error_propagates(self):
        with patch(
            "price_predictor.agents.news_impact.agent.fetch_news",
            new_callable=AsyncMock,
        ) as mock:
            mock.side_effect = NewsFetchError("GDELT down")
            result = await fetch_recent_news_tool("Reliance", days_back=7)

        assert result["status"] == "error"
        assert "GDELT down" in result["error_message"]

    @pytest.mark.asyncio
    async def test_value_error_propagates(self):
        with patch(
            "price_predictor.agents.news_impact.agent.fetch_news",
            new_callable=AsyncMock,
        ) as mock:
            mock.side_effect = ValueError("query too short")
            result = await fetch_recent_news_tool("Reliance", days_back=7)

        assert result["status"] == "error"
        assert "query too short" in result["error_message"]


# ═════════════════════════════════════════════════════════════
# fetch_recent_filings_tool
# ═════════════════════════════════════════════════════════════
class TestFetchRecentFilingsTool:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        with patch(
            "price_predictor.agents.news_impact.agent.fetch_filings",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = _filings_df(rows=3)
            result = await fetch_recent_filings_tool("RELIANCE", days_back=30)

        assert result["status"] == "success"
        assert result["nse_symbol"] == "RELIANCE"
        assert result["filing_count"] == 3
        assert "by_kind" in result
        assert sum(result["by_kind"].values()) == 3
        assert len(result["filings"]) == 3

    @pytest.mark.asyncio
    async def test_uppercases_symbol(self):
        with patch(
            "price_predictor.agents.news_impact.agent.fetch_filings",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = _filings_df(rows=1)
            result = await fetch_recent_filings_tool("reliance", days_back=10)

        assert result["nse_symbol"] == "RELIANCE"
        # And the underlying fetcher got the uppercased version too
        mock.assert_called_once()
        assert mock.call_args.args[0] == "RELIANCE"

    @pytest.mark.asyncio
    async def test_caps_filings_at_20(self):
        with patch(
            "price_predictor.agents.news_impact.agent.fetch_filings",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = _filings_df(rows=50)
            result = await fetch_recent_filings_tool("RELIANCE", days_back=30)

        assert result["filing_count"] == 50
        assert len(result["filings"]) == 20

    @pytest.mark.asyncio
    async def test_ns_suffix_rejected(self):
        """LLM might confuse yfinance and NSE conventions; tool rejects .NS."""
        result = await fetch_recent_filings_tool("RELIANCE.NS", days_back=30)
        assert result["status"] == "error"
        assert "bare ticker" in result["error_message"]

    @pytest.mark.asyncio
    async def test_empty_symbol_rejected(self):
        result = await fetch_recent_filings_tool("", days_back=30)
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_empty_filings_returns_success_with_zero(self):
        """Empty data is success-with-0, NOT an error (project rule)."""
        with patch(
            "price_predictor.agents.news_impact.agent.fetch_filings",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = pd.DataFrame(columns=[
                "symbol", "kind", "announced_at", "event_at", "event_type",
                "subject", "description", "attachment_url", "metadata",
            ])
            result = await fetch_recent_filings_tool("RELIANCE", days_back=7)

        assert result["status"] == "success"
        assert result["filing_count"] == 0
        assert result["filings"] == []
        assert result["by_kind"] == {}

    @pytest.mark.asyncio
    async def test_filings_fetch_error_propagates(self):
        with patch(
            "price_predictor.agents.news_impact.agent.fetch_filings",
            new_callable=AsyncMock,
        ) as mock:
            mock.side_effect = FilingsFetchError("NSE blocked")
            result = await fetch_recent_filings_tool("RELIANCE", days_back=30)

        assert result["status"] == "error"
        assert "NSE blocked" in result["error_message"]


# ═════════════════════════════════════════════════════════════
# fetch_estimates_tool
# ═════════════════════════════════════════════════════════════
class TestFetchEstimatesTool:
    @pytest.mark.asyncio
    async def test_happy_path_with_coverage(self):
        with patch(
            "price_predictor.agents.news_impact.agent.fetch_estimates",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = _estimates_with_coverage()
            result = await fetch_estimates_tool("RELIANCE.NS")

        assert result["status"] == "success"
        assert result["has_coverage"] is True
        s = result["summary"]
        assert s["next_quarter_eps_consensus"] == 15.5
        assert s["next_quarter_eps_num_analysts"] == 18
        assert s["next_quarter_revenue_consensus"] == 2_500_000_000_000.0
        assert s["price_target_mean"] == 3200.0
        assert s["current_price"] == 2900.0
        assert s["recommendations_current"]["strong_buy"] == 10
        assert s["recommendations_current"]["total"] == 32

    @pytest.mark.asyncio
    async def test_no_coverage_returns_nones_not_error(self):
        """has_coverage=False means analyst data unavailable -- NOT an error."""
        with patch(
            "price_predictor.agents.news_impact.agent.fetch_estimates",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = Estimates(
                symbol="OBSCURE.NS",
                fetched_at=datetime.now(UTC),
                earnings_estimates=[],
                revenue_estimates=[],
                recommendations=[],
                price_targets=None,
            )
            result = await fetch_estimates_tool("OBSCURE.NS")

        assert result["status"] == "success"
        assert result["has_coverage"] is False
        # All summary fields should be None
        assert all(v is None for v in result["summary"].values())

    @pytest.mark.asyncio
    async def test_empty_ticker_rejected(self):
        result = await fetch_estimates_tool("")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_estimates_fetch_error_propagates(self):
        with patch(
            "price_predictor.agents.news_impact.agent.fetch_estimates",
            new_callable=AsyncMock,
        ) as mock:
            mock.side_effect = EstimatesFetchError("yfinance rate limit")
            result = await fetch_estimates_tool("RELIANCE.NS")

        assert result["status"] == "error"
        assert "rate limit" in result["error_message"]


# ═════════════════════════════════════════════════════════════
# fetch_recent_prices_tool (thin wrapper)
# ═════════════════════════════════════════════════════════════
class TestFetchRecentPricesTool:
    def test_happy_path(self):
        # Patch the underlying sync fetch_ohlcv that fetch_prices_tool uses
        with patch("price_predictor.agents.price_agent.agent.fetch_ohlcv") as mock:
            mock.return_value = _prices_df()
            result = fetch_recent_prices_tool("RELIANCE.NS", days_back=30)

        assert result["status"] == "success"
        assert result["ticker"] == "RELIANCE.NS"
        assert "summary" in result
        # No bars in this wrapper (we hard-code include_bars=False)
        assert "bars" not in result

    def test_invalid_days_back(self):
        result = fetch_recent_prices_tool("RELIANCE.NS", days_back=1000)
        assert result["status"] == "error"
        assert "days_back" in result["error_message"]

    def test_propagates_underlying_errors(self):
        """Errors from fetch_prices_tool flow through unchanged."""
        with patch("price_predictor.agents.price_agent.agent.fetch_ohlcv") as mock:
            from price_predictor.data.prices import PriceFetchError
            mock.side_effect = PriceFetchError("upstream broke")
            result = fetch_recent_prices_tool("RELIANCE.NS", days_back=30)

        assert result["status"] == "error"
        assert "upstream broke" in result["error_message"]


# ═════════════════════════════════════════════════════════════
# Schema validation
# ═════════════════════════════════════════════════════════════
class TestCatalystSchema:
    def test_happy(self):
        c = Catalyst(
            description="Q4 EPS beat consensus by 12%",
            source="news",
            impact="positive",
        )
        assert c.source == "news"

    def test_description_too_short(self):
        with pytest.raises(ValidationError):
            Catalyst(description="short", source="news", impact="positive")

    def test_invalid_source(self):
        with pytest.raises(ValidationError):
            Catalyst(
                description="A perfectly valid description here",
                source="twitter",  # not allowed
                impact="positive",
            )

    def test_invalid_impact(self):
        with pytest.raises(ValidationError):
            Catalyst(
                description="A perfectly valid description here",
                source="news",
                impact="meh",  # not allowed
            )

    @pytest.mark.parametrize("impact", ["positive", "negative", "neutral", "mixed"])
    def test_all_four_impact_values_accepted(self, impact):
        """Regression: 'neutral' was missing, causing real Gemini calls to
        fail mid-stream when the model assessed a catalyst as neutral
        (perfectly reasonable -- e.g. a CEO interview with no new info).
        Schema must include all four; the *prompt* tells the LLM when to
        use each. See agent.py docstring on Catalyst.impact.
        """
        c = Catalyst(
            description="A perfectly valid description here",
            source="news",
            impact=impact,
        )
        assert c.impact == impact


class TestImpactAssessmentSchema:
    def _valid_kwargs(self) -> dict:
        return {
            "ticker": "RELIANCE",
            "sentiment": "bullish",
            "confidence": 0.7,
            "estimated_pct_move": 3.5,
            "reasoning": (
                "Recent news coverage shows three positive articles about new "
                "retail expansion. Filings indicate an upcoming board meeting. "
                "Price action over the past month has been steadily positive."
            ),
            "catalysts": [
                Catalyst(
                    description="Retail expansion announcement",
                    source="news",
                    impact="positive",
                ),
            ],
        }

    def test_happy(self):
        a = ImpactAssessment(**self._valid_kwargs())
        assert a.sentiment == "bullish"
        assert len(a.catalysts) == 1

    def test_empty_catalysts_allowed(self):
        kwargs = self._valid_kwargs()
        kwargs["catalysts"] = []
        a = ImpactAssessment(**kwargs)
        assert a.catalysts == []

    def test_confidence_out_of_range(self):
        kwargs = self._valid_kwargs()
        kwargs["confidence"] = 1.5
        with pytest.raises(ValidationError):
            ImpactAssessment(**kwargs)

    def test_estimated_move_clamped(self):
        kwargs = self._valid_kwargs()
        kwargs["estimated_pct_move"] = 50.0  # too extreme
        with pytest.raises(ValidationError):
            ImpactAssessment(**kwargs)

    def test_reasoning_too_short(self):
        kwargs = self._valid_kwargs()
        kwargs["reasoning"] = "too short"  # 9 chars, below floor of 20
        with pytest.raises(ValidationError):
            ImpactAssessment(**kwargs)

    def test_reasoning_accepts_honest_short_no_data_answer(self):
        """Regression: when all tools rate-limit/fail, the agent's honest
        short reasoning ('All tools rate-limited; no evidence to assess.')
        was being rejected by min_length=100 -> 500 to the user. Schemas
        encode invariants, not preferences. Length quality is steered by
        the prompt; the floor only rejects garbage.
        """
        kwargs = self._valid_kwargs()
        kwargs["reasoning"] = "All tools rate-limited; no evidence to assess."
        kwargs["catalysts"] = []  # honest: no data, no catalysts
        a = ImpactAssessment(**kwargs)
        assert len(a.reasoning) < 100, "This is the case the old schema rejected"
        assert len(a.reasoning) >= 20, "...but still above the new garbage floor"

    def test_too_many_catalysts(self):
        kwargs = self._valid_kwargs()
        kwargs["catalysts"] = [
            Catalyst(
                description=f"Catalyst number {i} description goes here",
                source="news", impact="positive",
            )
            for i in range(11)
        ]
        with pytest.raises(ValidationError):
            ImpactAssessment(**kwargs)


# ═════════════════════════════════════════════════════════════
# Agent factory smoke (no LLM calls)
# ═════════════════════════════════════════════════════════════
class TestAgentFactory:
    def test_make_news_impact_agent_structure(self):
        agent = make_news_impact_agent()
        assert agent.name == "news_impact"
        assert "Indian stock" in agent.description
        # 4 tools: news, filings, estimates, prices
        assert len(agent.tools) == 4
        tool_names = {t.__name__ for t in agent.tools}
        assert tool_names == {
            "fetch_recent_news_tool",
            "fetch_recent_filings_tool",
            "fetch_estimates_tool",
            "fetch_recent_prices_tool",
        }
        # Structured output bound
        assert agent.output_schema is ImpactAssessment

    def test_root_agent_module_level(self):
        """root_agent must exist at module level for ADK CLI discovery."""
        assert root_agent is not None
        assert root_agent.name == "news_impact"

    def test_prompt_has_known_ticker_gotchas(self):
        """Regression: live UI failure where agent didn't know HDFC Ltd merged
        into HDFC Bank in 2023. Prompt must teach the agent these domain
        facts so it self-resolves common ticker confusion."""
        agent = make_news_impact_agent()
        prompt = agent.instruction
        # The bug that motivated this fix
        assert "HDFC" in prompt and "HDFCBANK" in prompt
        # Section header so future devs know to add to this list
        assert "KNOWN TICKER GOTCHAS" in prompt

    def test_prompt_has_tool_error_recovery_rule(self):
        """The price tool returns 'suggested_ticker' on alias errors.
        The prompt must instruct the agent to USE that field instead of
        ignoring it and giving up."""
        agent = make_news_impact_agent()
        prompt = agent.instruction
        assert "suggested_ticker" in prompt
        assert "retry" in prompt.lower()
