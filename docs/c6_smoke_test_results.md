# C.6 — Manual Smoke Test of `technical_agent` in `adk web`

## How to launch

```bash
cd price_predictor
uv run adk web src/price_predictor/agents
# open http://127.0.0.1:8000 → pick "technical_agent" from dropdown
```

## Known caveat: yfinance rate-limits (RESOLVED post-C — see Issue B below)

**Original (C.6 era)**: Yahoo Finance throttles aggressively. When
rate-limited with a single-provider chain, every tool call returns:

```json
{
  "status": "error",
  "error_message": "All price providers failed (tried in order: ['yfinance']).
                    Last error: yfinance returned no data ..."
}
```

The agent handles this CORRECTLY — it apologizes and does NOT fabricate
indicator values. **This is success behavior, not failure.**

**Now (post-addendum)**: With `PRICE_CHAIN=yfinance,stooq,alpha_vantage`
and keys in `.env`, Yahoo throttling triggers automatic fallback to
Stooq (free, no rate limit in practice). The error path above is now
the **last-resort** state, not the routine one.

## Smoke-test prompts and expected behavior

For each prompt, watch the "Function Calls" section in adk web's UI.

| # | Prompt | Expected tool calls | Pass criteria |
|---|---|---|---|
| 1 | "How is RELIANCE looking technically?" | get_trend, get_momentum, get_volatility, get_levels (all 4) | Narrative covers all four; no "buy/sell" verbs |
| 2 | "What's the RSI on TCS?" | get_momentum only | Single call; mentions RSI value |
| 3 | "What's a reasonable stop-loss on INFY?" | get_volatility only | Quotes suggested_stop_distance in Rs |
| 4 | "Did HDFCBANK break out recently?" | get_levels only | Mentions breakout_state + swing-high level |
| 5 | "Should I buy ICICIBANK right now?" | All 4 | EXPLICIT no-advice disclaimer |
| 6 | "How is HDFC looking?" (delisted alias) | get_trend (HDFC) → fails → get_trend (HDFCBANK.NS) auto-retry | Self-recovery without asking user |

## Issues fixed during C.6

### Bug A (FIXED): noisy LLM-chain fallback
The `.env` had `groq/llama-3.3-70b-versatile` as primary. Groq rejects
the assistant message shape ADK builds for multi-turn tool conversations
("messages.2: for 'role:assistant'..."). Resilient layer fell back to
Gemini → worked, but every conversation wasted a roundtrip + spammed
logs.

**Fix**: reordered `CHAIN_AGENTIC` in `.env` and `.env.example` to put
Gemini first (it's the model ADK was designed against), Groq last as
emergency backup.

### Issue B (RESOLVED): yfinance rate-limit

**Original problem (C.6 era)**: `PRICE_CHAIN=yfinance` only. Yahoo throttles
bursty agent flows (4 tools per question), leaving the user with no data.

**Resolution (post-C addendum)**: Filled out the resilient chain that was
already scaffolded in B.1.

- Added `StooqProvider` (free CSV download, captcha-only key, no signup)
- Added `AlphaVantageProvider` (free 25 req/day, paid tier toggleable)
- New `USE_PAID_PRICES` toggle parallels `USE_PAID` for LLMs
- Default `PRICE_CHAIN=yfinance,stooq,alpha_vantage` falls through cleanly
  when Yahoo throttles
- Integration test `test_fetch_ohlcv_real_reliance` now passes off-corp
  in <4s, exercising the real chain end-to-end

See `implementation_flow.md` → "Provider expansion" section for details.
