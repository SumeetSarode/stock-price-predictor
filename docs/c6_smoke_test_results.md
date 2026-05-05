# C.6 — Manual Smoke Test of `technical_agent` in `adk web`

## How to launch

```bash
cd price_predictor
uv run adk web src/price_predictor/agents
# open http://127.0.0.1:8000 → pick "technical_agent" from dropdown
```

## Known caveat: yfinance rate-limits

Yahoo Finance throttles aggressively. When rate-limited, every tool call
returns:

```json
{
  "status": "error",
  "error_message": "All price providers failed (tried in order: ['yfinance']).
                    Last error: yfinance returned no data ..."
}
```

The agent handles this CORRECTLY — it apologizes and does NOT fabricate
indicator values. **This is success behavior, not failure.** To test
the happy path, either:
  - Wait 30-60min for Yahoo to cool down, or
  - Add a second provider (planned, not v1-critical), or
  - Add a MockProvider for offline testing (planned)

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

### Issue B (DEFERRED): yfinance rate-limit
Not a code bug. Documented above. To address robustly, add a 2nd price
provider (e.g. Stooq) — captured as a follow-up, not v1-critical for
the agent layer.
