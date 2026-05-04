"""Knowledge base (KB) package.

The KB holds the agent's "world knowledge" -- domain facts that are stable
enough to be data, not API calls. v1 contains the Nifty 50 stock registry.

This is intentionally NOT exposed as ADK tools. KB lookups are knowledge,
not actions; tool calls cost an LLM round-trip and should be reserved for
side-effecting operations or live data fetches.
"""
