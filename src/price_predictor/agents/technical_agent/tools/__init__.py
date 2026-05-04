"""ADK tool wrappers for the technical_agent.

Each tool is an async function the LLM can call. They follow a uniform
shape:

    async def get_<cluster>(ticker: str, sensitivity: str = "standard") -> dict

INTERNAL FLOW (same for every tool)
===================================
    1. Validate inputs (ticker non-empty, sensitivity is a known preset)
    2. Resolve the cache & fetch ~1y of OHLCV (cache-backed)
    3. Run the cluster's *_snapshot() primitive from analysis/
    4. Run the cluster-specific signal classifier
    5. Build a rationale[] list of human-readable bullet points
    6. Return a uniform dict (see TOOL_RESPONSE_SCHEMA in _types.py)

ERROR CONTRACT
==============
On any failure (bad ticker, network down, insufficient history), the tool
returns a dict with status=error -- it NEVER raises. The LLM needs to see
the error to apologize / retry / suggest alternatives.
"""
