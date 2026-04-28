# Best Practices — Price Predictor Project

> Living document. Add to it whenever we discover a new gotcha or pattern.
>
> Items marked **[VERIFY]** are based on training-data knowledge and should be
> double-checked against current official docs (especially for fast-moving
> libraries like ADK).
>
> Items marked **[CONFIRMED FROM SOURCE]** were verified by reading the
> installed package source code in this project.

---

## Table of Contents

1. [Universal Principles](#universal-principles)
2. [Google ADK 1.31](#google-adk-131)
3. [LiteLLM](#litellm)
4. [Pydantic v2 / pydantic-settings](#pydantic-v2--pydantic-settings)
5. [loguru](#loguru)
6. [httpx](#httpx)
7. [pandas / pandas-ta](#pandas--pandas-ta)
8. [yfinance](#yfinance)
9. [typer](#typer)
10. [pytest / pytest-asyncio / pytest-cov](#pytest--pytest-asyncio--pytest-cov)
11. [ruff](#ruff)
12. [uv](#uv)
13. [Project Conventions (Cross-Cutting)](#project-conventions-cross-cutting)
14. [How to Use This Doc](#how-to-use-this-doc)

---

## Universal Principles

### Critical patterns (DO)
- ✅ **Zen of Python** (`import this`) — applies to ALL languages
- ✅ **Type hints** on every public function signature (helps IDEs, mypy, AND ADK schema generation)
- ✅ **Docstrings** on every public function (Google style: Args / Returns / Raises)
- ✅ **DRY** — extract helpers when the same logic appears 3+ times (rule of three)
- ✅ **YAGNI** — build only what the current iteration needs; don't speculate
- ✅ **SOLID** — but pragmatically; don't over-engineer for a 5-file project
- ✅ **Files under 600 lines** — split into cohesive submodules above that
- ✅ **Explicit > implicit** — pass dependencies as args, don't reach for globals
- ✅ **Errors as values** for tools (return error dict); exceptions for true bugs
- ✅ **Commit often** — every passing iteration gets a commit

### Anti-patterns (DON'T)
- ❌ Premature abstraction (wait for the third use case)
- ❌ Module-level side effects (no I/O, no API calls at import time)
- ❌ Mutable default arguments (`def f(x=[]):` — classic Python footgun)
- ❌ `from x import *` (pollutes namespace, breaks tooling)
- ❌ Catching bare `Exception` and swallowing (always re-raise or log)

---

## Google ADK 1.31

### Overview
ADK (Agent Development Kit) is Google's framework for building LLM-powered
agents. We use ADK 1.31.1 with the `LiteLlm` model adapter to use Groq instead
of Gemini.

### Critical patterns (DO)

#### Tools
- ✅ **No default values on tool parameters** [CONFIRMED FROM SOURCE]
  - ADK strips defaults from the JSON schema sent to the LLM
  - The LLM literally never sees your defaults
  - Make params required; document defaults in docstring + agent instruction
  - File: `google/adk/tools/_automatic_function_calling_util.py` —
    `_remove_default(schema)` removes them
- ✅ **Return `dict` (not `str`) from tools** [VERIFY]
  - LLMs handle structured data better
  - Use `{"status": "success", ...}` / `{"status": "error", "error_message": ...}` shape
  - Add additional fields as needed (timestamps, IDs, payloads)
- ✅ **Type hints on every tool parameter** — used to generate the JSON schema
- ✅ **Comprehensive docstrings on tools** — this is the LLM's user manual
  - One-line summary
  - Args section: describe each parameter (especially valid values, defaults)
  - Returns section: describe the dict shape
- ✅ **Errors return strings, not exceptions** in the dict's `error_message` field
  - LLMs can react to "I got an error string back"; uncaught exceptions break the run
- ✅ **Wrap agents in factory functions** — `make_xxx_agent() -> LlmAgent`
  - Lazy instantiation
  - Easier to test and parameterize
- ✅ **Use `LiteLlm` adapter** for non-Gemini models
  - `from google.adk.models.lite_llm import LiteLlm`
  - Pass `api_key` explicitly
- ✅ **Use `InMemorySessionService`** for dev/tests; switch to
  `DatabaseSessionService` or `VertexAiSessionService` for prod
  [CONFIRMED FROM SOURCE: docstring says "not suitable for multi-threaded production"]
- ✅ **Async-first** — use `runner.run_async()` not `runner.run()` in production
- ✅ **One Runner per app** — sessions are managed per-runner
- ✅ **Stream events** with `async for event in runner.run_async(...)`

#### Agents
- ✅ Required fields: `name`, `model`, `instruction`, `tools` (and `description`
  if used as a sub-agent)
- ✅ `name` should be lowercase_snake_case
- ✅ `description` is a one-liner (used by parent agents to decide delegation)
- ✅ `instruction` is the system prompt (3–10 sentences usually); supports
  template placeholders `{var_name}` resolved from session state
- ✅ Use `static_instruction` for content that never changes (better caching)
- ✅ Use `output_key` to save the final response into session state for
  downstream agents to read

### Anti-patterns (DON'T)
- ❌ Tool params with default values (LLM never sees them — useless)
- ❌ Tools that return raw strings (use dicts with `status` field)
- ❌ Tools that raise exceptions to the LLM (return error dicts instead)
- ❌ Tools without type hints (schema generation fails or is incorrect)
- ❌ Tools without docstrings (LLM has no idea what the tool does)
- ❌ Module-level `agent = LlmAgent(...)` (creates at import; use factory)
- ❌ Sync `runner.run()` in production (blocks event loop)
- ❌ `InMemorySessionService` in production multi-threaded apps
- ❌ `global_instruction` field — DEPRECATED in 1.31, use `GlobalInstructionPlugin` [CONFIRMED FROM SOURCE]

### Conventions for THIS project
- 🎯 **Model selection lives in env, NOT code.**
  - `.env` defines `PRIMARY_MODEL` + `SECONDARY_MODEL` (LiteLLM `provider/model` format)
  - `settings.primary_model` / `settings.secondary_model` expose them
  - **Required — no defaults in `settings.py`**. Same pattern as API keys.
  - Why: dual sources of truth (defaults in code AND env file) cause drift.
    The `.env.example` documents valid values; `.env` provides them.
  - Reasoning: Twelve-Factor App principle — config in environment, code is config-agnostic
- 🎯 Recommended primary: `groq/llama-3.3-70b-versatile` (fast, free tier)
- 🎯 Recommended secondary: `gemini/gemini-2.5-flash` (note: needs `max_tokens >= 100`
  for thinking budget)
- 🎯 **Model factory pattern** (see `src/price_predictor/llm/factory.py`):
  - ONE function: `make_model(model_name: str) -> LiteLlm`
  - Provider auto-detected from `provider/model` prefix
  - **Factory knows nothing about model names** — it only knows how to build
    a model from a name passed in. Defaults live in `settings`, not factory.
  - Add a new provider = add ONE entry to `_API_KEY_GETTERS` map
  - **No convenience wrappers** like `make_groq_model()` — use
    `make_model(settings.primary_model)` instead (one function, fewer abstractions)
- 🎯 Agents pull model from settings:
  ```python
  from config.settings import settings
  from price_predictor.llm.factory import make_model
  ...
  model = make_model(settings.primary_model)   # most agents
  model = make_model(settings.secondary_model) # reviewer/critic agents
  ```
- 🎯 Tool dict shape:
  ```python
  # Success
  {"status": "success", "<domain_field>": <value>, ...}
  # Error
  {"status": "error", "error_message": "human-readable description"}
  ```
- 🎯 Agent name format: lowercase_snake_case (`hello_agent`, `analyst_agent`)
- 🎯 **Agent identity rule: file name = factory name (minus `make_`) = agent's `name` attribute**
  - File: `agents/<name>_agent.py`
  - Factory: `make_<name>_agent() -> LlmAgent`
  - Agent `name=`: `"<name>_agent"`
  - Example: `agents/hello_agent.py` → `make_hello_agent()` → `name="hello_agent"`
- 🎯 All agents go in `src/price_predictor/agents/`
- 🎯 Reusable tool functions go in domain modules (`analysis/`, `data/`, etc.),
  imported by agent files

### Things to verify
- [VERIFY] ADK 1.31 may have new features not covered here — check
  https://google.github.io/adk-docs/
- [VERIFY] Per-arg parameter descriptions: ADK source says "Do not support
  parameter description for now" — argument descriptions go in the function
  docstring and are extracted as a single block

### References
- Official docs: https://google.github.io/adk-docs/
- Source on disk: `.venv/lib/python3.13/site-packages/google/adk/`
- Tutorials: https://google.github.io/adk-docs/tutorials/

---

## LiteLLM

### Overview
LiteLLM normalizes 100+ LLM providers behind one OpenAI-style API. We use it
both directly (`from litellm import completion`) and via ADK's `LiteLlm` adapter.

### Critical patterns (DO)
- ✅ **Provider/model naming**: `groq/llama-3.3-70b-versatile`,
  `gemini/gemini-2.5-flash`, `openai/gpt-4o`, etc.
- ✅ **Pass `api_key` explicitly** when calling `completion(...)`
  - Avoids hidden dependency on `os.environ`
  - Easier to swap keys per call
- ✅ **Use `acompletion` (async)** for production agent code
- ✅ **Use `completion` (sync)** for smoke tests / one-shot scripts
- ✅ **Set `max_tokens`** explicitly — protects against runaway token usage
  - Gemini 2.5+ models need `max_tokens >= 100` to leave room for thinking tokens
- ✅ **Catch `RateLimitError`** and back off / retry
- ✅ **Response shape**: `response.choices[0].message.content` (always)
- ✅ **`finish_reason`**: check it — `"stop"` = clean, `"length"` = truncated,
  `"tool_calls"` = wants to call a tool

### Anti-patterns (DON'T)
- ❌ Setting `max_tokens=10` for Gemini 2.5+ (all tokens go to thinking, content empty)
- ❌ Asserting LLM responses match exact strings (non-deterministic even with `temperature=0`)
- ❌ Storing API keys in code (use `settings`)

### Conventions for THIS project
- 🎯 Model defaults from `make_groq_model()` factory in `llm/factory.py`
- 🎯 API keys always sourced from `config.settings` (never `os.environ`)
- 🎯 Smoke tests use sync `completion`; agent code uses async via `runner.run_async()`

### Things to verify
- [VERIFY] LiteLLM model names change occasionally — list available models with
  the provider's API if a model is suddenly 404'ing

### References
- Docs: https://docs.litellm.ai/
- Provider list: https://docs.litellm.ai/docs/providers

---

## Pydantic v2 / pydantic-settings

### Critical patterns (DO)
- ✅ **`SecretStr` for secrets** — masks in repr, requires `.get_secret_value()` to extract
- ✅ **`Field(default=..., description=..., min_length=..., ...)`** for rich field config
- ✅ **`field_validator` for custom validation** with `@classmethod` decorator
- ✅ **`model_validator(mode="after")`** for cross-field validation
- ✅ **`BaseSettings + SettingsConfigDict(env_file=".env")`** for config classes
- ✅ **Module-level singleton instance** for settings: `settings = Settings()` at
  module bottom
- ✅ **Type hints are validation rules** — Pydantic enforces them at runtime
- ✅ **Use `Annotated[type, Field(...)]`** when you need both type and metadata
- ✅ **No defaults for project-specific config** (API keys, model choice, DB urls).
  Required → fails fast at startup with clear error if env var missing.
- ✅ **Defaults OK for universal config** (log level, data dir) — same value
  works for everyone.

### Anti-patterns (DON'T)
- ❌ Re-instantiating `Settings()` in different modules (lose singleton benefit)
- ❌ Storing raw strings for secrets (use `SecretStr`)
- ❌ Using v1 syntax (`@validator`, `BaseModel.Config`) — we're on v2
- ❌ Calling `.dict()` or `.json()` — use `.model_dump()` / `.model_dump_json()`

### Conventions for THIS project
- 🎯 Single `Settings` class in `config/settings.py`
- 🎯 Singleton: `settings = Settings()` exported from the same module
- 🎯 Secrets always `SecretStr`; access with `.get_secret_value()` only when calling APIs
- 🎯 `.env` is gitignored; provide `.env.example` for onboarding

### References
- Pydantic v2 docs: https://docs.pydantic.dev/latest/
- pydantic-settings: https://docs.pydantic.dev/latest/concepts/pydantic_settings/

---

## loguru

### Critical patterns (DO)
- ✅ **Configure ONCE at app startup** — call `setup_logging()` from main entry points
- ✅ **Three handlers**: console (colorized), file (rotating), errors-only file
- ✅ **`logger.bind(stock="RELIANCE")`** for context that follows the logger
- ✅ **`logger.exception("...")`** inside `except` — auto-includes traceback
- ✅ **Rotation**: by size (`"10 MB"`) or time (`"1 day"`)
- ✅ **Retention**: keep N days/weeks (`"14 days"`)
- ✅ **Compression**: `"gz"` for old logs to save space
- ✅ **`enqueue=True`** for thread/process safety in concurrent code

### Anti-patterns (DON'T)
- ❌ Calling `setup_logging()` at module import time (side effect)
- ❌ Mixing `logging` (stdlib) and `loguru` — pick one (we use loguru)
- ❌ Logging secrets (passwords, API keys) — sanitize before logging
- ❌ Using `print()` for anything that should be persisted

### Conventions for THIS project
- 🎯 `setup_logging()` lives in `src/price_predictor/logging_config.py`
- 🎯 Three sinks: console / `predictor.log` (10MB rotation, 14d retention) / `errors.log`
- 🎯 Stock-tagged logger via `get_stock_logger("TICKER")` helper
- 🎯 Default log level from `settings.log_level`

### References
- Docs: https://loguru.readthedocs.io/

---

## httpx

### Critical patterns (DO)
- ✅ **Use `httpx.AsyncClient`** for async code (default in this project)
- ✅ **Always set timeouts** — `timeout=httpx.Timeout(10.0)` minimum
- ✅ **Reuse clients** — one `AsyncClient` per app, not per request (connection pooling)
- ✅ **Use `async with httpx.AsyncClient() as client:`** for short-lived scripts
- ✅ **`response.raise_for_status()`** to convert 4xx/5xx into exceptions
- ✅ **Catch `httpx.HTTPStatusError`** (server returned bad code) and
  `httpx.RequestError` (network/timeout) separately
- ✅ **Set `User-Agent`** — many APIs require it

### Anti-patterns (DON'T)
- ❌ No timeout (default is `5s` for connect, but you should set explicitly)
- ❌ Creating new client per request (defeats pooling)
- ❌ Mixing `requests` and `httpx` — pick one (we use `httpx` for async)

### Conventions for THIS project
- 🎯 Async client only (we're an async-first agent project)
- 🎯 Timeout: 30s default for external APIs, configurable per call
- 🎯 Long-lived `AsyncClient` instances managed via dependency injection

### References
- Docs: https://www.python-httpx.org/

---

## pandas / pandas-ta

### Critical patterns (DO)
- ✅ **Vectorized operations** — never `iterrows()` if you can avoid it
- ✅ **Explicit copies** — `df.copy()` when you'll mutate, to silence
  `SettingWithCopyWarning`
- ✅ **DatetimeIndex** for time series — enables `.rolling()`, `.resample()`,
  partial date slicing
- ✅ **`pd.Timestamp` with timezone** — always tz-aware for stock data
  (markets close at specific local times)
- ✅ **`df.empty`** check before operations on potentially-empty DataFrames
- ✅ **Type hints with `pd.DataFrame` / `pd.Series`** in function signatures

### Anti-patterns (DON'T)
- ❌ `iterrows()` in hot loops (slow, type-unsafe)
- ❌ Modifying a slice without `.copy()` (silent bugs via chained assignment)
- ❌ Mixing tz-naive and tz-aware datetimes (raises errors or silent miscalc)
- ❌ Using `df.append()` — DEPRECATED, use `pd.concat([df1, df2])`

### pandas-ta specifics
- ✅ Indicators are **methods on DataFrame** via `.ta.<indicator>()` accessor
  - `df.ta.sma(length=20)`, `df.ta.rsi(length=14)`
- ✅ Or pass a Series: `ta.sma(df["Close"], length=20)`
- ✅ Most indicators **return a Series or DataFrame**; assign back yourself
  - `df["sma_20"] = df.ta.sma(length=20)`
- ✅ **`length` is the lookback window** (varies by indicator)

### Anti-patterns
- ❌ Calling indicators on too-small DataFrames (need at least `length` rows)
- ❌ Forgetting that indicators produce NaN at the start (drop or skip them)

### Conventions for THIS project
- 🎯 All stock DataFrames have a tz-aware DatetimeIndex (Asia/Kolkata for IST stocks)
- 🎯 Column names: lowercase_snake_case (`open`, `close`, `high`, `low`, `volume`,
  `sma_20`, `rsi_14`)
- 🎯 Indicator columns suffixed with their `length` parameter

### References
- pandas: https://pandas.pydata.org/docs/
- pandas-ta: https://github.com/twopirllc/pandas-ta

---

## yfinance

### Critical patterns (DO)
- ✅ **`.NS` suffix** for NSE stocks (`RELIANCE.NS`, `TCS.NS`)
- ✅ **`.BO` suffix** for BSE stocks
- ✅ **Cache results** — yfinance is slow and rate-limited
- ✅ **Handle empty DataFrames** — yfinance returns empty (not error) on bad ticker
- ✅ **Use `auto_adjust=True`** in newer versions (default changed)
- ✅ **`period`** for relative ranges (`"1y"`, `"6mo"`); **`start`/`end`** for absolute
- ✅ **Wrap in retry logic** — Yahoo's API is flaky

### Anti-patterns (DON'T)
- ❌ Calling `yf.download()` in tight loops without caching (rate limit / IP ban)
- ❌ Trusting yfinance for real-time data (it's delayed 15+ min)
- ❌ Using yfinance for production trading (TOS prohibits it; for research only)

### Conventions for THIS project
- 🎯 All Indian stock fetches use `.NS` suffix
- 🎯 Local cache layer (`data/cache/<ticker>_<period>.parquet`) — TTL-based
- 🎯 Wrap in `data/yfinance_client.py` with retry + cache

### References
- Docs: https://github.com/ranaroussi/yfinance

---

## typer

### Critical patterns (DO)
- ✅ **Type hints on CLI args** — typer infers everything from them
- ✅ **`Annotated[type, typer.Option(...)]`** for help/aliases/validators
- ✅ **Sub-apps** for grouping commands (`app.add_typer(predict_app, name="predict")`)
- ✅ **`typer.echo()` / `typer.secho()`** for output (not `print`)
- ✅ **`typer.Exit(code=N)`** for explicit exit codes

### Anti-patterns (DON'T)
- ❌ `print()` in CLI commands (no color, no piping support)
- ❌ Untyped CLI args (typer can't generate the parser correctly)

### Conventions for THIS project
- 🎯 One root `typer.Typer()` in `cli/main.py`
- 🎯 Sub-apps per domain (`cli/predict.py`, `cli/backtest.py`, etc.)
- 🎯 All commands accept `--verbose` and `--config-file` global options

### References
- Docs: https://typer.tiangolo.com/

---

## pytest / pytest-asyncio / pytest-cov

### Critical patterns (DO)
- ✅ **File naming**: `test_*.py`
- ✅ **Function naming**: `test_*`
- ✅ **Fixtures** for shared setup (`@pytest.fixture`)
- ✅ **`conftest.py`** for fixtures shared across multiple test files
- ✅ **`@pytest.mark.parametrize`** for testing variations of the same logic
- ✅ **Custom markers** for categorization: `@pytest.mark.integration`,
  `@pytest.mark.slow`
- ✅ **Register markers in `pyproject.toml`** to silence warnings
- ✅ **`asyncio_mode = "auto"`** for pytest-asyncio (treats `async def test_*` automatically)
- ✅ **`assert x, "helpful message"`** for failure context
- ✅ **`pytest.raises(ExpectedException)`** for testing error paths
- ✅ **Integration tests behind a marker**: `pytest -m "not integration"` for fast loop

### Anti-patterns (DON'T)
- ❌ Tests with implicit dependencies (use fixtures)
- ❌ Tests that share state (each test should be isolated)
- ❌ Asserting LLM output matches exact strings (non-deterministic)
- ❌ Tests requiring network without `@pytest.mark.integration`

### Conventions for THIS project
- 🎯 `tests/` is flat (no `tests/unit/`, `tests/integration/` split)
- 🎯 Integration tests marked `@pytest.mark.integration`
- 🎯 Fast-loop command: `uv run pytest -m "not integration"`
- 🎯 Coverage minimum: aim for 80% on non-agent code; agents harder to unit-test

### References
- pytest: https://docs.pytest.org/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/
- pytest-cov: https://pytest-cov.readthedocs.io/

---

## ruff

### Critical patterns (DO)
- ✅ **Run `uv run ruff check` before every commit**
- ✅ **`ruff check --fix`** for auto-fixable issues
- ✅ **`ruff format`** for formatting (replaces black)
- ✅ **Configure rule selection in `pyproject.toml`** under `[tool.ruff.lint]`
- ✅ **Configure isort** under `[tool.ruff.lint.isort]` — set
  `known-first-party` for our packages

### Anti-patterns (DON'T)
- ❌ Disabling rules globally — disable per line with `# noqa: <rule>` if needed
- ❌ Mixing ruff + black + isort (ruff replaces both)

### Conventions for THIS project
- 🎯 Ruleset: `E, W, F, I, B, UP, SIM, RUF` (errors, warnings, pyflakes, isort, bugbear, pyupgrade, simplify, ruff-specific)
- 🎯 Ignored: `E501` (line length — formatter handles it)
- 🎯 Line length: 100
- 🎯 Target Python: 3.13
- 🎯 First-party packages: `config`, `price_predictor`
- 🎯 Pre-commit hook (future iteration)

### References
- Docs: https://docs.astral.sh/ruff/

---

## uv

### Critical patterns (DO)
- ✅ **`uv add <pkg>`** to add a dependency (updates `pyproject.toml` + `uv.lock`)
- ✅ **`python scripts/freeze.py`** after `uv add` — regenerates `requirements.txt` from
  `uv.lock` so pip-only installers (e.g., on Walmart-internal systems without uv)
  stay in sync. Commit both `uv.lock` AND `requirements.txt`.
- ✅ **`uv add --dev <pkg>`** for dev-only deps (pytest, ruff, etc.)
- ✅ **`uv sync`** to install from lockfile (CI / fresh clones)
- ✅ **`uv run <cmd>`** to execute in the project's venv (auto-syncs)
- ✅ **`uv venv`** to create a fresh venv for a new project
- ✅ **Walmart index URL** for all installs:
  `--index-url https://pypi.ci.artifacts.walmart.com/artifactory/api/pypi/external-pypi/simple --allow-insecure-host pypi.ci.artifacts.walmart.com`
- ✅ **Commit `uv.lock`** to git (reproducible builds)

### Anti-patterns (DON'T)
- ❌ `pip install` directly into the venv (bypasses lockfile)
- ❌ Editing `pyproject.toml` deps by hand for additions (use `uv add`)
- ❌ Forgetting `uv.lock` in git commits
- ❌ Editing `requirements.txt` by hand (it's auto-generated — run `freeze.py`)

### Conventions for THIS project
- 🎯 All deps managed via `uv add` (no manual edits)
- 🎯 Dev deps separated (`pytest`, `ruff`, `pytest-cov`, `pytest-asyncio`)
- 🎯 Project venv at `./.venv/` (not `~/.code-puppy-venv` — that's CodePuppy's own)

### References
- Docs: https://docs.astral.sh/uv/

---

## Project Conventions (Cross-Cutting)

### File / package layout
```
price_predictor/
├── config/                # Settings (Pydantic-settings)
├── src/price_predictor/
│   ├── agents/            # ADK agents (one per file: <name>_agent.py)
│   ├── analysis/          # Pure-function analytics (technical indicators, etc.)
│   ├── backtest/          # Backtest engine
│   ├── cli/               # Typer CLI commands
│   ├── concurrency/       # Async helpers
│   ├── data/              # Data fetchers (yfinance, GDELT, etc.)
│   ├── kb/                # Knowledge base (RAG)
│   ├── llm/               # Model factories
│   ├── prediction/        # Prediction schemas + post-processing
│   ├── tracking/          # Predictions vs actuals tracking
│   ├── utils/             # Cross-cutting helpers (see rules below)
│   └── logging_config.py
├── tests/
├── data/
│   ├── cache/             # API response cache (gitignored)
│   ├── kb/                # Vector DB (gitignored)
│   ├── logs/              # Log files (gitignored)
│   └── outputs/           # Run outputs (gitignored)
├── docs/                  # This file + ADRs
├── scripts/               # One-off scripts (e.g., freeze.py)
└── pyproject.toml
```

### `__init__.py` policy
- 🎯 **Default: empty.** Do not re-export anything by default.
- 🎯 **Exception 1**: top-level package `__init__.py` may export a `main()` entry point
  for the `[project.scripts]` console script.
- 🎯 **Exception 2**: a sub-package may re-export its public API IF the package has
  many internal modules and a clear "public face" (e.g., `from .factory import
  make_groq_model` in `llm/__init__.py`). Don't do this preemptively — only when
  the import path becomes annoying.
- 🎯 NEVER use `from .x import *` in `__init__.py` (breaks tooling, hides surface).

### `utils/` package — guardrails to prevent dumping-ground syndrome
- 🎯 **Sub-modules MUST be domain-named**: `utils/dates.py`, `utils/retries.py`,
  `utils/cache.py`. **BANNED**: `utils/helpers.py`, `utils/misc.py`, `utils/utils.py`.
- 🎯 **Single Responsibility per file** — one concern each.
- 🎯 **Code review question**: "Could this go in an existing domain package
  (analysis, data, llm)?" Default YES; only fall back to `utils/` if truly
  cross-cutting (used by 2+ other packages).
- 🎯 **If a utility grows past ~200 lines, promote it** to its own top-level
  package (e.g., `utils/cache.py` → `cache/`).
- 🎯 **No agents in utils.** Agents go in `agents/`.
- 🎯 **No tools in utils** unless they wrap genuinely-shared logic. Tools that
  belong to one agent stay in that agent's file.

### Tool placement convention
- 🎯 **One agent uses it**: tool function lives IN the agent file (e.g.,
  `get_current_time` in `hello_agent.py`).
- 🎯 **Multiple agents use it**: extract to the relevant domain package (e.g.,
  `data/yfinance_tools.py`, `analysis/indicators.py`).
- 🎯 **Cross-cutting and reusable**: `utils/<domain>.py`.
- 🎯 Never duplicate a tool function — extract on the second use case.

### Sync vs async tools (ADK)
- 🎯 **Sync tool**: pure-CPU work, fast (< 100ms), no I/O. Default choice.
- 🎯 **Async tool**: any network or disk I/O — use `async def` + `await`.
- 🎯 ADK supports both natively; choose based on what the function actually does.
- 🎯 **Don't make sync tools async "just in case"** — async has overhead and
  forces all callers to await.

### ADK Runner / Session convention (will codify in iteration 1.3)
- 🎯 **One Runner per app**, created in `cli/main.py` (or test fixture).
- 🎯 **One SessionService per app**: `InMemorySessionService` for dev/tests;
  `DatabaseSessionService("sqlite:///data/sessions.db")` for prod.
- 🎯 **Session per conversation** — one session_id per user interaction sequence.
- 🎯 **app_name** matches the project name: `"price_predictor"`.

### Imports
- ✅ Standard library first
- ✅ Third-party second
- ✅ First-party last (`config.*`, `price_predictor.*`)
- ✅ Blank lines between groups (ruff isort handles this)
- ✅ Imports at top of file (NEVER inside functions, except for circular-import workarounds)

### Naming
- 🎯 Modules / files: `lowercase_snake_case.py`
- 🎯 Classes: `PascalCase`
- 🎯 Functions / variables: `lowercase_snake_case`
- 🎯 Constants: `UPPER_SNAKE_CASE`
- 🎯 Private: leading underscore (`_helper`)
- 🎯 Agent files: filename matches agent name exactly (`hello_agent.py`, not `hello.py`), expose `make_<name>_agent()`

### Async vs sync
- 🎯 Agent code: async (use `acompletion`, `runner.run_async`)
- 🎯 Tools: can be sync OR async (ADK supports both)
- 🎯 Smoke tests / one-shot scripts: sync OK
- 🎯 Data fetchers: async (httpx)

### Errors
- 🎯 Tools: return `{"status": "error", "error_message": ...}`
- 🎯 Library code: raise specific exceptions (`ValueError`, `KeyError`)
- 🎯 CLI: catch + show user-friendly message; non-zero exit code

### Testing
- 🎯 Every PUBLIC function gets a test (eventually)
- 🎯 Integration tests behind `@pytest.mark.integration`
- 🎯 Fixtures shared via `conftest.py`
- 🎯 Coverage tracked via `pytest-cov`; report in CI

### Git
- 🎯 Commit per passing iteration
- 🎯 Conventional commit prefix: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`
- 🎯 Never force-push (Walmart rule)

---

## How to Use This Doc

### When specifying a new feature
1. Identify which libraries are involved
2. Re-read the relevant sections
3. Apply the patterns; flag any deviations explicitly
4. If you find a NEW pattern or gotcha, ADD IT HERE

### When reviewing code
1. Check against the relevant library section
2. Flag anti-patterns
3. Suggest the project convention as the fix

### When discovering a new gotcha
1. Add a new bullet under the relevant section
2. Mark with [CONFIRMED FROM SOURCE] if you verified, [VERIFY] if uncertain
3. Commit the doc update along with whatever code triggered the discovery

### When in doubt
- Prefer the convention to a one-off solution
- Prefer the documented approach over cleverness
- Read the source code (it's right there in `.venv/lib/python3.13/site-packages/`)

---

## Changelog

- **2026-04-28** — Initial version. Covers: Universal, ADK, LiteLLM, Pydantic,
  loguru, httpx, pandas/pandas-ta, yfinance, typer, pytest, ruff, uv. ADK
  best practices verified directly from source code.
- **2026-04-28** — Post-audit refinements:
  - Locked agent identity rule (file = factory = `name` attribute, exact match)
  - Updated ruff conventions to match actual config (B/UP/SIM/RUF added)
  - Documented `requirements.txt` regeneration workflow via `scripts/freeze.py`
  - Added `__init__.py` policy (default empty)
  - Added `utils/` package guardrails (domain-named sub-modules only)
  - Added tool placement convention (per-agent vs domain vs utils)
  - Added sync-vs-async tools guidance
  - Added Runner/Session convention preview
- **2026-04-28** — Locked **model factory pattern**: single `make_model()`
  function + provider dispatch. Model NAMES live in env (`PRIMARY_MODEL` /
  `SECONDARY_MODEL`), exposed via `settings.primary_model` /
  `settings.secondary_model`. Factory has no hardcoded model defaults —
  Twelve-Factor App principle (config in env, not code).
- **2026-04-28** — Removed defaults for `primary_model` / `secondary_model` in
  `settings.py`. Required env vars now (matches API key pattern). Avoids
  dual sources of truth between code and env.
