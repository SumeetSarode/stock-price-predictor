"""Provider facts, statuses and verdict rules for the free-model sweep. No I/O: everything here is
pure, so the rules that decide "useful / not useful / unknown" can be tested without a network.

Vocabulary (kept deliberately small):

  Candidate.status   TESTABLE       we can call it from here
                     NEEDS_SIGNUP   free, but needs an account/key we do not hold (what would unlock it is recorded)
                     TERMS_BLOCKED  the provider's terms forbid our use; never called
                     NOT_FREE       a paid plan whose marginal cost merely shows as $0
                     NOT_CHAT       by name an embedding/audio/image model; not a text-chat model
                     EXHAUSTED      free but out of quota right now; retry later
                     NO_ENDPOINT    no way to reach it from here (no list, no base URL)

  final verdict      USEFUL         our two pipeline steps passed in EVERY run
                     PARTIAL        some runs passed both steps
                     NOT_USEFUL     tested; it will not work (permanent failure) or is not actually free
                     INCONCLUSIVE   failures look transient (rate limit, timeout, 5xx): retry with --resume
                     UNTESTED       never reached the pipeline test (budget, signup, terms, ...)

"USEFUL" means compatible with our two steps. It does NOT mean accurate: judging accuracy needs the
backtest, not a schema check.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

TESTABLE, NEEDS_SIGNUP, TERMS_BLOCKED, NOT_FREE, NOT_CHAT, EXHAUSTED, NO_ENDPOINT = (
    "TESTABLE", "NEEDS_SIGNUP", "TERMS_BLOCKED", "NOT_FREE", "NOT_CHAT", "EXHAUSTED", "NO_ENDPOINT")
USEFUL, PARTIAL, NOT_USEFUL, INCONCLUSIVE, UNTESTED = (
    "USEFUL", "PARTIAL", "NOT_USEFUL", "INCONCLUSIVE", "UNTESTED")

NON_CHAT = re.compile(
    r"embed|whisper|tts|speech|transcri|moderat|guard|ocr|rerank|image|veo|flux|diffusion|lyria|audio|"
    r"\blive\b|robotics|\baqa\b|computer-use", re.I)
SUBSCRIPTION_PROVIDER = re.compile(r"(^|-)plan(-|$)", re.I)   # kimi-code-plan-cn, alibaba-token-plan, ...
# Free tiers the projects' own docs/trackers say are gone or locked: not leads, whatever a catalog says.
RETIRED_PROVIDERS = {
    "opencode": "free roster locked to the OpenCode client since 2026-09 (403 FreeTierError; freellmapi #1249/#1265, OmniRoute #15152)",
    "sambanova": "free tier permanently retired (402 once the one-time trial credit lapses)",
    "reka": "no free tier for new accounts since 2026-09 (prepaid credits required)",
    "chutes": "every model demands a non-zero balance (402)",
}
TERMS_BLOCKED_PROVIDERS = {"cohere": "Cohere API terms prohibit personal/household use"}

# Failure classes (see sweep_probe.classify_http and model_survey_offvpn.classify)
TRANSIENT = {"rate_limited", "overloaded", "timeout", "network", "empty"}
PERMANENT = {"not_found", "unsupported", "bad_output", "unsupported_params"}


@dataclass(frozen=True)
class Spec:
    base: str                       # OpenAI-compatible base URL
    models_url: str                 # the provider's OWN list of what it serves
    list_auth: str | None           # None = public list; else "bearer" or a header name
    free_rule: str                  # "all" (every served model) | "suffix" (only ids ending ':free')
    free_basis: str                 # why we believe it is free: goes in the report
    pace_s: float                   # seconds between calls to this provider
    budget: int                     # max upstream calls per run (protect YOUR OWN daily quota)
    env: tuple[str, ...]            # .env names that may hold the key
    litellm_prefix: str | None = None   # set when the app's make_model() supports it (production path)
    anonymous: bool = False
    fallback: tuple[str, ...] = ()  # used only if the provider's list cannot be fetched


SPECS: dict[str, Spec] = {
    "google": Spec("https://generativelanguage.googleapis.com/v1beta/openai",
                   "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000", "x-goog-api-key",
                   "all", "Gemini API free tier (rate-limited; unpaid content may be used to improve Google products)",
                   7, 40, ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "gemini"),
    "groq": Spec("https://api.groq.com/openai/v1", "https://api.groq.com/openai/v1/models", "bearer", "all",
                 "Groq free tier (per-model daily token limits)", 3, 40, ("GROQ_API_KEY",), "groq"),
    "openrouter": Spec("https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1/models", None, "suffix",
                       ":free routes (about 50 requests/day without credits)", 4, 30, ("OPENROUTER_API_KEY",),
                       "openrouter"),
    "mistral": Spec("https://api.mistral.ai/v1", "https://api.mistral.ai/v1/models", "bearer", "all",
                    "Mistral free 'Experiment' tier", 2, 40, ("MISTRAL_API_KEY",)),
    "cerebras": Spec("https://api.cerebras.ai/v1", "https://api.cerebras.ai/v1/models", "bearer", "all",
                     "Cerebras free tier", 3, 30, ("CEREBRAS_API_KEY",)),
    "nvidia": Spec("https://integrate.api.nvidia.com/v1", "https://integrate.api.nvidia.com/v1/models", "bearer",
                   "all", "NVIDIA NIM trial/developer tier (read its terms before production use)", 2, 60,
                   ("NVIDIA_NIM_API_KEY", "NVIDIA_API_KEY")),
    "ovh": Spec("https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
                "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/models", None, "all",
                "OVH anonymous tier (about 2 requests/min/model)", 35, 20, (), None, True,
    # base/models_url with "{account}": the key is stored as 'account_id:api_token' (see split_account_key)
    "cloudflare": Spec("https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1",
                       "https://api.cloudflare.com/client/v4/accounts/{account}/ai/models/search?task=Text%20Generation"
                       "&per_page=200", "bearer", "all",
                       "Cloudflare Workers AI free allocation (10,000 neurons/day; big models use it up fast)", 2, 60,
                       ("CLOUDFLARE_API_KEY",)),
    "ollama": Spec("https://ollama.com/v1", "https://ollama.com/v1/models", "bearer", "all",
                   "Ollama Cloud free tier (hourly and weekly usage limits)", 3, 60, ("OLLAMA_API_KEY",)),
    "siliconflow": Spec("https://api.siliconflow.com/v1", "https://api.siliconflow.com/v1/models?type=text&sub_type=chat",
                        "bearer", "all", "SiliconFlow (free models plus sign-up credit; the rest draw on the balance)",
                        2, 60, ("SILICONFLOW_API_KEY",)),
                ("Qwen3.8-27B", "Qwen3.5-397B-A17B", "gpt-oss-120b")),
    "kilo": Spec("https://api.kilo.ai/api/gateway/v1", "https://api.kilo.ai/api/gateway/v1/models", None, "suffix",
                 "Kilo anonymous ':free' routes (200 req/hr/IP; prompts and outputs are logged for training)", 20, 30,
                 (), None, True, ("nvidia/nemotron-3-super-120b-a12b:free", "nvidia/nemotron-3-ultra-550b-a55b:free")),
}

# models.dev provider id -> the name we use, so a provider we already test is never listed as "needs signup"
LEDGER_ALIASES = {"ovhcloud": "ovh", "zhipuai": "zhipu"}

def split_account_key(spec: Spec, key: str | None) -> tuple[Spec, str | None]:
    """Providers whose URLs carry an account id keep their key as 'account:token': -> (spec with real URLs, token)."""
    if not key or "{account}" not in spec.base:
        return spec, key
    account, _, token = key.partition(":")
    return replace(spec, base=spec.base.format(account=account),
                   models_url=spec.models_url.format(account=account)), token


# freellmapi platforms whose keys may be handed to its container when you opt in (--with-keys).
# Cohere is absent on purpose (terms), and so are the retired/locked ones (sambanova, opencode, reka).
# GitHub Models is absent on purpose: it was retired upstream on 30 July 2026, and the gateway now rejects the name
# and aborts the WHOLE container over it (see free_model_sweep_offvpn.boot_freellmapi for the general safeguard).
FREELLMAPI_PLATFORMS = (
    "google", "groq", "cerebras", "electronhub", "experiential", "bai", "anyapi", "radeon", "nvidia", "mistral",
    "openrouter", "cloudflare", "zhipu", "ollama", "pollinations", "llm7", "huggingface", "agnes",
    "siliconflow", "routeway", "bazaarlink", "ainative", "aion", "requesty", "navy", "nara", "sealion",
    "orcarouter", "unorouter", "xkiro", "longcat", "aihorde")

# freellmapi platform name -> extra .env names, beyond the generic {PLATFORM}_API_KEY
KEY_ALIASES = {"google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "nvidia": ("NVIDIA_NIM_API_KEY", "NVIDIA_API_KEY"),
               "huggingface": ("HF_TOKEN", "HUGGINGFACE_API_KEY"), "github": ("GITHUB_MODELS_TOKEN",)}


@dataclass
class Candidate:
    route: str                      # direct | anon | freellmapi | omniroute-base | omniroute-web | strata | ledger
    provider: str
    model: str                      # the id as THIS route names it
    status: str = TESTABLE
    base: str | None = None
    key: str | None = None          # never serialised
    litellm_id: str | None = None   # production path id (gemini/..., groq/...), when the app supports it
    anonymous: bool = False
    note: str = ""
    hint_structured: bool | None = None
    sources: tuple[str, ...] = ()

    @property
    def uid(self) -> str:
        return f"{self.route}|{self.provider}|{self.model}"

    @property
    def budget_key(self) -> str:
        """Direct providers share one allowance per provider; each gateway route gets its own."""
        return self.provider if self.route in ("direct", "anon") else f"{self.route}:{self.provider}"

    @property
    def test_id(self) -> str:
        return self.litellm_id or self.model

    def row(self) -> dict:
        """Everything needed to rebuild this candidate EXCEPT the key, which is never serialised."""
        return {"uid": self.uid, "route": self.route, "provider": self.provider, "model": self.model,
                "status": self.status, "note": self.note, "hint_structured": self.hint_structured,
                "sources": list(self.sources), "base": self.base, "litellm_id": self.litellm_id,
                "anonymous": self.anonymous}

    @classmethod
    def from_row(cls, row: dict, keys: dict[str, str] | None = None) -> Candidate:
        """Inverse of row(): the key comes from `keys` (read from .env), never from the file."""
        return cls(row["route"], row["provider"], row["model"], row.get("status", TESTABLE), row.get("base"),
                   key, row.get("litellm_id"), bool(row.get("anonymous")),
                   row.get("note", ""), row.get("hint_structured"), tuple(row.get("sources", ())))
        key = (keys or {}).get(row["provider"])
        if row["provider"] in SPECS:
            key = split_account_key(SPECS[row["provider"]], key)[1]


def classify_candidate(provider: str, model: str) -> tuple[str, str]:
    """Static triage by name/provider alone: -> (status, note). TESTABLE means 'worth probing'."""
    if provider in TERMS_BLOCKED_PROVIDERS:
        return TERMS_BLOCKED, TERMS_BLOCKED_PROVIDERS[provider]
    if provider in RETIRED_PROVIDERS:
        return NOT_FREE, f"free tier gone: {RETIRED_PROVIDERS[provider]}"
    if SUBSCRIPTION_PROVIDER.search(provider):
        return NOT_FREE, "paid subscription plan; $0 is the marginal cost inside the plan, not a free tier"
    if NON_CHAT.search(model):
        return NOT_CHAT, "by name an embedding/audio/image model, not a text-chat model"
    return TESTABLE, ""


def final_verdict(chat: dict | None, json_row: dict | None, runs: list[dict]) -> tuple[str, str]:
    """Pure verdict for one candidate from its probe rows.

    chat/json_row: {"verdict": PASS|FAIL, "klass": ...} or None.   runs: one dict per pipeline run,
    {"full": bool, "klasses": [failure classes]}. Returns (verdict, one-line reason).
    """
    if runs:
        full = sum(1 for r in runs if r["full"])
        if full == len(runs):
            return USEFUL, f"both steps passed in {full}/{len(runs)} runs"
        if full:
            return PARTIAL, f"both steps passed in {full}/{len(runs)} runs"
        klasses = {k for r in runs for k in r["klasses"]}
        if klasses & TRANSIENT:
            return INCONCLUSIVE, f"no full run; transient failures {sorted(klasses & TRANSIENT)}: retry with --resume"
        return NOT_USEFUL, f"no run passed both steps; failures {sorted(klasses) or ['other']}"
    if chat is None:
        return UNTESTED, "never probed"
    if chat["verdict"] != "PASS":
        k = chat.get("klass", "other")
        if k == "payment":
            return NOT_USEFUL, "not free: the provider demands payment (402)"
        if k in PERMANENT:
            return NOT_USEFUL, f"does not answer chat requests ({k})"
        if k == "auth":
            return INCONCLUSIVE, "auth rejected: key invalid, expired, or an account/verification is needed"
        return INCONCLUSIVE, f"chat probe failed transiently ({k}): retry with --resume"
    note = f"; json_schema probe {json_row['verdict']}" if json_row else ""
    return UNTESTED, f"answers chat but never reached the pipeline test (budget/ordering){note}"
