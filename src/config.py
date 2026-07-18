import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER      = os.getenv("LLM_PROVIDER",   "nvidia")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY",  "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY",  "")
NVIDIA_API_KEY    = os.getenv("NVIDIA_API_KEY",  "")
NVIDIA_BASE_URL   = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

PLANNER_MODEL   = os.getenv("PLANNER_MODEL",   "openai/gpt-oss-120b")
RERANKER_MODEL  = os.getenv("RERANKER_MODEL",  "openai/gpt-oss-120b")
SCORER_MODEL    = os.getenv("SCORER_MODEL",    "openai/gpt-oss-120b")
REFLECTOR_MODEL = os.getenv("REFLECTOR_MODEL", "openai/gpt-oss-120b")
NAIVE_MODEL     = os.getenv("NAIVE_MODEL",     "openai/gpt-oss-120b")

# EMBED_PROVIDER defaults to match LLM_PROVIDER when not explicitly set.
# nvidia uses their own OpenAI-compatible embedding endpoint.
_default_embed_provider = LLM_PROVIDER if LLM_PROVIDER in ("nvidia", "openai", "gemini") else "gemini"
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", _default_embed_provider)

_default_embed_model = {
    "nvidia": "nvidia/nv-embedqa-e5-v5",
    "openai": "text-embedding-3-small",
    "gemini": "gemini-embedding-2",
}.get(EMBED_PROVIDER, "nvidia/nv-embedqa-e5-v5")
EMBED_MODEL = os.getenv("EMBED_MODEL", _default_embed_model)

BM25_TOP_K     = int(os.getenv("BM25_TOP_K",    "8"))
VECTOR_TOP_K   = int(os.getenv("VECTOR_TOP_K",  "8"))
RRF_K          = int(os.getenv("RRF_K",         "60"))
RERANK_TOP_N   = int(os.getenv("RERANK_TOP_N",  "3"))
LOW_RETRIEVAL_FLOOR = float(os.getenv("LOW_RETRIEVAL_FLOOR", "0.02"))

CONFIDENCE_FLOOR = float(os.getenv("CONFIDENCE_FLOOR", "0.7"))

# Score at or above which a requirement counts as met. Used both to derive the
# "matched" status and to accept a required requirement, so the badge shown in
# the UI and the verdict can never disagree.
REQUIRED_ACCEPT_SCORE = float(os.getenv("REQUIRED_ACCEPT_SCORE", "0.75"))
# Score below which a requirement counts as missing rather than partially met.
PARTIAL_MATCH_SCORE = float(os.getenv("PARTIAL_MATCH_SCORE", "0.25"))


def validate_config() -> None:
    if LLM_PROVIDER == "nvidia" and not NVIDIA_API_KEY:
        raise EnvironmentError("NVIDIA_API_KEY is not set.")
    if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
        raise EnvironmentError("GEMINI_API_KEY is not set.")
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    if LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
    if EMBED_PROVIDER == "nvidia" and not NVIDIA_API_KEY:
        raise EnvironmentError("NVIDIA_API_KEY is not set (required for embeddings).")
    if EMBED_PROVIDER == "gemini" and not GEMINI_API_KEY:
        raise EnvironmentError("GEMINI_API_KEY is not set (required for embeddings).")
    if EMBED_PROVIDER == "openai" and not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY is not set (required for embeddings).")
