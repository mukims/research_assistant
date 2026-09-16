"""
Central configuration for the Research Assistant pipeline.

All model names, paths and tunable constants live here so that changing a model
or a path only requires editing one file.
"""

import os

try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except Exception:
    pass

# ─── Roots ───────────────────────────────────────────────────────────────────
# config.py now lives inside the package, so the project root is two levels up.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)

# Everything the pipeline *writes* is anchored here, separate from the source
# tree. Set CITATION_DATA_DIR to a writable path (e.g. /data) on a read-only or
# ephemeral host.
DATA_DIR = os.environ.get("CITATION_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))

# Also load from DATA_DIR/.env if present
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(DATA_DIR, ".env"), override=True)
except ImportError:
    pass


# ─── Env helpers ─────────────────────────────────────────────────────────────

def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# ─── LLM / Embedding Backend ─────────────────────────────────────────────────
# "ollama" (default) talks to a local Ollama daemon. "openai" talks to any
# OpenAI-compatible chat endpoint (including Google Gemini) — set OPENAI_BASE_URL
# and OPENAI_API_KEY, or set GEMINI_API_KEY.
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

LLM_BACKEND     = os.environ.get("LLM_BACKEND")
if LLM_BACKEND is None:
    # When GEMINI_API_KEY is supplied, default to OpenAI-compatible Gemini endpoint
    LLM_BACKEND = "openai" if GEMINI_API_KEY else "ollama"
LLM_BACKEND     = LLM_BACKEND.lower()

# When Gemini is used, default to the official Google OpenAI-compatible endpoint
# and gemini-3.5-flash-lite, but keep EMBED_BACKEND="ollama" by default so existing
# 25k chunks in ChromaDB work without re-indexing.
if (LLM_BACKEND == "openai" or GEMINI_API_KEY) and GEMINI_API_KEY:
    _default_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    _default_llm_model = "gemini-3.5-flash-lite"
    _default_embed_backend = "ollama"
    if os.environ.get("CITATION_LLM_MODEL") in (None, "", "gemma4:e2b"):
        os.environ["CITATION_LLM_MODEL"] = _default_llm_model
    if os.environ.get("CITATION_CHAT_MODEL") in (None, "", "gemma4:e2b"):
        os.environ["CITATION_CHAT_MODEL"] = _default_llm_model
else:
    _default_base_url = "https://router.huggingface.co/v1"
    _default_llm_model = "gemma4:e2b"
    _default_embed_backend = LLM_BACKEND

EMBED_BACKEND   = os.environ.get("CITATION_EMBED_BACKEND", _default_embed_backend).lower()

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", _default_base_url)
if "generativelanguage.googleapis.com" in OPENAI_BASE_URL and GEMINI_API_KEY:
    OPENAI_API_KEY = GEMINI_API_KEY
else:
    OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN") or GEMINI_API_KEY
HF_TOKEN        = os.environ.get("HF_TOKEN") or OPENAI_API_KEY

# ─── Models ──────────────────────────────────────────────────────────────────
LLM_MODEL       = os.environ.get("CITATION_LLM_MODEL", _default_llm_model)
CHAT_MODEL      = os.environ.get("CITATION_CHAT_MODEL", _default_llm_model)
EMBED_MODEL     = os.environ.get("CITATION_EMBED_MODEL", "nomic-embed-text")


# ─── Chat Model Runtime Options ──────────────────────────────────────────────
CHAT_OLLAMA_OPTIONS = {
    "num_ctx": 4096,           # Context window
}
CHAT_WINDOW_TOKENS          = _env_int(
    "CITATION_CHAT_WINDOW_TOKENS",
    32768 if LLM_BACKEND == "openai" else CHAT_OLLAMA_OPTIONS.get("num_ctx", 4096),
)
CHAT_CONDENSE               = _env_bool("CITATION_CHAT_CONDENSE", True)
CHAT_CONTEXT_MAX_CHARS      = _env_int("CITATION_CHAT_CONTEXT_MAX_CHARS", 7000)
CHAT_ANSWER_RESERVE_TOKENS  = _env_int("CITATION_CHAT_ANSWER_RESERVE_TOKENS", 700)
CHAT_PER_DOC_CAP            = _env_int("CITATION_CHAT_PER_DOC_CAP", 3)
CHAT_FOCUS_TOP_K            = _env_int("CITATION_CHAT_FOCUS_TOP_K", 2)
CHAT_TEMPERATURE            = float(os.environ.get("CITATION_CHAT_TEMPERATURE", "0.3"))


# ─── Index version ───────────────────────────────────────────────────────────
# Every on-disk name the index uses is derived from this, so a v2 index (new
# extractor, new chunker, prefixed embeddings) lives beside v1 and never
# opens a v1 file for writing. Default 1: nothing changes until set.
INDEX_VERSION = _env_int("CITATION_INDEX_VERSION", 1)
_INDEX_SUFFIX = "" if INDEX_VERSION == 1 else f"_v{INDEX_VERSION}"

# ─── Vector Database ──────────────────────────────────────────────────────────
VECTORDB_PATH    = os.path.join(DATA_DIR, "physics_vectordb")
COLLECTION_NAME  = f"physics_papers{_INDEX_SUFFIX}"       # detail chunks (stage-2 retrieval)
SUMMARY_COLLECTION_NAME = f"physics_summaries{_INDEX_SUFFIX}"   # one summary per document (stage 1)
BM25_INDEX_PATH  = os.path.join(DATA_DIR, f"bm25_index{_INDEX_SUFFIX}.pkl")

# ─── Directories ──────────────────────────────────────────────────────────────
RAW_DIR          = os.path.join(DATA_DIR, "raw")
DRAFTS_DIR       = os.path.join(DATA_DIR, "drafts")
PULLED_PDFS_DIR  = os.path.join(DATA_DIR, "pulled_pdfs")
# Figure/table crops written during ingestion. Each crop is handed straight to
# the VLM and never read back — only the generated description enters the
# corpus — so this is a debugging artefact, not corpus data, and is safe to
# delete between runs. It previously resolved to ../extracted_data/images, a
# sibling of the project, which put the output outside the repo, outside
# version control and outside any backup taken of it.
IMAGES_DIR       = os.environ.get(
    "CITATION_IMAGES_DIR", os.path.join(DATA_DIR, "images")
)

# ─── Data Files ───────────────────────────────────────────────────────────────
EXTRACTED_CITATIONS_PATH = os.path.join(DATA_DIR, "extracted_citations.json")
DOWNLOADED_JSON_PATH     = os.path.join(DATA_DIR, "downloaded.json")
FAILED_DOWNLOADS_PATH    = os.path.join(DATA_DIR, "failed_downloads.json")
# Agent 0's manifest: one record per research query it has seeded from, keyed
# by the query string. Same crash-safe rewrite-after-each-write pattern as
# downloaded.json.
SEED_PAPERS_PATH         = os.path.join(DATA_DIR, "seed_papers.json")
INGESTED_MANIFEST_PATH   = os.path.join(DATA_DIR, f"ingested{_INDEX_SUFFIX}.json")
PIPELINE_STATUS_PATH     = os.path.join(DATA_DIR, "pipeline_status.json")
INGEST_LOCK_PATH         = os.path.join(DATA_DIR, "ingest.lock")

# ─── Detectron2 / layout detection ──────────────────────────────────────────
# Layout detection (figure/table crops + a VLM description of each) is the
# heaviest, most install-fragile stage: detectron2 is not on PyPI and needs a
# matching torch build. With it OFF, ingestion falls back to text-only
# extraction (PyMuPDF) — which is what the Hugging Face Space runs.
LAYOUT_DETECTION = _env_bool("CITATION_LAYOUT_DETECTION", True)
# May be absent: when the file is not there, layoutparser downloads the
# PubLayNet weights for DETECTRON_CONFIG on first use.
DETECTRON_WEIGHTS = os.environ.get(
    "CITATION_DETECTRON_WEIGHTS", os.path.join(PROJECT_ROOT, "model_final.pth")
)
_default_detectron_config = (
    os.path.join(PROJECT_ROOT, "publaynet_config.yaml")
    if os.path.exists(os.path.join(PROJECT_ROOT, "publaynet_config.yaml"))
    else "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config"
)
DETECTRON_CONFIG = (
    os.environ.get("CITATION_DETECTRON_CONFIG", "").strip()
    or _default_detectron_config
)
DETECTRON_LABEL_MAP = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
DETECTRON_SCORE_THRESH = 0.5

# ─── Ingestion Tunables ───────────────────────────────────────────────────────
CHUNK_MIN_LENGTH         = 10       # Discard text chunks shorter than this
EMBED_BATCH_SIZE         = 1000     # ChromaDB upsert batch size
EMBED_MAX_CHARS          = 4000     # Truncate documents to this length before embedding
SEMANTIC_CHUNKER_TYPE    = "percentile"
SEMANTIC_CHUNKER_AMOUNT  = 90       # 90th percentile breakpoint

# v2 chunker (INDEX_VERSION >= 2): sentence windows packed within a section.
# ~1,200 chars is ~300 tokens; six of them fill ~1,800 tokens of a 4k window.
CHUNK_TARGET_CHARS       = _env_int("CITATION_CHUNK_TARGET_CHARS", 1200)
CHUNK_MAX_CHARS          = _env_int("CITATION_CHUNK_MAX_CHARS", 1800)
CHUNK_MIN_CHARS          = 200      # shorter windows are dropped (captions exempt)
CHUNK_MIN_ALPHA          = 0.6      # alphabetic ratio below this is font-map garbage

# Figure/table handling. The layout pass always crops the image and keeps its
# caption as the searchable text. Set CITATION_FIGURE_VLM=1 to also run a VLM
# description of every crop at ingest time (one model call per figure — off by
# default; crops can be described on demand instead).
FIGURE_VLM = _env_bool("CITATION_FIGURE_VLM", False)

# Per-document summary written to SUMMARY_COLLECTION_NAME at ingest time — the
# stage-1 "is this paper even relevant" index. One model call per paper.
SUMMARY_MODEL            = os.environ.get("CITATION_SUMMARY_MODEL", "") or None  # None → LLM_MODEL
SUMMARY_MAX_CHARS        = _env_int("CITATION_SUMMARY_MAX_CHARS", 8000)

# ─── Agent 5 — Batch Citer ───────────────────────────────────────────────────
# Sentences per citation-need request. One request for a whole draft makes the
# entire run hostage to a single malformed reply; smaller batches confine that
# to the batch. Too small wastes calls, since each one re-sends the framing.
CITATION_CHECK_BATCH_SIZE = 20

# ─── Agent 8 — Verifier (judgement) ──────────────────────────────────────────
# Default citation check model: gemini-3.5-flash-lite when GEMINI_API_KEY is present
_default_judgement_model = "gemini-3.5-flash-lite" if GEMINI_API_KEY else None
JUDGEMENT_MODEL       = os.environ.get("CITATION_JUDGEMENT_MODEL") or _default_judgement_model or None
# The judgement prompt is a rubric, not a generation task, and its regression
# cases assert exact verdicts — sampling makes both meaningless.
JUDGEMENT_TEMPERATURE = 0.0
# Chunks judged per cited source. 1 = judge the single best-matching chunk.
JUDGEMENT_TOP_K       = _env_int("CITATION_JUDGEMENT_TOP_K", 1)
# Evidence escalation. The top JUDGEMENT_TOP_K hit(s), each with
# JUDGEMENT_NEIGHBOUR_WINDOW adjacent chunks, are judged first. If that
# verdict reports it did not see enough (sufficiency != sufficient, or
# Unclear / Does not support), the top JUDGEMENT_ESCALATE_TOP_K hits are
# judged once more. 0 disables the second look. The assembled evidence is
# capped so prompt (~5.4k tokens) + evidence stays inside num_ctx below.
JUDGEMENT_ESCALATE_TOP_K     = _env_int("CITATION_JUDGEMENT_ESCALATE_TOP_K", 3)
JUDGEMENT_NEIGHBOUR_WINDOW   = _env_int("CITATION_JUDGEMENT_NEIGHBOUR_WINDOW", 1)
JUDGEMENT_EVIDENCE_MAX_CHARS = _env_int("CITATION_JUDGEMENT_EVIDENCE_MAX_CHARS", 6000)
# The V1.6 rubric is ~5,400 tokens and a full evidence block adds ~1,500 more,
# so the old 4096 no longer held even the prompt: Ollama truncates it —
# silently, from the tail, where the worked examples and the output schema
# live — and the model answers with a bare code fence that fails to parse.
# 12288 leaves room for the rubric, full evidence and a context block; a test
# pins the invariant. Ignored by the openai backend.
JUDGEMENT_OLLAMA_OPTIONS = {"num_ctx": _env_int("CITATION_JUDGEMENT_NUM_CTX", 12288)}
CITATION_AUDIT_CONTEXTUALIZE_QUERIES = _env_bool("CITATION_AUDIT_CONTEXTUALIZE_QUERIES", True)


# ─── Search Tunables ──────────────────────────────────────────────────────────
RRF_K            = 60               # Reciprocal Rank Fusion constant
DEFAULT_TOP_K    = 3                # Default number of results to return

# Two-stage retrieval: rank documents by summary similarity, keep the top
# DOC_SELECT_K, optionally have the LLM drop the off-topic ones (DOC_GATE),
# then run the detail search only over what survives.
DOC_SELECT_K     = _env_int("CITATION_DOC_SELECT_K", 6)
DOC_GATE         = _env_bool("CITATION_DOC_GATE", True)
# One gate call listing every shortlisted summary instead of one call per
# summary. Falls back to per-summary calls when the reply cannot be aligned.
GATE_BATCHED     = _env_bool("CITATION_GATE_BATCHED", True)

# ─── Synthesis (Tab 1 related work) ───────────────────────────────────────────
# map_reduce: per-paper notes, then one synthesis over the notes (1 + N + 1
# calls). single: one call over per-paper passages. Both at a low
# temperature — the model's default of 1.0 is for creative writing, not for a
# grounded overview. The reduce runs at 8k context (as the judge does) so
# eight papers' notes plus a 600-token answer fit.
SYNTHESIS_MODE                = os.environ.get("CITATION_SYNTHESIS_MODE", "map_reduce").lower()
SYNTHESIS_PER_PAPER_CHUNKS    = _env_int("CITATION_SYNTHESIS_PER_PAPER_CHUNKS", 4)
SYNTHESIS_PER_PAPER_MAX_CHARS = _env_int("CITATION_SYNTHESIS_PER_PAPER_MAX_CHARS", 5000)
SYNTHESIS_TEMPERATURE         = float(os.environ.get("CITATION_SYNTHESIS_TEMPERATURE", "0.2"))
SYNTHESIS_OLLAMA_OPTIONS      = {"num_ctx": 4096}



# ─── Orchestrator Tunables ────────────────────────────────────────────────────
PDF_COOLDOWN_SECONDS     = 30
DRAFT_COOLDOWN_SECONDS   = 2
MANUAL_COOLDOWN_SECONDS  = 5
DEFAULT_WORKERS          = 1

# ─── Contact Address ─────────────────────────────────────────────────────────
# Unpaywall requires a contact email on every request; OpenAlex and Crossref
# use it to route you to their faster "polite" pools. Set UNPAYWALL_EMAIL in
# your environment; the placeholder below is only a fallback so the pipeline
# does not silently send someone else's address.
UNPAYWALL_EMAIL   = os.environ.get("UNPAYWALL_EMAIL", "abcdef_12345@gmail.com")

# ─── Agent 0 — Discoverer ────────────────────────────────────────────────────
# Agent 0 searches a scholarly index whose results come back ranked by
# relevance to a natural-language query and already carry an open-access PDF
# URL — so it never has to resolve a DOI or guess a download location the way
# Agent 2 does for reference chains.
#
# Providers are tried in order until one yields a result whose PDF actually
# downloads. arXiv is first: for a physics tool its preprints are almost always
# what you want and the PDF never 404s, whereas OpenAlex's top hits are often
# paywalled publisher links. OpenAlex (better metadata, no rate limit) and
# Semantic Scholar (keyless pool heavily throttled — set S2_API_KEY) back it up.
SEARCH_PROVIDERS  = os.environ.get(
    "CITATION_SEARCH_PROVIDERS", "arxiv,openalex,semanticscholar"
).split(",")
SEARCH_LIMIT      = 15    # Candidates to rank through looking for an OA PDF

OPENALEX_URL      = "https://api.openalex.org/works"
# OpenAlex gives requests that supply a contact address the faster "polite"
# pool. Reuses the same address Unpaywall needs.
OPENALEX_MAILTO   = os.environ.get("OPENALEX_MAILTO", UNPAYWALL_EMAIL)

S2_API_KEY        = os.environ.get("S2_API_KEY")
S2_SEARCH_URL     = "https://api.semanticscholar.org/graph/v1/paper/search"

# ─── Agent 1 — Extractor (GROBID) ───────────────────────────────────────────
# Default is the public GROBID instance running as a Hugging Face Space, so the
# pipeline works with no local Java service. Point this at http://localhost:8070
# when running your own GROBID (faster, private, no shared rate limit).
GROBID_SERVER            = os.environ.get("GROBID_SERVER","http://localhost:8070")
GROBID_BATCH_CONCURRENCY = _env_int("GROBID_BATCH_CONCURRENCY", 2)
GROBID_DOCKER_IMAGE      = os.environ.get("GROBID_DOCKER_IMAGE", "grobid/grobid:0.8.1")
GROBID_CONTAINER_NAME    = os.environ.get("GROBID_CONTAINER_NAME", "grobid")
GROBID_START_COMMAND     = os.environ.get("GROBID_START_COMMAND", "")
GROBID_JAR_PATH          = os.environ.get("GROBID_JAR_PATH", "")

# v2 ingestion sends every PDF to processFulltextDocument. One TEI per PDF is
# cached here — the same directory Agent 1 writes, so a seed paper it has
# already processed is never sent twice.
GROBID_TEI_DIR           = os.path.join(RAW_DIR, "grobid_output")
GROBID_FULLTEXT_TIMEOUT  = _env_int("GROBID_FULLTEXT_TIMEOUT", 300)

# ─── Agent 2 — Fetcher ───────────────────────────────────────────────────────
MAX_CITATION_LEN   = 500   # Skip citations longer than this (likely malformed)
ARXIV_RATE_LIMIT   = 3     # Seconds between arXiv requests
UNPAYWALL_SLEEP    = 0.5   # Courtesy sleep after Unpaywall downloads
# ─── Rendering / DPI ─────────────────────────────────────────────────────────
PDF_RENDER_DPI = 72
