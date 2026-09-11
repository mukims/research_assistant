"""
Backend-agnostic LLM + embedding access.

Every model call in the pipeline goes through ``chat()``, ``chat_stream()`` or
``get_embeddings()`` here rather than a provider SDK directly, so a deployment
switches backend by setting environment variables (see config.py):

    LLM_BACKEND   = ollama | openai
    EMBED_BACKEND = ollama | openai | huggingface

``ollama`` is the default and needs a local daemon. ``openai`` talks to any
OpenAI-compatible endpoint (OPENAI_BASE_URL / OPENAI_API_KEY) — on Hugging Face
Spaces that is the HF router. ``huggingface`` embeddings use
``huggingface_hub.InferenceClient`` feature-extraction.

Heavy provider SDKs are imported lazily inside each backend so a checkout using
only one of them does not need the others installed.
"""

from dataclasses import dataclass

from research_assistant.config import (
    LLM_BACKEND,
    EMBED_BACKEND,
    LLM_MODEL,
    EMBED_MODEL,
    OPENAI_BASE_URL,
    OPENAI_API_KEY,
    HF_TOKEN,
    INDEX_VERSION,
)
from research_assistant.shared.log import get_logger

logger = get_logger("llm")


@dataclass
class ChatResult:
    content: str
    prompt_tokens: object = "N/A"
    completion_tokens: object = "N/A"


# ─── Chat ────────────────────────────────────────────────────────────────────


def chat(messages, model=None, images=None, temperature=None, options=None) -> ChatResult:
    """Run a chat completion.

    Args:
        messages: list of ``{"role", "content"}`` dicts.
        model:    model id; defaults to config.LLM_MODEL.
        images:   optional list of local image paths for a vision request
                  (attached to the last user message).
        temperature: sampling temperature. None leaves the provider's own
                  default in place, so existing callers are unaffected.
        options:  ollama runtime options (e.g. JUDGEMENT_OLLAMA_OPTIONS).
                  The openai backend has no equivalent and ignores it.
    """
    model = model or LLM_MODEL
    if LLM_BACKEND == "openai":
        return _openai_chat(messages, model, images, temperature)
    if LLM_BACKEND == "ollama":
        return _ollama_chat(messages, model, images, temperature, options)
    raise ValueError(f"Unknown LLM_BACKEND {LLM_BACKEND!r} (expected 'ollama' or 'openai')")


def _ollama_chat(messages, model, images, temperature=None, options=None) -> ChatResult:
    import ollama

    if images:
        messages = list(messages)
        messages[-1] = {**messages[-1], "images": list(images)}

    # Copied, not mutated: callers pass a module-level config dict.
    opts = dict(options) if options else {}
    if temperature is not None:
        opts["temperature"] = temperature

    kwargs = {"model": model, "messages": messages, "stream": False}
    if opts:
        kwargs["options"] = opts

    resp = ollama.chat(**kwargs)
    try:
        content = resp.message.content
    except AttributeError:
        content = resp["message"]["content"]
    return ChatResult(
        content=content,
        prompt_tokens=getattr(resp, "prompt_eval_count", "N/A"),
        completion_tokens=getattr(resp, "eval_count", "N/A"),
    )


def _b64_data_url(path):
    import base64
    import mimetypes

    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as fh:
        return f"data:{mime};base64,{base64.b64encode(fh.read()).decode()}"


def _openai_chat(messages, model, images, temperature=None) -> ChatResult:
    from openai import OpenAI

    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

    if images:
        messages = list(messages)
        last = messages[-1]
        parts = [{"type": "text", "text": last["content"]}]
        parts += [
            {"type": "image_url", "image_url": {"url": _b64_data_url(p)}} for p in images
        ]
        messages[-1] = {**last, "content": parts}

    kwargs = {"model": model, "messages": messages, "stream": False}
    if temperature is not None:
        kwargs["temperature"] = temperature

    resp = client.chat.completions.create(**kwargs)
    usage = getattr(resp, "usage", None)
    return ChatResult(
        content=resp.choices[0].message.content,
        prompt_tokens=getattr(usage, "prompt_tokens", "N/A"),
        completion_tokens=getattr(usage, "completion_tokens", "N/A"),
    )


# ─── Streaming chat ──────────────────────────────────────────────────────────
# Agent 7 streams token-by-token. Kept behind the same backend switch as chat()
# so no agent has to import a provider SDK to do it.


def chat_stream(messages, model=None, options=None):
    """Yield content deltas as plain strings.

    Args:
        messages: list of ``{"role", "content"}`` dicts.
        model:    model id; defaults to config.LLM_MODEL.
        options:  ollama runtime options (e.g. CHAT_OLLAMA_OPTIONS). The
                  OpenAI-compatible backend has no equivalent and ignores it.

    Deliberately not a generator itself, so an unknown backend raises at the
    call rather than on the first ``next()``.
    """
    model = model or LLM_MODEL
    if LLM_BACKEND == "openai":
        return _openai_stream(messages, model)
    if LLM_BACKEND == "ollama":
        return _ollama_stream(messages, model, options)
    raise ValueError(
        f"Unknown LLM_BACKEND {LLM_BACKEND!r} (expected 'ollama' or 'openai')"
    )


def _ollama_stream(messages, model, options):
    import ollama

    kwargs = {"model": model, "messages": messages, "stream": True}
    if options:
        kwargs["options"] = options
    for chunk in ollama.chat(**kwargs):
        try:
            content = chunk.message.content
        except AttributeError:
            content = (chunk.get("message") or {}).get("content")
        if content:
            yield content


def _openai_stream(messages, model):
    from openai import OpenAI

    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True
    )
    for chunk in stream:
        # Role-only opening frames and usage-only terminal frames carry no text.
        if not getattr(chunk, "choices", None):
            continue
        content = getattr(chunk.choices[0].delta, "content", None)
        if content:
            yield content


# ─── Embeddings ──────────────────────────────────────────────────────────────
# Returns an object implementing the LangChain Embeddings interface
# (embed_documents / embed_query) so it can be handed straight to
# SemanticChunker as well as used directly.


class _InferenceClientEmbeddings:
    """huggingface_hub.InferenceClient feature-extraction as a LangChain Embeddings."""

    def __init__(self, model):
        from huggingface_hub import InferenceClient

        self._model = model
        self._client = InferenceClient(token=HF_TOKEN)

    def embed_documents(self, texts):
        out = self._client.feature_extraction(list(texts), model=self._model)
        return [row.tolist() if hasattr(row, "tolist") else list(row) for row in out]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class _OpenAIEmbeddings:
    """OpenAI-compatible /embeddings endpoint as a LangChain Embeddings."""

    def __init__(self, model):
        from openai import OpenAI

        self._model = model
        self._client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

    def embed_documents(self, texts):
        resp = self._client.embeddings.create(model=self._model, input=list(texts))
        return [d.embedding for d in resp.data]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


NOMIC_DOC_PREFIX = "search_document: "
NOMIC_QUERY_PREFIX = "search_query: "


class _PrefixedEmbeddings:
    """nomic-embed-text's task prefixes. Applied only to a v2 index: v1 vectors
    were computed without them, and prefixed queries against unprefixed
    documents are worse than neither."""

    def __init__(self, inner, doc_prefix, query_prefix):
        self._inner = inner
        self._dp = doc_prefix
        self._qp = query_prefix

    def embed_documents(self, texts):
        return self._inner.embed_documents([self._dp + t for t in texts])

    def embed_query(self, text):
        return self._inner.embed_query(self._qp + text)


_embeddings_singleton = None


def get_embeddings(model=None):
    """Return a process-wide singleton embeddings object for EMBED_BACKEND."""
    global _embeddings_singleton
    if _embeddings_singleton is None:
        model = model or EMBED_MODEL
        if EMBED_BACKEND == "huggingface":
            logger.info("Embeddings: huggingface InferenceClient (%s)", model)
            _embeddings_singleton = _InferenceClientEmbeddings(model)
        elif EMBED_BACKEND == "openai":
            logger.info("Embeddings: OpenAI-compatible endpoint (%s)", model)
            _embeddings_singleton = _OpenAIEmbeddings(model)
        elif EMBED_BACKEND == "ollama":
            from langchain_ollama import OllamaEmbeddings

            logger.info("Embeddings: local Ollama (%s)", model)
            _embeddings_singleton = OllamaEmbeddings(model=model)
        else:
            raise ValueError(
                f"Unknown EMBED_BACKEND {EMBED_BACKEND!r} "
                "(expected 'ollama', 'openai' or 'huggingface')"
            )
        if INDEX_VERSION >= 2 and model.startswith("nomic-embed"):
            logger.info("Embeddings: nomic task prefixes on (index v%d)", INDEX_VERSION)
            _embeddings_singleton = _PrefixedEmbeddings(
                _embeddings_singleton, NOMIC_DOC_PREFIX, NOMIC_QUERY_PREFIX
            )
    return _embeddings_singleton
