"""First-run setup checks: what is missing, and the command that fixes it.

A new user used to learn that Ollama was down, or that a model had not been pulled,
minutes into the first pipeline run. These checks answer that up front. They are pure
(no Streamlit): the network, GROBID and store probes are parameters so tests inject
fakes, and every check turns any exception into ok=False with the text in `detail`,
so a broken probe can never take the page down.
"""

import os
from dataclasses import dataclass

import requests

from research_assistant import config

_OLLAMA_TIMEOUT_S = 3
_BACKEND_FIX = "Start Ollama (ollama serve) or set GEMINI_API_KEY in .env"
_GROBID_FIX = (
    "docker run --rm -d --name grobid -p 8070:8070 grobid/grobid:0.8.1 "
    "(without it, references fall back to regex)"
)


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""
    blocking: bool = True


def _model_present(model: str, tags: list) -> bool:
    """Ollama lists `gemma4:e2b` or `embeddinggemma-2:latest`. A name with no tag
    matches `m`, `m:latest` and `m:<any tag>`; a name with a tag matches only itself."""
    if ":" in model:
        return model in tags
    return any(t == model or t.startswith(model + ":") for t in tags)


def _ollama_tags(http_get):
    """(tags, None) when Ollama answers /api/tags, else (None, reason)."""
    from research_assistant.shared import llm

    try:
        resp = http_get(f"{llm._ollama_host()}/api/tags", timeout=_OLLAMA_TIMEOUT_S)
        if resp.status_code != 200:
            return None, f"Ollama answered HTTP {resp.status_code}"
        return [m.get("name", "") for m in resp.json().get("models", [])], None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _model_check(name: str, model: str, tags: list) -> Check:
    ok = _model_present(model, tags)
    return Check(
        name,
        ok,
        f"{model} is pulled" if ok else f"{model} is not in Ollama's model list",
        "" if ok else f"ollama pull {model}",
    )


def _vector_store_check(open_store) -> Check:
    try:
        if open_store is None:
            from research_assistant.shared import db

            open_store = db.open_store
        open_store()
        return Check("Vector store", True, "opens")
    except Exception as exc:
        return Check("Vector store", False, f"{type(exc).__name__}: {exc}", "deploy/local.sh up")


def _data_folder_check() -> Check:
    path = config.DATA_DIR
    try:
        os.makedirs(path, exist_ok=True)
        if os.access(path, os.W_OK):
            return Check("Data folder", True, f"{path} is writable")
        detail = f"{path} is not writable"
    except Exception as exc:
        detail = f"{path}: {type(exc).__name__}: {exc}"
    return Check("Data folder", False, detail, f"Make {path} writable or set CITATION_DATA_DIR")


def _grobid_check(grobid_alive) -> Check:
    try:
        if grobid_alive is None:
            from research_assistant.shared.grobid_manager import is_grobid_alive as grobid_alive
        alive = bool(grobid_alive())
        detail = "reachable" if alive else "not reachable"
    except Exception as exc:
        alive, detail = False, f"{type(exc).__name__}: {exc}"
    return Check("GROBID", alive, detail, "" if alive else _GROBID_FIX, blocking=False)


def run_checks(http_get=requests.get, grobid_alive=None, open_store=None) -> list:
    """Every setup check, in the order the sidebar lists them.

    The model checks are Ollama-only and are left out, not failed, when the backend is
    unreachable: the Model backend line already says so and the model list is unknown."""
    checks = []
    uses_ollama_chat = config.LLM_BACKEND != "openai"
    uses_ollama_embed = config.EMBED_BACKEND == "ollama"

    tags = None
    if uses_ollama_chat or uses_ollama_embed:
        tags, why = _ollama_tags(http_get)
    if uses_ollama_chat:
        checks.append(
            Check("Model backend", tags is not None, "Ollama is running" if tags is not None else why, "" if tags is not None else _BACKEND_FIX)
        )
    else:
        has_key = bool(config.OPENAI_API_KEY)
        checks.append(
            Check("Model backend", has_key, "API key is set" if has_key else "No OPENAI_API_KEY / GEMINI_API_KEY / HF_TOKEN set", "" if has_key else _BACKEND_FIX)
        )

    if tags is not None and uses_ollama_chat:
        checks.append(_model_check("Chat model", config.CHAT_MODEL, tags))
    if tags is not None and uses_ollama_embed:
        checks.append(_model_check("Embedding model", config.EMBED_MODEL, tags))

    checks.append(_vector_store_check(open_store))
    checks.append(_data_folder_check())
    checks.append(_grobid_check(grobid_alive))
    email = config.UNPAYWALL_EMAIL
    checks.append(
        Check(
            "Contact email",
            bool(email),
            email or "UNPAYWALL_EMAIL is not set",
            "" if email else "export UNPAYWALL_EMAIL=you@university.edu",
            blocking=False,
        )
    )
    return checks
