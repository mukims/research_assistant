"""Auth, the cached search index, and the small helpers every tab shares."""

import json
import os
import tempfile

import streamlit as st

from research_assistant import config
from research_assistant.shared import pipeline_status

def _check_auth() -> bool:
    app_pwd = os.environ.get("APP_PASSWORD")
    if not app_pwd:
        try:
            if hasattr(st, "secrets") and "password" in st.secrets:
                app_pwd = st.secrets["password"]
        except Exception:
            app_pwd = None

    if not app_pwd:
        return True
    if st.session_state.get("authenticated", False):
        return True

    st.markdown("### 🔐 Marvin the Citebot — Access Verification")
    st.caption("This research instance is currently protected during review. Please enter the access code to continue.")
    with st.form("auth_form", clear_on_submit=False):
        entered = st.text_input("Access Code", type="password", placeholder="Enter access password")
        submitted = st.form_submit_button("Unlock", type="primary")
        if submitted:
            if entered.strip() == app_pwd.strip():
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Invalid access code. Please verify the code and try again.", icon="🚫")
    return False


@st.cache_resource(show_spinner=False)
def get_cached_search_resources(mtime: float):
    """Load ChromaDB collection and BM25 index once into memory for all sessions.
    
    Keyed on the modification time of BM25_INDEX_PATH so re-indexing automatically
    refreshes the shared cache without needing server restarts.
    """
    from research_assistant.shared.db import load_search_resources
    return load_search_resources()


def _get_resources_mtime() -> float:
    if os.path.exists(config.BM25_INDEX_PATH):
        return os.path.getmtime(config.BM25_INDEX_PATH)
    return 0.0


def _is_ingest_locked() -> bool:
    return pipeline_status._is_ingest_locked()


def _graph():
    # Rebuilt per run so each build starts from a clean checkpointer.
    # orchestrate.py is a root-level entry point, not a package module —
    # it must be imported bare, not as research_assistant.orchestrate.
    import orchestrate

    return orchestrate.build_graph()


@st.cache_data(ttl=30, show_spinner=False)
def _corpus_stats():
    """(chunks, papers) — cheap, tolerant of a missing/empty store."""

    chunks = 0
    try:
        from research_assistant.shared.db import open_store

        chunks = (
            open_store()
            .get_collection(config.COLLECTION_NAME)
            .count()
        )
    except Exception:
        pass

    papers = 0
    for path in (config.SEED_PAPERS_PATH, config.DOWNLOADED_JSON_PATH):
        try:
            with open(path) as fh:
                papers += len(json.load(fh))
        except Exception:
            pass

    try:
        from research_assistant.shared import manifest
        manifest_count = len(manifest.load())
        papers = max(papers, manifest_count)
    except Exception:
        pass
    return chunks, papers


def _manifest(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _render_passages(passages, label="Retrieved context"):
    """The retrieved-chunk expander, shared by every tab that shows one."""
    if not passages:
        return
    with st.expander(f"{label} · {len(passages)} passage(s)"):
        for i, p in enumerate(passages, 1):
            m = p.get("metadata") or {}
            st.caption(
                f"{i}. **{m.get('citation_source', '?')}** — "
                f"{m.get('document', '?')} · p.{m.get('page', '?')}"
            )
            st.text((p.get("text") or "")[:900])
            if i < len(passages):
                st.divider()


def _stage_upload(uploaded_file) -> str:
    """Write an uploaded PDF somewhere the pipeline can seed from, and return it.

    Agent 0 copies the paper into RAW_DIR under a key derived from its DOI,
    arXiv id or content hash, so this staging copy is pure duplication — it
    exists only because the graph's state carries a path, not bytes. So it goes
    to a temp file the caller deletes once the run finishes, rather than under
    DATA_DIR where copies would pile up and two uploads sharing a filename
    would overwrite each other.
    """
    fd, path = tempfile.mkstemp(prefix="seed_upload_", suffix=".pdf")
    with os.fdopen(fd, "wb") as fh:
        fh.write(uploaded_file.getbuffer())
    return path


def _get_model_choices():
    """Build list of available chat models, prioritizing local and popular Ollama models."""
    from research_assistant.shared.llm import list_ollama_models

    choices = []

    # 1. Local installed Ollama models (detected from daemon)
    for m in list_ollama_models():
        choices.append({"label": f"🦙 Ollama: {m} (installed)", "model": m, "backend": "ollama"})

    # 2. Recommended / popular Ollama models
    standard_ollama = getattr(
        config,
        "OLLAMA_CHAT_MODELS",
        ("gemma4:e2b", "llama3.2:latest", "mistral:latest", "qwen2.5:latest", "deepseek-r1:8b"),
    )
    for m in standard_ollama:
        if not any(c["model"] == m for c in choices):
            choices.append({"label": f"🦙 Ollama: {m}", "model": m, "backend": "ollama"})

    # 3. Hosted / Gemini / OpenAI options
    if config.GEMINI_API_KEY:
        choices.append({"label": "✨ Gemini: gemini-3.5-flash-lite (fast)", "model": "gemini-3.5-flash-lite", "backend": "openai"})
        choices.append({"label": "✨ Gemini: gemini-2.5-pro (deep reasoning)", "model": "gemini-2.5-pro", "backend": "openai"})
    elif config.OPENAI_API_KEY:
        choices.append({"label": "🌐 OpenAI: gpt-4o-mini", "model": "gpt-4o-mini", "backend": "openai"})
    else:
        choices.append({"label": "✨ Gemini: gemini-3.5-flash-lite", "model": "gemini-3.5-flash-lite", "backend": "openai"})

    # 4. Custom Ollama model entry
    choices.append({"label": "⚙️ Custom Ollama model…", "model": "__custom__", "backend": "ollama"})
    return choices


def _render_sidebar_backend_controls():
    """Render chat model selector (with Ollama models) and embedding info in sidebar."""
    choices = _get_model_choices()
    labels = [c["label"] for c in choices]
    active_m = st.session_state.get("chat_model") or config.CHAT_MODEL
    default_idx = 0
    for idx, c in enumerate(choices):
        if c["model"] == active_m:
            default_idx = idx
            break

    sel_label = st.selectbox(
        "Chat Model Engine",
        options=labels,
        index=default_idx,
        key="sidebar_chat_model_selector",
        help="Select LLM: pick from local Ollama models or cloud endpoints.",
    )
    chosen = next((c for c in choices if c["label"] == sel_label), choices[0])

    if chosen["model"] == "__custom__":
        custom_name = st.text_input(
            "Custom Ollama Name",
            value=st.session_state.get("custom_ollama_model", "gemma4:e2b"),
            key="sidebar_custom_ollama_input",
            placeholder="e.g. phi3:mini",
        )
        final_model = custom_name.strip() or "gemma4:e2b"
        final_backend = "ollama"
        st.session_state["custom_ollama_model"] = final_model
    else:
        final_model = chosen["model"]
        final_backend = chosen["backend"]

    st.session_state["chat_model"] = final_model
    st.session_state["chat_backend"] = final_backend

    if "chat_agent" in st.session_state and st.session_state["chat_agent"] is not None:
        agent = st.session_state["chat_agent"]
        if getattr(agent, "model", None) != final_model or getattr(agent, "backend", None) != final_backend:
            agent.set_model(final_model, backend=final_backend)

    b_badge = "🦙 Ollama Local" if final_backend == "ollama" else "✨ Cloud API"
    st.caption(f"**Backend:** {b_badge}")
    st.caption(f"**Embeddings** — `{config.EMBED_MODEL}` · {config.EMBED_BACKEND}")

