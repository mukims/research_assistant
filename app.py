"""
Streamlit front-end for the Research Assistant pipeline (also the container entry
point).

This file lays out the page: the sidebar, then six tabs. What each tab does lives in
research_assistant/ui/, one module per tab (audit_tab, idea_tab, draft_tab, chat_tab,
help_tab; the planner tab is research_assistant/tasks/ui.py). The modules they share
are common (auth, the cached search index), results (showing one finished run),
audit (the citation-audit report), jobs (background runs), status and grobid (the
sidebar widgets).
"""

import os

import streamlit as st

from research_assistant import config
from research_assistant.ui.audit_tab import render_audit_tab
from research_assistant.ui.chat_tab import render_chat_tab
from research_assistant.ui.common import _check_auth, _corpus_stats
from research_assistant.ui.draft_tab import render_draft_tab
from research_assistant.ui.grobid import _render_grobid_controls
from research_assistant.ui.help_tab import render_help_tab
from research_assistant.ui.idea_tab import render_idea_tab
from research_assistant.ui.status import _render_live_pipeline_status, _render_sidebar_pipeline_status

st.set_page_config(page_title="Marvin the Citebot", page_icon="🤖", layout="wide")
os.makedirs(config.DATA_DIR, exist_ok=True)

if not _check_auth():
    st.stop()


# ─── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("Pipeline Status")
    _render_sidebar_pipeline_status()
    st.divider()

    st.subheader("Corpus")
    chunks, papers = _corpus_stats()
    c1, c2 = st.columns(2)
    c1.metric("Papers", papers)
    c2.metric("Chunks", chunks)

    st.subheader("Backend")
    st.caption(f"**Chat** — `{config.LLM_MODEL}` · {config.LLM_BACKEND}")
    st.caption(f"**Embeddings** — `{config.EMBED_MODEL}` · {config.EMBED_BACKEND}")
    st.caption(
        f"**Layout** — {'on' if config.LAYOUT_DETECTION else 'text-only'}"
    )

    st.subheader("Server & Services")
    _render_grobid_controls()
    if config.LLM_BACKEND == "openai" and not config.OPENAI_API_KEY:
        st.error("No OPENAI_API_KEY / HF_TOKEN set.", icon="🚫")


# ─── Main ───────────────────────────────────────────────────────────────────

st.title("🤖 Marvin the Citebot")
st.caption(
    "Autonomous AI Research Assistant. Give it a research idea → "
    "it builds a corpus from the literature and tells you what's already been done. "
    "Or hand it a paper to audit its citations against the evidence."
)

tab_planner, tab_audit, tab_idea, tab_draft, tab_chat, tab_help = st.tabs(
    ["Agent task planner", "Citation auditor", "Research idea", "Cite a draft", "Research chat",
     "How to use"]
)

with tab_planner:
    _render_live_pipeline_status("tab_planner")
    from research_assistant.tasks.ui import render_planner_tab
    render_planner_tab()

with tab_audit:
    render_audit_tab()

with tab_idea:
    render_idea_tab()

with tab_draft:
    render_draft_tab(chunks)

with tab_chat:
    render_chat_tab(chunks)

with tab_help:
    render_help_tab()
