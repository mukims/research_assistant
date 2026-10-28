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
from research_assistant.ui.common import _check_auth, _corpus_stats, _render_sidebar_backend_controls
from research_assistant.ui.corpus_tab import render_corpus_tab
from research_assistant.ui.draft_tab import render_draft_tab
from research_assistant.ui.grobid import _render_grobid_controls
from research_assistant.ui.help_tab import render_help_tab
from research_assistant.ui.idea_tab import render_idea_tab
from research_assistant.ui.status import _render_live_pipeline_status, _render_sidebar_pipeline_status

st.set_page_config(page_title="Marvin the Citebot", page_icon="🤖", layout="wide")
os.makedirs(config.DATA_DIR, exist_ok=True)

if not _check_auth():
    st.stop()

# ─── UI Styling ─────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Marvin Modern Theme Enhancements */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        font-weight: 600;
        border-bottom: 2px solid #6366f1;
    }
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 12px;
        padding: 12px 16px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .stButton > button {
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)



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
    _render_sidebar_backend_controls()

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

tab_planner, tab_audit, tab_idea, tab_draft, tab_chat, tab_corpus, tab_help = st.tabs(
    ["Agent task planner", "Citation auditor", "Research idea", "Cite a draft", "Research chat",
     "Corpus library", "How to use"]
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

with tab_corpus:
    render_corpus_tab(chunks, papers)

with tab_help:
    render_help_tab()
