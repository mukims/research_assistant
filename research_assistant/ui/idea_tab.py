"""Tab: research a topic from a research idea."""

import os
import streamlit as st

from research_assistant.ui.common import _corpus_stats, _stage_upload
from research_assistant.ui.jobs import _handle_job_state, _render_recent_runs, _start_pipeline_job
from research_assistant.ui.results import _render_build
from research_assistant.ui.status import _render_live_pipeline_status

STARTER_IDEAS = [
    "Topological protection in disordered quantum wires",
    "Twisted bilayer graphene flat-band superconductivity",
    "Majorana bound states in semiconductor-superconductor nanowires",
    "Solid-state electrolyte interface stability in lithium batteries",
]


def render_idea_tab():
    """Tab: research a topic from a research idea."""
    _render_live_pipeline_status("tab_idea")

    st.subheader("🔍 Explore Research Ideas & Discover Literature")
    st.caption(
        "Give Marvin a topic, hypothesis, or research question. The assistant searches arXiv, Semantic Scholar, and OpenAlex, "
        "downloads relevant open-access papers, indexes them into your local corpus, and synthesizes what has already been done."
    )

    with st.expander("💡 Example Research Topics (Click to prefill)", expanded=False):
        scols = st.columns(len(STARTER_IDEAS))
        for idx, starter in enumerate(STARTER_IDEAS):
            if scols[idx].button(f"✨ {starter}", key=f"idea_starter_{idx}", use_container_width=True):
                st.session_state["idea_prefilled_query"] = starter
                st.rerun()

    prefill = st.session_state.pop("idea_prefilled_query", "")

    with st.form("idea_form"):
        query = st.text_input(
            "Research idea",
            value=prefill,
            placeholder="topological protection in disordered quantum wires",
            help="Enter a research topic, question, or hypothesis.",
        )
        col_url, col_file = st.columns(2)
        with col_url:
            seed_url = st.text_input(
                "Seed paper URL (optional)",
                placeholder="https://arxiv.org/abs/2401.12345 — or leave blank to search automatically",
                help="Accepts an arXiv link or a direct .pdf URL to seed discovery directly.",
            )
        with col_file:
            uploaded_seed = st.file_uploader(
                "Seed paper PDF (optional)",
                type=["pdf"],
                key="idea_seed_file_uploader",
                help="Upload a seed PDF directly instead of fetching from external repositories.",
            )

        with st.expander("⚙️ Synthesis & Pipeline Controls", expanded=False):
            c1, c2, c3, c4, c5 = st.columns(5)
            ask = c1.toggle(
                "Synthesize answer",
                value=True,
                help="Formulate related-work synthesis across discovered papers.",
            )
            audit_seed = c2.toggle(
                "Audit seed citations",
                value=False,
                help="Audit in-text citations from the seed paper against retrieved evidence using Rubric V1.6.",
            )
            synthesis_mode = c3.selectbox(
                "Synthesis Depth",
                options=["map_reduce", "single"],
                index=0,
                format_func=lambda m: "Deep (Map-Reduce notes)" if m == "map_reduce" else "Fast (Single-pass)",
                help="Map-Reduce reads every shortlisted paper and extracts notes before synthesis; Single-pass synthesizes directly.",
            )
            force = c4.toggle(
                "Force re-run",
                value=False,
                help="Re-run pipeline stages even if already cached.",
            )
            enable_ocr = c5.toggle(
                "Enable OCR",
                value=True,
                help="Enable OCR for scanned seed papers or notes lacking a digital text layer.",
            )

        submitted_idea = st.form_submit_button("Explore Research Idea", type="primary")

    if submitted_idea:
        q = query.strip()
        seed_file_path = None
        if uploaded_seed is not None:
            seed_file_path = _stage_upload(uploaded_seed)
            if not q:
                # Infer query from uploaded file name if query wasn't typed
                q = os.path.splitext(uploaded_seed.name)[0].replace("_", " ").replace("-", " ")

        if not q and not seed_file_path and not seed_url.strip():
            st.warning("Please enter a research idea, provide a seed URL, or upload a seed PDF.", icon="⚠️")
        else:
            seed_url_val = seed_url.strip() or None
            for key in ("idea_result", "idea_query"):
                st.session_state.pop(key, None)
            _start_pipeline_job(
                query=q,
                seed_url_val=seed_url_val,
                seed_file_path=seed_file_path,
                ask=ask,
                force=force,
                audit_citations=audit_seed,
                synthesis_mode=synthesis_mode,
                origin="idea",
                enable_ocr=enable_ocr,
            )

    if _handle_job_state("idea"):
        pass  # a run is in progress; its live view is on screen
    elif st.session_state.get("idea_result"):
        idea_res = st.session_state["idea_result"]
        _corpus_stats.clear()
        _render_build(idea_res, st.session_state.get("idea_query", ""), origin="idea")
    else:
        _render_recent_runs("idea")
        st.markdown("---")
        st.markdown(
            "#### 💡 How Literature Discovery Works\n\n"
            "1. **Enter an idea**: Describe a research topic or paste a known arXiv / PDF link.\n"
            "2. **Discovery (Agent 0)**: Searches academic repositories for candidate literature and downloads the seed paper.\n"
            "3. **Citation Chasing (Agents 1 & 2)**: Extracts references and fetches connected open-access papers.\n"
            "4. **Corpus Indexing (Agent 3)**: Chunks, embeds, and builds the BM25 keyword index.\n"
            "5. **Synthesis**: Writes a structured related-work summary highlighting established findings and open gaps."
        )
