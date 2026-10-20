"""Tab: research a topic from a research idea."""

import streamlit as st

from research_assistant.ui.common import _corpus_stats
from research_assistant.ui.jobs import _handle_job_state, _render_recent_runs, _start_pipeline_job
from research_assistant.ui.results import _render_build
from research_assistant.ui.status import _render_live_pipeline_status

def render_idea_tab():
    """Tab: research a topic from a research idea."""
    _render_live_pipeline_status("tab_idea")

    st.subheader("🔍 Explore Research Ideas & Discover Literature")
    st.caption(
        "Give Marvin a topic, hypothesis, or research question. The assistant searches arXiv, Semantic Scholar, and OpenAlex, "
        "downloads relevant open-access papers, indexes them into your local corpus, and synthesizes what has already been done."
    )

    with st.form("idea_form"):
        query = st.text_input(
            "Research idea",
            placeholder="topological protection in disordered quantum wires",
            help="Enter a research topic, question, or hypothesis.",
        )
        seed_url = st.text_input(
            "Seed paper URL (optional)",
            placeholder="https://arxiv.org/abs/2401.12345 — leave blank to search automatically",
            help="Used when the search finds no open-access PDF. Accepts an arXiv link or a direct .pdf URL.",
        )
        c1, c2 = st.columns(2)
        ask = c1.toggle("Synthesize answer", value=True, help="Formulate related-work synthesis across discovered papers.")
        force = c2.toggle("Force re-run", value=False)
        submitted_idea = st.form_submit_button("Explore Research Idea", type="primary")

    if submitted_idea:
        if not query.strip():
            st.warning("Please enter a research idea to search.", icon="⚠️")
        else:
            q = query.strip()
            seed_url_val = seed_url.strip() or None
            for key in ("idea_result", "idea_query"):
                st.session_state.pop(key, None)
            _start_pipeline_job(
                query=q,
                seed_url_val=seed_url_val,
                seed_file_path=None,
                ask=ask,
                force=force,
                audit_citations=False,
                origin="idea",
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
