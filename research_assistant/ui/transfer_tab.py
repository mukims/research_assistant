"""Streamlit tab for Cross-Disciplinary Technique Transfer."""

import html
import json
import os
import streamlit as st

from research_assistant import config
from research_assistant.transfer.models import RunResult, result_from_dict, result_to_dict
from research_assistant.transfer.render import render_markdown, render_tree
from research_assistant.ui.status import _render_live_pipeline_status


TRANSFER_EXAMPLES = [
    "Inverse problem parameter recovery under correlated noise",
    "Topological protection in disordered one-dimensional lattices",
    "Phase retrieval in high-dimensional non-convex optimization",
    "Diffusion-based generative sampling with physical conservation laws",
]


def render_transfer_tab():
    """Render the Cross-Disciplinary Technique Transfer tab."""
    _render_live_pipeline_status("tab_transfer")

    st.subheader("🔄 Cross-Disciplinary Technique Transfer")
    st.caption(
        "Bridge methodologies across scientific disciplines. Enter a target bottleneck, problem formulation, "
        "or computational challenge, and Marvin explores transfer cards from other literature domains, "
        "maps foundational analogies, analyzes identifiability, tests critic objections, and generates novel hypotheses."
    )

    with st.expander("💡 Example Bottlenecks & Problems (Click to prefill)", expanded=False):
        scols = st.columns(len(TRANSFER_EXAMPLES))
        for idx, starter in enumerate(TRANSFER_EXAMPLES):
            if scols[idx].button(f"✨ {starter}", key=f"transfer_starter_{idx}", use_container_width=True):
                st.session_state["transfer_prefilled_query"] = starter
                st.rerun()

    prefill = st.session_state.pop("transfer_prefilled_query", "")

    with st.form("transfer_form"):
        query = st.text_input(
            "Target Problem or Bottleneck",
            value=prefill,
            placeholder="e.g. recovering continuous parameters from noisy spectra",
            help="Describe the physical problem, bottleneck, or mathematical challenge.",
        )

        col_cands, col_year, col_temp = st.columns(3)
        with col_cands:
            candidates = st.slider("Candidate cards to explore", min_value=1, max_value=10, value=3)
        with col_year:
            before_year = st.number_input("Prior art cutoff year (optional)", min_value=1900, max_value=2030, value=2024)
        with col_temp:
            temperature = st.slider("Proposal temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        run_btn = st.form_submit_button("🚀 Find Cross-Disciplinary Transfers", type="primary", use_container_width=True)

    if run_btn and query.strip():
        with st.status(f"Searching technique transfers for '{query}'…", expanded=True) as status_box:
            st.write("🔍 Identifying candidate problem cards & source methodologies…")
            try:
                from research_assistant.transfer.engine import default_engine

                def on_progress(done: int, total: int, label: str):
                    status_box.write(f"⚙️ [{done}/{total}] {label}")

                engine = default_engine()
                # Run the transfer engine search
                result = engine.run(
                    query.strip(),
                    candidates=candidates,
                    temperature=temperature,
                    prior_art_before_year=int(before_year) if before_year else None,
                    progress_cb=on_progress,
                )
                st.session_state["latest_transfer_result"] = result_to_dict(result)
                status_box.update(label="✅ Technique transfer search complete!", state="complete", expanded=False)
            except Exception as exc:
                status_box.update(label=f"❌ Transfer search failed: {exc}", state="error")
                st.error(f"Transfer engine execution error: {exc}")

    # Display results if present in session state
    saved_result_dict = st.session_state.get("latest_transfer_result")
    if saved_result_dict:
        try:
            result = result_from_dict(saved_result_dict)
            st.divider()
            st.markdown(f"### 📋 Transfer Results: *{html.escape(result.query)}*")

            t_tab_report, t_tab_tree, t_tab_json = st.tabs(["📄 Full Transfer Report", "🌳 Transfer Tree", "💾 Raw JSON"])

            with t_tab_report:
                report_md = render_markdown(result)
                st.markdown(report_md)

            with t_tab_tree:
                mermaid_tree = render_tree(result)
                st.markdown("#### Methodology Search Tree")
                from research_assistant.tasks.ui import render_mermaid
                render_mermaid(mermaid_tree, height=500)

            with t_tab_json:
                st.json(saved_result_dict)

        except Exception as exc:
            st.warning(f"Could not render stored transfer result: {exc}")
