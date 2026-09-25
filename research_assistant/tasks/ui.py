"""
Streamlit UI Component for Scientific Agent Task Planner & Replication DAGs.
"""

import json
import os
import re
import tempfile
from typing import List, Optional

import streamlit as st

from research_assistant import config
from research_assistant.tasks.extractor import deconstruct_paper_into_task_graph
from research_assistant.tasks.graph import (
    export_antigravity_specs,
    render_markdown_plan,
    render_mermaid,
)
from research_assistant.tasks.models import (
    AgentRole,
    Task,
    TaskCategory,
    TaskGraph,
    TaskStatus,
)
from research_assistant.tasks.roles import get_role_spec
from research_assistant.tasks.runner import ExecutionMode, TaskRunner


def _list_candidate_papers() -> List[str]:
    """Finds candidate papers (PDFs or TEI XMLs) in the data directory."""
    candidates = []
    pulled_dir = getattr(config, "PULLED_PDFS_DIR", os.path.join(config.DATA_DIR, "pulled_pdfs"))
    if os.path.exists(pulled_dir):
        for f in sorted(os.listdir(pulled_dir)):
            if f.lower().endswith(".pdf"):
                candidates.append(os.path.join(pulled_dir, f))

    raw_dir = getattr(config, "RAW_DIR", os.path.join(config.DATA_DIR, "raw"))
    if os.path.exists(raw_dir):
        for f in sorted(os.listdir(raw_dir)):
            if f.lower().endswith(".pdf"):
                candidates.append(os.path.join(raw_dir, f))

    audit_tei = os.path.join(config.DATA_DIR, "audit", "tei")
    if os.path.exists(audit_tei):
        for f in sorted(os.listdir(audit_tei)):
            if f.lower().endswith((".xml", ".tei.xml")):
                candidates.append(os.path.join(audit_tei, f))

    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen and os.path.isfile(c):
            seen.add(c)
            unique_candidates.append(c)

    return unique_candidates


def _render_mermaid_interactive(mermaid_code: str, height: int = 420):
    """Renders Mermaid diagram in an iframe with interactive pan/zoom support."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
      <style>
        body {{
          margin: 0;
          padding: 10px;
          background-color: transparent;
          font-family: system-ui, -apple-system, sans-serif;
          display: flex;
          justify-content: center;
        }}
        .mermaid {{
          width: 100%;
          text-align: center;
        }}
      </style>
    </head>
    <body>
      <div class="mermaid">
{mermaid_code}
      </div>
      <script>
        mermaid.initialize({{
          startOnLoad: true,
          theme: 'default',
          securityLevel: 'loose',
          flowchart: {{
            useMaxWidth: true,
            htmlLabels: true,
            curve: 'basis'
          }}
        }});
      </script>
    </body>
    </html>
    """
    st.components.v1.html(html, height=height, scrolling=True)


def render_planner_tab():
    """Renders the complete Agent Task Planner interface."""
    st.subheader("🧬 Scientific Paper-to-Agent Task Decomposition")
    st.caption(
        "Deconstruct complex scientific papers into structured Directed Acyclic Graphs (DAGs) "
        "of dependency-ordered tasks assigned to specialized AI subagents for autonomous replication."
    )

    col_input, col_opts = st.columns([2, 1])

    with col_input:
        input_source = st.radio(
            "Paper Source",
            ["Select Ingested Paper", "Upload New Paper / TEI XML / Text"],
            horizontal=True,
            key="planner_source_choice",
        )

        paper_path = None
        paper_title = None

        if input_source == "Select Ingested Paper":
            candidates = _list_candidate_papers()
            if candidates:
                labels = [os.path.basename(c) for c in candidates]
                selected_idx = st.selectbox(
                    "Choose paper from corpus",
                    range(len(candidates)),
                    format_func=lambda i: labels[i],
                    key="planner_paper_select",
                )
                paper_path = candidates[selected_idx]
                paper_title = None
            else:
                st.info("No ingested papers found in corpus yet. Please upload a paper below.")

        if input_source == "Upload New Paper / TEI XML / Text" or not paper_path:
            uploaded = st.file_uploader(
                "Upload Paper (PDF, TEI XML, or TXT)",
                type=["pdf", "xml", "txt"],
                key="planner_file_upload",
            )
            if uploaded:
                tmp_dir = os.path.join(config.DATA_DIR, "tmp_uploads")
                os.makedirs(tmp_dir, exist_ok=True)
                raw_name = getattr(uploaded, "name", "") or "uploaded_paper"
                base_name = os.path.basename(raw_name).strip(" .")
                if not base_name:
                    base_name = "uploaded_paper"
                ext = os.path.splitext(base_name)[1].lower()
                if ext not in (".pdf", ".xml", ".txt"):
                    data_head = uploaded.getvalue()[:1024]
                    if data_head.startswith(b"%PDF-"):
                        ext = ".pdf"
                    elif b"<TEI" in data_head or b"<tei" in data_head or b"<?xml" in data_head:
                        ext = ".xml"
                    else:
                        ext = ".txt"
                safe_stem = re.sub(r"[^\w\-.]", "_", os.path.splitext(base_name)[0]).strip(" .") or "uploaded_paper"
                safe_name = f"{safe_stem}{ext}"
                paper_path = os.path.join(tmp_dir, safe_name)
                with open(paper_path, "wb") as f:
                    f.write(uploaded.getvalue())
                paper_title = None

    with col_opts:
        st.markdown("##### Configuration")
        heuristic_mode = st.checkbox(
            "⚡ Fast Heuristic Mode",
            value=False,
            help="Generate a deterministic 5-task replication graph instantly without querying LLM.",
            key="planner_heuristic_toggle",
        )
        exec_mode_choice = st.selectbox(
            "Execution Mode",
            ["simulation", "autonomous"],
            format_func=lambda m: "🧪 Dry-Run Simulation" if m == "simulation" else "🤖 Autonomous Agent Execution",
            key="planner_exec_mode",
        )

        st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
        btn_deconstruct = st.button(
            "⚡ Deconstruct into Task Graph",
            type="primary",
            use_container_width=True,
            disabled=not paper_path,
            key="planner_btn_deconstruct",
        )

    # Trigger Deconstruction
    if btn_deconstruct and paper_path:
        with st.spinner(f"Mining paper methodology and synthesizing DAG for '{os.path.basename(paper_path)}'..."):
            try:
                base_id = os.path.splitext(os.path.basename(paper_path))[0]
                safe_pid = re.sub(r"[^\w\-]", "_", base_id).strip("_") or "paper"
                graph = deconstruct_paper_into_task_graph(
                    tei_or_path=paper_path,
                    paper_id=safe_pid,
                    paper_title=paper_title,
                    force_heuristic=heuristic_mode,
                )
                st.session_state["planner_active_graph"] = graph
                st.session_state["planner_last_report"] = None
                st.success(f"Successfully synthesized DAG for '{graph.paper_title}' with {len(graph.tasks)} tasks!")
            except Exception as e:
                st.error(f"Failed to deconstruct paper: {e}")

    # Display Active Graph
    graph: Optional[TaskGraph] = st.session_state.get("planner_active_graph")
    if graph:
        st.divider()
        st.markdown(f"### 📋 Research Plan: {graph.paper_title}")

        # Metrics bar
        try:
            waves = graph.topological_waves()
        except Exception as e:
            st.error(f"Cannot schedule task waves: {e}")
            waves = [[t for t in graph.tasks.values()]]
        roles = {t.assigned_role for t in graph.tasks.values()}
        deliverables = sum(len(t.outputs) for t in graph.tasks.values())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Tasks", len(graph.tasks))
        m2.metric("Execution Waves", len(waves))
        m3.metric("Agent Roles", len(roles))
        m4.metric("Deliverable Artifacts", deliverables)

        # Tabs for Graph Views
        tab_flow, tab_schedule, tab_specs = st.tabs(["📊 Interactive DAG Flowchart", "📑 Execution Schedule & Tasks", "🤖 Antigravity Agent Specs"])

        with tab_flow:
            mermaid_str = render_mermaid(graph)
            _render_mermaid_interactive(mermaid_str)
            with st.expander("Show Mermaid Source Code"):
                st.code(mermaid_str, language="mermaid")

        with tab_schedule:
            for wave_idx, wave_tasks in enumerate(waves):
                st.markdown(f"#### 🌊 Wave {wave_idx + 1} ({len(wave_tasks)} Parallel Tasks)")
                for task in wave_tasks:
                    role_spec = get_role_spec(task.assigned_role)
                    status_icon = "🟢" if task.status == TaskStatus.COMPLETED else ("🟡" if task.status == TaskStatus.READY else "⚪")

                    with st.expander(f"{status_icon} `{task.task_id}`: {task.title} — {role_spec.name}"):
                        st.markdown(f"**Objective**: {task.description}")
                        col_in, col_out = st.columns(2)
                        with col_in:
                            st.markdown("**Inputs / Prerequisites:**")
                            if task.inputs:
                                for inp in task.inputs:
                                    st.markdown(f"- `{inp}`")
                            else:
                                st.caption("None (Initial root task)")
                        with col_out:
                            st.markdown("**Required Deliverables:**")
                            if task.outputs:
                                for out in task.outputs:
                                    st.markdown(f"- `{out}`")
                            else:
                                st.caption("None declared")

                        if task.acceptance_criteria:
                            st.markdown("**Acceptance Criteria:**")
                            for crit_idx, crit in enumerate(task.acceptance_criteria):
                                st.checkbox(crit, value=(task.status == TaskStatus.COMPLETED), disabled=True, key=f"crit_{task.task_id}_{crit_idx}")

                        if task.paper_context_excerpts:
                            st.markdown("**Paper Excerpt Context:**")
                            for excerpt in task.paper_context_excerpts:
                                st.info(f"“{excerpt}”")

        with tab_specs:
            specs = export_antigravity_specs(graph)
            st.caption("Standardized agent invocation payloads for Google Antigravity autonomous multi-agent teams.")
            st.json(specs)

        # Execution Section
        st.divider()
        st.markdown("### 🚀 Dispatch & Execution")
        col_run_btn, col_down_json, col_down_md, col_down_ag = st.columns([2, 1, 1, 1])

        with col_run_btn:
            if st.button("🚀 Execute Task Graph Now", type="primary", use_container_width=True, key="planner_btn_exec"):
                workspace = os.path.join(config.DATA_DIR, "workspace_runs", graph.paper_id)
                mode = ExecutionMode.AUTONOMOUS if exec_mode_choice == "autonomous" else ExecutionMode.SIMULATION

                prog_bar = st.progress(0.0, text="Initializing wave dispatch...")
                status_text = st.empty()

                total = len(graph.tasks)
                executed = 0

                def on_task_update(t: Task, status: TaskStatus):
                    nonlocal executed
                    if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.BLOCKED):
                        executed += 1
                        pct = min(1.0, executed / max(1, total))
                        prog_bar.progress(pct, text=f"Executed {executed}/{total} tasks ({status.value})...")
                    status_text.caption(f"Currently processing: `{t.task_id}` ({t.title}) -> **{status.value}**")

                runner = TaskRunner(
                    graph=graph,
                    workspace_dir=workspace,
                    mode=mode,
                    on_task_update=on_task_update,
                )

                with st.spinner("Executing task waves..."):
                    report = runner.execute()
                    st.session_state["planner_last_report"] = report
                    st.rerun()

        with col_down_json:
            st.download_button(
                "📥 JSON Plan",
                data=json.dumps(graph.to_dict(), indent=2),
                file_name=f"{graph.paper_id}_plan.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_down_md:
            st.download_button(
                "📥 Markdown Plan",
                data=render_markdown_plan(graph),
                file_name=f"{graph.paper_id}_plan.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_down_ag:
            st.download_button(
                "📥 Antigravity Specs",
                data=json.dumps(export_antigravity_specs(graph), indent=2),
                file_name=f"{graph.paper_id}_antigravity.json",
                mime="application/json",
                use_container_width=True,
            )

        # Render Execution Report if available
        report = st.session_state.get("planner_last_report")
        if report:
            st.markdown("---")
            st.markdown("### 🏆 Replication Scorecard")
            sc = report.scorecard
            rating = sc.get("rating", "UNKNOWN")
            rate = sc.get("replication_rate", 0.0)

            col_grade, col_stats = st.columns([1, 2])
            with col_grade:
                if rating == "CERTIFIED_REPLICABLE":
                    st.success(f"### Grade: {rating}\n**Replication Success: {rate:.1%}**")
                elif rating == "PARTIALLY_REPLICATED":
                    st.warning(f"### Grade: {rating}\n**Replication Success: {rate:.1%}**")
                else:
                    st.error(f"### Grade: {rating}\n**Replication Success: {rate:.1%}**")

            with col_stats:
                st.markdown(
                    f"- **Tasks Completed**: `{report.completed_count} / {report.total_tasks}`\n"
                    f"- **Tasks Failed**: `{report.failed_count}`\n"
                    f"- **Tasks Blocked**: `{report.blocked_count}`\n"
                    f"- **Execution Period**: `{report.started_at}` to `{report.completed_at}`"
                )

            st.markdown("#### Task Results Table")
            table_rows = []
            for tid, res in report.results.items():
                t = graph.get_task(tid)
                table_rows.append({
                    "Task ID": tid,
                    "Title": t.title if t else tid,
                    "Role": t.assigned_role.value if t else "-",
                    "Status": res.status.value,
                    "Duration (s)": res.duration_seconds,
                    "Artifacts": len(res.output_artifacts),
                    "Error": res.error_message or "-",
                })
            st.dataframe(table_rows, use_container_width=True)
