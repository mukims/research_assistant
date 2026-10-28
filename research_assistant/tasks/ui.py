"""
Streamlit UI Component for Scientific Agent Task Planner & Replication DAGs.
"""

import json
import os
import re
import sys
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
    Task,
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


ROLE_ICONS = {
    "devops_agent": "🛠️",
    "data_engineer": "📊",
    "model_architect": "🧠",
    "experiment_runner": "🧪",
    "evaluation_agent": "📈",
    "audit_agent": "⚖️",
    "scientific_critic": "🔬",
}


def _list_existing_workspace_runs() -> List[str]:
    """Finds existing deconstructed paper runs in workspace_runs/ that contain plan.json."""
    ws_base = os.environ.get("CITATION_WORKSPACE_DIR") or os.path.join(config.PROJECT_ROOT, "workspace_runs")
    if not os.path.exists(ws_base):
        return []
    runs = []
    for item in sorted(os.listdir(ws_base)):
        plan_file = os.path.join(ws_base, item, "plan.json")
        if os.path.isfile(plan_file):
            runs.append(item)
    return runs


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
        var attempts = 0;
        function renderMermaid() {{
          if (typeof mermaid !== 'undefined') {{
            try {{
              mermaid.initialize({{
                startOnLoad: false,
                theme: 'default',
                securityLevel: 'loose',
                flowchart: {{
                  useMaxWidth: true,
                  htmlLabels: true,
                  curve: 'basis'
                }}
              }});
              mermaid.run({{
                querySelector: '.mermaid'
              }});
            }} catch(e) {{
              console.error('Mermaid render error:', e);
            }}
          }} else if (attempts < 50) {{
            attempts++;
            setTimeout(renderMermaid, 100);
          }}
        }}
        if (document.readyState === 'complete' || document.readyState === 'interactive') {{
          renderMermaid();
        }} else {{
          window.addEventListener('DOMContentLoaded', renderMermaid);
          window.addEventListener('load', renderMermaid);
        }}
      </script>
    </body>
    </html>
    """
    st.components.v1.html(html, height=height, scrolling=True)



def _is_test_env() -> bool:
    """Returns True if running under pytest, unittest, or headless test harness."""
    return (
        any(m in sys.modules for m in ("pytest", "unittest"))
        or (len(sys.argv) > 0 and ("pytest" in sys.argv[0] or any("test" in a for a in sys.argv)))
    )


def render_planner_tab():
    """Renders the complete Agent Task Planner interface."""
    # Pre-load existing workspace DAG if none is in session state (in live interactive app)
    if not st.session_state.get("planner_active_graph") and not _is_test_env():
        existing_runs = _list_existing_workspace_runs()
        if existing_runs:
            preferred = "sample_corpus_paper" if "sample_corpus_paper" in existing_runs else existing_runs[0]
            ws_base = os.environ.get("CITATION_WORKSPACE_DIR") or os.path.join(config.PROJECT_ROOT, "workspace_runs")
            plan_file = os.path.join(ws_base, preferred, "plan.json")
            try:
                with open(plan_file, "r", encoding="utf-8") as f:
                    st.session_state["planner_active_graph"] = TaskGraph.from_dict(json.load(f))
            except Exception:
                pass

    st.subheader("🧬 Scientific Paper-to-Agent Task Decomposition")
    st.caption(
        "Deconstruct complex scientific papers into structured Directed Acyclic Graphs (DAGs) "
        "of dependency-ordered tasks assigned to specialized AI subagents for autonomous replication."
    )

    col_input, col_opts = st.columns([2, 1])

    with col_input:
        existing_workspaces = _list_existing_workspace_runs()
        source_options = ["Select Ingested Paper", "Upload New Paper / TEI XML / Text"]
        if existing_workspaces:
            source_options.append("Load Existing Workspace Plan")

        input_source = st.radio(
            "Paper Source",
            source_options,
            horizontal=True,
            key="planner_source_choice",
        )

        paper_path = None
        paper_title = None

        if input_source == "Load Existing Workspace Plan":
            chosen_ws = st.selectbox(
                "Choose workspace plan to inspect",
                existing_workspaces,
                key="planner_ws_select",
            )
            ws_base = os.environ.get("CITATION_WORKSPACE_DIR") or os.path.join(config.PROJECT_ROOT, "workspace_runs")
            plan_file = os.path.join(ws_base, chosen_ws, "plan.json")
            if os.path.isfile(plan_file):
                paper_path = plan_file
            if st.button("📂 Load Selected Plan", key="planner_btn_load_ws"):
                try:
                    with open(plan_file, "r", encoding="utf-8") as f:
                        st.session_state["planner_active_graph"] = TaskGraph.from_dict(json.load(f))
                    st.session_state["planner_last_report"] = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load plan: {e}")

        elif input_source == "Select Ingested Paper":
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

        elif input_source == "Upload New Paper / TEI XML / Text":
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

                # Automatically make root dir for uploaded paper and drop initial markdown files
                try:
                    from research_assistant.tasks.workspace import materialize_paper_upload
                    paper_ws = materialize_paper_upload(uploaded.getvalue(), filename=safe_name)
                    st.caption(f"📁 Staged paper workspace: `{paper_ws}`")
                except Exception as exc:
                    pass

        # If user selected an ingested paper or uploaded file wasn't provided, ensure paper_path exists if possible
        if not paper_path and input_source == "Select Ingested Paper":
            candidates = _list_candidate_papers()
            if candidates:
                paper_path = candidates[0]

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
                active_model = st.session_state.get("chat_model") or config.LLM_MODEL
                active_backend = st.session_state.get("chat_backend")
                graph = deconstruct_paper_into_task_graph(
                    tei_or_path=paper_path,
                    paper_id=safe_pid,
                    paper_title=paper_title,
                    model=active_model,
                    backend=active_backend,
                    force_heuristic=heuristic_mode,
                    full_phases=True,
                )
                st.session_state["planner_active_graph"] = graph
                st.session_state["planner_last_report"] = None
                from research_assistant.tasks.workspace import get_paper_workspace_dir
                ws_dir = get_paper_workspace_dir(graph.paper_id)
                st.success(f"Successfully synthesized DAG for '{graph.paper_title}' with {len(graph.tasks)} tasks!")
                st.info(f"📁 Created paper root directory at `{ws_dir}` with all Markdown files (`plan.md`, `dag.md`, `scorecard.md`, `paper_summary.md`, `rationale.md`, `rationales/*.md`, `README.md`).")
            except Exception as e:
                st.error(f"Failed to deconstruct paper: {e}")

    # Display Active Graph
    graph: Optional[TaskGraph] = st.session_state.get("planner_active_graph")
    if graph:
        # Ensure any generic titles or single-role assignments from prior runs are sanitized
        for t in graph.tasks.values():
            if not t.title or t.title.strip() == "Execution Task":
                from research_assistant.tasks.iteration import _resolve_title
                t.title = _resolve_title({"title": ""}, t.category, t.description)
        roles = {t.assigned_role for t in graph.tasks.values()}
        if len(graph.tasks) >= 3 and len(roles) <= 1:
            from research_assistant.tasks.iteration import _resolve_role
            for t in graph.tasks.values():
                t.assigned_role = _resolve_role({}, t.category, t.title, t.description)

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
            with st.expander("🔍 Show Mermaid Source Code"):
                st.code(mermaid_str, language="mermaid")

            st.markdown("#### ⚡ Topological Wave Pipeline Flow")
            st.caption("Sequential execution stages. Tasks within each wave run concurrently in isolated agent environments.")

            for wave_idx, wave_tasks in enumerate(waves):
                wave_num = wave_idx + 1
                st.markdown(f"**🌊 Stage {wave_num} ({len(wave_tasks)} Concurrent Task{'s' if len(wave_tasks) > 1 else ''})**")

                cols = st.columns(min(len(wave_tasks), 3))
                for t_idx, task in enumerate(wave_tasks):
                    col = cols[t_idx % len(cols)]
                    with col:
                        role_val = task.assigned_role.value if hasattr(task.assigned_role, "value") else str(task.assigned_role)
                        role_icon = ROLE_ICONS.get(role_val, "📋")
                        status_badge = (
                            "🟢 Completed" if task.status == TaskStatus.COMPLETED
                            else ("🟡 In Progress" if task.status == TaskStatus.RUNNING
                            else ("⚪ Ready" if task.status == TaskStatus.READY else "🔴 Blocked"))
                        )
                        deliverables_str = ", ".join(f"`{out}`" for out in task.outputs) if task.outputs else "None specified"
                        st.markdown(
                            f"""
                            <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-bottom: 10px; background-color: #f8fafc;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                    <span style="font-size: 11px; font-weight: 700; color: #475569; background: #e2e8f0; padding: 2px 6px; border-radius: 4px;">{task.task_id}</span>
                                    <span style="font-size: 11px; font-weight: 600;">{status_badge}</span>
                                </div>
                                <div style="font-weight: 600; font-size: 13px; color: #0f172a; margin-bottom: 4px;">{task.title}</div>
                                <div style="font-size: 12px; color: #64748b; margin-bottom: 6px;">{role_icon} {role_val}</div>
                                <div style="font-size: 11px; color: #334155;"><strong>Deliverables:</strong> {deliverables_str}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                if wave_idx < len(waves) - 1:
                    st.markdown("<div style='text-align: center; color: #94a3b8; font-size: 16px; margin: 4px 0 12px 0;'>⬇️ <em>Dependencies cascade downstream</em> ⬇️</div>", unsafe_allow_html=True)

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
                from research_assistant.tasks.workspace import get_paper_workspace_dir
                workspace = get_paper_workspace_dir(graph.paper_id)
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

                active_model = st.session_state.get("chat_model") or config.LLM_MODEL
                active_backend = st.session_state.get("chat_backend")
                runner = TaskRunner(
                    graph=graph,
                    workspace_dir=workspace,
                    mode=mode,
                    model=active_model,
                    backend=active_backend,
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

    else:
        st.divider()
        st.info("ℹ️ No active task graph currently loaded in session.")
        col_hint1, col_hint2 = st.columns(2)
        with col_hint1:
            st.markdown(
                "**Quick Start:**\n"
                "- Select a paper from the corpus above and click **⚡ Deconstruct into Task Graph**.\n"
                "- Or choose **Load Existing Workspace Plan** to inspect pre-synthesized replication DAGs."
            )
        with col_hint2:
            existing_runs = _list_existing_workspace_runs()
            if existing_runs and st.button("📂 Load Default Sample Corpus Plan", type="secondary", key="planner_btn_quick_load"):
                preferred = "sample_corpus_paper" if "sample_corpus_paper" in existing_runs else existing_runs[0]
                ws_base = os.environ.get("CITATION_WORKSPACE_DIR") or os.path.join(config.PROJECT_ROOT, "workspace_runs")
                plan_file = os.path.join(ws_base, preferred, "plan.json")
                try:
                    with open(plan_file, "r", encoding="utf-8") as f:
                        st.session_state["planner_active_graph"] = TaskGraph.from_dict(json.load(f))
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load plan: {e}")
