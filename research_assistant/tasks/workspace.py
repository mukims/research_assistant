"""
Paper Workspace Manager & Markdown Bundle Generator.

Manages dedicated root directories for scientific papers upon upload or deconstruction,
and materializes comprehensive Markdown artifacts (plan.md, dag.md, scorecard.md,
paper_summary.md, rationale.md, rationales/*.md, and README.md).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional

from research_assistant import config
from research_assistant.shared.log import get_logger
from research_assistant.tasks.models import TaskGraph
from research_assistant.tasks.graph import (
    export_antigravity_specs,
    render_markdown_plan,
    render_mermaid,
)

logger = get_logger("task_workspace")


def get_paper_workspace_dir(paper_id: str, base_dir: Optional[str] = None) -> str:
    """
    Computes and creates the dedicated root directory for a paper.
    Enforces storage invariant safety (never inside persistent data or vectordb).
    """
    clean_id = re.sub(r"[^\w\-]", "_", str(paper_id).lstrip(".")).strip("_") or "paper"

    workspace_base = base_dir or os.environ.get("CITATION_WORKSPACE_DIR") or os.path.join(config.PROJECT_ROOT, "workspace_runs")

    abs_data = os.path.abspath(getattr(config, "DATA_DIR", os.path.join(config.PROJECT_ROOT, "data")))
    abs_vdb = os.path.abspath(getattr(config, "VECTORDB_PATH", os.path.join(abs_data, "physics_vectordb")))
    abs_ws = os.path.abspath(workspace_base)

    if (
        abs_ws == abs_data
        or abs_ws == abs_vdb
        or abs_ws.startswith(abs_data + os.sep)
        or abs_ws.startswith(abs_vdb + os.sep)
    ):
        workspace_base = os.path.join(os.path.dirname(abs_data), "workspace_runs")

    target_dir = os.path.abspath(os.path.join(workspace_base, clean_id))
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def render_paper_summary_markdown(
    title: str,
    paper_id: str,
    context: Dict[str, Any],
    source_filename: Optional[str] = None,
) -> str:
    """
    Renders an extracted paper summary document in Markdown.
    Contains metadata, abstract, methodology sections, and tables/figures.
    """
    source_label = source_filename or "N/A"
    abstract_text = context.get("abstract") or "No abstract provided in source document."

    lines = [
        f"# Paper Summary: {title}",
        f"- **Paper ID**: `{paper_id}`",
        f"- **Source File**: `{source_label}`",
        f"- **Extracted Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Abstract",
        abstract_text.strip(),
        "",
        "## Extracted Methodology & Theoretical Framework",
    ]

    sections = context.get("sections") or []
    if not sections:
        lines.append("*(No discrete methodology sections extracted)*")
    else:
        for idx, sec in enumerate(sections, 1):
            if isinstance(sec, str):
                sec_clean = sec.strip()
                if sec_clean.startswith("#"):
                    lines.append(sec_clean)
                else:
                    lines.append(f"### Section {idx}\n{sec_clean}")
            elif isinstance(sec, (list, tuple)) and len(sec) >= 2:
                head, paras = sec[0], sec[1]
                lines.append(f"### {head}")
                if isinstance(paras, list):
                    lines.append("\n\n".join(str(p).strip() for p in paras if p))
                else:
                    lines.append(str(paras).strip())
            lines.append("")

    tables = context.get("tables") or []
    if tables:
        lines.append("## Tables, Figures & Empirical Benchmarks")
        for t in tables:
            lines.append(f"- {str(t).strip()}")
        lines.append("")

    deref = context.get("dereferenced_citations") or []
    if deref:
        lines.append("## Dereferenced Methodological Citations")
        for c in deref:
            ref_title = c.get("title") or c.get("key") or "Reference"
            matched = c.get("matched_document_title") or c.get("matched_document_key") or "External"
            lines.append(f"- **{ref_title}** → `{matched}`")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_workspace_readme(
    graph: TaskGraph,
    manifest: Dict[str, str],
    source_filename: Optional[str] = None,
) -> str:
    """
    Renders a comprehensive README.md index for the paper root directory.
    """
    try:
        waves = graph.topological_waves()
        wave_count = len(waves)
    except Exception:
        wave_count = 1

    roles = {t.assigned_role.value if hasattr(t.assigned_role, "value") else str(t.assigned_role) for t in graph.tasks.values()}
    total_outputs = sum(len(t.outputs) for t in graph.tasks.values())
    source_label = source_filename or "source_paper"

    lines = [
        f"# {graph.paper_title}",
        "",
        f"- **Paper ID**: `{graph.paper_id}`",
        f"- **Source File**: `{source_label}`",
        f"- **Total Tasks**: {len(graph.tasks)}",
        f"- **Execution Waves**: {wave_count}",
        f"- **Specialized Agent Roles**: {len(roles)} ({', '.join(sorted(roles))})",
        f"- **Expected Deliverable Artifacts**: {total_outputs}",
        f"- **Created At**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Workspace Markdown Files",
        "",
        "| File | Description |",
        "| :--- | :--- |",
        "| [`plan.md`](./plan.md) | Complete task replication DAG, wave sequencing, role assignments & artifact contracts |",
        "| [`dag.md`](./dag.md) | Mermaid flowchart diagram of dependency relationships |",
        "| [`scorecard.md`](./scorecard.md) | Two-level critic rubric scorecard evaluating topology, safety & contracts |",
        "| [`paper_summary.md`](./paper_summary.md) | Extracted abstract, theoretical formulation, methodology sections & tables |",
        "| [`rationale.md`](./rationale.md) | Primary scientific hypothesis, physical invariants & complexity bounds |",
        "| [`rationales/`](./rationales/) | Directory of individual task rationales for each subagent |",
        "",
        "## Machine-Readable Specifications",
        "- [`plan.json`](./plan.json): Serialized TaskGraph DAG definition.",
        "- [`antigravity_specs.json`](./antigravity_specs.json): Subagent execution specifications for Google Antigravity.",
        "",
        "## Execution Waves Summary",
    ]

    try:
        for w_idx, wave in enumerate(waves, 1):
            lines.append(f"### Wave {w_idx} ({len(wave)} tasks)")
            for task in wave:
                r_name = task.assigned_role.value if hasattr(task.assigned_role, "value") else str(task.assigned_role)
                lines.append(f"- **`{task.task_id}`**: {task.title} (`{r_name}`)")
            lines.append("")
    except Exception:
        pass

    return "\n".join(lines).strip() + "\n"


def materialize_paper_upload(
    source: Any,
    filename: Optional[str] = None,
    base_dir: Optional[str] = None,
) -> str:
    """
    Called whenever a paper is uploaded.
    Creates a dedicated root directory for the paper, copies/saves the manuscript,
    and drops initial Markdown files (README.md, paper_summary.md).
    """
    raw_name = filename or getattr(source, "name", None) or "uploaded_paper.pdf"
    base_name = os.path.basename(str(raw_name))
    stem = os.path.splitext(base_name)[0]
    clean_id = re.sub(r"[^\w\-_.]", "_", stem).strip("_") or "paper"

    workspace_dir = get_paper_workspace_dir(clean_id, base_dir=base_dir)

    # Save a copy of the uploaded file inside the paper workspace
    dest_path = os.path.join(workspace_dir, base_name)
    if isinstance(source, (bytes, bytearray)):
        with open(dest_path, "wb") as f:
            f.write(source)
    elif isinstance(source, str) and os.path.isfile(source):
        if os.path.abspath(source) != os.path.abspath(dest_path):
            shutil.copy2(source, dest_path)
    elif hasattr(source, "getvalue") and callable(source.getvalue):
        with open(dest_path, "wb") as f:
            f.write(source.getvalue())

    # Extract context and render paper_summary.md
    try:
        from research_assistant.tasks.extractor import extract_paper_planning_context
        context = extract_paper_planning_context(dest_path if os.path.exists(dest_path) else source)
    except Exception as exc:
        logger.warning("Could not extract paper context for upload: %s", exc)
        context = {"title": stem, "abstract": "", "sections": []}

    title = context.get("title") or stem
    summary_md = render_paper_summary_markdown(title, clean_id, context, source_filename=base_name)
    summary_path = os.path.join(workspace_dir, "paper_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)

    # Write initial README.md
    readme_path = os.path.join(workspace_dir, "README.md")
    readme_content = f"""# {title}

- **Paper ID**: `{clean_id}`
- **Source File**: `{base_name}`
- **Status**: Uploaded / Staged (Ready for Task Deconstruction)
- **Staged At**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

## Available Markdown Files
- [`paper_summary.md`](./paper_summary.md): Extracted abstract, theoretical formulation, and methodology sections.
- [`{base_name}`](./{base_name}): Original manuscript file.
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    logger.info("Materialized paper upload in workspace: %s", workspace_dir)
    return workspace_dir


def materialize_deconstructed_paper_bundle(
    graph: TaskGraph,
    workspace_dir: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    tei_or_path: Optional[Any] = None,
) -> Dict[str, str]:
    """
    Called whenever a paper is deconstructed.
    Materializes a complete root directory for the paper and drops ALL markdown files:
    1. plan.md
    2. dag.md
    3. scorecard.md
    4. paper_summary.md
    5. rationale.md
    6. rationales/<task_id>_rationale.md
    7. README.md
    Also drops plan.json and antigravity_specs.json.
    """
    from research_assistant.tasks.critic import evaluate_workflow
    from research_assistant.tasks.runner import generate_task_rationale

    ws_dir = get_paper_workspace_dir(graph.paper_id, base_dir=workspace_dir)
    manifest: Dict[str, str] = {}

    source_filename = None
    if tei_or_path and isinstance(tei_or_path, (str, os.PathLike)):
        p_str = os.fspath(tei_or_path)
        if os.path.isfile(p_str):
            source_filename = os.path.basename(p_str)
            target_copy = os.path.join(ws_dir, source_filename)
            if os.path.abspath(p_str) != os.path.abspath(target_copy):
                try:
                    shutil.copy2(p_str, target_copy)
                    manifest[source_filename] = target_copy
                except Exception as e:
                    logger.debug("Could not copy paper into workspace: %s", e)

    # 1. plan.md
    plan_md = render_markdown_plan(graph)
    plan_path = os.path.join(ws_dir, "plan.md")
    with open(plan_path, "w", encoding="utf-8") as f:
        f.write(plan_md)
    manifest["plan.md"] = plan_path

    # 2. dag.md (Mermaid flowchart)
    mermaid_code = render_mermaid(graph)
    dag_md = f"""# Replication DAG: {graph.paper_title}

**Paper ID**: `{graph.paper_id}`  
**Tasks**: {len(graph.tasks)}

```mermaid
{mermaid_code}
```
"""
    dag_path = os.path.join(ws_dir, "dag.md")
    with open(dag_path, "w", encoding="utf-8") as f:
        f.write(dag_md)
    manifest["dag.md"] = dag_path

    # 3. scorecard.md (Critic rubric scorecard)
    try:
        scorecard = evaluate_workflow(graph)
        scorecard_md = scorecard.render_markdown()
    except Exception as exc:
        scorecard_md = f"# Replication Workflow Rubric Scorecard: {graph.paper_title}\n\n*(Scorecard evaluation error: {exc})*\n"
    scorecard_path = os.path.join(ws_dir, "scorecard.md")
    with open(scorecard_path, "w", encoding="utf-8") as f:
        f.write(scorecard_md)
    manifest["scorecard.md"] = scorecard_path

    # 4. paper_summary.md
    if context:
        summary_md = render_paper_summary_markdown(graph.paper_title, graph.paper_id, context, source_filename)
        summary_path = os.path.join(ws_dir, "paper_summary.md")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_md)
        manifest["paper_summary.md"] = summary_path

    # 5. rationales/<task_id>_rationale.md and root rationale.md
    rationales_dir = os.path.join(ws_dir, "rationales")
    os.makedirs(rationales_dir, exist_ok=True)
    first_rationale = None

    for task in graph.tasks.values():
        rat_text = generate_task_rationale(task, ws_dir)
        task_rat_path = os.path.join(rationales_dir, f"{task.task_id}_rationale.md")
        with open(task_rat_path, "w", encoding="utf-8") as f:
            f.write(rat_text)
        manifest[f"rationales/{task.task_id}_rationale.md"] = task_rat_path
        if first_rationale is None:
            first_rationale = rat_text

    # Write root rationale.md
    if first_rationale:
        root_rat_path = os.path.join(ws_dir, "rationale.md")
        with open(root_rat_path, "w", encoding="utf-8") as f:
            f.write(first_rationale)
        manifest["rationale.md"] = root_rat_path

    # 6. README.md
    readme_md = render_workspace_readme(graph, manifest, source_filename)
    readme_path = os.path.join(ws_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_md)
    manifest["README.md"] = readme_path

    # 7. Machine-readable JSON specifications
    plan_json_path = os.path.join(ws_dir, "plan.json")
    with open(plan_json_path, "w", encoding="utf-8") as f:
        json.dump(graph.to_dict(), f, indent=2)
    manifest["plan.json"] = plan_json_path

    ag_specs_path = os.path.join(ws_dir, "antigravity_specs.json")
    with open(ag_specs_path, "w", encoding="utf-8") as f:
        json.dump(export_antigravity_specs(graph), f, indent=2)
    manifest["antigravity_specs.json"] = ag_specs_path

    logger.info("Materialized %d files for paper '%s' in %s", len(manifest), graph.paper_id, ws_dir)
    return manifest
