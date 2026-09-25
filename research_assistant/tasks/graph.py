"""
Task Graph synthesis, topological wave scheduling, and export visualizers.
"""

import json
import os
import re
from typing import Any, Dict, List

from research_assistant.tasks.models import (
    AgentRole,
    Task,
    TaskCategory,
    TaskGraph,
    TaskStatus,
)


def render_mermaid(graph: TaskGraph) -> str:
    """Renders a Mermaid flowchart representing the Task DAG with role icons and status styles."""
    lines = [
        "flowchart TD",
        "    %% Category Styling",
        "    classDef env fill:#E0F2FE,stroke:#0284C7,stroke-width:2px,color:#0369A1;",
        "    classDef data fill:#FEF3C7,stroke:#D97706,stroke-width:2px,color:#B45309;",
        "    classDef model fill:#DCFCE7,stroke:#16A34A,stroke-width:2px,color:#15803D;",
        "    classDef exp fill:#F3E8FF,stroke:#9333EA,stroke-width:2px,color:#7E22CE;",
        "    classDef eval fill:#FCE7F3,stroke:#DB2777,stroke-width:2px,color:#BE185D;",
        "    classDef audit fill:#F1F5F9,stroke:#475569,stroke-width:2px,color:#334155;",
        "",
    ]

    CATEGORY_CLASSES = {
        TaskCategory.ENVIRONMENT_SETUP: "env",
        TaskCategory.DATA_ACQUISITION: "data",
        TaskCategory.DATA_PREPROCESSING: "data",
        TaskCategory.MODEL_IMPLEMENTATION: "model",
        TaskCategory.TRAINING_OR_SIMULATION: "exp",
        TaskCategory.ABLATION_STUDY: "exp",
        TaskCategory.METRIC_EVALUATION: "eval",
        TaskCategory.VERIFICATION_AUDIT: "audit",
    }

    ROLE_ICONS = {
        AgentRole.DEVOPS_AGENT: "🛠️",
        AgentRole.DATA_ENGINEER: "📊",
        AgentRole.MODEL_ARCHITECT: "🧠",
        AgentRole.EXPERIMENT_RUNNER: "🧪",
        AgentRole.EVALUATION_AGENT: "📈",
        AgentRole.AUDIT_AGENT: "⚖️",
    }

    def _safe_id(name: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        if clean and clean[0].isdigit():
            clean = f"n_{clean}"
        return clean or "node"

    # Declare nodes
    for t_id, task in graph.tasks.items():
        nid = _safe_id(t_id)
        icon = ROLE_ICONS.get(task.assigned_role, "📋")
        safe_title = re.sub(r"[\r\n]+", " ", task.title).replace('"', "'")
        safe_title = safe_title.replace("[", "(").replace("]", ")")
        lines.append(f'    {nid}["{icon} {safe_title}"]')

    lines.append("")

    # Declare edges (dependencies: parent -> child)
    for t_id, task in graph.tasks.items():
        nid = _safe_id(t_id)
        for dep in task.dependencies:
            if dep in graph.tasks:
                dep_id = _safe_id(dep)
                lines.append(f"    {dep_id} --> {nid}")

    lines.append("")

    # Assign CSS classes
    for t_id, task in graph.tasks.items():
        nid = _safe_id(t_id)
        css_class = CATEGORY_CLASSES.get(task.category, "audit")
        lines.append(f"    class {nid} {css_class};")

    return "\n".join(lines)


def render_markdown_plan(graph: TaskGraph) -> str:
    """Generates an executive Markdown research execution plan from the Task DAG."""
    try:
        waves = graph.topological_waves()
    except Exception:
        waves = [[t for t in graph.tasks.values()]]

    lines = [
        f"# Research Execution Plan: {graph.paper_title}",
        "",
        f"**Paper ID:** `{graph.paper_id}`  ",
        f"**Total Execution Tasks:** {len(graph.tasks)} across {len(waves)} sequential waves  ",
        "",
        "---",
        "",
        "## 1. Execution Schedule (Topological Waves)",
        "",
        "| Wave | Task ID | Category | Assigned Agent | Dependencies | Deliverables |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for wave_idx, wave in enumerate(waves, 1):
        for task in wave:
            deps_str = ", ".join(f"`{d}`" for d in task.dependencies) if task.dependencies else "None"
            outputs_str = ", ".join(task.outputs[:2]) if task.outputs else "Report"
            lines.append(
                f"| Wave {wave_idx} | `{task.task_id}` | {task.category.value} | `{task.assigned_role.value}` | {deps_str} | {outputs_str} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Dependency Architecture (DAG)",
        "",
        "```mermaid",
        render_mermaid(graph),
        "```",
        "",
        "---",
        "",
        "## 3. Detailed Task Specifications",
        "",
    ])

    for t_id, task in graph.tasks.items():
        lines.extend([
            f"### `{task.task_id}`: {task.title}",
            f"- **Category:** `{task.category.value}`",
            f"- **Assigned Role:** `{task.assigned_role.value}`",
            f"- **Prerequisites:** {', '.join(f'`{d}`' for d in task.dependencies) if task.dependencies else 'None'}",
            f"- **Inputs Required:** {', '.join(task.inputs) if task.inputs else 'None'}",
            f"- **Outputs Deliverable:** {', '.join(task.outputs) if task.outputs else 'None'}",
            "",
            "#### Description & Instructions",
            task.description,
            "",
        ])

        if task.acceptance_criteria:
            lines.append("#### Acceptance Criteria (Pass/Fail)")
            for ac in task.acceptance_criteria:
                lines.append(f"- [ ] {ac}")
            lines.append("")

        if task.paper_context_excerpts:
            lines.append("#### Relevant Paper Excerpts")
            for excerpt in task.paper_context_excerpts:
                lines.append(f"> \"{excerpt.strip()}\"")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def export_antigravity_specs(graph: TaskGraph) -> List[Dict[str, Any]]:
    """
    Transforms the TaskGraph into a list of Subagent invocation specifications
    compatible with the Antigravity / AGY subagent framework.
    """
    specs = []
    for task in graph.tasks.values():
        role_label = task.assigned_role.value.replace("_", " ").title()
        prompt_parts = [
            f"# Mission: {task.title}",
            "",
            f"**Task ID:** {task.task_id}",
            f"**Category:** {task.category.value}",
            f"**Paper:** {graph.paper_title}",
            "",
            "## Objectives & Instructions",
            task.description,
            "",
            "## Required Deliverables",
        ]
        for out in task.outputs:
            prompt_parts.append(f"- {out}")

        if task.acceptance_criteria:
            prompt_parts.append("\n## Acceptance Criteria")
            for crit in task.acceptance_criteria:
                prompt_parts.append(f"- [ ] {crit}")

        if task.paper_context_excerpts:
            prompt_parts.append("\n## Reference Paper Context")
            for ex in task.paper_context_excerpts:
                prompt_parts.append(f"> {ex}")

        specs.append({
            "TypeName": "self",
            "Role": role_label,
            "Prompt": "\n".join(prompt_parts),
            "Metadata": {
                "task_id": task.task_id,
                "dependencies": task.dependencies,
                "inputs": task.inputs,
                "outputs": task.outputs,
            },
        })
    return specs


def save_task_graph(graph: TaskGraph, path: str) -> None:
    """Atomically saves the TaskGraph to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temp_path = f"{path}.tmp.{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(graph.to_dict(), f, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)


def load_task_graph(path: str) -> TaskGraph:
    """Loads a TaskGraph from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TaskGraph.from_dict(data)
