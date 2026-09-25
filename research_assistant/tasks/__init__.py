"""
Scientific Paper-to-Agent Task Decomposition Pipeline.

Deconstructs scientific manuscripts into dependency-ordered Directed Acyclic
Graphs (DAGs) of executable tasks for autonomous AI agents.
"""

from research_assistant.tasks.models import (
    AgentRole,
    Task,
    TaskCategory,
    TaskGraph,
    TaskStatus,
)
from research_assistant.tasks.graph import (
    export_antigravity_specs,
    load_task_graph,
    render_markdown_plan,
    render_mermaid,
    save_task_graph,
)
from research_assistant.tasks.extractor import (
    deconstruct_paper_into_task_graph,
    extract_paper_planning_context,
)
from research_assistant.tasks.roles import (
    RoleSpec,
    format_agent_prompt,
    get_all_roles,
    get_role_spec,
)
from research_assistant.tasks.runner import (
    ExecutionMode,
    ExecutionReport,
    TaskExecutionResult,
    TaskRunner,
)

__all__ = [
    "Task",
    "TaskGraph",
    "TaskCategory",
    "TaskStatus",
    "AgentRole",
    "export_antigravity_specs",
    "load_task_graph",
    "render_markdown_plan",
    "render_mermaid",
    "save_task_graph",
    "deconstruct_paper_into_task_graph",
    "extract_paper_planning_context",
    "RoleSpec",
    "format_agent_prompt",
    "get_all_roles",
    "get_role_spec",
    "ExecutionMode",
    "ExecutionReport",
    "TaskExecutionResult",
    "TaskRunner",
]
