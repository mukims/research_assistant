"""
Scientific Paper-to-Agent Task Decomposition Pipeline.

Deconstructs scientific manuscripts into dependency-ordered Directed Acyclic
Graphs (DAGs) of executable tasks for autonomous AI agents.
"""

from research_assistant.tasks.models import (
    AgentRole,
    ArtifactContract,
    ArtifactType,
    Task,
    TaskCategory,
    TaskContextPack,
    TaskGraph,
    TaskStatus,
)
from research_assistant.tasks.retrieval import (
    build_paper_reference_subgraph,
    compile_task_context_pack,
    dereference_method_citations,
    format_query_results_markdown,
    populate_graph_context_packs,
    query_task_database,
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
    generate_task_rationale,
)
from research_assistant.tasks.workspace import (
    get_paper_workspace_dir,
    materialize_deconstructed_paper_bundle,
    materialize_paper_upload,
    render_paper_summary_markdown,
)

__all__ = [
    "Task",
    "TaskGraph",
    "TaskCategory",
    "TaskContextPack",
    "TaskStatus",
    "AgentRole",
    "ArtifactContract",
    "ArtifactType",
    "generate_task_rationale",
    "get_paper_workspace_dir",
    "materialize_deconstructed_paper_bundle",
    "materialize_paper_upload",
    "render_paper_summary_markdown",
    "build_paper_reference_subgraph",
    "compile_task_context_pack",
    "dereference_method_citations",
    "format_query_results_markdown",
    "populate_graph_context_packs",
    "query_task_database",
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
