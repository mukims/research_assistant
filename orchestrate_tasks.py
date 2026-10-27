#!/usr/bin/env python3
"""
Scientific Paper-to-Agent Task Orchestrator CLI.

Deconstructs scientific papers into dependency graphs and orchestrates agent execution.

Usage:
  python orchestrate_tasks.py plan <paper_path> [--format json|markdown|mermaid|antigravity] [--output <path>] [--heuristic]
  python orchestrate_tasks.py run <plan_json> [--workspace <path>] [--mode simulation|autonomous] [--report-out <path>]
  python orchestrate_tasks.py show <plan_json> [--format markdown|mermaid]
"""

import argparse
import json
import os
import sys

from research_assistant.tasks import (
    ExecutionMode,
    TaskRunner,
    deconstruct_paper_into_task_graph,
    export_antigravity_specs,
    format_query_results_markdown,
    load_task_graph,
    query_task_database,
    render_markdown_plan,
    render_mermaid,
)


def cmd_plan(args: argparse.Namespace) -> int:
    paper_path = args.paper_path
    if not os.path.exists(paper_path):
        print(f"Error: Paper file '{paper_path}' not found.", file=sys.stderr)
        return 1

    print(f"Deconstructing paper: {paper_path} (heuristic={args.heuristic})...")
    try:
        graph = deconstruct_paper_into_task_graph(
            tei_or_path=paper_path,
            paper_id=args.paper_id,
            force_heuristic=args.heuristic,
            full_phases=getattr(args, "full_phases", False),
            build_refs=getattr(args, "build_refs", False),
        )
        from research_assistant.tasks.workspace import get_paper_workspace_dir
        ws_dir = get_paper_workspace_dir(graph.paper_id)
        print(f"Paper root workspace created at: {ws_dir} (all Markdown files materialized)")
    except Exception as exc:
        print(f"Error: Failed to deconstruct paper '{paper_path}': {exc}", file=sys.stderr)
        return 1

    fmt = args.format.lower()
    if fmt == "json":
        content = json.dumps(graph.to_dict(), indent=2)
    elif fmt == "markdown":
        content = render_markdown_plan(graph)
    elif fmt == "mermaid":
        content = render_mermaid(graph)
    elif fmt == "antigravity":
        content = json.dumps(export_antigravity_specs(graph), indent=2)
    else:
        print(f"Error: Unknown format '{fmt}'.", file=sys.stderr)
        return 1

    if args.output:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Plan exported successfully to: {args.output}")
        except Exception as exc:
            print(f"Error: Failed to write plan to '{args.output}': {exc}", file=sys.stderr)
            return 1
    else:
        print(content)

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    plan_path = args.plan_json
    if not os.path.exists(plan_path):
        print(f"Error: Plan file '{plan_path}' not found.", file=sys.stderr)
        return 1

    try:
        graph = load_task_graph(plan_path)
    except Exception as exc:
        print(f"Error: Failed to load plan file '{plan_path}': {exc}", file=sys.stderr)
        return 1

    from research_assistant.tasks.workspace import get_paper_workspace_dir
    workspace = args.workspace or get_paper_workspace_dir(graph.paper_id)
    mode = ExecutionMode.AUTONOMOUS if args.mode == "autonomous" else ExecutionMode.SIMULATION

    print(f"Executing Task Graph: {graph.paper_title} ({len(graph.tasks)} tasks)")
    print(f"Mode: {mode.value} | Workspace: {workspace}")

    def on_update(task, status):
        print(f"  [{status.value.upper()}] Task: {task.task_id} - {task.title}")

    runner = TaskRunner(
        graph=graph,
        workspace_dir=workspace,
        mode=mode,
        on_task_update=on_update,
    )
    report = runner.execute()

    print("\n" + report.render_markdown())

    if args.report_out:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.report_out)), exist_ok=True)
            with open(args.report_out, "w", encoding="utf-8") as f:
                if args.report_out.endswith(".json"):
                    json.dump(report.to_dict(), f, indent=2)
                else:
                    f.write(report.render_markdown())
            print(f"Execution report saved to: {args.report_out}")
        except Exception as exc:
            print(f"Error: Failed to write execution report to '{args.report_out}': {exc}", file=sys.stderr)
            return 1

    return 0 if report.failed_count == 0 else 1


def cmd_show(args: argparse.Namespace) -> int:
    plan_path = args.plan_json
    if not os.path.exists(plan_path):
        print(f"Error: Plan file '{plan_path}' not found.", file=sys.stderr)
        return 1

    try:
        graph = load_task_graph(plan_path)
    except Exception as exc:
        print(f"Error: Failed to load plan file '{plan_path}': {exc}", file=sys.stderr)
        return 1

    fmt = args.format.lower()
    if fmt == "mermaid":
        print(render_mermaid(graph))
    elif fmt == "markdown":
        print(render_markdown_plan(graph))
    elif fmt == "json":
        print(json.dumps(graph.to_dict(), indent=2))
    elif fmt == "antigravity":
        print(json.dumps(export_antigravity_specs(graph), indent=2))
    else:
        print(f"Error: Unknown format '{fmt}'.", file=sys.stderr)
        return 1
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    plan_path = args.plan_json
    if not os.path.exists(plan_path):
        print(f"Error: Plan file '{plan_path}' not found.", file=sys.stderr)
        return 1

    try:
        graph = load_task_graph(plan_path)
    except Exception as exc:
        print(f"Error: Failed to load plan file '{plan_path}': {exc}", file=sys.stderr)
        return 1

    try:
        result = query_task_database(
            query=args.query,
            graph_or_subgraph=graph,
            task_id=getattr(args, "task_id", None),
            scope=getattr(args, "scope", "subgraph"),
            top_k=getattr(args, "top_k", 5),
            expand_window=getattr(args, "expand_window", 1),
        )
    except Exception as exc:
        print(f"Error: Database query failed: {exc}", file=sys.stderr)
        return 1

    fmt = getattr(args, "format", "markdown").lower()
    if fmt == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(format_query_results_markdown(result))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Scientific Paper-to-Agent Task Orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # plan command
    p_plan = subparsers.add_parser("plan", help="Deconstruct a paper into an executable task DAG")
    p_plan.add_argument("paper_path", help="Path to TEI XML, PDF, or text file")
    p_plan.add_argument("--paper-id", help="Optional custom identifier for the paper")
    p_plan.add_argument("--format", choices=["json", "markdown", "mermaid", "antigravity"], default="json")
    p_plan.add_argument("--output", "-o", help="File to write the generated plan to")
    p_plan.add_argument("--heuristic", action="store_true", help="Force fast offline heuristic mode")
    p_plan.add_argument("--full-phases", action="store_true", help="Generate all 6 mandatory scientific replication phases")
    p_plan.add_argument("--build-refs", action="store_true", help="Extract, fetch, and ingest missing references around the paper")

    # run command
    p_run = subparsers.add_parser("run", help="Execute a planned task graph")
    p_run.add_argument("plan_json", help="Path to TaskGraph JSON file")
    p_run.add_argument("--workspace", "-w", help="Working directory for task outputs")
    p_run.add_argument("--mode", choices=["simulation", "autonomous"], default="simulation")
    p_run.add_argument("--report-out", help="Path to write execution scorecard report")

    # show command
    p_show = subparsers.add_parser("show", help="Display a task graph as markdown, mermaid, json, or antigravity")
    p_show.add_argument("plan_json", help="Path to TaskGraph JSON file")
    p_show.add_argument("--format", choices=["markdown", "mermaid", "json", "antigravity"], default="markdown")

    # query command
    p_query = subparsers.add_parser("query", help="Query the paper's reference database for a task")
    p_query.add_argument("plan_json", help="Path to TaskGraph JSON file")
    p_query.add_argument("--query", "-q", required=True, help="Methodological or equation query string")
    p_query.add_argument("--task-id", "-t", help="Optional task ID for task-scoped context")
    p_query.add_argument("--scope", choices=["subgraph", "seed", "cited", "global"], default="subgraph")
    p_query.add_argument("--top-k", type=int, default=5, help="Number of passages to return")
    p_query.add_argument("--expand-window", type=int, default=1, help="Neighbor chunk expansion window")
    p_query.add_argument("--format", choices=["markdown", "json"], default="markdown")

    args = parser.parse_args()
    if args.command == "plan":
        return cmd_plan(args)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "show":
        return cmd_show(args)
    elif args.command == "query":
        return cmd_query(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
