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
    load_task_graph,
    render_markdown_plan,
    render_mermaid,
    save_task_graph,
)


def cmd_plan(args: argparse.Namespace) -> int:
    paper_path = args.paper_path
    if not os.path.exists(paper_path):
        print(f"Error: Paper file '{paper_path}' not found.", file=sys.stderr)
        return 1

    print(f"Deconstructing paper: {paper_path} (heuristic={args.heuristic})...")
    graph = deconstruct_paper_into_task_graph(
        tei_or_path=paper_path,
        paper_id=args.paper_id,
        force_heuristic=args.heuristic,
    )

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
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Plan exported successfully to: {args.output}")
    else:
        print(content)

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    plan_path = args.plan_json
    if not os.path.exists(plan_path):
        print(f"Error: Plan file '{plan_path}' not found.", file=sys.stderr)
        return 1

    graph = load_task_graph(plan_path)
    workspace = args.workspace or os.path.join("workspace_runs", graph.paper_id)
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
        os.makedirs(os.path.dirname(os.path.abspath(args.report_out)), exist_ok=True)
        with open(args.report_out, "w", encoding="utf-8") as f:
            if args.report_out.endswith(".json"):
                json.dump(report.to_dict(), f, indent=2)
            else:
                f.write(report.render_markdown())
        print(f"Execution report saved to: {args.report_out}")

    return 0 if report.failed_count == 0 else 1


def cmd_show(args: argparse.Namespace) -> int:
    plan_path = args.plan_json
    if not os.path.exists(plan_path):
        print(f"Error: Plan file '{plan_path}' not found.", file=sys.stderr)
        return 1

    graph = load_task_graph(plan_path)
    fmt = args.format.lower()
    if fmt == "mermaid":
        print(render_mermaid(graph))
    else:
        print(render_markdown_plan(graph))
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

    # run command
    p_run = subparsers.add_parser("run", help="Execute a planned task graph")
    p_run.add_argument("plan_json", help="Path to TaskGraph JSON file")
    p_run.add_argument("--workspace", "-w", help="Working directory for task outputs")
    p_run.add_argument("--mode", choices=["simulation", "autonomous"], default="simulation")
    p_run.add_argument("--report-out", help="Path to write execution scorecard report")

    # show command
    p_show = subparsers.add_parser("show", help="Display a task graph as markdown or mermaid")
    p_show.add_argument("plan_json", help="Path to TaskGraph JSON file")
    p_show.add_argument("--format", choices=["markdown", "mermaid"], default="markdown")

    args = parser.parse_args()
    if args.command == "plan":
        return cmd_plan(args)
    elif args.command == "run":
        return cmd_run(args)
    elif args.command == "show":
        return cmd_show(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
