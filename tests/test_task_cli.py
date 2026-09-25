"""
Unit and integration tests for orchestrate_tasks.py CLI.
"""

import json
import os
import subprocess
import sys
import pytest


def test_cli_help():
    res = subprocess.run([sys.executable, "orchestrate_tasks.py", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "plan" in res.stdout
    assert "run" in res.stdout
    assert "show" in res.stdout


def test_cli_plan_heuristic(tmp_path):
    paper = tmp_path / "paper.txt"
    paper.write_text("Title: Quantum ML\nMethod: Train neural networks on quantum states.")
    out_json = tmp_path / "plan.json"

    res = subprocess.run([
        sys.executable, "orchestrate_tasks.py", "plan",
        str(paper),
        "--paper-id", "qml_paper",
        "--heuristic",
        "--format", "json",
        "-o", str(out_json),
    ], capture_output=True, text=True)

    assert res.returncode == 0
    assert out_json.exists()

    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["paper_id"] == "qml_paper"
    assert len(data["tasks"]) == 5


def test_cli_plan_formats(tmp_path):
    paper = tmp_path / "paper.txt"
    paper.write_text("Method: Sample methodology.")

    for fmt in ["markdown", "mermaid", "antigravity"]:
        out_file = tmp_path / f"plan.{fmt}"
        res = subprocess.run([
            sys.executable, "orchestrate_tasks.py", "plan",
            str(paper),
            "--heuristic",
            "--format", fmt,
            "-o", str(out_file),
        ], capture_output=True, text=True)
        assert res.returncode == 0
        assert out_file.exists()
        assert len(out_file.read_text(encoding="utf-8")) > 50


def test_cli_show(tmp_path):
    paper = tmp_path / "paper.txt"
    paper.write_text("Method: Sample methodology.")
    plan_file = tmp_path / "plan.json"

    # Generate plan
    subprocess.run([
        sys.executable, "orchestrate_tasks.py", "plan",
        str(paper), "--heuristic", "-o", str(plan_file),
    ], check=True)

    # Show markdown
    res = subprocess.run([
        sys.executable, "orchestrate_tasks.py", "show",
        str(plan_file), "--format", "markdown",
    ], capture_output=True, text=True)
    assert res.returncode == 0
    assert "# Research Execution Plan" in res.stdout

    # Show mermaid
    res_m = subprocess.run([
        sys.executable, "orchestrate_tasks.py", "show",
        str(plan_file), "--format", "mermaid",
    ], capture_output=True, text=True)
    assert res_m.returncode == 0
    assert "flowchart TD" in res_m.stdout


def test_cli_run_simulation(tmp_path):
    paper = tmp_path / "paper.txt"
    paper.write_text("Method: Sample methodology.")
    plan_file = tmp_path / "plan.json"
    workspace = tmp_path / "cli_workspace"
    report_file = tmp_path / "report.md"

    # Plan
    subprocess.run([
        sys.executable, "orchestrate_tasks.py", "plan",
        str(paper), "--heuristic", "-o", str(plan_file),
    ], check=True)

    # Run
    res = subprocess.run([
        sys.executable, "orchestrate_tasks.py", "run",
        str(plan_file),
        "--workspace", str(workspace),
        "--mode", "simulation",
        "--report-out", str(report_file),
    ], capture_output=True, text=True)

    assert res.returncode == 0
    assert report_file.exists()
    report_content = report_file.read_text(encoding="utf-8")
    assert "Replication Scorecard" in report_content
    assert "CERTIFIED_REPLICABLE" in report_content
