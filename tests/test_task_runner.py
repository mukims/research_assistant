"""
Unit tests for TaskRunner wave execution and replication scorecards.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from research_assistant.tasks.models import (
    AgentRole,
    Task,
    TaskCategory,
    TaskGraph,
    TaskStatus,
)
from research_assistant.tasks.runner import (
    ExecutionMode,
    ExecutionReport,
    TaskExecutionResult,
    TaskRunner,
)


def _sample_graph() -> TaskGraph:
    graph = TaskGraph(paper_id="quantum_test", paper_title="Quantum AI")
    t1 = Task(
        task_id="t1",
        title="Setup Env",
        category=TaskCategory.ENVIRONMENT_SETUP,
        assigned_role=AgentRole.DEVOPS_AGENT,
        outputs=["env.lock"],
    )
    t2 = Task(
        task_id="t2",
        title="Prep Data",
        category=TaskCategory.DATA_PREPROCESSING,
        assigned_role=AgentRole.DATA_ENGINEER,
        dependencies=["t1"],
        outputs=["data/processed.h5"],
    )
    t3 = Task(
        task_id="t3",
        title="Run Model",
        category=TaskCategory.MODEL_IMPLEMENTATION,
        assigned_role=AgentRole.MODEL_ARCHITECT,
        dependencies=["t1"],
        outputs=["model.py"],
    )
    t4 = Task(
        task_id="t4",
        title="Evaluate",
        category=TaskCategory.METRIC_EVALUATION,
        assigned_role=AgentRole.EVALUATION_AGENT,
        dependencies=["t2", "t3"],
        outputs=["metrics.json"],
    )
    for t in [t1, t2, t3, t4]:
        graph.add_task(t)
    return graph


def test_task_runner_simulation(tmp_path):
    graph = _sample_graph()
    workspace = tmp_path / "run_sim"

    updates = []
    def on_update(task, status):
        updates.append((task.task_id, status))

    runner = TaskRunner(
        graph=graph,
        workspace_dir=str(workspace),
        mode=ExecutionMode.SIMULATION,
        on_task_update=on_update,
    )
    report = runner.execute()

    assert report.total_tasks == 4
    assert report.completed_count == 4
    assert report.failed_count == 0
    assert report.blocked_count == 0
    assert report.scorecard["rating"] == "CERTIFIED_REPLICABLE"
    assert report.scorecard["replication_rate"] == 1.0

    # Verify physical output files were created in workspace
    assert (workspace / "env.lock").exists()
    assert (workspace / "data" / "processed.h5").exists()
    assert (workspace / "model.py").exists()
    assert (workspace / "metrics.json").exists()

    # Check updates list has events
    assert len(updates) > 0

    # Markdown rendering check
    md = report.render_markdown()
    assert "# Replication Scorecard: Quantum AI" in md
    assert "CERTIFIED_REPLICABLE" in md
    assert "| `t1` | ✅ completed |" in md


def test_task_runner_blocked_downstream(tmp_path):
    graph = _sample_graph()
    workspace = tmp_path / "run_blocked"

    runner = TaskRunner(
        graph=graph,
        workspace_dir=str(workspace),
        mode=ExecutionMode.SIMULATION,
    )

    # Force t1 to simulate failure by patching _simulate_task
    def mock_simulate(task, t_start):
        if task.task_id == "t1":
            return TaskExecutionResult(
                task_id="t1",
                status=TaskStatus.FAILED,
                duration_seconds=0.1,
                error_message="Environment build error",
            )
        return TaskExecutionResult(task_id=task.task_id, status=TaskStatus.COMPLETED, duration_seconds=0.1)

    with patch.object(runner, "_simulate_task", side_effect=mock_simulate):
        report = runner.execute()

    assert report.completed_count == 0
    assert report.failed_count == 1
    assert report.blocked_count == 3  # t2, t3, t4 are blocked
    assert report.scorecard["rating"] == "REPLICATION_FAILED"
    assert report.results["t1"].status == TaskStatus.FAILED
    assert report.results["t2"].status == TaskStatus.BLOCKED
    assert report.results["t3"].status == TaskStatus.BLOCKED
    assert report.results["t4"].status == TaskStatus.BLOCKED


def test_task_runner_invalid_graph(tmp_path):
    graph = TaskGraph(paper_id="bad", paper_title="Bad Graph")
    t1 = Task(task_id="t1", title="T1", dependencies=["non_existent"])
    graph.add_task(t1)

    runner = TaskRunner(graph=graph, workspace_dir=str(tmp_path))
    with pytest.raises(ValueError, match="Cannot execute invalid TaskGraph"):
        runner.execute()


@patch("research_assistant.shared.llm.chat")
def test_task_runner_autonomous(mock_chat, tmp_path):
    mock_res = MagicMock()
    mock_res.content = "def train(): print('Model synthesized successfully')"
    mock_chat.return_value = mock_res

    graph = TaskGraph(paper_id="auto_test", paper_title="Auto Test")
    t1 = Task(
        task_id="t1",
        title="Generate Model",
        category=TaskCategory.MODEL_IMPLEMENTATION,
        assigned_role=AgentRole.MODEL_ARCHITECT,
        outputs=["src/model.py"],
    )
    graph.add_task(t1)

    workspace = tmp_path / "run_auto"
    runner = TaskRunner(
        graph=graph,
        workspace_dir=str(workspace),
        mode=ExecutionMode.AUTONOMOUS,
    )
    report = runner.execute()

    assert report.completed_count == 1
    assert (workspace / "src" / "model.py").exists()
    content = (workspace / "src" / "model.py").read_text(encoding="utf-8")
    assert "def train()" in content
