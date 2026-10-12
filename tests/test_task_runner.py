"""
Unit tests for TaskRunner wave execution and replication scorecards.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

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


class TestTaskRunner(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_task_runner_simulation(self):
        graph = _sample_graph()
        workspace = os.path.join(self.tmp_dir, "run_sim")

        updates = []
        def on_update(task, status):
            updates.append((task.task_id, status))

        runner = TaskRunner(
            graph=graph,
            workspace_dir=workspace,
            mode=ExecutionMode.SIMULATION,
            on_task_update=on_update,
        )
        report = runner.execute()

        self.assertEqual(report.total_tasks, 4)
        self.assertEqual(report.completed_count, 4)
        self.assertEqual(report.failed_count, 0)
        self.assertEqual(report.blocked_count, 0)
        self.assertEqual(report.scorecard["rating"], "CERTIFIED_REPLICABLE")
        self.assertEqual(report.scorecard["replication_rate"], 1.0)

        # Verify physical output files were created in workspace
        self.assertTrue(os.path.exists(os.path.join(workspace, "env.lock")))
        self.assertTrue(os.path.exists(os.path.join(workspace, "data", "processed.h5")))
        self.assertTrue(os.path.exists(os.path.join(workspace, "model.py")))
        self.assertTrue(os.path.exists(os.path.join(workspace, "metrics.json")))

        # Check updates list has events
        self.assertGreater(len(updates), 0)

        # Markdown rendering check
        md = report.render_markdown()
        self.assertIn("# Replication Scorecard: Quantum AI", md)
        self.assertIn("CERTIFIED_REPLICABLE", md)
        self.assertIn("| `t1` | ✅ completed |", md)

    def test_task_runner_blocked_downstream(self):
        graph = _sample_graph()
        workspace = os.path.join(self.tmp_dir, "run_blocked")

        runner = TaskRunner(
            graph=graph,
            workspace_dir=workspace,
            mode=ExecutionMode.SIMULATION,
        )

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

        self.assertEqual(report.completed_count, 0)
        self.assertEqual(report.failed_count, 1)
        self.assertEqual(report.blocked_count, 3)  # t2, t3, t4 are blocked
        self.assertEqual(report.scorecard["rating"], "REPLICATION_FAILED")
        self.assertEqual(report.results["t1"].status, TaskStatus.FAILED)
        self.assertEqual(report.results["t2"].status, TaskStatus.BLOCKED)
        self.assertEqual(report.results["t3"].status, TaskStatus.BLOCKED)
        self.assertEqual(report.results["t4"].status, TaskStatus.BLOCKED)

    def test_task_runner_invalid_graph(self):
        graph = TaskGraph(paper_id="bad", paper_title="Bad Graph")
        t1 = Task(task_id="t1", title="T1", dependencies=["non_existent"])
        graph.add_task(t1)

        runner = TaskRunner(graph=graph, workspace_dir=self.tmp_dir)
        with self.assertRaises(ValueError):
            runner.execute()

    @patch("research_assistant.shared.llm.chat")
    def test_task_runner_autonomous(self, mock_chat):
        mock_res = MagicMock()
        mock_res.content = "def train(): print('Model synthesized successfully')\n"
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

        workspace = os.path.join(self.tmp_dir, "run_auto")
        runner = TaskRunner(
            graph=graph,
            workspace_dir=workspace,
            mode=ExecutionMode.AUTONOMOUS,
        )
        report = runner.execute()

        self.assertEqual(report.completed_count, 1)
        out_path = os.path.join(workspace, "src", "model.py")
        self.assertTrue(os.path.exists(out_path))
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("def train()", content)


if __name__ == "__main__":
    unittest.main()
