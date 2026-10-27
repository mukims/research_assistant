"""
Unit tests for TaskRunner Two-Cycle Reflexion loop and error recovery.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from research_assistant.config import DATA_DIR, VECTORDB_PATH
from research_assistant.tasks.models import (
    AgentRole,
    ArtifactContract,
    ArtifactType,
    Task,
    TaskCategory,
    TaskGraph,
    TaskStatus,
)
from research_assistant.tasks.runner import (
    ExecutionMode,
    FailureDiagnostics,
    TaskRunner,
    parse_execution_error,
)


class TestTaskRunnerReflection(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_simulation_mode_creates_contract_compliant_artifacts(self):
        graph = TaskGraph(paper_id="sim_paper", paper_title="Simulation Test")
        t = Task(
            task_id="t_metric",
            title="Compute Metrics",
            category=TaskCategory.METRIC_EVALUATION,
            outputs=["results/eval.json"],
            artifact_contracts=[
                ArtifactContract(
                    path="results/eval.json",
                    artifact_type=ArtifactType.METRICS_JSON,
                    required_keys=["accuracy", "f1"],
                )
            ],
        )
        graph.add_task(t)

        runner = TaskRunner(
            graph=graph,
            workspace_dir=self.tmp_dir,
            mode=ExecutionMode.SIMULATION,
        )
        report = runner.execute()

        self.assertEqual(report.completed_count, 1)
        self.assertEqual(report.failed_count, 0)
        out_file = os.path.join(self.tmp_dir, "results/eval.json")
        self.assertTrue(os.path.exists(out_file))

    @patch("research_assistant.shared.llm.chat")
    def test_autonomous_mode_cycle_1_pass_cleanly(self, mock_chat):
        mock_chat.return_value = MagicMock(content="def train(): pass")

        graph = TaskGraph(paper_id="auto_paper", paper_title="Auto Test")
        t = Task(
            task_id="t_code",
            title="Write Code",
            category=TaskCategory.MODEL_IMPLEMENTATION,
            outputs=["model.py"],
            artifact_contracts=[
                ArtifactContract(
                    path="model.py",
                    artifact_type=ArtifactType.CODE,
                )
            ],
        )
        graph.add_task(t)

        runner = TaskRunner(
            graph=graph,
            workspace_dir=self.tmp_dir,
            mode=ExecutionMode.AUTONOMOUS,
        )
        report = runner.execute()

        self.assertEqual(report.completed_count, 1)
        # Should only call chat once for Cycle 1
        self.assertEqual(mock_chat.call_count, 1)
        self.assertIn("Cycle 1 Acceptance Harness: PASSED", report.results["t_code"].logs)

    @patch("research_assistant.shared.llm.chat")
    def test_autonomous_mode_two_cycle_reflection_recovery(self, mock_chat):
        # Cycle 1 returns invalid syntax; Cycle 2 returns valid syntax
        mock_c1 = MagicMock(content="broken code :::")
        mock_c2 = MagicMock(content="def fixed_code(): return 1\n")
        mock_chat.side_effect = [mock_c1, mock_c2]

        graph = TaskGraph(paper_id="reflect_paper", paper_title="Reflection Test")
        t = Task(
            task_id="t_syntax",
            title="Refine Syntax",
            category=TaskCategory.MODEL_IMPLEMENTATION,
            outputs=["algo.py"],
            artifact_contracts=[
                ArtifactContract(
                    path="algo.py",
                    artifact_type=ArtifactType.CODE,
                )
            ],
        )
        graph.add_task(t)

        runner = TaskRunner(
            graph=graph,
            workspace_dir=self.tmp_dir,
            mode=ExecutionMode.AUTONOMOUS,
        )
        report = runner.execute()

        self.assertEqual(report.completed_count, 1)
        # Should call chat twice: once for Cycle 1, once for Cycle 2 Refinement
        self.assertEqual(mock_chat.call_count, 2)
        logs = report.results["t_syntax"].logs
        self.assertIn("Triggering Cycle 2 Reflexion", logs)
        self.assertIn("Cycle 2 Acceptance Harness: PASSED on retest", logs)

    @patch("research_assistant.shared.llm.chat")
    def test_autonomous_mode_fails_when_cycle_2_also_fails(self, mock_chat):
        # Both Cycle 1 and Cycle 2 return invalid syntax
        mock_chat.return_value = MagicMock(content="broken syntax :::")

        graph = TaskGraph(paper_id="fail_paper", paper_title="Failure Test")
        t = Task(
            task_id="t_fail",
            title="Always Fails",
            category=TaskCategory.MODEL_IMPLEMENTATION,
            outputs=["fail.py"],
            artifact_contracts=[
                ArtifactContract(
                    path="fail.py",
                    artifact_type=ArtifactType.CODE,
                )
            ],
        )
        graph.add_task(t)

        runner = TaskRunner(
            graph=graph,
            workspace_dir=self.tmp_dir,
            mode=ExecutionMode.AUTONOMOUS,
        )
        report = runner.execute()

        self.assertEqual(report.completed_count, 0)
        self.assertEqual(report.failed_count, 1)
        self.assertIn("Cycle 2 Acceptance Harness: FAILED after refinement", report.results["t_fail"].logs)

    def test_execute_with_reflection_direct_cycle1_success(self):
        """Direct test of execute_with_reflection completing in Cycle 1."""
        graph = TaskGraph(paper_id="p1", paper_title="Direct Reflection")
        t = Task(task_id="t_direct", title="Direct Task", outputs=["out.txt"])
        graph.add_task(t)
        runner = TaskRunner(graph=graph, workspace_dir=self.tmp_dir, mode=ExecutionMode.AUTONOMOUS)

        with patch.object(runner, "_execute_harness_subprocess", return_value=(0, "OK")):
            status = runner.execute_with_reflection(t, max_reflection_cycles=2)
            self.assertEqual(status, TaskStatus.COMPLETED)
            self.assertEqual(t.metadata.get("reflection_cycles"), 1)

    def test_execute_with_reflection_direct_cycle2_recovery(self):
        """Direct test of execute_with_reflection recovering in Cycle 2."""
        graph = TaskGraph(paper_id="p1", paper_title="Direct Recovery")
        t = Task(task_id="t_recov", title="Recoverable Task", outputs=["fix.txt"])
        graph.add_task(t)
        runner = TaskRunner(graph=graph, workspace_dir=self.tmp_dir, mode=ExecutionMode.AUTONOMOUS)

        side_effects = [(1, "AssertionError: initial failure"), (0, "OK")]
        with patch.object(runner, "_execute_harness_subprocess", side_effect=side_effects):
            status = runner.execute_with_reflection(t, max_reflection_cycles=2)
            self.assertEqual(status, TaskStatus.COMPLETED)
            self.assertEqual(t.metadata.get("reflection_cycles"), 2)

    def test_execute_with_reflection_direct_max_cycles_exhausted(self):
        """Direct test of execute_with_reflection failing when max cycles exhausted."""
        graph = TaskGraph(paper_id="p1", paper_title="Direct Failure")
        t = Task(task_id="t_exhaust", title="Exhausted Task", outputs=["never.txt"])
        graph.add_task(t)
        runner = TaskRunner(graph=graph, workspace_dir=self.tmp_dir, mode=ExecutionMode.AUTONOMOUS)

        with patch.object(runner, "_execute_harness_subprocess", return_value=(1, "Persistent Failure")):
            status = runner.execute_with_reflection(t, max_reflection_cycles=2)
            self.assertEqual(status, TaskStatus.FAILED)
            self.assertEqual(t.metadata.get("reflection_cycles"), 2)
            self.assertEqual(t.metadata["failure_details"]["exit_code"], 1)

    def test_parse_execution_error_diagnostics(self):
        """Verifies parse_execution_error extracts traceback, classification, and assertion line."""
        raw_traceback = (
            "Traceback (most recent call last):\n"
            "  File 'test_task.py', line 22, in test_output\n"
            "    self.assertTrue(False)\n"
            "AssertionError: Deliverable missing: data/train.pt\n"
        )
        task = Task(task_id="t1", title="T1", outputs=["data/train.pt"])
        diag = parse_execution_error(1, raw_traceback, task=task, workspace_dir=self.tmp_dir)

        self.assertEqual(diag.exit_code, 1)
        self.assertEqual(diag.error_type, "AssertionError")
        self.assertEqual(diag.failing_assertion, "AssertionError: Deliverable missing: data/train.pt")
        self.assertIn("Traceback (most recent call last):", diag.stack_trace)
        self.assertIn("data/train.pt", diag.missing_files)
        self.assertIn("AssertionError", diag.summary())

    def test_storage_protection_invariant(self):
        """Verifies TaskRunner refuses to use persistent vector DB or DATA_DIR as workspace."""
        graph = TaskGraph(paper_id="p1", paper_title="P1")
        with self.assertRaises(ValueError) as ctx:
            TaskRunner(graph=graph, workspace_dir=VECTORDB_PATH)
        self.assertIn("Safety violation", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx_data:
            TaskRunner(graph=graph, workspace_dir=DATA_DIR)
        self.assertIn("Safety violation", str(ctx_data.exception))

    @patch("research_assistant.shared.llm.chat")
    def test_warm_state_preservation(self, mock_chat):
        """Verifies Cycle 2 operates on warm workspace without destroying existing deliverables."""
        mock_c1 = MagicMock(content="broken code :::")
        mock_c2 = MagicMock(content="def fixed_model(): return 42\n")
        mock_chat.side_effect = [mock_c1, mock_c2]

        graph = TaskGraph(paper_id="p1", paper_title="Warm State Test")
        t = Task(
            task_id="t_warm",
            title="Warm Task",
            inputs=["data/train.pt"],
            outputs=["src/model.py"],
            artifact_contracts=[ArtifactContract(path="src/model.py", artifact_type=ArtifactType.CODE)],
        )
        graph.add_task(t)
        runner = TaskRunner(graph=graph, workspace_dir=self.tmp_dir, mode=ExecutionMode.AUTONOMOUS)

        # Pre-populate upstream artifact data/train.pt in warm workspace
        os.makedirs(os.path.join(self.tmp_dir, "data"), exist_ok=True)
        p1 = os.path.join(self.tmp_dir, "data/train.pt")
        with open(p1, "w") as f:
            f.write("Initial valid training tensor data")

        status = runner.execute_with_reflection(t, max_reflection_cycles=2)
        self.assertEqual(status, TaskStatus.COMPLETED)
        self.assertEqual(t.metadata.get("reflection_cycles"), 2)

        # Verify warm state file was not destroyed or overwritten
        self.assertTrue(os.path.exists(p1))
        with open(p1, "r") as f:
            self.assertEqual(f.read(), "Initial valid training tensor data")

        # Verify refined deliverable exists and has refined code
        p_out = os.path.join(self.tmp_dir, "src/model.py")
        self.assertTrue(os.path.exists(p_out))
        with open(p_out, "r") as f:
            self.assertIn("def fixed_model", f.read())

    @patch("research_assistant.shared.llm.chat")
    def test_authentic_code_subprocess_execution_and_reflection(self, mock_chat):
        """Verifies entrypoint code execution runs in authentic subprocess and recovers from runtime errors."""
        c1_code = "print('starting')\n1 / 0\n"
        c2_code = (
            "import json\n"
            "with open('result.json', 'w') as f:\n"
            "    json.dump({'metric': 0.85}, f)\n"
            "print('done')\n"
        )
        mock_chat.side_effect = [MagicMock(content=c1_code), MagicMock(content=c2_code)]

        graph = TaskGraph(paper_id="p1", paper_title="Subprocess Test")
        t = Task(
            task_id="t_exec",
            title="Execute Code",
            outputs=["run.py", "result.json"],
            artifact_contracts=[
                ArtifactContract(path="run.py", artifact_type=ArtifactType.CODE),
                ArtifactContract(path="result.json", artifact_type=ArtifactType.METRICS_JSON, required_keys=["metric"]),
            ],
        )
        graph.add_task(t)
        runner = TaskRunner(graph=graph, workspace_dir=self.tmp_dir, mode=ExecutionMode.AUTONOMOUS)

        status = runner.execute_with_reflection(t, max_reflection_cycles=2)
        self.assertEqual(status, TaskStatus.COMPLETED)
        self.assertEqual(t.metadata.get("reflection_cycles"), 2)

        # Verify result.json was authentically generated by the subprocess execution
        res_json = os.path.join(self.tmp_dir, "result.json")
        self.assertTrue(os.path.exists(res_json))
        import json
        with open(res_json) as f:
            data = json.load(f)
        self.assertEqual(data.get("metric"), 0.85)


if __name__ == "__main__":
    unittest.main()
