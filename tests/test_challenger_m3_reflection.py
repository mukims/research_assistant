"""
Challenger Verification Suite for Milestone 3 (M3) Two-Cycle Reflexion Runner.
Agent: teamwork_preview_challenger_m3_2

Adversarially challenges and empirically verifies:
1. Cycle 1 Clean Pass:
   - When test harness exits 0 on Cycle 1, status is COMPLETED and reflection_cycles == 1.
2. Cycle 2 Recovery:
   - When Cycle 1 fails (exit 1), stderr/traceback and failed assertion line are captured in diagnostics,
     reflection prompt is formed with error details, and on Cycle 2 pass, status is COMPLETED and reflection_cycles == 2.
3. Strict 2-Cycle Cap:
   - When harness persistently fails, loop terminates at exactly 2 cycles and marks FAILED without infinite loop.
4. Warm-State Workspace Preservation:
   - In Cycle 2, existing intact files from Cycle 1 are preserved in the workspace directory.
5. Workspace Safety Invariant:
   - Creating a TaskRunner with workspace_dir=VECTORDB_PATH or workspace_dir=DATA_DIR raises ValueError ("Safety violation").
6. Subprocess Timeout Guardrail:
   - A hanging harness script triggers timeout handling and returns failure code cleanly.
7. ChromaDB Persistent Storage Invariant:
   - Assert data/physics_vectordb collection physics_papers has exactly 75,039 chunks.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import MagicMock, call, patch

import chromadb

from research_assistant.config import (
    COLLECTION_NAME,
    DATA_DIR,
    VECTORDB_PATH,
)
from research_assistant.tasks.harness import (
    generate_task_test_script,
    run_task_test_script,
)
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


class TestChallengerCycle1CleanPass(unittest.TestCase):
    """
    Stress-test Requirement 1: Cycle 1 Clean Pass.
    When test harness exits 0 on Cycle 1, status is COMPLETED and reflection_cycles == 1.
    Cycle 2 must NEVER be invoked.
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="challenger_m3_c1_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_cycle1_clean_pass_direct_execution(self):
        """Direct call to execute_with_reflection completes on Cycle 1 when harness returns 0."""
        graph = TaskGraph(paper_id="p_c1", paper_title="Cycle 1 Clean Pass Direct")
        task = Task(
            task_id="t_c1_direct",
            title="Clean Pass Task",
            category=TaskCategory.MODEL_IMPLEMENTATION,
            outputs=["model.py"],
            artifact_contracts=[ArtifactContract(path="model.py", artifact_type=ArtifactType.CODE)],
        )
        graph.add_task(task)

        runner = TaskRunner(graph=graph, workspace_dir=self.tmp_dir, mode=ExecutionMode.AUTONOMOUS)

        with patch("research_assistant.shared.llm.chat") as mock_chat:
            mock_chat.return_value = MagicMock(content="def forward_pass(): return 42\n")
            with patch.object(runner, "_execute_harness_subprocess", return_value=(0, "OK")):
                status = runner.execute_with_reflection(task, max_reflection_cycles=2)

                self.assertEqual(status, TaskStatus.COMPLETED)
                self.assertEqual(task.status, TaskStatus.COMPLETED)
                self.assertEqual(task.metadata.get("reflection_cycles"), 1)
                self.assertEqual(mock_chat.call_count, 1)
                self.assertIn("Cycle 1 Acceptance Harness: PASSED cleanly", task.execution_log)
                self.assertNotIn("Triggering Cycle 2 Reflexion", task.execution_log)
                self.assertNotIn("Cycle 2", task.execution_log)

    def test_cycle1_clean_pass_with_real_subprocess_harness(self):
        """End-to-end execution where real subprocess harness executes and exits 0 on Cycle 1."""
        graph = TaskGraph(paper_id="p_c1_real", paper_title="Cycle 1 Real Subprocess")
        task = Task(
            task_id="t_c1_real",
            title="Real Subprocess Task",
            category=TaskCategory.DATA_PREPROCESSING,
            outputs=["data/processed.json"],
            artifact_contracts=[
                ArtifactContract(
                    path="data/processed.json",
                    artifact_type=ArtifactType.METRICS_JSON,
                    required_keys=["sample_count", "split_ratio"],
                )
            ],
            acceptance_criteria=["Clean exit code 0", "Non-empty output"],
        )
        graph.add_task(task)

        runner = TaskRunner(graph=graph, workspace_dir=self.tmp_dir, mode=ExecutionMode.AUTONOMOUS)

        valid_json = '{"sample_count": 1000, "split_ratio": 0.8}'
        with patch("research_assistant.shared.llm.chat") as mock_chat:
            mock_chat.return_value = MagicMock(content=valid_json)
            report = runner.execute()

            self.assertEqual(report.completed_count, 1)
            self.assertEqual(report.failed_count, 0)
            res = report.results["t_c1_real"]
            self.assertEqual(res.status, TaskStatus.COMPLETED)
            self.assertEqual(task.metadata.get("reflection_cycles"), 1)
            self.assertEqual(mock_chat.call_count, 1)
            self.assertIn("Cycle 1 Acceptance Harness: PASSED cleanly", res.logs)

            # Empirically verify deliverable on disk
            deliverable_path = os.path.join(self.tmp_dir, "data/processed.json")
            self.assertTrue(os.path.exists(deliverable_path))
            self.assertGreater(os.path.getsize(deliverable_path), 0)

    def test_cycle1_clean_pass_when_chat_fails_gracefully_with_stub(self):
        """When chat raises network error, deliverable stub is materialized and passes if valid."""
        graph = TaskGraph(paper_id="p_c1_stub", paper_title="Cycle 1 Stub Pass")
        task = Task(
            task_id="t_c1_stub",
            title="Stub Task",
            category=TaskCategory.ENVIRONMENT_SETUP,
            outputs=["env.txt"],
            acceptance_criteria=["Clean exit code 0"],
        )
        graph.add_task(task)

        runner = TaskRunner(graph=graph, workspace_dir=self.tmp_dir, mode=ExecutionMode.AUTONOMOUS)

        with patch("research_assistant.shared.llm.chat", side_effect=ConnectionError("Ollama offline")):
            with patch.object(runner, "_execute_harness_subprocess", return_value=(0, "OK")):
                status = runner.execute_with_reflection(task, max_reflection_cycles=2)
                self.assertEqual(status, TaskStatus.COMPLETED)
                self.assertEqual(task.metadata.get("reflection_cycles"), 1)
                self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "env.txt")))


class TestChallengerCycle2Recovery(unittest.TestCase):
    """
    Stress-test Requirement 2: Cycle 2 Recovery.
    When Cycle 1 fails (exit 1), stderr/traceback and failed assertion line are captured in diagnostics,
    reflection prompt is formed with error details, and on Cycle 2 pass, status is COMPLETED and reflection_cycles == 2.
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="challenger_m3_c2_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_cycle2_recovery_diagnostics_and_prompt_formation(self):
        """
        Verify Cycle 1 failure captures failing assertion and traceback in diagnostics,
        embeds them into Cycle 2 reflection prompt, and completes in Cycle 2.
        """
        graph = TaskGraph(paper_id="p_c2_prompt", paper_title="Cycle 2 Prompt Verification")
        task = Task(
            task_id="t_c2_prompt",
            title="Recoverable Script",
            category=TaskCategory.MODEL_IMPLEMENTATION,
            description="Implement PyTorch convolutional architecture with finite loss.",
            outputs=["model.py"],
            artifact_contracts=[ArtifactContract(path="model.py", artifact_type=ArtifactType.CODE)],
        )
        graph.add_task(task)

        runner = TaskRunner(graph=graph, workspace_dir=self.tmp_dir, mode=ExecutionMode.AUTONOMOUS)

        failing_traceback = (
            "Traceback (most recent call last):\n"
            "  File 'test_harness.py', line 45, in test_artifact_contracts_schema\n"
            "    self.fail('Python syntax error in model.py: invalid syntax line 3')\n"
            "AssertionError: Python syntax error in model.py: invalid syntax line 3\n"
        )

        recorded_chat_messages = []

        def mock_chat_fn(messages, **kwargs):
            recorded_chat_messages.append(messages)
            if len(recorded_chat_messages) == 1:
                return MagicMock(content="broken syntax ::::")
            else:
                return MagicMock(content="def clean_model(): return True\n")

        # Cycle 1 returns exit 1 with failing traceback; Cycle 2 returns exit 0
        harness_returns = [(1, failing_traceback), (0, "OK")]

        with patch("research_assistant.shared.llm.chat", side_effect=mock_chat_fn) as mock_chat:
            with patch.object(runner, "_execute_harness_subprocess", side_effect=harness_returns):
                status = runner.execute_with_reflection(task, max_reflection_cycles=2)

                self.assertEqual(status, TaskStatus.COMPLETED)
                self.assertEqual(task.status, TaskStatus.COMPLETED)
                self.assertEqual(task.metadata.get("reflection_cycles"), 2)
                self.assertEqual(mock_chat.call_count, 2)

                # Inspect Cycle 2 messages to verify reflection prompt structure
                c2_call = recorded_chat_messages[1]
                self.assertEqual(len(c2_call), 4)  # system, user (c1), assistant (c1), user (refine)
                refine_message = c2_call[3]["content"]

                # Empirically assert error details are captured in reflection prompt
                self.assertIn("# Two-Cycle Reflexion: Refinement Cycle", refine_message)
                self.assertIn("AssertionError", refine_message)
                self.assertIn("Python syntax error in model.py", refine_message)
                self.assertIn("Traceback (most recent call last):", refine_message)
                self.assertIn(self.tmp_dir, refine_message)
                self.assertIn(task.description, refine_message)

                # Execution logs must confirm both cycles
                self.assertIn("Cycle 1 Test Failed", task.execution_log)
                self.assertIn("Triggering Cycle 2 Reflexion", task.execution_log)
                self.assertIn("Cycle 2 Acceptance Harness: PASSED on retest", task.execution_log)

    def test_cycle2_recovery_with_real_subprocess_harness(self):
        """End-to-end Cycle 2 recovery executing real subprocess test harnesses."""
        graph = TaskGraph(paper_id="p_c2_real", paper_title="Real Subprocess Cycle 2 Recovery")
        task = Task(
            task_id="t_c2_real",
            title="Recoverable Code File",
            category=TaskCategory.MODEL_IMPLEMENTATION,
            outputs=["src/pipeline.py"],
            artifact_contracts=[ArtifactContract(path="src/pipeline.py", artifact_type=ArtifactType.CODE)],
        )
        graph.add_task(task)

        runner = TaskRunner(graph=graph, workspace_dir=self.tmp_dir, mode=ExecutionMode.AUTONOMOUS)

        # Cycle 1 produces broken Python code (syntax error) -> harness subprocess will exit 1
        # Cycle 2 produces valid Python code -> harness subprocess will exit 0
        cycle1_output = "def broken_fn(\n   syntax error here !!!"
        cycle2_output = "def valid_fn():\n    return 'success'\n"

        with patch("research_assistant.shared.llm.chat") as mock_chat:
            mock_chat.side_effect = [MagicMock(content=cycle1_output), MagicMock(content=cycle2_output)]
            report = runner.execute()

            self.assertEqual(report.completed_count, 1)
            self.assertEqual(report.failed_count, 0)
            res = report.results["t_c2_real"]
            self.assertEqual(res.status, TaskStatus.COMPLETED)
            self.assertEqual(task.metadata.get("reflection_cycles"), 2)
            self.assertEqual(mock_chat.call_count, 2)
            self.assertIn("Triggering Cycle 2 Reflexion", res.logs)
            self.assertIn("Cycle 2 Acceptance Harness: PASSED on retest", res.logs)

            # Verify refined file on disk contains valid function
            out_file = os.path.join(self.tmp_dir, "src/pipeline.py")
            self.assertTrue(os.path.exists(out_file))
            with open(out_file, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("def valid_fn", content)
                self.assertNotIn("syntax error here", content)

    def test_cycle2_recovery_captures_missing_deliverables(self):
        """When deliverables are completely missing, diagnostics record missing_files and passes in Cycle 2."""
        task = Task(
            task_id="t_missing",
            title="Missing File Recovery",
            outputs=["out/file_a.txt", "out/file_b.txt"],
        )
        # Manually create only file_a, leaving file_b missing
        os.makedirs(os.path.join(self.tmp_dir, "out"), exist_ok=True)
        with open(os.path.join(self.tmp_dir, "out/file_a.txt"), "w") as f:
            f.write("content")

        raw_err = "AssertionError: Deliverable missing: out/file_b.txt\n"
        diag = parse_execution_error(1, raw_err, task=task, workspace_dir=self.tmp_dir)

        self.assertEqual(diag.error_type, "AssertionError")
        self.assertIn("out/file_b.txt", diag.missing_files)
        self.assertNotIn("out/file_a.txt", diag.missing_files)


class TestChallengerStrictTwoCycleCap(unittest.TestCase):
    """
    Stress-test Requirement 3: Strict 2-Cycle Cap.
    When harness persistently fails, loop terminates at exactly 2 cycles and marks FAILED without infinite loop.
    Bounds checks on max_reflection_cycles.
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="challenger_m3_cap_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_persistent_failure_terminates_at_exactly_two_cycles(self):
        """When harness persistently fails, loop terminates at cycle 2 and marks TaskStatus.FAILED."""
        graph = TaskGraph(paper_id="p_cap", paper_title="Two Cycle Cap Test")
        task = Task(
            task_id="t_cap",
            title="Always Failing Task",
            category=TaskCategory.TRAINING_OR_SIMULATION,
            outputs=["logs/train.log"],
        )
        graph.add_task(task)

        runner = TaskRunner(graph=graph, workspace_dir=self.tmp_dir, mode=ExecutionMode.AUTONOMOUS)

        persistent_error = "AssertionError: Loss is NaN in epoch 1\n"
        with patch("research_assistant.shared.llm.chat") as mock_chat:
            mock_chat.return_value = MagicMock(content="failing attempt")
            with patch.object(runner, "_execute_harness_subprocess", return_value=(1, persistent_error)):
                status = runner.execute_with_reflection(task, max_reflection_cycles=2)

                self.assertEqual(status, TaskStatus.FAILED)
                self.assertEqual(task.status, TaskStatus.FAILED)
                self.assertEqual(task.metadata.get("reflection_cycles"), 2)
                self.assertEqual(mock_chat.call_count, 2)

                # Verify failure_details metadata
                details = task.metadata.get("failure_details", {})
                self.assertEqual(details.get("exit_code"), 1)
                self.assertEqual(details.get("error_type"), "AssertionError")
                self.assertIn("Loss is NaN", details.get("assertion", ""))
                self.assertIn("Cycle 2 Acceptance Harness: FAILED", task.execution_log)

    def test_arbitrary_large_max_cycles_strictly_capped_at_two(self):
        """Passing max_reflection_cycles=100 or 10 is strictly bounded to at most 2 cycles."""
        graph = TaskGraph(paper_id="p_large_cap", paper_title="Large Cap Test")
        task = Task(task_id="t_large_cap", title="Large Cap Task", outputs=["res.txt"])
        graph.add_task(task)

        runner = TaskRunner(graph=graph, workspace_dir=self.tmp_dir, mode=ExecutionMode.AUTONOMOUS)

        with patch("research_assistant.shared.llm.chat") as mock_chat:
            mock_chat.return_value = MagicMock(content="broken code")
            with patch.object(runner, "_execute_harness_subprocess", return_value=(1, "AssertionError: Persistent")):
                # Pass 100 cycles - must strictly terminate after 2
                status = runner.execute_with_reflection(task, max_reflection_cycles=100)

                self.assertEqual(status, TaskStatus.FAILED)
                self.assertEqual(task.metadata.get("reflection_cycles"), 2)
                self.assertEqual(mock_chat.call_count, 2)

    def test_single_cycle_bound(self):
        """Passing max_reflection_cycles=1 terminates after exactly 1 cycle."""
        graph = TaskGraph(paper_id="p_one_cap", paper_title="One Cap Test")
        task = Task(task_id="t_one_cap", title="One Cap Task", outputs=["res.txt"])
        graph.add_task(task)

        runner = TaskRunner(graph=graph, workspace_dir=self.tmp_dir, mode=ExecutionMode.AUTONOMOUS)

        with patch("research_assistant.shared.llm.chat") as mock_chat:
            mock_chat.return_value = MagicMock(content="broken code")
            with patch.object(runner, "_execute_harness_subprocess", return_value=(1, "AssertionError: Failed")):
                status = runner.execute_with_reflection(task, max_reflection_cycles=1)

                self.assertEqual(status, TaskStatus.FAILED)
                self.assertEqual(mock_chat.call_count, 1)

    def test_zero_or_negative_max_cycles_executes_safely(self):
        """Passing max_reflection_cycles=0 or -5 executes safely without infinite loop."""
        graph = TaskGraph(paper_id="p_zero_cap", paper_title="Zero Cap Test")
        task = Task(task_id="t_zero_cap", title="Zero Cap Task", outputs=["res.txt"])
        graph.add_task(task)

        runner = TaskRunner(graph=graph, workspace_dir=self.tmp_dir, mode=ExecutionMode.AUTONOMOUS)

        with patch("research_assistant.shared.llm.chat") as mock_chat:
            mock_chat.return_value = MagicMock(content="valid code")
            with patch.object(runner, "_execute_harness_subprocess", return_value=(0, "OK")):
                status = runner.execute_with_reflection(task, max_reflection_cycles=0)
                self.assertEqual(status, TaskStatus.COMPLETED)
                self.assertEqual(task.metadata.get("reflection_cycles"), 1)


class TestChallengerWarmStateWorkspacePreservation(unittest.TestCase):
    """
    Stress-test Requirement 4: Warm-State Workspace Preservation.
    In Cycle 2, existing intact files from Cycle 1 are preserved in the workspace directory.
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="challenger_m3_warm_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_intact_auxiliary_and_upstream_files_preserved_in_cycle2(self):
        """
        Files present in workspace prior to or during Cycle 1 (e.g. data splits, checkpoints)
        must remain byte-identical and un-corrupted during Cycle 2 refinement.
        """
        # Create an upstream dataset and checkpoint file in workspace
        os.makedirs(os.path.join(self.tmp_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(self.tmp_dir, "checkpoints"), exist_ok=True)

        upstream_dataset = os.path.join(self.tmp_dir, "data/splits.csv")
        with open(upstream_dataset, "w", encoding="utf-8") as f:
            f.write("id,feature,label\n1,0.55,1\n2,0.92,0\n")

        upstream_weights = os.path.join(self.tmp_dir, "checkpoints/init_weights.bin")
        dummy_weights = b"\x00\x01\x02\x03\x04\x05\xaa\xbb\xcc\xdd" * 50
        with open(upstream_weights, "wb") as f:
            f.write(dummy_weights)

        # Build task that refines code.py
        graph = TaskGraph(paper_id="p_warm", paper_title="Warm State Preservation")
        task = Task(
            task_id="t_warm",
            title="Warm Refinement Task",
            category=TaskCategory.MODEL_IMPLEMENTATION,
            inputs=["data/splits.csv", "checkpoints/init_weights.bin"],
            outputs=["src/model.py"],
            artifact_contracts=[ArtifactContract(path="src/model.py", artifact_type=ArtifactType.CODE)],
        )
        graph.add_task(task)

        runner = TaskRunner(graph=graph, workspace_dir=self.tmp_dir, mode=ExecutionMode.AUTONOMOUS)

        with patch("research_assistant.shared.llm.chat") as mock_chat:
            mock_chat.side_effect = [
                MagicMock(content="broken code :::"),
                MagicMock(content="def refined_model(): return 100\n"),
            ]
            status = runner.execute_with_reflection(task, max_reflection_cycles=2)

            self.assertEqual(status, TaskStatus.COMPLETED)
            self.assertEqual(task.metadata.get("reflection_cycles"), 2)

            # 1. Assert upstream dataset file remained 100% byte-identical
            self.assertTrue(os.path.exists(upstream_dataset))
            with open(upstream_dataset, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "id,feature,label\n1,0.55,1\n2,0.92,0\n")

            # 2. Assert binary weights file remained 100% byte-identical
            self.assertTrue(os.path.exists(upstream_weights))
            with open(upstream_weights, "rb") as f:
                self.assertEqual(f.read(), dummy_weights)

            # 3. Assert target deliverable was successfully created and refined
            model_file = os.path.join(self.tmp_dir, "src/model.py")
            self.assertTrue(os.path.exists(model_file))
            with open(model_file, "r", encoding="utf-8") as f:
                self.assertIn("def refined_model", f.read())

    def test_partially_materialized_files_preserve_valid_deliverables(self):
        """When task specifies multiple outputs, Cycle 2 updates targets on warm state."""
        graph = TaskGraph(paper_id="p_multi", paper_title="Multi Deliverable Preservation")
        task = Task(
            task_id="t_multi",
            title="Multi Deliverable",
            outputs=["out/valid.txt", "out/target.py"],
            artifact_contracts=[
                ArtifactContract(path="out/target.py", artifact_type=ArtifactType.CODE),
            ],
        )
        graph.add_task(task)

        runner = TaskRunner(graph=graph, workspace_dir=self.tmp_dir, mode=ExecutionMode.AUTONOMOUS)

        # Pre-seed out/valid.txt in workspace
        os.makedirs(os.path.join(self.tmp_dir, "out"), exist_ok=True)
        valid_path = os.path.join(self.tmp_dir, "out/valid.txt")
        with open(valid_path, "w", encoding="utf-8") as f:
            f.write("Valid intact data from earlier stage")

        with patch("research_assistant.shared.llm.chat") as mock_chat:
            mock_chat.side_effect = [
                MagicMock(content="broken :::"),
                MagicMock(content="def resolved(): pass\n"),
            ]
            status = runner.execute_with_reflection(task, max_reflection_cycles=2)
            self.assertEqual(status, TaskStatus.COMPLETED)

            # Both files must exist on warm state
            self.assertTrue(os.path.exists(valid_path))
            self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "out/target.py")))


class TestChallengerWorkspaceSafetyInvariant(unittest.TestCase):
    """
    Stress-test Requirement 5: Workspace Safety Invariant.
    Creating a TaskRunner with workspace_dir=VECTORDB_PATH or workspace_dir=DATA_DIR raises ValueError ("Safety violation").
    """

    def setUp(self):
        self.graph = TaskGraph(paper_id="p_safe", paper_title="Safety Guardrail Test")

    def test_direct_vectordb_path_raises_safety_violation(self):
        """Initializing TaskRunner with workspace_dir=VECTORDB_PATH raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            TaskRunner(graph=self.graph, workspace_dir=VECTORDB_PATH)
        self.assertIn("Safety violation", str(ctx.exception))
        self.assertIn("cannot point to persistent data storage", str(ctx.exception))

    def test_direct_data_dir_raises_safety_violation(self):
        """Initializing TaskRunner with workspace_dir=DATA_DIR raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            TaskRunner(graph=self.graph, workspace_dir=DATA_DIR)
        self.assertIn("Safety violation", str(ctx.exception))
        self.assertIn("cannot point to persistent data storage", str(ctx.exception))

    def test_relative_path_resolving_to_vectordb_path_raises_safety_violation(self):
        """Relative path resolving to VECTORDB_PATH (e.g. data/physics_vectordb) raises ValueError."""
        rel_vdb = os.path.relpath(VECTORDB_PATH)
        with self.assertRaises(ValueError) as ctx:
            TaskRunner(graph=self.graph, workspace_dir=rel_vdb)
        self.assertIn("Safety violation", str(ctx.exception))

    def test_relative_path_resolving_to_data_dir_raises_safety_violation(self):
        """Relative path resolving to DATA_DIR (e.g. data) raises ValueError."""
        rel_data = os.path.relpath(DATA_DIR)
        with self.assertRaises(ValueError) as ctx:
            TaskRunner(graph=self.graph, workspace_dir=rel_data)
        self.assertIn("Safety violation", str(ctx.exception))

    def test_trailing_slashes_and_dot_slashes_resolving_to_persistent_storage(self):
        """Paths with trailing slashes or './' still trigger the safety violation."""
        trailing_vdb = VECTORDB_PATH.rstrip("/") + "/"
        with self.assertRaises(ValueError) as ctx:
            TaskRunner(graph=self.graph, workspace_dir=trailing_vdb)
        self.assertIn("Safety violation", str(ctx.exception))

        trailing_data = DATA_DIR.rstrip("/") + "/."
        with self.assertRaises(ValueError) as ctx:
            TaskRunner(graph=self.graph, workspace_dir=trailing_data)
        self.assertIn("Safety violation", str(ctx.exception))

    def test_safe_workspace_directory_succeeds(self):
        """A normal workspace directory outside DATA_DIR initializes without error."""
        safe_tmp = tempfile.mkdtemp(prefix="safe_ws_")
        try:
            runner = TaskRunner(graph=self.graph, workspace_dir=safe_tmp)
            self.assertEqual(runner.workspace_dir, os.path.abspath(safe_tmp))
        finally:
            shutil.rmtree(safe_tmp, ignore_errors=True)


class TestChallengerSubprocessTimeoutGuardrail(unittest.TestCase):
    """
    Stress-test Requirement 6: Subprocess Timeout Guardrail.
    A hanging harness script triggers timeout handling and returns failure code cleanly.
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="challenger_m3_timeout_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_run_task_test_script_handles_hanging_script_with_timeout(self):
        """
        Executes a test harness containing an infinite sleep.
        run_task_test_script must abort at timeout_seconds and return (False, '', '...timed out...').
        """
        hanging_script = (
            "import time\n"
            "import unittest\n"
            "class TestHanging(unittest.TestCase):\n"
            "    def test_hang(self):\n"
            "        time.sleep(30)\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )

        t_start = time.time()
        success, stdout, stderr = run_task_test_script(
            hanging_script,
            workspace_dir=self.tmp_dir,
            timeout_seconds=2,
        )
        elapsed = time.time() - t_start

        self.assertFalse(success)
        self.assertIn("timed out", stderr.lower())
        self.assertLess(elapsed, 8.0, "Subprocess timeout failed to terminate hanging process promptly!")

    def test_execute_harness_subprocess_returns_failure_code_on_timeout(self):
        """TaskRunner._execute_harness_subprocess returns (1, '...timed out...') when harness hangs."""
        graph = TaskGraph(paper_id="p_hang", paper_title="Hanging Harness")
        runner = TaskRunner(graph=graph, workspace_dir=self.tmp_dir)

        hanging_script = (
            "import time\n"
            "import unittest\n"
            "class TestHang(unittest.TestCase):\n"
            "    def test_hang(self):\n"
            "        time.sleep(15)\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )

        returncode, output = runner._execute_harness_subprocess(
            script_content=hanging_script,
            timeout_seconds=2,
        )

        self.assertEqual(returncode, 1)
        self.assertIn("timed out", output.lower())

    def test_parse_execution_error_classifies_timeout_expired(self):
        """parse_execution_error classifies timeout output as TimeoutExpired."""
        raw_timeout = "Test harness timed out after 30s"
        diag = parse_execution_error(1, raw_timeout, workspace_dir=self.tmp_dir)

        self.assertEqual(diag.exit_code, 1)
        self.assertEqual(diag.error_type, "TimeoutExpired")
        self.assertIn("Test harness timed out", diag.failing_assertion)
        self.assertIn("TimeoutExpired", diag.summary())

    def test_execute_with_reflection_recovers_after_cycle1_timeout(self):
        """
        Cycle 1 harness times out; diagnostics identify TimeoutExpired;
        Cycle 2 LLM refines to fast code and passes cleanly.
        """
        graph = TaskGraph(paper_id="p_recov_timeout", paper_title="Timeout Recovery")
        task = Task(task_id="t_time_recov", title="Timeout Task", outputs=["res.txt"])
        graph.add_task(task)

        runner = TaskRunner(graph=graph, workspace_dir=self.tmp_dir, mode=ExecutionMode.AUTONOMOUS)

        # Mock Cycle 1 timeout followed by Cycle 2 clean pass
        side_effects = [(1, "Test harness timed out after 2s"), (0, "OK")]
        with patch("research_assistant.shared.llm.chat") as mock_chat:
            mock_chat.return_value = MagicMock(content="refined non-hanging code")
            with patch.object(runner, "_execute_harness_subprocess", side_effect=side_effects):
                status = runner.execute_with_reflection(task, max_reflection_cycles=2)

                self.assertEqual(status, TaskStatus.COMPLETED)
                self.assertEqual(task.metadata.get("reflection_cycles"), 2)
                self.assertIn("Cycle 1 Test Failed", task.execution_log)
                self.assertIn("Cycle 2 Acceptance Harness: PASSED on retest", task.execution_log)


class TestChallengerChromaDBPersistentStorageInvariant(unittest.TestCase):
    """
    Stress-test Requirement 7: ChromaDB Persistent Storage Invariant.
    Assert data/physics_vectordb collection physics_papers has exactly 75,039 chunks.
    Verify read-only non-destructive access.
    """

    def test_chromadb_physics_papers_chunk_count_equals_75039(self):
        """Empirically connect to persistent vector DB and assert chunk count is exactly 75,039."""
        client = chromadb.PersistentClient(path=VECTORDB_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()

        self.assertEqual(
            count,
            75039,
            f"ChromaDB persistent collection '{COLLECTION_NAME}' chunk count corrupted! Expected 75039, got {count}",
        )

    def test_chromadb_collection_read_only_schema_integrity(self):
        """Peek at collection records and assert 768-dim embeddings without mutating persistent storage."""
        client = chromadb.PersistentClient(path=VECTORDB_PATH)
        collection = client.get_collection(COLLECTION_NAME)

        peek_data = collection.peek(10)
        self.assertIn("ids", peek_data)
        self.assertIn("documents", peek_data)
        self.assertIn("metadatas", peek_data)
        self.assertIn("embeddings", peek_data)

        self.assertEqual(len(peek_data["ids"]), 10)
        self.assertEqual(len(peek_data["documents"]), 10)
        self.assertEqual(len(peek_data["embeddings"]), 10)

        # Embedding dimension check for nomic-embed-text
        for emb in peek_data["embeddings"]:
            self.assertEqual(len(emb), 768, "Embedding dimension mismatch! Expected 768 dimensions.")

        # Re-verify chunk count has not changed after peeking
        post_count = collection.count()
        self.assertEqual(post_count, 75039)


if __name__ == "__main__":
    unittest.main()
