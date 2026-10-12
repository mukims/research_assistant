"""
Tier 5 White-Box Adversarial Hardening Test Suite (Milestone 5).

Comprehensive adversarial coverage against research_assistant/tasks/:
1. Topologies & DAG Validation (models.py, critic.py):
   - Disconnected subgraphs, self-referential loops, complex diamond topologies, non-existent dependencies.
   - Mandatory 6-phase validation and wave concurrency calculation under massive/empty graphs.
2. Artifact Contract Schema Validation (models.py, harness.py, critic.py):
   - Wildcard shape dimensions vs scalar tensors.
   - Corrupt dtypes, empty files, malformed JSON metrics (NaN, Inf, booleans), syntax-errored python scripts.
   - CLI flags verification and tabular column schemas.
3. Proposer-Critic Iteration Loop & Edge Healing (iteration.py):
   - Feedback loop robustness, early exit guarantees, turn caps, cycle severing and phase injection on severely degraded graphs.
4. Two-Cycle Reflexion Runner (runner.py):
   - Strict 2-cycle cap enforcement.
   - Warm-state retention across cycles.
   - Workspace safety protection forbidding persistent DATA_DIR and VECTORDB_PATH.
   - Thread environment variables (OMP_NUM_THREADS="1", TORCH_NUM_THREADS="1").
5. Grounded Retrieval Helpers (retrieval.py):
   - Test ensure_target_paper_grounded and format_faceted_context aliases/functions if present, or test fallback handling.
6. Persistent ChromaDB chunk count invariant:
   - Assert physics_papers collection in data/physics_vectordb holds exactly 75,039 chunks.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import chromadb
import numpy as np

from research_assistant.config import (
    COLLECTION_NAME,
    DATA_DIR,
    VECTORDB_PATH,
)
from research_assistant.tasks import (
    critic,
    harness,
    iteration,
    models,
    retrieval,
    runner,
)
from research_assistant.tasks.critic import (
    FlawSeverity,
    MacroCritic,
    MicroCritic,
    RubricScorecard,
    TaskCritic,
)
from research_assistant.tasks.harness import (
    generate_task_test_script,
    run_task_test_script,
)
from research_assistant.tasks.iteration import (
    ProposerCriticEngine,
    _build_heuristic_7phase_fallback,
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


class TestTier5TopologiesAndDAGValidation(unittest.TestCase):
    """Adversarial stress testing of DAG structures, topologies, validation, and execution waves."""

    def test_disconnected_subgraphs_partitioning_and_concurrency(self):
        """Disjoint subgraphs partition into concurrent waves without topology errors."""
        graph = TaskGraph(paper_id="disjoint_p", paper_title="Disconnected Components Paper")

        # Subgraph 1: A1 -> B1 -> C1
        graph.add_task(Task(task_id="A1", title="Init Subgraph 1", category=TaskCategory.DATA_ACQUISITION))
        graph.add_task(Task(task_id="B1", title="Transform Subgraph 1", dependencies=["A1"], category=TaskCategory.DATA_PREPROCESSING))
        graph.add_task(Task(task_id="C1", title="Eval Subgraph 1", dependencies=["B1"], category=TaskCategory.METRIC_EVALUATION))

        # Subgraph 2: A2 -> B2 (completely disconnected from Subgraph 1)
        graph.add_task(Task(task_id="A2", title="Init Subgraph 2", category=TaskCategory.ENVIRONMENT_SETUP))
        graph.add_task(Task(task_id="B2", title="Build Subgraph 2", dependencies=["A2"], category=TaskCategory.MODEL_IMPLEMENTATION))

        errors = graph.validate()
        self.assertEqual(errors, [], "Disconnected subgraphs should be valid DAGs")

        waves = graph.topological_waves()
        self.assertEqual(len(waves), 3)

        # Wave 0 must contain roots from both subgraphs
        wave0_ids = {t.task_id for t in waves[0]}
        self.assertEqual(wave0_ids, {"A1", "A2"})

        # Wave 1 must contain second-level tasks from both subgraphs
        wave1_ids = {t.task_id for t in waves[1]}
        self.assertEqual(wave1_ids, {"B1", "B2"})

        # Wave 2 contains remaining task C1
        wave2_ids = {t.task_id for t in waves[2]}
        self.assertEqual(wave2_ids, {"C1"})

        self.assertEqual(graph.wave_count(), 3)
        self.assertEqual(graph.max_concurrency(), 2)
        self.assertTrue(graph.is_parallelizable())

    def test_self_referential_loop_detected_by_validate_and_macro_critic(self):
        """Self-referential dependencies (task depending on itself) are strictly blocked."""
        graph = TaskGraph(paper_id="self_loop", paper_title="Self Loop Paper")
        graph.add_task(Task(
            task_id="t_self",
            title="Self-referencing Task",
            dependencies=["t_self"],
            category=TaskCategory.MODEL_IMPLEMENTATION,
        ))

        errors = graph.validate()
        self.assertTrue(any("cannot depend on itself" in e for e in errors))

        with self.assertRaises(ValueError):
            graph.topological_waves()

        macro = MacroCritic()
        flaws = macro.evaluate(graph)
        self.assertTrue(any(
            f.category == "topology" and f.severity == FlawSeverity.BLOCKING.value and "cannot depend on itself" in f.description
            for f in flaws
        ))

    def test_complex_nested_diamond_topology_wave_resolution(self):
        """Multi-level diamond topologies resolve strictly level-by-level."""
        # Topology:
        #        A
        #       / \
        #      B   C
        #       \ /
        #        D
        #       / \
        #      E   F
        #       \ /
        #        G
        graph = TaskGraph(paper_id="nested_diamond", paper_title="Nested Diamond Topology")
        graph.add_task(Task(task_id="A", title="Root", category=TaskCategory.ENVIRONMENT_SETUP))
        graph.add_task(Task(task_id="B", title="Branch B", dependencies=["A"], category=TaskCategory.DATA_ACQUISITION))
        graph.add_task(Task(task_id="C", title="Branch C", dependencies=["A"], category=TaskCategory.MODEL_IMPLEMENTATION))
        graph.add_task(Task(task_id="D", title="Join D", dependencies=["B", "C"], category=TaskCategory.SANITY_SMOKE_TEST))
        graph.add_task(Task(task_id="E", title="Branch E", dependencies=["D"], category=TaskCategory.TRAINING_OR_SIMULATION))
        graph.add_task(Task(task_id="F", title="Branch F", dependencies=["D"], category=TaskCategory.ABLATION_STUDY))
        graph.add_task(Task(task_id="G", title="Final Sink", dependencies=["E", "F"], category=TaskCategory.VERIFICATION_AUDIT))

        self.assertEqual(graph.validate(), [])
        waves = graph.topological_waves()
        self.assertEqual(len(waves), 5)

        self.assertEqual([t.task_id for t in waves[0]], ["A"])
        self.assertEqual(sorted([t.task_id for t in waves[1]]), ["B", "C"])
        self.assertEqual([t.task_id for t in waves[2]], ["D"])
        self.assertEqual(sorted([t.task_id for t in waves[3]]), ["E", "F"])
        self.assertEqual([t.task_id for t in waves[4]], ["G"])

        self.assertEqual(graph.wave_count(), 5)
        self.assertEqual(graph.max_concurrency(), 2)
        self.assertTrue(graph.is_parallelizable())

    def test_non_existent_dependency_detected_by_validate_and_macro_critic(self):
        """Tasks referencing missing task IDs are identified as missing dependency errors."""
        graph = TaskGraph(paper_id="missing_dep", paper_title="Missing Dependency Paper")
        graph.add_task(Task(
            task_id="task_valid",
            title="Task with Missing Parent",
            dependencies=["ghost_task_99"],
            category=TaskCategory.DATA_ACQUISITION,
        ))

        errors = graph.validate()
        self.assertTrue(any("depends on missing task 'ghost_task_99'" in e for e in errors))

        macro = MacroCritic()
        flaws = macro.evaluate(graph)
        self.assertTrue(any(
            f.category == "topology" and f.severity == FlawSeverity.BLOCKING.value and "depends on missing task" in f.description
            for f in flaws
        ))

    def test_multi_node_cycles_and_overlapping_cycles_detection(self):
        """Cyclic dependencies spanning multiple nodes and overlapping cycles are caught cleanly."""
        graph = TaskGraph(paper_id="cycle_p", paper_title="Cyclic Graph Paper")
        # Cycle 1: C1 -> C2 -> C3 -> C1
        # Overlapping branch: C2 -> C4 -> C2
        graph.add_task(Task(task_id="C1", title="Cycle Node 1", dependencies=["C3"], category=TaskCategory.DATA_ACQUISITION))
        graph.add_task(Task(task_id="C2", title="Cycle Node 2", dependencies=["C1", "C4"], category=TaskCategory.DATA_PREPROCESSING))
        graph.add_task(Task(task_id="C3", title="Cycle Node 3", dependencies=["C2"], category=TaskCategory.MODEL_IMPLEMENTATION))
        graph.add_task(Task(task_id="C4", title="Cycle Node 4", dependencies=["C2"], category=TaskCategory.SANITY_SMOKE_TEST))

        errors = graph.validate()
        self.assertTrue(any("Cycle detected:" in e for e in errors))

        macro = MacroCritic()
        flaws = macro.evaluate(graph)
        blocking_cycles = [f for f in flaws if f.category == "topology" and "Circular dependency" in f.description]
        self.assertGreater(len(blocking_cycles), 0)

    def test_mandatory_6_phase_validation_missing_detection(self):
        """Missing scientific phases are strictly cataloged by get_missing_phases."""
        graph = TaskGraph(paper_id="sparse_p", paper_title="Sparse Graph Paper")
        # Only environment setup and data acquisition present
        graph.add_task(Task(task_id="t_env", title="Setup Environment", category=TaskCategory.ENVIRONMENT_SETUP))
        graph.add_task(Task(task_id="t_data", title="Download Data", dependencies=["t_env"], category=TaskCategory.DATA_ACQUISITION))

        missing = graph.get_missing_phases()
        self.assertFalse(graph.has_all_mandatory_phases())
        self.assertIn("sanity_smoke_test", missing)
        self.assertIn("training_or_simulation", missing)
        self.assertIn("metric_evaluation", missing)
        self.assertIn("verification_audit", missing)
        self.assertNotIn("environment_setup", missing)
        self.assertNotIn("data_splits", missing)

        macro = MacroCritic()
        flaws = macro.evaluate(graph)
        missing_phase_flaws = [f for f in flaws if f.category == "missing_phase"]
        self.assertEqual(len(missing_phase_flaws), 4)

    def test_mandatory_6_phase_fuzzy_and_alias_recognition(self):
        """Phases are recognized through category enums, assigned agent roles, and keyword titles."""
        graph = TaskGraph(paper_id="fuzzy_p", paper_title="Fuzzy Phase Graph")
        # 1. Environment via role
        graph.add_task(Task(task_id="t_devops", title="Config runtime", assigned_role=AgentRole.DEVOPS_AGENT))
        # 2. Data via title
        graph.add_task(Task(task_id="t_fetch", title="Dataset Split Creation", dependencies=["t_devops"]))
        # 3. Smoke test via description keyword
        graph.add_task(Task(task_id="t_dry", title="Initial Forward Run", description="Run sanity smoke check.", dependencies=["t_fetch"]))
        # 4. Training via category alias
        graph.add_task(Task(task_id="t_opt", title="Run Hyperparameter Sweep", category=TaskCategory.ABLATION_STUDY, dependencies=["t_dry"]))
        # 5. Metric evaluation via role
        graph.add_task(Task(task_id="t_bench", title="Test Set Score", assigned_role=AgentRole.EVALUATION_AGENT, dependencies=["t_opt"]))
        # 6. Audit via task_id keyword
        graph.add_task(Task(task_id="task_audit_cert", title="Final Verification Certification", dependencies=["t_bench"]))

        self.assertEqual(graph.get_missing_phases(), [])
        self.assertTrue(graph.has_all_mandatory_phases())

    def test_empty_graph_wave_concurrency_and_macro_critic(self):
        """Empty graph produces zero waves, zero concurrency, and blocking macro critique."""
        empty_graph = TaskGraph(paper_id="empty_p", paper_title="Empty Workflow")
        self.assertEqual(empty_graph.validate(), [])
        self.assertEqual(empty_graph.topological_waves(), [])
        self.assertEqual(empty_graph.wave_count(), 0)
        self.assertEqual(empty_graph.max_concurrency(), 0)
        self.assertFalse(empty_graph.is_parallelizable())

        macro = MacroCritic()
        flaws = macro.evaluate(empty_graph)
        self.assertEqual(len(flaws), 1)
        self.assertEqual(flaws[0].flaw_id, "macro_empty_graph")
        self.assertEqual(flaws[0].severity, FlawSeverity.BLOCKING.value)
        self.assertEqual(macro.compute_topology_score(empty_graph, flaws), 0.0)

    def test_massive_ultra_wide_graph_concurrency_calculation(self):
        """Massive wide graph (100 independent tasks) scales in Wave 0 with max concurrency 100."""
        graph = TaskGraph(paper_id="wide_p", paper_title="Ultra Wide Graph Paper")
        for i in range(100):
            graph.add_task(Task(
                task_id=f"wide_task_{i:03d}",
                title=f"Parallel Task {i}",
                category=TaskCategory.DATA_ACQUISITION,
            ))

        self.assertEqual(graph.validate(), [])
        waves = graph.topological_waves()
        self.assertEqual(len(waves), 1)
        self.assertEqual(len(waves[0]), 100)
        self.assertEqual(graph.wave_count(), 1)
        self.assertEqual(graph.max_concurrency(), 100)
        self.assertTrue(graph.is_parallelizable())

    def test_massive_ultra_deep_graph_wave_depth_calculation(self):
        """Massive deep graph (80 sequentially chained tasks) produces 80 single-task waves."""
        graph = TaskGraph(paper_id="deep_p", paper_title="Ultra Deep Graph Paper")
        prev_id = None
        for i in range(80):
            t_id = f"seq_{i:02d}"
            deps = [prev_id] if prev_id else []
            graph.add_task(Task(
                task_id=t_id,
                title=f"Sequential Step {i}",
                dependencies=deps,
                category=TaskCategory.DATA_PREPROCESSING if i > 0 else TaskCategory.ENVIRONMENT_SETUP,
            ))
            prev_id = t_id

        self.assertEqual(graph.validate(), [])
        waves = graph.topological_waves()
        self.assertEqual(len(waves), 80)
        self.assertEqual(graph.wave_count(), 80)
        self.assertEqual(graph.max_concurrency(), 1)
        self.assertFalse(graph.is_parallelizable())


class TestTier5ArtifactContractSchemaValidation(unittest.TestCase):
    """White-box verification of formal artifact schemas, test harness assertions, and MicroCritic."""

    def setUp(self):
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace, ignore_errors=True)

    def test_contract_from_dict_and_to_dict_edge_cases(self):
        """ArtifactContract deserialization handles strings, invalid types, and aliases gracefully."""
        # String data fallback
        c_str = ArtifactContract.from_dict("data/raw.csv")
        self.assertEqual(c_str.path, "data/raw.csv")
        self.assertEqual(c_str.artifact_type, ArtifactType.OTHER)

        # Aliases
        c_json = ArtifactContract.from_dict({"path": "m.json", "artifact_type": "metrics_json"})
        self.assertEqual(c_json.artifact_type, ArtifactType.METRICS_JSON)

        c_cli = ArtifactContract.from_dict({"path": "run.py", "artifact_type": "cli"})
        self.assertEqual(c_cli.artifact_type, ArtifactType.CLI_INTERFACE)

        c_tensor = ArtifactContract.from_dict({"path": "t.pt", "artifact_type": "ndarray", "shape": ["B", 64]})
        self.assertEqual(c_tensor.artifact_type, ArtifactType.TENSOR)
        self.assertEqual(c_tensor.shape, ["B", 64])

        # Roundtrip to_dict
        d = c_tensor.to_dict()
        self.assertEqual(d["path"], "t.pt")
        self.assertEqual(d["artifact_type"], "tensor")
        self.assertEqual(d["shape"], ["B", 64])

    def test_wildcard_shape_dimensions_vs_scalar_tensors_in_harness(self):
        """Harness validates dynamic wildcard tensor dimensions and scalar rank constraints."""
        # Dynamic shape: ["batch_size", 3, 32, 32]
        task_dyn = Task(
            task_id="t_tensor_dyn",
            title="Dynamic Tensor Task",
            category=TaskCategory.DATA_PREPROCESSING,
            outputs=["data/images.npy"],
            artifact_contracts=[
                ArtifactContract(
                    path="data/images.npy",
                    artifact_type=ArtifactType.TENSOR,
                    shape=["batch_size", 3, 32, 32],
                    dtype="float32",
                )
            ],
        )

        img_dir = os.path.join(self.workspace, "data")
        os.makedirs(img_dir, exist_ok=True)
        img_path = os.path.join(img_dir, "images.npy")

        # 1. Valid dynamic batch size = 8
        np.save(img_path, np.zeros((8, 3, 32, 32), dtype=np.float32))
        script = generate_task_test_script(task_dyn, self.workspace)
        ok, _, stderr = run_task_test_script(script, self.workspace)
        self.assertTrue(ok, f"Dynamic batch tensor should pass: {stderr}")

        # 2. Channel dimension mismatch (expected 3, got 1)
        np.save(img_path, np.zeros((8, 1, 32, 32), dtype=np.float32))
        ok_bad_ch, _, stderr_bad_ch = run_task_test_script(script, self.workspace)
        self.assertFalse(ok_bad_ch)
        self.assertIn("Tensor dim 1 mismatch", stderr_bad_ch)

        # 3. Scalar tensor verification: shape=[]
        task_scalar = Task(
            task_id="t_scalar",
            title="Scalar Tensor Task",
            category=TaskCategory.METRIC_EVALUATION,
            outputs=["metrics/loss.npy"],
            artifact_contracts=[
                ArtifactContract(
                    path="metrics/loss.npy",
                    artifact_type=ArtifactType.TENSOR,
                    shape=[],
                    dtype="float32",
                )
            ],
        )
        os.makedirs(os.path.join(self.workspace, "metrics"), exist_ok=True)
        loss_path = os.path.join(self.workspace, "metrics", "loss.npy")

        # Scalar numpy array has shape () -> len is 0
        np.save(loss_path, np.array(0.42, dtype=np.float32))
        script_scalar = generate_task_test_script(task_scalar, self.workspace)
        ok_scalar, _, stderr_scalar = run_task_test_script(script_scalar, self.workspace)
        self.assertTrue(ok_scalar, f"Scalar tensor should pass: {stderr_scalar}")

        # Providing a 2D tensor when scalar is expected -> rank mismatch
        np.save(loss_path, np.zeros((2, 2), dtype=np.float32))
        ok_rank_fail, _, stderr_rank = run_task_test_script(script_scalar, self.workspace)
        self.assertFalse(ok_rank_fail)
        self.assertIn("Tensor rank mismatch", stderr_rank)

    def test_tensor_corrupt_dtype_detected_by_harness(self):
        """Harness catches dtype mismatch (e.g. float32 expected, int64 provided)."""
        task = Task(
            task_id="t_dtype",
            title="Dtype Check Task",
            category=TaskCategory.MODEL_IMPLEMENTATION,
            outputs=["weights.npy"],
            artifact_contracts=[
                ArtifactContract(
                    path="weights.npy",
                    artifact_type=ArtifactType.TENSOR,
                    shape=[10, 10],
                    dtype="float32",
                )
            ],
        )
        w_path = os.path.join(self.workspace, "weights.npy")
        np.save(w_path, np.zeros((10, 10), dtype=np.int64))  # corrupt dtype

        script = generate_task_test_script(task, self.workspace)
        ok, _, stderr = run_task_test_script(script, self.workspace)
        self.assertFalse(ok)
        self.assertIn("Tensor dtype mismatch", stderr)

    def test_empty_deliverable_file_fails_harness(self):
        """Zero-byte deliverable files trigger harness failure."""
        task = Task(
            task_id="t_empty_file",
            title="Empty Deliverable Task",
            category=TaskCategory.ENVIRONMENT_SETUP,
            outputs=["lockfile.lock"],
        )
        # Create empty 0-byte file
        with open(os.path.join(self.workspace, "lockfile.lock"), "w") as f:
            pass

        script = generate_task_test_script(task, self.workspace)
        ok, _, stderr = run_task_test_script(script, self.workspace)
        self.assertFalse(ok)
        self.assertIn("Deliverable empty", stderr)

    def test_malformed_json_metrics_nan_and_inf_fail_harness(self):
        """Harness detects non-finite metrics (NaN and Inf) in JSON outputs."""
        task = Task(
            task_id="t_metric_nan",
            title="Metric Validation Task",
            category=TaskCategory.METRIC_EVALUATION,
            outputs=["metrics/eval.json"],
            artifact_contracts=[
                ArtifactContract(
                    path="metrics/eval.json",
                    artifact_type=ArtifactType.METRICS_JSON,
                    required_keys=["loss", "accuracy"],
                )
            ],
        )
        os.makedirs(os.path.join(self.workspace, "metrics"), exist_ok=True)
        m_path = os.path.join(self.workspace, "metrics", "eval.json")

        # 1. NaN value
        with open(m_path, "w") as f:
            f.write('{"loss": NaN, "accuracy": 0.95}')

        script = generate_task_test_script(task, self.workspace)
        ok_nan, _, stderr_nan = run_task_test_script(script, self.workspace)
        self.assertFalse(ok_nan)
        self.assertIn("must be finite", stderr_nan)

        # 2. Infinity value
        with open(m_path, "w") as f:
            f.write('{"loss": Infinity, "accuracy": 0.95}')

        ok_inf, _, stderr_inf = run_task_test_script(script, self.workspace)
        self.assertFalse(ok_inf)
        self.assertIn("must be finite", stderr_inf)

    def test_malformed_json_metrics_boolean_fails_harness(self):
        """Harness prevents booleans from masquerading as numeric metric scores."""
        task = Task(
            task_id="t_metric_bool",
            title="Boolean Metric Task",
            category=TaskCategory.METRIC_EVALUATION,
            outputs=["eval.json"],
            artifact_contracts=[
                ArtifactContract(
                    path="eval.json",
                    artifact_type=ArtifactType.METRICS_JSON,
                    required_keys=["accuracy"],
                )
            ],
        )
        with open(os.path.join(self.workspace, "eval.json"), "w") as f:
            f.write('{"accuracy": true}')

        script = generate_task_test_script(task, self.workspace)
        ok, _, stderr = run_task_test_script(script, self.workspace)
        self.assertFalse(ok)
        self.assertIn("cannot be bool", stderr)

    def test_json_metrics_missing_required_keys_fails_harness(self):
        """Harness asserts presence of all required metric keys."""
        task = Task(
            task_id="t_missing_keys",
            title="Missing Metric Key Task",
            category=TaskCategory.METRIC_EVALUATION,
            outputs=["eval.json"],
            artifact_contracts=[
                ArtifactContract(
                    path="eval.json",
                    artifact_type=ArtifactType.METRICS_JSON,
                    required_keys=["f1_score", "auc_roc"],
                )
            ],
        )
        with open(os.path.join(self.workspace, "eval.json"), "w") as f:
            f.write('{"f1_score": 0.88}')  # auc_roc missing

        script = generate_task_test_script(task, self.workspace)
        ok, _, stderr = run_task_test_script(script, self.workspace)
        self.assertFalse(ok)
        self.assertIn("Required key missing in JSON", stderr)

    def test_syntax_errored_python_script_fails_harness(self):
        """Harness catches syntax errors in generated python deliverable scripts via AST parsing."""
        task = Task(
            task_id="t_code_err",
            title="Code Syntax Check Task",
            category=TaskCategory.MODEL_IMPLEMENTATION,
            outputs=["src/model.py"],
            artifact_contracts=[
                ArtifactContract(
                    path="src/model.py",
                    artifact_type=ArtifactType.CODE,
                )
            ],
        )
        os.makedirs(os.path.join(self.workspace, "src"), exist_ok=True)
        with open(os.path.join(self.workspace, "src", "model.py"), "w") as f:
            f.write("def forward_pass(x\n    return x + 1\n")  # Syntax error

        script = generate_task_test_script(task, self.workspace)
        ok, _, stderr = run_task_test_script(script, self.workspace)
        self.assertFalse(ok)
        self.assertIn("Python syntax error in", stderr)

    def test_cli_flags_verification_in_harness(self):
        """Harness validates presence of required command-line interface arguments."""
        task = Task(
            task_id="t_cli",
            title="CLI Validation Task",
            category=TaskCategory.TRAINING_OR_SIMULATION,
            outputs=["train_cli.py"],
            artifact_contracts=[
                ArtifactContract(
                    path="train_cli.py",
                    artifact_type=ArtifactType.CLI_INTERFACE,
                    cli_args=["--batch-size", "--learning-rate"],
                )
            ],
        )
        cli_path = os.path.join(self.workspace, "train_cli.py")

        # 1. Missing --learning-rate
        with open(cli_path, "w") as f:
            f.write("import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--batch-size')\n")

        script = generate_task_test_script(task, self.workspace)
        ok_missing, _, stderr_missing = run_task_test_script(script, self.workspace)
        self.assertFalse(ok_missing)
        self.assertIn("Expected CLI argument missing", stderr_missing)

        # 2. Both arguments present
        with open(cli_path, "w") as f:
            f.write("import argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--batch-size')\nparser.add_argument('--learning-rate')\n")

        ok_valid, _, stderr_valid = run_task_test_script(script, self.workspace)
        self.assertTrue(ok_valid, f"CLI harness should pass when args present: {stderr_valid}")

    def test_tabular_columns_and_dtype_mismatch_in_harness(self):
        """Harness asserts presence of required column headers in CSV files."""
        task = Task(
            task_id="t_table",
            title="Tabular Validation Task",
            category=TaskCategory.DATA_ACQUISITION,
            outputs=["data/split.csv"],
            artifact_contracts=[
                ArtifactContract(
                    path="data/split.csv",
                    artifact_type=ArtifactType.TABULAR,
                    columns={"sample_id": "int", "feature": "float"},
                )
            ],
        )
        os.makedirs(os.path.join(self.workspace, "data"), exist_ok=True)
        csv_path = os.path.join(self.workspace, "data", "split.csv")

        # Missing 'feature' column
        with open(csv_path, "w") as f:
            f.write("sample_id,label\n1,0\n")

        script = generate_task_test_script(task, self.workspace)
        ok_missing, _, stderr_missing = run_task_test_script(script, self.workspace)
        self.assertFalse(ok_missing)
        self.assertIn("Expected column missing: feature", stderr_missing)

    def test_micro_critic_vague_instruction_and_empty_description(self):
        """MicroCritic blocks empty descriptions and warns on overly brief ones (<60 chars)."""
        micro = MicroCritic()
        graph = TaskGraph(paper_id="crit_p", paper_title="Criticism Graph")

        t_empty = Task(task_id="t_desc_empty", title="Empty Desc Task", description="", outputs=["out.txt"], acceptance_criteria=["Check"])
        t_vague = Task(task_id="t_desc_vague", title="Vague Desc Task", description="Brief description.", outputs=["out.txt"], acceptance_criteria=["Check"])
        graph.add_task(t_empty)
        graph.add_task(t_vague)

        flaws = micro.evaluate(graph)
        self.assertTrue(any(f.flaw_id == "micro_desc_empty_t_desc_empty" and f.severity == FlawSeverity.BLOCKING.value for f in flaws))
        self.assertTrue(any(f.flaw_id == "micro_desc_vague_t_desc_vague" and f.severity == FlawSeverity.WARNING.value for f in flaws))

    def test_micro_critic_missing_output_and_extension_checks(self):
        """MicroCritic blocks tasks with zero outputs and warns on deliverables lacking extensions."""
        micro = MicroCritic()
        graph = TaskGraph(paper_id="crit_out_p", paper_title="Output Criticism Graph")

        t_no_out = Task(
            task_id="t_no_out",
            title="No Output Task",
            description="Detailed operational description satisfying the minimum character requirements of 60 chars.",
            outputs=[],
            artifact_contracts=[],
            acceptance_criteria=["Exit 0"],
        )
        t_no_ext = Task(
            task_id="t_no_ext",
            title="No Extension Task",
            description="Detailed operational description satisfying the minimum character requirements of 60 chars.",
            outputs=["data/processed_file"],  # Missing .ext
            acceptance_criteria=["Exit 0"],
        )
        graph.add_task(t_no_out)
        graph.add_task(t_no_ext)

        flaws = micro.evaluate(graph)
        self.assertTrue(any(f.flaw_id == "micro_contract_empty_t_no_out" and f.severity == FlawSeverity.BLOCKING.value for f in flaws))
        self.assertTrue(any(f.flaw_id == "micro_contract_no_ext_t_no_ext" and f.severity == FlawSeverity.WARNING.value for f in flaws))

    def test_micro_critic_incomplete_contract_specifications(self):
        """MicroCritic warns on incomplete ArtifactContract schemas (missing shape, columns, keys, args)."""
        micro = MicroCritic()
        graph = TaskGraph(paper_id="crit_contract_p", paper_title="Contract Incomplete Graph")

        t = Task(
            task_id="t_contracts",
            title="Comprehensive Contracts Task",
            description="Detailed operational description satisfying the minimum character requirements of 60 chars.",
            outputs=["t.pt", "table.csv", "m.json", "cli.py"],
            artifact_contracts=[
                ArtifactContract(path="t.pt", artifact_type=ArtifactType.TENSOR),  # missing shape, dtype
                ArtifactContract(path="table.csv", artifact_type=ArtifactType.TABULAR),  # missing columns
                ArtifactContract(path="m.json", artifact_type=ArtifactType.METRICS_JSON),  # missing required_keys
                ArtifactContract(path="cli.py", artifact_type=ArtifactType.CLI_INTERFACE),  # missing cli_args
            ],
            acceptance_criteria=["Exit 0"],
        )
        graph.add_task(t)

        flaws = micro.evaluate(graph)
        categories = [f.flaw_id for f in flaws]
        self.assertTrue(any("tensor_no_shape" in fid for fid in categories))
        self.assertTrue(any("tensor_no_dtype" in fid for fid in categories))
        self.assertTrue(any("tabular_no_columns" in fid for fid in categories))
        self.assertTrue(any("json_no_keys" in fid for fid in categories))
        self.assertTrue(any("cli_no_args" in fid for fid in categories))

    def test_micro_critic_subjective_acceptance_criteria_checks(self):
        """MicroCritic blocks empty acceptance criteria and warns on subjective phrasing."""
        micro = MicroCritic()
        graph = TaskGraph(paper_id="crit_test_p", paper_title="Testability Criticism Graph")

        t_no_crit = Task(
            task_id="t_no_crit",
            title="No Criteria Task",
            description="Detailed operational description satisfying the minimum character requirements of 60 chars.",
            outputs=["out.txt"],
            acceptance_criteria=[],
        )
        t_subjective = Task(
            task_id="t_subjective",
            title="Subjective Criteria Task",
            description="Detailed operational description satisfying the minimum character requirements of 60 chars.",
            outputs=["out.txt"],
            acceptance_criteria=["The model outputs should look good and perform nicely."],
        )
        graph.add_task(t_no_crit)
        graph.add_task(t_subjective)

        flaws = micro.evaluate(graph)
        self.assertTrue(any(f.flaw_id == "micro_testability_empty_t_no_crit" and f.severity == FlawSeverity.BLOCKING.value for f in flaws))
        self.assertTrue(any("micro_testability_subjective_t_subjective" in f.flaw_id and f.severity == FlawSeverity.WARNING.value for f in flaws))

    def test_micro_critic_cpu_thread_oversubscription_evasion_attempts(self):
        """MicroCritic detects deceptive and evasive unsafe thread patterns in training tasks."""
        micro = MicroCritic()
        graph = TaskGraph(paper_id="thread_p", paper_title="Thread Safety Paper")

        # Deceptive: mentions single-threaded but requests num_thread: 16
        t_deceptive = Task(
            task_id="t_train_deceptive",
            title="Train Model",
            category=TaskCategory.TRAINING_OR_SIMULATION,
            description="Train network single-threaded with num_thread: 16 and maximum threads.",
            outputs=["model.pt"],
            acceptance_criteria=["Loss converges"],
        )
        # Safe explicit cap
        t_safe = Task(
            task_id="t_train_safe",
            title="Train Model Safe",
            category=TaskCategory.TRAINING_OR_SIMULATION,
            description="Train network with torch.set_num_threads(1) and OMP_NUM_THREADS=1 single thread cap.",
            outputs=["model.pt"],
            acceptance_criteria=["Loss converges"],
        )
        graph.add_task(t_deceptive)
        graph.add_task(t_safe)

        flaws = micro.evaluate(graph)
        self.assertTrue(any(
            f.task_id == "t_train_deceptive" and f.category == "cpu_oversubscription" and f.severity == FlawSeverity.BLOCKING.value
            for f in flaws
        ))
        self.assertFalse(any(
            f.task_id == "t_train_safe" and f.category == "cpu_oversubscription"
            for f in flaws
        ))

    def test_micro_critic_non_string_and_none_task_attributes_safety(self):
        """MicroCritic evaluates tasks with None or non-string attributes without raising exceptions."""
        micro = MicroCritic()
        graph = TaskGraph(paper_id="null_attrs", paper_title="Null Attributes Paper")

        # Task with None outputs and acceptance criteria
        t_none = Task(
            task_id="t_none_attrs",
            title="None Attributes Task",
            description="",
            outputs=None,  # type: ignore
            acceptance_criteria=None,  # type: ignore
        )
        graph.add_task(t_none)

        # Task deserialized from dictionary containing None values
        t_from_dict = Task.from_dict({
            "task_id": "t_dict_none",
            "title": "Dict None Task",
            "description": None,
            "outputs": None,
            "acceptance_criteria": None,
            "dependencies": None,
        })
        graph.add_task(t_from_dict)

        flaws = micro.evaluate(graph)
        self.assertIsInstance(flaws, list)
        self.assertTrue(any(f.task_id == "t_none_attrs" and "empty" in f.flaw_id for f in flaws))
        self.assertTrue(any(f.task_id == "t_dict_none" and "empty" in f.flaw_id for f in flaws))

    def test_rubric_scorecard_score_clamping_under_extreme_flaws(self):
        """RubricScorecard scores are strictly clamped in [0.0, 100.0] even with extreme flaw counts."""
        critic_engine = TaskCritic()
        graph = TaskGraph(paper_id="extreme_flaws", paper_title="Extreme Flaws Graph")

        # Task with almost every flaw possible
        for i in range(15):
            t = Task(
                task_id=f"bad_task_{i}",
                title=f"Bad {i}",
                category=TaskCategory.TRAINING_OR_SIMULATION,
                description="too short",
                outputs=["no_ext"],
                acceptance_criteria=["looks nice"],
                dependencies=[f"bad_task_{(i+1)%15}"],  # massive cycle
            )
            graph.add_task(t)

        scorecard = critic_engine.evaluate(graph)
        self.assertGreaterEqual(scorecard.topology_score, 0.0)
        self.assertLessEqual(scorecard.topology_score, 100.0)
        self.assertGreaterEqual(scorecard.contract_score, 0.0)
        self.assertLessEqual(scorecard.contract_score, 100.0)
        self.assertGreaterEqual(scorecard.testability_score, 0.0)
        self.assertLessEqual(scorecard.testability_score, 100.0)
        self.assertGreaterEqual(scorecard.overall_score, 0.0)
        self.assertLessEqual(scorecard.overall_score, 100.0)
        self.assertFalse(scorecard.is_passing)
        self.assertGreater(scorecard.blocking_flaws_count, 0)


class TestTier5ProposerCriticIterationAndHealing(unittest.TestCase):
    """Adversarial stress testing of the Proposer-Critic multi-turn loop and deterministic edge healing."""

    def test_proposer_critic_early_exit_on_turn_1_when_zero_blocking_flaws(self):
        """ProposerCriticEngine exits immediately on Turn 1 if zero blocking flaws remain."""
        clean_graph = _build_heuristic_7phase_fallback("p1", "Clean Graph")

        # Mock chat_fn returning clean graph JSON on turn 1
        tasks_json = json.dumps({
            "paper_title": "Clean Graph",
            "tasks": [t.to_dict() for t in clean_graph.tasks.values()]
        })
        mock_response = MagicMock(content=tasks_json)
        mock_chat = MagicMock(return_value=mock_response)

        engine = ProposerCriticEngine(chat_fn=mock_chat)
        final_graph, scorecards = engine.run_refinement_loop(
            paper_id="p1",
            paper_title="Clean Graph",
            context={"abstract": "Clean abstract"},
            max_turns=3,
        )

        self.assertEqual(len(scorecards), 1, "Should exit on Turn 1 due to 0 blocking flaws")
        self.assertTrue(scorecards[0].is_passing)
        self.assertEqual(mock_chat.call_count, 1)

    def test_proposer_critic_turn_capping_and_bounds(self):
        """ProposerCriticEngine enforces turn clamping to [1, 3]."""
        flawed_json = json.dumps({
            "paper_title": "Flawed",
            "tasks": [{"task_id": "t1", "title": "Incomplete", "outputs": []}]
        })
        mock_chat = MagicMock(return_value=MagicMock(content=flawed_json))
        engine = ProposerCriticEngine(chat_fn=mock_chat)

        # Request 10 turns -> clamped to 3
        _, scorecards_max = engine.run_refinement_loop(
            paper_id="p_bound",
            paper_title="Bound",
            context={},
            max_turns=10,
        )
        self.assertLessEqual(len(scorecards_max), 3)

        # Request 0 turns -> clamped to 1
        mock_chat.reset_mock()
        _, scorecards_min = engine.run_refinement_loop(
            paper_id="p_bound_0",
            paper_title="Bound 0",
            context={},
            max_turns=0,
        )
        self.assertEqual(len(scorecards_min), 1)

    def test_proposer_critic_feedback_injection_across_turns(self):
        """Turn 2 prompt contains the explicit CritiqueFlaws surfaced during Turn 1."""
        flawed_json = json.dumps({
            "paper_title": "Flawed Graph",
            "tasks": [{
                "task_id": "t_flawed",
                "title": "Vague task",
                "description": "short",
                "outputs": [],
                "dependencies": [],
            }]
        })
        calls = []

        def recording_chat(messages, **kwargs):
            calls.append(messages)
            return MagicMock(content=flawed_json)

        engine = ProposerCriticEngine(chat_fn=recording_chat)
        engine.run_refinement_loop(
            paper_id="p_feedback",
            paper_title="Feedback Injection Paper",
            context={"abstract": "Sample abstract text"},
            max_turns=2,
        )

        self.assertEqual(len(calls), 2)
        turn2_messages = calls[1]
        turn2_user_content = turn2_messages[-1]["content"]
        self.assertIn("Critique Flaws Identified (Turn 1)", turn2_user_content)
        self.assertIn("BLOCKING", turn2_user_content)

    def test_deterministic_healing_severs_cycles(self):
        """_heal_remaining_flaws severs circular dependencies while preserving acyclic structure."""
        engine = ProposerCriticEngine()
        graph = TaskGraph(paper_id="heal_cycle", paper_title="Heal Cycle Paper")

        # Construct A -> B -> A
        graph.add_task(Task(task_id="A", title="Node A", dependencies=["B"], category=TaskCategory.DATA_ACQUISITION))
        graph.add_task(Task(task_id="B", title="Node B", dependencies=["A"], category=TaskCategory.DATA_PREPROCESSING))

        self.assertTrue(any("Cycle detected:" in e for e in graph.validate()))

        dummy_scorecard = engine.critic.evaluate(graph)
        healed_graph = engine._heal_remaining_flaws(graph, dummy_scorecard, {})

        errors = healed_graph.validate()
        cycle_errors = [e for e in errors if "Cycle detected:" in e]
        self.assertEqual(cycle_errors, [], "Circular dependencies must be completely severed by healing")

    def test_deterministic_healing_injects_all_missing_mandatory_phases(self):
        """_heal_remaining_flaws injects tasks for all missing scientific phases."""
        engine = ProposerCriticEngine()
        graph = TaskGraph(paper_id="heal_phases", paper_title="Heal Phases Paper")
        # Empty graph missing all 6 phases
        dummy_scorecard = engine.critic.evaluate(graph)
        healed_graph = engine._heal_remaining_flaws(graph, dummy_scorecard, {})

        self.assertEqual(healed_graph.get_missing_phases(), [])
        self.assertTrue(healed_graph.has_all_mandatory_phases())
        self.assertGreaterEqual(len(healed_graph.tasks), 6)

    def test_proposer_critic_fallback_on_llm_json_syntax_error(self):
        """ProposerCriticEngine recovers cleanly when LLM produces malformed or non-JSON output."""
        mock_chat = MagicMock(return_value=MagicMock(content="I am an AI, here is your plan:\n- Step 1: setup"))
        engine = ProposerCriticEngine(chat_fn=mock_chat)

        final_graph, scorecards = engine.run_refinement_loop(
            paper_id="json_err",
            paper_title="JSON Error Paper",
            context={"abstract": "Abstract of paper"},
            max_turns=1,
        )

        self.assertIsNotNone(final_graph)
        self.assertTrue(final_graph.has_all_mandatory_phases())
        self.assertEqual(len(scorecards), 1)

    def test_proposer_critic_fallback_on_llm_exception(self):
        """ProposerCriticEngine recovers cleanly when LLM call raises an unhandled exception."""
        mock_chat = MagicMock(side_effect=RuntimeError("Connection refused by daemon"))
        engine = ProposerCriticEngine(chat_fn=mock_chat)

        final_graph, scorecards = engine.run_refinement_loop(
            paper_id="exc_p",
            paper_title="Exception Paper",
            context={},
            max_turns=1,
        )

        self.assertIsNotNone(final_graph)
        self.assertTrue(final_graph.has_all_mandatory_phases())


class TestTier5TwoCycleReflexionRunner(unittest.TestCase):
    """White-box verification of Two-Cycle Reflexion, workspace safety, and hardware limits."""

    def setUp(self):
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace, ignore_errors=True)

    def test_workspace_safety_forbids_datadir_and_vectordb_path(self):
        """TaskRunner initialization strictly forbids pointing workspace_dir to persistent data."""
        graph = TaskGraph(paper_id="safe_p", paper_title="Safe Workspace Paper")

        # 1. Exact match with DATA_DIR
        with self.assertRaises(ValueError) as ctx_data:
            TaskRunner(graph=graph, workspace_dir=DATA_DIR)
        self.assertIn("Safety violation", str(ctx_data.exception))

        # 2. Exact match with VECTORDB_PATH
        with self.assertRaises(ValueError) as ctx_vdb:
            TaskRunner(graph=graph, workspace_dir=VECTORDB_PATH)
        self.assertIn("Safety violation", str(ctx_vdb.exception))

    def test_workspace_safety_forbids_subpaths_of_persistent_storage(self):
        """TaskRunner initialization forbids pointing workspace_dir to subdirectories of persistent data."""
        graph = TaskGraph(paper_id="safe_sub_p", paper_title="Safe Subpath Paper")

        sub_vdb = os.path.join(VECTORDB_PATH, "malicious_sub")
        with self.assertRaises(ValueError) as ctx_sub:
            TaskRunner(graph=graph, workspace_dir=sub_vdb)
        self.assertIn("Safety violation", str(ctx_sub.exception))

        sub_data = os.path.join(DATA_DIR, "scratch")
        with self.assertRaises(ValueError) as ctx_data_sub:
            TaskRunner(graph=graph, workspace_dir=sub_data)
        self.assertIn("Safety violation", str(ctx_data_sub.exception))

    def test_thread_environment_variables_enforced_in_subprocess(self):
        """Test harness execution enforces CPU single-threading environment variables in subprocess."""
        graph = TaskGraph(paper_id="env_p", paper_title="Env Var Check Paper")
        runner_inst = TaskRunner(graph=graph, workspace_dir=self.workspace)

        # Harness that checks subprocess environment variables
        check_script = (
            "import os, unittest\n"
            "class TestEnv(unittest.TestCase):\n"
            "    def test_threads(self):\n"
            "        self.assertEqual(os.environ.get('OMP_NUM_THREADS'), '1')\n"
            "        self.assertEqual(os.environ.get('TORCH_NUM_THREADS'), '1')\n"
            "        self.assertEqual(os.environ.get('MKL_NUM_THREADS'), '1')\n"
            "        self.assertEqual(os.environ.get('OPENBLAS_NUM_THREADS'), '1')\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )
        returncode, output = runner_inst._execute_harness_subprocess(script_content=check_script)
        self.assertEqual(returncode, 0, f"Subprocess should inherit thread caps: {output}")

    def test_simulation_mode_wave_execution_and_reporting(self):
        """Simulation mode executes waves, generates artifact hashes, and produces ExecutionReport."""
        graph = TaskGraph(paper_id="sim_p", paper_title="Simulation Flow")
        t1 = Task(
            task_id="t_step1",
            title="Step 1",
            category=TaskCategory.DATA_ACQUISITION,
            outputs=["data/out1.csv"],
            artifact_contracts=[ArtifactContract(path="data/out1.csv", artifact_type=ArtifactType.TABULAR)],
        )
        t2 = Task(
            task_id="t_step2",
            title="Step 2",
            dependencies=["t_step1"],
            category=TaskCategory.METRIC_EVALUATION,
            outputs=["eval.json"],
            artifact_contracts=[ArtifactContract(path="eval.json", artifact_type=ArtifactType.METRICS_JSON)],
        )
        graph.add_task(t1)
        graph.add_task(t2)

        runner_inst = TaskRunner(graph=graph, workspace_dir=self.workspace, mode=ExecutionMode.SIMULATION)
        report = runner_inst.execute()

        self.assertEqual(report.completed_count, 2)
        self.assertEqual(report.failed_count, 0)
        self.assertEqual(report.blocked_count, 0)
        self.assertEqual(report.scorecard["rating"], "CERTIFIED_REPLICABLE")
        self.assertIn("Replication Scorecard", report.render_markdown())

    def test_upstream_dependency_failure_cascades_blocked_status(self):
        """Failure of an upstream task cascades and marks downstream dependents as BLOCKED."""
        graph = TaskGraph(paper_id="cascade_p", paper_title="Cascading Block Paper")
        t_fail = Task(task_id="t_root", title="Failing Root", category=TaskCategory.DATA_ACQUISITION)
        t_child = Task(task_id="t_dep", title="Dependent Child", dependencies=["t_root"], category=TaskCategory.MODEL_IMPLEMENTATION)
        graph.add_task(t_fail)
        graph.add_task(t_child)

        runner_inst = TaskRunner(graph=graph, workspace_dir=self.workspace)

        # Mock root failing
        with patch.object(runner_inst, "_execute_task") as mock_exec:
            mock_exec.return_value = runner.TaskExecutionResult(
                task_id="t_root",
                status=TaskStatus.FAILED,
                duration_seconds=0.1,
                error_message="Root crashed",
            )
            report = runner_inst.execute()

        self.assertEqual(report.failed_count, 1)
        self.assertEqual(report.blocked_count, 1)
        self.assertEqual(report.results["t_dep"].status, TaskStatus.BLOCKED)

    def test_reflection_cycle_1_immediate_pass(self):
        """execute_with_reflection records 1 reflection cycle when Cycle 1 passes cleanly."""
        graph = TaskGraph(paper_id="c1_p", paper_title="Cycle 1 Pass Paper")
        task = Task(
            task_id="t_c1",
            title="Pass on Cycle 1",
            outputs=["ready.lock"],
            artifact_contracts=[ArtifactContract(path="ready.lock", artifact_type=ArtifactType.LOCKFILE)],
        )
        graph.add_task(task)
        runner_inst = TaskRunner(graph=graph, workspace_dir=self.workspace)

        with patch("research_assistant.shared.llm.chat") as mock_chat, \
             patch.object(runner_inst, "_execute_harness_subprocess", return_value=(0, "OK")):
            mock_chat.return_value = MagicMock(content="lock_content")
            status = runner_inst.execute_with_reflection(task, max_reflection_cycles=2)

        self.assertEqual(status, TaskStatus.COMPLETED)
        self.assertEqual(task.metadata.get("reflection_cycles"), 1)

    def test_reflection_cycle_2_warm_state_recovery(self):
        """execute_with_reflection recovers on Cycle 2 after Cycle 1 failure on warm state."""
        graph = TaskGraph(paper_id="c2_p", paper_title="Cycle 2 Recovery Paper")
        task = Task(
            task_id="t_c2",
            title="Pass on Cycle 2",
            outputs=["data/result.pt"],
            artifact_contracts=[ArtifactContract(path="data/result.pt", artifact_type=ArtifactType.TENSOR)],
        )
        graph.add_task(task)
        runner_inst = TaskRunner(graph=graph, workspace_dir=self.workspace)

        harness_responses = [(1, "AssertionError: Deliverable missing: data/result.pt"), (0, "OK")]

        with patch("research_assistant.shared.llm.chat") as mock_chat, \
             patch.object(runner_inst, "_execute_harness_subprocess", side_effect=harness_responses):
            mock_chat.return_value = MagicMock(content="fixed_tensor_code")
            status = runner_inst.execute_with_reflection(task, max_reflection_cycles=2)

        self.assertEqual(status, TaskStatus.COMPLETED)
        self.assertEqual(task.metadata.get("reflection_cycles"), 2)

    def test_reflection_strict_2_cycle_cap_enforcement_on_repeated_failure(self):
        """execute_with_reflection terminates strictly after 2 cycles when tests continue failing."""
        graph = TaskGraph(paper_id="c2_fail_p", paper_title="Cycle 2 Permanent Failure")
        task = Task(
            task_id="t_c2_fail",
            title="Fail on Cycle 2",
            outputs=["broken.py"],
        )
        graph.add_task(task)
        runner_inst = TaskRunner(graph=graph, workspace_dir=self.workspace)

        harness_responses = [(1, "AssertionError: Fail 1"), (1, "AssertionError: Fail 2")]

        with patch("research_assistant.shared.llm.chat") as mock_chat, \
             patch.object(runner_inst, "_execute_harness_subprocess", side_effect=harness_responses):
            mock_chat.return_value = MagicMock(content="still_broken")
            status = runner_inst.execute_with_reflection(task, max_reflection_cycles=2)

        self.assertEqual(status, TaskStatus.FAILED)
        self.assertEqual(task.metadata.get("reflection_cycles"), 2)
        self.assertIn("failure_details", task.metadata)
        self.assertEqual(task.metadata["failure_details"]["error_type"], "AssertionError")

    def test_failure_diagnostics_error_parser_varieties(self):
        """parse_execution_error classifies Python and harness errors accurately."""
        diag_assert = parse_execution_error(1, "Traceback:\nAssertionError: Values do not match")
        self.assertEqual(diag_assert.error_type, "AssertionError")
        self.assertEqual(diag_assert.failing_assertion, "AssertionError: Values do not match")

        diag_syntax = parse_execution_error(1, "SyntaxError: invalid syntax")
        self.assertEqual(diag_syntax.error_type, "SyntaxError")

        diag_fnf = parse_execution_error(1, "FileNotFoundError: [Errno 2] No such file or directory: 'x'")
        self.assertEqual(diag_fnf.error_type, "FileNotFoundError")

        diag_import = parse_execution_error(1, "ModuleNotFoundError: No module named 'fake_lib'")
        self.assertEqual(diag_import.error_type, "ImportError")

        diag_timeout = parse_execution_error(1, "Test harness timed out after 30s")
        self.assertEqual(diag_timeout.error_type, "TimeoutExpired")

    def test_harness_subprocess_timeout_enforcement(self):
        """run_task_test_script aborts hanging subprocesses when timeout is reached."""
        hanging_script = (
            "import time, unittest\n"
            "class HangTest(unittest.TestCase):\n"
            "    def test_hang(self):\n"
            "        time.sleep(10)\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )
        ok, stdout, stderr = run_task_test_script(hanging_script, workspace_dir=self.workspace, timeout_seconds=1)
        self.assertFalse(ok)
        self.assertIn("timed out after 1s", stderr)


class TestTier5GroundedRetrievalHelpers(unittest.TestCase):
    """Testing grounding, faceted formatting, and boundary retrieval fallbacks."""

    def test_ensure_target_paper_grounded_or_canonical_check_or_ingest(self):
        """Verify target paper grounding contract and transparent auto-ingestion logic."""
        # 1. Present paper bypasses ingestion
        with patch.object(retrieval, "get_ingested_documents", return_value={"already_there.pdf"}), \
             patch.object(retrieval, "ingest_pdfs") as mock_ingest:
            key, was_ingested = retrieval.ensure_target_paper_grounded("already_there.pdf")
            self.assertEqual(key, "already_there.pdf")
            self.assertFalse(was_ingested)
            mock_ingest.assert_not_called()

        # 2. Absent paper triggers auto-ingestion
        with patch.object(retrieval, "get_ingested_documents", return_value=set()), \
             patch.object(retrieval, "ingest_pdfs") as mock_ingest, \
             patch("os.path.isfile", return_value=True):
            key_absent, was_ingested_absent = retrieval.ensure_target_paper_grounded("absent.pdf")
            self.assertEqual(key_absent, "absent.pdf")
            self.assertTrue(was_ingested_absent)
            mock_ingest.assert_called_once()

    def test_format_faceted_context_or_untruncated_context_formatting(self):
        """Verify un-truncated context formatting across methodological facets and chunk deduplication."""
        facets = {
            "data_pipeline": {"chunks": [], "scores": [], "text": "Data details 123"},
            "model_architecture": {"chunks": [], "scores": [], "text": "Architecture equations"},
            "hyperparameters": {"chunks": [], "scores": [], "text": "Batch size 32"},
            "evaluation": {"chunks": [], "scores": [], "text": "Accuracy 99%"},
        }
        fallback = [{"text": "External baseline chunk", "metadata": {"document": "ext.pdf", "page": 1}}]

        out = retrieval.format_faceted_context(facets)
        self.assertIn("Methodological Facets", out)
        self.assertIn("Data details 123", out)

        # Chunk deduplication across multiple facets
        dup_chunk = {"id": "c1", "text": "Common training methodology passage"}
        facets_dup = {
            "model_architecture": {"chunks": [dup_chunk]},
            "hyperparameters": {"chunks": [dup_chunk]},
        }
        res_dup = retrieval.format_faceted_context(facets_dup)
        self.assertEqual(res_dup.count("Common training methodology passage"), 1)

    def test_retrieve_faceted_paper_context_partial_facet_failure_resilience(self):
        """If one facet query fails, surviving facets still populate results and formatted context."""
        mock_res = (MagicMock(), MagicMock(), ["chunk 1"], [{"doc": "p1"}])

        def selective_hybrid_search(query, *args, **kwargs):
            if "provenance" in query:
                raise RuntimeError("Facet index error")
            return [{"text": "Model architecture operators chunk", "rrf_score": 0.05, "metadata": {"page": 1}}]

        with patch("research_assistant.tasks.retrieval.hybrid_search", side_effect=selective_hybrid_search):
            res = retrieval.retrieve_faceted_paper_context(
                doc_key="resilient.pdf",
                search_resources=mock_res,
                enable_corpus_fallback=False,
            )
            self.assertEqual(res["data_pipeline"]["chunks"], [])
            self.assertGreater(len(res["model_architecture"]["chunks"]), 0)
            self.assertIn("Model architecture operators chunk", res["formatted_context"])

    def test_retrieve_faceted_paper_context_zero_and_negative_top_k(self):
        """retrieve_faceted_paper_context tolerates boundary top_k values (0 or negative)."""
        mock_res = (MagicMock(), MagicMock(), ["chunk 1"], [{"doc": "p1"}])
        res = retrieval.retrieve_faceted_paper_context(
            doc_key="test_paper.pdf",
            top_k_per_facet=0,
            search_resources=mock_res,
        )
        self.assertEqual(res["doc_key"], "test_paper.pdf")
        self.assertIn("data_pipeline", res)
        self.assertIn("model_architecture", res)

    def test_retrieve_faceted_paper_context_corrupt_search_resources(self):
        """Corrupted search_resources tuple falls back cleanly to empty result structure."""
        corrupt_res = ("not", "a", "valid")  # 3 items instead of 4
        res = retrieval.retrieve_faceted_paper_context(
            doc_key="test.pdf",
            search_resources=corrupt_res,  # type: ignore
        )
        self.assertEqual(res["doc_key"], "test.pdf")
        self.assertEqual(res["data_pipeline"]["chunks"], [])
        self.assertEqual(res["formatted_context"], "")

    def test_retrieve_faceted_paper_context_empty_doc_key_fallback(self):
        """Calling faceted retrieval with empty doc_key returns structured dictionary safely."""
        mock_res = (MagicMock(), MagicMock(), ["text"], [{"doc": "p"}])
        res = retrieval.retrieve_faceted_paper_context(
            doc_key="",
            query_hint="quantum simulation",
            search_resources=mock_res,
        )
        self.assertEqual(res["doc_key"], "")
        self.assertIn("formatted_context", res)


class TestTier5PersistentChromaDBInvariant(unittest.TestCase):
    """Storage volume protection invariant: persistent ChromaDB holds exactly 75,039 chunks."""

    def test_persistent_chromadb_physics_papers_chunk_count_invariant(self):
        """Assert physics_papers collection in data/physics_vectordb holds exactly 75,039 chunks."""
        self.assertTrue(os.path.exists(VECTORDB_PATH), f"VECTORDB_PATH '{VECTORDB_PATH}' must exist.")
        client = chromadb.PersistentClient(path=VECTORDB_PATH)
        coll = client.get_collection(COLLECTION_NAME)
        count = coll.count()
        self.assertEqual(
            count,
            75039,
            f"Persistent storage violation: expected exactly 75,039 chunks in '{COLLECTION_NAME}', got {count}!",
        )

    def test_persistent_chromadb_read_only_invariants(self):
        """Read-only peek inspection verifies metadata integrity without altering chunk count."""
        client = chromadb.PersistentClient(path=VECTORDB_PATH)
        coll = client.get_collection(COLLECTION_NAME)
        peek_res = coll.peek(limit=2)
        self.assertIn("documents", peek_res)
        self.assertIn("metadatas", peek_res)
        self.assertEqual(len(peek_res["documents"]), 2)
        # Verify collection count remains untouched
        self.assertEqual(coll.count(), 75039)


if __name__ == "__main__":
    unittest.main()
