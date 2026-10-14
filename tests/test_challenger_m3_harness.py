"""
Empirical Challenger Test Suite for Milestone 3 (M3).
Author: Empirical Challenger 1 (teamwork_preview_challenger_m3_1)

Adversarially stress-tests:
1. Valid artifacts passing generated test scripts:
   - Valid tensors (.npy, .pt) with matching shapes and dtypes.
   - Valid tabular (.csv, .parquet) with matching columns and types.
   - Valid metrics.json with all required keys and numeric finite values.
   - Valid python code passes syntax compilation.
2. Invalid/corrupted artifacts failing generated test scripts:
   - Mismatched tensor shapes (e.g. expected [10, 64] got [10, 32], rank mismatch).
   - Mismatched tensor dtypes (e.g. expected float32 got int64).
   - Missing tabular columns (.csv and .parquet).
   - Missing JSON metric keys, non-finite values (NaN, Inf), bools, non-numeric strings.
   - Python code with syntax errors fails compilation check.
   - Missing deliverable files fail existence check.
   - Empty (0-byte) output files fail non-emptiness check.
3. Subagent prompt formatting in roles.py:
   - Prompt contains schemas, artifact contracts, and generated test harness script.
   - Verification command and upstream deliverables formatting.
4. ChromaDB persistent storage invariant:
   - Collection 'physics_papers' in 'data/physics_vectordb' contains exactly 75,039 chunks.
"""

from __future__ import annotations

import ast
import json
import math
import os
import shutil
import tempfile
import unittest

try:
    import chromadb
except ImportError:  # CI installs requirements-test.txt only
    chromadb = None

from tests.optional_deps import requires_real_store, requires_torch
import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # CI installs requirements-test.txt only
    torch = None

from research_assistant.config import (
    COLLECTION_NAME,
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
)
from research_assistant.tasks.roles import (
    format_agent_prompt,
    format_subagent_prompt,
    get_role_spec,
)

# Under CPU-safe execution limits, give subprocesses up to 60s to load torch/pandas
SUBPROCESS_TIMEOUT = 60


class TestChallengerValidArtifactsPassing(unittest.TestCase):
    """Stress-tests that valid artifacts strictly pass generated test scripts."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="challenger_valid_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_valid_numpy_tensor_passes(self):
        """Verifies valid .npy tensor matching expected shape and dtype passes."""
        arr = np.ones((10, 64), dtype=np.float32)
        out_rel = "tensors/latent_embeddings.npy"
        out_abs = os.path.join(self.tmp_dir, out_rel)
        os.makedirs(os.path.dirname(out_abs), exist_ok=True)
        np.save(out_abs, arr)

        task = Task(
            task_id="task_embed_valid",
            title="Generate Latent Embeddings",
            category=TaskCategory.DATA_PREPROCESSING,
            outputs=[out_rel],
            artifact_contracts=[
                ArtifactContract(
                    path=out_rel,
                    artifact_type=ArtifactType.TENSOR,
                    shape=[10, 64],
                    dtype="float32",
                )
            ],
        )

        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir, timeout_seconds=SUBPROCESS_TIMEOUT)
        self.assertTrue(ok, f"Valid numpy tensor failed verification: {stderr}")

    @requires_torch
    def test_valid_torch_tensor_passes(self):
        """Verifies valid .pt PyTorch tensor matching expected shape and dtype passes."""
        tensor = torch.randn(16, 32, dtype=torch.float32)
        out_rel = "weights/checkpoint.pt"
        out_abs = os.path.join(self.tmp_dir, out_rel)
        os.makedirs(os.path.dirname(out_abs), exist_ok=True)
        torch.save(tensor, out_abs)

        task = Task(
            task_id="task_torch_valid",
            title="Save PyTorch Weights",
            category=TaskCategory.TRAINING_OR_SIMULATION,
            outputs=[out_rel],
            artifact_contracts=[
                ArtifactContract(
                    path=out_rel,
                    artifact_type=ArtifactType.TENSOR,
                    shape=[16, 32],
                    dtype="float32",
                )
            ],
        )

        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir, timeout_seconds=SUBPROCESS_TIMEOUT)
        self.assertTrue(ok, f"Valid torch tensor failed verification: {stderr}")

    def test_valid_csv_tabular_passes(self):
        """Verifies valid CSV with expected columns and dtypes passes."""
        out_rel = "tables/dataset.csv"
        out_abs = os.path.join(self.tmp_dir, out_rel)
        os.makedirs(os.path.dirname(out_abs), exist_ok=True)

        df = pd.DataFrame({
            "sample_id": ["s1", "s2", "s3"],
            "measurement": [1.25, 2.35, 3.45],
            "count": [10, 20, 30],
        })
        df.to_csv(out_abs, index=False)

        task = Task(
            task_id="task_csv_valid",
            title="Export CSV Dataset",
            category=TaskCategory.DATA_ACQUISITION,
            outputs=[out_rel],
            artifact_contracts=[
                ArtifactContract(
                    path=out_rel,
                    artifact_type=ArtifactType.TABULAR,
                    columns={"sample_id": "str", "measurement": "float", "count": "int"},
                )
            ],
        )

        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir, timeout_seconds=SUBPROCESS_TIMEOUT)
        self.assertTrue(ok, f"Valid CSV failed verification: {stderr}")

    def test_valid_parquet_tabular_passes(self):
        """Verifies valid Parquet with expected columns and dtypes passes."""
        out_rel = "data/features.parquet"
        out_abs = os.path.join(self.tmp_dir, out_rel)
        os.makedirs(os.path.dirname(out_abs), exist_ok=True)

        df = pd.DataFrame({
            "feature_id": [101, 102, 103],
            "embedding_norm": [0.42, 0.77, 0.99],
        })
        df.to_parquet(out_abs, index=False)

        task = Task(
            task_id="task_parquet_valid",
            title="Export Parquet Features",
            category=TaskCategory.DATA_PREPROCESSING,
            outputs=[out_rel],
            artifact_contracts=[
                ArtifactContract(
                    path=out_rel,
                    artifact_type=ArtifactType.TABULAR,
                    columns={"feature_id": "int", "embedding_norm": "float"},
                )
            ],
        )

        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir, timeout_seconds=SUBPROCESS_TIMEOUT)
        self.assertTrue(ok, f"Valid Parquet failed verification: {stderr}")

    def test_valid_metrics_json_numeric_finite_passes(self):
        """Verifies valid JSON metric file with required keys and finite numbers passes."""
        out_rel = "metrics/results.json"
        out_abs = os.path.join(self.tmp_dir, out_rel)
        os.makedirs(os.path.dirname(out_abs), exist_ok=True)

        metrics_payload = {
            "loss": 0.042,
            "accuracy": 0.985,
            "f1": 0.978,
            "latency_ms": 12.4,
            "epochs_trained": 50,
        }
        with open(out_abs, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f)

        task = Task(
            task_id="task_metrics_valid",
            title="Evaluate Metrics",
            category=TaskCategory.METRIC_EVALUATION,
            outputs=[out_rel],
            artifact_contracts=[
                ArtifactContract(
                    path=out_rel,
                    artifact_type=ArtifactType.METRICS_JSON,
                    required_keys=["loss", "accuracy", "f1"],
                )
            ],
        )

        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir, timeout_seconds=SUBPROCESS_TIMEOUT)
        self.assertTrue(ok, f"Valid JSON metrics failed verification: {stderr}")

    def test_valid_python_code_passes(self):
        """Verifies valid Python source file with function and classes passes."""
        out_rel = "src/model.py"
        out_abs = os.path.join(self.tmp_dir, out_rel)
        os.makedirs(os.path.dirname(out_abs), exist_ok=True)

        py_content = (
            "import math\n\n"
            "class BaselineModel:\n"
            "    def __init__(self, hidden_dim: int = 128):\n"
            "        self.dim = hidden_dim\n\n"
            "    def predict(self, val: float) -> float:\n"
            "        return math.tanh(val) * self.dim\n"
        )
        with open(out_abs, "w", encoding="utf-8") as f:
            f.write(py_content)

        task = Task(
            task_id="task_code_valid",
            title="Implement Baseline Model",
            category=TaskCategory.MODEL_IMPLEMENTATION,
            outputs=[out_rel],
            artifact_contracts=[
                ArtifactContract(
                    path=out_rel,
                    artifact_type=ArtifactType.CODE,
                )
            ],
        )

        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir, timeout_seconds=SUBPROCESS_TIMEOUT)
        self.assertTrue(ok, f"Valid Python code failed verification: {stderr}")


class TestChallengerInvalidArtifactsFailing(unittest.TestCase):
    """Stress-tests that corrupted, invalid, or mismatched artifacts fail generated test scripts."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="challenger_invalid_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_tensor_shape_dimension_mismatch_fails(self):
        """Asserts tensor shape dimension mismatch (expected [10, 64] got [10, 32]) fails."""
        arr = np.zeros((10, 32), dtype=np.float32)
        out_rel = "mismatch_dim.npy"
        np.save(os.path.join(self.tmp_dir, out_rel), arr)

        task = Task(
            task_id="t_shape_mismatch",
            title="Check Shape",
            outputs=[out_rel],
            artifact_contracts=[
                ArtifactContract(
                    path=out_rel,
                    artifact_type=ArtifactType.TENSOR,
                    shape=[10, 64],
                    dtype="float32",
                )
            ],
        )

        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir, timeout_seconds=SUBPROCESS_TIMEOUT)
        self.assertFalse(ok, "Mismatched tensor shape should have failed verification")
        self.assertIn("Tensor dim 1 mismatch", stderr)

    @requires_torch
    def test_tensor_rank_mismatch_fails(self):
        """Asserts tensor rank mismatch (expected 3D [10, 64, 4] got 2D [10, 64]) fails."""
        tensor = torch.zeros(10, 64, dtype=torch.float32)
        out_rel = "mismatch_rank.pt"
        torch.save(tensor, os.path.join(self.tmp_dir, out_rel))

        task = Task(
            task_id="t_rank_mismatch",
            title="Check Rank",
            outputs=[out_rel],
            artifact_contracts=[
                ArtifactContract(
                    path=out_rel,
                    artifact_type=ArtifactType.TENSOR,
                    shape=[10, 64, 4],
                    dtype="float32",
                )
            ],
        )

        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir, timeout_seconds=SUBPROCESS_TIMEOUT)
        self.assertFalse(ok, "Mismatched tensor rank should have failed verification")
        self.assertIn("Tensor rank mismatch", stderr)

    def test_tensor_dtype_mismatch_fails(self):
        """Asserts tensor dtype mismatch (expected float32 got int64) fails."""
        arr = np.ones((10, 64), dtype=np.int64)
        out_rel = "mismatch_dtype.npy"
        np.save(os.path.join(self.tmp_dir, out_rel), arr)

        task = Task(
            task_id="t_dtype_mismatch",
            title="Check Dtype",
            outputs=[out_rel],
            artifact_contracts=[
                ArtifactContract(
                    path=out_rel,
                    artifact_type=ArtifactType.TENSOR,
                    shape=[10, 64],
                    dtype="float32",
                )
            ],
        )

        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir, timeout_seconds=SUBPROCESS_TIMEOUT)
        self.assertFalse(ok, "Mismatched tensor dtype should have failed verification")
        self.assertIn("Tensor dtype mismatch", stderr)

    @requires_torch
    def test_tensor_torch_dtype_mismatch_fails(self):
        """Asserts PyTorch tensor dtype mismatch (expected float32 got int32) fails."""
        tensor = torch.ones(8, 16, dtype=torch.int32)
        out_rel = "mismatch_torch_dtype.pt"
        torch.save(tensor, os.path.join(self.tmp_dir, out_rel))

        task = Task(
            task_id="t_torch_dtype_mismatch",
            title="Check Torch Dtype",
            outputs=[out_rel],
            artifact_contracts=[
                ArtifactContract(
                    path=out_rel,
                    artifact_type=ArtifactType.TENSOR,
                    shape=[8, 16],
                    dtype="float32",
                )
            ],
        )

        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir, timeout_seconds=SUBPROCESS_TIMEOUT)
        self.assertFalse(ok, "Mismatched torch dtype should have failed verification")
        self.assertIn("Tensor dtype mismatch", stderr)

    def test_csv_missing_column_fails(self):
        """Asserts CSV missing required column fails verification."""
        out_rel = "missing_col.csv"
        df = pd.DataFrame({"sample_id": ["s1"], "feature_a": [1.0]})
        df.to_csv(os.path.join(self.tmp_dir, out_rel), index=False)

        task = Task(
            task_id="t_csv_missing_col",
            title="Check CSV Columns",
            outputs=[out_rel],
            artifact_contracts=[
                ArtifactContract(
                    path=out_rel,
                    artifact_type=ArtifactType.TABULAR,
                    columns={"sample_id": "str", "feature_a": "float", "target_label": "str"},
                )
            ],
        )

        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir, timeout_seconds=SUBPROCESS_TIMEOUT)
        self.assertFalse(ok, "CSV missing column should have failed verification")
        self.assertIn("Expected column missing: target_label", stderr)

    def test_parquet_missing_column_fails(self):
        """Asserts Parquet missing required column fails verification."""
        out_rel = "missing_col.parquet"
        df = pd.DataFrame({"idx": [1, 2], "score": [0.5, 0.9]})
        df.to_parquet(os.path.join(self.tmp_dir, out_rel), index=False)

        task = Task(
            task_id="t_parquet_missing_col",
            title="Check Parquet Columns",
            outputs=[out_rel],
            artifact_contracts=[
                ArtifactContract(
                    path=out_rel,
                    artifact_type=ArtifactType.TABULAR,
                    columns={"idx": "int", "score": "float", "rank": "int"},
                )
            ],
        )

        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir, timeout_seconds=SUBPROCESS_TIMEOUT)
        self.assertFalse(ok, "Parquet missing column should have failed verification")
        self.assertIn("Expected column missing: rank", stderr)

    def test_metrics_json_missing_keys_fails(self):
        """Asserts JSON metric deliverable missing required keys fails verification."""
        out_rel = "partial_metrics.json"
        with open(os.path.join(self.tmp_dir, out_rel), "w") as f:
            json.dump({"loss": 0.12}, f)

        task = Task(
            task_id="t_missing_keys",
            title="Check Metric Keys",
            outputs=[out_rel],
            artifact_contracts=[
                ArtifactContract(
                    path=out_rel,
                    artifact_type=ArtifactType.METRICS_JSON,
                    required_keys=["loss", "eval_accuracy", "f1"],
                )
            ],
        )

        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir, timeout_seconds=SUBPROCESS_TIMEOUT)
        self.assertFalse(ok, "JSON metrics missing required keys should have failed")
        self.assertIn("Required key missing in JSON", stderr)

    def test_metrics_json_nan_value_fails(self):
        """Asserts JSON metric deliverable containing NaN value fails verification."""
        out_rel = "nan_metrics.json"
        with open(os.path.join(self.tmp_dir, out_rel), "w") as f:
            f.write('{"loss": NaN, "accuracy": 0.95}')

        task = Task(
            task_id="t_nan_metrics",
            title="Check NaN Metric",
            outputs=[out_rel],
            artifact_contracts=[
                ArtifactContract(
                    path=out_rel,
                    artifact_type=ArtifactType.METRICS_JSON,
                    required_keys=["loss", "accuracy"],
                )
            ],
        )

        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir, timeout_seconds=SUBPROCESS_TIMEOUT)
        self.assertFalse(ok, "JSON metrics with NaN value should have failed verification")
        self.assertIn("must be finite", stderr)

    def test_metrics_json_inf_value_fails(self):
        """Asserts JSON metric deliverable containing Infinity value fails verification."""
        out_rel = "inf_metrics.json"
        with open(os.path.join(self.tmp_dir, out_rel), "w") as f:
            f.write('{"loss": Infinity, "accuracy": 0.95}')

        task = Task(
            task_id="t_inf_metrics",
            title="Check Inf Metric",
            outputs=[out_rel],
            artifact_contracts=[
                ArtifactContract(
                    path=out_rel,
                    artifact_type=ArtifactType.METRICS_JSON,
                    required_keys=["loss", "accuracy"],
                )
            ],
        )

        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir, timeout_seconds=SUBPROCESS_TIMEOUT)
        self.assertFalse(ok, "JSON metrics with Infinity value should have failed verification")
        self.assertIn("must be finite", stderr)

    def test_metrics_json_bool_value_fails(self):
        """Asserts JSON metric deliverable containing boolean for metric fails verification."""
        out_rel = "bool_metrics.json"
        with open(os.path.join(self.tmp_dir, out_rel), "w") as f:
            json.dump({"loss": False, "accuracy": True}, f)

        task = Task(
            task_id="t_bool_metrics",
            title="Check Bool Metric",
            outputs=[out_rel],
            artifact_contracts=[
                ArtifactContract(
                    path=out_rel,
                    artifact_type=ArtifactType.METRICS_JSON,
                    required_keys=["loss"],
                )
            ],
        )

        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir, timeout_seconds=SUBPROCESS_TIMEOUT)
        self.assertFalse(ok, "JSON metrics with bool value should have failed verification")
        self.assertIn("cannot be bool", stderr)

    def test_metrics_json_non_numeric_string_fails(self):
        """Asserts JSON metric deliverable containing non-numeric string fails verification."""
        out_rel = "str_metrics.json"
        with open(os.path.join(self.tmp_dir, out_rel), "w") as f:
            json.dump({"loss": "not_a_number"}, f)

        task = Task(
            task_id="t_str_metrics",
            title="Check Non-numeric String Metric",
            outputs=[out_rel],
            artifact_contracts=[
                ArtifactContract(
                    path=out_rel,
                    artifact_type=ArtifactType.METRICS_JSON,
                    required_keys=["loss"],
                )
            ],
        )

        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir, timeout_seconds=SUBPROCESS_TIMEOUT)
        self.assertFalse(ok, "JSON metrics with non-numeric string should have failed")
        self.assertIn("must be numeric", stderr)

    def test_python_code_syntax_error_fails(self):
        """Asserts Python code containing syntax error fails compilation check."""
        out_rel = "bad_code.py"
        with open(os.path.join(self.tmp_dir, out_rel), "w") as f:
            f.write("def broken_syntax(:\n    pass\n")

        task = Task(
            task_id="t_bad_code",
            title="Check Syntax Error",
            outputs=[out_rel],
            artifact_contracts=[
                ArtifactContract(
                    path=out_rel,
                    artifact_type=ArtifactType.CODE,
                )
            ],
        )

        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir, timeout_seconds=SUBPROCESS_TIMEOUT)
        self.assertFalse(ok, "Python code with syntax error should have failed verification")
        self.assertIn("Python syntax error", stderr)

    def test_missing_deliverable_file_fails(self):
        """Asserts non-existent deliverable file fails existence check."""
        out_rel = "does_not_exist.bin"
        task = Task(
            task_id="t_missing_file",
            title="Check Missing File",
            outputs=[out_rel],
        )

        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir, timeout_seconds=SUBPROCESS_TIMEOUT)
        self.assertFalse(ok, "Missing deliverable should have failed verification")
        self.assertIn("Deliverable missing", stderr)

    def test_empty_zero_byte_file_fails(self):
        """Asserts empty (0-byte) deliverable file fails non-emptiness check."""
        out_rel = "empty_artifact.txt"
        with open(os.path.join(self.tmp_dir, out_rel), "w") as f:
            pass  # 0 bytes

        task = Task(
            task_id="t_empty_file",
            title="Check Empty File",
            outputs=[out_rel],
        )

        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir, timeout_seconds=SUBPROCESS_TIMEOUT)
        self.assertFalse(ok, "Empty (0-byte) deliverable should have failed verification")
        self.assertIn("Deliverable empty", stderr)


class TestChallengerRolesPromptFormatting(unittest.TestCase):
    """Stress-tests subagent prompt formatting in roles.py."""

    def test_prompt_contains_schemas_artifact_contracts_and_harness(self):
        """Verifies subagent prompt contains schemas, artifact contracts, and test harness."""
        task = Task(
            task_id="task_deconstruct_01",
            title="Train Physics Invariant Network",
            category=TaskCategory.TRAINING_OR_SIMULATION,
            assigned_role=AgentRole.EXPERIMENT_RUNNER,
            description="Train network enforcing energy conservation invariant.",
            inputs=["data/train.parquet", "src/network.py"],
            outputs=["checkpoints/best_model.pt", "eval/metrics.json"],
            acceptance_criteria=["Exit code 0", "Loss < 0.05", "Non-empty model weights"],
            paper_context_excerpts=["Equations (3)-(5) define the Hamiltonian neural constraint."],
            artifact_contracts=[
                ArtifactContract(
                    path="checkpoints/best_model.pt",
                    artifact_type=ArtifactType.TENSOR,
                    shape=[128, 64],
                    dtype="float32",
                    description="Trained model weights",
                ),
                ArtifactContract(
                    path="eval/metrics.json",
                    artifact_type=ArtifactType.METRICS_JSON,
                    required_keys=["final_loss", "energy_violation_rate"],
                    description="Final evaluation metrics",
                ),
            ],
        )

        prompt = format_subagent_prompt(task, workspace_dir="/workspace/replication")

        # 1. Assert header and task details
        self.assertIn("# Scientific Replication Task: Train Physics Invariant Network", prompt)
        self.assertIn("**Task ID**: `task_deconstruct_01`", prompt)
        self.assertIn("**Assigned Role**: Experiment & Simulation Runner (experiment_runner)", prompt)
        self.assertIn("**Workspace Directory**: `/workspace/replication`", prompt)

        # 2. Assert Inputs and Outputs
        self.assertIn("data/train.parquet, src/network.py", prompt)
        self.assertIn("checkpoints/best_model.pt, eval/metrics.json", prompt)

        # 3. Assert Formal Contracts Section
        self.assertIn("## Formal Deliverable Artifact Contracts & Schemas", prompt)
        self.assertIn("checkpoints/best_model.pt", prompt)
        self.assertIn("type: `tensor`", prompt)
        self.assertIn("shape: `[128, 64]`", prompt)
        self.assertIn("dtype: `float32`", prompt)
        self.assertIn("eval/metrics.json", prompt)
        self.assertIn("type: `json`", prompt)
        self.assertIn("required_keys: `['final_loss', 'energy_violation_rate']`", prompt)

        # 4. Assert Acceptance Criteria
        self.assertIn("- Exit code 0", prompt)
        self.assertIn("- Loss < 0.05", prompt)

        # 5. Assert Automated Acceptance Test Harness
        self.assertIn("## Automated Acceptance Test Harness", prompt)
        self.assertIn("class TestTaskAcceptance_task_deconstruct_01(unittest.TestCase):", prompt)
        self.assertIn("**Verification Command**:", prompt)
        self.assertIn("python -m unittest test_task_deconstruct_01.py", prompt)

    def test_prompt_formatting_with_upstream_artifacts(self):
        """Verifies upstream deliverables section rendered when upstream_artifacts is passed."""
        task = Task(
            task_id="task_eval_upstream",
            title="Evaluate Upstream Model",
            category=TaskCategory.METRIC_EVALUATION,
            assigned_role=AgentRole.EVALUATION_AGENT,
            inputs=["weights/model.pt"],
            outputs=["eval/report.json"],
        )
        upstream = {
            "weights/model.pt": "verified PyTorch tensor with shape [64, 32], dtype float32",
        }
        prompt = format_subagent_prompt(task, upstream_artifacts=upstream)

        self.assertIn("## Verified Upstream Deliverables on Disk", prompt)
        self.assertIn("- `weights/model.pt`: verified PyTorch tensor with shape [64, 32], dtype float32", prompt)

    def test_prompt_alias_and_initial_task_formatting(self):
        """Verifies format_agent_prompt alias and root task prerequisite formatting."""
        self.assertIs(format_agent_prompt, format_subagent_prompt)

        root_task = Task(
            task_id="task_root_env",
            title="Setup Python Environment",
            category=TaskCategory.ENVIRONMENT_SETUP,
            assigned_role=AgentRole.DEVOPS_AGENT,
            inputs=[],
            outputs=["env_ready.lock"],
        )
        prompt = format_agent_prompt(root_task)
        self.assertIn("None (Initial task)", prompt)


@requires_real_store
class TestChallengerChromaDBPersistentStorageInvariant(unittest.TestCase):
    """Stress-tests ChromaDB persistent storage invariant."""

    def test_physics_vectordb_exact_chunk_count_75039(self):
        """Asserts persistent collection 'physics_papers' contains exactly 75,039 chunks."""
        self.assertTrue(os.path.exists(VECTORDB_PATH), f"Vector DB path does not exist: {VECTORDB_PATH}")
        client = chromadb.PersistentClient(path=VECTORDB_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()

        self.assertEqual(
            count,
            75039,
            f"INVARIANT VIOLATION: Expected 75,039 chunks in ChromaDB '{COLLECTION_NAME}', got {count}!",
        )

    def test_physics_vectordb_schema_integrity(self):
        """Asserts collection schema has 768-dim embeddings and valid metadata."""
        client = chromadb.PersistentClient(path=VECTORDB_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        sample = collection.peek(3)

        self.assertEqual(len(sample["ids"]), 3)
        self.assertEqual(len(sample["embeddings"]), 3)
        self.assertEqual(len(sample["metadatas"]), 3)
        self.assertEqual(len(sample["documents"]), 3)

        # Verify embedding dimension
        emb_dim = len(sample["embeddings"][0])
        self.assertEqual(
            emb_dim,
            768,
            f"Embedding dimension mismatch: expected 768, got {emb_dim}",
        )

        # Verify documents and metadatas are non-empty
        for i in range(3):
            self.assertGreater(len(sample["documents"][i]), 0)
            self.assertIn("document", sample["metadatas"][i])


if __name__ == "__main__":
    unittest.main()
