"""
Unit tests for executable test harness generation and schema contract assertion.
"""

import ast
import json
import math
import os
import shutil
import tempfile
import unittest

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


class TestTaskHarness(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_generate_and_run_basic_harness_pass(self):
        task = Task(
            task_id="t_env",
            title="Setup Env",
            category=TaskCategory.ENVIRONMENT_SETUP,
            outputs=["env.lock"],
        )
        # Create output file
        with open(os.path.join(self.tmp_dir, "env.lock"), "w") as f:
            f.write("python=3.13\n")

        script = generate_task_test_script(task, self.tmp_dir)
        self.assertIn("TestTaskAcceptance_t_env", script)
        self.assertIn("env.lock", script)

        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir)
        self.assertTrue(ok, f"Harness failed unexpectedly: {stderr}")

    def test_harness_fails_when_deliverable_missing(self):
        task = Task(
            task_id="t_missing",
            title="Missing File Task",
            category=TaskCategory.DATA_ACQUISITION,
            outputs=["data/raw.csv"],
        )
        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir)
        self.assertFalse(ok)
        self.assertIn("Deliverable missing", stderr)

    def test_harness_validates_json_metrics_contract(self):
        task = Task(
            task_id="t_eval",
            title="Eval Task",
            category=TaskCategory.METRIC_EVALUATION,
            outputs=["metrics.json"],
            artifact_contracts=[
                ArtifactContract(
                    path="metrics.json",
                    artifact_type=ArtifactType.METRICS_JSON,
                    required_keys=["accuracy", "f1"],
                )
            ],
        )
        # Write valid JSON with required keys
        with open(os.path.join(self.tmp_dir, "metrics.json"), "w") as f:
            f.write('{"accuracy": 0.95, "f1": 0.94}')

        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir)
        self.assertTrue(ok, f"Harness should have passed: {stderr}")

        # Invalidate JSON key
        with open(os.path.join(self.tmp_dir, "metrics.json"), "w") as f:
            f.write('{"accuracy": 0.95}')  # missing f1

        ok_bad, _, stderr_bad = run_task_test_script(script, self.tmp_dir)
        self.assertFalse(ok_bad)
        self.assertIn("Required key missing in JSON", stderr_bad)

    def test_harness_validates_python_code_syntax(self):
        task = Task(
            task_id="t_code",
            title="Model Code",
            category=TaskCategory.MODEL_IMPLEMENTATION,
            outputs=["model.py"],
            artifact_contracts=[
                ArtifactContract(
                    path="model.py",
                    artifact_type=ArtifactType.CODE,
                )
            ],
        )
        # Write invalid Python
        with open(os.path.join(self.tmp_dir, "model.py"), "w") as f:
            f.write("def broken_syntax(:\n")

        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir)
        self.assertFalse(ok)
        self.assertIn("Python syntax error", stderr)

        # Fix Python code
        with open(os.path.join(self.tmp_dir, "model.py"), "w") as f:
            f.write("def valid_syntax():\n    return 42\n")

        ok_fixed, _, stderr_fixed = run_task_test_script(script, self.tmp_dir)
        self.assertTrue(ok_fixed, f"Valid code should pass: {stderr_fixed}")

    def test_harness_validates_tabular_columns(self):
        task = Task(
            task_id="t_csv",
            title="Data CSV",
            category=TaskCategory.DATA_PREPROCESSING,
            outputs=["dataset.csv"],
            artifact_contracts=[
                ArtifactContract(
                    path="dataset.csv",
                    artifact_type=ArtifactType.TABULAR,
                    columns={"sample_id": "str", "measurement": "float"},
                )
            ],
        )
        with open(os.path.join(self.tmp_dir, "dataset.csv"), "w") as f:
            f.write("sample_id,measurement,label\n1,0.5,active\n")

        script = generate_task_test_script(task, self.tmp_dir)
        ok, stdout, stderr = run_task_test_script(script, self.tmp_dir)
        self.assertTrue(ok, f"Tabular contract should pass: {stderr}")

    def test_harness_generates_valid_python_ast(self):
        """Verifies generate_task_test_script produces syntactically valid Python code."""
        task = Task(
            task_id="t_all_contracts",
            title="Comprehensive Task",
            category=TaskCategory.MODEL_IMPLEMENTATION,
            outputs=["model.py", "eval.json", "data.csv", "weights.pt", "cli.py"],
            acceptance_criteria=["Exit code 0", "Non-empty outputs"],
            artifact_contracts=[
                ArtifactContract(path="model.py", artifact_type=ArtifactType.CODE),
                ArtifactContract(path="eval.json", artifact_type=ArtifactType.METRICS_JSON, required_keys=["acc"]),
                ArtifactContract(path="data.csv", artifact_type=ArtifactType.TABULAR, columns={"id": "str"}),
                ArtifactContract(path="weights.pt", artifact_type=ArtifactType.TENSOR, shape=[10, 20], dtype="float32"),
                ArtifactContract(path="cli.py", artifact_type=ArtifactType.CLI_INTERFACE, cli_args=["--lr"]),
            ],
        )
        script = generate_task_test_script(task, self.tmp_dir)
        parsed = ast.parse(script)
        self.assertIsNotNone(parsed)

    def test_harness_validates_numpy_tensor_shape_and_dtype(self):
        """Verifies tensor shape and dtype assertions on NumPy deliverables."""
        import numpy as np

        tensor_path = os.path.join(self.tmp_dir, "features.npy")
        arr = np.zeros((4, 8), dtype=np.float32)
        np.save(tensor_path, arr)

        # Matching contract
        task = Task(
            task_id="t_numpy",
            title="NumPy Tensor Task",
            outputs=["features.npy"],
            artifact_contracts=[
                ArtifactContract(
                    path="features.npy",
                    artifact_type=ArtifactType.TENSOR,
                    shape=[4, 8],
                    dtype="float32",
                )
            ],
        )
        script = generate_task_test_script(task, self.tmp_dir)
        ok, _, stderr = run_task_test_script(script, self.tmp_dir)
        self.assertTrue(ok, f"Expected pass, got: {stderr}")

        # Shape mismatch
        task_bad_shape = Task(
            task_id="t_numpy",
            title="NumPy Tensor Task",
            outputs=["features.npy"],
            artifact_contracts=[
                ArtifactContract(
                    path="features.npy",
                    artifact_type=ArtifactType.TENSOR,
                    shape=[4, 10],
                    dtype="float32",
                )
            ],
        )
        script_bad = generate_task_test_script(task_bad_shape, self.tmp_dir)
        ok_bad, _, stderr_bad = run_task_test_script(script_bad, self.tmp_dir)
        self.assertFalse(ok_bad)
        self.assertIn("Tensor dim 1 mismatch", stderr_bad)

    def test_harness_validates_torch_tensor(self):
        """Verifies PyTorch tensor loading, rank, dimension, and dtype checking."""
        import torch

        os.makedirs(os.path.join(self.tmp_dir, "checkpoints"), exist_ok=True)
        pt_path = os.path.join(self.tmp_dir, "checkpoints/model.pt")
        tensor = torch.randn(2, 5, dtype=torch.float32)
        torch.save(tensor, pt_path)

        task = Task(
            task_id="t_torch",
            title="PyTorch Model Weights",
            outputs=["checkpoints/model.pt"],
            artifact_contracts=[
                ArtifactContract(
                    path="checkpoints/model.pt",
                    artifact_type=ArtifactType.TENSOR,
                    shape=[2, 5],
                    dtype="float32",
                )
            ],
        )
        script = generate_task_test_script(task, self.tmp_dir)
        ok, _, stderr = run_task_test_script(script, self.tmp_dir)
        self.assertTrue(ok, f"Expected PyTorch tensor pass, got: {stderr}")

    def test_harness_validates_safetensors_tensor(self):
        """Verifies Safetensors tensor contract verification."""
        try:
            from safetensors.numpy import save_file
            import numpy as np
        except ImportError:
            self.skipTest("safetensors not installed")

        st_path = os.path.join(self.tmp_dir, "model.safetensors")
        save_file({"weight": np.zeros((3, 6), dtype=np.float32)}, st_path)

        task = Task(
            task_id="t_st",
            title="Safetensors Weights",
            outputs=["model.safetensors"],
            artifact_contracts=[
                ArtifactContract(
                    path="model.safetensors",
                    artifact_type=ArtifactType.TENSOR,
                    shape=[3, 6],
                    dtype="float32",
                )
            ],
        )
        script = generate_task_test_script(task, self.tmp_dir)
        ok, _, stderr = run_task_test_script(script, self.tmp_dir)
        self.assertTrue(ok, f"Expected Safetensors pass, got: {stderr}")

    def test_harness_validates_tabular_parquet(self):
        """Verifies Parquet tabular contract verification."""
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")

        df = pd.DataFrame({"id": [1, 2, 3], "score": [0.1, 0.2, 0.3]})
        parquet_path = os.path.join(self.tmp_dir, "data.parquet")
        df.to_parquet(parquet_path)

        task = Task(
            task_id="t_parquet",
            title="Parquet Data",
            outputs=["data.parquet"],
            artifact_contracts=[
                ArtifactContract(
                    path="data.parquet",
                    artifact_type=ArtifactType.TABULAR,
                    columns={"id": "int", "score": "float"},
                )
            ],
        )
        script = generate_task_test_script(task, self.tmp_dir)
        ok, _, stderr = run_task_test_script(script, self.tmp_dir)
        self.assertTrue(ok, f"Expected Parquet pass, got: {stderr}")

    def test_harness_validates_json_metrics_numeric_and_finite(self):
        """Verifies JSON metric contract rejects non-numeric or non-finite values."""
        json_path = os.path.join(self.tmp_dir, "metrics.json")
        task = Task(
            task_id="t_num_metrics",
            title="Numeric Metrics",
            outputs=["metrics.json"],
            artifact_contracts=[
                ArtifactContract(
                    path="metrics.json",
                    artifact_type=ArtifactType.METRICS_JSON,
                    required_keys=["loss"],
                )
            ],
        )
        script = generate_task_test_script(task, self.tmp_dir)

        # String non-numeric value fails
        with open(json_path, "w") as f:
            f.write('{"loss": "very_low"}')
        ok_str, _, stderr_str = run_task_test_script(script, self.tmp_dir)
        self.assertFalse(ok_str)
        self.assertIn("must be numeric", stderr_str)

        # Bool value fails
        with open(json_path, "w") as f:
            f.write('{"loss": true}')
        ok_bool, _, stderr_bool = run_task_test_script(script, self.tmp_dir)
        self.assertFalse(ok_bool)
        self.assertIn("cannot be bool", stderr_bool)

        # Valid finite float passes
        with open(json_path, "w") as f:
            f.write('{"loss": 0.042}')
        ok_float, _, stderr_float = run_task_test_script(script, self.tmp_dir)
        self.assertTrue(ok_float, f"Expected finite float pass, got: {stderr_float}")

    def test_harness_validates_cli_interface_flags(self):
        """Verifies CLI interface flag existence in deliverables."""
        cli_path = os.path.join(self.tmp_dir, "train.py")
        with open(cli_path, "w") as f:
            f.write("parser.add_argument('--learning-rate')\nparser.add_argument('--batch-size')\n")

        task = Task(
            task_id="t_cli",
            title="CLI Script",
            outputs=["train.py"],
            artifact_contracts=[
                ArtifactContract(
                    path="train.py",
                    artifact_type=ArtifactType.CLI_INTERFACE,
                    cli_args=["--learning-rate", "--batch-size"],
                )
            ],
        )
        script = generate_task_test_script(task, self.tmp_dir)
        ok, _, stderr = run_task_test_script(script, self.tmp_dir)
        self.assertTrue(ok, f"Expected CLI flags pass, got: {stderr}")

        # Missing flag
        task_missing_flag = Task(
            task_id="t_cli",
            title="CLI Script",
            outputs=["train.py"],
            artifact_contracts=[
                ArtifactContract(
                    path="train.py",
                    artifact_type=ArtifactType.CLI_INTERFACE,
                    cli_args=["--epochs"],
                )
            ],
        )
        script_missing = generate_task_test_script(task_missing_flag, self.tmp_dir)
        ok_miss, _, stderr_miss = run_task_test_script(script_missing, self.tmp_dir)
        self.assertFalse(ok_miss)
        self.assertIn("Expected CLI argument missing", stderr_miss)

    def test_run_task_test_script_env_injection(self):
        """Verifies that run_task_test_script injects thread caps and custom environment variables."""
        test_py = (
            "import os\n"
            "import unittest\n"
            "class TestEnv(unittest.TestCase):\n"
            "    def test_env_caps(self):\n"
            "        self.assertEqual(os.environ.get('OMP_NUM_THREADS'), '1')\n"
            "        self.assertEqual(os.environ.get('TORCH_NUM_THREADS'), '1')\n"
            "        self.assertEqual(os.environ.get('CUSTOM_TEST_KEY'), 'custom_value')\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )
        ok, _, stderr = run_task_test_script(
            test_py,
            self.tmp_dir,
            env={"CUSTOM_TEST_KEY": "custom_value"},
        )
        self.assertTrue(ok, f"Environment injection failed: {stderr}")


if __name__ == "__main__":
    unittest.main()
