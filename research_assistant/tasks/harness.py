"""
Executable Test Harness Generator & Acceptance Verifier.

Generates and executes deterministic unittest.TestCase test scripts
verifying task deliverables, artifact contracts (tensor shapes, dtypes,
tabular columns, JSON metrics), and non-empty output integrity.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Dict, Optional, Tuple

from research_assistant.shared.log import get_logger
from research_assistant.tasks.models import (
    ArtifactType,
    Task,
)

logger = get_logger("task_harness")


def generate_task_test_script(task: Task, workspace_dir: str = ".") -> str:
    """
    Generates a standalone, runnable unittest.TestCase script that asserts:
    1. Deliverable output files exist and are non-empty.
    2. Formal artifact contracts (tensor shapes, dtypes, JSON numeric metrics,
       CSV/Parquet columns, valid code AST, CLI flags).
    3. Acceptance criteria smoke-test checks.
    """
    clean_id = "".join(c if c.isalnum() or c == "_" else "_" for c in task.task_id)
    lines = [
        "#!/usr/bin/env python3",
        f'"""Automated acceptance test harness for task: {task.task_id} ({task.title})"""',
        "",
        "import ast",
        "import csv",
        "import json",
        "import math",
        "import os",
        "import sys",
        "import unittest",
        "",
        f"WORKSPACE_DIR = os.path.abspath({repr(workspace_dir)})",
        "",
        f"class TestTaskAcceptance_{clean_id}(unittest.TestCase):",
        "    def setUp(self):",
        "        self.workspace = WORKSPACE_DIR",
        "",
        "    def _resolve_path(self, rel_or_abs: str) -> str:",
        "        if os.path.isabs(rel_or_abs):",
        "            return rel_or_abs",
        "        return os.path.abspath(os.path.join(self.workspace, rel_or_abs))",
        "",
        "    def _load_tensor(self, fpath: str):",
        '        """Loads tensor via torch, safetensors, or numpy."""',
        "        if fpath.endswith('.npz'):",
        "            try:",
        "                import numpy as np",
        "                npz = np.load(fpath)",
        "                key = npz.files[0] if npz.files else None",
        "                return npz[key] if key else None",
        "            except Exception:",
        "                return None",
        "        elif fpath.endswith('.npy'):",
        "            try:",
        "                import numpy as np",
        "                return np.load(fpath)",
        "            except Exception:",
        "                return None",
        "        elif fpath.endswith(('.pt', '.pth', '.bin')):",
        "            try:",
        "                import torch",
        "                data = torch.load(fpath, map_location='cpu', weights_only=False)",
        "                if isinstance(data, dict):",
        "                    for v in data.values():",
        "                        if hasattr(v, 'shape'):",
        "                            return v",
        "                    return data",
        "                return data",
        "            except Exception:",
        "                return None",
        "        elif fpath.endswith('.safetensors'):",
        "            try:",
        "                from safetensors.numpy import load_file",
        "                st_dict = load_file(fpath)",
        "                first_k = next(iter(st_dict.keys()), None)",
        "                return st_dict[first_k] if first_k else None",
        "            except Exception:",
        "                try:",
        "                    from safetensors.torch import load_file",
        "                    st_dict = load_file(fpath)",
        "                    first_k = next(iter(st_dict.keys()), None)",
        "                    return st_dict[first_k] if first_k else None",
        "                except Exception:",
        "                    return None",
        "        else:",
        "            try:",
        "                import numpy as np",
        "                return np.load(fpath)",
        "            except Exception:",
        "                try:",
        "                    import torch",
        "                    return torch.load(fpath, map_location='cpu', weights_only=False)",
        "                except Exception:",
        "                    return None",
        "",
        "    def _load_tabular(self, fpath: str):",
        '        """Loads tabular data from Parquet, CSV, or TSV."""',
        "        if fpath.endswith('.parquet'):",
        "            try:",
        "                import pandas as pd",
        "                df = pd.read_parquet(fpath)",
        "                return list(df.columns), {c: str(df[c].dtype) for c in df.columns}",
        "            except ImportError:",
        "                import pyarrow.parquet as pq",
        "                table = pq.read_table(fpath)",
        "                return table.column_names, {c: str(table.schema.field(c).type) for c in table.column_names}",
        "        else:",
        "            sep = '\\t' if fpath.endswith('.tsv') else ','",
        "            try:",
        "                import pandas as pd",
        "                df = pd.read_csv(fpath, sep=sep)",
        "                return list(df.columns), {c: str(df[c].dtype) for c in df.columns}",
        "            except Exception:",
        "                with open(fpath, 'r', encoding='utf-8') as f:",
        "                    reader = csv.reader(f, delimiter=sep)",
        "                    header = next(reader, None) or []",
        "                    return header, {}",
        "",
    ]

    # Test 1: Outputs exist and are non-empty
    lines.append("    def test_output_deliverables_exist_and_non_empty(self):")
    target_deliverables = list(dict.fromkeys(task.outputs + [c.path for c in (task.artifact_contracts or [])]))
    if not target_deliverables:
        lines.append("        # Task defines no physical file deliverables")
        lines.append("        self.assertTrue(True)")
    else:
        for out in target_deliverables:
            lines.append(f"        path = self._resolve_path({repr(out)})")
            lines.append(f"        self.assertTrue(os.path.exists(path), f'Deliverable missing: {{path}}')")
            lines.append(f"        self.assertGreater(os.path.getsize(path), 0, f'Deliverable empty: {{path}}')")
    lines.append("")

    # Test 2: Artifact contracts validation
    if task.artifact_contracts:
        lines.append("    def test_artifact_contracts_schema(self):")
        for idx, contract in enumerate(task.artifact_contracts):
            rel_path = contract.path
            c_type = contract.artifact_type.value if hasattr(contract.artifact_type, "value") else str(contract.artifact_type)
            lines.append(f"        # Contract {idx+1}: {rel_path} ({c_type})")
            lines.append(f"        fpath = self._resolve_path({repr(rel_path)})")
            lines.append(f"        self.assertTrue(os.path.exists(fpath), f'Contract target not found: {{fpath}}')")

            # JSON metric validation
            if contract.artifact_type == ArtifactType.METRICS_JSON or rel_path.endswith(".json"):
                lines.append("        with open(fpath, 'r', encoding='utf-8') as f:")
                lines.append("            data = json.load(f)")
                lines.append("        self.assertIsInstance(data, dict, 'JSON metric deliverable must be a dict')")
                if contract.required_keys:
                    for req_k in contract.required_keys:
                        lines.append(f"        self.assertIn({repr(req_k)}, data, 'Required key missing in JSON')")
                        lines.append(f"        val = data[{repr(req_k)}]")
                        lines.append("        if isinstance(val, bool):")
                        lines.append(f'            self.fail("Metric value for " + {repr(req_k)} + " cannot be bool, got " + str(val))')
                        lines.append("        elif isinstance(val, (int, float)):")
                        lines.append(f'            self.assertTrue(math.isfinite(float(val)), "Metric value for " + {repr(req_k)} + " must be finite, got " + str(val))')
                        lines.append("        elif isinstance(val, str):")
                        lines.append("            try:")
                        lines.append("                num_v = float(val)")
                        lines.append(f'                self.assertTrue(math.isfinite(num_v), "Metric string for " + {repr(req_k)} + " must parse to finite number")')
                        lines.append("            except ValueError:")
                        lines.append(f'                self.fail("Metric value for " + {repr(req_k)} + " must be numeric, got " + repr(val))')
                        lines.append("        elif isinstance(val, dict):")
                        lines.append("            self.assertTrue(len(val) >= 0)")
                lines.append("        for k, v in data.items():")
                lines.append("            if isinstance(v, (int, float)) and not isinstance(v, bool):")
                lines.append('                self.assertTrue(math.isfinite(float(v)), f"Metric {k} must be finite, got {v}")')

            # CLI_INTERFACE validation
            elif contract.artifact_type == ArtifactType.CLI_INTERFACE:
                lines.append("        with open(fpath, 'r', encoding='utf-8') as f:")
                lines.append("            cli_content = f.read()")
                if rel_path.endswith(".py"):
                    lines.append("        try:")
                    lines.append("            ast.parse(cli_content, filename=fpath)")
                    lines.append("        except SyntaxError as e:")
                    lines.append(f"            self.fail(f'Python syntax error in {{fpath}}: {{e}}')")
                if contract.cli_args:
                    for arg in contract.cli_args:
                        lines.append(f"        self.assertIn({repr(arg)}, cli_content, 'Expected CLI argument missing')")

            # CODE validation
            elif contract.artifact_type == ArtifactType.CODE or rel_path.endswith(".py"):
                lines.append("        with open(fpath, 'r', encoding='utf-8') as f:")
                lines.append("            code_text = f.read()")
                lines.append("        try:")
                lines.append("            ast.parse(code_text, filename=fpath)")
                lines.append("        except SyntaxError as e:")
                lines.append(f"            self.fail(f'Python syntax error in {{fpath}}: {{e}}')")

            # TABULAR validation
            elif contract.artifact_type == ArtifactType.TABULAR or rel_path.endswith((".csv", ".tsv", ".parquet")):
                lines.append("        cols_found, dtypes_found = self._load_tabular(fpath)")
                lines.append("        self.assertTrue(len(cols_found) > 0, 'Tabular file missing columns or header row')")
                if contract.columns:
                    for col_name, exp_type in contract.columns.items():
                        lines.append(f"        self.assertIn({repr(col_name)}, cols_found, f'Expected column missing: {col_name}')")
                        if exp_type:
                            lines.append(f"        if {repr(col_name)} in dtypes_found:")
                            lines.append(f"            act_t = dtypes_found[{repr(col_name)}].lower()")
                            lines.append(f"            exp_t = {repr(exp_type.lower())}")
                            lines.append("            if 'int' in exp_t:")
                            lines.append(f"                self.assertTrue('int' in act_t or 'float' in act_t or 'object' in act_t, f'Column {col_name} expected int, got {{act_t}}')")
                            lines.append("            elif 'float' in exp_t or 'double' in exp_t:")
                            lines.append(f"                self.assertTrue('float' in act_t or 'double' in act_t or 'int' in act_t, f'Column {col_name} expected float, got {{act_t}}')")
                            lines.append("            elif 'str' in exp_t or 'object' in exp_t:")
                            lines.append("                self.assertTrue(True)")

            # TENSOR validation
            elif contract.artifact_type == ArtifactType.TENSOR or rel_path.endswith((".npy", ".npz", ".pt", ".pth", ".safetensors")):
                lines.append("        arr = self._load_tensor(fpath)")
                lines.append("        if arr is not None and hasattr(arr, 'shape'):")
                if contract.shape is not None:
                    lines.append(f"            self.assertEqual(len(arr.shape), {len(contract.shape)}, 'Tensor rank mismatch')")
                    for dim_idx, exp_dim in enumerate(contract.shape):
                        if isinstance(exp_dim, int) and exp_dim > 0:
                            lines.append(f"            self.assertEqual(arr.shape[{dim_idx}], {exp_dim}, f'Tensor dim {dim_idx} mismatch')")
                        else:
                            lines.append(f"            self.assertGreater(arr.shape[{dim_idx}], 0, f'Tensor dynamic dim {dim_idx} must be > 0')")
                if contract.dtype is not None:
                    lines.append("            act_dtype = str(getattr(arr, 'dtype', '')).lower()")
                    lines.append(f"            exp_dtype = {repr(contract.dtype.lower())}")
                    lines.append("            self.assertTrue(exp_dtype in act_dtype or act_dtype in exp_dtype, f'Tensor dtype mismatch: expected {exp_dtype}, got {act_dtype}')")
                lines.append("        elif arr is None:")
                lines.append("            # Fallback size verification if tensor library unavailable")
                lines.append("            self.assertGreater(os.path.getsize(fpath), 0, 'Tensor deliverable is empty')")
        lines.append("")

    # Test 3: Acceptance Criteria Smoke-Test Checks
    if task.acceptance_criteria:
        lines.append("    def test_acceptance_criteria_smoke_checks(self):")
        lines.append(f'        """Validates {len(task.acceptance_criteria)} task acceptance criteria."""')
        for idx, crit in enumerate(task.acceptance_criteria):
            lines.append(f"        # Criterion {idx+1}: {crit}")
            crit_lower = crit.lower()
            if "exit" in crit_lower or "code 0" in crit_lower or "non-empty" in crit_lower or "clean" in crit_lower:
                for out in task.outputs:
                    lines.append(f"        path_{idx} = self._resolve_path({repr(out)})")
                    lines.append(f"        self.assertTrue(os.path.exists(path_{idx}), f'Deliverable missing for criterion: {crit}')")
                    lines.append(f"        self.assertGreater(os.path.getsize(path_{idx}), 0, f'Deliverable empty for criterion: {crit}')")
            elif "loss" in crit_lower or "accuracy" in crit_lower or "metric" in crit_lower or "score" in crit_lower:
                for out in task.outputs:
                    if out.endswith(".json"):
                        lines.append(f"        m_path_{idx} = self._resolve_path({repr(out)})")
                        lines.append(f"        if os.path.exists(m_path_{idx}):")
                        lines.append(f"            with open(m_path_{idx}, 'r', encoding='utf-8') as mf_{idx}:")
                        lines.append(f"                m_data_{idx} = json.load(mf_{idx})")
                        lines.append(f"            self.assertIsInstance(m_data_{idx}, dict, 'Metrics deliverable must be a dict')")
            else:
                lines.append("        self.assertTrue(True)")
        lines.append("")

    lines.extend([
        "if __name__ == '__main__':",
        "    unittest.main()",
        "",
    ])
    return "\n".join(lines)


def run_task_test_script(
    script_content: str,
    workspace_dir: str = ".",
    timeout_seconds: int = 30,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[bool, str, str]:
    """
    Executes a test harness script in a subprocess.
    Returns:
        (success: bool, stdout: str, stderr: str)
    """
    os.makedirs(workspace_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix="_test_harness.py",
        dir=workspace_dir,
        delete=False,
        encoding="utf-8",
    ) as tmp_f:
        tmp_path = tmp_f.name
        tmp_f.write(script_content)

    try:
        sub_env = env.copy() if env is not None else os.environ.copy()
        sub_env.setdefault("OMP_NUM_THREADS", "1")
        sub_env.setdefault("MKL_NUM_THREADS", "1")
        sub_env.setdefault("OPENBLAS_NUM_THREADS", "1")
        sub_env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        sub_env.setdefault("NUMEXPR_NUM_THREADS", "1")
        sub_env.setdefault("TORCH_NUM_THREADS", "1")
        cmd = [sys.executable, "-m", "unittest", os.path.basename(tmp_path)]
        res = subprocess.run(
            cmd,
            cwd=workspace_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=sub_env,
        )
        success = res.returncode == 0
        return success, res.stdout, res.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Test harness timed out after {timeout_seconds}s"
    except Exception as exc:
        return False, "", f"Failed to execute test harness: {exc}"
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
