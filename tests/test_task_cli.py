"""
Unit and integration tests for orchestrate_tasks.py CLI.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TestTaskCLI(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._temp_dir.name)

    def tearDown(self):
        self._temp_dir.cleanup()

    def test_cli_help(self):
        res = subprocess.run([sys.executable, "orchestrate_tasks.py", "--help"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("plan", res.stdout)
        self.assertIn("run", res.stdout)
        self.assertIn("show", res.stdout)

    def test_cli_plan_heuristic(self):
        paper = self.tmp_path / "paper.txt"
        paper.write_text("Title: Quantum ML\nMethod: Train neural networks on quantum states.")
        out_json = self.tmp_path / "plan.json"

        res = subprocess.run([
            sys.executable, "orchestrate_tasks.py", "plan",
            str(paper),
            "--paper-id", "qml_paper",
            "--heuristic",
            "--format", "json",
            "-o", str(out_json),
        ], capture_output=True, text=True)

        self.assertEqual(res.returncode, 0)
        self.assertTrue(out_json.exists())

        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["paper_id"], "qml_paper")
        self.assertEqual(len(data["tasks"]), 5)

    def test_cli_plan_formats(self):
        paper = self.tmp_path / "paper.txt"
        paper.write_text("Method: Sample methodology.")

        for fmt in ["markdown", "mermaid", "antigravity"]:
            out_file = self.tmp_path / f"plan.{fmt}"
            res = subprocess.run([
                sys.executable, "orchestrate_tasks.py", "plan",
                str(paper),
                "--heuristic",
                "--format", fmt,
                "-o", str(out_file),
            ], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0)
            self.assertTrue(out_file.exists())
            self.assertGreater(len(out_file.read_text(encoding="utf-8")), 50)

    def test_cli_show(self):
        paper = self.tmp_path / "paper.txt"
        paper.write_text("Method: Sample methodology.")
        plan_file = self.tmp_path / "plan.json"

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
        self.assertEqual(res.returncode, 0)
        self.assertIn("# Research Execution Plan", res.stdout)

        # Show mermaid
        res_m = subprocess.run([
            sys.executable, "orchestrate_tasks.py", "show",
            str(plan_file), "--format", "mermaid",
        ], capture_output=True, text=True)
        self.assertEqual(res_m.returncode, 0)
        self.assertIn("flowchart TD", res_m.stdout)

    def test_cli_run_simulation(self):
        paper = self.tmp_path / "paper.txt"
        paper.write_text("Method: Sample methodology.")
        plan_file = self.tmp_path / "plan.json"
        workspace = self.tmp_path / "cli_workspace"
        report_file = self.tmp_path / "report.md"

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

        self.assertEqual(res.returncode, 0)
        self.assertTrue(report_file.exists())
        report_content = report_file.read_text(encoding="utf-8")
        self.assertIn("Replication Scorecard", report_content)
        self.assertIn("CERTIFIED_REPLICABLE", report_content)

    def test_cli_plan_missing_file(self):
        missing = self.tmp_path / "does_not_exist.pdf"
        res = subprocess.run([
            sys.executable, "orchestrate_tasks.py", "plan",
            str(missing), "--heuristic",
        ], capture_output=True, text=True)
        self.assertEqual(res.returncode, 1)
        self.assertIn("Error: Paper file", res.stderr)

    def test_cli_show_missing_file(self):
        missing = self.tmp_path / "does_not_exist.json"
        res = subprocess.run([
            sys.executable, "orchestrate_tasks.py", "show",
            str(missing),
        ], capture_output=True, text=True)
        self.assertEqual(res.returncode, 1)
        self.assertIn("Error: Plan file", res.stderr)

    def test_cli_show_corrupt_file(self):
        corrupt = self.tmp_path / "corrupt_plan.json"
        corrupt.write_text("invalid json content")
        res = subprocess.run([
            sys.executable, "orchestrate_tasks.py", "show",
            str(corrupt),
        ], capture_output=True, text=True)
        self.assertEqual(res.returncode, 1)
        self.assertIn("Error: Failed to load plan file", res.stderr)

    def test_cli_show_json_and_antigravity(self):
        paper = self.tmp_path / "paper.txt"
        paper.write_text("Method: Sample methodology.")
        plan_file = self.tmp_path / "plan.json"

        subprocess.run([
            sys.executable, "orchestrate_tasks.py", "plan",
            str(paper), "--heuristic", "-o", str(plan_file),
        ], check=True)

        res_json = subprocess.run([
            sys.executable, "orchestrate_tasks.py", "show",
            str(plan_file), "--format", "json",
        ], capture_output=True, text=True)
        self.assertEqual(res_json.returncode, 0)
        parsed = json.loads(res_json.stdout)
        self.assertIn("tasks", parsed)

        res_anti = subprocess.run([
            sys.executable, "orchestrate_tasks.py", "show",
            str(plan_file), "--format", "antigravity",
        ], capture_output=True, text=True)
        self.assertEqual(res_anti.returncode, 0)
        parsed_anti = json.loads(res_anti.stdout)
        self.assertIsInstance(parsed_anti, list)
        self.assertIn("Role", parsed_anti[0])

    def test_cli_run_missing_file(self):
        missing = self.tmp_path / "does_not_exist.json"
        res = subprocess.run([
            sys.executable, "orchestrate_tasks.py", "run",
            str(missing),
        ], capture_output=True, text=True)
        self.assertEqual(res.returncode, 1)
        self.assertIn("Error: Plan file", res.stderr)

    def test_cli_run_corrupt_file(self):
        corrupt = self.tmp_path / "corrupt_plan.json"
        corrupt.write_text("invalid json content")
        res = subprocess.run([
            sys.executable, "orchestrate_tasks.py", "run",
            str(corrupt),
        ], capture_output=True, text=True)
        self.assertEqual(res.returncode, 1)
        self.assertIn("Error: Failed to load plan file", res.stderr)

    def test_cli_plan_output_write_error(self):
        paper = self.tmp_path / "paper.txt"
        paper.write_text("Method: Sample methodology.")
        # Try to write into a path that cannot be created (a file used as directory)
        blocking_file = self.tmp_path / "blocker"
        blocking_file.write_text("not a dir")
        bad_out = blocking_file / "sub" / "plan.json"

        res = subprocess.run([
            sys.executable, "orchestrate_tasks.py", "plan",
            str(paper), "--heuristic", "-o", str(bad_out),
        ], capture_output=True, text=True)
        self.assertEqual(res.returncode, 1)
        self.assertIn("Error: Failed to write plan", res.stderr)

    def test_cli_query_subcommand(self):
        paper = self.tmp_path / "paper_query.txt"
        paper.write_text("Abstract: Quantum circuit training with parameter-shift gradients.")
        plan_file = self.tmp_path / "plan_query.json"

        subprocess.run([
            sys.executable, "orchestrate_tasks.py", "plan",
            str(paper), "--heuristic", "-o", str(plan_file),
        ], check=True, capture_output=True, text=True)

        res_md = subprocess.run([
            sys.executable, "orchestrate_tasks.py", "query",
            str(plan_file), "--task-id", "task_03_model",
            "--query", "parameter-shift gradients", "--format", "markdown",
        ], capture_output=True, text=True)
        self.assertEqual(res_md.returncode, 0)
        self.assertIn("# Database Query Results", res_md.stdout)

        res_json = subprocess.run([
            sys.executable, "orchestrate_tasks.py", "query",
            str(plan_file), "--task-id", "task_03_model",
            "--query", "parameter-shift gradients", "--format", "json",
        ], capture_output=True, text=True)
        self.assertEqual(res_json.returncode, 0)
        payload = json.loads(res_json.stdout)
        self.assertEqual(payload["task_id"], "task_03_model")
        self.assertIn("results", payload)


if __name__ == "__main__":
    unittest.main()

