"""
Tests for the planner's execution sandbox: model-written code must not see the
app's secrets, and must still be able to import its sibling modules.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from research_assistant.tasks.sandbox import ENV_ALLOWLIST, python_argv, sandbox_env


class TestTaskSandbox(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._temp_dir.name)

    def tearDown(self):
        self._temp_dir.cleanup()

    def test_secrets_never_reach_generated_code(self):
        base = {
            "PATH": "/bin",
            "GEMINI_API_KEY": "g",
            "OPENAI_API_KEY": "o",
            "APP_PASSWORD": "p",
            "HF_TOKEN": "h",
            "AWS_SECRET_ACCESS_KEY": "a",
        }
        env = sandbox_env("/w", base)
        self.assertEqual(env["PATH"], "/bin")
        self.assertEqual(env["HOME"], "/w")
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(env["CITATION_DISABLE_DOTENV"], "1")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
            self.assertEqual(env[name], "1")
        leaked = {"GEMINI_API_KEY", "OPENAI_API_KEY", "APP_PASSWORD", "HF_TOKEN", "AWS_SECRET_ACCESS_KEY"} & set(env)
        self.assertEqual(leaked, set())

    def test_only_allowlisted_keys_are_copied(self):
        base = {k: "x" for k in ENV_ALLOWLIST}
        base["PYTHONPATH"] = "/evil"
        base["HOME"] = "/home/user"
        env = sandbox_env("/w", base)
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["HOME"], "/w")

    def test_child_process_sees_no_key(self):
        out = subprocess.run(
            python_argv("-c", "import os; print(os.environ.get('GEMINI_API_KEY'))"),
            env=sandbox_env(str(self.tmp_path), {**os.environ, "GEMINI_API_KEY": "leak"}),
            cwd=self.tmp_path, capture_output=True, text=True,
        )
        self.assertEqual(out.stdout.strip(), "None")

    def test_python_argv_isolates_but_keeps_script_dir_on_path(self):
        self.assertEqual(python_argv("x.py")[1:], ["-E", "-s", "x.py"])
        self.assertNotIn("-I", python_argv())

    def test_workspace_sibling_import_still_works(self):
        (self.tmp_path / "model.py").write_text("X = 1\n")
        (self.tmp_path / "run_main.py").write_text("from model import X\nprint(X)\n")
        out = subprocess.run(
            python_argv("run_main.py"), env=sandbox_env(str(self.tmp_path)),
            cwd=self.tmp_path, capture_output=True, text=True,
        )
        self.assertEqual(out.stdout.strip(), "1")

    def test_runner_and_harness_use_the_sandbox(self):
        from research_assistant.tasks import harness, runner
        from research_assistant.tasks.models import Task, TaskGraph
        from research_assistant.tasks.runner import ExecutionMode

        captured = []

        def fake_run(argv, **kwargs):
            captured.append((argv, kwargs.get("env")))
            return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

        ws = str(self.tmp_path)
        (self.tmp_path / "run_main.py").write_text("print('hi')\n")
        task = Task(task_id="t1", title="T", description="d", outputs=["run_main.py"])
        graph = TaskGraph(paper_id="p", paper_title="P", tasks=[task])
        tr = runner.TaskRunner(graph=graph, workspace_dir=ws, mode=ExecutionMode.SIMULATION)

        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "leak", "APP_PASSWORD": "leak"}), \
                mock.patch.object(runner.subprocess, "run", side_effect=fake_run), \
                mock.patch.object(harness.subprocess, "run", side_effect=fake_run):
            tr._run_entrypoint_subprocess(task)
            tr._execute_harness_subprocess("import unittest\n", task=task)
            # A caller-supplied env must be allowlisted, never passed through raw.
            harness.run_task_test_script("import unittest\n", ws, env={"PATH": "/bin", "GEMINI_API_KEY": "leak"})

        self.assertGreaterEqual(len(captured), 3)
        for argv, env in captured:
            self.assertEqual(argv[1:3], ["-E", "-s"])
            self.assertNotIn("GEMINI_API_KEY", env)
            self.assertNotIn("APP_PASSWORD", env)
            self.assertEqual(env["CITATION_DISABLE_DOTENV"], "1")


if __name__ == "__main__":
    unittest.main()
