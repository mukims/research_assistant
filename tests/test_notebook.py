"""The pipeline test-bench notebook (notebooks/pipeline.ipynb).

A notebook rots silently: nothing imports it, so a renamed function or a stray
prompt only shows up when someone runs it for an hour. These checks are static
(stdlib only, so they run in CI) except TestHelpers, which executes the notebook's
own helper cell in a subprocess. A subprocess because use() drops every
research_assistant module from sys.modules, which would corrupt the rest of this
test run if done in-process.
"""

import ast
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "pipeline.ipynb"
PROJECT = ("research_assistant", "orchestrate")


def _import(name):
    """Import a project module; skip when a third-party dependency is absent (CI installs few)."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if (error.name or "").split(".")[0] in PROJECT:
            raise  # one of our own modules vanished: the bug this test exists to catch
        raise unittest.SkipTest(f"needs {error.name}, which this environment lacks")


def _cells():
    with open(NOTEBOOK, encoding="utf-8") as fh:
        return json.load(fh)["cells"]


def _code(cell):
    """Cell source as Python: IPython magics and shell escapes removed."""
    lines = "".join(cell["source"]).splitlines()
    return "\n".join("" if line.lstrip().startswith(("%", "!")) else line for line in lines)


def _code_cells():
    return [c for c in _cells() if c["cell_type"] == "code"]


def _helper_cell():
    found = [c for c in _code_cells() if "def use(" in _code(c)]
    assert len(found) == 1, "exactly one cell must define use()"
    return found[0]


class TestNotebookFile(unittest.TestCase):
    def test_is_a_valid_nbformat_4_notebook(self):
        with open(NOTEBOOK, encoding="utf-8") as fh:
            nb = json.load(fh)
        self.assertEqual(nb["nbformat"], 4)
        self.assertGreater(len(nb["cells"]), 10)
        for cell in nb["cells"]:
            self.assertIn(cell["cell_type"], ("markdown", "code"))
            self.assertIn("source", cell)

    def test_is_committed_without_outputs(self):
        for cell in _code_cells():
            self.assertEqual(cell.get("outputs"), [], "".join(cell["source"])[:60])
            self.assertIsNone(cell.get("execution_count"))

    def test_every_code_cell_parses(self):
        for cell in _code_cells():
            with self.subTest(cell="".join(cell["source"])[:50]):
                ast.parse(_code(cell))

    def test_never_prompts(self):
        for cell in _code_cells():
            for node in ast.walk(ast.parse(_code(cell))):
                if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "input":
                    self.fail("input() would block a headless run: " + _code(cell)[:60])

    def test_every_import_from_the_project_resolves(self):
        for cell in _code_cells():
            for node in ast.walk(ast.parse(_code(cell))):
                if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    if node.module.split(".")[0] not in PROJECT:
                        continue
                    module = _import(node.module)
                    for alias in node.names:
                        with self.subTest(name=f"{node.module}.{alias.name}"):
                            if not hasattr(module, alias.name):  # `from package import submodule`
                                _import(f"{node.module}.{alias.name}")

    def test_every_stage_cell_states_its_success_criterion(self):
        stages = [c for c in _code_cells() if "with stage(" in _code(c)]
        self.assertGreater(len(stages), 8)
        for cell in stages:
            with self.subTest(cell=_code(cell).strip().splitlines()[0][:60]):
                self.assertTrue(any(isinstance(n, ast.Assert) for n in ast.walk(ast.parse(_code(cell)))))

    def test_scratch_data_is_git_ignored(self):
        self.assertIn(".data/", (ROOT / "notebooks" / ".gitignore").read_text().split())


class TestHelpers(unittest.TestCase):
    """Run the helper cell for real, in a fresh interpreter."""

    def _run(self, body):
        helper = _code(_helper_cell())
        program = textwrap.dedent(
            f"""
            import os, sys
            from pathlib import Path
            for key in [k for k in os.environ if k.startswith("CITATION_")]:
                del os.environ[key]
            REPO = Path({str(ROOT)!r})
            sys.path.insert(0, str(REPO))
            SUMMARY_MODEL, LLM_BACKEND, EMBED_BACKEND = "summary-model", "ollama", "ollama"
            """
        ) + "\n" + helper + "\n" + textwrap.dedent(body)
        result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, cwd=tempfile.gettempdir())
        for name in re.findall(r"No module named '([^']+)'", result.stderr) if result.returncode else ():
            if name.split(".")[0] not in PROJECT:
                self.skipTest(f"needs {name}, which this environment lacks")
        return result

    def test_refuses_the_real_data_directory(self):
        result = self._run(
            """
            for bad in (REPO / "data", REPO / "data" / "physics_vectordb"):
                try:
                    check_data_dir(bad)
                except RuntimeError:
                    continue
                raise SystemExit(f"accepted {bad}")
            check_data_dir(Path(__import__("tempfile").mkdtemp()))
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr[-600:])

    def test_use_switches_every_model_and_the_data_directory(self):
        result = self._run(
            """
            import tempfile
            a, b = tempfile.mkdtemp(), tempfile.mkdtemp()
            use("embed-a", "llm-a", a)
            import research_assistant.config as config
            first = config
            assert (config.EMBED_MODEL, config.LLM_MODEL, config.CHAT_MODEL) == ("embed-a", "llm-a", "llm-a"), config.EMBED_MODEL
            assert config.JUDGEMENT_MODEL == "llm-a", config.JUDGEMENT_MODEL
            assert config.SUMMARY_MODEL == "summary-model", config.SUMMARY_MODEL
            assert config.INDEX_VERSION == 2 and config.DATA_DIR == a
            use("embed-b", "llm-b", b)
            import research_assistant.config as config
            assert config is not first, "modules were not re-imported"
            assert (config.EMBED_MODEL, config.LLM_MODEL, config.CHAT_MODEL, config.JUDGEMENT_MODEL) == ("embed-b", "llm-b", "llm-b", "llm-b")
            assert config.DATA_DIR == b and config.SUMMARY_MODEL == "summary-model"
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr[-800:])

    def test_a_stray_gemini_key_cannot_reroute_the_comparison(self):
        result = self._run(
            """
            import tempfile
            os.environ["GEMINI_API_KEY"] = "stray"
            # "gemma4:e2b" is the repo's default name, the one config.py rewrites to a Gemini model when a key is present
            use("embed-a", "gemma4:e2b", tempfile.mkdtemp())
            import research_assistant.config as config
            assert (config.LLM_MODEL, config.CHAT_MODEL, config.JUDGEMENT_MODEL) == ("gemma4:e2b",) * 3, (config.LLM_MODEL, config.CHAT_MODEL, config.JUDGEMENT_MODEL)
            assert config.LLM_BACKEND == "ollama", config.LLM_BACKEND
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr[-800:])


if __name__ == "__main__":
    unittest.main()
