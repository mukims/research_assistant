"""
Empirical Challenger Verification Suite for Milestone 4 (R4: Storage Protection & Test Environment Alignment).
Agent: teamwork_preview_challenger_m4_2

Adversarially tests and empirically verifies:
1. TaskRunner(graph, workspace_dir=VECTORDB_PATH) raises ValueError ("Safety violation").
2. TaskRunner(graph, workspace_dir=DATA_DIR) raises ValueError ("Safety violation").
3. TaskRunner(graph, workspace_dir=os.path.join(VECTORDB_PATH, "sub")) raises ValueError ("Safety violation").
4. Additional storage invariants:
   - DATA_DIR subdirectories raise ValueError ("Safety violation").
   - Deeply nested subdirectories raise ValueError ("Safety violation").
   - Relative paths to persistent directories raise ValueError ("Safety violation").
   - Trailing slashes and redundant dot-slashes raise ValueError ("Safety violation").
   - Sibling directory paths outside persistent storage do NOT false-positive.
   - Legitimate workspace directory outside DATA_DIR/VECTORDB_PATH initializes cleanly.
5. ChromaDB collection 'physics_papers' at 'data/physics_vectordb' holds exactly 75,039 chunks without data loss.
6. All test files matching 'tests/test_*.py' have zero 'import pytest' statements (AST and regex verification).
"""

from __future__ import annotations

import ast
import glob
import os
import re
import shutil
import tempfile
import unittest

try:
    import chromadb
except ImportError:  # CI installs requirements-test.txt only
    chromadb = None

from tests.optional_deps import requires_real_store

from research_assistant.config import (
    COLLECTION_NAME,
    DATA_DIR,
    PROJECT_ROOT,
    VECTORDB_PATH,
)
from research_assistant.tasks.models import TaskGraph
from research_assistant.tasks.runner import TaskRunner


class TestChallengerStorageGuardrails(unittest.TestCase):
    """
    Adversarially tests storage protection invariants in TaskRunner.
    Ensures persistent database and data volumes are immune to destruction or overwrites.
    """

    def setUp(self):
        self.graph = TaskGraph(paper_id="challenger_p1", paper_title="Storage Protection Invariant Test")

    def test_direct_vectordb_path_raises_safety_violation(self):
        """TaskRunner(graph, workspace_dir=VECTORDB_PATH) raises ValueError with 'Safety violation'."""
        with self.assertRaises(ValueError) as ctx:
            TaskRunner(graph=self.graph, workspace_dir=VECTORDB_PATH)
        self.assertIn("Safety violation", str(ctx.exception))
        self.assertIn("cannot point to persistent data storage", str(ctx.exception))

    def test_direct_data_dir_raises_safety_violation(self):
        """TaskRunner(graph, workspace_dir=DATA_DIR) raises ValueError with 'Safety violation'."""
        with self.assertRaises(ValueError) as ctx:
            TaskRunner(graph=self.graph, workspace_dir=DATA_DIR)
        self.assertIn("Safety violation", str(ctx.exception))
        self.assertIn("cannot point to persistent data storage", str(ctx.exception))

    def test_vectordb_subdirectory_raises_safety_violation(self):
        """TaskRunner(graph, workspace_dir=os.path.join(VECTORDB_PATH, 'sub')) raises ValueError."""
        sub_vdb = os.path.join(VECTORDB_PATH, "sub")
        with self.assertRaises(ValueError) as ctx:
            TaskRunner(graph=self.graph, workspace_dir=sub_vdb)
        self.assertIn("Safety violation", str(ctx.exception))
        self.assertIn("cannot point to persistent data storage", str(ctx.exception))

    def test_data_dir_subdirectory_raises_safety_violation(self):
        """TaskRunner(graph, workspace_dir=os.path.join(DATA_DIR, 'sub')) raises ValueError."""
        sub_data = os.path.join(DATA_DIR, "sub")
        with self.assertRaises(ValueError) as ctx:
            TaskRunner(graph=self.graph, workspace_dir=sub_data)
        self.assertIn("Safety violation", str(ctx.exception))
        self.assertIn("cannot point to persistent data storage", str(ctx.exception))

    def test_deeply_nested_vectordb_subdirectory_raises_safety_violation(self):
        """Deeply nested subdirectories within VECTORDB_PATH raise ValueError."""
        deep_vdb = os.path.join(VECTORDB_PATH, "deep", "nested", "sub_dir")
        with self.assertRaises(ValueError) as ctx:
            TaskRunner(graph=self.graph, workspace_dir=deep_vdb)
        self.assertIn("Safety violation", str(ctx.exception))

    def test_relative_paths_resolving_to_persistent_storage(self):
        """Relative paths resolving to persistent storage raise ValueError."""
        rel_vdb = os.path.relpath(VECTORDB_PATH)
        with self.assertRaises(ValueError) as ctx:
            TaskRunner(graph=self.graph, workspace_dir=rel_vdb)
        self.assertIn("Safety violation", str(ctx.exception))

        rel_data = os.path.relpath(DATA_DIR)
        with self.assertRaises(ValueError) as ctx:
            TaskRunner(graph=self.graph, workspace_dir=rel_data)
        self.assertIn("Safety violation", str(ctx.exception))

        rel_sub = os.path.join(rel_vdb, "sub")
        with self.assertRaises(ValueError) as ctx:
            TaskRunner(graph=self.graph, workspace_dir=rel_sub)
        self.assertIn("Safety violation", str(ctx.exception))

    def test_trailing_slashes_and_dots_raise_safety_violation(self):
        """Paths with trailing slashes or './' raise ValueError."""
        trailing_vdb = VECTORDB_PATH.rstrip(os.sep) + os.sep
        with self.assertRaises(ValueError) as ctx:
            TaskRunner(graph=self.graph, workspace_dir=trailing_vdb)
        self.assertIn("Safety violation", str(ctx.exception))

        dot_data = os.path.join(DATA_DIR, ".")
        with self.assertRaises(ValueError) as ctx:
            TaskRunner(graph=self.graph, workspace_dir=dot_data)
        self.assertIn("Safety violation", str(ctx.exception))

    def test_sibling_directory_does_not_false_positive(self):
        """A directory that shares a prefix with DATA_DIR but is not inside it must succeed."""
        tmp_parent = tempfile.mkdtemp(prefix="safe_challenger_parent_")
        try:
            # Create a sibling path resembling data_backup
            sibling_dir = os.path.join(tmp_parent, os.path.basename(DATA_DIR) + "_backup")
            runner = TaskRunner(graph=self.graph, workspace_dir=sibling_dir)
            self.assertEqual(runner.workspace_dir, os.path.abspath(sibling_dir))
            self.assertTrue(os.path.isdir(sibling_dir))
        finally:
            shutil.rmtree(tmp_parent, ignore_errors=True)

    def test_safe_workspace_outside_data_dir_succeeds(self):
        """A standard temporary workspace outside persistent data storage initializes without error."""
        tmp_dir = tempfile.mkdtemp(prefix="safe_task_ws_")
        try:
            runner = TaskRunner(graph=self.graph, workspace_dir=tmp_dir)
            self.assertEqual(runner.workspace_dir, os.path.abspath(tmp_dir))
            self.assertTrue(os.path.isdir(tmp_dir))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


@requires_real_store
class TestChallengerChromaDBIntegrity(unittest.TestCase):
    """
    Empirically verifies the persistent ChromaDB collection state.
    Asserts exact chunk count (75,039) and structural integrity of physics_papers.
    """

    def test_chromadb_physics_papers_chunk_count_is_75039(self):
        """ChromaDB collection 'physics_papers' at 'data/physics_vectordb' holds exactly 75,039 chunks."""
        vectordb_dir = os.path.join(PROJECT_ROOT, "data", "physics_vectordb")
        self.assertTrue(os.path.exists(vectordb_dir), f"ChromaDB directory not found: {vectordb_dir}")

        client = chromadb.PersistentClient(path=vectordb_dir)
        collection = client.get_collection("physics_papers")

        count = collection.count()
        self.assertEqual(
            count,
            75039,
            f"Expected exactly 75,039 chunks in collection 'physics_papers', found {count}"
        )

    def test_chromadb_via_config_path_matches_exact_count(self):
        """Verify ChromaDB through VECTORDB_PATH and COLLECTION_NAME configuration constants."""
        client = chromadb.PersistentClient(path=VECTORDB_PATH)
        collection = client.get_collection(COLLECTION_NAME)

        count = collection.count()
        self.assertEqual(
            count,
            75039,
            f"Configured COLLECTION_NAME '{COLLECTION_NAME}' at VECTORDB_PATH holds {count} chunks, expected 75,039"
        )

    def test_chromadb_query_integrity(self):
        """Verify the collection can be queried and records have intact IDs, documents, and metadatas."""
        client = chromadb.PersistentClient(path=VECTORDB_PATH)
        collection = client.get_collection("physics_papers")

        sample = collection.get(limit=5, include=["documents", "metadatas"])
        self.assertEqual(len(sample["ids"]), 5)
        self.assertEqual(len(sample["documents"]), 5)
        self.assertEqual(len(sample["metadatas"]), 5)
        for doc in sample["documents"]:
            self.assertIsInstance(doc, str)
            self.assertGreater(len(doc), 0)


class TestChallengerTestRunnerIsolationAndPytestAbsence(unittest.TestCase):
    """
    Empirically inspects all test files in tests/test_*.py to guarantee complete absence
    of 'import pytest' statements and full compatibility with python -m unittest.
    """

    def setUp(self):
        self.tests_dir = os.path.join(PROJECT_ROOT, "tests")
        self.test_files = sorted(glob.glob(os.path.join(self.tests_dir, "test_*.py")))

    def test_test_files_suite_size(self):
        """Ensures that a non-trivial test suite was discovered (at least 50 test files)."""
        self.assertGreaterEqual(
            len(self.test_files),
            50,
            f"Expected at least 50 test files in {self.tests_dir}, found {len(self.test_files)}"
        )

    def test_all_test_files_have_zero_pytest_ast_imports(self):
        """AST parsing across all tests/test_*.py confirms zero Import or ImportFrom referencing pytest."""
        violations: list[str] = []

        for file_path in self.test_files:
            rel_path = os.path.relpath(file_path, PROJECT_ROOT)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            try:
                tree = ast.parse(content, filename=file_path)
            except SyntaxError as e:
                violations.append(f"{rel_path}: SyntaxError during AST parse: {e}")
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "pytest" or alias.name.startswith("pytest."):
                            violations.append(f"{rel_path}:{node.lineno}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module and (node.module == "pytest" or node.module.startswith("pytest.")):
                        violations.append(f"{rel_path}:{node.lineno}: from {node.module} import ...")

        self.assertEqual(
            violations,
            [],
            f"Found forbidden pytest imports in test files:\n" + "\n".join(violations)
        )

    def test_all_test_files_have_zero_pytest_regex_imports(self):
        """Line-by-line regex confirms no uncommented lines begin with 'import pytest' or 'from pytest'."""
        pytest_import_pattern = re.compile(r"^\s*(import\s+pytest\b|from\s+pytest\b)")
        violations: list[str] = []

        for file_path in self.test_files:
            rel_path = os.path.relpath(file_path, PROJECT_ROOT)
            with open(file_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, 1):
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    if pytest_import_pattern.match(line):
                        violations.append(f"{rel_path}:{line_idx}: {stripped}")

        self.assertEqual(
            violations,
            [],
            f"Found forbidden regex pytest imports in test files:\n" + "\n".join(violations)
        )


if __name__ == "__main__":
    unittest.main()
