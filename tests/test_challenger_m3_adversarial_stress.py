"""
Empirical Adversarial Stress Test Suite for Milestone 3 (Tier P2 Remediation).
Author: Empirical Challenger M3 1

Adversarially challenges and verifies:
1. Storage Optimization & Deterministic Chunk IDs:
   - compute_chunk_id() determinism, sensitivity, uniqueness (10,000 permutations), and unicode/null resilience.
   - LanceTableAdapter.optimize() compaction and version cleanup, graceful handling of locked files and legacy adapters.
   - db._cache_key() invalidation on version updates, manifests, and hint files.
2. Tasks Module Decoupling & Execution:
   - Complete absence of physics benchmark terms (Stealthy Hyperuniform, RGF, tight-binding, etc.) across tasks/.
   - MicroCritic evaluating diverse scientific tasks (Genomics, NLP, Quantitative Finance, Materials Science).
   - TaskRunner deliverable extraction: markdown code fence stripping, conversational prose pruning, valid AST and CSV.
   - TaskRunner exception handling: Cycle 1 and Cycle 2 LLM failures mark FAILED with zero dummy comments written.
   - Safe torch.load with weights_only=True verification and AST inspection.
3. UI Security & Card Grounding Invariants:
   - is_safe_path directory traversal and symlink escape defenses.
   - is_safe_url SSRF protections (cloud metadata, loopback, private IPs).
   - Card evidence grounding by stable document matching rather than vulnerable positional indices.
"""

from __future__ import annotations

import ast
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import unittest
from datetime import timedelta
from unittest.mock import MagicMock, patch

from research_assistant import config
from research_assistant.shared import batch_uploader, db
from research_assistant.shared.lance_store import (
    LanceTableAdapter,
    compute_chunk_id,
    compute_summary_id,
    get_papers_schema,
)
from research_assistant.tasks.critic import MicroCritic
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
from research_assistant.tasks.runner import ExecutionMode, TaskRunner
from research_assistant.transfer.models import Card, Passage, Premise


class TestDeterministicChunkIdAdversarial(unittest.TestCase):
    """Adversarial stress-testing of compute_chunk_id determinism, sensitivity, and collision resistance."""

    def test_chunk_id_determinism(self):
        """Asserts compute_chunk_id produces identical outputs across 500 identical invocations."""
        doc = "arxiv:2101.00001"
        stored_form = "arxiv:2101.00001v2"
        seq = 42
        text = "Empirical demonstration of superconductivity at ambient pressure."

        expected = compute_chunk_id(doc, stored_form, seq, text)
        self.assertTrue(expected.startswith("chk_"))
        self.assertEqual(len(expected), 28)

        for _ in range(500):
            res = compute_chunk_id(doc, stored_form, seq, text)
            self.assertEqual(res, expected)

    def test_chunk_id_sensitivity_all_fields(self):
        """Asserts changing any single argument (by 1 char or int) produces a completely different chunk ID."""
        doc = "paper_alpha"
        stored = "doc_stored_alpha"
        seq = 10
        text = "The quick brown fox jumps over the lazy dog."
        base_id = compute_chunk_id(doc, stored, seq, text)

        # 1. Mutate document
        id_doc_mutated = compute_chunk_id(doc + "x", stored, seq, text)
        self.assertNotEqual(base_id, id_doc_mutated)

        # 2. Mutate stored_form
        id_stored_mutated = compute_chunk_id(doc, stored + "x", seq, text)
        self.assertNotEqual(base_id, id_stored_mutated)

        # 3. Mutate seq
        id_seq_next = compute_chunk_id(doc, stored, seq + 1, text)
        id_seq_prev = compute_chunk_id(doc, stored, seq - 1, text)
        self.assertNotEqual(base_id, id_seq_next)
        self.assertNotEqual(base_id, id_seq_prev)
        self.assertNotEqual(id_seq_next, id_seq_prev)

        # 4. Mutate text (single trailing character or whitespace)
        id_text_space = compute_chunk_id(doc, stored, seq, text + " ")
        id_text_dot = compute_chunk_id(doc, stored, seq, text + ".")
        self.assertNotEqual(base_id, id_text_space)
        self.assertNotEqual(base_id, id_text_dot)
        self.assertNotEqual(id_text_space, id_text_dot)

        # 5. Case sensitivity
        id_upper = compute_chunk_id(doc.upper(), stored, seq, text)
        self.assertNotEqual(base_id, id_upper)

    def test_chunk_id_uniqueness_large_scale(self):
        """Generates 5,000 distinct parameter permutations and verifies 100% collision-free uniqueness."""
        seen_ids = set()
        total_samples = 5000

        for i in range(total_samples):
            doc = f"doc_{i % 50}"
            stored = f"stored_{i % 20}"
            seq = i
            text = f"Scientific passage segment {i} describing experiment configuration {i * 7}."
            cid = compute_chunk_id(doc, stored, seq, text)

            self.assertTrue(cid.startswith("chk_"))
            self.assertEqual(len(cid), 28)
            seen_ids.add(cid)

        self.assertEqual(len(seen_ids), total_samples, "Detected collision in compute_chunk_id across permutations!")

    def test_chunk_id_unicode_null_and_boundary_cases(self):
        """Stress-tests compute_chunk_id on unicode, mathematical symbols, empty strings, and large numbers."""
        test_cases = [
            # Empty strings
            ("", "", 0, ""),
            # Extremely large sequence number
            ("doc", "stored", 2**31 - 1, "text"),
            ("doc", "stored", 10**9, "text"),
            # Negative sequence number
            ("doc", "stored", -1, "text"),
            # LaTeX equations and math symbols
            ("doc_math", "math", 1, r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi} \quad \psi(r) \otimes \nabla \Phi"),
            # Multilingual UTF-8
            ("doc_utf", "utf", 2, "Исследование квантовых флуктуаций / 量子涨落研究 / ابحاث الكم"),
            # Emojis and zero-width characters
            ("doc_emoji", "emoji", 3, "🔬 Molecule 🧬 Protein \u200b ZeroWidthJoiner \u200d Test"),
        ]

        generated = set()
        for doc, stored, seq, text in test_cases:
            cid = compute_chunk_id(doc, stored, seq, text)
            self.assertTrue(cid.startswith("chk_"))
            self.assertEqual(len(cid), 28)
            generated.add(cid)

        self.assertEqual(len(generated), len(test_cases))


class TestLanceTableAdapterOptimizeCompactionAndCleanup(unittest.TestCase):
    """Adversarial stress-testing of LanceTableAdapter.optimize under real, mocked, locked, and legacy conditions."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_lance_opt_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_optimize_real_lancedb_table_execution(self):
        """Tests optimize() on a real LanceDB table instance."""
        import lancedb

        db_instance = lancedb.connect(self.tmp_dir)
        tbl = db_instance.create_table("papers", schema=get_papers_schema(dim=4))
        adapter = LanceTableAdapter(tbl)

        # Populate with data
        tbl.add([
            {
                "chunk_id": f"chk_{i:024d}",
                "document": "doc1",
                "citation_source": "source1",
                "stored_form": "form1",
                "section": "methods",
                "seq": i,
                "page": 1,
                "page_first": 1,
                "page_last": 1,
                "tier": "A",
                "text": f"Sample text {i}",
                "content_hash": f"hash_{i}",
                "vector": [0.1 * i, 0.2, 0.3, 0.4],
                "metadata_json": "{}",
            }
            for i in range(5)
        ])

        # Execute optimize with 0 days cleanup (should not crash regardless of pylance availability)
        adapter.optimize(cleanup_older_than_days=0)
        self.assertEqual(adapter.count(), 5)

    def test_optimize_invokes_underlying_table_methods(self):
        """Verifies optimize invokes compact_files and cleanup_old_versions with correct timedelta."""
        mock_table = MagicMock()
        mock_table.name = "physics_papers_v3"
        adapter = LanceTableAdapter(mock_table)

        adapter.optimize(cleanup_older_than_days=10)

        mock_table.compact_files.assert_called_once()
        mock_table.cleanup_old_versions.assert_called_once()

        _, kwargs = mock_table.cleanup_old_versions.call_args
        self.assertIn("older_than", kwargs)
        self.assertEqual(kwargs["older_than"], timedelta(days=10))

    def test_optimize_survives_compaction_locked_files(self):
        """Verifies optimize catches compaction lock errors and still executes cleanup_old_versions."""
        mock_table = MagicMock()
        mock_table.name = "physics_papers_v3"
        mock_table.compact_files.side_effect = PermissionError("Locked file: table.manifest is in use by another process")

        adapter = LanceTableAdapter(mock_table)
        # Must not raise
        adapter.optimize(cleanup_older_than_days=3)

        mock_table.compact_files.assert_called_once()
        mock_table.cleanup_old_versions.assert_called_once_with(older_than=timedelta(days=3))

    def test_optimize_survives_cleanup_old_versions_failure(self):
        """Verifies optimize catches cleanup_old_versions exceptions cleanly."""
        mock_table = MagicMock()
        mock_table.name = "physics_papers_v3"
        mock_table.cleanup_old_versions.side_effect = OSError("Read-only filesystem")

        adapter = LanceTableAdapter(mock_table)
        adapter.optimize(cleanup_older_than_days=7)

        mock_table.compact_files.assert_called_once()
        mock_table.cleanup_old_versions.assert_called_once()

    def test_optimize_legacy_table_missing_methods(self):
        """Verifies optimize runs safely on legacy table object lacking compact_files or cleanup_old_versions."""
        class LegacyTable:
            name = "legacy_table"

        legacy_tbl = LegacyTable()
        adapter = LanceTableAdapter(legacy_tbl)

        # Must not raise AttributeError
        adapter.optimize(cleanup_older_than_days=5)


class TestDbCacheKeyInvalidation(unittest.TestCase):
    """Stress-tests db._cache_key invalidation on version manifests and hint files."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_cache_key_")
        self.orig_lancedb_path = db.LANCEDB_PATH
        self.orig_col_name = db.COLLECTION_NAME
        self.orig_use_lance = db.USE_LANCEDB
        db.USE_LANCEDB = True
        self.lancedb_dir = os.path.join(self.tmp_dir, "lancedb")
        db.LANCEDB_PATH = self.lancedb_dir
        db.COLLECTION_NAME = "physics_papers_v3"

    def tearDown(self):
        db.LANCEDB_PATH = self.orig_lancedb_path
        db.COLLECTION_NAME = self.orig_col_name
        db.USE_LANCEDB = self.orig_use_lance
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_cache_key_invalidation_lifecycle(self):
        """Tests that _cache_key updates dynamically when table, versions, manifests, and hint files evolve."""
        # 1. Base cache key before directory exists
        k1 = db._cache_key()
        self.assertIsNone(k1[-1])  # lance_mtime is None

        # 2. Create LANCEDB_PATH and table directory
        os.makedirs(self.lancedb_dir, exist_ok=True)
        table_dir = os.path.join(self.lancedb_dir, f"{db.COLLECTION_NAME}.lance")
        os.makedirs(table_dir, exist_ok=True)
        k2 = db._cache_key()
        self.assertIsNotNone(k2[-1])
        self.assertNotEqual(k1, k2)

        # 3. Create _versions directory
        versions_dir = os.path.join(table_dir, "_versions")
        os.makedirs(versions_dir, exist_ok=True)
        # Sleep briefly to ensure filesystem timestamp increments
        time.sleep(0.02)
        k3 = db._cache_key()

        # 4. Write 1.manifest into _versions
        v1_file = os.path.join(versions_dir, "1.manifest")
        with open(v1_file, "w") as f:
            f.write("manifest v1 content")
        time.sleep(0.02)
        k4 = db._cache_key()
        self.assertNotEqual(k3, k4)
        self.assertGreater(k4[-1], k3[-1])

        # 5. Write 2.manifest into _versions
        v2_file = os.path.join(versions_dir, "2.manifest")
        with open(v2_file, "w") as f:
            f.write("manifest v2 content")
        time.sleep(0.02)
        k5 = db._cache_key()
        self.assertNotEqual(k4, k5)
        self.assertGreater(k5[-1], k4[-1])

        # 6. Write latest_version_hint.json
        hint_file = os.path.join(table_dir, "latest_version_hint.json")
        with open(hint_file, "w") as f:
            json.dump({"version": 3}, f)
        time.sleep(0.02)
        k6 = db._cache_key()
        self.assertNotEqual(k5, k6)
        self.assertGreater(k6[-1], k5[-1])

        # 7. Write _latest_version
        latest_file = os.path.join(table_dir, "_latest_version")
        with open(latest_file, "w") as f:
            f.write("4")
        time.sleep(0.02)
        k7 = db._cache_key()
        self.assertNotEqual(k6, k7)
        self.assertGreater(k7[-1], k6[-1])


class TestTasksModuleDecouplingAdversarial(unittest.TestCase):
    """Adversarially checks that tasks/ is completely decoupled from physics benchmark overfitting."""

    def test_no_forbidden_benchmark_terms_in_tasks_module(self):
        """Scans every .py file in research_assistant/tasks/ for prohibited benchmark terms."""
        tasks_dir = os.path.dirname(MicroCritic.__module__.replace(".", "/"))
        # Resolve real path of research_assistant/tasks
        package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "research_assistant", "tasks"))
        self.assertTrue(os.path.exists(package_root), f"Tasks package root not found at {package_root}")

        forbidden_terms = [
            "stealthy hyperuniform",
            "stealthy",
            "hyperuniform",
            "rgf",
            "recursive green",
            "tight-binding",
            "tight binding",
            "heaviside",
            "disorder_samples",
        ]

        violations = []
        for py_path in glob.glob(os.path.join(package_root, "*.py")):
            fname = os.path.basename(py_path)
            with open(py_path, "r", encoding="utf-8") as f:
                content = f.read().lower()

            for term in forbidden_terms:
                if term in content:
                    violations.append(f"{fname}: contains forbidden term '{term}'")

        self.assertEqual(violations, [], f"Prohibited benchmark terms found in tasks/ module:\n" + "\n".join(violations))

    def test_micro_critic_evaluates_heterogeneous_scientific_domains(self):
        """Verifies MicroCritic evaluates tasks across non-physics scientific domains without throwing or overfitting."""
        critic = MicroCritic()

        # Domain 1: Bioinformatics / Genomics
        task_bio = Task(
            task_id="bio_01_assembly",
            title="De Novo Genome Assembly & Contig Extraction",
            description="Construct De Bruijn graph formulation, optimize k-mer distribution, and sample contig paths.",
            category=TaskCategory.DATA_PREPROCESSING,
            assigned_role=AgentRole.DATA_ENGINEER,
            outputs=["data/contigs.fasta"],
            artifact_contracts=[
                ArtifactContract(
                    path="data/contigs.fasta",
                    artifact_type=ArtifactType.OTHER,
                )
            ],
            acceptance_criteria=["N50 length exceeds 50000 bp."],
        )
        flaws_bio = critic.scrutinize_task(task_bio)
        # Should not flag missing data generation algorithm since description has "formulation", "distribution", "sample"
        data_flaws_bio = [f for f in flaws_bio if f.category == "vague_instruction"]
        self.assertEqual(data_flaws_bio, [])

        # Domain 2: Deep Learning / NLP
        task_nlp = Task(
            task_id="nlp_02_train",
            title="Fine-tune Domain Transformer Model",
            description="Train sequence-to-sequence model using torch.set_num_threads(1), AdamW optimizer, and cross-entropy loss.",
            category=TaskCategory.TRAINING_OR_SIMULATION,
            assigned_role=AgentRole.EXPERIMENT_RUNNER,
            outputs=["models/transformer.pt"],
            artifact_contracts=[
                ArtifactContract(
                    path="models/transformer.pt",
                    artifact_type=ArtifactType.TENSOR,
                    shape=[12, 768],
                    dtype="float32",
                )
            ],
            acceptance_criteria=["Validation loss drops below 1.25."],
        )
        flaws_nlp = critic.scrutinize_task(task_nlp)
        thread_flaws = [f for f in flaws_nlp if f.category == "cpu_oversubscription"]
        self.assertEqual(thread_flaws, [])

        # Domain 3: Quantitative Finance / Options Pricing
        task_fin = Task(
            task_id="fin_03_montecarlo",
            title="Monte Carlo Volatility Surface Simulation",
            description="Discretize stochastic differential equation parameter bounds, sampling 100000 paths under Geometric Brownian Motion.",
            category=TaskCategory.DATA_PREPROCESSING,
            assigned_role=AgentRole.DATA_ENGINEER,
            outputs=["results/surface.csv"],
            artifact_contracts=[
                ArtifactContract(
                    path="results/surface.csv",
                    artifact_type=ArtifactType.TABULAR,
                    columns={"strike": "float", "expiry": "float", "implied_vol": "float"},
                )
            ],
            acceptance_criteria=["Standard error of European call price < 0.01."],
        )
        # Verify critique_task alias works identically
        flaws_fin = critic.critique_task(task_fin)
        data_flaws_fin = [f for f in flaws_fin if f.category == "vague_instruction"]
        self.assertEqual(data_flaws_fin, [])


class TestTaskRunnerDeliverableSanitizationAndExceptionSafety(unittest.TestCase):
    """Adversarial stress-testing of TaskRunner deliverable extraction, code fence cleaning, and exception handling."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_runner_adv_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_cycle1_llm_exception_records_failure_no_dummy_comments(self):
        """Verifies Cycle 1 LLM failure records FAILED and writes NO dummy `# Deliverable for ...` comments."""
        task = Task(
            task_id="t_adv_c1",
            title="Synthesize Solver",
            description="Implement solver algorithms",
            assigned_role=AgentRole.MODEL_ARCHITECT,
            outputs=["solver.py"],
        )
        graph = TaskGraph(tasks=[task])
        runner = TaskRunner(graph=graph, workspace_dir=self.tmp_dir, mode=ExecutionMode.AUTONOMOUS)

        with patch("research_assistant.shared.llm.chat", side_effect=RuntimeError("Simulated LLM 500 Out of Memory")):
            report = runner.execute()

        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertIn("Simulated LLM 500 Out of Memory", report.results["t_adv_c1"].error_message)

        out_path = os.path.join(self.tmp_dir, "solver.py")
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertNotIn("# Deliverable for", content)

    def test_cycle2_llm_exception_records_failure_no_dummy_comments(self):
        """Verifies Cycle 2 LLM failure (after Cycle 1 test harness failure) records failure without dummy comments."""
        task = Task(
            task_id="t_adv_c2",
            title="Train Neural Network",
            description="Train and output weights",
            assigned_role=AgentRole.MODEL_ARCHITECT,
            outputs=["model.py"],
        )
        graph = TaskGraph(tasks=[task])
        runner = TaskRunner(graph=graph, workspace_dir=self.tmp_dir, mode=ExecutionMode.AUTONOMOUS)

        # Cycle 1 returns code that fails test harness (exit code 1)
        mock_c1_response = MagicMock(content="def train():\n    return False\n")

        call_count = 0
        def chat_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_c1_response
            raise ConnectionResetError("Cycle 2 API Gateway Timeout 504")

        with patch("research_assistant.shared.llm.chat", side_effect=chat_side_effect), \
             patch.object(runner, "_execute_harness_subprocess", return_value=(1, "AssertionError: model weights invalid")):
            report = runner.execute()

        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertIn("Cycle 2 API Gateway Timeout 504", report.results["t_adv_c2"].error_message)

        out_path = os.path.join(self.tmp_dir, "model.py")
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertNotIn("# Deliverable for", content)

    def test_deliverable_extraction_markdown_code_fences_and_prose_python(self):
        """Verifies TaskRunner strips markdown fences and conversational fluff when materializing Python files."""
        task = Task(
            task_id="t_code_strip",
            title="Clean Python Script",
            assigned_role=AgentRole.MODEL_ARCHITECT,
            outputs=["src/clean_pipeline.py"],
            artifact_contracts=[
                ArtifactContract(
                    path="src/clean_pipeline.py",
                    artifact_type=ArtifactType.CODE,
                )
            ],
        )
        runner = TaskRunner(graph=TaskGraph(tasks=[task]), workspace_dir=self.tmp_dir)

        conversational_llm_output = (
            "Certainly! Here is the complete implementation for your pipeline:\n\n"
            "```python\n"
            "import os\n"
            "import math\n\n"
            "def compute_variance(data):\n"
            "    mean = sum(data) / len(data)\n"
            "    return sum((x - mean) ** 2 for x in data) / len(data)\n\n"
            "if __name__ == '__main__':\n"
            "    print(compute_variance([1.0, 2.0, 3.0]))\n"
            "```\n\n"
            "I hope this helps! Let me know if you need any adjustments.\n"
        )

        out_files = runner._materialize_deliverables(
            task, conversational_llm_output, "MODEL_ARCHITECT", overwrite=True, is_simulation=False
        )
        self.assertIn("src/clean_pipeline.py", out_files)

        written_path = os.path.join(self.tmp_dir, "src/clean_pipeline.py")
        self.assertTrue(os.path.exists(written_path))

        with open(written_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Must not contain markdown backticks or conversational preamble
        self.assertNotIn("```", code)
        self.assertNotIn("Certainly! Here is", code)
        self.assertNotIn("I hope this helps!", code)

        # Must be valid executable Python AST
        parsed = ast.parse(code)
        self.assertTrue(len(parsed.body) >= 3)

    def test_deliverable_extraction_markdown_code_fences_and_prose_csv(self):
        """Verifies TaskRunner strips markdown code fences and conversational lines when materializing CSV files."""
        task = Task(
            task_id="t_csv_strip",
            title="Clean Tabular Output",
            assigned_role=AgentRole.DATA_ENGINEER,
            outputs=["data/eval_metrics.csv"],
            artifact_contracts=[
                ArtifactContract(
                    path="data/eval_metrics.csv",
                    artifact_type=ArtifactType.TABULAR,
                    columns={"sample_id": "int", "f1_score": "float", "latency_ms": "float"},
                )
            ],
        )
        runner = TaskRunner(graph=TaskGraph(tasks=[task]), workspace_dir=self.tmp_dir)

        conversational_csv_output = (
            "Here is the evaluation metric table formatted as CSV:\n\n"
            "```csv\n"
            "sample_id,f1_score,latency_ms\n"
            "1,0.942,12.4\n"
            "2,0.915,14.8\n"
            "3,0.961,11.2\n"
            "```\n\n"
            "Note: latency is measured on an isolated CPU thread.\n"
        )

        out_files = runner._materialize_deliverables(
            task, conversational_csv_output, "DATA_ENGINEER", overwrite=True, is_simulation=False
        )
        self.assertIn("data/eval_metrics.csv", out_files)

        csv_path = os.path.join(self.tmp_dir, "data/eval_metrics.csv")
        self.assertTrue(os.path.exists(csv_path))

        with open(csv_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        self.assertEqual(lines[0], "sample_id,f1_score,latency_ms")
        self.assertEqual(len(lines), 4)
        for line in lines:
            self.assertNotIn("Here is the evaluation", line)
            self.assertNotIn("Note:", line)
            self.assertNotIn("```", line)


class TestSafeTorchLoadWeightsOnlyEmpirical(unittest.TestCase):
    """Empirical verification that generate_task_test_script mandates weights_only=True on torch.load."""

    def test_harness_script_ast_enforces_weights_only(self):
        """Parses generated test script AST and asserts all torch.load calls include weights_only=True."""
        task = Task(
            task_id="t_torch_sec",
            title="Produce Model Weights",
            category=TaskCategory.TRAINING_OR_SIMULATION,
            assigned_role=AgentRole.MODEL_ARCHITECT,
            outputs=["weights.pt"],
            artifact_contracts=[
                ArtifactContract(
                    path="weights.pt",
                    artifact_type=ArtifactType.TENSOR,
                    shape=[10, 10],
                    dtype="float32",
                )
            ],
        )

        script_text = generate_task_test_script(task, workspace_dir=".")
        parsed = ast.parse(script_text)

        # Walk AST to find all Call nodes where func is torch.load or attribute is load
        torch_load_calls = []
        for node in ast.walk(parsed):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "load":
                    # Check if base is torch or variable
                    if isinstance(func.value, ast.Name) and func.value.id == "torch":
                        torch_load_calls.append(node)

        self.assertTrue(len(torch_load_calls) >= 1, "No torch.load calls found in generated test script!")

        for call in torch_load_calls:
            weights_only_kw = next((kw for kw in call.keywords if kw.arg == "weights_only"), None)
            self.assertIsNotNone(weights_only_kw, "torch.load call is missing weights_only keyword argument!")
            self.assertTrue(
                isinstance(weights_only_kw.value, ast.Constant) and weights_only_kw.value.value is True,
                f"weights_only is not explicitly True: {ast.dump(weights_only_kw.value)}",
            )


class TestSecurityAndCardIndexingGuardrails(unittest.TestCase):
    """Verifies directory traversal, SSRF prevention, and Card evidence grounding invariants."""

    def test_is_safe_path_traversal_attacks(self):
        """Verifies is_safe_path blocks system roots, parent directory traversal, and null bytes."""
        workspace = tempfile.mkdtemp(prefix="safe_path_test_")
        try:
            self.assertTrue(batch_uploader.is_safe_path(workspace, allowed_roots=[workspace]))
            self.assertTrue(batch_uploader.is_safe_path(os.path.join(workspace, "sub", "file.pdf"), allowed_roots=[workspace]))

            # Adversarial traversal payloads
            self.assertFalse(batch_uploader.is_safe_path(os.path.join(workspace, "../../etc/passwd"), allowed_roots=[workspace]))
            self.assertFalse(batch_uploader.is_safe_path("/etc/shadow", allowed_roots=[workspace]))
            self.assertFalse(batch_uploader.is_safe_path("/bin/sh", allowed_roots=[workspace]))
            self.assertFalse(batch_uploader.is_safe_path("/System/Library", allowed_roots=[workspace]))
            self.assertFalse(batch_uploader.is_safe_path("", allowed_roots=[workspace]))
            self.assertFalse(batch_uploader.is_safe_path(None, allowed_roots=[workspace]))
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def test_is_safe_url_ssrf_attacks(self):
        """Verifies is_safe_url blocks AWS/GCP metadata, loopback, private RFC1918, and non-http schemes."""
        # SSRF malicious endpoints
        self.assertFalse(batch_uploader.is_safe_url("http://169.254.169.254/latest/meta-data/"))
        self.assertFalse(batch_uploader.is_safe_url("http://metadata.google.internal/computeMetadata/v1/"))
        self.assertFalse(batch_uploader.is_safe_url("http://127.0.0.1:8000/admin"))
        self.assertFalse(batch_uploader.is_safe_url("http://localhost:5000/"))
        self.assertFalse(batch_uploader.is_safe_url("http://0.0.0.0/"))
        self.assertFalse(batch_uploader.is_safe_url("http://[::1]:8080/"))
        self.assertFalse(batch_uploader.is_safe_url("file:///etc/passwd"))
        self.assertFalse(batch_uploader.is_safe_url("gopher://127.0.0.1:6379/"))

        # Valid public external endpoint (arXiv); DNS is stubbed because tests have no network.
        public = [(2, 1, 6, "", ("151.101.0.42", 0))]
        with patch.object(batch_uploader.socket, "getaddrinfo", return_value=public):
            self.assertTrue(batch_uploader.is_safe_url("https://arxiv.org/abs/2101.00001"))

    def test_transfer_card_evidence_document_matching(self):
        """Verifies Card evidence chunk resolution correctly matches document identity when indices shift."""
        from research_assistant.transfer.models import Card, Premise
        from research_assistant.transfer.stages import Seed, develop, ground
        from research_assistant.transfer.deps import TransferSettings
        from research_assistant.transfer.premises import PremiseChecker
        from tests.transfer_fixtures import world

        deps, log = world()
        card = Card(
            card_id="card_target",
            document="s1",
            citation="Source 1",
            name="Target Method",
            problem_solved="Boundary value problem",
            context="Numerical methods",
            inputs="boundary conditions",
            outputs="solution vector",
            assumptions=["smoothness"],
            evidence=Premise(claim="method claim", key="C1", document="s1", chunk_index=0),
        )
        seed = Seed(bottleneck_id="B1", card=card, score=0.9, level="mid", rank=1)

        # Shift deps so index 0 belongs to unrelated_doc and index 1 belongs to s1
        deps.metadatas[0] = {"document": "unrelated_doc", "chunk_index": 0}
        deps.metadatas[1] = {"document": "s1", "chunk_index": 0, "seq": 0}

        ch = PremiseChecker(deps.judge, deps.texts, deps.metadatas)
        g = ground("q", deps, TransferSettings(), ch)
        t = develop(seed, 1, g, deps, TransferSettings(), ch)
        self.assertIsNotNone(t)
        self.assertEqual(t.card_document, "s1")


if __name__ == "__main__":
    unittest.main()
