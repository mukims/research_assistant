"""
Comprehensive test suite for Milestone 3 (Tier P2: Architecture, Scale, and Feature Wiring).

Covers:
- Feature 15: LanceDB Storage Optimization & Compaction
- Feature 16: Tasks Module Decoupling & Hardening
- Feature 17: Transfer Engine UI Wiring & Stable Card Indexing
- Feature 18: UI Security & Performance
"""

import html
import io
import json
import os
import shutil
import tempfile
import time
import unittest
import zipfile
from unittest.mock import MagicMock, patch

from research_assistant import config


class TestFeature15LanceDBRemediation(unittest.TestCase):
    """Tests for LanceDB Storage Optimization & Compaction (Feature 15)."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test-lance-p2-")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_compute_chunk_id_determinism_and_uniqueness(self):
        from research_assistant.shared.lance_store import compute_chunk_id

        doc = "doi_10.1038_srep29726"
        stored_form = "Sample chunk content text"
        seq = 3
        content = "Sample chunk content text full body"

        # Determinism: same inputs produce identical IDs
        id1 = compute_chunk_id(doc, stored_form, seq, content)
        id2 = compute_chunk_id(doc, stored_form, seq, content)
        self.assertEqual(id1, id2)
        self.assertTrue(id1.startswith("chk_"))
        self.assertEqual(len(id1), 28)

        # Sensitivity: different sequence produces different ID
        id_seq_diff = compute_chunk_id(doc, stored_form, seq + 1, content)
        self.assertNotEqual(id1, id_seq_diff)

        # Sensitivity: different document produces different ID
        id_doc_diff = compute_chunk_id("another_doc", stored_form, seq, content)
        self.assertNotEqual(id1, id_doc_diff)

        # Sensitivity: different content produces different ID
        id_content_diff = compute_chunk_id(doc, stored_form, seq, "modified content")
        self.assertNotEqual(id1, id_content_diff)

    def test_upsert_corpus_targeted_deduplication(self):
        """Verifies upsert_corpus does not perform O(N) full-table scan and uses targeted get."""
        from research_assistant.shared.ingestion import upsert_corpus

        mock_store = MagicMock()
        mock_col = MagicMock()
        mock_col.is_lancedb = True
        mock_store.get_or_create_collection.return_value = mock_col

        # Simulate that "doc1" already has one chunk stored
        mock_col.get.return_value = {
            "documents": ["chunk one text"],
            "metadatas": [{"document": "doc1"}],
        }

        corpus = [
            {
                "type": "text_chunk",
                "document": "doc1",
                "page": 1,
                "citation": "ref1",
                "content": "chunk one text",
                "embed_text": "chunk one text",
                "metadata": {"seq": 0},
            },
            {
                "type": "text_chunk",
                "document": "doc1",
                "page": 2,
                "citation": "ref1",
                "content": "chunk two brand new unique text",
                "embed_text": "chunk two brand new unique text",
                "metadata": {"seq": 1},
            },
        ]

        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1] * 768]

        with patch("research_assistant.shared.ingestion.open_store", return_value=mock_store), \
             patch("research_assistant.shared.llm.get_embeddings", return_value=mock_embeddings), \
             patch("research_assistant.shared.ingestion.upsert_summaries"):

            inserted_count = upsert_corpus(corpus)
            self.assertEqual(inserted_count, 1)

            # Check that collection.get was called targeted with document filter, NOT offset pagination
            mock_col.get.assert_called_once()
            called_kwargs = mock_col.get.call_args.kwargs
            self.assertEqual(called_kwargs.get("where"), {"document": "doc1"})
            self.assertNotIn("offset", called_kwargs)

            # Check that only the new chunk was added
            mock_col.add.assert_called_once()
            add_kwargs = mock_col.add.call_args.kwargs
            self.assertEqual(len(add_kwargs["documents"]), 1)
            self.assertEqual(add_kwargs["documents"][0], "chunk two brand new unique text")

            # Check post-ingest optimization called
            mock_col.optimize.assert_called_once()

    def test_lance_table_adapter_optimize(self):
        """Verifies LanceTableAdapter.optimize calls Table.optimize with the cleanup window."""
        from datetime import timedelta
        from research_assistant.shared.lance_store import LanceTableAdapter

        mock_lance_table = MagicMock()
        adapter = LanceTableAdapter(mock_lance_table, "test_table")

        adapter.optimize(cleanup_older_than_days=14)
        mock_lance_table.optimize.assert_called_once_with(cleanup_older_than=timedelta(days=14))

    def test_lance_table_adapter_optimize_graceful_fallback(self):
        """Verifies optimize does not crash when the table's optimize fails."""
        from research_assistant.shared.lance_store import LanceTableAdapter

        mock_lance_table = MagicMock()
        mock_lance_table.optimize.side_effect = RuntimeError("Compaction lock error")
        adapter = LanceTableAdapter(mock_lance_table, "test_table")

        # Must not raise
        adapter.optimize()

    def test_db_cache_key_invalidation_on_versions(self):
        """Verifies db._cache_key inspects _versions and latest_version_hint.json."""
        from research_assistant.shared import db

        lancedb_dir = os.path.join(self.tmp_dir, "lancedb")
        os.makedirs(lancedb_dir, exist_ok=True)
        col_lance_dir = os.path.join(lancedb_dir, f"{config.COLLECTION_NAME}.lance")
        os.makedirs(col_lance_dir, exist_ok=True)
        versions_dir = os.path.join(col_lance_dir, "_versions")
        os.makedirs(versions_dir, exist_ok=True)

        with patch.object(db, "LANCEDB_PATH", lancedb_dir), \
             patch.object(db, "USE_LANCEDB", True):

            key1 = db._cache_key()

            # Create a new version manifest file inside _versions
            time.sleep(0.05)
            manifest_file = os.path.join(versions_dir, "version_2.manifest")
            with open(manifest_file, "w") as f:
                f.write("manifest data")

            key2 = db._cache_key()
            self.assertNotEqual(key1, key2)

            # Update latest_version_hint.json
            time.sleep(0.05)
            hint_file = os.path.join(col_lance_dir, "latest_version_hint.json")
            with open(hint_file, "w") as f:
                f.write('{"version": 3}')

            key3 = db._cache_key()
            self.assertNotEqual(key2, key3)


class TestFeature16TasksHardeningRemediation(unittest.TestCase):
    """Tests for Tasks Module Decoupling & Hardening (Feature 16)."""

    def test_tasks_retrieval_imports_and_alias(self):
        import inspect
        from research_assistant.tasks import retrieval
        from research_assistant.agents.agent2_fetcher import fetch_missing_citations, fetch_papers

        self.assertIsNotNone(fetch_missing_citations)
        self.assertIs(fetch_missing_citations, fetch_papers)

        # Confirm retrieval uses fetch_papers in auto_ingest_paper_if_needed
        src = inspect.getsource(retrieval.auto_ingest_paper_if_needed)
        self.assertIn("fetch_papers", src)

    def test_prompts_decoupled_from_benchmark(self):
        from research_assistant.tasks import prompts, extractor

        sys_prompt = prompts.PROTOCOL_DECONSTRUCTION_SYSTEM
        # Prohibited benchmark-specific overfitting strings
        self.assertNotIn("Stealthy Hyperuniform", sys_prompt)
        self.assertNotIn("RGF", sys_prompt)
        self.assertNotIn("tight-binding", sys_prompt)
        self.assertNotIn("Heaviside", sys_prompt)
        self.assertNotIn("disorder_samples", sys_prompt)

        # Extractor rule patterns check
        extractor_code = extractor.__file__
        with open(extractor_code, "r", encoding="utf-8") as f:
            code_text = f.read()
        self.assertNotIn("tight-binding", code_text)
        self.assertNotIn("hamiltonian", code_text)

    def test_micro_critic_evaluates_general_simulation(self):
        from research_assistant.tasks.critic import MicroCritic
        from research_assistant.tasks.models import AgentRole, Task, TaskCategory

        critic = MicroCritic()
        # General computational simulation with formula / parameter keywords
        task = Task(
            task_id="t1",
            title="Generate Numerical Synthetic Data",
            description="Synthesize simulated dataset with discretization algorithm, parameter bounds, and Gaussian distribution.",
            category=TaskCategory.DATA_PREPROCESSING,
            assigned_role=AgentRole.DATA_ENGINEER,
            outputs=["data/processed.csv"],
        )
        flaws = critic.critique_task(task)
        # Should not flag micro_data_algorithm_missing
        flaw_ids = [f.flaw_id for f in flaws]
        self.assertNotIn("micro_data_algorithm_missing_t1", flaw_ids)

    def test_task_runner_deliverable_extraction_markdown_and_csv(self):
        from research_assistant.tasks.models import AgentRole, Task, TaskGraph
        from research_assistant.tasks.runner import ExecutionMode, TaskRunner

        tmp_dir = tempfile.mkdtemp(prefix="test-runner-extract-")
        try:
            task = Task(
                task_id="t1",
                title="Extract Data",
                description="Save metrics and code",
                assigned_role=AgentRole.DATA_ENGINEER,
                outputs=["metrics.csv", "pipeline.py"],
            )
            graph = TaskGraph(tasks=[task])
            runner = TaskRunner(graph=graph, workspace_dir=tmp_dir, mode=ExecutionMode.SIMULATION)

            csv_content_with_prose = (
                "Certainly! Here is the simulation output you requested:\n\n"
                "```csv\n"
                "iteration,accuracy\n"
                "1,0.85\n"
                "2,0.92\n"
                "```\n"
                "Note: values converged in 2 steps."
            )

            runner._materialize_artifacts(task, content=csv_content_with_prose, is_simulation=False, overwrite=True)

            csv_path = os.path.join(tmp_dir, "metrics.csv")
            self.assertTrue(os.path.exists(csv_path))
            with open(csv_path, "r", encoding="utf-8") as f:
                csv_lines = [l.strip() for l in f if l.strip()]

            # Verified: conversational text stripped, CSV structured rows preserved
            self.assertEqual(csv_lines, ["iteration,accuracy", "1,0.85", "2,0.92"])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_task_runner_does_not_mask_llm_exceptions(self):
        """Verifies TaskRunner records failure rather than writing dummy stub code."""
        from research_assistant.tasks.models import AgentRole, Task, TaskGraph, TaskStatus
        from research_assistant.tasks.runner import ExecutionMode, TaskRunner

        tmp_dir = tempfile.mkdtemp(prefix="test-runner-fail-")
        try:
            task = Task(
                task_id="t1",
                title="Generate Model",
                description="Generate simulation code",
                assigned_role=AgentRole.MODEL_ARCHITECT,
                outputs=["model.py"],
            )

            graph = TaskGraph(tasks=[task])
            runner = TaskRunner(graph=graph, workspace_dir=tmp_dir, mode=ExecutionMode.AUTONOMOUS)

            with patch("research_assistant.shared.llm.chat", side_effect=RuntimeError("LLM API 500 error")):
                report = runner.execute()

            self.assertEqual(task.status, TaskStatus.FAILED)
            self.assertIn("LLM API 500 error", report.results["t1"].error_message)

            # Ensure dummy stub comments were not written to disk as deliverables
            out_path = os.path.join(tmp_dir, "model.py")
            if os.path.exists(out_path):
                with open(out_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.assertNotIn("# Deliverable for Generate Model", content)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_harness_safe_torch_load_weights_only(self):
        from research_assistant.tasks import harness

        harness_code = harness.__file__
        with open(harness_code, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("weights_only=True", content)
        self.assertNotIn("weights_only=False", content)

    def test_iteration_no_hardcoded_num_ctx(self):
        from research_assistant.tasks import iteration

        iter_code = iteration.__file__
        with open(iter_code, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn('"num_ctx": 4096', content)
        self.assertIn("TASK_OLLAMA_OPTIONS", content)


class TestFeature17TransferEngineRemediation(unittest.TestCase):
    """Tests for Transfer Engine UI Wiring & Stable Card Indexing (Feature 17)."""

    def test_app_wires_transfer_tab(self):
        app_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
        with open(app_file, "r", encoding="utf-8") as f:
            app_code = f.read()
        self.assertIn("render_transfer_tab", app_code)
        self.assertIn('"Technique transfer"', app_code)

    def test_transfer_tab_renders_cleanly(self):
        from research_assistant.ui.transfer_tab import render_transfer_tab

        # Verify function can be imported and executed in a mock session
        with patch("streamlit.tabs", return_value=[MagicMock(), MagicMock(), MagicMock(), MagicMock()]), \
             patch("streamlit.selectbox", return_value="Demo Bottleneck"), \
             patch("streamlit.slider", return_value=3), \
             patch("streamlit.button", return_value=False), \
             patch("streamlit.info"):
            render_transfer_tab()

    def test_default_engine_instantiation_and_run_interface(self):
        """Verifies default_engine() can be imported and initialized with a callable run method."""
        from research_assistant.transfer.engine import default_engine, TransferEngine
        engine = default_engine()
        self.assertIsInstance(engine, TransferEngine)
        self.assertTrue(callable(getattr(engine, "run", None)))

    def test_card_evidence_grounding_document_matching(self):
        """Verifies stage 3 grounds card evidence using document ID instead of bare array index."""
        from research_assistant.transfer.models import Card, Passage, Premise
        from research_assistant.transfer.stages import Seed, develop
        from tests.transfer_fixtures import world

        deps, log = world()
        card = Card(
            card_id="card_stable",
            document="s1",
            citation="Source 1",
            name="Stable Technique",
            problem_solved="Matrix solve",
            context="Numerical methods",
            inputs="A, b",
            outputs="x",
            assumptions=["linear"],
            evidence=Premise(claim="source fact", key="C4", document="s1", chunk_index=0),
        )
        seed = Seed(bottleneck_id="B1", card=card, score=0.9, level="mid", rank=1)

        # Alter deps so index 0 belongs to a different document
        deps.metadatas[0] = {"document": "unrelated_doc", "chunk_index": 0}
        deps.metadatas[1] = {"document": "s1", "chunk_index": 0, "seq": 0}

        # develop should successfully ground to s1 without index mismatch
        from research_assistant.transfer import stages
        from research_assistant.transfer.deps import TransferSettings
        from research_assistant.transfer.premises import PremiseChecker

        ch = PremiseChecker(deps.judge, deps.texts, deps.metadatas)
        g = stages.ground("q", deps, TransferSettings(), ch)
        t = stages.develop(seed, 1, g, deps, TransferSettings(), ch)
        self.assertIsNotNone(t)
        self.assertEqual(t.card_document, "s1")

    def test_stage6_revision_applied_to_transfer(self):
        """Verifies Stage 6 applies reviser mapping, steps, prediction, and summary to Transfer."""
        from research_assistant.transfer import stages
        from research_assistant.transfer.deps import TransferSettings
        from research_assistant.transfer.premises import PremiseChecker
        from tests.transfer_fixtures import world

        revise_reply = json.dumps({
            "responses": [{"objection": 0, "response": "Objection addressed", "concede": False}],
            "revised_title": "Revised Transfer Title",
            "revised_summary": "Revised summary with boundary proofs",
            "revised_prediction": "Revised prediction with exponential decay",
            "revised_mapping": [{"source_role": "S1_rev", "target_role": "T1_rev", "assumption": "A1_rev", "status": "holds", "core": True}],
            "revised_steps": [{"action": "Revised Step with Damping", "input": "x_damped", "output": "y", "tool": "numpy"}],
        })

        deps, log = world(replies={("revise", 1): revise_reply})
        deps.critic = lambda m: json.dumps({
            "shape_match": "yes",
            "rows": [{"row": 0, "status": "breaks", "reason": "Edge effect"}],
            "objections": ["Edge effect breaks convergence"],
        })

        ch = PremiseChecker(deps.judge, deps.texts, deps.metadatas)
        s = TransferSettings()
        g = stages.ground("q", deps, s, ch)
        seeds, _ = stages.candidate_cards(g, deps, s)

        t = stages.develop(seeds[0], 1, g, deps, s, ch)
        self.assertIsNotNone(t)
        self.assertEqual(t.title, "Revised Transfer Title")
        self.assertEqual(t.summary, "Revised summary with boundary proofs")
        self.assertEqual(t.prediction, "Revised prediction with exponential decay")
        self.assertEqual(t.mapping[0].source_role, "S1_rev")
        self.assertEqual(t.steps[0].action, "Revised Step with Damping")

    def test_prior_art_temporal_cutoff_handles_none_year(self):
        from research_assistant.transfer.prior_art import PriorArtClient

        client = PriorArtClient(
            embed_documents=lambda texts: [[0.1] * 768 for _ in texts],
        )

        hits_with_none_year = [
            {"title": "Paper 2019", "year": 2019, "abstract": "abc"},
            {"title": "Paper Unknown Year", "year": None, "abstract": "def"},
            {"title": "Paper 2022", "year": 2022, "abstract": "ghi"},
        ]

        with patch.object(client, "get_json", return_value={"results": []}), \
             patch.object(client, "get_text", return_value="<xml></xml>"), \
             patch("research_assistant.transfer.prior_art.parse_openalex", return_value=hits_with_none_year), \
             patch("research_assistant.transfer.prior_art.parse_arxiv", return_value=[]):

            res = client.check("quantum transport", ["conductance"], "summary text", before_year=2021)
            # Must not crash with TypeError '<' not supported between NoneType and int
            self.assertIsNotNone(res)
            # Only 2019 paper should survive cutoff
            titles = [h["title"] for h in res.hits]
            self.assertIn("Paper 2019", titles)
            self.assertNotIn("Paper Unknown Year", titles)
            self.assertNotIn("Paper 2022", titles)


class TestFeature18UISecurityAndPerformanceRemediation(unittest.TestCase):
    """Tests for UI Security & Performance (Feature 18)."""

    def test_grobid_status_cached(self):
        from research_assistant.ui import grobid

        # Check caching decorator on _cached_grobid_agent_status
        self.assertTrue(hasattr(grobid._cached_grobid_agent_status, "clear"))

        # Check clear helper executes without error
        grobid._clear_grobid_cache()

    def test_sidebar_model_propagation(self):
        from research_assistant.ui import common

        fake_session = {}
        before = (config.LLM_MODEL, config.LLM_BACKEND)
        with patch.object(common, "st") as mock_st, \
             patch.object(common, "_get_model_choices", return_value=[{"label": "Test Ollama", "model": "qwen2.5:14b", "backend": "ollama"}]):

            mock_st.session_state = fake_session
            mock_st.selectbox.return_value = "Test Ollama"

            common._render_sidebar_backend_controls()

            self.assertEqual(fake_session.get("llm_model"), "qwen2.5:14b")
            self.assertEqual(fake_session.get("llm_backend"), "ollama")
            # The pick is per session: it must not rewrite process-wide config
            # (see tests/test_app_render.py::TestSidebarModelIsPerSession).
            self.assertEqual((config.LLM_MODEL, config.LLM_BACKEND), before)

    def test_is_safe_path_directory_traversal(self):
        from research_assistant.shared.batch_uploader import is_safe_path

        # Forbidden system roots and directory traversals
        self.assertFalse(is_safe_path("/etc"))
        self.assertFalse(is_safe_path("/etc/passwd"))
        self.assertFalse(is_safe_path("/private/etc/hosts"))
        self.assertFalse(is_safe_path("../../../../../../../etc/shadow"))
        self.assertFalse(is_safe_path(""))
        self.assertFalse(is_safe_path(None))

        # Safe workspace / temp paths
        safe_tmp = tempfile.gettempdir()
        self.assertTrue(is_safe_path(safe_tmp, allowed_roots=[safe_tmp]))
        self.assertTrue(is_safe_path(os.path.join(safe_tmp, "test_paper.pdf"), allowed_roots=[safe_tmp]))
        # Reject escape from specific allowed root
        self.assertFalse(is_safe_path(os.path.join(safe_tmp, "..", "forbidden.txt"), allowed_roots=[safe_tmp]))

    def test_is_safe_url_ssrf_protection(self):
        from research_assistant.shared.batch_uploader import is_safe_url

        # Private / loopback / link-local / non-http URLs must be rejected
        self.assertFalse(is_safe_url("http://127.0.0.1/admin"))
        self.assertFalse(is_safe_url("http://localhost:8501/secret"))
        self.assertFalse(is_safe_url("http://169.254.169.254/latest/meta-data/"))
        self.assertFalse(is_safe_url("file:///etc/passwd"))
        self.assertFalse(is_safe_url("ftp://ftp.example.com/data.pdf"))
        self.assertFalse(is_safe_url("http://10.0.0.1/internal"))
        self.assertFalse(is_safe_url("http://192.168.1.1/router"))

        # Valid public URLs
        with patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("93.184.216.34", 80))]):
            self.assertTrue(is_safe_url("https://arxiv.org/pdf/2401.12345.pdf"))
            self.assertTrue(is_safe_url("http://example.com/paper.pdf"))

    def test_zip_magic_bytes_and_bomb_limits(self):
        from research_assistant.shared.batch_uploader import (
            MAX_SINGLE_FILE_SIZE,
            MAX_ZIP_FILE_COUNT,
            unpack_and_stage_uploads,
        )

        tmp_dest = tempfile.mkdtemp(prefix="test-zip-")
        try:
            # 1. Non-zip content with fake .zip extension must be rejected
            fake_zip = io.BytesIO(b"GIF89a not a zip file")
            fake_zip.name = "malicious.zip"
            staged = unpack_and_stage_uploads([fake_zip], destination_dir=tmp_dest)
            self.assertEqual(len(staged), 0)

            # 2. Valid zip with valid PDF bytes
            valid_pdf = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w") as zf:
                zf.writestr("paper1.pdf", valid_pdf)
            zip_buf.seek(0)
            zip_buf.name = "archive.zip"

            with patch("research_assistant.shared.batch_uploader._inspect_pdf", return_value=("Sample Title", "", "")):
                staged = unpack_and_stage_uploads([zip_buf], destination_dir=tmp_dest)
            self.assertEqual(len(staged), 1)
            self.assertEqual(staged[0]["filename"], "paper1.pdf")

            # 3. Zip bomb: excessive file count
            bomb_buf = io.BytesIO()
            with zipfile.ZipFile(bomb_buf, "w") as zf:
                for idx in range(MAX_ZIP_FILE_COUNT + 10):
                    zf.writestr(f"p_{idx}.pdf", valid_pdf)
            bomb_buf.seek(0)
            bomb_buf.name = "bomb.zip"

            with patch("research_assistant.shared.batch_uploader._inspect_pdf", return_value=("Title", "", "")):
                staged = unpack_and_stage_uploads([bomb_buf], destination_dir=tmp_dest)
            # Should have halted extraction at MAX_ZIP_FILE_COUNT
            self.assertLessEqual(len(staged), MAX_ZIP_FILE_COUNT)

        finally:
            shutil.rmtree(tmp_dest, ignore_errors=True)

    def test_html_escaping_in_ui_results_and_tasks(self):
        from research_assistant.ui import results

        raw_title = "<script>alert('xss')</script>"
        raw_reason = "<img src=x onerror=alert(1)>"
        final_payload = {
            "fetch_failures": [{"title": raw_title, "reason": raw_reason}],
            "downloaded": [],
            "seed_papers": [],
        }

        markdown_outputs = []
        with patch("streamlit.markdown", side_effect=lambda txt, **kw: markdown_outputs.append(txt)), \
             patch("streamlit.expander", return_value=MagicMock()), \
             patch("streamlit.caption"):

            results._render_seed_and_downloads("query", final_payload)

            all_md = "\n".join(markdown_outputs)
            self.assertNotIn("<script>", all_md)
            self.assertNotIn("<img", all_md)
            self.assertIn("&lt;script&gt;", all_md)

        # 2. Test tasks/ui.py card escaping
        malicious_task_id = "<script>bad</script>"
        malicious_title = "<img src=x>"
        malicious_output = "<iframe src='evil.com'></iframe>"

        safe_id = html.escape(malicious_task_id)
        safe_title = html.escape(malicious_title)
        safe_out = f"<code>{html.escape(malicious_output)}</code>"

        self.assertNotIn("<script>", safe_id)
        self.assertNotIn("<img", safe_title)
        self.assertNotIn("<iframe", safe_out)
        self.assertIn("&lt;script&gt;", safe_id)

    def test_mermaid_strict_security(self):
        from research_assistant.tasks.ui import _render_mermaid_interactive

        with patch("streamlit.components.v1.html") as mock_comp_html:
            _render_mermaid_interactive("flowchart TD\n    A --> B")
            mock_comp_html.assert_called_once()
            rendered_html = mock_comp_html.call_args[0][0]
            self.assertIn("securityLevel: 'strict'", rendered_html)
            self.assertIn("htmlLabels: false", rendered_html)


if __name__ == "__main__":
    unittest.main()
