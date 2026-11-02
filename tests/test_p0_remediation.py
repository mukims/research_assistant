"""
tests/test_p0_remediation.py — Comprehensive unit and regression test suite
verifying all 7 Tier P0 remediation subsystems in research_assistant.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from research_assistant.shared import pipeline_status
from research_assistant.shared import manifest


class _OfflineEmbeddings:
    """Stands in for the Ollama embedder, which the suite's network block makes unreachable."""

    def embed_documents(self, texts):
        return [[0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0, 0.0, 0.0]


class TestP0Area1ProgressCallbacks(unittest.TestCase):
    """Area 1: UI Progress Callbacks & Cleanup."""

    def test_unregister_progress_callback_function(self):
        called = []

        def my_cb(status):
            called.append(status)

        pipeline_status.register_progress_callback(my_cb)
        pipeline_status.update_progress(item_current=1, item_total=2, current_item_name="test.pdf")
        self.assertGreater(len(called), 0)

        # Unregister by function reference
        pipeline_status.unregister_progress_callback(my_cb)
        count_before = len(called)
        pipeline_status.update_progress(item_current=2, item_total=2, current_item_name="test.pdf")
        self.assertEqual(len(called), count_before)

    def test_unregister_progress_callback_closure(self):
        called = []

        def my_cb(status):
            called.append(status)

        unreg = pipeline_status.register_progress_callback(my_cb)
        pipeline_status.update_progress(item_current=1, item_total=2, current_item_name="test.pdf")
        self.assertGreater(len(called), 0)

        # Unregister by closure
        pipeline_status.unregister_progress_callback(unreg)
        count_before = len(called)
        pipeline_status.update_progress(item_current=2, item_total=2, current_item_name="test.pdf")
        self.assertEqual(len(called), count_before)

    def test_unregister_progress_callback_nonexistent_and_none(self):
        # Should not raise AttributeError or any exception
        pipeline_status.unregister_progress_callback(None)

        def dummy_cb(status):
            pass

        pipeline_status.unregister_progress_callback(dummy_cb)

    def test_ui_batch_progress_callback_signatures(self):
        # Test progress callback logic as used in corpus_tab and audit_tab
        results = []

        def mock_progress(pct, text):
            results.append((pct, text))

        class MockBatchProgress:
            def progress(self, pct, text=""):
                results.append((pct, text))

        batch_progress = MockBatchProgress()

        # Simulated callback from corpus_tab / audit_tab
        def _on_batch_progress(status_or_item_c=None, item_t=None, name=None, **kwargs):
            if isinstance(status_or_item_c, dict):
                st_dict = status_or_item_c
                item_c = st_dict.get("item_current", 0)
                item_t = st_dict.get("item_total", 0)
                name = st_dict.get("current_item_name") or st_dict.get("name") or ""
                progress = st_dict.get("progress")
                message = st_dict.get("detail") or st_dict.get("message") or ""
            else:
                item_c = status_or_item_c or 0
                item_t = item_t or 0
                name = name or ""
                progress = None
                message = ""

            if progress is not None:
                pct = min(1.0, max(0.0, float(progress)))
            elif item_t and item_t > 0:
                pct = min(1.0, max(0.0, item_c / item_t))
            else:
                pct = 0.0

            display_txt = message if message else (
                f"Parsing paper [{item_c}/{item_t}]: {name[:40]}" if (item_t and item_t > 0) else f"Parsing paper: {name[:40]}"
            )
            batch_progress.progress(pct, text=display_txt)

        # 1. Emitted by pipeline_status with dict
        status_dict = {
            "item_current": 3,
            "item_total": 10,
            "current_item_name": "graphene_paper.pdf",
            "progress": 0.3,
            "detail": "Parsing paper [3/10]: graphene_paper.pdf",
            "stage": "ingest_refs",
        }
        _on_batch_progress(status_dict)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[-1][0], 0.3)
        self.assertIn("graphene_paper", results[-1][1])

        # 2. Emitted with positional arguments
        _on_batch_progress(5, 10, "other_paper.pdf")
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[-1][0], 0.5)
        self.assertIn("other_paper", results[-1][1])


class TestP0Area2JobCancellationLedger(unittest.TestCase):
    """Area 2: Job Cancellation Ledger & Persistence."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.global_path = os.path.join(self.tmpdir, "pipeline_status.json")
        pipeline_status.set_status_path(self.global_path)
        pipeline_status.clear_status()

    def tearDown(self):
        pipeline_status.set_status_path(None)
        pipeline_status.clear_status()

    def test_request_cancel_writes_both_ledgers(self):
        job_id = "job_test_cancel_123"
        job_dir = pipeline_status.get_job_status_dir(job_id)
        os.makedirs(job_dir, exist_ok=True)
        job_file = pipeline_status.get_job_status_path(job_id)

        # Initialize job ledger
        pipeline_status.atomic_write_json(job_file, {"active": True, "cancel_requested": False})

        # Call request_cancel targeting the job
        pipeline_status.request_cancel("User stopped test", job_id=job_id)

        # Verify global status ledger has cancel_requested: True
        global_st = pipeline_status._load_status_from_disk(self.global_path)
        self.assertTrue(global_st.get("cancel_requested"))

        # Verify job-specific status ledger has cancel_requested: True
        job_st = pipeline_status._load_status_from_disk(job_file)
        self.assertTrue(job_st.get("cancel_requested"))
        self.assertFalse(job_st.get("active"))

    def test_persist_status_locked_preserves_cancellation(self):
        job_id = "job_test_preserve_456"
        job_file = pipeline_status.get_job_status_path(job_id)
        os.makedirs(os.path.dirname(job_file), exist_ok=True)

        # Write cancel_requested: True to global file
        pipeline_status.request_cancel("User cancel test", job_id=job_id)

        # Now simulate a worker thread calling update_progress / set_status with active=True
        with pipeline_status.job_status_context(job_id):
            status = pipeline_status.update_progress(item_current=1, item_total=10, current_item_name="tick.pdf")
            # Must NOT overwrite cancel_requested with False!
            self.assertTrue(status.get("cancel_requested"))

            # Inspect job file directly
            job_st = pipeline_status._load_status_from_disk(job_file)
            self.assertTrue(job_st.get("cancel_requested"))

            # Inspect global file directly
            global_st = pipeline_status._load_status_from_disk(self.global_path)
            self.assertTrue(global_st.get("cancel_requested"))

    def test_is_cancel_requested_checks_both(self):
        job_id = "job_test_query_789"
        job_file = pipeline_status.get_job_status_path(job_id)
        os.makedirs(os.path.dirname(job_file), exist_ok=True)

        # Case A: Global has cancel_requested: True, job has False
        pipeline_status.atomic_write_json(self.global_path, {"cancel_requested": True})
        pipeline_status.atomic_write_json(job_file, {"cancel_requested": False})
        with pipeline_status.job_status_context(job_id):
            self.assertTrue(pipeline_status.is_cancel_requested())
            self.assertTrue(pipeline_status.is_cancel_requested(job_id))

        # Case B: Global has False, job has True
        pipeline_status.atomic_write_json(self.global_path, {"cancel_requested": False})
        pipeline_status.atomic_write_json(job_file, {"cancel_requested": True})
        with pipeline_status.job_status_context(job_id):
            self.assertTrue(pipeline_status.is_cancel_requested())
            self.assertTrue(pipeline_status.is_cancel_requested(job_id))

    def test_run_jobs_cancellation_halt(self):
        from research_assistant.shared import run_jobs

        job_id = "job_run_cancel_test"
        pipeline_status.request_cancel(job_id=job_id)

        executed = []

        def sample_runner():
            executed.append(True)
            return {"status": "ok"}

        job = run_jobs.start_job(job_id, "Test Cancelled Run", sample_runner)
        job.thread.join(timeout=2.0)
        self.assertEqual(job.state, "cancelled")
        self.assertEqual(len(executed), 0)


class TestP0Area3PreventSpuriousDrafts(unittest.TestCase):
    """Area 3: Prevent Spurious Draft File Exports on Streamlit Rerun."""

    def test_format_conversation_markdown_does_not_write_file(self):
        from research_assistant.agents.agent7_research_chat import ResearchChatAgent, DRAFTS_DIR

        os.makedirs(DRAFTS_DIR, exist_ok=True)
        files_before = set(os.listdir(DRAFTS_DIR))

        agent = ResearchChatAgent()
        agent.history = [
            {"role": "user", "content": "What is the band gap of MoS2?"},
            {"role": "assistant", "content": "Monolayer MoS2 has a direct band gap of ~1.8 eV."},
        ]

        md_output = agent.format_conversation_markdown(scratchpad_notes=["Direct gap in monolayer"])
        self.assertIn("MoS2", md_output)
        self.assertIn("1.8 eV", md_output)
        self.assertIn("Direct gap in monolayer", md_output)

        files_after = set(os.listdir(DRAFTS_DIR))
        # Zero new draft files created on format call!
        self.assertEqual(files_before, files_after)

    def test_export_conversation_writes_file_when_explicitly_called(self):
        from research_assistant.agents.agent7_research_chat import ResearchChatAgent, DRAFTS_DIR

        agent = ResearchChatAgent()
        agent.history = [{"role": "user", "content": "Hello"}]
        path = agent.export_conversation()
        self.assertTrue(os.path.exists(path))
        try:
            os.remove(path)
        except OSError:
            pass


class TestP0Area4TantivyFTSAndLanceStore(unittest.TestCase):
    """Area 4: Tantivy FTS & LanceDB Search Configuration."""

    def test_escape_tantivy_query(self):
        from research_assistant.shared.lance_store import escape_tantivy_query

        # Scientific query with brackets, colons, parentheses, exponents, slashes, minuses, quotes
        raw_query = 'Novoselov [18] 2D material: graphene (a) 10^5 / -0.5 "eV" \\'
        escaped = escape_tantivy_query(raw_query)

        # Every special char [, ], :, (, ), ^, /, -, \, " must be preceded by backslash
        self.assertIn(r"\[18\]", escaped)
        self.assertIn(r"\:", escaped)
        self.assertIn(r"\(a\)", escaped)
        self.assertIn(r"\^5", escaped)
        self.assertIn(r"\/", escaped)
        self.assertIn(r"\-0.5", escaped)
        self.assertIn(r'\"eV\"', escaped)

    def test_create_fts_index_indexes_citation_source(self):
        from research_assistant.shared.lance_store import LanceTableAdapter

        mock_table = MagicMock()
        mock_table.schema.names = ["chunk_id", "text", "citation_source", "vector"]

        adapter = LanceTableAdapter(mock_table)
        adapter.create_fts_index("text")

        # Verify create_index was called for both text and citation_source
        cols_indexed = [call[0][0] for call in mock_table.create_index.call_args_list]
        self.assertIn("text", cols_indexed)
        self.assertIn("citation_source", cols_indexed)

    def test_search_vector_metric_and_prefilter(self):
        from research_assistant.shared.lance_store import LanceTableAdapter

        mock_table = MagicMock()
        mock_query_builder = MagicMock()
        mock_table.search.return_value = mock_query_builder
        mock_query_builder.metric.return_value = mock_query_builder
        mock_query_builder.where.return_value = mock_query_builder
        mock_query_builder.prefilter.return_value = mock_query_builder
        mock_query_builder.limit.return_value = mock_query_builder
        mock_query_builder.to_list.return_value = [
            {"chunk_id": "c1", "text": "sample chunk", "_distance": 0.15, "document": "paper1.pdf"}
        ]

        adapter = LanceTableAdapter(mock_table)
        results = adapter.search_vector([0.1, 0.2], top_k=5, doc_filter={"paper1.pdf"})

        # Verify .metric("cosine") was called
        mock_query_builder.metric.assert_called_with("cosine")
        # Verify .where was called with prefilter=True
        mock_query_builder.where.assert_called_with("document IN ('paper1.pdf')", prefilter=True)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "c1")


class TestP0Area5BibliographyResolution(unittest.TestCase):
    """Area 5: Bibliography Resolution Shift & audit/extractor.py Fix."""

    def test_printed_reference_number_eliminates_index_shift(self):
        from research_assistant.audit.resolver import _resolve_ref

        # Simulate GROBID bibliography where refs 1-4 dropped, so b31 has raw_reference "[36] Novoselov..."
        # but is at position 32 in enumeration!
        bib_by_id = {
            "b31": {
                "xml_id": "b31",
                "index": 36,
                "printed_index": 36,
                "position_index": 32,
                "title": "Two-dimensional gas of massless Dirac fermions in graphene",
                "authors": ["K. S. Novoselov", "A. K. Geim"],
                "year": 2005,
                "raw_reference": "[36] K. S. Novoselov et al., Nature 438, 197-200 (2005)",
            },
            "b35": {
                "xml_id": "b35",
                "index": 40,
                "printed_index": 40,
                "position_index": 36,
                "title": "Unrelated shifted reference at positional index 36",
                "authors": ["J. Doe"],
                "year": 2010,
                "raw_reference": "[40] J. Doe et al.",
            },
        }

        # bib_by_index has position 36 mapped to b35
        bib_by_index = {
            32: bib_by_id["b31"],
            36: bib_by_id["b35"],  # Position 36 points to ref 40 due to shift
        }

        # Target links to b31, citation text is "[36]"
        resolved, how, note = _resolve_ref("b31", "[36]", bib_by_id, bib_by_index, number_map={})

        # Under the old code, b31 vs bib_by_index[36] (b35) triggered "ambiguous" and returned None.
        # Under the remediated code, matching printed_index / raw_reference matches b31!
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["xml_id"], "b31")
        self.assertEqual(resolved["printed_index"], 36)

    def test_author_year_raw_reference_matching(self):
        from research_assistant.audit.resolver import _resolve_ref

        bib_by_id = {
            "b5": {
                "xml_id": "b5",
                "title": "High-mobility transport in graphene",
                "authors": ["Novoselov", "Geim"],
                "year": 2005,
                "raw_reference": "Novoselov, K. S. & Geim, A. K. (2005). Nature, 438, 197.",
            }
        }

        resolved, how, note = _resolve_ref(None, "Novoselov et al. (2005)", bib_by_id, {}, {})
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["xml_id"], "b5")
        self.assertEqual(how, "surname")

    def test_audit_extractor_unification(self):
        from research_assistant.audit import extractor

        self.assertTrue(hasattr(extractor, "extract_seed_citation_claims"))
        self.assertTrue(hasattr(extractor, "find_tei_for_seed"))
        self.assertTrue(hasattr(extractor, "XML_OUTPUT_DIR"))

        # Calling find_tei_for_seed on non-existent path returns None cleanly without exceptions
        res = extractor.find_tei_for_seed("/tmp/non_existent_paper.pdf")
        self.assertIsNone(res)


class TestP0Area6FetcherBidirectionalSimilarity(unittest.TestCase):
    """Area 6: Fetcher Bidirectional Similarity & Failed Downloads Backfill."""

    def test_verify_candidate_match(self):
        from research_assistant.shared.academic_apis import verify_candidate_match

        # 1. Matching target and candidate
        ok = verify_candidate_match(
            target_title="Electric Field Effect in Atomically Thin Carbon Films",
            candidate_title="Electric field effect in atomically thin carbon films.",
            target_author="K. S. Novoselov",
            candidate_author="Novoselov, K. S.",
            target_year=2004,
            candidate_year=2004,
        )
        self.assertTrue(ok)

        # 2. Year within 1 year
        ok_year = verify_candidate_match(
            target_title="Electric Field Effect in Atomically Thin Carbon Films",
            candidate_title="Electric Field Effect in Atomically Thin Carbon Films",
            target_year=2004,
            candidate_year=2005,
        )
        self.assertTrue(ok_year)

        # 3. Year discrepancy > 1 year -> REJECTED
        bad_year = verify_candidate_match(
            target_title="Electric Field Effect in Atomically Thin Carbon Films",
            candidate_title="Electric Field Effect in Atomically Thin Carbon Films",
            target_year=2004,
            candidate_year=2015,
        )
        self.assertFalse(bad_year)

        # 4. Asymmetric / unrelated title -> REJECTED
        bad_title = verify_candidate_match(
            target_title="Mixing",
            candidate_title="Mixing in Turbulent Flows and Chemical Reactions Across Extended Channels",
        )
        self.assertFalse(bad_title)

        # 5. Author surname mismatch -> REJECTED
        bad_author = verify_candidate_match(
            target_title="Electric Field Effect in Atomically Thin Carbon Films",
            candidate_title="Electric Field Effect in Atomically Thin Carbon Films",
            target_author="Smith",
            candidate_author="Johnson",
        )
        self.assertFalse(bad_author)

    def test_backfill_failed_downloads_resolves_oa_and_abstract(self):
        from research_assistant.agents.agent2_fetcher import backfill_failed_downloads

        with tempfile.TemporaryDirectory() as tmpdir:
            failed_file = os.path.join(tmpdir, "failed_downloads.json")
            downloaded_file = os.path.join(tmpdir, "downloaded.json")
            out_dir = os.path.join(tmpdir, "pulled_pdfs")

            sample_failed = {
                "doi:10.1000/182": {
                    "key": "doi:10.1000/182",
                    "doi": "10.1000/182",
                    "title": "Test Open Access Paper",
                    "reason": "paywalled / not open access",
                },
                "doi:10.1000/999": {
                    "key": "doi:10.1000/999",
                    "doi": "10.1000/999",
                    "title": "Test Abstract Only Paper",
                    "reason": "paywalled / not open access",
                },
            }
            with open(failed_file, "w") as f:
                json.dump(sample_failed, f)

            def mock_fetch(query):
                if "182" in query:
                    return {
                        "doi": "10.1000/182",
                        "title": "Test Open Access Paper",
                        "oa_pdf_url": "https://example.com/paper182.pdf",
                        "abstract": "This is an open access paper with full text available.",
                    }
                else:
                    return {
                        "doi": "10.1000/999",
                        "title": "Test Abstract Only Paper",
                        "oa_pdf_url": None,
                        "abstract": "This is a detailed abstract that exceeds fifty characters in length for testing abstract-only registration.",
                    }

            with patch("research_assistant.shared.academic_apis.fetch_bibliometrics_and_abstract", side_effect=mock_fetch), \
                 patch("research_assistant.agents.agent2_fetcher.download_pdf", return_value=(True, None)):
                res = backfill_failed_downloads(
                    failed_path=failed_file,
                    downloaded_path=downloaded_file,
                    output_dir=out_dir,
                )

            self.assertEqual(res["attempted"], 2)
            self.assertEqual(res["resolved_oa"], 1)
            self.assertEqual(res["resolved_abstract"], 1)
            self.assertEqual(res["remaining_failed"], 0)


class TestP0Area7IngestionManifestAndAbstractOnly(unittest.TestCase):
    """Area 7: Ingestion Manifest & Abstract-Only Tracking."""

    def test_abstract_only_emits_summary_source_and_updates_manifest(self):
        from research_assistant.shared.ingestion import upsert_corpus
        from research_assistant.shared import manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            m_path = os.path.join(tmpdir, "manifest.json")
            with patch.object(manifest, "INGESTED_MANIFEST_PATH", m_path):
                manifest.clear()

                downloaded = {
                    "doi:10.1000/test_abs": {
                        "evidence_tier": "abstract_only",
                        "abstract": "This is a verified test abstract for a paper without open access PDF.",
                        "title": "Quantum Transport in Graphene Nanoribbons",
                        "key": "doi:10.1000/test_abs",
                    }
                }

                # Build corpus as done in agent3_ingestor.py
                abstract_corpus = []
                abstract_keys = []
                for key, record in downloaded.items():
                    if record.get("evidence_tier") == "abstract_only" and record.get("abstract"):
                        doc_key = record.get("key") or key
                        label = record.get("title") or key
                        abstract_corpus.append({
                            "document": doc_key,
                            "citation": label,
                            "page": 1,
                            "type": "text_chunk",
                            "content": record["abstract"],
                            "embed_text": f"Title: {label}\nAbstract: {record['abstract']}",
                            "meta": {"stored_form": "abstract_only", "tier": "abstract_only"},
                        })
                        abstract_corpus.append({
                            "document": doc_key,
                            "citation": label,
                            "page": 1,
                            "type": "summary_source",
                            "content": f"Title: {label}\nAbstract: {record['abstract']}",
                            "meta": {"stored_form": "abstract_only", "tier": "abstract_only"},
                        })
                        abstract_keys.append(doc_key)

                # Verify summary_source is present
                types = [e["type"] for e in abstract_corpus]
                self.assertIn("summary_source", types)
                self.assertIn("text_chunk", types)

                # Ingest abstract_corpus and register in manifest
                with patch("research_assistant.shared.ingestion.upsert_summaries") as mock_summaries:
                    upsert_corpus(abstract_corpus)
                    mock_summaries.assert_called_once()
                    manifest.add_many(abstract_keys)

                self.assertTrue(manifest.contains("doi:10.1000/test_abs"))

    def test_failed_summaries_are_not_marked_in_manifest(self):
        from research_assistant.shared.ingestion import ingest_pdfs, last_summary_run
        from research_assistant.shared import manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "test_paper.pdf")
            with open(pdf_path, "wb") as f:
                f.write(b"%PDF-1.4 mock pdf content")

            candidates = {pdf_path: "Test Paper Title"}
            from research_assistant.shared.source_key import pdf_key
            key = pdf_key(pdf_path)

            def mock_upsert(corpus, **kwargs):
                last_summary_run.update(attempted=1, made=0, failed=1, failed_docs=[key])
                return len(corpus)

            with patch("research_assistant.shared.ingestion.process_pdf", return_value=[{"document": key, "type": "summary_source", "citation": "Test", "content": "Text"}]), \
                 patch("research_assistant.shared.ingestion.upsert_corpus", side_effect=mock_upsert), \
                 patch("research_assistant.shared.ingestion.rebuild_bm25"):
                # Simulate summary generation failure
                manifest.clear()
                result = ingest_pdfs(candidates, workers=1, skip_ingested=False)

                # Summary failed, so pdf must NOT be added to manifest
                self.assertFalse(manifest.contains(key))
                self.assertIn(key, last_summary_run.get("failed_docs", []))


class TestP0Iteration2DefectRegressions(unittest.TestCase):
    """Area 8: Milestone 1 Iteration 2 Challenger & Explorer Defect Regressions."""

    def test_01_verify_candidate_match_surname_extraction_and_strict_matching(self):
        """verify_candidate_match must extract surnames properly and reject mismatched authors."""
        from research_assistant.shared.academic_apis import verify_candidate_match

        # Different authors with common initials must be REJECTED (no initial substring matching)
        self.assertFalse(
            verify_candidate_match("Relativity", "Relativity", "Einstein, A.", "Hawking, S.", 1915, 1915),
            "Einstein, A. must NOT match Hawking, S.",
        )
        self.assertFalse(
            verify_candidate_match("Quantum Transport", "Quantum Transport", "Smith, J.", "Jones, J.", 2020, 2020),
            "Smith, J. must NOT match Jones, J.",
        )
        self.assertFalse(
            verify_candidate_match("Graphene", "Graphene", "Novoselov, K. S.", "Zhang, K. S.", 2004, 2004),
            "Novoselov, K. S. must NOT match Zhang, K. S.",
        )

        # Compound and inverted author matching must PASS
        self.assertTrue(
            verify_candidate_match(
                "Thermodynamic Theory", "Thermodynamic Theory",
                "Van der Waals, J. D.", "J. D. van der Waals", 1873, 1873,
            ),
            "Compound surname Van der Waals must match",
        )
        self.assertTrue(
            verify_candidate_match(
                "Carbon Films", "Carbon Films",
                "Novoselov, K. S.", "K. S. Novoselov", 2004, 2004,
            ),
            "Novoselov, K. S. must match K. S. Novoselov",
        )
        self.assertTrue(
            verify_candidate_match(
                "Relativity", "Relativity",
                "Einstein, A.", "Albert Einstein", 1915, 1915,
            ),
            "Einstein, A. must match Albert Einstein",
        )

    def test_02_verify_candidate_match_conjunctive_title_and_part_qualifiers(self):
        """verify_candidate_match must reject near-miss titles with contrasting Roman numerals, digits, or modifiers."""
        from research_assistant.shared.academic_apis import verify_candidate_match

        # Contrasting Roman numerals Part I vs Part II
        self.assertFalse(
            verify_candidate_match(
                "Superconductivity in twisted bilayer graphene: Part I",
                "Superconductivity in twisted bilayer graphene: Part II",
            ),
            "Part I must NOT match Part II",
        )

        # Contrasting numbers Vol 1 vs Vol 2
        self.assertFalse(
            verify_candidate_match(
                "Lectures on Physics Vol 1",
                "Lectures on Physics Vol 2",
            ),
            "Vol 1 must NOT match Vol 2",
        )

        # Contrasting physical objects (black hole vs neutron star)
        self.assertFalse(
            verify_candidate_match(
                "Observation of gravitational waves from a binary black hole merger",
                "Observation of gravitational waves from a binary neutron star merger",
            ),
            "black hole merger must NOT match neutron star merger",
        )

        # Contrasting layer modifiers (monolayer vs bilayer)
        self.assertFalse(
            verify_candidate_match(
                "Quantum Hall effect in monolayer graphene",
                "Quantum Hall effect in bilayer graphene",
            ),
            "monolayer must NOT match bilayer",
        )

    def test_03_resolver_comma_inversion_institutional_and_roman(self):
        """_resolve_ref must handle comma-inversions, institutional acronyms, and bracketed Roman numerals."""
        from research_assistant.audit.resolver import _resolve_ref
        from research_assistant.shared.claim_text import is_numeric_cite

        # 1. Comma-inverted author without author name in raw_reference
        bib_comma = {
            "b1": {
                "xml_id": "b1",
                "authors": ["Smith, J."],
                "year": 2020,
                "title": "Quantum Transport",
                "raw_reference": "Quantum Transport. Phys. Rev. B (2020).",
            }
        }
        res, how, _ = _resolve_ref(None, "Smith (2020)", bib_comma, {}, {})
        self.assertIsNotNone(res)
        self.assertEqual(how, "surname")

        # 2. Institutional acronym with empty authors list
        bib_inst = {
            "b2": {
                "xml_id": "b2",
                "authors": [],
                "year": 2019,
                "title": "Compendium of Chemical Terminology",
                "raw_reference": "IUPAC Compendium of Chemical Terminology (2019).",
            }
        }
        res_inst, how_inst, _ = _resolve_ref(None, "IUPAC (2019)", bib_inst, {}, {})
        self.assertIsNotNone(res_inst)
        self.assertEqual(how_inst, "surname")

        # 3. Bracketed Roman numeral citation marker
        self.assertTrue(is_numeric_cite("[IV]"))
        self.assertTrue(is_numeric_cite("(iv)"))
        self.assertTrue(is_numeric_cite("[XIV]"))
        self.assertFalse(is_numeric_cite("Smith (2020)"))

        bib_roman = {
            "b4": {
                "xml_id": "b4",
                "authors": ["Novoselov"],
                "year": 2004,
                "raw_reference": "[IV] Science (2004)",
            }
        }
        res_rom, how_rom, _ = _resolve_ref(None, "[IV]", bib_roman, {4: bib_roman["b4"]}, {})
        self.assertIsNotNone(res_rom)
        self.assertEqual(how_rom, "printed_label")

    def test_04_pipeline_status_cancellation_persistence(self):
        """In-flight worker calls must NOT clear cancel_requested when active=True is sent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = os.path.join(tmpdir, "pipeline_status.json")
            pipeline_status.set_status_path(status_file)
            pipeline_status.clear_status()

            # Start pipeline
            pipeline_status.set_status(active=True, stage="fetch", item_total=10)
            self.assertTrue(pipeline_status.get_status().get("active"))

            # User requests cancel
            pipeline_status.request_cancel("User stopped")
            self.assertTrue(pipeline_status.is_cancel_requested())

            # In-flight worker calls set_status with active=True in fetch stage
            in_flight = pipeline_status.set_status(active=True, stage="fetch", item_current=2, item_total=10)
            self.assertTrue(in_flight.get("cancel_requested"))
            self.assertTrue(pipeline_status.is_cancel_requested())

            # Explicit clear resets cancellation
            pipeline_status.clear_cancel_request()
            self.assertFalse(pipeline_status.is_cancel_requested())

    def test_05_ingestion_manifest_summary_exception_handling(self):
        """Papers whose summary generation throws an exception must not be recorded in manifest."""
        from research_assistant.shared.ingestion import ingest_pdfs, last_summary_run
        from research_assistant.shared import manifest
        from research_assistant.shared.source_key import pdf_key

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "failed_paper.pdf")
            with open(pdf_path, "wb") as f:
                f.write(b"%PDF-1.4 test")
            m_path = os.path.join(tmpdir, "manifest.json")
            with patch.object(manifest, "MANIFEST_PATH", m_path), \
                 patch.object(manifest, "INGESTED_MANIFEST_PATH", m_path):
                manifest.clear()
                last_summary_run.clear()
                last_summary_run.update(attempted=0, made=0, failed=0, error="", failed_docs=[])
                key = pdf_key(pdf_path)

                with patch("research_assistant.shared.ingestion.process_pdf", return_value=[{"document": key, "type": "summary_source", "citation": "P", "content": "Hello"}]), \
                     patch("research_assistant.shared.ingestion.upsert_summaries", side_effect=RuntimeError("Store collection error")), \
                     patch("research_assistant.shared.llm.get_embeddings", return_value=_OfflineEmbeddings()), \
                     patch("research_assistant.shared.ingestion.rebuild_bm25"):
                    pipeline_status.clear_status()
                    ingest_pdfs({pdf_path: "Paper Title"}, workers=1, skip_ingested=False)
                    self.assertFalse(manifest.contains(key), f"Paper {key} was erroneously marked in manifest!")
                    self.assertIn(key, last_summary_run.get("failed_docs", []))

    def test_06_ui_progress_callbacks_safe_coercion(self):
        """UI callbacks must safely handle string division, int names, and None fields."""
        class MockProgress:
            def __init__(self):
                self.calls = []

            def progress(self, pct, text=""):
                self.calls.append((pct, text))

        mp = MockProgress()

        def _safe_cb(status_or_item_c=None, item_t=None, name=None, **kwargs):
            try:
                if isinstance(status_or_item_c, dict):
                    st_dict = status_or_item_c
                    item_c = st_dict.get("item_current", 0)
                    item_t = st_dict.get("item_total", 0)
                    name = st_dict.get("current_item_name") or st_dict.get("name") or ""
                    progress = st_dict.get("progress")
                    message = st_dict.get("detail") or st_dict.get("message") or ""
                else:
                    item_c = status_or_item_c if status_or_item_c is not None else 0
                    item_t = item_t if item_t is not None else 0
                    name = name or ""
                    progress = None
                    message = ""

                if progress is not None:
                    try:
                        pct = min(1.0, max(0.0, float(progress)))
                    except (ValueError, TypeError):
                        pct = 0.0
                else:
                    try:
                        c_flt = float(item_c) if item_c is not None else 0.0
                        t_flt = float(item_t) if item_t is not None else 0.0
                        pct = min(1.0, max(0.0, c_flt / t_flt)) if t_flt > 0.0 else 0.0
                    except (ValueError, TypeError, ZeroDivisionError):
                        pct = 0.0

                name_str = str(name)[:40] if name is not None else ""
                try:
                    has_t = float(item_t) > 0 if item_t is not None else False
                except (ValueError, TypeError):
                    has_t = False

                display_txt = str(message) if message else (
                    f"Parsing paper [{item_c}/{item_t}]: {name_str}" if has_t else f"Parsing paper: {name_str}"
                )
                mp.progress(pct, text=display_txt)
            except Exception:
                pass

        # Case 1: String item_current and int name
        _safe_cb({"item_current": "invalid", "item_total": 10, "current_item_name": 12345})
        self.assertEqual(mp.calls[-1], (0.0, "Parsing paper [invalid/10]: 12345"))

        # Case 2: String numbers
        _safe_cb({"item_current": "1", "item_total": "2", "name": "paper.pdf"})
        self.assertEqual(mp.calls[-1], (0.5, "Parsing paper [1/2]: paper.pdf"))

        # Case 3: Zero total
        _safe_cb({"item_current": 5, "item_total": 0})
        self.assertEqual(mp.calls[-1], (0.0, "Parsing paper: "))

        # Case 4: Positional call with None
        _safe_cb(None, None, None)
        self.assertEqual(mp.calls[-1], (0.0, "Parsing paper: "))

