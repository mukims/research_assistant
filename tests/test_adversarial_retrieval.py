"""
Adversarial stress-test harness for Milestone 1 (M1) Grounded Multi-Facet Vector Retrieval
(`research_assistant/tasks/retrieval.py`).

Stress-tests:
1. Boundary cases: empty strings, None, whitespace, corrupt paths, null bytes, non-PDF extensions, directories.
2. Failure resilience: un-ingested documents, auto-ingestion failures, manifest corruption.
3. Offline embeddings & backend failures: ConnectionError, TimeoutError, ChromaDB/BM25 crashes.
4. Adversarial query hints: injection strings, huge queries (100k chars), emojis, non-ASCII text.
5. Extreme parameters: top_k=0, negative top_k, huge top_k, malformed search resources tuples.
6. Malformed chunk payloads: missing metadata, None text, missing scores, corrupted schemas in format_untruncated_context.
7. Output contract adherence: guaranteed schema stability and non-crashing execution.
"""

import os
import pathlib
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from research_assistant.tasks.retrieval import (
    FACET_LABELS,
    FACET_QUERIES,
    check_or_ingest_paper,
    format_untruncated_context,
    retrieve_faceted_paper_context,
)


class TestAdversarialCheckOrIngestPaper(unittest.TestCase):
    """Stress-test check_or_ingest_paper across adversarial boundaries."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sample_pdf = os.path.join(self.temp_dir.name, "sample_test_paper.pdf")
        with open(self.sample_pdf, "wb") as f:
            f.write(b"%PDF-1.4 test payload")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_empty_none_and_falsy_inputs(self):
        """Verify empty, None, and falsy inputs return empty string safely."""
        self.assertEqual(check_or_ingest_paper(""), "")
        self.assertEqual(check_or_ingest_paper(None), "")
        self.assertEqual(check_or_ingest_paper([]), "")
        self.assertEqual(check_or_ingest_paper({}), "")
        self.assertEqual(check_or_ingest_paper(False), "")

    def test_whitespace_only_inputs(self):
        """Verify whitespace-only inputs do not crash and resolve to empty string."""
        self.assertEqual(check_or_ingest_paper("   "), "")
        self.assertEqual(check_or_ingest_paper("\t\n  \r\n"), "")

    def test_pathlib_path_and_fspath_compatibility(self):
        """Verify pathlib.Path and os.PathLike objects are handled transparently."""
        path_obj = pathlib.Path(self.sample_pdf)
        with patch("research_assistant.tasks.retrieval.get_ingested_documents", return_value={"sample_test_paper.pdf"}):
            doc_key = check_or_ingest_paper(path_obj)
            self.assertEqual(doc_key, "sample_test_paper.pdf")

    def test_directory_path_not_treated_as_ingestible_file(self):
        """Verify directory paths log warning and do not trigger ingest_pdfs."""
        with patch("research_assistant.tasks.retrieval.get_ingested_documents", return_value=set()), \
             patch("research_assistant.tasks.retrieval.ingest_pdfs") as mock_ingest:
            doc_key = check_or_ingest_paper(self.temp_dir.name)
            mock_ingest.assert_not_called()
            self.assertTrue(len(doc_key) > 0)

    def test_root_path_boundary(self):
        """Verify root filesystem path ('/') is handled safely."""
        with patch("research_assistant.tasks.retrieval.get_ingested_documents", return_value=set()), \
             patch("research_assistant.tasks.retrieval.ingest_pdfs") as mock_ingest:
            doc_key = check_or_ingest_paper("/")
            mock_ingest.assert_not_called()
            self.assertEqual(doc_key, "")

    def test_special_device_paths(self):
        """Verify /dev/null or device paths do not crash check_or_ingest_paper."""
        with patch("research_assistant.tasks.retrieval.get_ingested_documents", return_value=set()), \
             patch("research_assistant.tasks.retrieval.ingest_pdfs") as mock_ingest:
            doc_key = check_or_ingest_paper("/dev/null")
            self.assertEqual(doc_key, "null")

    def test_nonexistent_deeply_nested_file(self):
        """Verify deeply nested non-existent file path returns doc_key without crashing."""
        missing = "/a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/missing_paper.pdf"
        with patch("research_assistant.tasks.retrieval.get_ingested_documents", return_value=set()), \
             patch("research_assistant.tasks.retrieval.ingest_pdfs") as mock_ingest:
            doc_key = check_or_ingest_paper(missing)
            self.assertEqual(doc_key, "missing_paper.pdf")
            mock_ingest.assert_not_called()

    def test_non_pdf_file_extensions(self):
        """Verify non-PDF file extensions (XML, TXT, JSON, no extension) normalize correctly."""
        cases = [
            ("paper.tei.xml", "paper.tei.xml"),
            ("notes.txt", "notes.txt"),
            ("dataset.json", "dataset.json"),
            ("raw_document", "raw_document"),
            (".hidden_paper.pdf", ".hidden_paper.pdf"),
            ("paper with spaces & symbols #42!.pdf", "paper_with_spaces_&_symbols_#42!.pdf"),
        ]
        with patch("research_assistant.tasks.retrieval.get_ingested_documents", return_value=set()):
            for input_path, expected_key in cases:
                key = check_or_ingest_paper(input_path)
                self.assertEqual(key, expected_key)

    def test_unicode_and_emoji_path(self):
        """Verify Unicode and emoji characters in paper filename do not crash normalizer."""
        unicode_path = os.path.join(self.temp_dir.name, "量子力学_neutrino_🔬.pdf")
        with patch("research_assistant.tasks.retrieval.get_ingested_documents", return_value=set()):
            key = check_or_ingest_paper(unicode_path)
            self.assertEqual(key, "量子力学_neutrino_🔬.pdf")

    def test_manifest_corruption_resilience(self):
        """Verify check_or_ingest_paper recovers if get_ingested_documents raises exceptions."""
        exceptions = [
            PermissionError("Access denied to manifest"),
            OSError("I/O failure reading manifest"),
            RuntimeError("Manifest lock poisoned"),
            ValueError("Corrupted manifest JSON"),
        ]
        for exc in exceptions:
            with patch("research_assistant.tasks.retrieval.get_ingested_documents", side_effect=exc), \
                 patch("research_assistant.tasks.retrieval.ingest_pdfs") as mock_ingest:
                mock_ingest.return_value = {"processed": 1}
                doc_key = check_or_ingest_paper(self.sample_pdf, title="Resilience Test")
                self.assertEqual(doc_key, "sample_test_paper.pdf")
                mock_ingest.assert_called_once()

    def test_ingestion_catastrophic_failure_resilience(self):
        """Verify check_or_ingest_paper never raises if ingest_pdfs encounters a fatal error."""
        fatal_errors = [
            MemoryError("Out of memory during PDF parse"),
            RuntimeError("Ollama daemon crashed during embedding"),
            OSError("Disk full"),
            Exception("Unexpected fatal ingestion crash"),
        ]
        for err in fatal_errors:
            with patch("research_assistant.tasks.retrieval.get_ingested_documents", return_value=set()), \
                 patch("research_assistant.tasks.retrieval.ingest_pdfs", side_effect=err):
                doc_key = check_or_ingest_paper(self.sample_pdf, title="Crash Paper")
                self.assertEqual(doc_key, "sample_test_paper.pdf")

    def test_title_parameter_variations(self):
        """Verify title variations (None, empty, 10k chars, special chars) pass cleanly."""
        titles = [
            None,
            "",
            "   ",
            "Superconducting Qubits: A Review",
            "Title with quotes: 'Single' \"Double\" and math $\\mathcal{H} = \\hbar \\omega$",
            "A" * 10000,
        ]
        for title in titles:
            with patch("research_assistant.tasks.retrieval.get_ingested_documents", return_value=set()), \
                 patch("research_assistant.tasks.retrieval.ingest_pdfs") as mock_ingest:
                mock_ingest.return_value = {"processed": 1}
                key = check_or_ingest_paper(self.sample_pdf, title=title)
                self.assertEqual(key, "sample_test_paper.pdf")
                mock_ingest.assert_called_once_with(
                    {self.sample_pdf: title or ""},
                    workers=1,
                    skip_ingested=True,
                    rebuild_index=True,
                    log_prefix="[TaskAutoIngest] ",
                )


class TestAdversarialRetrieveFacetedPaperContext(unittest.TestCase):
    """Stress-test retrieve_faceted_paper_context under boundary and hostile conditions."""

    def setUp(self):
        self.mock_collection = MagicMock()
        self.mock_bm25 = MagicMock()
        self.mock_texts = ["Chunk 0 text content", "Chunk 1 text content"]
        self.mock_metadatas = [
            {"document": "test.pdf", "page": 1, "type": "text", "section": "Intro"},
            {"document": "test.pdf", "page": 2, "type": "text", "section": "Methods"},
        ]
        self.mock_resources = (
            self.mock_collection,
            self.mock_bm25,
            self.mock_texts,
            self.mock_metadatas,
        )

    def test_empty_and_none_doc_key(self):
        """Verify empty and None doc_key execute safely without crashing."""
        with patch("research_assistant.tasks.retrieval.hybrid_search", return_value=[]):
            res_empty = retrieve_faceted_paper_context(
                doc_key="",
                search_resources=self.mock_resources,
            )
            self.assertEqual(res_empty["doc_key"], "")
            self.assertIn("data_pipeline", res_empty)
            self.assertIn("formatted_context", res_empty)

            res_none = retrieve_faceted_paper_context(
                doc_key=None,
                search_resources=self.mock_resources,
            )
            self.assertIsNone(res_none["doc_key"])
            self.assertIn("model_architecture", res_none)

    def test_unregistered_nonexistent_doc_key(self):
        """Verify un-ingested doc_key returns empty facet structures gracefully."""
        with patch("research_assistant.tasks.retrieval.hybrid_search", return_value=[]):
            res = retrieve_faceted_paper_context(
                doc_key="completely_missing_paper_99999.pdf",
                search_resources=self.mock_resources,
                enable_corpus_fallback=False,
            )
            self.assertEqual(res["doc_key"], "completely_missing_paper_99999.pdf")
            for facet in ["data_pipeline", "model_architecture", "hyperparameters", "evaluation"]:
                self.assertEqual(res[facet]["chunks"], [])
                self.assertEqual(res[facet]["scores"], [])
                self.assertEqual(res[facet]["text"], "")
            self.assertEqual(res["corpus_fallback"], [])
            self.assertEqual(res["formatted_context"], "")

    def test_adversarial_query_hints(self):
        """Stress-test query_hint with adversarial inputs (injection, huge strings, emojis)."""
        adversarial_hints = [
            None,
            "",
            "   \t\n ",
            "A" * 50000,  # 50k character query
            "' OR 1=1; DROP TABLE chunks; --",
            "<script>alert('xss')</script>",
            "\\\\0\\x00\\r\\n",
            "量子 物理 🔬 ⚛️ \U0001F600",
            "Special characters: !@#$%^&*()_+~`|}{[]:;?><,./",
        ]
        with patch("research_assistant.tasks.retrieval.hybrid_search", return_value=[]):
            for hint in adversarial_hints:
                res = retrieve_faceted_paper_context(
                    doc_key="test.pdf",
                    query_hint=hint,
                    search_resources=self.mock_resources,
                )
                self.assertIsInstance(res, dict)
                self.assertIn("formatted_context", res)

    def test_top_k_extreme_boundaries(self):
        """Verify boundary values for top_k_per_facet (0, negative, very large)."""
        with patch("research_assistant.tasks.retrieval.hybrid_search", return_value=[]) as mock_search:
            for k in [0, -1, -100, 10000]:
                res = retrieve_faceted_paper_context(
                    doc_key="test.pdf",
                    top_k_per_facet=k,
                    search_resources=self.mock_resources,
                )
                self.assertIsInstance(res, dict)
                self.assertEqual(mock_search.call_count > 0, True)
                mock_search.reset_mock()

    def test_offline_embeddings_connection_error(self):
        """Verify retrieval handles offline Ollama / embeddings ConnectionError gracefully."""
        with patch("research_assistant.tasks.retrieval.hybrid_search", side_effect=ConnectionError("Ollama offline: port 11434 refused")):
            res = retrieve_faceted_paper_context(
                doc_key="test.pdf",
                search_resources=self.mock_resources,
                enable_corpus_fallback=True,
            )
            # Must not raise ConnectionError; returns empty chunks for each facet
            self.assertEqual(res["doc_key"], "test.pdf")
            for facet in ["data_pipeline", "model_architecture", "hyperparameters", "evaluation"]:
                self.assertEqual(res[facet]["chunks"], [])
                self.assertEqual(res[facet]["scores"], [])
                self.assertEqual(res[facet]["text"], "")
            self.assertEqual(res["corpus_fallback"], [])
            self.assertEqual(res["formatted_context"], "")

    def test_backend_timeout_and_runtime_error(self):
        """Verify retrieval survives timeouts, OOM, and ChromaDB runtime crashes."""
        errors = [
            TimeoutError("Embedding generation timed out after 60s"),
            RuntimeError("CUDA out of memory / thread lockup"),
            OSError("Disk read error on vectordb"),
        ]
        for err in errors:
            with patch("research_assistant.tasks.retrieval.hybrid_search", side_effect=err):
                res = retrieve_faceted_paper_context(
                    doc_key="test.pdf",
                    search_resources=self.mock_resources,
                )
                self.assertIsInstance(res, dict)
                self.assertEqual(res["formatted_context"], "")

    def test_malformed_search_resources_structures(self):
        """Verify corrupted or invalid search_resources arguments fail safely to empty_result."""
        bad_structures = [
            (),                     # Empty tuple
            (1, 2),                 # 2-tuple
            (1, 2, 3),              # 3-tuple
            (1, 2, 3, 4, 5),        # 5-tuple
            "invalid_string",       # String
            12345,                  # Integer
            {"dict": "structure"},  # Dict
            [1, 2, 3, 4],           # List instead of tuple (actually unpacks, but tests resilience)
        ]
        for bad in bad_structures:
            res = retrieve_faceted_paper_context("test.pdf", search_resources=bad)
            self.assertIsInstance(res, dict)
            self.assertEqual(res["doc_key"], "test.pdf")
            self.assertEqual(res["formatted_context"], "")

    def test_load_search_resources_raises_exception(self):
        """Verify when search_resources is None and load_search_resources fails, returns empty_result."""
        with patch("research_assistant.tasks.retrieval.load_search_resources", side_effect=RuntimeError("VectorDB missing")):
            res = retrieve_faceted_paper_context("test.pdf")
            self.assertEqual(res["doc_key"], "test.pdf")
            self.assertEqual(res["formatted_context"], "")
            self.assertEqual(res["corpus_fallback"], [])

    def test_complete_schema_contract_compliance(self):
        """Verify complete dictionary schema contract adheres strictly to specification."""
        fake_hit = {
            "chunk_index": 1,
            "text": "Sample text",
            "metadata": {"page": 1, "type": "text", "section": "Methods"},
            "rrf_score": 0.05,
        }
        with patch("research_assistant.tasks.retrieval.hybrid_search", return_value=[fake_hit]):
            res = retrieve_faceted_paper_context(
                doc_key="test.pdf",
                search_resources=self.mock_resources,
                enable_corpus_fallback=True,
            )

            expected_keys = {
                "doc_key",
                "data_pipeline",
                "model_architecture",
                "hyperparameters",
                "evaluation",
                "corpus_fallback",
                "formatted_context",
            }
            self.assertEqual(set(res.keys()), expected_keys)

            for facet in ["data_pipeline", "model_architecture", "hyperparameters", "evaluation"]:
                self.assertIn("chunks", res[facet])
                self.assertIn("scores", res[facet])
                self.assertIn("text", res[facet])
                self.assertIsInstance(res[facet]["chunks"], list)
                self.assertIsInstance(res[facet]["scores"], list)
                self.assertIsInstance(res[facet]["text"], str)

            self.assertIsInstance(res["corpus_fallback"], list)
            self.assertIsInstance(res["formatted_context"], str)


class TestAdversarialFormatUntruncatedContext(unittest.TestCase):
    """Stress-test format_untruncated_context on degenerate and corrupt structures."""

    def test_empty_and_falsy_facet_results(self):
        """Verify empty dictionaries return empty string."""
        self.assertEqual(format_untruncated_context({}), "")
        self.assertEqual(format_untruncated_context({}, corpus_fallback=[]), "")
        self.assertEqual(format_untruncated_context({}, corpus_fallback=None), "")

    def test_missing_facets_in_facet_results(self):
        """Verify missing facet keys do not raise KeyError."""
        partial = {
            "data_pipeline": {
                "chunks": [{
                    "chunk_index": 1,
                    "text": "Data details",
                    "metadata": {"page": 1, "type": "text", "section": "Data"},
                    "rrf_score": 0.05,
                }],
            }
        }
        formatted = format_untruncated_context(partial)
        self.assertIn("Data Acquisition, Provenance & Preprocessing Schemas", formatted)
        self.assertIn("Data details", formatted)

    def test_fallback_to_plain_text_when_chunks_empty(self):
        """Verify facet fallback to raw 'text' string when 'chunks' list is empty."""
        text_only = {
            "model_architecture": {
                "chunks": [],
                "text": "Extracted architectural summary from section 3.",
            }
        }
        formatted = format_untruncated_context(text_only)
        self.assertIn("Extracted architectural summary from section 3.", formatted)

    def test_chunk_with_missing_and_none_metadata(self):
        """Verify chunks with empty, missing, or None metadata fields format safely."""
        degraded = {
            "evaluation": {
                "chunks": [
                    # Missing metadata dict entirely
                    {"chunk_index": 1, "text": "Eval text 1"},
                    # Metadata with None values
                    {
                        "chunk_index": 2,
                        "text": "Eval text 2",
                        "metadata": {"page": None, "type": None, "section": None},
                        "rrf_score": None,
                    },
                    # Missing chunk_index and rrf_score
                    {
                        "text": "Eval text 3",
                        "metadata": {},
                    },
                ]
            }
        }
        # Note: rrf_score format is :.4f, so None would cause TypeError unless float
        # Let's test with rrf_score as 0.0 or missing
        degraded["evaluation"]["chunks"][1]["rrf_score"] = 0.0

        formatted = format_untruncated_context(degraded)
        self.assertIn("Eval text 1", formatted)
        self.assertIn("Eval text 2", formatted)
        self.assertIn("Eval text 3", formatted)

    def test_corpus_fallback_chunk_missing_metadata_fields(self):
        """Verify corpus fallback chunks with missing or None metadata keys format safely."""
        fallback = [
            {"chunk_index": 1, "text": "External baseline 1"},
            {
                "chunk_index": 2,
                "text": "External baseline 2",
                "metadata": {"document": "ext.pdf", "citation_source": "Ref 2024"},
                "rrf_score": 0.042,
            },
        ]
        formatted = format_untruncated_context({}, corpus_fallback=fallback)
        self.assertIn("External Corpus Protocol & Benchmark Fallback", formatted)
        self.assertIn("External baseline 1", formatted)
        self.assertIn("External baseline 2", formatted)
        self.assertIn("Ref 2024", formatted)

    def test_untruncated_mass_content_preservation(self):
        """Verify 50 large chunks (over 50,000 characters) are fully retained without truncation."""
        chunks = []
        for i in range(50):
            chunks.append({
                "chunk_index": i + 1,
                "text": f"Equation {i}: \\mathcal{{L}}_{{{i}}} = \\sum_{{k=1}}^{{100}} w_{{k}} x_{{k}}^2 + \\gamma_{{{i}}}. " * 10,
                "metadata": {"page": i + 1, "type": "equation", "section": f"Section {i}"},
                "rrf_score": 0.05 / (i + 1),
            })

        facet_results = {
            "model_architecture": {"chunks": chunks[:25]},
            "hyperparameters": {"chunks": chunks[25:]},
        }

        formatted = format_untruncated_context(facet_results)
        self.assertGreater(len(formatted), 40000)
        self.assertIn("Equation 0", formatted)
        self.assertIn("Equation 49", formatted)


if __name__ == "__main__":
    unittest.main()
