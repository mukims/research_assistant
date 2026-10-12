"""
Challenger Verification Suite for Milestone 1 (M1) Grounded Multi-Facet Vector Retrieval.
Agent: teamwork_preview_challenger_m1_2

Adversarially challenges and empirically verifies:
1. 4 distinct methodological facets and 75k corpus fallback logic.
2. format_untruncated_context with large payloads (>25,000 characters) preserving equations and tables.
3. Persistent storage integrity (physics_papers collection = 75,039 chunks in ChromaDB).
4. Live hybrid search resource interaction with 75k corpus.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import chromadb

from research_assistant.config import (
    COLLECTION_NAME,
    VECTORDB_PATH,
)
from research_assistant.shared.db import load_search_resources
from research_assistant.tasks.retrieval import (
    FACET_LABELS,
    FACET_QUERIES,
    check_or_ingest_paper,
    format_untruncated_context,
    retrieve_faceted_paper_context,
)


class TestM1PersistentStorageIntegrity(unittest.TestCase):
    """Verify persistent vector storage integrity invariant."""

    def test_chromadb_physics_papers_chunk_count(self):
        """Empirically assert physics_papers collection contains exactly 75,039 chunks."""
        client = chromadb.PersistentClient(path=VECTORDB_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        self.assertEqual(
            count,
            75039,
            f"Persistent collection '{COLLECTION_NAME}' corrupted or altered! Expected 75039, got {count}",
        )

    def test_chromadb_schema_and_embedding_dimension(self):
        """Verify chunk schema, embedding dimension (768), and metadata integrity."""
        client = chromadb.PersistentClient(path=VECTORDB_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        sample = collection.peek(5)

        self.assertIn("ids", sample)
        self.assertIn("documents", sample)
        self.assertIn("metadatas", sample)
        self.assertIn("embeddings", sample)

        self.assertEqual(len(sample["ids"]), 5)
        self.assertEqual(len(sample["documents"]), 5)
        self.assertEqual(len(sample["metadatas"]), 5)
        self.assertEqual(len(sample["embeddings"]), 5)

        # Dimension check: nomic-embed-text generates 768-dim embeddings
        for emb in sample["embeddings"]:
            self.assertEqual(len(emb), 768, "Embedding dimension mismatch! Expected 768.")

        # Document key check in metadata
        for meta in sample["metadatas"]:
            self.assertIn("document", meta)
            self.assertTrue(len(meta["document"]) > 0)


class TestM1FourFacetsAndCorpusFallback(unittest.TestCase):
    """Empirically test 4 facets retrieval and 75k corpus fallback."""

    def test_facet_definitions_coverage(self):
        """Assert all 4 required methodological facets exist with distinct query semantics."""
        expected_facets = {
            "data_pipeline",
            "model_architecture",
            "hyperparameters",
            "evaluation",
        }
        self.assertEqual(set(FACET_QUERIES.keys()), expected_facets)
        self.assertEqual(set(FACET_LABELS.keys()), expected_facets)

        # 1. Data acquisition, provenance, train/val/test splits, normalization, preprocessing
        self.assertIn("data acquisition", FACET_QUERIES["data_pipeline"])
        self.assertIn("train validation test split", FACET_QUERIES["data_pipeline"])
        self.assertIn("preprocessing", FACET_QUERIES["data_pipeline"])

        # 2. Core model architecture, mathematical operators, loss functions, layer dimensions
        self.assertIn("model architecture", FACET_QUERIES["model_architecture"])
        self.assertIn("loss function", FACET_QUERIES["model_architecture"])
        self.assertIn("mathematical operators", FACET_QUERIES["model_architecture"])

        # 3. Hyperparameters, learning rate schedules, optimizer, batch sizes, seeds
        self.assertIn("hyperparameters", FACET_QUERIES["hyperparameters"])
        self.assertIn("learning rate", FACET_QUERIES["hyperparameters"])
        self.assertIn("optimizer configuration", FACET_QUERIES["hyperparameters"])
        self.assertIn("batch size", FACET_QUERIES["hyperparameters"])
        self.assertIn("random seeds", FACET_QUERIES["hyperparameters"])

        # 4. Evaluation metrics, benchmarks, test procedures, comparative target values
        self.assertIn("evaluation metrics", FACET_QUERIES["evaluation"])
        self.assertIn("benchmark", FACET_QUERIES["evaluation"])
        self.assertIn("target values", FACET_QUERIES["evaluation"])

    def test_retrieve_faceted_paper_context_structure(self):
        """Verify faceted retrieval executes all 4 queries with target document scoping."""
        mock_collection = MagicMock()
        mock_bm25 = MagicMock()
        mock_texts = ["Chunk 0", "Chunk 1"]
        mock_metadatas = [{"document": "test_paper.pdf"}, {"document": "test_paper.pdf"}]
        resources = (mock_collection, mock_bm25, mock_texts, mock_metadatas)

        recorded_searches = []

        def spy_hybrid_search(query, *args, **kwargs):
            recorded_searches.append({
                "query": query,
                "doc_filter": kwargs.get("doc_filter"),
                "exclude_docs": kwargs.get("exclude_docs"),
                "top_k": kwargs.get("top_k"),
            })
            return [{
                "chunk_index": 1,
                "text": f"Result for {query[:20]}",
                "metadata": {"document": "test_paper.pdf", "page": 1, "type": "text"},
                "rrf_score": 0.05,
            }]

        with patch("research_assistant.tasks.retrieval.hybrid_search", side_effect=spy_hybrid_search):
            result = retrieve_faceted_paper_context(
                doc_key="test_paper.pdf",
                query_hint="neural quantum states",
                top_k_per_facet=7,
                enable_corpus_fallback=True,
                search_resources=resources,
            )

        # 4 facet queries + 1 corpus fallback query = 5 searches
        self.assertEqual(len(recorded_searches), 5)

        # First 4 searches must be scoped strictly to the target paper
        for i in range(4):
            self.assertEqual(recorded_searches[i]["doc_filter"], {"test_paper.pdf"})
            self.assertIsNone(recorded_searches[i]["exclude_docs"])
            self.assertEqual(recorded_searches[i]["top_k"], 7)
            self.assertIn("neural quantum states", recorded_searches[i]["query"])

        # 5th search is corpus fallback: un-scoped across corpus (doc_filter=None), excludes target
        fallback_search = recorded_searches[4]
        self.assertIsNone(fallback_search["doc_filter"])
        self.assertEqual(fallback_search["exclude_docs"], {"test_paper.pdf"})
        self.assertIn("baseline benchmark protocol", fallback_search["query"])
        self.assertIn("neural quantum states", fallback_search["query"])

        # Result verification
        self.assertEqual(result["doc_key"], "test_paper.pdf")
        for f in ["data_pipeline", "model_architecture", "hyperparameters", "evaluation"]:
            self.assertIn(f, result)
            self.assertEqual(len(result[f]["chunks"]), 1)
            self.assertEqual(result[f]["scores"], [0.05])
            self.assertTrue(len(result[f]["text"]) > 0)

        self.assertEqual(len(result["corpus_fallback"]), 1)
        self.assertTrue(len(result["formatted_context"]) > 0)

    def test_corpus_fallback_toggle(self):
        """Assert corpus fallback is omitted when enable_corpus_fallback is False."""
        resources = (MagicMock(), MagicMock(), ["c0"], [{"document": "paper.pdf"}])

        search_calls = []
        def spy_search(query, *args, **kwargs):
            search_calls.append(kwargs)
            return []

        with patch("research_assistant.tasks.retrieval.hybrid_search", side_effect=spy_search):
            result = retrieve_faceted_paper_context(
                doc_key="paper.pdf",
                enable_corpus_fallback=False,
                search_resources=resources,
            )

        # Exactly 4 searches (facets only), no fallback search
        self.assertEqual(len(search_calls), 4)
        self.assertEqual(result["corpus_fallback"], [])

    def test_real_bm25_search_resources_integration(self):
        """Empirically test retrieval against real 75,039-chunk BM25 index with dummy dense embedder."""
        # Load the real persistent index from disk
        resources = load_search_resources()
        collection, bm25, texts, metadatas = resources

        self.assertEqual(len(texts), 75039)
        self.assertEqual(len(metadatas), 75039)

        # Create a mock embeddings model returning a constant 768-dim vector to bypass Ollama daemon
        dummy_emb = [0.0] * 768
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = dummy_emb

        # Pick a real document from the loaded metadatas
        sample_doc = metadatas[100]["document"]

        # Run faceted retrieval with real search resources and mock embedder
        with patch("research_assistant.shared.search._get_embeddings", return_value=mock_embedder):
            result = retrieve_faceted_paper_context(
                doc_key=sample_doc,
                query_hint="Hamiltonian ground state",
                top_k_per_facet=3,
                enable_corpus_fallback=True,
                search_resources=resources,
            )

        self.assertEqual(result["doc_key"], sample_doc)
        self.assertIn("data_pipeline", result)
        self.assertIn("model_architecture", result)
        self.assertIn("hyperparameters", result)
        self.assertIn("evaluation", result)
        self.assertIn("corpus_fallback", result)
        # Chunks retrieved from real 75k corpus
        self.assertGreater(len(result["corpus_fallback"]), 0)


class TestM1UntruncatedContextPayload(unittest.TestCase):
    """Empirically test format_untruncated_context with large payloads (>25,000 characters)."""

    def test_payload_exceeding_25000_chars_retains_equations_and_tables(self):
        """Stress-test payload > 25,000 chars ensuring math equations and tables are never truncated."""
        # 1. Complex mathematical formulation block (equations)
        equations_block = (
            "Mathematical Formulations:\n"
            "\\begin{equation}\n"
            "\\mathcal{L}_{total}(\\theta) = \\mathbb{E}_{x \\sim \\mathcal{D}} \\left[ "
            "\\frac{1}{2} \\| y - f_{\\theta}(x) \\|_2^2 + \\lambda \\sum_{l=1}^L \\| W^{(l)} \\|_F^2 "
            "+ \\beta \\mathrm{KL}(q_{\\phi}(z|x) \\parallel p(z)) \\right]\n"
            "\\end{equation}\n"
            "Hamiltonian operator:\n"
            "\\hat{H} = -J \\sum_{\\langle i,j \\rangle} \\sigma_i^z \\sigma_j^z - h \\sum_i \\sigma_i^x\n"
        ) * 20  # ~7,500 chars

        # 2. Large hyperparameter grid table (>150 rows, ~15,000 chars)
        table_rows = ["| Layer ID | Operator Type | In Features | Out Features | Activation | Dropout | Weight Decay | Learning Rate |"]
        table_rows.append("|:---|:---|:---|:---|:---|:---|:---|:---|")
        for i in range(160):
            table_rows.append(
                f"| layer_{i:04d} | Conv2dTranspose | {64 * (i % 8 + 1)} | {128 * (i % 8 + 1)} "
                f"| LeakyReLU(0.2) | 0.15 | 1e-4 | {1e-3 / (1 + i * 0.05):.6f} |"
            )
        hyperparams_table = "\n".join(table_rows)

        # 3. Data pipeline specification (~6,000 chars)
        data_spec = (
            "Data Pipeline Schema:\n"
            "- Dataset: 500,000 quantum spin configuration tensors.\n"
            "- Train/Val/Test Split: 70% / 15% / 15% (350k / 75k / 75k).\n"
            "- Preprocessing: Z-score standard scaling per subsystem; one-hot basis encoding.\n"
            "- Data Loader: PyTorch DistributedSampler with pin_memory=True, batch_size=256.\n"
        ) * 15

        # 4. Evaluation benchmark table (~8,000 chars)
        eval_rows = ["| Benchmark Name | Metric | Baseline Target | Tolerable Margin | Reference DOI |"]
        eval_rows.append("|:---|:---|:---|:---|:---|")
        for j in range(80):
            eval_rows.append(f"| QuantumBench_{j:02d} | State Fidelity F | 0.99{j%10} | +/- 0.002 | 10.1103/PhysRevB.{j:05d} |")
        eval_table = "\n".join(eval_rows)

        # 5. External fallback protocol (~5,000 chars)
        fallback_protocol = (
            "External Reference Protocol (Scott & Caves 2006):\n"
            "- Benchmark State Preparation: Symmetric Haar-random pure state generator.\n"
            "- Cross-Entropy Benchmarking threshold: XEB > 0.0024 with 5-sigma statistical confidence.\n"
        ) * 16

        total_input_chars = (
            len(equations_block)
            + len(hyperparams_table)
            + len(data_spec)
            + len(eval_table)
            + len(fallback_protocol)
        )
        self.assertGreater(
            total_input_chars,
            25000,
            f"Input payload too small: {total_input_chars} chars. Must exceed 25,000.",
        )
        self.assertGreater(total_input_chars, 35000)

        facet_results = {
            "data_pipeline": {
                "chunks": [{
                    "chunk_index": 1,
                    "text": data_spec,
                    "metadata": {"page": 2, "type": "text", "section": "Data Pipeline"},
                    "rrf_score": 0.045,
                }],
            },
            "model_architecture": {
                "chunks": [{
                    "chunk_index": 2,
                    "text": equations_block,
                    "metadata": {"page": 3, "type": "equation", "section": "Model Architecture"},
                    "rrf_score": 0.042,
                }],
            },
            "hyperparameters": {
                "chunks": [{
                    "chunk_index": 3,
                    "text": hyperparams_table,
                    "metadata": {"page": 5, "type": "table", "section": "Hyperparameters"},
                    "rrf_score": 0.038,
                }],
            },
            "evaluation": {
                "chunks": [{
                    "chunk_index": 4,
                    "text": eval_table,
                    "metadata": {"page": 7, "type": "table", "section": "Evaluation Metrics"},
                    "rrf_score": 0.035,
                }],
            },
        }

        corpus_fallback = [{
            "chunk_index": 99,
            "text": fallback_protocol,
            "metadata": {
                "document": "scott_caves_2006.pdf",
                "page": 4,
                "citation_source": "Scott & Caves (2006)",
            },
            "rrf_score": 0.021,
        }]

        formatted = format_untruncated_context(facet_results, corpus_fallback)

        # Empirically verify length surpasses 25,000 characters
        self.assertGreater(
            len(formatted),
            25000,
            f"Formatted context was truncated! Length is only {len(formatted)} characters.",
        )
        self.assertGreater(len(formatted), 35000)

        # Empirically verify that math equations survived un-truncated
        self.assertIn("\\mathcal{L}_{total}(\\theta)", formatted)
        self.assertIn("\\hat{H} = -J \\sum_{\\langle i,j \\rangle}", formatted)
        self.assertIn("\\mathrm{KL}(q_{\\phi}(z|x)", formatted)

        # Empirically verify that the hyperparameter table survived completely down to row 119
        self.assertIn("layer_0000", formatted)
        self.assertIn("layer_0060", formatted)
        self.assertIn("layer_0119", formatted)

        # Empirically verify evaluation table and fallback protocol survived
        self.assertIn("QuantumBench_00", formatted)
        self.assertIn("QuantumBench_49", formatted)
        self.assertIn("Scott & Caves (2006)", formatted)
        self.assertIn("Cross-Entropy Benchmarking threshold", formatted)

        # Assert no truncation markers
        self.assertNotIn("[truncated]", formatted)
        self.assertNotIn("... [more]", formatted)

    def test_extreme_payload_100k_characters(self):
        """Stress-test format_untruncated_context with extreme >100,000 character payload."""
        large_chunk = "Text block line with detailed algorithmic parameters. " * 500  # ~27k chars
        facet_results = {
            "data_pipeline": {"chunks": [{"chunk_index": 1, "text": large_chunk, "metadata": {}}]},
            "model_architecture": {"chunks": [{"chunk_index": 2, "text": large_chunk, "metadata": {}}]},
            "hyperparameters": {"chunks": [{"chunk_index": 3, "text": large_chunk, "metadata": {}}]},
            "evaluation": {"chunks": [{"chunk_index": 4, "text": large_chunk, "metadata": {}}]},
        }
        corpus_fallback = [{"chunk_index": 5, "text": large_chunk, "metadata": {}}]

        formatted = format_untruncated_context(facet_results, corpus_fallback)
        self.assertGreater(len(formatted), 100000)


if __name__ == "__main__":
    unittest.main()
