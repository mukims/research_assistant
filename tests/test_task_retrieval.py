"""
Unit tests for grounded multi-facet vector retrieval (research_assistant/tasks/retrieval.py).
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

from research_assistant.tasks.retrieval import (
    FACET_QUERIES,
    check_or_ingest_paper,
    format_untruncated_context,
    retrieve_faceted_paper_context,
)


class TestTaskRetrieval(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sample_pdf = os.path.join(self.temp_dir.name, "quantum_state_tomography.pdf")
        with open(self.sample_pdf, "wb") as f:
            f.write(b"%PDF-1.4 sample content")

    def tearDown(self):
        self.temp_dir.cleanup()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Grounding and Auto-Ingestion Tests
    # ──────────────────────────────────────────────────────────────────────────

    @patch("research_assistant.tasks.retrieval.get_ingested_documents")
    @patch("research_assistant.tasks.retrieval.ingest_pdfs")
    def test_check_or_ingest_paper_already_ingested(self, mock_ingest, mock_get_ingested):
        mock_get_ingested.return_value = {"quantum_state_tomography.pdf", "other_paper.pdf"}

        doc_key = check_or_ingest_paper(self.sample_pdf, title="Quantum State Tomography")

        self.assertEqual(doc_key, "quantum_state_tomography.pdf")
        mock_ingest.assert_not_called()

    @patch("research_assistant.tasks.retrieval.get_ingested_documents")
    @patch("research_assistant.tasks.retrieval.ingest_pdfs")
    def test_check_or_ingest_paper_missing_triggers_auto_ingest(self, mock_ingest, mock_get_ingested):
        mock_get_ingested.return_value = {"other_paper.pdf"}
        mock_ingest.return_value = {"processed": 1, "inserted": 10, "failed": []}

        doc_key = check_or_ingest_paper(self.sample_pdf, title="Quantum State Tomography")

        self.assertEqual(doc_key, "quantum_state_tomography.pdf")
        mock_ingest.assert_called_once_with(
            {self.sample_pdf: "Quantum State Tomography"},
            workers=1,
            skip_ingested=True,
            rebuild_index=True,
            log_prefix="[TaskAutoIngest] ",
        )

    @patch("research_assistant.tasks.retrieval.get_ingested_documents")
    @patch("research_assistant.tasks.retrieval.ingest_pdfs")
    def test_check_or_ingest_paper_nonexistent_file(self, mock_ingest, mock_get_ingested):
        mock_get_ingested.return_value = set()
        missing_file = os.path.join(self.temp_dir.name, "missing_paper.pdf")

        doc_key = check_or_ingest_paper(missing_file)

        self.assertEqual(doc_key, "missing_paper.pdf")
        mock_ingest.assert_not_called()

    def test_check_or_ingest_paper_empty_input(self):
        self.assertEqual(check_or_ingest_paper(""), "")
        self.assertEqual(check_or_ingest_paper(None), "")

    @patch("research_assistant.tasks.retrieval.get_ingested_documents")
    @patch("research_assistant.tasks.retrieval.ingest_pdfs")
    def test_check_or_ingest_paper_handles_ingest_exception(self, mock_ingest, mock_get_ingested):
        mock_get_ingested.return_value = set()
        mock_ingest.side_effect = RuntimeError("PDF engine failure")

        doc_key = check_or_ingest_paper(self.sample_pdf, title="Failing Paper")

        self.assertEqual(doc_key, "quantum_state_tomography.pdf")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Faceted Retrieval Tests
    # ──────────────────────────────────────────────────────────────────────────

    @patch("research_assistant.tasks.retrieval.load_search_resources")
    @patch("research_assistant.tasks.retrieval.hybrid_search")
    def test_retrieve_faceted_paper_context_all_four_facets(self, mock_search, mock_load):
        mock_resources = (MagicMock(), MagicMock(), ["chunk 0 text"], [{"document": "quantum.pdf"}])
        mock_load.return_value = mock_resources

        def fake_hybrid_search(query, *args, **kwargs):
            doc_filter = kwargs.get("doc_filter")
            if doc_filter == {"quantum.pdf"}:
                if "provenance" in query:
                    return [{
                        "chunk_index": 1,
                        "text": "Dataset split: 80% train, 10% validation, 10% test. Preprocessing: Z-score normalization.",
                        "metadata": {"page": 2, "type": "text", "section": "3. Data Preparation"},
                        "rrf_score": 0.033,
                    }]
                elif "neural network" in query:
                    return [{
                        "chunk_index": 2,
                        "text": "Architecture: 4-layer MLP with ComplexLinear layers. Loss function: L = D_KL(P || Q) + lambda * ||W||_2.",
                        "metadata": {"page": 3, "type": "text", "section": "4. Model Architecture"},
                        "rrf_score": 0.031,
                    }]
                elif "learning rate" in query:
                    return [{
                        "chunk_index": 3,
                        "text": "Optimizer: AdamW (lr=1e-3, beta1=0.9, beta2=0.999), batch size 64, 100 epochs, seed 42.",
                        "metadata": {"page": 4, "type": "text", "section": "5. Training Setup"},
                        "rrf_score": 0.029,
                    }]
                elif "evaluation metrics" in query:
                    return [{
                        "chunk_index": 4,
                        "text": "Evaluation benchmark: Quantum State Fidelity F(rho, sigma) >= 0.985 across 10,000 Haar random states.",
                        "metadata": {"page": 5, "type": "text", "section": "6. Evaluation & Results"},
                        "rrf_score": 0.035,
                    }]
            return [{
                "chunk_index": 99,
                "text": "External baseline benchmark: standard Haar state protocol from Scott & Caves (2006).",
                "metadata": {"document": "external_scott2006.pdf", "page": 1, "citation_source": "Scott 2006"},
                "rrf_score": 0.015,
            }]

        mock_search.side_effect = fake_hybrid_search

        result = retrieve_faceted_paper_context(
            doc_key="quantum.pdf",
            query_hint="variational quantum tomography",
            top_k_per_facet=3,
            enable_corpus_fallback=True,
        )

        self.assertEqual(result["doc_key"], "quantum.pdf")
        for facet in ["data_pipeline", "model_architecture", "hyperparameters", "evaluation"]:
            self.assertIn(facet, result)
            self.assertEqual(len(result[facet]["chunks"]), 1)
            self.assertEqual(len(result[facet]["scores"]), 1)
            self.assertTrue(len(result[facet]["text"]) > 0)

        # Corpus fallback populated
        self.assertEqual(len(result["corpus_fallback"]), 1)
        self.assertIn("Scott 2006", result["corpus_fallback"][0]["metadata"]["citation_source"])

        # Formatted context contains all facets without truncation
        formatted = result["formatted_context"]
        self.assertIn("Data Acquisition, Provenance & Preprocessing Schemas", formatted)
        self.assertIn("Core Model Architecture, Mathematical Operators & Loss Formulations", formatted)
        self.assertIn("Hyperparameters, Optimizer Configurations & Training Schedules", formatted)
        self.assertIn("Evaluation Metrics, Benchmark Procedures & Target Replication Bounds", formatted)
        self.assertIn("External Corpus Protocol & Benchmark Fallback", formatted)
        self.assertIn("Z-score normalization", formatted)
        self.assertIn("ComplexLinear", formatted)
        self.assertIn("AdamW", formatted)
        self.assertIn("F(rho, sigma)", formatted)

    @patch("research_assistant.tasks.retrieval.hybrid_search")
    def test_retrieve_faceted_paper_context_with_passed_resources(self, mock_search):
        mock_resources = (MagicMock(), MagicMock(), ["chunk 0"], [{"document": "test.pdf"}])
        mock_search.return_value = []

        result = retrieve_faceted_paper_context(
            doc_key="test.pdf",
            search_resources=mock_resources,
            enable_corpus_fallback=False,
        )

        self.assertEqual(result["doc_key"], "test.pdf")
        self.assertEqual(result["corpus_fallback"], [])
        self.assertEqual(result["data_pipeline"]["chunks"], [])

    @patch("research_assistant.tasks.retrieval.load_search_resources")
    def test_retrieve_faceted_paper_context_load_resources_failure(self, mock_load):
        mock_load.side_effect = RuntimeError("ChromaDB connection failure")

        result = retrieve_faceted_paper_context("unreachable.pdf")

        self.assertEqual(result["doc_key"], "unreachable.pdf")
        self.assertEqual(result["data_pipeline"]["chunks"], [])
        self.assertEqual(result["formatted_context"], "")

    @patch("research_assistant.tasks.retrieval.load_search_resources")
    @patch("research_assistant.tasks.retrieval.hybrid_search")
    def test_retrieve_faceted_paper_context_corpus_fallback_disabled(self, mock_search, mock_load):
        mock_load.return_value = (MagicMock(), MagicMock(), [], [])
        mock_search.return_value = []

        result = retrieve_faceted_paper_context(
            doc_key="no_fallback.pdf",
            enable_corpus_fallback=False,
        )

        self.assertEqual(result["corpus_fallback"], [])
        # 4 facet searches only, no fallback search
        self.assertEqual(mock_search.call_count, 4)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Context Formatting & Un-Truncated Preservation Tests
    # ──────────────────────────────────────────────────────────────────────────

    def test_format_untruncated_context_preserves_length_and_math(self):
        # Create very large text exceeding 10,000 characters to verify no [:10000] truncation
        huge_hyperparam_table = "\n".join(
            f"| layer_{i:03d} | dim_{2**i} | lr=0.00{i} | loss=MSE + alpha * ||theta||^2 |"
            for i in range(300)
        )
        self.assertGreater(len(huge_hyperparam_table), 12000)

        facet_results = {
            "data_pipeline": {
                "chunks": [{
                    "chunk_index": 1,
                    "text": "Dataset: 1M samples generated via Hamiltonians H = sum_i J_i sigma_i.",
                    "metadata": {"page": 1, "type": "text"},
                    "rrf_score": 0.05,
                }]
            },
            "model_architecture": {
                "chunks": [{
                    "chunk_index": 2,
                    "text": "Forward pass: y = sigma(W_2 * ReLU(W_1 * x + b_1) + b_2).",
                    "metadata": {"page": 2, "type": "text"},
                    "rrf_score": 0.04,
                }]
            },
            "hyperparameters": {
                "chunks": [{
                    "chunk_index": 3,
                    "text": huge_hyperparam_table,
                    "metadata": {"page": 3, "type": "table"},
                    "rrf_score": 0.03,
                }]
            },
            "evaluation": {
                "chunks": [{
                    "chunk_index": 4,
                    "text": "Target Metric: Validation Fidelity >= 0.999.",
                    "metadata": {"page": 4, "type": "text"},
                    "rrf_score": 0.02,
                }]
            },
        }

        formatted = format_untruncated_context(facet_results)

        # Must retain complete table well beyond 10k chars
        self.assertGreater(len(formatted), 12000)
        self.assertIn("layer_299", formatted)
        self.assertIn("H = sum_i J_i sigma_i", formatted)
        self.assertIn("y = sigma(W_2 * ReLU(W_1 * x + b_1) + b_2)", formatted)
        self.assertIn("Target Metric: Validation Fidelity >= 0.999.", formatted)

    def test_reference_subgraph_and_citation_dereferencing(self):
        """
        Verifies Stage 1 (Paper-Scoped Reference Subgraph Builder),
        Stage 2 (Citation Dereferencer), and Stage 3 (TaskContextPack & Query DB).
        """
        from research_assistant.tasks.models import Task, TaskCategory, TaskGraph
        from research_assistant.tasks.retrieval import (
            build_paper_reference_subgraph,
            dereference_method_citations,
            populate_graph_context_packs,
            query_task_database,
        )
        from research_assistant.tasks.roles import format_subagent_prompt

        sample_tei = """<?xml version="1.0" encoding="UTF-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <teiHeader>
            <fileDesc><titleStmt><title type="main">Quantum Variational Solver Paper</title></titleStmt></fileDesc>
          </teiHeader>
          <text>
            <body>
              <div>
                <head>2. Methodology</head>
                <p>We adopt the parameter-shift gradient estimator following <ref type="bibr" target="#b0">[1]</ref> and normalize input Hamiltonians using the protocol of <ref type="bibr" target="#b1">[2]</ref>.</p>
              </div>
            </body>
            <back>
              <div type="references">
                <listBibl>
                  <biblStruct xml:id="b0">
                    <analytic><title type="main">Parameter-Shift Quantum Gradients</title></analytic>
                    <monogr><imprint><date type="published" when="2021"/></imprint></monogr>
                    <idno type="DOI">10.1103/PhysRevA.99.032331</idno>
                  </biblStruct>
                  <biblStruct xml:id="b1">
                    <analytic><title type="main">Unpublished Normalization Protocol</title></analytic>
                    <monogr><imprint><date type="published" when="2022"/></imprint></monogr>
                  </biblStruct>
                </listBibl>
              </div>
            </back>
          </text>
        </TEI>
        """

        fake_manifest = {
            "ref_1": {
                "xml_id": "b0",
                "cited_by": "quantum_solver.pdf",
                "title": "Parameter-Shift Quantum Gradients",
                "doi": "10.1103/PhysRevA.99.032331",
                "path": "/tmp/pulled_pdfs/param_shift_2021.pdf",
            }
        }

        with patch("research_assistant.shared.seed_audit._load_downloaded_manifest", return_value=fake_manifest), \
             patch("research_assistant.tasks.retrieval.get_ingested_documents", return_value={"quantum_solver.pdf", "param_shift_2021.pdf"}):
            subgraph = build_paper_reference_subgraph(
                sample_tei,
                paper_title="quantum_solver.pdf",
            )

        self.assertEqual(subgraph["seed_doc_key"], "quantum_solver.pdf")
        self.assertIn("param_shift_2021.pdf", subgraph["cited_doc_keys"])
        self.assertIn("param_shift_2021.pdf", subgraph["method_doc_keys"])
        self.assertEqual(len(subgraph["method_citations"]), 1)
        self.assertEqual(subgraph["method_citations"][0]["cited_doc_key"], "param_shift_2021.pdf")
        self.assertEqual(len(subgraph["unresolved_method_citations"]), 1)
        self.assertEqual(
            subgraph["unresolved_method_citations"][0]["ref_title"],
            "Unpublished Normalization Protocol",
        )

        # Stage 2: Dereference method citations against mock search resources
        mock_collection = MagicMock()
        mock_bm25 = MagicMock()
        texts = [
            "Preceding context for parameter-shift derivation.",
            "Exact parameter-shift rule: d/dtheta <H> = 0.5 * (<H(theta + pi/2)> - <H(theta - pi/2)>).",
            "Following context with variance bounds.",
            "Seed paper method overview chunk.",
        ]
        metadatas = [
            {"document": "param_shift_2021.pdf", "section": "Methods", "page": 3, "type": "text"},
            {"document": "param_shift_2021.pdf", "section": "Methods", "page": 3, "type": "text"},
            {"document": "param_shift_2021.pdf", "section": "Methods", "page": 4, "type": "text"},
            {"document": "quantum_solver.pdf", "section": "Methodology", "page": 2, "type": "text"},
        ]

        def fake_hybrid_search(query, collection, bm25, texts, metadatas, top_k=5, doc_filter=None, **kwargs):
            if doc_filter and "param_shift_2021.pdf" in doc_filter:
                return [{
                    "chunk_index": 1,
                    "text": texts[1],
                    "metadata": metadatas[1],
                    "rrf_score": 0.032,
                }]
            return [{
                "chunk_index": 3,
                "text": texts[3],
                "metadata": metadatas[3],
                "rrf_score": 0.031,
            }]

        with patch("research_assistant.tasks.retrieval.hybrid_search", side_effect=fake_hybrid_search):
            deref = dereference_method_citations(
                subgraph,
                search_resources=(mock_collection, mock_bm25, texts, metadatas),
            )
            self.assertEqual(len(deref), 1)
            self.assertEqual(deref[0]["cited_doc_key"], "param_shift_2021.pdf")
            # Neighbor chunks (0 and 2) should be expanded into the passage text
            passage_text = deref[0]["passages"][0]["text"]
            self.assertIn("Exact parameter-shift rule", passage_text)
            self.assertIn("Preceding context", passage_text)
            self.assertIn("Following context", passage_text)

            # Stage 3: Populate TaskContextPack on a TaskGraph and test serialization + prompt + query
            graph = TaskGraph(paper_id="quantum_solver.pdf", paper_title="Quantum Variational Solver Paper")
            t_model = Task(
                task_id="task_03_model",
                title="Implement Parameter-Shift Gradient Estimator",
                category=TaskCategory.MODEL_IMPLEMENTATION,
                description="Implement parameter-shift gradient estimator [1] for the variational circuit.",
                inputs=["env_ready.lock"],
                outputs=["src/gradients.py"],
                acceptance_criteria=["Gradient matches finite difference within 1e-6"],
            )
            graph.add_task(t_model)

            populate_graph_context_packs(
                graph,
                subgraph=subgraph,
                dereferenced_citations=deref,
                search_resources=(mock_collection, mock_bm25, texts, metadatas),
            )

            self.assertIsNotNone(t_model.context_pack)
            self.assertFalse(t_model.context_pack.is_empty())
            self.assertTrue(len(t_model.context_pack.seed_passages) >= 1)
            self.assertTrue(len(t_model.context_pack.cited_protocols) >= 1)
            self.assertEqual(len(t_model.context_pack.unresolved_citations), 1)

            # Round-trip serialization check
            restored = TaskGraph.from_dict(graph.to_dict())
            restored_task = restored.get_task("task_03_model")
            self.assertIsNotNone(restored_task.context_pack)
            self.assertEqual(
                restored_task.context_pack.cited_protocols[0]["cited_doc_key"],
                "param_shift_2021.pdf",
            )
            self.assertIn("param_shift_2021.pdf", restored.subgraph_doc_keys())

            # Subagent prompt check
            prompt = format_subagent_prompt(restored_task, plan_path="plan.json")
            self.assertIn("## Grounded Reference Database Context Pack", prompt)
            self.assertIn("Dereferenced Cited Protocols", prompt)
            self.assertIn("param_shift_2021.pdf", prompt)
            self.assertIn("python orchestrate_tasks.py query plan.json --task-id task_03_model", prompt)

            # On-demand query_task_database check
            q_res = query_task_database(
                query="parameter shift gradient formula",
                graph_or_subgraph=restored,
                task_id="task_03_model",
                scope="cited",
                search_resources=(mock_collection, mock_bm25, texts, metadatas),
            )
            self.assertEqual(q_res["scope"], "cited")
            self.assertTrue(len(q_res["results"]) >= 1)
            self.assertEqual(q_res["results"][0]["document"], "param_shift_2021.pdf")
            self.assertIn("Exact parameter-shift rule", q_res["results"][0]["text"])

    def test_budget_context_blocks_enforces_token_limits(self):
        from research_assistant.tasks.retrieval import budget_context_blocks

        blocks = [
            {"text": "A" * 1000, "score": 0.95},
            {"text": "B" * 2000, "score": 0.85},
            {"text": "C" * 3000, "score": 0.70},
        ]
        # 1000 chars ~ 285 tokens. With max_tokens=600 (~2100 chars), block C must be truncated/excluded
        budgeted = budget_context_blocks(blocks, max_tokens=600, chars_per_token=3.5)
        total_len = sum(len(b["text"]) for b in budgeted)
        self.assertLessEqual(total_len, 2200)
        self.assertGreaterEqual(len(budgeted), 1)


if __name__ == "__main__":
    unittest.main()
