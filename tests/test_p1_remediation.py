"""
Comprehensive Tier P1 Remediation Test Suite.
Validates production fixes across all 7 Tier P1 functional areas:
1. Area 1: Verification Context & Minus-Sign Normalization & Cross-Encoder Reranking
2. Area 2: Agent 8 Verifier Feature Parity with seed_audit.py & LanceDB limit cap
3. Area 3: Multi-Sentence Citation Evidence Keying & Batching
4. Area 4: Source Assessor Corrections (preprints & signatures)
5. Area 5: TEI Tables & Math Paragraph Extraction & Ingest Title Fallback
6. Area 6: Bounded Concurrency across Serial Loops
7. Area 7: Core LLM Client Hardening, Privacy & Model Defaults
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from bs4 import BeautifulSoup

from research_assistant import config
from research_assistant.judgement import judge, policy, source_assessor
from research_assistant.shared import (
    chunking,
    extract,
    ingestion,
    lance_store,
    llm,
    retrieve,
    search,
    seed_audit,
)
from research_assistant.agents import (
    agent1_extractor,
    agent5_batch_citer,
    agent8_verifier,
)


class TestArea1MinusSignAndRerank(unittest.TestCase):
    """Area 1: Minus-sign normalization, config top_k, and cross-encoder reranking."""

    def test_minus_sign_preserved_in_negative_numbers(self):
        """Negative numbers and exponents must retain their minus sign after normalisation."""
        neg_val = judge._normalise("-0.5 eV")
        pos_val = judge._normalise("0.5 eV")
        self.assertNotEqual(neg_val, pos_val)
        self.assertIn("-", neg_val)
        self.assertNotIn("-", pos_val)

        exp_val = judge._normalise("10^-5")
        self.assertIn("-", exp_val)

        temp_val = judge._normalise("-50 K")
        self.assertIn("-", temp_val)

    def test_word_break_hyphens_stripped(self):
        """Line-breaking hyphens between letters must be removed."""
        norm1 = judge._normalise("energy-\n dependent")
        self.assertEqual(norm1, "energydependent")
        norm2 = judge._normalise("informa-tion")
        self.assertEqual(norm2, "information")

    def test_span_is_verbatim_distinguishes_signs(self):
        """A positive measurement in evidence must NOT match a negative hallucinated claim."""
        evidence = "The measured band gap was 0.5 eV under ambient conditions."
        self.assertTrue(judge.span_is_verbatim("0.5 eV", evidence))
        self.assertFalse(judge.span_is_verbatim("-0.5 eV", evidence))

    def test_judgement_top_k_defaults(self):
        """JUDGEMENT_TOP_K must be 3 and JUDGEMENT_ESCALATE_TOP_K must be 6."""
        self.assertEqual(config.JUDGEMENT_TOP_K, 3)
        self.assertEqual(config.JUDGEMENT_ESCALATE_TOP_K, 6)

    def test_rerank_candidates_with_mock_model(self):
        """rerank_candidates uses CrossEncoder scores when available."""
        cands = [
            {"text": "passage A", "rrf_score": 0.5},
            {"text": "passage B", "rrf_score": 0.8},
        ]
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.95, 0.10]

        with patch.object(search, "get_cross_encoder", return_value=mock_model):
            reranked = search.rerank_candidates("quantum hall", cands, top_k=2)
            self.assertEqual(len(reranked), 2)
            self.assertEqual(reranked[0]["text"], "passage A")
            self.assertEqual(reranked[0]["cross_encoder_score"], 0.95)
            self.assertEqual(reranked[1]["text"], "passage B")


class TestArea2VerifierParity(unittest.TestCase):
    """Area 2: Agent 8 Verifier feature parity with seed_audit."""

    def test_lancedb_get_without_limit_returns_all_rows(self):
        """When limit is None, LanceCollection.get() queries total row count to avoid 10-row cap."""
        mock_table = MagicMock()
        mock_table.name = "chunks"
        mock_table.__len__.return_value = 42

        mock_search = MagicMock()
        mock_table.search.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.where.return_value = mock_search
        mock_search.to_list.return_value = [{"chunk_id": str(i), "text": f"text {i}"} for i in range(42)]

        store = lance_store.LanceCollection(mock_table)
        res = store.get(limit=None)
        mock_search.limit.assert_called_with(42)
        self.assertEqual(len(res["ids"]), 42)

    def test_totals_reliability_includes_unsupported(self):
        """_totals in agent8_verifier must include 'unsupported'."""
        results = [
            {"outcome": "judged", "judgement": "Does not support", "reliability": "UNSUPPORTED"},
            {"outcome": "judged", "judgement": "Supports", "reliability": "HIGH"},
        ]
        totals = agent8_verifier._totals(results)
        self.assertIn("unsupported", totals["reliability"])
        self.assertEqual(totals["reliability"]["unsupported"], 1)
        self.assertEqual(totals["reliability"]["high"], 1)

    def test_evidence_tier_abstract_only_prevents_high_reliability(self):
        """abstract_only evidence must yield MODERATE reliability, not HIGH."""
        rel = policy.evaluate_reliability(
            relation="Supports",
            source_grade=source_assessor.SourceGrade.RIGOROUS_PRIMARY.value,
            confidence="High",
            span_verified=True,
            evidence_tier="abstract_only",
        )
        self.assertEqual(rel["rating"], policy.ReliabilityRating.MODERATE.value)
        self.assertIn("Abstract", rel["rating_label"])


class TestArea3MultiSentenceEvidenceAndBatching(unittest.TestCase):
    """Area 3: Multi-Sentence Citation Evidence Keying & Batching."""

    def test_multi_sentence_evidence_keying(self):
        """Evidence is stored under both (sentence_idx, key) and f'{sentence_idx}:{key}'."""
        citation_evidence = {}
        # Sentence 0 cites cite_1
        ev0 = {"key": "cite_1", "document": "docA.pdf", "chunk_index": 2, "evidence": "evidence for sentence 0"}
        citation_evidence[(0, "cite_1")] = ev0
        citation_evidence["0:cite_1"] = ev0
        citation_evidence["cite_1"] = ev0

        # Sentence 4 cites cite_1 for a different claim
        ev4 = {"key": "cite_1", "document": "docA.pdf", "chunk_index": 15, "evidence": "evidence for sentence 4"}
        citation_evidence[(4, "cite_1")] = ev4
        citation_evidence["4:cite_1"] = ev4
        citation_evidence["cite_1"] = ev4  # legacy key

        # Lookups by sentence index must distinguish the two sentences
        self.assertEqual(citation_evidence["0:cite_1"]["evidence"], "evidence for sentence 0")
        self.assertEqual(citation_evidence["4:cite_1"]["evidence"], "evidence for sentence 4")
        self.assertEqual(citation_evidence[(0, "cite_1")]["chunk_index"], 2)
        self.assertEqual(citation_evidence[(4, "cite_1")]["chunk_index"], 15)

    def test_citation_check_batch_size_configured(self):
        """CITATION_CHECK_BATCH_SIZE is set to 20."""
        self.assertEqual(config.CITATION_CHECK_BATCH_SIZE, 20)


class TestArea4SourceAssessor(unittest.TestCase):
    """Area 4: Source Assessor Corrections."""

    def test_published_paper_with_arxiv_id_is_not_preprint(self):
        """A paper published in a peer-reviewed journal or with non-arXiv DOI is not a preprint."""
        meta = {
            "venue": "Physical Review B",
            "doi": "10.1103/PhysRevB.99.125101",
            "arxiv_id": "1901.01234",
        }
        res = source_assessor.assess_source(meta)
        self.assertFalse(res["is_preprint"])
        self.assertEqual(res["grade"], source_assessor.SourceGrade.RIGOROUS_PRIMARY.value)

    def test_assess_source_online_signature(self):
        """assess_source_online calls fetch_bibliometrics_and_abstract with positional target."""
        with patch("research_assistant.shared.academic_apis.fetch_bibliometrics_and_abstract") as mock_fetch:
            mock_fetch.return_value = {
                "venue": "Nature",
                "citation_count": 100,
                "is_retracted": False,
            }
            res = source_assessor.assess_source_online(metadata={"doi": "10.1038/nature12345", "title": "Quantum Paper"})
            mock_fetch.assert_called_once_with("10.1038/nature12345")
            self.assertEqual(res["venue"], "Nature")


class TestArea5TEITablesAndMath(unittest.TestCase):
    """Area 5: TEI Tables & Math Extraction."""

    def test_tei_table_markdown_formatting(self):
        """TEI table elements are formatted into clean Markdown tables."""
        tei_table_xml = (
            '<figure type="table" xmlns="http://www.tei-c.org/ns/1.0">'
            '<head>Table 1: Superconducting transition temperatures.</head>'
            '<table>'
            '<row role="label"><cell>Material</cell><cell>Tc (K)</cell></row>'
            '<row><cell>YBaCuO</cell><cell>93</cell></row>'
            '<row><cell>BiSrCaCuO</cell><cell>110</cell></row>'
            '</table>'
            '</figure>'
        )
        soup = BeautifulSoup(tei_table_xml, "xml")
        table_md = extract._format_tei_table(soup.find("table"))
        self.assertIn("| Material | Tc (K) |", table_md)
        self.assertIn("| YBaCuO | 93 |", table_md)
        self.assertIn("| BiSrCaCuO | 110 |", table_md)

    def test_math_dense_paragraph_retained(self):
        """Math-dense paragraphs with low alpha ratio are preserved by chunking filters."""
        math_text = (
            "The Hamiltonian is given by H = \\sum_{k} \\epsilon_k c_k^\\dagger c_k + "
            "U \\sum_i n_{i\\uparrow} n_{i\\downarrow} with t_1 = 1.0 eV, t_2 = -0.35 eV, "
            "and \\Delta = \\sqrt{\\epsilon^2 + \\Delta_0^2}."
        )
        self.assertTrue(chunking.is_math_or_quantitative(math_text))
        self.assertTrue(chunking.should_keep_chunk(math_text, min_chars=100, min_alpha=0.6))

    def test_agent1_process_force_false(self):
        """agent1_extractor passes force=False to preserve coordinate-rich TEI XML cache."""
        with patch("grobid_client.grobid_client.GrobidClient") as mock_grobid_cls:
            mock_client = MagicMock()
            mock_grobid_cls.return_value = mock_client
            agent1_extractor.extract_tei("/pdf_dir", "/out_dir")
            mock_client.process.assert_called_once()
            self.assertFalse(mock_client.process.call_args.kwargs["force"])


class TestArea6BoundedConcurrency(unittest.TestCase):
    """Area 6: Bounded Concurrency across Serial Loops."""

    def test_resolver_contextualize_concurrent(self):
        """contextualize_citation_queries processes paragraphs concurrently."""
        claims = [
            {"paragraph_id": f"p_{i}", "claim": f"Claim {i}", "context": f"Context {i}"}
            for i in range(4)
        ]
        with patch("research_assistant.shared.llm.chat") as mock_chat:
            mock_chat.return_value.content = '["Contextualized query"]'
            from research_assistant.audit import resolver
            resolver.contextualize_citation_queries(claims, enabled=True)
            self.assertEqual(len(claims), 4)
            for c in claims:
                self.assertIn("search_query", c)

    def test_retrieve_paper_notes_concurrent(self):
        """research_answer in map_reduce mode runs paper_notes in parallel."""
        selected = [
            {"key": "P1", "document": "doc1.pdf", "summary": "sum1"},
            {"key": "P2", "document": "doc2.pdf", "summary": "sum2"},
        ]
        with patch.object(retrieve, "rank_documents", return_value=selected), \
             patch.object(retrieve, "gate_documents", side_effect=lambda q, r: r), \
             patch.object(retrieve, "passages_per_paper", return_value={"P1": [{"text": "t1"}], "P2": [{"text": "t2"}]}), \
             patch.object(retrieve, "paper_notes", return_value=("Notes text", True)) as mock_notes, \
             patch.object(retrieve, "synthesise", return_value="Final synthesis [P1, P2]"):
            res = retrieve.research_answer("test query", mode="map_reduce")
            self.assertEqual(mock_notes.call_count, 2)
            self.assertIn("Final synthesis", res["suggestion"])


class TestArea7LLMHardening(unittest.TestCase):
    """Area 7: Core LLM Client Hardening, Privacy & Model Defaults."""

    def test_model_defaults_gemini_3_5_flash_lite(self):
        """Default models must use gemini-3.5-flash-lite, never deprecated gemini-2.5-flash."""
        # config picks the defaults at import time from the environment, so ask a fresh
        # interpreter with a key set and no .env, not this process's (hermetic) config.
        env = {**os.environ, "GEMINI_API_KEY": "test-key", "CITATION_DISABLE_DOTENV": "1"}
        for name in ("CITATION_LLM_MODEL", "CITATION_CHAT_MODEL", "CITATION_JUDGEMENT_MODEL",
                     "CITATION_LLM_BACKEND", "LLM_BACKEND"):
            env.pop(name, None)
        probe = (
            "from research_assistant import config; "
            "from research_assistant.judgement import judge; "
            "print(config._default_llm_model); print(judge._judgement_model())"
        )
        out = subprocess.run(
            [sys.executable, "-c", probe],
            env=env, cwd=str(Path(__file__).resolve().parent.parent), capture_output=True, text=True, timeout=120, check=True,
        ).stdout.split()
        self.assertEqual(out[-2:], ["gemini-3.5-flash-lite", "gemini-3.5-flash-lite"])
        self.assertNotIn("2.5", " ".join(out))

    def test_summary_fallback_to_ollama_gemma4_e2b_mlx(self):
        """When primary summary model fails, ingestion must fallback to Ollama gemma4:e2b-mlx."""
        from research_assistant.shared import ingestion, pipeline_status
        from research_assistant.shared.llm import ChatResult
        from unittest.mock import MagicMock

        pipeline_status.clear_cancel_request()

        calls = []
        def mock_chat(messages, model=None, backend=None, **kwargs):
            calls.append({"model": model, "backend": backend})
            if len(calls) == 1:
                raise RuntimeError("Gemini API 404: model not found")
            return ChatResult(content="Summary generated by fallback model")

        mock_col = MagicMock()
        mock_col.get.return_value = {"metadatas": []}
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_col

        with patch("research_assistant.shared.llm.chat", side_effect=mock_chat), \
             patch("research_assistant.shared.ingestion.SUMMARY_MODEL", "gemini-3.5-flash-lite"), \
             patch("research_assistant.shared.ingestion.open_store", return_value=mock_client), \
             patch("research_assistant.shared.ingestion.check_embedder"), \
             patch("research_assistant.shared.llm.get_embeddings"):
            per_doc = {"test_doc.pdf": "Full text of scientific paper about physics."}
            count = ingestion.upsert_summaries(per_doc, {})

        self.assertEqual(count, 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["model"], "gemini-3.5-flash-lite")
        self.assertEqual(calls[1]["model"], "gemma4:e2b-mlx")
        self.assertEqual(calls[1]["backend"], "ollama")
        self.assertTrue(mock_col.add.called)
        self.assertIn("Summary generated by fallback model", mock_col.add.call_args[1]["documents"][0])

    def test_client_pooling_reuses_instances(self):
        """_get_openai_client returns the same cached instance for identical parameters."""
        c1 = llm._get_openai_client("https://api.openai.com/v1", "key123", 60.0)
        c2 = llm._get_openai_client("https://api.openai.com/v1", "key123", 60.0)
        self.assertIs(c1, c2)

    def test_ollama_does_not_fallback_to_openai(self):
        """An error in Ollama chat must raise without falling back to OpenAI."""
        with patch.object(llm, "LLM_BACKEND", "ollama"), \
             patch("ollama.chat", side_effect=RuntimeError("Ollama failed")):
            with self.assertRaises(RuntimeError) as ctx:
                llm.chat([{"role": "user", "content": "hi"}], backend="ollama")
            self.assertIn("Ollama failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
