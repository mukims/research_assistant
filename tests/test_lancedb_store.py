"""
Tests for LanceDB storage, hybrid search, neighbour expansion, and dead-letter quarantine.
"""

import os
import shutil
import tempfile
import unittest

from research_assistant.shared.lance_store import (
    LanceStoreClient,
    LanceTableAdapter,
    compute_chunk_id,
    compute_summary_id,
    get_lance_connection,
)
from research_assistant.shared.search import expand_neighbours, hybrid_search


class TestLanceDBStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test-lance-")
        self.client = LanceStoreClient(self.tmp)
        self.table_adapter = self.client.get_or_create_collection("test_papers")

        # Insert 6 structured chunks across two papers
        # Paper A: chunks with seq 0, 1, 2
        # Paper B: chunks with seq 0, 1, 2
        self.chunks_data = [
            {
                "chunk_id": compute_chunk_id("paper_a.pdf", "text_chunk", 0, "Introduction to quantum Hall effect in graphene."),
                "document": "paper_a.pdf",
                "citation_source": "Novoselov 2005",
                "stored_form": "text_chunk",
                "section": "intro",
                "seq": 0,
                "page": 1,
                "page_first": 1,
                "page_last": 1,
                "tier": "full_text",
                "text": "Introduction to quantum Hall effect in graphene.",
                "content_hash": "hash_a0",
                "vector": [1.0] + [0.0] * 767,
                "metadata_json": '{"section": "intro", "figure_id": null}',
            },
            {
                "chunk_id": compute_chunk_id("paper_a.pdf", "text_chunk", 1, "The Dirac points exhibit linear energy dispersion."),
                "document": "paper_a.pdf",
                "citation_source": "Novoselov 2005",
                "stored_form": "text_chunk",
                "section": "methods",
                "seq": 1,
                "page": 2,
                "page_first": 2,
                "page_last": 2,
                "tier": "full_text",
                "text": "The Dirac points exhibit linear energy dispersion.",
                "content_hash": "hash_a1",
                "vector": [0.8, 0.6] + [0.0] * 766,
                "metadata_json": '{"section": "methods"}',
            },
            {
                "chunk_id": compute_chunk_id("paper_a.pdf", "text_chunk", 2, "Conclusion: Landau levels observed at room temperature."),
                "document": "paper_a.pdf",
                "citation_source": "Novoselov 2005",
                "stored_form": "text_chunk",
                "section": "conclusion",
                "seq": 2,
                "page": 3,
                "page_first": 3,
                "page_last": 3,
                "tier": "full_text",
                "text": "Conclusion: Landau levels observed at room temperature.",
                "content_hash": "hash_a2",
                "vector": [0.5, 0.5] + [0.0] * 766,
                "metadata_json": '{"section": "conclusion"}',
            },
            {
                "chunk_id": compute_chunk_id("paper_b.pdf", "text_chunk", 0, "High-temperature superconductivity in cuprates."),
                "document": "paper_b.pdf",
                "citation_source": "Bednorz 1986",
                "stored_form": "text_chunk",
                "section": "intro",
                "seq": 0,
                "page": 1,
                "page_first": 1,
                "page_last": 1,
                "tier": "full_text",
                "text": "High-temperature superconductivity in cuprates.",
                "content_hash": "hash_b0",
                "vector": [0.0, 1.0] + [0.0] * 766,
                "metadata_json": '{"section": "intro"}',
            },
            {
                "chunk_id": compute_chunk_id("paper_b.pdf", "figure_image", 1, "Figure crop of superconducting phase diagram."),
                "document": "paper_b.pdf",
                "citation_source": "Bednorz 1986",
                "stored_form": "figure_image",
                "section": "figures",
                "seq": 1,
                "page": 2,
                "page_first": 2,
                "page_last": 2,
                "tier": "full_text",
                "text": "Figure crop of superconducting phase diagram.",
                "content_hash": "hash_b1",
                "vector": [0.0, 0.9] + [0.0] * 766,
                "metadata_json": '{"section": "figures", "type": "figure_image"}',
            },
            {
                "chunk_id": compute_chunk_id("paper_b.pdf", "text_chunk", 2, "Cooper pairs form via antiferromagnetic fluctuations."),
                "document": "paper_b.pdf",
                "citation_source": "Bednorz 1986",
                "stored_form": "text_chunk",
                "section": "theory",
                "seq": 2,
                "page": 3,
                "page_first": 3,
                "page_last": 3,
                "tier": "full_text",
                "text": "Cooper pairs form via antiferromagnetic fluctuations.",
                "content_hash": "hash_b2",
                "vector": [0.0, 0.8] + [0.0] * 766,
                "metadata_json": '{"section": "theory"}',
            },
        ]
        self.table_adapter.table.add(self.chunks_data)
        self.table_adapter.create_fts_index("text")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_content_addressed_chunk_id_determinism(self):
        cid1 = compute_chunk_id("paper.pdf", "text_chunk", 0, "sample text")
        cid2 = compute_chunk_id("paper.pdf", "text_chunk", 0, "sample text")
        cid3 = compute_chunk_id("paper.pdf", "text_chunk", 1, "sample text")
        self.assertEqual(cid1, cid2)
        self.assertNotEqual(cid1, cid3)
        self.assertTrue(cid1.startswith("chk_"))

    def test_native_hybrid_search(self):
        fake_emb = type("Emb", (), {"embed_query": lambda self, q: [1.0] + [0.0] * 767})()
        results = hybrid_search(
            query="quantum Hall graphene",
            collection=self.table_adapter,
            bm25=None,
            texts=[],
            metadatas=[],
            top_k=3,
            embeddings_model=fake_emb,
        )
        self.assertTrue(len(results) > 0)
        top = results[0]
        self.assertEqual(top["metadata"]["document"], "paper_a.pdf")
        self.assertIn("quantum Hall", top["text"])

    def test_hybrid_search_with_doc_filter(self):
        fake_emb = type("Emb", (), {"embed_query": lambda self, q: [0.0, 1.0] + [0.0] * 766})()
        results = hybrid_search(
            query="superconductivity",
            collection=self.table_adapter,
            bm25=None,
            texts=[],
            metadatas=[],
            top_k=3,
            embeddings_model=fake_emb,
            doc_filter={"paper_b.pdf"},
        )
        for r in results:
            self.assertEqual(r["metadata"]["document"], "paper_b.pdf")

    def test_hybrid_search_with_exclude_types(self):
        fake_emb = type("Emb", (), {"embed_query": lambda self, q: [0.0, 0.9] + [0.0] * 766})()
        results = hybrid_search(
            query="phase diagram",
            collection=self.table_adapter,
            bm25=None,
            texts=[],
            metadatas=[],
            top_k=5,
            embeddings_model=fake_emb,
            exclude_types={"figure_image"},
        )
        for r in results:
            self.assertNotEqual(r["metadata"]["stored_form"], "figure_image")

    def test_sequence_based_neighbour_expansion(self):
        # Middle chunk seq=1 of paper_a
        hit = {
            "chunk_index": 1,
            "text": "The Dirac points exhibit linear energy dispersion.",
            "metadata": {
                "document": "paper_a.pdf",
                "stored_form": "text_chunk",
                "seq": 1,
            },
        }
        # Provide LanceSequenceProxy with table_adapter attribute
        proxy = type("Proxy", (), {"table_adapter": self.table_adapter})()
        expand_neighbours([hit], proxy, [], window=1)
        self.assertEqual(hit["context_before"], "Introduction to quantum Hall effect in graphene.")
        self.assertEqual(hit["context_after"], "Conclusion: Landau levels observed at room temperature.")

    def test_adapter_add_upsert_delete(self):
        new_table = self.client.get_or_create_collection("test_dynamic_ops")
        new_table.add(
            ids=["chk_dyn_1", "chk_dyn_2"],
            documents=["First dynamic test document.", "Second dynamic test document."],
            metadatas=[{"document": "test1.pdf", "seq": 0}, {"document": "test2.pdf", "seq": 1}],
            embeddings=[[0.1] * 768, [0.2] * 768],
        )
        self.assertEqual(new_table.count(), 2)
        res = new_table.get(ids=["chk_dyn_1"])
        self.assertEqual(len(res["ids"]), 1)
        self.assertEqual(res["documents"][0], "First dynamic test document.")

        # Upsert: update text of chk_dyn_1 and add chk_dyn_3
        new_table.upsert(
            ids=["chk_dyn_1", "chk_dyn_3"],
            documents=["Updated first test doc.", "Third test doc."],
            metadatas=[{"document": "test1.pdf", "seq": 0}, {"document": "test3.pdf", "seq": 0}],
            embeddings=[[0.3] * 768, [0.4] * 768],
        )
        self.assertEqual(new_table.count(), 3)
        res_updated = new_table.get(ids=["chk_dyn_1"])
        self.assertEqual(res_updated["documents"][0], "Updated first test doc.")

        # Delete by id
        new_table.delete(ids=["chk_dyn_2"])
        self.assertEqual(new_table.count(), 2)

    def test_sequence_proxy_caching_and_slicing(self):
        from research_assistant.shared.db import LanceSequenceProxy
        proxy_text = LanceSequenceProxy(self.table_adapter, "text", cache_size=2)
        proxy_meta = LanceSequenceProxy(self.table_adapter, "metadata", cache_size=2)

        self.assertEqual(len(proxy_text), 6)
        # Slicing
        slice_texts = proxy_text[1:3]
        self.assertEqual(len(slice_texts), 2)
        self.assertIn("Dirac points", slice_texts[0])

        # Negative index
        last_meta = proxy_meta[-1]
        self.assertEqual(last_meta["document"], "paper_b.pdf")

    def test_dead_letter_quarantine_table(self):
        q_table = self.client.get_or_create_collection("quarantine_chunks")
        q_record = [{
            "chunk_id": "chk_bad_1",
            "document": "corrupt.pdf",
            "stored_form": "text_chunk",
            "seq": 0,
            "text": "corrupt tokens that failed embedding",
            "error_reason": "embedding_failure_timeout",
            "created_at": "2026-10-27T15:00:00Z",
        }]
        q_table.table.add(q_record)
        self.assertEqual(q_table.count(), 1)
        res = q_table.get(ids=["chk_bad_1"])
        self.assertEqual(res["ids"], ["chk_bad_1"])


if __name__ == "__main__":
    unittest.main()
