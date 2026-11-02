"""LanceDB is the default store, so it gets the same ingest-then-search coverage Chroma has.

The vector tables used to be hard-coded to 768 dimensions: an embedder of any other
width failed on the first add() with "Cannot cast to FixedSizeList(768)". The width now
comes from the embedder (llm.embedding_dim -> embedder_record -> get_or_create_collection),
and an existing table of another width is refused with the fix in the message instead of
failing (or worse, truncating) later.

tests/conftest.py turns LanceDB off by default; these tests opt in by patching the names
db.py imported from config.
"""

import shutil
import tempfile
import types
import unittest
from unittest.mock import patch

import research_assistant.shared.db as db
import research_assistant.shared.ingestion as ing
import research_assistant.shared.lance_store as lance_store
from research_assistant.shared.lance_store import EmbeddingDimensionMismatch, LanceStoreClient
from research_assistant.shared.search import hybrid_search


def _row(dim, hot=0, text="a sentence about physics"):
    return dict(
        embeddings=[[1.0 if i == hot else 0.0 for i in range(dim)]],
        documents=[text],
        metadatas=[{"document": "p.pdf", "citation_source": "P 2020", "type": "text_chunk", "page": 1, "seq": 0}],
        ids=["chk_1"],
    )


class _TmpDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test-lance-ingest-")
        self.addCleanup(shutil.rmtree, self.tmp, True)


class TestTableWidth(_TmpDirCase):
    def test_new_table_takes_the_embedder_width(self):
        client = LanceStoreClient(self.tmp)
        table = client.get_or_create_collection("physics_papers_v3", {"embedding_dim": 3})
        table.add(**_row(3))
        self.assertEqual(table.count(), 1)

    def test_existing_768_table_still_opens_with_768_embedder(self):
        client = LanceStoreClient(self.tmp)
        table = client.get_or_create_collection("physics_papers_v3", {"embedding_dim": 768})
        table.add(**_row(768))

        reopened = LanceStoreClient(self.tmp).get_or_create_collection(
            "physics_papers_v3", {"embedding_dim": 768}
        )
        self.assertEqual(reopened.count(), 1)

    def test_width_mismatch_names_the_fix(self):
        client = LanceStoreClient(self.tmp)
        client.get_or_create_collection("physics_papers_v3", {"embedding_dim": 3})
        with self.assertRaisesRegex(EmbeddingDimensionMismatch, "CITATION_INDEX_VERSION"):
            client.get_or_create_collection("physics_papers_v3", {"embedding_dim": 5})

    def test_the_mismatch_is_a_value_error_naming_both_widths(self):
        client = LanceStoreClient(self.tmp)
        client.get_or_create_collection("physics_summaries_v3", {"embedding_dim": 3})
        with self.assertRaisesRegex(ValueError, r"physics_summaries_v3 holds 3-d vectors .* 5-d"):
            client.get_or_create_collection("physics_summaries_v3", {"embedding_dim": 5})

    def test_a_papers_or_summaries_table_needs_a_width(self):
        client = LanceStoreClient(self.tmp)
        for name in ("physics_papers_v3", "physics_summaries_v3"):
            for metadata in (None, {}, {"embedder": "x"}):
                with self.assertRaisesRegex(ValueError, "embedding_dim"):
                    client.get_or_create_collection(name, metadata)

    def test_quarantine_table_needs_no_width(self):
        table = LanceStoreClient(self.tmp).get_or_create_collection("quarantine_chunks")
        self.assertEqual(table.count(), 0)


class _KeywordEmbedder:
    """Deterministic 3-d embedder: how often each of three words occurs."""

    WORDS = ("alpha", "beta", "gamma")

    def _vec(self, text):
        low = text.lower()
        return [float(low.count(w)) for w in self.WORDS]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class TestIngestThenSearch(_TmpDirCase):
    def _entry(self, document, content):
        return {
            "document": document, "citation": f"{document} citation", "page": 1,
            "type": "text_chunk", "content": content, "embed_text": content, "meta": {"seq": 0},
        }

    def test_ingest_then_search_on_lancedb(self):
        db.clear_search_cache()
        self.addCleanup(db.clear_search_cache)
        embedder = _KeywordEmbedder()
        with patch.object(db, "USE_LANCEDB", True), \
             patch.object(db, "LANCEDB_PATH", self.tmp), \
             patch.object(lance_store, "LANCEDB_PATH", self.tmp), \
             patch("research_assistant.shared.llm.get_embeddings", return_value=embedder), \
             patch("research_assistant.shared.search._get_embeddings", return_value=embedder):
            added = ing.upsert_corpus([
                self._entry("a.pdf", "alpha alpha particles scatter off heavy nuclei"),
                self._entry("b.pdf", "beta decay changes a neutron into a proton"),
            ])
            self.assertEqual(added, 2)

            collection, bm25, texts, metadatas = db.load_search_resources()
            self.assertTrue(getattr(collection, "is_lancedb", False))
            hits = hybrid_search("alpha particles", collection, bm25, texts, metadatas, top_k=2)

        self.assertTrue(hits)
        self.assertEqual(hits[0]["metadata"]["document"], "a.pdf")
        self.assertIn("alpha", hits[0]["text"])


class TestEmbedderRecord(unittest.TestCase):
    def test_record_carries_the_embedding_width(self):
        record = db.embedder_record(_KeywordEmbedder())
        self.assertEqual(record["embedding_dim"], 3)

    def test_embedding_dim_probes_once_per_embeddings_object(self):
        from research_assistant.shared import llm

        calls = []
        emb = types.SimpleNamespace(embed_query=lambda t: (calls.append(t), [0.0] * 7)[1])
        self.assertEqual(llm.embedding_dim(emb), 7)
        self.assertEqual(llm.embedding_dim(emb), 7)
        self.assertEqual(calls, ["dimension probe"])


if __name__ == "__main__":
    unittest.main()
