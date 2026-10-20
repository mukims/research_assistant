"""Tests for what actually reaches the corpus.

upsert_corpus() is the one place chunks become citable records, and it had no
test: the heavy collaborators (ChromaDB, the embedding stack)
are lazily imported inside the function, so the existing suite patched the
whole function out rather than exercising it. The fakes below stand in for
those three, which is enough to test the bookkeeping — which is where the
attribution bugs live.
"""

import sys
import types
import unittest
from unittest.mock import patch

import research_assistant.shared.ingestion as ing


class FakeCollection:
    """Just enough ChromaDB to exercise the id allocation and dedup logic."""

    def __init__(self):
        self.ids, self.documents, self.metadatas, self.vectors = [], [], [], []
        self.metadata = None
        self.name = "fake"

    def add(self, embeddings, documents, metadatas, ids):
        for vec, doc, meta, cid in zip(embeddings, documents, metadatas, ids):
            if cid in self.ids:
                raise ValueError(f"duplicate id {cid}")
            self.vectors.append(vec)
            self.ids.append(cid)
            self.documents.append(doc)
            self.metadatas.append(meta)

    def get(self, include=None, limit=None, offset=0, **kw):
        end = len(self.ids) if limit is None else offset + limit
        return {
            "ids": self.ids[offset:end],
            "documents": self.documents[offset:end],
            "metadatas": self.metadatas[offset:end],
        }

    def chunks_for(self, document):
        return [
            doc
            for doc, meta in zip(self.documents, self.metadatas)
            if meta["document"] == document
        ]


class UpsertTestCase(unittest.TestCase):
    def setUp(self):
        self.collection = FakeCollection()

        chroma = types.ModuleType("chromadb")
        chroma.PersistentClient = lambda path: types.SimpleNamespace(
            get_or_create_collection=lambda name, metadata=None: self.collection,
            get_collection=lambda name: self.collection,
        )

        patcher = patch.dict(sys.modules, {"chromadb": chroma})
        patcher.start()
        self.addCleanup(patcher.stop)

        embeddings = types.SimpleNamespace(
            embed_documents=lambda texts: [[0.0] for _ in texts],
            embed_query=lambda text: [0.0],
        )
        # get_embeddings is imported inside upsert_corpus from shared.llm.
        import research_assistant.shared.llm as llm_mod

        patcher = patch.object(llm_mod, "get_embeddings", lambda *a, **k: embeddings)
        patcher.start()
        self.addCleanup(patcher.stop)

        # The stage-1 summary index is a separate concern with its own LLM call.
        patcher = patch.object(ing, "upsert_summaries", lambda *a, **k: 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _entry(self, document, content, citation=None, page=1):
        return {
            "document": document,
            "citation": citation or f"{document} citation",
            "page": page,
            "type": "text_chunk",
            "content": content,
        }


class TestCrossDocumentDeduplication(UpsertTestCase):
    """A chunk shared by two papers must be indexed once per paper.

    Deduplication was global and keyed on the chunk text alone, so the second
    paper to contain a shared sentence lost it — and every retrieval of that
    passage cited whichever paper happened to be ingested first. Boilerplate
    makes this common: shared method descriptions, standard equations,
    "the remainder of this paper is organized as follows".
    """

    BOILERPLATE = "The remainder of this paper is organized as follows. " * 2

    def test_shared_text_is_kept_for_both_documents(self):
        ing.upsert_corpus([self._entry("a.pdf", self.BOILERPLATE)])
        ing.upsert_corpus([self._entry("b.pdf", self.BOILERPLATE)])

        self.assertEqual(len(self.collection.chunks_for("a.pdf")), 1)
        self.assertEqual(
            len(self.collection.chunks_for("b.pdf")), 1,
            "the second paper's copy was dropped — its passage now cites the first",
        )

    def test_shared_text_keeps_each_document_its_own_citation(self):
        ing.upsert_corpus([self._entry("a.pdf", self.BOILERPLATE, citation="Alice 2020")])
        ing.upsert_corpus([self._entry("b.pdf", self.BOILERPLATE, citation="Bob 2021")])

        citations = {m["document"]: m["citation_source"] for m in self.collection.metadatas}
        self.assertEqual(citations, {"a.pdf": "Alice 2020", "b.pdf": "Bob 2021"})

    def test_repeated_text_within_one_document_is_still_deduplicated(self):
        """Re-ingesting a paper must not double its chunks."""
        entry = self._entry("a.pdf", self.BOILERPLATE)
        ing.upsert_corpus([entry])
        ing.upsert_corpus([entry])

        self.assertEqual(len(self.collection.chunks_for("a.pdf")), 1)

    def test_duplicate_text_on_two_pages_of_one_document_is_deduplicated(self):
        header = "Physical Review Letters 130, 123456 (2023). " * 2
        ing.upsert_corpus([
            self._entry("a.pdf", header, page=1),
            self._entry("a.pdf", header, page=2),
        ])

        self.assertEqual(len(self.collection.chunks_for("a.pdf")), 1)


class TestChunkIds(UpsertTestCase):
    def test_ids_stay_contiguous_across_separate_upserts(self):
        """Retrieval treats the integer in chunk_N as a position in the loaded
        list, so a gap makes dense hits resolve to the wrong text."""
        ing.upsert_corpus([self._entry("a.pdf", "First paper body text here.")])
        ing.upsert_corpus([self._entry("b.pdf", "Second paper body text here.")])

        self.assertEqual(self.collection.ids, ["chunk_0", "chunk_1"])


class TestLongChunkDeduplication(UpsertTestCase):
    """add() truncates each document to EMBED_MAX_CHARS before storing it.

    Dedup compared the full in-memory text against that truncated stored text,
    so any chunk longer than the limit never matched itself and was re-inserted
    on every ingest — the corpus grew a fresh copy of every long chunk each run.
    """

    def test_chunk_longer_than_the_embed_limit_is_not_reinserted(self):
        long_text = "Superconductivity in twisted bilayer graphene. " * 200
        self.assertGreater(len(long_text), ing.EMBED_MAX_CHARS)

        entry = self._entry("a.pdf", long_text)
        ing.upsert_corpus([entry])
        ing.upsert_corpus([entry])

        self.assertEqual(len(self.collection.chunks_for("a.pdf")), 1)


class TestPrechunkedEntries(UpsertTestCase):
    """v2 hands upsert_corpus chunks it has already made."""

    def _chunk(self, document, content, type="text_chunk", embed_text=None, meta=None):
        return {
            "document": document, "citation": f"{document} title", "page": 2, "type": type,
            "content": content, "embed_text": embed_text, "meta": meta or {},
        }

    def test_prechunked_text_is_stored_verbatim_not_rechunked(self):
        content = "Para one.\n\nPara two."             # the fake chunker would split on the blank line
        ing.upsert_corpus([self._chunk("a.pdf", content)])
        self.assertEqual(self.collection.chunks_for("a.pdf"), [content])

    def test_embed_text_is_embedded_and_content_is_stored(self):
        seen = []
        import research_assistant.shared.llm as llm_mod
        emb = types.SimpleNamespace(embed_documents=lambda texts: (seen.extend(texts), [[0.0] for _ in texts])[1],
                                    embed_query=lambda t: [0.0])
        with patch.object(llm_mod, "get_embeddings", lambda *a, **k: emb):
            ing.upsert_corpus([self._chunk("a.pdf", "stored text", embed_text="Title: T. Section: S. stored text")])
        self.assertEqual(seen, ["Title: T. Section: S. stored text"])
        self.assertEqual(self.collection.documents, ["stored text"])

    def _figure_image(self, document="a.pdf", vector=(0.5, 0.25)):
        return self._chunk(document, "Image of Figure 1: caption text", type="figure_image",
                           meta={"figure_id": "fig_0"}) | {"embedding": list(vector)}

    def test_a_figure_image_keeps_its_own_vector_and_is_not_embedded_as_text(self):
        seen = []
        import research_assistant.shared.llm as llm_mod
        emb = types.SimpleNamespace(embed_documents=lambda texts: (seen.extend(texts), [[9.0] for _ in texts])[1],
                                    embed_query=lambda t: [0.0])
        with patch.object(llm_mod, "get_embeddings", lambda *a, **k: emb):
            ing.upsert_corpus([self._chunk("a.pdf", "body text " * 5), self._figure_image()])
        self.assertEqual(seen, ["body text " * 5])
        self.assertEqual(self.collection.vectors, [[9.0], [0.5, 0.25]])
        self.assertEqual(self.collection.metadatas[1]["type"], "figure_image")
        self.assertEqual(self.collection.metadatas[1]["figure_id"], "fig_0")

    def test_mixed_batch_keeps_vectors_with_their_own_entries(self):
        import research_assistant.shared.llm as llm_mod
        emb = types.SimpleNamespace(embed_documents=lambda texts: [[float(len(t))] for t in texts],
                                    embed_query=lambda t: [0.0])
        entries = [self._chunk("a.pdf", "first body text " * 3), self._figure_image(vector=(7.0,)),
                   self._chunk("a.pdf", "second body text " * 3), self._figure_image(vector=(8.0,)) | {"content": "Image of Figure 2: other"}]
        with patch.object(llm_mod, "get_embeddings", lambda *a, **k: emb):
            ing.upsert_corpus(entries)
        self.assertEqual([m["type"] for m in self.collection.metadatas],
                         ["text_chunk", "figure_image", "text_chunk", "figure_image"])
        self.assertEqual(self.collection.vectors[1], [7.0])
        self.assertEqual(self.collection.vectors[3], [8.0])
        self.assertEqual(self.collection.vectors[0], [float(len("first body text " * 3))])

    def test_a_new_collection_records_its_embedder(self):
        recorded = {}
        chroma = sys.modules["chromadb"]
        original = chroma.PersistentClient

        def client(path):
            real = original(path)
            inner = real.get_or_create_collection

            def goc(name, metadata=None):
                recorded[name] = metadata
                return inner(name, metadata=metadata)
            real.get_or_create_collection = goc
            return real
        with patch.object(chroma, "PersistentClient", client):
            ing.upsert_corpus([self._chunk("a.pdf", "body text " * 5)])
        self.assertEqual(recorded[ing.COLLECTION_NAME]["embedder"], ing.EMBED_MODEL)

    def test_a_collection_built_with_another_embedder_is_refused(self):
        self.collection.metadata = {"embedder": "some-other-model"}
        with self.assertRaisesRegex(RuntimeError, "some-other-model"):
            ing.upsert_corpus([self._chunk("a.pdf", "body text " * 5)])

    def test_meta_is_merged_unprefixed(self):
        ing.upsert_corpus([self._chunk("a.pdf", "x" * 30, meta={"section": "methods", "seq": 3, "flag": False})])
        m = self.collection.metadatas[0]
        self.assertEqual((m["section"], m["seq"], m["flag"]), ("methods", 3, False))
        self.assertEqual(m["type"], "text_chunk")
        self.assertEqual(m["citation_source"], "a.pdf title")

    def test_summary_source_feeds_the_summary_index_and_is_not_stored(self):
        seen = {}
        with patch.object(ing, "upsert_summaries", lambda per_doc_text, per_doc_citation: seen.update(per_doc_text) or 0):
            ing.upsert_corpus([
                self._chunk("a.pdf", "body chunk " * 5),
                {"document": "a.pdf", "citation": "a.pdf title", "page": 0, "type": "summary_source",
                 "content": "abstract + intro + conclusion"},
            ])
        self.assertEqual(seen, {"a.pdf": "abstract + intro + conclusion"})
        self.assertEqual(self.collection.chunks_for("a.pdf"), ["body chunk " * 5])

    def test_embed_documents_safe_normal(self):
        emb_mock = types.SimpleNamespace(
            embed_documents=lambda texts: [[float(len(t))] for t in texts],
            embed_query=lambda text: [float(len(text))],
        )
        texts = ["text1", "text2", "text3"]
        res = ing._embed_documents_safe(emb_mock, texts, sub_batch_size=2)
        self.assertEqual(len(res), 3)
        self.assertEqual(res, [[5.0], [5.0], [5.0]])

    def test_embed_documents_safe_sub_batch_fallback(self):
        calls = []

        def failing_embed(texts):
            calls.append(texts)
            if len(texts) > 1:
                raise RuntimeError("Batch too large for runner")
            return [[1.0] for _ in texts]

        emb_mock = types.SimpleNamespace(
            embed_documents=failing_embed,
            embed_query=lambda text: [1.0],
        )
        texts = ["a", "b", "c"]
        res = ing._embed_documents_safe(emb_mock, texts, sub_batch_size=2)
        self.assertEqual(len(res), 3)
        self.assertEqual(res, [[1.0], [1.0], [1.0]])
        # First tried batch of 2 (failed), then 1-by-1
        self.assertTrue(any(len(c) == 1 for c in calls))


if __name__ == "__main__":
    unittest.main()



class SummaryFailureTests(unittest.TestCase):
    """A model that cannot summarise must not look like a clean run."""

    def _run(self, chat_side_effect):
        col = FakeCollection()
        chroma = types.ModuleType("chromadb")
        chroma.PersistentClient = lambda path: types.SimpleNamespace(
            get_or_create_collection=lambda name, metadata=None: col,
        )
        emb = types.SimpleNamespace(embed_documents=lambda texts: [[0.0] for _ in texts])
        events = []
        import research_assistant.shared.llm as llm_mod
        with patch.dict(sys.modules, {"chromadb": chroma}), \
             patch.object(llm_mod, "get_embeddings", lambda *a, **k: emb), \
             patch.object(llm_mod, "chat", side_effect=chat_side_effect), \
             patch.object(ing.pipeline_status, "add_event", events.append), \
             patch.object(ing.pipeline_status, "update_progress", lambda **k: None), \
             patch.object(ing.pipeline_status, "is_cancel_requested", lambda: False):
            made = ing.upsert_summaries({"a.pdf": "text a", "b.pdf": "text b"}, {})
        return made, events

    def test_every_failure_is_counted_and_announced(self):
        made, events = self._run(RuntimeError("model failed to load"))
        self.assertEqual(made, 0)
        self.assertEqual(ing.last_summary_run["attempted"], 2)
        self.assertEqual(ing.last_summary_run["failed"], 2)
        self.assertIn("model failed to load", ing.last_summary_run["error"])
        self.assertTrue(any("2 of 2 summaries failed" in e for e in events), events)

    def test_partial_failure_reports_the_remainder(self):
        ok = types.SimpleNamespace(content=" a summary ")
        made, events = self._run([ok, RuntimeError("boom")])
        self.assertEqual(made, 1)
        self.assertEqual(ing.last_summary_run["failed"], 1)
        self.assertTrue(any("1 of 2 summaries failed" in e for e in events), events)

    def test_empty_summary_counts_as_failure(self):
        empty = types.SimpleNamespace(content="   ")
        made, _ = self._run([empty, empty])
        self.assertEqual(made, 0)
        self.assertEqual(ing.last_summary_run["failed"], 2)
