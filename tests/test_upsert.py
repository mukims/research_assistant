"""Tests for what actually reaches the corpus.

upsert_corpus() is the one place chunks become citable records, and it had no
test: the heavy collaborators (ChromaDB, SemanticChunker, the embedding stack)
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
        self.ids, self.documents, self.metadatas = [], [], []

    def add(self, embeddings, documents, metadatas, ids):
        for doc, meta, cid in zip(documents, metadatas, ids):
            if cid in self.ids:
                raise ValueError(f"duplicate id {cid}")
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

        # SemanticChunker splits on blank lines here; the real breakpoint logic
        # is an embedding-distance calculation and is not what these test.
        splitter = types.ModuleType("langchain_experimental.text_splitter")

        class FakeChunker:
            def __init__(self, embeddings, **kw):
                pass

            def create_documents(self, texts):
                return [
                    types.SimpleNamespace(page_content=part)
                    for text in texts
                    for part in text.split("\n\n")
                    if part.strip()
                ]

        splitter.SemanticChunker = FakeChunker
        experimental = types.ModuleType("langchain_experimental")
        experimental.text_splitter = splitter

        patcher = patch.dict(
            sys.modules,
            {
                "chromadb": chroma,
                "langchain_experimental": experimental,
                "langchain_experimental.text_splitter": splitter,
            },
        )
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
            "type": "text",
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


if __name__ == "__main__":
    unittest.main()
