"""Tests for hybrid retrieval.

Two things matter here. The top-k candidates must be the genuine top-k by score
now that the sparse side partitions instead of fully sorting, and a dense hit
whose id falls outside the loaded chunks must be dropped rather than used to
index the wrong document.
"""

import unittest

import numpy as np

from research_assistant.shared.search import hybrid_search


class FakeBM25:
    def __init__(self, scores):
        self._scores = np.asarray(scores, dtype=float)

    def get_scores(self, tokenized_query):
        return self._scores


class FakeCollection:
    """Returns a fixed dense ranking, ignoring the query embedding."""

    def __init__(self, ids):
        self._ids = list(ids)

    def query(self, **kwargs):
        return {
            "ids": [self._ids],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }


class FakeEmbeddings:
    def embed_query(self, text):
        return [0.0, 0.0, 0.0]


def _corpus(n):
    texts = [f"chunk text {i}" for i in range(n)]
    metadatas = [{"citation_source": f"src{i}", "document": f"doc{i}.pdf"} for i in range(n)]
    return texts, metadatas


class TestSparseRanking(unittest.TestCase):
    """With no dense hits, the result order is purely the BM25 ranking."""

    def _search(self, scores, top_k=3, dense_ids=()):
        texts, metadatas = _corpus(len(scores))
        return hybrid_search(
            "q",
            FakeCollection(dense_ids),
            FakeBM25(scores),
            texts,
            metadatas,
            top_k=top_k,
            embeddings_model=FakeEmbeddings(),
        )

    def test_returns_the_highest_scoring_chunks(self):
        scores = [0.0] * 100
        scores[42], scores[7], scores[93] = 9.0, 8.0, 7.0

        results = self._search(scores, top_k=3)

        self.assertEqual([r["chunk_index"] for r in results], [42, 7, 93])

    def test_partitioning_agrees_with_a_full_sort(self):
        """The corpus is far larger than k_cand, so the partition path runs."""
        rng = np.random.default_rng(0)
        scores = rng.random(5000)
        expected = list(np.argsort(scores)[::-1][:5])

        results = self._search(scores, top_k=5)

        self.assertEqual([r["chunk_index"] for r in results], expected)

    def test_small_corpus_below_the_candidate_cut(self):
        """Fewer chunks than candidates: the full-sort branch must still work."""
        scores = [0.1, 0.9, 0.5]

        results = self._search(scores, top_k=2)

        self.assertEqual([r["chunk_index"] for r in results], [1, 2])

    def test_results_carry_the_matching_text_and_metadata(self):
        scores = [0.0] * 10
        scores[4] = 5.0

        results = self._search(scores, top_k=1)

        self.assertEqual(results[0]["text"], "chunk text 4")
        self.assertEqual(results[0]["metadata"]["citation_source"], "src4")

    def test_all_equal_scores_do_not_crash(self):
        results = self._search([1.0] * 50, top_k=3)
        self.assertEqual(len(results), 3)


class TestDenseIdHandling(unittest.TestCase):
    def _search(self, dense_ids, n=20, top_k=3):
        # Distinct descending sparse scores so the sparse ordering is
        # unambiguous and the dense contribution is what moves a chunk.
        texts, metadatas = _corpus(n)
        scores = [float(n - i) for i in range(n)]
        return hybrid_search(
            "q",
            FakeCollection(dense_ids),
            FakeBM25(scores),
            texts,
            metadatas,
            top_k=top_k,
            embeddings_model=FakeEmbeddings(),
        )

    def test_a_dense_hit_is_promoted_above_its_sparse_rank(self):
        """Chunk 9 is only 10th on the sparse side; the dense hit lifts it."""
        results = self._search(["chunk_9"], n=20, top_k=3)
        self.assertIn(9, [r["chunk_index"] for r in results])

    def test_out_of_range_dense_id_is_dropped(self):
        """A stale id must not index some unrelated chunk by position."""
        results = self._search(["chunk_999"], n=20, top_k=3)

        indices = [r["chunk_index"] for r in results]
        self.assertNotIn(999, indices)
        self.assertTrue(all(0 <= i < 20 for i in indices))

    def test_malformed_dense_id_is_ignored(self):
        """A bad id contributes nothing and does not disturb the sparse order."""
        with_bad = self._search(["not-a-chunk-id"], n=20, top_k=3)
        without = self._search([], n=20, top_k=3)

        self.assertEqual(
            [r["chunk_index"] for r in with_bad],
            [r["chunk_index"] for r in without],
        )

    def test_empty_dense_result_is_fine(self):
        results = self._search([], n=20)
        self.assertEqual(len(results), 3)

    def test_chunk_index_is_always_a_plain_int(self):
        """Sparse hits arrive as numpy integers; the field must not vary."""
        results = self._search(["chunk_9"], n=20, top_k=3)
        for r in results:
            self.assertIs(type(r["chunk_index"]), int)


class TestDocFilter(unittest.TestCase):
    """The sparse-side half of doc_filter (search.py:79-83) is what keeps
    stage-2 retrieval from leaking chunks out of papers the stage-1 gate
    already rejected. It was previously untested in either direction:
    replacing its predicate with `True` (a no-op filter) left the whole
    suite green. The dense side is exercised too (via collection.query's
    `where`), but FakeCollection ignores kwargs and returns a fixed id list
    regardless, so this test isolates the sparse-side predicate specifically
    by returning no dense hits at all."""

    def test_sparse_side_excludes_documents_outside_the_filter(self):
        # Index 0 is the single highest BM25 scorer, but its document is not
        # in doc_filter. A no-op predicate (e.g. mutated to `True`) would let
        # it leak through as the top result anyway.
        scores = [10.0, 5.0, 4.0, 3.0, 2.0]
        texts, metadatas = _corpus(len(scores))

        results = hybrid_search(
            "q",
            FakeCollection([]),   # no dense hits — isolates the sparse side
            FakeBM25(scores),
            texts,
            metadatas,
            top_k=2,
            embeddings_model=FakeEmbeddings(),
            doc_filter={"doc1.pdf", "doc2.pdf"},
        )

        indices = [r["chunk_index"] for r in results]
        self.assertNotIn(0, indices, "doc0.pdf is outside doc_filter and must not leak in")
        self.assertEqual(indices, [1, 2])
        for r in results:
            self.assertIn(r["metadata"]["document"], {"doc1.pdf", "doc2.pdf"})


class TestFusion(unittest.TestCase):
    def test_a_chunk_found_by_both_outranks_one_found_by_either(self):
        """That is the whole point of reciprocal rank fusion."""
        n = 50
        scores = [0.0] * n
        scores[10] = 5.0    # sparse rank 1
        scores[20] = 4.0    # sparse rank 2
        texts, metadatas = _corpus(n)

        results = hybrid_search(
            "q",
            FakeCollection(["chunk_20"]),   # dense also likes 20
            FakeBM25(scores),
            texts,
            metadatas,
            top_k=2,
            embeddings_model=FakeEmbeddings(),
        )

        self.assertEqual(results[0]["chunk_index"], 20)


class TestQueryTokenizerFollowsThePickle(unittest.TestCase):
    """The query is tokenised the way the index was built."""

    class _RecordingBM25:
        def __init__(self, version):
            self.tokenizer_version = version
            self.seen = None

        def get_scores(self, tokenized_query):
            self.seen = list(tokenized_query)
            return np.asarray([1.0, 0.5])

    def _run(self, bm25):
        texts, metadatas = _corpus(2)
        hybrid_search("The ﬁelds", FakeCollection([]), bm25, texts, metadatas,
                      top_k=1, embeddings_model=FakeEmbeddings())
        return bm25.seen

    def test_v2_pickle_gets_stemmed_normalised_tokens(self):
        self.assertEqual(self._run(self._RecordingBM25("v2")), ["field"])

    def test_v1_pickle_gets_legacy_tokens(self):
        self.assertEqual(self._run(self._RecordingBM25("v1")), ["the", "ﬁelds"])

    def test_untagged_object_is_treated_as_v1(self):
        bm25 = self._RecordingBM25("v1")
        del bm25.tokenizer_version
        self.assertEqual(self._run(bm25), ["the", "ﬁelds"])


class TestFilterBeforeThePoolIsCut(unittest.TestCase):
    """The sparse side must rank within the allowed chunks, not rank everything
    and then filter. Cutting a global top-k first meant that on a big corpus the
    cited paper's own keyword hits fell out of the pool and BM25 silently
    returned nothing for the one paper the auditor asked about."""

    N = 300

    def _corpus(self, doc_of):
        texts = [f"chunk {i}" for i in range(self.N)]
        metas = [{"document": doc_of(i), "type": "text_chunk"} for i in range(self.N)]
        return texts, metas

    def _scores(self):
        # Chunks 0-4 are paper a's, with the weakest scores; the other 295 all outrank them.
        return [1.0 - 0.1 * i if i < 5 else 50.0 + i for i in range(self.N)]

    def test_doc_filter_still_finds_a_paper_whose_chunks_score_low(self):
        texts, metas = self._corpus(lambda i: "a.pdf" if i < 5 else "b.pdf")
        out = hybrid_search("q", FakeCollection([]), FakeBM25(self._scores()), texts, metas, top_k=3,
                            embeddings_model=FakeEmbeddings(), doc_filter={"a.pdf"})
        self.assertEqual([r["chunk_index"] for r in out], [0, 1, 2])

    def test_exclude_docs_still_finds_what_is_left(self):
        texts, metas = self._corpus(lambda i: "a.pdf" if i < 5 else "b.pdf")
        out = hybrid_search("q", FakeCollection([]), FakeBM25(self._scores()), texts, metas, top_k=3,
                            embeddings_model=FakeEmbeddings(), exclude_docs={"b.pdf"})
        self.assertEqual([r["chunk_index"] for r in out], [0, 1, 2])

    def test_exclude_types_still_finds_what_is_left(self):
        texts, metas = self._corpus(lambda i: "a.pdf")
        for i in range(5, self.N):
            metas[i]["type"] = "figure_image"
        out = hybrid_search("q", FakeCollection([]), FakeBM25(self._scores()), texts, metas, top_k=3,
                            embeddings_model=FakeEmbeddings(), exclude_types={"figure_image"})
        self.assertEqual([r["chunk_index"] for r in out], [0, 1, 2])

    def test_an_unfiltered_search_is_unchanged(self):
        texts, metas = self._corpus(lambda i: "a.pdf" if i < 5 else "b.pdf")
        out = hybrid_search("q", FakeCollection([]), FakeBM25(self._scores()), texts, metas, top_k=3,
                            embeddings_model=FakeEmbeddings())
        self.assertEqual([r["chunk_index"] for r in out], [self.N - 1, self.N - 2, self.N - 3])

    def test_the_real_bm25_agrees_with_scoring_everything(self):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            self.skipTest("rank-bm25 not installed")
        from research_assistant.shared.tokenize import tokenize
        texts = ["graphene conductance", "graphene ribbons edge states", "unrelated stock market index",
                 "graphene edge conductance measurement", "market volatility", "graphene"]
        metas = [{"document": d, "type": "text_chunk"} for d in ("a.pdf", "a.pdf", "b.pdf", "a.pdf", "b.pdf", "b.pdf")]
        bm25 = BM25Okapi([tokenize(t) for t in texts])
        bm25.tokenizer_version = "v2"
        out = hybrid_search("graphene edge conductance", FakeCollection([]), bm25, texts, metas, top_k=3,
                            embeddings_model=FakeEmbeddings(), doc_filter={"a.pdf"})
        scores = bm25.get_scores(tokenize("graphene edge conductance"))
        expected = sorted((i for i, m in enumerate(metas) if m["document"] == "a.pdf" and scores[i] > 0),
                          key=lambda i: -scores[i])
        self.assertEqual([r["chunk_index"] for r in out if scores[r["chunk_index"]] > 0], expected[:3])


class TestExcludeTypes(unittest.TestCase):
    """A verdict must rest on what the paper says, not on a model's reading of
    a plot: Agent 8 excludes figure_description chunks from evidence."""

    class _Coll(FakeCollection):
        def __init__(self, ids):
            super().__init__(ids)
            self.where = None

        def query(self, **kwargs):
            self.where = kwargs.get("where")
            return super().query(**kwargs)

    def test_sparse_side_drops_excluded_types(self):
        texts = ["a", "b", "c"]
        metas = [{"document": "d.pdf", "type": "text_chunk"},
                 {"document": "d.pdf", "type": "figure_description"},
                 {"document": "d.pdf", "type": "caption"}]
        out = hybrid_search("q", FakeCollection([]), FakeBM25([1.0, 3.0, 2.0]), texts, metas,
                            top_k=3, embeddings_model=FakeEmbeddings(), exclude_types={"figure_description"})
        self.assertEqual([r["chunk_index"] for r in out], [2, 0])

    def test_dense_where_combines_document_and_type_filters(self):
        coll = self._Coll([])
        texts, metas = _corpus(2)
        hybrid_search("q", coll, FakeBM25([1.0, 1.0]), texts, metas, top_k=1,
                      embeddings_model=FakeEmbeddings(), doc_filter={"doc0.pdf"}, exclude_types={"figure_description"})
        self.assertEqual(coll.where, {"$and": [{"document": {"$in": ["doc0.pdf"]}},
                                               {"type": {"$nin": ["figure_description"]}}]})

    def test_the_shared_evidence_exclusion_drops_images_on_both_sides(self):
        from research_assistant.config import EXCLUDED_FROM_EVIDENCE
        texts = ["a", "image of the figure", "c"]
        metas = [{"document": "d.pdf", "type": "text_chunk"},
                 {"document": "d.pdf", "type": "figure_image"},
                 {"document": "d.pdf", "type": "caption"}]
        out = hybrid_search("q", FakeCollection([]), FakeBM25([1.0, 9.0, 2.0]), texts, metas,
                            top_k=3, embeddings_model=FakeEmbeddings(), exclude_types=EXCLUDED_FROM_EVIDENCE)
        self.assertEqual([r["chunk_index"] for r in out], [2, 0])
        coll = self._Coll([])
        hybrid_search("q", coll, FakeBM25([1.0, 1.0, 1.0]), texts, metas, top_k=1,
                      embeddings_model=FakeEmbeddings(), exclude_types=EXCLUDED_FROM_EVIDENCE)
        self.assertEqual(coll.where, {"type": {"$nin": ["figure_description", "figure_image"]}})

    def test_dense_where_is_none_when_nothing_is_filtered(self):
        coll = self._Coll([])
        texts, metas = _corpus(2)
        hybrid_search("q", coll, FakeBM25([1.0, 1.0]), texts, metas, top_k=1, embeddings_model=FakeEmbeddings())
        self.assertIsNone(coll.where)


class TestExcludeDocs(unittest.TestCase):
    """The citer's evaluation runs on a seed paper's own sentences. Two of the
    three seeds are in the chunk index, so without an exclusion retrieval
    would hand the citer the seed's own text and the score would be a lie."""

    class _Coll(FakeCollection):
        def __init__(self, ids):
            super().__init__(ids)
            self.where = None
            self.n_results = None

        def query(self, **kwargs):
            self.where = kwargs.get("where")
            self.n_results = kwargs.get("n_results")
            return super().query(**kwargs)

    def test_sparse_side_drops_excluded_documents(self):
        texts = ["a", "b", "c"]
        metas = [{"document": "seed.pdf"}, {"document": "ref.pdf"}, {"document": "seed.pdf"}]
        out = hybrid_search("q", FakeCollection([]), FakeBM25([3.0, 2.0, 1.0]), texts, metas,
                            top_k=3, embeddings_model=FakeEmbeddings(), exclude_docs={"seed.pdf"})
        self.assertEqual([r["chunk_index"] for r in out], [1])

    def test_exclusion_widens_the_candidate_pool_like_doc_filter(self):
        # The sparse side drops the excluded chunks after ranking; a seed's
        # own chunks dominate a query written from its sentences, so the
        # narrow 15-deep pool lost a mean 3.8 of 15 candidates on nature12952.
        coll = self._Coll([])
        texts, metas = _corpus(2)
        hybrid_search("q", coll, FakeBM25([1.0, 1.0]), texts, metas, top_k=1,
                      embeddings_model=FakeEmbeddings(), exclude_docs={"seed.pdf"})
        self.assertEqual(coll.n_results, 60)
        hybrid_search("q", coll, FakeBM25([1.0, 1.0]), texts, metas, top_k=1, embeddings_model=FakeEmbeddings())
        self.assertEqual(coll.n_results, 15)

    def test_dense_where_carries_the_exclusion(self):
        coll = self._Coll([])
        texts, metas = _corpus(2)
        hybrid_search("q", coll, FakeBM25([1.0, 1.0]), texts, metas, top_k=1,
                      embeddings_model=FakeEmbeddings(), exclude_docs={"doc0.pdf"})
        self.assertEqual(coll.where, {"document": {"$nin": ["doc0.pdf"]}})

    def test_exclusion_combines_with_the_other_filters(self):
        coll = self._Coll([])
        texts, metas = _corpus(2)
        hybrid_search("q", coll, FakeBM25([1.0, 1.0]), texts, metas, top_k=1,
                      embeddings_model=FakeEmbeddings(),
                      doc_filter={"doc0.pdf", "doc1.pdf"}, exclude_docs={"doc1.pdf"}, exclude_types={"caption"})
        self.assertEqual(coll.where, {"$and": [
            {"document": {"$in": ["doc0.pdf", "doc1.pdf"]}},
            {"document": {"$nin": ["doc1.pdf"]}},
            {"type": {"$nin": ["caption"]}},
        ]})

    def test_no_exclusion_changes_nothing(self):
        coll = self._Coll([])
        texts, metas = _corpus(2)
        hybrid_search("q", coll, FakeBM25([1.0, 1.0]), texts, metas, top_k=1,
                      embeddings_model=FakeEmbeddings(), exclude_docs=set())
        self.assertIsNone(coll.where)


from research_assistant.shared.search import expand_neighbours


class TestExpandNeighbours(unittest.TestCase):
    def _corpus(self):
        texts = ["a0", "a1", "a2", "b0", "b1"]
        metas = [{"document": "a.pdf"}] * 3 + [{"document": "b.pdf"}] * 2
        return texts, metas

    def test_adds_the_adjacent_chunks_of_the_same_document(self):
        texts, metas = self._corpus()
        out = expand_neighbours([{"chunk_index": 1, "text": "a1", "metadata": metas[1]}], texts, metas)
        self.assertEqual((out[0]["context_before"], out[0]["context_after"]), ("a0", "a2"))

    def test_never_crosses_into_another_document(self):
        texts, metas = self._corpus()
        out = expand_neighbours([{"chunk_index": 2, "text": "a2", "metadata": metas[2]}], texts, metas)
        self.assertEqual((out[0]["context_before"], out[0]["context_after"]), ("a1", ""))
        out = expand_neighbours([{"chunk_index": 3, "text": "b0", "metadata": metas[3]}], texts, metas)
        self.assertEqual((out[0]["context_before"], out[0]["context_after"]), ("", "b1"))

    def test_window_two_joins_both_sides(self):
        texts, metas = self._corpus()
        out = expand_neighbours([{"chunk_index": 2, "text": "a2", "metadata": metas[2]}], texts, metas, window=2)
        self.assertEqual(out[0]["context_before"], "a0\na1")

    def test_a_figure_image_or_old_description_is_never_context(self):
        texts = ["t0", "Figure 1: caption", "Image of Figure 1: caption", "t1", "Auto-generated description"]
        metas = [{"document": "a.pdf", "type": "text_chunk"}, {"document": "a.pdf", "type": "caption"},
                 {"document": "a.pdf", "type": "figure_image"}, {"document": "a.pdf", "type": "text_chunk"},
                 {"document": "a.pdf", "type": "figure_description"}]
        out = expand_neighbours([{"chunk_index": 1, "text": texts[1], "metadata": metas[1]}], texts, metas)
        self.assertEqual((out[0]["context_before"], out[0]["context_after"]), ("t0", ""))
        out = expand_neighbours([{"chunk_index": 3, "text": "t1", "metadata": metas[3]}], texts, metas)
        self.assertEqual((out[0]["context_before"], out[0]["context_after"]), ("", ""))

    def test_window_zero_and_out_of_range_are_safe(self):
        texts, metas = self._corpus()
        out = expand_neighbours([{"chunk_index": 0, "text": "a0", "metadata": metas[0]}], texts, metas, window=0)
        self.assertEqual((out[0]["context_before"], out[0]["context_after"]), ("", ""))
        out = expand_neighbours([{"chunk_index": 99, "text": "x", "metadata": {"document": "z"}}], texts, metas)
        self.assertEqual((out[0]["context_before"], out[0]["context_after"]), ("", ""))

    def test_rerank_candidates_preserves_clean_fallback(self):
        from research_assistant.shared.search import rerank_candidates
        candidates = [
            {"text": "irrelevant text", "score": 0.5},
            {"text": "quantum computing superconductor", "score": 0.9},
        ]
        reranked = rerank_candidates("quantum physics", candidates, top_k=2)
        self.assertEqual(len(reranked), 2)
        # Should have cross_encoder_score or keep original items
        self.assertTrue("text" in reranked[0])


if __name__ == "__main__":
    unittest.main()
