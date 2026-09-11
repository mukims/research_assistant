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


if __name__ == "__main__":
    unittest.main()
