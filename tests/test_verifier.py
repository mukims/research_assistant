"""Tests for agent 8's resolution logic.

Everything here is pure: mapping inversion, cite stripping, and document
resolution against a fake collection. No network, no ChromaDB, no model.
"""

import unittest

from research_assistant.agents.agent8_verifier import (
    citation_pairs,
    invert_citation_mapping,
    resolve_documents,
    strip_citations,
)


class FakeCollection:
    """Stands in for a ChromaDB collection's .get(where=...)."""

    def __init__(self, rows):
        # rows: list of (citation_source, document)
        self.rows = rows
        self.calls = []

    def get(self, where=None, include=None):
        self.calls.append(where)
        source = where["citation_source"]
        metas = [{"document": doc} for src, doc in self.rows if src == source]
        return {"ids": [str(i) for i in range(len(metas))], "metadatas": metas}


class TestInvertCitationMapping(unittest.TestCase):
    def test_inverts_source_to_key(self):
        self.assertEqual(
            invert_citation_mapping({"Smith 2020": "cite_1", "Jones 2019": "cite_2"}),
            {"cite_1": "Smith 2020", "cite_2": "Jones 2019"},
        )

    def test_empty_mapping(self):
        self.assertEqual(invert_citation_mapping({}), {})

    def test_two_sources_sharing_a_key_keeps_one(self):
        """Shouldn't happen — agent 5 mints a key per source — but inversion
        must not raise if it ever does."""
        inverted = invert_citation_mapping({"Smith 2020": "cite_1", "Jones 2019": "cite_1"})
        self.assertEqual(set(inverted), {"cite_1"})
        self.assertIn(inverted["cite_1"], {"Smith 2020", "Jones 2019"})


class TestStripCitations(unittest.TestCase):
    def test_removes_single_cite(self):
        self.assertEqual(
            strip_citations("Graphene conducts well \\cite{cite_1}."),
            "Graphene conducts well.",
        )

    def test_removes_multi_key_cite(self):
        self.assertEqual(
            strip_citations("Graphene conducts well \\cite{cite_1,cite_2}."),
            "Graphene conducts well.",
        )

    def test_removes_several_cites_in_one_sentence(self):
        self.assertEqual(
            strip_citations("A \\cite{cite_1} and B \\cite{cite_2} differ."),
            "A and B differ.",
        )

    def test_sentence_without_cites_is_unchanged(self):
        self.assertEqual(strip_citations("Nothing here."), "Nothing here.")

    def test_collapses_the_gap_left_behind(self):
        self.assertEqual(strip_citations("A \\cite{x}  B"), "A B")


class TestResolveDocuments(unittest.TestCase):
    def test_returns_documents_for_a_source(self):
        col = FakeCollection([("Smith 2020", "a.pdf"), ("Smith 2020", "a.pdf"),
                              ("Jones 2019", "b.pdf")])
        self.assertEqual(resolve_documents(col, "Smith 2020", {}), {"a.pdf"})

    def test_unknown_source_returns_empty_set(self):
        col = FakeCollection([("Smith 2020", "a.pdf")])
        self.assertEqual(resolve_documents(col, "Nobody 1999", {}), set())

    def test_result_is_cached_across_calls(self):
        """A source cited twenty times must cost one Chroma call, not twenty."""
        col = FakeCollection([("Smith 2020", "a.pdf")])
        cache = {}
        resolve_documents(col, "Smith 2020", cache)
        resolve_documents(col, "Smith 2020", cache)
        self.assertEqual(len(col.calls), 1)

    def test_empty_result_is_cached_too(self):
        col = FakeCollection([])
        cache = {}
        resolve_documents(col, "Nobody 1999", cache)
        resolve_documents(col, "Nobody 1999", cache)
        self.assertEqual(len(col.calls), 1)


class TestCitationPairs(unittest.TestCase):
    def test_one_pair_per_cited_sentence(self):
        pairs = citation_pairs(
            ["Uncited sentence.", "Cited one \\cite{cite_1}."],
            {"cite_1": "Smith 2020"},
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["sentence_index"], 1)
        self.assertEqual(pairs[0]["cite_key"], "cite_1")
        self.assertEqual(pairs[0]["citation_source"], "Smith 2020")
        self.assertEqual(pairs[0]["claim"], "Cited one.")
        self.assertIsNone(pairs[0]["outcome"])

    def test_two_sources_in_one_sentence_yield_two_pairs(self):
        pairs = citation_pairs(
            ["Both agree \\cite{cite_1,cite_2}."],
            {"cite_1": "Smith 2020", "cite_2": "Jones 2019"},
        )
        self.assertEqual(len(pairs), 2)
        self.assertEqual({p["cite_key"] for p in pairs}, {"cite_1", "cite_2"})
        # Both judge the same claim, with the LaTeX removed.
        self.assertEqual({p["claim"] for p in pairs}, {"Both agree."})

    def test_key_absent_from_mapping_is_orphaned(self):
        """Agent 5 warns about invented keys; this is where they surface."""
        pairs = citation_pairs(["Invented \\cite{cite_9}."], {"cite_1": "Smith 2020"})
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["outcome"], "orphaned")
        self.assertIsNone(pairs[0]["citation_source"])

    def test_pairs_are_ordered_by_sentence_then_key(self):
        pairs = citation_pairs(
            ["B \\cite{cite_2}.", "A \\cite{cite_1}."],
            {"cite_1": "Jones 2019", "cite_2": "Smith 2020"},
        )
        self.assertEqual([p["sentence_index"] for p in pairs], [0, 1])

    def test_no_citations_yields_nothing(self):
        self.assertEqual(citation_pairs(["Plain text."], {"cite_1": "Smith 2020"}), [])
