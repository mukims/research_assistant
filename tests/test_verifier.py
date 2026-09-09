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


import json
import os
import tempfile
from unittest.mock import patch

from research_assistant.agents.agent8_verifier import verify_draft
from research_assistant.judgement.judge import JudgementParseError


def _verdict(judgement="Supports"):
    return {
        "slots": {
            "finding": {"assertion": "a", "verdict": "Supports"},
            "scope": {"assertion": "b", "verdict": "Supports"},
            "strength": {"assertion": "c", "verdict": "Not applicable"},
        },
        "judgement": judgement,
        "evidence_sufficiency": "sufficient",
        "confidence": "High",
        "supporting_span": "span",
        "reason": "because",
    }


class _Resources:
    """The (collection, bm25, texts, metadatas) tuple load_search_resources returns."""

    def __init__(self, rows, hits):
        self.collection = FakeCollection(rows)
        self.hits = hits

    def as_tuple(self):
        return (self.collection, None, [], [])


class VerifyDraftTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, draft, mapping):
        draft_path = os.path.join(self.tmp.name, "cited_draft.txt")
        with open(draft_path, "w", encoding="utf-8") as fh:
            fh.write(draft)
        with open(os.path.join(self.tmp.name, "cited_draft_citations.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(mapping, fh)
        return draft_path

    def _run(self, draft, mapping, rows, hits, judge_side_effect=None):
        draft_path = self._write(draft, mapping)
        res = _Resources(rows, hits)
        with patch("research_assistant.agents.agent8_verifier.hybrid_search",
                   return_value=hits), \
             patch("research_assistant.agents.agent8_verifier.judge",
                   side_effect=judge_side_effect or (lambda c, e, **k: _verdict())):
            return draft_path, verify_draft(draft_path, search_resources=res.as_tuple())


class TestVerifyDraft(VerifyDraftTestCase):
    def test_happy_path_judges_the_citation(self):
        _, report = self._run(
            "Graphene conducts well \\cite{cite_1}.",
            {"Smith 2020": "cite_1"},
            [("Smith 2020", "a.pdf")],
            [{"text": "Graphene is highly conductive.", "metadata": {"document": "a.pdf"}}],
        )
        self.assertEqual(len(report["results"]), 1)
        entry = report["results"][0]
        self.assertEqual(entry["outcome"], "judged")
        self.assertEqual(entry["judgement"], "Supports")
        self.assertEqual(entry["evidence"], "Graphene is highly conductive.")

    def test_missing_citations_file_raises(self):
        draft_path = os.path.join(self.tmp.name, "cited_draft.txt")
        with open(draft_path, "w", encoding="utf-8") as fh:
            fh.write("Text \\cite{cite_1}.")
        with self.assertRaises(FileNotFoundError):
            verify_draft(draft_path, search_resources=(FakeCollection([]), None, [], []))

    def test_orphaned_key_is_not_judged(self):
        _, report = self._run(
            "Invented \\cite{cite_9}.", {"Smith 2020": "cite_1"},
            [("Smith 2020", "a.pdf")], [],
        )
        self.assertEqual(report["results"][0]["outcome"], "orphaned")
        self.assertEqual(report["totals"]["orphaned"], 1)

    def test_source_with_no_documents_is_unresolved(self):
        _, report = self._run(
            "Claim \\cite{cite_1}.", {"Ghost 1999": "cite_1"}, [], [],
        )
        self.assertEqual(report["results"][0]["outcome"], "unresolved")

    def test_empty_retrieval_is_no_evidence(self):
        _, report = self._run(
            "Claim \\cite{cite_1}.", {"Smith 2020": "cite_1"},
            [("Smith 2020", "a.pdf")], [],
        )
        self.assertEqual(report["results"][0]["outcome"], "no_evidence")

    def test_parse_failure_is_recorded_and_does_not_abort(self):
        """One malformed reply must not cost a 120-citation run."""
        calls = {"n": 0}

        def flaky(claim, evidence, **kwargs):
            calls["n"] += 1
            if "first" in claim:
                raise JudgementParseError("bad", raw="garbage")
            return _verdict()

        _, report = self._run(
            "The first claim \\cite{cite_1}. The second claim \\cite{cite_1}.",
            {"Smith 2020": "cite_1"},
            [("Smith 2020", "a.pdf")],
            [{"text": "evidence", "metadata": {"document": "a.pdf"}}],
            judge_side_effect=flaky,
        )
        outcomes = [r["outcome"] for r in report["results"]]
        self.assertEqual(outcomes, ["parse_failed", "judged"])
        self.assertEqual(report["results"][0]["raw"], "garbage")

    def test_both_output_files_are_written(self):
        draft_path, _ = self._run(
            "Claim \\cite{cite_1}.", {"Smith 2020": "cite_1"},
            [("Smith 2020", "a.pdf")],
            [{"text": "evidence", "metadata": {"document": "a.pdf"}}],
        )
        base = draft_path.replace(".txt", "")
        self.assertTrue(os.path.exists(base + "_verification.json"))
        self.assertTrue(os.path.exists(base + "_verification.md"))

    def test_totals_count_every_category(self):
        _, report = self._run(
            "A \\cite{cite_1}. B \\cite{cite_9}.",
            {"Smith 2020": "cite_1"},
            [("Smith 2020", "a.pdf")],
            [{"text": "evidence", "metadata": {"document": "a.pdf"}}],
        )
        self.assertEqual(report["totals"]["judged"], 1)
        self.assertEqual(report["totals"]["orphaned"], 1)
        self.assertEqual(report["totals"]["total"], 2)

    def test_report_lists_worst_verdicts_first(self):
        verdicts = iter([_verdict("Supports"), _verdict("Contradicts")])
        draft_path, _ = self._run(
            "Fine \\cite{cite_1}. Wrong \\cite{cite_2}.",
            {"Smith 2020": "cite_1", "Jones 2019": "cite_2"},
            [("Smith 2020", "a.pdf"), ("Jones 2019", "b.pdf")],
            [{"text": "evidence", "metadata": {"document": "a.pdf"}}],
            judge_side_effect=lambda c, e, **k: next(verdicts),
        )
        markdown = open(draft_path.replace(".txt", "_verification.md"),
                        encoding="utf-8").read()
        # The flagged sentence must appear before the clean one. Asserting on
        # the word "Contradicts" instead would pass trivially — it is also a
        # row label in the summary table at the top.
        self.assertLess(markdown.index("Wrong"), markdown.index("Fine"))
