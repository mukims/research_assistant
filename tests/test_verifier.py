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

    def _run(self, draft, mapping, rows, hits, judge_side_effect=None,
             search_side_effect=None):
        draft_path = self._write(draft, mapping)
        res = _Resources(rows, hits)
        # A side effect lets a test make retrieval raise; the default is the
        # same fixed hit list for every call.
        search_patch = ({"side_effect": search_side_effect} if search_side_effect
                        else {"return_value": hits})
        with patch("research_assistant.agents.agent8_verifier.hybrid_search",
                   **search_patch), \
             patch("research_assistant.agents.agent8_verifier.judge",
                   side_effect=judge_side_effect or (lambda c, e, **k: _verdict())):
            return draft_path, verify_draft(draft_path, search_resources=res.as_tuple())

    def _markdown(self, draft_path):
        with open(draft_path.replace(".txt", "_verification.md"), encoding="utf-8") as fh:
            return fh.read()

    def _json(self, draft_path):
        with open(draft_path.replace(".txt", "_verification.json"), encoding="utf-8") as fh:
            return json.load(fh)


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


class TestModelReplyCannotOverwriteTheRecord(VerifyDraftTestCase):
    """The record's provenance is the pipeline's, never the model's.

    `_validate` requires the six rubric fields but permits extra top-level
    keys, so a reply is free to echo `sentence_index`, `claim`, `evidence` or
    `outcome`. Merging the whole reply let those win, which destroyed exactly
    the provenance ARCHITECTURE §9.2 leans on to justify re-retrieval — and,
    when the echoed `sentence_index` was a string, raised TypeError in the
    Markdown writer after the JSON had already landed.
    """

    def _polluted(self, judgement="Contradicts"):
        verdict = _verdict(judgement)
        verdict.update({
            "sentence_index": "1",
            "sentence": "a sentence the model invented",
            "claim": "a claim the model invented",
            "evidence": "evidence the model invented",
            "cite_key": "cite_99",
            "citation_source": "Nobody 1999",
            "outcome": "looks_fine_to_me",
        })
        return verdict

    def _run_polluted(self):
        return self._run(
            "Graphene conducts well \\cite{cite_1}.",
            {"Smith 2020": "cite_1"},
            [("Smith 2020", "a.pdf")],
            [{"text": "Graphene is highly conductive.", "metadata": {"document": "a.pdf"}}],
            judge_side_effect=lambda c, e, **k: self._polluted(),
        )

    def test_pipeline_fields_survive_colliding_keys_in_the_reply(self):
        _, report = self._run_polluted()
        entry = report["results"][0]
        self.assertEqual(entry["sentence_index"], 0)
        self.assertEqual(entry["sentence"], "Graphene conducts well \\cite{cite_1}.")
        self.assertEqual(entry["claim"], "Graphene conducts well.")
        self.assertEqual(entry["evidence"], "Graphene is highly conductive.")
        self.assertEqual(entry["cite_key"], "cite_1")
        self.assertEqual(entry["citation_source"], "Smith 2020")
        self.assertEqual(entry["outcome"], "judged")

    def test_the_rubric_fields_still_land(self):
        """Guarding the record must not drop the verdict it is recording."""
        _, report = self._run_polluted()
        entry = report["results"][0]
        self.assertEqual(entry["judgement"], "Contradicts")
        self.assertEqual(entry["confidence"], "High")
        self.assertEqual(entry["evidence_sufficiency"], "sufficient")
        self.assertEqual(entry["supporting_span"], "span")
        self.assertEqual(entry["reason"], "because")
        self.assertEqual(set(entry["slots"]), {"finding", "scope", "strength"})

    def test_an_echoed_outcome_cannot_misbucket_the_totals(self):
        _, report = self._run_polluted()
        self.assertEqual(report["totals"]["judged"], 1)
        self.assertEqual(report["totals"]["Contradicts"], 1)
        self.assertNotIn("looks_fine_to_me", report["totals"])

    def test_both_artefacts_are_written_and_agree(self):
        """A string sentence_index used to raise in the Markdown writer after
        the JSON had already been written — one artefact, silently."""
        draft_path, report = self._run_polluted()
        self.assertTrue(os.path.exists(draft_path.replace(".txt", "_verification.md")))
        on_disk = self._json(draft_path)["results"][0]
        self.assertEqual(on_disk["sentence_index"], 0)
        self.assertEqual(on_disk["claim"], "Graphene conducts well.")
        self.assertIn("Sentence 1", self._markdown(draft_path))


class TestRetrievalFailure(VerifyDraftTestCase):
    """hybrid_search embeds the query — a network call under the hosted
    embedding backends — so it can raise, and spec §7 says nothing but a
    missing mapping aborts the run."""

    def _flaky_search(self, claim, *args, **kwargs):
        if "first" in claim:
            raise RuntimeError("chroma is unreachable")
        return [{"text": "evidence", "metadata": {"document": "a.pdf"}}]

    def _run_flaky(self):
        return self._run(
            "The first claim \\cite{cite_1}. The second claim \\cite{cite_1}.",
            {"Smith 2020": "cite_1"},
            [("Smith 2020", "a.pdf")],
            [{"text": "evidence", "metadata": {"document": "a.pdf"}}],
            search_side_effect=self._flaky_search,
        )

    def test_a_raising_search_is_recorded_and_does_not_abort(self):
        _, report = self._run_flaky()
        self.assertEqual([r["outcome"] for r in report["results"]],
                         ["retrieval_failed", "judged"])

    def test_the_exception_type_is_recorded(self):
        """'RuntimeError: chroma is unreachable' and 'the model replied with
        garbage' are different bugs; the report has to tell them apart."""
        _, report = self._run_flaky()
        entry = report["results"][0]
        self.assertEqual(entry["error_type"], "RuntimeError")
        self.assertIn("chroma is unreachable", entry["raw"])

    def test_totals_seed_and_count_the_new_outcome(self):
        _, report = self._run_flaky()
        self.assertEqual(report["totals"]["retrieval_failed"], 1)

    def test_totals_seed_retrieval_failed_even_when_none_occurred(self):
        _, report = self._run(
            "Claim \\cite{cite_1}.", {"Smith 2020": "cite_1"},
            [("Smith 2020", "a.pdf")],
            [{"text": "evidence", "metadata": {"document": "a.pdf"}}],
        )
        self.assertEqual(report["totals"]["retrieval_failed"], 0)

    def test_both_artefacts_still_land(self):
        draft_path, _ = self._run_flaky()
        self.assertTrue(os.path.exists(draft_path.replace(".txt", "_verification.json")))
        self.assertTrue(os.path.exists(draft_path.replace(".txt", "_verification.md")))

    def test_the_report_shows_it_in_the_table_and_the_not_judged_section(self):
        draft_path, _ = self._run_flaky()
        markdown = self._markdown(draft_path)
        self.assertIn("| Retrieval failed | 1 |", markdown)
        self.assertIn("retrieval_failed", markdown.split("## Not judged")[1])
        self.assertIn("`RuntimeError`", markdown)


class TestJudgingCallFailure(VerifyDraftTestCase):
    """A dead endpoint is not an unusable model reply."""

    def setUp(self):
        super().setUp()
        # @retry sleeps a second between attempts; the failure mode under test
        # is which bucket the exception lands in, not the backoff.
        patcher = patch("research_assistant.shared.retry.time.sleep")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run_down(self):
        def endpoint_is_down(claim, evidence, **kwargs):
            raise ConnectionError("connection refused")

        return self._run(
            "Claim \\cite{cite_1}.", {"Smith 2020": "cite_1"},
            [("Smith 2020", "a.pdf")],
            [{"text": "evidence", "metadata": {"document": "a.pdf"}}],
            judge_side_effect=endpoint_is_down,
        )

    def test_a_transport_error_is_call_failed_not_parse_failed(self):
        _, report = self._run_down()
        self.assertEqual(report["results"][0]["outcome"], "call_failed")
        self.assertEqual(report["totals"]["call_failed"], 1)
        self.assertEqual(report["totals"]["parse_failed"], 0)

    def test_the_exception_type_is_recorded(self):
        _, report = self._run_down()
        entry = report["results"][0]
        self.assertEqual(entry["error_type"], "ConnectionError")
        self.assertIn("connection refused", entry["raw"])

    def test_a_parse_error_is_still_parse_failed(self):
        """The split must not reclassify the case it was split away from."""
        def unusable(claim, evidence, **kwargs):
            raise JudgementParseError("bad", raw="garbage")

        _, report = self._run(
            "Claim \\cite{cite_1}.", {"Smith 2020": "cite_1"},
            [("Smith 2020", "a.pdf")],
            [{"text": "evidence", "metadata": {"document": "a.pdf"}}],
            judge_side_effect=unusable,
        )
        self.assertEqual(report["results"][0]["outcome"], "parse_failed")
        self.assertEqual(report["totals"]["call_failed"], 0)

    def test_the_report_shows_it_in_the_table_and_the_not_judged_section(self):
        draft_path, _ = self._run_down()
        markdown = self._markdown(draft_path)
        self.assertIn("| Model call failed | 1 |", markdown)
        self.assertIn("call_failed", markdown.split("## Not judged")[1])
        self.assertIn("`ConnectionError`", markdown)


class TestReportedModel(VerifyDraftTestCase):
    """The record must name the model that produced the verdicts.

    judge() passes JUDGEMENT_MODEL to chat(), which falls back to LLM_MODEL —
    so naming LLM_MODEL unconditionally misattributes every verdict whenever
    CITATION_JUDGEMENT_MODEL is set, which is precisely when someone is
    auditing on a model other than the drafting one.
    """

    def _run_simple(self):
        return self._run(
            "Claim \\cite{cite_1}.", {"Smith 2020": "cite_1"},
            [("Smith 2020", "a.pdf")],
            [{"text": "evidence", "metadata": {"document": "a.pdf"}}],
        )

    def test_judgement_model_wins_when_set(self):
        with patch("research_assistant.agents.agent8_verifier.JUDGEMENT_MODEL",
                   "auditor-model"):
            draft_path, report = self._run_simple()
        self.assertEqual(report["model"], "auditor-model")
        self.assertIn("auditor-model", self._markdown(draft_path))

    def test_falls_back_to_the_pipeline_model_when_unset(self):
        from research_assistant.config import LLM_MODEL

        with patch("research_assistant.agents.agent8_verifier.JUDGEMENT_MODEL", None):
            _, report = self._run_simple()
        self.assertEqual(report["model"], LLM_MODEL)


class TestEvidenceIsQuotedInTheReport(VerifyDraftTestCase):
    """ARCHITECTURE §9.2 promises the report names the judged chunk."""

    def _run_flagged(self, evidence, supporting_span=None):
        verdict = _verdict("Does not support")
        verdict["supporting_span"] = supporting_span
        return self._run(
            "Claim \\cite{cite_1}.", {"Smith 2020": "cite_1"},
            [("Smith 2020", "a.pdf")],
            [{"text": evidence, "metadata": {"document": "a.pdf"}}],
            judge_side_effect=lambda c, e, **k: verdict,
        )

    def test_a_flagged_entry_quotes_the_chunk_that_was_judged(self):
        """'Does not support' is the most actionable verdict and the one where
        supporting_span is legitimately null — without the chunk the reader
        gets three assertions and no evidence text at all."""
        draft_path, _ = self._run_flagged("The chunk\nthat was judged.")
        markdown = self._markdown(draft_path)
        self.assertIn("**Evidence judged:**", markdown)
        # Flattened: a bare '> ' prefix would leave line two outside the quote.
        self.assertIn("> The chunk that was judged.", markdown)

    def test_long_evidence_is_truncated(self):
        draft_path, _ = self._run_flagged("x" * 900)
        markdown = self._markdown(draft_path)
        self.assertIn("x" * 400 + "…", markdown)
        self.assertNotIn("x" * 401, markdown)


class TestNotJudgedIsFullyAccountedFor(VerifyDraftTestCase):
    """The UI's 'Not judged' tile is `total - judged`; that number is only
    honest if every non-judged entry lands in a named bucket."""

    def test_every_outcome_has_a_seeded_bucket(self):
        _, report = self._run(
            "A \\cite{cite_1}. B \\cite{cite_9}.",
            {"Smith 2020": "cite_1"},
            [("Smith 2020", "a.pdf")],
            [{"text": "evidence", "metadata": {"document": "a.pdf"}}],
        )
        totals = report["totals"]
        buckets = ("orphaned", "unresolved", "no_evidence",
                   "retrieval_failed", "parse_failed", "call_failed")
        for name in buckets:
            self.assertIn(name, totals, f"{name} must be seeded even at zero")
        self.assertEqual(totals["total"] - totals["judged"],
                         sum(totals[name] for name in buckets))
