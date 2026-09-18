"""The citer harness with a stubbed citer: the seed is excluded from its own
search, the need check is batched per seed, results carry what the metrics
need, and one sentence's failure is recorded rather than fatal."""

import unittest
from unittest.mock import patch

from research_assistant.agents.agent5_batch_citer import CiteResult
from scripts import evaluate_citer as ec


def _case(id, seed="arxiv_2108.10114v3", kind="cited", sentence="A claim with enough words to be eligible.", author=("a.pdf",)):
    return {"id": id, "seed": seed, "kind": kind, "sentence": sentence, "context": f"«{sentence}»",
            "section": "results", "section_heading": "", "paragraph_id": "p_0", "sentence_index": 0,
            "author_documents": list(author), "author_refs": [], "roles": ["evidential"]}


class TestSeedDocuments(unittest.TestCase):
    METAS = [{"document": "doi_10.1038_nature12952.pdf"}, {"document": "arxiv_2007.12504.pdf"},
             {"document": "doi_10.1103_physrevb.102.075409.pdf"}, {}]

    def test_names_that_differ_by_source_still_match(self):
        self.assertEqual(ec.seed_documents("nature12952", self.METAS), {"doi_10.1038_nature12952.pdf"})
        self.assertEqual(ec.seed_documents("arxiv_2007.12504v1", self.METAS), {"arxiv_2007.12504.pdf"})

    def test_a_seed_not_in_the_index_excludes_nothing(self):
        self.assertEqual(ec.seed_documents("arxiv_2108.10114v3", self.METAS), set())

    def test_a_short_document_name_is_not_swallowed_by_the_seed_stem(self):
        # "a" is a substring of "arxiv_2108…"; the match runs one way only.
        self.assertEqual(ec.seed_documents("arxiv_2108.10114v3", self.METAS + [{"document": "a.pdf"}]), set())


class TestRun(unittest.TestCase):
    RESOURCES = (None, None, [], [{"document": "arxiv_2108.10114v3.pdf"}, {"document": "a.pdf"}])

    def test_seed_is_excluded_need_is_batched_and_results_are_scored(self):
        cases = [_case("c_1"), _case("c_2", sentence="Short one."), _case("u_1", kind="uncited", author=())]
        seen = {}

        def fake_cite(sentence, resources, key_registry, **kw):
            seen[sentence] = kw
            key_registry.setdefault("Doe 2020", "cite_1")
            return CiteResult(original=sentence, cited_text=f"{sentence[:-1]} \\cite{{cite_1}}.", keys=["cite_1"],
                              candidates=[{"key": "cite_1", "citation": "Doe 2020", "document": "a.pdf"}], query=sentence)

        with patch.object(ec.citer, "_batch_needs_citation", return_value=[True, False]) as need, \
             patch.object(ec.citer, "cite_sentence", side_effect=fake_cite):
            out = ec.run(cases, self.RESOURCES, runs=1)

        # "Short one." has 2 words and never reaches the need check; the other two are batched once
        need.assert_called_once()
        self.assertEqual(need.call_args[0][0], [cases[0]["sentence"], cases[2]["sentence"]])
        self.assertEqual(out["excluded"], {"arxiv_2108.10114v3": ["arxiv_2108.10114v3.pdf"]})
        self.assertEqual(seen[cases[0]["sentence"]]["exclude_docs"], {"arxiv_2108.10114v3.pdf"})
        self.assertEqual(seen[cases[0]["sentence"]]["context"], cases[0]["context"])
        by_id = {r["case"]["id"]: r for r in out["results"]}
        self.assertEqual(by_id["c_1"]["needs_cite"], True)
        self.assertEqual(by_id["c_1"]["cite"]["keys"], ["cite_1"])
        self.assertEqual(by_id["c_2"]["needs_cite"], False)
        self.assertIsNone(by_id["c_2"]["cite"])
        self.assertEqual(by_id["u_1"]["needs_cite"], False)
        self.assertAlmostEqual(out["summary"]["end_to_end"], 1 / 2)
        self.assertEqual(out["runs"], 1)

    def test_a_failing_sentence_is_recorded_not_fatal(self):
        cases = [_case("c_1"), _case("c_2")]
        calls = iter([RuntimeError("model down"),
                      CiteResult(original="x", cited_text="x", keys=[], candidates=[], query="x",
                                 skip_reason="no relevant context found in database")])

        def fake_cite(sentence, resources, key_registry, **kw):
            v = next(calls)
            if isinstance(v, Exception):
                raise v
            return v

        with patch.object(ec.citer, "_batch_needs_citation", return_value=[True, True]), \
             patch.object(ec.citer, "cite_sentence", side_effect=fake_cite):
            out = ec.run(cases, self.RESOURCES, runs=1)
        by_id = {r["case"]["id"]: r for r in out["results"]}
        self.assertIn("model down", by_id["c_1"]["error"])
        self.assertIsNone(by_id["c_2"]["error"])
        self.assertEqual(out["summary"]["errors"], 1)

    def test_a_failed_need_check_marks_every_case_of_that_seed(self):
        with patch.object(ec.citer, "_batch_needs_citation", side_effect=ValueError("misaligned")), \
             patch.object(ec.citer, "cite_sentence") as cite:
            out = ec.run([_case("c_1"), _case("c_2")], self.RESOURCES, runs=1)
        cite.assert_not_called()
        self.assertTrue(all("misaligned" in r["error"] for r in out["results"]))

    def test_judge_gate_is_forwarded_and_named_in_the_result(self):
        seen = {}

        def fake_cite(sentence, resources, key_registry, **kw):
            seen.update(kw)
            return CiteResult(original=sentence, cited_text=sentence, query=sentence, skip_reason="no candidate passed the judge")

        with patch.object(ec.citer, "_batch_needs_citation", return_value=[True]), \
             patch.object(ec.citer, "cite_sentence", side_effect=fake_cite):
            out = ec.run([_case("c_1")], self.RESOURCES, runs=1, judge_gate=True)
        self.assertIs(seen["judge_gate"], True)
        self.assertIs(out["judge_gate"], True)

    def test_contextualize_builds_queries_per_seed_and_forwards_them(self):
        seen = {}

        def fake_cite(sentence, resources, key_registry, **kw):
            seen[sentence] = kw
            return CiteResult(original=sentence, cited_text=sentence, query=kw.get("query") or sentence,
                              skip_reason="no relevant context found in database")

        def fake_ctx(claims, model=None):
            for c in claims:
                c["search_query"] = "Q: " + c["claim"]

        cases = [_case("c_1"), _case("c_2", sentence="Another claim with enough words for the check.")]
        with patch.object(ec.citer, "_batch_needs_citation", return_value=[True, True]), \
             patch.object(ec.citer, "cite_sentence", side_effect=fake_cite), \
             patch("research_assistant.shared.seed_audit.contextualize_citation_queries", side_effect=fake_ctx) as ctx:
            out = ec.run(cases, self.RESOURCES, runs=1, contextualize=True)
        ctx.assert_called_once()
        self.assertEqual(seen[cases[0]["sentence"]]["query"], "Q: " + cases[0]["sentence"])
        self.assertIs(out["contextualize"], True)
        with patch.object(ec.citer, "_batch_needs_citation", return_value=[True, True]), \
             patch.object(ec.citer, "cite_sentence", side_effect=fake_cite):
            ec.run(cases, self.RESOURCES, runs=1, contextualize=False)
        self.assertIsNone(seen[cases[0]["sentence"]]["query"])
