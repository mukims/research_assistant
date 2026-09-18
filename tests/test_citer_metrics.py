"""Scoring for the citer evaluation. Every number is defined in the spec's
table (§3.2); these pin each one on hand-built results."""

import unittest

from research_assistant.eval import citer_metrics as cm


def _case(id, kind="cited", author=("a.pdf",), roles=("evidential",), section="results"):
    return {"id": id, "seed": "s", "kind": kind, "sentence": "S.", "context": "«S.»",
            "section": section, "section_heading": "", "paragraph_id": "p_0", "sentence_index": 0,
            "author_documents": list(author), "author_refs": [], "roles": list(roles)}


def _cite(keys, candidates, skip_reason=None):
    return {"original": "S.", "cited_text": "S \\cite{x}.", "keys": list(keys),
            "candidates": [{"key": k, "citation": k, "document": d} for k, d in candidates],
            "reasoning": "", "skip_reason": skip_reason, "query": "S.", "verdicts": [], "partial": False}


def _r(case, needs_cite, cite=None, run=1, error=None):
    return {"case": case, "needs_cite": needs_cite, "cite": cite, "run": run, "error": error}


class TestCitedDocuments(unittest.TestCase):
    def test_only_keys_that_made_the_sentence_count(self):
        cite = _cite(["cite_2"], [("cite_1", "a.pdf"), ("cite_2", "b.pdf"), ("cite_3", None)])
        self.assertEqual(cm.cited_documents(cite), {"b.pdf"})
        self.assertEqual(cm.cited_documents(None), set())


class TestScore(unittest.TestCase):
    def _results(self):
        return [
            # cited, needed, hit: cites the author's paper
            _r(_case("c1"), True, _cite(["cite_1"], [("cite_1", "a.pdf"), ("cite_2", "z.pdf")])),
            # cited, needed, miss: cites a paper the author did not
            _r(_case("c2", author=("a.pdf", "b.pdf"), roles=("evidential", "method"), section="methods"),
               True, _cite(["cite_2"], [("cite_1", "a.pdf"), ("cite_2", "z.pdf")])),
            # cited, needed, declined
            _r(_case("c3"), True, _cite([], [("cite_1", "a.pdf")], skip_reason="context retrieved but the model did not cite it")),
            # cited, need missed
            _r(_case("c4"), False),
            # uncited: one correctly left alone, one wrongly flagged
            _r(_case("u1", kind="uncited", author=(), roles=()), False),
            _r(_case("u2", kind="uncited", author=(), roles=()), True, _cite(["cite_1"], [("cite_1", "a.pdf")])),
        ]

    def test_need_numbers(self):
        s = cm.score(self._results())
        self.assertEqual((s["n_cited"], s["n_uncited"]), (4, 2))
        self.assertAlmostEqual(s["need_recall"], 3 / 4)
        self.assertAlmostEqual(s["need_specificity"], 1 / 2)

    def test_target_numbers_are_micro_over_sentences_that_cited(self):
        s = cm.score(self._results())
        # c1 cited {a} ∩ {a} = 1 of 1; c2 cited {z} ∩ {a,b} = 0 of 1 → precision 1/2
        self.assertAlmostEqual(s["target_precision"], 1 / 2)
        # author docs over all cited cases: c1 1, c2 2, c3 1, c4 1 = 5; found 1 → recall 1/5
        self.assertAlmostEqual(s["target_recall"], 1 / 5)
        self.assertAlmostEqual(s["sentence_hit"], 1 / 2)        # of the two that cited
        self.assertAlmostEqual(s["end_to_end"], 1 / 4)          # of all cited cases

    def test_declined_is_counted_with_its_reasons(self):
        s = cm.score(self._results())
        self.assertEqual(s["declined"]["n"], 1)
        self.assertEqual(s["declined"]["reasons"], {"context retrieved but the model did not cite it": 1})

    def test_slices_by_role_and_section(self):
        s = cm.score(self._results())
        self.assertEqual(s["by_role"]["evidential"]["n"], 4)
        self.assertAlmostEqual(s["by_role"]["evidential"]["end_to_end"], 1 / 4)
        self.assertEqual(s["by_role"]["method"]["n"], 1)
        self.assertEqual(s["by_section"]["methods"]["n"], 1)

    def test_errors_are_excluded_and_counted(self):
        rs = self._results() + [_r(_case("c9"), True, error="boom")]
        s = cm.score(rs)
        self.assertEqual(s["errors"], 1)
        self.assertEqual(s["n_cited"], 4)

    def test_stability_over_runs(self):
        hit = _cite(["cite_1"], [("cite_1", "a.pdf")])
        miss = _cite(["cite_1"], [("cite_1", "z.pdf")])
        rs = [_r(_case("c1"), True, hit, run=1), _r(_case("c1"), True, hit, run=2),
              _r(_case("c2"), True, hit, run=1), _r(_case("c2"), True, miss, run=2)]
        self.assertAlmostEqual(cm.score(rs)["stability"], 1 / 2)
        self.assertIsNone(cm.score(rs[:2:2] + rs[2:3])["stability"])   # one run: undefined

    def test_render_states_the_floor_and_compare_shows_deltas(self):
        s = cm.score(self._results())
        text = cm.render(s)
        self.assertIn(cm.FLOOR_NOTE.split(":")[0], text)
        self.assertIn("end_to_end", text)
        out = cm.compare({"summary": s}, {"summary": s})
        self.assertIn("end_to_end", out)
        self.assertIn("25%", out)
