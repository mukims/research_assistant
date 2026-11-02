import unittest

from research_assistant.judgement.judge import JudgementParseError
from research_assistant.transfer import models, premises

TEXTS = ["the sentence before", "the cited chunk", "a figure", "after"]
METAS = [{"document": "a.pdf", "type": t} for t in ("text_chunk", "text_chunk", "figure_image", "text_chunk")]
OK = {"judgement": "Supports", "span_verified": True, "supporting_span": "the cited chunk", "reason": "r"}


class TestPremiseChecker(unittest.TestCase):
    def test_judged_against_its_chunk_and_its_text_neighbour_not_the_figure(self):
        seen = []
        p = premises.PremiseChecker(lambda c, e: (seen.append(e), OK)[1], TEXTS, METAS).check(models.Premise("A.", "C1", "a.pdf", 1))
        self.assertIn("the cited chunk", seen[0]); self.assertIn("the sentence before", seen[0]); self.assertNotIn("a figure", seen[0])
        self.assertTrue(p.accepted)

    def test_the_same_claim_on_the_same_chunk_is_judged_once(self):
        ch = premises.PremiseChecker(lambda c, e: OK, TEXTS, METAS)
        ch.check_all([models.Premise("A  claim.", "C1", "a.pdf", 1), models.Premise("a claim.", "C1", "a.pdf", 1)])
        self.assertEqual(ch.calls, 1)

    def test_a_judge_failure_earns_nothing_and_the_next_premise_is_still_judged(self):
        def judge(c, e):
            if c == "bad": raise JudgementParseError("Reply was not valid JSON", raw="{")
            return OK
        bad, good = premises.PremiseChecker(judge, TEXTS, METAS).check_all(
            [models.Premise("bad", "C1", "a.pdf", 1), models.Premise("good", "C1", "a.pdf", 1)])
        self.assertEqual(bad.credit, 0.0); self.assertIn("JudgementParseError", bad.error); self.assertTrue(good.accepted)

    def test_premise_is_checked_against_its_own_paper(self):
        # Position 1 belongs to a.pdf; b.pdf's chunk with seq 1 sits at position 3 (per-document seq on LanceDB).
        texts = ["a zero", "a one", "a two", "b one"]
        metas = [
            {"document": "a.pdf", "type": "text_chunk", "seq": 0},
            {"document": "a.pdf", "type": "text_chunk", "seq": 1},
            {"document": "a.pdf", "type": "text_chunk", "seq": 2},
            {"document": "b.pdf", "type": "text_chunk", "seq": 1},
        ]
        seen = []
        premises.PremiseChecker(lambda c, e: (seen.append(e), OK)[1], texts, metas).check(
            models.Premise("B.", "C1", "b.pdf", 1))
        self.assertIn("b one", seen[0])
        self.assertNotIn("a one", seen[0])

    def test_unresolvable_premise_is_an_error_not_a_judgement(self):
        calls = []
        p = premises.PremiseChecker(lambda c, e: (calls.append(e), OK)[1], TEXTS, METAS).check(
            models.Premise("A.", "C1", "b.pdf", 1))
        self.assertEqual(calls, [])
        self.assertEqual((p.judgement, p.span_verified, p.supporting_span, p.reason), (None, None, None, None))
        self.assertEqual(p.error, "evidence chunk not found in b.pdf")
        self.assertEqual(p.credit, 0.0)

