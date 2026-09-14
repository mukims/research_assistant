import unittest

from research_assistant.judgement.policy import ReliabilityRating, evaluate_reliability
from research_assistant.judgement.source_assessor import SourceGrade


class TestReliabilityPolicy(unittest.TestCase):
    def test_supports_with_rigorous_source_is_high(self):
        res = evaluate_reliability(
            relation="Supports",
            source_grade=SourceGrade.RIGOROUS_PRIMARY.value,
            confidence="High",
            span_verified=True,
        )
        self.assertEqual(res["rating"], ReliabilityRating.HIGH.value)
        self.assertIn("🟢", res["badge"])

    def test_supports_with_standard_source_is_high(self):
        res = evaluate_reliability(
            relation="Supports",
            source_grade=SourceGrade.STANDARD_PRIMARY.value,
            confidence="High",
            span_verified=True,
        )
        self.assertEqual(res["rating"], ReliabilityRating.HIGH.value)

    def test_supports_with_preprint_is_moderate(self):
        res = evaluate_reliability(
            relation="Supports",
            source_grade=SourceGrade.PREPRINT_UNREVIEWED.value,
            confidence="High",
            span_verified=True,
        )
        self.assertEqual(res["rating"], ReliabilityRating.MODERATE.value)
        self.assertIn("🟡", res["badge"])
        self.assertIn("preprint", res["explanation"].lower())

    def test_supports_with_review_is_moderate(self):
        res = evaluate_reliability(
            relation="Supports",
            source_grade=SourceGrade.SECONDARY_REVIEW.value,
            confidence="High",
            span_verified=True,
        )
        self.assertEqual(res["rating"], ReliabilityRating.MODERATE.value)

    def test_partially_supports_is_moderate(self):
        res = evaluate_reliability(
            relation="Partially supports",
            source_grade=SourceGrade.STANDARD_PRIMARY.value,
            confidence="High",
            span_verified=True,
        )
        self.assertEqual(res["rating"], ReliabilityRating.MODERATE.value)

    def test_supports_with_retracted_source_is_low(self):
        res = evaluate_reliability(
            relation="Supports",
            source_grade=SourceGrade.RETRACTED_OR_FLAWED.value,
            confidence="High",
            span_verified=True,
        )
        self.assertEqual(res["rating"], ReliabilityRating.LOW.value)
        self.assertIn("retracted", res["explanation"].lower())

    def test_supports_with_unverified_span_is_low(self):
        res = evaluate_reliability(
            relation="Supports",
            source_grade=SourceGrade.RIGOROUS_PRIMARY.value,
            confidence="High",
            span_verified=False,
        )
        self.assertEqual(res["rating"], ReliabilityRating.LOW.value)
        self.assertIn("span", res["explanation"].lower())

    def test_contradicts_is_contradicted(self):
        res = evaluate_reliability(
            relation="Contradicts",
            source_grade=SourceGrade.STANDARD_PRIMARY.value,
            confidence="High",
            span_verified=True,
        )
        self.assertEqual(res["rating"], ReliabilityRating.CONTRADICTED.value)
        self.assertIn("🔴", res["badge"])

    def test_does_not_support_is_unsupported(self):
        res = evaluate_reliability(
            relation="Does not support",
            source_grade=SourceGrade.STANDARD_PRIMARY.value,
        )
        self.assertEqual(res["rating"], ReliabilityRating.UNSUPPORTED.value)
        self.assertIn("does not report", res["explanation"])

    def test_judged_unclear_is_unresolved_with_judge_explanation(self):
        res = evaluate_reliability(relation="Unclear / insufficient evidence",
                                   source_grade=SourceGrade.STANDARD_PRIMARY.value)
        self.assertEqual(res["rating"], ReliabilityRating.UNRESOLVED.value)
        self.assertIn("could not decide", res["explanation"])

    def test_not_assessed_outcomes_are_unresolved_with_their_own_reason(self):
        for outcome, phrase in (
            ("cap_exceeded", "budget"),
            ("not_attempted", "backend"),
            ("retrieval_failed", "retrieval"),
            ("call_failed", "backend"),
            ("no_evidence", "no passages"),
            ("cluster_skipped", "cites"),
            ("not_a_claim", "not a verifiable claim"),
            ("malformed_claim", "sentence"),
            ("unresolved_ref", "bibliography"),
            ("not_downloaded", "not in the corpus"),
            ("deferred_paywalled", "deferred"),
        ):
            res = evaluate_reliability(relation=None, source_grade=None, outcome=outcome)
            self.assertEqual(res["rating"], ReliabilityRating.UNRESOLVED.value, outcome)
            self.assertTrue(res["explanation"].startswith("Not assessed"), outcome)
            self.assertIn(phrase, res["explanation"], outcome)

    def test_missing_confidence_does_not_downgrade(self):
        res = evaluate_reliability(relation="Supports", source_grade=SourceGrade.STANDARD_PRIMARY.value,
                                   confidence=None, span_verified=True)
        self.assertEqual(res["rating"], ReliabilityRating.HIGH.value)

    def test_deferred_is_unresolved(self):
        res = evaluate_reliability(
            relation=None,
            source_grade=SourceGrade.UNKNOWN.value,
            outcome="deferred_paywalled",
        )
        self.assertEqual(res["rating"], ReliabilityRating.UNRESOLVED.value)


if __name__ == "__main__":
    unittest.main()
