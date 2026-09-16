"""Candidates for the operator to label: abstract sentences with the body
chunk that should support them, and the claim/evidence pairs Agent 8 has
already judged. Pure helpers only; the script does the Chroma I/O."""

import unittest

from research_assistant.judgement import harvest as hv


class TestResplit(unittest.TestCase):
    def test_glued_sentences_get_a_space(self):
        self.assertEqual(
            hv.resplit_glued("training data.This paper explores models."),
            "training data. This paper explores models.",
        )

    def test_decimals_and_abbreviations_are_untouched(self):
        self.assertEqual(hv.resplit_glued("a gap of 0.5 eV. Fig.2 shows"), "a gap of 0.5 eV. Fig.2 shows")


class TestAbstractSentences(unittest.TestCase):
    ABSTRACT = (
        "Current neural networks use quantum chemistry only as training data."
        "This paper explores models that use quantum chemistry as an integral part of the prediction process, "
        "implemented as a DFTB layer for deep learning. Short. "
        + "x" * 400
        + ". We thank the funding agency for support."
    )

    def test_length_filter_and_resplit(self):
        out = hv.abstract_sentences(self.ABSTRACT)
        self.assertEqual(out, [
            "This paper explores models that use quantum chemistry as an integral part of the prediction process, "
            "implemented as a DFTB layer for deep learning.",
        ])

    def test_acknowledgement_like_sentences_are_dropped(self):
        self.assertEqual(
            hv.abstract_sentences("We thank the agency for support of this long and detailed programme of work today."),
            [],
        )


class TestCandidates(unittest.TestCase):
    def test_evidence_wraps_neighbours(self):
        hit = {
            "text": "MID",
            "context_before": "PRE",
            "context_after": "POST",
            "metadata": {"section": "results"},
        }
        self.assertEqual(hv.evidence_from_hit(hit), "PRE\nMID\nPOST")
        self.assertEqual(hv.evidence_from_hit({"text": "MID", "metadata": {}}), "MID")

    def test_make_candidate_shape(self):
        hit = {"text": "MID", "metadata": {"section": "results"}}
        c = hv.make_candidate("a_0001", "Claim.", hit, "d.pdf", "Title", "abstract")
        self.assertEqual(
            c,
            {
                "id": "a_0001",
                "source": "abstract",
                "claim": "Claim.",
                "citation_evidence": "MID",
                "document": "d.pdf",
                "citation_source": "Title",
                "section": "results",
                "hidden": {"model_judgement": None},
            },
        )

    def test_candidates_from_verification_keep_only_judged_and_hide_the_verdict(self):
        report = {
            "results": [
                {
                    "outcome": "judged",
                    "claim": "C1",
                    "evidence": "E1",
                    "citation_source": "S1",
                    "judgement": "Supports",
                },
                {"outcome": "orphaned", "claim": "C2"},
                {
                    "outcome": "judged",
                    "claim": "C3",
                    "evidence": "E3",
                    "citation_source": "S3",
                    "judgement": "Does not support",
                },
            ]
        }
        out = hv.candidates_from_verification(report, prefix="v1")
        self.assertEqual([c["id"] for c in out], ["v1_0001", "v1_0002"])
        self.assertEqual(out[1]["hidden"], {"model_judgement": "Does not support"})
        self.assertEqual(out[0]["source"], "verification")
        self.assertIsNone(out[0]["document"])

    def test_candidates_from_verification_carry_the_context_the_judge_saw(self):
        report = {"results": [{
            "outcome": "judged", "claim": "C1", "evidence": "E1", "citation_source": "S1", "judgement": "Supports",
            "context": "B. «C1» A.", "section_heading": "2. Results",
            "artifacts": [{"id": "fig_0", "kind": "figure", "label": "Fig. 1", "caption": "cap"}],
        }, {"outcome": "judged", "claim": "C2", "evidence": "E2", "citation_source": "S2", "judgement": "Supports"}]}
        out = hv.candidates_from_verification(report, prefix="v1")
        self.assertEqual((out[0]["context"], out[0]["section_heading"], out[0]["artifacts"][0]["label"]),
                         ("B. «C1» A.", "2. Results", "Fig. 1"))
        self.assertEqual((out[1]["context"], out[1]["section_heading"], out[1]["artifacts"]), (None, None, None))


if __name__ == "__main__":
    unittest.main()
