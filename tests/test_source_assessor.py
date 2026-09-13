import unittest

from research_assistant.judgement.source_assessor import SourceGrade, assess_source


class TestSourceAssessor(unittest.TestCase):
    def test_retracted_paper_is_flagged(self):
        res = assess_source({
            "title": "Superconductivity at Room Temperature",
            "venue": "Nature",
            "is_retracted": True,
        })
        self.assertEqual(res["grade"], SourceGrade.RETRACTED_OR_FLAWED.value)
        self.assertTrue(res["is_retracted"])
        self.assertIn("Retracted", res["grade_label"])

    def test_retraction_in_title_or_venue(self):
        res = assess_source({
            "title": "RETRACTED: Low-temperature anomalies in hydrides",
            "venue": "Physical Review B",
        })
        self.assertEqual(res["grade"], SourceGrade.RETRACTED_OR_FLAWED.value)

    def test_arxiv_preprint_is_flagged(self):
        res = assess_source({
            "title": "Topological quantum computation",
            "arxiv_id": "2401.12345",
            "venue": "arXiv preprint arXiv:2401.12345",
        })
        self.assertEqual(res["grade"], SourceGrade.PREPRINT_UNREVIEWED.value)
        self.assertTrue(res["is_preprint"])
        self.assertIn("Preprint", res["grade_label"])

    def test_biorxiv_preprint_is_flagged(self):
        res = assess_source({
            "title": "Structural basis of ribozyme cleavage",
            "doi": "10.1101/2023.01.01.123456",
            "venue": "bioRxiv",
        })
        self.assertEqual(res["grade"], SourceGrade.PREPRINT_UNREVIEWED.value)
        self.assertTrue(res["is_preprint"])

    def test_review_paper_is_flagged(self):
        res = assess_source({
            "title": "A comprehensive review of 2D materials in catalysis",
            "venue": "Chemical Reviews",
            "doi": "10.1021/acs.chemrev.12345",
        })
        self.assertEqual(res["grade"], SourceGrade.SECONDARY_REVIEW.value)
        self.assertIn("Review", res["grade_label"])

    def test_rigorous_primary_peer_reviewed(self):
        res = assess_source({
            "title": "High-mobility transport in monolayer WSe2",
            "venue": "Physical Review Letters",
            "doi": "10.1103/PhysRevLett.120.123456",
            "citation_count": 85,
        })
        self.assertEqual(res["grade"], SourceGrade.RIGOROUS_PRIMARY.value)
        self.assertIn("Rigorous", res["grade_label"])

    def test_standard_primary_peer_reviewed(self):
        res = assess_source({
            "title": "Synthesis of MnFe2O4 nanoparticles via hydrothermal route",
            "venue": "Journal of Alloys and Compounds",
            "doi": "10.1016/j.jallcom.2021.123456",
            "citation_count": 5,
        })
        self.assertEqual(res["grade"], SourceGrade.STANDARD_PRIMARY.value)

    def test_fallback_on_empty_metadata(self):
        res = assess_source({})
        self.assertEqual(res["grade"], SourceGrade.UNKNOWN.value)


if __name__ == "__main__":
    unittest.main()
