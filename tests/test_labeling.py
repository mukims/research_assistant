import unittest

from research_assistant.judgement import labeling as lb


def _cand(**over):
    c = {
        "id": "a_0001",
        "source": "abstract",
        "claim": "MoS2 networks conduct by hopping at low T.",
        "citation_evidence": "Transport is dominated by hopping at low temperature.",
        "document": "d.pdf",
        "citation_source": "Title",
        "section": "results",
        "hidden": {"model_judgement": "Supports"},
    }
    c.update(over)
    return c


class TestIds(unittest.TestCase):
    def test_next_id_counts_past_existing(self):
        self.assertEqual(lb.next_id(set(), "h"), "h_001")
        self.assertEqual(lb.next_id({"h_001", "h_002", "t_009"}, "h"), "h_003")


class TestApplyLabel(unittest.TestCase):
    def test_supports_with_negation_yields_two_cases(self):
        cases = lb.apply_label(
            _cand(),
            "s",
            set(),
            note="clear",
            negation="MoS2 networks do not conduct by hopping at low T.",
        )
        self.assertEqual(len(cases), 2)
        h, n = cases
        self.assertEqual((h["id"], h["source"], h["expected_judgement"]), ("h_001", "human", "Supports"))
        self.assertEqual((h["document"], h["citation_source"], h["notes"]), ("d.pdf", "Title", "clear"))
        self.assertEqual(h["model_judgement_at_harvest"], "Supports")
        self.assertEqual(
            (n["id"], n["source"], n["expected_judgement"], n["origin"]),
            ("n_001", "human_negation", "Contradicts", "h_001"),
        )
        self.assertEqual(n["citation_evidence"], h["citation_evidence"])
        self.assertTrue(n["claim"].startswith("MoS2 networks do not"))

    def test_other_verdicts_yield_one_case_and_ignore_negation(self):
        cases = lb.apply_label(_cand(), "d", {"h_001"}, negation="ignored")
        self.assertEqual(len(cases), 1)
        self.assertEqual((cases[0]["id"], cases[0]["expected_judgement"]), ("h_002", "Does not support"))

    def test_bad_key_raises(self):
        with self.assertRaises(KeyError):
            lb.apply_label(_cand(), "z", set())

    def test_cases_validate(self):
        from research_assistant.judgement.evalset import validate_case

        for c in lb.apply_label(_cand(), "s", set(), negation="No."):
            validate_case(c)


class TestUnlabelled(unittest.TestCase):
    def test_already_labelled_candidates_are_skipped_by_content(self):
        cand = _cand()
        done = lb.apply_label(cand, "s", set())
        self.assertEqual(
            lb.unlabelled([cand, _cand(id="a_0002", claim="Other.")], done),
            [_cand(id="a_0002", claim="Other.")],
        )


if __name__ == "__main__":
    unittest.main()
