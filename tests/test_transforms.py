"""Each transform's verdict follows from the rubric by construction, so the
generator must be deterministic and must refuse to produce a case whose
label it cannot guarantee."""

import unittest

from research_assistant.judgement import transforms as tf
from research_assistant.judgement.evalset import validate_case


def _seed(
    id="h_001",
    claim="Covalent MoS2 networks show a 10% increase in photoconductivity at low temperature.",
    evidence=(
        "Films were prepared by exfoliation. The covalent MoS2 networks show a 10% increase in photoconductivity "
        "at low temperature compared to pristine films. This is attributed to passivation."
    ),
    document="d.pdf",
    citation_source="Title",
):
    return validate_case({
        "id": id,
        "source": "human",
        "claim": claim,
        "citation_evidence": evidence,
        "expected_judgement": "Supports",
        "document": document,
        "citation_source": citation_source,
    })


class TestKeySentence(unittest.TestCase):
    def test_picks_the_sentence_with_most_shared_tokens(self):
        idx, sents = tf.key_sentence(_seed()["claim"], _seed()["citation_evidence"])
        self.assertEqual(idx, 1)
        self.assertTrue(sents[1].startswith("The covalent MoS2 networks show"))


class TestCrossPair(unittest.TestCase):
    def test_pairs_claims_with_evidence_from_other_papers(self):
        seeds = [
            _seed("h_001", document="a.pdf"),
            _seed(
                "h_002",
                claim="Graphene is a semimetal with linear dispersion near K.",
                evidence="Near the K point graphene's bands are linear; it is a semimetal.",
                document="b.pdf",
            ),
        ]
        out = tf.cross_pair(seeds, limit=10, existing_ids=set())
        self.assertEqual(len(out), 2)
        for c in out:
            self.assertEqual(
                (c["source"], c["transform"], c["expected_judgement"]),
                ("transform", "cross_pair", "Does not support"),
            )
            self.assertIsNone(c["document"])
        self.assertEqual(out[0]["claim"], seeds[0]["claim"])
        self.assertEqual(out[0]["citation_evidence"], seeds[1]["citation_evidence"])

    def test_same_paper_is_never_paired(self):
        seeds = [
            _seed("h_001", document="a.pdf"),
            _seed("h_002", claim="Other claim.", evidence="Other evidence.", document="a.pdf"),
        ]
        self.assertEqual(tf.cross_pair(seeds, 10, set()), [])

    def test_limit_and_ids(self):
        seeds = [_seed(f"h_{i:03d}", document=f"{i}.pdf") for i in range(1, 6)]
        out = tf.cross_pair(seeds, limit=3, existing_ids={"t_001"})
        self.assertEqual([c["id"] for c in out], ["t_002", "t_003", "t_004"])


class TestScopeSwap(unittest.TestCase):
    def test_swaps_a_token_present_in_both_claim_and_evidence(self):
        c = tf.scope_swap(_seed(), set())
        self.assertIn("WSe2", c["claim"])
        self.assertNotIn("MoS2", c["claim"])
        self.assertIn("MoS2", c["citation_evidence"])  # evidence untouched
        self.assertEqual(c["expected_judgement"], "Does not support")
        self.assertIn("Unclear / insufficient evidence", c["accept"])
        self.assertEqual((c["transform"], c["origin"], c["document"]), ("scope_swap", "h_001", "d.pdf"))

    def test_no_shared_scope_token_means_no_case(self):
        self.assertIsNone(
            tf.scope_swap(_seed(claim="The effect is large.", evidence="The effect is large indeed."), set())
        )


class TestNumberSwap(unittest.TestCase):
    def test_doubles_a_number_present_in_both(self):
        c = tf.number_swap(_seed(), set())
        self.assertIn("20%", c["claim"])
        self.assertNotIn("10%", c["claim"])
        self.assertEqual(c["expected_judgement"], "Contradicts")

    def test_large_numbers_are_halved(self):
        c = tf.number_swap(
            _seed(claim="Capacitance reaches 623 F/g.", evidence="A capacitance of 623 F/g was measured."),
            set(),
        )
        self.assertIn("311.5", c["claim"])

    def test_number_only_in_the_claim_is_not_swapped(self):
        self.assertIsNone(tf.number_swap(_seed(claim="A 10% increase.", evidence="An increase was seen."), set()))


class TestDeleteKeySentence(unittest.TestCase):
    def test_removes_the_key_sentence_and_accepts_both_negatives(self):
        c = tf.delete_key_sentence(_seed(), set())
        self.assertNotIn("10% increase", c["citation_evidence"])
        self.assertIn("Films were prepared", c["citation_evidence"])
        self.assertEqual(c["expected_judgement"], "Does not support")
        self.assertIn("Unclear / insufficient evidence", c["accept"])

    def test_single_sentence_evidence_yields_nothing(self):
        self.assertIsNone(
            tf.delete_key_sentence(_seed(evidence="Only the key sentence about MoS2 networks here."), set())
        )


class TestHedge(unittest.TestCase):
    def test_assertive_verb_in_the_key_sentence_is_weakened(self):
        c = tf.hedge_evidence(_seed(), set())
        self.assertIn("may suggest a 10% increase", c["citation_evidence"])
        self.assertEqual(c["expected_judgement"], "Partially supports")
        self.assertEqual(c["claim"], _seed()["claim"])

    def test_no_assertive_verb_means_no_case(self):
        self.assertIsNone(
            tf.hedge_evidence(
                _seed(evidence="A 10% increase in photoconductivity at low temperature for MoS2 networks."),
                set(),
            )
        )


class TestGenerate(unittest.TestCase):
    def test_only_supports_seeds_are_used_and_all_cases_validate(self):
        seeds = [
            _seed("h_001", document="a.pdf"),
            _seed(
                "h_002",
                claim="Graphene is a semimetal with a 5 meV gap.",
                evidence="Graphene shows a 5 meV gap; it is a semimetal.",
                document="b.pdf",
            ),
            validate_case({
                "id": "h_003",
                "source": "human",
                "claim": "x",
                "citation_evidence": "y",
                "expected_judgement": "Contradicts",
            }),
        ]
        out = tf.generate(seeds, cross_pairs=4)
        self.assertTrue(all(c["source"] == "transform" for c in out))
        self.assertTrue(all(c["origin"] in {"h_001", "h_002"} for c in out))
        self.assertEqual(len({c["id"] for c in out}), len(out))
        for c in out:
            validate_case(c)
        self.assertGreaterEqual(sum(c["transform"] == "cross_pair" for c in out), 2)


if __name__ == "__main__":
    unittest.main()
