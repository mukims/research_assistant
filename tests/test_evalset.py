"""The evaluation set's contract: every case is well-formed, ids are unique,
and nothing the prompt already contains is ever scored as held-out."""

import json
import os
import tempfile
import unittest

from research_assistant.judgement import evalset as es
from research_assistant.judgement.judge import PROMPT_TEMPLATE


def _case(**over):
    base = {
        "id": "h_001",
        "source": "human",
        "claim": "MoS2 networks conduct by hopping.",
        "citation_evidence": "Hopping dominates transport in the networks.",
        "expected_judgement": "Supports",
    }
    base.update(over)
    return base


class TestValidateCase(unittest.TestCase):
    def test_defaults_are_filled(self):
        c = es.validate_case(_case())
        self.assertEqual(c["accept"], ["Supports"])
        self.assertIsNone(c["transform"])
        self.assertIsNone(c["origin"])
        self.assertIsNone(c["document"])
        self.assertEqual(c["notes"], "")

    def test_prompt_examples_without_source_default_to_prompt_example(self):
        c = es.validate_case({
            "id": "case_A",
            "name": "x",
            "claim": "c",
            "citation_evidence": "e",
            "expected_judgement": "Supports",
        })
        self.assertEqual(c["source"], "prompt_example")

    def test_bad_judgement_source_or_accept_raise(self):
        with self.assertRaises(ValueError):
            es.validate_case(_case(expected_judgement="supports"))
        with self.assertRaises(ValueError):
            es.validate_case(_case(source="llm"))
        with self.assertRaises(ValueError):
            es.validate_case(_case(accept=["Supports", "maybe"]))
        with self.assertRaises(ValueError):
            es.validate_case(_case(claim=""))

    def test_accept_always_includes_expected(self):
        c = es.validate_case(
            _case(expected_judgement="Does not support", accept=["Unclear / insufficient evidence"])
        )
        self.assertEqual(set(c["accept"]), {"Does not support", "Unclear / insufficient evidence"})

    def test_context_fields_default_to_none_and_tags_to_a_list(self):
        c = es.validate_case(_case())
        self.assertIsNone(c["context"])
        self.assertIsNone(c["section_heading"])
        self.assertIsNone(c["artifacts"])
        self.assertEqual(c["tags"], [])
        c = es.validate_case(_case(tags=["figure_ref"], context="«x»", section_heading="2. Results"))
        self.assertEqual((c["tags"], c["context"], c["section_heading"]), (["figure_ref"], "«x»", "2. Results"))


class TestLoadCases(unittest.TestCase):
    def test_loads_several_files_skips_missing_and_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "a.jsonl")
            b = os.path.join(d, "b.jsonl")
            with open(a, "w") as f:
                f.write(json.dumps(_case(id="h_001")) + "\n\n" + json.dumps(_case(id="h_002")) + "\n")
            with open(b, "w") as f:
                f.write(json.dumps(_case(id="t_001", source="transform", transform="cross_pair", origin="h_001")) + "\n")
            cases = es.load_cases([a, b, os.path.join(d, "missing.jsonl")])
            self.assertEqual([c["id"] for c in cases], ["h_001", "h_002", "t_001"])
            with open(b, "a") as f:
                f.write(json.dumps(_case(id="h_001")) + "\n")
            with self.assertRaises(ValueError):
                es.load_cases([a, b])

    def test_the_shipped_prompt_examples_load(self):
        cases = es.load_cases([es.CASES_DIR / "cases.jsonl"])
        self.assertEqual(len(cases), 7)
        self.assertTrue(all(c["source"] == "prompt_example" for c in cases))


class TestHeldOut(unittest.TestCase):
    def test_prompt_examples_are_not_held_out(self):
        cases = es.load_cases([es.CASES_DIR / "cases.jsonl"])
        self.assertFalse(any(es.held_out(c, PROMPT_TEMPLATE) for c in cases))

    def test_a_fresh_case_is_held_out(self):
        self.assertTrue(es.held_out(es.validate_case(_case()), PROMPT_TEMPLATE))

    def test_evidence_in_the_prompt_is_enough_to_refuse(self):
        prompt_evidence = json.loads(open(es.CASES_DIR / "cases.jsonl").readline())["citation_evidence"]
        c = es.validate_case(_case(claim="A brand new claim about ferrites.", citation_evidence=prompt_evidence))
        self.assertFalse(es.held_out(c, PROMPT_TEMPLATE))

    def test_split(self):
        fresh = es.validate_case(_case(id="h_9"))
        inprompt = es.load_cases([es.CASES_DIR / "cases.jsonl"])[0]
        scored, refused = es.split_held_out([fresh, inprompt], PROMPT_TEMPLATE)
        self.assertEqual([c["id"] for c in scored], ["h_9"])
        self.assertEqual([c["id"] for c in refused], ["case_A"])


if __name__ == "__main__":
    unittest.main()
