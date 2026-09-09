"""Tests for the judgement module.

The offline tests here need no API key and no network: parsing and validation
are ordinary Python, and that is what makes them runnable in CI. The live
regression suite against cases.jsonl is opt-in via RUN_LLM_TESTS=1.
"""

import json
import os
import unittest
from pathlib import Path

import research_assistant.judgement.judge as judge_mod
from research_assistant.judgement.judge import (
    JudgementParseError,
    build_prompt,
    parse_judgement,
)

CASES_PATH = Path(judge_mod.__file__).parent / "cases" / "cases.jsonl"

VALID_JUDGEMENTS = {
    "Supports",
    "Partially supports",
    "Contradicts",
    "Does not support",
    "Unclear / insufficient evidence",
}


def load_cases():
    with CASES_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _good(**overrides):
    """A well-formed judgement payload, minus whatever a test overrides."""
    payload = {
        "slots": {
            "finding": {"assertion": "a", "verdict": "Supports"},
            "scope": {"assertion": "b", "verdict": "Supports"},
            "strength": {"assertion": "c", "verdict": "Not applicable"},
        },
        "judgement": "Supports",
        "evidence_sufficiency": "sufficient",
        "confidence": "High",
        "supporting_span": "some span",
        "reason": "because",
    }
    payload.update(overrides)
    return payload


class TestCases(unittest.TestCase):
    def test_cases_are_well_formed(self):
        cases = load_cases()
        self.assertEqual(len(cases), 6)
        for case in cases:
            self.assertTrue(case["claim"])
            self.assertTrue(case["citation_evidence"])
            self.assertIn(case["expected_judgement"], VALID_JUDGEMENTS)


class TestBuildPrompt(unittest.TestCase):
    def test_both_placeholders_are_filled(self):
        prompt = build_prompt("TEST CLAIM", "TEST EVIDENCE")
        self.assertIn("TEST CLAIM", prompt)
        self.assertIn("TEST EVIDENCE", prompt)
        self.assertNotIn("{{CLAIM}}", prompt)
        self.assertNotIn("{{CITATION_EVIDENCE}}", prompt)


class TestParseJudgement(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(parse_judgement(json.dumps(_good()))["judgement"], "Supports")

    def test_json_fence_is_stripped(self):
        raw = "```json\n" + json.dumps(_good()) + "\n```"
        self.assertEqual(parse_judgement(raw)["judgement"], "Supports")

    def test_bare_fence_is_stripped(self):
        raw = "```\n" + json.dumps(_good()) + "\n```"
        self.assertEqual(parse_judgement(raw)["judgement"], "Supports")

    def test_surrounding_whitespace_is_tolerated(self):
        self.assertEqual(
            parse_judgement("\n\n" + json.dumps(_good()) + "\n\n")["judgement"],
            "Supports",
        )

    def test_invalid_json_raises_with_raw_attached(self):
        with self.assertRaises(JudgementParseError) as ctx:
            parse_judgement("not json at all")
        self.assertEqual(ctx.exception.raw, "not json at all")

    def test_missing_field_raises(self):
        payload = _good()
        del payload["confidence"]
        with self.assertRaises(JudgementParseError):
            parse_judgement(json.dumps(payload))

    def test_lowercased_judgement_is_rejected(self):
        """A near-miss category would otherwise distort the report's counts."""
        with self.assertRaises(JudgementParseError):
            parse_judgement(json.dumps(_good(judgement="supports")))

    def test_unknown_confidence_is_rejected(self):
        with self.assertRaises(JudgementParseError):
            parse_judgement(json.dumps(_good(confidence="Very high")))

    def test_unknown_sufficiency_is_rejected(self):
        with self.assertRaises(JudgementParseError):
            parse_judgement(json.dumps(_good(evidence_sufficiency="lots")))

    def test_missing_slot_is_rejected(self):
        payload = _good()
        del payload["slots"]["strength"]
        with self.assertRaises(JudgementParseError):
            parse_judgement(json.dumps(payload))

    def test_extra_slot_is_rejected(self):
        payload = _good()
        payload["slots"]["vibes"] = {"assertion": "x", "verdict": "Supports"}
        with self.assertRaises(JudgementParseError):
            parse_judgement(json.dumps(payload))

    def test_json_array_is_rejected(self):
        with self.assertRaises(JudgementParseError):
            parse_judgement("[1, 2, 3]")


@unittest.skipUnless(
    os.getenv("RUN_LLM_TESTS") == "1",
    "Set RUN_LLM_TESTS=1 to run live LLM tests.",
)
class TestLiveJudgement(unittest.TestCase):
    def test_every_case_reaches_its_expected_judgement(self):
        from research_assistant.judgement.judge import judge

        failures = []
        for case in load_cases():
            result = judge(case["claim"], case["citation_evidence"])
            if result["judgement"] != case["expected_judgement"]:
                failures.append(
                    f"{case['id']} ({case['name']}): "
                    f"expected {case['expected_judgement']!r}, "
                    f"got {result['judgement']!r}"
                )
        self.assertEqual(failures, [], "\n".join(failures))
