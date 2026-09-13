"""The harness loop with a stub judge: held-out refusal, in-prompt cases
kept out of the headline, per-run results, verifier-mode draft shape."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import evaluate_judge as ej
from research_assistant.judgement.evalset import CASES_DIR, validate_case


def _case(id, expected="Supports", **over):
    d = {
        "id": id,
        "source": "human",
        "claim": f"claim {id}",
        "citation_evidence": f"evidence {id}",
        "expected_judgement": expected,
        "document": "d.pdf",
        "citation_source": "Title",
    }
    d.update(over)
    return validate_case(d)


class TestJudgeMode(unittest.TestCase):
    def test_runs_scores_and_refuses_in_prompt_cases(self):
        in_prompt = ej.load_cases([CASES_DIR / "cases.jsonl"])[0]
        cases = [_case("h_1"), _case("h_2", "Contradicts"), in_prompt]
        replies = {
            "claim h_1": "Supports",
            "claim h_2": "Does not support",
            in_prompt["claim"]: in_prompt["expected_judgement"],
        }
        with patch.object(ej, "judge", side_effect=lambda c, e: {"judgement": replies[c]}):
            out = ej.run(cases, mode="judge", runs=2)
        self.assertEqual(out["summary"]["n"], 4)  # 2 held-out cases × 2 runs
        self.assertAlmostEqual(out["summary"]["strict_acc"], 0.5)
        self.assertEqual([c["id"] for c in out["refused"]], [])
        self.assertEqual(out["in_prompt"]["n"], 2)
        self.assertAlmostEqual(out["in_prompt"]["strict_acc"], 1.0)
        self.assertEqual(len(out["results"]), 6)

    def test_a_held_out_case_whose_text_is_in_the_prompt_is_refused(self):
        in_prompt = ej.load_cases([CASES_DIR / "cases.jsonl"])[0]
        leak = _case("h_9", citation_evidence=in_prompt["citation_evidence"])
        with patch.object(ej, "judge", side_effect=lambda c, e: {"judgement": "Supports"}):
            out = ej.run([leak], mode="judge", runs=1)
        self.assertEqual([c["id"] for c in out["refused"]], ["h_9"])
        self.assertEqual(out["summary"]["n"], 0)

    def test_errors_are_recorded_not_raised(self):
        with patch.object(ej, "judge", side_effect=RuntimeError("boom")):
            out = ej.run([_case("h_1")], mode="judge", runs=1)
        self.assertEqual(out["summary"]["errors"], 1)
        self.assertIn("boom", out["results"][0]["error"])


class TestVerifierMode(unittest.TestCase):
    def test_writes_a_one_sentence_draft_and_uses_the_entry(self):
        seen = {}

        def fake_verify(draft_path, citations_path=None, **kw):
            seen["draft"] = open(draft_path).read()
            seen["mapping"] = json.load(open(draft_path.replace(".txt", "_citations.json")))
            return {
                "results": [
                    {
                        "outcome": "judged",
                        "judgement": "Supports",
                        "escalated": True,
                        "first_judgement": "Unclear / insufficient evidence",
                    }
                ]
            }

        with patch.object(ej, "verify_draft", fake_verify):
            out = ej.run([_case("h_1")], mode="verifier", runs=1)
        self.assertEqual(seen["draft"], "claim h_1 \\cite{cite_1}.")
        self.assertEqual(seen["mapping"], {"Title": "cite_1"})
        self.assertEqual(out["results"][0]["got"], "Supports")
        self.assertEqual(out["summary"]["signals"]["escalated"], 1.0)

    def test_cases_without_a_paper_are_skipped_in_verifier_mode(self):
        with patch.object(ej, "verify_draft", side_effect=AssertionError("must not be called")):
            out = ej.run(
                [_case("t_1", source="transform", transform="cross_pair", document=None, citation_source=None)],
                mode="verifier",
                runs=1,
            )
        self.assertEqual(out["summary"]["n"], 0)
        self.assertEqual(out["skipped_no_paper"], 1)


if __name__ == "__main__":
    unittest.main()
