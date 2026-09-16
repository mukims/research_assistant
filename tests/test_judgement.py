"""Tests for the judgement module.

The offline tests here need no API key and no network: parsing and validation
are ordinary Python, and that is what makes them runnable in CI. The live
regression suite against cases.jsonl is opt-in via RUN_LLM_TESTS=1.
"""

import copy
import json
import os
import types
import unittest
from pathlib import Path
from unittest.mock import patch

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
        self.assertEqual(len(cases), 7)
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

    def test_context_block_is_absent_by_default(self):
        prompt = build_prompt("claim text", "evidence text")
        self.assertNotIn("{{CONTEXT_BLOCK}}", prompt)
        self.assertNotIn("**Context", prompt)
        self.assertLess(prompt.index("**Claim:**"), prompt.index("claim text"))

    def test_context_block_precedes_the_claim(self):
        prompt = build_prompt("claim text", "evidence text", context="Before. «claim text» After.")
        self.assertIn("**Context", prompt)
        self.assertIn("«claim text»", prompt)
        self.assertLess(prompt.index("**Context"), prompt.index("**Claim:**"))
        self.assertIn("judge only that sentence", prompt)


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

    def test_slot_value_as_string_is_rejected(self):
        payload = _good()
        payload["slots"]["finding"] = "oops"
        with self.assertRaises(JudgementParseError):
            parse_judgement(json.dumps(payload))

    def test_slot_missing_assertion_is_rejected(self):
        payload = _good()
        del payload["slots"]["scope"]["assertion"]
        with self.assertRaises(JudgementParseError):
            parse_judgement(json.dumps(payload))

    def test_slot_missing_verdict_is_rejected(self):
        payload = _good()
        del payload["slots"]["strength"]["verdict"]
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


class TestBackendSelection(unittest.TestCase):
    """A key in the environment is a default, not an override: an operator who
    sets LLM_BACKEND=ollama to run the audit on a local model must get the
    local model, whether or not a Gemini key happens to be present."""

    def _call(self, **config):
        captured = {}

        def fake_chat(messages, model=None, temperature=None, options=None, backend=None):
            captured.update(model=model, backend=backend)
            return types.SimpleNamespace(content=json.dumps(_reply()))

        patches = [patch.object(judge_mod, k, v) for k, v in config.items()]
        for p_ in patches:
            p_.start(); self.addCleanup(p_.stop)
        with patch.object(judge_mod, "chat", side_effect=fake_chat):
            judge_mod.judge("claim", "evidence")
        return captured

    def test_a_gemini_key_alone_still_routes_to_gemini(self):
        got = self._call(GEMINI_API_KEY="k", LLM_BACKEND="openai", JUDGEMENT_MODEL="gemini-3.5-flash-lite")
        self.assertEqual((got["backend"], got["model"]), ("openai", "gemini-3.5-flash-lite"))

    def test_an_explicit_ollama_backend_wins_over_the_key(self):
        got = self._call(GEMINI_API_KEY="k", LLM_BACKEND="ollama", JUDGEMENT_MODEL="gemma4:e2b")
        self.assertEqual((got["backend"], got["model"]), ("ollama", "gemma4:e2b"))

    def test_no_key_leaves_the_choice_to_chat(self):
        got = self._call(GEMINI_API_KEY=None, LLM_BACKEND="ollama", JUDGEMENT_MODEL=None)
        self.assertEqual((got["backend"], got["model"]), (None, None))


def _reply(finding="Supports", scope="Supports", strength="Not applicable", judgement="Supports",
            span="the sentence", confidence="High", sufficiency="sufficient"):
    return {
        "slots": {"finding": {"assertion": "f", "verdict": finding},
                  "scope": {"assertion": "s", "verdict": scope},
                  "strength": {"assertion": "t", "verdict": strength}},
        "judgement": judgement, "evidence_sufficiency": sufficiency, "confidence": confidence,
        "supporting_span": span, "reason": "r",
    }


class TestDeriveJudgement(unittest.TestCase):
    """The rubric's Step 3, in code, first match wins."""

    def test_scope_does_not_support_beats_everything(self):
        self.assertEqual(judge_mod.derive_judgement(_reply(finding="Contradicts", scope="Does not support")["slots"]),
                         "Does not support")

    def test_contradicts(self):
        self.assertEqual(judge_mod.derive_judgement(_reply(finding="Contradicts")["slots"]), "Contradicts")

    def test_finding_does_not_support(self):
        self.assertEqual(judge_mod.derive_judgement(_reply(finding="Does not support")["slots"]), "Does not support")

    def test_insufficient_finding_or_scope_is_unclear(self):
        self.assertEqual(judge_mod.derive_judgement(_reply(finding="Insufficient")["slots"]),
                         "Unclear / insufficient evidence")
        self.assertEqual(judge_mod.derive_judgement(_reply(scope="Insufficient")["slots"]),
                         "Unclear / insufficient evidence")

    def test_all_supports_or_not_applicable_is_supports(self):
        self.assertEqual(judge_mod.derive_judgement(_reply(strength="Supports")["slots"]), "Supports")
        self.assertEqual(judge_mod.derive_judgement(_reply(strength="Not applicable")["slots"]), "Supports")

    def test_partial_scope_or_strength_is_partially_supports(self):
        self.assertEqual(judge_mod.derive_judgement(_reply(scope="Partially supports")["slots"]), "Partially supports")
        self.assertEqual(judge_mod.derive_judgement(_reply(strength="Insufficient")["slots"]), "Partially supports")

    def test_out_of_vocabulary_slot_counts_as_insufficient(self):
        # "Partially supports" is not a finding verdict — seen from gemma4:e2b twice on 2026-09-12.
        self.assertEqual(judge_mod.derive_judgement(_reply(finding="Partially supports")["slots"]),
                         "Unclear / insufficient evidence")


class TestSpanIsVerbatim(unittest.TestCase):
    EVIDENCE = "The ﬁlms showed a 10% increase in Δσ_ph — “as expected”.  Next sentence."

    def test_exact_substring(self):
        self.assertTrue(judge_mod.span_is_verbatim("Next sentence.", self.EVIDENCE))

    def test_whitespace_case_ligature_and_quote_differences_are_tolerated(self):
        self.assertTrue(judge_mod.span_is_verbatim('the films showed a 10% increase in Δσ_ph - "as expected".', self.EVIDENCE))

    def test_paraphrase_is_not_verbatim(self):
        self.assertFalse(judge_mod.span_is_verbatim("Films increased by ten percent.", self.EVIDENCE))

    def test_ellipsis_joined_span_is_verbatim_when_every_piece_is(self):
        """The rubric asks for one sentence; models sometimes join two with
        "..." — each piece grounded, the join not. Grounded is what matters."""
        self.assertTrue(judge_mod.span_is_verbatim("The ﬁlms showed a 10% increase ... Next sentence.", self.EVIDENCE))
        self.assertTrue(judge_mod.span_is_verbatim("The films showed […] Next sentence.", self.EVIDENCE))
        self.assertFalse(judge_mod.span_is_verbatim("The films showed ... something invented here.", self.EVIDENCE))

    def test_no_span_is_none(self):
        self.assertIsNone(judge_mod.span_is_verbatim(None, self.EVIDENCE))
        self.assertIsNone(judge_mod.span_is_verbatim("", self.EVIDENCE))
        self.assertIsNone(judge_mod.span_is_verbatim("null", self.EVIDENCE))


class TestEnforceRubric(unittest.TestCase):
    def test_clean_reply_passes_through_with_derived_fields(self):
        out = judge_mod.enforce_rubric(_reply(span="the sentence"), "here is the sentence indeed")
        self.assertEqual(out["judgement"], "Supports")
        self.assertEqual(out["model_judgement"], "Supports")
        self.assertFalse(out["rubric_mismatch"])
        self.assertEqual(out["rubric_violations"], [])
        self.assertTrue(out["span_verified"])
        self.assertEqual(out["confidence"], "High")

    def test_model_aggregate_that_breaks_the_rules_is_overridden_and_recorded(self):
        r = _reply(scope="Does not support", judgement="Supports")
        out = judge_mod.enforce_rubric(r, "the sentence")
        self.assertEqual(out["judgement"], "Does not support")
        self.assertEqual(out["model_judgement"], "Supports")
        self.assertTrue(out["rubric_mismatch"])
        self.assertEqual(out["confidence"], "Medium")

    def test_out_of_vocabulary_slot_is_recorded_and_caps_confidence(self):
        r = _reply(finding="Partially supports", judgement="Partially supports")
        out = judge_mod.enforce_rubric(r, "the sentence")
        self.assertEqual(out["rubric_violations"], ["finding: 'Partially supports'"])
        self.assertEqual(out["judgement"], "Unclear / insufficient evidence")
        self.assertTrue(out["rubric_mismatch"])
        self.assertEqual(out["confidence"], "Medium")

    def test_unverified_span_caps_confidence_but_keeps_the_verdict(self):
        out = judge_mod.enforce_rubric(_reply(span="not in there"), "the evidence text")
        self.assertFalse(out["span_verified"])
        self.assertEqual(out["judgement"], "Supports")
        self.assertEqual(out["confidence"], "Medium")

    def test_absence_cannot_be_asserted_from_insufficient_evidence(self):
        """Seen on 2026-09-14 (gemini-3.5-flash-lite, arxiv_2007.12504v1): five
        of six verdicts were "Does not support" at High confidence with
        evidence_sufficiency "insufficient" and a null span — "the retrieved
        passages don't mention it" reported as "the paper does not say it".
        Three chunks cannot establish absence; the rubric's own reading of
        "insufficient" is "too fragmentary to assess"."""
        r = _reply(finding="Does not support", scope="Does not support", judgement="Does not support",
                   span=None, sufficiency="insufficient")
        out = judge_mod.enforce_rubric(r, "unrelated evidence text")
        self.assertEqual(out["judgement"], "Unclear / insufficient evidence")
        self.assertEqual(out["model_judgement"], "Does not support")
        self.assertTrue(out["rubric_mismatch"])
        self.assertIn("absence", " ".join(out["rubric_violations"]))
        self.assertEqual(out["confidence"], "Medium")

    def test_does_not_support_stands_when_evidence_was_partial_or_sufficient(self):
        for suff in ("partial", "sufficient"):
            r = _reply(finding="Does not support", judgement="Does not support", span=None, sufficiency=suff)
            out = judge_mod.enforce_rubric(r, "evidence")
            self.assertEqual(out["judgement"], "Does not support", suff)
            self.assertFalse(out["rubric_mismatch"], suff)

    def test_span_that_cites_another_work_is_flagged(self):
        """"In specific cases, this ergodic hypothesis has been proved 18 ." —
        the cited paper is attributing the statement to reference 18. The
        verdict stands; the record says the support is secondhand."""
        for span in ("In specific cases, this ergodic hypothesis has been proved 18 .",
                     "This was shown by Smith et al. for graphene.",
                     "The effect is well established [12, 13] in this regime.",
                     "As reported in Ref. 7, the gap opens."):
            out = judge_mod.enforce_rubric(_reply(span=span), span)
            self.assertTrue(out["span_cites_others"], span)
        out = judge_mod.enforce_rubric(_reply(span="We measured a gap of 18 meV at 4 K."), "We measured a gap of 18 meV at 4 K.")
        self.assertFalse(out["span_cites_others"])
        self.assertIsNone(judge_mod.enforce_rubric(_reply(span=None), "x")["span_cites_others"])

    def test_slot_assertion_that_drifts_from_the_claim_is_flagged(self):
        """Seen on 2026-09-14: for the claim about δE_cor scaling, the model's
        finding assertion was "temperature has little effect on the inversion
        procedure" — imported from the context sentence. The verdict stands;
        the record says the decomposition drifted and confidence is capped."""
        claim = ("In our case, where we consider variations of the system chemical potential, "
                 "δE cor scales with the mean level density times the transmission, which makes "
                 "δE cor much larger than typical thermal broadening.")
        r = _reply()
        r["slots"]["finding"]["assertion"] = "temperature has little effect on the inversion procedure because the correlation energy is large"
        r["slots"]["scope"]["assertion"] = "variations of the system chemical potential"
        out = judge_mod.enforce_rubric(r, "the sentence", claim=claim)
        self.assertTrue(any(v.startswith("finding: assertion drift") for v in out["rubric_violations"]), out["rubric_violations"])
        self.assertFalse(any(v.startswith("scope:") for v in out["rubric_violations"]))
        self.assertEqual(out["confidence"], "Medium")
        self.assertEqual(out["judgement"], "Supports")

    def test_the_prompts_own_examples_do_not_trip_the_drift_check(self):
        """The seven worked examples are the floor: a guard that flags them
        flags correct decompositions."""
        import re as _re
        text = judge_mod.PROMPT_TEMPLATE
        blocks = _re.findall(r"Claim: \*(.+?)\*\nEvidence: \*(.+?)\*\n.*?```\n(\{.*?\})\n```", text, _re.S)
        self.assertEqual(len(blocks), 7)
        for claim, evidence, payload in blocks:
            r = json.loads(payload)
            out = judge_mod.enforce_rubric(r, evidence, claim=claim)
            self.assertFalse(any("assertion drift" in v for v in out["rubric_violations"]), (claim, out["rubric_violations"]))

    def test_no_claim_means_no_drift_check(self):
        r = _reply(); r["slots"]["finding"]["assertion"] = "something entirely unrelated to anything"
        out = judge_mod.enforce_rubric(r, "the sentence")
        self.assertEqual(out["rubric_violations"], [])

    def test_low_confidence_is_not_raised_to_medium(self):
        out = judge_mod.enforce_rubric(_reply(span="not in there", confidence="Low"), "the evidence text")
        self.assertEqual(out["confidence"], "Low")

    def test_judge_applies_enforcement(self):
        reply = json.dumps(_reply(scope="Does not support", judgement="Supports", span="evidence"))
        with patch.object(judge_mod, "chat", return_value=types.SimpleNamespace(content=reply)):
            out = judge_mod.judge("claim", "the evidence")
        self.assertEqual(out["judgement"], "Does not support")
        self.assertTrue(out["rubric_mismatch"])


class TestComposeContext(unittest.TestCase):
    """The judge's context block is assembled in one place, from the citing
    paper's parts only. The cited paper's summary is never one of them."""

    def test_window_alone_is_the_window(self):
        self.assertEqual(judge_mod.compose_context("Before. «claim» After."), "Before. «claim» After.")

    def test_section_and_artifacts_precede_the_window(self):
        out = judge_mod.compose_context(
            "Before. «claim» After.",
            section="2. Results > 2.1. THz",
            artifacts=[{"label": "Fig. 1", "caption": "THz  spectra\nof films."}, {"label": "Fig. 2", "caption": ""}],
        )
        self.assertEqual(out, "[Section: 2. Results > 2.1. THz]\n[Fig. 1: THz spectra of films.]\nBefore. «claim» After.")

    def test_caption_is_capped(self):
        out = judge_mod.compose_context("«c»", artifacts=[{"label": "Fig. 1", "caption": "x" * 500}])
        self.assertEqual(out, f"[Fig. 1: {'x' * judge_mod.ARTIFACT_CAPTION_CHARS}]\n«c»")

    def test_no_window_is_none_even_with_a_section(self):
        self.assertIsNone(judge_mod.compose_context("", section="2. Results"))
        self.assertIsNone(judge_mod.compose_context(None, artifacts=[{"label": "Fig. 1", "caption": "c"}]))

    def test_header_names_the_parts_and_keeps_the_never_as_evidence_rule(self):
        prompt = build_prompt("claim", "evidence", context="«claim»")
        self.assertIn("captions of figures", prompt)
        self.assertIn("never as evidence", prompt)

