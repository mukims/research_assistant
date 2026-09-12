# tests/test_retrieve.py
"""The synthesis path: gate, shortlist, per-paper passages, notes, synthesis.

Everything below stubs the model and the index; the live behaviour is
checked in the plan's Task 7. What these pin is the bookkeeping that makes
the output honest: verdicts aligned to their summaries, every shortlisted
paper read, keys checked against the shortlist.
"""

import types
import unittest
from unittest.mock import patch

from research_assistant.shared import retrieve as rt


def _reply(text):
    return types.SimpleNamespace(content=text)


def _ranked(n):
    return [{"document": f"d{i}.pdf", "citation": f"Paper {i}", "summary": f"summary {i}", "score": 1 - i / 10}
            for i in range(1, n + 1)]


class TestParseGateVerdicts(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(rt.parse_gate_verdicts("1: YES\n2: NO\n3: yes", 3), [True, False, True])

    def test_tolerates_punctuation_and_preamble(self):
        self.assertEqual(rt.parse_gate_verdicts("Here you go:\n1) YES\n2. NO\n", 2), [True, False])

    def test_out_of_order_is_indexed_by_number(self):
        self.assertEqual(rt.parse_gate_verdicts("2: NO\n1: YES", 2), [True, False])

    def test_missing_duplicate_or_extra_is_none(self):
        self.assertIsNone(rt.parse_gate_verdicts("1: YES", 2))
        self.assertIsNone(rt.parse_gate_verdicts("1: YES\n1: NO", 2))
        self.assertIsNone(rt.parse_gate_verdicts("1: YES\n2: NO\n3: YES", 2))
        self.assertIsNone(rt.parse_gate_verdicts("YES NO", 2))


class TestGateDocuments(unittest.TestCase):
    def test_batched_gate_is_one_call_and_keeps_the_yeses(self):
        with patch.object(rt, "GATE_BATCHED", True), \
             patch.object(rt, "chat", return_value=_reply("1: YES\n2: NO\n3: YES")) as chat:
            kept = rt.gate_documents("idea", _ranked(3))
        self.assertEqual(chat.call_count, 1)
        self.assertEqual([r["document"] for r in kept], ["d1.pdf", "d3.pdf"])
        prompt = chat.call_args.args[0][-1]["content"]
        self.assertIn("summary 2", prompt); self.assertIn("idea", prompt)

    def test_misaligned_reply_falls_back_to_per_summary_calls(self):
        replies = iter([_reply("1: YES\n1: YES"), _reply("YES"), _reply("NO"), _reply("YES")])
        with patch.object(rt, "GATE_BATCHED", True), patch.object(rt, "chat", side_effect=lambda *a, **k: next(replies)) as chat:
            kept = rt.gate_documents("idea", _ranked(3))
        self.assertEqual(chat.call_count, 4)
        self.assertEqual([r["document"] for r in kept], ["d1.pdf", "d3.pdf"])

    def test_everything_rejected_keeps_the_top_summary(self):
        with patch.object(rt, "GATE_BATCHED", True), patch.object(rt, "chat", return_value=_reply("1: NO\n2: NO")):
            kept = rt.gate_documents("idea", _ranked(2))
        self.assertEqual([r["document"] for r in kept], ["d1.pdf"])

    def test_batched_off_uses_per_summary_calls(self):
        with patch.object(rt, "GATE_BATCHED", False), patch.object(rt, "chat", return_value=_reply("YES")) as chat:
            rt.gate_documents("idea", _ranked(3))
        self.assertEqual(chat.call_count, 3)
