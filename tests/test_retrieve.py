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


class TestShortlist(unittest.TestCase):
    def test_dedupe_by_title_keeps_the_higher_score(self):
        ranked = [{"document": "arxiv_1.pdf", "citation": "Conductance Quantization in Ribbons", "summary": "a", "score": 0.7},
                  {"document": "doi_1.pdf", "citation": "conductance quantization in ribbons.", "summary": "b", "score": 0.9},
                  {"document": "d3.pdf", "citation": "Other", "summary": "c", "score": 0.8}]
        out = rt.dedupe_shortlist(ranked)
        self.assertEqual([r["document"] for r in out], ["doi_1.pdf", "d3.pdf"])

    def test_keys_are_assigned_in_order(self):
        sel = _ranked(3)
        keys = rt.assign_keys(sel)
        self.assertEqual([r["key"] for r in sel], ["P1", "P2", "P3"])
        self.assertEqual(keys, {"P1": "Paper 1", "P2": "Paper 2", "P3": "Paper 3"})


class TestPassagesPerPaper(unittest.TestCase):
    def test_one_search_per_paper_restricted_to_it(self):
        sel = _ranked(2); rt.assign_keys(sel)
        calls = []

        def fake_search(query, collection, bm25, texts, metadatas, top_k, doc_filter=None, exclude_types=None, **kw):
            calls.append((top_k, doc_filter, exclude_types))
            doc = next(iter(doc_filter))
            return [{"chunk_index": 1, "text": f"chunk of {doc}", "metadata": {"document": doc, "page": 3, "section": "results"}}]

        with patch.object(rt, "load_search_resources", return_value=(None, None, [], [])), \
             patch.object(rt, "hybrid_search", side_effect=fake_search):
            out = rt.passages_per_paper("idea", sel, per_paper=4)
        self.assertEqual(calls, [(4, {"d1.pdf"}, {"figure_description"}), (4, {"d2.pdf"}, {"figure_description"})])
        self.assertEqual(list(out), ["P1", "P2"])
        self.assertEqual(out["P1"][0]["key"], "P1")

    def test_format_passages_caps_and_labels(self):
        hits = [{"text": "A" * 30, "metadata": {"page": 2, "section": "methods"}},
                {"text": "B" * 30, "metadata": {"page": 5, "section": "results"}}]
        out = rt.format_passages(hits, max_chars=60)
        self.assertTrue(out.startswith("(p.2, methods) " + "A" * 30))
        self.assertLessEqual(len(out), 60 + len("\n\n"))
        self.assertIn("(p.5, results)", out)

