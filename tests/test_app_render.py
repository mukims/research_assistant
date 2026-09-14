# tests/test_app_render.py
"""The one pure piece of Tab 1's synthesis rendering: which warnings the
reader sees. Everything else in app.py is Streamlit calls."""

import unittest

import app


class TestSynthesisWarnings(unittest.TestCase):
    def test_clean_answer_has_no_warnings(self):
        self.assertEqual(app._synthesis_warnings({"unverified_citations": [], "irrelevant_cited": []}), [])

    def test_unverified_and_irrelevant_keys_are_named(self):
        w = app._synthesis_warnings({"unverified_citations": ["P7"], "irrelevant_cited": ["P2"], "keys": {"P2": "Paper 2"}})
        self.assertEqual(len(w), 2)
        self.assertIn("P7", w[0]); self.assertIn("P2", w[1]); self.assertIn("Paper 2", w[1])

    def test_old_answers_without_the_fields_are_fine(self):
        self.assertEqual(app._synthesis_warnings({"suggestion": "x"}), [])


class TestAuditItemBadge(unittest.TestCase):
    def test_badges_follow_outcome_before_judgement(self):
        self.assertEqual(app._audit_item_badge({"outcome": "judged", "judgement": "Supports"}), "🟢 Supports")
        self.assertEqual(app._audit_item_badge({"outcome": "judged", "judgement": "Does not support"}), "🟠 Does Not Support")
        self.assertEqual(app._audit_item_badge({"outcome": "judged", "judgement": "Unclear / insufficient evidence"}), "⚪ Unclear / Insufficient Evidence")
        self.assertEqual(app._audit_item_badge({"outcome": "cap_exceeded", "judgement": None}), "⏸ Not Assessed (budget)")
        self.assertEqual(app._audit_item_badge({"outcome": "not_attempted", "judgement": None}), "⏸ Not Assessed (backend down)")
        self.assertEqual(app._audit_item_badge({"outcome": "not_a_claim", "judgement": None, "role": "software"}), "🔧 Not a Claim (software)")
        self.assertEqual(app._audit_item_badge({"outcome": "deferred_paywalled", "judgement": None}), "⏳ Deferred (Pending Evidence)")
        self.assertEqual(app._audit_item_badge({"outcome": "not_downloaded", "judgement": None}), "🔒 Paywalled / Not In Corpus")
        # No outcome ever falls through to a verdict-looking label.
        self.assertNotIn("Unclear", app._audit_item_badge({"outcome": "retrieval_failed", "judgement": None}))
