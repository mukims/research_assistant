import os
import tempfile
import unittest

from research_assistant.transfer import render, store
from tests.transfer_fixtures import sample_result


class TestRender(unittest.TestCase):
    def test_hostile_text_cannot_break_a_label(self):
        label = render.mermaid_label('A "b" [c] {d} #1 <e> & `f` |g|\nh ' + "w" * 80, limit=60)
        for ch in '"[]{}#<>&`|\n':
            self.assertNotIn(ch, label)
        self.assertLessEqual(len(label), 60)

    def test_the_report_labels_the_proposal_and_sets_aside_with_the_reason(self):
        md = render.render_markdown(sample_result())
        for text in ("**Proposal (not stated by any source):**", "## Set aside", "not_identifiable", "```mermaid", "✅ Supports"):
            self.assertIn(text, md)

    def test_the_tree_marks_the_best_and_the_set_aside(self):
        tree = render.render_tree(sample_result())
        for line in ("R --> B1", "B1 --> T_B1_T1", "class T_B1_T1 best", "class T_B1_T2 aside"):
            self.assertIn(line, tree)

    def test_save_and_load_round_trip_under_runs(self):
        with tempfile.TemporaryDirectory() as d:
            path = store.save_result(sample_result(), directory=d)
            self.assertEqual(path, os.path.join(d, "20261021-120000-abcdef12.json"))
            self.assertEqual(store.load_result(path), sample_result())
