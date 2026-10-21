import tempfile
import unittest

import numpy as np

from research_assistant.transfer import cards, review
from tests.transfer_fixtures import card


class TestReview(unittest.TestCase):
    def test_the_gate_needs_seventy_percent_usable_and_sixty_percent_specific(self):
        rows = [
            {"card_id": str(i), "rating": r, "specific": s}
            for i, (r, s) in enumerate(
                [("correct", "y")] * 6 + [("partial", "y")] + [("wrong", "n")] * 3
            )
        ]
        self.assertEqual(
            review.score_review(rows),
            {"rated": 10, "correct_or_partial": 0.7, "specific": 0.7, "passed": True},
        )
        rows[0]["rating"] = "wrong"
        self.assertFalse(review.score_review(rows)["passed"])

    def test_blank_rows_are_ignored(self):
        self.assertEqual(
            review.score_review([{"card_id": "1", "rating": "", "specific": ""}])["rated"],
            0,
        )

    def test_the_sheet_has_one_block_and_one_csv_row_per_card(self):
        with tempfile.TemporaryDirectory() as d:
            store = cards.CardStore(d)
            store.append("a.pdf", [card("a.pdf", True)], np.array([[1.0, 0.0]]))
            md, csv_text = review.review_sheet(store.load(), ["a.pdf"])
        self.assertIn("a.pdf#0", md)
        self.assertEqual(
            csv_text.splitlines(),
            ["card_id,rating,specific", "a.pdf#0,,"],
        )

    def test_the_sheet_lists_each_mechanism_step_in_order(self):
        c = card("a.pdf", True)
        c.mechanism = ["Rescale the spectrum.", "Run the recursion."]
        with tempfile.TemporaryDirectory() as d:
            store = cards.CardStore(d)
            store.append("a.pdf", [c], np.array([[1.0, 0.0]]))
            md, _ = review.review_sheet(store.load(), ["a.pdf"])
        self.assertIn("1. Rescale the spectrum.\n", md)
        self.assertIn("2. Run the recursion.\n", md)
        self.assertLess(md.index("Mechanism"), md.index("Evidence Claim"))
