import json
import os
import unittest

from research_assistant import config
from research_assistant.transfer import models
from tests.transfer_fixtures import good_transfer, premise, sample_result


class TestModels(unittest.TestCase):
    def test_premise_credit_follows_the_citers_rule(self):
        self.assertEqual(premise("Supports").credit, 1.0); self.assertEqual(premise("Partially supports").credit, 0.5)
        for j, s in [("Supports", None), ("Supports", False), ("Unclear / insufficient evidence", True), (None, None)]:
            self.assertEqual(premise(j, span=s).credit, 0.0, (j, s))
        self.assertTrue(premise("Contradicts").contradicts)

    def test_a_card_is_verified_only_by_an_accepted_or_partial_evidence_premise(self):
        _, _, card = good_transfer(); self.assertTrue(card.verified)
        card.evidence = premise("Unclear / insufficient evidence"); self.assertFalse(card.verified)

    def test_a_run_and_a_card_survive_json(self):
        r = sample_result()
        self.assertEqual(models.result_from_dict(json.loads(json.dumps(models.result_to_dict(r)))), r)
        _, _, c = good_transfer(); self.assertEqual(models.card_from_dict(json.loads(json.dumps(models.card_to_dict(c)))), c)

    def test_the_transfer_directory_is_under_data_dir(self):
        self.assertTrue(os.path.realpath(config.TRANSFER_DIR).startswith(os.path.realpath(config.DATA_DIR)))
