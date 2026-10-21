import json
import os
import tempfile
import unittest

from research_assistant.eval import transfer_cases

MINIMAL_CASE = {
    "id": "min",
    "kind": "positive",
    "query": "query text",
    "generic": "generic bottleneck",
    "cutoff_year": 2020,
    "corpus": "main",
    "papers": {"target": [], "source": [], "distractor": [], "hidden": []},
    "exclude_documents": [],
    "expected": {
        "source_titles": [],
        "technique_terms": [],
        "mapping_terms": [],
        "critic_terms": [],
        "set_aside": None,
    },
}


class TestCases(unittest.TestCase):
    def test_the_public_cases_load_and_a_private_key_is_merged(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "ae_neg.json"), "w") as fh:
                json.dump({**MINIMAL_CASE, "id": "ae_neg", "kind": "negative", "corpus": "main"}, fh)
            cases = transfer_cases.load_cases(private_dir=d)
        self.assertEqual([c.id for c in cases], ["nqs", "kpm", "ae_neg"])

    def test_a_case_missing_a_field_is_rejected_by_name(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "bad.json"), "w") as fh:
                json.dump({"id": "bad", "kind": "positive"}, fh)
            with self.assertRaisesRegex(ValueError, "bad"):
                transfer_cases.load_cases(private_dir=d)
