"""Tests for the ingestion manifest.

This records what has already been parsed. Parsing is the expensive stage —
layout detection per page plus a VLM call per figure — so a key that fails to
persist means the PDF is re-parsed in full on every future run.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import research_assistant.shared.manifest as m


class ManifestTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "ingested.json")
        patcher = patch.object(m, "MANIFEST_PATH", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)


class TestLoadAndAdd(ManifestTestCase):
    def test_missing_file_is_empty_not_an_error(self):
        self.assertEqual(m.load(), {})
        self.assertFalse(m.contains("paper.pdf"))

    def test_add_then_contains(self):
        m.add("paper.pdf")
        self.assertTrue(m.contains("paper.pdf"))

    def test_add_persists_to_disk(self):
        m.add("paper.pdf")
        with open(self.path) as fh:
            self.assertIn("paper.pdf", json.load(fh))

    def test_adding_twice_does_not_duplicate(self):
        """The old text file appended unconditionally and grew forever."""
        m.add("paper.pdf")
        m.add("paper.pdf")
        self.assertEqual(list(m.load()), ["paper.pdf"])

    def test_add_many_writes_every_key(self):
        m.add_many(["a.pdf", "b.pdf", "c.pdf"])
        self.assertEqual(sorted(m.load()), ["a.pdf", "b.pdf", "c.pdf"])

    def test_add_many_preserves_existing_entries(self):
        m.add("old.pdf")
        m.add_many(["new.pdf"])
        self.assertEqual(sorted(m.load()), ["new.pdf", "old.pdf"])


class TestCorruption(ManifestTestCase):
    def test_unreadable_manifest_is_treated_as_empty(self):
        """Corrupt state must not crash ingestion — it re-parses at worst."""
        with open(self.path, "w") as fh:
            fh.write("{not json")
        self.assertEqual(m.load(), {})

    def test_wrong_toplevel_type_is_treated_as_empty(self):
        with open(self.path, "w") as fh:
            json.dump(["a.pdf"], fh)
        self.assertEqual(m.load(), {})

    def test_writing_over_a_corrupt_manifest_recovers_it(self):
        with open(self.path, "w") as fh:
            fh.write("{not json")
        m.add("paper.pdf")
        self.assertEqual(list(m.load()), ["paper.pdf"])


class TestAtomicity(ManifestTestCase):
    def test_no_temp_files_left_behind(self):
        m.add("paper.pdf")
        leftovers = [f for f in os.listdir(self.tmp.name) if f != "ingested.json"]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
