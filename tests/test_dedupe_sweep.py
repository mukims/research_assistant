"""Tests for the end-of-run deduplication sweep (shared.dedupe_sweep)."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from research_assistant.agents import agent2_fetcher
from research_assistant.shared import dedupe_sweep
from research_assistant.shared import ingestion
from research_assistant.shared.paper_index import PaperEntry, PaperIndex


class TestDedupeSweep(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.index_path = os.path.join(self.tmp.name, "paper_index.json")
        self.report_path = os.path.join(self.tmp.name, "dedupe_report.json")

    def test_sweep_reports_papers_sharing_a_doi(self):
        idx = PaperIndex(self.index_path)
        idx.add(PaperEntry(document="doc1.pdf", path="/p/doc1.pdf", dois=("10.1000/1",)))
        idx.add(PaperEntry(document="doc2.pdf", path="/p/doc2.pdf", dois=("10.1000/1",)))
        idx.save()

        res = dedupe_sweep.run_sweep(self.index_path, self.report_path)
        self.assertEqual(len(res["groups"]), 1)
        self.assertEqual(res["groups"][0]["reason"], "doi")
        self.assertEqual(res["groups"][0]["documents"], ["doc1.pdf", "doc2.pdf"])

        with open(self.report_path, "r", encoding="utf-8") as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk["groups"], res["groups"])
        self.assertEqual(on_disk["papers"], 2)

    def test_clean_index_reports_no_groups(self):
        idx = PaperIndex(self.index_path)
        idx.add(PaperEntry(document="doc1.pdf", path="/p/doc1.pdf", dois=("10.1000/1",)))
        idx.add(PaperEntry(document="doc2.pdf", path="/p/doc2.pdf", dois=("10.1000/2",)))
        idx.save()

        res = dedupe_sweep.run_sweep(self.index_path, self.report_path)
        self.assertEqual(res["groups"], [])
        self.assertTrue(os.path.exists(self.report_path))

    def test_unreadable_index_does_not_raise(self):
        with open(self.index_path, "w") as f:
            f.write("{corrupt json")

        res = dedupe_sweep.run_sweep(self.index_path, self.report_path)
        self.assertEqual(res.get("groups"), [])

    def test_sweep_deletes_nothing(self):
        file1 = os.path.join(self.tmp.name, "doc1.pdf")
        file2 = os.path.join(self.tmp.name, "doc2.pdf")
        with open(file1, "wb") as f:
            f.write(b"%PDF-doc1")
        with open(file2, "wb") as f:
            f.write(b"%PDF-doc2")

        idx = PaperIndex(self.index_path)
        idx.add(PaperEntry(document="doc1.pdf", path=file1, dois=("10.1000/1",)))
        idx.add(PaperEntry(document="doc2.pdf", path=file2, dois=("10.1000/1",)))
        idx.save()

        dedupe_sweep.run_sweep(self.index_path, self.report_path)
        self.assertTrue(os.path.exists(file1))
        self.assertTrue(os.path.exists(file2))

    def test_ingest_pdfs_runs_the_sweep(self):
        with patch.object(ingestion, "run_sweep") as mock_sweep:
            ingestion.ingest_pdfs([])
            mock_sweep.assert_called_once()

    def test_fetch_papers_runs_the_sweep(self):
        with patch.object(agent2_fetcher, "_load_references", return_value=[]), \
             patch.object(agent2_fetcher, "_checkpoint"), \
             patch.object(agent2_fetcher, "run_sweep") as mock_sweep:
            agent2_fetcher.fetch_papers()
            mock_sweep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
