"""Tests for corpus_tab and agent6_manual_ingestor batch ingestion."""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from research_assistant.agents import agent6_manual_ingestor


class TestManualIngestorDirectory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    @patch("research_assistant.shared.batch_uploader.scan_and_stage_directory")
    @patch("research_assistant.agents.agent6_manual_ingestor.ingest_pdfs")
    def test_ingest_directory_calls_ingest_pdfs(self, mock_ingest, mock_scan):
        mock_scan.return_value = [
            {"path": "/tmp/a.pdf", "title": "Paper A"},
            {"path": "/tmp/b.pdf", "title": "Paper B"},
        ]
        mock_ingest.return_value = {"processed": 2, "inserted": 10, "skipped": 0, "failed": []}

        res = agent6_manual_ingestor.ingest_directory("/tmp/folder", workers=2, force=True)
        self.assertEqual(res["processed"], 2)
        mock_scan.assert_called_once_with("/tmp/folder", recursive=True)
        mock_ingest.assert_called_once_with(
            {"/tmp/a.pdf": "Paper A", "/tmp/b.pdf": "Paper B"},
            workers=2,
            skip_ingested=False,
        )

    @patch("research_assistant.shared.batch_uploader.scan_and_stage_directory")
    def test_ingest_directory_empty_folder(self, mock_scan):
        mock_scan.return_value = []
        res = agent6_manual_ingestor.ingest_directory("/tmp/empty")
        self.assertEqual(res["processed"], 0)


if __name__ == "__main__":
    unittest.main()
