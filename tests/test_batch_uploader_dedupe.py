"""Tests for batch_uploader layer-1 deduplication (pasted DOIs and arXiv IDs)."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from research_assistant.agents import agent2_fetcher
from research_assistant.shared import batch_uploader
from research_assistant.shared.paper_index import PaperEntry, PaperIndex

CORPUS_PDF_BYTES = b"%PDF-1.4 corpus copy for batch uploader dedupe\n%%EOF"


class TestBatchUploaderDedupe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dest_dir = os.path.join(self.tmp.name, "staged")
        os.makedirs(self.dest_dir, exist_ok=True)

        self.corpus_path = os.path.join(self.tmp.name, "corpus.pdf")
        with open(self.corpus_path, "wb") as f:
            f.write(CORPUS_PDF_BYTES)

        self.index_path = os.path.join(self.tmp.name, "paper_index.json")
        self.index = PaperIndex(self.index_path)
        self.index.add(PaperEntry(
            document="doi_10.1038_nature05180.pdf",
            path=self.corpus_path,
            dois=("10.1038/nature05180",),
            arxiv_ids=("2007.12504",),
        ))
        self.index.save()

    def test_pasted_doi_in_corpus_is_staged_without_downloading(self):
        with patch.object(batch_uploader, "PAPER_INDEX_PATH", self.index_path), \
             patch.object(batch_uploader.requests, "get") as mock_get, \
             patch.object(agent2_fetcher, "try_unpaywall") as mock_unpaywall, \
             patch.object(agent2_fetcher, "try_europepmc") as mock_europepmc:
            staged = batch_uploader.fetch_and_stage_urls_or_dois(
                ["10.1038/nature05180"],
                destination_dir=self.dest_dir,
            )
            mock_get.assert_not_called()
            mock_unpaywall.assert_not_called()
            mock_europepmc.assert_not_called()
            self.assertEqual(len(staged), 1)
            with open(staged[0]["path"], "rb") as f:
                self.assertEqual(f.read(), CORPUS_PDF_BYTES)

    def test_pasted_arxiv_id_in_corpus_is_staged_without_downloading(self):
        with patch.object(batch_uploader, "PAPER_INDEX_PATH", self.index_path), \
             patch.object(batch_uploader.requests, "get") as mock_get:
            staged = batch_uploader.fetch_and_stage_urls_or_dois(
                ["arXiv:2007.12504v1"],
                destination_dir=self.dest_dir,
            )
            mock_get.assert_not_called()
            self.assertEqual(len(staged), 1)
            with open(staged[0]["path"], "rb") as f:
                self.assertEqual(f.read(), CORPUS_PDF_BYTES)

    def test_pasted_doi_not_in_corpus_is_downloaded_as_before(self):
        empty_index_path = os.path.join(self.tmp.name, "empty_index.json")
        PaperIndex(empty_index_path).save()

        def fake_unpaywall(doi, dest_path):
            with open(dest_path, "wb") as f:
                f.write(b"%PDF-1.4 unpaywall fresh\n%%EOF")
            return True, None

        with patch.object(batch_uploader, "PAPER_INDEX_PATH", empty_index_path), \
             patch.object(agent2_fetcher, "try_unpaywall", side_effect=fake_unpaywall) as mock_unpaywall:
            staged = batch_uploader.fetch_and_stage_urls_or_dois(
                ["10.1038/nature05180"],
                destination_dir=self.dest_dir,
            )
            self.assertEqual(mock_unpaywall.call_count, 1)
            self.assertEqual(len(staged), 1)


if __name__ == "__main__":
    unittest.main()
