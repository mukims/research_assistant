"""Tests for scripts/build_paper_index.py."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from research_assistant.shared.paper_header import PaperHeader
from research_assistant.shared.paper_index import PaperEntry, PaperIndex


class TestBuildPaperIndex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        self.pulled_dir = os.path.join(self.tmp.name, "pulled_pdfs")
        self.raw_dir = os.path.join(self.tmp.name, "raw")
        os.makedirs(self.pulled_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)

        self.index_path = os.path.join(self.tmp.name, "paper_index.json")
        self.report_path = os.path.join(self.tmp.name, "dedupe_report.json")
        self.downloaded_path = os.path.join(self.tmp.name, "downloaded.json")

    def test_backfill_skips_papers_already_indexed(self):
        from scripts.build_paper_index import build_index

        pdf_a = os.path.join(self.pulled_dir, "a.pdf")
        pdf_b = os.path.join(self.pulled_dir, "b.pdf")
        with open(pdf_a, "wb") as f:
            f.write(b"%PDF-a")
        with open(pdf_b, "wb") as f:
            f.write(b"%PDF-b")

        idx = PaperIndex(self.index_path)
        idx.add(PaperEntry(document="a.pdf", path=pdf_a, title="Paper A", first_authors=("Smith",)))
        idx.save()

        calls = []

        def fake_read_header(path, convert_if_missing=False):
            calls.append(path)
            return PaperHeader(title="Paper B", first_authors=("Jones",), doi=None, arxiv_id=None)

        with patch("scripts.build_paper_index.read_header", side_effect=fake_read_header):
            build_index(
                index_path=self.index_path,
                report_path=self.report_path,
                pulled_dir=self.pulled_dir,
                raw_dir=self.raw_dir,
                manifest_paths=[self.downloaded_path],
            )

        self.assertNotIn(pdf_a, calls)
        self.assertIn(pdf_b, calls)

    def test_backfill_takes_dois_from_manifests(self):
        from scripts.build_paper_index import build_index

        pdf_b = os.path.join(self.pulled_dir, "b.pdf")
        with open(pdf_b, "wb") as f:
            f.write(b"%PDF-b")

        manifest_data = {
            "ref_b": {
                "path": pdf_b,
                "doi": "10.1038/nature05180",
            }
        }
        with open(self.downloaded_path, "w") as f:
            json.dump(manifest_data, f)

        header_b = PaperHeader(title="Paper B", first_authors=("Jones",), doi=None, arxiv_id=None)

        with patch("scripts.build_paper_index.read_header", return_value=header_b):
            build_index(
                index_path=self.index_path,
                report_path=self.report_path,
                pulled_dir=self.pulled_dir,
                raw_dir=self.raw_dir,
                manifest_paths=[self.downloaded_path],
            )

        idx = PaperIndex.load(self.index_path)
        entry_b = idx.get("b.pdf")
        self.assertIsNotNone(entry_b)
        self.assertEqual(entry_b.dois, ("10.1038/nature05180",))

    def test_dry_run_writes_nothing(self):
        from scripts.build_paper_index import build_index

        pdf_b = os.path.join(self.pulled_dir, "b.pdf")
        with open(pdf_b, "wb") as f:
            f.write(b"%PDF-b")

        build_index(
            index_path=self.index_path,
            report_path=self.report_path,
            pulled_dir=self.pulled_dir,
            raw_dir=self.raw_dir,
            manifest_paths=[self.downloaded_path],
            dry_run=True,
        )

        self.assertFalse(os.path.exists(self.index_path))


if __name__ == "__main__":
    unittest.main()
