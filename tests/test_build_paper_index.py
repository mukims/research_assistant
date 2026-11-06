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

        self.grobid_patch = patch("scripts.build_paper_index.grobid_alive", return_value=True)
        self.grobid_patch.start()
        self.addCleanup(self.grobid_patch.stop)

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

    def test_backfill_keeps_entries_written_meanwhile(self):
        from scripts.build_paper_index import build_index
        from research_assistant.shared.paper_index import edit, PaperEntry

        pdf_1 = os.path.join(self.pulled_dir, "p1.pdf")
        pdf_2 = os.path.join(self.pulled_dir, "p2.pdf")
        with open(pdf_1, "wb") as f:
            f.write(b"%PDF-1")
        with open(pdf_2, "wb") as f:
            f.write(b"%PDF-2")

        first_call = True

        def fake_read_header(path, convert_if_missing=False):
            nonlocal first_call
            if first_call:
                first_call = False
                with edit(self.index_path) as idx:
                    idx.add(PaperEntry(document="extra_X.pdf", path="/some/extra_X.pdf", title="Paper X"))
            return PaperHeader(title=f"Title {os.path.basename(path)}", first_authors=("Smith",), doi=None, arxiv_id=None)

        with patch("scripts.build_paper_index.read_header", side_effect=fake_read_header):
            res = build_index(
                index_path=self.index_path,
                report_path=self.report_path,
                pulled_dir=self.pulled_dir,
                raw_dir=self.raw_dir,
                manifest_paths=[self.downloaded_path],
            )

        reloaded = PaperIndex.load(self.index_path)
        self.assertIsNotNone(reloaded.get("extra_X.pdf"))
        self.assertIsNotNone(reloaded.get("p1.pdf"))
        self.assertIsNotNone(reloaded.get("p2.pdf"))

    def test_backfill_refuses_to_start_without_grobid(self):
        from scripts.build_paper_index import build_index

        pdf_1 = os.path.join(self.pulled_dir, "p1.pdf")
        with open(pdf_1, "wb") as f:
            f.write(b"%PDF-1")

        with patch("scripts.build_paper_index.grobid_alive", return_value=False):
            res = build_index(
                index_path=self.index_path,
                report_path=self.report_path,
                pulled_dir=self.pulled_dir,
                raw_dir=self.raw_dir,
                manifest_paths=[self.downloaded_path],
            )
            self.assertIn("error", res)
            self.assertIn("GROBID is down", res["error"])
            self.assertFalse(os.path.exists(self.index_path))

            # --dry-run still works without GROBID
            dry_res = build_index(
                index_path=self.index_path,
                report_path=self.report_path,
                pulled_dir=self.pulled_dir,
                raw_dir=self.raw_dir,
                manifest_paths=[self.downloaded_path],
                dry_run=True,
            )
            self.assertNotIn("error", dry_res)
            self.assertEqual(dry_res["total"], 1)

    def test_backfill_stops_cleanly_when_grobid_goes_down(self):
        from scripts.build_paper_index import build_index

        for i in range(30):
            pdf_path = os.path.join(self.pulled_dir, f"paper_{i:02d}.pdf")
            with open(pdf_path, "wb") as f:
                f.write(b"%PDF-test")

        # Probe returns True initially, then False at checkpoint
        probe_responses = [True, False]

        def fake_grobid_alive(*args, **kwargs):
            if probe_responses:
                return probe_responses.pop(0)
            return False

        def fake_read_header(path, convert_if_missing=False):
            return PaperHeader(title=f"Title {os.path.basename(path)}", first_authors=("Smith",), doi=None, arxiv_id=None)

        with patch("scripts.build_paper_index.grobid_alive", side_effect=fake_grobid_alive), \
             patch("scripts.build_paper_index.read_header", side_effect=fake_read_header):
            res = build_index(
                index_path=self.index_path,
                report_path=self.report_path,
                pulled_dir=self.pulled_dir,
                raw_dir=self.raw_dir,
                manifest_paths=[self.downloaded_path],
            )
            self.assertIn("error", res)
            self.assertEqual(res["indexed"], 25)

        reloaded = PaperIndex.load(self.index_path)
        self.assertEqual(len(reloaded), 25)


if __name__ == "__main__":
    unittest.main()
