"""The pure parts of the v3 migration: which PDFs the old manifests name, and where they are."""

import json
import os
import sys
import tempfile
import unittest

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from scripts import migrate_to_v3 as mig


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").close()


class TestOldDocumentKeys(unittest.TestCase):
    def test_union_of_both_manifests_pdfs_apart_from_other_entries(self):
        with tempfile.TemporaryDirectory() as d:
            json.dump({"a.pdf": "t", "b.pdf": "t", "paper.tei.xml": "t"}, open(os.path.join(d, "ingested.json"), "w"))
            json.dump({"b.pdf": "t", "c.pdf": "t"}, open(os.path.join(d, "ingested_v2.json"), "w"))
            pdfs, other = mig.old_document_keys(d)
        self.assertEqual(pdfs, ["a.pdf", "b.pdf", "c.pdf"])
        self.assertEqual(other, ["paper.tei.xml"])

    def test_a_missing_manifest_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(mig.old_document_keys(d), ([], []))


class TestFindPdfs(unittest.TestCase):
    def test_matches_on_the_normalised_key_and_reports_the_rest(self):
        with tempfile.TemporaryDirectory() as d:
            _touch(os.path.join(d, "pulled", "My Paper.PDF"))
            _touch(os.path.join(d, "raw", "processed", "seed.pdf"))
            found, missing = mig.find_pdfs(["my_paper.pdf", "seed.pdf", "gone.pdf"],
                                           [os.path.join(d, "pulled"), os.path.join(d, "raw")])
        self.assertEqual(sorted(found), ["my_paper.pdf", "seed.pdf"])
        self.assertTrue(found["my_paper.pdf"].endswith("My Paper.PDF"))
        self.assertEqual(missing, ["gone.pdf"])

    def test_the_first_directory_wins_when_a_paper_is_in_two(self):
        with tempfile.TemporaryDirectory() as d:
            _touch(os.path.join(d, "one", "p.pdf"))
            _touch(os.path.join(d, "two", "p.pdf"))
            found, _ = mig.find_pdfs(["p.pdf"], [os.path.join(d, "one"), os.path.join(d, "two")])
        self.assertIn(os.sep + "one" + os.sep, found["p.pdf"])


class TestRunBatches(unittest.TestCase):
    FOUND = {f"p{i}.pdf": f"/x/p{i}.pdf" for i in range(5)}
    TODO = sorted(FOUND)

    def _ingest(self, calls, pymupdf_on=None):
        def ingest(paths):
            calls.append(list(paths))
            fell = 1 if pymupdf_on is not None and len(calls) == pymupdf_on else 0
            return {"processed": len(paths), "inserted": 2 * len(paths), "failed": [],
                    "figure_images": len(paths), "summaries": {"failed": 0},
                    "extraction": {"grobid": len(paths) - fell, "pymupdf": fell}}
        return ingest

    def test_every_batch_runs_when_grobid_stays_up(self):
        calls = []
        totals, stopped = mig.run_batches(self.TODO, self.FOUND, self._ingest(calls), lambda: True, 2, say=lambda *_: None)
        self.assertEqual(stopped, "")
        self.assertEqual([len(c) for c in calls], [2, 2, 1])
        self.assertEqual((totals["processed"], totals["inserted"], totals["figure_images"]), (5, 10, 5))

    def test_a_dead_grobid_stops_before_the_next_batch_is_started(self):
        calls, answers = [], iter([True, False])
        totals, stopped = mig.run_batches(self.TODO, self.FOUND, self._ingest(calls), lambda: next(answers), 2,
                                          say=lambda *_: None)
        self.assertEqual(len(calls), 1)
        self.assertIn("GROBID stopped answering", stopped)
        self.assertIn("p2.pdf", stopped)
        self.assertEqual(totals["processed"], 2)

    def test_the_first_batch_that_fell_back_to_pymupdf_stops_the_run(self):
        calls = []
        totals, stopped = mig.run_batches(self.TODO, self.FOUND, self._ingest(calls, pymupdf_on=2), lambda: True, 2,
                                          say=lambda *_: None)
        self.assertEqual(len(calls), 2)
        self.assertIn("fell back to PyMuPDF", stopped)
        self.assertIn("p2.pdf .. p3.pdf", stopped)
        self.assertEqual(totals["pymupdf"], 1)


if __name__ == "__main__":
    unittest.main()
