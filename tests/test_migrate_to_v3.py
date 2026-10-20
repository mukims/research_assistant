"""The pure parts of the v3 migration: which PDFs the old manifests name, and where they are."""

import json
import os
import sys
import tempfile
import types
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

    def test_papers_only_the_old_store_knows_are_included(self):
        """The store is the record of what is indexed: hand-added papers were in no manifest."""
        with tempfile.TemporaryDirectory() as d:
            json.dump({"a.pdf": "t"}, open(os.path.join(d, "ingested.json"), "w"))
            json.dump({"labels": {"a.pdf": "A", "handadded.pdf": "H"}}, open(os.path.join(d, "migration_labels.json"), "w"))
            pdfs, _ = mig.old_document_keys(d)
        self.assertEqual(pdfs, ["a.pdf", "handadded.pdf"])


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


class TestLabelPrecedence(unittest.TestCase):
    def test_the_old_index_label_beats_downloaded_json_beats_the_file_name(self):
        with tempfile.TemporaryDirectory() as d:
            json.dump({"k": {"key": "k", "path": "/vm/data/pulled_pdfs/both.pdf", "provider": "u",
                             "fetched_at": "2026-09-10T20:05:57Z", "title": "Nat. Phys"},
                       "j": {"key": "j", "path": "/vm/data/pulled_pdfs/manifest_only.pdf", "provider": "u",
                             "fetched_at": "2026-09-10T20:05:57Z", "title": "Manifest Title"}},
                      open(os.path.join(d, "downloaded.json"), "w"))
            json.dump({"labels": {"both.pdf": "The Real Title", "store_only.pdf": "Store Title"}},
                      open(os.path.join(d, "migration_labels.json"), "w"))
            by_key = mig.label_map(d)
            got = mig.labeled(["/x/Both.pdf", "/x/manifest_only.pdf", "/x/store_only.pdf", "/x/neither.pdf"], by_key)
        self.assertEqual(list(got.values()), ["The Real Title", "Manifest Title", "Store Title", "neither"])

    def test_the_label_source_is_counted(self):
        with tempfile.TemporaryDirectory() as d:
            json.dump({"labels": {"a.pdf": "A"}}, open(os.path.join(d, "migration_labels.json"), "w"))
            self.assertEqual(mig.exported_labels(d), {"a.pdf": "A"})
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(mig.exported_labels(d), {})


class TestLabels(unittest.TestCase):
    """A paper's label becomes citation_source on every chunk and so the cite key:
    it must come from the SOURCE data directory, not the (possibly empty) target."""

    def _source(self, d, records):
        json.dump(records, open(os.path.join(d, "downloaded.json"), "w"))

    def test_title_from_the_source_manifest_by_file_name_whatever_the_recorded_directory(self):
        with tempfile.TemporaryDirectory() as d:
            self._source(d, {"doi:10.1/x": {"key": "doi:10.1/x", "path": "/home/user/data/pulled_pdfs/doi_10.1_x.pdf",
                                            "provider": "unpaywall", "fetched_at": "2026-09-10T20:05:57Z",
                                            "title": "A Real Title"}})
            by_base = mig.label_map(d)
        self.assertEqual(by_base["doi_10.1_x.pdf"], "A Real Title")
        self.assertEqual(mig.labeled(["/elsewhere/doi_10.1_x.pdf", "/elsewhere/seed.pdf"], by_base),
                         {"/elsewhere/doi_10.1_x.pdf": "A Real Title", "/elsewhere/seed.pdf": "seed"})

    def test_no_manifest_means_filename_stems(self):
        with tempfile.TemporaryDirectory() as d:
            by_base = mig.label_map(d)
        self.assertEqual(by_base, {})
        self.assertEqual(mig.labeled(["/x/a.b.pdf"], by_base), {"/x/a.b.pdf": "a.b"})


class TestTeiCache(unittest.TestCase):
    def test_finds_any_suffix_for_the_pdfs_own_file_stem(self):
        with tempfile.TemporaryDirectory() as d:
            _touch(os.path.join(d, "My Paper.fulltext.tei.xml"))
            self.assertTrue(mig.tei_for(d, "/x/My Paper.pdf").endswith("My Paper.fulltext.tei.xml"))
            self.assertIsNone(mig.tei_for(d, "/x/other.pdf"))

    def test_seeding_copies_only_what_is_missing_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            for n in ("a", "b"):
                open(os.path.join(src, f"{n}.grobid.tei.xml"), "w").write("source")
            open(os.path.join(dst, "b.grobid.tei.xml"), "w").write("keep me")
            copied, had, uncached = mig.seed_tei_cache(["/x/a.pdf", "/x/b.pdf", "/x/c.pdf"], src, dst)
            self.assertEqual((copied, had, uncached), (1, 1, 1))
            self.assertEqual(open(os.path.join(dst, "a.grobid.tei.xml")).read(), "source")
            self.assertEqual(open(os.path.join(dst, "b.grobid.tei.xml")).read(), "keep me")

    def test_a_tei_with_figures_but_no_boxes_is_not_copied_or_counted(self):
        stale = b'<TEI><figure xml:id="fig_0"><head>Fig</head></figure></TEI>'
        usable = b'<TEI><figure xml:id="fig_0" coords="1,2,3,4,5"><head>Fig</head></figure></TEI>'
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            open(os.path.join(src, "old.grobid.tei.xml"), "wb").write(stale)
            open(os.path.join(src, "new.grobid.tei.xml"), "wb").write(usable)
            copied, had, uncached = mig.seed_tei_cache(["/x/old.pdf", "/x/new.pdf"], src, dst)
            self.assertEqual((copied, had, uncached), (1, 0, 1))
            self.assertEqual(os.listdir(dst), ["new.grobid.tei.xml"])

    def test_same_source_and_target_copies_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            _touch(os.path.join(d, "a.grobid.tei.xml"))
            self.assertEqual(mig.seed_tei_cache(["/x/a.pdf", "/x/c.pdf"], d, d), (0, 1, 1))


class TestStaleIndex(unittest.TestCase):
    def test_a_missing_index_or_one_older_than_the_manifest_is_stale(self):
        with tempfile.TemporaryDirectory() as d:
            bm25, man = os.path.join(d, "bm25.pkl"), os.path.join(d, "ingested_v3.json")
            self.assertFalse(mig.bm25_is_stale(bm25, man))          # nothing ingested yet
            _touch(man)
            self.assertTrue(mig.bm25_is_stale(bm25, man))           # papers, no index: a hard kill
            _touch(bm25)
            os.utime(man, (os.path.getmtime(bm25) + 10,) * 2)
            self.assertTrue(mig.bm25_is_stale(bm25, man))
            os.utime(man, (os.path.getmtime(bm25) - 10,) * 2)
            self.assertFalse(mig.bm25_is_stale(bm25, man))


class TestVersionAndSummaryChecks(unittest.TestCase):
    def test_ollama_older_than_the_embedder_needs_is_a_problem(self):
        self.assertIsNone(mig.ollama_version_problem("0.40.5"))
        self.assertIsNone(mig.ollama_version_problem("0.36.0"))
        self.assertIn("0.36", mig.ollama_version_problem("0.35.9"))
        self.assertIn("version", mig.ollama_version_problem("garbage"))

    def test_a_summary_model_that_does_not_answer_is_a_problem(self):
        def broken(*a, **k):
            raise RuntimeError("model failed to load")
        self.assertIn("model failed to load", mig.summary_model_problem(broken))
        self.assertIsNone(mig.summary_model_problem(lambda *a, **k: types.SimpleNamespace(content="ok")))
        self.assertIn("empty", mig.summary_model_problem(lambda *a, **k: types.SimpleNamespace(content=" ")))


class TestVerifyCounts(unittest.TestCase):
    def _meta(self, doc, type_):
        return {"document": doc, "type": type_}

    def test_reports_what_is_missing_and_what_is_only_worth_a_note(self):
        metas = ([self._meta("a.pdf", "text_chunk")] * 3 + [self._meta("a.pdf", "caption")] * 2
                 + [self._meta("a.pdf", "figure_image")] * 2
                 + [self._meta("b.pdf", "text_chunk")]
                 + [self._meta("c.pdf", "caption")])             # c: a caption, no text, no image
        r = mig.verify_counts(metas, manifest_keys={"a.pdf", "b.pdf", "c.pdf", "d.pdf"},
                              summary_docs={"a.pdf", "b.pdf"}, expected_keys={"a.pdf", "b.pdf", "c.pdf", "d.pdf", "e.pdf"})
        self.assertEqual(r["types"], {"text_chunk": 4, "caption": 3, "figure_image": 2})
        self.assertEqual(r["not_migrated"], ["e.pdf"])
        self.assertEqual(r["no_summary"], ["c.pdf", "d.pdf"])
        self.assertEqual(r["no_text"], ["c.pdf", "d.pdf"])
        self.assertEqual(r["figures_without_image"], 1)
        self.assertEqual(r["papers_with_chunks"], 3)

    def test_a_complete_migration_has_no_problems(self):
        metas = [self._meta("a.pdf", "text_chunk"), self._meta("a.pdf", "caption"), self._meta("a.pdf", "figure_image")]
        r = mig.verify_counts(metas, {"a.pdf"}, {"a.pdf"}, {"a.pdf"})
        self.assertEqual((r["not_migrated"], r["no_summary"], r["no_text"], r["figures_without_image"]), ([], [], [], 0))


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
