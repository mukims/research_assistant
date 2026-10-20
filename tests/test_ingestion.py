"""Tests for the shared batch-ingestion bookkeeping.

These cover the check/mark contract rather than the parsing itself. Parsing is
the expensive part of the pipeline (GROBID per paper plus an image embedding
per figure), so the rules about *when* it is skipped are what protect the
runtime — and they were previously implemented four separate times, with
different answers in each.
"""

import concurrent.futures
import os
import tempfile
import unittest
from unittest.mock import patch

import research_assistant.shared.ingestion as ing
import research_assistant.shared.manifest as manifest_mod


def _touch(directory, name):
    d = getattr(directory, "name", directory)
    path = os.path.join(d, name)
    with open(path, "wb") as f:
        f.write(b"%PDF-1.4 fake")
    return path


class TestPdfKey(unittest.TestCase):
    def test_normalises_case_spaces_and_path(self):
        """The key is the basename, lowercased, with spaces collapsed to underscores."""
        self.assertEqual(ing.pdf_key("/a/b/My Paper.PDF"), "my_paper.pdf")
        self.assertEqual(ing.pdf_key("Gabbett et al sub to nat mat.pdf"),
                         "gabbett_et_al_sub_to_nat_mat.pdf")

    def test_is_stable_across_equivalent_paths(self):
        """Relative and absolute paths to the same file share one identity."""
        self.assertEqual(ing.pdf_key("pulled_pdfs/x.pdf"), ing.pdf_key("/tmp/pulled_pdfs/x.pdf"))


class ManifestBackedTestCase(unittest.TestCase):
    """Base for tests that exercise the (real) JSON manifest, directly or via
    ingest_pdfs()'s manifest.add_many() call.

    Isolates each test's manifest to a temp file so tests never read or write
    the process's real manifest path.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(
            manifest_mod, "MANIFEST_PATH", os.path.join(self.tmp.name, "ingested.json")
        )
        patcher.start()
        self.addCleanup(patcher.stop)


class IngestPdfsCase(ManifestBackedTestCase):
    """Base class wiring up the expensive collaborators as recording fakes."""

    def setUp(self):
        super().setUp()

        self.processed = []      # paths handed to process_pdf
        self.rebuilds = 0
        self.ingested = set()    # what get_ingested_documents reports
        self.corpus_per_pdf = 1  # entries each process_pdf call returns
        self.inserted = 0        # what upsert_corpus reports

        def fake_process(path, label, *a, **k):
            self.processed.append(path)
            return [{"content": f"{label}-{i}"} for i in range(self.corpus_per_pdf)]

        def fake_upsert(corpus):
            return self.inserted

        def fake_rebuild():
            self.rebuilds += 1

        # A BM25 path that does not exist, so the "index missing" branch is
        # exercised only where a test intends it.
        self.bm25_path = os.path.join(self.tmp.name, "bm25.pkl")

        # mark_document_ingested no longer exists — ingest_pdfs() now marks
        # via manifest.add_many() directly, against the temp manifest this
        # base class already isolates. Tests assert on manifest_mod.load().
        for target, repl in [
            ("process_pdf", fake_process),
            ("upsert_corpus", fake_upsert),
            ("rebuild_bm25", fake_rebuild),
            ("get_ingested_documents", lambda *a, **k: set(self.ingested)),
        ]:
            patcher = patch.object(ing, target, repl)
            patcher.start()
            self.addCleanup(patcher.stop)

        patcher = patch.object(ing, "BM25_INDEX_PATH", self.bm25_path)
        patcher.start()
        self.addCleanup(patcher.stop)


class TestSkipAndMark(IngestPdfsCase):
    def test_already_ingested_pdf_is_not_reparsed(self):
        """The whole point of the manifest: never parse the same paper twice."""
        pdf = _touch(self.tmp.name, "paper.pdf")
        self.ingested = {"paper.pdf"}

        result = ing.ingest_pdfs([pdf])

        self.assertEqual(self.processed, [])
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(result["processed"], 0)

    def test_force_reingest_bypasses_the_skip(self):
        """skip_ingested=False is how Agent 3's --force re-runs everything."""
        pdf = _touch(self.tmp.name, "paper.pdf")
        self.ingested = {"paper.pdf"}

        ing.ingest_pdfs([pdf], skip_ingested=False)

        self.assertEqual(self.processed, [pdf])

    def test_pdf_yielding_no_content_is_still_marked(self):
        """A corrupt or duplicate paper must not be retried on every future run.

        This is the regression that made the orchestrator re-parse the same
        seven unreadable PDFs on every startup: they produced no chunks, so
        nothing recorded them as seen.
        """
        pdf = _touch(self.tmp.name, "empty.pdf")
        self.corpus_per_pdf = 0

        ing.ingest_pdfs([pdf])

        self.assertEqual(list(manifest_mod.load()), ["empty.pdf"])

    def test_every_processed_pdf_is_marked(self):
        a, b = _touch(self.tmp.name, "a.pdf"), _touch(self.tmp.name, "b.pdf")
        ing.ingest_pdfs([a, b])
        self.assertEqual(sorted(manifest_mod.load()), ["a.pdf", "b.pdf"])

    def test_missing_file_is_reported_not_marked(self):
        """A path that does not exist is a failure, not a silently-seen document."""
        result = ing.ingest_pdfs([os.path.join(self.tmp.name, "nope.pdf")])

        self.assertEqual(result["failed"], [os.path.join(self.tmp.name, "nope.pdf")])
        self.assertEqual(self.processed, [])
        self.assertEqual(manifest_mod.load(), {})

    def test_crashed_worker_is_reported_not_marked(self):
        """Finding I1: a path whose worker raised must land in result["failed"]
        (as it already did) but must NOT be marked in the manifest — marking
        it hides the crash forever, since only --force (which reprocesses
        everything) would ever revisit it. A sibling PDF that succeeded in
        the same batch must still be marked normally.

        ProcessPoolExecutor is swapped for ThreadPoolExecutor here purely so
        the fake process function (a closure over test state) does not need
        to be pickled across a process boundary — the control flow in
        ingest_pdfs() being exercised is identical either way.
        """
        good = _touch(self.tmp.name, "good.pdf")
        bad = _touch(self.tmp.name, "bad.pdf")

        def fake_process(path, label, *a, **k):
            if path == bad:
                raise RuntimeError("worker crashed")
            return [{"content": "ok"}]

        with patch.object(ing, "process_pdf", fake_process), \
             patch.object(ing.concurrent.futures, "ProcessPoolExecutor",
                          concurrent.futures.ThreadPoolExecutor):
            result = ing.ingest_pdfs([good, bad], workers=2)

        self.assertIn(bad, result["failed"])
        marked = manifest_mod.load()
        self.assertIn(ing.pdf_key(good), marked)
        self.assertNotIn(ing.pdf_key(bad), marked)


class TestIndexRebuild(IngestPdfsCase):
    def test_batch_rebuilds_index_once_not_per_file(self):
        """Rebuilding re-tokenises the whole collection, so per-file is quadratic."""
        pdfs = [_touch(self.tmp.name, f"p{i}.pdf") for i in range(5)]
        self.inserted = 10

        ing.ingest_pdfs(pdfs)

        self.assertEqual(len(self.processed), 5)
        self.assertEqual(self.rebuilds, 1)

    def test_no_rebuild_when_nothing_was_inserted(self):
        """An all-duplicate batch leaves the collection unchanged."""
        open(self.bm25_path, "wb").close()   # index already exists
        pdf = _touch(self.tmp.name, "dup.pdf")
        self.inserted = 0

        ing.ingest_pdfs([pdf])

        self.assertEqual(self.rebuilds, 0)

    def test_rebuild_when_index_is_missing_even_if_nothing_inserted(self):
        """Search needs the index to exist at all, duplicates notwithstanding."""
        pdf = _touch(self.tmp.name, "dup.pdf")
        self.inserted = 0

        ing.ingest_pdfs([pdf])

        self.assertEqual(self.rebuilds, 1)

    def test_rebuild_can_be_deferred_by_the_caller(self):
        pdf = _touch(self.tmp.name, "p.pdf")
        self.inserted = 3

        ing.ingest_pdfs([pdf], rebuild_index=False)

        self.assertEqual(self.rebuilds, 0)


class TestInputForms(IngestPdfsCase):
    def test_list_input_labels_documents_by_filename_stem(self):
        """Passing a bare list is how the orchestrator syncs pulled_pdfs/."""
        pdf = _touch(self.tmp.name, "Some_Paper_12.pdf")

        with patch.object(ing, "upsert_corpus", side_effect=lambda c: len(c)) as upsert:
            ing.ingest_pdfs([pdf])

        corpus = upsert.call_args[0][0]
        self.assertEqual(corpus[0]["content"], "Some_Paper_12-0")

    def test_dict_input_preserves_the_citation_label(self):
        """Agent 3 passes downloaded.json, whose values are full citation strings."""
        pdf = _touch(self.tmp.name, "x.pdf")

        with patch.object(ing, "upsert_corpus", side_effect=lambda c: len(c)) as upsert:
            ing.ingest_pdfs({pdf: "Novoselov et al., Science 306, 666 (2004)"})

        corpus = upsert.call_args[0][0]
        self.assertTrue(corpus[0]["content"].startswith("Novoselov et al."))

    def test_empty_input_is_a_no_op(self):
        result = ing.ingest_pdfs([])
        self.assertEqual(result["processed"], 0)
        self.assertEqual(self.rebuilds, 0)


class TestMarkingUsesTheManifest(ManifestBackedTestCase):
    def test_every_attempted_pdf_is_marked_even_when_it_yields_nothing(self):
        """A corrupt or empty PDF must still be marked, or it is re-parsed forever."""
        pdf = _touch(self.tmp.name, "empty.pdf")
        with patch.object(ing, "process_pdf", return_value=[]), \
             patch.object(ing, "rebuild_bm25"):
            ing.ingest_pdfs({pdf: "label"}, rebuild_index=False)
        self.assertIn(ing.pdf_key(pdf), manifest_mod.load())


class TestIngestReportsExtractionModes(IngestPdfsCase):
    def test_result_counts_documents_by_extraction_mode(self):
        good = _touch(self.tmp.name, "good.pdf")
        scan = _touch(self.tmp.name, "scan.pdf")

        def fake_process(path, label, *a, **k):
            mode = "pymupdf" if "scan" in path else "grobid"
            return [{"content": "x", "document": ing.pdf_key(path), "extraction": mode}]

        with patch.object(ing, "process_pdf", fake_process):
            result = ing.ingest_pdfs([good, scan], rebuild_index=False)

        self.assertEqual(result["extraction"], {"grobid": 1, "pymupdf": 1})

    def test_every_page_of_one_document_counts_once(self):
        """The tally is per document, not per corpus entry."""
        pdf = _touch(self.tmp.name, "many_pages.pdf")

        def fake_process(path, label, *a, **k):
            key = ing.pdf_key(path)
            return [
                {"content": f"page-{i}", "document": key, "extraction": "pymupdf"}
                for i in range(5)
            ]

        with patch.object(ing, "process_pdf", fake_process):
            result = ing.ingest_pdfs([pdf], rebuild_index=False)

        self.assertEqual(result["extraction"], {"grobid": 0, "pymupdf": 1})


class TestConcurrentIngestion(IngestPdfsCase):
    """Two ingests must never touch the corpus at the same time.

    watch.py runs the raw/ pipeline on one thread while its pulled_pdfs/
    watcher drains on another — and Agent 2 downloads *into* pulled_pdfs/, so
    a fetch reliably arms that second thread mid-run. Both paths reach
    upsert_corpus(), which allocates chunk ids by reading the collection's
    current maximum and counting up. Overlapping readers get the same maximum
    and mint the same ids; whichever add() loses is dropped, and both then
    race to rewrite the one BM25 pickle.
    """

    def setUp(self):
        super().setUp()
        patcher = patch.object(
            ing, "INGEST_LOCK_PATH", os.path.join(self.tmp.name, "ingest.lock")
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_second_ingest_waits_for_the_first(self):
        import threading
        import time

        events = []

        def slow_upsert(corpus):
            events.append("enter")
            time.sleep(0.05)
            events.append("exit")
            return 0

        with patch.object(ing, "upsert_corpus", slow_upsert):
            a = _touch(self.tmp.name, "a.pdf")
            b = _touch(self.tmp.name, "b.pdf")
            threads = [
                threading.Thread(target=ing.ingest_pdfs, args=({p: p},))
                for p in (a, b)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.assertEqual(
            events, ["enter", "exit", "enter", "exit"],
            "ingests overlapped — chunk-id allocation and the BM25 rebuild raced",
        )


class TestProcessPdfIsTheV2Processor(ManifestBackedTestCase):
    """There is one ingest path: process_pdf delegates to ingest_v2, and the
    result reports how each document was extracted and how many figure images
    were embedded."""

    def _entries(self, path, label, images_dir=None):
        key = ing.pdf_key(path)
        base = {"document": key, "citation": label, "page": 0, "extraction": "grobid"}
        return [
            {**base, "type": "text_chunk", "content": "c" * 300, "embed_text": "h c", "meta": {"seq": 0}},
            {**base, "type": "figure_image", "content": "Image of Figure 1: x", "embedding": [0.1],
             "meta": {"extraction": "image"}},
        ]

    def test_process_pdf_delegates_and_takes_no_figure_switch(self):
        with patch("research_assistant.shared.ingest_v2.process_pdf_v2", return_value=[]) as v2:
            ing.process_pdf("/x/a.pdf", "A")
        v2.assert_called_once_with("/x/a.pdf", "A", images_dir=None)

    def test_result_reports_modes_and_embedded_figure_images(self):
        pdf = _touch(self.tmp, "a.pdf")
        with patch("research_assistant.shared.ingest_v2.process_pdf_v2", side_effect=self._entries), \
             patch.object(ing, "upsert_corpus", return_value=2), patch.object(ing, "rebuild_bm25"):
            result = ing.ingest_pdfs({pdf: "A"})
        self.assertEqual(result["extraction"], {"grobid": 1, "pymupdf": 0})
        self.assertEqual(result["figure_images"], 1)
        self.assertNotIn("described", result)

    def test_a_pymupdf_fallback_is_warned_about(self):
        pdf = _touch(self.tmp, "a.pdf")

        def pymupdf_entries(path, label, images_dir=None):
            return [{**self._entries(path, label)[0], "extraction": "pymupdf"}]

        with patch("research_assistant.shared.ingest_v2.process_pdf_v2", side_effect=pymupdf_entries), \
             patch.object(ing, "upsert_corpus", return_value=1), patch.object(ing, "rebuild_bm25"), \
             self.assertLogs("ingestion", level="WARNING") as logs:
            ing.ingest_pdfs({pdf: "A"})
        self.assertTrue(any("fell back to PyMuPDF" in m for m in logs.output), logs.output)


if __name__ == "__main__":
    unittest.main()
