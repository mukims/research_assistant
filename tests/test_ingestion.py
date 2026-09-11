"""Tests for the shared batch-ingestion bookkeeping.

These cover the check/mark contract rather than the parsing itself. Parsing is
the expensive part of the pipeline (layout detection per page plus a VLM call
per figure), so the rules about *when* it is skipped are what protect the
runtime — and they were previously implemented four separate times, with
different answers in each.
"""

import concurrent.futures
import os
import tempfile
import types
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


class TestProcessPdfFallback(unittest.TestCase):
    """Tests verifying process_pdf() robustly falls back to text-only extraction."""

    def test_process_pdf_falls_back_when_layout_detection_disabled(self):
        with patch.object(ing, "LAYOUT_DETECTION", False), \
             patch.object(ing, "_extract_text_only", return_value=[{"type": "text", "content": "extracted"}]) as mock_extract, \
             patch.object(ing, "_process_pdf_layout") as mock_layout:
            result = ing.process_pdf("paper.pdf", "citation_label")

        mock_extract.assert_called_once_with("paper.pdf", "citation_label")
        mock_layout.assert_not_called()
        self.assertEqual(result, [{"type": "text", "content": "extracted", "extraction": "text_only"}])

    def test_process_pdf_falls_back_on_import_error(self):
        with patch.object(ing, "LAYOUT_DETECTION", True), \
             patch.object(ing, "_process_pdf_layout", side_effect=ImportError("No module named 'layoutparser'")), \
             patch.object(ing, "_extract_text_only", return_value=[{"type": "text", "content": "fallback_text"}]) as mock_extract:
            result = ing.process_pdf("paper.pdf", "citation_label")

        mock_extract.assert_called_once_with("paper.pdf", "citation_label")
        self.assertEqual(result, [{"type": "text", "content": "fallback_text", "extraction": "text_only"}])

    def test_process_pdf_falls_back_on_yaml_scanner_error(self):
        try:
            from yaml.scanner import ScannerError
            yaml_err = ScannerError(None, None, "mapping values are not allowed here", None)
        except ImportError:
            yaml_err = Exception("mapping values are not allowed here")

        with patch.object(ing, "LAYOUT_DETECTION", True), \
             patch.object(ing, "_process_pdf_layout", side_effect=yaml_err), \
             patch.object(ing, "_extract_text_only", return_value=[{"type": "text", "content": "recovered_content"}]) as mock_extract:
            result = ing.process_pdf("doi_10.1038_paper.pdf", "doi_label")

        mock_extract.assert_called_once_with("doi_10.1038_paper.pdf", "doi_label")
        self.assertEqual(result, [{"type": "text", "content": "recovered_content", "extraction": "text_only"}])

    def test_process_pdf_falls_back_on_layout_runtime_error(self):
        with patch.object(ing, "LAYOUT_DETECTION", True), \
             patch.object(ing, "_process_pdf_layout", side_effect=RuntimeError("CUDA out of memory in detect()")), \
             patch.object(ing, "_extract_text_only", return_value=[{"type": "text", "content": "text_after_oom"}]) as mock_extract:
            result = ing.process_pdf("large_paper.pdf", "citation_label")

        mock_extract.assert_called_once_with("large_paper.pdf", "citation_label")
        self.assertEqual(result, [{"type": "text", "content": "text_after_oom", "extraction": "text_only"}])


    def test_process_pdf_passes_detectron_config(self):
        with patch.object(ing, "LAYOUT_DETECTION", True), \
             patch.object(ing, "_process_pdf_layout", return_value=[{"type": "text"}]) as mock_layout:
            result = ing.process_pdf("paper.pdf", "citation_label", detectron_config="/custom/config.yaml")

        mock_layout.assert_called_once_with(
            "paper.pdf", "citation_label", None, None, detectron_config="/custom/config.yaml"
        )
        self.assertEqual(result, [{"type": "text", "extraction": "layout"}])

    def test_extract_text_only_tolerates_corrupt_page(self):
        fake_pdf = [unittest.mock.MagicMock(), unittest.mock.MagicMock()]
        # Page 0 raises error on get_text
        fake_pdf[0].get_text.side_effect = RuntimeError("Corrupted page stream")
        # Page 1 succeeds
        fake_pdf[1].get_text.return_value = [
            (0, 0, 100, 100, "Valid text on page 2", 0, 0)
        ]

        fake_fitz = unittest.mock.MagicMock()
        fake_fitz.open.return_value = fake_pdf

        with patch.dict("sys.modules", {"fitz": fake_fitz}):
            result = ing._extract_text_only("corrupt_page.pdf", "citation")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["page"], 1)
        self.assertEqual(result[0]["content"], "Valid text on page 2")


class TestDetectronModelLoading(unittest.TestCase):
    """Tests for _get_detectron_model error handling, caching, and fallback."""

    def setUp(self):
        self._orig_cache = dict(ing._detectron_model_cache)
        ing._detectron_model_cache.clear()

    def tearDown(self):
        ing._detectron_model_cache.clear()
        ing._detectron_model_cache.update(self._orig_cache)

    def test_a_failing_config_is_not_retried_for_every_document(self):
        """A load that cannot succeed must be attempted once, not per PDF.

        process_pdf swallows the failure into a text-only fallback, so nothing
        upstream stops the next document trying again. Without remembering the
        failure, a broken config makes every paper in the batch pay for a full
        Detectron2 construction that is guaranteed to fail.
        """
        attempts = []

        def boom(*a, **k):
            attempts.append(1)
            raise RuntimeError("bad config")

        fake_lp = types.SimpleNamespace(Detectron2LayoutModel=boom)
        with patch.dict("sys.modules", {"layoutparser": fake_lp}):
            with self.assertRaises(RuntimeError):
                ing._get_detectron_model("w.pth", "/nonexistent/config.yaml")
            after_first_call = len(attempts)

            for _ in range(3):
                with self.assertRaises(RuntimeError):
                    ing._get_detectron_model("w.pth", "/nonexistent/config.yaml")

        self.assertEqual(
            len(attempts), after_first_call,
            "a failed model load must be remembered, not retried for every document",
        )

    def test_get_detectron_model_uses_cache(self):
        fake_model = object()
        ing._detectron_model_cache[("weights.pth", "config.yaml")] = fake_model

        model = ing._get_detectron_model("weights.pth", "config.yaml")
        self.assertIs(model, fake_model)

    def test_get_detectron_model_falls_back_to_local_config_on_remote_error(self):
        try:
            from yaml.scanner import ScannerError
            remote_err = ScannerError(None, None, "mapping values are not allowed here", None)
        except ImportError:
            remote_err = Exception("mapping values are not allowed here")

        calls = []
        fake_lp_model = object()

        def fake_detectron2_init(config_path, model_path, extra_config, label_map):
            calls.append(config_path)
            if config_path.startswith("lp://"):
                raise remote_err
            return fake_lp_model

        fake_lp = unittest.mock.MagicMock()
        fake_lp.Detectron2LayoutModel.side_effect = fake_detectron2_init

        with patch.dict("sys.modules", {"layoutparser": fake_lp}), \
             patch("os.path.exists", return_value=True):
            model = ing._get_detectron_model(
                weights_path="model.pth",
                config_path="lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config",
            )

        self.assertIs(model, fake_lp_model)
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0].startswith("lp://"))
        self.assertTrue(calls[1].endswith("publaynet_config.yaml"))

    def test_get_detectron_model_raises_if_fallback_fails_too(self):
        fake_lp = unittest.mock.MagicMock()
        fake_lp.Detectron2LayoutModel.side_effect = RuntimeError("Fatal model init error")

        with patch.dict("sys.modules", {"layoutparser": fake_lp}), \
             patch("os.path.exists", return_value=True):
            with self.assertRaises(RuntimeError):
                ing._get_detectron_model(
                    weights_path="model.pth",
                    config_path="lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config",
                )

    def test_get_detectron_model_does_not_loop_if_local_config_fails(self):
        calls = []
        fake_lp = unittest.mock.MagicMock()

        def fake_detectron2_init(config_path, model_path, extra_config, label_map):
            calls.append(config_path)
            raise ValueError("Invalid local config")

        fake_lp.Detectron2LayoutModel.side_effect = fake_detectron2_init

        local_cfg = os.path.join(ing.PROJECT_ROOT, "publaynet_config.yaml")
        with patch.dict("sys.modules", {"layoutparser": fake_lp}), \
             patch("os.path.exists", return_value=True):
            with self.assertRaises(ValueError):
                ing._get_detectron_model(weights_path="model.pth", config_path=local_cfg)

        self.assertEqual(len(calls), 1)

    def test_get_detectron_model_does_not_loop_on_relative_path_to_local_config(self):
        calls = []
        fake_lp = unittest.mock.MagicMock()

        def fake_detectron2_init(config_path, model_path, extra_config, label_map):
            calls.append(config_path)
            raise ValueError("Corrupt local config")

        fake_lp.Detectron2LayoutModel.side_effect = fake_detectron2_init

        # Relative path referring to the same local publaynet_config.yaml
        with patch.dict("sys.modules", {"layoutparser": fake_lp}), \
             patch("os.path.exists", return_value=True):
            with self.assertRaises(ValueError):
                ing._get_detectron_model(weights_path="model.pth", config_path="publaynet_config.yaml")

        self.assertEqual(len(calls), 1)

    def test_get_detectron_model_defaults_weights_to_detectron_weights_if_present(self):
        calls = []
        fake_lp_model = object()

        def fake_detectron2_init(config_path, model_path, extra_config, label_map):
            calls.append((config_path, model_path))
            return fake_lp_model

        fake_lp = unittest.mock.MagicMock()
        fake_lp.Detectron2LayoutModel.side_effect = fake_detectron2_init

        with patch.dict("sys.modules", {"layoutparser": fake_lp}), \
             patch.object(ing, "DETECTRON_WEIGHTS", "/repo/model_final.pth"), \
             patch("os.path.exists", lambda p: p == "/repo/model_final.pth"):
            model = ing._get_detectron_model(weights_path=None, config_path="config.yaml")

        self.assertIs(model, fake_lp_model)
        self.assertEqual(calls[0][1], "/repo/model_final.pth")

    def test_get_detectron_model_custom_config_does_not_poison_default_cache(self):
        fake_lp = unittest.mock.MagicMock()
        fake_custom_model = object()
        fake_lp.Detectron2LayoutModel.return_value = fake_custom_model

        with patch.dict("sys.modules", {"layoutparser": fake_lp}), \
             patch.object(ing, "DETECTRON_CONFIG", "default_config.yaml"):
            model = ing._get_detectron_model("shared_weights.pth", "custom_config.yaml")

        self.assertIs(model, fake_custom_model)
        # Verify default weights key was NOT poisoned with custom config model
        self.assertNotIn("shared_weights.pth", ing._detectron_model_cache)

    def test_get_detectron_model_fallback_caches_local_config_key(self):
        try:
            from yaml.scanner import ScannerError
            remote_err = ScannerError(None, None, "bad yaml", None)
        except ImportError:
            remote_err = Exception("bad yaml")

        fake_lp_model = object()

        def fake_detectron2_init(config_path, model_path, extra_config, label_map):
            if config_path.startswith("lp://"):
                raise remote_err
            return fake_lp_model

        fake_lp = unittest.mock.MagicMock()
        fake_lp.Detectron2LayoutModel.side_effect = fake_detectron2_init

        local_cfg = os.path.join(ing.PROJECT_ROOT, "publaynet_config.yaml")
        with patch.dict("sys.modules", {"layoutparser": fake_lp}), \
             patch("os.path.exists", return_value=True):
            model = ing._get_detectron_model(
                weights_path="model.pth",
                config_path="lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config",
            )

        self.assertIs(model, fake_lp_model)
        # Both the original requested remote config and the resolved local config are cached
        self.assertIn(("model.pth", "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config"), ing._detectron_model_cache)
        self.assertIn(("model.pth", local_cfg), ing._detectron_model_cache)


class TestDetectronConfigResolution(unittest.TestCase):
    """Tests for DETECTRON_CONFIG preference of local file over remote."""

    def test_prefers_local_publaynet_config_when_present(self):
        import importlib
        import research_assistant.config as conf

        with patch("os.path.exists", lambda path: path.endswith("publaynet_config.yaml")), \
             patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CITATION_DETECTRON_CONFIG", None)
            importlib.reload(conf)
            self.assertTrue(
                conf.DETECTRON_CONFIG.endswith("publaynet_config.yaml"),
                f"Expected publaynet_config.yaml, got {conf.DETECTRON_CONFIG}",
            )

    def test_respects_env_var_override(self):
        import importlib
        import research_assistant.config as conf

        with patch.dict(os.environ, {"CITATION_DETECTRON_CONFIG": "/custom/path/config.yaml"}):
            importlib.reload(conf)
            self.assertEqual(conf.DETECTRON_CONFIG, "/custom/path/config.yaml")

    def test_empty_env_var_falls_back_to_local_config(self):
        import importlib
        import research_assistant.config as conf

        with patch("os.path.exists", lambda path: path.endswith("publaynet_config.yaml")), \
             patch.dict(os.environ, {"CITATION_DETECTRON_CONFIG": ""}):
            importlib.reload(conf)
            self.assertTrue(
                conf.DETECTRON_CONFIG.endswith("publaynet_config.yaml"),
                f"Expected publaynet_config.yaml, got {conf.DETECTRON_CONFIG}",
            )

    def test_whitespace_env_var_falls_back_to_local_config(self):
        import importlib
        import research_assistant.config as conf

        with patch("os.path.exists", lambda path: path.endswith("publaynet_config.yaml")), \
             patch.dict(os.environ, {"CITATION_DETECTRON_CONFIG": "   \t\n"}):
            importlib.reload(conf)
            self.assertTrue(
                conf.DETECTRON_CONFIG.endswith("publaynet_config.yaml"),
                f"Expected publaynet_config.yaml, got {conf.DETECTRON_CONFIG}",
            )

    def test_defaults_to_remote_when_local_file_absent(self):
        import importlib
        import research_assistant.config as conf

        with patch("os.path.exists", return_value=False), \
             patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CITATION_DETECTRON_CONFIG", None)
            importlib.reload(conf)
            self.assertTrue(
                conf.DETECTRON_CONFIG.startswith("lp://"),
                f"Expected lp://..., got {conf.DETECTRON_CONFIG}",
            )
        # Reload once more to restore original config state
        importlib.reload(conf)


class TestMarkingUsesTheManifest(ManifestBackedTestCase):
    def test_every_attempted_pdf_is_marked_even_when_it_yields_nothing(self):
        """A corrupt or empty PDF must still be marked, or it is re-parsed forever."""
        pdf = _touch(self.tmp.name, "empty.pdf")
        with patch.object(ing, "process_pdf", return_value=[]), \
             patch.object(ing, "rebuild_bm25"):
            ing.ingest_pdfs({pdf: "label"}, rebuild_index=False)
        self.assertIn(ing.pdf_key(pdf), manifest_mod.load())


class TestExtractionModeIsRecorded(unittest.TestCase):
    """A silent degradation to text-only has to be visible in the result.

    Layout detection failing costs every figure and table in the corpus, but
    the fallback is per-PDF and only logs a warning — so a run can quietly
    produce a corpus with no figures at all. Agent 1 records its
    GROBID-vs-regex fallback per run under `extraction` for exactly this
    reason; ingestion needs the same.
    """

    def test_layout_extraction_tags_its_entries(self):
        with patch.object(ing, "LAYOUT_DETECTION", True), \
             patch.object(ing, "_process_pdf_layout",
                          return_value=[{"type": "text", "content": "x"}]):
            corpus = ing.process_pdf("paper.pdf", "label")

        self.assertEqual(corpus[0]["extraction"], "layout")

    def test_text_only_fallback_tags_its_entries(self):
        """The tag is what makes an unexpected fallback countable downstream."""
        with patch.object(ing, "LAYOUT_DETECTION", True), \
             patch.object(ing, "_process_pdf_layout",
                          side_effect=RuntimeError("CUDA out of memory in detect()")), \
             patch.object(ing, "_extract_text_only",
                          return_value=[{"type": "text", "content": "x"}]):
            corpus = ing.process_pdf("paper.pdf", "label")

        self.assertEqual(corpus[0]["extraction"], "text_only")


class TestIngestReportsExtractionModes(IngestPdfsCase):
    def test_result_counts_documents_by_extraction_mode(self):
        good = _touch(self.tmp.name, "good.pdf")
        scan = _touch(self.tmp.name, "scan.pdf")

        def fake_process(path, label, *a, **k):
            mode = "text_only" if "scan" in path else "layout"
            return [{"content": "x", "document": ing.pdf_key(path), "extraction": mode}]

        with patch.object(ing, "process_pdf", fake_process):
            result = ing.ingest_pdfs([good, scan], rebuild_index=False)

        self.assertEqual(result["extraction"], {"layout": 1, "text_only": 1})

    def test_every_page_of_one_document_counts_once(self):
        """The tally is per document, not per corpus entry."""
        pdf = _touch(self.tmp.name, "many_pages.pdf")

        def fake_process(path, label, *a, **k):
            key = ing.pdf_key(path)
            return [
                {"content": f"page-{i}", "document": key, "extraction": "text_only"}
                for i in range(5)
            ]

        with patch.object(ing, "process_pdf", fake_process):
            result = ing.ingest_pdfs([pdf], rebuild_index=False)

        self.assertEqual(result["extraction"], {"layout": 0, "text_only": 1})


if __name__ == "__main__":
    unittest.main()


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


class TestVersionDispatch(ManifestBackedTestCase):
    """Under INDEX_VERSION=2 the v2 processor runs, with the run's figure switch."""

    def _v2_entries(self, path, label, describe_figures=False, images_dir=None):
        key = ing.pdf_key(path)
        out = [{"document": key, "citation": label, "page": 0, "type": "text_chunk",
                "content": "c" * 300, "embed_text": "h c", "meta": {"seq": 0}, "extraction": "grobid"}]
        if describe_figures:
            out.append({"document": key, "citation": label, "page": 0, "type": "figure_description",
                        "content": "d" * 300, "embed_text": "h d", "meta": {"extraction": "vlm"},
                        "extraction": "grobid"})
        return out

    def test_v2_uses_process_pdf_v2_and_reports_modes(self):
        pdf = _touch(self.tmp, "a.pdf")
        with patch.object(ing, "INDEX_VERSION", 2), \
             patch("research_assistant.shared.ingest_v2.process_pdf_v2", side_effect=self._v2_entries) as v2, \
             patch.object(ing, "upsert_corpus", return_value=1), patch.object(ing, "rebuild_bm25"):
            result = ing.ingest_pdfs({pdf: "A"}, describe_figures=True)
        v2.assert_called_once()
        self.assertTrue(v2.call_args.kwargs["describe_figures"])
        self.assertEqual(result["extraction"], {"grobid": 1, "pymupdf": 0})
        self.assertEqual(result["described"], 1)

    def test_describe_defaults_to_config_figure_vlm(self):
        pdf = _touch(self.tmp, "a.pdf")
        with patch.object(ing, "INDEX_VERSION", 2), patch.object(ing, "FIGURE_VLM", True), \
             patch("research_assistant.shared.ingest_v2.process_pdf_v2", side_effect=self._v2_entries) as v2, \
             patch.object(ing, "upsert_corpus", return_value=1), patch.object(ing, "rebuild_bm25"):
            ing.ingest_pdfs({pdf: "A"})
        self.assertTrue(v2.call_args.kwargs["describe_figures"])

    def test_v1_path_is_untouched(self):
        pdf = _touch(self.tmp, "a.pdf")
        with patch.object(ing, "INDEX_VERSION", 1), patch.object(ing, "LAYOUT_DETECTION", False), \
             patch.object(ing, "_extract_text_only", return_value=[]) as v1, \
             patch("research_assistant.shared.ingest_v2.process_pdf_v2") as v2, \
             patch.object(ing, "rebuild_bm25"):
            result = ing.ingest_pdfs({pdf: "A"})
        v1.assert_called_once()
        v2.assert_not_called()
        self.assertEqual(result["extraction"], {"layout": 0, "text_only": 0})
