"""Tests for Agent 1.

Two properties matter. Where a PDF is filed after processing is the whole
record of what happened — a paper under processed/ is treated as done and
never looked at again. And extraction must degrade to the regex path rather
than stopping when GROBID is unreachable, because a dead server used to end
the run.

The last group covers the merged run_extractor()'s payload shape: it must
keep "articles" (the citing papers' own metadata) alongside "references",
and its return dict must use "reference_count" rather than reusing
"references" for both a list (on disk) and an int (in the return value).
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import research_assistant.agents.agent1_extractor as ex
from research_assistant.schemas import Reference


class TestUniqueDestination(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_uses_the_plain_name_when_free(self):
        self.assertEqual(
            ex._unique_destination(self.tmp.name, "paper.pdf"),
            os.path.join(self.tmp.name, "paper.pdf"),
        )

    def test_suffixes_rather_than_overwriting(self):
        """Two different papers with one filename must not silently collide."""
        open(os.path.join(self.tmp.name, "paper.pdf"), "w").close()
        self.assertNotEqual(
            ex._unique_destination(self.tmp.name, "paper.pdf"),
            os.path.join(self.tmp.name, "paper.pdf"),
        )


class TestRegexReferences(unittest.TestCase):
    def _extract(self, text):
        with patch.object(ex, "_pdftotext", return_value=text):
            return ex._references_from_regex("some.pdf")

    def test_bracket_numbered_references(self):
        refs = self._extract(
            "Introduction text\n"
            "References\n"
            "[1] Smith J. A first paper. Nature, 2019.\n"
            "[2] Doe A. A second paper. PRL, 2020.\n"
        )
        self.assertEqual(len(refs), 2)
        self.assertIn("Smith", refs[0].raw_reference)

    def test_dot_numbered_references(self):
        refs = self._extract(
            "References\n"
            "1. Smith J. A first paper. Nature, 2019.\n"
            "2. Doe A. A second paper. PRL, 2020.\n"
        )
        self.assertEqual(len(refs), 2)

    def test_every_reference_is_tagged_as_regex_extracted(self):
        """Provenance must be visible downstream — these are lower quality."""
        refs = self._extract("References\n[1] Smith J. A paper. Nature, 2019.\n")
        self.assertEqual(refs[0].extraction_method, "regex")
        self.assertEqual(refs[0].source_file, "some.pdf")

    def test_structured_fields_are_absent_not_invented(self):
        refs = self._extract("References\n[1] Smith J. A paper. Nature, 2019.\n")
        self.assertIsNone(refs[0].doi)
        self.assertIsNone(refs[0].doi_confidence)

    def test_no_recognisable_references_yields_nothing(self):
        self.assertEqual(self._extract("Just prose, no reference list."), [])


class TestFallbackRouting(unittest.TestCase):
    def _ref(self, method):
        return Reference(
            raw_reference="r", source_file="p.pdf", extraction_method=method
        )

    def test_grobid_used_when_available_and_productive(self):
        with patch.object(ex, "_references_from_grobid",
                          return_value=[self._ref("grobid")]) as grobid, \
             patch.object(ex, "_references_from_regex") as regex:
            refs, method = ex.extract_references("p.pdf", grobid_ok=True)
        self.assertEqual(method, "grobid")
        self.assertEqual(len(refs), 1)
        grobid.assert_called_once()
        regex.assert_not_called()

    def test_regex_used_when_grobid_is_down(self):
        """The case that used to end the whole run."""
        with patch.object(ex, "_references_from_grobid") as grobid, \
             patch.object(ex, "_references_from_regex",
                          return_value=[self._ref("regex")]):
            refs, method = ex.extract_references("p.pdf", grobid_ok=False)
        self.assertEqual(method, "regex")
        self.assertEqual(len(refs), 1)
        grobid.assert_not_called()

    def test_regex_used_when_grobid_returns_nothing_for_this_pdf(self):
        """Per-PDF fallback, not just per-run."""
        with patch.object(ex, "_references_from_grobid", return_value=[]), \
             patch.object(ex, "_references_from_regex",
                          return_value=[self._ref("regex")]):
            refs, method = ex.extract_references("p.pdf", grobid_ok=True)
        self.assertEqual(method, "regex")

    def test_both_paths_empty_reports_none(self):
        with patch.object(ex, "_references_from_grobid", return_value=[]), \
             patch.object(ex, "_references_from_regex", return_value=[]):
            refs, method = ex.extract_references("p.pdf", grobid_ok=True)
        self.assertEqual((refs, method), ([], "none"))


class TestGrobidProbe(unittest.TestCase):
    def test_unreachable_server_is_false_not_an_exception(self):
        import requests

        with patch.object(ex.requests, "get",
                          side_effect=requests.RequestException("refused")):
            self.assertFalse(ex.grobid_alive())


class TestTeiParseMemoization(unittest.TestCase):
    """Directed change 2: _references_from_grobid and _article_from_grobid
    both read the same TEI file for any PDF that took the GROBID path, and
    the Finding 1 fix makes _article_from_grobid run in more cases, so the
    duplicate parse became more frequent, not less. The memoized helper must
    still hand each caller its own copy — a cache that returns the same
    mutable dict/list to everyone would let one caller's mutation corrupt
    what another caller already received."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        ex._parse_tei_file_cached.cache_clear()
        self.addCleanup(ex._parse_tei_file_cached.cache_clear)

        self.tei_path = os.path.join(self.tmp.name, "a.grobid.tei.xml")
        open(self.tei_path, "w").close()

        patcher = patch.object(ex, "_tei_path_for", return_value=self.tei_path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_second_caller_hits_the_cache_not_a_second_parse(self):
        article = {"source_file": "a.pdf", "title": "T", "doi": None, "authors": []}
        references = [{
            "raw_reference": "r", "source_file": "a.pdf", "title": None,
            "container": None, "authors": [], "year": None, "doi": None,
            "doi_confidence": None, "xml_id": None,
        }]
        with patch.object(ex, "parse_tei_file",
                           return_value=(article, references)) as parse_mock:
            refs = ex._references_from_grobid("a.pdf")
            got_article = ex._article_from_grobid("a.pdf")

        parse_mock.assert_called_once_with(self.tei_path)
        self.assertEqual(len(refs), 1)
        self.assertEqual(got_article, article)

    def test_mutating_a_returned_article_does_not_corrupt_the_cache(self):
        """_article_from_grobid hands back the article dict as-is (no
        reconstruction, unlike the Reference objects _references_from_grobid
        builds), so this is exactly where a shared-object cache would leak a
        mutation from one caller into the next."""
        article = {"source_file": "a.pdf", "title": "T", "doi": None, "authors": []}
        with patch.object(ex, "parse_tei_file", return_value=(article, [])):
            first = ex._article_from_grobid("a.pdf")
            first["title"] = "mutated"
            second = ex._article_from_grobid("a.pdf")

        self.assertEqual(second["title"], "T")


class RunExtractorTestCase(unittest.TestCase):
    """Isolates run_extractor() to a temp RAW_DIR/output path per test, so it
    never touches the real data/ directory."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw_dir = os.path.join(self.tmp.name, "raw")
        os.makedirs(self.raw_dir)
        self.citations_path = os.path.join(self.tmp.name, "extracted_citations.json")

        self._patch(ex, "RAW_DIR", self.raw_dir)
        self._patch(ex, "XML_OUTPUT_DIR", os.path.join(self.raw_dir, "grobid_output"))
        self._patch(ex, "EXTRACTED_CITATIONS_PATH", self.citations_path)

    def _patch(self, target, name, value):
        patcher = patch.object(target, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _touch_pdf(self, name):
        path = os.path.join(self.raw_dir, name)
        open(path, "wb").close()
        return path


class TestRunExtractorPayload(RunExtractorTestCase):
    def test_no_pdfs_returns_reference_count_zero(self):
        """The empty-run early return must use the same key shape as the
        full run — a caller should not have to special-case it."""
        self.assertEqual(
            ex.run_extractor(),
            {"reference_count": 0, "processed": 0, "failed": 0},
        )

    def test_return_value_uses_reference_count_not_references(self):
        """references (the list) lives in the JSON payload; the return dict
        must not reuse the same key for an int — one name, one type."""
        self._touch_pdf("a.pdf")
        with patch.object(ex, "grobid_alive", return_value=False), \
             patch.object(ex, "extract_references",
                          return_value=([Reference(raw_reference="r", source_file="a.pdf",
                                                    extraction_method="regex")], "regex")):
            result = ex.run_extractor()
        self.assertEqual(result["reference_count"], 1)
        self.assertNotIn("references", result)

    def test_payload_on_disk_keeps_articles_alongside_references(self):
        """Ruling G2: articles (citing papers' own metadata) must not be
        dropped just because nothing downstream reads it yet."""
        self._touch_pdf("a.pdf")
        article = {"source_file": "a.pdf", "title": "T", "doi": None, "authors": []}
        with patch.object(ex, "grobid_alive", return_value=True), \
             patch.object(ex, "run_grobid_batch"), \
             patch.object(ex, "extract_references",
                          return_value=([Reference(raw_reference="r", source_file="a.pdf",
                                                    extraction_method="grobid")], "grobid")), \
             patch.object(ex, "_article_from_grobid", return_value=article):
            ex.run_extractor()

        with open(self.citations_path) as fh:
            payload = json.load(fh)

        self.assertEqual(payload["articles"], [article])
        self.assertEqual(len(payload["references"]), 1)
        self.assertIn("summary", payload)
        self.assertEqual(payload["extraction"], {"grobid": 1, "regex": 0, "none": 0})

    def test_regex_only_pdf_contributes_no_article_record(self):
        """A PDF that fell back to regex has no TEI to read a header from —
        that is expected, not a bug to paper over with a placeholder.

        This covers only the *global* fallback (grobid_alive() is False for
        the whole run). See
        test_grobid_ok_but_pdf_used_regex_still_contributes_article below for
        the per-PDF fallback case, which this test cannot catch: here
        grobid_ok is False, so the buggy `if method == "grobid"` gate and the
        correct `if grobid_ok` gate agree (both skip) — the regression only
        shows up when grobid_ok is True but method isn't "grobid"."""
        self._touch_pdf("a.pdf")
        with patch.object(ex, "grobid_alive", return_value=False), \
             patch.object(ex, "extract_references",
                          return_value=([Reference(raw_reference="r", source_file="a.pdf",
                                                    extraction_method="regex")], "regex")):
            ex.run_extractor()

        with open(self.citations_path) as fh:
            payload = json.load(fh)
        self.assertEqual(payload["articles"], [])

    def test_grobid_ok_but_pdf_used_regex_still_contributes_article(self):
        """Finding 1 (Important): situation (b) — GROBID is up and processed
        this PDF fine (TEI header, title, authors, DOI all parseable), but
        found no reference list for it. extract_references then reports
        method="regex" for this PDF, but the citing paper's own metadata is
        still sitting on disk in the TEI and must not be silently discarded.

        Gating article collection on `method == "grobid"` (the bug) drops it
        here, because method describes only where the *references* came
        from, not whether GROBID ran. Gating on `grobid_ok` (the fix) keeps
        it, since _article_from_grobid does its own os.path.exists check and
        is safe to call regardless of method."""
        self._touch_pdf("a.pdf")
        article = {"source_file": "a.pdf", "title": "T", "doi": None, "authors": []}
        with patch.object(ex, "grobid_alive", return_value=True), \
             patch.object(ex, "run_grobid_batch"), \
             patch.object(ex, "extract_references",
                          return_value=([Reference(raw_reference="r", source_file="a.pdf",
                                                    extraction_method="regex")], "regex")), \
             patch.object(ex, "_article_from_grobid", return_value=article) as article_mock:
            ex.run_extractor()

        with open(self.citations_path) as fh:
            payload = json.load(fh)
        self.assertEqual(payload["articles"], [article])
        article_mock.assert_called_once_with(os.path.join(self.raw_dir, "a.pdf"))

    def test_batch_failure_downgrades_grobid_ok_for_the_whole_run(self):
        """Directed change 3: run_grobid_batch() raising (a client
        construction or .process() failure, distinct from grobid_alive()
        itself returning False) must downgrade grobid_ok to False for the
        rest of the run, so every PDF is handled via the regex/none path
        instead of the run crashing."""
        self._touch_pdf("a.pdf")
        self._touch_pdf("b.pdf")
        with patch.object(ex, "grobid_alive", return_value=True), \
             patch.object(ex, "run_grobid_batch", side_effect=RuntimeError("boom")), \
             patch.object(ex, "_references_from_grobid") as grobid_refs, \
             patch.object(ex, "_references_from_regex",
                          return_value=[Reference(raw_reference="r", source_file="x.pdf",
                                                   extraction_method="regex")]):
            result = ex.run_extractor()

        grobid_refs.assert_not_called()
        self.assertEqual(result, {"reference_count": 2, "processed": 2, "failed": 0})

    def test_grobid_batch_is_called_once_up_front_not_per_pdf(self):
        """Ruling G3: run_grobid_batch is the batch call that honours
        GROBID_BATCH_CONCURRENCY. Calling it once per PDF would silently lose
        that concurrency, so extract_references must never trigger it."""
        self._touch_pdf("a.pdf")
        self._touch_pdf("b.pdf")
        with patch.object(ex, "grobid_alive", return_value=True), \
             patch.object(ex, "run_grobid_batch") as batch, \
             patch.object(ex, "extract_references", return_value=([], "none")):
            ex.run_extractor()
        batch.assert_called_once_with(self.raw_dir, ex.XML_OUTPUT_DIR)


if __name__ == "__main__":
    unittest.main()
