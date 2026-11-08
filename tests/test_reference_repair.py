# tests/test_reference_repair.py
"""Repair of stored papers whose text chunks hold reference lists (no real data/ touched)."""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from research_assistant.shared.paper_index import PaperEntry, PaperIndex
from research_assistant.shared.reference_repair import (
    LeakRepair,
    apply_reference_repair,
    scan_reference_leaks,
)
from tests.test_bibliography import APS, APS2, ELS

HEAVY = " ".join([APS, APS2, ELS])        # three reference marks
CLEAN = "clean body text"


class FakeCollection:
    """Rows of {"document", "stored_form", "text", "citation_source"} behind a Chroma-like get/delete."""

    def __init__(self, rows):
        self.rows = [dict(r) for r in rows]
        self.version = 7

    @staticmethod
    def _match(row, where):
        if not where:
            return True
        if "$and" in where:
            return all(FakeCollection._match(row, w) for w in where["$and"])
        for k, v in where.items():
            col = "stored_form" if k == "type" else k
            if isinstance(v, dict) and "$in" in v:
                if row.get(col) not in v["$in"]:
                    return False
            elif row.get(col) != v:
                return False
        return True

    def get(self, where=None, include=None, limit=None, offset=0):
        hits = [r for r in self.rows if self._match(r, where)]
        offset = offset or 0
        hits = hits[offset: offset + limit] if limit is not None else hits[offset:]
        return {
            "ids": [str(i) for i, _ in enumerate(hits)],
            "documents": [r["text"] for r in hits],
            "metadatas": [{"document": r["document"], "type": r["stored_form"],
                           "citation_source": r["citation_source"]} for r in hits],
        }

    def delete(self, where=None, ids=None):
        self.rows = [r for r in self.rows if not self._match(r, where)]

    def count(self, **where):
        return sum(1 for r in self.rows if self._match(r, where))

    def texts(self, **where):
        return [r["text"] for r in self.rows if self._match(r, where)]


def row(document, stored_form, text):
    return {"document": document, "stored_form": stored_form, "text": text, "citation_source": f"cite {document}"}


def chunk_entry(document, text):
    return {"document": document, "citation": f"cite {document}", "page": 1, "type": "text_chunk",
            "content": text, "embed_text": text, "meta": {"seq": 0}}


def fake_upsert_into(col):
    def upsert(entries):
        for e in entries:
            col.rows.append(row(e["document"], e["type"], e["content"]))
        return len(entries)
    return upsert


class TestScan(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.col = FakeCollection([
            row("a.pdf", "text_chunk", "intro text"),
            row("a.pdf", "text_chunk", HEAVY),
            row("a.pdf", "caption", "Figure 1. A plot."),
            row("a.pdf", "figure_image", "[Figure 1] A plot."),
            row("b.pdf", "text_chunk", "only clean prose here"),
            row("stub:x", "text_chunk", HEAVY),            # not a PDF: never a candidate
        ])
        self.index = PaperIndex(os.path.join(self.tmp, "index.json"))
        self.stub_clean = lambda d, p: [chunk_entry(d, CLEAN)]
        self.stub_same = lambda d, p: [chunk_entry(d, HEAVY)]

    def test_only_papers_with_reference_heavy_chunks_are_candidates(self):
        repairs, skips, fresh = scan_reference_leaks(self.col, self.index, fresh_chunks=self.stub_clean)
        self.assertEqual([r.document for r in repairs], ["a.pdf"])
        self.assertNotIn("b.pdf", [s.document for s in skips])
        self.assertEqual(repairs[0].stored_chunks, 2)
        self.assertEqual(repairs[0].stored_heavy, 1)
        self.assertEqual((repairs[0].fresh_chunks, repairs[0].fresh_heavy), (1, 0))
        self.assertEqual([e["content"] for e in fresh["a.pdf"]], [CLEAN])

    def test_a_paper_is_repaired_only_when_fresh_chunks_are_cleaner(self):
        repairs, skips, _ = scan_reference_leaks(self.col, self.index, fresh_chunks=self.stub_same)
        self.assertEqual(repairs, [])
        self.assertEqual([(s.document, s.reason) for s in skips], [("a.pdf", "not_improved")])

    def test_a_paper_with_no_fresh_chunks_is_skipped_not_wiped(self):
        repairs, skips, _ = scan_reference_leaks(self.col, self.index, fresh_chunks=lambda d, p: [])
        self.assertEqual([(s.document, s.reason) for s in skips], [("a.pdf", "no_fresh_chunks")])
        apply_reference_repair(repairs, {}, self.col, upsert=fake_upsert_into(self.col))
        self.assertEqual(self.col.count(document="a.pdf", stored_form="text_chunk"), 2)   # untouched

    def test_a_repair_without_fresh_entries_never_deletes(self):
        # A repair handed to apply with no fresh rows (a hand-edited call) must not wipe the paper.
        repair = LeakRepair("a.pdf", "/p/a.pdf", 2, 1, 1, 0)
        result = apply_reference_repair([repair], {}, self.col, upsert=fake_upsert_into(self.col))
        self.assertEqual(self.col.count(document="a.pdf", stored_form="text_chunk"), 2)
        self.assertEqual(result["repaired"], [])
        self.assertEqual([e["document"] for e in result["errors"]], ["a.pdf"])

    def test_extract_errors_are_skips(self):
        def boom(d, p):
            raise RuntimeError("bad cache")
        repairs, skips, _ = scan_reference_leaks(self.col, self.index, fresh_chunks=boom)
        self.assertEqual(repairs, [])
        self.assertEqual([(s.document, s.reason) for s in skips], [("a.pdf", "extract_failed: bad cache")])

    def test_fresh_entries_carry_the_stored_document_name(self):
        # process_pdf_v2 keys entries by the PDF's file name, which can differ from the stored name.
        repairs, _, fresh = scan_reference_leaks(
            self.col, self.index, fresh_chunks=lambda d, p: [chunk_entry("renamed.pdf", CLEAN)])
        self.assertEqual([e["document"] for e in fresh["a.pdf"]], ["a.pdf"])

    def test_pdf_path_comes_from_the_index_when_the_file_exists(self):
        pdf = os.path.join(self.tmp, "real_a.pdf")
        open(pdf, "wb").close()
        self.index.add(PaperEntry(document="a.pdf", path=pdf))
        seen = []
        scan_reference_leaks(self.col, self.index, fresh_chunks=lambda d, p: seen.append(p) or [])
        self.assertEqual(seen, [pdf])

    def test_apply_replaces_only_text_chunk_rows(self):
        repairs, _, fresh = scan_reference_leaks(self.col, self.index, fresh_chunks=self.stub_clean)
        result = apply_reference_repair(repairs, fresh, self.col, upsert=fake_upsert_into(self.col))
        self.assertEqual(self.col.texts(document="a.pdf", stored_form="text_chunk"), [CLEAN])
        self.assertEqual(self.col.count(document="a.pdf", stored_form="caption"), 1)
        self.assertEqual(self.col.count(document="a.pdf", stored_form="figure_image"), 1)
        self.assertEqual(self.col.count(document="b.pdf"), 1)
        self.assertEqual(result["repaired"], ["a.pdf"])
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["versions_before"], {"papers": 7})

    def test_a_failed_upsert_is_reported_as_deleted_not_reinserted(self):
        self.col.rows.append(row("c.pdf", "text_chunk", HEAVY))
        repairs, _, fresh = scan_reference_leaks(self.col, self.index, fresh_chunks=self.stub_clean)
        self.assertEqual([r.document for r in repairs], ["a.pdf", "c.pdf"])
        good = fake_upsert_into(self.col)

        def upsert_failing_for_a(entries):
            if entries[0]["document"] == "a.pdf":
                raise RuntimeError("boom")
            return good(entries)

        result = apply_reference_repair(repairs, fresh, self.col, upsert=upsert_failing_for_a)
        self.assertEqual(result["repaired"], ["c.pdf"])
        self.assertEqual(result["errors"], [{"document": "a.pdf", "error": "deleted_not_reinserted: boom"}])

    def test_missing_docling_cache_is_skipped_without_conversion(self):
        with patch("research_assistant.config.DOCLING_CACHE_DIR", os.path.join(self.tmp, "no_cache")), \
             patch("research_assistant.shared.ingest_v2.process_pdf_v2",
                   side_effect=AssertionError("process_pdf_v2 must not run without a cache")):
            _, skips, _ = scan_reference_leaks(self.col, self.index)
        self.assertEqual([(s.document, s.reason) for s in skips], [("a.pdf", "no_docling_cache")])

    def test_the_default_extractor_refuses_docling_conversion(self):
        # A cache file that exists but cannot be read would make extract_docling convert the PDF.
        from research_assistant.shared import reference_repair
        cache_dir = os.path.join(self.tmp, "cache")
        os.makedirs(cache_dir)
        pdf = os.path.join(self.tmp, "a.pdf")
        open(pdf, "wb").close()
        with open(os.path.join(cache_dir, "a.docling.json"), "w") as fh:
            fh.write("{not json")
        with patch("research_assistant.config.DOCLING_CACHE_DIR", cache_dir), \
             patch("research_assistant.shared.docling_extract.get_digital_converter",
                   side_effect=AssertionError("conversion attempted")) as conv:
            with self.assertRaises(Exception) as ctx:
                reference_repair._fresh_chunks_from_cache("a.pdf", pdf, "cite a")
        conv.assert_not_called()
        self.assertNotIsInstance(ctx.exception, AssertionError)


class TestCli(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.col = FakeCollection([row("a.pdf", "text_chunk", HEAVY)])

        class Client:
            def get_collection(inner, name):
                return self.col

        self.patches = [
            patch("research_assistant.shared.db.open_store", return_value=Client()),
            patch("research_assistant.shared.paper_index.PaperIndex.load",
                  return_value=PaperIndex(os.path.join(self.tmp, "index.json"))),
            patch("research_assistant.shared.reference_repair._fresh_chunks_from_cache",
                  side_effect=lambda d, p, c: [chunk_entry(d, CLEAN)]),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.plan = os.path.join(self.tmp, "plan.json")

    def run_main(self, argv):
        from scripts.repair_reference_leak import main
        out = io.StringIO()
        with redirect_stdout(out):
            rc = main(argv + ["--plan", self.plan])
        return rc, out.getvalue()

    def test_dry_run_writes_the_plan_and_changes_nothing(self):
        rc, out = self.run_main([])
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(self.plan))
        self.assertIn("1 papers, 1 stored chunks → 1 fresh chunks, 1 reference-heavy chunks removed", out)
        self.assertEqual(self.col.texts(document="a.pdf"), [HEAVY])

    def test_apply_refuses_when_the_scan_differs_from_the_plan(self):
        rc, _ = self.run_main([])
        self.assertEqual(rc, 0)
        self.col.rows.append(row("new.pdf", "text_chunk", HEAVY))
        rc, out = self.run_main(["--apply"])
        self.assertEqual(rc, 1)
        self.assertIn("store changed since the dry run", out)
        self.assertEqual(self.col.texts(document="a.pdf"), [HEAVY])

    def test_apply_without_a_plan_refuses(self):
        rc, out = self.run_main(["--apply"])
        self.assertEqual(rc, 1)
        self.assertIn("dry run", out)


if __name__ == "__main__":
    unittest.main()
