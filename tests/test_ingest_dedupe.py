"""Tests for Layer 2 deduplication at ingestion."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from research_assistant.shared import ingestion as ing
from research_assistant.shared import manifest as manifest_mod
from research_assistant.shared.paper_header import PaperHeader
from research_assistant.shared.paper_index import PaperEntry, PaperIndex


class TestIngestDedupe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        self.pulled_dir = os.path.join(self.tmp.name, "pulled_pdfs")
        os.makedirs(self.pulled_dir, exist_ok=True)
        self.raw_dir = os.path.join(self.tmp.name, "raw")
        os.makedirs(self.raw_dir, exist_ok=True)

        self.manifest_path = os.path.join(self.tmp.name, "ingested.json")
        self.index_path = os.path.join(self.tmp.name, "paper_index.json")
        self.downloaded_path = os.path.join(self.tmp.name, "downloaded.json")
        self.lock_path = os.path.join(self.tmp.name, "ingest.lock")

        # Create corpus paper P
        self.p_file = os.path.join(self.tmp.name, "existing_p.pdf")
        with open(self.p_file, "wb") as f:
            f.write(b"%PDF-1.4 existing paper P")

        self.paper_p = PaperEntry(
            document="existing_p.pdf",
            path=self.p_file,
            title="Half-Metallic Graphene Nanoribbons",
            first_authors=("Son",),
            dois=("10.1038/nature05180",),
        )

        idx = PaperIndex(self.index_path)
        idx.add(self.paper_p)
        idx.save()

        self.patches = [
            patch.object(ing, "PULLED_PDFS_DIR", self.pulled_dir),
            patch.object(ing, "RAW_DIR", self.raw_dir),
            patch.object(ing, "PAPER_INDEX_PATH", self.index_path),
            patch.object(ing, "DOWNLOADED_JSON_PATH", self.downloaded_path),
            patch.object(ing, "INGEST_LOCK_PATH", self.lock_path),
            patch.object(manifest_mod, "MANIFEST_PATH", self.manifest_path),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_title_author_duplicate_in_pulled_pdfs_is_dropped_and_deleted(self):
        new_pdf = os.path.join(self.pulled_dir, "new_dup.pdf")
        with open(new_pdf, "wb") as f:
            f.write(b"%PDF-1.4 new duplicate")

        candidates = {new_pdf: "new_dup"}
        corpus = [{"document": "new_dup.pdf", "text": "chunk"}]
        header = PaperHeader(
            title="Half-Metallic Graphene Nanoribbons",
            first_authors=("Son",),
            doi=None,
            arxiv_id=None,
        )

        with patch("research_assistant.shared.ingestion.read_header", return_value=header):
            kept_cands, kept_corp, duplicates, headers = ing._drop_title_duplicates(candidates, corpus)

        self.assertNotIn(new_pdf, kept_cands)
        self.assertEqual(kept_corp, [])
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0]["path"], new_pdf)
        self.assertEqual(duplicates[0]["document"], "new_dup.pdf")
        self.assertEqual(duplicates[0]["kept"], "existing_p.pdf")
        self.assertTrue(duplicates[0]["deleted"])
        self.assertFalse(os.path.exists(new_pdf))

    def test_duplicate_outside_pulled_pdfs_is_dropped_but_kept_on_disk(self):
        new_pdf = os.path.join(self.raw_dir, "raw_dup.pdf")
        with open(new_pdf, "wb") as f:
            f.write(b"%PDF-1.4 raw duplicate")

        candidates = {new_pdf: "raw_dup"}
        corpus = [{"document": "raw_dup.pdf", "text": "chunk"}]
        header = PaperHeader(
            title="Half-Metallic Graphene Nanoribbons",
            first_authors=("Son",),
            doi=None,
            arxiv_id=None,
        )

        with patch("research_assistant.shared.ingestion.read_header", return_value=header):
            kept_cands, kept_corp, duplicates, headers = ing._drop_title_duplicates(candidates, corpus)

        self.assertNotIn(new_pdf, kept_cands)
        self.assertEqual(kept_corp, [])
        self.assertEqual(len(duplicates), 1)
        self.assertFalse(duplicates[0]["deleted"])
        self.assertTrue(os.path.exists(new_pdf))

    def test_unreadable_header_keeps_the_pdf(self):
        new_pdf = os.path.join(self.pulled_dir, "unreadable.pdf")
        with open(new_pdf, "wb") as f:
            f.write(b"%PDF-1.4 unreadable")

        candidates = {new_pdf: "unreadable"}
        corpus = [{"document": "unreadable.pdf", "text": "chunk"}]

        with patch("research_assistant.shared.ingestion.read_header", return_value=None):
            kept_cands, kept_corp, duplicates, headers = ing._drop_title_duplicates(candidates, corpus)

        self.assertIn(new_pdf, kept_cands)
        self.assertEqual(kept_corp, corpus)
        self.assertEqual(duplicates, [])
        self.assertTrue(os.path.exists(new_pdf))

    def test_match_whose_file_is_gone_does_not_drop(self):
        # Remove P's file from disk
        os.unlink(self.p_file)

        new_pdf = os.path.join(self.pulled_dir, "new_cand.pdf")
        with open(new_pdf, "wb") as f:
            f.write(b"%PDF-1.4 candidate")

        candidates = {new_pdf: "new_cand"}
        corpus = [{"document": "new_cand.pdf", "text": "chunk"}]
        header = PaperHeader(
            title="Half-Metallic Graphene Nanoribbons",
            first_authors=("Son",),
            doi=None,
            arxiv_id=None,
        )

        with patch("research_assistant.shared.ingestion.read_header", return_value=header):
            kept_cands, kept_corp, duplicates, headers = ing._drop_title_duplicates(candidates, corpus)

        self.assertIn(new_pdf, kept_cands)
        self.assertEqual(duplicates, [])
        self.assertTrue(os.path.exists(new_pdf))

    def test_duplicate_is_aliased_and_downloaded_json_repointed(self):
        new_pdf = os.path.join(self.pulled_dir, "dup_alias.pdf")
        with open(new_pdf, "wb") as f:
            f.write(b"%PDF-1.4 dup alias")

        # downloaded.json with a record pointing to new_pdf
        with open(self.downloaded_path, "w") as f:
            json.dump({"ref1": {"path": new_pdf, "doi": "10.1038/nature05180"}}, f)

        candidates = {new_pdf: "dup_alias"}
        corpus = [{"document": "dup_alias.pdf", "text": "chunk"}]
        header = PaperHeader(
            title="Half-Metallic Graphene Nanoribbons",
            first_authors=("Son",),
            doi=None,
            arxiv_id=None,
        )

        with patch("research_assistant.shared.ingestion.read_header", return_value=header):
            ing._drop_title_duplicates(candidates, corpus)

        idx = PaperIndex.load(self.index_path)
        resolved = idx.resolve("dup_alias.pdf")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.document, "existing_p.pdf")

        with open(self.downloaded_path) as f:
            d_data = json.load(f)
        self.assertEqual(d_data["ref1"]["path"], self.p_file)

    def test_ingested_pdf_is_added_to_the_index(self):
        pdf_path = os.path.join(self.pulled_dir, "unique_paper.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 unique")

        header = PaperHeader(
            title="Novel 2D Materials",
            first_authors=("Novoselov",),
            doi="10.1126/science.1001",
            arxiv_id="cond-mat/0501001",
        )

        def fake_process(path, label, *a, **k):
            return [{"document": "unique_paper.pdf", "text": "chunk"}]

        with patch("research_assistant.shared.ingestion.process_pdf", side_effect=fake_process), \
             patch("research_assistant.shared.ingestion.read_header", return_value=header), \
             patch("research_assistant.shared.ingestion.upsert_corpus", return_value=1), \
             patch("research_assistant.shared.ingestion.rebuild_bm25"):
            result = ing.ingest_pdfs([pdf_path])

        self.assertEqual(result["inserted"], 1)
        idx = PaperIndex.load(self.index_path)
        entry = idx.get("unique_paper.pdf")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.title, "Novel 2D Materials")
        self.assertEqual(entry.first_authors, ("Novoselov",))
        self.assertIn("10.1126/science.1001", entry.dois)
        self.assertIn("cond-mat/0501001", entry.arxiv_ids)

    def test_duplicate_never_reaches_the_store(self):
        dup_pdf = os.path.join(self.pulled_dir, "dup_paper.pdf")
        with open(dup_pdf, "wb") as f:
            f.write(b"%PDF-1.4 duplicate")

        uniq_pdf = os.path.join(self.pulled_dir, "uniq_paper.pdf")
        with open(uniq_pdf, "wb") as f:
            f.write(b"%PDF-1.4 unique")

        dup_header = PaperHeader(
            title="Half-Metallic Graphene Nanoribbons",
            first_authors=("Son",),
            doi=None,
            arxiv_id=None,
        )
        uniq_header = PaperHeader(
            title="Unique Paper Title",
            first_authors=("Geim",),
            doi=None,
            arxiv_id=None,
        )

        def fake_read_header(path, *a, **k):
            if "dup_paper" in path:
                return dup_header
            return uniq_header

        def fake_process(path, label, *a, **k):
            doc = "dup_paper.pdf" if "dup_paper" in path else "uniq_paper.pdf"
            return [{"document": doc, "text": f"chunk for {doc}"}]

        recorded_upsert = []

        def fake_upsert(corpus):
            recorded_upsert.extend(corpus)
            return len(corpus)

        with patch("research_assistant.shared.ingestion.process_pdf", side_effect=fake_process), \
             patch("research_assistant.shared.ingestion.read_header", side_effect=fake_read_header), \
             patch("research_assistant.shared.ingestion.upsert_corpus", side_effect=fake_upsert), \
             patch("research_assistant.shared.ingestion.rebuild_bm25"):
            result = ing.ingest_pdfs([dup_pdf, uniq_pdf])

        # Verify only unique paper reached upsert_corpus
        upserted_docs = {e["document"] for e in recorded_upsert}
        self.assertIn("uniq_paper.pdf", upserted_docs)
        self.assertNotIn("dup_paper.pdf", upserted_docs)

        # Verify duplicate is not in ingest manifest
        manifest_docs = manifest_mod.load()
        self.assertIn("uniq_paper.pdf", manifest_docs)
        self.assertNotIn("dup_paper.pdf", manifest_docs)

        # Verify result["duplicates"]
        self.assertEqual(len(result["duplicates"]), 1)
        self.assertEqual(result["duplicates"][0]["document"], "dup_paper.pdf")


if __name__ == "__main__":
    unittest.main()
