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
        manifest_mod.add(self.paper_p.document)
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

    def test_a_null_path_record_in_downloaded_json_does_not_stop_the_drop(self):
        # 2026-11-08: abstract-only records carry "path": null and crashed repoint_manifest
        # after the duplicate PDF had already been deleted.
        manifest_mod.add(self.paper_p.document)
        new_pdf = os.path.join(self.pulled_dir, "new_dup.pdf")
        with open(new_pdf, "wb") as f:
            f.write(b"%PDF-1.4 new duplicate")
        with open(self.downloaded_path, "w") as f:
            json.dump({"raw:abs": {"path": None, "title": "Abstract only"},
                       "doi:new": {"path": new_pdf, "title": "Half-Metallic Graphene Nanoribbons"}}, f)
        header = PaperHeader(title="Half-Metallic Graphene Nanoribbons", first_authors=("Son",),
                             doi=None, arxiv_id=None)

        with patch("research_assistant.shared.ingestion.read_header", return_value=header):
            kept_cands, _, duplicates, _ = ing._drop_title_duplicates({new_pdf: "new_dup"}, [])

        self.assertNotIn(new_pdf, kept_cands)
        self.assertEqual([d["kept"] for d in duplicates], ["existing_p.pdf"])
        with open(self.downloaded_path) as f:
            downloaded = json.load(f)
        self.assertEqual(downloaded["doi:new"]["path"], self.p_file)
        self.assertIsNone(downloaded["raw:abs"]["path"])

    def test_a_drop_whose_bookkeeping_fails_keeps_its_pdf(self):
        # Record the alias and repoint the manifest before deleting, so a failure there
        # never leaves a deleted PDF that nothing points at.
        new_pdf = os.path.join(self.pulled_dir, "new_dup.pdf")
        with open(new_pdf, "wb") as f:
            f.write(b"%PDF-1.4 new duplicate")
        index = PaperIndex.load(self.index_path)
        with patch.object(ing, "repoint_manifest", side_effect=RuntimeError("manifest unwritable")):
            with self.assertRaises(RuntimeError):
                ing._execute_drops({new_pdf: self.paper_p}, index)
        self.assertTrue(os.path.exists(new_pdf))

    def test_duplicate_outside_pulled_pdfs_is_dropped_but_kept_on_disk(self):
        manifest_mod.add(self.paper_p.document)
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

    def test_match_that_was_never_ingested_does_not_drop(self):
        new_pdf = os.path.join(self.pulled_dir, "new_dup.pdf")
        with open(new_pdf, "wb") as f:
            f.write(b"%PDF-1.4 new duplicate")

        with open(self.downloaded_path, "w") as f:
            json.dump({"ref1": {"path": new_pdf, "doi": "10.1038/nature05180"}}, f)

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

        self.assertIn(new_pdf, kept_cands)
        self.assertEqual(kept_corp, corpus)
        self.assertEqual(duplicates, [])
        self.assertTrue(os.path.exists(new_pdf))

        idx = PaperIndex.load(self.index_path)
        self.assertIsNone(idx.resolve("new_dup.pdf"))

        with open(self.downloaded_path) as f:
            d_data = json.load(f)
        self.assertEqual(d_data["ref1"]["path"], new_pdf)

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
        manifest_mod.add(self.paper_p.document)
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
        manifest_mod.add(self.paper_p.document)
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

    def test_store_duplicate_is_dropped_before_parsing(self):
        manifest_mod.add(self.paper_p.document)
        dup_pdf = os.path.join(self.pulled_dir, "dup_pdf.pdf")
        with open(dup_pdf, "wb") as f:
            f.write(b"%PDF dup")
        uniq_pdf = os.path.join(self.pulled_dir, "uniq_pdf.pdf")
        with open(uniq_pdf, "wb") as f:
            f.write(b"%PDF uniq")

        dup_header = PaperHeader(
            title="Half-Metallic Graphene Nanoribbons",
            first_authors=("Son",),
            doi=None,
            arxiv_id=None,
        )
        uniq_header = PaperHeader(
            title="Unique Paper",
            first_authors=("Geim",),
            doi=None,
            arxiv_id=None,
        )

        parsed_paths = []
        def fake_process(path, label, *a, **k):
            parsed_paths.append(path)
            doc = os.path.basename(path)
            return [{"document": doc, "text": f"chunk {doc}"}]

        def fake_read_header(path, *a, **k):
            if "dup_pdf" in path:
                return dup_header
            return uniq_header

        with patch("research_assistant.shared.ingestion.is_grobid_alive", return_value=True), \
             patch("research_assistant.shared.ingestion.read_header", side_effect=fake_read_header), \
             patch("research_assistant.shared.ingestion.process_pdf", side_effect=fake_process), \
             patch("research_assistant.shared.ingestion.upsert_corpus", return_value=1), \
             patch("research_assistant.shared.ingestion.rebuild_bm25"):
            result = ing.ingest_pdfs([dup_pdf, uniq_pdf])

        self.assertEqual(parsed_paths, [uniq_pdf])
        self.assertFalse(os.path.exists(dup_pdf))
        self.assertEqual([d["kept"] for d in result["duplicates"]], ["existing_p.pdf"])

    def test_two_copies_in_one_batch_keep_the_doi_file(self):
        title_abc_path = os.path.join(self.pulled_dir, "title_abc.pdf")
        with open(title_abc_path, "wb") as f:
            f.write(b"%PDF title_abc")
        doi_path = os.path.join(self.pulled_dir, "doi_10.1_x.pdf")
        with open(doi_path, "wb") as f:
            f.write(b"%PDF doi")

        with open(self.downloaded_path, "w") as f:
            json.dump({"ref": {"path": title_abc_path}}, f)

        header = PaperHeader(
            title="Spin Transport in Graphene Bilayers",
            first_authors=("Tombros",),
            doi=None,
            arxiv_id=None,
        )

        parsed_paths = []
        def fake_process(path, label, *a, **k):
            parsed_paths.append(path)
            doc = os.path.basename(path)
            return [{"document": doc, "text": f"chunk {doc}"}]

        with patch("research_assistant.shared.ingestion.is_grobid_alive", return_value=True), \
             patch("research_assistant.shared.ingestion.read_header", return_value=header), \
             patch("research_assistant.shared.ingestion.process_pdf", side_effect=fake_process), \
             patch("research_assistant.shared.ingestion.upsert_corpus", return_value=1), \
             patch("research_assistant.shared.ingestion.rebuild_bm25"):
            result = ing.ingest_pdfs([title_abc_path, doi_path])

        self.assertEqual(parsed_paths, [doi_path])
        self.assertFalse(os.path.exists(title_abc_path))
        self.assertEqual(result["duplicates"][0]["document"], "title_abc.pdf")
        self.assertEqual(result["duplicates"][0]["kept"], "doi_10.1_x.pdf")
        self.assertEqual(PaperIndex.load(self.index_path).resolve("title_abc.pdf").document, "doi_10.1_x.pdf")
        self.assertEqual(json.load(open(self.downloaded_path))["ref"]["path"], doi_path)
        self.assertIn("doi_10.1_x.pdf", manifest_mod.load())
        self.assertNotIn("title_abc.pdf", manifest_mod.load())

    def test_same_title_different_authors_in_one_batch_are_both_parsed(self):
        a_path = os.path.join(self.pulled_dir, "paper_a.pdf")
        with open(a_path, "wb") as f:
            f.write(b"%PDF a")
        b_path = os.path.join(self.pulled_dir, "paper_b.pdf")
        with open(b_path, "wb") as f:
            f.write(b"%PDF b")

        header_a = PaperHeader(title="A Survey of Large Language Models", first_authors=("Zhao",), doi=None, arxiv_id=None)
        header_b = PaperHeader(title="A Survey of Large Language Models", first_authors=("Chang",), doi=None, arxiv_id=None)

        def fake_read_header(path, *a, **k):
            if "paper_a" in path:
                return header_a
            return header_b

        parsed_paths = []
        def fake_process(path, label, *a, **k):
            parsed_paths.append(path)
            doc = os.path.basename(path)
            return [{"document": doc, "text": f"chunk {doc}"}]

        with patch("research_assistant.shared.ingestion.is_grobid_alive", return_value=True), \
             patch("research_assistant.shared.ingestion.read_header", side_effect=fake_read_header), \
             patch("research_assistant.shared.ingestion.process_pdf", side_effect=fake_process), \
             patch("research_assistant.shared.ingestion.upsert_corpus", return_value=1), \
             patch("research_assistant.shared.ingestion.rebuild_bm25"):
            result = ing.ingest_pdfs([a_path, b_path])

        self.assertEqual(sorted(parsed_paths), sorted([a_path, b_path]))
        self.assertEqual(result["duplicates"], [])

    def test_ingested_match_behind_a_pending_entry_still_drops(self):
        q_file = os.path.join(self.tmp.name, "pending_q.pdf")
        with open(q_file, "wb") as f:
            f.write(b"%PDF pending Q")

        idx = PaperIndex(self.index_path)
        idx.add(PaperEntry(document="pending_q.pdf", path=q_file, title="Half-Metallic Graphene Nanoribbons", first_authors=("Son",)))
        idx.add(self.paper_p)
        idx.save()

        manifest_mod.add("existing_p.pdf")

        dup_pdf = os.path.join(self.pulled_dir, "dup_candidate.pdf")
        with open(dup_pdf, "wb") as f:
            f.write(b"%PDF dup cand")

        header = PaperHeader(title="Half-Metallic Graphene Nanoribbons", first_authors=("Son",), doi=None, arxiv_id=None)

        parsed_paths = []
        def fake_process(path, label, *a, **k):
            parsed_paths.append(path)
            doc = os.path.basename(path)
            return [{"document": doc, "text": f"chunk {doc}"}]

        with patch("research_assistant.shared.ingestion.is_grobid_alive", return_value=True), \
             patch("research_assistant.shared.ingestion.read_header", return_value=header), \
             patch("research_assistant.shared.ingestion.process_pdf", side_effect=fake_process), \
             patch("research_assistant.shared.ingestion.upsert_corpus", return_value=1), \
             patch("research_assistant.shared.ingestion.rebuild_bm25"):
            result = ing.ingest_pdfs([dup_pdf])

        self.assertEqual(parsed_paths, [])
        self.assertEqual(result["duplicates"][0]["kept"], "existing_p.pdf")

    def test_grobid_without_a_header_falls_back_to_docling_first_pages(self):
        manifest_mod.add(self.paper_p.document)
        dup_pdf = os.path.join(self.pulled_dir, "dup.pdf")
        with open(dup_pdf, "wb") as f:
            f.write(b"%PDF dup")

        read_header_kwargs = []
        p_header = PaperHeader(title="Half-Metallic Graphene Nanoribbons", first_authors=("Son",), doi=None, arxiv_id=None)

        def fake_read_header(path, *a, **k):
            read_header_kwargs.append(dict(k))
            if k.get("convert_if_missing"):
                return p_header
            return None

        parsed_paths = []
        def fake_process(path, label, *a, **k):
            parsed_paths.append(path)
            return [{"document": "dup.pdf", "text": "chunk"}]

        with patch("research_assistant.shared.ingestion.is_grobid_alive", return_value=True), \
             patch("research_assistant.shared.ingestion.read_header", side_effect=fake_read_header), \
             patch("research_assistant.shared.ingestion.process_pdf", side_effect=fake_process), \
             patch("research_assistant.shared.ingestion.upsert_corpus", return_value=1), \
             patch("research_assistant.shared.ingestion.rebuild_bm25"):
            result = ing.ingest_pdfs([dup_pdf])

        self.assertEqual(parsed_paths, [])
        self.assertTrue(any(k.get("convert_if_missing") for k in read_header_kwargs))

    def test_readable_grobid_header_never_converts(self):
        dup_pdf = os.path.join(self.pulled_dir, "dup.pdf")
        with open(dup_pdf, "wb") as f:
            f.write(b"%PDF dup")

        read_header_kwargs = []
        p_header = PaperHeader(title="Half-Metallic Graphene Nanoribbons", first_authors=("Son",), doi=None, arxiv_id=None)

        def fake_read_header(path, *a, **k):
            read_header_kwargs.append(dict(k))
            return p_header

        with patch("research_assistant.shared.ingestion.is_grobid_alive", return_value=True), \
             patch("research_assistant.shared.ingestion.read_header", side_effect=fake_read_header), \
             patch("research_assistant.shared.ingestion.process_pdf", return_value=[]), \
             patch("research_assistant.shared.ingestion.upsert_corpus", return_value=1), \
             patch("research_assistant.shared.ingestion.rebuild_bm25"):
            ing.ingest_pdfs([dup_pdf])

        self.assertFalse(any(k.get("convert_if_missing") for k in read_header_kwargs))

    def test_pdf_with_a_docling_cache_is_not_converted_again(self):
        pdf = os.path.join(self.pulled_dir, "cached.pdf")
        with open(pdf, "wb") as f:
            f.write(b"%PDF cached")

        cache_dir = os.path.join(self.tmp.name, "docling_cache")
        os.makedirs(cache_dir, exist_ok=True)

        read_header_kwargs = []
        def fake_read_header(path, *a, **k):
            read_header_kwargs.append(dict(k))
            return None

        parsed_paths = []
        def fake_process(path, label, *a, **k):
            parsed_paths.append(path)
            return [{"document": "cached.pdf", "text": "chunk"}]

        from research_assistant import config
        from research_assistant.shared import docling_extract
        with patch.object(config, "DOCLING_CACHE_DIR", cache_dir):
            cache_file = docling_extract._docling_cache_path(pdf)
            with open(cache_file, "w") as f:
                f.write("{}")

            with patch("research_assistant.shared.ingestion.is_grobid_alive", return_value=True), \
                 patch("research_assistant.shared.ingestion.read_header", side_effect=fake_read_header), \
                 patch("research_assistant.shared.ingestion.process_pdf", side_effect=fake_process), \
                 patch("research_assistant.shared.ingestion.upsert_corpus", return_value=1), \
                 patch("research_assistant.shared.ingestion.rebuild_bm25"):
                result = ing.ingest_pdfs([pdf])

        self.assertFalse(any(k.get("convert_if_missing") for k in read_header_kwargs))
        self.assertEqual(parsed_paths, [pdf])

    def test_unreadable_header_is_parsed(self):
        pdf = os.path.join(self.pulled_dir, "unreadable.pdf")
        with open(pdf, "wb") as f:
            f.write(b"%PDF unreadable")

        parsed_paths = []
        def fake_process(path, label, *a, **k):
            parsed_paths.append(path)
            return [{"document": "unreadable.pdf", "text": "chunk"}]

        with patch("research_assistant.shared.ingestion.is_grobid_alive", return_value=True), \
             patch("research_assistant.shared.ingestion.read_header", return_value=None), \
             patch("research_assistant.shared.ingestion.process_pdf", side_effect=fake_process), \
             patch("research_assistant.shared.ingestion.upsert_corpus", return_value=1), \
             patch("research_assistant.shared.ingestion.rebuild_bm25"):
            result = ing.ingest_pdfs([pdf])

        self.assertEqual(parsed_paths, [pdf])
        self.assertEqual(result["duplicates"], [])

    def test_post_parse_check_reuses_pre_parse_headers(self):
        pdf = os.path.join(self.pulled_dir, "unique.pdf")
        with open(pdf, "wb") as f:
            f.write(b"%PDF unique")

        header = PaperHeader(title="Novel 2D Physics", first_authors=("Geim",), doi=None, arxiv_id=None)

        call_paths = []
        def fake_read_header(path, *a, **k):
            call_paths.append(path)
            return header

        def fake_process(path, label, *a, **k):
            return [{"document": "unique.pdf", "text": "chunk"}]

        with patch("research_assistant.shared.ingestion.is_grobid_alive", return_value=True), \
             patch("research_assistant.shared.ingestion.read_header", side_effect=fake_read_header), \
             patch("research_assistant.shared.ingestion.process_pdf", side_effect=fake_process), \
             patch("research_assistant.shared.ingestion.upsert_corpus", return_value=1), \
             patch("research_assistant.shared.ingestion.rebuild_bm25"):
            ing.ingest_pdfs([pdf])

        self.assertEqual(call_paths.count(pdf), 1)

    def test_batch_of_only_duplicates_parses_and_stores_nothing(self):
        manifest_mod.add(self.paper_p.document)
        dup_pdf = os.path.join(self.pulled_dir, "dup.pdf")
        with open(dup_pdf, "wb") as f:
            f.write(b"%PDF dup")

        dup_header = PaperHeader(title="Half-Metallic Graphene Nanoribbons", first_authors=("Son",), doi=None, arxiv_id=None)

        parsed_paths = []
        def fake_process(path, label, *a, **k):
            parsed_paths.append(path)
            return [{"document": "dup.pdf", "text": "chunk"}]

        upsert_calls = 0
        def fake_upsert(corpus):
            nonlocal upsert_calls
            upsert_calls += 1
            return len(corpus)

        with patch("research_assistant.shared.ingestion.is_grobid_alive", return_value=True), \
             patch("research_assistant.shared.ingestion.read_header", return_value=dup_header), \
             patch("research_assistant.shared.ingestion.process_pdf", side_effect=fake_process), \
             patch("research_assistant.shared.ingestion.upsert_corpus", side_effect=fake_upsert), \
             patch("research_assistant.shared.ingestion.rebuild_bm25"):
            result = ing.ingest_pdfs([dup_pdf])

        self.assertEqual(parsed_paths, [])
        self.assertEqual(upsert_calls, 0)
        self.assertEqual(len(result["duplicates"]), 1)

    def test_grobid_down_skips_the_pre_parse_check(self):
        manifest_mod.add(self.paper_p.document)
        dup_pdf = os.path.join(self.pulled_dir, "dup.pdf")
        with open(dup_pdf, "wb") as f:
            f.write(b"%PDF dup")

        dup_header = PaperHeader(title="Half-Metallic Graphene Nanoribbons", first_authors=("Son",), doi=None, arxiv_id=None)

        read_header_kwargs = []
        def fake_read_header(path, *a, **k):
            read_header_kwargs.append(dict(k))
            return dup_header

        parsed_paths = []
        def fake_process(path, label, *a, **k):
            parsed_paths.append(path)
            return [{"document": "dup.pdf", "text": "chunk"}]

        with patch("research_assistant.shared.ingestion.is_grobid_alive", return_value=False), \
             patch("research_assistant.shared.ingestion.read_header", side_effect=fake_read_header), \
             patch("research_assistant.shared.ingestion.process_pdf", side_effect=fake_process), \
             patch("research_assistant.shared.ingestion.upsert_corpus", return_value=1), \
             patch("research_assistant.shared.ingestion.rebuild_bm25"):
            result = ing.ingest_pdfs([dup_pdf])

        self.assertEqual(parsed_paths, [dup_pdf])
        self.assertFalse(any(k.get("convert_if_missing") for k in read_header_kwargs))
        self.assertEqual(len(result["duplicates"]), 1)

    def test_cancel_during_header_check_parses_nothing(self):
        pdf = os.path.join(self.pulled_dir, "cand.pdf")
        with open(pdf, "wb") as f:
            f.write(b"%PDF cand")

        parsed_paths = []
        def fake_process(path, label, *a, **k):
            parsed_paths.append(path)
            return [{"document": "cand.pdf", "text": "chunk"}]

        with patch("research_assistant.shared.ingestion.is_grobid_alive", return_value=True), \
             patch("research_assistant.shared.pipeline_status.is_cancel_requested", return_value=True), \
             patch("research_assistant.shared.ingestion.process_pdf", side_effect=fake_process), \
             patch("research_assistant.shared.ingestion.upsert_corpus", return_value=1), \
             patch("research_assistant.shared.ingestion.rebuild_bm25"):
            result = ing.ingest_pdfs([pdf])

        self.assertEqual(parsed_paths, [])
        self.assertFalse(manifest_mod.load())

    def test_force_reingest_of_two_stored_copies_keeps_one(self):
        x_path = os.path.join(self.pulled_dir, "title_x.pdf")
        with open(x_path, "wb") as f:
            f.write(b"%PDF-1.4 paper X")
        y_path = os.path.join(self.pulled_dir, "title_y.pdf")
        with open(y_path, "wb") as f:
            f.write(b"%PDF-1.4 paper Y")

        idx = PaperIndex(self.index_path)
        idx.add(PaperEntry(
            document="title_x.pdf",
            path=x_path,
            title="Spin Transport in Graphene Bilayers",
            first_authors=("Tombros",),
        ))
        idx.add(PaperEntry(
            document="title_y.pdf",
            path=y_path,
            title="Spin Transport in Graphene Bilayers",
            first_authors=("Tombros",),
        ))
        idx.save()

        manifest_mod.add("title_x.pdf")
        manifest_mod.add("title_y.pdf")

        header = PaperHeader(
            title="Spin Transport in Graphene Bilayers",
            first_authors=("Tombros",),
            doi=None,
            arxiv_id=None,
        )

        parsed_paths = []
        def fake_process(path, label, *a, **k):
            parsed_paths.append(path)
            doc = os.path.basename(path)
            return [{"document": doc, "text": f"chunk {doc}"}]

        with patch("research_assistant.shared.ingestion.is_grobid_alive", return_value=True), \
             patch("research_assistant.shared.ingestion.read_header", return_value=header), \
             patch("research_assistant.shared.ingestion.process_pdf", side_effect=fake_process), \
             patch("research_assistant.shared.ingestion.upsert_corpus", return_value=1), \
             patch("research_assistant.shared.ingestion.rebuild_bm25"):
            result = ing.ingest_pdfs([y_path, x_path], skip_ingested=False)

        self.assertTrue(os.path.exists(x_path))
        self.assertFalse(os.path.exists(y_path))
        self.assertEqual(parsed_paths, [x_path])
        self.assertEqual([(d["document"], d["kept"]) for d in result["duplicates"]], [("title_y.pdf", "title_x.pdf")])
        self.assertEqual(PaperIndex.load(self.index_path).resolve("title_x.pdf").document, "title_x.pdf")


if __name__ == "__main__":
    unittest.main()
