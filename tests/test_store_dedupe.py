"""Tests for store duplicate cleanup (shared.store_dedupe)."""

import os
import tempfile
import time
import unittest

from research_assistant.shared import manifest
from research_assistant.shared.store_dedupe import (
    TITLE_MIN,
    Group,
    StoreDoc,
    plan_cleanup,
    title_key,
)


class TestPlanCleanup(unittest.TestCase):
    def test_two_pdfs_same_title_shared_surname_keeps_more_chunks(self):
        title = "A Thorough Study of Graphene Superlattices"
        doc_a = StoreDoc(
            document="paper_a.pdf",
            kind="pdf",
            title=title,
            surnames=frozenset(["zhang"]),
            chunks=59,
            has_doi=False,
        )
        doc_b = StoreDoc(
            document="paper_b.pdf",
            kind="pdf",
            title=title,
            surnames=frozenset(["zhang"]),
            chunks=40,
            has_doi=False,
        )
        groups = plan_cleanup([doc_a, doc_b])
        self.assertEqual(len(groups), 1)
        g = groups[0]
        self.assertEqual(g.keep, "paper_a.pdf")
        self.assertEqual(g.remove, ("paper_b.pdf",))
        self.assertEqual(g.reason, "pdf+pdf")
        self.assertEqual(g.title_key, title_key(title))

    def test_two_pdfs_same_title_no_shared_surname_not_grouped(self):
        title = "A Survey of Large Language Models and Applications"
        doc_a = StoreDoc(
            document="paper_a.pdf",
            kind="pdf",
            title=title,
            surnames=frozenset(["wang"]),
            chunks=50,
            has_doi=True,
        )
        doc_b = StoreDoc(
            document="paper_b.pdf",
            kind="pdf",
            title=title,
            surnames=frozenset(["li"]),
            chunks=60,
            has_doi=True,
        )
        groups = plan_cleanup([doc_a, doc_b])
        self.assertEqual(groups, [])

    def test_chunk_tie_prefers_doi_then_smallest_key(self):
        title = "Topological Insulators and Superconductors"
        # Tie on chunks (50), doc_b has DOI
        doc_a = StoreDoc(
            document="paper_a.pdf",
            kind="pdf",
            title=title,
            surnames=frozenset(["hasan"]),
            chunks=50,
            has_doi=False,
        )
        doc_b = StoreDoc(
            document="paper_b.pdf",
            kind="pdf",
            title=title,
            surnames=frozenset(["hasan"]),
            chunks=50,
            has_doi=True,
        )
        groups = plan_cleanup([doc_a, doc_b])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].keep, "paper_b.pdf")
        self.assertEqual(groups[0].remove, ("paper_a.pdf",))

        # Both have DOI and equal chunks -> smallest document key wins
        doc_c = StoreDoc(
            document="paper_c.pdf",
            kind="pdf",
            title=title,
            surnames=frozenset(["hasan"]),
            chunks=50,
            has_doi=True,
        )
        groups2 = plan_cleanup([doc_c, doc_b])
        self.assertEqual(len(groups2), 1)
        self.assertEqual(groups2[0].keep, "paper_b.pdf")  # "paper_b.pdf" < "paper_c.pdf"
        self.assertEqual(groups2[0].remove, ("paper_c.pdf",))

    def test_pdf_and_stubs_keep_pdf(self):
        title = "Universal Quantum Computing with Anyons"
        pdf = StoreDoc(
            document="paper.pdf",
            kind="pdf",
            title=title,
            surnames=frozenset(["kitaev"]),
            chunks=67,
            has_doi=True,
        )
        stub_x = StoreDoc(
            document="doi:x",
            kind="stub",
            title=title,
            surnames=frozenset(),
            chunks=1,
            has_doi=True,
        )
        stub_y = StoreDoc(
            document="raw:y",
            kind="stub",
            title=title,
            surnames=frozenset(),
            chunks=1,
            has_doi=False,
        )
        groups = plan_cleanup([pdf, stub_y, stub_x])
        self.assertEqual(len(groups), 1)
        g = groups[0]
        self.assertEqual(g.keep, "paper.pdf")
        self.assertEqual(g.remove, ("doi:x", "raw:y"))  # sorted
        self.assertEqual(g.reason, "pdf+stub+stub")

    def test_stubs_keep_doi_key(self):
        title = "Machine Learning for Quantum Matter"
        # raw:a (has_doi) + doi:b -> keep doi:b
        s_raw_a = StoreDoc(
            document="raw:a",
            kind="stub",
            title=title,
            surnames=frozenset(),
            chunks=1,
            has_doi=True,
        )
        s_doi_b = StoreDoc(
            document="doi:b",
            kind="stub",
            title=title,
            surnames=frozenset(),
            chunks=1,
            has_doi=False,
        )
        groups1 = plan_cleanup([s_raw_a, s_doi_b])
        self.assertEqual(len(groups1), 1)
        self.assertEqual(groups1[0].keep, "doi:b")
        self.assertEqual(groups1[0].remove, ("raw:a",))
        self.assertEqual(groups1[0].reason, "stub+stub")

        # raw:a (has_doi) + raw:b (no doi) -> keep raw:a
        s_raw_b = StoreDoc(
            document="raw:b",
            kind="stub",
            title=title,
            surnames=frozenset(),
            chunks=1,
            has_doi=False,
        )
        groups2 = plan_cleanup([s_raw_a, s_raw_b])
        self.assertEqual(len(groups2), 1)
        self.assertEqual(groups2[0].keep, "raw:a")
        self.assertEqual(groups2[0].remove, ("raw:b",))

    def test_short_or_missing_title_never_grouped(self):
        s1 = StoreDoc(document="s1", kind="stub", title="Introduction", surnames=frozenset(), chunks=1, has_doi=False)
        s2 = StoreDoc(document="s2", kind="stub", title="Introduction", surnames=frozenset(), chunks=1, has_doi=False)
        self.assertEqual(plan_cleanup([s1, s2]), [])

        s_none1 = StoreDoc(document="n1", kind="stub", title=None, surnames=frozenset(), chunks=1, has_doi=False)
        s_none2 = StoreDoc(document="n2", kind="stub", title=None, surnames=frozenset(), chunks=1, has_doi=False)
        self.assertEqual(plan_cleanup([s_none1, s_none2]), [])

    def test_same_doi_different_titles_not_grouped(self):
        p1 = StoreDoc(
            document="p1.pdf",
            kind="pdf",
            title="Lipschitz Generative Adversarial Networks",
            surnames=frozenset(["smith"]),
            chunks=20,
            has_doi=True,
        )
        p2 = StoreDoc(
            document="p2.pdf",
            kind="pdf",
            title="Deep Learning for Autonomous Vehicles",
            surnames=frozenset(["smith"]),
            chunks=30,
            has_doi=True,
        )
        self.assertEqual(plan_cleanup([p1, p2]), [])

    def test_keep_override(self):
        title = "A Thorough Study of Graphene Superlattices"
        doc_a = StoreDoc(
            document="paper_a.pdf",
            kind="pdf",
            title=title,
            surnames=frozenset(["zhang"]),
            chunks=59,
            has_doi=False,
        )
        doc_b = StoreDoc(
            document="paper_b.pdf",
            kind="pdf",
            title=title,
            surnames=frozenset(["zhang"]),
            chunks=40,
            has_doi=False,
        )
        # Even though doc_a has more chunks, keep forces paper_b.pdf
        groups = plan_cleanup([doc_a, doc_b], keep=["paper_b.pdf"])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].keep, "paper_b.pdf")
        self.assertEqual(groups[0].remove, ("paper_a.pdf",))

    def test_skip_override(self):
        title = "A Thorough Study of Graphene Superlattices"
        doc_a = StoreDoc(
            document="paper_a.pdf",
            kind="pdf",
            title=title,
            surnames=frozenset(["zhang"]),
            chunks=59,
            has_doi=False,
        )
        doc_b = StoreDoc(
            document="paper_b.pdf",
            kind="pdf",
            title=title,
            surnames=frozenset(["zhang"]),
            chunks=40,
            has_doi=False,
        )
        # skip paper_a.pdf drops the whole group
        groups = plan_cleanup([doc_a, doc_b], skip=["paper_a.pdf"])
        self.assertEqual(groups, [])

    def test_drop_makes_single_group(self):
        groups = plan_cleanup([], drop=["doi:10.1000/999"])
        self.assertEqual(len(groups), 1)
        g = groups[0]
        self.assertEqual(g.keep, "")
        self.assertEqual(g.remove, ("doi:10.1000/999",))
        self.assertEqual(g.reason, "drop")


class TestManifestDiscard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.manifest_path = os.path.join(self.tmp.name, "ingested.json")
        self.patch = unittest.mock.patch.object(manifest, "MANIFEST_PATH", self.manifest_path)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_discard_many(self):
        manifest.add("a")
        manifest.add("b")
        self.assertEqual(manifest.load().keys(), {"a", "b"})

        count = manifest.discard_many(["a", "zz"])
        self.assertEqual(count, 1)
        self.assertEqual(manifest.load().keys(), {"b"})

        # On no match, returns 0 and does not rewrite file (mtime unchanged)
        mtime_before = os.path.getmtime(self.manifest_path)
        time.sleep(0.01)
        count_zero = manifest.discard_many(["nonexistent", "other"])
        self.assertEqual(count_zero, 0)
        mtime_after = os.path.getmtime(self.manifest_path)
        self.assertEqual(mtime_before, mtime_after)


class TestScanStore(unittest.TestCase):
    def test_scan_store_counts_chunks_and_titles(self):
        from research_assistant.shared.paper_index import PaperEntry, PaperIndex
        from research_assistant.shared.store_dedupe import scan_store

        # Fake collection with 3 chunks for paper_a.pdf, 2 for stub_b, 1 for stub_c
        metas = [
            {"document": "paper_a.pdf", "citation_source": "Wrong Title In Store"},
            {"document": "paper_a.pdf", "citation_source": "Wrong Title In Store"},
            {"document": "paper_a.pdf", "citation_source": "Wrong Title In Store"},
            {"document": "raw:b", "citation_source": "A Study of Quantum Hall Systems"},
            {"document": "raw:b", "citation_source": "A Study of Quantum Hall Systems 2"},
            {"document": "doi:10.1000/c", "citation_source": "Another Unique Paper Title"},
        ]

        class PagedCollection:
            def __init__(self, items, page_size=2):
                self.items = items
                self.page_size = page_size

            def get(self, include=None, limit=None, offset=0):
                offset = offset or 0
                limit = limit or self.page_size
                return {"metadatas": self.items[offset : offset + limit]}

        col = PagedCollection(metas, page_size=2)

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        idx_path = os.path.join(tmp.name, "index.json")
        idx = PaperIndex(idx_path)
        idx.add(
            PaperEntry(
                document="paper_a.pdf",
                path="/p/paper_a.pdf",
                title="A Proper Title From Index",
                first_authors=("Zhang",),
                dois=("10.1000/a",),
            )
        )
        idx.save()

        downloaded = {
            "raw:b": {"doi": "10.1000/b_doi", "title": "A Study of Quantum Hall Systems"},
        }

        docs = scan_store(col, idx, downloaded)
        doc_map = {d.document: d for d in docs}

        self.assertEqual(len(docs), 3)

        # paper_a.pdf
        doc_a = doc_map["paper_a.pdf"]
        self.assertEqual(doc_a.kind, "pdf")
        self.assertEqual(doc_a.chunks, 3)
        self.assertEqual(doc_a.title, "A Proper Title From Index")
        self.assertEqual(doc_a.surnames, frozenset(["zhang"]))
        self.assertTrue(doc_a.has_doi)

        # raw:b
        doc_b = doc_map["raw:b"]
        self.assertEqual(doc_b.kind, "stub")
        self.assertEqual(doc_b.chunks, 2)
        self.assertEqual(doc_b.title, "A Study of Quantum Hall Systems")
        self.assertEqual(doc_b.surnames, frozenset())
        self.assertTrue(doc_b.has_doi)  # from downloaded

        # doi:10.1000/c
        doc_c = doc_map["doi:10.1000/c"]
        self.assertEqual(doc_c.kind, "stub")
        self.assertEqual(doc_c.chunks, 1)
        self.assertEqual(doc_c.title, "Another Unique Paper Title")
        self.assertTrue(doc_c.has_doi)  # from doi: prefix


class TestDedupeStoreScript(unittest.TestCase):
    def test_dry_run_writes_plan_only(self):
        import json
        from unittest.mock import MagicMock, patch
        from scripts import dedupe_store
        from research_assistant.shared.paper_index import PaperEntry, PaperIndex

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)

        plan_path = os.path.join(tmp.name, "plan.json")
        manifest_path = os.path.join(tmp.name, "manifest.json")
        index_path = os.path.join(tmp.name, "index.json")

        # Set up manifest
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"p1.pdf": "t1", "p2.pdf": "t2"}, f)
        mtime_before = os.path.getmtime(manifest_path)

        # Set up index
        idx = PaperIndex(index_path)
        title = "A Shared Title For Two Different PDF Chunks"
        idx.add(PaperEntry(document="p1.pdf", path="/p/p1.pdf", title=title, first_authors=("Smith",), dois=("10.1/1",)))
        idx.add(PaperEntry(document="p2.pdf", path="/p/p2.pdf", title=title, first_authors=("Smith",)))
        idx.save()

        # Fake store collection
        metas = [
            {"document": "p1.pdf", "citation_source": title},
            {"document": "p1.pdf", "citation_source": title},
            {"document": "p2.pdf", "citation_source": title},
        ]
        mock_col = MagicMock()
        mock_col.get.side_effect = [
            {"metadatas": metas},
            {"metadatas": []},
        ]
        mock_col.delete.side_effect = RuntimeError("delete should not be called in dry run!")

        mock_store = MagicMock()
        mock_store.get_collection.return_value = mock_col

        argv = ["--plan", plan_path]

        with patch("research_assistant.shared.db.open_store", return_value=mock_store), \
             patch("research_assistant.config.PAPER_INDEX_PATH", index_path), \
             patch("research_assistant.shared.manifest.MANIFEST_PATH", manifest_path), \
             patch("research_assistant.config.DOWNLOADED_JSON_PATH", os.path.join(tmp.name, "none.json")):
            ret = dedupe_store.main(argv)

        self.assertEqual(ret, 0)
        self.assertTrue(os.path.exists(plan_path))

        with open(plan_path, "r", encoding="utf-8") as f:
            plan = json.load(f)

        self.assertEqual(plan["totals"]["groups"], 1)
        self.assertEqual(plan["totals"]["remove_docs"], 1)
        self.assertEqual(plan["totals"]["remove_pdf_docs"], 1)
        self.assertEqual(plan["totals"]["remove_stub_docs"], 0)
        self.assertEqual(plan["totals"]["remove_chunks"], 1)  # p2.pdf had 1 chunk

        # No delete was called
        mock_col.delete.assert_not_called()

        # Manifest unchanged
        self.assertEqual(os.path.getmtime(manifest_path), mtime_before)

    def test_dry_run_then_apply_same_store_succeeds(self):
        import json
        from contextlib import contextmanager
        from unittest.mock import MagicMock, patch
        from scripts import dedupe_store
        from research_assistant.shared.paper_index import PaperEntry, PaperIndex

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)

        plan_path = os.path.join(tmp.name, "plan.json")
        manifest_path = os.path.join(tmp.name, "manifest.json")
        index_path = os.path.join(tmp.name, "index.json")

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"p1.pdf": "t1", "p2.pdf": "t2"}, f)

        idx = PaperIndex(index_path)
        title = "A Shared Title For Two Different PDF Chunks"
        idx.add(PaperEntry(document="p1.pdf", path="/p/p1.pdf", title=title, first_authors=("Smith",), dois=("10.1/1",)))
        idx.add(PaperEntry(document="p2.pdf", path="/p/p2.pdf", title=title, first_authors=("Smith",)))
        idx.save()

        metas = [
            {"document": "p1.pdf", "citation_source": title},
            {"document": "p2.pdf", "citation_source": title},
        ]

        class FakeCol:
            def __init__(self):
                self.table = MagicMock()
                self.table.version = 1
                self.delete_calls = []

            def get(self, include=None, limit=None, offset=0):
                if offset == 0:
                    return {"metadatas": metas}
                return {"metadatas": []}

            def delete(self, where=None):
                self.delete_calls.append(where)

        papers_col = FakeCol()
        summaries_col = FakeCol()

        mock_store = MagicMock()
        def get_col(name):
            if "summaries" in name:
                return summaries_col
            return papers_col
        mock_store.get_collection.side_effect = get_col

        @contextmanager
        def fake_lock():
            yield

        with patch("research_assistant.shared.db.open_store", return_value=mock_store), \
             patch("research_assistant.config.PAPER_INDEX_PATH", index_path), \
             patch("research_assistant.shared.manifest.MANIFEST_PATH", manifest_path), \
             patch("research_assistant.shared.manifest.rebuild_from_collection", return_value=1), \
             patch("research_assistant.shared.dedupe_sweep.run_sweep"), \
             patch("research_assistant.shared.db.clear_search_cache"), \
             patch("research_assistant.shared.ingestion._ingest_lock", fake_lock), \
             patch("research_assistant.config.DOWNLOADED_JSON_PATH", os.path.join(tmp.name, "none.json")):

            # 1. Dry run
            ret_dry = dedupe_store.main(["--plan", plan_path])
            self.assertEqual(ret_dry, 0)
            self.assertTrue(os.path.exists(plan_path))

            # 2. Apply on the same store
            ret_apply = dedupe_store.main(["--apply", "--plan", plan_path])
            self.assertEqual(ret_apply, 0)

            # Assert fake delete was called for the removed document (p2.pdf) in both tables
            expected_where = {"document": "p2.pdf"}
            self.assertIn(expected_where, papers_col.delete_calls)
            self.assertIn(expected_where, summaries_col.delete_calls)


class TestApplyCleanup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.manifest_path = os.path.join(self.tmp.name, "manifest.json")
        self.index_path = os.path.join(self.tmp.name, "index.json")
        self.quarantine_dir = os.path.join(self.tmp.name, "quarantine")
        self.patch_manifest = unittest.mock.patch("research_assistant.shared.manifest.MANIFEST_PATH", self.manifest_path)
        self.patch_manifest.start()
        self.addCleanup(self.patch_manifest.stop)

    def test_apply_deletes_both_tables_and_updates_manifest_and_index(self):
        import json
        from unittest.mock import MagicMock
        from research_assistant.shared import manifest
        from research_assistant.shared.paper_index import PaperEntry, PaperIndex
        from research_assistant.shared.store_dedupe import Group, apply_cleanup

        manifest.add("k.pdf")
        manifest.add("r.pdf")

        idx = PaperIndex(self.index_path)
        idx.add(PaperEntry(document="k.pdf", path="/p/k.pdf", title="Title K"))
        idx.add(PaperEntry(document="r.pdf", path="/p/r.pdf", title="Title R"))
        idx.save()

        papers = MagicMock()
        papers.table.version = 10
        summaries = MagicMock()
        summaries.table.version = 5

        g = Group(title_key="title", keep="k.pdf", remove=("r.pdf",), reason="pdf+pdf")
        res = apply_cleanup([g], papers, summaries, self.index_path, self.quarantine_dir)

        papers.delete.assert_called_once_with(where={"document": "r.pdf"})
        summaries.delete.assert_called_once_with(where={"document": "r.pdf"})

        self.assertNotIn("r.pdf", manifest.load())
        self.assertIn("k.pdf", manifest.load())

        idx_after = PaperIndex.load(self.index_path)
        self.assertIsNone(idx_after.get("r.pdf"))
        self.assertEqual(idx_after.resolve("r.pdf").document, "k.pdf")

        self.assertEqual(res["versions_before"], {"papers": 10, "summaries": 5})
        self.assertEqual(res["removed"], ["r.pdf"])
        self.assertEqual(res["errors"], [])

    def test_apply_moves_pdf_to_quarantine_never_deletes_file(self):
        import os
        from unittest.mock import MagicMock
        from research_assistant.shared import manifest
        from research_assistant.shared.paper_index import PaperEntry, PaperIndex
        from research_assistant.shared.store_dedupe import Group, apply_cleanup

        manifest.add("r.pdf")
        pdf_file = os.path.join(self.tmp.name, "r.pdf")
        with open(pdf_file, "wb") as f:
            f.write(b"PDF CONTENT FOR QUARANTINE")

        idx = PaperIndex(self.index_path)
        idx.add(PaperEntry(document="r.pdf", path=pdf_file, title="Title"))
        idx.save()

        papers = MagicMock()
        papers.table.version = 1
        summaries = MagicMock()
        summaries.table.version = 1

        g = Group(title_key="title", keep="k.pdf", remove=("r.pdf",), reason="pdf+pdf")
        res = apply_cleanup([g], papers, summaries, self.index_path, self.quarantine_dir)

        self.assertFalse(os.path.exists(pdf_file))
        self.assertEqual(len(res["moved"]), 1)
        src, dst = res["moved"][0]
        self.assertEqual(src, pdf_file)
        self.assertTrue(os.path.exists(dst))
        with open(dst, "rb") as f:
            self.assertEqual(f.read(), b"PDF CONTENT FOR QUARANTINE")

    def test_delete_error_leaves_manifest_and_index_for_that_doc(self):
        from unittest.mock import MagicMock
        from research_assistant.shared import manifest
        from research_assistant.shared.paper_index import PaperEntry, PaperIndex
        from research_assistant.shared.store_dedupe import Group, apply_cleanup

        manifest.add("fail.pdf")
        pdf_file = os.path.join(self.tmp.name, "fail.pdf")
        with open(pdf_file, "wb") as f:
            f.write(b"NOT TO BE DELETED")

        idx = PaperIndex(self.index_path)
        idx.add(PaperEntry(document="fail.pdf", path=pdf_file, title="Title"))
        idx.save()

        papers = MagicMock()
        papers.table.version = 1
        papers.delete.side_effect = RuntimeError("LanceDB locked")
        summaries = MagicMock()
        summaries.table.version = 1

        g = Group(title_key="title", keep="k.pdf", remove=("fail.pdf",), reason="pdf+pdf")
        res = apply_cleanup([g], papers, summaries, self.index_path, self.quarantine_dir)

        self.assertEqual(len(res["errors"]), 1)
        self.assertEqual(res["errors"][0]["document"], "fail.pdf")
        self.assertIn("fail.pdf", manifest.load())
        idx_after = PaperIndex.load(self.index_path)
        self.assertIsNotNone(idx_after.get("fail.pdf"))
        self.assertTrue(os.path.exists(pdf_file))

    def test_drop_group_removes_index_entry_without_alias(self):
        from unittest.mock import MagicMock
        from research_assistant.shared import manifest
        from research_assistant.shared.paper_index import PaperEntry, PaperIndex
        from research_assistant.shared.store_dedupe import Group, apply_cleanup

        manifest.add("drop.pdf")
        idx = PaperIndex(self.index_path)
        idx.add(PaperEntry(document="drop.pdf", path="/p/drop.pdf", title="To Drop"))
        idx.save()

        papers = MagicMock()
        papers.table.version = 1
        summaries = MagicMock()
        summaries.table.version = 1

        g = Group(title_key="", keep="", remove=("drop.pdf",), reason="drop")
        apply_cleanup([g], papers, summaries, self.index_path, self.quarantine_dir)

        self.assertNotIn("drop.pdf", manifest.load())
        idx_after = PaperIndex.load(self.index_path)
        self.assertIsNone(idx_after.get("drop.pdf"))
        self.assertIsNone(idx_after.resolve("drop.pdf"))

    def test_apply_refuses_when_plan_differs(self):
        import json
        from unittest.mock import MagicMock, patch
        from scripts import dedupe_store
        from research_assistant.shared.paper_index import PaperIndex

        plan_path = os.path.join(self.tmp.name, "plan.json")
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump({
                "groups": [{"title_key": "old", "keep": "a", "remove": ["b"], "reason": "pdf+pdf"}],
                "args": {"keep": [], "skip": [], "drop": [], "plan": plan_path},
            }, f)

        idx = PaperIndex(self.index_path)
        idx.save()

        mock_store = MagicMock()
        mock_col = MagicMock()
        mock_col.get.return_value = {"metadatas": []}
        mock_store.get_collection.return_value = mock_col

        argv = ["--apply", "--plan", plan_path]

        with patch("research_assistant.shared.db.open_store", return_value=mock_store), \
             patch("research_assistant.config.PAPER_INDEX_PATH", self.index_path), \
             patch("research_assistant.config.DOWNLOADED_JSON_PATH", os.path.join(self.tmp.name, "none.json")):
            ret = dedupe_store.main(argv)

        self.assertNotEqual(ret, 0)

    def test_apply_holds_ingest_lock(self):
        import json
        from contextlib import contextmanager
        from unittest.mock import MagicMock, patch
        from scripts import dedupe_store
        from research_assistant.shared.paper_index import PaperIndex

        plan_path = os.path.join(self.tmp.name, "plan.json")
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump({
                "groups": [],
                "args": {"keep": [], "skip": [], "drop": [], "plan": plan_path},
            }, f)

        idx = PaperIndex(self.index_path)
        idx.save()

        mock_store = MagicMock()
        mock_col = MagicMock()
        mock_col.get.return_value = {"metadatas": []}
        mock_store.get_collection.return_value = mock_col

        lock_acquired = []

        @contextmanager
        def fake_lock():
            lock_acquired.append(True)
            yield
            lock_acquired.append(False)

        argv = ["--apply", "--plan", plan_path]

        with patch("research_assistant.shared.db.open_store", return_value=mock_store), \
             patch("research_assistant.config.PAPER_INDEX_PATH", self.index_path), \
             patch("research_assistant.shared.manifest.rebuild_from_collection", return_value=0), \
             patch("research_assistant.shared.dedupe_sweep.run_sweep"), \
             patch("research_assistant.shared.db.clear_search_cache"), \
             patch("research_assistant.shared.ingestion._ingest_lock", fake_lock), \
             patch("research_assistant.config.DOWNLOADED_JSON_PATH", os.path.join(self.tmp.name, "none.json")):
            ret = dedupe_store.main(argv)

        self.assertEqual(ret, 0)
        self.assertEqual(lock_acquired, [True, False])

    def test_move_failure_is_recorded_and_apply_continues(self):
        import shutil
        from unittest.mock import MagicMock, patch
        from research_assistant.shared import manifest
        from research_assistant.shared.paper_index import PaperEntry, PaperIndex
        from research_assistant.shared.store_dedupe import Group, apply_cleanup

        manifest.add("k.pdf")
        manifest.add("r1.pdf")
        manifest.add("r2.pdf")

        r1_path = os.path.join(self.tmp.name, "r1.pdf")
        with open(r1_path, "wb") as f:
            f.write(b"R1 CONTENT")

        r2_path = os.path.join(self.tmp.name, "r2.pdf")
        with open(r2_path, "wb") as f:
            f.write(b"R2 CONTENT")

        idx = PaperIndex(self.index_path)
        idx.add(PaperEntry(document="k.pdf", path="/p/k.pdf", title="Title"))
        idx.add(PaperEntry(document="r1.pdf", path=r1_path, title="Title"))
        idx.add(PaperEntry(document="r2.pdf", path=r2_path, title="Title"))
        idx.save()

        papers = MagicMock()
        papers.table.version = 1
        summaries = MagicMock()
        summaries.table.version = 1

        g = Group(title_key="title", keep="k.pdf", remove=("r1.pdf", "r2.pdf"), reason="pdf+pdf+pdf")

        real_move = shutil.move
        def fake_move(src, dst):
            if src == r1_path:
                raise OSError("Disk full")
            return real_move(src, dst)

        with patch("shutil.move", side_effect=fake_move):
            res = apply_cleanup([g], papers, summaries, self.index_path, self.quarantine_dir)

        # Assert errors has one entry with stage == "move" for r1.pdf
        self.assertEqual(len(res["errors"]), 1)
        self.assertEqual(res["errors"][0]["document"], "r1.pdf")
        self.assertEqual(res["errors"][0]["stage"], "move")

        # Assert the first PDF is still at its original path
        self.assertTrue(os.path.exists(r1_path))

        # Assert the second is still removed and moved
        self.assertIn("r2.pdf", res["removed"])
        self.assertFalse(os.path.exists(r2_path))
        quarantined_r2 = os.path.join(self.quarantine_dir, "r2.pdf")
        self.assertTrue(os.path.exists(quarantined_r2))
        self.assertEqual(len(res["moved"]), 1)
        self.assertEqual(res["moved"][0], [r2_path, quarantined_r2])


if __name__ == "__main__":
    unittest.main()
