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


if __name__ == "__main__":
    unittest.main()
