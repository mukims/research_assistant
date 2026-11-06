"""Tests for the corpus paper index (shared.paper_index)."""

import json
import os
import tempfile
import threading
import unittest

from research_assistant.shared.paper_index import (
    PaperEntry,
    PaperIndex,
    edit,
    normalise_arxiv,
    repoint_manifest,
    surname_key,
)


class TestPaperIndex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.index_path = os.path.join(self.tmp.name, "paper_index.json")

    def test_find_by_ids_matches_doi_regardless_of_prefix_and_case(self):
        idx = PaperIndex(self.index_path)
        entry = PaperEntry(
            document="doi_10.1038_nature05180.pdf",
            path="/corpus/nature05180.pdf",
            dois=("10.1038/nature05180",),
        )
        idx.add(entry)
        match = idx.find_by_ids(doi="https://doi.org/10.1038/NATURE05180")
        self.assertIsNotNone(match)
        self.assertEqual(match.document, "doi_10.1038_nature05180.pdf")

    def test_find_by_ids_matches_arxiv_across_versions(self):
        idx = PaperIndex(self.index_path)
        entry = PaperEntry(
            document="arxiv_2007.12504.pdf",
            path="/corpus/arxiv_2007.12504.pdf",
            arxiv_ids=("2007.12504",),
        )
        idx.add(entry)
        match = idx.find_by_ids(arxiv_id="arXiv:2007.12504v2")
        self.assertIsNotNone(match)
        self.assertEqual(match.document, "arxiv_2007.12504.pdf")

    def test_find_by_ids_with_no_ids_finds_nothing(self):
        idx = PaperIndex(self.index_path)
        entry = PaperEntry(
            document="no_ids.pdf",
            path="/corpus/no_ids.pdf",
            title="Some Title",
        )
        idx.add(entry)
        self.assertIsNone(idx.find_by_ids(None, None))

    def test_title_author_match_ignores_case_punctuation_and_filler_words(self):
        idx = PaperIndex(self.index_path)
        entry = PaperEntry(
            document="son2006.pdf",
            path="/corpus/son2006.pdf",
            title="Half-Metallic Graphene Nanoribbons",
            first_authors=("Son",),
        )
        idx.add(entry)
        match = idx.find_by_title_author("half metallic graphene nanoribbons.", ["SON"])
        self.assertIsNotNone(match)
        self.assertEqual(match.document, "son2006.pdf")

    def test_same_title_different_first_author_is_not_a_match(self):
        idx = PaperIndex(self.index_path)
        entry = PaperEntry(
            document="son2006.pdf",
            path="/corpus/son2006.pdf",
            title="Half-Metallic Graphene Nanoribbons",
            first_authors=("Son",),
        )
        idx.add(entry)
        match = idx.find_by_title_author("Half-Metallic Graphene Nanoribbons", ["Geim"])
        self.assertIsNone(match)

    def test_title_or_author_missing_never_matches(self):
        idx = PaperIndex(self.index_path)
        entry = PaperEntry(
            document="son2006.pdf",
            path="/corpus/son2006.pdf",
            title="Half-Metallic Graphene Nanoribbons",
            first_authors=("Son",),
        )
        idx.add(entry)
        self.assertIsNone(idx.find_by_title_author("Half-Metallic Graphene Nanoribbons", []))
        self.assertIsNone(idx.find_by_title_author(None, ["Son"]))

    def test_any_shared_surname_matches(self):
        idx = PaperIndex(self.index_path)
        entry = PaperEntry(
            document="zaghouani.pdf",
            path="/corpus/zaghouani.pdf",
            title="Quantum Transport",
            first_authors=("Zaghouani", "Müller"),
        )
        idx.add(entry)
        match = idx.find_by_title_author("Quantum Transport", ["Muller"])
        self.assertIsNotNone(match)
        self.assertEqual(match.document, "zaghouani.pdf")

    def test_surname_key_handles_compound_and_encoding_artifacts(self):
        self.assertEqual(surname_key("Essam Ghareeb"), "ghareeb")
        self.assertEqual(surname_key("Ghareeb"), "ghareeb")
        self.assertEqual(surname_key("Kulyt˙e"), "kulyte")
        self.assertEqual(surname_key("Kulytė"), "kulyte")
        self.assertIsNone(surname_key(""))

    def test_exclude_skips_the_paper_itself(self):
        idx = PaperIndex(self.index_path)
        entry = PaperEntry(
            document="son2006.pdf",
            path="/corpus/son2006.pdf",
            title="Half-Metallic Graphene Nanoribbons",
            first_authors=("Son",),
        )
        idx.add(entry)
        match = idx.find_by_title_author(
            "Half-Metallic Graphene Nanoribbons",
            ["Son"],
            exclude="son2006.pdf",
        )
        self.assertIsNone(match)

    def test_alias_resolves_to_the_kept_paper(self):
        idx = PaperIndex(self.index_path)
        entry = PaperEntry(
            document="doi_y.pdf",
            path="/corpus/doi_y.pdf",
            title="Some Paper",
        )
        idx.add(entry)
        idx.add_alias("title_x.pdf", "doi_y.pdf")
        resolved = idx.resolve("title_x.pdf")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.document, "doi_y.pdf")

    def test_round_trips_through_disk(self):
        with edit(self.index_path) as idx:
            idx.add(PaperEntry(
                document="doc1.pdf",
                path="/tmp/doc1.pdf",
                title="Doc 1",
                first_authors=("Author",),
                dois=("10.1000/1",),
                arxiv_ids=("2001.00001",),
            ))
            idx.add_alias("alias1.pdf", "doc1.pdf")

        loaded = PaperIndex.load(self.index_path)
        self.assertEqual(len(loaded), 1)
        entry = loaded.get("doc1.pdf")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.document, "doc1.pdf")
        self.assertEqual(entry.title, "Doc 1")
        self.assertEqual(entry.first_authors, ("Author",))
        self.assertEqual(entry.dois, ("10.1000/1",))
        self.assertEqual(entry.arxiv_ids, ("2001.00001",))
        resolved = loaded.resolve("alias1.pdf")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.document, "doc1.pdf")

    def test_unreadable_file_loads_as_empty(self):
        with open(self.index_path, "w") as f:
            f.write("{not json")
        loaded = PaperIndex.load(self.index_path)
        self.assertEqual(len(loaded), 0)

    def test_concurrent_edits_lose_nothing(self):
        threads = []
        for i in range(8):
            entry = PaperEntry(
                document=f"doc_{i}.pdf",
                path=f"/corpus/doc_{i}.pdf",
                title=f"Paper {i}",
            )

            def worker(e=entry):
                with edit(self.index_path) as idx:
                    idx.add(e)

            t = threading.Thread(target=worker)
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        loaded = PaperIndex.load(self.index_path)
        self.assertEqual(len(loaded), 8)

    def test_duplicate_groups_reports_shared_doi_and_shared_title_author(self):
        idx = PaperIndex(self.index_path)
        # A, B share a DOI
        idx.add(PaperEntry(document="A", path="/p/A.pdf", dois=("10.1000/same",)))
        idx.add(PaperEntry(document="B", path="/p/B.pdf", dois=("10.1000/same",)))
        # C, D share title+author
        idx.add(PaperEntry(
            document="C",
            path="/p/C.pdf",
            title="Shared Title For CD",
            first_authors=("Smith",),
        ))
        idx.add(PaperEntry(
            document="D",
            path="/p/D.pdf",
            title="Shared Title For CD",
            first_authors=("Smith",),
        ))
        # E is unique
        idx.add(PaperEntry(
            document="E",
            path="/p/E.pdf",
            title="Unique Title",
            first_authors=("Jones",),
            dois=("10.1000/unique",),
        ))

        groups = idx.duplicate_groups()
        simplified = [
            {"reason": g["reason"], "documents": g["documents"]}
            for g in groups
        ]
        expected = [
            {"reason": "doi", "documents": ["A", "B"]},
            {"reason": "title_author", "documents": ["C", "D"]},
        ]
        # Sort simplified and expected by reason to compare deterministically
        self.assertEqual(
            sorted(simplified, key=lambda x: (x["reason"], x["documents"])),
            sorted(expected, key=lambda x: (x["reason"], x["documents"])),
        )

    def test_repoint_manifest_moves_matching_records_only(self):
        manifest_path = os.path.join(self.tmp.name, "downloaded.json")
        data = {
            "key1": {"path": "/old/dir/target.pdf", "title": "Paper 1"},
            "key2": {"path": "/old/dir/other.pdf", "title": "Paper 2"},
        }
        with open(manifest_path, "w") as f:
            json.dump(data, f)

        count = repoint_manifest(manifest_path, "/different/dir/target.pdf", "/new/dir/target.pdf")
        self.assertEqual(count, 1)

        with open(manifest_path) as f:
            updated = json.load(f)

        self.assertEqual(updated["key1"]["path"], "/new/dir/target.pdf")
        self.assertEqual(updated["key2"]["path"], "/old/dir/other.pdf")


if __name__ == "__main__":
    unittest.main()
