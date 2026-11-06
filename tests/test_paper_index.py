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
    is_placeholder_doi,
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

    def test_repoint_path_updates_matching_entry_only(self):
        idx = PaperIndex(self.index_path)
        old_path = os.path.join(self.tmp.name, "raw", "target.pdf")
        other_path = os.path.join(self.tmp.name, "raw", "other.pdf")
        new_path = os.path.join(self.tmp.name, "processed", "target.pdf")

        e1 = PaperEntry(document="doc1", path=old_path, title="T1")
        e2 = PaperEntry(document="doc2", path=other_path, title="T2")
        idx.add(e1)
        idx.add(e2)

        count = idx.repoint_path(old_path, new_path)
        self.assertEqual(count, 1)
        self.assertEqual(idx.get("doc1").path, new_path)
        self.assertEqual(idx.get("doc1").title, "T1")
        self.assertEqual(idx.get("doc2").path, other_path)

    def test_placeholder_dois_are_recognised(self):
        placeholders = [
            "10.1007/sxxxxx-yyy-zzzz-1",
            "10.1109/xxx.0000.0000000",
            "10.1145/nnnnnnn.nnnnnnn",
            "10.1145/xxxxxx",
            "10.1364/ao.xx.xxxxxx",
            "10.15607/rss.2024.xx.015",
        ]
        for doi in placeholders:
            self.assertTrue(is_placeholder_doi(doi), f"{doi} should be recognized as placeholder")

        legitimate = [
            "10.1038/nature05180",
            "10.1103/physrevx.10.011001",
            "10.1145/3586183.3606763",
            None,
            "",
        ]
        for doi in legitimate:
            self.assertFalse(is_placeholder_doi(doi), f"{doi} should NOT be recognized as placeholder")

    def test_add_drops_placeholder_dois(self):
        idx = PaperIndex(self.index_path)
        entry = PaperEntry(
            document="doc1",
            path="/corpus/doc1.pdf",
            dois=("10.1145/xxxxxx", "10.1038/nature05180"),
        )
        idx.add(entry)
        saved_entry = idx.get("doc1")
        self.assertIsNotNone(saved_entry)
        self.assertEqual(saved_entry.dois, ("10.1038/nature05180",))

    def test_find_by_ids_ignores_placeholder_doi(self):
        idx = PaperIndex(self.index_path)
        entry = PaperEntry(
            document="doc1",
            path="/corpus/doc1.pdf",
            dois=("10.1145/xxxxxx",),
            arxiv_ids=("2001.00001",),
        )
        idx._papers["doc1"] = entry

        # Placeholder DOI alone does not match
        self.assertIsNone(idx.find_by_ids(doi="10.1145/xxxxxx"))

        # arXiv ID in same call still matches
        match = idx.find_by_ids(doi="10.1145/xxxxxx", arxiv_id="2001.00001")
        self.assertIsNotNone(match)
        self.assertEqual(match.document, "doc1")

    def test_duplicate_groups_skip_placeholder_doi(self):
        data = {
            "version": 1,
            "papers": {
                "A": {"document": "A", "path": "/p/A.pdf", "dois": ["10.1145/nnnnnnn.nnnnnnn"]},
                "B": {"document": "B", "path": "/p/B.pdf", "dois": ["10.1145/nnnnnnn.nnnnnnn"]},
            },
            "aliases": {},
        }
        with open(self.index_path, "w") as f:
            json.dump(data, f)

        idx = PaperIndex.load(self.index_path)
        groups = idx.duplicate_groups()
        doi_groups = [g for g in groups if g["reason"] == "doi"]
        self.assertEqual(doi_groups, [])

    def test_retire_merges_ids_and_aliases(self):
        idx = PaperIndex(self.index_path)
        keeper = PaperEntry(
            document="keeper.pdf",
            path="/corpus/keeper.pdf",
            dois=("10.1000/keeper",),
            arxiv_ids=("2001.00001",),
        )
        retired = PaperEntry(
            document="retired.pdf",
            path="/corpus/retired.pdf",
            dois=("10.1000/retired",),
            arxiv_ids=("2001.00002",),
        )
        idx.add(keeper)
        idx.add(retired)

        idx.retire("retired.pdf", "keeper.pdf")

        # Keeper gains retired entry's DOI and arXiv ID
        updated_keeper = idx.get("keeper.pdf")
        self.assertIsNotNone(updated_keeper)
        self.assertIn("10.1000/retired", updated_keeper.dois)
        self.assertIn("10.1000/keeper", updated_keeper.dois)
        self.assertIn("2001.00001", updated_keeper.arxiv_ids)
        self.assertIn("2001.00002", updated_keeper.arxiv_ids)

        # Retired entry is deleted
        self.assertIsNone(idx.get("retired.pdf"))

        # Alias is registered
        resolved = idx.resolve("retired.pdf")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.document, "keeper.pdf")

        # Dropping an entry with keeper == "" deletes entry without alias
        to_drop = PaperEntry(document="drop.pdf", path="/p/drop.pdf")
        idx.add(to_drop)
        idx.retire("drop.pdf", "")
        self.assertIsNone(idx.get("drop.pdf"))
        self.assertIsNone(idx.resolve("drop.pdf"))

    def test_retire_missing_is_noop(self):
        idx = PaperIndex(self.index_path)
        # Retiring a missing doc does nothing and raises nothing
        idx.retire("missing.pdf", "keeper.pdf")
        self.assertIsNone(idx.get("missing.pdf"))


if __name__ == "__main__":
    unittest.main()
