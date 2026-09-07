"""Tests for Agent 0's seed manifest, which now routes through SeedPaper.

The one thing actually changed by the port: seeds[query] used to be a bare
dict any caller could grow arbitrary keys into (the old **extra splat on
_download_and_record was a direct route for that). Routing both manifest
sites through SeedPaper(...).to_dict() means a malformed record fails at
write time, and the manifest a caller like orchestrate.py reads back is
always a valid SeedPaper shape.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import research_assistant.agents.agent0_discoverer as a0
from research_assistant.schemas import SeedPaper


class TestDownloadAndRecordSeedPaperWiring(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw_dir_patch = patch.object(a0, "RAW_DIR", self.tmp.name)
        self.seed_path_patch = patch.object(
            a0, "SEED_PAPERS_PATH", os.path.join(self.tmp.name, "seed_papers.json")
        )
        self.raw_dir_patch.start()
        self.seed_path_patch.start()
        self.addCleanup(self.raw_dir_patch.stop)
        self.addCleanup(self.seed_path_patch.stop)

    def test_manifest_record_round_trips_through_seed_paper(self):
        with patch.object(a0, "download_pdf", return_value=(True, None)):
            dest = a0._download_and_record(
                "some query", "arxiv:1234.5678", "https://arxiv.org/pdf/1234.5678",
                {}, force=False, arxiv_id="1234.5678", source="manual-url",
            )
        self.assertIsNotNone(dest)
        seeds = a0._load_seeds()
        record = seeds["some query"]
        # Must be exactly the shape SeedPaper produces, not a hand-rolled dict:
        # a malformed record would raise here instead of round-tripping.
        self.assertEqual(SeedPaper.from_dict(record).to_dict(), record)
        self.assertEqual(record["source"], "manual-url")
        self.assertEqual(record["arxiv_id"], "1234.5678")

    def test_rejects_undeclared_keyword_arguments(self):
        """The **extra splat is gone — an unknown kwarg must now be a TypeError,
        not a key silently smuggled into the manifest."""
        with patch.object(a0, "download_pdf", return_value=(True, None)):
            with self.assertRaises(TypeError):
                a0._download_and_record(
                    "q", "k", "https://example.com/p.pdf", {}, False,
                    unexpected_field="sneaky",
                )


class TestDiscoverFromFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw_dir_patch = patch.object(a0, "RAW_DIR", os.path.join(self.tmp.name, "raw"))
        self.seed_path_patch = patch.object(
            a0, "SEED_PAPERS_PATH", os.path.join(self.tmp.name, "seed_papers.json")
        )
        self.raw_dir_patch.start()
        self.seed_path_patch.start()
        self.addCleanup(self.raw_dir_patch.stop)
        self.addCleanup(self.seed_path_patch.stop)

    def test_discover_from_file_with_query(self):
        # Create a mock valid PDF file
        pdf_content = b"%PDF-1.4 mock content with arxiv:2401.99999 and /Title (Quantum Wires)"
        pdf_path = os.path.join(self.tmp.name, "sample_paper.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_content)

        dest = a0.discover_from_file("topological wires", pdf_path)
        self.assertIsNotNone(dest)
        self.assertTrue(os.path.exists(dest))
        self.assertTrue(dest.startswith(os.path.join(self.tmp.name, "raw")))

        seeds = a0._load_seeds()
        self.assertIn("topological wires", seeds)
        rec = seeds["topological wires"]
        self.assertEqual(rec["source"], "upload")
        self.assertEqual(rec["arxiv_id"], "2401.99999")
        self.assertEqual(rec["title"], "Quantum Wires")
        # Validate SeedPaper schema compliance
        self.assertEqual(SeedPaper.from_dict(rec).to_dict(), rec)

        # Verify get_seed and get_seed_by_path
        self.assertEqual(a0.get_seed("topological wires"), rec)
        found_rec, q = a0.get_seed_by_path(dest)
        self.assertEqual(found_rec, rec)
        self.assertEqual(q, "topological wires")

    def test_discover_from_file_without_query_infers_topic(self):
        # PDF with Title metadata but no user query
        pdf_content = b"%PDF-1.4 mock content with /Title (Superconducting Qubits In Silicon)"
        pdf_path = os.path.join(self.tmp.name, "spin_qubits_2024.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_content)

        dest = a0.discover_from_file("", pdf_path)
        self.assertIsNotNone(dest)
        seeds = a0._load_seeds()
        # Query inferred from title
        self.assertIn("Superconducting Qubits In Silicon", seeds)
        rec = seeds["Superconducting Qubits In Silicon"]
        self.assertEqual(rec["source"], "upload")
        self.assertEqual(rec["title"], "Superconducting Qubits In Silicon")

        found_rec, q = a0.get_seed_by_path(dest)
        self.assertEqual(q, "Superconducting Qubits In Silicon")

    def test_discover_from_file_fallback_to_filename_stem(self):
        # PDF with no title metadata
        pdf_content = b"%PDF-1.4 binary content with no recognizable title tag"
        pdf_path = os.path.join(self.tmp.name, "majorana_zero_modes.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_content)

        dest = a0.discover_from_file("", pdf_path)
        self.assertIsNotNone(dest)
        seeds = a0._load_seeds()
        self.assertIn("majorana zero modes", seeds)
        rec = seeds["majorana zero modes"]
        self.assertEqual(rec["title"], "majorana zero modes")
        self.assertTrue(rec["key"].startswith("file:"))

    def test_discover_from_file_bytes_input(self):
        pdf_content = b"%PDF-1.4 uploaded from stream directly"
        dest = a0.discover_from_file("my bytes query", pdf_content, filename="direct_stream.pdf")
        self.assertIsNotNone(dest)
        self.assertTrue(os.path.exists(dest))
        seeds = a0._load_seeds()
        self.assertIn("my bytes query", seeds)

    def test_two_papers_with_the_same_inferred_query_both_keep_a_record(self):
        """An inferred query must not silently overwrite another paper's record.

        The seed manifest is keyed by query. When the caller supplies one that
        is fine — re-running a query is meant to re-seed it. But on a blank
        upload the query is *inferred* from the title or the filename stem, and
        two different papers can infer the same string. Both are written to
        RAW_DIR under distinct content-hash keys, so both must survive in the
        manifest too; otherwise the first paper sits on disk with no seed
        record, the UI shows the wrong seed card, and path -> seed lookup
        fails for it.
        """
        a = a0.discover_from_file("", b"%PDF-1.4 first paper body", filename="paper.pdf")
        b = a0.discover_from_file("", b"%PDF-1.4 second paper body", filename="paper.pdf")

        self.assertNotEqual(a, b, "different content should land in different files")
        seeds = a0._load_seeds()
        self.assertEqual(len(seeds), 2)
        self.assertEqual({s["path"] for s in seeds.values()}, {a, b})

    def test_reuploading_the_same_paper_keeps_a_single_record(self):
        """Same bytes means the same key — idempotent, not a second entry."""
        a = a0.discover_from_file("", b"%PDF-1.4 identical body", filename="paper.pdf")
        b = a0.discover_from_file("", b"%PDF-1.4 identical body", filename="paper.pdf")

        self.assertEqual(a, b)
        self.assertEqual(len(a0._load_seeds()), 1)

    def test_a_supplied_query_still_re_seeds_in_place(self):
        """Only the inferred case disambiguates; an explicit query keeps its slot.

        `discover()` and `discover_from_url()` both rely on `seeds[query]` to
        answer "have I already seeded this query?", so a supplied query has to
        stay a stable key.
        """
        a0.discover_from_file("my topic", b"%PDF-1.4 first paper body", filename="a.pdf")
        a0.discover_from_file("my topic", b"%PDF-1.4 second paper body", filename="b.pdf")

        self.assertEqual(list(a0._load_seeds()), ["my topic"])

    def test_discover_from_file_rejects_invalid_pdf(self):
        not_a_pdf = os.path.join(self.tmp.name, "fake.pdf")
        with open(not_a_pdf, "wb") as f:
            f.write(b"<html>This is an html error page, not a pdf</html>")

        dest = a0.discover_from_file("some query", not_a_pdf)
        self.assertIsNone(dest)
        seeds = a0._load_seeds()
        self.assertEqual(seeds, {})

    def test_discover_from_file_nonexistent_path(self):
        dest = a0.discover_from_file("some query", os.path.join(self.tmp.name, "does_not_exist.pdf"))
        self.assertIsNone(dest)

    def test_discover_from_file_hex_title(self):
        # Hex title for 'Topological Wires'
        pdf_content = b"%PDF-1.4 mock content\n/Title <546F706F6C6F676963616C205769726573>\n"
        pdf_path = os.path.join(self.tmp.name, "hex_title.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_content)

        dest = a0.discover_from_file("", pdf_path)
        self.assertIsNotNone(dest)
        seeds = a0._load_seeds()
        self.assertIn("Topological Wires", seeds)
        self.assertEqual(seeds["Topological Wires"]["title"], "Topological Wires")

    def test_discover_from_file_xmp_dc_title_and_doi(self):
        pdf_content = (
            b"%PDF-1.4 mock content\n"
            b"<x:xmpmeta><rdf:RDF>"
            b"<dc:title><rdf:Alt><rdf:li xml:lang='x-default'>Quantum Error Correction</rdf:li></rdf:Alt></dc:title>"
            b"<prism:doi>10.1103/PhysRevA.99.123456</prism:doi>"
            b"</rdf:RDF></x:xmpmeta>"
        )
        pdf_path = os.path.join(self.tmp.name, "xmp_paper.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_content)

        dest = a0.discover_from_file("", pdf_path)
        self.assertIsNotNone(dest)
        seeds = a0._load_seeds()
        self.assertIn("Quantum Error Correction", seeds)
        rec = seeds["Quantum Error Correction"]
        self.assertEqual(rec["title"], "Quantum Error Correction")
        self.assertEqual(rec["doi"], "10.1103/physreva.99.123456")
        self.assertEqual(rec["key"], "doi:10.1103/physreva.99.123456")

    def test_discover_from_file_ignores_unprefixed_doi_in_bytes(self):
        # Random binary containing 10.1234/something should NOT be picked up as paper DOI
        pdf_content = b"%PDF-1.4 arbitrary text with 10.9999/random_string without doi label"
        pdf_path = os.path.join(self.tmp.name, "no_doi.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_content)

        dest = a0.discover_from_file("", pdf_path)
        self.assertIsNotNone(dest)
        seeds = a0._load_seeds()
        rec = next(iter(seeds.values()))
        self.assertIsNone(rec["doi"])
        self.assertTrue(rec["key"].startswith("file:"))

    def test_get_seed_by_path_returns_most_recent(self):
        seeds = {
            "First Older Topic": {"path": "/path/to/paper.pdf", "title": "Old"},
            "Second Latest Topic": {"path": "/path/to/paper.pdf", "title": "New"},
        }
        with patch.object(a0, "_load_seeds", return_value=seeds):
            rec, q = a0.get_seed_by_path("/path/to/paper.pdf")
            self.assertEqual(q, "Second Latest Topic")
            self.assertEqual(rec["title"], "New")


if __name__ == "__main__":
    unittest.main()
