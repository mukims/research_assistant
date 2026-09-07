"""Tests for Agent 2's paper naming and Agent 3's manifest parsing.

Filenames are how the fetcher decides whether it already has a paper. They used
to embed the citation's position in the fetch queue, which is recomputed every
run, so the same work downloaded under a new name each time its reference string
was spelled differently — 21 of the 167 PDFs in the working corpus were
byte-identical duplicates of another file.

fetch_papers() names files with filename_for(source_key(ref)) — TestPaperFilename
exercises that real path directly. An earlier draft carried a standalone naming
function with its own tests; nothing in production ever called it, so it and
its tests were deleted rather than kept as false coverage.

TestManifestParsing and TestFetchPapersSuccessRecord cover the schema wiring
that replaced Agent 3's legacy-manifest-shape shim: `_pdfs_from_manifest` now
builds a `schemas.DownloadedPaper` from each record instead of guessing at the
manifest's shape, and Agent 2's success path must populate every field that
schema declares (`provider` and `fetched_at` in particular did not exist prior
to this schema wiring).
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from research_assistant.agents import agent2_fetcher
from research_assistant.schemas import DownloadedPaper
from research_assistant.shared.fetch import filename_for
from research_assistant.shared.source_key import source_key


class TestPaperFilename(unittest.TestCase):
    """filename_for(source_key(ref)) is what fetch_papers() actually calls to
    build a paper's destination path — these tests exercise that real path,
    not a stand-alone naming function nothing else calls."""

    def test_same_doi_gives_same_name_regardless_of_queue_position(self):
        """The property the old scheme lacked: identity, not fetch order."""
        a = {
            "title": "Half-metallic graphene nanoribbons",
            "doi": "10.1038/nature05180",
            "doi_confidence": "high",
            "raw_reference": "Son, Y. et al. Half-metallic graphene nanoribbons. Nature 2006.",
            "source_file": "seed.pdf",
        }
        b = {
            "title": "Half-metallic graphene nanoribbons",
            "doi": "10.1038/nature05180",
            "doi_confidence": "high",
            "raw_reference": "Y. Son et al., Nature 444, 347 (2006).",
            "source_file": "other.pdf",
        }
        self.assertEqual(filename_for(source_key(a)), filename_for(source_key(b)))

    def test_title_capitalisation_does_not_split_one_paper(self):
        """With no DOI, identity falls back to the folded-title hash, which
        lowercases before hashing — capitalisation must not create a second
        file for the same work."""
        a = {"title": "Half-metallic graphene nanoribbons", "raw_reference": "x", "source_file": "s.pdf"}
        b = {"title": "Half-Metallic Graphene Nanoribbons", "raw_reference": "x", "source_file": "s.pdf"}
        self.assertEqual(filename_for(source_key(a)), filename_for(source_key(b)))

    def test_distinct_papers_get_distinct_names(self):
        a = {"title": "Paper A", "doi": "10.1038/nature05180", "source_file": "s.pdf"}
        b = {"title": "Paper B", "doi": "10.1038/nature12952", "source_file": "s.pdf"}
        self.assertNotEqual(filename_for(source_key(a)), filename_for(source_key(b)))

    def test_same_title_different_doi_stays_distinct(self):
        """Titles collide across papers; the DOI is what disambiguates."""
        a = {"title": "Graphene", "doi": "10.1038/aaa", "source_file": "s.pdf"}
        b = {"title": "Graphene", "doi": "10.1038/bbb", "source_file": "s.pdf"}
        self.assertNotEqual(filename_for(source_key(a)), filename_for(source_key(b)))

    def test_filename_is_a_sanitised_form_of_the_source_key(self):
        ref = {"doi": "10.1038/nature05180", "doi_confidence": "high", "source_file": "s.pdf"}
        key = source_key(ref)
        self.assertEqual(key, "doi:10.1038/nature05180")
        self.assertEqual(filename_for(key), "doi_10.1038_nature05180.pdf")


class _FakeCrossrefResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class TestResolveDoi(unittest.TestCase):
    """Finding I3: `return doi, "grobid-unverified" if doi else (None, None)`
    parses as `return doi, ("grobid-unverified" if doi else (None, None))` —
    a tuple written into what should be a plain (doi, how) pair, so
    doi_source ends up `[null, null]` in downloaded.json. Line 106 was
    already parenthesised correctly; only line 103 had the bug."""

    def _ref(self, **overrides):
        ref = {"doi": "10.1000/xyz", "doi_confidence": "low", "title": "Some title"}
        ref.update(overrides)
        return ref

    def test_crossref_request_failure_returns_a_plain_two_tuple(self):
        import requests as requests_mod

        with patch.object(agent2_fetcher.requests, "get",
                           side_effect=requests_mod.RequestException("boom")):
            result = agent2_fetcher.resolve_doi(self._ref())

        self.assertEqual(result, ("10.1000/xyz", "grobid-unverified"))
        self.assertIsInstance(result[1], str)

    def test_crossref_no_results_returns_a_plain_two_tuple(self):
        with patch.object(agent2_fetcher.requests, "get",
                           return_value=_FakeCrossrefResponse({"message": {"items": []}})):
            result = agent2_fetcher.resolve_doi(self._ref())

        self.assertEqual(result, ("10.1000/xyz", "grobid-unverified"))

    def test_no_doi_and_crossref_failure_returns_none_none(self):
        import requests as requests_mod

        with patch.object(agent2_fetcher.requests, "get",
                           side_effect=requests_mod.RequestException("boom")):
            result = agent2_fetcher.resolve_doi(self._ref(doi=None))

        self.assertEqual(result, (None, None))

    def test_crossref_success_resolves_via_crossref(self):
        with patch.object(agent2_fetcher.requests, "get",
                           return_value=_FakeCrossrefResponse(
                               {"message": {"items": [{"DOI": "10.9999/found"}]}})):
            result = agent2_fetcher.resolve_doi(self._ref())

        self.assertEqual(result, ("10.9999/found", "crossref"))


class TestManifestParsing(unittest.TestCase):
    def test_malformed_entry_is_skipped_not_treated_as_a_path(self):
        """The old shim turned a record with no path into a missing file."""
        from research_assistant.agents.agent3_ingestor import _pdfs_from_manifest

        pdfs = _pdfs_from_manifest({
            "doi:10.1/good": {
                "key": "doi:10.1/good", "path": "a.pdf",
                "provider": "arxiv", "fetched_at": "t", "title": "Good",
            },
            "doi:10.1/bad": {"key": "doi:10.1/bad", "provider": "arxiv"},
        })
        self.assertEqual(pdfs, {"a.pdf": "Good"})

    def test_title_is_preferred_over_the_opaque_key_as_the_label(self):
        from research_assistant.agents.agent3_ingestor import _pdfs_from_manifest

        pdfs = _pdfs_from_manifest({
            "doi:10.1/x": {
                "key": "doi:10.1/x", "path": "a.pdf", "provider": "arxiv",
                "fetched_at": "t", "title": "Real Title",
                "raw_reference": "Smith et al.",
            },
        })
        self.assertEqual(pdfs, {"a.pdf": "Real Title"})

    def test_record_with_path_but_missing_provider_and_fetched_at_is_skipped(self):
        """The legacy shim accepted any record with a `path`, full stop. The
        schema-validated replacement additionally requires `provider` and
        `fetched_at` (both required DownloadedPaper fields with no default) —
        a record that has `path` but lacks them must still be rejected, not
        silently treated as a well-formed download."""
        from research_assistant.agents.agent3_ingestor import _pdfs_from_manifest

        pdfs = _pdfs_from_manifest({
            "k": {"key": "k", "path": "a.pdf"},
        })
        self.assertEqual(pdfs, {})


class TestFetchPapersSuccessRecord(unittest.TestCase):
    """Regression guard: fetch_papers()'s success path must populate every
    field DownloadedPaper declares. Unlike a hand-built DownloadedPaper that
    merely round-trips through to_dict/from_dict, this calls fetch_papers()
    itself — mocked at the network and filesystem seams — and inspects the
    record actually written to downloaded.json, so deleting a field
    assignment inside fetch_papers() (e.g. `cited_by=ref.get("source_file")`)
    makes this test fail."""

    def _run_fetch_and_get_record(self):
        ref = {
            "raw_reference": "Son, Y. et al. Half-metallic graphene nanoribbons. Nature 2006.",
            "source_file": "seed.pdf",
            "title": "Half-metallic graphene nanoribbons",
            "doi": "10.1038/nature05180",
            "doi_confidence": "high",
            "arxiv_id": "cond-mat/0611602",
            "xml_id": "b12",
        }

        def fake_unpaywall(doi, dest_path):
            with open(dest_path, "wb") as fh:
                fh.write(b"%PDF-1.4 fake pdf bytes")
            return True, None

        with tempfile.TemporaryDirectory() as tmp:
            downloaded_path = os.path.join(tmp, "downloaded.json")
            failed_path = os.path.join(tmp, "failed_downloads.json")
            pdfs_dir = os.path.join(tmp, "pulled_pdfs")
            citations_path = os.path.join(tmp, "extracted_citations.json")

            with patch.object(agent2_fetcher, "DOWNLOADED_JSON_PATH", downloaded_path), \
                 patch.object(agent2_fetcher, "FAILED_DOWNLOADS_PATH", failed_path), \
                 patch.object(agent2_fetcher, "PULLED_PDFS_DIR", pdfs_dir), \
                 patch.object(agent2_fetcher, "EXTRACTED_CITATIONS_PATH", citations_path), \
                 patch.object(agent2_fetcher, "_load_references", return_value=[ref]), \
                 patch.object(agent2_fetcher, "resolve_doi", return_value=("10.1038/nature05180", "grobid")), \
                 patch.object(agent2_fetcher, "try_unpaywall", side_effect=fake_unpaywall), \
                 patch.object(agent2_fetcher, "try_europepmc", return_value=(False, "not tried")), \
                 patch.object(agent2_fetcher, "try_arxiv", return_value=(False, "not tried")):
                agent2_fetcher.fetch_papers()

            with open(downloaded_path) as f:
                downloaded = json.load(f)

        self.assertEqual(len(downloaded), 1)
        return next(iter(downloaded.values()))

    def test_fetch_papers_populates_every_downloadedpaper_field(self):
        record = self._run_fetch_and_get_record()
        for field_name in DownloadedPaper.__dataclass_fields__:
            with self.subTest(field=field_name):
                self.assertIsNotNone(record.get(field_name))

    def test_fetch_papers_record_matches_the_reference_it_came_from(self):
        record = self._run_fetch_and_get_record()
        self.assertEqual(record["provider"], "unpaywall")
        self.assertEqual(record["doi"], "10.1038/nature05180")
        self.assertEqual(record["doi_source"], "grobid")
        self.assertEqual(record["cited_by"], "seed.pdf")
        self.assertEqual(record["xml_id"], "b12")
        self.assertEqual(record["arxiv_id"], "cond-mat/0611602")
        self.assertTrue(record["authoritative"])


if __name__ == "__main__":
    unittest.main()
