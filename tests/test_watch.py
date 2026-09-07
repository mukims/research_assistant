"""Tests for watch.py's citation-label handling (Finding I2).

sync_database() and PulledPDFHandler._drain() both used to pass a bare list
of paths to ingest_pdfs(), which falls back to the filename stem for every
entry -- e.g. "doi_10.1038_nature05180" -- and that opaque label becomes
citation_source on every chunk, then the \\cite{} key downstream. Once
ingest_pdfs marks a path in the manifest, Agent 3 can never correct it later,
so the label has to be right the first time under watch.py's default path.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import watch


class TestLabeledPdfs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.manifest_path = os.path.join(self.tmp.name, "downloaded.json")
        patcher = patch.object(watch, "DOWNLOADED_JSON_PATH", self.manifest_path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write_manifest(self, data):
        with open(self.manifest_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    def test_uses_the_parsed_title_not_the_opaque_filename_stem(self):
        path = "/data/pulled_pdfs/doi_10.1038_nature05180.pdf"
        self._write_manifest({
            "doi:10.1038/nature05180": {
                "key": "doi:10.1038/nature05180",
                "path": path,
                "provider": "unpaywall",
                "fetched_at": "2026-01-01T00:00:00Z",
                "title": "The electronic properties of graphene",
            }
        })

        labels = watch._labeled_pdfs([path])

        self.assertEqual(labels[path], "The electronic properties of graphene")

    def test_falls_back_to_the_filename_stem_with_no_manifest_record(self):
        """The only case an opaque label is acceptable: nothing on record."""
        path = "/data/pulled_pdfs/manually_dropped.pdf"
        self._write_manifest({})

        labels = watch._labeled_pdfs([path])

        self.assertEqual(labels[path], "manually_dropped")

    def test_missing_manifest_file_falls_back_for_every_path(self):
        path = "/data/pulled_pdfs/x.pdf"
        # self.manifest_path is deliberately never created.

        labels = watch._labeled_pdfs([path])

        self.assertEqual(labels[path], "x")

    def test_malformed_manifest_entry_does_not_crash_the_lookup(self):
        """A record missing a required field must be skipped, not raise --
        agent3_ingestor._pdfs_from_manifest already logs and continues."""
        path = "/data/pulled_pdfs/y.pdf"
        self._write_manifest({"bad": {"key": "bad"}})  # missing path/provider/fetched_at

        labels = watch._labeled_pdfs([path])

        self.assertEqual(labels[path], "y")


if __name__ == "__main__":
    unittest.main()
