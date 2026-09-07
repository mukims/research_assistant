"""Tests for the contracts between pipeline stages.

The failure these are designed out of: agent 3 previously accepted two
incompatible downloaded.json shapes, and the wrong one "made every entry look
like a missing file and ingested nothing at all".
"""

import unittest

from research_assistant.schemas import (
    DownloadedPaper,
    Reference,
    SchemaError,
    SeedPaper,
)


class TestReference(unittest.TestCase):
    def test_round_trip_preserves_every_field(self):
        ref = Reference(
            raw_reference="Smith et al. Nature 2020",
            source_file="seed.pdf",
            title="A paper",
            container="Nature",
            authors=("Smith, J.", "Doe, A."),
            year=2020,
            doi="10.1038/x",
            doi_confidence="high",
            xml_id="b12",
            extraction_method="grobid",
        )
        self.assertEqual(Reference.from_dict(ref.to_dict()), ref)

    def test_authors_from_json_list_becomes_a_tuple(self):
        """JSON has no tuples; a decoded list must not break equality."""
        ref = Reference.from_dict(
            {"raw_reference": "r", "source_file": "s.pdf", "authors": ["A", "B"]}
        )
        self.assertEqual(ref.authors, ("A", "B"))

    def test_unknown_keys_are_ignored(self):
        """A manifest written by an older run still loads."""
        ref = Reference.from_dict(
            {"raw_reference": "r", "source_file": "s.pdf", "legacy_field": 1}
        )
        self.assertEqual(ref.raw_reference, "r")

    def test_missing_required_field_raises(self):
        with self.assertRaises(SchemaError):
            Reference.from_dict({"raw_reference": "r"})

    def test_defaults_fill_in_for_absent_optional_fields(self):
        ref = Reference.from_dict({"raw_reference": "r", "source_file": "s.pdf"})
        self.assertIsNone(ref.doi)
        self.assertEqual(ref.authors, ())
        self.assertEqual(ref.extraction_method, "grobid")


class TestDownloadedPaper(unittest.TestCase):
    def test_missing_path_raises_rather_than_half_building(self):
        """The exact bug this schema exists to prevent."""
        with self.assertRaises(SchemaError):
            DownloadedPaper.from_dict(
                {"key": "doi:10.1/x", "provider": "arxiv", "fetched_at": "now"}
            )

    def test_round_trip(self):
        paper = DownloadedPaper(
            key="doi:10.1/x",
            path="data/pulled_pdfs/doi_10.1_x.pdf",
            provider="unpaywall",
            fetched_at="2026-09-07T00:00:00Z",
            title="A paper",
        )
        self.assertEqual(DownloadedPaper.from_dict(paper.to_dict()), paper)

    def test_non_dict_input_raises(self):
        with self.assertRaises(SchemaError):
            DownloadedPaper.from_dict("not a dict")


class TestSeedPaper(unittest.TestCase):
    def test_round_trip(self):
        seed = SeedPaper(
            key="arxiv:2401.12345",
            path="data/raw/arxiv_2401.12345.pdf",
            fetched_at="2026-09-07T00:00:00Z",
            source="search",
            title="A seed",
        )
        self.assertEqual(SeedPaper.from_dict(seed.to_dict()), seed)

    def test_missing_source_raises(self):
        with self.assertRaises(SchemaError):
            SeedPaper.from_dict({"key": "k", "path": "p", "fetched_at": "t"})


if __name__ == "__main__":
    unittest.main()
