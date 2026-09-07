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

    def test_explicit_null_on_optional_field_yields_the_default(self):
        """An explicit `null` must be treated the same as absent, not passed
        through raw -- otherwise `authors` ends up `None` instead of `()`
        and callers hit `TypeError: 'NoneType' object is not iterable`."""
        ref = Reference.from_dict(
            {"raw_reference": "r", "source_file": "s.pdf", "authors": None}
        )
        self.assertEqual(ref.authors, ())

    def test_explicit_null_on_required_field_still_raises(self):
        """Dropping nulls must not extend to required fields: a manifest
        emitting `"source_file": null` is exactly the malformed input this
        module exists to catch."""
        with self.assertRaises(SchemaError):
            Reference.from_dict({"raw_reference": "r", "source_file": None})

    def test_malformed_value_raises_schema_error_not_type_error(self):
        """A present-but-wrong-shaped value must fail as SchemaError, not
        leak the raw TypeError from the tuple() coercion or the
        constructor call."""
        with self.assertRaises(SchemaError):
            Reference.from_dict(
                {"raw_reference": "r", "source_file": "s.pdf", "authors": 5}
            )

    def test_bare_string_authors_raises_rather_than_exploding_into_characters(self):
        """Finding I3's second half: tuple() does not reject a plain string --
        tuple("Smith, J.") silently becomes a 9-tuple of individual
        characters, which then produces a different source_key than the
        list-of-names it was meant to be. A bare string must be rejected
        outright, not coerced."""
        with self.assertRaises(SchemaError):
            Reference.from_dict(
                {"raw_reference": "r", "source_file": "s.pdf", "authors": "Smith, J."}
            )

    def test_non_string_optional_scalar_field_raises(self):
        """Finding I3: a str | None field (e.g. title) must reject a non-str,
        non-None value rather than silently storing it."""
        with self.assertRaises(SchemaError):
            Reference.from_dict(
                {"raw_reference": "r", "source_file": "s.pdf", "title": ["not", "a", "string"]}
            )


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

    def test_agent2_fetcher_fields_round_trip(self):
        """agent2_fetcher.py in the source repo writes these four fields;
        from_dict silently drops unknown keys, so without them declared a
        round trip would lose `cited_by` -- the provenance link back to the
        paper that cited this one -- on every run."""
        paper = DownloadedPaper(
            key="doi:10.1/x",
            path="data/pulled_pdfs/doi_10.1_x.pdf",
            provider="crossref",
            fetched_at="2026-09-07T00:00:00Z",
            doi_source="crossref",
            authoritative=True,
            cited_by="seed.pdf",
            xml_id="b12",
        )
        round_tripped = DownloadedPaper.from_dict(paper.to_dict())
        self.assertEqual(round_tripped, paper)
        self.assertEqual(round_tripped.cited_by, "seed.pdf")
        self.assertEqual(round_tripped.xml_id, "b12")
        self.assertEqual(round_tripped.doi_source, "crossref")
        self.assertTrue(round_tripped.authoritative)

    def test_tuple_written_into_doi_source_raises_schema_error(self):
        """The exact shape Finding I3's operator-precedence bug in
        agent2_fetcher.resolve_doi() produced: a JSON-decoded 2-list landing
        in a str | None field instead of a string."""
        with self.assertRaises(SchemaError):
            DownloadedPaper.from_dict({
                "key": "doi:10.1/x", "path": "p", "provider": "unpaywall",
                "fetched_at": "now", "doi_source": [None, None],
            })


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
