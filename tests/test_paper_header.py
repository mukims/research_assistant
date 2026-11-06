"""Tests for reading paper headers (shared.paper_header)."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

from research_assistant import config
from research_assistant.shared.paper_header import (
    PaperHeader,
    docling_header,
    grobid_header,
    header_from_docling,
    header_from_grobid_tei,
    read_header,
)


GROBID_TEI_FIXTURE = """<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title level="a" type="main">Half-metallic graphene nanoribbons</title>
      </titleStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author>
              <persName>
                <forename type="first">Young-Woo</forename>
                <surname>Son</surname>
              </persName>
            </author>
            <author>
              <persName>
                <forename type="first">Marvin</forename>
                <surname>Cohen</surname>
              </persName>
            </author>
            <idno type="DOI">10.1038/nature05180</idno>
            <idno type="arXiv">cond-mat/0611602</idno>
          </analytic>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
  </teiHeader>
</TEI>"""

COVER_PAGE_FIXTURE = {
    "texts": [
        {"label": "text", "text": "See discussions, stats, and author profiles for this publication at: https://www.researchgate.net/...", "prov": [{"page_no": 1}]},
        {"label": "section_header", "text": "Pedagogical introduction to equilibrium Green's functions: Condensed-matter examples with numerical implementations", "prov": [{"page_no": 1}]},
        {"label": "text", "text": "Article in Revista Brasileira de Ensino de Física · October 2016", "prov": [{"page_no": 1}]},
        {"label": "text", "text": "DOI: 10.1590/1806-9126-RBEF-2016-0087", "prov": [{"page_no": 1}]},
        {"label": "text", "text": "CITATIONS", "prov": [{"page_no": 1}]},
        {"label": "text", "text": "25", "prov": [{"page_no": 1}]},
        {"label": "text", "text": "READS", "prov": [{"page_no": 1}]},
        {"label": "text", "text": "4,187", "prov": [{"page_no": 1}]},
        {"label": "text", "text": "3 authors, including:", "prov": [{"page_no": 1}]},
        {"label": "text", "text": "Mariana M. Odashima", "prov": [{"page_no": 1}]},
        {"label": "text", "text": "Universidade Federal de Uberlândia (UFU)", "prov": [{"page_no": 1}]},
    ]
}


class TestPaperHeader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_grobid_tei_yields_title_surname_doi_and_arxiv(self):
        header = header_from_grobid_tei(GROBID_TEI_FIXTURE)
        self.assertEqual(
            header,
            PaperHeader(
                "Half-metallic graphene nanoribbons",
                ("Son",),
                "10.1038/nature05180",
                "cond-mat/0611602",
            ),
        )

    def test_grobid_unreachable_returns_none(self):
        with patch("research_assistant.shared.paper_header.requests.post", side_effect=requests.ConnectionError("unreachable")):
            header = grobid_header("/tmp/any.pdf")
            self.assertIsNone(header)

    def test_grobid_non_200_returns_none(self):
        fake_resp = MagicMock()
        fake_resp.status_code = 503
        with patch("research_assistant.shared.paper_header.requests.post", return_value=fake_resp):
            header = grobid_header("/tmp/any.pdf")
            self.assertIsNone(header)

    def test_grobid_posts_header_request_without_consolidation(self):
        recorded = {}

        def fake_post(url, data=None, files=None, headers=None, timeout=None):
            recorded["url"] = url
            recorded["data"] = data
            resp = MagicMock()
            resp.status_code = 200
            resp.text = GROBID_TEI_FIXTURE
            return resp

        # Create dummy pdf
        dummy_pdf = os.path.join(self.tmp.name, "paper.pdf")
        with open(dummy_pdf, "wb") as f:
            f.write(b"%PDF-fake")

        with patch("research_assistant.shared.paper_header.requests.post", side_effect=fake_post):
            header = grobid_header(dummy_pdf)
            self.assertIsNotNone(header)
            self.assertTrue(recorded["url"].endswith("/api/processHeaderDocument"))
            self.assertEqual(recorded["data"]["consolidateHeader"], "0")

    def test_read_header_merges_both_sources(self):
        g = PaperHeader("T", ("Zaghouani",), None, None)
        d = PaperHeader("T", ("Müller",), None, None)
        with patch("research_assistant.shared.paper_header.grobid_header", return_value=g), \
             patch("research_assistant.shared.paper_header.docling_header", return_value=d):
            merged = read_header("/tmp/paper.pdf")
            self.assertIsNotNone(merged)
            self.assertEqual(merged.title, "T")
            self.assertEqual(merged.first_authors, ("Zaghouani", "Müller"))

    def test_read_header_keeps_one_spelling_per_surname(self):
        g = PaperHeader("T", ("Essam Ghareeb",), None, None)
        d = PaperHeader("T", ("Ghareeb",), None, None)
        with patch("research_assistant.shared.paper_header.grobid_header", return_value=g), \
             patch("research_assistant.shared.paper_header.docling_header", return_value=d):
            merged = read_header("/tmp/paper.pdf")
            self.assertIsNotNone(merged)
            self.assertEqual(merged.first_authors, ("Essam Ghareeb",))

    def test_read_header_falls_back_to_docling_when_grobid_is_down(self):
        d = PaperHeader("Some Title", ("Odashima",), "10.1590/1", None)
        with patch("research_assistant.shared.paper_header.grobid_header", return_value=None), \
             patch("research_assistant.shared.paper_header.docling_header", return_value=d):
            merged = read_header("/tmp/paper.pdf")
            self.assertEqual(merged, d)

    def test_read_header_is_none_without_title_or_author(self):
        with patch("research_assistant.shared.paper_header.grobid_header", return_value=None), \
             patch("research_assistant.shared.paper_header.docling_header", return_value=None):
            self.assertIsNone(read_header("/tmp/paper.pdf"))

        g_no_author = PaperHeader("Some Title", (), None, None)
        with patch("research_assistant.shared.paper_header.grobid_header", return_value=g_no_author), \
             patch("research_assistant.shared.paper_header.docling_header", return_value=None):
            self.assertIsNone(read_header("/tmp/paper.pdf"))

    def test_researchgate_cover_page_yields_title_author_and_doi(self):
        header = header_from_docling(COVER_PAGE_FIXTURE)
        self.assertTrue(header.title.startswith("Pedagogical introduction to equilibrium Green's functions"))
        self.assertEqual(header.first_authors, ("Odashima",))
        self.assertEqual(header.doi, "10.1590/1806-9126-rbef-2016-0087")

    def test_title_label_wins_over_section_header(self):
        doc = {
            "texts": [
                {"label": "section_header", "text": "Introduction to the problem space", "prov": [{"page_no": 1}]},
                {"label": "title", "text": "Half-metallic graphene nanoribbons", "prov": [{"page_no": 1}]},
                {"label": "text", "text": "Young-Woo Son, Marvin L. Cohen and Steven G. Louie", "prov": [{"page_no": 1}]},
            ]
        }
        header = header_from_docling(doc)
        self.assertEqual(header.title, "Half-metallic graphene nanoribbons")
        self.assertEqual(header.first_authors, ("Son",))

    def test_superscript_markers_are_ignored(self):
        doc = {
            "texts": [
                {"label": "title", "text": "Half-metallic graphene nanoribbons", "prov": [{"page_no": 1}]},
                {"label": "text", "text": "Young-Woo Son1*, Marvin L. Cohen1,2", "prov": [{"page_no": 1}]},
            ]
        }
        header = header_from_docling(doc)
        self.assertEqual(header.first_authors, ("Son",))

    def test_trailing_marks_are_stripped_from_the_surname(self):
        doc1 = {
            "texts": [
                {"label": "title", "text": "Title One", "prov": [{"page_no": 1}]},
                {"label": "text", "text": "Ulrich Eckern\u2217", "prov": [{"page_no": 1}]},
            ]
        }
        header1 = header_from_docling(doc1)
        self.assertEqual(header1.first_authors, ("Eckern",))

        doc2 = {
            "texts": [
                {"label": "title", "text": "Title Two", "prov": [{"page_no": 1}]},
                {"label": "text", "text": "Shantanu Mishra'+", "prov": [{"page_no": 1}]},
            ]
        }
        header2 = header_from_docling(doc2)
        self.assertEqual(header2.first_authors, ("Mishra",))

    def test_bare_initial_is_not_a_surname(self):
        doc = {
            "texts": [
                {"label": "title", "text": "Some Title", "prov": [{"page_no": 1}]},
                {"label": "text", "text": "Alice B", "prov": [{"page_no": 1}]},
            ]
        }
        header = header_from_docling(doc)
        self.assertEqual(header.first_authors, ())

    def test_venue_or_repository_banner_is_not_a_title(self):
        doc = {
            "texts": [
                {"label": "section_header", "text": "European Journal of Public Health Studies", "prov": [{"page_no": 1}]},
                {"label": "section_header", "text": "Institutional Repository - Research Portal", "prov": [{"page_no": 1}]},
                {"label": "section_header", "text": "Carrier density and magnetism in graphene zigzag nanoribbons", "prov": [{"page_no": 1}]},
            ]
        }
        header = header_from_docling(doc)
        self.assertEqual(header.title, "Carrier density and magnetism in graphene zigzag nanoribbons")

    def test_affiliation_is_never_taken_as_the_author(self):
        doc = {
            "texts": [
                {"label": "title", "text": "Title Here", "prov": [{"page_no": 1}]},
                {"label": "text", "text": "Department of Physics, University of California, Berkeley", "prov": [{"page_no": 1}]},
                {"label": "text", "text": "Abstract", "prov": [{"page_no": 1}]},
            ]
        }
        header = header_from_docling(doc)
        self.assertEqual(header.first_authors, ())

    def test_generic_section_header_is_not_a_title(self):
        doc = {
            "texts": [
                {"label": "section_header", "text": "Abstract", "prov": [{"page_no": 1}]},
                {"label": "section_header", "text": "Introduction", "prov": [{"page_no": 1}]},
            ]
        }
        header = header_from_docling(doc)
        self.assertIsNone(header.title)
        self.assertEqual(header.first_authors, ())

    def test_only_page_one_is_read(self):
        doc = {
            "texts": [
                {"label": "title", "text": "Title Here", "prov": [{"page_no": 1}]},
                {"label": "text", "text": "Mariana M. Odashima", "prov": [{"page_no": 2}]},
            ]
        }
        header = header_from_docling(doc)
        self.assertEqual(header.first_authors, ())

    def test_docling_header_without_cache_returns_none(self):
        with patch.object(config, "DOCLING_CACHE_DIR", self.tmp.name), \
             patch("research_assistant.shared.docling_extract.get_digital_converter", side_effect=AssertionError("Should not call")):
            header = docling_header("/path/to/paper.pdf", convert_if_missing=False)
            self.assertIsNone(header)

    def test_docling_header_with_corrupt_cache_returns_none(self):
        cache_file = os.path.join(self.tmp.name, "corrupt.docling.json")
        with open(cache_file, "w") as f:
            f.write("{not json")
        with patch.object(config, "DOCLING_CACHE_DIR", self.tmp.name):
            header = docling_header(os.path.join(self.tmp.name, "corrupt.pdf"), convert_if_missing=False)
            self.assertIsNone(header)

    def test_docling_header_converts_two_pages_when_asked(self):
        recorded = {}
        fake_conv_res = MagicMock()
        fake_conv_res.document.model_dump.return_value = COVER_PAGE_FIXTURE

        fake_converter = MagicMock()
        def fake_convert(pdf_path, page_range=None):
            recorded["page_range"] = page_range
            return fake_conv_res

        fake_converter.convert = fake_convert

        with patch.object(config, "DOCLING_CACHE_DIR", self.tmp.name), \
             patch("research_assistant.shared.docling_extract.get_digital_converter", return_value=fake_converter):
            dummy_pdf = os.path.join(self.tmp.name, "uncached.pdf")
            header = docling_header(dummy_pdf, convert_if_missing=True)
            self.assertEqual(recorded["page_range"], (1, 2))
            self.assertIsNotNone(header)
            self.assertEqual(header.first_authors, ("Odashima",))


if __name__ == "__main__":
    unittest.main()
