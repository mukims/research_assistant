"""
Unit tests for research_assistant.shared.seed_audit
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from bs4 import BeautifulSoup

from research_assistant.shared.seed_audit import (
    _clean_claim_punctuation,
    _match_downloaded_paper,
    audit_seed_citations,
    extract_seed_citation_claims,
    find_tei_for_seed,
)

SAMPLE_TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
    <teiHeader>
        <fileDesc>
            <titleStmt>
                <title level="a" type="main">Seed Paper Title</title>
            </titleStmt>
            <sourceDesc>
                <biblStruct>
                    <analytic>
                        <title level="a" type="main">Seed Paper Title</title>
                        <author><persName><forename>Alice</forename><surname>Smith</surname></persName></author>
                    </analytic>
                </biblStruct>
            </sourceDesc>
        </fileDesc>
    </teiHeader>
    <text>
        <body>
            <p>Recent experiments demonstrate that graphene exhibits extraordinary electronic transport <ref type="bibr" target="#b0">[1]</ref>. This was verified at room temperature.</p>
            <p>In another study, numerical simulations showed that disorder can surprisingly improve inversion accuracy <ref type="bibr">[2]</ref> under certain conditions.</p>
        </body>
        <back>
            <div type="references">
                <listBibl>
                    <biblStruct xml:id="b0">
                        <analytic>
                            <title level="a" type="main">Electric field effect in atomically thin carbon films</title>
                            <author><persName><forename>K.</forename><surname>Novoselov</surname></persName></author>
                            <author><persName><forename>A.</forename><surname>Geim</surname></persName></author>
                            <idno type="DOI">10.1126/science.1102896</idno>
                        </analytic>
                        <monogr>
                            <title level="j">Science</title>
                            <imprint><date when="2004">2004</date></imprint>
                        </monogr>
                    </biblStruct>
                    <biblStruct xml:id="b1">
                        <analytic>
                            <title level="a" type="main">Disorder-assisted quantum inversion in nanoribbons</title>
                            <author><persName><forename>Bob</forename><surname>Jones</surname></persName></author>
                        </analytic>
                        <monogr>
                            <imprint><date when="2020">2020</date></imprint>
                        </monogr>
                    </biblStruct>
                </listBibl>
            </div>
        </back>
    </text>
</TEI>
"""


class TestSeedAudit(unittest.TestCase):
    def test_clean_claim_punctuation(self):
        self.assertEqual(
            _clean_claim_punctuation("Experiments confirmed this [ ] ."),
            "Experiments confirmed this.",
        )
        self.assertEqual(
            _clean_claim_punctuation("It was shown ( ) , that conductance varies ."),
            "It was shown, that conductance varies.",
        )

    def test_extract_seed_citation_claims_with_target(self):
        soup = BeautifulSoup(SAMPLE_TEI_XML, "xml")
        claims = extract_seed_citation_claims(soup)
        self.assertEqual(len(claims), 2)

        # First claim with explicit target="#b0"
        c1 = claims[0]
        self.assertIn("Recent experiments demonstrate that graphene exhibits", c1["claim"])
        self.assertNotIn("__CITE_", c1["claim"])
        self.assertEqual(c1["target"], "b0")
        self.assertIsNotNone(c1["ref"])
        self.assertEqual(c1["ref"]["index"], 1)
        self.assertEqual(c1["ref"]["doi"], "10.1126/science.1102896")
        self.assertIn("Novoselov", c1["ref"]["authors"][0])

    def test_extract_seed_citation_claims_numeric_fallback(self):
        soup = BeautifulSoup(SAMPLE_TEI_XML, "xml")
        claims = extract_seed_citation_claims(soup)

        # Second claim with untargeted [2]
        c2 = claims[1]
        self.assertIn("numerical simulations showed that disorder can surprisingly improve inversion", c2["claim"])
        self.assertEqual(c2["cite_text"], "[2]")
        self.assertIsNotNone(c2["ref"])
        self.assertEqual(c2["ref"]["index"], 2)
        self.assertEqual(c2["ref"]["title"], "Disorder-assisted quantum inversion in nanoribbons")

    def test_match_downloaded_paper(self):
        manifest = {
            "doi:10.1126/science.1102896": {
                "key": "doi:10.1126/science.1102896",
                "path": "/path/to/novoselov2004.pdf",
                "doi": "10.1126/science.1102896",
                "title": "Electric Field Effect in Atomically Thin Carbon Films",
                "cited_by": "seed.pdf",
                "xml_id": "b0",
            },
            "other_paper": {
                "key": "other_key",
                "path": "/path/to/other.pdf",
                "doi": "10.1000/xyz",
                "title": "Unrelated Work",
            },
        }

        # Match by xml_id + cited_by
        ref1 = {"xml_id": "b0", "doi": "10.1126/science.1102896", "title": "Electric field effect"}
        matched = _match_downloaded_paper(ref1, "seed.pdf", manifest)
        self.assertIsNotNone(matched)
        self.assertEqual(matched["key"], "doi:10.1126/science.1102896")

        # Match by DOI even if cited_by differs
        ref2 = {"xml_id": "b99", "doi": "10.1126/science.1102896", "title": "Random title"}
        matched_doi = _match_downloaded_paper(ref2, "another_seed.pdf", manifest)
        self.assertIsNotNone(matched_doi)

        # No match
        ref3 = {"xml_id": "b5", "doi": "10.9999/notfound", "title": "Unknown Title"}
        self.assertIsNone(_match_downloaded_paper(ref3, "seed.pdf", manifest))

    def test_audit_seed_citations_graceful_missing_tei(self):
        report = audit_seed_citations("/non/existent/path.pdf")
        self.assertIn("error", report)
        self.assertEqual(report["totals"]["total"], 0)
        self.assertEqual(report["results"], [])

    @patch("research_assistant.shared.seed_audit._judge_once")
    @patch("research_assistant.shared.seed_audit.hybrid_search")
    @patch("research_assistant.shared.seed_audit.find_tei_for_seed")
    @patch("research_assistant.shared.seed_audit._load_downloaded_manifest")
    def test_audit_seed_citations_mocked(self, mock_manifest, mock_find_tei, mock_search, mock_judge):
        with tempfile.NamedTemporaryFile("w", suffix=".tei.xml", delete=False, encoding="utf-8") as tf:
            tf.write(SAMPLE_TEI_XML)
            tei_file = tf.name

        mock_find_tei.return_value = tei_file

        with tempfile.NamedTemporaryFile("w", suffix=".pdf", delete=False) as dummy_pdf:
            dummy_pdf_path = dummy_pdf.name

        mock_manifest.return_value = {
            "doi:10.1126/science.1102896": {
                "key": "doi:10.1126/science.1102896",
                "path": dummy_pdf_path,
                "doi": "10.1126/science.1102896",
                "title": "Electric field effect in atomically thin carbon films",
                "cited_by": "seed.pdf",
                "xml_id": "b0",
            }
        }

        mock_search.return_value = [
            {"text": "We observed strong electric field effect in graphene.", "metadata": {"document": os.path.basename(dummy_pdf_path)}}
        ]

        mock_judge.return_value = {
            "judgement": "Supports",
            "confidence": "High",
            "supporting_span": "We observed strong electric field effect in graphene.",
            "reason": "Direct evidence of electric field effect.",
            "slots": {"finding": "Supports", "scope": "Supports", "strength": "Supports"},
            "evidence_sufficiency": "sufficient",
        }

        fake_resources = (MagicMock(), MagicMock(), [], [])
        report = audit_seed_citations("seed.pdf", search_resources=fake_resources, max_claims=5)

        self.assertNotIn("error", report)
        self.assertEqual(report["totals"]["total"], 2)
        self.assertEqual(report["totals"]["downloaded"], 1)
        self.assertEqual(report["totals"]["judged"], 1)
        self.assertEqual(report["totals"]["Supports"], 1)
        self.assertEqual(report["totals"]["not_downloaded"], 1)

        # Cleanup
        os.unlink(tei_file)
        os.unlink(dummy_pdf_path)


if __name__ == "__main__":
    unittest.main()
