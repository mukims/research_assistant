"""
Unit tests for research_assistant.shared.seed_audit
"""

import contextlib
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
    compute_totals,
    cross_check_seed_audit,
    explain_rubric_verdict,
    extract_seed_citation_claims,
    find_tei_for_seed,
    generate_seed_audit_markdown,
    get_cached_seed_audit,
    get_deferred_missing_references,
    save_and_register_reference_pdf,
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

MULTI_PARAGRAPH_TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
    <teiHeader>
        <fileDesc>
            <titleStmt>
                <title level="a" type="main">Multi-Paragraph Graphene Paper</title>
            </titleStmt>
            <sourceDesc>
                <biblStruct>
                    <analytic>
                        <title level="a" type="main">Multi-Paragraph Graphene Paper</title>
                        <author><persName><forename>Alice</forename><surname>Smith</surname></persName></author>
                    </analytic>
                </biblStruct>
            </sourceDesc>
        </fileDesc>
    </teiHeader>
    <text>
        <body>
            <p xml:id="p_0">Recent experiments demonstrate that graphene exhibits extraordinary electronic transport <ref type="bibr" target="#b0">[1]</ref>. Another study confirmed ballistic quantum oscillations in high-mobility samples <ref type="bibr" target="#b1">[2]</ref>.</p>
            <p xml:id="p_1">Numerical simulations showed that disorder can surprisingly improve inversion accuracy <ref type="bibr" target="#b2">[3]</ref> under topological symmetry.</p>
            <p xml:id="p_2">This concluding paragraph discusses prospective technological applications and synthesis methods without citing literature.</p>
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
                            <title level="a" type="main">Ballistic transport in high-mobility graphene</title>
                            <author><persName><forename>Bob</forename><surname>Jones</surname></persName></author>
                            <idno type="DOI">10.1038/nphys1234</idno>
                        </analytic>
                        <monogr>
                            <imprint><date when="2020">2020</date></imprint>
                        </monogr>
                    </biblStruct>
                    <biblStruct xml:id="b2">
                        <analytic>
                            <title level="a" type="main">Disorder-assisted quantum inversion in nanoribbons</title>
                            <author><persName><forename>Carol</forename><surname>White</surname></persName></author>
                            <idno type="DOI">10.1103/PhysRevB.99.012345</idno>
                        </analytic>
                        <monogr>
                            <imprint><date when="2022">2022</date></imprint>
                        </monogr>
                    </biblStruct>
                </listBibl>
            </div>
        </back>
    </text>
</TEI>
"""


class TestSeedAudit(unittest.TestCase):
    def setUp(self):
        self._audit_tmp = tempfile.TemporaryDirectory()
        self._audit_patcher = patch("research_assistant.shared.seed_audit.AUDIT_DIR", self._audit_tmp.name)
        self._audit_patcher.start()

    def tearDown(self):
        self._audit_patcher.stop()
        self._audit_tmp.cleanup()

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
        self.assertEqual(report["totals"]["deferred_paywalled"], 1)
        self.assertEqual(report["totals"]["not_downloaded"], 0)
        self.assertIn("reliability", report["totals"])
        self.assertEqual(report["totals"]["reliability"]["high"], 1)
        self.assertEqual(report["totals"]["reliability"]["unresolved"], 1)
        self.assertEqual(report["results"][0]["reliability"], "HIGH")
        self.assertIn("🟢", report["results"][0]["reliability_badge"])

        # Cleanup
        os.unlink(tei_file)
        os.unlink(dummy_pdf_path)

    def test_generate_seed_audit_markdown_and_explain(self):
        sample_report = {
            "seed_path": "/data/test_paper.pdf",
            "seed_name": "test_paper.pdf",
            "generated": "2026-09-13T12:00:00",
            "model": "gemini-3.5-flash-lite",
            "totals": {
                "total": 3,
                "downloaded": 2,
                "judged": 2,
                "Supports": 1,
                "Partially supports": 1,
                "Contradicts": 0,
                "Does not support": 0,
                "Unclear / insufficient evidence": 0,
                "not_downloaded": 1,
            },
            "results": [
                {
                    "sentence": "Graphene conducts electricity well [1].",
                    "claim": "Graphene conducts electricity well.",
                    "cite_text": "[1]",
                    "ref": {
                        "index": 1,
                        "title": "Graphene Conductivity",
                        "authors": ["A. Author"],
                        "year": 2021,
                        "doi": "10.1000/182",
                    },
                    "outcome": "judged",
                    "judgement": "Supports",
                    "confidence": "High",
                    "evidence_sufficiency": "sufficient",
                    "supporting_span": "Graphene has high electrical conductivity.",
                    "reason": "Explicit empirical confirmation.",
                    "slots": {
                        "finding": {"assertion": "conducts well", "verdict": "Supports"},
                        "scope": {"assertion": "graphene", "verdict": "Supports"},
                        "strength": {"assertion": "well", "verdict": "Supports"},
                    },
                },
                {
                    "sentence": "All polymers are highly elastic [2].",
                    "claim": "All polymers are highly elastic.",
                    "cite_text": "[2]",
                    "ref": {
                        "index": 2,
                        "title": "Elasticity of PANI",
                        "authors": ["B. Builder"],
                        "year": 2022,
                    },
                    "outcome": "judged",
                    "judgement": "Partially supports",
                    "confidence": "Medium",
                    "evidence_sufficiency": "partial",
                    "supporting_span": "PANI shows notable elasticity.",
                    "reason": "Only PANI tested, generalized to all polymers.",
                    "slots": {
                        "finding": {"assertion": "elastic", "verdict": "Supports"},
                        "scope": {"assertion": "all polymers", "verdict": "Partially supports"},
                        "strength": {"assertion": "highly", "verdict": "Supports"},
                    },
                },
                {
                    "sentence": "Ancient methods were used [3].",
                    "claim": "Ancient methods were used.",
                    "cite_text": "[3]",
                    "outcome": "not_downloaded",
                    "judgement": "Not downloaded",
                },
            ],
        }

        # Test explain_rubric_verdict
        expl_supp = explain_rubric_verdict(sample_report["results"][0])
        self.assertIn("Fully supported", expl_supp)

        expl_part = explain_rubric_verdict(sample_report["results"][1])
        self.assertIn("Partial support", expl_part)

        expl_pw = explain_rubric_verdict(sample_report["results"][2])
        self.assertIn("paywalled", expl_pw.lower())

        # Test generate_seed_audit_markdown
        md = generate_seed_audit_markdown(sample_report)
        self.assertIn("# 🔍 In-Text Citation Audit Report", md)
        self.assertIn("Graphene Conductivity", md)
        self.assertIn("Slot Decomposition Analysis", md)
        self.assertIn("Why this verdict?", md)
        self.assertIn("Executive Summary", md)

    def test_extract_seed_citation_claims_paragraph_grouping(self):
        """Test multi-paragraph TEI XML verifies claims receive paragraph_id, paragraph_index, and paragraph_refs."""
        soup = BeautifulSoup(MULTI_PARAGRAPH_TEI_XML, "xml")
        claims = extract_seed_citation_claims(soup)

        self.assertEqual(len(claims), 3)

        # First paragraph claims
        c0, c1 = claims[0], claims[1]
        self.assertEqual(c0["paragraph_id"], "p_0")
        self.assertEqual(c0["paragraph_index"], 0)
        self.assertEqual(c0["target"], "b0")
        self.assertEqual(len(c0["paragraph_refs"]), 2)
        ref_ids_p0 = {r["xml_id"] for r in c0["paragraph_refs"]}
        self.assertEqual(ref_ids_p0, {"b0", "b1"})

        self.assertEqual(c1["paragraph_id"], "p_0")
        self.assertEqual(c1["paragraph_index"], 0)
        self.assertEqual(c1["target"], "b1")
        self.assertEqual(len(c1["paragraph_refs"]), 2)
        self.assertEqual({r["xml_id"] for r in c1["paragraph_refs"]}, {"b0", "b1"})

        # Second paragraph claim
        c2 = claims[2]
        self.assertEqual(c2["paragraph_id"], "p_1")
        self.assertEqual(c2["paragraph_index"], 1)
        self.assertEqual(c2["target"], "b2")
        self.assertEqual(len(c2["paragraph_refs"]), 1)
        self.assertEqual(c2["paragraph_refs"][0]["xml_id"], "b2")

        # Verify all legacy keys are preserved
        for c in claims:
            for key in (
                "sentence",
                "claim",
                "cite_text",
                "target",
                "ref",
                "paragraph_id",
                "paragraph_index",
                "paragraph_refs",
            ):
                self.assertIn(key, c)

    @patch("research_assistant.shared.seed_audit._judge_once")
    @patch("research_assistant.shared.seed_audit.hybrid_search")
    @patch("research_assistant.shared.seed_audit.find_tei_for_seed")
    @patch("research_assistant.shared.seed_audit._load_downloaded_manifest")
    def test_audit_seed_citations_deferral_ratio_greater_than_half(
        self, mock_manifest, mock_find_tei, mock_search, mock_judge
    ):
        """Paragraph with paywall ratio > 0.50 (2/2 = 1.0) must be deferred, bypassing search & judge."""
        tei_content = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
    <teiHeader><fileDesc><titleStmt><title>Paywall Test</title></titleStmt><sourceDesc><p></p></sourceDesc></fileDesc></teiHeader>
    <text>
        <body>
            <p xml:id="p_paywall">First claim asserting novel superconducting phase transition <ref type="bibr" target="#b1">[1]</ref>. Second claim asserting anomalous Hall conductance in the same regime <ref type="bibr" target="#b2">[2]</ref>.</p>
        </body>
        <back>
            <div type="references">
                <listBibl>
                    <biblStruct xml:id="b1">
                        <analytic><title level="a" type="main">Superconducting Transition</title></analytic>
                        <monogr><imprint><date when="2021">2021</date></imprint></monogr>
                    </biblStruct>
                    <biblStruct xml:id="b2">
                        <analytic><title level="a" type="main">Anomalous Hall Conductance</title></analytic>
                        <monogr><imprint><date when="2022">2022</date></imprint></monogr>
                    </biblStruct>
                </listBibl>
            </div>
        </back>
    </text>
</TEI>"""
        with tempfile.NamedTemporaryFile("w", suffix=".tei.xml", delete=False, encoding="utf-8") as tf:
            tf.write(tei_content)
            tei_file = tf.name

        try:
            mock_find_tei.return_value = tei_file
            mock_manifest.return_value = {}  # Neither reference is downloaded (2/2 = 1.0 > 0.50)

            fake_resources = (MagicMock(), MagicMock(), [], [])
            report = audit_seed_citations("seed.pdf", search_resources=fake_resources)

            self.assertNotIn("error", report)
            t = report["totals"]
            self.assertEqual(t["total"], 2)
            self.assertEqual(t["downloaded"], 0)
            self.assertEqual(t["judged"], 0)
            self.assertEqual(t["Supports"], 0)
            self.assertEqual(t["not_downloaded"], 0)
            self.assertEqual(t["deferred_paywalled"], 2)

            for res in report["results"]:
                self.assertEqual(res["outcome"], "deferred_paywalled")
                self.assertEqual(res["judgement"], "Deferred (pending paywalled evidence)")
                self.assertIn("Evaluation deferred:", res["reason"])
                self.assertIn("2/2", res["reason"])
                self.assertIn(">50% paywalled", res["reason"])
                self.assertEqual(res["paywall_ratio"], 1.0)
                self.assertEqual(len(res["paragraph_missing_refs"]), 2)
                missing_xml_ids = {r["xml_id"] for r in res["paragraph_missing_refs"]}
                self.assertEqual(missing_xml_ids, {"b1", "b2"})

            # Neither retrieval nor LLM should be called
            mock_search.assert_not_called()
            mock_judge.assert_not_called()
        finally:
            os.unlink(tei_file)

    @patch("research_assistant.shared.seed_audit._judge_once")
    @patch("research_assistant.shared.seed_audit.hybrid_search")
    @patch("research_assistant.shared.seed_audit.find_tei_for_seed")
    @patch("research_assistant.shared.seed_audit._load_downloaded_manifest")
    def test_audit_seed_citations_ratio_half_or_less_evaluates(
        self, mock_manifest, mock_find_tei, mock_search, mock_judge
    ):
        """Paragraph with paywall ratio <= 0.50 (1/2 = 0.50) evaluates downloaded and marks missing as not_downloaded."""
        tei_content = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
    <teiHeader><fileDesc><titleStmt><title>Boundary Ratio Test</title></titleStmt><sourceDesc><p></p></sourceDesc></fileDesc></teiHeader>
    <text>
        <body>
            <p xml:id="p_boundary">Conductivity in graphene rises sharply near the Dirac point <ref type="bibr" target="#b0">[1]</ref>. Furthermore ballistic transport occurs over micron scales <ref type="bibr" target="#b1">[2]</ref>.</p>
        </body>
        <back>
            <div type="references">
                <listBibl>
                    <biblStruct xml:id="b0">
                        <analytic><title level="a" type="main">Graphene Dirac Conductivity</title><idno type="DOI">10.1126/science.1102896</idno></analytic>
                        <monogr><imprint><date when="2004">2004</date></imprint></monogr>
                    </biblStruct>
                    <biblStruct xml:id="b1">
                        <analytic><title level="a" type="main">Ballistic Transport Study</title><idno type="DOI">10.1038/nphys1234</idno></analytic>
                        <monogr><imprint><date when="2020">2020</date></imprint></monogr>
                    </biblStruct>
                </listBibl>
            </div>
        </back>
    </text>
</TEI>"""
        with tempfile.NamedTemporaryFile("w", suffix=".tei.xml", delete=False, encoding="utf-8") as tf:
            tf.write(tei_content)
            tei_file = tf.name

        with tempfile.NamedTemporaryFile("w", suffix=".pdf", delete=False) as dummy_pdf:
            dummy_pdf_path = dummy_pdf.name

        try:
            mock_find_tei.return_value = tei_file
            # Only b0 is in the manifest; b1 is missing (1 missing / 2 total = 0.50 <= 0.50)
            mock_manifest.return_value = {
                "doi:10.1126/science.1102896": {
                    "key": "doi:10.1126/science.1102896",
                    "path": dummy_pdf_path,
                    "doi": "10.1126/science.1102896",
                    "title": "Graphene Dirac Conductivity",
                    "cited_by": "seed.pdf",
                    "xml_id": "b0",
                }
            }

            mock_search.return_value = [
                {
                    "text": "Conductivity increases at the Dirac point.",
                    "metadata": {"document": os.path.basename(dummy_pdf_path)},
                }
            ]
            mock_judge.return_value = {
                "judgement": "Supports",
                "confidence": "High",
                "supporting_span": "Conductivity increases at the Dirac point.",
                "reason": "Explicit empirical match.",
                "slots": {"finding": "Supports", "scope": "Supports", "strength": "Supports"},
                "evidence_sufficiency": "sufficient",
            }

            fake_resources = (MagicMock(), MagicMock(), [], [])
            report = audit_seed_citations("seed.pdf", search_resources=fake_resources)

            self.assertNotIn("error", report)
            t = report["totals"]
            self.assertEqual(t["total"], 2)
            self.assertEqual(t["downloaded"], 1)
            self.assertEqual(t["judged"], 1)
            self.assertEqual(t["Supports"], 1)
            self.assertEqual(t["not_downloaded"], 1)
            self.assertEqual(t["deferred_paywalled"], 0)

            # Check individual items
            judged_item = next(r for r in report["results"] if r.get("target") == "b0")
            self.assertEqual(judged_item["outcome"], "judged")
            self.assertEqual(judged_item["judgement"], "Supports")

            missing_item = next(r for r in report["results"] if r.get("target") == "b1")
            self.assertEqual(missing_item["outcome"], "not_downloaded")
            self.assertEqual(missing_item["judgement"], "Not downloaded")

            # Search and judge called exactly once for the downloaded claim
            self.assertEqual(mock_search.call_count, 1)
            self.assertEqual(mock_judge.call_count, 1)
        finally:
            os.unlink(tei_file)
            os.unlink(dummy_pdf_path)

    @patch("research_assistant.shared.seed_audit._judge_once")
    @patch("research_assistant.shared.seed_audit.hybrid_search")
    @patch("research_assistant.shared.seed_audit.find_tei_for_seed")
    @patch("research_assistant.shared.seed_audit._load_downloaded_manifest")
    def test_audit_seed_citations_mixed_paragraphs(
        self, mock_manifest, mock_find_tei, mock_search, mock_judge
    ):
        """Document with both deferred and non-deferred paragraphs verifies selective evaluation."""
        tei_content = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
    <teiHeader><fileDesc><titleStmt><title>Mixed Paragraphs</title></titleStmt><sourceDesc><p></p></sourceDesc></fileDesc></teiHeader>
    <text>
        <body>
            <p xml:id="p_evaluated">Paragraph one shows high electrical conductivity in monolayer graphene <ref type="bibr" target="#b0">[1]</ref>.</p>
            <p xml:id="p_deferred">Paragraph two discusses coupled thermal and thermoelectric transport <ref type="bibr" target="#b0">[1]</ref>. It also claims topological edge states emerge <ref type="bibr" target="#b1">[2]</ref>. Finally phonon scattering dominates at elevated temperatures <ref type="bibr" target="#b2">[3]</ref>.</p>
        </body>
        <back>
            <div type="references">
                <listBibl>
                    <biblStruct xml:id="b0">
                        <analytic><title level="a" type="main">Graphene Transport</title><idno type="DOI">10.1126/science.1102896</idno></analytic>
                        <monogr><imprint><date when="2004">2004</date></imprint></monogr>
                    </biblStruct>
                    <biblStruct xml:id="b1">
                        <analytic><title level="a" type="main">Topological Edge States</title><idno type="DOI">10.1038/nphys1234</idno></analytic>
                        <monogr><imprint><date when="2020">2020</date></imprint></monogr>
                    </biblStruct>
                    <biblStruct xml:id="b2">
                        <analytic><title level="a" type="main">Phonon Scattering</title><idno type="DOI">10.1103/PhysRevB.99.012345</idno></analytic>
                        <monogr><imprint><date when="2022">2022</date></imprint></monogr>
                    </biblStruct>
                </listBibl>
            </div>
        </back>
    </text>
</TEI>"""
        with tempfile.NamedTemporaryFile("w", suffix=".tei.xml", delete=False, encoding="utf-8") as tf:
            tf.write(tei_content)
            tei_file = tf.name

        with tempfile.NamedTemporaryFile("w", suffix=".pdf", delete=False) as dummy_pdf:
            dummy_pdf_path = dummy_pdf.name

        try:
            mock_find_tei.return_value = tei_file
            # Only b0 is in the manifest; b1 and b2 are missing
            mock_manifest.return_value = {
                "doi:10.1126/science.1102896": {
                    "key": "doi:10.1126/science.1102896",
                    "path": dummy_pdf_path,
                    "doi": "10.1126/science.1102896",
                    "title": "Graphene Transport",
                    "cited_by": "seed.pdf",
                    "xml_id": "b0",
                }
            }

            mock_search.return_value = [
                {
                    "text": "High conductivity in monolayer graphene.",
                    "metadata": {"document": os.path.basename(dummy_pdf_path)},
                }
            ]
            mock_judge.return_value = {
                "judgement": "Supports",
                "confidence": "High",
                "supporting_span": "High conductivity in monolayer graphene.",
                "reason": "Corroborated.",
                "slots": {"finding": "Supports", "scope": "Supports", "strength": "Supports"},
                "evidence_sufficiency": "sufficient",
            }

            fake_resources = (MagicMock(), MagicMock(), [], [])
            report = audit_seed_citations("seed.pdf", search_resources=fake_resources)

            self.assertNotIn("error", report)
            t = report["totals"]
            self.assertEqual(t["total"], 4)
            self.assertEqual(t["downloaded"], 2)
            self.assertEqual(t["judged"], 1)
            self.assertEqual(t["Supports"], 1)
            self.assertEqual(t["deferred_paywalled"], 3)
            self.assertEqual(t["not_downloaded"], 0)

            # Paragraph 1 claim is evaluated
            p1_items = [r for r in report["results"] if r.get("paragraph_id") == "p_evaluated"]
            self.assertEqual(len(p1_items), 1)
            self.assertEqual(p1_items[0]["outcome"], "judged")
            self.assertEqual(p1_items[0]["judgement"], "Supports")

            # Paragraph 2 claims are ALL deferred (including the one citing b0, because p2 ratio = 2/3 = 0.67 > 0.50)
            p2_items = [r for r in report["results"] if r.get("paragraph_id") == "p_deferred"]
            self.assertEqual(len(p2_items), 3)
            for item in p2_items:
                self.assertEqual(item["outcome"], "deferred_paywalled")
                self.assertEqual(item["judgement"], "Deferred (pending paywalled evidence)")

            # Retrieval and LLM should only be invoked for Paragraph 1 (1 call)
            self.assertEqual(mock_search.call_count, 1)
            self.assertEqual(mock_judge.call_count, 1)
        finally:
            os.unlink(tei_file)
            os.unlink(dummy_pdf_path)

    @patch("research_assistant.shared.seed_audit._judge_once")
    @patch("research_assistant.shared.seed_audit.hybrid_search")
    @patch("research_assistant.shared.seed_audit.find_tei_for_seed")
    @patch("research_assistant.shared.seed_audit._load_downloaded_manifest")
    def test_audit_seed_citations_totals_math(
        self, mock_manifest, mock_find_tei, mock_search, mock_judge
    ):
        """Strict verification of totals sum invariant and judged count matching."""
        tei_content = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
    <teiHeader><fileDesc><titleStmt><title>Totals Math Test</title></titleStmt><sourceDesc><p></p></sourceDesc></fileDesc></teiHeader>
    <text>
        <body>
            <p xml:id="p_bench">Claim one asserting extraordinary electronic mobility in flakes <ref type="bibr" target="#b0">[1]</ref>. Claim two asserting thermal conductance across boundary interfaces <ref type="bibr" target="#b0">[1]</ref>. Claim three asserting negative Poisson ratio in lattice <ref type="bibr" target="#b0">[1]</ref>. Claim four asserting bandgap opening under shear strain <ref type="bibr" target="#b0">[1]</ref>. Claim five asserting fragile phase coherence at elevated temperatures <ref type="bibr" target="#b0">[1]</ref>.</p>
            <p xml:id="p_half">Claim six asserting ballistic transport in graphene nanoribbons <ref type="bibr" target="#b0">[1]</ref>. Claim seven asserting localized edge reconstruction states <ref type="bibr" target="#b1">[2]</ref>.</p>
            <p xml:id="p_defer">Claim eight asserting non-local entanglement across long distances <ref type="bibr" target="#b1">[2]</ref>. Claim nine asserting topological protection in fractional quantum Hall <ref type="bibr" target="#b2">[3]</ref>.</p>
        </body>
        <back>
            <div type="references">
                <listBibl>
                    <biblStruct xml:id="b0">
                        <analytic><title level="a" type="main">Graphene Physics</title><idno type="DOI">10.1126/science.1102896</idno></analytic>
                        <monogr><imprint><date when="2004">2004</date></imprint></monogr>
                    </biblStruct>
                    <biblStruct xml:id="b1">
                        <analytic><title level="a" type="main">Nanoribbon States</title><idno type="DOI">10.1038/nphys1234</idno></analytic>
                        <monogr><imprint><date when="2020">2020</date></imprint></monogr>
                    </biblStruct>
                    <biblStruct xml:id="b2">
                        <analytic><title level="a" type="main">Topological Protection</title><idno type="DOI">10.1103/PhysRevB.99.012345</idno></analytic>
                        <monogr><imprint><date when="2022">2022</date></imprint></monogr>
                    </biblStruct>
                </listBibl>
            </div>
        </back>
    </text>
</TEI>"""
        with tempfile.NamedTemporaryFile("w", suffix=".tei.xml", delete=False, encoding="utf-8") as tf:
            tf.write(tei_content)
            tei_file = tf.name

        with tempfile.NamedTemporaryFile("w", suffix=".pdf", delete=False) as dummy_pdf:
            dummy_pdf_path = dummy_pdf.name

        try:
            mock_find_tei.return_value = tei_file
            # Only b0 is in the manifest
            mock_manifest.return_value = {
                "doi:10.1126/science.1102896": {
                    "key": "doi:10.1126/science.1102896",
                    "path": dummy_pdf_path,
                    "doi": "10.1126/science.1102896",
                    "title": "Graphene Physics",
                    "cited_by": "seed.pdf",
                    "xml_id": "b0",
                }
            }

            mock_search.return_value = [
                {"text": "Relevant passage text.", "metadata": {"document": os.path.basename(dummy_pdf_path)}}
            ]

            # Return 6 judged verdicts covering all 5 rubric judgement types
            mock_judge.side_effect = [
                {
                    "judgement": "Supports",
                    "confidence": "High",
                    "supporting_span": "evidence",
                    "reason": "r",
                    "slots": {},
                },
                {
                    "judgement": "Partially supports",
                    "confidence": "Medium",
                    "supporting_span": "evidence",
                    "reason": "r",
                    "slots": {},
                },
                {
                    "judgement": "Contradicts",
                    "confidence": "High",
                    "supporting_span": "evidence",
                    "reason": "r",
                    "slots": {},
                },
                {
                    "judgement": "Does not support",
                    "confidence": "Low",
                    "supporting_span": "evidence",
                    "reason": "r",
                    "slots": {},
                },
                {
                    "judgement": "Unclear / insufficient evidence",
                    "confidence": "Low",
                    "supporting_span": "evidence",
                    "reason": "r",
                    "slots": {},
                },
                {
                    "judgement": "Supports",
                    "confidence": "High",
                    "supporting_span": "evidence",
                    "reason": "r",
                    "slots": {},
                },
            ]

            fake_resources = (MagicMock(), MagicMock(), [], [])
            report = audit_seed_citations("seed.pdf", search_resources=fake_resources, max_claims=10)

            self.assertNotIn("error", report)
            t = report["totals"]

            # Strict totals sum invariant
            expected_sum = (
                t["Supports"]
                + t["Partially supports"]
                + t["Contradicts"]
                + t["Does not support"]
                + t["Unclear / insufficient evidence"]
                + t["not_downloaded"]
                + t["deferred_paywalled"]
            )
            self.assertEqual(expected_sum, t["total"])
            self.assertEqual(t["judged"] + sum(t["not_assessed"].values()), t["total"])

            # Strict judged count invariant
            self.assertEqual(t["judged"], sum(1 for r in report["results"] if r.get("outcome") == "judged"))

            # Total matches results length
            self.assertEqual(t["total"], len(report["results"]))

            # Verify every category is populated (> 0)
            self.assertGreater(t["Supports"], 0)
            self.assertGreater(t["Partially supports"], 0)
            self.assertGreater(t["Contradicts"], 0)
            self.assertGreater(t["Does not support"], 0)
            self.assertGreater(t["Unclear / insufficient evidence"], 0)
            self.assertGreater(t["not_downloaded"], 0)
            self.assertGreater(t["deferred_paywalled"], 0)
        finally:
            os.unlink(tei_file)
            os.unlink(dummy_pdf_path)

    def test_get_deferred_missing_references(self):
        """Test deduplication, metadata fields, and aggregated unique affected_claims across deferred paragraphs."""
        ref1 = {
            "xml_id": "b1",
            "index": 1,
            "title": "Electronic Transport in Graphene",
            "authors": ["A. Novoselov", "K. Geim"],
            "year": 2004,
            "doi": "10.1126/science.1102896",
            "raw_reference": "Novoselov et al. Science 2004.",
        }
        ref2 = {
            "xml_id": "b2",
            "index": 2,
            "title": "Quantum Hall Effect in 2D Systems",
            "authors": ["B. Zhang", "C. Dan"],
            "year": 2006,
            "doi": "10.1038/nature04235",
            "raw_reference": "Zhang et al. Nature 2006.",
        }
        ref3 = {
            "xml_id": "b3",
            "index": 3,
            "title": "Phonon Dispersion in 2D Lattices",
            "authors": ["E. Fermi"],
            "year": 2010,
            "doi": "10.1103/PhysRevB.80.012345",
            "raw_reference": "Fermi E. PRB 2010.",
        }

        # Multi-paragraph report with deferred items sharing ref2
        sample_report = {
            "results": [
                # Paragraph 1 claims citing ref1 and ref2
                {
                    "outcome": "deferred_paywalled",
                    "sentence": "Graphene demonstrates ballistic conduction at room temperature [1, 2].",
                    "paragraph_missing_refs": [ref1, ref2],
                },
                {
                    "outcome": "deferred_paywalled",
                    "sentence": "Room temperature mobility exceeds 100,000 cm2/Vs under gate voltage [1, 2].",
                    "paragraph_missing_refs": [ref1, ref2],
                },
                # Paragraph 2 claims citing ref2 and ref3 (ref2 shared!)
                {
                    "outcome": "deferred_paywalled",
                    "sentence": "Quantum Hall plateaus are observed at room temperature [2, 3].",
                    "paragraph_missing_refs": [ref2, ref3],
                },
                {
                    "outcome": "deferred_paywalled",
                    "sentence": "Phonon backscattering is suppressed near the Dirac cone [2, 3].",
                    "paragraph_missing_refs": [ref2, ref3],
                },
                # Evaluated (non-deferred) claim
                {
                    "outcome": "judged",
                    "sentence": "Evaluated claim from non-deferred paragraph.",
                    "paragraph_missing_refs": [],
                },
            ]
        }

        missing = get_deferred_missing_references(sample_report)

        # Deduplication: exactly 3 unique missing references (b1, b2, b3)
        self.assertEqual(len(missing), 3)

        by_xml = {m["xml_id"]: m for m in missing}
        self.assertIn("b1", by_xml)
        self.assertIn("b2", by_xml)
        self.assertIn("b3", by_xml)

        # Shared ref2 should aggregate all 4 affected claims without duplicates
        b2_entry = by_xml["b2"]
        self.assertEqual(len(b2_entry["affected_claims"]), 4)
        self.assertIn(
            "Graphene demonstrates ballistic conduction at room temperature [1, 2].",
            b2_entry["affected_claims"],
        )
        self.assertIn(
            "Room temperature mobility exceeds 100,000 cm2/Vs under gate voltage [1, 2].",
            b2_entry["affected_claims"],
        )
        self.assertIn(
            "Quantum Hall plateaus are observed at room temperature [2, 3].",
            b2_entry["affected_claims"],
        )
        self.assertIn(
            "Phonon backscattering is suppressed near the Dirac cone [2, 3].",
            b2_entry["affected_claims"],
        )

        # ref1 should have the 2 claims from paragraph 1
        b1_entry = by_xml["b1"]
        self.assertEqual(len(b1_entry["affected_claims"]), 2)

        # ref3 should have the 2 claims from paragraph 2
        b3_entry = by_xml["b3"]
        self.assertEqual(len(b3_entry["affected_claims"]), 2)

        # Verify metadata integrity
        self.assertEqual(b2_entry["title"], "Quantum Hall Effect in 2D Systems")
        self.assertEqual(b2_entry["authors"], ["B. Zhang", "C. Dan"])
        self.assertEqual(b2_entry["year"], 2006)
        self.assertEqual(b2_entry["doi"], "10.1038/nature04235")
        self.assertEqual(b2_entry["raw_reference"], "Zhang et al. Nature 2006.")

        # Graceful handling of empty or missing report
        self.assertEqual(get_deferred_missing_references({}), [])
        self.assertEqual(get_deferred_missing_references({"results": []}), [])

    def test_generate_seed_audit_markdown_with_deferred(self):
        """Verify markdown renders Executive Summary deferred row and Section 3 table with DOI links, and explain_rubric_verdict on deferred."""
        report = {
            "seed_path": "/data/test_seed.pdf",
            "seed_name": "test_seed.pdf",
            "generated": "2026-09-13T12:00:00",
            "model": "gemini-3.5-flash-lite",
            "totals": {
                "total": 2,
                "downloaded": 0,
                "judged": 0,
                "Supports": 0,
                "Partially supports": 0,
                "Contradicts": 0,
                "Does not support": 0,
                "Unclear / insufficient evidence": 0,
                "not_downloaded": 0,
                "deferred_paywalled": 2,
            },
            "results": [
                {
                    "sentence": "Superconducting phases appear in twisted bilayer graphene [1].",
                    "claim": "Superconducting phases appear in twisted bilayer graphene.",
                    "cite_text": "[1]",
                    "outcome": "deferred_paywalled",
                    "judgement": "Deferred (pending paywalled evidence)",
                    "reason": "Evaluation deferred: 1/1 references cited in this paragraph are missing from the corpus (>50% paywalled).",
                    "ref": {
                        "xml_id": "b0",
                        "index": 1,
                        "title": "Unconventional superconductivity in magic-angle graphene",
                        "authors": ["Y. Cao", "P. Jarillo-Herrero"],
                        "year": 2018,
                        "doi": "10.1038/nature26160",
                    },
                    "paragraph_missing_refs": [
                        {
                            "xml_id": "b0",
                            "index": 1,
                            "title": "Unconventional superconductivity in magic-angle graphene",
                            "authors": ["Y. Cao", "P. Jarillo-Herrero"],
                            "year": 2018,
                            "doi": "10.1038/nature26160",
                        }
                    ],
                },
                {
                    "sentence": "Correlated insulator states accompany the superconducting domes [1].",
                    "claim": "Correlated insulator states accompany the superconducting domes.",
                    "cite_text": "[1]",
                    "outcome": "deferred_paywalled",
                    "judgement": "Deferred (pending paywalled evidence)",
                    "reason": "Evaluation deferred: 1/1 references cited in this paragraph are missing from the corpus (>50% paywalled).",
                    "ref": {
                        "xml_id": "b0",
                        "index": 1,
                        "title": "Unconventional superconductivity in magic-angle graphene",
                        "authors": ["Y. Cao", "P. Jarillo-Herrero"],
                        "year": 2018,
                        "doi": "10.1038/nature26160",
                    },
                    "paragraph_missing_refs": [
                        {
                            "xml_id": "b0",
                            "index": 1,
                            "title": "Unconventional superconductivity in magic-angle graphene",
                            "authors": ["Y. Cao", "P. Jarillo-Herrero"],
                            "year": 2018,
                            "doi": "10.1038/nature26160",
                        }
                    ],
                },
            ],
        }

        md = generate_seed_audit_markdown(report)

        # Executive summary row for deferred items
        self.assertIn("⏳ **Deferred (Pending Evidence)**", md)
        self.assertIn("| **2** | 100% |", md)

        # Section 3 table and headers
        self.assertIn("## 3. Missing References Required for Deferred Paragraphs (1)", md)
        self.assertIn(
            "| Reference / Title | Authors | Year | DOI | Required By Deferred Claim(s) |", md
        )
        self.assertIn("Unconventional superconductivity in magic-angle graphene", md)
        self.assertIn("Y. Cao, P. Jarillo-Herrero", md)
        self.assertIn("2018", md)

        # DOI formatted as markdown hyperlink
        self.assertIn("[`10.1038/nature26160`](https://doi.org/10.1038/nature26160)", md)

        # Affected claims rendered
        self.assertIn("Superconducting phases appear in twisted bilayer graphene [1].", md)

        # Test explain_rubric_verdict on deferred item with reason
        explanation_with_reason = explain_rubric_verdict(report["results"][0])
        self.assertIn("Evaluation deferred:", explanation_with_reason)
        self.assertIn("Upload the missing reference PDF(s)", explanation_with_reason)

        # Test explain_rubric_verdict on deferred item without custom reason
        explanation_default = explain_rubric_verdict({"outcome": "deferred_paywalled"})
        self.assertIn("Evaluation deferred", explanation_default)
        self.assertIn("Upload the missing reference PDF(s)", explanation_default)

        # Test explain_rubric_verdict on item with judgement "Deferred (pending paywalled evidence)"
        explanation_by_judgement = explain_rubric_verdict(
            {"judgement": "Deferred (pending paywalled evidence)"}
        )
        self.assertIn("Evaluation deferred", explanation_by_judgement)

    @patch("research_assistant.agents.agent6_manual_ingestor.ingest_manual_pdf")
    def test_upload_and_registration_helpers(self, mock_agent6_ingest):
        """Test save_and_register_reference_pdf using temporary directories for file writing, atomic manifest update, and Agent 6 call."""
        mock_agent6_ingest.return_value = {"status": "ok", "chunks": 8}

        with tempfile.TemporaryDirectory() as tmp_pulled_dir, tempfile.TemporaryDirectory() as tmp_manifest_dir:
            manifest_path = os.path.join(tmp_manifest_dir, "downloaded.json")
            pdf_bytes = b"%PDF-1.5 \x00\x01\x02 test pdf raw bytes"
            ref_info = {
                "xml_id": "b10",
                "doi": "10.1016/j.cell.2023.01.005",
                "title": "Cellular Mechanisms of Quantum Biology",
                "authors": ["Jane Scientist"],
                "year": 2023,
                "raw_reference": "Scientist J. Cell 2023.",
            }

            # 1. Ingest with custom filename and ingest=True
            entry = save_and_register_reference_pdf(
                pdf_bytes=pdf_bytes,
                ref_info=ref_info,
                seed_pdf_name="seed_paper.pdf",
                original_filename="Quantum Bio (2023) [Final].pdf",
                pulled_pdfs_dir=tmp_pulled_dir,
                downloaded_manifest_path=manifest_path,
                ingest=True,
            )

            # Verify file was written with sanitized filename
            expected_sanitized_name = "Quantum_Bio__2023___Final_.pdf"
            expected_file_path = os.path.join(tmp_pulled_dir, expected_sanitized_name)
            self.assertEqual(entry["path"], expected_file_path)
            self.assertTrue(os.path.exists(expected_file_path))
            with open(expected_file_path, "rb") as f:
                self.assertEqual(f.read(), pdf_bytes)

            # Verify manifest was atomically written and contains correct fields
            self.assertTrue(os.path.exists(manifest_path))
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            expected_key = "doi:10.1016/j.cell.2023.01.005"
            self.assertIn(expected_key, manifest)
            manifest_entry = manifest[expected_key]
            self.assertEqual(manifest_entry["key"], expected_key)
            self.assertEqual(manifest_entry["path"], expected_file_path)
            self.assertEqual(manifest_entry["provider"], "manual_upload")
            self.assertEqual(manifest_entry["title"], "Cellular Mechanisms of Quantum Biology")
            self.assertEqual(manifest_entry["doi"], "10.1016/j.cell.2023.01.005")
            self.assertEqual(manifest_entry["xml_id"], "b10")
            self.assertEqual(manifest_entry["cited_by"], "seed_paper.pdf")
            self.assertTrue(entry["ingested"])
            self.assertEqual(entry["ingest_result"], {"status": "ok", "chunks": 8})

            # Verify Agent 6 was invoked
            mock_agent6_ingest.assert_called_once_with(
                expected_file_path,
                citation_string="Cellular Mechanisms of Quantum Biology",
            )
            mock_agent6_ingest.reset_mock()

            # 2. Ingest without DOI or original_filename (fallback naming), and ingest=False
            ref_info_no_doi = {
                "xml_id": "b20",
                "title": "Quantum Coherence in Tubulins",
            }
            entry2 = save_and_register_reference_pdf(
                pdf_bytes=b"%PDF-1.5 second file",
                ref_info=ref_info_no_doi,
                seed_pdf_name="seed_paper.pdf",
                original_filename="reference.pdf",
                pulled_pdfs_dir=tmp_pulled_dir,
                downloaded_manifest_path=manifest_path,
                ingest=False,
            )

            # Fallback filename should use xml_id
            self.assertEqual(os.path.basename(entry2["path"]), "b20.pdf")
            self.assertTrue(os.path.exists(entry2["path"]))
            self.assertFalse(entry2["ingested"])
            mock_agent6_ingest.assert_not_called()

            # Verify manifest contains BOTH entries (atomic write does not overwrite previous keys)
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_updated = json.load(f)
            self.assertIn(expected_key, manifest_updated)
            self.assertIn("xml:b20:seed_paper.pdf", manifest_updated)

            # 3. Verify validation error on empty bytes
            with self.assertRaises(ValueError):
                save_and_register_reference_pdf(
                    pdf_bytes=b"",
                    ref_info=ref_info,
                    seed_pdf_name="seed_paper.pdf",
                    pulled_pdfs_dir=tmp_pulled_dir,
                    downloaded_manifest_path=manifest_path,
                )


class TestSeedAuditCaching(unittest.TestCase):
    def test_get_cached_seed_audit_hit_and_miss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("research_assistant.shared.seed_audit.AUDIT_DIR", tmpdir):
                # 1. Non-existent cache returns None
                self.assertIsNone(get_cached_seed_audit("/path/to/nonexistent_paper.pdf"))

                # 2. Corrupt json returns None without crashing
                corrupt_file = os.path.join(tmpdir, "corrupt_audit.json")
                with open(corrupt_file, "w") as f:
                    f.write("{invalid-json")
                self.assertIsNone(get_cached_seed_audit("corrupt.pdf"))

                # 3. Valid cache is successfully loaded
                valid_data = {
                    "seed_name": "valid_paper.pdf",
                    "totals": {"total": 1, "Supports": 1},
                    "results": [{"claim": "Claim 1", "judgement": "Supports", "outcome": "judged"}],
                }
                valid_file = os.path.join(tmpdir, "valid_paper_audit.json")
                with open(valid_file, "w") as f:
                    json.dump(valid_data, f)

                loaded = get_cached_seed_audit("/somewhere/valid_paper.pdf")
                self.assertIsNotNone(loaded)
                self.assertEqual(loaded["seed_name"], "valid_paper.pdf")
                self.assertEqual(len(loaded["results"]), 1)

    def test_cross_check_seed_audit_fast_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("research_assistant.shared.seed_audit.AUDIT_DIR", tmpdir), \
                 patch("research_assistant.shared.seed_audit._load_downloaded_manifest", return_value={}):
                cached = {
                    "seed_name": "sample_seed.pdf",
                    "generated": "2026-09-13T10:00:00Z",
                    "totals": {"total": 2, "judged": 1, "not_downloaded": 1},
                    "results": [
                        {
                            "claim": "Graphene is 2D carbon.",
                            "judgement": "Supports",
                            "outcome": "judged",
                            "confidence": "High",
                            "ref": {
                                "title": "Physical Review B Paper",
                                "venue": "Physical Review B",
                                "doi": "10.1103/PhysRevB.99.123456",
                            },
                        },
                        {
                            "claim": "Unchecked statement.",
                            "judgement": "Unclear / insufficient evidence",
                            "outcome": "not_downloaded",
                            "ref": {"title": "Paywalled Paper", "doi": "10.1016/j.paywall.2020"},
                        },
                    ],
                }

                report, summary = cross_check_seed_audit("sample_seed.pdf", cached)
                self.assertTrue(summary["from_cache"])
                self.assertIn("duration_seconds", summary)
                self.assertLess(summary["duration_seconds"], 1.0)
                self.assertEqual(summary["newly_judged_count"], 0)
                self.assertEqual(report["totals"]["total"], 2)
                self.assertIn("reliability", report["totals"])
                # First result is Physical Review B -> Rigorous Primary -> High reliability
                self.assertEqual(report["results"][0]["reliability"], "HIGH")
                # Second result is not downloaded -> Unresolved
                self.assertEqual(report["results"][1]["reliability"], "UNRESOLVED")

    def test_cross_check_seed_audit_with_newly_downloaded_reference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_manifest = {
                "arxiv_2201.00001": {
                    "key": "arxiv:2201.00001",
                    "title": "Quantum Transport Measurement",
                    "venue": "Physical Review Letters",
                    "doi": "10.1103/PhysRevLett.120.00001",
                    "path": "/data/pulled/arxiv_2201.00001.pdf",
                }
            }
            cached = {
                "seed_name": "seed.pdf",
                "totals": {"total": 1, "not_downloaded": 1, "judged": 0},
                "results": [
                    {
                        "claim": "Quantum transport shows conductance quantization.",
                        "judgement": "Unclear / insufficient evidence",
                        "outcome": "not_downloaded",
                        "ref": {
                            "title": "Quantum Transport Measurement",
                            "venue": "Physical Review Letters",
                            "doi": "10.1103/PhysRevLett.120.00001",
                        },
                    }
                ],
            }

            with patch("research_assistant.shared.seed_audit.AUDIT_DIR", tmpdir), \
                 patch("research_assistant.shared.seed_audit._load_downloaded_manifest", return_value=fake_manifest), \
                 patch("research_assistant.shared.seed_audit._judge_claim_entry") as mock_judge:

                def fake_judge(item, *args, **kwargs):
                    item["outcome"] = "judged"
                    item["judgement"] = "Supports"
                    item["confidence"] = "High"
                    item["span_verified"] = True
                    return item

                mock_judge.side_effect = fake_judge

                report, summary = cross_check_seed_audit(
                    "seed.pdf",
                    cached,
                    search_resources=(MagicMock(), MagicMock(), [], []),
                    max_new_claims=1,
                )
                self.assertEqual(summary["newly_judged_count"], 1)
                self.assertEqual(report["totals"]["Supports"], 1)
                self.assertEqual(report["totals"]["reliability"]["high"], 1)
                mock_judge.assert_called_once()

    def test_audit_seed_citations_skip_if_cached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cached_report = {
                "seed_name": "paper.pdf",
                "totals": {"total": 1, "judged": 1, "Supports": 1},
                "results": [{"claim": "Test claim", "judgement": "Supports", "outcome": "judged"}],
            }
            with patch("research_assistant.shared.seed_audit.AUDIT_DIR", tmpdir), \
                 patch("research_assistant.shared.seed_audit.get_cached_seed_audit", return_value=cached_report) as mock_get_cached, \
                 patch("research_assistant.shared.seed_audit.cross_check_seed_audit") as mock_cross_check:

                mock_cross_check.return_value = (cached_report, {"duration_seconds": 0.02, "from_cache": True})

                # Call with skip_if_cached=True (default)
                rep = audit_seed_citations("paper.pdf")
                self.assertTrue(rep.get("from_cache"))
                mock_get_cached.assert_called_once()
                mock_cross_check.assert_called_once()

                # Call with force=True -> bypasses cache
                mock_get_cached.reset_mock()
                mock_cross_check.reset_mock()
                with patch("research_assistant.shared.seed_audit.find_tei_for_seed", return_value=None):
                    rep_force = audit_seed_citations("paper.pdf", force=True)
                    mock_get_cached.assert_not_called()
                    mock_cross_check.assert_not_called()
                    self.assertFalse(rep_force.get("from_cache", False))


AUTHOR_YEAR_TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>
<div><head>Introduction</head>
<p><s>Target-oriented methods have been proposed by several groups (<ref type="bibr" target="#b27">Vasconcelos et al. 2017</ref>; <ref type="bibr" target="#b6">C.A.N. da Costa et al. 2018</ref>).</s><s>A similar idea was explored by <ref type="bibr" target="#b16">Landa et al. (2006)</ref> with a path-integral formulation of depth migration.</s></p>
</div>
<div><head>Results</head>
<p><s>Following <ref type="bibr" target="#b25">Silva et al. (2021)</ref>, the computational time for each method is described next.</s><s>The PGF time grows with M, while the RPGF time decreases.</s></p>
<p><s>We use UMFPACK (<ref type="bibr" target="#b9">Davis 2004</ref>) for LU factorization of sparse matrices.</s></p>
</div></body>
<back><listBibl>
<biblStruct xml:id="b6"><analytic><title level="a" type="main">Target-level waveform inversion</title><author><persName><forename>C</forename><surname>Costa</surname></persName></author></analytic><monogr><title level="j">Geophysical Prospecting</title><imprint><date when="2018">2018</date></imprint></monogr></biblStruct>
<biblStruct xml:id="b9"><analytic><title level="a" type="main">Algorithm 832</title><author><persName><forename>T</forename><surname>Davis</surname></persName></author></analytic><monogr><title level="j">ACM Trans. Math. Softw.</title><imprint><date when="2004">2004</date></imprint></monogr></biblStruct>
<biblStruct xml:id="b16"><analytic><title level="a" type="main">Path-integral seismic imaging</title><author><persName><forename>E</forename><surname>Landa</surname></persName></author></analytic><monogr><title level="j">Geophysical Prospecting</title><imprint><date when="2006">2006</date></imprint></monogr></biblStruct>
<biblStruct xml:id="b25"><analytic><title level="a" type="main">Target-oriented inversion using the patched green's function method</title><author><persName><forename>D</forename><surname>Silva</surname></persName></author></analytic><monogr><title level="j">Geophysics</title><imprint><date when="2021">2021</date></imprint></monogr></biblStruct>
<biblStruct xml:id="b27"><analytic><title level="a" type="main">Subsurface-domain objective functions</title><author><persName><forename>I</forename><surname>Vasconcelos</surname></persName></author></analytic><monogr><title level="j">Geophysics</title><imprint><date when="2017">2017</date></imprint></monogr></biblStruct>
<biblStruct xml:id="b40"><monogr><title level="m">Semiconductor Nanostructures</title><author><persName><forename>T</forename><surname>Ihn</surname></persName></author><imprint><date when="2010">2010</date></imprint></monogr></biblStruct>
</listBibl></back></text></TEI>
"""

# A footnote GROBID swept into the bibliography: b20 is not reference [21].
# GROBID linked [21] to b53 elsewhere; a targetless [21] must follow that, and
# a targetless number nobody linked must not be guessed from list position.
FOOTNOTE_BIB_TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>
<p><s>We compute everything with the kernel polynomial method <ref type="bibr" target="#b53">[21]</ref> on large flakes.</s></p>
<p><s>The same method was used for the disordered case <ref type="bibr">[21]</ref> and the clean case <ref type="bibr">[22]</ref> respectively.</s></p>
</body>
<back><listBibl>
<biblStruct xml:id="b19"><analytic><title level="a" type="main">Machine learning phases of matter</title></analytic></biblStruct>
<biblStruct xml:id="b20"><note type="raw_reference">An alternative definition of β(E), shown in the SM [21], may extend the validity</note></biblStruct>
<biblStruct xml:id="b21"><analytic><title level="a" type="main">Z2pack</title></analytic></biblStruct>
<biblStruct xml:id="b53"><analytic><title level="a" type="main">KITE: high-performance accurate modelling of electronic structure</title></analytic></biblStruct>
</listBibl></back></text></TEI>
"""


class TestExtractionOnClaimText(unittest.TestCase):
    def _claims(self, xml):
        return extract_seed_citation_claims(BeautifulSoup(xml, "xml"))

    def test_sentence_tags_give_one_claim_per_sentence(self):
        claims = self._claims(AUTHOR_YEAR_TEI_XML)
        by_claim = {c["claim"] for c in claims}
        self.assertIn("Target-oriented methods have been proposed by several groups.", by_claim)
        self.assertIn("A similar idea was explored by Landa et al. (2006) with a path-integral formulation of depth migration.", by_claim)
        self.assertIn("Following Silva et al. (2021), the computational time for each method is described next.", by_claim)

    def test_no_placeholder_ever_leaks(self):
        for c in self._claims(AUTHOR_YEAR_TEI_XML):
            self.assertNotIn("__CITE_", c["claim"]); self.assertNotIn("⟦", c["claim"])
            self.assertNotIn("__CITE_", c["sentence"]); self.assertNotIn("⟦", c["sentence"])
            self.assertEqual(c["claim_quality"], [], c["claim"])

    def test_display_sentence_and_context(self):
        landa = next(c for c in self._claims(AUTHOR_YEAR_TEI_XML) if c["ref"]["xml_id"] == "b16")
        self.assertEqual(landa["sentence"], "A similar idea was explored by [Landa et al. (2006)] with a path-integral formulation of depth migration.")
        self.assertTrue(landa["context"].startswith("Target-oriented methods have been proposed by several groups ( [Vasconcelos"))
        self.assertIn("«A similar idea", landa["context"])
        self.assertEqual(landa["sentence_index"], 1)
        self.assertEqual(landa["cite_count"], 1)

    def test_section_and_role(self):
        claims = self._claims(AUTHOR_YEAR_TEI_XML)
        landa = next(c for c in claims if c["ref"]["xml_id"] == "b16")
        silva = next(c for c in claims if c["ref"]["xml_id"] == "b25")
        davis = next(c for c in claims if c["ref"]["xml_id"] == "b9")
        self.assertEqual(landa["section"], "introduction"); self.assertEqual(landa["role"], "evidential")
        self.assertEqual(silva["section"], "results"); self.assertEqual(silva["role"], "method")
        self.assertEqual(davis["role"], "software")

    def test_paragraph_refs_count_only_evidential_resolved_references(self):
        davis = next(c for c in self._claims(AUTHOR_YEAR_TEI_XML) if c["ref"]["xml_id"] == "b9")
        self.assertEqual(davis["paragraph_refs"], [])

    def test_venue_and_monograph_on_ref(self):
        claims = self._claims(AUTHOR_YEAR_TEI_XML)
        landa = next(c for c in claims if c["ref"]["xml_id"] == "b16")
        self.assertEqual(landa["ref"]["venue"], "Geophysical Prospecting")
        self.assertFalse(landa["ref"]["is_monograph"])

    def test_targetless_number_follows_what_grobid_linked_elsewhere(self):
        claims = self._claims(FOOTNOTE_BIB_TEI_XML)
        second = [c for c in claims if c["sentence"].startswith("The same method")]
        twenty_one = next(c for c in second if c["cite_text"] == "[21]")
        self.assertEqual(twenty_one["ref"]["xml_id"], "b53")
        self.assertEqual(twenty_one["resolution"], "number_map")
        self.assertTrue(twenty_one["resolved"])

    def test_unlinked_number_is_not_guessed_when_positions_are_inconsistent(self):
        claims = self._claims(FOOTNOTE_BIB_TEI_XML)
        twenty_two = next(c for c in claims if c["cite_text"] == "[22]")
        self.assertFalse(twenty_two["resolved"])
        self.assertEqual(twenty_two["resolution"], "unresolved")

    def test_position_fallback_still_works_when_consistent(self):
        # SAMPLE_TEI_XML: [1]→b0 is linked, so [2] at position 2 is safe.
        claims = extract_seed_citation_claims(BeautifulSoup(SAMPLE_TEI_XML, "xml"))
        self.assertEqual(claims[1]["resolution"], "position")
        self.assertTrue(claims[1]["resolved"])


@contextlib.contextmanager
def _audit_dirs():
    """Run an audit with its output redirected to a temporary directory."""
    import research_assistant.shared.seed_audit as sa
    with tempfile.TemporaryDirectory() as tmp, patch.object(sa, "AUDIT_DIR", tmp):
        yield tmp


class TestHonestOutcomes(unittest.TestCase):
    def test_failure_outcomes_carry_no_judgement(self):
        from research_assistant.shared.seed_audit import _judge_claim_entry
        item = {"claim": "Graphene is a semimetal with linear dispersion.", "document": "x.pdf", "context": ""}
        with patch("research_assistant.shared.seed_audit.hybrid_search", side_effect=ConnectionError("Failed to connect to Ollama")):
            _judge_claim_entry(item, MagicMock(), MagicMock(), [], [])
        self.assertEqual(item["outcome"], "retrieval_failed")
        self.assertIsNone(item["judgement"])
        self.assertIn("Ollama", item["reason"])

        item = {"claim": "Graphene is a semimetal with linear dispersion.", "document": "x.pdf", "context": ""}
        with patch("research_assistant.shared.seed_audit.hybrid_search", return_value=[]):
            _judge_claim_entry(item, MagicMock(), MagicMock(), [], [])
        self.assertEqual(item["outcome"], "no_evidence")
        self.assertIsNone(item["judgement"])

    def test_compute_totals_counts_verdicts_over_judged_only(self):
        results = [
            {"outcome": "judged", "judgement": "Supports", "downloaded": True, "reliability": "HIGH"},
            {"outcome": "judged", "judgement": "Does not support", "downloaded": True, "reliability": "UNSUPPORTED"},
            {"outcome": "judged", "judgement": "Unclear / insufficient evidence", "downloaded": True, "reliability": "UNRESOLVED"},
            {"outcome": "retrieval_failed", "judgement": None, "downloaded": True, "reliability": "UNRESOLVED"},
            {"outcome": "cap_exceeded", "judgement": None, "downloaded": True, "reliability": "UNRESOLVED"},
            {"outcome": "not_a_claim", "judgement": None, "downloaded": True, "reliability": "UNRESOLVED"},
            {"outcome": "not_downloaded", "judgement": None, "downloaded": False, "reliability": "UNRESOLVED"},
            {"outcome": "deferred_paywalled", "judgement": None, "downloaded": False, "reliability": "UNRESOLVED"},
        ]
        t = compute_totals(results)
        self.assertEqual(t["total"], 8)
        self.assertEqual(t["judged"], 3)
        self.assertEqual(t["Supports"], 1)
        self.assertEqual(t["Does not support"], 1)
        self.assertEqual(t["Unclear / insufficient evidence"], 1)   # the judged one only
        self.assertEqual(t["not_assessed"], {"retrieval_failed": 1, "cap_exceeded": 1, "not_a_claim": 1,
                                             "not_downloaded": 1, "deferred_paywalled": 1})
        self.assertEqual(t["coverage"], {"downloaded": 6, "attempted": 4, "judged": 3})
        self.assertEqual(t["reliability"]["unsupported"], 1)
        self.assertEqual(t["judged"] + sum(t["not_assessed"].values()), t["total"])

    def test_explain_is_outcome_first(self):
        self.assertIn("budget", explain_rubric_verdict({"outcome": "cap_exceeded", "judgement": None}))
        self.assertIn("backend", explain_rubric_verdict({"outcome": "not_attempted", "judgement": None}))
        self.assertIn("software", explain_rubric_verdict({"outcome": "not_a_claim", "role": "software", "judgement": None}))
        self.assertIn("fragment", explain_rubric_verdict({"outcome": "malformed_claim", "judgement": None}))
        self.assertIn("bibliography", explain_rubric_verdict({"outcome": "unresolved_ref", "judgement": None}))
        self.assertIn("cites", explain_rubric_verdict({"outcome": "cluster_skipped", "judgement": None}))
        # The "insufficient evidence" sentence is only reachable from a judged item.
        self.assertNotIn("fragmentary", explain_rubric_verdict({"outcome": "cap_exceeded", "judgement": None}))
        self.assertIn("fragmentary", explain_rubric_verdict({"outcome": "judged", "judgement": "Unclear / insufficient evidence"}))

    @patch("research_assistant.shared.seed_audit._judge_once")
    @patch("research_assistant.shared.seed_audit.hybrid_search")
    @patch("research_assistant.shared.seed_audit.find_tei_for_seed")
    @patch("research_assistant.shared.seed_audit._load_downloaded_manifest")
    def test_non_claims_are_partitioned_before_judging(self, mock_manifest, mock_find_tei, mock_search, mock_judge):
        with tempfile.NamedTemporaryFile("w", suffix=".tei.xml", delete=False, encoding="utf-8") as tf:
            tf.write(AUTHOR_YEAR_TEI_XML); tei_file = tf.name
        with tempfile.NamedTemporaryFile("w", suffix=".pdf", delete=False) as dummy_pdf:
            pdf = dummy_pdf.name
        mock_find_tei.return_value = tei_file
        mock_manifest.return_value = {
            k: {"key": k, "path": pdf, "xml_id": xid, "cited_by": "seed.pdf", "title": t}
            for k, xid, t in (
                ("a", "b9", "Algorithm 832"), ("b", "b16", "Path-integral seismic imaging"),
                ("c", "b25", "Target-oriented inversion using the patched green's function method"),
                ("d", "b27", "Subsurface-domain objective functions"), ("e", "b6", "Target-level waveform inversion"),
            )
        }
        mock_search.return_value = [{"text": "Evidence.", "metadata": {"document": os.path.basename(pdf), "section": "results", "page": 3}}]
        mock_judge.return_value = {
            "judgement": "Supports", "confidence": "High", "supporting_span": "Evidence.", "reason": "r",
            "slots": {"finding": {"assertion": "a", "verdict": "Supports"}, "scope": {"assertion": "s", "verdict": "Supports"},
                      "strength": {"assertion": "t", "verdict": "Not applicable"}},
            "evidence_sufficiency": "sufficient",
        }
        with _audit_dirs():
            report = audit_seed_citations("seed.pdf", search_resources=(MagicMock(), MagicMock(), [], []),
                                          max_claims=10, skip_if_cached=False)
        by_xid = {r["ref"]["xml_id"]: r for r in report["results"]}
        self.assertEqual(by_xid["b9"]["outcome"], "not_a_claim")     # software
        self.assertEqual(by_xid["b25"]["outcome"], "not_a_claim")    # method
        self.assertEqual(by_xid["b16"]["outcome"], "judged")
        self.assertEqual(mock_judge.call_count, 3)                    # b16, b27, b6
        self.assertEqual(report["totals"]["not_assessed"]["not_a_claim"], 2)
        self.assertEqual(by_xid["b9"]["reliability_explanation"][:12], "Not assessed")
        os.unlink(tei_file); os.unlink(pdf)


BREAKER_TEI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>
<div><head>Results</head>
<p xml:id="p_0"><s>Graphene shows extraordinary electronic mobility in suspended flakes <ref type="bibr" target="#b0">[1]</ref>.</s><s>Thermal conductance across grain boundaries is strongly suppressed <ref type="bibr" target="#b1">[2]</ref>.</s><s>A negative Poisson ratio was reported in the rippled lattice <ref type="bibr" target="#b2">[3]</ref>.</s><s>Shear strain opens a bandgap of several hundred meV <ref type="bibr" target="#b3">[4]</ref>.</s><s>Phase coherence survives up to room temperature in these devices <ref type="bibr" target="#b4">[5]</ref>.</s></p>
</div></body>
<back><listBibl>
<biblStruct xml:id="b0"><analytic><title level="a" type="main">Paper b0</title></analytic><monogr><title level="j">Nano Letters</title></monogr></biblStruct>
<biblStruct xml:id="b1"><analytic><title level="a" type="main">Paper b1</title></analytic><monogr><title level="j">Nano Letters</title></monogr></biblStruct>
<biblStruct xml:id="b2"><analytic><title level="a" type="main">Paper b2</title></analytic><monogr><title level="j">Nano Letters</title></monogr></biblStruct>
<biblStruct xml:id="b3"><analytic><title level="a" type="main">Paper b3</title></analytic><monogr><title level="j">Nano Letters</title></monogr></biblStruct>
<biblStruct xml:id="b4"><analytic><title level="a" type="main">Paper b4</title></analytic><monogr><title level="j">Nano Letters</title></monogr></biblStruct>
</listBibl></back></text></TEI>
"""


class TestBudgetAndBreaker(unittest.TestCase):
    def _c(self, section, cite_count, p, s, xid):
        return {"section": section, "cite_count": cite_count, "paragraph_index": p, "sentence_index": s,
                "ref": {"xml_id": xid}, "claim": f"claim {xid}"}

    def test_results_before_intro_and_singles_before_clusters(self):
        from research_assistant.shared.seed_audit import prioritise_claims
        intro_cluster = [self._c("other", 8, 0, 0, f"b{i}") for i in range(8)]
        intro_single = self._c("introduction", 1, 1, 0, "b20")
        methods_single = self._c("methods", 1, 5, 0, "b30")
        results_pair = [self._c("results", 2, 9, 2, "b40"), self._c("results", 2, 9, 2, "b41")]
        ordered, skipped = prioritise_claims(intro_cluster + [intro_single, methods_single] + results_pair)
        self.assertEqual([c["ref"]["xml_id"] for c in ordered[:4]], ["b40", "b41", "b30", "b20"])
        self.assertEqual(len(ordered), 4 + 3)          # at most 3 of the 8-cite sentence
        self.assertEqual(len(skipped), 5)
        self.assertTrue(all(c["outcome"] == "cluster_skipped" for c in skipped))

    @patch("research_assistant.shared.seed_audit._judge_once")
    @patch("research_assistant.shared.seed_audit.hybrid_search")
    @patch("research_assistant.shared.seed_audit.find_tei_for_seed")
    @patch("research_assistant.shared.seed_audit._load_downloaded_manifest")
    def test_three_consecutive_backend_failures_stop_the_run(self, mock_manifest, mock_find_tei, mock_search, mock_judge):
        # Five evidential claims on five references, all "downloaded": three
        # failures trip the breaker and two are left not_attempted.
        with tempfile.NamedTemporaryFile("w", suffix=".tei.xml", delete=False, encoding="utf-8") as tf:
            tf.write(BREAKER_TEI_XML); tei_file = tf.name
        with tempfile.NamedTemporaryFile("w", suffix=".pdf", delete=False) as dummy_pdf:
            pdf = dummy_pdf.name
        mock_find_tei.return_value = tei_file
        mock_manifest.return_value = {
            x: {"key": x, "path": pdf, "xml_id": x, "cited_by": "seed.pdf", "title": f"Paper {x}"}
            for x in ("b0", "b1", "b2", "b3", "b4")
        }
        mock_search.side_effect = ConnectionError("Failed to connect to Ollama")
        with _audit_dirs():
            report = audit_seed_citations("seed.pdf", search_resources=(MagicMock(), MagicMock(), [], []),
                                          max_claims=50, skip_if_cached=False)
        outcomes = [r["outcome"] for r in report["results"] if r.get("downloaded")]
        self.assertEqual(outcomes.count("retrieval_failed"), 3)
        self.assertEqual(outcomes.count("not_attempted"), 2)
        self.assertIsNotNone(report["aborted"])
        self.assertEqual(report["aborted"]["after_attempted"], 3)
        self.assertIn("Ollama", report["aborted"]["reason"])
        self.assertEqual(mock_judge.call_count, 0)
        self.assertEqual(report["totals"]["Unclear / insufficient evidence"], 0)
        os.unlink(tei_file); os.unlink(pdf)

    @patch("research_assistant.shared.seed_audit._judge_once")
    @patch("research_assistant.shared.seed_audit.hybrid_search")
    @patch("research_assistant.shared.seed_audit.find_tei_for_seed")
    @patch("research_assistant.shared.seed_audit._load_downloaded_manifest")
    def test_history_copy_is_written(self, mock_manifest, mock_find_tei, mock_search, mock_judge):
        with _audit_dirs() as tmp:
            with tempfile.NamedTemporaryFile("w", suffix=".tei.xml", delete=False, encoding="utf-8") as tf:
                tf.write(SAMPLE_TEI_XML); tei_file = tf.name
            mock_find_tei.return_value = tei_file
            mock_manifest.return_value = {}
            audit_seed_citations("seed.pdf", search_resources=(MagicMock(), MagicMock(), [], []), skip_if_cached=False)
            self.assertTrue(os.path.exists(os.path.join(tmp, "seed_audit.json")))
            history = os.listdir(os.path.join(tmp, "history"))
            self.assertEqual(len(history), 1)
            self.assertTrue(history[0].startswith("seed_") and history[0].endswith("_audit.json"))
            os.unlink(tei_file)


if __name__ == "__main__":
    unittest.main()

