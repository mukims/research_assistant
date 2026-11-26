"""Regression tests for the 7-step Citation Auditor remediation plan."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from research_assistant.agents.agent8_verifier import verify_draft
from research_assistant.audit.report import generate_seed_audit_markdown
from research_assistant.audit.resolver import _resolve_ref, enrich_item_bibliometrics
from research_assistant.judgement.judge import (
    build_citing_paper_note,
    compose_context,
    filter_relevant_artifacts,
    guard_abstract_only_verdict,
    infer_evidence_tier,
)
from research_assistant.judgement.policy import ReliabilityRating, evaluate_reliability
from research_assistant.judgement.source_assessor import SourceGrade, assess_source
from research_assistant.shared.chunking import (
    chunk_document,
    is_markdown_table,
    split_markdown_table_preserving_header,
)
from research_assistant.shared.claim_text import (
    classify_citation_role,
    describe_citation_locus,
    expand_numeric_cite_markers,
    is_non_citation_ref,
    split_author_year_cites,
)
from research_assistant.shared.search import (
    extract_salient_claim_terms,
    rerank_candidates,
    resolve_cross_encoder_target,
    supplement_intra_doc_lexical_hits,
)
from research_assistant.shared.seed_audit import (
    audit_seed_citations,
    compute_totals,
    cross_check_seed_audit,
    detect_uncited_claims,
    extract_seed_citation_claims,
    extract_seed_self_identity,
    extract_uncited_candidate_sentences,
)


_SAMPLE_TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <body>
      <div xmlns="http://www.tei-c.org/ns/1.0">
        <head>Results</head>
        <p><s>Room-temperature superconductivity occurs at 15 C in carbonaceous sulfur hydride <ref type="bibr" target="#b0">[1]</ref>.</s>
           <s>Hydrothermal MnFe2O4 electrodes achieve stable cycling over 5000 cycles <ref type="bibr" target="#b1">[2]</ref>.</s></p>
      </div>
    </body>
    <back>
      <listBibl>
        <biblStruct xml:id="b0">
          <analytic>
            <title level="a">Room-temperature superconductivity in a carbonaceous sulfur hydride</title>
            <author><persName><surname>Snider</surname></persName></author>
            <idno type="DOI">10.1038/s41586-020-2801-z</idno>
          </analytic>
          <monogr>
            <title level="j">Nature</title>
            <imprint><date when="2020" /></imprint>
          </monogr>
        </biblStruct>
        <biblStruct xml:id="b1">
          <analytic>
            <title level="a">Hydrothermal synthesis and cycling of MnFe2O4 electrodes</title>
            <author><persName><surname>Patel</surname></persName></author>
            <idno type="DOI">10.1016/j.jallcom.2021.999999</idno>
          </analytic>
          <monogr>
            <title level="j">Journal of Alloys and Compounds</title>
            <imprint><date when="2021" /></imprint>
          </monogr>
        </biblStruct>
      </listBibl>
    </back>
  </text>
</TEI>
"""


def _fake_supports_verdict(span_text: str):
    return {
        "slots": {
            "finding": {"assertion": "reported finding", "verdict": "Supports"},
            "scope": {"assertion": "reported scope", "verdict": "Supports"},
            "strength": {"assertion": "reported strength", "verdict": "Not applicable"},
        },
        "judgement": "Supports",
        "model_judgement": "Supports",
        "evidence_sufficiency": "sufficient",
        "confidence": "High",
        "supporting_span": span_text,
        "span_verified": True,
        "span_cites_others": False,
        "rubric_mismatch": False,
        "rubric_violations": [],
        "reason": "Supported verbatim by retrieved passage.",
    }


class TestStep1MetadataWiring(unittest.TestCase):
    def test_doi_url_prefix_normalised_for_preprint_detection(self):
        res = assess_source({
            "title": "Preprint with full URL DOI",
            "doi": "https://doi.org/10.1101/2024.05.01.592000",
        })
        self.assertEqual(res["grade"], SourceGrade.PREPRINT_UNREVIEWED.value)
        self.assertTrue(res["is_preprint"])

    def test_chunk_storage_type_does_not_pollute_bibliometric_metadata(self):
        item = {"ref": {"title": "Some Paper", "venue": "Journal of Alloys and Compounds"}}
        enrich_item_bibliometrics(
            item,
            dl_entry={"citation_count": 75, "is_retracted": False, "doi": "https://doi.org/10.1016/j.jallcom.2021.1"},
            hit_meta={"type": "text_chunk", "seq": 4, "document": "p.pdf"},
        )
        self.assertNotIn("type", item["metadata"])
        self.assertNotIn("seq", item["metadata"])
        self.assertEqual(item["metadata"]["citation_count"], 75)
        self.assertEqual(item["metadata"]["doi"], "10.1016/j.jallcom.2021.1")
        self.assertEqual(item["ref"]["doi"], "10.1016/j.jallcom.2021.1")
        grade = assess_source(metadata=item["metadata"], ref_info=item["ref"])
        self.assertEqual(grade["grade"], SourceGrade.RIGOROUS_PRIMARY.value)

    def test_audit_and_cross_check_propagate_retraction_and_citation_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed_pdf = os.path.join(tmp, "seed.pdf")
            with open(seed_pdf, "wb") as f:
                f.write(b"%PDF-1.4 dummy")
            tei_path = os.path.join(tmp, "seed.tei.xml")
            with open(tei_path, "w", encoding="utf-8") as f:
                f.write(_SAMPLE_TEI)
            pdf0 = os.path.join(tmp, "snider2020.pdf")
            pdf1 = os.path.join(tmp, "patel2021.pdf")
            for p in (pdf0, pdf1):
                with open(p, "wb") as f:
                    f.write(b"%PDF-1.4 dummy")

            manifest = {
                "doi:10.1038/s41586-020-2801-z": {
                    "key": "doi:10.1038/s41586-020-2801-z",
                    "path": pdf0,
                    "doi": "10.1038/s41586-020-2801-z",
                    "title": "Room-temperature superconductivity in a carbonaceous sulfur hydride",
                    "is_retracted": True,
                    "citation_count": 620,
                    "evidence_tier": "full_text",
                },
                "doi:10.1016/j.jallcom.2021.999999": {
                    "key": "doi:10.1016/j.jallcom.2021.999999",
                    "path": pdf1,
                    "doi": "10.1016/j.jallcom.2021.999999",
                    "title": "Hydrothermal synthesis and cycling of MnFe2O4 electrodes",
                    "is_retracted": False,
                    "citation_count": 92,
                    "evidence_tier": "full_text",
                },
            }

            texts = [
                "Room-temperature superconductivity occurs at 15 C in carbonaceous sulfur hydride.",
                "Hydrothermal MnFe2O4 electrodes achieve stable cycling over 5000 cycles.",
            ]
            metadatas = [
                {"document": "snider2020.pdf", "type": "text_chunk", "section": "results"},
                {"document": "patel2021.pdf", "type": "text_chunk", "section": "results"},
            ]

            def fake_search(query, collection, bm25, t, m, top_k=4, doc_filter=None, **kw):
                out = []
                for idx, meta in enumerate(metadatas):
                    if not doc_filter or meta["document"] in doc_filter:
                        out.append({
                            "chunk_index": idx,
                            "text": texts[idx],
                            "metadata": meta,
                            "rrf_score": 0.5,
                        })
                return out

            def fake_judge(claim, evidence, context=None):
                return _fake_supports_verdict(evidence.splitlines()[0])

            audit_dir = os.path.join(tmp, "audits")
            with (
                patch("research_assistant.shared.seed_audit.AUDIT_DIR", audit_dir),
                patch("research_assistant.shared.seed_audit.find_tei_for_seed", return_value=tei_path),
                patch("research_assistant.shared.seed_audit._load_downloaded_manifest", return_value=manifest),
                patch("research_assistant.shared.seed_audit.attach_cited_summaries"),
                patch("research_assistant.shared.seed_audit.contextualize_citation_queries"),
                patch("research_assistant.shared.seed_audit.hybrid_search", side_effect=fake_search),
                patch("research_assistant.shared.seed_audit._judge_once", side_effect=fake_judge),
            ):
                report = audit_seed_citations(
                    seed_pdf,
                    search_resources=(object(), object(), texts, metadatas),
                    force=True,
                )
                by_doc = {r["document"]: r for r in report["results"]}
                # Snider 2020 is retracted -> RETRACTED_OR_FLAWED -> LOW reliability despite Supports + High conf
                self.assertEqual(by_doc["snider2020.pdf"]["source_grade"], SourceGrade.RETRACTED_OR_FLAWED.value)
                self.assertEqual(by_doc["snider2020.pdf"]["reliability"], ReliabilityRating.LOW.value)
                # Patel 2021 is in J. Alloys Compd. (normally STANDARD_PRIMARY), but citation_count=92 >= 50 -> RIGOROUS_PRIMARY
                self.assertEqual(by_doc["patel2021.pdf"]["source_grade"], SourceGrade.RIGOROUS_PRIMARY.value)
                self.assertEqual(by_doc["patel2021.pdf"]["reliability"], ReliabilityRating.HIGH.value)

                # Simulate legacy cached report with empty item['metadata'] and verify cross_check retrofits it
                legacy = dict(report)
                legacy["results"] = []
                for r in report["results"]:
                    rc = dict(r)
                    rc["metadata"] = {}
                    rc["source_grade"] = SourceGrade.UNKNOWN.value
                    legacy["results"].append(rc)
                refreshed, _ = cross_check_seed_audit(
                    seed_pdf,
                    legacy,
                    search_resources=(object(), object(), texts, metadatas),
                    max_new_claims=0,
                )
                ref_by_doc = {r["document"]: r for r in refreshed["results"]}
                self.assertEqual(ref_by_doc["snider2020.pdf"]["source_grade"], SourceGrade.RETRACTED_OR_FLAWED.value)
                self.assertEqual(ref_by_doc["snider2020.pdf"]["reliability"], ReliabilityRating.LOW.value)
                self.assertEqual(ref_by_doc["patel2021.pdf"]["source_grade"], SourceGrade.RIGOROUS_PRIMARY.value)

    def test_agent8_verifier_propagates_manifest_retraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            draft_path = os.path.join(tmp, "draft.txt")
            cites_path = os.path.join(tmp, "draft_citations.json")
            ev_path = os.path.join(tmp, "draft_citation_evidence.json")
            with open(draft_path, "w", encoding="utf-8") as fh:
                fh.write("Room-temperature superconductivity occurs at 15 C \\cite{cite_1}.\n")
            with open(cites_path, "w", encoding="utf-8") as fh:
                json.dump({"Snider et al. 2020": "cite_1"}, fh)
            with open(ev_path, "w", encoding="utf-8") as fh:
                json.dump({
                    "cite_1": {
                        "document": "snider2020.pdf",
                        "evidence": "Room-temperature superconductivity occurs at 15 C.",
                        "metadata": {"venue": "Nature", "type": "text_chunk"},
                    }
                }, fh)
            manifest = {
                "doi:10.1038/s41586-020-2801-z": {
                    "path": "/corpus/snider2020.pdf",
                    "doi": "10.1038/s41586-020-2801-z",
                    "title": "Room-temperature superconductivity in a carbonaceous sulfur hydride",
                    "is_retracted": True,
                    "citation_count": 620,
                }
            }
            with (
                patch("research_assistant.agents.agent8_verifier._load_downloaded_manifest", return_value=manifest),
                patch(
                    "research_assistant.agents.agent8_verifier._judge_once",
                    return_value=_fake_supports_verdict("Room-temperature superconductivity occurs at 15 C."),
                ),
            ):
                rep = verify_draft(draft_path, search_resources=(None, None, [], []))
            entry = rep["results"][0]
            self.assertEqual(entry["source_grade"], SourceGrade.RETRACTED_OR_FLAWED.value)
            self.assertEqual(entry["reliability"], ReliabilityRating.LOW.value)
            self.assertEqual(rep["totals"]["reliability"]["low"], 1)


class TestStep2GrobidNoiseAndAuthorYear(unittest.TestCase):
    def test_non_citation_ref_filters_equations_code_and_lifespans(self):
        for noisy in (
            "",
            "(1.3)",
            "(1.198)",
            "(1.191",
            ".198)",
            "1.3.6",
            "[__name__]",
            "Fisher (1890-1962)",
            "Chapter 14)",
            "(Section B.6)",
            "[1:3]",
            "[0,",
            "(3, 4) X tensor([[ 0., 1., 2., 3.",
            "[ 4., 5., 6., 7.",
        ):
            self.assertTrue(is_non_citation_ref(noisy, has_valid_target=False), noisy)
        for valid in ("[12]", "[13–20]", "Bollobás (1999)", "(Kingma and Welling, 2014)"):
            self.assertFalse(is_non_citation_ref(valid, has_valid_target=False), valid)
        # Linked target overrides heuristic unless empty
        self.assertFalse(is_non_citation_ref("(1.3)", has_valid_target=True))

    def test_split_author_year_cites(self):
        self.assertEqual(
            split_author_year_cites("(Kingma and Welling, 2014, Rezende et al., 2014)"),
            ["(Kingma and Welling, 2014)", "(Rezende et al., 2014)"],
        )
        self.assertEqual(
            split_author_year_cites("(Goodfellow et al., 2014; Song and Ermon, 2019)"),
            ["(Goodfellow et al., 2014)", "(Song and Ermon, 2019)"],
        )
        self.assertEqual(split_author_year_cites("[12, 13]"), ["[12, 13]"])

    def test_accent_insensitive_and_first_author_ranked_resolution(self):
        bib = {
            "b1": {
                "xml_id": "b1",
                "authors": ["Y. Bengio", "I. Goodfellow"],
                "year": 2014,
                "title": "Coauthor paper",
                "raw_reference": "Bengio, Y. and Goodfellow, I. (2014).",
            },
            "b2": {
                "xml_id": "b2",
                "authors": ["I. Goodfellow", "J. Pouget-Abadie", "M. Mirza"],
                "year": 2014,
                "title": "Generative Adversarial Nets",
                "raw_reference": "Goodfellow, I. et al. (2014). Generative Adversarial Nets.",
            },
            "b3": {
                "xml_id": "b3",
                "authors": ["B. Bollobás"],
                "year": 1999,
                "title": "Linear Analysis",
                "raw_reference": "Bollobás, B. (1999). Linear Analysis.",
            },
        }
        hit_gan, how, _ = _resolve_ref("", "(Goodfellow et al., 2014)", bib, {}, {})
        self.assertEqual(how, "surname")
        self.assertEqual(hit_gan["xml_id"], "b2")

        hit_accent, how2, _ = _resolve_ref("", "Bollobas (1999)", bib, {}, {})
        self.assertEqual(how2, "surname")
        self.assertEqual(hit_accent["xml_id"], "b3")

        hit_accent2, _, _ = _resolve_ref("", "Bollobás (1999)", {"b3": {**bib["b3"], "authors": ["B. Bollobas"]}}, {}, {})
        self.assertEqual(hit_accent2["xml_id"], "b3")

    def test_extract_claims_drops_equation_noise_and_stitches_split_author_year_refs(self):
        tei = """<?xml version="1.0" encoding="UTF-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <text>
            <body>
              <div>
                <p><s>Substituting Eq. <ref type="bibr">(1.3)</ref> and <ref type="bibr">(1.191</ref><ref type="bibr">.198)</ref> yields the bound.</s>
                   <s>Pioneered by Ronald <ref type="bibr">Fisher (1890</ref><ref type="bibr">Fisher ( -1962) )</ref> in statistics.</s>
                   <s>Variational autoencoders were introduced simultaneously <ref type="bibr">(Kingma and</ref> <ref type="bibr">Welling, 2014, Rezende et al., 2014)</ref> for latent inference.</s></p>
              </div>
            </body>
            <back>
              <listBibl>
                <biblStruct xml:id="b0">
                  <analytic>
                    <title level="a">Auto-Encoding Variational Bayes</title>
                    <author><persName><surname>Kingma</surname></persName></author>
                    <author><persName><surname>Welling</surname></persName></author>
                  </analytic>
                  <monogr><imprint><date when="2014" /></imprint></monogr>
                </biblStruct>
                <biblStruct xml:id="b1">
                  <analytic>
                    <title level="a">Stochastic Backpropagation and Approximate Inference</title>
                    <author><persName><surname>Rezende</surname></persName></author>
                    <author><persName><surname>Mohamed</surname></persName></author>
                    <author><persName><surname>Wierstra</surname></persName></author>
                  </analytic>
                  <monogr><imprint><date when="2014" /></imprint></monogr>
                </biblStruct>
              </listBibl>
            </back>
          </text>
        </TEI>
        """
        from bs4 import BeautifulSoup
        claims = extract_seed_citation_claims(BeautifulSoup(tei, "xml"))
        # Equation refs and Fisher lifespan produce 0 claims; stitched VAE ref produces 2 resolved claims
        self.assertEqual(len(claims), 2)
        self.assertTrue(all(c["resolved"] for c in claims))
        self.assertEqual({c["ref"]["xml_id"] for c in claims}, {"b0", "b1"})


class TestStep3ScorecardSeparation(unittest.TestCase):
    def test_reliability_breakdown_partitions_total_and_separates_non_claims_and_paywalled(self):
        results = [
            {"outcome": "judged", "judgement": "Supports", "downloaded": True, "reliability": "HIGH"},
            {"outcome": "judged", "judgement": "Partially supports", "downloaded": True, "reliability": "MODERATE"},
            {"outcome": "judged", "judgement": "Supports", "downloaded": True, "reliability": "LOW"},
            {"outcome": "judged", "judgement": "Contradicts", "downloaded": True, "reliability": "CONTRADICTED"},
            {"outcome": "judged", "judgement": "Does not support", "downloaded": True, "reliability": "UNSUPPORTED"},
            {"outcome": "judged", "judgement": "Unclear / insufficient evidence", "downloaded": True, "reliability": "UNRESOLVED"},
            {"outcome": "cap_exceeded", "judgement": None, "downloaded": True, "reliability": "UNRESOLVED"},
            {"outcome": "not_a_claim", "role": "software", "judgement": None, "downloaded": False, "reliability": "UNRESOLVED"},
            {"outcome": "malformed_claim", "judgement": None, "downloaded": False, "reliability": "UNRESOLVED"},
            {"outcome": "not_downloaded", "judgement": None, "downloaded": False, "reliability": "UNRESOLVED"},
            {"outcome": "deferred_paywalled", "judgement": None, "downloaded": False, "reliability": "UNRESOLVED"},
        ]
        totals = compute_totals(results)
        # Legacy reliability key remains 100% intact (6 UNRESOLVED items total)
        self.assertEqual(totals["reliability"]["unresolved"], 6)
        self.assertEqual(totals["non_evidential"], 2)
        self.assertEqual(totals["evidential_citations"], 9)

        rb = totals["reliability_breakdown"]
        self.assertEqual(rb["inconclusive"], 2)            # Unclear + cap_exceeded
        self.assertEqual(rb["unavailable_paywalled"], 2)   # not_downloaded + deferred_paywalled
        self.assertEqual(rb["non_evidential_skipped"], 2)  # not_a_claim + malformed_claim
        self.assertEqual(sum(rb.values()), len(results))
        self.assertEqual(
            rb["inconclusive"] + rb["unavailable_paywalled"] + rb["non_evidential_skipped"],
            totals["reliability"]["unresolved"],
        )

        # Markdown renders the separated rows cleanly
        for r in results:
            r.setdefault("ref", {"xml_id": "b0", "index": 1, "title": "T", "authors": ["A"], "year": 2020})
            r.setdefault("sentence", "S."); r.setdefault("claim", "S.")
            r.setdefault("paragraph_id", "p0"); r.setdefault("paragraph_refs", [])
        md = generate_seed_audit_markdown({
            "seed_name": "test.pdf",
            "results": results,
            "totals": totals,
        })
        self.assertIn("Inconclusive / Unjudged (In Corpus)", md)
        self.assertIn("Unavailable / Paywalled", md)
        self.assertIn("Non-Evidential (Excluded)", md)
        self.assertNotIn("Unresolved / Pending", md)


class TestStep4MultiCitationAttribution(unittest.TestCase):
    def test_describe_citation_locus_isolates_clauses_and_coordinate_items(self):
        cites = {
            0: {"txt": "[2]"},
            1: {"txt": "[3]"},
            2: {"txt": "[14]"},
        }
        sent = (
            "Inverse problems are widely used in medical imaging ⟦C0⟧ and seismology ⟦C1⟧ , "
            "while spatial mapping of quantum devices remains challenging ⟦C2⟧ ."
        )
        loc0 = describe_citation_locus(sent, 0, cites)
        loc1 = describe_citation_locus(sent, 1, cites)
        loc2 = describe_citation_locus(sent, 2, cites)

        self.assertEqual(loc0["mode"], "multi_locus")
        self.assertIn("medical imaging", loc0["target_clause"])
        self.assertNotIn("seismology", loc0["target_clause"])
        self.assertNotIn("quantum devices", loc0["target_clause"])
        self.assertIn("[THIS CITATION: 2]", loc0["focal_sentence"])

        self.assertEqual(loc1["mode"], "multi_locus")
        self.assertIn("seismology", loc1["target_clause"])
        self.assertNotIn("quantum devices", loc1["target_clause"])

        self.assertEqual(loc2["mode"], "multi_locus")
        self.assertIn("quantum devices", loc2["target_clause"])
        self.assertNotIn("seismology", loc2["target_clause"])

    def test_describe_citation_locus_identifies_contiguous_clusters_and_single_cites(self):
        cites_cluster = {0: {"txt": "[13,"}, 1: {"txt": "14,"}, 2: {"txt": "15]"}}
        sent_cluster = "Graphene nanoribbons can be patterned by lithography, unzipping, or synthesis ⟦C0⟧ ⟦C1⟧ ⟦C2⟧ ."
        loc_c = describe_citation_locus(sent_cluster, 0, cites_cluster)
        self.assertEqual(loc_c["mode"], "cluster")
        self.assertIsNone(loc_c["target_clause"])
        self.assertIn("1 of 3 references", loc_c["attribution_note"])

        cites_single = {0: {"txt": "[5]"}}
        sent_single = "Graphene exhibits linear dispersion near the Dirac point ⟦C0⟧ ."
        loc_s = describe_citation_locus(sent_single, 0, cites_single)
        self.assertEqual(loc_s["mode"], "single")
        self.assertIsNone(loc_s["attribution_note"])


class TestStep5IntraPaperRetrievalAndReranking(unittest.TestCase):
    def test_salient_terms_preserve_formulas_and_numbers_without_title_noise(self):
        terms = extract_salient_claim_terms(
            "MnFe2O4@PANI electrodes reach 623 F/g in 1M H2SO4 using GGA-PBE."
        )
        joined = " ".join(terms)
        self.assertIn("623", joined)
        self.assertIn("MnFe2O4@PANI", joined)
        self.assertIn("GGA-PBE", joined)

    def test_lexical_supplement_rescues_exact_results_chunk_missed_by_dense_pool(self):
        texts = [
            "Introduction: Supercapacitors have attracted broad interest [1-12].",
            "Methods: Samples were dried overnight at 60 C.",
            "Results: Symmetric MnFe2O4@PANI cells deliver 623 F/g in 1M H2SO4.",
        ]
        metas = [
            {"document": "p.pdf", "type": "text_chunk", "section": "introduction"},
            {"document": "p.pdf", "type": "text_chunk", "section": "methods"},
            {"document": "p.pdf", "type": "text_chunk", "section": "results"},
        ]
        initial_hits = [{"chunk_index": 0, "text": texts[0], "metadata": metas[0], "rrf_score": 0.03}]
        merged = supplement_intra_doc_lexical_hits(
            ["MnFe2O4@PANI reaches 623 F/g in 1M H2SO4"],
            {"p.pdf"},
            None,
            texts,
            metas,
            initial_hits,
        )
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[1]["chunk_index"], 2)

        reranked = rerank_candidates(
            "supercapacitor electrode performance",
            merged,
            top_k=2,
            claim_text="MnFe2O4@PANI reaches 623 F/g in 1M H2SO4",
            primary_k=1,
        )
        # Results chunk with exact quantitative & formula match is promoted to #0
        self.assertEqual(reranked[0]["chunk_index"], 2)

    def test_section_and_adjacency_diversity_prevents_intro_monopoly(self):
        cands = [
            {"chunk_index": 0, "text": "Graphene transport overview [1] [2] [3] [4].", "metadata": {"document": "g.pdf", "section": "introduction"}, "rrf_score": 0.033},
            {"chunk_index": 1, "text": "Adjacent intro sentence on graphene transport.", "metadata": {"document": "g.pdf", "section": "introduction"}, "rrf_score": 0.032},
            {"chunk_index": 4, "text": "Background on graphene transport models.", "metadata": {"document": "g.pdf", "section": "introduction"}, "rrf_score": 0.031},
            {"chunk_index": 8, "text": "Abstract summary of graphene transport.", "metadata": {"document": "g.pdf", "section": "abstract"}, "rrf_score": 0.030},
            {"chunk_index": 15, "text": "Measured graphene transport mobility reaches 200000 cm2/Vs at 300 K.", "metadata": {"document": "g.pdf", "section": "results"}, "rrf_score": 0.028},
        ]
        out = rerank_candidates(
            "graphene transport mobility 200000 cm2/Vs",
            cands,
            top_k=5,
            claim_text="Graphene transport mobility reaches 200000 cm2/Vs.",
            neighbour_window=1,
            primary_k=3,
        )
        top3_indices = {h["chunk_index"] for h in out[:3]}
        # Results chunk 15 is in top 3, and adjacent chunks (0, 1) do not both consume primary top-3 slots
        self.assertIn(15, top3_indices)
        self.assertFalse({0, 1}.issubset(top3_indices))


class TestStep6SelectiveContextInjection(unittest.TestCase):
    def test_filter_relevant_artifacts_keeps_only_claim_referenced_figures_and_tables(self):
        arts = [
            {"label": "Fig. 1", "caption": "Schematic of device geometry."},
            {"label": "Fig. 2", "caption": "Nyquist impedance spectra at 300 K."},
            {"label": "Fig. 3", "caption": "Cycling stability over 5000 cycles."},
            {"label": "Table 1", "caption": "Comparison of specific capacitance."},
        ]
        # Claim mentions Fig. 2b only -> keeps Fig. 2, drops Fig. 1, Fig. 3, Table 1
        kept = filter_relevant_artifacts(
            "As shown in Fig. 2b, the charge-transfer semicircle shrinks markedly.", arts
        )
        self.assertEqual([a["label"] for a in kept], ["Fig. 2"])

        # Plural Figs. 1 and 3
        kept_plural = filter_relevant_artifacts(
            "Data in Figs. 1 and 3 confirm uniform morphology and cycling retention.", arts
        )
        self.assertEqual([a["label"] for a in kept_plural], ["Fig. 1", "Fig. 3"])

        # Plain prose sentence in same paragraph -> keeps 0 captions
        kept_none = filter_relevant_artifacts(
            "Prior studies attributed this enhancement to hydrogen bonding.", arts
        )
        self.assertEqual(kept_none, [])

        ctx = compose_context("«As shown in Fig. 2b...»", section="2. Results > 2.2. EIS", artifacts=kept)
        self.assertIn("[Section: 2. Results > 2.2. EIS]", ctx)
        self.assertIn("[Fig. 2: Nyquist impedance spectra at 300 K.]", ctx)
        self.assertNotIn("Fig. 1", ctx)


class TestStep7EvalExpansionAndUncitedDetector(unittest.TestCase):
    def test_regenerated_transforms_have_zero_prompt_leakage(self):
        from research_assistant.judgement.evalset import load_cases, split_held_out
        from research_assistant.judgement.judge import PROMPT_TEMPLATE

        cases = load_cases()
        held, refused = split_held_out(cases, PROMPT_TEMPLATE)
        non_prompt_refused = [c["id"] for c in refused if c["source"] != "prompt_example"]
        self.assertEqual(non_prompt_refused, [])
        self.assertGreaterEqual(len(held), 55)

    def test_uncited_claim_detector_extracts_factual_uncited_sentences_and_skips_discourse(self):
        tei = """<?xml version="1.0" encoding="UTF-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <text>
            <body>
              <div>
                <head>Introduction</head>
                <p><s>In this paper, we present a variational framework for transport simulations.</s>
                   <s>Prior experiments established that suspended graphene mobility exceeds 200000 cm2/Vs at low temperature.</s>
                   <s>Landauer transport was formulated for mesoscopic conductors <ref type="bibr" target="#b0">[1]</ref>.</s></p>
              </div>
            </body>
          </text>
        </TEI>
        """
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(tei, "xml")
        cands = extract_uncited_candidate_sentences(soup)
        self.assertEqual(len(cands), 1)
        self.assertIn("suspended graphene mobility", cands[0]["sentence"])

        flagged = detect_uncited_claims(soup, batch_checker=lambda batch: [True for _ in batch])
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0]["section"], "introduction")


class TestRound2Improvement1AbstractOnlyGuard(unittest.TestCase):
    def test_abstract_only_caps_does_not_support_to_unclear(self):
        v = {
            "judgement": "Does not support",
            "evidence_sufficiency": "partial",
            "confidence": "High",
            "supporting_span": None,
            "reason": "The abstract discusses PbTe thermopower without listing 170 uV/K.",
            "rubric_violations": [],
        }
        guarded = guard_abstract_only_verdict(v, "abstract_only")
        self.assertEqual(guarded["judgement"], "Unclear / insufficient evidence")
        self.assertEqual(guarded["model_judgement"], "Does not support")
        self.assertTrue(guarded["rubric_mismatch"])
        self.assertIn("abstract", guarded["reason"].lower())

        # Full-text Does not support is untouched
        untouched = guard_abstract_only_verdict(v, "full_text")
        self.assertEqual(untouched["judgement"], "Does not support")

        # Policy also maps abstract_only Does not support to UNRESOLVED rather than UNSUPPORTED
        rel = evaluate_reliability("Does not support", evidence_tier="abstract_only")
        self.assertEqual(rel["rating"], ReliabilityRating.UNRESOLVED.value)

    def test_infer_evidence_tier_handles_stored_form_and_prior_tier(self):
        self.assertEqual(
            infer_evidence_tier([{"metadata": {"stored_form": "abstract_only"}}]),
            "abstract_only",
        )
        self.assertEqual(
            infer_evidence_tier([{"metadata": {"document": "x"}}], prior_tier="abstract_only"),
            "abstract_only",
        )
        self.assertEqual(
            infer_evidence_tier([{"metadata": {"stored_form": "text_chunk"}}], prior_tier="abstract_only"),
            "full_text",
        )


class TestRound2Improvement2CitingPaperSelfIdentity(unittest.TestCase):
    def test_extracts_self_coined_framework_from_abstract_and_title(self):
        from bs4 import BeautifulSoup
        tei = """<?xml version="1.0" encoding="UTF-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <teiHeader>
            <fileDesc>
              <titleStmt><title type="main">Reimagining research papers as interactive AI agents</title></titleStmt>
            </fileDesc>
            <profileDesc>
              <abstract><p>Here we introduce Paper2Agent, an automated framework that converts papers into agents.</p></abstract>
            </profileDesc>
          </teiHeader>
        </TEI>
        """
        title, names = extract_seed_self_identity(BeautifulSoup(tei, "xml"))
        self.assertEqual(title, "Reimagining research papers as interactive AI agents")
        self.assertEqual(names, ["Paper2Agent"])

        note = build_citing_paper_note(
            "Paper2Agent represents each scientific paper as a Model Context Protocol (MCP) server.",
            seed_title=title,
            seed_self_names=names,
        )
        self.assertIsNotNone(note)
        self.assertIn("Paper2Agent", note)
        self.assertIn("Do NOT require", note)

        # Unrelated claim without Paper2Agent gets None (zero distraction)
        self.assertIsNone(
            build_citing_paper_note(
                "Prior tools focused on containerization for reproducibility.",
                seed_title=title,
                seed_self_names=names,
            )
        )


class TestRound2Improvement3RangeExpansionAndMethodRole(unittest.TestCase):
    def test_expand_numeric_cite_markers(self):
        self.assertEqual(expand_numeric_cite_markers("[13–16]"), ["[13]", "[14]", "[15]", "[16]"])
        self.assertEqual(expand_numeric_cite_markers("[4, 7, 9-10]"), ["[4]", "[7]", "[9]", "[10]"])
        self.assertEqual(expand_numeric_cite_markers("41–43"), ["41", "42", "43"])
        self.assertEqual(expand_numeric_cite_markers("[12]"), ["[12]"])
        self.assertEqual(expand_numeric_cite_markers("[9,"), ["[9,"])

    def test_computational_setup_role_classified_as_method(self):
        sent = (
            "All calculations were carried out using a double-zeta basis set with polarization "
            "orbitals within the GGA-PBE exchange-correlation functional ⟦C0⟧ ⟦C1⟧ ."
        )
        self.assertEqual(classify_citation_role(sent, "⟦C0⟧", {"title": "Generalized Gradient Approximation Made Simple"}), "method")
        self.assertEqual(classify_citation_role(sent, "⟦C1⟧", {"title": "SIESTA"}), "method")

    def test_extract_claims_expands_single_and_split_numeric_ranges(self):
        from bs4 import BeautifulSoup
        tei = """<?xml version="1.0" encoding="UTF-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <text>
            <body>
              <div>
                <p><s>Graphene nanoribbons have been fabricated by lithography, unzipping, and synthesis <ref type="bibr">[1–</ref><ref type="bibr">3]</ref>.</s></p>
              </div>
            </body>
            <back>
              <listBibl>
                <biblStruct xml:id="b0"><analytic><title level="a">STM lithography of nanoribbons</title><author><persName><surname>Tapaszto</surname></persName></author></analytic><monogr><imprint><date when="2008"/></imprint></monogr></biblStruct>
                <biblStruct xml:id="b1"><analytic><title level="a">Longitudinal unzipping of nanotubes</title><author><persName><surname>Kosynkin</surname></persName></author></analytic><monogr><imprint><date when="2009"/></imprint></monogr></biblStruct>
                <biblStruct xml:id="b2"><analytic><title level="a">Atomically precise bottom-up nanoribbons</title><author><persName><surname>Cai</surname></persName></author></analytic><monogr><imprint><date when="2010"/></imprint></monogr></biblStruct>
              </listBibl>
            </back>
          </text>
        </TEI>
        """
        claims = extract_seed_citation_claims(BeautifulSoup(tei, "xml"))
        self.assertEqual(len(claims), 3)
        self.assertEqual([c["ref"]["xml_id"] for c in claims], ["b0", "b1", "b2"])


class TestRound2Improvement4TableChunking(unittest.TestCase):
    def test_split_markdown_table_preserves_header_and_dedupes_caption_table_md(self):
        from research_assistant.shared.extract import Document, Figure, Paragraph, Section

        rows = ["| Material | S (uV/K) | PF (mW/mK2) |", "| --- | --- | --- |"]
        for i in range(12):
            rows.append(f"| Compound_{i:02d} | {150 + i * 10} | {1.2 + i * 0.2:.1f} |")
        tbl_md = "\n".join(rows)
        self.assertTrue(is_markdown_table(tbl_md))

        split_chunks = split_markdown_table_preserving_header(
            tbl_md, hard_max=220, caption_prefix="Table 1: Thermoelectric metrics."
        )
        self.assertGreaterEqual(len(split_chunks), 3)
        for ch in split_chunks:
            self.assertIn("Table 1: Thermoelectric metrics.", ch)
            self.assertIn("| Material | S (uV/K) | PF (mW/mK2) |", ch)
            self.assertIn("| --- | --- | --- |", ch)

        # Verify extract_tei-style Figure (where caption already had table_md appended) does NOT duplicate table_md
        short_tbl = "| Mat | S |\n| --- | --- |\n| PbTe | 170 |"
        fig = Figure(
            "tab_1", "table", "Table 1", f"Thermopower.\n\n{short_tbl}", 2, None, table_markdown=short_tbl
        )
        doc = Document("d.pdf", "Title", [], [Section("Results", "results", [Paragraph(tbl_md, 2)])], [fig], "grobid")
        doc_chunks = chunk_document(doc, target=180, hard_max=240)
        cap_chunks = [c for c in doc_chunks if c.type == "caption"]
        self.assertEqual(len(cap_chunks), 1)
        self.assertEqual(cap_chunks[0].text.count("| PbTe | 170 |"), 1)
        # Numeric Markdown table in section paragraphs survived low alpha_ratio and kept headers in every chunk
        text_chunks = [c for c in doc_chunks if c.type == "text"]
        self.assertGreaterEqual(len(text_chunks), 2)
        for tc in text_chunks:
            self.assertIn("| Material | S (uV/K) | PF (mW/mK2) |", tc.text)


class TestRound2Improvement5SelectiveVotingAndCrossEncoder(unittest.TestCase):
    def test_selective_voting_casts_tie_breakers_only_on_disputed_or_hedged_verdicts(self):
        import types
        from research_assistant.judgement import judge as judge_mod

        def _raw_reply(verdict_name: str, finding: str = "Supports"):
            return types.SimpleNamespace(content=json.dumps({
                "slots": {
                    "finding": {"assertion": "graphene mobility", "verdict": finding},
                    "scope": {"assertion": "graphene", "verdict": "Supports"},
                    "strength": {"assertion": "high", "verdict": "Partially supports" if verdict_name == "Partially supports" else "Supports"},
                },
                "judgement": verdict_name,
                "evidence_sufficiency": "sufficient",
                "confidence": "High",
                "supporting_span": "graphene mobility is high",
                "reason": "because",
            }))

        # 1. Supports on pass 1 returns immediately in 1 call (zero extra cost)
        with (
            patch.object(judge_mod, "JUDGEMENT_VOTES", 1),
            patch.object(judge_mod, "JUDGEMENT_SELECTIVE_VOTES", True),
            patch.object(judge_mod, "chat", side_effect=[_raw_reply("Supports")]) as m_chat,
        ):
            res_sup = judge_mod.judge("graphene mobility is high", "graphene mobility is high")
            self.assertEqual(res_sup["judgement"], "Supports")
            self.assertEqual(m_chat.call_count, 1)

        # 2. Fluke Contradicts on pass 1 followed by 2 Supports tie-breakers resolves to Supports (3 calls)
        with (
            patch.object(judge_mod, "JUDGEMENT_VOTES", 1),
            patch.object(judge_mod, "JUDGEMENT_SELECTIVE_VOTES", True),
            patch.object(
                judge_mod,
                "chat",
                side_effect=[
                    _raw_reply("Contradicts", finding="Contradicts"),
                    _raw_reply("Supports", finding="Supports"),
                    _raw_reply("Supports", finding="Supports"),
                ],
            ) as m_chat2,
        ):
            res_tie = judge_mod.judge("graphene mobility is high", "graphene mobility is high")
            self.assertEqual(res_tie["judgement"], "Supports")
            self.assertEqual(m_chat2.call_count, 3)
            self.assertEqual(res_tie["votes"], {"Contradicts": 1, "Supports": 2})

    def test_resolve_cross_encoder_target_auto_discovers_local_dir_and_respects_explicit_disable(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                resolve_cross_encoder_target(use_flag=False, model_name="m", local_dir=tmp, explicit_env=None)
            )
            with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as fh:
                fh.write("{}")
            # Local weights auto-discovered offline
            self.assertEqual(
                resolve_cross_encoder_target(use_flag=False, model_name="m", local_dir=tmp, explicit_env=None),
                tmp,
            )
            # Explicit CITATION_USE_CROSS_ENCODER=0 overrides auto-discovery
            self.assertIsNone(
                resolve_cross_encoder_target(use_flag=False, model_name="m", local_dir=tmp, explicit_env="0")
            )


class TestRound2Improvement6VerifierEvalCoverage(unittest.TestCase):
    def test_corpus_backed_verifier_gold_cases_exceed_thirty_across_all_verdict_classes(self):
        from research_assistant.judgement.evalset import load_cases, split_held_out
        from research_assistant.judgement.harvest import candidates_from_verification
        from research_assistant.judgement.judge import PROMPT_TEMPLATE, VALID_JUDGEMENTS

        cases = load_cases()
        held, refused = split_held_out(cases, PROMPT_TEMPLATE)
        leaked = [c["id"] for c in refused if c["source"] != "prompt_example"]
        self.assertEqual(leaked, [])
        self.assertGreaterEqual(len(held), 75)

        verifier_cases = [c for c in held if c.get("document") and c.get("citation_source")]
        self.assertGreaterEqual(len(verifier_cases), 30)
        covered_labels = {c["expected_judgement"] for c in verifier_cases}
        self.assertEqual(covered_labels, set(VALID_JUDGEMENTS))

        # Also verify harvest preserves document when present
        cands = candidates_from_verification(
            {
                "results": [
                    {
                        "outcome": "judged",
                        "claim": "C1",
                        "evidence": "E1",
                        "document": "doc1.pdf",
                        "citation_source": "Src1",
                        "judgement": "Supports",
                    }
                ]
            },
            prefix="v1",
        )
        self.assertEqual(cands[0]["document"], "doc1.pdf")

    def test_span_is_verbatim_tolerates_grobid_footnote_space_before_punctuation(self):
        from research_assistant.judgement.judge import span_is_verbatim

        ev = "First, GWAS do not directly pinpoint relevant genes, as most associated variants are non-coding 13 . Moreover, 0.5 eV and -0.43 eV remain exact."
        span = "First, GWAS do not directly pinpoint relevant genes, as most associated variants are non-coding 13."
        self.assertTrue(span_is_verbatim(span, ev))
        # Different decimal or sign still fails
        self.assertFalse(span_is_verbatim("Moreover, 0.43 eV remain exact.", ev))


if __name__ == "__main__":
    unittest.main()
