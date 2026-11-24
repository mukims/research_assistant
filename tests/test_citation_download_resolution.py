"""Regression tests for end-to-end citation extraction, OA PDF downloading, and audit resolution."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from bs4 import BeautifulSoup

import orchestrate
from research_assistant.agents import agent1_extractor, agent2_fetcher
from research_assistant.audit.resolver import _load_downloaded_manifest, _match_downloaded_paper, _resolve_ref
from research_assistant.shared.academic_apis import (
    _collect_openalex_oa_urls,
    clean_reference_title,
    title_variants,
    verify_candidate_match,
)
from research_assistant.shared.claim_text import is_numeric_cite
from research_assistant.shared.fetch import candidate_pdf_urls
from research_assistant.shared.paper_index import PaperEntry, PaperIndex
from research_assistant.shared.seed_audit import extract_seed_citation_claims
from research_assistant.shared.source_key import (
    extract_arxiv_from_text,
    extract_doi_from_text,
    normalise_doi,
    source_key,
)


class TestIdentifierCleaningAndExtraction(unittest.TestCase):
    def test_normalise_doi_strips_grobid_tail_noise_and_preserves_sici_parens(self):
        self.assertEqual(
            normalise_doi("10.1007/978-981-96-6259-3website"),
            "10.1007/978-981-96-6259-3",
        )
        self.assertEqual(
            normalise_doi("10.1109/tpami.2024.3386927/mm1"),
            "10.1109/tpami.2024.3386927",
        )
        self.assertEqual(
            normalise_doi("https://dx.doi.org/10.1016/0003-4916(91)90372-f)"),
            "10.1016/0003-4916(91)90372-f",
        )

    def test_extract_arxiv_and_doi_from_raw_text(self):
        self.assertEqual(
            extract_arxiv_from_text("doi:10.48550/arXiv.2401.12345v2"),
            "2401.12345v2",
        )
        self.assertEqual(
            extract_arxiv_from_text("J. Smith et al., arXiv preprint arXiv:2305.09876 (2023)."),
            "2305.09876",
        )
        self.assertEqual(
            extract_arxiv_from_text("Available at https://arxiv.org/abs/cond-mat/0611602."),
            "cond-mat/0611602",
        )
        self.assertEqual(
            extract_doi_from_text("K. Novoselov, Science 306, 666 (2004), 10.1126/science.1102896."),
            "10.1126/science.1102896",
        )


class TestExtractorReferenceCompleteness(unittest.TestCase):
    def test_grobid_parse_extracts_arxiv_pmid_and_keeps_titleless_software_or_preprint_refs(self):
        tei = """<?xml version="1.0" encoding="UTF-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <text><back><listBibl>
            <biblStruct xml:id="b0">
              <analytic>
                <title>Quantum transport in nanowires</title>
                <author><persName><forename>J.</forename><surname>Smith</surname></persName></author>
                <idno type="arXiv">arXiv:2308.11188v1</idno>
                <idno type="PMID">34567890</idno>
              </analytic>
              <monogr><imprint><date when="2023"/></imprint></monogr>
            </biblStruct>
            <biblStruct xml:id="b28">
              <analytic>
                <author><persName><forename>J.</forename><surname>Bloom</surname></persName></author>
              </analytic>
              <monogr>
                <ptr target="https://arxiv.org/abs/2406.01234"/>
                <imprint><date when="2024"/></imprint>
              </monogr>
            </biblStruct>
          </listBibl></back></text>
        </TEI>"""
        with tempfile.NamedTemporaryFile("w", suffix=".grobid.tei.xml", delete=False, encoding="utf-8") as fh:
            fh.write(tei)
            tei_path = fh.name
        try:
            agent1_extractor._parse_tei_file_cached.cache_clear()
            with patch.object(agent1_extractor, "_tei_path_for", return_value=tei_path):
                refs = agent1_extractor._references_from_grobid("seed.pdf")
            self.assertEqual(len(refs), 2)
            self.assertEqual(refs[0].arxiv_id, "2308.11188v1")
            self.assertEqual(refs[0].pmid, "34567890")
            self.assertEqual(refs[1].arxiv_id, "2406.01234")
            self.assertIn("Bloom", refs[1].raw_reference)
        finally:
            os.unlink(tei_path)


class TestCandidateMatchingAndOaUrls(unittest.TestCase):
    def test_landing_urls_rewrite_to_direct_pdf_candidates(self):
        self.assertEqual(
            candidate_pdf_urls("https://arxiv.org/abs/2401.12345v1")[0],
            "https://arxiv.org/pdf/2401.12345v1.pdf",
        )
        self.assertEqual(
            candidate_pdf_urls("https://openreview.net/forum?id=AbCdEf123")[0],
            "https://openreview.net/pdf?id=AbCdEf123",
        )
        self.assertEqual(
            candidate_pdf_urls("https://aclanthology.org/2023.emnlp-main.10/")[0],
            "https://aclanthology.org/2023.emnlp-main.10.pdf",
        )
        self.assertEqual(
            candidate_pdf_urls("https://www.biorxiv.org/content/10.1101/2025.01.02.123456v1")[0],
            "https://www.biorxiv.org/content/10.1101/2025.01.02.123456v1.full.pdf",
        )
        self.assertEqual(
            candidate_pdf_urls("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7654321/")[0],
            "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7654321/fullTextPdf",
        )

    def test_openalex_collects_repository_pdf_urls_ahead_of_html_landing_url(self):
        work = {
            "open_access": {"oa_url": "https://doi.org/10.1038/s41586-026-0001"},
            "best_oa_location": {
                "pdf_url": None,
                "landing_page_url": "https://doi.org/10.1038/s41586-026-0001",
                "source": {"type": "journal"},
            },
            "locations": [
                {
                    "pdf_url": "https://arxiv.org/pdf/2501.99999",
                    "landing_page_url": "https://arxiv.org/abs/2501.99999",
                    "source": {"type": "repository"},
                }
            ],
            "ids": {"pmcid": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC999888"},
        }
        urls = _collect_openalex_oa_urls(work)
        self.assertEqual(urls[0], "https://arxiv.org/pdf/2501.99999.pdf")
        self.assertIn("https://www.ebi.ac.uk/europepmc/webservices/rest/PMC999888/fullTextPdf", urls)
        self.assertEqual(urls[-1], "https://doi.org/10.1038/s41586-026-0001")

    def test_verify_candidate_match_handles_hyphen_concatenation_glued_author_and_consortium(self):
        self.assertTrue(
            verify_candidate_match(
                "Stealthyhyperuniform wave dynamics in two-dimensional photonic crystals",
                "Stealthy hyperuniform wave dynamics in two-dimensional photonic crystals",
                "M. Barsukova",
                "Barsukova",
                2026,
                2026,
            )
        )
        self.assertTrue(
            verify_candidate_match(
                "Ferreira, Decoding disorder signatures of AuCl3 and vacancies in graphene",
                "Decoding disorder signatures of AuCl3 and vacancies in graphene",
                "F. R. Duarte",
                "Duarte",
                2024,
                2024,
            )
        )
        self.assertTrue(
            verify_candidate_match(
                "Hierarchical reasoning model. 2025",
                "Hierarchical reasoning model",
                "Wang",
                "Wang",
                2025,
                2025,
            )
        )
        self.assertTrue(
            verify_candidate_match(
                "Gemma 3 Technical Report",
                "Gemma 3 Technical Report",
                "Gemma Team",
                "Aishwarya Kamath",
                2025,
                2025,
                candidate_authors=["Aishwarya Kamath", "Gemma Team"],
            )
        )
        # Still rejects mismatched regular authors
        self.assertFalse(
            verify_candidate_match(
                "Gemma 3 Technical Report",
                "Gemma 3 Technical Report",
                "Smith, J.",
                "Kamath, A.",
                2025,
                2025,
                candidate_authors=["Aishwarya Kamath"],
            )
        )


class TestFetcherAndAuditLinking(unittest.TestCase):
    def test_resolve_doi_skips_noisy_first_crossref_item_and_matches_second(self):
        payload = {
            "message": {
                "items": [
                    {
                        "DOI": "10.1000/book-review",
                        "title": ["Book Review: Unrelated Commentary"],
                        "author": [{"family": "Reviewer"}],
                        "issued": {"date-parts": [[2020]]},
                    },
                    {
                        "DOI": "10.1103/PhysRevB.97.054105",
                        "title": ["Effect of imperfections on the hyperuniformity of many-particle systems"],
                        "author": [{"family": "Kim"}],
                        "issued": {"date-parts": [[2018]]},
                    },
                ]
            }
        }

        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return payload

        ref = {
            "title": "Effect of imperfections on the hyperuniformity of many-particle systems",
            "authors": ["J. Kim", "S. Torquato"],
            "year": 2018,
        }
        with patch.object(agent2_fetcher.requests, "get", return_value=_Resp()):
            doi, how = agent2_fetcher.resolve_doi(ref)
        self.assertEqual((doi, how), ("10.1103/physrevb.97.054105", "crossref"))

    def test_fetch_papers_upgrades_legacy_abstract_only_record_when_oa_pdf_exists(self):
        ref = {
            "raw_reference": "Rule A. Ten simple rules for Jupyter notebooks. PLoS Comput Biol 2019.",
            "source_file": "new_seed.pdf",
            "title": "Ten simple rules for writing and sharing computational analyses in Jupyter Notebooks",
            "authors": ["A. Rule"],
            "year": 2019,
            "xml_id": "b18",
        }
        key = source_key(ref)
        with tempfile.TemporaryDirectory() as tmp:
            dl_path = os.path.join(tmp, "downloaded.json")
            fail_path = os.path.join(tmp, "failed.json")
            pdfs_dir = os.path.join(tmp, "pulled_pdfs")
            idx_path = os.path.join(tmp, "paper_index.json")
            os.makedirs(pdfs_dir)

            # Pre-seed legacy abstract_only entry (without oa_exhausted)
            with open(dl_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        key: {
                            "key": key,
                            "path": None,
                            "evidence_tier": "abstract_only",
                            "abstract": "Short abstract that lacked body details for citation verification.",
                            "provider": "openalex",
                            "fetched_at": "2026-10-01T00:00:00Z",
                            "title": ref["title"],
                        }
                    },
                    fh,
                )

            def fake_dl(url, dest):
                if "fullTextPdf" in url or url.endswith(".pdf"):
                    with open(dest, "wb") as out:
                        out.write(b"%PDF-1.4 real full text")
                    return True, None
                return False, "not a PDF"

            fake_biblio = {
                "title": ref["title"],
                "abstract": "Short abstract that lacked body details for citation verification.",
                "doi": "10.1371/journal.pcbi.1007007",
                "oa_pdf_url": "https://doi.org/10.1371/journal.pcbi.1007007",
                "oa_pdf_urls": [
                    "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6657844/fullTextPdf",
                    "https://doi.org/10.1371/journal.pcbi.1007007",
                ],
            }

            status_path = os.path.join(tmp, "pipeline_status.json")
            from research_assistant.shared import pipeline_status
            pipeline_status.set_status_path(status_path)
            try:
                with patch.object(agent2_fetcher, "DOWNLOADED_JSON_PATH", dl_path), \
                     patch.object(agent2_fetcher, "FAILED_DOWNLOADS_PATH", fail_path), \
                     patch.object(agent2_fetcher, "PULLED_PDFS_DIR", pdfs_dir), \
                     patch.object(agent2_fetcher, "PAPER_INDEX_PATH", idx_path), \
                     patch.object(agent2_fetcher, "_load_references", return_value=[ref]), \
                     patch.object(agent2_fetcher, "resolve_doi", return_value=(None, None)), \
                     patch.object(agent2_fetcher, "try_europepmc", return_value=(False, "not in Europe PMC")), \
                     patch.object(agent2_fetcher, "try_arxiv", return_value=(False, "not on arXiv")), \
                     patch.object(agent2_fetcher, "fetch_bibliometrics_and_abstract", return_value=fake_biblio), \
                     patch.object(agent2_fetcher, "download_pdf", side_effect=fake_dl), \
                     patch.object(agent2_fetcher.time, "sleep", lambda s: None):
                    agent2_fetcher.fetch_papers()
            finally:
                pipeline_status.set_status_path(None)

            with open(dl_path, encoding="utf-8") as fh:
                updated = json.load(fh)
            self.assertEqual(updated[key]["evidence_tier"], "full_text")
            self.assertTrue(os.path.isfile(updated[key]["path"]))

    def test_ref_11_marker_is_numeric_and_resolves_against_bibliography(self):
        self.assertTrue(is_numeric_cite("Ref. 11"))
        self.assertTrue(is_numeric_cite("Refs. 11, 12"))
        self.assertFalse(is_numeric_cite("Refael et al. (2020)"))

        entry = {"xml_id": "b10", "printed_index": 11, "title": "Target Paper", "authors": ["A. Author"], "year": 2021}
        resolved, how, _ = _resolve_ref("", "Ref. 11", {"b10": entry}, {11: entry}, {})
        self.assertEqual(resolved, entry)
        self.assertIn(how, ("printed_label", "position"))

    def test_match_downloaded_paper_via_source_key_arxiv_and_paper_index_alias(self):
        ref = {"xml_id": "b7", "arxiv_id": "2406.10833v2", "title": None, "authors": ["K. Lee"], "year": 2024}
        manifest = {
            "arxiv:2406.10833": {
                "key": "arxiv:2406.10833",
                "path": "/tmp/arxiv_2406.10833.pdf",
                "arxiv_id": "2406.10833",
                "cited_by": "older_seed.pdf",
                "xml_id": "b1",
            }
        }
        matched = _match_downloaded_paper(ref, "new_seed.pdf", manifest)
        self.assertIsNotNone(matched)
        self.assertEqual(matched["key"], "arxiv:2406.10833")

        # Verify sibling paper_index.json enrichment and alias resolution
        with tempfile.TemporaryDirectory() as tmp:
            keeper_pdf = os.path.join(tmp, "keeper_doc.pdf")
            with open(keeper_pdf, "wb") as f:
                f.write(b"%PDF-1.4")
            idx = PaperIndex(os.path.join(tmp, "paper_index.json"))
            idx.add(
                PaperEntry(
                    document="keeper_doc.pdf",
                    path=keeper_pdf,
                    title="Canonical Corpus Paper",
                    dois=("10.1038/canon1",),
                )
            )
            idx.add_alias("title:36de9d75ff887cac", "keeper_doc.pdf")
            idx.save()

            dl_path = os.path.join(tmp, "downloaded.json")
            with open(dl_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "title:36de9d75ff887cac": {
                            "key": "title:36de9d75ff887cac",
                            "path": None,
                            "evidence_tier": "abstract_only",
                            "abstract": "Some abstract...",
                            "title": "Canonical Corpus Paper",
                        }
                    },
                    f,
                )
            enriched = _load_downloaded_manifest(dl_path, tmp)
            self.assertEqual(enriched["title:36de9d75ff887cac"]["canonical_document"], "keeper_doc.pdf")
            self.assertEqual(enriched["title:36de9d75ff887cac"]["evidence_tier"], "full_text")

    def test_orchestrate_runs_audit_even_when_ask_is_false(self):
        self.assertEqual(
            orchestrate._after_ingest_refs({"ask": False, "audit_citations": True}),
            "respond",
        )
        self.assertEqual(
            orchestrate._after_ingest_refs({"ask": False, "audit_citations": False}),
            orchestrate.END,
        )
        with patch("research_assistant.shared.seed_audit.audit_seed_citations", return_value={"totals": {"total": 3}, "results": []}) as mock_audit:
            out = orchestrate.respond({"query": "q", "ask": False, "audit_citations": True, "seed_path": "seed.pdf"})
        self.assertNotIn("answer", out)
        self.assertEqual(out["citation_audit"]["totals"]["total"], 3)
        mock_audit.assert_called_once()


class TestJobCancellationAndSourceScoping(unittest.TestCase):
    def test_cancelled_job_does_not_poison_next_paper_or_unwinding_thread(self):
        import threading
        import time
        from research_assistant import config
        from research_assistant.shared import pipeline_status, run_jobs
        import research_assistant.shared.seed_audit as sa

        with tempfile.TemporaryDirectory() as tmp:
            global_status = os.path.join(tmp, "pipeline_status.json")
            audit_dir = os.path.join(tmp, "audits")
            os.makedirs(audit_dir, exist_ok=True)
            pipeline_status.set_status_path(None)
            run_jobs._reset_registry_for_tests()
            try:
                with patch.object(config, "DATA_DIR", tmp), \
                     patch.object(config, "PIPELINE_STATUS_PATH", global_status, create=True), \
                     patch.object(sa, "AUDIT_DIR", audit_dir):
                    job_a_started = threading.Event()
                    job_a_release = threading.Event()

                    def runner_a():
                        pipeline_status.set_status(active=True, stage="fetch", detail="fetching job A")
                        job_a_started.set()
                        job_a_release.wait(5)
                        # Simulate old thread logging stop event as it unwinds after job B starts
                        pipeline_status.add_event("⏹️ Fetch stopped by user (0 downloaded)")
                        return {"stopped": "cancelled"}

                    run_jobs.start_job("job_a", "paper_a.pdf", runner_a, origin="audit")
                    self.assertTrue(job_a_started.wait(3))
                    # User clicks Stop in sidebar on job_a
                    run_jobs.cancel_job("job_a", reason="User stopped pipeline via sidebar")
                    self.assertTrue(pipeline_status.is_cancel_requested("job_a"))

                    # User immediately submits a new paper (job_b) while job_a thread is still unwinding
                    job_b_seen = {}

                    def runner_b():
                        pipeline_status.set_status(active=True, stage="discover", detail="starting job B")
                        job_a_release.set()
                        time.sleep(0.05)
                        job_b_seen["cancelled"] = pipeline_status.is_cancel_requested()
                        return {"seed_path": "paper_b.pdf"}

                    job_b = run_jobs.start_job("job_b", "paper_b.pdf", runner_b, origin="audit", clear_cancel=True)
                    job_b.thread.join(timeout=5)
                    self.assertEqual(job_b.state, "done")
                    self.assertIs(job_b_seen.get("cancelled"), False)
            finally:
                run_jobs._reset_registry_for_tests()
                pipeline_status.set_status_path(None)

    def test_busy_refusal_preserves_running_job_cancel_flag(self):
        import threading
        from research_assistant import config
        from research_assistant.shared import pipeline_status, run_jobs
        import research_assistant.shared.seed_audit as sa

        with tempfile.TemporaryDirectory() as tmp:
            global_status = os.path.join(tmp, "pipeline_status.json")
            audit_dir = os.path.join(tmp, "audits")
            os.makedirs(audit_dir, exist_ok=True)
            pipeline_status.set_status_path(None)
            run_jobs._reset_registry_for_tests()
            gate = threading.Event()
            try:
                with patch.object(config, "DATA_DIR", tmp), \
                     patch.object(config, "PIPELINE_STATUS_PATH", global_status, create=True), \
                     patch.object(sa, "AUDIT_DIR", audit_dir):
                    job_a = run_jobs.start_job(
                        "job_live", "live.pdf", lambda: (gate.wait(5), {})[1], origin="audit"
                    )
                    pipeline_status.request_cancel("stop live job", job_id="job_live")
                    with self.assertRaises(run_jobs.JobBusy):
                        run_jobs.start_job("job_other", "other.pdf", lambda: {}, origin="audit", clear_cancel=True)
                    # Cancel on job_live must still be set!
                    self.assertTrue(pipeline_status.is_cancel_requested("job_live"))
                    gate.set()
                    job_a.thread.join(timeout=5)
            finally:
                gate.set()
                run_jobs._reset_registry_for_tests()
                pipeline_status.set_status_path(None)

    def test_fetch_papers_source_file_scopes_to_current_seed_paper(self):
        from research_assistant.shared import pipeline_status

        old_ref = {
            "raw_reference": "Old Paper. 2020.",
            "source_file": "old_seed.pdf",
            "title": "Historical Unrelated Reference",
            "authors": ["O. Author"],
            "year": 2020,
            "xml_id": "b1",
        }
        new_ref = {
            "raw_reference": "New Paper. arXiv:2402.00001 (2024).",
            "source_file": "new_seed.pdf",
            "title": "Target New Seed Reference",
            "authors": ["N. Author"],
            "year": 2024,
            "arxiv_id": "2402.00001",
            "xml_id": "b2",
        }
        attempted_titles = []
        with tempfile.TemporaryDirectory() as tmp:
            dl_path = os.path.join(tmp, "downloaded.json")
            fail_path = os.path.join(tmp, "failed.json")
            pdfs_dir = os.path.join(tmp, "pulled_pdfs")
            idx_path = os.path.join(tmp, "paper_index.json")
            status_path = os.path.join(tmp, "status.json")
            os.makedirs(pdfs_dir)
            pipeline_status.set_status_path(status_path)

            def fake_arxiv(ref, dest):
                attempted_titles.append(ref.get("title"))
                with open(dest, "wb") as out:
                    out.write(b"%PDF-1.4 arxiv pdf")
                return True, None

            try:
                with patch.object(agent2_fetcher, "DOWNLOADED_JSON_PATH", dl_path), \
                     patch.object(agent2_fetcher, "FAILED_DOWNLOADS_PATH", fail_path), \
                     patch.object(agent2_fetcher, "PULLED_PDFS_DIR", pdfs_dir), \
                     patch.object(agent2_fetcher, "PAPER_INDEX_PATH", idx_path), \
                     patch.object(agent2_fetcher, "_load_references", return_value=[old_ref, new_ref]), \
                     patch.object(agent2_fetcher, "resolve_doi", return_value=(None, None)), \
                     patch.object(agent2_fetcher, "try_arxiv", side_effect=fake_arxiv), \
                     patch.object(agent2_fetcher, "fetch_bibliometrics_and_abstract", return_value={}), \
                     patch.object(agent2_fetcher.time, "sleep", lambda s: None):
                    agent2_fetcher.fetch_papers(source_file="new_seed.pdf")
            finally:
                pipeline_status.set_status_path(None)

            self.assertEqual(attempted_titles, ["Target New Seed Reference"])


class TestMultiBibliographyAndGrobidEdgeCases(unittest.TestCase):
    def test_empty_analytic_title_recovers_from_note_and_initials_recover_surname(self):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <text>
            <body>
              <div><p><s>Quantum chemistry uses ML <ref type="bibr" target="#b0">[1]</ref> and Wannier functions <ref type="bibr" target="#b1">[2]</ref>.</s></p></div>
            </body>
            <back>
              <listBibl>
                <biblStruct xml:id="b0">
                  <analytic>
                    <title level="a" type="main">Quantum chemistry in the age of machine learning</title>
                    <author><persName><forename type="first">P</forename><forename type="middle">O</forename></persName></author>
                  </analytic>
                  <monogr><title level="j">J. Phys. Chem. Lett</title><imprint><date when="2020">2020</date></imprint></monogr>
                  <note type="raw_reference">P. O. Dral, Quantum chemistry in the age of machine learning, J. Phys. Chem. Lett. 11, 2336 (2020).</note>
                </biblStruct>
                <biblStruct xml:id="b1">
                  <analytic>
                    <title/>
                    <author><persName><forename type="first">G</forename><surname>Pizzi</surname></persName></author>
                  </analytic>
                  <monogr><title level="j">J. Phys.: Condens. Matter</title><imprint><date when="2020">2020</date></imprint></monogr>
                  <note>Wannier90 as a community code: new features and applications</note>
                  <note type="raw_reference">G. Pizzi et al., Wannier90 as a community code: new features and applications, J. Phys.: Condens. Matter 32, 165902 (2020).</note>
                </biblStruct>
              </listBibl>
            </back>
          </text>
        </TEI>
        """
        soup = BeautifulSoup(xml, "xml")
        bibls = soup.find_all("biblStruct")
        r0 = agent1_extractor.parse_reference(bibls[0], source_file="seed.pdf")
        r1 = agent1_extractor.parse_reference(bibls[1], source_file="seed.pdf")
        self.assertEqual(r0["authors"], ["P. O. Dral"])
        self.assertEqual(r1["title"], "Wannier90 as a community code: new features and applications")

        claims = extract_seed_citation_claims(BeautifulSoup(xml, "xml"))
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0]["ref"]["authors"], ["P. O. Dral"])
        self.assertEqual(claims[1]["ref"]["title"], "Wannier90 as a community code: new features and applications")

    def test_multi_bibliography_pdf_alignment_separates_main_and_supplement_and_endnotes(self):
        import pymupdf

        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = os.path.join(tmp, "dual_bib.pdf")
            doc = pymupdf.open()
            p1 = doc.new_page()
            p1.insert_text((50, 80), "Main body discusses graphene transport [1], see SM [2], and PAOFLOW [3].")
            p2 = doc.new_page()
            p2.insert_text(
                (50, 80),
                "References\n"
                "[1] A. H. Castro Neto et al., The electronic properties of graphene, Rev. Mod. Phys. 81, 109 (2009).\n"
                "[2] See Supplemental Material at [URL will be inserted by publisher] for derivations.\n"
                "[3] M. B. Nardelli et al., PAOFLOW: A utility to construct Hamiltonians, Comput. Mater. Sci. 143, 462 (2018).\n"
                "[4] J. French, P. Mello, and A. Pandey, Ergodic behavior in nuclear reactions, Phys. Lett. B 80, 17 (1978).\n",
            )
            p3 = doc.new_page()
            p3.insert_text((50, 80), "Supplemental Material\nDFT calculations used the SIESTA package [1] and GGA-PBE functional [2].")
            p4 = doc.new_page()
            p4.insert_text(
                (50, 80),
                "References\n"
                "[1] J. M. Soler et al., The SIESTA method for ab initio simulations, J. Phys.: Condens. Matter 14, 2745 (2002).\n"
                "[2] J. P. Perdew, K. Burke, and M. Ernzerhof, Generalized gradient approximation made simple, Phys. Rev. Lett. 77, 3865 (1996).\n"
                "[3] J. C. Slater and G. F. Koster, Simplified LCAO method, Phys. Rev. 94, 1498 (1954).\n",
            )
            doc.save(pdf_path)
            doc.close()

            tei_xml = """<?xml version="1.0" encoding="UTF-8"?>
            <TEI xmlns="http://www.tei-c.org/ns/1.0">
              <text>
                <body>
                  <div><p><s>Main body discusses graphene transport <ref type="bibr">[1]</ref>, see SM <ref type="bibr" target="#b4">[2]</ref>, and PAOFLOW <ref type="bibr" target="#b1">[3]</ref>.</s></p></div>
                  <div><head>Supplemental Material</head><p><s>DFT calculations used the SIESTA package <ref type="bibr">[1]</ref> and GGA-PBE functional <ref type="bibr">[2]</ref>.</s></p></div>
                </body>
                <back>
                  <listBibl>
                    <biblStruct xml:id="b0">
                      <analytic><title>The electronic properties of graphene</title><author><persName><surname>Castro Neto</surname></persName></author></analytic>
                      <monogr><title level="j">Rev. Mod. Phys.</title><imprint><date when="2009">2009</date></imprint></monogr>
                    </biblStruct>
                    <biblStruct xml:id="b1">
                      <analytic><title>PAOFLOW: A utility to construct Hamiltonians</title><author><persName><surname>Nardelli</surname></persName></author></analytic>
                      <monogr><title level="j">Comput. Mater. Sci.</title><imprint><date when="2018">2018</date></imprint></monogr>
                    </biblStruct>
                    <biblStruct xml:id="b2">
                      <analytic><title>Ergodic behavior in nuclear reactions</title><author><persName><surname>French</surname></persName></author></analytic>
                      <monogr><title level="j">Phys. Lett. B</title><imprint><date when="1978">1978</date></imprint></monogr>
                    </biblStruct>
                    <biblStruct xml:id="b3">
                      <analytic><title>The SIESTA method for ab initio simulations</title><author><persName><surname>Soler</surname></persName></author></analytic>
                      <monogr><title level="j">J. Phys.: Condens. Matter</title><imprint><date when="2002">2002</date></imprint></monogr>
                    </biblStruct>
                    <biblStruct xml:id="b4">
                      <analytic><title>Generalized gradient approximation made simple</title><author><persName><surname>Perdew</surname></persName></author></analytic>
                      <monogr><title level="j">Phys. Rev. Lett.</title><imprint><date when="1996">1996</date></imprint></monogr>
                    </biblStruct>
                    <biblStruct xml:id="b5">
                      <analytic><title>Simplified LCAO method</title><author><persName><surname>Slater</surname></persName></author></analytic>
                      <monogr><title level="j">Phys. Rev.</title><imprint><date when="1954">1954</date></imprint></monogr>
                    </biblStruct>
                  </listBibl>
                </back>
              </text>
            </TEI>
            """
            claims = extract_seed_citation_claims(BeautifulSoup(tei_xml, "xml"), seed_pdf_path=pdf_path)
            self.assertTrue(all(c["resolved"] for c in claims))
            by_cite_and_claim = {(c["cite_text"], c["claim"][:20]): c for c in claims}

            # Main [1] -> Castro Neto (b0)
            self.assertEqual(by_cite_and_claim[("[1]", "Main body discusses ")]["ref"]["xml_id"], "b0")
            # Main [2] -> Printed endnote ("See Supplemental Material..."), NOT b4!
            self.assertTrue(by_cite_and_claim[("[2]", "Main body discusses ")]["ref"]["is_endnote"])
            self.assertEqual(by_cite_and_claim[("[2]", "Main body discusses ")]["role"], "pointer")
            # Main [3] -> PAOFLOW (b1), NOT b2 (list position 3)!
            self.assertEqual(by_cite_and_claim[("[3]", "Main body discusses ")]["ref"]["xml_id"], "b1")
            # Supplement [1] -> SIESTA (b3), NOT Castro Neto (b0)!
            self.assertEqual(by_cite_and_claim[("[1]", "DFT calculations use")]["ref"]["xml_id"], "b3")
            # Supplement [2] -> Perdew GGA-PBE (b4)!
            self.assertEqual(by_cite_and_claim[("[2]", "DFT calculations use")]["ref"]["xml_id"], "b4")

    def test_try_arxiv_accepts_preprint_posted_two_years_before_journal_publication(self):
        atom_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/abs/0709.1163v2</id>
            <published>2007-09-07T00:00:00Z</published>
            <title>The electronic properties of graphene</title>
            <author><name>A. H. Castro Neto</name></author>
            <link title="pdf" href="http://arxiv.org/pdf/0709.1163v2" rel="related" type="application/pdf"/>
          </entry>
        </feed>
        """

        class _Resp:
            content = atom_xml

            def raise_for_status(self):
                return None

        ref = {
            "title": "The electronic properties of graphene",
            "authors": ["A H Castro Neto", "F Guinea"],
            "year": 2009,
        }
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "out.pdf")
            with patch.object(agent2_fetcher.requests, "get", return_value=_Resp()), \
                 patch.object(agent2_fetcher, "download_pdf", return_value=(True, None)) as mock_dl:
                ok, reason = agent2_fetcher.try_arxiv(ref, dest)
            self.assertTrue(ok)
            self.assertIsNone(reason)
            mock_dl.assert_called_once()


if __name__ == "__main__":
    unittest.main()
