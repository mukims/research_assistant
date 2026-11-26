"""Regression tests for the 8 high-severity pre-demo bugs across Tools 01, 02, 05, 06, and 07."""

import json
import os
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import fitz

from research_assistant.agents import agent2_fetcher, agent5_batch_citer, agent8_verifier
from research_assistant.judgement.source_assessor import SourceGrade, assess_source
from research_assistant.schemas import DownloadedPaper
from research_assistant.shared import batch_uploader, docling_extract, extract as ex
from research_assistant.shared.draft_input import sidecar_path
from research_assistant.shared.ingest_v2 import process_pdf_v2
from research_assistant.shared.paper_index import PaperEntry, PaperIndex
from research_assistant.tasks.models import AgentRole, Task, TaskCategory, TaskGraph, TaskStatus
from research_assistant.tasks.runner import ExecutionMode, TaskRunner
from research_assistant.transfer import prior_art
from research_assistant.ui.corpus_tab import build_corpus_explorer_records


class TestTool05OcrRoutingAndModes(unittest.TestCase):
    """Tool 05 Bugs 1 & 2: enable_ocr=True on digital PDFs + VLM OCR wiring."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pdf_path = os.path.join(self.tmp.name, "paper.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Clean digital PDF text layer. " * 20)
        doc.save(self.pdf_path)
        doc.close()

    def _fake_docling_doc(self, text="Digital body paragraph with detailed results."):
        class FakeItem:
            def __init__(self, label, txt, page=1):
                self.label = label
                self.text = txt
                self.prov = [types.SimpleNamespace(page_no=page)]

        class FakeDoc:
            name = "paper.pdf"
            pictures = []
            tables = []
            texts = [text] if text else []

            def iterate_items(self):
                if not text:
                    return []
                return [
                    (FakeItem("section_header", "1. Introduction"), 1),
                    (FakeItem("text", text), 1),
                ]

            def model_dump(self, mode="json"):
                return {}

        return FakeDoc()

    def test_enable_ocr_true_uses_digital_converter_on_digital_pdf(self):
        digital_conv = MagicMock()
        digital_conv.convert.return_value = types.SimpleNamespace(document=self._fake_docling_doc())

        with patch.object(docling_extract, "needs_ocr", return_value=False) as mock_needs, \
             patch.object(docling_extract, "get_digital_converter", return_value=digital_conv) as mock_dig, \
             patch.object(docling_extract, "get_ocr_converter") as mock_ocr:
            doc = docling_extract.extract_docling(
                self.pdf_path,
                enable_ocr=True,
                use_cache=False,
            )

        mock_needs.assert_called_once_with(self.pdf_path)
        mock_dig.assert_called_once()
        mock_ocr.assert_not_called()
        self.assertEqual(doc.extraction, "docling")

    def test_enable_ocr_false_never_runs_ocr_even_when_scanned(self):
        digital_conv = MagicMock()
        digital_conv.convert.return_value = types.SimpleNamespace(document=self._fake_docling_doc(text=""))

        with patch.object(docling_extract, "needs_ocr", return_value=True), \
             patch.object(docling_extract, "get_digital_converter", return_value=digital_conv) as mock_dig, \
             patch.object(docling_extract, "get_ocr_converter") as mock_ocr:
            doc = docling_extract.extract_docling(
                self.pdf_path,
                enable_ocr=False,
                use_cache=False,
            )

        mock_dig.assert_called_once()
        mock_ocr.assert_not_called()
        self.assertEqual(doc.extraction, "docling")

    def test_vlm_ocr_engine_wired_through_process_pdf_v2_for_scanned_pdfs(self):
        vlm_doc = ex.Document(
            key="paper.pdf",
            title="Handwritten Notes",
            abstract=[],
            sections=[
                ex.Section(
                    "",
                    "other",
                    [ex.Paragraph("Transcribed VLM formula and notes content exceeding minimum chunk length requirements for indexing." * 3, 0)],
                )
            ],
            figures=[],
            extraction="ocr",
        )

        with patch.object(docling_extract, "needs_ocr", return_value=True), \
             patch.object(docling_extract, "get_ocr_converter") as mock_easyocr, \
             patch("research_assistant.shared.ocr.extract_ocr", return_value=vlm_doc) as mock_vlm:
            entries = process_pdf_v2(
                self.pdf_path,
                "Handwritten Notes",
                enable_ocr=True,
                ocr_engine="vlm",
            )

        mock_easyocr.assert_not_called()
        mock_vlm.assert_called_once_with(self.pdf_path, key="paper.pdf", engine="vlm")
        text_chunks = [e for e in entries if e["type"] == "text_chunk"]
        self.assertEqual(len(text_chunks), 1)
        self.assertEqual(text_chunks[0]["extraction"], "ocr")

    def test_ingestion_buckets_docling_ocr_and_docling_cached_modes(self):
        import research_assistant.shared.ingestion as ing
        import research_assistant.shared.manifest as manifest_mod

        p1 = os.path.join(self.tmp.name, "cached.pdf")
        p2 = os.path.join(self.tmp.name, "scanned.pdf")
        for p in (p1, p2):
            with open(p, "wb") as fh:
                fh.write(b"%PDF-1.4 fake")

        def fake_process(path, label, *a, **kw):
            mode = "docling_cached" if path == p1 else "docling_ocr"
            return [{"content": "ok", "document": ing.pdf_key(path), "extraction": mode}]

        with patch.object(manifest_mod, "MANIFEST_PATH", os.path.join(self.tmp.name, "ingested.json")), \
             patch.object(ing, "INGEST_LOCK_PATH", os.path.join(self.tmp.name, "ingest.lock")), \
             patch.object(ing, "is_grobid_alive", return_value=False), \
             patch.object(ing, "_drop_title_duplicates", side_effect=lambda cands, corp, **kw: (cands, corp, [], {})), \
             patch.object(ing, "_add_ingested_to_paper_index"), \
             patch.object(ing, "run_sweep"), \
             patch.object(ing, "process_pdf", fake_process), \
             patch.object(ing, "upsert_corpus", return_value=2), \
             patch.object(ing, "rebuild_bm25"):
            res = ing.ingest_pdfs([p1, p2], workers=1, rebuild_index=False)

        self.assertEqual(res["extraction"]["docling"], 1)
        self.assertEqual(res["extraction"]["ocr"], 1)


class TestTool05EuropePmcTupleUnpack(unittest.TestCase):
    """Tool 05 Bug 3: (False, reason) tuple from try_unpaywall must not block try_europepmc."""

    def test_unpaywall_false_tuple_falls_through_to_europepmc(self):
        with tempfile.TemporaryDirectory() as tmp:
            index_path = os.path.join(tmp, "paper_index.json")
            PaperIndex(index_path).save()
            dest_dir = os.path.join(tmp, "staged")
            os.makedirs(dest_dir, exist_ok=True)

            pdf_doc = fitz.open()
            pg = pdf_doc.new_page()
            pg.insert_text((50, 50), "Europe PMC Open Access Biomedical Article")
            valid_pdf_bytes = pdf_doc.tobytes()
            pdf_doc.close()

            def fake_europepmc(doi, ref, dest_path):
                with open(dest_path, "wb") as fh:
                    fh.write(valid_pdf_bytes)
                return True, None

            with patch.object(batch_uploader, "PAPER_INDEX_PATH", index_path), \
                 patch.object(agent2_fetcher, "try_unpaywall", return_value=(False, "paywalled / not open access")) as mock_upw, \
                 patch.object(agent2_fetcher, "try_europepmc", side_effect=fake_europepmc) as mock_epmc:
                staged = batch_uploader.fetch_and_stage_urls_or_dois(
                    ["10.1093/nar/gkae001"],
                    destination_dir=dest_dir,
                )

            mock_upw.assert_called_once()
            mock_epmc.assert_called_once()
            self.assertEqual(len(staged), 1)
            leftover_temps = [f for f in os.listdir(dest_dir) if f.startswith("temp_doi_fetch_")]
            self.assertEqual(leftover_temps, [])


class TestTool05CorpusExplorerMetadataJoin(unittest.TestCase):
    """Tool 05 Bug 4: Corpus Explorer resolves titles/DOIs/arXiv IDs via PaperIndex & manifests."""

    def test_explorer_joins_paper_index_downloaded_and_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            pi = PaperIndex(os.path.join(tmp, "paper_index.json"))
            pi.add(
                PaperEntry(
                    document="doi_10.1038_nature13831.pdf",
                    path="/data/pulled_pdfs/doi_10.1038_nature13831.pdf",
                    title="Experimental realization of the topological Haldane model",
                    first_authors=("Jotzu",),
                    dois=("10.1038/nature13831",),
                    arxiv_ids=("1406.7874",),
                )
            )
            pi.add_alias("old_duplicate_name.pdf", "doi_10.1038_nature13831.pdf")

            downloaded = {
                "doi:10.1038/nature13831": {
                    "key": "doi:10.1038/nature13831",
                    "path": "/data/pulled_pdfs/doi_10.1038_nature13831.pdf",
                    "title": "Experimental realization of the topological Haldane model",
                    "venue": "Nature",
                    "doi": "10.1038/nature13831",
                },
                "doi:10.1103/physrevb.99.01": {
                    "key": "doi:10.1103/physrevb.99.01",
                    "path": None,
                    "evidence_tier": "abstract_only",
                    "title": "Abstract-only Condensed Matter Paper",
                    "venue": "Physical Review B",
                    "doi": "10.1103/physrevb.99.01",
                },
            }
            seeds = {
                "quantum error correction": {
                    "key": "arxiv:2401.99999",
                    "path": "/data/raw/arxiv_2401.99999.pdf",
                    "title": "Fault-Tolerant Quantum Error Correction",
                    "arxiv_id": "2401.99999",
                    "url": "https://arxiv.org/pdf/2401.99999.pdf",
                }
            }
            ingested_map = {
                "doi_10.1038_nature13831.pdf": "2026-11-25T10:00:00",
                "old_duplicate_name.pdf": "2026-11-25T10:01:00",
                "arxiv_2401.99999.pdf": "2026-11-25T10:02:00",
                "doi:10.1103/physrevb.99.01": "2026-11-25T10:03:00",
            }

            records = build_corpus_explorer_records(
                ingested_map,
                downloaded=downloaded,
                seeds=seeds,
                paper_index=pi,
            )
            by_key = {r["key"]: r for r in records}

            self.assertEqual(by_key["doi_10.1038_nature13831.pdf"]["title"], "Experimental realization of the topological Haldane model")
            self.assertEqual(by_key["doi_10.1038_nature13831.pdf"]["doi"], "10.1038/nature13831")
            self.assertEqual(by_key["doi_10.1038_nature13831.pdf"]["arxiv_id"], "1406.7874")
            self.assertEqual(by_key["doi_10.1038_nature13831.pdf"]["venue"], "Nature")

            # Aliased PDF resolves through PaperIndex.resolve()
            self.assertEqual(by_key["old_duplicate_name.pdf"]["title"], "Experimental realization of the topological Haldane model")

            # Seed paper matched by path basename rather than query key
            self.assertEqual(by_key["arxiv_2401.99999.pdf"]["title"], "Fault-Tolerant Quantum Error Correction")
            self.assertEqual(by_key["arxiv_2401.99999.pdf"]["url"], "https://arxiv.org/pdf/2401.99999.pdf")

            # Filter query matches resolved title/venue
            filtered = build_corpus_explorer_records(
                ingested_map,
                downloaded=downloaded,
                seeds=seeds,
                paper_index=pi,
                filter_query="haldane",
            )
            self.assertEqual(len(filtered), 2)


class TestTool01VenuePersistenceAndGrading(unittest.TestCase):
    """Tool 01 Bug 5: DownloadedPaper venue persistence & source_assessor grading fixes."""

    def test_downloaded_paper_persists_venue_from_fetcher(self):
        ref = {
            "raw_reference": "Vaswani et al. Attention Is All You Need. NeurIPS 2017.",
            "source_file": "seed.pdf",
            "title": "Attention Is All You Need",
            "container": "Advances in Neural Information Processing Systems",
            "doi": "10.5555/3295222.3295349",
            "doi_confidence": "high",
            "xml_id": "b1",
        }

        def fake_unpaywall(doi, dest_path):
            with open(dest_path, "wb") as fh:
                fh.write(b"%PDF-1.4 fake")
            return True, None

        with tempfile.TemporaryDirectory() as tmp:
            dl_path = os.path.join(tmp, "downloaded.json")
            fail_path = os.path.join(tmp, "failed.json")
            pdfs_dir = os.path.join(tmp, "pdfs")
            cites_path = os.path.join(tmp, "cites.json")

            with patch.object(agent2_fetcher, "DOWNLOADED_JSON_PATH", dl_path), \
                 patch.object(agent2_fetcher, "FAILED_DOWNLOADS_PATH", fail_path), \
                 patch.object(agent2_fetcher, "PULLED_PDFS_DIR", pdfs_dir), \
                 patch.object(agent2_fetcher, "EXTRACTED_CITATIONS_PATH", cites_path), \
                 patch.object(agent2_fetcher, "_load_references", return_value=[ref]), \
                 patch.object(agent2_fetcher, "resolve_doi", return_value=("10.5555/3295222.3295349", "grobid")), \
                 patch.object(agent2_fetcher, "try_unpaywall", side_effect=fake_unpaywall):
                agent2_fetcher.fetch_papers()

            with open(dl_path, encoding="utf-8") as fh:
                saved = json.load(fh)

        rec = next(iter(saved.values()))
        self.assertEqual(rec["venue"], "Advances in Neural Information Processing Systems")
        parsed = DownloadedPaper.from_dict(rec)
        self.assertEqual(parsed.venue, "Advances in Neural Information Processing Systems")

    def test_well_cited_paper_without_venue_is_rigorous_primary(self):
        res = assess_source({
            "title": "Deep Residual Learning for Image Recognition",
            "doi": "10.1109/CVPR.2016.90",
            "citation_count": 180000,
        })
        self.assertEqual(res["grade"], SourceGrade.RIGOROUS_PRIMARY.value)

    def test_science_and_cell_false_positives_rejected_and_ict_clinical_venues_recognized(self):
        for std_venue in (
            "Lecture Notes in Computer Science",
            "Applied Surface Science",
            "Materials Science Forum",
            "Fuel Cell Science and Technology",
        ):
            res = assess_source({"title": "Empirical study of caching", "venue": std_venue, "citation_count": 5})
            self.assertEqual(res["grade"], SourceGrade.STANDARD_PRIMARY.value, std_venue)

        for rig_venue in (
            "Nature",
            "Nature Machine Intelligence",
            "Science",
            "Science Robotics",
            "Cell",
            "Molecular Cell",
            "NeurIPS",
            "Advances in Neural Information Processing Systems",
            "International Conference on Machine Learning",
            "ICLR",
            "CVPR",
            "ACL",
            "ACM CCS",
            "IEEE S&P",
            "New England Journal of Medicine",
            "NEJM",
            "The Lancet",
            "JAMA",
        ):
            res = assess_source({"title": "Primary experimental architecture", "venue": rig_venue, "citation_count": 10})
            self.assertEqual(res["grade"], SourceGrade.RIGOROUS_PRIMARY.value, rig_venue)


class TestTool02MdTexSidecarNonOverwrite(unittest.TestCase):
    """Tool 02 Bug 6: --out paper.md / paper.tex must never overwrite the draft with sidecars."""

    def test_sidecar_path_handles_md_tex_txt_and_extensionless(self):
        self.assertEqual(sidecar_path("/a/b.txt/paper.md", "_report.md"), "/a/b.txt/paper_report.md")
        self.assertEqual(sidecar_path("/a/paper.tex", "_citations.json"), "/a/paper_citations.json")
        self.assertEqual(sidecar_path("/a/paper.txt", "_verification.md"), "/a/paper_verification.md")
        self.assertEqual(sidecar_path("/a/paper", "_report.md"), "/a/paper_report.md")

    def test_batch_citer_and_verifier_preserve_md_and_tex_outputs(self):
        for ext in (".md", ".tex"):
            with self.subTest(ext=ext), tempfile.TemporaryDirectory() as tmp:
                in_path = os.path.join(tmp, f"input_draft{ext}")
                out_path = os.path.join(tmp, f"cited_draft{ext}")
                original_text = "Graphene exhibits high carrier mobility at room temperature.\n"
                with open(in_path, "w", encoding="utf-8") as fh:
                    fh.write(original_text)

                fake_hit = {
                    "text": "Graphene exhibits high carrier mobility at room temperature.",
                    "metadata": {
                        "document": "graphene.pdf",
                        "citation_source": "Novoselov et al. 2004",
                        "venue": "Science",
                        "type": "text_chunk",
                    },
                    "rrf_score": 0.9,
                }
                fake_verdict = {
                    "slots": {
                        "finding": {"assertion": "reported finding", "verdict": "Supports"},
                        "scope": {"assertion": "reported scope", "verdict": "Supports"},
                        "strength": {"assertion": "reported strength", "verdict": "Not applicable"},
                    },
                    "judgement": "Supports",
                    "model_judgement": "Supports",
                    "confidence": "High",
                    "evidence_sufficiency": "sufficient",
                    "supporting_span": "Graphene exhibits high carrier mobility at room temperature.",
                    "span_verified": True,
                    "span_cites_others": False,
                    "rubric_mismatch": False,
                    "rubric_violations": [],
                    "reason": "Direct match.",
                }

                with patch.object(agent5_batch_citer, "_batch_needs_citation", return_value=[True]), \
                     patch.object(agent5_batch_citer, "hybrid_search", return_value=[fake_hit]), \
                     patch.object(agent5_batch_citer, "expand_neighbours", side_effect=lambda hits, *a, **kw: hits), \
                     patch.object(agent5_batch_citer, "judge", return_value=fake_verdict):
                    written = agent5_batch_citer.run_batch_citer(
                        in_path,
                        out_path,
                        search_resources=(object(), object(), [fake_hit["text"]], [fake_hit["metadata"]]),
                    )

                self.assertEqual(written, out_path)
                with open(out_path, encoding="utf-8") as fh:
                    cited_body = fh.read()
                self.assertIn("\\cite{cite_1}", cited_body)
                self.assertNotIn("# Citation Report", cited_body)
                self.assertTrue(os.path.exists(os.path.join(tmp, "cited_draft_citations.json")))
                self.assertTrue(os.path.exists(os.path.join(tmp, "cited_draft_citation_evidence.json")))
                self.assertTrue(os.path.exists(os.path.join(tmp, "cited_draft_report.md")))

                with patch.object(agent8_verifier, "_load_downloaded_manifest", return_value={}), \
                     patch.object(agent8_verifier, "_judge_once", return_value=fake_verdict):
                    ver_rep = agent8_verifier.verify_draft(
                        out_path,
                        search_resources=(None, None, [], []),
                    )

                self.assertEqual(ver_rep["totals"]["judged"], 1)
                with open(out_path, encoding="utf-8") as fh:
                    still_cited_body = fh.read()
                self.assertEqual(still_cited_body, cited_body)
                self.assertTrue(os.path.exists(os.path.join(tmp, "cited_draft_verification.json")))
                self.assertTrue(os.path.exists(os.path.join(tmp, "cited_draft_verification.md")))


class TestTool07PriorArtResilienceAndTelemetry(unittest.TestCase):
    """Tool 07 Bug 7: Decoupled provider error handling, logging, and API key support."""

    def test_single_provider_failure_preserves_surviving_provider_hits(self):
        oa_payload = {
            "results": [
                {
                    "display_name": "Kernel polynomial method in condensed matter",
                    "publication_year": 2006,
                    "doi": "10.1103/RevModPhys.78.275",
                    "abstract_inverted_index": {"kernel": [0], "polynomial": [1], "method": [2]},
                }
            ]
        }

        def failing_arxiv(_url, _params):
            raise RuntimeError("HTTP 429 Too Many Requests from arXiv")

        client = prior_art.PriorArtClient(
            get_json=lambda _u, _p: oa_payload,
            get_text=failing_arxiv,
            embed_documents=lambda texts: [[1.0, 0.0]] * len(texts),
        )
        with self.assertLogs("transfer.prior_art", level="WARNING") as cm:
            res = client.check("kernel polynomial method", ["density of states"], "Chebyshev expansion")

        self.assertEqual(res.status, "done_before")
        self.assertEqual(len(res.hits), 1)
        self.assertIn("arxiv:", res.error or "")
        self.assertTrue(any("arXiv prior-art lookup failed" in msg for msg in cm.output))

    def test_total_outage_logs_warning_and_records_error_reason(self):
        def boom(_u, _p):
            raise ConnectionError("provider unreachable")

        client = prior_art.PriorArtClient(
            get_json=boom,
            get_text=boom,
            embed_documents=lambda texts: [[1.0, 0.0]] * len(texts),
        )
        with self.assertLogs("transfer.prior_art", level="WARNING") as cm:
            res = client.check("tensor network", ["entanglement"], "summary")

        self.assertEqual(res.status, "unchecked")
        self.assertIn("openalex:", res.error or "")
        self.assertIn("arxiv:", res.error or "")
        self.assertTrue(any("Prior-art check failed across all providers" in msg for msg in cm.output))


class TestTool06SimulationBadgeHonesty(unittest.TestCase):
    """Tool 06 Bug 8: Simulation mode never awards CERTIFIED_REPLICABLE or passes broken harnesses."""

    def test_simulation_mode_awards_dry_run_validated_not_certified_replicable(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = os.path.join(tmp, "sim_ws")
            graph = TaskGraph(paper_id="demo_p", paper_title="Demo Paper")
            graph.add_task(
                Task(
                    task_id="t1",
                    title="Setup",
                    category=TaskCategory.ENVIRONMENT_SETUP,
                    assigned_role=AgentRole.DEVOPS_AGENT,
                    outputs=["env.txt"],
                )
            )
            runner = TaskRunner(graph=graph, workspace_dir=ws, mode=ExecutionMode.SIMULATION)
            report = runner.execute()

            self.assertEqual(report.scorecard["rating"], "DRY_RUN_VALIDATED")
            self.assertNotEqual(report.scorecard["rating"], "CERTIFIED_REPLICABLE")
            md = report.render_markdown()
            self.assertIn("DRY_RUN_VALIDATED", md)
            self.assertNotIn("CERTIFIED_REPLICABLE", md)
            self.assertIn("Simulation Dry Run", md)

    def test_simulation_mode_fails_task_when_test_harness_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = os.path.join(tmp, "sim_fail_ws")
            graph = TaskGraph(paper_id="fail_p", paper_title="Failing Contract Paper")
            graph.add_task(
                Task(
                    task_id="t1",
                    title="Broken Harness Task",
                    category=TaskCategory.MODEL_IMPLEMENTATION,
                    assigned_role=AgentRole.MODEL_ARCHITECT,
                    outputs=["model.py"],
                    test_harness_script="import unittest\nclass T(unittest.TestCase):\n    def test_fail(self):\n        self.fail('contract violated')\nif __name__ == '__main__':\n    unittest.main()\n",
                )
            )
            runner = TaskRunner(graph=graph, workspace_dir=ws, mode=ExecutionMode.SIMULATION)
            report = runner.execute()

            self.assertEqual(report.results["t1"].status, TaskStatus.FAILED)
            self.assertEqual(report.failed_count, 1)
            self.assertEqual(report.completed_count, 0)
            self.assertEqual(report.scorecard["rating"], "DRY_RUN_FAILED")


if __name__ == "__main__":
    unittest.main()
