"""Adversarial stress-test harness for Milestone M1 (Core Discovery Engine).

Tests adversarial inputs and edge cases against:
1. filter_open_access (missing URLs, malformed URLs, non-HTTP schemes, synthesis of arXiv direct links)
2. evaluate_candidates (empty lists, single candidate, 50 candidates, tie-breaking, LLM failures, heuristic fallback, non-integer year vulnerability)
3. download_seed_cascade (HTTP 404, HTML error pages without %PDF-, connection timeouts, partial downloads, full failure)
4. discover atomic registration (strict SeedPaper schema adherence, TypeError on unexpected fields, trace decoupling)
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

import research_assistant.agents.agent0_discoverer as a0
from research_assistant.schemas import SeedPaper, SchemaError


class TestAdversarialFilterOpenAccess(unittest.TestCase):
    """Stress-test filter_open_access with adversarial and degenerate URLs."""

    def test_missing_urls_filtered_out(self):
        candidates = [
            {"title": "Paper No URL"},
            {"title": "Paper None URL", "pdf_url": None},
            {"title": "Paper Empty URL", "pdf_url": ""},
            {"title": "Paper Whitespace URL", "pdf_url": "   \t\n  "},
            {"title": "Paper Non-string int URL", "pdf_url": 12345},
            {"title": "Paper Non-string list URL", "pdf_url": ["https://example.com/p.pdf"]},
            {"title": "Paper Non-string dict URL", "pdf_url": {"url": "https://example.com/p.pdf"}},
            {"title": "Paper Bool URL", "pdf_url": True},
        ]
        filtered = a0.filter_open_access(candidates)
        self.assertEqual(len(filtered), 0)

    def test_malformed_urls_filtered_out(self):
        candidates = [
            {"title": "Paper 1", "pdf_url": "not_a_url"},
            {"title": "Paper 2", "pdf_url": "://bad"},
            {"title": "Paper 3", "pdf_url": "example.com/paper.pdf"},
            {"title": "Paper 4", "pdf_url": "http://"},
            {"title": "Paper 5", "pdf_url": "https://"},
            {"title": "Paper 6", "pdf_url": "htp://example.com/p.pdf"},
            {"title": "Paper 7", "pdf_url": "//example.com/p.pdf"},
        ]
        filtered = a0.filter_open_access(candidates)
        urls = [c["pdf_url"] for c in filtered]
        self.assertNotIn("not_a_url", urls)
        self.assertNotIn("://bad", urls)
        self.assertNotIn("example.com/paper.pdf", urls)
        self.assertNotIn("htp://example.com/p.pdf", urls)
        self.assertNotIn("//example.com/p.pdf", urls)

    def test_unsupported_and_dangerous_schemes_filtered_out(self):
        candidates = [
            {"title": "FTP paper", "pdf_url": "ftp://files.example.com/paper.pdf"},
            {"title": "SFTP paper", "pdf_url": "sftp://files.example.com/paper.pdf"},
            {"title": "Javascript XSS paper", "pdf_url": "javascript:alert(document.cookie)"},
            {"title": "Data URI paper", "pdf_url": "data:application/pdf;base64,JVBERi0xLjQK"},
            {"title": "File URI paper", "pdf_url": "file:///etc/passwd"},
            {"title": "Gopher paper", "pdf_url": "gopher://gopher.floodgap.com"},
            {"title": "Mailto paper", "pdf_url": "mailto:author@example.com"},
        ]
        filtered = a0.filter_open_access(candidates)
        self.assertEqual(len(filtered), 0)

    def test_synthesis_of_arxiv_direct_links(self):
        candidates = [
            # Missing pdf_url but valid arxiv_id
            {
                "title": "Quantum Error Correction",
                "arxiv_id": "2301.12345",
                "pdf_url": None,
            },
            # /abs/ abstract link transformed to /pdf/
            {
                "title": "Topological Phases",
                "arxiv_id": "2302.54321",
                "pdf_url": "https://arxiv.org/abs/2302.54321",
            },
            # /abs/ link with version suffix
            {
                "title": "Majorana Zero Modes",
                "arxiv_id": "2303.99999v2",
                "pdf_url": "http://arxiv.org/abs/2303.99999v2",
            },
            # Non-arXiv publisher link preserved even if arxiv_id exists
            {
                "title": "Non-Abelian Anyons",
                "arxiv_id": "2304.11111",
                "pdf_url": "https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.100.000000",
            },
        ]
        filtered = a0.filter_open_access(candidates)
        self.assertEqual(len(filtered), 4)

        # Candidate 0 synthesized
        self.assertEqual(filtered[0]["pdf_url"], "https://arxiv.org/pdf/2301.12345.pdf")
        # Candidate 1 transformed from /abs/ to /pdf/
        self.assertEqual(filtered[1]["pdf_url"], "https://arxiv.org/pdf/2302.54321.pdf")
        # Candidate 2 transformed from /abs/ to /pdf/
        self.assertEqual(filtered[2]["pdf_url"], "https://arxiv.org/pdf/2303.99999v2.pdf")
        # Candidate 3 kept non-arxiv direct publisher link
        self.assertEqual(
            filtered[3]["pdf_url"],
            "https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.100.000000",
        )

    def test_deduplication_by_source_key_in_filtering(self):
        candidates = [
            {
                "title": "Universal Quantum Computing",
                "doi": "10.1103/PhysRevA.52.3457",
                "pdf_url": "https://arxiv.org/pdf/quant-ph/9503016.pdf",
            },
            {
                "title": "Universal Quantum Computing (Copy 2)",
                "doi": "10.1103/physreva.52.3457",  # same normalized DOI
                "pdf_url": "https://example.com/another_copy.pdf",
            },
        ]
        filtered = a0.filter_open_access(candidates)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["pdf_url"], "https://arxiv.org/pdf/quant-ph/9503016.pdf")

    def test_candidate_without_source_key_identity_filtered(self):
        candidates = [
            {"pdf_url": "https://example.com/orphan.pdf", "title": ""}
        ]
        filtered = a0.filter_open_access(candidates)
        self.assertEqual(len(filtered), 0)


class TestAdversarialCandidateEvaluation(unittest.TestCase):
    """Stress-test evaluate_candidates with empty, scale, tie-breaking, and failure modes."""

    def setUp(self):
        self.context = {
            "core_problem": "Topological quantum computation with Majorana zero modes",
            "primary_domain": "Condensed Matter Physics",
            "subtopics": ["Majorana wires", "Non-Abelian braiding", "Quantum memory"],
            "key_concepts": ["topological phase", "Majorana fermion", "Andreev bound state"],
            "synonyms": ["Majorana bound states", "topological superconductor"],
            "seed_criteria": "Foundational theoretical survey or benchmark experiment",
            "ideal_paper_type": "comprehensive_review",
        }

    def test_empty_candidate_list(self):
        self.assertEqual(a0.evaluate_candidates([], self.context), [])

    def test_single_candidate(self):
        cands = [{
            "title": "A Review of Majorana Nanowires",
            "arxiv_id": "2401.00001",
            "year": 2024,
            "pdf_url": "https://arxiv.org/pdf/2401.00001.pdf",
        }]
        ranked = a0.evaluate_candidates(cands, self.context)
        self.assertEqual(len(ranked), 1)
        self.assertIn("score", ranked[0])
        self.assertIn("reasoning", ranked[0])
        self.assertGreaterEqual(ranked[0]["score"], 0.0)

    def test_fifty_candidates_scale_and_shortlist(self):
        candidates = []
        for i in range(1, 51):
            candidates.append({
                "title": f"Candidate Paper {i:02d}: Study of Quantum Phenomena in Wire {i}",
                "doi": f"10.1000/mock.{i:04d}",
                "arxiv_id": f"2401.{i:05d}",
                "year": 2000 + (i % 25),
                "pdf_url": f"https://arxiv.org/pdf/2401.{i:05d}.pdf",
                "authors": [f"Author {i} A", f"Author {i} B"],
            })

        # By default, max_candidates_to_score is 12
        ranked_default = a0.evaluate_candidates(candidates, self.context)
        self.assertEqual(len(ranked_default), 12)

        # When requesting all 50
        ranked_all = a0.evaluate_candidates(candidates, self.context, max_candidates_to_score=50)
        self.assertEqual(len(ranked_all), 50)
        for c in ranked_all:
            self.assertIn("score", c)
            self.assertIn("reasoning", c)

    def test_tie_breaking_deterministic_order(self):
        """Verify strict deterministic tie-breaking across all 4 tie-breaker dimensions:
        1. score (higher first)
        2. doi presence (1 > 0)
        3. arxiv_id presence (1 > 0)
        4. year (more recent first)
        5. title (descending order)
        """
        # Test 1: Equal score tie-break by DOI presence
        cand_with_doi = {
            "title": "Quantum Study",
            "doi": "10.1000/cand1",
            "arxiv_id": "2001.00001",
            "year": 2020,
            "pdf_url": "https://example.com/1.pdf",
        }
        cand_without_doi = {
            "title": "Quantum Study",
            "doi": None,
            "arxiv_id": "2001.00002",
            "year": 2020,
            "pdf_url": "https://example.com/2.pdf",
        }
        with patch.object(a0, "chat", side_effect=RuntimeError("offline")):
            # Both have same year, title, and arxiv provenance bonus, yielding equal heuristic scores
            ranked = a0.evaluate_candidates([cand_without_doi, cand_with_doi], self.context)
            self.assertEqual(ranked[0]["doi"], "10.1000/cand1")

        # Test 2: Equal score & equal DOI status tie-break by arXiv ID presence
        mock_reply = MagicMock()
        mock_reply.content = json.dumps([
            {"index": 0, "score": 8.0, "reasoning": "equal"},
            {"index": 1, "score": 8.0, "reasoning": "equal"},
        ])
        cand_with_arx = {
            "title": "Same Title",
            "doi": "10.1000/same",
            "arxiv_id": "2001.00001",
            "year": 2020,
            "pdf_url": "https://example.com/1.pdf",
        }
        cand_no_arx = {
            "title": "Same Title",
            "doi": "10.1000/same",
            "arxiv_id": None,
            "year": 2020,
            "pdf_url": "https://example.com/2.pdf",
        }
        with patch.object(a0, "chat", return_value=mock_reply):
            ranked = a0.evaluate_candidates([cand_no_arx, cand_with_arx], self.context)
            self.assertEqual(ranked[0]["arxiv_id"], "2001.00001")

        # Test 3: Equal score, equal DOI, equal arXiv tie-break by Year
        cand_older = {
            "title": "Same Title",
            "doi": "10.1000/same",
            "arxiv_id": "2001.00001",
            "year": 2015,
            "pdf_url": "https://example.com/1.pdf",
        }
        cand_newer = {
            "title": "Same Title",
            "doi": "10.1000/same",
            "arxiv_id": "2001.00001",
            "year": 2024,
            "pdf_url": "https://example.com/2.pdf",
        }
        with patch.object(a0, "chat", return_value=mock_reply):
            ranked = a0.evaluate_candidates([cand_older, cand_newer], self.context)
            self.assertEqual(ranked[0]["year"], 2024)

        # Test 4: Equal score, DOI, arXiv, year tie-break by Title
        cand_apple = {
            "title": "Apple Architecture",
            "doi": "10.1000/same",
            "arxiv_id": "2001.00001",
            "year": 2024,
            "pdf_url": "https://example.com/1.pdf",
        }
        cand_zebra = {
            "title": "Zebra Architecture",
            "doi": "10.1000/same",
            "arxiv_id": "2001.00001",
            "year": 2024,
            "pdf_url": "https://example.com/2.pdf",
        }
        with patch.object(a0, "chat", return_value=mock_reply):
            ranked = a0.evaluate_candidates([cand_apple, cand_zebra], self.context)
            self.assertEqual(ranked[0]["title"], "Zebra Architecture")

    def test_llm_failure_modes_gracefully_fallback(self):
        cands = [
            {"title": "Paper 1", "year": 2022, "pdf_url": "https://arxiv.org/pdf/2201.00001.pdf"},
            {"title": "Paper 2", "year": 2023, "pdf_url": "https://arxiv.org/pdf/2301.00001.pdf"},
        ]

        failure_cases = [
            RuntimeError("Ollama connection refused"),
            TimeoutError("LLM call timed out after 30s"),
            Exception("CUDA out of memory"),
        ]

        for exc in failure_cases:
            with patch.object(a0, "chat", side_effect=exc):
                ranked = a0.evaluate_candidates(cands, self.context)
                self.assertEqual(len(ranked), 2)
                for r in ranked:
                    self.assertEqual(r["eval_method"], "heuristic_fallback")
                    self.assertIsInstance(r["score"], float)
                    self.assertTrue("Heuristic ranking" in r["reasoning"])

    def test_llm_malformed_json_fallback(self):
        cands = [
            {"title": "Paper 1", "year": 2022, "pdf_url": "https://arxiv.org/pdf/2201.00001.pdf"},
        ]
        malformed_replies = [
            "Here is the evaluation: sorry, I cannot parse candidates.",
            "{'status': 'error', 'message': 'invalid format'}",
            json.dumps({"some_key": "not a list"}),
            json.dumps(["string in list", 12345]),
            "",
        ]
        for reply in malformed_replies:
            mock_res = MagicMock()
            mock_res.content = reply
            with patch.object(a0, "chat", return_value=mock_res):
                ranked = a0.evaluate_candidates(cands, self.context)
                self.assertEqual(len(ranked), 1)
                self.assertEqual(ranked[0]["eval_method"], "heuristic_fallback")

    def test_vulnerability_non_integer_year_string_handling(self):
        """Stress test: evaluate_candidates should handle candidates with non-integer year
        strings (e.g. '2024-05', 'unknown', 'in press') without raising ValueError.

        Finding: Currently, evaluate_candidates lines 1086 and 1155 execute:
            int(c.get("year") or 0)
        which raises ValueError when year is a non-integer string like '2024-05' or 'unknown'.
        """
        cands = [
            {
                "title": "Quantum Error Correction in 2024",
                "year": "2024-05",  # typical ISO date string from external provider
                "pdf_url": "https://example.com/p1.pdf",
            },
            {
                "title": "Early Review",
                "year": "unknown",  # non-numeric indicator
                "pdf_url": "https://example.com/p2.pdf",
            },
        ]
        # Demonstrate the vulnerability directly
        with self.assertRaises(ValueError):
            a0.evaluate_candidates(cands, self.context)


class TestAdversarialDownloadSeedCascade(unittest.TestCase):
    """Stress-test download_seed_cascade with network and content failures."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw_patch = patch.object(a0, "RAW_DIR", self.tmp.name)
        self.raw_patch.start()
        self.addCleanup(self.raw_patch.stop)

    def test_cascade_past_http_404_to_working_candidate(self):
        candidates = [
            {"title": "Dead 404 Link", "arxiv_id": "2001.00001", "pdf_url": "https://example.com/404.pdf"},
            {"title": "Working Link", "arxiv_id": "2001.00002", "pdf_url": "https://example.com/valid.pdf"},
        ]

        def mock_download(url, dest):
            if "404.pdf" in url or "2001.00001" in url:
                return False, "HTTP 404"
            with open(dest, "wb") as f:
                f.write(b"%PDF-1.4 valid binary")
            return True, None

        with patch.object(a0, "download_pdf", side_effect=mock_download):
            cand, key, dest = a0.download_seed_cascade(candidates)
            self.assertIsNotNone(dest)
            self.assertEqual(cand["arxiv_id"], "2001.00002")
            self.assertTrue(os.path.exists(dest))

    def test_cascade_past_html_error_page_without_magic_bytes(self):
        candidates = [
            {"title": "Paywalled Paper", "arxiv_id": "2001.00001", "pdf_url": "https://paywall.com/paper.pdf"},
            {"title": "Open Access Paper", "arxiv_id": "2001.00002", "pdf_url": "https://open.org/paper.pdf"},
        ]

        def mock_download(url, dest):
            if "paywall.com" in url or "2001.00001" in url:
                return False, "not a PDF (content-type text/html)"
            with open(dest, "wb") as f:
                f.write(b"%PDF-1.7 valid content")
            return True, None

        with patch.object(a0, "download_pdf", side_effect=mock_download):
            cand, key, dest = a0.download_seed_cascade(candidates)
            self.assertIsNotNone(dest)
            self.assertEqual(cand["arxiv_id"], "2001.00002")
            self.assertEqual(cand["download_status"], "downloaded")

    def test_cascade_past_network_timeout(self):
        candidates = [
            {"title": "Timing Out Host", "arxiv_id": "2001.00001", "pdf_url": "https://slow.com/p1.pdf"},
            {"title": "Fast Host", "arxiv_id": "2001.00002", "pdf_url": "https://fast.com/p2.pdf"},
        ]

        def mock_download(url, dest):
            if "slow.com" in url or "2001.00001" in url:
                return False, "download error: Connection timed out"
            with open(dest, "wb") as f:
                f.write(b"%PDF-1.5 fast")
            return True, None

        with patch.object(a0, "download_pdf", side_effect=mock_download):
            cand, key, dest = a0.download_seed_cascade(candidates)
            self.assertIsNotNone(dest)
            self.assertEqual(cand["arxiv_id"], "2001.00002")

    def test_cascade_past_corrupted_partial_download(self):
        candidates = [
            {"title": "Truncated Stream", "arxiv_id": "2001.00001", "pdf_url": "https://drop.com/p1.pdf"},
            {"title": "Complete Stream", "arxiv_id": "2001.00002", "pdf_url": "https://complete.com/p2.pdf"},
        ]

        def mock_download(url, dest):
            if "drop.com" in url or "2001.00001" in url:
                return False, "incomplete download (1024 of 500000 bytes)"
            with open(dest, "wb") as f:
                f.write(b"%PDF-1.6 complete")
            return True, None

        with patch.object(a0, "download_pdf", side_effect=mock_download):
            cand, key, dest = a0.download_seed_cascade(candidates)
            self.assertIsNotNone(dest)
            self.assertEqual(cand["arxiv_id"], "2001.00002")

    def test_all_candidates_fail_returns_none_tuple(self):
        candidates = [
            {"title": "Broken 1", "arxiv_id": "2001.00001", "pdf_url": "https://broken1.com/p1.pdf"},
            {"title": "Broken 2", "arxiv_id": "2001.00002", "pdf_url": "https://broken2.com/p2.pdf"},
            {"title": "Broken 3", "arxiv_id": "2001.00003", "pdf_url": "https://broken3.com/p3.pdf"},
        ]
        with patch.object(a0, "download_pdf", return_value=(False, "HTTP 500 Server Error")):
            cand, key, dest = a0.download_seed_cascade(candidates)
            self.assertIsNone(cand)
            self.assertIsNone(key)
            self.assertIsNone(dest)

    def test_arxiv_fallback_retry_success(self):
        candidates = [
            {
                "title": "Quantum Supremacy",
                "arxiv_id": "1910.11333",
                "pdf_url": "https://publisher.nature.com/articles/s41586-019-1666-5.pdf",
            }
        ]

        attempts = []

        def mock_download(url, dest):
            attempts.append(url)
            if "nature.com" in url:
                return False, "HTTP 403 Forbidden"
            if "arxiv.org/pdf/1910.11333.pdf" in url:
                with open(dest, "wb") as f:
                    f.write(b"%PDF-1.4 arxiv copy")
                return True, None
            return False, "unexpected url"

        with patch.object(a0, "download_pdf", side_effect=mock_download):
            cand, key, dest = a0.download_seed_cascade(candidates)
            self.assertIsNotNone(dest)
            self.assertEqual(len(attempts), 2)
            self.assertIn("nature.com", attempts[0])
            self.assertIn("arxiv.org/pdf/1910.11333.pdf", attempts[1])
            self.assertEqual(cand["pdf_url"], "https://arxiv.org/pdf/1910.11333.pdf")

    def test_disk_cache_with_valid_pdf_magic_bytes(self):
        cand = {
            "title": "Pre-cached Paper",
            "arxiv_id": "2001.00001",
            "pdf_url": "https://arxiv.org/pdf/2001.00001.pdf",
        }
        key = a0.source_key(cand)
        dest = os.path.join(self.tmp.name, a0.filename_for(key))
        with open(dest, "wb") as f:
            f.write(b"%PDF-1.5 already here")

        with patch.object(a0, "download_pdf") as mock_dl:
            c, k, d = a0.download_seed_cascade([cand], force=False)
            self.assertEqual(d, dest)
            self.assertEqual(c["download_status"], "cached")
            mock_dl.assert_not_called()

    def test_disk_cache_with_corrupt_file_forces_redownload(self):
        cand = {
            "title": "Corrupt Disk File Paper",
            "arxiv_id": "2001.00001",
            "pdf_url": "https://arxiv.org/pdf/2001.00001.pdf",
        }
        key = a0.source_key(cand)
        dest = os.path.join(self.tmp.name, a0.filename_for(key))
        with open(dest, "wb") as f:
            f.write(b"not a pdf")

        def mock_download(url, path):
            with open(path, "wb") as f:
                f.write(b"%PDF-1.5 redownloaded")
            return True, None

        with patch.object(a0, "download_pdf", side_effect=mock_download) as mock_dl:
            c, k, d = a0.download_seed_cascade([cand], force=False)
            self.assertEqual(d, dest)
            self.assertEqual(c["download_status"], "downloaded")
            mock_dl.assert_called_once()


class TestAdversarialDiscoverRegistration(unittest.TestCase):
    """Stress-test discover registration and strict SeedPaper schema enforcement."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw_patch = patch.object(a0, "RAW_DIR", os.path.join(self.tmp.name, "raw"))
        self.seeds_patch = patch.object(
            a0, "SEED_PAPERS_PATH", os.path.join(self.tmp.name, "seed_papers.json")
        )
        self.traces_patch = patch.object(
            a0, "DISCOVERY_TRACES_PATH", os.path.join(self.tmp.name, "discovery_traces.json")
        )
        self.raw_patch.start()
        self.seeds_patch.start()
        self.traces_patch.start()
        self.addCleanup(self.raw_patch.stop)
        self.addCleanup(self.seeds_patch.stop)
        self.addCleanup(self.traces_patch.stop)

    def test_seed_paper_strictly_requires_canonical_fields(self):
        """SeedPaper has 8 fields: key, path, fetched_at, source, title, url, doi, arxiv_id."""
        sp = SeedPaper(
            key="arxiv:2401.00001",
            path="/path/to/paper.pdf",
            fetched_at="2026-09-26T12:00:00Z",
            source="search",
            title="A Good Paper",
            url="https://arxiv.org/pdf/2401.00001.pdf",
            doi="10.1000/123",
            arxiv_id="2401.00001",
        )
        d = sp.to_dict()
        self.assertEqual(len(d), 8)
        self.assertEqual(
            set(d.keys()),
            {"key", "path", "fetched_at", "source", "title", "url", "doi", "arxiv_id"},
        )

    def test_seed_paper_rejects_unexpected_fields_with_type_error(self):
        """Passing undeclared kwargs to SeedPaper must raise TypeError."""
        with self.assertRaises(TypeError):
            SeedPaper(
                key="k",
                path="p",
                fetched_at="t",
                source="search",
                unrecognized_metadata="malicious_payload",
            )

    def test_seed_paper_from_dict_rejects_missing_required_fields(self):
        with self.assertRaises(SchemaError):
            SeedPaper.from_dict({"key": "k", "path": "p"})

    def test_discover_end_to_end_persists_valid_seed_and_decoupled_trace(self):
        query = "fault-tolerant surface codes"

        context_return = {
            "core_problem": "Surface Code Quantum Computing",
            "primary_domain": "Quantum Information & Computing",
            "subtopics": ["Syndrome extraction", "Lattice surgery"],
            "key_concepts": ["stabilizer code", "surface code threshold"],
            "synonyms": ["planar code"],
            "seed_criteria": "Foundational threshold paper",
            "ideal_paper_type": "foundational_theory",
        }

        mock_candidate = {
            "title": "Fault-Tolerant Quantum Computation with Surface Codes",
            "doi": "10.1103/PhysRevA.86.032324",
            "arxiv_id": "1208.0928",
            "year": 2012,
            "authors": ["Austin G. Fowler", "Matteo Mariantoni"],
            "pdf_url": "https://arxiv.org/pdf/1208.0928.pdf",
        }

        def mock_download(url, dest):
            with open(dest, "wb") as f:
                f.write(b"%PDF-1.5 valid surface code paper")
            return True, None

        with patch.object(a0, "formulate_context", return_value=context_return), \
             patch.object(a0, "iterative_search", return_value=([mock_candidate], {"iterations": []})), \
             patch.object(a0, "download_pdf", side_effect=mock_download):

            path = a0.discover(query, force=True)
            self.assertIsNotNone(path)
            self.assertTrue(os.path.exists(path))

            # Verify seed_papers.json strictly adheres to SeedPaper
            seeds = a0._load_seeds()
            self.assertIn(query, seeds)
            rec = seeds[query]

            # Round-trip schema check
            validated = SeedPaper.from_dict(rec)
            self.assertEqual(validated.to_dict(), rec)
            self.assertEqual(rec["title"], mock_candidate["title"])
            self.assertEqual(rec["doi"], mock_candidate["doi"])
            self.assertEqual(rec["arxiv_id"], mock_candidate["arxiv_id"])
            self.assertEqual(rec["source"], "search")
            self.assertEqual(set(rec.keys()), {"key", "path", "fetched_at", "source", "title", "url", "doi", "arxiv_id"})

            # Verify decoupled discovery trace was written
            trace = a0.get_discovery_trace(query)
            self.assertIsNotNone(trace)
            self.assertEqual(trace["query"], query)
            self.assertEqual(trace["context_profile"]["primary_domain"], "Quantum Information & Computing")
            self.assertEqual(trace["selected_seed"]["title"], mock_candidate["title"])


if __name__ == "__main__":
    unittest.main()
