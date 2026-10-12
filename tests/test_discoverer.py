"""Tests for Agent 0's seed manifest, which now routes through SeedPaper.

The one thing actually changed by the port: seeds[query] used to be a bare
dict any caller could grow arbitrary keys into (the old **extra splat on
_download_and_record was a direct route for that). Routing both manifest
sites through SeedPaper(...).to_dict() means a malformed record fails at
write time, and the manifest a caller like orchestrate.py reads back is
always a valid SeedPaper shape.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import research_assistant.agents.agent0_discoverer as a0
from research_assistant.schemas import SeedPaper


class TestDownloadAndRecordSeedPaperWiring(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw_dir_patch = patch.object(a0, "RAW_DIR", self.tmp.name)
        self.seed_path_patch = patch.object(
            a0, "SEED_PAPERS_PATH", os.path.join(self.tmp.name, "seed_papers.json")
        )
        self.raw_dir_patch.start()
        self.seed_path_patch.start()
        self.addCleanup(self.raw_dir_patch.stop)
        self.addCleanup(self.seed_path_patch.stop)

    def test_manifest_record_round_trips_through_seed_paper(self):
        with patch.object(a0, "download_pdf", return_value=(True, None)):
            dest = a0._download_and_record(
                "some query", "arxiv:1234.5678", "https://arxiv.org/pdf/1234.5678",
                {}, force=False, arxiv_id="1234.5678", source="manual-url",
            )
        self.assertIsNotNone(dest)
        seeds = a0._load_seeds()
        record = seeds["some query"]
        # Must be exactly the shape SeedPaper produces, not a hand-rolled dict:
        # a malformed record would raise here instead of round-tripping.
        self.assertEqual(SeedPaper.from_dict(record).to_dict(), record)
        self.assertEqual(record["source"], "manual-url")
        self.assertEqual(record["arxiv_id"], "1234.5678")

    def test_rejects_undeclared_keyword_arguments(self):
        """The **extra splat is gone — an unknown kwarg must now be a TypeError,
        not a key silently smuggled into the manifest."""
        with patch.object(a0, "download_pdf", return_value=(True, None)):
            with self.assertRaises(TypeError):
                a0._download_and_record(
                    "q", "k", "https://example.com/p.pdf", {}, False,
                    unexpected_field="sneaky",
                )


class TestDiscoverFromFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw_dir_patch = patch.object(a0, "RAW_DIR", os.path.join(self.tmp.name, "raw"))
        self.seed_path_patch = patch.object(
            a0, "SEED_PAPERS_PATH", os.path.join(self.tmp.name, "seed_papers.json")
        )
        self.raw_dir_patch.start()
        self.seed_path_patch.start()
        self.addCleanup(self.raw_dir_patch.stop)
        self.addCleanup(self.seed_path_patch.stop)

    def test_discover_from_file_with_query(self):
        # Create a mock valid PDF file
        pdf_content = b"%PDF-1.4 mock content with arxiv:2401.99999 and /Title (Quantum Wires)"
        pdf_path = os.path.join(self.tmp.name, "sample_paper.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_content)

        dest = a0.discover_from_file("topological wires", pdf_path)
        self.assertIsNotNone(dest)
        self.assertTrue(os.path.exists(dest))
        self.assertTrue(dest.startswith(os.path.join(self.tmp.name, "raw")))

        seeds = a0._load_seeds()
        self.assertIn("topological wires", seeds)
        rec = seeds["topological wires"]
        self.assertEqual(rec["source"], "upload")
        self.assertEqual(rec["arxiv_id"], "2401.99999")
        self.assertEqual(rec["title"], "Quantum Wires")
        # Validate SeedPaper schema compliance
        self.assertEqual(SeedPaper.from_dict(rec).to_dict(), rec)

        # Verify get_seed and get_seed_by_path
        self.assertEqual(a0.get_seed("topological wires"), rec)
        found_rec, q = a0.get_seed_by_path(dest)
        self.assertEqual(found_rec, rec)
        self.assertEqual(q, "topological wires")

    def test_discover_from_file_without_query_infers_topic(self):
        # PDF with Title metadata but no user query
        pdf_content = b"%PDF-1.4 mock content with /Title (Superconducting Qubits In Silicon)"
        pdf_path = os.path.join(self.tmp.name, "spin_qubits_2024.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_content)

        dest = a0.discover_from_file("", pdf_path)
        self.assertIsNotNone(dest)
        seeds = a0._load_seeds()
        # Query inferred from title
        self.assertIn("Superconducting Qubits In Silicon", seeds)
        rec = seeds["Superconducting Qubits In Silicon"]
        self.assertEqual(rec["source"], "upload")
        self.assertEqual(rec["title"], "Superconducting Qubits In Silicon")

        found_rec, q = a0.get_seed_by_path(dest)
        self.assertEqual(q, "Superconducting Qubits In Silicon")

    def test_discover_from_file_fallback_to_filename_stem(self):
        # PDF with no title metadata
        pdf_content = b"%PDF-1.4 binary content with no recognizable title tag"
        pdf_path = os.path.join(self.tmp.name, "majorana_zero_modes.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_content)

        dest = a0.discover_from_file("", pdf_path)
        self.assertIsNotNone(dest)
        seeds = a0._load_seeds()
        self.assertIn("majorana zero modes", seeds)
        rec = seeds["majorana zero modes"]
        self.assertEqual(rec["title"], "majorana zero modes")
        self.assertTrue(rec["key"].startswith("file:"))

    def test_discover_from_file_bytes_input(self):
        pdf_content = b"%PDF-1.4 uploaded from stream directly"
        dest = a0.discover_from_file("my bytes query", pdf_content, filename="direct_stream.pdf")
        self.assertIsNotNone(dest)
        self.assertTrue(os.path.exists(dest))
        seeds = a0._load_seeds()
        self.assertIn("my bytes query", seeds)

    def test_two_papers_with_the_same_inferred_query_both_keep_a_record(self):
        """An inferred query must not silently overwrite another paper's record.

        The seed manifest is keyed by query. When the caller supplies one that
        is fine — re-running a query is meant to re-seed it. But on a blank
        upload the query is *inferred* from the title or the filename stem, and
        two different papers can infer the same string. Both are written to
        RAW_DIR under distinct content-hash keys, so both must survive in the
        manifest too; otherwise the first paper sits on disk with no seed
        record, the UI shows the wrong seed card, and path -> seed lookup
        fails for it.
        """
        a = a0.discover_from_file("", b"%PDF-1.4 first paper body", filename="paper.pdf")
        b = a0.discover_from_file("", b"%PDF-1.4 second paper body", filename="paper.pdf")

        self.assertNotEqual(a, b, "different content should land in different files")
        seeds = a0._load_seeds()
        self.assertEqual(len(seeds), 2)
        self.assertEqual({s["path"] for s in seeds.values()}, {a, b})

    def test_reuploading_the_same_paper_keeps_a_single_record(self):
        """Same bytes means the same key — idempotent, not a second entry."""
        a = a0.discover_from_file("", b"%PDF-1.4 identical body", filename="paper.pdf")
        b = a0.discover_from_file("", b"%PDF-1.4 identical body", filename="paper.pdf")

        self.assertEqual(a, b)
        self.assertEqual(len(a0._load_seeds()), 1)

    def test_a_supplied_query_still_re_seeds_in_place(self):
        """Only the inferred case disambiguates; an explicit query keeps its slot.

        `discover()` and `discover_from_url()` both rely on `seeds[query]` to
        answer "have I already seeded this query?", so a supplied query has to
        stay a stable key.
        """
        a0.discover_from_file("my topic", b"%PDF-1.4 first paper body", filename="a.pdf")
        a0.discover_from_file("my topic", b"%PDF-1.4 second paper body", filename="b.pdf")

        self.assertEqual(list(a0._load_seeds()), ["my topic"])

    def test_discover_from_file_rejects_invalid_pdf(self):
        not_a_pdf = os.path.join(self.tmp.name, "fake.pdf")
        with open(not_a_pdf, "wb") as f:
            f.write(b"<html>This is an html error page, not a pdf</html>")

        dest = a0.discover_from_file("some query", not_a_pdf)
        self.assertIsNone(dest)
        seeds = a0._load_seeds()
        self.assertEqual(seeds, {})

    def test_discover_from_file_nonexistent_path(self):
        dest = a0.discover_from_file("some query", os.path.join(self.tmp.name, "does_not_exist.pdf"))
        self.assertIsNone(dest)

    def test_discover_from_file_hex_title(self):
        # Hex title for 'Topological Wires'
        pdf_content = b"%PDF-1.4 mock content\n/Title <546F706F6C6F676963616C205769726573>\n"
        pdf_path = os.path.join(self.tmp.name, "hex_title.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_content)

        dest = a0.discover_from_file("", pdf_path)
        self.assertIsNotNone(dest)
        seeds = a0._load_seeds()
        self.assertIn("Topological Wires", seeds)
        self.assertEqual(seeds["Topological Wires"]["title"], "Topological Wires")

    def test_discover_from_file_xmp_dc_title_and_doi(self):
        pdf_content = (
            b"%PDF-1.4 mock content\n"
            b"<x:xmpmeta><rdf:RDF>"
            b"<dc:title><rdf:Alt><rdf:li xml:lang='x-default'>Quantum Error Correction</rdf:li></rdf:Alt></dc:title>"
            b"<prism:doi>10.1103/PhysRevA.99.123456</prism:doi>"
            b"</rdf:RDF></x:xmpmeta>"
        )
        pdf_path = os.path.join(self.tmp.name, "xmp_paper.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_content)

        dest = a0.discover_from_file("", pdf_path)
        self.assertIsNotNone(dest)
        seeds = a0._load_seeds()
        self.assertIn("Quantum Error Correction", seeds)
        rec = seeds["Quantum Error Correction"]
        self.assertEqual(rec["title"], "Quantum Error Correction")
        self.assertEqual(rec["doi"], "10.1103/physreva.99.123456")
        self.assertEqual(rec["key"], "doi:10.1103/physreva.99.123456")

    def test_discover_from_file_ignores_unprefixed_doi_in_bytes(self):
        # Random binary containing 10.1234/something should NOT be picked up as paper DOI
        pdf_content = b"%PDF-1.4 arbitrary text with 10.9999/random_string without doi label"
        pdf_path = os.path.join(self.tmp.name, "no_doi.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_content)

        dest = a0.discover_from_file("", pdf_path)
        self.assertIsNotNone(dest)
        seeds = a0._load_seeds()
        rec = next(iter(seeds.values()))
        self.assertIsNone(rec["doi"])
        self.assertTrue(rec["key"].startswith("file:"))

    def test_get_seed_by_path_returns_most_recent(self):
        seeds = {
            "First Older Topic": {"path": "/path/to/paper.pdf", "title": "Old"},
            "Second Latest Topic": {"path": "/path/to/paper.pdf", "title": "New"},
        }
        with patch.object(a0, "_load_seeds", return_value=seeds):
            rec, q = a0.get_seed_by_path("/path/to/paper.pdf")
            self.assertEqual(q, "Second Latest Topic")
            self.assertEqual(rec["title"], "New")


class TestContextFormulation(unittest.TestCase):
    def test_formulate_context_valid_llm_json(self):
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "core_problem": "Topological Superconductivity in Majorana Nanowires",
            "primary_domain": "Condensed Matter Physics",
            "subtopics": ["Majorana Zero Modes", "Topological Phase Transitions", "Quantum Wires"],
            "key_concepts": ["Majorana fermions", "topological order", "Andreev bound states"],
            "synonyms": ["Majorana bound states", "topological superconductor"],
            "seed_criteria": "Foundational theoretical framework or experimental signature",
            "ideal_paper_type": "foundational_theory",
        })
        with patch.object(a0, "chat", return_value=mock_response) as mock_chat:
            profile = a0.formulate_context("majorana nanowires")
            self.assertEqual(profile["primary_domain"], "Condensed Matter Physics")
            self.assertEqual(profile["ideal_paper_type"], "foundational_theory")
            self.assertEqual(len(profile["subtopics"]), 3)
            mock_chat.assert_called_once()
            _, kwargs = mock_chat.call_args
            self.assertEqual(kwargs.get("options", {}).get("num_ctx"), 4096)

    def test_formulate_context_code_fence_stripping(self):
        mock_response = MagicMock()
        mock_response.content = """Here is the scientific context profile:
```json
{
    "core_problem": "Lattice Based Cryptography",
    "primary_domain": "Quantum Information & Computing",
    "subtopics": ["Learning With Errors", "Short Integer Solution"],
    "key_concepts": ["LWE", "post-quantum cryptography", "ring-LWE"],
    "synonyms": ["lattice cryptography", "quantum-resistant encryption"],
    "seed_criteria": "Canonical complexity reductions and cryptographic schemes",
    "ideal_paper_type": "foundational_theory"
}
```
Hope this helps!"""
        with patch.object(a0, "chat", return_value=mock_response):
            profile = a0.formulate_context("lattice cryptography")
            self.assertEqual(profile["primary_domain"], "Quantum Information & Computing")
            self.assertIn("LWE", profile["key_concepts"])

    def test_formulate_context_partial_fields_merged_with_heuristic(self):
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "core_problem": "CRISPR-Cas9 Gene Editing",
            "subtopics": ["Base Editing", "Prime Editing"],
        })
        with patch.object(a0, "chat", return_value=mock_response):
            profile = a0.formulate_context("CRISPR-Cas9 Gene Editing")
            self.assertEqual(profile["core_problem"], "CRISPR-Cas9 Gene Editing")
            self.assertIn("Base Editing", profile["subtopics"])
            self.assertEqual(profile["primary_domain"], "Biophysics & Molecular Biology")
            self.assertTrue(profile["ideal_paper_type"])
            self.assertTrue(profile["seed_criteria"])

    def test_formulate_context_offline_exception_fallback(self):
        with patch.object(a0, "chat", side_effect=RuntimeError("Ollama offline")):
            profile = a0.formulate_context("neural network diffusion models")
            self.assertEqual(profile["primary_domain"], "Machine Learning & Artificial Intelligence")
            self.assertEqual(profile["core_problem"], "neural network diffusion models")
            self.assertTrue(len(profile["key_concepts"]) >= 3)
            self.assertTrue(len(profile["subtopics"]) >= 2)

    def test_formulate_context_domain_classification(self):
        cases = [
            ("superconducting qubit quantum error correction", "Quantum Information & Computing"),
            ("cryo-em structure of ribosome protein folding", "Biophysics & Molecular Biology"),
            ("fourier neural operators for extreme weather prediction", "Machine Learning & Artificial Intelligence"),
            ("dark matter neutrino cosmological inflation", "High Energy Physics & Cosmology"),
        ]
        with patch.object(a0, "chat", side_effect=Exception("offline")):
            for query, expected_domain in cases:
                profile = a0.formulate_context(query)
                self.assertEqual(profile["primary_domain"], expected_domain, f"Failed for query: {query}")


class TestCandidatePoolDeduplication(unittest.TestCase):
    def test_deduplicate_by_doi(self):
        pool = a0.CandidatePool()
        cand1 = {
            "title": "Majorana bound states in semiconductor nanowires",
            "doi": "10.1103/PhysRevLett.105.077001",
            "arxiv_id": "1003.1145",
            "pdf_url": "https://arxiv.org/pdf/1003.1145.pdf",
        }
        cand2 = {
            "title": "Majorana Bound States in Semiconductor Nanowires",
            "doi": "10.1103/physrevlett.105.077001",
            "arxiv_id": None,
            "pdf_url": "https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.105.077001",
        }
        self.assertTrue(pool.add(cand1, provider="openalex", iteration=1))
        self.assertFalse(pool.add(cand2, provider="semanticscholar", iteration=2))
        candidates = pool.to_list()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["hit_count"], 2)
        self.assertIn("openalex", candidates[0]["providers_seen"])
        self.assertIn("semanticscholar", candidates[0]["providers_seen"])

    def test_deduplicate_by_arxiv_id_stripping_version(self):
        pool = a0.CandidatePool()
        cand1 = {
            "title": "Quantum Supremacy Using a Programmable Superconducting Processor",
            "arxiv_id": "1910.11333v1",
            "pdf_url": "https://arxiv.org/pdf/1910.11333v1.pdf",
        }
        cand2 = {
            "title": "Quantum Supremacy Using a Programmable Superconducting Processor",
            "arxiv_id": "1910.11333v2",
            "pdf_url": "https://arxiv.org/pdf/1910.11333.pdf",
        }
        self.assertTrue(pool.add(cand1, provider="arxiv", iteration=1))
        self.assertFalse(pool.add(cand2, provider="arxiv", iteration=2))
        candidates = pool.to_list()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["hit_count"], 2)

    def test_deduplicate_by_title_hash_when_no_id(self):
        pool = a0.CandidatePool()
        cand1 = {
            "title": "Introduction to Quantum Error Correction and Fault-Tolerant Quantum Memory",
            "authors": ["Daniel Gottesman"],
            "year": "2010",
            "pdf_url": "https://example.com/gottesman.pdf",
        }
        cand2 = {
            "title": "Introduction to quantum error correction and fault-tolerant quantum memory.",
            "authors": ["D. Gottesman"],
            "year": "2010",
            "pdf_url": "https://example.com/gottesman2.pdf",
        }
        self.assertTrue(pool.add(cand1, provider="openalex", iteration=1))
        self.assertFalse(pool.add(cand2, provider="openalex", iteration=2))
        self.assertEqual(len(pool.to_list()), 1)

    def test_metadata_enrichment_across_iterations(self):
        pool = a0.CandidatePool()
        c1 = {"title": "Equivariant Diffusion", "arxiv_id": "2203.17003", "pdf_url": None}
        c2 = {
            "title": "Equivariant Diffusion for Molecule Generation",
            "arxiv_id": "2203.17003",
            "doi": "10.48550/arXiv.2203.17003",
            "pdf_url": "https://arxiv.org/pdf/2203.17003.pdf",
            "authors": ["Emiel Hoogeboom", "Victor Garcia Satorras"],
            "year": "2022",
        }
        pool.add(c1, provider="openalex", iteration=1)
        pool.add(c2, provider="arxiv", iteration=2)
        res = pool.to_list()[0]
        self.assertEqual(res["doi"], "10.48550/arXiv.2203.17003")
        self.assertEqual(res["pdf_url"], "https://arxiv.org/pdf/2203.17003.pdf")
        self.assertEqual(len(res["authors"]), 2)
        self.assertEqual(res["year"], "2022")


class TestIterativeSearch(unittest.TestCase):
    def test_iterative_search_three_iterations_accumulated(self):
        context = {
            "core_problem": "Topological quantum computation",
            "primary_domain": "Condensed Matter Physics",
            "subtopics": ["Braiding", "Majorana modes"],
            "key_concepts": ["non-abelian anyons", "topological qubit"],
            "synonyms": ["fault-tolerant quantum memory"],
            "ideal_paper_type": "comprehensive_review",
            "seed_criteria": "Foundational review paper",
        }
        mock_p1 = MagicMock(return_value=[
            {"title": "Paper 1", "arxiv_id": "2101.00001", "pdf_url": "https://arxiv.org/pdf/2101.00001.pdf"},
        ])
        mock_p2 = MagicMock(return_value=[
            {"title": "Paper 2", "arxiv_id": "2101.00002", "pdf_url": "https://arxiv.org/pdf/2101.00002.pdf"},
        ])

        with patch.dict(a0._PROVIDERS, {"openalex": mock_p1, "arxiv": mock_p2}), \
             patch.object(a0, "chat", side_effect=Exception("no llm")):
            candidates, trace = a0.iterative_search(context, providers=["openalex", "arxiv"], limit=2)
            self.assertEqual(len(trace["iterations"]), 3)
            self.assertEqual(trace["summary"]["total_iterations"], 3)
            self.assertTrue(len(candidates) >= 2)

    def test_iterative_search_provider_fault_tolerance(self):
        context = {
            "core_problem": "Deep learning architectures",
            "primary_domain": "Machine Learning & Artificial Intelligence",
            "subtopics": ["Transformers"],
            "key_concepts": ["attention"],
            "synonyms": ["self-attention"],
            "ideal_paper_type": "foundational_theory",
            "seed_criteria": "Attention is all you need paper",
        }
        failing_provider = MagicMock(side_effect=Exception("HTTP 500 Internal Error"))
        succeeding_provider = MagicMock(return_value=[
            {"title": "Attention Is All You Need", "arxiv_id": "1706.03762", "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf"}
        ])

        with patch.dict(a0._PROVIDERS, {"openalex": failing_provider, "arxiv": succeeding_provider}), \
             patch.object(a0, "chat", side_effect=Exception("offline")):
            candidates, trace = a0.iterative_search(context, providers=["openalex", "arxiv"], limit=1)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["arxiv_id"], "1706.03762")
            self.assertTrue(any("openalex" in err for err in trace["summary"]["errors_encountered"]))


class TestCandidateEvaluation(unittest.TestCase):
    def test_evaluate_candidates_llm_scoring(self):
        context = {
            "core_problem": "Quantum key distribution protocols",
            "primary_domain": "Quantum Information & Computing",
            "subtopics": ["BB84 Protocol"],
            "key_concepts": ["quantum cryptography", "photon polarization"],
            "seed_criteria": "Foundational QKD protocol paper",
        }
        candidates = [
            {"title": "Weak QKD implementation", "arxiv_id": "2001.00001", "pdf_url": "https://arxiv.org/pdf/2001.00001.pdf"},
            {"title": "Quantum Cryptography: Public Key Distribution and Coin Tossing", "arxiv_id": "2001.00002", "pdf_url": "https://arxiv.org/pdf/2001.00002.pdf"},
        ]
        mock_reply = MagicMock()
        mock_reply.content = json.dumps([
            {"index": 0, "score": 9.8, "reasoning": "Seminal foundation paper"},
            {"index": 1, "score": 6.5, "reasoning": "Practical but narrow"},
        ])
        with patch.object(a0, "chat", return_value=mock_reply):
            ranked = a0.evaluate_candidates(candidates, context)
            self.assertEqual(len(ranked), 2)
            self.assertGreater(ranked[0]["score"], ranked[1]["score"])
            self.assertEqual(ranked[0]["score"], 9.8)
            self.assertEqual(ranked[0]["eval_method"], "llm")
            self.assertEqual(ranked[1]["score"], 6.5)

    def test_evaluate_candidates_heuristic_fallback(self):
        context = {
            "core_problem": "Topological quantum computing with Majorana nanowires",
            "primary_domain": "Condensed Matter Physics",
            "subtopics": ["Majorana Zero Modes"],
            "key_concepts": ["topological superconductivity", "Majorana nanowires"],
            "synonyms": ["Majorana bound states review"],
            "seed_criteria": "Canonical survey paper",
        }
        candidates = [
            {"title": "Arbitrary un-related paper", "year": "2015", "pdf_url": "https://example.com/p1.pdf"},
            {"title": "A Survey of Majorana Nanowires and Topological Superconductivity", "year": "2024", "arxiv_id": "2401.00001", "pdf_url": "https://arxiv.org/pdf/2401.00001.pdf"},
        ]
        with patch.object(a0, "chat", side_effect=Exception("LLM unavailable")):
            ranked = a0.evaluate_candidates(candidates, context)
            self.assertEqual(ranked[0]["arxiv_id"], "2401.00001")
            self.assertGreater(ranked[0]["score"], ranked[1]["score"])
            self.assertEqual(ranked[0]["eval_method"], "heuristic_fallback")


class TestDownloadSeedCascade(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw_patch = patch.object(a0, "RAW_DIR", self.tmp.name)
        self.raw_patch.start()
        self.addCleanup(self.raw_patch.stop)

    def test_download_seed_cascade_cascades_on_failure(self):
        ranked = [
            {"title": "Paper 1 (broken)", "arxiv_id": "2001.00001", "pdf_url": "https://example.com/404.pdf"},
            {"title": "Paper 2 (working)", "arxiv_id": "2001.00002", "pdf_url": "https://example.com/valid.pdf"},
        ]

        def fake_download(url, dest):
            if "2001.00001" in url or "404.pdf" in url:
                return False, "HTTP 404 Not Found"
            with open(dest, "wb") as f:
                f.write(b"%PDF-1.5 valid content")
            return True, None

        with patch.object(a0, "download_pdf", side_effect=fake_download):
            cand, key, dest = a0.download_seed_cascade(ranked)
            self.assertIsNotNone(dest)
            self.assertEqual(cand["arxiv_id"], "2001.00002")
            self.assertTrue(os.path.exists(dest))

    def test_download_seed_cascade_all_fail_returns_none(self):
        ranked = [
            {"title": "Paper 1", "arxiv_id": "2001.00001", "pdf_url": "https://example.com/fail1.pdf"},
            {"title": "Paper 2", "arxiv_id": "2001.00002", "pdf_url": "https://example.com/fail2.pdf"},
        ]
        with patch.object(a0, "download_pdf", return_value=(False, "Connection timeout")):
            cand, key, dest = a0.download_seed_cascade(ranked)
            self.assertIsNone(cand)
            self.assertIsNone(key)
            self.assertIsNone(dest)


class TestDiscoveryTracePersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.trace_patch = patch.object(
            a0, "DISCOVERY_TRACES_PATH", os.path.join(self.tmp.name, "discovery_traces.json")
        )
        self.trace_patch.start()
        self.addCleanup(self.trace_patch.stop)

    def test_save_and_get_discovery_trace(self):
        trace_data = {
            "query": "topological quantum computing",
            "context_profile": {"primary_domain": "Condensed Matter Physics"},
            "ranked_candidates": [{"title": "Paper 1", "score": 9.5}],
        }
        a0._save_discovery_trace("topological quantum computing", trace_data)
        loaded = a0.get_discovery_trace("topological quantum computing")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["context_profile"]["primary_domain"], "Condensed Matter Physics")
        self.assertEqual(a0.get_discovery_trace("nonexistent"), None)


if __name__ == "__main__":
    unittest.main()
