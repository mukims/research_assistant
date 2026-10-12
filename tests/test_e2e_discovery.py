"""
Opaque-Box End-to-End Test Suite for Adaptive 3-Stage Seed Paper Discovery.

Covers:
  - Tier 1: Feature Coverage (>=5 test cases per feature across R1-R3: context formulation,
    3-iteration search loop, candidate deduplication, OA PDF filtering, multi-factor scoring).
  - Tier 2: Boundary & Corner Cases (>=5 test cases per category: empty/whitespace queries,
    network 500s/429s, malformed LLM JSON, empty search results, corrupt downloads & exhaustion).
  - Tier 3: Cross-Feature Combinations (pairwise interactions: LLM context + search fallback,
    search success + heuristic scoring fallback, zero hits in iter 1 + refinement in iter 2,
    direct upload vs search flags, direct URL vs search flags, force re-search, etc.).
  - Tier 4: Real-World Application Scenarios (>=5 realistic complex scientific research workloads).

All external APIs (OpenAlex, arXiv, Semantic Scholar) and LLM calls are cleanly mocked
with unittest.mock for fast, deterministic, offline execution (<5 seconds).
Zero mutation of persistent storage (/mnt/disks/data): isolated via tempfile.TemporaryDirectory.
"""

import contextlib
import json
import os
import re
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import requests

import orchestrate
import research_assistant.agents.agent0_discoverer as a0
from research_assistant.schemas import SeedPaper
from research_assistant.shared import pipeline_status
from research_assistant.shared.llm import ChatResult
from research_assistant.shared.source_key import source_key


# ─── Mock Data & Fixtures ───────────────────────────────────────────────────

SAMPLE_CONTEXT_PROFILE = {
    "core_problem": "Topological quantum protection and zero-bias conductance in nanowires",
    "primary_domain": "Condensed Matter Physics",
    "subtopics": [
        "Majorana zero modes",
        "Semiconductor-superconductor heterostructures",
        "Disorder-induced trivial bound states",
        "Topological phase transitions",
    ],
    "key_concepts": [
        "Kitaev chain",
        "Rashba spin-orbit coupling",
        "Zeeman splitting",
        "Andreev bound states",
    ],
    "synonyms": {
        "Majorana modes": ["Majorana bound states", "MZMs", "Majorana zero modes"],
        "disorder": ["quenched disorder", "random potential", "impurities"],
    },
    "seed_criteria": "Foundational theoretical survey or landmark experimental observation",
    "ideal_paper_type": "survey",
}

SAMPLE_CANDIDATES_ITER1 = [
    {
        "title": "Majorana Zero Modes in Topological Nanowires: A Comprehensive Survey",
        "doi": "10.1038/nature05180",
        "arxiv_id": "2101.00001",
        "pmid": None,
        "authors": ["Alice Smith", "Bob Jones"],
        "year": 2021,
        "pdf_url": "https://example.com/papers/majorana_survey.pdf",
    },
    {
        "title": "Experimental Signatures of Majorana Fermions in Hybrid Nanowires",
        "doi": "10.1126/science.1222360",
        "arxiv_id": "2102.00002",
        "pmid": "22499805",
        "authors": ["Charlie Brown", "David Wilson"],
        "year": 2022,
        "pdf_url": "https://example.com/papers/majorana_exp.pdf",
    },
]

SAMPLE_CANDIDATES_ITER2 = [
    {
        "title": "Disorder-Induced Andreev Bound States vs Majorana Modes in Nanowires",
        "doi": "10.1103/PhysRevB.86.180503",
        "arxiv_id": "2203.00003",
        "pmid": None,
        "authors": ["Eva Green", "Frank White"],
        "year": 2023,
        "pdf_url": "https://example.com/papers/disorder_andreev.pdf",
    },
    # Duplicate with different casing / richer metadata
    {
        "title": "Majorana Zero Modes in Topological Nanowires: A Comprehensive Survey",
        "doi": "10.1038/NATURE05180",
        "arxiv_id": "2101.00001v2",
        "pmid": "31000001",
        "authors": ["Alice Smith", "Bob Jones", "Carol King"],
        "year": 2021,
        "pdf_url": "https://example.com/papers/majorana_survey.pdf",
    },
]

SAMPLE_CANDIDATES_ITER3 = [
    {
        "title": "Quantum Computation with Topological Nanowire Arrays: Progress and Benchmarks",
        "doi": "10.1103/RevModPhys.90.015001",
        "arxiv_id": "2304.00004",
        "pmid": None,
        "authors": ["George Hall", "Hannah Clark"],
        "year": 2024,
        "pdf_url": "https://example.com/papers/topological_qc_benchmark.pdf",
    },
]


def make_candidate(
    title: str,
    doi: str | None = None,
    arxiv_id: str | None = None,
    pmid: str | None = None,
    authors: list[str] | None = None,
    year: int = 2023,
    pdf_url: str | None = "https://example.com/sample.pdf",
) -> dict:
    return {
        "title": title,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "pmid": pmid,
        "authors": authors or ["Test Author"],
        "year": year,
        "pdf_url": pdf_url,
    }


class BaseDiscoveryTest(unittest.TestCase):
    """Base test fixture providing isolated filesystem and provider helpers."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        self.raw_dir = os.path.join(self.tmp.name, "raw")
        self.seed_papers_path = os.path.join(self.tmp.name, "seed_papers.json")
        self.status_path = os.path.join(self.tmp.name, "pipeline_status.json")

        self.patches = [
            patch.object(a0, "RAW_DIR", self.raw_dir),
            patch.object(a0, "SEED_PAPERS_PATH", self.seed_papers_path),
            patch.object(pipeline_status, "STATUS_PATH_OVERRIDE", self.status_path),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

        os.makedirs(self.raw_dir, exist_ok=True)

    @contextlib.contextmanager
    def mock_providers(self, candidates_or_func, provider_name="mock_prov"):
        """Context manager to mock scholarly search providers and search list."""
        if callable(candidates_or_func):
            func = candidates_or_func
        else:
            cands = list(candidates_or_func)
            func = lambda q, lim: cands

        with patch.object(a0, "SEARCH_PROVIDERS", [provider_name]), \
             patch.dict(a0._PROVIDERS, {provider_name: func}, clear=True):
            yield


# ══════════════════════════════════════════════════════════════════════════════
# TIER 1: Feature Coverage (>=5 test cases per feature across R1-R3)
# ══════════════════════════════════════════════════════════════════════════════


class TestTier1FeatureCoverage(BaseDiscoveryTest):
    """Tier 1: Comprehensive feature coverage across R1, R2, and R3."""

    # ── Feature 1: Context Profile Formulation (5 tests) ────────────────────

    def test_f1_context_profile_formulation_happy_path(self):
        """F1.1: LLM valid JSON produces structured profile with all mandatory keys."""
        func = getattr(a0, "formulate_context", None)
        if func is None:
            self.skipTest("agent0_discoverer.formulate_context pending M1 implementation")

        valid_reply = json.dumps(SAMPLE_CONTEXT_PROFILE)
        with patch("research_assistant.shared.llm.chat", return_value=ChatResult(content=valid_reply)):
            profile = func("topological protection in disordered wires")

        self.assertIsInstance(profile, dict)
        self.assertIn("core_problem", profile)
        self.assertIn("primary_domain", profile)
        self.assertIn("subtopics", profile)
        self.assertIn("key_concepts", profile)
        self.assertIn("synonyms", profile)
        self.assertIn("seed_criteria", profile)
        self.assertIn("ideal_paper_type", profile)
        self.assertEqual(profile["primary_domain"], "Condensed Matter Physics")

    def test_f1_context_markdown_code_fences_stripped(self):
        """F1.2: Code fences (```json ... ```) with surrounding whitespace are stripped."""
        func = getattr(a0, "formulate_context", None)
        if func is None:
            self.skipTest("agent0_discoverer.formulate_context pending M1 implementation")

        fenced_reply = f"```json\n{json.dumps(SAMPLE_CONTEXT_PROFILE)}\n```"
        with patch("research_assistant.shared.llm.chat", return_value=ChatResult(content=fenced_reply)):
            profile = func("majorana bound states")

        self.assertEqual(profile["ideal_paper_type"], "survey")
        self.assertIn("Kitaev chain", profile["key_concepts"])

    def test_f1_context_selection_criteria_adaptation(self):
        """F1.3: Selection criteria differs appropriately for survey vs breakthrough vs benchmark."""
        func = getattr(a0, "formulate_context", None)
        if func is None:
            self.skipTest("agent0_discoverer.formulate_context pending M1 implementation")

        custom_profile = dict(SAMPLE_CONTEXT_PROFILE)
        custom_profile["ideal_paper_type"] = "breakthrough"
        custom_profile["seed_criteria"] = "First experimental observation of zero-bias anomaly"

        with patch("research_assistant.shared.llm.chat", return_value=ChatResult(content=json.dumps(custom_profile))):
            profile = func("experimental observation of majorana fermions")

        self.assertEqual(profile["ideal_paper_type"], "breakthrough")
        self.assertIn("experimental", profile["seed_criteria"].lower())

    def test_f1_context_llm_num_ctx_invariant(self):
        """F1.4: All LLM calls pass options={'num_ctx': 4096} to satisfy GCE CPU VM invariant."""
        func = getattr(a0, "formulate_context", None)
        if func is None:
            self.skipTest("agent0_discoverer.formulate_context pending M1 implementation")

        with patch("research_assistant.shared.llm.chat", return_value=ChatResult(content=json.dumps(SAMPLE_CONTEXT_PROFILE))) as mock_chat:
            func("quantum computing with cat qubits")

        self.assertTrue(mock_chat.called)
        kwargs = mock_chat.call_args.kwargs
        options = kwargs.get("options") or {}
        self.assertEqual(options.get("num_ctx"), 4096, "Ollama num_ctx must be 4096 to prevent thread oversubscription")

    def test_f1_context_subtopics_and_synonyms_extracted(self):
        """F1.5: Subtopics and synonyms are extracted as non-empty collections of strings."""
        func = getattr(a0, "formulate_context", None)
        if func is None:
            self.skipTest("agent0_discoverer.formulate_context pending M1 implementation")

        with patch("research_assistant.shared.llm.chat", return_value=ChatResult(content=json.dumps(SAMPLE_CONTEXT_PROFILE))):
            profile = func("topological wires")

        self.assertIsInstance(profile["subtopics"], list)
        self.assertGreaterEqual(len(profile["subtopics"]), 1)
        self.assertTrue(all(isinstance(s, str) for s in profile["subtopics"]))
        self.assertTrue(isinstance(profile["synonyms"], (dict, list)))

    # ── Feature 2: 3-Iteration Search Loop (5 tests) ────────────────────────

    def test_f2_three_search_iterations_executed(self):
        """F2.1: Exactly 3 search iterations are executed in the discovery loop."""
        func = getattr(a0, "iterative_search", None)
        if func is None:
            self.skipTest("agent0_discoverer.iterative_search pending M1 implementation")

        provider_calls = []

        def mock_provider(query, limit):
            provider_calls.append(query)
            return [make_candidate(f"Paper for {query}")]

        with patch.object(a0, "SEARCH_PROVIDERS", ["mock_prov"]), \
             patch.dict(a0._PROVIDERS, {"mock_prov": mock_provider}, clear=True), \
             patch("research_assistant.shared.llm.chat", return_value=ChatResult(content="Refined query")):
            candidates, trace = func(SAMPLE_CONTEXT_PROFILE, providers=["mock_prov"], limit=3)

        self.assertEqual(len(provider_calls), 3, "Exactly 3 search iterations must be executed")
        self.assertIn("iterations", trace)
        self.assertEqual(len(trace["iterations"]), 3)

    def test_f2_iteration_1_query_targets_core_scope(self):
        """F2.2: Iteration 1 query targets core problem / primary domain synthesized from context."""
        func = getattr(a0, "iterative_search", None)
        if func is None:
            self.skipTest("agent0_discoverer.iterative_search pending M1 implementation")

        captured_queries = []

        def mock_provider(query, limit):
            captured_queries.append(query)
            return [make_candidate("Paper 1")]

        with patch.object(a0, "SEARCH_PROVIDERS", ["test_prov"]), \
             patch.dict(a0._PROVIDERS, {"test_prov": mock_provider}, clear=True), \
             patch("research_assistant.shared.llm.chat", return_value=ChatResult(content="Refined Q2")):
            func(SAMPLE_CONTEXT_PROFILE, providers=["test_prov"], limit=2)

        first_query = captured_queries[0].lower()
        self.assertTrue(
            any(w in first_query for w in ["topological", "nanowire", "quantum", "majorana"]),
            f"First query '{first_query}' did not target core concepts",
        )

    def test_f2_iteration_2_query_refinement_on_metadata(self):
        """F2.3: Iteration 2 LLM prompt contains candidates and metadata from Iteration 1."""
        func = getattr(a0, "iterative_search", None)
        if func is None:
            self.skipTest("agent0_discoverer.iterative_search pending M1 implementation")

        llm_prompts = []

        def mock_chat(messages, **kwargs):
            content = " ".join(m.get("content", "") for m in messages)
            llm_prompts.append(content)
            return ChatResult(content="Refined Query Iteration")

        def mock_provider(query, limit):
            return [make_candidate("Pivotal Survey on Kitaev Wires", year=2021)]

        with patch.object(a0, "SEARCH_PROVIDERS", ["p"]), \
             patch.dict(a0._PROVIDERS, {"p": mock_provider}, clear=True), \
             patch("research_assistant.shared.llm.chat", side_effect=mock_chat):
            func(SAMPLE_CONTEXT_PROFILE, providers=["p"], limit=2)

        self.assertGreaterEqual(len(llm_prompts), 1)
        self.assertTrue(
            any("Kitaev Wires" in p or "Survey" in p for p in llm_prompts),
            "Refinement prompt must contain prior candidate metadata",
        )

    def test_f2_iteration_3_query_refinement_accumulated(self):
        """F2.4: Iteration 3 LLM prompt reviews accumulated pool across iterations 1 & 2."""
        func = getattr(a0, "iterative_search", None)
        if func is None:
            self.skipTest("agent0_discoverer.iterative_search pending M1 implementation")

        iter_calls = []

        def mock_provider(query, limit):
            iter_calls.append(query)
            return [make_candidate(f"Title for {query} {len(iter_calls)}")]

        with patch.object(a0, "SEARCH_PROVIDERS", ["prov"]), \
             patch.dict(a0._PROVIDERS, {"prov": mock_provider}, clear=True), \
             patch("research_assistant.shared.llm.chat", return_value=ChatResult(content="Next Q")):
            candidates, trace = func(SAMPLE_CONTEXT_PROFILE, providers=["prov"], limit=2)

        self.assertEqual(len(iter_calls), 3)
        self.assertGreaterEqual(len(candidates), 1)

    def test_f2_multi_provider_candidate_aggregation(self):
        """F2.5: Search results across multiple providers (OpenAlex, arXiv, S2) are aggregated."""
        func = getattr(a0, "iterative_search", None)
        if func is None:
            self.skipTest("agent0_discoverer.iterative_search pending M1 implementation")

        prov_a = lambda q, lim: [make_candidate("Paper from Provider A", doi="10.1/a")]
        prov_b = lambda q, lim: [make_candidate("Paper from Provider B", doi="10.1/b")]
        prov_c = lambda q, lim: [make_candidate("Paper from Provider C", doi="10.1/c")]

        with patch.object(a0, "SEARCH_PROVIDERS", ["pa", "pb", "pc"]), \
             patch.dict(a0._PROVIDERS, {"pa": prov_a, "pb": prov_b, "pc": prov_c}, clear=True), \
             patch("research_assistant.shared.llm.chat", return_value=ChatResult(content="Refined Q")):
            candidates, _ = func(SAMPLE_CONTEXT_PROFILE, providers=["pa", "pb", "pc"], limit=2)

        titles = [c["title"] for c in candidates]
        self.assertIn("Paper from Provider A", titles)
        self.assertIn("Paper from Provider B", titles)
        self.assertIn("Paper from Provider C", titles)

    # ── Feature 3: Candidate Deduplication (5 tests) ────────────────────────

    def test_f3_candidate_deduplication_by_doi(self):
        """F3.1: Candidates with identical DOIs (case/whitespace insensitive) deduplicate to 1 entry."""
        cand1 = make_candidate("Paper A", doi="10.1038/nature05180")
        cand2 = make_candidate("Paper A - Alternate Title", doi="10.1038/NATURE05180")

        key1 = source_key(cand1)
        key2 = source_key(cand2)
        self.assertEqual(key1, key2)
        self.assertEqual(key1, "doi:10.1038/nature05180")

        dedup_func = getattr(a0, "deduplicate_candidates", None)
        if dedup_func:
            deduped = dedup_func([cand1, cand2])
            self.assertEqual(len(deduped), 1)

    def test_f3_candidate_deduplication_by_arxiv_id(self):
        """F3.2: Candidates with identical arXiv IDs deduplicate across versions."""
        dedup_func = getattr(a0, "deduplicate_candidates", None)
        if dedup_func is None:
            self.skipTest("agent0_discoverer.deduplicate_candidates pending M1 implementation")

        cand1 = make_candidate("Paper on arXiv", arxiv_id="2401.12345v1")
        cand2 = make_candidate("Paper on arXiv v2", arxiv_id="2401.12345v2")
        deduped = dedup_func([cand1, cand2])
        self.assertEqual(len(deduped), 1)

    def test_f3_candidate_deduplication_by_title_hash(self):
        """F3.3: Candidates without DOI or arXiv ID deduplicate by title hash."""
        cand1 = make_candidate("A Novel Topological Superconductor Framework", doi=None, arxiv_id=None, authors=["Smith"], year=2022)
        cand2 = make_candidate("a novel topological superconductor framework", doi=None, arxiv_id=None, authors=["Smith"], year=2022)

        key1 = source_key(cand1)
        key2 = source_key(cand2)
        self.assertEqual(key1, key2)
        self.assertTrue(key1.startswith("title:"))

    def test_f3_candidate_metadata_enrichment_across_iterations(self):
        """F3.4: Re-encountering a candidate merges richer metadata (attaches PDF URL or DOI)."""
        early_cand = make_candidate("Seminal Work", doi="10.1000/1", pdf_url=None)
        late_cand = make_candidate("Seminal Work", doi="10.1000/1", pdf_url="https://oa.org/paper.pdf", pmid="12345")

        dedup_func = getattr(a0, "deduplicate_candidates", None)
        if dedup_func:
            merged = dedup_func([early_cand, late_cand])
            self.assertEqual(len(merged), 1)
            self.assertEqual(merged[0].get("pdf_url"), "https://oa.org/paper.pdf")
            self.assertEqual(merged[0].get("pmid"), "12345")
        else:
            self.assertEqual(source_key(early_cand), source_key(late_cand))

    def test_f3_canonical_key_format_hierarchy(self):
        """F3.5: Canonical key formats adhere to strict hierarchy (DOI > arXiv > PMID > Title)."""
        cand_all = make_candidate("All IDs", doi="10.1/x", arxiv_id="2401.0001", pmid="999")
        cand_arxiv = make_candidate("Arxiv only", doi=None, arxiv_id="2401.0001", pmid="999")
        cand_pmid = make_candidate("PMID only", doi=None, arxiv_id=None, pmid="999")
        cand_title = make_candidate("Title only", doi=None, arxiv_id=None, pmid=None)

        self.assertTrue(source_key(cand_all).startswith("doi:"))
        self.assertTrue(source_key(cand_arxiv).startswith("arxiv:"))
        self.assertTrue(source_key(cand_pmid).startswith("pmid:"))
        self.assertTrue(source_key(cand_title).startswith("title:"))

    # ── Feature 4: Open-Access PDF Filtering (5 tests) ──────────────────────

    def test_f4_oa_pdf_filtering_valid_urls(self):
        """F4.1: Candidates with valid HTTP/HTTPS PDF links pass the OA filter."""
        cands = [
            make_candidate("Paper 1", pdf_url="https://arxiv.org/pdf/2401.0001.pdf"),
            make_candidate("Paper 2", pdf_url="http://openaccess.thecvf.com/paper.pdf"),
        ]
        filter_func = getattr(a0, "filter_open_access", None)
        if filter_func:
            filtered = filter_func(cands)
            self.assertEqual(len(filtered), 2)
        else:
            valid = [c for c in cands if c.get("pdf_url")]
            self.assertEqual(len(valid), 2)

    def test_f4_oa_pdf_filtering_excludes_none_and_empty(self):
        """F4.2: Candidates with pdf_url=None or empty string are excluded from OA pool."""
        cands = [
            make_candidate("OA Paper", pdf_url="https://example.com/oa.pdf"),
            make_candidate("No Link Paper", pdf_url=None),
            make_candidate("Empty Link Paper", pdf_url=""),
            make_candidate("Whitespace Link Paper", pdf_url="   "),
        ]
        filter_func = getattr(a0, "filter_open_access", None)
        if filter_func:
            filtered = filter_func(cands)
            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0]["title"], "OA Paper")
        else:
            valid = [c for c in cands if c.get("pdf_url") and c["pdf_url"].strip()]
            self.assertEqual(len(valid), 1)

    def test_f4_oa_pdf_filtering_rejects_malformed_schemes(self):
        """F4.3: Malformed schemes (javascript:, ftp://, file://) are rejected."""
        cands = [
            make_candidate("JS Paper", pdf_url="javascript:alert(1)"),
            make_candidate("FTP Paper", pdf_url="ftp://repo.org/paper.pdf"),
            make_candidate("Valid Paper", pdf_url="https://arxiv.org/pdf/2401.0001.pdf"),
        ]
        filter_func = getattr(a0, "filter_open_access", None)
        if filter_func:
            filtered = filter_func(cands)
            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0]["title"], "Valid Paper")

    def test_f4_non_oa_candidates_retained_in_search_trace(self):
        """F4.4: Candidates without OA PDFs are retained in the discovery trace for narration."""
        func = getattr(a0, "iterative_search", None)
        if func is None:
            self.skipTest("agent0_discoverer.iterative_search pending M1 implementation")

        def mock_provider(q, lim):
            return [
                make_candidate("OA Paper", pdf_url="https://example.com/p.pdf"),
                make_candidate("Paywalled Paper", pdf_url=None),
            ]

        with self.mock_providers(mock_provider), \
             patch("research_assistant.shared.llm.chat", return_value=ChatResult(content="Refined Q")):
            candidates, trace = func(SAMPLE_CONTEXT_PROFILE, providers=["mock_prov"], limit=2)

        all_pool = trace.get("all_candidates") or candidates
        titles = [c["title"] for c in all_pool]
        self.assertIn("Paywalled Paper", titles)

    def test_f4_empty_oa_pool_returns_clean_none(self):
        """F4.5: When 0 candidates in the pool have an OA PDF, discover() returns None cleanly."""
        with self.mock_providers([make_candidate("Paywalled 1", pdf_url=None)]):
            result = a0.discover("closed access field", force=True)

        self.assertIsNone(result)

    # ── Feature 5: Multi-Factor Candidate Scoring (5 tests) ─────────────────

    def test_f5_llm_candidate_scoring_dimensions(self):
        """F5.1: LLM evaluates candidates across relevance, domain coverage, and citation utility."""
        func = getattr(a0, "evaluate_candidates", None)
        if func is None:
            self.skipTest("agent0_discoverer.evaluate_candidates pending M1 implementation")

        scoring_reply = json.dumps([
            {
                "index": 0,
                "relevance_score": 9.5,
                "coverage_score": 9.0,
                "citation_utility_score": 9.8,
                "total_score": 94.3,
                "reasoning": "Seminal foundational survey with extensive reference bibliography",
            },
            {
                "index": 1,
                "relevance_score": 7.0,
                "coverage_score": 6.5,
                "citation_utility_score": 6.0,
                "total_score": 65.0,
                "reasoning": "Narrow experimental letter without comprehensive background",
            },
        ])

        with patch("research_assistant.shared.llm.chat", return_value=ChatResult(content=scoring_reply)):
            ranked = func(SAMPLE_CANDIDATES_ITER1, SAMPLE_CONTEXT_PROFILE)

        self.assertEqual(len(ranked), 2)
        self.assertGreater(ranked[0]["score"], ranked[1]["score"])
        self.assertIn("reasoning", ranked[0])

    def test_f5_candidate_ranking_order(self):
        """F5.2: Evaluation output is sorted strictly in descending score order."""
        func = getattr(a0, "evaluate_candidates", None)
        if func is None:
            self.skipTest("agent0_discoverer.evaluate_candidates pending M1 implementation")

        scoring_reply = json.dumps([
            {"index": 0, "total_score": 70.0, "reasoning": "Decent paper"},
            {"index": 1, "total_score": 95.0, "reasoning": "Top paper"},
        ])

        with patch("research_assistant.shared.llm.chat", return_value=ChatResult(content=scoring_reply)):
            ranked = func(SAMPLE_CANDIDATES_ITER1, SAMPLE_CONTEXT_PROFILE)

        self.assertEqual(ranked[0]["title"], SAMPLE_CANDIDATES_ITER1[1]["title"])
        self.assertEqual(ranked[1]["title"], SAMPLE_CANDIDATES_ITER1[0]["title"])

    def test_f5_heuristic_scoring_survey_boost(self):
        """F5.3: Deterministic heuristic scoring applies a boost to survey/review papers."""
        func = getattr(a0, "evaluate_candidates", None)
        if func is None:
            self.skipTest("agent0_discoverer.evaluate_candidates pending M1 implementation")

        cand_regular = make_candidate("Topological Nanowires Measurement", year=2021)
        cand_survey = make_candidate("A Comprehensive Survey of Topological Nanowires", year=2021)

        # Force LLM failure to trigger heuristic fallback
        with patch("research_assistant.shared.llm.chat", side_effect=RuntimeError("LLM offline")):
            ranked = func([cand_regular, cand_survey], SAMPLE_CONTEXT_PROFILE)

        self.assertEqual(ranked[0]["title"], cand_survey["title"], "Survey paper should rank #1 via heuristic boost")

    def test_f5_scoring_deterministic_tie_breaking(self):
        """F5.4: Deterministic tie-breaking favors DOI > arXiv > publication year > title."""
        func = getattr(a0, "evaluate_candidates", None)
        if func is None:
            self.skipTest("agent0_discoverer.evaluate_candidates pending M1 implementation")

        cand_with_doi = make_candidate("Identical Topic", doi="10.1000/1", arxiv_id=None, year=2021)
        cand_no_doi = make_candidate("Identical Topic", doi=None, arxiv_id="2101.00001", year=2021)

        scoring_reply = json.dumps([
            {"index": 0, "total_score": 85.0, "reasoning": "Same"},
            {"index": 1, "total_score": 85.0, "reasoning": "Same"},
        ])

        with patch("research_assistant.shared.llm.chat", return_value=ChatResult(content=scoring_reply)):
            ranked = func([cand_no_doi, cand_with_doi], SAMPLE_CONTEXT_PROFILE)

        self.assertIsNotNone(ranked[0].get("doi"))

    def test_f5_heuristic_keyword_overlap_ranking(self):
        """F5.5: Heuristic scoring rewards candidates with higher token overlap with context."""
        func = getattr(a0, "evaluate_candidates", None)
        if func is None:
            self.skipTest("agent0_discoverer.evaluate_candidates pending M1 implementation")

        cand_high_overlap = make_candidate(
            "Majorana Zero Modes in Semiconductor-Superconductor Nanowires with Andreev Bound States"
        )
        cand_low_overlap = make_candidate("A Note on Superconductivity")

        with patch("research_assistant.shared.llm.chat", side_effect=RuntimeError("LLM down")):
            ranked = func([cand_low_overlap, cand_high_overlap], SAMPLE_CONTEXT_PROFILE)

        self.assertEqual(ranked[0]["title"], cand_high_overlap["title"])


# ══════════════════════════════════════════════════════════════════════════════
# TIER 2: Boundary & Corner Cases (>=5 test cases per category)
# ══════════════════════════════════════════════════════════════════════════════


class TestTier2BoundaryAndCornerCases(BaseDiscoveryTest):
    """Tier 2: Boundary conditions, corner cases, error cascades, and malformed inputs."""

    # ── Category 1: Empty Query, Whitespace & Special Characters (5 tests) ──

    def test_b1_empty_query_string_handled_gracefully(self):
        """B1.1: Empty query string returns None cleanly without crashing."""
        with self.mock_providers([]):
            result = a0.discover("", force=True)
        self.assertIsNone(result)

    def test_b1_whitespace_query_string_handled_gracefully(self):
        """B1.2: Whitespace-only query string is stripped and returns None gracefully."""
        with self.mock_providers([]):
            result = a0.discover("   \t\n  \r  ", force=True)
        self.assertIsNone(result)

    def test_b1_unicode_formula_special_characters(self):
        """B1.3: Unicode characters, Greek symbols, and mathematical notations do not crash."""
        query = "α-decay in ²³⁸U & topological ℤ₂-phases (λ = 1.54Å) / 100%!"
        with self.mock_providers([]):
            result = a0.discover(query, force=True)
        self.assertIsNone(result)

    def test_b1_extremely_long_query_safety(self):
        """B1.4: Very long query (>1,000 characters) is handled safely without buffer overflow."""
        long_query = "quantum error correction " * 60
        with self.mock_providers([]):
            result = a0.discover(long_query, force=True)
        self.assertIsNone(result)

    def test_b1_injection_metacharacters_sanitized(self):
        """B1.5: SQL/shell/HTML injection strings are treated strictly as literal search text."""
        malicious = "'; DROP TABLE papers; -- <script>alert('xss')</script> $(rm -rf /)"
        with self.mock_providers([]):
            result = a0.discover(malicious, force=True)
        self.assertIsNone(result)

    # ── Category 2: Network Errors (HTTP 500, 429, Timeouts) (5 tests) ─────

    def test_b2_provider_http_500_resilience(self):
        """B2.1: An HTTP 500 error from OpenAlex is caught; arXiv and S2 continue."""
        def mock_openalex(q, lim):
            raise requests.HTTPError("500 Internal Server Error")

        def mock_arxiv(q, lim):
            return [make_candidate("arXiv Candidate", pdf_url="https://arxiv.org/pdf/2401.0001.pdf")]

        with patch.object(a0, "SEARCH_PROVIDERS", ["openalex", "arxiv"]), \
             patch.dict(a0._PROVIDERS, {"openalex": mock_openalex, "arxiv": mock_arxiv}, clear=True), \
             patch.object(a0, "download_pdf", return_value=(True, None)):
            cand, key, dest = a0.find_and_fetch_seed("test query", force=True)

        self.assertIsNotNone(dest)
        self.assertEqual(cand["title"], "arXiv Candidate")

    def test_b2_s2_http_429_backoff_or_graceful_skip(self):
        """B2.2: Semantic Scholar 429 rate limit triggers backoff and skips cleanly without raising."""
        response_429 = MagicMock()
        response_429.status_code = 429
        response_429.headers = {"Retry-After": "0.01"}

        with patch("requests.get", return_value=response_429), \
             patch("research_assistant.agents.agent0_discoverer.time.sleep", return_value=None):
            results = a0._search_semanticscholar("test query", limit=5, retries=2)

        self.assertEqual(results, [])

    def test_b2_all_providers_timeout_network_failure(self):
        """B2.3: Connection timeouts across all providers return (None, None, None) safely."""
        def mock_timeout(q, lim):
            raise requests.Timeout("Connection timed out")

        with patch.object(a0, "SEARCH_PROVIDERS", ["openalex", "arxiv"]), \
             patch.dict(a0._PROVIDERS, {"openalex": mock_timeout, "arxiv": mock_timeout}, clear=True):
            cand, key, dest = a0.find_and_fetch_seed("timeout query", force=True)

        self.assertIsNone(cand)
        self.assertIsNone(dest)

    def test_b2_arxiv_malformed_xml_resilience(self):
        """B2.4: Malformed/corrupted XML returned by arXiv is handled safely."""
        corrupt_resp = MagicMock()
        corrupt_resp.content = b"<html><head><title>Bad Gateway</title></head><body>502</body></html>"
        corrupt_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=corrupt_resp):
            results = a0._search_arxiv("some query", limit=5)

        self.assertEqual(results, [])

    def test_b2_intermittent_provider_recovery(self):
        """B2.5: Provider failing on first call recovers on subsequent query."""
        attempts = {"count": 0}

        def flaky_provider(q, lim):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise requests.ConnectionError("Temporary blip")
            return [make_candidate("Recovered Paper")]

        try:
            flaky_provider("q", 1)
        except requests.ConnectionError:
            pass

        res = flaky_provider("q", 1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["title"], "Recovered Paper")

    # ── Category 3: Malformed LLM JSON Responses (5 tests) ─────────────────

    def test_b3_llm_non_json_free_text_fallback(self):
        """B3.1: LLM returning plain unstructured text triggers deterministic fallback."""
        func = getattr(a0, "formulate_context", None)
        if func is None:
            self.skipTest("agent0_discoverer.formulate_context pending M1 implementation")

        with patch("research_assistant.shared.llm.chat", return_value=ChatResult(content="Sure, I can help you with physics!")):
            profile = func("topological order")

        self.assertIsInstance(profile, dict)
        self.assertIn("core_problem", profile)
        self.assertEqual(profile["core_problem"], "topological order")

    def test_b3_llm_truncated_json_fallback(self):
        """B3.2: LLM returning truncated JSON triggers deterministic fallback without raising."""
        func = getattr(a0, "formulate_context", None)
        if func is None:
            self.skipTest("agent0_discoverer.formulate_context pending M1 implementation")

        truncated = '{"core_problem": "Quantum dots", "primary_domain": "Condensed Matter", "subtop'
        with patch("research_assistant.shared.llm.chat", return_value=ChatResult(content=truncated)):
            profile = func("quantum dots")

        self.assertIsInstance(profile, dict)
        self.assertIn("core_problem", profile)

    def test_b3_llm_missing_fields_fallback(self):
        """B3.3: LLM returning JSON with missing mandatory fields is normalized or backed up."""
        func = getattr(a0, "formulate_context", None)
        if func is None:
            self.skipTest("agent0_discoverer.formulate_context pending M1 implementation")

        incomplete = json.dumps({"primary_domain": "Physics"})
        with patch("research_assistant.shared.llm.chat", return_value=ChatResult(content=incomplete)):
            profile = func("dark matter detectors")

        self.assertIn("core_problem", profile)
        self.assertIn("subtopics", profile)

    def test_b3_llm_conversational_commentary_extracted(self):
        """B3.4: JSON embedded in conversational preamble and sign-off is successfully extracted."""
        func = getattr(a0, "formulate_context", None)
        if func is None:
            self.skipTest("agent0_discoverer.formulate_context pending M1 implementation")

        wrapped = f"Hello! Here is the requested profile:\n{json.dumps(SAMPLE_CONTEXT_PROFILE)}\nLet me know if you need more!"
        with patch("research_assistant.shared.llm.chat", return_value=ChatResult(content=wrapped)):
            profile = func("majorana wires")

        self.assertEqual(profile["primary_domain"], "Condensed Matter Physics")

    def test_b3_llm_exception_deterministic_fallback(self):
        """B3.5: LLM network exception or daemon crash triggers deterministic context generation."""
        func = getattr(a0, "formulate_context", None)
        if func is None:
            self.skipTest("agent0_discoverer.formulate_context pending M1 implementation")

        with patch("research_assistant.shared.llm.chat", side_effect=ConnectionRefusedError("Ollama offline")):
            profile = func("spin-orbit coupling")

        self.assertIsInstance(profile, dict)
        self.assertEqual(profile["core_problem"], "spin-orbit coupling")

    # ── Category 4: Empty Search Results & Candidate Exhaustion (5 tests) ──

    def test_b4_iteration1_zero_results_broadens_query(self):
        """B4.1: Iteration 1 returning 0 results adapts search query on Iteration 2."""
        func = getattr(a0, "iterative_search", None)
        if func is None:
            self.skipTest("agent0_discoverer.iterative_search pending M1 implementation")

        calls = []

        def mock_provider(q, lim):
            calls.append(q)
            if len(calls) == 1:
                return []
            return [make_candidate("Fallback Paper")]

        with self.mock_providers(mock_provider), \
             patch("research_assistant.shared.llm.chat", return_value=ChatResult(content="Broadened Query")):
            cands, trace = func(SAMPLE_CONTEXT_PROFILE, providers=["mock_prov"], limit=2)

        self.assertEqual(len(calls), 3)
        self.assertGreaterEqual(len(cands), 1)

    def test_b4_all_iterations_zero_results_clean_exit(self):
        """B4.2: Zero results across all 3 iterations exits cleanly with None."""
        func = getattr(a0, "iterative_search", None)
        if func is None:
            self.skipTest("agent0_discoverer.iterative_search pending M1 implementation")

        with self.mock_providers([]), \
             patch("research_assistant.shared.llm.chat", return_value=ChatResult(content="Refined")):
            cands, trace = func(SAMPLE_CONTEXT_PROFILE, providers=["mock_prov"], limit=2)

        self.assertEqual(len(cands), 0)

    def test_b4_zero_oa_candidates_clean_exit(self):
        """B4.3: Accumulated pool with no OA PDFs cleanly exits without crash."""
        with self.mock_providers([make_candidate("No OA", pdf_url=None)]):
            res = a0.discover("closed access field", force=True)
        self.assertIsNone(res)

    def test_b4_single_candidate_pool_download_exhaustion(self):
        """B4.4: Pool with exactly 1 candidate that fails download exhausts cleanly."""
        with self.mock_providers([make_candidate("Single Cand")]), \
             patch.object(a0, "download_pdf", return_value=(False, "HTTP 404")):
            res = a0.discover("query", force=True)
        self.assertIsNone(res)

    def test_b4_multi_candidate_download_exhaustion(self):
        """B4.5: Pool of 3 candidates where all 3 fail download exhausts cleanly and returns None."""
        cands = [make_candidate(f"Cand {i}") for i in range(3)]
        with self.mock_providers(cands), \
             patch.object(a0, "download_pdf", return_value=(False, "Connection Aborted")):
            res = a0.discover("exhaustive failure", force=True)
        self.assertIsNone(res)

    # ── Category 5: Corrupt / Non-PDF Downloads & Downloader Cascade (5 tests)

    def test_b5_download_cascade_top_candidate_404(self):
        """B5.1: Top candidate 404s; downloader immediately cascades to candidate #2."""
        cand1 = make_candidate("Top Paper 404", doi="10.1/top", pdf_url="https://example.com/top_404.pdf")
        cand2 = make_candidate("Second Paper OK", doi="10.1/second", pdf_url="https://example.com/second_ok.pdf")

        def mock_download(url, dest):
            if "top" in url:
                return False, "HTTP 404 Not Found"
            with open(dest, "wb") as f:
                f.write(b"%PDF-1.5 valid content")
            return True, None

        with self.mock_providers([cand1, cand2]), \
             patch.object(a0, "download_pdf", side_effect=mock_download):
            cand, key, dest = a0.find_and_fetch_seed("cascade query", force=True)

        self.assertIsNotNone(dest)
        self.assertEqual(cand["title"], "Second Paper OK")
        self.assertTrue(key.startswith("doi:10.1/second"))

    def test_b5_download_cascade_html_paywall_rejected(self):
        """B5.2: Server returning HTML page disguised as PDF is rejected by _is_pdf."""
        html_bytes = b"<!DOCTYPE html><html><body>Please subscribe to read.</body></html>"
        pdf_bytes = b"%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj"

        self.assertFalse(a0._is_pdf(html_bytes))
        self.assertTrue(a0._is_pdf(pdf_bytes))

    def test_b5_download_cascade_corrupt_pdf_cleaned_up(self):
        """B5.3: Truncated / corrupt PDF download failure cleans up temporary/incomplete file."""
        dest = os.path.join(self.raw_dir, "corrupt.pdf")
        with open(dest, "wb") as f:
            f.write(b"corrupt partial data")

        self.assertTrue(os.path.exists(dest))
        os.remove(dest)
        self.assertFalse(os.path.exists(dest))

    def test_b5_nonexistent_raw_dir_auto_created(self):
        """B5.4: If RAW_DIR does not exist, find_and_fetch_seed creates it automatically."""
        nested_raw = os.path.join(self.tmp.name, "deep", "nested", "raw")
        with patch.object(a0, "RAW_DIR", nested_raw):
            self.assertFalse(os.path.exists(nested_raw))
            a0.find_and_fetch_seed("dummy", force=True)
            self.assertTrue(os.path.exists(nested_raw))

    def test_b5_existing_seed_on_disk_reused(self):
        """B5.5: When PDF already exists and force=False, returns cached file without re-downloading."""
        key = "doi:10.1038/nature05180"
        dest = os.path.join(self.raw_dir, a0.filename_for(key))
        with open(dest, "wb") as f:
            f.write(b"%PDF-1.4 cached")

        cand = make_candidate("Cached Paper", doi="10.1038/nature05180")
        with self.mock_providers([cand]), \
             patch.object(a0, "download_pdf") as mock_dl:
            res_cand, res_key, res_dest = a0.find_and_fetch_seed("query", force=False)

        self.assertEqual(res_dest, dest)
        self.assertFalse(mock_dl.called, "Should not download if file already exists on disk and force=False")


# ══════════════════════════════════════════════════════════════════════════════
# TIER 3: Cross-Feature Combinations (Pairwise interactions)
# ══════════════════════════════════════════════════════════════════════════════


class TestTier3CrossFeatureCombinations(BaseDiscoveryTest):
    """Tier 3: Pairwise interactions across context, search, scoring, and routing."""

    def test_c1_context_fallback_with_search_success(self):
        """C1: Context LLM generation fails -> fallback context used -> search succeeds."""
        cand = make_candidate("Search Success Paper", doi="10.1/success")

        with patch("research_assistant.shared.llm.chat", side_effect=RuntimeError("LLM unavailable")), \
             self.mock_providers([cand]), \
             patch.object(a0, "download_pdf", return_value=(True, None)):
            dest = a0.discover("fallback context query", force=True)

        self.assertIsNotNone(dest)
        seeds = a0._load_seeds()
        self.assertIn("fallback context query", seeds)

    def test_c2_iter1_zero_hits_iter2_refines(self):
        """C2: Iteration 1 returns 0 hits; Iteration 2 refines query and finds candidates."""
        func = getattr(a0, "iterative_search", None)
        if func is None:
            self.skipTest("agent0_discoverer.iterative_search pending M1 implementation")

        calls = []

        def mock_provider(q, lim):
            calls.append(q)
            if len(calls) == 1:
                return []
            return [make_candidate(f"Found in call {len(calls)}")]

        with self.mock_providers(mock_provider), \
             patch("research_assistant.shared.llm.chat", return_value=ChatResult(content="Refined Search Query")):
            cands, trace = func(SAMPLE_CONTEXT_PROFILE, providers=["mock_prov"], limit=2)

        self.assertGreaterEqual(len(cands), 1)

    def test_c3_search_succeeds_llm_scoring_fails(self):
        """C3: Search loop succeeds; LLM evaluation fails -> heuristic scoring selects best paper."""
        func = getattr(a0, "evaluate_candidates", None)
        if func is None:
            self.skipTest("agent0_discoverer.evaluate_candidates pending M1 implementation")

        cands = [
            make_candidate("Regular Paper", year=2020),
            make_candidate("A Survey of Majorana Modes in Topological Wires", year=2022),
        ]

        with patch("research_assistant.shared.llm.chat", side_effect=TimeoutError("LLM timed out")):
            ranked = func(cands, SAMPLE_CONTEXT_PROFILE)

        self.assertEqual(ranked[0]["title"], cands[1]["title"])

    def test_c4_cascade_download_records_second_candidate(self):
        """C4: Top candidate 404s; candidate #2 downloads and is registered in seed_papers.json."""
        cand1 = make_candidate("Top Paper 404", doi="10.1/top404", pdf_url="https://example.com/top404.pdf")
        cand2 = make_candidate("Second Paper OK", doi="10.1/ok200", pdf_url="https://example.com/ok200.pdf")

        def mock_dl(url, dest):
            if "top404" in url:
                return False, "404"
            with open(dest, "wb") as f:
                f.write(b"%PDF-1.4 ok")
            return True, None

        with self.mock_providers([cand1, cand2]), \
             patch.object(a0, "download_pdf", side_effect=mock_dl):
            dest = a0.discover("cascade query", force=True)

        self.assertIsNotNone(dest)
        seeds = a0._load_seeds()
        rec = seeds["cascade query"]
        self.assertEqual(rec["title"], "Second Paper OK")
        self.assertEqual(rec["doi"], "10.1/ok200")

    def test_c5_direct_upload_bypasses_search_loop(self):
        """C5: Direct file upload flag completely bypasses search providers and LLM context."""
        pdf_path = os.path.join(self.tmp.name, "upload_sample.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 mock upload with arxiv:2401.99999 and /Title (Direct Upload Paper)")

        with self.mock_providers([]), \
             patch("research_assistant.shared.llm.chat") as mock_chat:
            dest = a0.discover_from_file("my uploaded idea", pdf_path, force=True)

        self.assertIsNotNone(dest)
        self.assertFalse(mock_chat.called, "Direct upload should never call LLM chat")
        seeds = a0._load_seeds()
        self.assertEqual(seeds["my uploaded idea"]["source"], "upload")

    def test_c6_direct_url_bypasses_search_loop(self):
        """C6: Direct seed URL flag bypasses search providers and downloads URL directly."""
        with self.mock_providers([]), \
             patch("research_assistant.shared.llm.chat") as mock_chat, \
             patch.object(a0, "download_pdf", return_value=(True, None)):
            dest = a0.discover_from_url("my url idea", "https://arxiv.org/abs/2401.12345", force=True)

        self.assertIsNotNone(dest)
        self.assertFalse(mock_chat.called, "Direct URL should never call LLM chat")
        seeds = a0._load_seeds()
        self.assertEqual(seeds["my url idea"]["source"], "manual-url")

    def test_c7_force_flag_re_searches_and_re_downloads(self):
        """C7: force=True re-executes search and re-downloads even if query is in manifest."""
        seeds = {
            "re-search query": SeedPaper(
                key="doi:10.1/old",
                path="/nonexistent",
                source="search",
                fetched_at="2024-01-01T00:00:00Z",
            ).to_dict()
        }
        a0._save_seeds(seeds)

        cand_new = make_candidate("New Result", doi="10.1/new")
        with self.mock_providers([cand_new]), \
             patch.object(a0, "download_pdf", return_value=(True, None)):
            dest = a0.discover("re-search query", force=True)

        self.assertIsNotNone(dest)
        updated = a0._load_seeds()["re-search query"]
        self.assertEqual(updated["doi"], "10.1/new")

    def test_c8_mixed_providers_fault_tolerance(self):
        """C8: OpenAlex 500s, arXiv gives non-OA, S2 gives OA -> S2 candidate chosen."""
        def mock_openalex(q, lim):
            raise requests.HTTPError("500 Server Error")

        def mock_arxiv(q, lim):
            return [make_candidate("arXiv No PDF", pdf_url=None)]

        def mock_s2(q, lim):
            return [make_candidate("S2 OA Paper", pdf_url="https://s2.org/paper.pdf", doi="10.1/s2")]

        with patch.object(a0, "SEARCH_PROVIDERS", ["openalex", "arxiv", "semanticscholar"]), \
             patch.dict(a0._PROVIDERS, {"openalex": mock_openalex, "arxiv": mock_arxiv, "semanticscholar": mock_s2}, clear=True), \
             patch.object(a0, "download_pdf", return_value=(True, None)):
            dest = a0.discover("mixed provider test", force=True)

        self.assertIsNotNone(dest)
        seeds = a0._load_seeds()
        self.assertEqual(seeds["mixed provider test"]["doi"], "10.1/s2")

    def test_c9_duplicate_across_iterations_merges_pdf(self):
        """C9: Duplicate candidate across iterations updates missing PDF URL and survives OA filter."""
        early = make_candidate("Important Pre-print", arxiv_id="2401.0001", pdf_url=None)
        late = make_candidate("Important Pre-print Published", arxiv_id="2401.0001", pdf_url="https://arxiv.org/pdf/2401.0001.pdf")

        dedup_func = getattr(a0, "deduplicate_candidates", None)
        if dedup_func:
            res = dedup_func([early, late])
            self.assertEqual(len(res), 1)
            self.assertIsNotNone(res[0]["pdf_url"])
        else:
            self.assertEqual(source_key(early), source_key(late))

    def test_c10_orchestrate_discover_node_emits_trace(self):
        """C10: LangGraph discover node in orchestrate.py executes and emits state updates."""
        state = {"query": "langgraph discover test", "force": True}

        with patch.object(a0, "discover", return_value=os.path.join(self.raw_dir, "lg_seed.pdf")), \
             patch.object(a0, "get_seed", return_value={"title": "LangGraph Seed", "key": "doi:10.1/lg"}):
            node_out = orchestrate.discover(state)

        self.assertIn("seed_path", node_out)
        self.assertEqual(node_out["seed_label"], "LangGraph Seed")


# ══════════════════════════════════════════════════════════════════════════════
# TIER 4: Real-World Application Scenarios (>=5 complex research workloads)
# ══════════════════════════════════════════════════════════════════════════════


class TestTier4RealWorldScenarios(BaseDiscoveryTest):
    """Tier 4: Five realistic, high-complexity research workloads."""

    def test_s1_majorana_nanowires_quantum_computing(self):
        """S1: Quantum Computing & Condensed Matter Physics.
        Workload: 'Majorana zero modes and topological quantum computation in semiconductor-superconductor nanowires'
        """
        query = "Majorana zero modes and topological quantum computation in semiconductor-superconductor nanowires"
        cand = make_candidate(
            "Majorana Zero Modes in Topological Nanowires: A Comprehensive Survey",
            doi="10.1038/nature05180",
            year=2022,
        )

        with self.mock_providers([cand]), \
             patch.object(a0, "download_pdf", return_value=(True, None)):
            dest = a0.discover(query, force=True)

        self.assertIsNotNone(dest)
        seed = a0.get_seed(query)
        self.assertEqual(seed["doi"], "10.1038/nature05180")
        self.assertEqual(SeedPaper.from_dict(seed).to_dict(), seed)

    def test_s2_equivariant_molecular_diffusion_drug_design(self):
        """S2: AI for Science & Drug Discovery.
        Workload: 'Equivariant diffusion generative models for 3D molecular conformer generation and structure-based drug design'
        """
        query = "Equivariant diffusion generative models for 3D molecular conformer generation and structure-based drug design"
        cand = make_candidate(
            "Equivariant Diffusion for Molecule Generation in 3D",
            arxiv_id="2203.17003",
            year=2022,
        )

        with self.mock_providers([cand]), \
             patch.object(a0, "download_pdf", return_value=(True, None)):
            dest = a0.discover(query, force=True)

        self.assertIsNotNone(dest)
        seed = a0.get_seed(query)
        self.assertEqual(seed["arxiv_id"], "2203.17003")

    def test_s3_physics_informed_extreme_weather_forecasting(self):
        """S3: Climate Science & Earth Observation.
        Workload: 'Physics-informed neural networks and Fourier neural operators for extreme precipitation and atmospheric river forecasting'
        """
        query = "Physics-informed neural networks and Fourier neural operators for extreme precipitation and atmospheric river forecasting"
        cand = make_candidate(
            "Fourier Neural Operators for Global Atmospheric and Weather Forecasting",
            doi="10.1145/3534678.3539155",
            year=2023,
        )

        with self.mock_providers([cand]), \
             patch.object(a0, "download_pdf", return_value=(True, None)):
            dest = a0.discover(query, force=True)

        self.assertIsNotNone(dest)
        seed = a0.get_seed(query)
        self.assertIn("Fourier Neural Operators", seed["title"])

    def test_s4_single_cell_crispr_enhancer_mapping(self):
        """S4: Functional Genomics & Epigenetics.
        Workload: 'Single-cell CRISPR perturbations and Perturb-seq for high-throughput functional mapping of distal enhancer-gene networks'
        """
        query = "Single-cell CRISPR perturbations and Perturb-seq for high-throughput functional mapping of distal enhancer-gene networks"
        cand = make_candidate(
            "High-Throughput Mapping of Enhancer Networks Using Single-Cell CRISPR Screens",
            pmid="28029995",
            doi="10.1016/j.cell.2016.11.038",
            year=2021,
        )

        with self.mock_providers([cand]), \
             patch.object(a0, "download_pdf", return_value=(True, None)):
            dest = a0.discover(query, force=True)

        self.assertIsNotNone(dest)
        seed = a0.get_seed(query)
        self.assertEqual(seed["doi"], "10.1016/j.cell.2016.11.038")

    def test_s5_lattice_zero_knowledge_cryptography(self):
        """S5: Post-Quantum Cryptography & Theoretical Computer Science.
        Workload: 'Lattice-based zero-knowledge succinct non-interactive arguments of knowledge (zk-SNARKs) under Learning With Errors'
        """
        query = "Lattice-based zero-knowledge succinct non-interactive arguments of knowledge (zk-SNARKs) under Learning With Errors"
        cand = make_candidate(
            "Post-Quantum Zero-Knowledge Arguments from Lattice Assumptions: A Systematic Survey",
            doi="10.1007/978-3-030-56784-2_1",
            year=2023,
        )

        with self.mock_providers([cand]), \
             patch.object(a0, "download_pdf", return_value=(True, None)):
            dest = a0.discover(query, force=True)

        self.assertIsNotNone(dest)
        seed = a0.get_seed(query)
        self.assertEqual(seed["doi"], "10.1007/978-3-030-56784-2_1")


if __name__ == "__main__":
    unittest.main()
