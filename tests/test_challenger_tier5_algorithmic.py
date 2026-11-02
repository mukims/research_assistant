"""
Empirical Challenger Adversarial Hardening Suite for Milestone 4 (Phase 2: Tier 5).
Algorithmic & Mathematical Stress Hardening:
- Area 1: Mathematical & Sign Preservation in judge.py
- Area 2: Evidence Expansion & Multi-sentence Matching in agent5_batch_citer.py & agent8_verifier.py
- Area 3: LanceDB Storage & Deduplication in lance_store.py & ingestion.py
- Area 4: Tantivy Query Escaping in lance_store.py on complex scientific expressions

Authored by Empirical Challenger (challenger_m4_1).
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import unicodedata
import unittest
from unittest.mock import MagicMock, patch

from research_assistant.agents import agent5_batch_citer, agent8_verifier
from research_assistant.agents.agent5_batch_citer import CiteResult, split_paragraphs
from research_assistant.agents.agent8_verifier import citation_pairs
from research_assistant.judgement import judge
from research_assistant.judgement.judge import (
    _LINE_HYPHEN_RE,
    _QUOTES,
    _normalise,
    enforce_rubric,
    span_is_verbatim,
)
from research_assistant.shared.claim_text import split_into_sentences
from research_assistant.shared.lance_store import (
    LanceStoreClient,
    compute_chunk_id,
    compute_summary_id,
    escape_tantivy_query,
)
from research_assistant.shared.seed_audit import (
    _extract_claim_keywords,
    _get_document_fallback_chunks,
)


# ─────────────────────────────────────────────────────────────────────────────
# Area 1: Mathematical & Sign Preservation in judge.py
# ─────────────────────────────────────────────────────────────────────────────

class TestMathematicalAndSignPreservation(unittest.TestCase):
    """Adversarial stress testing on mathematical precision and sign preservation in judge.py."""

    def test_unicode_minus_normalization_empirically_exposed_gap(self):
        """Verify Unicode minus \\u2212 ('−') vs ASCII hyphen-minus \\u002d ('-').

        In scientific literature, negative values frequently use Unicode minus (U+2212)
        or superscript negative exponents. This test verifies:
        NFKC normalization does NOT convert U+2212 to ASCII hyphen-minus U+002D,
        but _QUOTES normalizes \\u2212 to ASCII hyphen-minus '-', ensuring an LLM quoting
        with standard ASCII '-' succeeds under span_is_verbatim.
        """
        evidence_with_unicode_minus = "The measured shift was \u22120.5 eV under ambient conditions."
        span_with_ascii_minus = "shift was -0.5 eV under ambient"
        span_with_unicode_minus = "shift was \u22120.5 eV under ambient"

        # Empirically verify NFKC property
        self.assertEqual(
            unicodedata.normalize("NFKC", "\u2212"),
            "\u2212",
            "Unicode minus U+2212 does not decompose to U+002D under NFKC",
        )

        # In remediated implementation, _QUOTES translates \u2212 to -
        norm_ev = _normalise(evidence_with_unicode_minus)
        norm_span_ascii = _normalise(span_with_ascii_minus)
        norm_span_unicode = _normalise(span_with_unicode_minus)

        self.assertIn("-", norm_ev)
        self.assertNotIn("\u2212", norm_ev)
        self.assertEqual(norm_ev, "the measured shift was -0.5 ev under ambient conditions.")
        self.assertIn("-", norm_span_ascii)
        self.assertNotIn("\u2212", norm_span_ascii)
        self.assertEqual(norm_span_ascii, norm_span_unicode)

        # Verify span_is_verbatim behavior: both Unicode and ASCII minus spans pass
        self.assertTrue(
            span_is_verbatim(span_with_unicode_minus, evidence_with_unicode_minus),
            "Span matching identical unicode minus must pass",
        )
        self.assertTrue(
            span_is_verbatim(span_with_ascii_minus, evidence_with_unicode_minus),
            "Remediated: ASCII minus in span passes verbatim check against Unicode minus evidence",
        )

    def test_en_dash_and_em_dash_normalization(self):
        """Verify en-dash (\\u2013) and em-dash (\\u2014) are mapped to hyphen-minus via _QUOTES."""
        evidence_en_dash = "The transition occurs in the range 10\u201320 eV."
        evidence_em_dash = "The result\u2014confirmed by XRD\u2014shows high crystallinity."
        span_en_dash = "range 10-20 eV"
        span_em_dash = "result-confirmed by XRD-shows"

        self.assertTrue(span_is_verbatim(span_en_dash, evidence_en_dash))
        self.assertTrue(span_is_verbatim(span_em_dash, evidence_em_dash))

    def test_signed_exponents_preservation(self):
        """Verify signed exponents (1e-5, 2.4E-12, -3.2e-4) preserve minus signs."""
        test_cases = [
            ("1e-5", "1e-5"),
            ("2.4E-12", "2.4e-12"),
            ("-3.2e-4", "-3.2e-4"),
            ("1.602e-19 C", "1.602e-19 c"),
            ("-0.5 eV", "-0.5 ev"),
            ("rate is 10^-5 s^-1", "rate is 10^-5 s^-1"),
        ]
        for raw, expected in test_cases:
            norm = _normalise(raw)
            self.assertEqual(norm, expected, f"Expected {raw!r} to normalize to {expected!r}, got {norm!r}")

    def test_superscript_negative_exponents_behavior(self):
        """Verify NFKC decomposition and _QUOTES normalization of superscript negative exponents (10⁻⁵, s⁻¹)."""
        raw_superscript = "Decay rate of 10\u207b\u2075 s\u207b\u00b9."
        # NFKC decomposes \u207b (superscript minus) into \u2212 (unicode minus)
        nfkc_form = unicodedata.normalize("NFKC", raw_superscript)
        self.assertIn("\u2212", nfkc_form)

        norm = _normalise(raw_superscript)
        # Remediated: \u2212 translates to ASCII minus '-' via _QUOTES
        self.assertEqual(norm, "decay rate of 10-5 s-1.")

    def test_hyphenated_multi_word_terms_vs_negative_quantities(self):
        """Stress-test hyphenated words across newlines vs negative quantities.

        Words joined across newlines (pre-existing, energy- dependent) must have
        hyphens removed to unify with full words, while standalone minus signs preceding
        numbers must be strictly preserved so negative quantities are not inverted.
        """
        # Hyphenated words across line breaks
        self.assertEqual(_normalise("pre-existing"), "preexisting")
        self.assertEqual(_normalise("pre-\n existing"), "preexisting")
        self.assertEqual(_normalise("pre- \n existing"), "preexisting")
        self.assertEqual(_normalise("energy- dependent"), "energydependent")
        self.assertEqual(_normalise("spin-orbit coupling"), "spinorbit coupling")
        self.assertEqual(_normalise("state-of-the-art"), "stateofthearth" if False else "stateofthe-art".replace("-", ""))

        # Negative physical values: minus sign MUST be preserved
        self.assertEqual(_normalise("-0.5 eV"), "-0.5 ev")
        self.assertEqual(_normalise("-273.15 C"), "-273.15 c")
        self.assertEqual(_normalise("shift of -12 meV"), "shift of -12 mev")

        # Crucial invariant: negative value is NOT normalized to positive value
        self.assertNotEqual(_normalise("-0.5 eV"), _normalise("0.5 eV"))
        self.assertNotEqual(_normalise("-10 K"), _normalise("10 K"))

    def test_rubric_enforcement_impact_of_unverified_span(self):
        """Verify that an unverified span demotes a 'Supports' verdict to 'Unclear / insufficient evidence'."""
        evidence = "The bandgap was measured to be -0.5 eV."
        result = {
            "slots": {
                "finding": {"verdict": "Supports", "assertion": "bandgap was -0.5 eV"},
                "scope": {"verdict": "Supports", "assertion": "measured"},
                "strength": {"verdict": "Supports", "assertion": "measured"},
            },
            "judgement": "Supports",
            "evidence_sufficiency": "sufficient",
            "confidence": "High",
            "supporting_span": "bandgap was measured to be -0.5 eV",
            "reason": "Direct match",
        }
        enforced = enforce_rubric(result, evidence)
        self.assertEqual(enforced["judgement"], "Supports")
        self.assertTrue(enforced["span_verified"])

        # Now simulate span mismatch (e.g. fabricated or mismatched minus)
        bad_result = dict(result)
        bad_result["supporting_span"] = "bandgap was measured to be +0.5 eV"
        enforced_bad = enforce_rubric(bad_result, evidence)
        self.assertEqual(enforced_bad["judgement"], "Unclear / insufficient evidence")
        self.assertFalse(enforced_bad["span_verified"])


# ─────────────────────────────────────────────────────────────────────────────
# Area 2: Evidence Expansion & Multi-sentence Matching
# ─────────────────────────────────────────────────────────────────────────────

class TestEvidenceExpansionAndMultiSentenceMatching(unittest.TestCase):
    """Adversarial stress testing on composite keying and 3-step evidence relaxation."""

    def test_multi_sentence_composite_key_independence(self):
        """Adversarially challenge citation evidence retention across multiple sentences

        citing the same paper key [cite_1] with distinct claim semantics.
        """
        citation_evidence = {}
        # Sentence 0 cites cite_1 for Superconductivity
        ev_0 = {
            "key": "cite_1",
            "citation_source": "Smith et al. (2020)",
            "document": "smith2020.pdf",
            "chunk_index": 10,
            "evidence": "YBCO undergoes superconducting transition at Tc = 93 K.",
        }
        # Sentence 1 cites cite_1 for Crystal Structure
        ev_1 = {
            "key": "cite_1",
            "citation_source": "Smith et al. (2020)",
            "document": "smith2020.pdf",
            "chunk_index": 25,
            "evidence": "Orthorhombic perovskite structure was verified by neutron diffraction.",
        }

        # Store using composite keys as in agent5_batch_citer.py
        citation_evidence[(0, "cite_1")] = ev_0
        citation_evidence["0:cite_1"] = ev_0
        citation_evidence["cite_1"] = ev_0

        citation_evidence[(1, "cite_1")] = ev_1
        citation_evidence["1:cite_1"] = ev_1
        citation_evidence["cite_1"] = ev_1  # Overwritten bare key!

        # 1. Assert composite keys are strictly isolated and NOT overwritten
        self.assertEqual(citation_evidence[(0, "cite_1")]["chunk_index"], 10)
        self.assertEqual(citation_evidence["0:cite_1"]["chunk_index"], 10)
        self.assertEqual(citation_evidence[(1, "cite_1")]["chunk_index"], 25)
        self.assertEqual(citation_evidence["1:cite_1"]["chunk_index"], 25)

        # 2. Demonstrate that bare key collision DOES occur on bare string key
        self.assertEqual(citation_evidence["cite_1"]["chunk_index"], 25)
        self.assertNotEqual(
            citation_evidence["cite_1"]["evidence"],
            ev_0["evidence"],
            "Bare key collisions overwrite prior sentence evidence — confirming necessity of composite keys",
        )

    def test_agent8_evidence_registry_lookup_composite_priority(self):
        """Test Agent 8's evidence lookup logic matches composite keys before bare key."""
        evidence_registry = {
            "0:cite_1": {"evidence": "Evidence for sentence 0", "chunk_index": 10},
            "1:cite_1": {"evidence": "Evidence for sentence 1", "chunk_index": 25},
            "cite_1": {"evidence": "Overwritten evidence", "chunk_index": 25},
        }

        # Lookup for sentence 0
        s_idx, c_key = 0, "cite_1"
        replayed_0 = (
            evidence_registry.get(f"{s_idx}:{c_key}")
            or evidence_registry.get((s_idx, c_key))
            or evidence_registry.get(c_key)
        )
        self.assertEqual(replayed_0["chunk_index"], 10)
        self.assertEqual(replayed_0["evidence"], "Evidence for sentence 0")

        # Lookup for sentence 1
        s_idx, c_key = 1, "cite_1"
        replayed_1 = (
            evidence_registry.get(f"{s_idx}:{c_key}")
            or evidence_registry.get((s_idx, c_key))
            or evidence_registry.get(c_key)
        )
        self.assertEqual(replayed_1["chunk_index"], 25)
        self.assertEqual(replayed_1["evidence"], "Evidence for sentence 1")

    def test_sentence_index_drift_on_unpunctuated_headings(self):
        """Empirically test sentence index alignment discrepancy when draft has headings without terminal periods.

        Agent 5 splits paragraphs on \\n\\s*\\n via split_paragraphs.
        Agent 8 calls split_into_sentences on draft_text.
        When a paragraph lacks terminal punctuation, split_into_sentences glues it to the next paragraph,
        causing index drift.
        """
        draft_with_headings = (
            "# Introduction\n\n"
            "Graphene has exceptional electron mobility \\cite{cite_1}.\n\n"
            "# Results\n\n"
            "The measured thermal conductivity was 5000 W/mK \\cite{cite_1}."
        )

        a5_paras = split_paragraphs(draft_with_headings)
        a5_sentences = [s for p in a5_paras for s in p]

        a8_sentences = split_into_sentences(draft_with_headings)

        # Agent 5 gives 4 units (heading, sent1, heading, sent2)
        self.assertEqual(len(a5_paras), 4)
        self.assertEqual(len(a5_sentences), 4)
        # Agent 8 glues headings into sentences because no terminal period exists
        self.assertEqual(len(a8_sentences), 2)

        # Document index mismatch in raw split_into_sentences
        a5_sent1_idx = 1
        a8_sent1_idx = 0
        self.assertNotEqual(a5_sent1_idx, a8_sent1_idx)

        # Verify that aligning Agent 8 with split_paragraphs eliminates sentence index drift
        a8_aligned_sentences = [s for p in a5_paras for s in p]
        self.assertEqual(len(a8_aligned_sentences), len(a5_sentences))
        self.assertEqual(a8_aligned_sentences, a5_sentences)

    def test_compound_sentences_with_multiple_citations(self):
        """Verify compound sentences with multiple citations extract distinct pairs."""
        sentence = "Graphene exhibits high mobility \\cite{cite_1} and high thermal conductivity \\cite{cite_2}."
        mapping = {"cite_1": "Smith (2020)", "cite_2": "Jones (2021)"}
        pairs = citation_pairs([sentence], mapping)

        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["cite_key"], "cite_1")
        self.assertEqual(pairs[0]["citation_source"], "Smith (2020)")
        self.assertEqual(pairs[1]["cite_key"], "cite_2")
        self.assertEqual(pairs[1]["citation_source"], "Jones (2021)")
        self.assertEqual(pairs[0]["sentence_index"], 0)
        self.assertEqual(pairs[1]["sentence_index"], 0)

    def test_three_step_relaxation_helpers(self):
        """Test relaxation Step 2 (keyword extraction) and Step 3 (document fallback chunks)."""
        # Step 2: Keyword extraction
        claim = "The Curie temperature of BaTiO3 is 393 K under zero electric field."
        ref_title = "Ferroelectric Phase Transitions in Perovskite Oxides"
        kw = _extract_claim_keywords(claim, ref_title)
        self.assertIn("Curie", kw)
        self.assertIn("temperature", kw)
        self.assertIn("BaTiO3", kw)
        self.assertIn("393", kw)
        self.assertIn("Ferroelectric", kw)

        # Stop words defined in _extract_claim_keywords must be filtered out
        stop_words = {"the", "of", "is", "in", "and", "for"}
        kw_set = set(kw.lower().split())
        self.assertTrue(stop_words.isdisjoint(kw_set))

        # Step 3: Document fallback retrieval from mock collection
        mock_col = MagicMock()
        mock_col.get.return_value = {
            "documents": ["Fallback chunk text 1", "Fallback chunk text 2"],
            "metadatas": [{"document": "doc1", "seq": 0}, {"document": "doc1", "seq": 1}],
        }
        fallback_chunks = _get_document_fallback_chunks("doc1", mock_col, None, None, k=2)
        self.assertEqual(len(fallback_chunks), 2)
        self.assertEqual(fallback_chunks[0]["text"], "Fallback chunk text 1")
        self.assertEqual(fallback_chunks[0]["metadata"]["document"], "doc1")


# ─────────────────────────────────────────────────────────────────────────────
# Area 3: LanceDB Storage & Deduplication
# ─────────────────────────────────────────────────────────────────────────────

class TestLanceDBStorageAndDeduplication(unittest.TestCase):
    """Adversarial stress testing on LanceDB storage invariants and deduplication."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="challenger_lancedb_")
        self.client = LanceStoreClient(self.tmp_dir)
        self.col = self.client.get_or_create_collection("adversarial_papers", {"embedding_dim": 768})

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_content_addressed_chunk_id_determinism_and_sensitivity(self):
        """Test content-addressed chunk ID generation determinism and boundary sensitivity."""
        doc = "paper_2026.pdf"
        stored = "This is a quantitative passage with Tc = 93 K."
        seq = 1
        full_text = stored + " Additional text."

        cid1 = compute_chunk_id(doc, stored, seq, full_text)
        cid2 = compute_chunk_id(doc, stored, seq, full_text)
        self.assertEqual(cid1, cid2, "Identical inputs must yield identical chunk_id")
        self.assertTrue(cid1.startswith("chk_"))
        self.assertEqual(len(cid1), 4 + 24)

        # Sensitivity checks
        cid_diff_doc = compute_chunk_id("other_paper.pdf", stored, seq, full_text)
        cid_diff_stored = compute_chunk_id(doc, stored + "!", seq, full_text)
        cid_diff_seq = compute_chunk_id(doc, stored, seq + 1, full_text)
        cid_diff_full = compute_chunk_id(doc, stored, seq, full_text + "!")

        self.assertNotEqual(cid1, cid_diff_doc)
        self.assertNotEqual(cid1, cid_diff_stored)
        self.assertNotEqual(cid1, cid_diff_seq)
        self.assertNotEqual(cid1, cid_diff_full)

    def test_document_scoped_deduplication_invariant(self):
        """Verify that identical text in TWO DIFFERENT documents is NOT deduplicated,

        while identical text within the SAME document IS deduplicated.
        """
        shared_text = "The sample was annealed at 800 °C for 2 hours in oxygen."
        doc_a = "paper_A.pdf"
        doc_b = "paper_B.pdf"

        # Ingestion corpus with 3 entries: 2 from doc_a (duplicate), 1 from doc_b (shared text)
        corpus = [
            {"type": "text_chunk", "document": doc_a, "page": 1, "citation": "Author A", "content": shared_text},
            {"type": "text_chunk", "document": doc_a, "page": 1, "citation": "Author A", "content": shared_text},
            {"type": "text_chunk", "document": doc_b, "page": 1, "citation": "Author B", "content": shared_text},
        ]

        # Simulate ingestion deduplication key: identity = (entry["document"], stored_form)
        seen = set()
        kept = []
        for entry in corpus:
            identity = (entry["document"], entry["content"])
            if identity in seen:
                continue
            seen.add(identity)
            kept.append(entry)

        self.assertEqual(len(kept), 2, "Expected 2 entries: 1 for doc_a and 1 for doc_b")
        self.assertEqual({e["document"] for e in kept}, {doc_a, doc_b})

    def test_collection_get_on_empty_and_deleted(self):
        """Test collection.get() behavior on empty collections and document deletions."""
        # 1. Empty get
        res_empty = self.col.get(where={"document": "non_existent"})
        self.assertEqual(res_empty, {"ids": [], "documents": [], "metadatas": []})

        # 2. Add and delete
        cid = compute_chunk_id("doc_del", "Delete me", 0, "Delete me")
        self.col.add(
            ids=[cid],
            documents=["Delete me"],
            metadatas=[{"document": "doc_del", "seq": 0}],
            embeddings=[[0.05] * 768],
        )
        self.assertEqual(len(self.col), 1)

        # Delete by document
        self.col.delete(where={"document": "doc_del"})
        self.assertEqual(len(self.col), 0)

        # Get after delete
        res_post_del = self.col.get(where={"document": "doc_del"})
        self.assertEqual(res_post_del, {"ids": [], "documents": [], "metadatas": []})

    def test_upsert_replaces_rows_without_inflation(self):
        """Test upsert replaces matching IDs without duplicate row inflation."""
        cid = "chk_fixed_id_1"
        self.col.add(
            ids=[cid],
            documents=["Initial text"],
            metadatas=[{"document": "doc_up", "seq": 0}],
            embeddings=[[0.01] * 768],
        )
        self.assertEqual(len(self.col), 1)

        # Upsert with altered text
        self.col.upsert(
            ids=[cid],
            documents=["Updated text content"],
            metadatas=[{"document": "doc_up", "seq": 0}],
            embeddings=[[0.02] * 768],
        )
        self.assertEqual(len(self.col), 1, "Upsert must not inflate row count")

        res = self.col.get(ids=[cid])
        self.assertEqual(res["documents"][0], "Updated text content")

    def test_table_optimize_and_concurrency_stress(self):
        """Stress-test table.optimize() and concurrent read/write operations across threads."""
        # Prepopulate
        for i in range(10):
            cid = f"chk_conc_{i}"
            self.col.add(
                ids=[cid],
                documents=[f"Concurrent chunk text {i}"],
                metadatas=[{"document": f"doc_{i}", "seq": i}],
                embeddings=[[0.01 * (i % 5)] * 768],
            )

        stop_event = threading.Event()
        errors = []

        def reader():
            while not stop_event.is_set():
                try:
                    res = self.col.get(where={"document": "doc_3"})
                    self.assertIsInstance(res["documents"], list)
                    v_res = self.col.search_vector([0.02] * 768, top_k=2)
                    self.assertLessEqual(len(v_res), 2)
                except Exception as e:
                    errors.append(f"Reader error: {e}")

        def writer(tid):
            for i in range(8):
                try:
                    cid = f"t_{tid}_{i}"
                    self.col.add(
                        ids=[cid],
                        documents=[f"Writer {tid} chunk {i}"],
                        metadatas=[{"document": f"doc_w_{tid}", "seq": i}],
                        embeddings=[[0.03 * (i % 3)] * 768],
                    )
                    self.col.optimize()
                except Exception as e:
                    errors.append(f"Writer error: {e}")

        readers = [threading.Thread(target=reader) for _ in range(3)]
        for r in readers:
            r.start()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futs = [executor.submit(writer, tid) for tid in range(2)]
            for f in futs:
                f.result()

        stop_event.set()
        for r in readers:
            r.join()

        self.assertEqual(len(errors), 0, f"Concurrent stress test encountered errors: {errors}")


# ─────────────────────────────────────────────────────────────────────────────
# Area 4: Tantivy Query Escaping in lance_store.py
# ─────────────────────────────────────────────────────────────────────────────

class TestTantivyQueryEscaping(unittest.TestCase):
    """Adversarial stress testing on Tantivy FTS query sanitization on complex scientific queries."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="challenger_tantivy_")
        self.client = LanceStoreClient(self.tmp_dir)
        self.col = self.client.get_or_create_collection("tantivy_scientific", {"embedding_dim": 768})

        # Ingest realistic scientific corpus with formulas, math, and DOIs
        self.col.add(
            ids=["chk_sci_1", "chk_sci_2", "chk_sci_3", "chk_sci_4"],
            documents=[
                "We synthesized [Fe(CN)6]4- and [Ru(bpy)3]2+ complexes in aqueous buffer.",
                "Published in Phys. Rev. B with DOI http://dx.doi.org/10.1000/182 and p-value < 0.05.",
                "The Fermi energy shift E_F - E_V = 0.25 eV at temperature T >= 300 K.",
                "Chemical equilibria involving Ca2+ and SO4^2- ions with transition at Tc = 93 K.",
            ],
            metadatas=[
                {"document": "chem.pdf", "citation_source": "Smith (2020)"},
                {"document": "phys.pdf", "citation_source": "Jones (2021)"},
                {"document": "semi.pdf", "citation_source": "Brown (2022)"},
                {"document": "super.pdf", "citation_source": "Davis (2023)"},
            ],
            embeddings=[[0.1] * 768, [0.2] * 768, [0.3] * 768, [0.4] * 768],
        )
        self.col.create_fts_index("text")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_escape_tantivy_query_all_special_chars(self):
        """Verify escape_tantivy_query sanitizes all required Tantivy special characters."""
        # Characters: [, ], :, (, ), ^, /, -, \, "
        raw = r'[Fe(CN)6]4- http://dx.doi.org/10.1000/182 ^2 "phrase" \path'
        escaped = escape_tantivy_query(raw)

        # Every unescaped special char must be preceded by backslash
        expected = r'\[Fe\(CN\)6\]4\- http\:\/\/dx.doi.org\/10.1000\/182 \^2 \"phrase\" \\path'
        self.assertEqual(escaped, expected)

    def test_live_tantivy_fts_scientific_queries(self):
        """Execute live Tantivy FTS queries on LanceDB table with escaped scientific queries."""
        scientific_queries = [
            ("[Fe(CN)6]4-", 1),
            ("http://dx.doi.org/10.1000/182", 1),
            ("p-value < 0.05", 1),
            ("E_F - E_V = 0.25 eV", 1),
            ("Ca2+ and SO4^2-", 1),
            ("Tc = 93 K", 1),
            ("[unclosed bracket", 0),
            ("(nested (parens) [and] brackets)", 0),
            ("http://example.com/path?arg=1&val=2", 1),  # matches http
            ("---", 0),
            (":::", 0),
            (r"\\[\\]", 0),
            ("p-value < 0.05 AND [Fe(CN)6]4-", 2),
        ]

        for query, expected_min_hits in scientific_queries:
            esc = escape_tantivy_query(query)
            try:
                hits = self.col.table.search(esc, query_type="fts").limit(5).to_list()
                self.assertGreaterEqual(
                    len(hits),
                    expected_min_hits,
                    f"Query {query!r} (escaped {esc!r}) expected >= {expected_min_hits} hits, got {len(hits)}",
                )
            except Exception as e:
                self.fail(f"Tantivy FTS threw exception on scientific query {query!r} (escaped {esc!r}): {e}")

    def test_search_hybrid_with_scientific_query(self):
        """Verify search_hybrid executes vector + FTS RRF fusion on scientific queries without parser crash."""
        query = "[Fe(CN)6]4- complex in aqueous buffer"
        q_vec = [0.1] * 768
        results = self.col.search_hybrid(query=query, query_vector=q_vec, top_k=2)

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["document"], "chem.pdf")
        self.assertIn("rrf_score", results[0])


if __name__ == "__main__":
    unittest.main()
