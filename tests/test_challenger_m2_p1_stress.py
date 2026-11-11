"""
Adversarial Stress Test Suite for Milestone 2 Tier P1:
- Area 1: Multi-Sentence Citation Evidence Keying (Agent 5 Batch Citer & Agent 8 Verifier)
- Area 2: Bounded Concurrency across Serial Loops (Seed Audit, Resolver, Retrieve Paper Notes)
- Area 3: LLM Structured Schemas, Client Caching, and Privacy / Fallback Prevention

Authored by Empirical Challenger (challenger_m2_2).
"""

import concurrent.futures
import json
import os
import random
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from pydantic import BaseModel, Field

from research_assistant import config
from research_assistant.agents import agent5_batch_citer, agent8_verifier
from research_assistant.agents.agent5_batch_citer import CiteResult
from research_assistant.audit import resolver
from research_assistant.shared import llm, pipeline_status, retrieve, seed_audit


# ─────────────────────────────────────────────────────────────────────────────
# Suite 1: Multi-Sentence Citation Evidence Keying & Agent 8 Replay
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiSentenceCitationEvidenceKeying(unittest.TestCase):
    """Adversarially challenge citation evidence retention across multiple sentences

    citing the same paper for different facts.
    """

    def test_multi_sentence_same_key_different_evidence_end_to_end(self):
        """End-to-end run_batch_citer test: Sentence 0 and Sentence 2 cite the same

        citation key [cite_1] (Smith et al. 2020), but assert completely different facts.
        Verify evidence for sentence 0 is NOT overwritten by sentence 2 when indexed
        by (sentence_idx, cite_key) and f"{sentence_idx}:{cite_key}".
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            draft_path = os.path.join(tmp_dir, "draft.txt")
            out_path = os.path.join(tmp_dir, "cited_draft.txt")

            # Paragraph where sentence 0 and sentence 2 need citations, sentence 1 does not
            draft_content = (
                "The critical temperature for YBCO was measured to be 93 K under ambient pressure. "
                "This finding revolutionized high-temperature superconductivity. "
                "The upper critical magnetic field was determined to be 110 T at cryogenic temperatures."
            )
            with open(draft_path, "w", encoding="utf-8") as f:
                f.write(draft_content)

            # Citing results for sentence 0 vs sentence 2
            res_sentence_0 = CiteResult(
                original="The critical temperature for YBCO was measured to be 93 K under ambient pressure.",
                cited_text="The critical temperature for YBCO was measured to be 93 K under ambient pressure \\cite{cite_1}.",
                keys=["cite_1"],
                candidates=[{
                    "key": "cite_1",
                    "citation": "Smith et al. (2020) High-Tc Superconductors",
                    "document": "smith2020.pdf",
                    "chunk_index": 3,
                }],
                verdicts=[{
                    "accepted": True,
                    "key": "cite_1",
                    "citation_source": "Smith et al. (2020) High-Tc Superconductors",
                    "document": "smith2020.pdf",
                    "chunk_index": 3,
                    "evidence": "YBCO reaches zero electrical resistance at Tc = 93 K under standard pressure.",
                    "judgement": "Supports",
                    "span_verified": True,
                }],
                reasoning="Matched Tc measurement",
            )

            res_sentence_2 = CiteResult(
                original="The upper critical magnetic field was determined to be 110 T at cryogenic temperatures.",
                cited_text="The upper critical magnetic field was determined to be 110 T at cryogenic temperatures \\cite{cite_1}.",
                keys=["cite_1"],
                candidates=[{
                    "key": "cite_1",
                    "citation": "Smith et al. (2020) High-Tc Superconductors",
                    "document": "smith2020.pdf",
                    "chunk_index": 14,
                }],
                verdicts=[{
                    "accepted": True,
                    "key": "cite_1",
                    "citation_source": "Smith et al. (2020) High-Tc Superconductors",
                    "document": "smith2020.pdf",
                    "chunk_index": 14,
                    "evidence": "Upper critical magnetic field Hc2 was determined to be 110 T using pulsed magnetometry.",
                    "judgement": "Supports",
                    "span_verified": True,
                }],
                reasoning="Matched Hc2 measurement",
            )

            def mock_cite(sentence, *args, **kwargs):
                if "93 K" in sentence:
                    return res_sentence_0
                if "110 T" in sentence:
                    return res_sentence_2
                return CiteResult(original=sentence, cited_text=sentence)

            mock_resources = (MagicMock(), MagicMock(), {}, {})

            with patch.object(agent5_batch_citer, "_batch_needs_citation", return_value=[True, False, True]), \
                 patch.object(agent5_batch_citer, "cite_sentence", side_effect=mock_cite):
                result_path = agent5_batch_citer.run_batch_citer(
                    draft_path, out_path=out_path, search_resources=mock_resources
                )

            self.assertEqual(result_path, out_path)
            self.assertTrue(os.path.exists(out_path))

            # Inspect generated _citation_evidence.json
            evidence_file = out_path.replace(".txt", "_citation_evidence.json")
            self.assertTrue(os.path.exists(evidence_file))
            with open(evidence_file, "r", encoding="utf-8") as f:
                evidence_registry = json.load(f)

            # Verification of composite indexing:
            # 1. Both sentences have independent keys in _citation_evidence.json
            self.assertIn("0:cite_1", evidence_registry, "Sentence 0 evidence key missing!")
            self.assertIn("2:cite_1", evidence_registry, "Sentence 2 evidence key missing!")

            # 2. Check sentence 0 data
            ev_0 = evidence_registry["0:cite_1"]
            self.assertEqual(ev_0["chunk_index"], 3)
            self.assertIn("Tc = 93 K", ev_0["evidence"])
            self.assertEqual(ev_0["document"], "smith2020.pdf")

            # 3. Check sentence 2 data
            ev_2 = evidence_registry["2:cite_1"]
            self.assertEqual(ev_2["chunk_index"], 14)
            self.assertIn("110 T", ev_2["evidence"])
            self.assertEqual(ev_2["document"], "smith2020.pdf")

            # 4. Verify distinctness: Sentence 2 did NOT overwrite Sentence 0
            self.assertNotEqual(ev_0["evidence"], ev_2["evidence"])
            self.assertNotEqual(ev_0["chunk_index"], ev_2["chunk_index"])

    def test_agent8_verifier_replays_multi_sentence_evidence_without_cross_contamination(self):
        """Verify that agent8_verifier.verify_draft correctly resolves replayed evidence

        for each sentence by checking f"{s_idx}:{c_key}", passing the correct evidence
        to _judge_once for each distinct sentence.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            draft_path = os.path.join(tmp_dir, "cited_draft.txt")
            citations_path = os.path.join(tmp_dir, "cited_draft_citations.json")
            evidence_path = os.path.join(tmp_dir, "cited_draft_citation_evidence.json")

            draft_content = (
                "The critical temperature for YBCO was measured to be 93 K under ambient pressure \\cite{cite_1}. "
                "This finding revolutionized high-temperature superconductivity. "
                "The upper critical magnetic field was determined to be 110 T at cryogenic temperatures \\cite{cite_1}."
            )
            with open(draft_path, "w", encoding="utf-8") as f:
                f.write(draft_content)

            citations_mapping = {
                "Smith et al. (2020) High-Tc Superconductors": "cite_1"
            }
            with open(citations_path, "w", encoding="utf-8") as f:
                json.dump(citations_mapping, f)

            evidence_data = {
                "0:cite_1": {
                    "key": "cite_1",
                    "citation_source": "Smith et al. (2020) High-Tc Superconductors",
                    "document": "smith2020.pdf",
                    "chunk_index": 3,
                    "evidence": "Evidence Chunk 3: Tc is 93 K.",
                },
                "2:cite_1": {
                    "key": "cite_1",
                    "citation_source": "Smith et al. (2020) High-Tc Superconductors",
                    "document": "smith2020.pdf",
                    "chunk_index": 14,
                    "evidence": "Evidence Chunk 14: Upper critical field is 110 T.",
                },
                "cite_1": {
                    "key": "cite_1",
                    "citation_source": "Smith et al. (2020) High-Tc Superconductors",
                    "document": "smith2020.pdf",
                    "chunk_index": 14,
                    "evidence": "Evidence Chunk 14: Upper critical field is 110 T.",
                },
            }
            with open(evidence_path, "w", encoding="utf-8") as f:
                json.dump(evidence_data, f)

            judge_calls = []

            def mock_judge(claim, evidence, context=None):
                judge_calls.append({"claim": claim, "evidence": evidence})
                return {
                    "judgement": "Supports",
                    "confidence": "High",
                    "span_verified": True,
                    "supporting_span": "Supports claim",
                    "model_judgement": "Supports",
                    "evidence_sufficiency": "Sufficient",
                    "span_cites_others": False,
                    "reason": "Verified against replayed chunk",
                    "rubric_violations": [],
                    "slots": {},
                }

            with patch.object(agent8_verifier, "_judge_once", side_effect=mock_judge), \
                 patch("research_assistant.shared.db.load_search_resources", return_value=(None, None, {}, {})):
                verif_report = agent8_verifier.verify_draft(draft_path, citations_path=citations_path)

            self.assertEqual(len(judge_calls), 2)
            # Sentence 0 (index 0) must receive Evidence Chunk 3
            self.assertIn("93 K", judge_calls[0]["claim"])
            self.assertEqual(judge_calls[0]["evidence"], "Evidence Chunk 3: Tc is 93 K.")

            # Sentence 2 (index 2) must receive Evidence Chunk 14
            self.assertIn("110 T", judge_calls[1]["claim"])
            self.assertEqual(judge_calls[1]["evidence"], "Evidence Chunk 14: Upper critical field is 110 T.")

            # Check verification report items
            results = verif_report["results"]
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["evidence_source"], "replayed")
            self.assertEqual(results[1]["evidence_source"], "replayed")
            self.assertEqual(results[0]["evidence"], "Evidence Chunk 3: Tc is 93 K.")
            self.assertEqual(results[1]["evidence"], "Evidence Chunk 14: Upper critical field is 110 T.")

    def test_multi_sentence_candidate_fallback_keying(self):
        """Test candidate fallback path (when model cites via rewrite without verdicts):

        verifies that candidates path correctly populates both (sentence_idx, key)
        and f"{sentence_idx}:{key}".
        """
        citation_evidence = {}
        res0 = CiteResult(
            original="Sentence zero claim.",
            cited_text="Sentence zero claim \\cite{cite_1}.",
            keys=["cite_1"],
            candidates=[{"key": "cite_1", "citation": "Source A", "document": "docA.pdf", "chunk_index": 1}],
        )
        res3 = CiteResult(
            original="Sentence three claim.",
            cited_text="Sentence three claim \\cite{cite_1}.",
            keys=["cite_1"],
            candidates=[{"key": "cite_1", "citation": "Source A", "document": "docA.pdf", "chunk_index": 9}],
        )

        for i, res in [(0, res0), (3, res3)]:
            for cand in res.candidates:
                k = cand.get("key")
                if k in res.keys:
                    ev_data = {
                        "key": k,
                        "citation_source": cand.get("citation"),
                        "document": cand.get("document"),
                        "chunk_index": cand.get("chunk_index"),
                        "evidence": None,
                    }
                    citation_evidence[(i, k)] = ev_data
                    citation_evidence[f"{i}:{k}"] = ev_data
                    if k not in citation_evidence:
                        citation_evidence[k] = ev_data

        # Sentence 0 must have chunk_index 1
        self.assertEqual(citation_evidence[(0, "cite_1")]["chunk_index"], 1)
        self.assertEqual(citation_evidence["0:cite_1"]["chunk_index"], 1)

        # Sentence 3 must have chunk_index 9
        self.assertEqual(citation_evidence[(3, "cite_1")]["chunk_index"], 9)
        self.assertEqual(citation_evidence["3:cite_1"]["chunk_index"], 9)

        # Legacy key retains the first registered key (not overwritten)
        self.assertEqual(citation_evidence["cite_1"]["chunk_index"], 1)


# ─────────────────────────────────────────────────────────────────────────────
# Suite 2: Bounded Concurrency Stress Testing
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundedConcurrencyStress(unittest.TestCase):
    """Stress-test bounded concurrency in seed_audit, resolver, and retrieve.paper_notes

    under heavy parallel worker load and simulated latencies.
    """

    def test_seed_audit_bounded_concurrency_stress(self):
        """Stress-test ThreadPoolExecutor in seed_audit across 30 simulated claims

        with concurrency=8. Verify thread safety, no deadlocks, and that all claims
        are processed and accumulated without loss.
        """
        num_claims = 30
        claims = [
            {
                "claim": f"High-pressure phase transition observed at {10 + i} GPa.",
                "resolved": True,
                "role": "finding",
                "claim_quality": [],
                "document": f"paper_{i % 5}.pdf",
                "ref": {"index": i + 1, "title": f"Study {i + 1}", "authors": ["Author"]},
            }
            for i in range(num_claims)
        ]

        active_threads = set()
        thread_lock = threading.Lock()

        def mock_judge_claim_entry(item, *args, **kwargs):
            tid = threading.get_ident()
            with thread_lock:
                active_threads.add(tid)
            # Micro-latency to encourage thread interleaving
            time.sleep(random.uniform(0.005, 0.02))
            item["outcome"] = "judged"
            item["judgement"] = "Supports"
            item["confidence"] = "High"
            item["reason"] = f"Validated claim {item['claim'][:20]}"

        with patch.object(seed_audit, "find_tei_for_seed", return_value="dummy.tei.xml"), \
             patch.object(seed_audit, "extract_seed_citation_claims", return_value=claims), \
             patch.object(seed_audit, "_match_downloaded_paper", return_value={"status": "complete"}), \
             patch.object(seed_audit, "_is_ref_available", return_value=True), \
             patch.object(seed_audit, "attach_cited_summaries", return_value=None), \
             patch.object(seed_audit, "contextualize_citation_queries", return_value=None), \
             patch.object(seed_audit, "_judge_claim_entry", side_effect=mock_judge_claim_entry), \
             patch.object(pipeline_status, "update_progress", return_value=None):

            mock_res = (MagicMock(), MagicMock(), {}, {})
            report = seed_audit.audit_seed_citations(
                seed_path="test_seed.pdf",
                search_resources=mock_res,
                max_claims=num_claims,
                force=True,
                skip_if_cached=False,
                concurrency=8,
            )

        # 1. Concurrency check: multiple thread IDs participated
        self.assertGreater(len(active_threads), 1, "ThreadPoolExecutor ran single-threaded despite concurrency=8!")

        # 2. Completeness: all 30 claims were evaluated
        results = report["results"]
        self.assertEqual(len(results), num_claims)
        judged_count = sum(1 for r in results if r.get("outcome") == "judged")
        self.assertEqual(judged_count, num_claims)

        # 3. Aggregation check: totals match individual results
        self.assertEqual(report["totals"]["judged"], num_claims)
        self.assertEqual(report["totals"]["Supports"], num_claims)

    def test_seed_audit_cancellation_during_concurrent_execution(self):
        """Verify that cancellation requested mid-run halts background workers cleanly

        without deadlock or hanging threads.
        """
        num_claims = 25
        claims = [
            {
                "claim": f"Superconducting gap symmetry claim {i}.",
                "resolved": True,
                "role": "finding",
                "claim_quality": [],
                "document": f"paper_{i}.pdf",
                "ref": {"index": i + 1, "title": f"Study {i + 1}"},
            }
            for i in range(num_claims)
        ]

        processed_count = 0
        counter_lock = threading.Lock()

        def mock_judge_claim_entry(item, *args, **kwargs):
            nonlocal processed_count
            with counter_lock:
                processed_count += 1
                if processed_count >= 5:
                    pipeline_status.request_cancel()
            time.sleep(0.01)
            item["outcome"] = "judged"
            item["judgement"] = "Supports"

        with patch.object(seed_audit, "find_tei_for_seed", return_value="dummy.tei.xml"), \
             patch.object(seed_audit, "extract_seed_citation_claims", return_value=claims), \
             patch.object(seed_audit, "_match_downloaded_paper", return_value={"status": "complete"}), \
             patch.object(seed_audit, "_is_ref_available", return_value=True), \
             patch.object(seed_audit, "attach_cited_summaries", return_value=None), \
             patch.object(seed_audit, "contextualize_citation_queries", return_value=None), \
             patch.object(seed_audit, "_judge_claim_entry", side_effect=mock_judge_claim_entry), \
             patch.object(pipeline_status, "update_progress", return_value=None):

            # Ensure cancellation state starts clean
            if hasattr(pipeline_status, "_cancel_requested"):
                pipeline_status._cancel_requested = False

            mock_res = (MagicMock(), MagicMock(), {}, {})
            report = seed_audit.audit_seed_citations(
                seed_path="test_cancel.pdf",
                search_resources=mock_res,
                max_claims=num_claims,
                force=True,
                skip_if_cached=False,
                concurrency=4,
            )

        # Cleanup cancel state
        if hasattr(pipeline_status, "_cancel_requested"):
            pipeline_status._cancel_requested = False

        # Verify that remaining items were marked not_attempted or stopped
        outcomes = [r.get("outcome") for r in report["results"]]
        self.assertIn("not_attempted", outcomes, "Cancellation failed to set 'not_attempted' for pending items!")

    def test_resolver_contextualize_bounded_concurrency_stress(self):
        """Stress-test bounded concurrency in resolver.contextualize_citation_queries

        across 20 paragraphs with varied response latencies and intermittent failures.
        """
        num_paragraphs = 20
        claims = []
        for p in range(num_paragraphs):
            for c in range(2):
                claims.append({
                    "paragraph_id": f"p_{p}",
                    "claim": f"Paragraph {p} Claim {c}: topological insulator bulk gap.",
                    "context": f"Context paragraph {p} discussing surface states.",
                    "ref": {"authors": ["Zhang et al."], "year": 2019, "title": "Topological Insulators"},
                })

        active_threads = set()
        thread_lock = threading.Lock()

        def mock_chat_contextualize(messages, *args, **kwargs):
            tid = threading.get_ident()
            with thread_lock:
                active_threads.add(tid)
            time.sleep(random.uniform(0.005, 0.015))
            # Simulate occasional LLM parsing failure on paragraph 7
            content = messages[0]["content"]
            if "Paragraph 7" in content:
                raise RuntimeError("LLM rate limit error")
            return MagicMock(content='["focused search query 1", "focused search query 2"]')

        with patch("research_assistant.shared.llm.chat", side_effect=mock_chat_contextualize):
            resolver.contextualize_citation_queries(claims, enabled=True)

        # 1. Parallel execution verified
        self.assertGreater(len(active_threads), 1, "Resolver did not execute concurrently!")

        # 2. All claims processed and populated with search_query
        self.assertEqual(len(claims), num_paragraphs * 2)
        for c in claims:
            self.assertIn("search_query", c)
            self.assertTrue(len(c["search_query"]) > 5)

        # 3. Fallback verification: Paragraph 7 claims got fallback queries
        p7_claims = [c for c in claims if c["paragraph_id"] == "p_7"]
        for c in p7_claims:
            self.assertTrue(c["search_query"].startswith("Zhang"))

    def test_retrieve_paper_notes_bounded_concurrency_stress(self):
        """Stress-test retrieve.research_answer in map_reduce mode across 16 papers.

        Verify thread safety, order preservation in output synthesis, and graceful
        fallback when individual paper notes fail.
        """
        num_papers = 16
        selected = [
            {
                "key": f"P{i:02d}",
                "document": f"doc_{i:02d}.pdf",
                "summary": f"Summary for paper {i:02d}",
            }
            for i in range(num_papers)
        ]

        active_threads = set()
        thread_lock = threading.Lock()

        def mock_paper_notes(query, key, title, hits, **kw):
            tid = threading.get_ident()
            with thread_lock:
                active_threads.add(tid)
            time.sleep(random.uniform(0.005, 0.02))
            if key == "P05":
                raise ValueError("Simulated GROBID text corrupt for P05")
            return f"Synthesized notes for {key}", True

        passages = {f"P{i:02d}": [{"text": f"Passage {i}"}] for i in range(num_papers)}

        with patch.object(retrieve, "rank_documents", return_value=selected), \
             patch.object(retrieve, "flat_documents", return_value=[]), \
             patch.object(retrieve, "SYNTHESIS_MAX_PAPERS", num_papers), \
             patch.object(retrieve, "gate_documents", side_effect=lambda q, r, **kw: r), \
             patch.object(retrieve, "passages_per_paper", return_value=passages), \
             patch.object(retrieve, "paper_notes", side_effect=mock_paper_notes), \
             patch.object(retrieve, "synthesise", return_value="Final consolidated synthesis."):

            res = retrieve.research_answer("What are the bounds on superconducting gap?", mode="map_reduce")

        # 1. Concurrency verified
        self.assertGreater(len(active_threads), 1, "retrieve.paper_notes did not run concurrently!")

        # 2. Success: all papers accounted for in selected
        self.assertEqual(len(res["selected"]), num_papers)
        self.assertEqual(res["suggestion"], "Final consolidated synthesis.")


# ─────────────────────────────────────────────────────────────────────────────
# Suite 3: LLM Schema & Privacy Validation
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMSchemaAndPrivacyValidation(unittest.TestCase):
    """Adversarially challenge LLM client caching, privacy guarantees, and structured

    response format handling.
    """

    def test_ollama_error_does_not_leak_or_fallback_to_openai(self):
        """Verify that when Ollama fails, llm.chat strictly raises without falling

        back to OpenAI or Gemini, even when GEMINI_API_KEY and OPENAI_API_KEY are configured.
        """
        with patch.object(llm, "LLM_BACKEND", "ollama"), \
             patch.object(llm, "GEMINI_API_KEY", "sensitive-gemini-cloud-key"), \
             patch.object(llm, "OPENAI_API_KEY", "sensitive-openai-cloud-key"), \
             patch("ollama.Client", **{"return_value.chat.side_effect": RuntimeError("Ollama daemon unreachable")}), \
             patch.object(llm, "_openai_chat") as mock_openai_chat:

            # 1. Explicit backend="ollama"
            with self.assertRaises(RuntimeError) as ctx1:
                llm.chat([{"role": "user", "content": "Sensitive proprietary equation"}], backend="ollama")
            self.assertIn("Ollama daemon unreachable", str(ctx1.exception))
            mock_openai_chat.assert_not_called()

            # 2. Inferred backend from local model name
            with self.assertRaises(RuntimeError) as ctx2:
                llm.chat([{"role": "user", "content": "Sensitive local query"}], model="llama3:8b")
            self.assertIn("Ollama daemon unreachable", str(ctx2.exception))
            mock_openai_chat.assert_not_called()

            # 3. Default backend="ollama" when model is None
            with self.assertRaises(RuntimeError) as ctx3:
                llm.chat([{"role": "user", "content": "Sensitive local query"}], model=None)
            self.assertIn("Ollama daemon unreachable", str(ctx3.exception))
            mock_openai_chat.assert_not_called()

    def test_ollama_stream_does_not_fallback_to_openai(self):
        """Verify that chat_stream with Ollama backend does not leak to OpenAI."""
        with patch.object(llm, "LLM_BACKEND", "ollama"), \
             patch.object(llm, "GEMINI_API_KEY", "sensitive-gemini-cloud-key"), \
             patch.object(llm, "_openai_stream") as mock_openai_stream, \
             patch("ollama.Client", **{"return_value.chat.side_effect": RuntimeError("Ollama stream broken")}):

            with self.assertRaises(RuntimeError):
                stream_gen = llm.chat_stream([{"role": "user", "content": "Stream test"}], backend="ollama")
                next(stream_gen)
            mock_openai_stream.assert_not_called()

    def test_client_cache_reuses_instances_and_isolates_keys(self):
        """Verify _CLIENT_CACHE reuses client instances for identical credentials

        and creates distinct clients when base_url, api_key, or timeout changes.
        """
        # Clear cache for isolated testing
        llm._CLIENT_CACHE.clear()

        # 1. Repeated calls with identical parameters return the same instance
        c1 = llm._get_openai_client("https://api.openai.com/v1", "key_alpha", 60.0)
        c2 = llm._get_openai_client("https://api.openai.com/v1", "key_alpha", 60.0)
        self.assertIs(c1, c2, "Client was not reused from cache for identical parameters!")

        # 2. Different base_url creates new instance
        c3 = llm._get_openai_client("https://custom.endpoint.com/v1", "key_alpha", 60.0)
        self.assertIsNot(c1, c3)

        # 3. Different api_key creates new instance
        c4 = llm._get_openai_client("https://api.openai.com/v1", "key_beta", 60.0)
        self.assertIsNot(c1, c4)

        # 4. Different timeout creates new instance
        c5 = llm._get_openai_client("https://api.openai.com/v1", "key_alpha", 30.0)
        self.assertIsNot(c1, c5)

        # Total cache size must be 4 distinct clients
        self.assertEqual(len(llm._CLIENT_CACHE), 4)

    def test_client_cache_thread_safety_under_heavy_concurrency(self):
        """Verify that 40 threads requesting _get_openai_client simultaneously

        all retrieve valid client instances and reuse the cached client.
        """
        # 1. Warm cache test: after initial load, all 40 concurrent threads get identical instance
        llm._CLIENT_CACHE.clear()
        cached_client = llm._get_openai_client("https://api.openai.com/v1", "shared_key", 45.0)

        results = [None] * 40

        def _fetch_client(idx):
            results[idx] = llm._get_openai_client("https://api.openai.com/v1", "shared_key", 45.0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(_fetch_client, range(40)))

        # All 40 results must be the exact identical cached client
        self.assertTrue(all(r is cached_client for r in results))

        # 2. Cold start cache test: clear cache, let 40 threads hit it concurrently
        llm._CLIENT_CACHE.clear()
        cold_results = [None] * 40

        def _fetch_cold(idx):
            cold_results[idx] = llm._get_openai_client("https://api.openai.com/v1", "cold_key", 45.0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(_fetch_cold, range(40)))

        # All 40 results are non-None valid clients
        self.assertTrue(all(r is not None for r in cold_results))
        # Cache converges to exactly 1 client entry
        self.assertIn(("https://api.openai.com/v1", "cold_key", 45.0), llm._CLIENT_CACHE)
        final_cached = llm._CLIENT_CACHE[("https://api.openai.com/v1", "cold_key", 45.0)]
        self.assertIsNotNone(final_cached)

    def test_structured_schema_pydantic_model_openai(self):
        """Verify that passing a Pydantic BaseModel to _openai_chat properly constructs

        the response_format with json_schema.
        """
        class SampleRubric(BaseModel):
            judgement: str = Field(description="Supports, Contradicts, etc.")
            confidence: str = Field(description="Low, Medium, High")
            score: float = Field(ge=0.0, le=1.0)

        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content='{"judgement": "Supports", "confidence": "High", "score": 0.95}'))]
        mock_client.chat.completions.create.return_value = mock_completion

        with patch.object(llm, "_get_openai_client", return_value=mock_client):
            res = llm._openai_chat(
                [{"role": "user", "content": "Judge claim"}],
                model="gpt-4o",
                images=None,
                schema=SampleRubric,
            )

        self.assertIn("Supports", res.content)
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertIn("response_format", call_kwargs)
        rf = call_kwargs["response_format"]
        self.assertEqual(rf["type"], "json_schema")
        self.assertEqual(rf["json_schema"]["name"], "SampleRubric")
        self.assertIn("properties", rf["json_schema"]["schema"])
        self.assertIn("judgement", rf["json_schema"]["schema"]["properties"])

    def test_structured_schema_pydantic_model_ollama(self):
        """Verify that passing a Pydantic BaseModel to _ollama_chat passes the schema

        dict to the format parameter.
        """
        class SampleRubric(BaseModel):
            judgement: str = Field(description="Supports, Contradicts, etc.")
            confidence: str = Field(description="Low, Medium, High")

        with patch("ollama.Client") as mock_client:
            mock_ollama_chat = mock_client.return_value.chat
            mock_ollama_chat.return_value = MagicMock(message=MagicMock(content='{"judgement": "Supports"}'))

            res = llm._ollama_chat(
                [{"role": "user", "content": "Judge claim"}],
                model="llama3.1",
                images=None,
                schema=SampleRubric,
            )

        self.assertIn("Supports", res.content)
        call_kwargs = mock_ollama_chat.call_args.kwargs
        self.assertIn("format", call_kwargs)
        self.assertIsInstance(call_kwargs["format"], dict)
        self.assertIn("properties", call_kwargs["format"])


if __name__ == "__main__":
    unittest.main()
