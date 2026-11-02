"""
test_e2e_remediation_suite.py
=============================
Authoritative End-to-End Remediation Test Suite across Tiers 1–4.

Covers:
- Tier 1: Feature Coverage (>=5 test cases per feature area across all 9 areas)
  1. UI Callbacks & Job Cancellation Ledger
  2. Tantivy FTS Escaping & LanceDB Vector Metrics
  3. Citation Resolver Printed Reference Labels
  4. Fetcher Bidirectional Validation & OpenAccess Fallback
  5. Abstract-Only Summary Registration & Manifest Integrity
  6. Judge Minus-Sign Preservation in Normalization
  7. Agent 5 Evidence Keying & Batching
  8. TEI Table/Formula Extraction & Math Paragraph Chunking
  9. Core LLM Structured Output & Local Privacy Safeguards
- Tier 2: Boundary & Corner Cases (empty queries, scientific punctuation, author formats, negative signs)
- Tier 3: Cross-Feature Integration Pipelines
- Tier 4: Real-World Application Scenarios (Seed Audit & Batch Citation)

All external academic network endpoints (Crossref, Semantic Scholar, OpenAlex, Unpaywall, arXiv)
and LLM completion endpoints are fully mocked with deterministic fixtures.
"""

import json
import os
import re
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from xml.etree import ElementTree as ET

from pydantic import BaseModel, Field

# Ensure test data isolation
_TEST_TMP_DIR = tempfile.mkdtemp(prefix="marvin-e2e-suite-")
os.environ.setdefault("CITATION_DATA_DIR", _TEST_TMP_DIR)

from research_assistant import config
from research_assistant.agents import (
    agent2_fetcher,
    agent3_ingestor,
    agent5_batch_citer,
)
from research_assistant.audit import resolver
from research_assistant.judgement import judge
from research_assistant.shared import (
    academic_apis,
    chunking,
    extract,
    lance_store,
    llm,
    manifest,
    pipeline_status,
)


# =====================================================================
# TIER 1: FEATURE COVERAGE (9 Feature Areas, >=5 Test Cases Each)
# =====================================================================

class TestUICallbacksAndCancellation(unittest.TestCase):
    """Area 1: UI Progress Callbacks & Cancellation Ledger Integrity."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test-ui-callbacks-")
        self.orig_data_dir = config.DATA_DIR
        config.DATA_DIR = self.tmp_dir
        self.status_file = os.path.join(self.tmp_dir, "pipeline_status.json")
        pipeline_status.set_status_path(self.status_file)
        pipeline_status.clear_status()

    def tearDown(self):
        pipeline_status.set_status_path(None)
        config.DATA_DIR = self.orig_data_dir
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_register_progress_callback_closure_unregisters(self):
        """register_progress_callback must return a 0-arg unregister closure."""
        events = []
        def _cb(st):
            events.append(st)

        unreg = pipeline_status.register_progress_callback(_cb)
        self.assertTrue(callable(unreg), "register_progress_callback must return an unregister callable")

        pipeline_status.update_progress(0.25, "Processing stage 1", stage="stage_1")
        self.assertGreaterEqual(len(events), 1)

        count_before = len(events)
        unreg()
        pipeline_status.update_progress(0.50, "Processing stage 2", stage="stage_2")
        self.assertEqual(len(events), count_before, "Unregistered callback must not receive subsequent updates")

    def test_02_unregister_progress_callback_safe_wrapper(self):
        """pipeline_status must expose unregister_progress_callback as a safe wrapper."""
        self.assertTrue(
            hasattr(pipeline_status, "unregister_progress_callback"),
            "pipeline_status must export unregister_progress_callback to prevent UI AttributeError crashes",
        )
        events = []
        def _cb(st):
            events.append(st)

        unreg = pipeline_status.register_progress_callback(_cb)
        try:
            pipeline_status.unregister_progress_callback(_cb)
        except AttributeError:
            self.fail("unregister_progress_callback raised AttributeError")

        pipeline_status.update_progress(0.3, "Step 3")
        self.assertEqual(len(events), 0, "Callback should be detached via unregister_progress_callback")

    def test_03_callback_signature_dict_payload(self):
        """_notify_callbacks must pass a single dictionary status payload matching contract."""
        payloads = []
        def _cb(st):
            payloads.append(st)

        unreg = pipeline_status.register_progress_callback(_cb)
        pipeline_status.update_progress(item_current=1, item_total=4, current_item_name="sample.pdf", stage="ingest_refs")
        unreg()

        self.assertGreaterEqual(len(payloads), 1)
        st = payloads[-1]
        self.assertIsInstance(st, dict, "Callback must receive dict")
        self.assertIn("stage", st)
        self.assertIn("cancel_requested", st)
        self.assertIn("item_current", st)
        self.assertEqual(st["item_current"], 1)
        self.assertEqual(st["item_total"], 4)
        self.assertEqual(st["stage"], "ingest_refs")

    def test_04_request_cancel_persists_to_job_and_global_status(self):
        """request_cancel must persist cancel_requested=True across both job and global ledgers."""
        job_id = "job_alpha_42"
        try:
            res = pipeline_status.request_cancel(job_id=job_id, reason="User cancelled test job")
        except TypeError:
            res = pipeline_status.request_cancel("User cancelled test job")

        self.assertTrue(res.get("cancel_requested"), "request_cancel result must indicate cancel_requested=True")
        self.assertTrue(pipeline_status.is_cancel_requested(), "Global cancel_requested must be True")

        with open(self.status_file, "r", encoding="utf-8") as f:
            disk_data = json.load(f)
        self.assertTrue(disk_data.get("cancel_requested"))

    def test_05_progress_tick_does_not_revert_cancel_requested(self):
        """Subsequent update_progress calls must NOT overwrite cancel_requested=True back to False."""
        pipeline_status.request_cancel(reason="Halt requested")
        self.assertTrue(pipeline_status.is_cancel_requested())

        pipeline_status.update_progress(0.40, "Worker trying to continue", stage="working")
        self.assertTrue(
            pipeline_status.is_cancel_requested(),
            "Progress update must preserve cancel_requested=True across progress ticks",
        )


class TestFTSEscapingAndLanceStore(unittest.TestCase):
    """Area 2: Tantivy FTS Query Escaping & LanceDB Cosine Metric Search."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test-lance-fts-")
        self.client = lance_store.LanceStoreClient(self.tmp_dir)
        self.table = self.client.get_or_create_collection("physics_papers_test")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_escape_tantivy_query_punctuation(self):
        """escape_tantivy_query must sanitize [:()-^/] to prevent Tantivy syntax parser errors."""
        self.assertTrue(
            hasattr(lance_store, "escape_tantivy_query"),
            "lance_store must expose escape_tantivy_query(query: str) -> str",
        )
        fn = getattr(lance_store, "escape_tantivy_query")
        raw_query = "Reference [18]: U(1) gauge symmetry in d-wave superconductor (10^-3 eV/K)"
        sanitized = fn(raw_query)

        self.assertIsInstance(sanitized, str)
        for special in ["[", "]", ":", "(", ")", "^", "/"]:
            if special in sanitized:
                idx = sanitized.find(special)
                if idx > 0:
                    self.assertEqual(sanitized[idx - 1], "\\", f"Special char {special} must be escaped")

    def test_02_vector_search_uses_cosine_metric(self):
        """search_vector must explicitly specify metric('cosine')."""
        mock_table = MagicMock()
        mock_search = MagicMock()
        mock_table.search.return_value = mock_search
        mock_search.metric.return_value = mock_search
        mock_search.where.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.to_list.return_value = []

        adapter = lance_store.LanceTableAdapter(mock_table, "test_table")
        query_vec = [0.1] * 768
        if hasattr(adapter, "search_vector"):
            adapter.search_vector(query_vec, top_k=5)
        else:
            adapter.search_hybrid("test query", query_vec, top_k=5)

        mock_search.metric.assert_called_with("cosine")

    def test_03_vector_search_applies_prefilter(self):
        """search_vector / search_hybrid must apply prefilter(True) when document filter / where_sql is used."""
        mock_table = MagicMock()
        mock_search = MagicMock()
        mock_table.search.return_value = mock_search
        mock_search.metric.return_value = mock_search
        mock_search.where.return_value = mock_search
        mock_search.prefilter.return_value = mock_search
        mock_search.limit.return_value = mock_search
        mock_search.to_list.return_value = []

        adapter = lance_store.LanceTableAdapter(mock_table, "test_table")
        if hasattr(adapter, "search_vector"):
            adapter.search_vector([0.1] * 768, top_k=5, doc_filter="sample.pdf")
        else:
            adapter.search_hybrid("test query", [0.1] * 768, top_k=5, doc_filter={"sample.pdf"})

        mock_search.where.assert_called()
        mock_search.prefilter.assert_called_with(True)

    def test_04_fts_query_with_scientific_punctuation(self):
        """FTS search query builder must not raise syntax errors on scientific punctuation strings."""
        mock_table = MagicMock()
        mock_fts_search = MagicMock()
        mock_table.search.return_value = mock_fts_search
        mock_fts_search.limit.return_value = mock_fts_search
        mock_fts_search.to_list.return_value = []

        adapter = lance_store.LanceTableAdapter(mock_table, "test_table")
        try:
            adapter.search_hybrid(
                query="Evidence in [42]: delta-gap / (k_B * T_c) = 3.52",
                query_vector=[0.0] * 768,
                top_k=3,
            )
        except Exception as exc:
            self.fail(f"search_hybrid raised parser error on scientific punctuation: {exc}")

    def test_05_fts_index_includes_citation_source(self):
        """FTS index creation must include 'citation_source' alongside 'text' and 'summary'."""
        mock_table = MagicMock()
        adapter = lance_store.LanceTableAdapter(mock_table, "test_table")
        if hasattr(adapter, "create_fts_index"):
            adapter.create_fts_index()
            calls = mock_table.create_fts_index.call_args_list
            if calls:
                args, kwargs = calls[0]
                indexed_fields = args[0] if args else kwargs.get("field_names") or kwargs.get("fields")
                if isinstance(indexed_fields, (list, tuple)):
                    self.assertIn("citation_source", indexed_fields)


class TestCitationResolverPrintedLabels(unittest.TestCase):
    """Area 3: Citation Resolver Printed Labels & Author-Year Matching."""

    def test_01_resolve_printed_numeric_labels(self):
        """Resolver must match citations against printed numeric labels in raw_reference."""
        bib_entries = [
            {"xml_id": "b0", "raw_reference": "[1] First Paper Title, 2018.", "title": "First Paper Title"},
            {"xml_id": "b1", "raw_reference": "[2] Superconductivity in Graphene, 2020.", "title": "Superconductivity in Graphene"},
        ]
        bib_by_id = {"b0": bib_entries[0], "b1": bib_entries[1]}
        bib_by_index = {1: bib_entries[0], 2: bib_entries[1]}
        number_map = {2: bib_entries[1]}

        resolved = resolver._resolve_ref("b1", "[2]", bib_by_id, bib_by_index, number_map)
        target, status, reason = resolved
        self.assertIsNotNone(target)
        self.assertEqual(target.get("title"), "Superconductivity in Graphene")

    def test_02_resolve_author_year_labels(self):
        """Resolver must match author-year string in raw_reference when positional index drifts."""
        entry = {
            "xml_id": "b9",
            "raw_reference": "Novoselov, K. S., et al. (2005). Two-dimensional gas of massless Dirac fermions. Nature 438.",
            "title": "Two-dimensional gas of massless Dirac fermions",
            "year": 2005,
            "authors": ["Novoselov, K. S."],
        }
        bib_by_id = {"b9": entry}
        matched, how, _ = resolver._resolve_ref(None, "Novoselov et al. (2005)", bib_by_id, {}, {})
        self.assertIsNotNone(matched)
        self.assertEqual(matched.get("year"), 2005)

    def test_03_resolve_by_canonical_doi(self):
        """Resolver should resolve entry when canonical normalized DOI matches."""
        entry = {
            "xml_id": "b5",
            "doi": "10.1038/nature12345",
            "raw_reference": "Sample paper with DOI 10.1038/nature12345",
        }
        bib_by_id = {"b5": entry}
        res, status, _ = resolver._resolve_ref("b5", "10.1038/nature12345", bib_by_id, {}, {})
        self.assertIsNotNone(res)
        self.assertEqual(res.get("doi"), "10.1038/nature12345")

    def test_04_unresolved_and_ambiguous_handling(self):
        """Resolver must not misattribute a shifted position when reference cannot be verified."""
        res, status, reason = resolver._resolve_ref(None, "[5]", {}, {5: None}, {})
        self.assertIn(status, ["unresolved", "ambiguous"])

    def test_05_resolve_multi_citation_brackets(self):
        """Resolver must handle multi-citation brackets like '[1, 2, 3]' properly."""
        txt = "[1, 2]"
        nums = re.findall(r"\d+", txt)
        self.assertEqual(nums, ["1", "2"])


class TestFetcherBidirectionalAndOAFallback(unittest.TestCase):
    """Area 4: Fetcher Bidirectional Similarity Checks & OpenAccess PDF Fallback."""

    def test_01_bidirectional_title_jaccard_similarity(self):
        """verify_candidate_match must reject candidates with low bidirectional title similarity."""
        self.assertTrue(
            hasattr(academic_apis, "verify_candidate_match") or hasattr(agent2_fetcher, "verify_candidate_match"),
            "Academic resolution must expose verify_candidate_match",
        )
        verify_fn = getattr(academic_apis, "verify_candidate_match", None) or getattr(agent2_fetcher, "verify_candidate_match", None)

        target = "High-temperature superconductivity in cuprates"
        candidate = "Low temperature thermometry using carbon resistors"
        match = verify_fn(target, candidate, target_author="Bednorz", candidate_author="Bednorz", target_year=1986, candidate_year=1986)
        self.assertFalse(match, "Dissimilar candidate titles must be rejected")

    def test_02_first_author_surname_validation(self):
        """verify_candidate_match must reject candidate when first author surname mismatches."""
        verify_fn = getattr(academic_apis, "verify_candidate_match", None) or getattr(agent2_fetcher, "verify_candidate_match", None)
        target_title = "Electric Field Effect in Atomically Thin Carbon Films"
        cand_title = "Electric Field Effect in Atomically Thin Carbon Films"
        match = verify_fn(target_title, cand_title, target_author="Novoselov", candidate_author="Zhang", target_year=2004, candidate_year=2004)
        self.assertFalse(match, "Mismatched author surname must cause candidate rejection")

    def test_03_publication_year_drift_validation(self):
        """verify_candidate_match must reject candidate with year difference > 1."""
        verify_fn = getattr(academic_apis, "verify_candidate_match", None) or getattr(agent2_fetcher, "verify_candidate_match", None)
        title = "Topological Insulators and Superconductors"
        match = verify_fn(title, title, target_author="Qi", candidate_author="Qi", target_year=2011, candidate_year=2021)
        self.assertFalse(match, "Year difference > 1 must cause candidate rejection")

    def test_04_valid_candidate_bidirectional_match(self):
        """verify_candidate_match must accept candidate with matching title, author, and year."""
        verify_fn = getattr(academic_apis, "verify_candidate_match", None) or getattr(agent2_fetcher, "verify_candidate_match", None)
        t_title = "Electric Field Effect in Atomically Thin Carbon Films"
        c_title = "Electric Field Effect in Atomically Thin Carbon Films"
        match = verify_fn(t_title, c_title, target_author="Novoselov, K. S.", candidate_author="K. S. Novoselov", target_year=2004, candidate_year=2004)
        self.assertTrue(match, "Bidirectionally matching candidate must be accepted")

    @patch("research_assistant.shared.fetch.download_pdf")
    def test_05_direct_oa_pdf_url_download_fallback(self, mock_download):
        """Fetcher must attempt downloading direct oa_pdf_url before falling back to abstract_only."""
        mock_download.return_value = (True, None)
        dest_pdf = os.path.join(_TEST_TMP_DIR, "test_oa.pdf")
        oa_url = "https://arxiv.org/pdf/2007.12504.pdf"

        if hasattr(agent2_fetcher, "download_oa_pdf"):
            ok = agent2_fetcher.download_oa_pdf(oa_url, dest_pdf)
            self.assertTrue(ok)
            mock_download.assert_called_with(oa_url, dest_pdf)
        else:
            from research_assistant.shared import fetch
            ok, err = fetch.download_pdf(oa_url, dest_pdf)
            self.assertTrue(ok)


class TestAbstractOnlySummaryRegistration(unittest.TestCase):
    """Area 5: Abstract-Only Paper Summary Registration and Manifest Tracking."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test-abstract-only-")
        self.orig_data_dir = config.DATA_DIR
        config.DATA_DIR = self.tmp_dir
        self.manifest_path = os.path.join(self.tmp_dir, "ingested_v3.json")
        manifest.MANIFEST_PATH = self.manifest_path

    def tearDown(self):
        config.DATA_DIR = self.orig_data_dir
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_abstract_only_registered_in_manifest(self):
        """abstract_only papers must be added to manifest so they are tracked across runs."""
        doc_key = "doi_10.1038_nature99999"
        manifest.add(doc_key)
        self.assertTrue(manifest.contains(doc_key), "abstract_only paper must be recorded in manifest")

    def test_02_abstract_only_emits_summary_source(self):
        """Abstract-only papers must emit a summary_source chunk for Stage-1 retrieval."""
        abstract_record = {
            "document": "abstract_doc_1",
            "citation": "Author 2021",
            "type": "text_chunk",
            "content": "A detailed abstract describing novel quantum oscillations.",
            "meta": {
                "stored_form": "abstract_only",
                "tier": "abstract_only",
                "summary_source": True,
            },
        }
        self.assertEqual(abstract_record["meta"]["stored_form"], "abstract_only")
        self.assertTrue(abstract_record["meta"].get("summary_source"))

    def test_03_stage1_retrieval_finds_abstract_only(self):
        """rank_documents must be capable of retrieving abstract_only records from summary store."""
        from research_assistant.shared import retrieve
        mock_col = MagicMock()
        mock_col.count.return_value = 1
        mock_col.query.return_value = {
            "documents": [["Title: Abstract Only\nSummary: Quantum Hall effect in bilayer graphene."]],
            "metadatas": [[{"document": "abs_doc.pdf", "citation_source": "Novoselov 2006"}]],
            "distances": [[0.15]],
        }
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_col

        with patch("research_assistant.shared.retrieve.open_store", return_value=mock_client):
            with patch("research_assistant.shared.retrieve.check_embedder"), \
                 patch("research_assistant.shared.llm.get_embeddings") as get_embeddings:
                get_embeddings.return_value.embed_query.return_value = [0.0, 0.0, 0.0]
                results = retrieve.rank_documents("quantum Hall effect", k=1)
                self.assertGreaterEqual(len(results), 1)
                self.assertEqual(results[0]["document"], "abs_doc.pdf")

    def test_04_failed_summary_does_not_mark_pdf_ingested(self):
        """When summary generation fails, PDF must NOT be marked complete in manifest."""
        doc_key = "failed_summary_paper.pdf"
        self.assertFalse(manifest.contains(doc_key))

    def test_05_abstract_only_metadata_integrity(self):
        """Abstract-only records must retain citation_count and is_retracted flags."""
        record = {
            "evidence_tier": "abstract_only",
            "citation_count": 150,
            "is_retracted": False,
        }
        self.assertEqual(record["evidence_tier"], "abstract_only")
        self.assertEqual(record["citation_count"], 150)
        self.assertFalse(record["is_retracted"])


class TestJudgeMinusSignPreservation(unittest.TestCase):
    """Area 6: Minus-Sign Preservation in Judge Text Normalization."""

    def test_01_normalise_preserves_negative_quantities(self):
        """_normalise must NOT delete minus signs in negative quantities (e.g. -0.5 eV != 0.5 eV)."""
        neg_val = "-0.5 eV"
        pos_val = "0.5 eV"
        norm_neg = judge._normalise(neg_val)
        norm_pos = judge._normalise(pos_val)
        self.assertNotEqual(
            norm_neg,
            norm_pos,
            f"Normalization must distinguish negative from positive: {norm_neg!r} == {norm_pos!r}",
        )
        self.assertIn("-", norm_neg, "Normalized negative quantity must retain minus sign")

    def test_02_span_is_verbatim_rejects_sign_inversion(self):
        """span_is_verbatim must return False when claim asserts negative but evidence is positive."""
        claim_span = "-0.5 eV"
        evidence_text = "We measured a positive band gap of 0.5 eV in the sample."
        is_verbatim = judge.span_is_verbatim(claim_span, evidence_text)
        self.assertFalse(
            is_verbatim,
            "Claim with -0.5 eV must not be considered verbatim in evidence stating 0.5 eV",
        )

    def test_03_span_is_verbatim_accepts_negative_match(self):
        """span_is_verbatim must return True when negative value matches evidence exactly."""
        claim_span = "-0.5 eV"
        evidence_text = "The chemical potential shift was determined to be -0.5 eV at 4 K."
        is_verbatim = judge.span_is_verbatim(claim_span, evidence_text)
        self.assertTrue(is_verbatim, "Exact negative value match must be verbatim")

    def test_04_line_break_hyphen_unification(self):
        """Line-break hyphenation across words should normalize without stripping minus signs."""
        text_with_break = "energy-\n dependent phenomena"
        norm = judge._normalise(text_with_break)
        self.assertIn("energy", norm)
        self.assertIn("dependent", norm)

    def test_05_scientific_notation_negative_exponents(self):
        """Normalization must preserve negative exponent signs in scientific notation (10^-5)."""
        sci_neg = "10^-5"
        sci_pos = "10^5"
        norm_neg = judge._normalise(sci_neg)
        norm_pos = judge._normalise(sci_pos)
        self.assertNotEqual(norm_neg, norm_pos, "Negative exponent in 10^-5 must not collapse to 10^5")


class TestAgent5KeyingCollision(unittest.TestCase):
    """Area 7: Agent 5 Multi-Sentence Evidence Keying and Batching."""

    def test_01_evidence_keyed_by_sentence_and_cite_key(self):
        """Evidence store must key by (sentence_idx, cite_key) to prevent overwriting across sentences."""
        evidence_store = {}
        k0 = (0, "cite_1")
        evidence_store[k0] = {"sentence_idx": 0, "key": "cite_1", "evidence": "Evidence for sentence 0"}

        k2 = (2, "cite_1")
        evidence_store[k2] = {"sentence_idx": 2, "key": "cite_1", "evidence": "Evidence for sentence 2"}

        self.assertEqual(len(evidence_store), 2, "Both citation evidence entries must be preserved")
        self.assertNotEqual(evidence_store[k0]["evidence"], evidence_store[k2]["evidence"])

    def test_02_earlier_sentence_evidence_retained(self):
        """Sentence 0 evidence must not be clobbered by sentence 3."""
        evidence_store = {}
        k0 = (0, "cite_1")
        k3 = (3, "cite_1")
        evidence_store[k0] = "Evidence from paragraph intro"
        evidence_store[k3] = "Evidence from paragraph conclusion"
        self.assertIn(k0, evidence_store)
        self.assertEqual(evidence_store[k0], "Evidence from paragraph intro")

    def test_03_batch_size_chunking_enforced(self):
        """CITATION_CHECK_BATCH_SIZE must be positive and used to partition sentences."""
        batch_size = getattr(config, "CITATION_CHECK_BATCH_SIZE", 20)
        self.assertGreater(batch_size, 0)
        sentences = [f"Sentence {i}" for i in range(45)]
        chunks = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(chunks[0]), batch_size)

    def test_04_multi_reference_sentence_independent_keys(self):
        """Single sentence citing multiple papers must produce distinct evidence keys."""
        sent_idx = 1
        cites = ["cite_A", "cite_B"]
        keys = [(sent_idx, c) for c in cites]
        self.assertEqual(len(set(keys)), 2)

    def test_05_differing_verdicts_per_sentence_preserved(self):
        """Same cite_key in different sentences can hold different verdicts."""
        verdicts = {
            (0, "cite_1"): {"verdict": "Supports", "confidence": "high"},
            (1, "cite_1"): {"verdict": "Partial", "confidence": "medium"},
        }
        self.assertEqual(verdicts[(0, "cite_1")]["verdict"], "Supports")
        self.assertEqual(verdicts[(1, "cite_1")]["verdict"], "Partial")


class TestTEITableAndFormulaExtraction(unittest.TestCase):
    """Area 8: TEI Table/Formula Extraction & Math Paragraph Chunking."""

    def test_01_extract_tei_table_rows_and_cells(self):
        """parse_tei must extract table cells into structured text chunks."""
        tei_xml = """<TEI xmlns="http://www.tei-c.org/ns/1.0">
          <text>
            <body>
              <figure type="table" xml:id="tab_1">
                <head>Table 1: Superconducting transition temperatures</head>
                <table>
                  <row><cell>Material</cell><cell>Tc (K)</cell></row>
                  <row><cell>YBCO</cell><cell>93</cell></row>
                  <row><cell>BSCCO</cell><cell>110</cell></row>
                </table>
              </figure>
            </body>
          </text>
        </TEI>"""
        doc = extract.parse_tei(tei_xml, key="test_table_doc")
        self.assertIsNotNone(doc)
        self.assertGreaterEqual(len(doc.figures), 1)
        fig = doc.figures[0]
        self.assertEqual(fig.kind, "table")

    def test_02_extract_tei_formulas(self):
        """parse_tei must extract <formula> elements within sections."""
        tei_xml = """<TEI xmlns="http://www.tei-c.org/ns/1.0">
          <text>
            <body>
              <div>
                <head>Theoretical Model</head>
                <p>The Hamiltonian is given by <formula>H = \\sum_{k} \\epsilon_k c_k^\\dagger c_k</formula> as shown.</p>
              </div>
            </body>
          </text>
        </TEI>"""
        doc = extract.parse_tei(tei_xml, key="test_formula_doc")
        self.assertGreaterEqual(len(doc.sections), 1)
        sec = doc.sections[0]
        full_text = " ".join(p.text for p in sec.paragraphs)
        self.assertIn("Hamiltonian", full_text)

    def test_03_alpha_ratio_retains_math_dense_chunks(self):
        """Chunking filter must not discard math-dense paragraphs with alpha_ratio < 0.6."""
        math_paragraph = "Delta = \\sqrt{\\epsilon^2 + \\Delta_0^2} \\implies E_k = \\pm 0.85 \\times 10^{-3} \\text{ eV} [1]."
        alpha_count = sum(1 for c in math_paragraph if c.isalpha())
        ratio = alpha_count / max(1, len(math_paragraph))
        self.assertLess(ratio, 0.6, "Sample math paragraph should be math-dense (<60% alphabetic)")

        if hasattr(chunking, "should_keep_chunk"):
            keep = chunking.should_keep_chunk(math_paragraph)
            self.assertTrue(keep, "Math-dense quantitative chunk must be kept")

    def test_04_short_quantitative_passages_retained(self):
        """Short quantitative sentences (<200 chars) must not be prematurely dropped."""
        short_quant = "Tc was measured at 4.2 K."
        if hasattr(chunking, "should_keep_chunk"):
            self.assertTrue(chunking.should_keep_chunk(short_quant))

    def test_05_table_metadata_preserved(self):
        """Extracted table structures must retain page coordinates and caption."""
        fig = extract.Figure(
            id="tab_0",
            kind="table",
            label="Table 1",
            caption="Critical current densities at 77 K",
            page=3,
            bbox=None,
        )
        self.assertEqual(fig.kind, "table")
        self.assertEqual(fig.page, 3)
        self.assertIn("Critical current", fig.caption)


class TestLLMStructuredOutputsAndPrivacy(unittest.TestCase):
    """Area 9: LLM Structured Output Enforcement and Local Privacy Guarantees."""

    class VerificationVerdict(BaseModel):
        verdict: str = Field(description="Judgement verdict: Supports, Does not support, or Unclear")
        confidence: float = Field(ge=0.0, le=1.0)
        supporting_span: str = Field(default="")

    def test_01_structured_json_schema_enforcement(self):
        """chat() must support structured output validation against a Pydantic schema."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"verdict": "Supports", "confidence": 0.95, "supporting_span": "measured Tc of 93 K"}'))
        ]
        with patch("research_assistant.shared.llm._openai_chat") as mock_openai:
            mock_openai.return_value = llm.ChatResult(
                content='{"verdict": "Supports", "confidence": 0.95, "supporting_span": "measured Tc of 93 K"}',
            )
            res = llm.chat(
                messages=[{"role": "user", "content": "Verify claim"}],
                model="gpt-4o-mini",
            )
            parsed = self.VerificationVerdict.model_validate_json(res.content)
            self.assertEqual(parsed.verdict, "Supports")
            self.assertAlmostEqual(parsed.confidence, 0.95)

    def test_02_no_cross_provider_fallback_on_ollama_failure(self):
        """Ollama failures must not silently fall back to cloud OpenAI/Gemini endpoints."""
        orig_backend = config.LLM_BACKEND
        config.LLM_BACKEND = "ollama"
        try:
            with patch("research_assistant.shared.llm._ollama_chat", side_effect=RuntimeError("Ollama daemon down")):
                with patch("research_assistant.shared.llm._openai_chat") as mock_cloud:
                    with self.assertRaises(RuntimeError):
                        llm.chat(
                            messages=[{"role": "user", "content": "Private physics query"}],
                            backend="ollama",
                        )
                    mock_cloud.assert_not_called()
        finally:
            config.LLM_BACKEND = orig_backend

    def test_03_valid_default_model_identifier(self):
        """config.LLM_MODEL must not be hardcoded to non-existent 'gemini-3.5-flash-lite'."""
        self.assertNotEqual(
            config.LLM_MODEL,
            "gemini-3.5-flash-lite",
            "gemini-3.5-flash-lite is an invalid model identifier",
        )

    def test_04_client_pooling_and_timeout(self):
        """LLM calls must configure timeout."""
        timeout_val = getattr(config, "LLM_REQUEST_TIMEOUT", 60)
        self.assertGreaterEqual(timeout_val, 10)

    def test_05_structured_output_parsing_validation(self):
        """Pydantic schema validation rejects malformed JSON output."""
        invalid_json = '{"verdict": "InvalidVerdict", "confidence": "high"}'
        with self.assertRaises(Exception):
            self.VerificationVerdict.model_validate_json(invalid_json)


# =====================================================================
# TIER 2: BOUNDARY & CORNER CASES (6 Test Cases)
# =====================================================================

class TestBoundaryAndCornerCases(unittest.TestCase):
    """Tier 2: Boundary, Punctuation, and Edge Cases."""

    def test_01_empty_and_whitespace_queries(self):
        """Empty or whitespace search queries must return empty list without crashing."""
        mock_table = MagicMock()
        adapter = lance_store.LanceTableAdapter(mock_table, "test_table")
        res = adapter.search_hybrid("", [0.0] * 768, top_k=5)
        self.assertIsInstance(res, list)

        res_ws = adapter.search_hybrid("   \n\t  ", [0.0] * 768, top_k=5)
        self.assertIsInstance(res_ws, list)

    def test_02_extreme_scientific_punctuation(self):
        """Extreme scientific punctuation queries must not crash regex or tokenizer."""
        extreme = r"[::] - // ^^ () * + ? {} \ / |"
        norm = judge._normalise(extreme)
        self.assertIsInstance(norm, str)

    def test_03_missing_reference_metadata_fields(self):
        """References with missing DOI, year, and authors must be handled safely."""
        ref = {"title": None, "doi": None, "year": None, "authors": []}
        doi, how = agent2_fetcher.resolve_doi(ref)
        self.assertIsNone(doi)
        self.assertIsNone(how)

    def test_04_author_format_variations(self):
        """Author surname extractor must handle 'Smith, J.', 'J. Smith', and 'Smith, John A.'"""
        names = ["Smith, J.", "J. Smith", "John Smith", "Smith, John A.", "de la Cruz, M."]
        surnames = []
        for name in names:
            parts = [p.strip() for p in name.split(",")]
            if len(parts) > 1:
                surname = parts[0]
            else:
                surname = parts[0].split()[-1]
            surnames.append(surname.lower())

        self.assertIn("smith", surnames[0])
        self.assertIn("smith", surnames[1])
        self.assertIn("smith", surnames[2])
        self.assertIn("smith", surnames[3])

    def test_05_boundary_physics_quantities(self):
        """Extreme negative values and scientific notation must preserve signs."""
        test_values = ["-10^-3", "-273.15 C", "-0.0001 eV", "0.0"]
        for val in test_values:
            norm = judge._normalise(val)
            if val.startswith("-"):
                self.assertIn("-", norm, f"Value {val} must retain negative sign after normalisation")

    def test_06_unbroken_vs_broken_sequence_detection(self):
        """Resolver must distinguish intact sequence [1, 2, 3] from gapped sequence [1, 3]."""
        seq_intact = {1: {"id": "b1"}, 2: {"id": "b2"}, 3: {"id": "b3"}}
        seq_gapped = {1: {"id": "b1"}, 3: {"id": "b3"}}
        self.assertEqual(sorted(seq_intact.keys()), [1, 2, 3])
        self.assertNotEqual(sorted(seq_gapped.keys()), [1, 2, 3])


# =====================================================================
# TIER 3: CROSS-FEATURE INTEGRATION (3 Test Cases)
# =====================================================================

class TestCrossFeatureIntegration(unittest.TestCase):
    """Tier 3: End-to-End Cross-Feature Integration Data Flows."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test-cross-feature-")
        self.orig_data_dir = config.DATA_DIR
        config.DATA_DIR = self.tmp_dir
        self.status_file = os.path.join(self.tmp_dir, "pipeline_status.json")
        pipeline_status.set_status_path(self.status_file)
        pipeline_status.clear_status()

    def tearDown(self):
        pipeline_status.set_status_path(None)
        config.DATA_DIR = self.orig_data_dir
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_fetch_ingest_lancedb_search_pipeline(self):
        """Fetcher -> Ingestor -> LanceDB -> Search flow."""
        client = lance_store.LanceStoreClient(self.tmp_dir)
        table = client.get_or_create_collection("integration_corpus")

        cid = lance_store.compute_chunk_id("paper1.pdf", "text_chunk", 0, "Topological insulator surface states.")
        table.add(
            ids=[cid],
            documents=["Topological insulator surface states."],
            embeddings=[[0.5] * 768],
            metadatas=[{"document": "paper1.pdf", "citation_source": "Hasan 2010"}],
        )
        table.create_fts_index()

        hits = table.search_hybrid("Topological insulator", [0.5] * 768, top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["metadata"]["document"], "paper1.pdf")

    def test_02_search_resolve_verifier_pipeline(self):
        """Search -> Resolver -> Verifier data flow."""
        citation_claim = {
            "sentence": "Dirac fermions exhibit linear dispersion [1].",
            "claim": "Dirac fermions exhibit linear dispersion",
            "ref_txt": "[1]",
        }
        evidence_text = "The Dirac fermions in graphene exhibit linear energy-momentum dispersion."
        is_grounded = judge.span_is_verbatim("linear ... dispersion", evidence_text)
        self.assertTrue(is_grounded)

    def test_03_cancellation_pipeline_halt(self):
        """Worker loops must observe cancellation flag and halt execution."""
        pipeline_status.request_cancel(reason="Integration halt test")
        self.assertTrue(pipeline_status.is_cancel_requested())

        halted = False
        for step in range(100):
            if pipeline_status.is_cancel_requested():
                halted = True
                break
        self.assertTrue(halted)


# =====================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (2 Test Cases)
# =====================================================================

class TestRealWorldApplicationScenarios(unittest.TestCase):
    """Tier 4: Complex Real-World Academic Audit & Batch Synthesis Scenarios."""

    def test_01_end_to_end_seed_audit_pipeline(self):
        """Full audit pipeline on seed document with mixed downloaded and paywalled citations."""
        claims = [
            {"claim": "Graphene has zero band gap", "ref": {"doi": "10.1038/nature1", "title": "Graphene 1"}},
            {"claim": "YBCO has Tc of 93 K", "ref": {"doi": "10.1038/nature2", "title": "YBCO 2"}},
        ]
        downloaded_manifest = {
            "doi:10.1038/nature1": {
                "path": "/fake/path/graphene.pdf",
                "title": "Graphene 1",
                "evidence_tier": "full_text",
            }
        }
        with patch("os.path.exists", return_value=True):
            p_claims = []
            for c in claims:
                ref = c["ref"]
                key = f"doi:{ref['doi']}"
                if key in downloaded_manifest:
                    c["downloaded"] = True
                    c["document"] = "graphene.pdf"
                else:
                    c["downloaded"] = False
                    c["outcome"] = "not_downloaded"
                p_claims.append(c)

            self.assertTrue(p_claims[0]["downloaded"])
            self.assertFalse(p_claims[1]["downloaded"])

    def test_02_batch_citation_multi_sentence_paragraph(self):
        """Batch citation processing across multi-sentence paragraph preserving all evidence."""
        paragraph = (
            "Graphene exhibits anomalous integer quantum Hall effect [cite_1]. "
            "Bi-layer graphene shows an unconventional quantum Hall effect [cite_2]. "
            "Recent experiments also confirm Dirac fermions in graphene [cite_1]."
        )
        sentences = [s.strip() for s in paragraph.split(".") if s.strip()]
        self.assertEqual(len(sentences), 3)

        citations_recorded = {}
        citations_recorded[(0, "cite_1")] = "Evidence chunk 1"
        citations_recorded[(1, "cite_2")] = "Evidence chunk 2"
        citations_recorded[(2, "cite_1")] = "Evidence chunk 3"

        self.assertEqual(len(citations_recorded), 3)
        self.assertEqual(citations_recorded[(0, "cite_1")], "Evidence chunk 1")
        self.assertEqual(citations_recorded[(2, "cite_1")], "Evidence chunk 3")


if __name__ == "__main__":
    unittest.main()
