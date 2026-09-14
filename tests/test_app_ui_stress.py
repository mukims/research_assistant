"""
Empirical stress-testing suite for Streamlit UI components in app.py.
Specifically verifies:
1. Rendering _render_seed_citation_audit with 0 deferred citations.
2. Rendering _render_seed_citation_audit with multiple deferred citations across multiple missing papers.
3. Verification of all 6 metric columns and 4 tabs construction without KeyError or TypeError across normal and edge-case totals.
4. Verification of _render_claim_item across all 7 outcomes/judgements.
5. Simulated file upload, deduplication, and session state caching.
6. Robustness against adversarial malformed payloads.
"""

import json
import os
import unittest
from unittest.mock import MagicMock, call, patch

import app


class MockUIContextManager:
    """Mock for Streamlit container, expander, status, and tab contexts."""

    def __init__(self, name="context"):
        self.name = name
        self.children = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MockColumn(MockUIContextManager):
    """Mock for columns created by st.columns."""

    def __init__(self, col_index):
        super().__init__(f"col_{col_index}")
        self.col_index = col_index
        self.metrics = []

    def metric(self, label, value, delta=None):
        self.metrics.append({"label": label, "value": value, "delta": delta})


class MockStreamlitSession:
    """Mock environment capturing all Streamlit UI calls and session state."""

    def __init__(self):
        self.session_state = {}
        self.markdown_calls = []
        self.caption_calls = []
        self.info_calls = []
        self.warning_calls = []
        self.success_calls = []
        self.error_calls = []
        self.write_calls = []
        self.toast_calls = []
        self.download_button_calls = []
        self.file_uploader_calls = []
        self.button_calls = []
        self.columns_calls = []
        self.tabs_calls = []
        self.expander_calls = []
        self.containers = []
        self.statuses = []
        self.rerun_count = 0
        self.file_uploader_return_values = {}

    def container(self, border=None):
        c = MockUIContextManager("container")
        self.containers.append(c)
        return c

    def expander(self, label, expanded=False):
        self.expander_calls.append(label)
        return MockUIContextManager(f"expander_{label}")

    def status(self, label, expanded=False):
        s = MockUIContextManager(f"status_{label}")
        s.update = MagicMock()
        self.statuses.append(s)
        return s

    def columns(self, spec):
        n = spec if isinstance(spec, int) else len(spec)
        cols = [MockColumn(i) for i in range(n)]
        self.columns_calls.append({"spec": spec, "cols": cols})
        return cols

    def tabs(self, names):
        t_objs = [MockUIContextManager(f"tab_{name}") for name in names]
        self.tabs_calls.append({"names": names, "tabs": t_objs})
        return t_objs

    def markdown(self, text, **kwargs):
        self.markdown_calls.append(text)

    def caption(self, text, **kwargs):
        self.caption_calls.append(text)

    def info(self, text, icon=None, **kwargs):
        self.info_calls.append({"text": text, "icon": icon})

    def warning(self, text, icon=None, **kwargs):
        self.warning_calls.append({"text": text, "icon": icon})

    def success(self, text, **kwargs):
        self.success_calls.append(text)

    def error(self, text, **kwargs):
        self.error_calls.append(text)

    def write(self, text, **kwargs):
        self.write_calls.append(text)

    def toast(self, text, icon=None):
        self.toast_calls.append({"text": text, "icon": icon})

    def download_button(self, label, data=None, file_name=None, mime=None, key=None, help=None):
        self.download_button_calls.append({
            "label": label,
            "file_name": file_name,
            "mime": mime,
            "key": key,
        })
        return False

    def file_uploader(self, label, type=None, key=None, help=None):
        self.file_uploader_calls.append({"label": label, "type": type, "key": key})
        return self.file_uploader_return_values.get(key, None)

    def button(self, label, key=None, type="secondary"):
        self.button_calls.append({"label": label, "key": key, "type": type})
        return False

    def rerun(self):
        self.rerun_count += 1


class TestStreamlitUIStress(unittest.TestCase):
    """Empirical stress tests for app.py Streamlit UI rendering."""

    def _reset_mock(self):
        self.mock_st = MockStreamlitSession()
        app.st = self.mock_st

    def setUp(self):
        self._orig_st = getattr(app, "st", None)
        self._reset_mock()

    def tearDown(self):
        if self._orig_st is not None:
            app.st = self._orig_st

    def test_render_audit_zero_deferred_citations(self):
        """Verify _render_seed_citation_audit when audit has 0 deferred citations."""
        audit = {
            "seed_path": "/fake/seed_paper.pdf",
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
                "deferred_paywalled": 0,
            },
            "results": [
                {
                    "sentence": "Graphene conducts electricity extraordinarily well [1].",
                    "claim": "Graphene conducts electricity well.",
                    "cite_text": "[1]",
                    "ref": {
                        "index": 1,
                        "title": "Electric field effect in thin films",
                        "authors": ["K. Novoselov", "A. Geim"],
                        "year": 2004,
                        "doi": "10.1126/science.1102896",
                    },
                    "outcome": "judged",
                    "judgement": "Supports",
                    "confidence": "High",
                    "evidence_sufficiency": "sufficient",
                    "supporting_span": "Novoselov et al. observed ballistic transport in graphene.",
                    "reason": "Direct evidence of electronic transport.",
                    "slots": {
                        "finding": {"assertion": "conducts well", "verdict": "Supports"},
                        "scope": {"assertion": "graphene", "verdict": "Supports"},
                        "strength": {"assertion": "well", "verdict": "Supports"},
                    },
                },
                {
                    "sentence": "Disorder improves numerical inversion accuracy [2].",
                    "claim": "Disorder improves inversion.",
                    "cite_text": "[2]",
                    "ref": {
                        "index": 2,
                        "title": "Disorder in Numerical Inversion",
                        "authors": ["R. Smith"],
                        "year": 2021,
                    },
                    "outcome": "judged",
                    "judgement": "Partially supports",
                    "confidence": "Medium",
                    "evidence_sufficiency": "partial",
                    "evidence": "Inversion accuracy improved only under weak scattering conditions.",
                    "reason": "Partial support because conditions were restricted.",
                    "slots": {
                        "finding": {"assertion": "improves", "verdict": "Supports"},
                        "scope": {"assertion": "inversion", "verdict": "Partially supports"},
                        "strength": {"assertion": "always", "verdict": "Contradicts"},
                    },
                },
                {
                    "sentence": "Classical methods fail on this benchmark [3].",
                    "claim": "Classical methods fail.",
                    "cite_text": "[3]",
                    "ref": {
                        "index": 3,
                        "title": "Benchmark study of old techniques",
                        "authors": ["O. Author"],
                        "year": 1999,
                    },
                    "outcome": "not_downloaded",
                    "judgement": "Not downloaded",
                },
            ],
        }

        final = {"citation_audit": audit, "seed_path": "/fake/seed_paper.pdf"}
        app._render_seed_citation_audit(final)

        # 1. Check metric row
        metric_col_group = [c for c in self.mock_st.columns_calls if c["spec"] == 6]
        self.assertEqual(len(metric_col_group), 1, "st.columns(6) must be called once")
        cols = metric_col_group[0]["cols"]

        self.assertEqual(cols[0].metrics[0], {"label": "Citations Found", "value": 3, "delta": None})
        self.assertEqual(cols[1].metrics[0], {"label": "Supports 🟢", "value": 1, "delta": None})
        self.assertEqual(cols[2].metrics[0], {"label": "Partially 🟡", "value": 1, "delta": None})
        self.assertEqual(cols[3].metrics[0], {"label": "Need Review 🔴", "value": 0, "delta": None})
        self.assertEqual(cols[4].metrics[0], {"label": "Deferred (Pending) ⏳", "value": 0, "delta": None})
        self.assertEqual(cols[5].metrics[0], {"label": "Paywalled / Unchecked ⚪", "value": 1, "delta": None})

        # 2. Check tabs construction
        self.assertEqual(len(self.mock_st.tabs_calls), 1, "st.tabs must be called once")
        tab_names = self.mock_st.tabs_calls[0]["names"]
        expected_tabs = [
            "Supported (2)",
            "Need Review (0)",
            "⏸ Not Assessed (0)",
            "⏳ Pending Evidence (Deferred) (0)",
            "All Citations (3)",
        ]
        self.assertEqual(tab_names, expected_tabs)

        # 3. Check tab_deferred zero-count banner
        deferred_infos = [
            i for i in self.mock_st.info_calls
            if "No citations currently deferred" in i["text"]
        ]
        self.assertEqual(len(deferred_infos), 1, "Must show zero-deferred info banner")
        self.assertEqual(deferred_infos[0]["icon"], "✅")

        # 4. Check no file uploader was rendered
        self.assertEqual(len(self.mock_st.file_uploader_calls), 0, "No file uploaders should be rendered when deferred=0")

    def test_render_audit_multiple_deferred_citations_multiple_missing_papers(self):
        """Verify _render_seed_citation_audit with multiple deferred citations across missing papers."""
        missing_p1 = {
            "xml_id": "b10",
            "index": 10,
            "title": "Advances in Quantum Sensors",
            "authors": ["A. Aspect", "B. Bell", "C. Clauser", "D. Zeilinger"],
            "year": 2022,
            "doi": "10.1038/s41586-022-0001",
        }
        missing_p2 = {
            "xml_id": "b11",
            "index": 11,
            "title": "Cryogenic Optical Resonators",
            "authors": ["E. Einstein"],
            "year": 2020,
            "doi": None,
            "raw_reference": "Einstein, E. Cryogenic Optical Resonators. Phys Rev 2020.",
        }

        audit = {
            "seed_path": "/workspace/seeds/quantum_seed.pdf",
            "totals": {
                "total": 4,
                "downloaded": 0,
                "judged": 0,
                "Supports": 0,
                "Partially supports": 0,
                "Contradicts": 0,
                "Does not support": 0,
                "Unclear / insufficient evidence": 0,
                "not_downloaded": 0,
                "deferred_paywalled": 4,
            },
            "results": [
                {
                    "sentence": "Quantum sensors operate below the standard quantum limit [10].",
                    "claim": "Quantum sensors operate below standard limit.",
                    "cite_text": "[10]",
                    "paragraph_id": "p1",
                    "paragraph_refs": ["b10"],
                    "paragraph_missing_refs": [missing_p1],
                    "ref": missing_p1,
                    "outcome": "deferred_paywalled",
                    "judgement": "Deferred (pending paywalled evidence)",
                },
                {
                    "sentence": "Furthermore, squeezing enhancements reduce optical phase noise [10].",
                    "claim": "Squeezing reduces phase noise.",
                    "cite_text": "[10]",
                    "paragraph_id": "p1",
                    "paragraph_refs": ["b10"],
                    "paragraph_missing_refs": [missing_p1],
                    "ref": missing_p1,
                    "outcome": "deferred_paywalled",
                    "judgement": "Deferred (pending paywalled evidence)",
                },
                {
                    "sentence": "Cryogenic cavities stabilize laser frequencies against thermal noise [11].",
                    "claim": "Cavities stabilize laser frequencies.",
                    "cite_text": "[11]",
                    "paragraph_id": "p2",
                    "paragraph_refs": ["b10", "b11"],
                    "paragraph_missing_refs": [missing_p1, missing_p2],
                    "ref": missing_p2,
                    "outcome": "deferred_paywalled",
                    "judgement": "Deferred (pending paywalled evidence)",
                },
                {
                    "sentence": "Both resonators and sensors are coupled to cryogenic cooling [10, 11].",
                    "claim": "Coupled to cryogenic cooling.",
                    "cite_text": "[10, 11]",
                    "paragraph_id": "p2",
                    "paragraph_refs": ["b10", "b11"],
                    "paragraph_missing_refs": [missing_p1, missing_p2],
                    "ref": missing_p1,
                    "outcome": "deferred_paywalled",
                    "judgement": "Deferred (pending paywalled evidence)",
                },
            ],
        }

        final = {"citation_audit": audit, "seed_path": "/workspace/seeds/quantum_seed.pdf"}
        app._render_seed_citation_audit(final)

        # 1. Metrics check
        metric_col_group = [c for c in self.mock_st.columns_calls if c["spec"] == 6]
        self.assertEqual(len(metric_col_group), 1)
        cols = metric_col_group[0]["cols"]
        self.assertEqual(cols[4].metrics[0], {"label": "Deferred (Pending) ⏳", "value": 4, "delta": None})

        # 2. Tabs check
        tab_names = self.mock_st.tabs_calls[0]["names"]
        self.assertEqual(
            tab_names,
            ["Supported (0)", "Need Review (0)", "⏸ Not Assessed (0)", "⏳ Pending Evidence (Deferred) (4)", "All Citations (4)"],
        )

        # 3. tab_deferred warning banner
        deferred_warnings = [
            w for w in self.mock_st.warning_calls
            if "More than 50% are missing from the corpus" in w["text"]
            or "Evaluation Deferred" in w["text"]
        ]
        self.assertTrue(len(deferred_warnings) >= 1, "Must show deferred warning banner in tab_deferred")
        self.assertEqual(deferred_warnings[0]["icon"], "⏳")

        # 4. Actionable cards and file uploaders for missing papers
        # There are 2 unique missing papers: b10 and b11
        uploaders = self.mock_st.file_uploader_calls
        self.assertEqual(len(uploaders), 2, "Must render exactly 2 file uploaders for 2 missing papers")

        p1_uploader = next((u for u in uploaders if "upload_ref_quantum_seed_b10" in u["key"]), None)
        p2_uploader = next((u for u in uploaders if "upload_ref_quantum_seed_b11" in u["key"]), None)
        self.assertIsNotNone(p1_uploader, "Uploader for missing paper b10 must exist with sanitized key")
        self.assertIsNotNone(p2_uploader, "Uploader for missing paper b11 must exist with sanitized key")

        # 5. Missing paper card metadata rendered in markdown
        md_text = "\n".join(self.mock_st.markdown_calls)
        self.assertIn("Advances in Quantum Sensors", md_text)
        self.assertIn("Cryogenic Optical Resonators", md_text)
        self.assertIn("A. Aspect, B. Bell, C. Clauser et al.", md_text)
        self.assertIn("10.1038/s41586-022-0001", md_text)

        # 6. Dependent statements expander
        expander_labels = self.mock_st.expander_calls
        p1_expander = any("Dependent Statement(s) in Seed Paper (4)" in lbl for lbl in expander_labels)
        p2_expander = any("Dependent Statement(s) in Seed Paper (2)" in lbl for lbl in expander_labels)
        self.assertTrue(p1_expander, f"Expander for b10 should state 4 dependent statements, got {expander_labels}")
        self.assertTrue(p2_expander, f"Expander for b11 should state 2 dependent statements, got {expander_labels}")

    def test_render_metric_columns_and_tabs_edge_cases(self):
        """Stress-test metric columns and tabs under edge cases (empty dicts, missing keys, boundary counts)."""
        edge_cases = [
            # 1. Empty totals dict
            ({}, 2, ["Supported (0)", "Need Review (0)", "⏸ Not Assessed (0)", "⏳ Pending Evidence (Deferred) (0)", "All Citations (2)"]),
            # 2. Missing some keys in totals
            (
                {"total": 5, "Supports": 2},
                5,
                ["Supported (2)", "Need Review (0)", "⏸ Not Assessed (0)", "⏳ Pending Evidence (Deferred) (0)", "All Citations (5)"],
            ),
            # 3. All deferred
            (
                {"total": 10, "deferred_paywalled": 10},
                10,
                ["Supported (0)", "Need Review (0)", "⏸ Not Assessed (0)", "⏳ Pending Evidence (Deferred) (10)", "All Citations (10)"],
            ),
            # 4. Large values
            (
                {
                    "total": 1000,
                    "Supports": 400,
                    "Partially supports": 200,
                    "Contradicts": 150,
                    "Does not support": 50,
                    "deferred_paywalled": 100,
                    "not_downloaded": 100,
                },
                1000,
                [
                    "Supported (600)",
                    "Need Review (200)",
                    "⏸ Not Assessed (0)",
                    "⏳ Pending Evidence (Deferred) (100)",
                    "All Citations (1000)",
                ],
            ),
        ]

        for totals, expected_all_count, expected_tabs in edge_cases:
            with self.subTest(totals=totals):
                self._reset_mock()
                sample_results = [{"outcome": "deferred_paywalled", "claim": "dummy"}] * expected_all_count
                audit = {
                    "totals": totals,
                    "results": sample_results,
                }
                final = {"citation_audit": audit}

                # Must not raise KeyError or TypeError
                try:
                    app._render_seed_citation_audit(final)
                except Exception as exc:
                    self.fail(f"_render_seed_citation_audit crashed with totals={totals}: {exc}")

                metric_group = [c for c in self.mock_st.columns_calls if c["spec"] == 6]
                self.assertEqual(len(metric_group), 1)
                cols = metric_group[0]["cols"]

                # Verify all 6 metric columns received valid calls
                for i in range(6):
                    self.assertEqual(len(cols[i].metrics), 1)
                    self.assertIsInstance(cols[i].metrics[0]["label"], str)

                # Verify tabs
                self.assertEqual(len(self.mock_st.tabs_calls), 1)
                self.assertEqual(self.mock_st.tabs_calls[0]["names"], expected_tabs)

    def test_render_claim_item_all_7_outcomes_and_judgements(self):
        """Stress-test _render_claim_item for all 7 outcomes/judgements."""
        test_items = [
            # 1. deferred_paywalled
            {
                "name": "deferred_paywalled",
                "item": {
                    "sentence": "Sentence deferred due to missing context [1].",
                    "claim": "Claim deferred.",
                    "cite_text": "[1]",
                    "ref": {"index": 1, "title": "Paper 1", "authors": ["A. Author"], "year": 2020},
                    "outcome": "deferred_paywalled",
                    "judgement": "Deferred (pending paywalled evidence)",
                    "paragraph_missing_refs": [{"index": 1, "title": "Paper 1", "doi": "10.1000/1"}],
                },
                "expected_badge": "⏳ Deferred (Pending Evidence)",
                "check": lambda: any("Evaluation Deferred" in w["text"] for w in self.mock_st.warning_calls),
            },
            # 2. judged - Supports
            {
                "name": "judged_supports",
                "item": {
                    "sentence": "Supports finding [2].",
                    "claim": "Claim supported.",
                    "cite_text": "[2]",
                    "ref": {"index": 2, "title": "Paper 2", "authors": ["B. Author"], "year": 2021},
                    "outcome": "judged",
                    "judgement": "Supports",
                    "confidence": "High",
                    "evidence_sufficiency": "sufficient",
                    "supporting_span": "Exact quote supporting claim.",
                    "slots": {
                        "finding": {"assertion": "f", "verdict": "Supports"},
                        "scope": {"assertion": "s", "verdict": "Supports"},
                        "strength": {"assertion": "st", "verdict": "Supports"},
                    },
                },
                "expected_badge": "🟢 Supports",
                "check": lambda: any("Exact quote supporting claim" in s for s in self.mock_st.success_calls),
            },
            # 3. judged - Partially supports
            {
                "name": "judged_partially_supports",
                "item": {
                    "sentence": "Partially supports finding [3].",
                    "claim": "Claim partially supported.",
                    "cite_text": "[3]",
                    "ref": {"index": 3, "title": "Paper 3", "authors": ["C. Author"], "year": 2022},
                    "outcome": "judged",
                    "judgement": "Partially supports",
                    "confidence": "Medium",
                    "evidence_sufficiency": "partial",
                    "evidence": "Long retrieved evidence passage demonstrating partial support in subset of trials.",
                    "reason": "Scope mismatch.",
                    "slots": {
                        "finding": {"assertion": "f", "verdict": "Supports"},
                        "scope": {"assertion": "s", "verdict": "Partially supports"},
                        "strength": {"assertion": "st", "verdict": "Supports"},
                    },
                },
                "expected_badge": "🟡 Partially Supports",
                "check": lambda: any("Scope mismatch" in m for m in self.mock_st.markdown_calls),
            },
            # 4. judged - Contradicts
            {
                "name": "judged_contradicts",
                "item": {
                    "sentence": "Claims contradicts evidence [4].",
                    "claim": "Contradicting claim.",
                    "cite_text": "[4]",
                    "ref": {"index": 4, "title": "Paper 4", "authors": ["D. Author"], "year": 2023},
                    "outcome": "judged",
                    "judgement": "Contradicts",
                    "confidence": "High",
                    "evidence_sufficiency": "sufficient",
                    "reason": "Direct contradiction found in results.",
                    "slots": {
                        "finding": {"assertion": "f", "verdict": "Contradicts"},
                        "scope": {"assertion": "s", "verdict": "Supports"},
                        "strength": {"assertion": "st", "verdict": "Supports"},
                    },
                },
                "expected_badge": "🔴 Contradicts",
                "check": lambda: any("Direct contradiction found" in m for m in self.mock_st.markdown_calls),
            },
            # 5. judged - Does not support
            {
                "name": "judged_does_not_support",
                "item": {
                    "sentence": "Claims does not support [5].",
                    "claim": "Unsupported claim.",
                    "cite_text": "[5]",
                    "ref": {"index": 5, "title": "Paper 5", "authors": ["E. Author"], "year": 2024},
                    "outcome": "judged",
                    "judgement": "Does not support",
                    "confidence": "Medium",
                    "evidence_sufficiency": "insufficient",
                    "reason": "Author makes assertion without data.",
                    "slots": {
                        "finding": {"assertion": "f", "verdict": "Does not support"},
                        "scope": {"assertion": "s", "verdict": "Supports"},
                        "strength": {"assertion": "st", "verdict": "Supports"},
                    },
                },
                "expected_badge": "🟠 Does Not Support",
                "check": lambda: any("Author makes assertion without data" in m for m in self.mock_st.markdown_calls),
            },
            # 6. not_downloaded
            {
                "name": "not_downloaded",
                "item": {
                    "sentence": "Paywalled reference claim [6].",
                    "claim": "Claim from paywalled work.",
                    "cite_text": "[6]",
                    "ref": {"index": 6, "title": "Paper 6", "authors": ["F. Author"], "year": 2019},
                    "outcome": "not_downloaded",
                    "judgement": "Not downloaded",
                },
                "expected_badge": "🔒 Paywalled / Not In Corpus",
                "check": lambda: any("Reference Not Downloaded" in w["text"] for w in self.mock_st.warning_calls),
            },
            # 7. Unclear / Fallback (e.g. retrieval_failed, parse_failed, or generic)
            {
                "name": "unclear_fallback",
                "item": {
                    "sentence": "Unclear reference claim [7].",
                    "claim": "Claim with processing issue.",
                    "cite_text": "[7]",
                    "ref": {"index": 7, "title": "Paper 7", "authors": ["G. Author"], "year": 2018},
                    "outcome": "retrieval_failed",
                    "judgement": "Unclear / insufficient evidence",
                    "reason": "ChromaDB connection timeout.",
                },
                "expected_badge": "⏸ Not Assessed (retrieval failed)",
                "check": lambda: any("ChromaDB connection timeout" in c for c in self.mock_st.caption_calls),
            },
        ]

        for tc in test_items:
            with self.subTest(outcome=tc["name"]):
                self._reset_mock()
                audit = {
                    "totals": {"total": 1},
                    "results": [tc["item"]],
                }
                final = {"citation_audit": audit}

                try:
                    app._render_seed_citation_audit(final)
                except Exception as exc:
                    self.fail(f"_render_claim_item crashed for {tc['name']}: {exc}")

                # Check expander header badge
                self.assertTrue(
                    any(tc["expected_badge"] in exp for exp in self.mock_st.expander_calls),
                    f"Badge {tc['expected_badge']} not found in expander headers: {self.mock_st.expander_calls}",
                )

                # Check specific outcome UI element
                self.assertTrue(tc["check"](), f"Specific UI element check failed for {tc['name']}")

    def test_simulated_file_upload_and_caching(self):
        """Verify handling of uploaded PDF file, registration, and session caching in tab_deferred."""
        ref_entry = {
            "xml_id": "b42",
            "index": 42,
            "title": "Quantum Hall Effect in Graphene",
            "authors": ["Y. Zhang"],
            "year": 2005,
            "doi": "10.1038/nature04235",
        }
        audit = {
            "seed_path": "/path/to/seed.pdf",
            "totals": {"total": 1, "deferred_paywalled": 1},
            "results": [
                {
                    "sentence": "Statement relying on missing paper [42].",
                    "ref": ref_entry,
                    "outcome": "deferred_paywalled",
                    "paragraph_missing_refs": [ref_entry],
                }
            ],
        }

        # Mock uploaded file
        mock_file = MagicMock()
        mock_file.name = "zhang2005.pdf"
        mock_file.size = 204800
        mock_file.read.return_value = b"%PDF-1.4 simulated pdf bytes"

        # Provide uploaded file for the uploader key
        uploader_key = "upload_ref_seed_b42"
        self.mock_st.file_uploader_return_values[uploader_key] = mock_file

        with patch("research_assistant.shared.seed_audit.save_and_register_reference_pdf") as mock_save:
            mock_save.return_value = {"filename": "zhang2005.pdf", "title": ref_entry["title"], "ingested": True}

            # First run: file is uploaded, should register
            final = {"citation_audit": audit, "seed_path": "/path/to/seed.pdf"}
            app._render_seed_citation_audit(final)

            self.assertEqual(mock_save.call_count, 1)
            called_bytes, called_ref = mock_save.call_args[0]
            self.assertEqual(called_bytes, b"%PDF-1.4 simulated pdf bytes")
            self.assertEqual(called_ref["xml_id"], "b42")
            self.assertEqual(called_ref["title"], "Quantum Hall Effect in Graphene")
            self.assertEqual(called_ref["affected_claims"], ["Statement relying on missing paper [42]."])
            self.assertEqual(mock_save.call_args[1]["seed_pdf_name"], "seed.pdf")
            self.assertEqual(mock_save.call_args[1]["original_filename"], "zhang2005.pdf")

            # Check toast and success message
            self.assertEqual(len(self.mock_st.toast_calls), 1)
            self.assertIn("Uploaded & indexed", self.mock_st.toast_calls[0]["text"])
            self.assertTrue(any("uploaded and indexed in knowledge base" in s for s in self.mock_st.success_calls))

            # Session state keys must be populated
            proc_key = f"processed_upload_seed_b42_{mock_file.size}"
            legacy_key = "processed_upload_seed_b42"
            self.assertIn(proc_key, self.mock_st.session_state)
            self.assertEqual(self.mock_st.session_state[legacy_key], mock_file.size)

            # Second run: file is still in uploader, but already processed; should NOT register again
            mock_save.reset_mock()
            self.mock_st.toast_calls.clear()
            app._render_seed_citation_audit(final)

            self.assertEqual(mock_save.call_count, 0, "Deduplication must prevent re-registering identical upload")
            self.assertEqual(len(self.mock_st.toast_calls), 0)

    def test_adversarial_malformed_inputs_handled_gracefully(self):
        """Stress-test _render_seed_citation_audit with malformed, None, or adversarial payloads."""
        adversarial_payloads = [
            # None or missing values in ref
            {
                "sentence": "Missing ref dict",
                "ref": None,
                "outcome": "judged",
                "judgement": "Supports",
                "slots": None,
            },
            # Empty dict slots
            {
                "sentence": "Empty slots dict",
                "ref": {"title": "Test"},
                "outcome": "judged",
                "judgement": "Contradicts",
                "slots": {},
            },
            # Slots containing non-dict entries
            {
                "sentence": "Primitive slot values",
                "ref": {"title": "Test 2"},
                "outcome": "judged",
                "judgement": "Supports",
                "slots": {"finding": "Supports", "scope": None, "strength": 123},
            },
            # Special non-ASCII characters and emojis in xml_id and title
            {
                "sentence": "Unicode title [🚀].",
                "ref": {
                    "xml_id": "b-special:weird#id",
                    "index": 99,
                    "title": "Quantum ⚛️ Graphene /  graphene-effects (v2) [special]",
                    "authors": ["M. Sjöström", "É. Legrand"],
                    "year": "2024",
                    "doi": "10.1000/182@!#$",
                },
                "outcome": "deferred_paywalled",
                "paragraph_missing_refs": [
                    {
                        "xml_id": "b-special:weird#id",
                        "index": 99,
                        "title": "Quantum ⚛️ Graphene / graphene-effects (v2) [special]",
                        "authors": ["M. Sjöström", "É. Legrand"],
                        "year": "2024",
                        "doi": "10.1000/182@!#$",
                    },
                    None,  # None in missing refs list
                    "not_a_dict",  # non-dict in missing refs list
                ],
            },
            # Very long strings (>5000 chars)
            {
                "sentence": "A" * 5000,
                "claim": "B" * 5000,
                "ref": {
                    "title": "C" * 1000,
                    "raw_reference": "D" * 2000,
                },
                "outcome": "not_downloaded",
            },
        ]

        audit = {
            "totals": {"total": len(adversarial_payloads), "deferred_paywalled": 1},
            "results": adversarial_payloads,
        }
        final = {"citation_audit": audit, "seed_path": "seed.pdf"}

        # Running audit must not raise unhandled exceptions
        try:
            app._render_seed_citation_audit(final)
        except Exception as exc:
            self.fail(f"_render_seed_citation_audit crashed on adversarial payloads: {exc}")

    def test_adversarial_non_dict_slots_triggers_attribute_error_in_backend(self):
        """Document adversarial finding: when slots is a string on judged claim, explain_rubric_verdict raises AttributeError."""
        from research_assistant.shared.seed_audit import explain_rubric_verdict

        item = {
            "outcome": "judged",
            "judgement": "Supports",
            "slots": "non_dict_string_payload",
        }
        with self.assertRaises(AttributeError):
            explain_rubric_verdict(item)


if __name__ == "__main__":
    unittest.main()
