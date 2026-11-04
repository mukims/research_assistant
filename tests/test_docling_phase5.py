# tests/test_docling_phase5.py
"""
Unit tests for Phase 5 Docling extractor and rebuild script fixes.
Maintains unittest.TestCase style without importing pytest.
"""

import os
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import fields
from unittest.mock import MagicMock, patch

from research_assistant.shared.extract import Figure
from scripts.rebuild_corpus_docling import compute_abstract_id, delete_document_rows


class TestFigureDataclass(unittest.TestCase):
    def test_table_markdown_is_last_field_with_default(self):
        f_fields = fields(Figure)
        self.assertEqual(f_fields[-1].name, "table_markdown")
        self.assertEqual(f_fields[-1].default, "")
        self.assertEqual(f_fields[-2].name, "image_path")
        self.assertEqual(f_fields[-2].default, "")

    def test_positional_instantiation_backward_compatibility(self):
        fig = Figure(
            "fig_1",
            "figure",
            "Figure 1",
            "A test plot",
            1,
            (10, 10, 100, 100),
            ["ref 1"],
            "data/images/fig_1.png",
        )
        self.assertEqual(fig.image_path, "data/images/fig_1.png")
        self.assertEqual(fig.table_markdown, "")


class TestAbstractOnlyIdDeterminism(unittest.TestCase):
    def test_abstract_only_id_format(self):
        doc_key = "arxiv_2301.00001.pdf"
        cid = compute_abstract_id(doc_key)
        self.assertTrue(cid.startswith("chk_abs_"))
        self.assertEqual(len(cid), 24)  # "chk_abs_" (8) + 16 hex chars

    def test_different_hashseeds_produce_same_id(self):
        probe_code = (
            "import sys; "
            "from scripts.rebuild_corpus_docling import compute_abstract_id; "
            "print(compute_abstract_id('paper_test_determinism.pdf'))"
        )
        cmd = [sys.executable, "-c", probe_code]
        env1 = {**os.environ, "PYTHONHASHSEED": "0"}
        env2 = {**os.environ, "PYTHONHASHSEED": "999999"}

        out1 = subprocess.check_output(cmd, env=env1, text=True).strip()
        out2 = subprocess.check_output(cmd, env=env2, text=True).strip()

        self.assertEqual(out1, out2)
        self.assertTrue(out1.startswith("chk_abs_"))


class TestSyntheticDoclingExtraction(unittest.TestCase):
    def test_references_skipped_and_appendices_retained(self):
        from research_assistant.shared.docling_extract import extract_docling

        class FakeItem:
            def __init__(self, label, text, page=1):
                self.label = label
                self.text = text
                self.prov = [types.SimpleNamespace(page_no=page)]

        class FakeDoc:
            name = "synthetic_test.pdf"
            pictures = []
            tables = []
            texts = [1]

            def iterate_items(self):
                return [
                    (FakeItem("section_header", "1. Introduction"), 1),
                    (FakeItem("text", "Introductory discussion."), 1),
                    (FakeItem("section_header", "References"), 1),
                    (FakeItem("reference", "[1] First citation."), 2),
                    (FakeItem("reference", "[2] Second citation."), 2),
                    (FakeItem("section_header", "Appendix A: Mathematical Derivations"), 3),
                    (FakeItem("text", "Detailed derivation step."), 3),
                ]

            def model_dump(self, mode="json"):
                return {}

        with patch("research_assistant.shared.docling_extract.get_digital_converter") as mock_conv, \
             patch("research_assistant.shared.docling_extract.needs_ocr", return_value=False), \
             patch("os.path.isfile", return_value=True), \
             patch("builtins.open", unittest.mock.mock_open()):

            mock_conv_instance = MagicMock()
            mock_conv_instance.convert.return_value = types.SimpleNamespace(document=FakeDoc())
            mock_conv.return_value = mock_conv_instance

            doc = extract_docling("dummy.pdf", use_cache=False)

        section_titles = [s.heading for s in doc.sections]
        self.assertIn("1. Introduction", section_titles)
        self.assertIn("Appendix A: Mathematical Derivations", section_titles)
        self.assertNotIn("References", section_titles)

        # Check paragraphs
        all_text = " ".join(p.text for s in doc.sections for p in s.paragraphs)
        self.assertIn("Introductory discussion.", all_text)
        self.assertIn("Detailed derivation step.", all_text)
        self.assertNotIn("First citation", all_text)
        self.assertNotIn("Second citation", all_text)


class TestIdempotentBatchHandling(unittest.TestCase):
    def test_delete_document_rows_calls_table_delete(self):
        mock_client = MagicMock()
        mock_papers = MagicMock()
        mock_summaries = MagicMock()

        def get_col(name):
            if "summaries" in name:
                return mock_summaries
            return mock_papers

        mock_client.get_collection.side_effect = get_col

        delete_document_rows(mock_client, ["paper_1.pdf", "paper_2.pdf"])

        mock_papers.delete.assert_called_once_with(where={"document": {"$in": ["paper_1.pdf", "paper_2.pdf"]}})
        mock_summaries.delete.assert_called_once_with(where={"document": {"$in": ["paper_1.pdf", "paper_2.pdf"]}})


class TestDoclingCropPaddedAndMerge(unittest.TestCase):
    def test_subpanels_merged_and_fullpage_rejected(self):
        from research_assistant.shared.docling_extract import extract_docling

        class FakeBBox:
            def __init__(self, l, t, r, b):
                self.l, self.t, self.r, self.b = l, t, r, b
                self.coord_origin = "BOTTOMLEFT"

        class FakePic:
            def __init__(self, self_ref, cap, l, t, r, b, page=1):
                self.self_ref = self_ref
                self._cap = cap
                self.prov = [types.SimpleNamespace(page_no=page, bbox=FakeBBox(l, t, r, b))]
            def caption_text(self, doc):
                return self._cap
            def get_image(self, doc):
                return None

        class FakeDoc:
            name = "test_crop.pdf"
            tables = []
            texts = [1]
            def __init__(self):
                self.pictures = [
                    # Two touching subpanels on page 1
                    FakePic("#/pictures/0", "", 100, 500, 300, 310, page=1),
                    FakePic("#/pictures/1", "Figure 1: Two-panel plot", 100, 305, 300, 100, page=1),
                    # One full page false positive on page 2 (no caption, >80% height & width)
                    FakePic("#/pictures/2", "", 50, 780, 550, 20, page=2),
                ]
            def iterate_items(self):
                return []
            def model_dump(self, mode="json"):
                return {}

        fake_pdf = MagicMock()
        fake_page1 = MagicMock()
        fake_page1.rect = types.SimpleNamespace(width=612.0, height=792.0)
        fake_page2 = MagicMock()
        fake_page2.rect = types.SimpleNamespace(width=612.0, height=792.0)
        fake_pdf.__len__.return_value = 2
        fake_pdf.__getitem__.side_effect = lambda idx: fake_page1 if idx == 0 else fake_page2

        with patch("research_assistant.shared.docling_extract.get_digital_converter") as mock_conv, \
             patch("research_assistant.shared.docling_extract.needs_ocr", return_value=False), \
             patch("fitz.open", return_value=fake_pdf), \
             patch("os.path.isfile", return_value=True):

            mock_conv_instance = MagicMock()
            mock_conv_instance.convert.return_value = types.SimpleNamespace(document=FakeDoc())
            mock_conv.return_value = mock_conv_instance

            doc = extract_docling("dummy.pdf", use_cache=False)

        # The two subpanels must be merged into 1 figure, and the fullpage picture must be rejected!
        self.assertEqual(len(doc.figures), 1)
        self.assertEqual(doc.figures[0].label, "Figure 1")
        self.assertEqual(doc.figures[0].caption, "Figure 1: Two-panel plot")


if __name__ == "__main__":
    unittest.main()
