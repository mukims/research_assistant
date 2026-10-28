# tests/test_ocr.py
"""Tests for OCR detection, extraction, and ingestion integration."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import fitz

from research_assistant.shared import ocr
from research_assistant.shared import extract as ex
from research_assistant.shared.ingest_v2 import EXTRACTION_MODES_V2, process_pdf_v2


class TestIsScannedPdf(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_digital_pdf_returns_false(self):
        pdf_path = os.path.join(self.tmp.name, "digital.pdf")
        doc = fitz.open()
        page = doc.new_page()
        # Add plentiful digital text
        text = "This is a digital PDF with extractable font text. " * 10
        page.insert_text((50, 50), text)
        doc.save(pdf_path)
        doc.close()

        self.assertFalse(ocr.is_scanned_pdf(pdf_path, min_chars_per_page=50))

    def test_scanned_empty_pdf_returns_true(self):
        pdf_path = os.path.join(self.tmp.name, "scanned.pdf")
        doc = fitz.open()
        # Blank page (simulating an image or empty scan)
        doc.new_page()
        doc.save(pdf_path)
        doc.close()

        self.assertTrue(ocr.is_scanned_pdf(pdf_path, min_chars_per_page=50))

    def test_nonexistent_file_returns_false(self):
        self.assertFalse(ocr.is_scanned_pdf("/path/does/not/exist.pdf"))


class TestOcrPageImage(unittest.TestCase):
    def test_easyocr_dehyphenation_and_cleaning(self):
        mock_reader = MagicMock()
        # Mock easyocr result with hyphenated line break and noisy short text
        mock_reader.readtext.return_value = [
            ([[[10, 10], [100, 10], [100, 30], [10, 30]]], "semi-\nconductor materials"),
            ([[[10, 40], [100, 40], [100, 60], [10, 60]]], "exhibit novel electronic properties."),
            ([[[10, 70], [20, 70], [20, 80], [10, 80]]], "x"),  # Noise: < 4 chars
        ]

        with patch.object(ocr, "get_ocr_reader", return_value=mock_reader):
            # Test with dummy numpy array
            import numpy as np
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            paras = ocr.ocr_page_image(dummy_img, engine="easyocr")

            self.assertEqual(len(paras), 2)
            self.assertEqual(paras[0], "semiconductor materials")
            self.assertEqual(paras[1], "exhibit novel electronic properties.")

    def test_vlm_engine_transcription(self):
        mock_result = MagicMock()
        mock_result.content = "Section 1: Notes\n\nQuantum state superposition |ψ⟩ = α|0⟩ + β|1⟩."

        with patch("research_assistant.shared.llm.chat", return_value=mock_result):
            import numpy as np
            dummy_img = np.zeros((50, 50, 3), dtype=np.uint8)
            paras = ocr.ocr_page_image(dummy_img, engine="vlm")

            self.assertEqual(len(paras), 2)
            self.assertEqual(paras[0], "Section 1: Notes")
            self.assertIn("Quantum state superposition", paras[1])


class TestExtractOcr(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_extract_ocr_page_tracking_and_bib_cut(self):
        pdf_path = os.path.join(self.tmp.name, "sample_scan.pdf")
        doc = fitz.open()
        doc.new_page()  # Page 0
        doc.new_page()  # Page 1
        doc.save(pdf_path)
        doc.close()

        # Mock ocr_page_image: page 0 has intro text, page 1 has conclusion and references
        def _mock_ocr_page(img, engine="easyocr"):
            if not hasattr(_mock_ocr_page, "calls"):
                _mock_ocr_page.calls = 0
            _mock_ocr_page.calls += 1
            if _mock_ocr_page.calls == 1:
                return ["First page intro paragraph.", "Second paragraph on page 0."]
            return ["Conclusion on page 1.", "References", "1. Smith et al. 2024."]

        with patch.object(ocr, "ocr_page_image", side_effect=_mock_ocr_page):
            d = ocr.extract_ocr(pdf_path, key="sample_scan.pdf", engine="easyocr")

            self.assertEqual(d.key, "sample_scan.pdf")
            self.assertEqual(d.extraction, "ocr")
            self.assertEqual(len(d.sections), 1)

            paragraphs = d.sections[0].paragraphs
            self.assertEqual(len(paragraphs), 3)  # References and citation cut
            self.assertEqual(paragraphs[0].text, "First page intro paragraph.")
            self.assertEqual(paragraphs[0].page, 0)
            self.assertEqual(paragraphs[1].page, 0)
            self.assertEqual(paragraphs[2].text, "Conclusion on page 1.")
            self.assertEqual(paragraphs[2].page, 1)


class TestExtractDispatchOcr(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pdf_path = os.path.join(self.tmp.name, "test.pdf")
        doc = fitz.open()
        doc.new_page()
        doc.save(self.pdf_path)
        doc.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_scanned_pdf_routes_to_ocr_when_enabled(self):
        mock_ocr_doc = ex.Document(
            key="test.pdf",
            title="test.pdf",
            abstract=[],
            sections=[ex.Section("", "other", [ex.Paragraph("Scanned text", 0)])],
            figures=[],
            extraction="ocr",
        )

        with patch("research_assistant.shared.ocr.is_scanned_pdf", return_value=True), \
             patch("research_assistant.shared.ocr.extract_ocr", return_value=mock_ocr_doc) as mock_extract:
            doc = ex.extract(self.pdf_path, key="test.pdf", enable_ocr=True)

            mock_extract.assert_called_once_with(self.pdf_path, key="test.pdf", engine="easyocr")
            self.assertEqual(doc.extraction, "ocr")
            self.assertEqual(doc.sections[0].paragraphs[0].text, "Scanned text")

    def test_scanned_pdf_skips_ocr_when_disabled(self):
        empty_pymupdf = ex.Document(
            key="test.pdf",
            title="test.pdf",
            abstract=[],
            sections=[ex.Section("", "other", [])],
            figures=[],
            extraction="pymupdf",
        )

        with patch("research_assistant.shared.ocr.is_scanned_pdf", return_value=True), \
             patch("research_assistant.shared.ocr.extract_ocr") as mock_extract, \
             patch.object(ex, "extract_pymupdf", return_value=empty_pymupdf):
            doc = ex.extract(self.pdf_path, key="test.pdf", enable_ocr=False)

            mock_extract.assert_not_called()
            self.assertEqual(doc.extraction, "pymupdf")


class TestIngestV2Ocr(unittest.TestCase):
    def test_modes_contain_ocr(self):
        self.assertIn("ocr", EXTRACTION_MODES_V2)

    def test_process_pdf_v2_with_ocr_extraction(self):
        mock_ocr_doc = ex.Document(
            key="scanned_paper.pdf",
            title="Scanned Paper Title",
            abstract=[],
            sections=[ex.Section("", "other", [
                ex.Paragraph("This is an important extracted paragraph from the scanned manuscript with sufficient characters for chunking.", 0)
            ])],
            figures=[],
            extraction="ocr",
        )

        with patch("research_assistant.shared.ingest_v2.extract", return_value=mock_ocr_doc):
            entries = process_pdf_v2("/path/scanned_paper.pdf", "Scanned Paper Title")
            self.assertTrue(len(entries) > 0)
            for entry in entries:
                if entry.get("type") == "text_chunk":
                    self.assertEqual(entry.get("extraction"), "ocr")
                    self.assertEqual(entry.get("meta", {}).get("extraction"), "ocr")
