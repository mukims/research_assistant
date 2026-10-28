"""Tests for batch_uploader: unpack_and_stage_uploads, scan_and_stage_directory, fetch_and_stage_urls_or_dois."""

import io
import os
import tempfile
import unittest
import zipfile
from unittest.mock import patch, MagicMock

from research_assistant.shared import batch_uploader

DUMMY_PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n%%EOF\n"


class TestBatchUploader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dest_dir = os.path.join(self.tmp.name, "staged")
        os.makedirs(self.dest_dir, exist_ok=True)

    def test_extract_from_bytes_valid_pdf(self):
        rec = batch_uploader._extract_from_bytes(
            DUMMY_PDF_BYTES,
            "sample_paper.pdf",
            self.dest_dir,
        )
        self.assertIsNotNone(rec)
        self.assertTrue(os.path.exists(rec["path"]))
        self.assertEqual(rec["filename"], "sample_paper.pdf")
        self.assertTrue(rec["key"].startswith("file:"))

    def test_extract_from_bytes_invalid_skipped(self):
        rec = batch_uploader._extract_from_bytes(
            b"Not a PDF file at all",
            "readme.txt",
            self.dest_dir,
        )
        self.assertIsNone(rec)

    def test_unpack_and_stage_uploads_zip(self):
        # Create an in-memory ZIP containing two valid PDFs
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("paper1.pdf", DUMMY_PDF_BYTES + b"% paper1\n")
            zf.writestr("subfolder/paper2.pdf", DUMMY_PDF_BYTES + b"% paper2\n")
            zf.writestr("ignored.txt", b"hello world")
        zip_bytes = zip_buf.getvalue()

        staged = batch_uploader.unpack_and_stage_uploads(
            [zip_bytes],
            destination_dir=self.dest_dir,
        )
        self.assertEqual(len(staged), 2)
        for s in staged:
            self.assertTrue(os.path.exists(s["path"]))

    def test_scan_and_stage_directory(self):
        # Create folder structure with nested PDFs
        src_dir = os.path.join(self.tmp.name, "src")
        sub_dir = os.path.join(src_dir, "nested")
        os.makedirs(sub_dir, exist_ok=True)

        with open(os.path.join(src_dir, "doc1.pdf"), "wb") as f:
            f.write(DUMMY_PDF_BYTES + b"% doc1\n")
        with open(os.path.join(sub_dir, "doc2.pdf"), "wb") as f:
            f.write(DUMMY_PDF_BYTES + b"% doc2\n")
        with open(os.path.join(src_dir, "notes.txt"), "wb") as f:
            f.write(b"not a pdf")

        staged = batch_uploader.scan_and_stage_directory(
            src_dir,
            destination_dir=self.dest_dir,
            recursive=True,
        )
        self.assertEqual(len(staged), 2)

    def test_fetch_and_stage_urls_or_dois_arxiv(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = DUMMY_PDF_BYTES

        with patch("requests.get", return_value=mock_resp):
            staged = batch_uploader.fetch_and_stage_urls_or_dois(
                ["2401.12345", "https://arxiv.org/abs/2305.67890"],
                destination_dir=self.dest_dir,
            )
            self.assertGreaterEqual(len(staged), 1)


if __name__ == "__main__":
    unittest.main()
