"""Tests for shared.fetch.download_pdf.

A PDF that silently isn't a PDF is the most common failure in this pipeline,
and download_pdf is the one guard against it. It checked the %PDF- magic bytes
on the first chunk, which catches an HTML error page served with HTTP 200 — but
not a body that starts correctly and then stops early. requests' iter_content
ends without raising when a connection drops mid-stream, so a truncated
download was renamed into place and reported as a success; the damage surfaced
much later as a paper that extracts badly.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import requests

from research_assistant.shared import fetch


class FakeResponse:
    def __init__(self, chunks, status_code=200, headers=None, raise_mid_stream=False):
        self._chunks = chunks
        self.status_code = status_code
        self.headers = headers or {}
        self._raise_mid_stream = raise_mid_stream

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, chunk_size=8192):
        for chunk in self._chunks:
            yield chunk
        if self._raise_mid_stream:
            raise requests.ConnectionError("peer reset the connection")


class DownloadTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dest = os.path.join(self.tmp.name, "paper.pdf")

    def _download(self, response):
        with patch.object(fetch.requests, "get", return_value=response):
            return fetch.download_pdf("https://example.org/paper.pdf", self.dest)

    def _siblings(self):
        return sorted(os.listdir(self.tmp.name))


class TestTruncatedBody(DownloadTestCase):
    def test_body_shorter_than_content_length_is_rejected(self):
        ok, reason = self._download(
            FakeResponse([b"%PDF-1.4 ", b"x" * 100], headers={"Content-Length": "5000"})
        )
        self.assertFalse(ok, "a truncated PDF was accepted as a complete download")
        self.assertIn("incomplete", (reason or "").lower())

    def test_truncated_body_does_not_reach_the_destination(self):
        self._download(
            FakeResponse([b"%PDF-1.4 ", b"x" * 100], headers={"Content-Length": "5000"})
        )
        self.assertFalse(os.path.exists(self.dest))

    def test_truncated_body_leaves_no_part_file(self):
        self._download(
            FakeResponse([b"%PDF-1.4 ", b"x" * 100], headers={"Content-Length": "5000"})
        )
        self.assertEqual(self._siblings(), [])

    def test_connection_reset_mid_stream_leaves_no_part_file(self):
        """The .part file used to be orphaned on every mid-stream failure."""
        ok, reason = self._download(
            FakeResponse([b"%PDF-1.4 ", b"x" * 100], raise_mid_stream=True)
        )
        self.assertFalse(ok)
        self.assertEqual(self._siblings(), [])


class TestCompleteBody(DownloadTestCase):
    def test_matching_content_length_is_accepted(self):
        body = b"%PDF-1.4 " + b"x" * 91
        ok, reason = self._download(
            FakeResponse([body], headers={"Content-Length": str(len(body))})
        )
        self.assertTrue(ok, reason)
        with open(self.dest, "rb") as fh:
            self.assertEqual(fh.read(), body)

    def test_absent_content_length_is_still_accepted(self):
        """Chunked transfer encoding sends no length; that is not an error."""
        ok, reason = self._download(FakeResponse([b"%PDF-1.4 ", b"x" * 100]))
        self.assertTrue(ok, reason)
        self.assertTrue(os.path.exists(self.dest))

    def test_unparseable_content_length_is_ignored_not_fatal(self):
        ok, reason = self._download(
            FakeResponse([b"%PDF-1.4 ", b"x" * 100], headers={"Content-Length": "banana"})
        )
        self.assertTrue(ok, reason)


class TestExistingGuards(DownloadTestCase):
    """Behaviour that must survive the added length check."""

    def test_html_error_page_with_status_200_is_rejected(self):
        ok, reason = self._download(
            FakeResponse([b"<html>404</html>"], headers={"Content-Type": "text/html"})
        )
        self.assertFalse(ok)
        self.assertIn("not a PDF", reason)

    def test_non_200_is_rejected(self):
        ok, reason = self._download(FakeResponse([b"%PDF-1.4"], status_code=503))
        self.assertFalse(ok)
        self.assertEqual(reason, "HTTP 503")

    def test_empty_response_is_rejected(self):
        ok, reason = self._download(FakeResponse([]))
        self.assertFalse(ok)
        self.assertEqual(reason, "empty response")


if __name__ == "__main__":
    unittest.main()
