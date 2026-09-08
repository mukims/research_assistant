"""
Shared PDF download helper.

The stream-to-disk-with-validation logic that Agent 0 (seed discovery) and
Agent 2 (reference fetching) both need. Kept in one place so the "the server
returned an HTML error page with HTTP 200" guard behaves identically for both
callers — a PDF that silently isn't a PDF is the single most common failure in
this pipeline.
"""

import os
import re

import requests

from research_assistant.config import UNPAYWALL_EMAIL
from research_assistant.shared.log import get_logger

logger = get_logger("fetch")

# Crossref, arXiv, Unpaywall and Semantic Scholar all ask for identifying
# contact details. Being polite gets you the faster rate-limit pool rather
# than the shared one.
HEADERS = {"User-Agent": f"research_assistant/0.1 (mailto:{UNPAYWALL_EMAIL})"}


def filename_for(key: str) -> str:
    """Stable, filesystem-safe filename derived from a source key."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", key)[:120] + ".pdf"


def _is_pdf(first_bytes: bytes) -> bool:
    """Servers frequently return an HTML error page with status 200."""
    return first_bytes[:5] == b"%PDF-"


def _expected_length(headers) -> int | None:
    """Content-Length, when it can be trusted as the decoded body's size.

    A transfer-encoded or content-encoded response reports the size of the
    encoded stream while iter_content yields decoded bytes, so the two are not
    comparable and the check is skipped rather than guessed at.
    """
    if headers.get("Content-Encoding") or headers.get("Transfer-Encoding"):
        return None
    try:
        return int(headers.get("Content-Length"))
    except (TypeError, ValueError):
        return None


def _discard(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def download_pdf(url: str, dest_path: str) -> tuple[bool, str | None]:
    """Stream a PDF to *dest_path*, refusing anything that isn't actually a PDF.

    Returns ``(ok, reason)``. On success ``reason`` is ``None``; on failure
    ``dest_path`` is left untouched (the download goes to a ``.part`` file that
    is only renamed into place once the whole body is written, and is removed
    when it is not).

    Two things can arrive that are not a usable PDF. A server can send an HTML
    error page with HTTP 200 — caught by the magic-byte check. Or the body can
    start correctly and stop early: iter_content ends without raising when a
    connection drops mid-stream, so a short read used to be renamed into place
    and reported as a success, surfacing much later as a paper that extracts
    badly. Comparing the bytes written against Content-Length catches that.

    Network errors are caught and returned as a failure reason rather than
    raised, so a caller trying several sources in turn moves on to the next one
    instead of aborting the whole fetch.
    """
    tmp_path = dest_path + ".part"
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=30) as res:
            if res.status_code != 200:
                return False, f"HTTP {res.status_code}"

            chunks = res.iter_content(chunk_size=8192)
            try:
                first = next(chunks)
            except StopIteration:
                return False, "empty response"

            if not _is_pdf(first):
                ctype = res.headers.get("Content-Type", "unknown")
                return False, f"not a PDF (content-type {ctype})"

            written = 0
            with open(tmp_path, "wb") as fh:
                fh.write(first)
                written += len(first)
                for chunk in chunks:
                    fh.write(chunk)
                    written += len(chunk)

            expected = _expected_length(res.headers)
            if expected is not None and written < expected:
                _discard(tmp_path)
                return False, (
                    f"incomplete download ({written} of {expected} bytes)"
                )
    except requests.RequestException as exc:
        _discard(tmp_path)
        return False, f"download error: {exc}"
    except BaseException:
        _discard(tmp_path)
        raise

    os.replace(tmp_path, dest_path)
    return True, None
