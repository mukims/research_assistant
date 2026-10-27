"""PDF metadata inspection and extraction utilities.

Extracts titles, DOIs, arXiv IDs, and validates PDF file contents
without requiring any agent-specific or retrieval dependencies.
"""

from __future__ import annotations

import os
import re

from research_assistant.shared.source_key import normalise_doi

# ─── Direct-file inspection regexes ──────────────────────────────────────────

_DOI_RE = re.compile(
    rb"(?i)(?:doi[:\s/]|https?://doi\.org/|doi\.org\\/)(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)"
)
_ARXIV_RE_BYTES = re.compile(rb"(?i)arxiv[:\s/]+([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)")
_PDF_TITLE_RE = re.compile(rb"/Title\s*(?:\((.*?)\)|<([0-9a-fA-F]+)>)", re.DOTALL)
_XMP_TITLE_RE = re.compile(
    rb"<dc:title>[\s\S]*?<rdf:li[^>]*>([\s\S]*?)</rdf:li>", re.IGNORECASE
)
_XMP_TITLE_FALLBACK_RE = re.compile(rb"<dc:title>([\s\S]*?)</dc:title>", re.IGNORECASE)
_XMP_DOI_RE = re.compile(
    rb"<(?:prism:doi|dc:identifier)>\s*(?:doi:)?\s*(10\.\d{4,9}/[^<\s]+)\s*</",
    re.IGNORECASE,
)

_HEADER_JUNK_RE = re.compile(
    r"(?i)^(?:arxiv[:\s/]|https?://|doi[:\s/]|10\.\d{4,9}/|"
    r"journal of|physical review|nature\b|science\b|proceedings of|"
    r"ieee\b|springer|elsevier|volume\b|vol\.\b|issue\b|no\.\b|pp\.\b|pages\b|"
    r"published|accepted|received|issn\b|typeset|draft version)",
)

# What a title field holds when it is not a title (_clean_title). Narrower than
# _HEADER_JUNK_RE on purpose: "Nature of the ..." and "Science and ..." are titles.
_TITLE_ID_RE = re.compile(r"(?i)^(?:arxiv[:\s/]|https?://|doi[:\s/]|10\.\d{4,9}/)")
_TITLE_FILE_RE = re.compile(
    r"(?i)^microsoft (?:word|powerpoint|excel)\b|\.(?:docx?|tex|rtf|indd|odt)$"
)
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")  # no spaces between words
_TITLE_BANNER_WORDS = frozenset({
    "paper", "article", "letter", "letters", "research", "original", "open", "access",
    "preprint", "press", "under", "review", "accepted", "manuscript", "version", "draft",
    "submitted", "published", "proof",
})


def _decode_pdf_title(raw_str: bytes | str) -> str | None:
    """Decode PDF title string from literal bytes/str or hex string."""
    if isinstance(raw_str, str):
        raw_str = raw_str.encode("utf-8", "ignore")
    if not raw_str:
        return None
    if raw_str.startswith(b"\xfe\xff"):
        try:
            return raw_str.decode("utf-16-be").strip()
        except Exception:
            pass
    elif raw_str.startswith(b"\xff\xfe"):
        try:
            return raw_str.decode("utf-16-le").strip()
        except Exception:
            pass
    decoded = (
        raw_str.replace(b"\\(", b"(")
        .replace(b"\\)", b")")
        .replace(b"\\\\", b"\\")
        .replace(b"\\r", b" ")
        .replace(b"\\n", b" ")
        .replace(b"\\t", b" ")
    )
    try:
        text = decoded.decode("utf-8", "ignore").strip()
    except Exception:
        text = decoded.decode("latin-1", "ignore").strip()
    return re.sub(r"\s+", " ", text) if text else None


def _clean_title(title: str | None) -> str | None:
    if not title:
        return None
    t = title.strip()
    t = (
        t.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    t = re.sub(r"\s+", " ", t).strip()
    words = re.findall(r"[^\W\d_]{3,}", t)
    if (
        len(t) > 3
        and t.lower() not in ("untitled", "default", "none", "nan", "untitled document")
        and not _TITLE_ID_RE.match(t)
        and not _TITLE_FILE_RE.search(t)
        and (len(words) >= 2 or len(_CJK_RE.findall(t)) >= 4)
        and not (words and all(w.lower() in _TITLE_BANNER_WORDS for w in words))
    ):
        return t
    return None


def _clean_doi(raw_doi: str | None) -> str | None:
    if not raw_doi:
        return None
    d = raw_doi.strip().rstrip(".,;)>]").strip()
    m = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", d)
    if m:
        d = m.group(0).rstrip(".,;)>]")
        return normalise_doi(d) or d.lower()
    return None


def _is_pdf_content(first_chunk: bytes) -> bool:
    """PDF header (%PDF-) anywhere in first 1024 bytes per ISO 32000-1 §7.5.2."""
    return b"%PDF-" in first_chunk[:1024]


def _inspect_pdf(
    file_path: str, raw_bytes: bytes | None = None
) -> tuple[str | None, str | None, str | None]:
    """Extract (title, doi, arxiv_id) from PDF metadata or text if available."""
    title, doi, arxiv_id = None, None, None

    # 1. Try PyMuPDF if installed
    doc = None
    try:
        import fitz

        if file_path and os.path.exists(file_path):
            doc = fitz.open(file_path)
        elif raw_bytes:
            doc = fitz.open(stream=raw_bytes, filetype="pdf")

        if doc:
            meta = doc.metadata or {}
            t = _clean_title(meta.get("title"))
            if t:
                title = t

            for page_num in range(min(2, len(doc))):
                text = doc[page_num].get_text("text")
                if not arxiv_id:
                    m_arx = re.search(
                        r"(?i)arxiv[:\s/]+([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)", text
                    )
                    if m_arx:
                        arxiv_id = m_arx.group(1)
                if not doi:
                    m_doi = re.search(
                        r"(?i)(?:doi[:\s/]|https?://doi\.org/)(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)",
                        text,
                    )
                    if m_doi:
                        doi = _clean_doi(m_doi.group(1))
                if not title and page_num == 0:
                    page = doc[page_num]
                    by_size = {}
                    for block in page.get_text("dict", sort=True)["blocks"]:
                        for pdf_line in block.get("lines", []):
                            spans = [
                                s for s in pdf_line["spans"] if s["text"].strip()
                            ]
                            if spans and pdf_line["bbox"][1] < page.rect.height / 2:
                                size = round(max(s["size"] for s in spans))
                                by_size.setdefault(size, []).append(
                                    "".join(s["text"] for s in pdf_line["spans"]).strip()
                                )
                    for size in sorted(by_size, reverse=True)[:3]:
                        joined = " ".join(by_size[size])
                        cand_title = (
                            None
                            if _HEADER_JUNK_RE.search(joined)
                            else _clean_title(joined)
                        )
                        if cand_title and len(cand_title) <= 300:
                            title = cand_title
                            break
                if not title and page_num == 0:
                    lines = [
                        line.strip()
                        for line in text.splitlines()
                        if len(line.strip()) > 3
                    ]
                    for candidate_line in lines[:10]:
                        if _HEADER_JUNK_RE.search(candidate_line):
                            continue
                        cand_title = _clean_title(candidate_line)
                        if cand_title and len(cand_title) > 5:
                            title = cand_title
                            break
    except Exception:
        pass
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass

    # 2. Fallback regex on raw bytes
    if raw_bytes is None and file_path and os.path.exists(file_path):
        try:
            size = os.path.getsize(file_path)
            with open(file_path, "rb") as f:
                if size <= 1024 * 1024:
                    raw_bytes = f.read()
                else:
                    head = f.read(512 * 1024)
                    f.seek(max(0, size - 512 * 1024))
                    tail = f.read(512 * 1024)
                    raw_bytes = head + b"\n" + tail
        except OSError:
            raw_bytes = b""
    elif raw_bytes is None:
        raw_bytes = b""

    # Look for arXiv ID
    if not arxiv_id and raw_bytes:
        m_arx = _ARXIV_RE_BYTES.search(raw_bytes[:128 * 1024])
        if m_arx:
            try:
                arxiv_id = m_arx.group(1).decode("utf-8", "ignore")
            except Exception:
                pass

    # Look for DOI
    if not doi and raw_bytes:
        m_xmp_doi = _XMP_DOI_RE.search(raw_bytes)
        if m_xmp_doi:
            try:
                doi = _clean_doi(m_xmp_doi.group(1).decode("utf-8", "ignore"))
            except Exception:
                pass
        if not doi:
            m_doi = _DOI_RE.search(raw_bytes[:128 * 1024])
            if m_doi:
                try:
                    doi = _clean_doi(m_doi.group(1).decode("utf-8", "ignore"))
                except Exception:
                    pass

    # Look for Title
    if not title and raw_bytes:
        m_xmp = _XMP_TITLE_RE.search(raw_bytes) or _XMP_TITLE_FALLBACK_RE.search(
            raw_bytes
        )
        if m_xmp:
            try:
                cand = _clean_title(m_xmp.group(1).decode("utf-8", "ignore"))
                if cand:
                    title = cand
            except Exception:
                pass

        if not title:
            m_title = _PDF_TITLE_RE.search(raw_bytes)
            if m_title:
                raw_t = m_title.group(1)
                hex_t = m_title.group(2)
                if hex_t:
                    try:
                        raw_bytes_t = bytes.fromhex(hex_t.decode("ascii"))
                        cand = _decode_pdf_title(raw_bytes_t)
                        title = _clean_title(cand)
                    except Exception:
                        pass
                elif raw_t:
                    cand = _decode_pdf_title(raw_t)
                    title = _clean_title(cand)

    return title, doi, arxiv_id


inspect_pdf_metadata = _inspect_pdf
is_pdf_content = _is_pdf_content
clean_title = _clean_title
clean_doi = _clean_doi
decode_pdf_title = _decode_pdf_title
