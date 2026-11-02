"""
Batch Uploader & Staging Module (batch_uploader.py)

Safely unpacks, validates, inspects, and stages uploaded PDF research papers
and ZIP archives for direct indexing into ChromaDB and the BM25 vector index.
"""

import io
import os
import re
import hashlib
import zipfile
import ipaddress
import socket
from urllib.parse import urlparse
from typing import Any

from research_assistant.shared.pdf_metadata import (
    _inspect_pdf,
    _is_pdf_content,
)
from research_assistant.shared.source_key import normalise_doi
from research_assistant.shared.fetch import filename_for
from research_assistant.config import RAW_DIR
from research_assistant.shared.log import get_logger

logger = get_logger("batch_uploader")

MAX_TOTAL_UNCOMPRESSED_SIZE = 500 * 1024 * 1024  # 500 MB
MAX_SINGLE_FILE_SIZE = 100 * 1024 * 1024        # 100 MB
MAX_ZIP_FILE_COUNT = 500                        # 500 files


def is_safe_path(path: str, allowed_roots: list[str] | None = None) -> bool:
    """Validate that path does not traverse outside permitted directories or touch system roots."""
    if not path or not isinstance(path, str):
        return False
    if "\0" in path:
        return False
    norm_path = path.replace("\\", "/").strip()
    try:
        real_path = os.path.realpath(os.path.expanduser(norm_path))
    except (ValueError, OSError):
        return False

    forbidden = [
        "/etc", "/bin", "/sbin", "/usr", "/var/log", "/dev", "/proc", "/sys",
        "/private/etc", "/private/var/log", "/private/var/root", "/private/var/db",
        "/System", "/Library",
    ]
    if real_path == "/" or any(real_path == fb or real_path.startswith(fb + "/") for fb in forbidden):
        return False

    if allowed_roots is None:
        import tempfile
        allowed_roots = [
            os.getcwd(),
            os.path.expanduser("~"),
            tempfile.gettempdir(),
            "/tmp",
        ]
    try:
        real_allowed = [os.path.realpath(os.path.expanduser(r)) for r in allowed_roots if r]
    except (ValueError, OSError):
        return False
    return any(real_path == r or real_path.startswith(r.rstrip(os.sep) + os.sep) for r in real_allowed)


_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")


def is_safe_url(url: str) -> bool:
    """Validate that the URL uses HTTP/HTTPS and resolves to a public, non-private IP."""
    try:
        parsed = urlparse(url.strip())
        if parsed.scheme.lower() not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        if hostname.lower() in ("localhost", "127.0.0.1", "::1", "0.0.0.0", "metadata.google.internal"):
            return False
        addr_info = socket.getaddrinfo(hostname, None)
        for family, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or not ip.is_global
                or (ip.version == 4 and ip in _CGNAT_NETWORK)
                or (bool(getattr(ip, "ipv4_mapped", None)) and ip.ipv4_mapped in _CGNAT_NETWORK)
            ):
                return False
        return True
    except Exception:
        return False


def _safe_filename(name: str) -> str:
    """Sanitize filename to avoid path traversal or problematic characters."""
    base = os.path.basename(name.replace("\\", "/"))
    clean = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    return clean or "uploaded_paper.pdf"


def _extract_from_bytes(
    content: bytes,
    original_name: str,
    destination_dir: str,
) -> dict | None:
    """Validate and save a single PDF byte payload, extracting metadata."""
    if not _is_pdf_content(content[:1024]):
        logger.warning('Skipping non-PDF content for %s', original_name)
        return None

    stem = os.path.splitext(os.path.basename(original_name))[0]
    clean_stem = re.sub(r'[_-]+', ' ', stem).strip()
    title, doi, arxiv_id = _inspect_pdf('', raw_bytes=content)

    if arxiv_id:
        key = f'arxiv:{arxiv_id.strip().lower()}'
    elif doi:
        norm_d = normalise_doi(doi) or doi.lower().strip()
        key = f'doi:{norm_d}'
    else:
        digest = hashlib.sha1(content).hexdigest()[:16]
        key = f'file:{digest}'

    effective_title = title or clean_stem or 'Untitled Paper'
    safe_name = filename_for(key) if key else _safe_filename(original_name)
    if not safe_name.lower().endswith('.pdf'):
        safe_name += '.pdf'

    dest_path = os.path.join(destination_dir, safe_name)
    os.makedirs(destination_dir, exist_ok=True)

    try:
        with open(dest_path, 'wb') as fh:
            fh.write(content)
    except OSError as exc:
        logger.error('Failed writing staged PDF %s: %s', dest_path, exc)
        return None

    return {
        'path': dest_path,
        'filename': original_name,
        'title': effective_title,
        'key': key,
        'doi': doi,
        'arxiv_id': arxiv_id,
        'size': len(content),
    }


def unpack_and_stage_uploads(
    uploaded_items: list[Any],
    destination_dir: str = RAW_DIR,
) -> list[dict]:
    """
    Unpacks, validates, inspects, and stages uploaded files (PDFs and ZIP archives).

    Args:
        uploaded_items:  List of Streamlit UploadedFile objects, file paths, or file-like objects.
        destination_dir: Target directory where validated PDFs are staged (default: RAW_DIR).

    Returns:
        List of dicts describing staged papers:
        [{'path': ..., 'filename': ..., 'title': ..., 'key': ..., 'doi': ..., 'arxiv_id': ..., 'size': ...}, ...]
    """
    staged = []
    seen_keys = set()

    for item in uploaded_items:
        if isinstance(item, tuple) and len(item) == 2:
            item_obj, name = item
        else:
            item_obj = item
            name = getattr(item, 'name', '') or (str(item) if isinstance(item, (str, os.PathLike)) else 'file.pdf')

        name_lower = name.lower()

        # Read content bytes
        if hasattr(item_obj, 'read') and hasattr(item_obj, 'seek'):
            item_obj.seek(0)
            data = item_obj.read()
        elif hasattr(item_obj, 'getbuffer'):
            data = bytes(item_obj.getbuffer())
        elif isinstance(item_obj, (str, os.PathLike)) and os.path.exists(item_obj):
            try:
                with open(item_obj, 'rb') as f:
                    data = f.read()
            except OSError as exc:
                logger.error('Could not read file %s: %s', item_obj, exc)
                continue
        elif isinstance(item_obj, (bytes, bytearray)):
            data = bytes(item_obj)
        else:
            logger.warning('Unrecognized item type: %s', type(item_obj))
            continue

        if not data:
            logger.warning('Empty item skipped: %s', name)
            continue

        # Check if ZIP archive
        if name_lower.endswith('.zip') or data.startswith(b'PK\x03\x04'):
            logger.info('Extracting ZIP archive: %s (%d bytes)', name, len(data))
            try:
                total_uncompressed = 0
                file_count = 0
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for member in zf.infolist():
                        if member.is_dir():
                            continue
                        file_count += 1
                        if file_count > MAX_ZIP_FILE_COUNT:
                            logger.error("ZIP bomb detected: exceeds max file count (%d)", MAX_ZIP_FILE_COUNT)
                            break
                        if member.file_size > MAX_SINGLE_FILE_SIZE:
                            logger.error("ZIP member %s exceeds max single file size (%d bytes)", member.filename, MAX_SINGLE_FILE_SIZE)
                            continue
                        total_uncompressed += member.file_size
                        if total_uncompressed > MAX_TOTAL_UNCOMPRESSED_SIZE:
                            logger.error("ZIP bomb detected: exceeds max total uncompressed size (%d bytes)", MAX_TOTAL_UNCOMPRESSED_SIZE)
                            break
                        m_name = member.filename
                        # Skip hidden / macOS metadata
                        if m_name.startswith('__MACOSX') or '/.' in m_name or os.path.basename(m_name).startswith('.'):
                            continue
                        if not m_name.lower().endswith('.pdf'):
                            continue

                        pdf_bytes = zf.read(member)
                        rec = _extract_from_bytes(pdf_bytes, m_name, destination_dir)
                        if rec and rec['key'] not in seen_keys:
                            seen_keys.add(rec['key'])
                            staged.append(rec)
            except (zipfile.BadZipFile, OSError) as exc:
                logger.error('Failed reading ZIP file %s: %s', name, exc)
                continue
        else:
            # Single PDF file
            rec = _extract_from_bytes(data, name, destination_dir)
            if rec and rec['key'] not in seen_keys:
                seen_keys.add(rec['key'])
                staged.append(rec)

    logger.info('Staged %d valid research paper(s) into %s', len(staged), destination_dir)
    return staged


def scan_and_stage_directory(
    folder_path: str,
    destination_dir: str = RAW_DIR,
    recursive: bool = True,
) -> list[dict]:
    """Scan a local directory for all PDF documents, inspect, and stage them.

    Args:
        folder_path:     Root path of directory on disk to scan.
        destination_dir: Target directory where validated PDFs are staged (default: RAW_DIR).
        recursive:       Whether to scan subdirectories recursively.

    Returns:
        List of dicts describing staged papers.
    """
    real_folder = os.path.realpath(os.path.expanduser(folder_path.strip()))
    if not is_safe_path(real_folder):
        logger.error("Directory traversal blocked: %s is not a safe workspace path", folder_path)
        return []
    if not os.path.isdir(real_folder):
        logger.warning("Directory does not exist: %s", real_folder)
        return []

    pdf_files: list[str] = []
    if recursive:
        for root, _, files in os.walk(real_folder):
            for f in files:
                if f.lower().endswith(".pdf") and not f.startswith("."):
                    pdf_files.append(os.path.join(root, f))
    else:
        for f in os.listdir(real_folder):
            full_p = os.path.join(real_folder, f)
            if os.path.isfile(full_p) and f.lower().endswith(".pdf") and not f.startswith("."):
                pdf_files.append(full_p)

    logger.info("Found %d PDF file(s) in %s (recursive=%s)", len(pdf_files), real_folder, recursive)
    return unpack_and_stage_uploads(pdf_files, destination_dir=destination_dir)


def fetch_and_stage_urls_or_dois(
    items: list[str],
    destination_dir: str = RAW_DIR,
) -> list[dict]:
    """Fetch papers given a list of URLs, arXiv IDs, or DOIs and stage them.

    Args:
        items:           List of strings (URLs, DOIs like 10.1103/..., or arXiv IDs like 2401.12345).
        destination_dir: Target directory where fetched PDFs are staged.

    Returns:
        List of dicts describing successfully fetched and staged papers.
    """
    import requests

    staged: list[dict] = []
    seen_keys: set[str] = set()

    for raw in items:
        token = raw.strip()
        if not token or token.startswith("#"):
            continue

        data: bytes | None = None
        orig_name: str = "fetched_paper.pdf"

        # Check if URL
        if token.startswith(("http://", "https://")):
            url = token
            # Convert arXiv abs to pdf link if applicable
            if "arxiv.org/abs/" in url:
                url = url.replace("arxiv.org/abs/", "arxiv.org/pdf/") + ".pdf"
            if not is_safe_url(url):
                logger.warning("SSRF protection: rejected private, loopback, or invalid URL: %s", url)
                continue
            orig_name = os.path.basename(url.split("?")[0]) or "url_paper.pdf"
            try:
                resp = requests.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Marvin-Citebot/1.0"},
                    timeout=30,
                )
                if resp.status_code == 200 and _is_pdf_content(resp.content[:1024]):
                    data = resp.content
                else:
                    logger.warning("URL %s did not return valid PDF content (status %d)", url, resp.status_code)
            except Exception as exc:
                logger.warning("Failed fetching URL %s: %s", url, exc)

        # Check if arXiv ID (e.g. 2401.12345 or arXiv:2401.12345)
        elif re.match(r"^(?:arxiv:)?\d{4}\.\d{4,5}(?:v\d+)?$", token, re.I):
            aid = re.sub(r"^arxiv:", "", token, flags=re.I).strip()
            url = f"https://arxiv.org/pdf/{aid}.pdf"
            orig_name = f"arxiv_{aid}.pdf"
            try:
                resp = requests.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 Marvin-Citebot/1.0"},
                    timeout=30,
                )
                if resp.status_code == 200 and _is_pdf_content(resp.content[:1024]):
                    data = resp.content
            except Exception as exc:
                logger.warning("Failed fetching arXiv:%s: %s", aid, exc)

        # Check if DOI (e.g. 10.xxxx/yyyy)
        elif token.startswith("10.") and "/" in token:
            doi = token
            # Try Unpaywall / Europe PMC via Agent 2
            try:
                from research_assistant.agents import agent2_fetcher

                tmp_dest = os.path.join(destination_dir, "temp_doi_fetch.pdf")
                ok = agent2_fetcher.try_unpaywall(doi, tmp_dest)
                if not ok:
                    ok = agent2_fetcher.try_europepmc(doi, {"doi": doi}, tmp_dest)
                if ok and os.path.exists(tmp_dest):
                    with open(tmp_dest, "rb") as fh:
                        data = fh.read()
                    os.unlink(tmp_dest)
                    orig_name = f"doi_{doi.replace('/', '_')}.pdf"
            except Exception as exc:
                logger.warning("Failed resolving DOI %s: %s", doi, exc)

        if data:
            rec = _extract_from_bytes(data, orig_name, destination_dir)
            if rec and rec["key"] not in seen_keys:
                seen_keys.add(rec["key"])
                staged.append(rec)

    logger.info("Successfully fetched and staged %d paper(s)", len(staged))
    return staged

