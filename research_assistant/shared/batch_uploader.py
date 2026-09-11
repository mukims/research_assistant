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
from typing import BinaryIO, Any

from research_assistant.agents.agent0_discoverer import (
    _inspect_pdf,
    _is_pdf_content,
    _clean_title,
)
from research_assistant.shared.source_key import normalise_doi
from research_assistant.shared.fetch import filename_for
from research_assistant.config import RAW_DIR
from research_assistant.shared.log import get_logger

logger = get_logger("batch_uploader")


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
        if name_lower.endswith('.zip') or data[:4] == b'PK':
            logger.info('Extracting ZIP archive: %s (%d bytes)', name, len(data))
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for member in zf.infolist():
                        if member.is_dir():
                            continue
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
