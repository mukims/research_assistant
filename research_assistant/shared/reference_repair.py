# research_assistant/shared/reference_repair.py
"""Find and repair stored papers whose text chunks hold reference lists.

Papers ingested before the extractor learned to drop unlabelled reference runs
(shared.bibliography) can have chunks that are really bibliographies. A
candidate is a stored PDF with at least one reference-heavy ``text_chunk``. It
is re-chunked from its Docling cache only, never by a new conversion, and is
repaired only when the fresh chunks carry fewer reference-heavy chunks than the
stored ones. A repair replaces that paper's ``text_chunk`` rows and nothing
else: captions, formulas, figure images and the summary stay.

Driven by scripts/repair_reference_leak.py (dry run by default).
"""

from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional
from unittest import mock

from research_assistant import config
from research_assistant.shared.bibliography import is_reference_heavy_chunk
from research_assistant.shared.log import get_logger

logger = get_logger(__name__)

_PAGE = 5000


@dataclass
class LeakRepair:
    document: str
    pdf_path: str
    stored_chunks: int
    stored_heavy: int
    fresh_chunks: int
    fresh_heavy: int


@dataclass
class LeakSkip:
    document: str
    reason: str   # "no_docling_cache" | "no_fresh_chunks" | "not_improved" | "extract_failed: <error>"


class NoDoclingCache(Exception):
    """The paper has no Docling cache, so it cannot be re-chunked without a conversion."""


@contextmanager
def _cache_only():
    """Make any route other than the Docling cache fail loudly instead of re-parsing the PDF.

    extract_docling converts the PDF when its cache is unreadable, and process_pdf_v2
    falls back to the legacy extractor when Docling fails; both are refused here, and
    image embedding is off (only text chunks are kept).
    """
    from research_assistant.shared import docling_extract, ingest_v2

    def refuse(*_a, **_k):
        raise RuntimeError("refused: the repair reads the Docling cache only")

    with mock.patch.object(docling_extract, "get_digital_converter", refuse), \
         mock.patch.object(docling_extract, "get_ocr_converter", refuse), \
         mock.patch.object(ingest_v2, "extract", refuse), \
         mock.patch.object(ingest_v2, "EMBED_IMAGES", False):
        yield


def _fresh_chunks_from_cache(document: str, pdf_path: str, citation: str) -> list[dict]:
    """The paper's ``text_chunk`` entries, rebuilt from its Docling cache with the current extractor."""
    from research_assistant.shared import ingest_v2
    from research_assistant.shared.docling_extract import _docling_cache_path

    if not os.path.isfile(_docling_cache_path(pdf_path)):
        raise NoDoclingCache(pdf_path)
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")
    with tempfile.TemporaryDirectory(prefix="refrepair_img_") as images_dir, _cache_only():
        entries = ingest_v2.process_pdf_v2(
            pdf_path, citation_string=citation, images_dir=images_dir, extractor="docling",
        )
    return [e for e in entries if e.get("type") == "text_chunk"]


def _stored_text_chunks(col) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Every stored ``text_chunk`` text per document, and each document's citation_source."""
    texts: dict[str, list[str]] = defaultdict(list)
    citations: dict[str, str] = {}
    offset = 0
    while True:
        batch = col.get(where={"type": "text_chunk"}, include=["documents", "metadatas"],
                        limit=_PAGE, offset=offset)
        docs = (batch or {}).get("documents") or []
        metas = (batch or {}).get("metadatas") or []
        if not docs:
            break
        for text, meta in zip(docs, metas):
            document = (meta or {}).get("document")
            if not document:
                continue
            texts[document].append(text or "")
            citations.setdefault(document, (meta or {}).get("citation_source") or "")
        if len(docs) < _PAGE:
            break
        offset += _PAGE
    return texts, citations


def _pdf_path(index, document: str) -> str:
    entry = index.resolve(document) if index is not None else None
    if entry is not None and entry.path and os.path.isfile(entry.path):
        return entry.path
    return os.path.join(config.PULLED_PDFS_DIR, document)


def scan_reference_leaks(
    col, index, *, fresh_chunks: Optional[Callable[[str, str], list[dict]]] = None,
) -> tuple[list[LeakRepair], list[LeakSkip], dict[str, list[dict]]]:
    """Plan the repair: which stored papers get cleaner chunks from their Docling cache.

    Returns (repairs, skips, fresh) where ``fresh`` maps each repaired document to its
    fresh ``text_chunk`` entries. Reads the store; writes nothing.
    """
    texts, citations = _stored_text_chunks(col)
    repairs: list[LeakRepair] = []
    skips: list[LeakSkip] = []
    fresh: dict[str, list[dict]] = {}

    for document in sorted(texts):
        if not document.lower().endswith(".pdf"):
            continue
        stored = texts[document]
        stored_heavy = sum(1 for t in stored if is_reference_heavy_chunk(t))
        if stored_heavy == 0:
            continue
        pdf_path = _pdf_path(index, document)
        try:
            if fresh_chunks is not None:
                entries = fresh_chunks(document, pdf_path)
            else:
                entries = _fresh_chunks_from_cache(document, pdf_path, citations.get(document, ""))
        except NoDoclingCache:
            skips.append(LeakSkip(document, "no_docling_cache"))
            continue
        except Exception as exc:                      # noqa: BLE001 — one paper never stops the scan
            skips.append(LeakSkip(document, f"extract_failed: {exc}"))
            continue
        entries = [{**e, "document": document} for e in entries if e.get("type") == "text_chunk"]
        if not entries:
            skips.append(LeakSkip(document, "no_fresh_chunks"))
            continue
        fresh_heavy = sum(1 for e in entries if is_reference_heavy_chunk(e.get("content") or ""))
        if fresh_heavy >= stored_heavy:
            skips.append(LeakSkip(document, "not_improved"))
            continue
        repairs.append(LeakRepair(document, pdf_path, len(stored), stored_heavy, len(entries), fresh_heavy))
        fresh[document] = entries
    return repairs, skips, fresh


def _count_text_chunks(col, document: str) -> int:
    batch = col.get(where={"$and": [{"document": document}, {"type": "text_chunk"}]}, include=["metadatas"])
    return len((batch or {}).get("ids") or (batch or {}).get("metadatas") or [])


def _table_version(col) -> int:
    t = getattr(col, "table", col)
    v = getattr(t, "version", 0)
    if callable(v):
        v = v()
    return int(v) if v is not None else 0


def apply_reference_repair(repairs: list[LeakRepair], fresh: dict[str, list[dict]], col, *,
                           upsert: Optional[Callable[[list[dict]], object]] = None) -> dict:
    """Replace each repaired paper's ``text_chunk`` rows with its fresh entries.

    The caller holds the ingest lock and rebuilds the full-text index afterwards. A paper
    without fresh entries is never touched. A paper whose rows were deleted but whose
    upsert failed is reported as ``deleted_not_reinserted`` so it can be restored from
    the backup; the other papers continue.
    """
    if upsert is None:
        from research_assistant.shared.ingestion import upsert_corpus as upsert

    result = {"versions_before": {"papers": _table_version(col)}, "repaired": [], "errors": []}
    for r in repairs:
        entries = fresh.get(r.document) or []
        if not entries:
            result["errors"].append({"document": r.document, "error": "no_fresh_chunks: left untouched"})
            continue
        try:
            col.delete(where={"$and": [{"document": r.document}, {"type": "text_chunk"}]})
        except Exception as exc:                      # noqa: BLE001
            result["errors"].append({"document": r.document, "error": f"delete_failed: {exc}"})
            continue
        try:
            inserted = upsert(entries)
        except Exception as exc:                      # noqa: BLE001
            logger.error("%s: text chunks deleted but not reinserted: %s", r.document, exc)
            result["errors"].append({"document": r.document, "error": f"deleted_not_reinserted: {exc}"})
            continue
        # upsert_corpus stops early on a pending cancel and still returns the full count,
        # so trust only what the store now holds.
        stored = _count_text_chunks(col, r.document)
        if stored == 0 or (isinstance(inserted, int) and stored < inserted):
            logger.error("%s: only %d of %d text chunks stored after the repair", r.document, stored, len(entries))
            result["errors"].append({"document": r.document,
                                     "error": f"deleted_not_reinserted: {stored} of {len(entries)} text chunks stored"})
            continue
        result["repaired"].append(r.document)
    return result
