# research_assistant/shared/ingest_v2.py
"""PDF → corpus entries, v2.

extract() → chunk_document() → entries in the dict shape upsert_corpus()
consumes, so the lock, dedupe, id allocation, manifest and BM25 rebuild are
shared with v1 rather than copied. Three entry types are new and carry
`embed_text` (what is embedded) apart from `content` (what is stored, quoted,
cited and verified), and flat `meta`.

Every figure and table with a crop also gets an image entry: the crop embedded
by embeddinggemma-2 (config.EMBED_IMAGES), stored under the caption's text.
"""

from __future__ import annotations

import os

from research_assistant.config import (
    CHUNK_TARGET_CHARS, CHUNK_MAX_CHARS, CHUNK_MIN_CHARS, CHUNK_MIN_ALPHA,
    EMBED_IMAGES, FIGURE_CROP_DPI, IMAGES_DIR, SUMMARY_MAX_CHARS,
)
from research_assistant.shared.chunking import chunk_document
from research_assistant.shared.extract import extract, Document
from research_assistant.shared.figures import (
    FIGURE_IMAGE_PREFIX, crop_name, crop_figure,
)
from research_assistant.shared.log import get_logger

logger = get_logger("ingest_v2")

EXTRACTION_MODES_V2 = ("grobid", "pymupdf", "ocr")


def _pdf_key(pdf_path: str) -> str:
    # Same identity as ingestion.pdf_key(); repeated here to avoid importing
    # ingestion (which imports this module).
    return os.path.basename(pdf_path).strip().replace(" ", "_").lower()


def _header(title: str, section_label: str) -> str:
    return f"Title: {title}. Section: {section_label}. "


def _summary_source(doc: Document) -> str:
    parts = [p.text for p in doc.abstract]
    for kind in ("introduction", "conclusion"):
        for sec in doc.sections:
            if sec.kind == kind:
                parts.extend(p.text for p in sec.paragraphs)
    if not parts:                                   # PyMuPDF path: kind "other" only
        parts = [p.text for s in doc.sections for p in s.paragraphs]
    return "\n\n".join(parts)[:SUMMARY_MAX_CHARS]


def process_pdf_v2(
    pdf_path: str,
    citation_string: str,
    images_dir: str | None = None,
    enable_ocr: bool | None = None,
    ocr_engine: str | None = None,
) -> list[dict]:
    images_dir = images_dir or IMAGES_DIR
    key = _pdf_key(pdf_path)
    doc = extract(pdf_path, key=key, enable_ocr=enable_ocr, ocr_engine=ocr_engine)
    title = doc.title or citation_string
    chunks = chunk_document(doc, CHUNK_TARGET_CHARS, CHUNK_MAX_CHARS,
                            min_chars=CHUNK_MIN_CHARS, min_alpha=CHUNK_MIN_ALPHA)
    figures = {f.id: f for f in doc.figures}

    entries: list[dict] = []
    imaged = failed_images = 0
    embedder = None
    fig_index = 0
    for c in chunks:
        base = {
            "document": key, "citation": citation_string, "page": c.page_first,
            "extraction": doc.extraction,
        }
        meta = {
            "section": c.section, "section_raw": c.section_raw,
            "page_first": c.page_first, "page_last": c.page_last, "seq": c.seq,
            "extraction": doc.extraction,
        }
        if c.type == "text":
            entries.append({**base, "type": "text_chunk", "content": c.text,
                            "embed_text": _header(title, c.section_raw or c.section) + c.text,
                            "meta": meta})
            continue

        # caption chunk — always; crop — cheap, always attempted
        fig = figures.get(c.figure_id)
        image_path = ""
        if fig is not None:
            name = crop_name(key, fig, fig_index)
            fig_index += 1
            try:
                if crop_figure(pdf_path, fig, os.path.join(images_dir, name), FIGURE_CROP_DPI):
                    image_path = name
            except Exception as exc:                # noqa: BLE001
                logger.warning("Crop failed for %s %s: %s", key, fig.label, exc)

        cap_meta = {**meta, "figure_id": c.figure_id, "figure_kind": c.figure_kind,
                    "figure_label": c.figure_label, "image_path": image_path}
        cap_entry = {**base, "type": "caption", "content": c.text,
                     "embed_text": _header(title, c.figure_label) + c.text, "meta": cap_meta}
        entries.append(cap_entry)

        if EMBED_IMAGES and fig is not None and image_path:
            try:
                if embedder is None:
                    from research_assistant.shared.llm import get_embeddings
                    embedder = get_embeddings()
                vector = embedder.embed_image(os.path.join(images_dir, image_path))
            except Exception as exc:                # noqa: BLE001
                logger.error("Image embedding failed for %s %s: %s", key, fig.label, exc)
                failed_images += 1
                continue
            content = FIGURE_IMAGE_PREFIX.format(label=fig.label) + (fig.caption or c.text)
            entries.append({**base, "type": "figure_image", "content": content,
                            "embedding": vector,
                            "meta": {**cap_meta, "extraction": "image"}})
            imaged += 1

    entries.append({"document": key, "citation": citation_string, "page": 0,
                    "type": "summary_source", "content": _summary_source(doc),
                    "extraction": doc.extraction})

    logger.info("✓ %s: %s, %d text chunk(s), %d caption(s), %d image(s) embedded, %d bib entries kept out",
                key, doc.extraction, sum(e["type"] == "text_chunk" for e in entries),
                sum(e["type"] == "caption" for e in entries), imaged, doc.n_bib)
    if failed_images:
        # A figure without its image vector is still found by its caption, but
        # the run should say that some images are missing.
        logger.warning("%s: %d figure image(s) could not be embedded", key, failed_images)
    return entries
