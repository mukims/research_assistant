# research_assistant/shared/ocr.py
"""OCR extraction layer for scanned research papers and handwritten/typed notes.

Provides:
- is_scanned_pdf: Fast heuristic check using PyMuPDF to identify PDFs without a digital text layer.
- get_ocr_reader: Cached EasyOCR reader running locally on MPS GPU or CPU.
- ocr_page_image: OCR execution on an image using EasyOCR or Vision LLM.
- ocr_with_vlm: Multimodal VLM transcription for handwritten notes and formulas.
- extract_ocr: Converts a scanned PDF into a standard Document(..., extraction="ocr")
  that feeds seamlessly into sentence-window chunking, vector indexing, and BM25.
"""

from __future__ import annotations

import os
import re
import tempfile
from typing import Sequence

import numpy as np

from research_assistant import config
from research_assistant.shared.extract import Document, Paragraph, Section, _BIB_HEAD
from research_assistant.shared.log import get_logger
from research_assistant.shared.tokenize import normalize

logger = get_logger("ocr")

_OCR_READER = None


def is_scanned_pdf(pdf_path: str, min_chars_per_page: int | None = None) -> bool:
    """Check if a PDF is a scanned document lacking a digital text layer.

    Returns True if the average extracted digital text across pages is below
    the minimum character threshold (default from config.OCR_MIN_CHARS_PER_PAGE).
    """
    if not os.path.isfile(pdf_path):
        return False

    threshold = min_chars_per_page if min_chars_per_page is not None else config.OCR_MIN_CHARS_PER_PAGE

    try:
        import fitz

        doc = fitz.open(pdf_path)
        try:
            num_pages = len(doc)
            if num_pages == 0:
                return False

            total_chars = 0
            has_images = False
            for page in doc:
                text = page.get_text().strip()
                total_chars += len(text)
                if not has_images and len(page.get_images()) > 0:
                    has_images = True

            avg_chars = total_chars / max(1, num_pages)
            if avg_chars < threshold:
                return True

            return False
        finally:
            doc.close()
    except Exception as exc:
        logger.debug("is_scanned_pdf check failed for %s: %s", pdf_path, exc)
        return False


def get_ocr_reader(lang: Sequence[str] | None = None, gpu: bool | None = None):
    """Lazily instantiate and cache the EasyOCR reader.

    Uses model weights located in config.OCR_MODEL_DIR (data/models/ocr)
    with download disabled to operate safely in sandboxed/offline environments.
    """
    global _OCR_READER
    if _OCR_READER is not None:
        return _OCR_READER

    import easyocr
    import torch

    lang = list(lang) if lang is not None else ["en"]
    if gpu is None:
        gpu = torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )

    model_dir = config.OCR_MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)

    try:
        _OCR_READER = easyocr.Reader(
            lang,
            gpu=gpu,
            model_storage_directory=model_dir,
            download_enabled=False,
        )
    except Exception as exc:
        if gpu:
            logger.warning(
                "Failed to initialize EasyOCR with GPU acceleration (%s); falling back to CPU.",
                exc,
            )
            _OCR_READER = easyocr.Reader(
                lang,
                gpu=False,
                model_storage_directory=model_dir,
                download_enabled=False,
            )
        else:
            raise

    return _OCR_READER


def reset_ocr_reader() -> None:
    """Clear cached EasyOCR reader instance (useful for tests or memory cleanup)."""
    global _OCR_READER
    _OCR_READER = None


def ocr_with_vlm(image: str | bytes | np.ndarray, prompt: str | None = None) -> str:
    """Transcribe an image using a multimodal Vision LLM (e.g. Gemini or local Ollama vision).

    Suitable for handwritten lab notes, whiteboards, or non-standard mathematical formulas.
    """
    from PIL import Image
    from research_assistant.shared.llm import chat

    prompt = (
        prompt
        or "Transcribe all text from this scanned document page accurately. "
        "Preserve mathematical formulas, headings, bullet points, and paragraph structure. "
        "Output only the transcribed content without commentary."
    )

    tmp_path = None
    try:
        if isinstance(image, str) and os.path.isfile(image):
            img_path = image
        else:
            if isinstance(image, bytes):
                import io
                pil_img = Image.open(io.BytesIO(image))
            elif isinstance(image, np.ndarray):
                pil_img = Image.fromarray(image)
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                tmp_path = f.name
                pil_img.save(f, format="PNG")
            img_path = tmp_path

        res = chat(
            messages=[{"role": "user", "content": prompt}],
            images=[img_path],
            temperature=0.0,
        )
        return res.content or ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def ocr_page_image(
    image: str | bytes | np.ndarray,
    engine: str | None = None,
) -> list[str]:
    """Run OCR on a page image and return structured paragraphs.

    Args:
        image: File path, image bytes, or numpy RGB array.
        engine: "easyocr" (default) or "vlm".

    Returns:
        list of cleaned paragraph strings.
    """
    engine = (engine or config.OCR_ENGINE or "easyocr").lower()

    if engine == "vlm":
        text = ocr_with_vlm(image)
        raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        return [normalize(p.replace("\n", " ")) for p in raw_paras if len(p) > 4]

    # EasyOCR mode
    from PIL import Image

    if isinstance(image, str):
        img_np = np.array(Image.open(image).convert("RGB"))
    elif isinstance(image, bytes):
        import io
        img_np = np.array(Image.open(io.BytesIO(image)).convert("RGB"))
    elif isinstance(image, np.ndarray):
        img_np = image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    reader = get_ocr_reader()
    # EasyOCR with paragraph=True returns grouped bounding boxes and combined text
    results = reader.readtext(img_np, paragraph=True)
    if not results:
        # Fallback to non-paragraph detail mode if paragraph mode found nothing
        results_detail = reader.readtext(img_np, paragraph=False)
        results = [([item[0]], item[1]) for item in results_detail]

    # Sort results in top-to-bottom, left-to-right reading order
    # Each item: ([box], text) where box is 4 points [[x0,y0], [x1,y0], [x1,y1], [x0,y1]]
    def _sort_key(item):
        try:
            pts = item[0]
            if isinstance(pts, list) and len(pts) > 0 and isinstance(pts[0], list) and len(pts[0]) > 0:
                first_pt = pts[0]
                if isinstance(first_pt[0], (int, float)):
                    return (first_pt[1], first_pt[0])
                if isinstance(first_pt, list) and len(first_pt) > 0:
                    return (first_pt[0][1], first_pt[0][0])
        except Exception:
            pass
        return (0, 0)

    results.sort(key=_sort_key)

    paragraphs: list[str] = []
    for _bbox, text in results:
        if not text:
            continue
        # De-hyphenate words broken across lines: e.g. "semi-\nconductor" -> "semiconductor"
        cleaned = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
        cleaned = normalize(cleaned.replace("\n", " "))
        if len(cleaned) > 4:
            paragraphs.append(cleaned)

    return paragraphs


def extract_ocr(
    pdf_path: str,
    key: str | None = None,
    engine: str | None = None,
    dpi: int | None = None,
) -> Document:
    """Render PDF pages to images and extract text into a Document(..., extraction='ocr').

    Produces paragraphs with page tracking and cuts references/bibliography
    consistently with PyMuPDF extraction.
    """
    import fitz

    key = key or os.path.basename(pdf_path).strip().replace(" ", "_").lower()
    engine = engine or config.OCR_ENGINE or "easyocr"
    dpi = dpi or config.OCR_DPI or 150

    doc = fitz.open(pdf_path)
    pages: list[list[str]] = []
    try:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
            if pix.n >= 4:
                img_np = img_np[:, :, :3]
            page_paras = ocr_page_image(img_np, engine=engine)
            pages.append(page_paras)
    finally:
        doc.close()

    # Drop repeating running headers/footers repeated on >=30% of pages
    npages = len(pages)
    seen_on: dict[str, int] = {}
    for blocks in pages:
        for b in set(blocks):
            seen_on[b] = seen_on.get(b, 0) + 1
    repeated = {b for b, n in seen_on.items() if npages >= 4 and n >= max(2, 0.3 * npages)}

    flat: list[Paragraph] = []
    for i, blocks in enumerate(pages):
        for b in blocks:
            if b not in repeated:
                flat.append(Paragraph(b, i))

    cut = max((i for i, p in enumerate(flat) if _BIB_HEAD.match(p.text)), default=None)
    body = flat[:cut] if cut is not None else flat

    return Document(
        key=key,
        title=key,
        abstract=[],
        sections=[Section("", "other", body)],
        figures=[],
        extraction="ocr",
        n_bib=0,
    )
