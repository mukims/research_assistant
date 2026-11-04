# research_assistant/shared/docling_extract.py
"""Docling-based document extraction for scientific papers.

Converts full-text PDFs into canonical Document representations:
- Clean paragraph hierarchy and section kind categorization
- Accurate multi-panel and vector-plot figure detection + high-res image crops
- Table structure recognition with clean Markdown output (never cropped as images)
- Mathematical formula extraction
- Dynamic preflight OCR routing (native Apple Vision on macOS, EasyOCR on Linux)
- Reading-order traversal with automatic bibliography exclusion
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Optional, Tuple

from research_assistant import config
from research_assistant.shared.extract import (
    Document,
    Figure,
    Formula,
    Paragraph,
    Section,
    _BIB_HEAD,
    normalise_section_kind,
)
from research_assistant.shared.log import get_logger

try:
    from docling_core.types.doc import DocItemLabel
except ImportError:
    DocItemLabel = None

logger = get_logger("docling_extract")

_DIGITAL_CONVERTER = None
_OCR_CONVERTER = None


def needs_ocr(pdf_path: str, min_chars_per_page: Optional[int] = None) -> bool:
    """Fast PyMuPDF preflight check to determine if a PDF requires full-page OCR.

    Checks:
    1. Average characters per page below threshold (default from config.OCR_MIN_CHARS_PER_PAGE).
    2. Hybrid scans where >= 20% of pages contain images and < 20 digital characters.
    3. Encoding errors where > 5% of digital text consists of replacement characters (\\ufffd).
    """
    if not os.path.isfile(pdf_path):
        return False

    threshold = (
        min_chars_per_page
        if min_chars_per_page is not None
        else config.OCR_MIN_CHARS_PER_PAGE
    )

    try:
        import fitz

        doc = fitz.open(pdf_path)
        try:
            num_pages = len(doc)
            if num_pages == 0:
                return False

            total_chars = 0
            empty_or_image_pages = 0
            replacement_chars = 0

            for page in doc:
                text = page.get_text() or ""
                stripped = text.strip()
                char_count = len(stripped)
                total_chars += char_count

                if char_count < 20 and len(page.get_images()) > 0:
                    empty_or_image_pages += 1

                replacement_chars += text.count("\ufffd")

            avg_chars = total_chars / max(1, num_pages)
            if avg_chars < threshold:
                return True

            if (empty_or_image_pages / num_pages) >= 0.2:
                return True

            if total_chars > 0 and (replacement_chars / total_chars) > 0.05:
                return True

            return False
        finally:
            doc.close()
    except Exception as exc:
        logger.debug("needs_ocr check failed for %s: %s", pdf_path, exc)
        return False


def get_digital_converter():
    """Lazily instantiate the high-performance digital converter (no OCR)."""
    global _DIGITAL_CONVERTER
    if _DIGITAL_CONVERTER is not None:
        return _DIGITAL_CONVERTER

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=config.DOCLING_DO_TABLE_STRUCTURE,
        generate_picture_images=True,
        images_scale=config.DOCLING_IMAGE_SCALE,
    )
    _DIGITAL_CONVERTER = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    return _DIGITAL_CONVERTER


def get_ocr_converter():
    """Lazily instantiate the OCR-enabled converter."""
    global _OCR_CONVERTER
    if _OCR_CONVERTER is not None:
        return _OCR_CONVERTER

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        EasyOcrOptions,
        OcrMacOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    if sys.platform == "darwin":
        ocr_opts = OcrMacOptions(
            lang=["en-US"], recognition="accurate", force_full_page_ocr=False
        )
    else:
        ocr_opts = EasyOcrOptions(lang=["en"], force_full_page_ocr=False)

    opts = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=ocr_opts,
        do_table_structure=config.DOCLING_DO_TABLE_STRUCTURE,
        generate_picture_images=True,
        images_scale=config.DOCLING_IMAGE_SCALE,
    )
    _OCR_CONVERTER = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    return _OCR_CONVERTER


def _docling_cache_path(pdf_path: str) -> str:
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    return os.path.join(config.DOCLING_CACHE_DIR, f"{stem}.docling.json")


def extract_docling(
    pdf_path: str,
    key: Optional[str] = None,
    images_dir: Optional[str] = None,
    force_ocr: Optional[bool] = None,
    enable_ocr: Optional[bool] = None,
    use_cache: bool = True,
) -> Document:
    """Extract full-text, structured tables, and clean figure crops using Docling.

    Returns a canonical research_assistant Document instance.
    """
    disable_ocr = os.environ.get("DOCLING_DISABLE_OCR", "0").lower() in ("1", "true", "yes")
    if disable_ocr:
        force_ocr = False
    elif enable_ocr is not None:
        force_ocr = bool(enable_ocr)
    key = key or os.path.basename(pdf_path).strip().replace(" ", "_").lower()
    images_dir = images_dir or config.IMAGES_DIR
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(config.DOCLING_CACHE_DIR, exist_ok=True)

    from docling_core.types.doc.document import DoclingDocument

    cache_file = _docling_cache_path(pdf_path)
    doc = None
    extraction_mode = "docling"

    if use_cache and os.path.isfile(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as fh:
                doc_data = json.load(fh)
                doc = DoclingDocument.model_validate(doc_data)
                extraction_mode = "docling_cached"
                logger.debug("Loaded cached Docling document for %s", key)
        except Exception as exc:
            logger.warning("Failed to load cached Docling document %s: %s", cache_file, exc)
            doc = None

    if doc is None:
        do_ocr = False if disable_ocr else (force_ocr if force_ocr is not None else needs_ocr(pdf_path))
        converter = get_ocr_converter() if do_ocr else get_digital_converter()
        extraction_mode = "docling_ocr" if do_ocr else "docling"

        logger.info("Converting %s via %s…", os.path.basename(pdf_path), extraction_mode)
        try:
            conv_res = converter.convert(pdf_path)
            doc = conv_res.document
        except Exception as exc:
            if not do_ocr and not disable_ocr:
                logger.warning("Digital Docling conversion failed for %s (%s) — retrying with OCR.", key, exc)
                conv_res = get_ocr_converter().convert(pdf_path)
                doc = conv_res.document
                extraction_mode = "docling_ocr"
            elif do_ocr:
                logger.warning("OCR Docling conversion failed for %s (%s) — falling back to digital converter.", key, exc)
                conv_res = get_digital_converter().convert(pdf_path)
                doc = conv_res.document
                extraction_mode = "docling"
            else:
                raise

        # Fallback check: if digital conversion returned zero texts, retry with OCR if not disabled
        if not do_ocr and not disable_ocr and len(doc.texts) == 0:
            logger.info("Digital Docling produced 0 texts for %s — retrying with OCR…", key)
            try:
                conv_res = get_ocr_converter().convert(pdf_path)
                doc = conv_res.document
                extraction_mode = "docling_ocr"
            except Exception as exc:
                logger.warning("OCR fallback conversion failed for %s: %s", key, exc)

        # Cache converted Docling document JSON for fast subsequent runs
        try:
            with open(cache_file, "w", encoding="utf-8") as fh:
                json.dump(doc.model_dump(mode="json"), fh)
        except Exception as exc:
            logger.warning("Failed to write Docling cache for %s: %s", key, exc)

    # ─── 1. Map Body Sections & Paragraphs ────────────────────────────────────
    sections: list[Section] = []
    current_heading = ""
    current_paras: list[Paragraph] = []
    in_bibliography = False

    title = doc.name or key
    abstract: list[Paragraph] = []

    for item, _level in doc.iterate_items():
        label = str(getattr(item, "label", "")).lower()

        # Extract page (0-based)
        page_0 = 0
        if getattr(item, "prov", None) and len(item.prov) > 0:
            page_0 = max(0, item.prov[0].page_no - 1)

        # Skip reference items
        is_reference = False
        if DocItemLabel is not None and getattr(item, "label", None) == DocItemLabel.REFERENCE:
            is_reference = True
        elif "reference" in label:
            is_reference = True

        if is_reference:
            continue

        # Skip running page headers, footers, footnotes, and captions from main prose
        if any(skip in label for skip in ["page_header", "page_footer", "footnote", "caption"]):
            continue

        if "section_header" in label or "title" in label:
            heading_text = (getattr(item, "text", "") or "").strip()
            if not heading_text:
                continue

            # Check if this heading indicates the bibliography / references section
            if _BIB_HEAD.match(heading_text):
                if current_paras:
                    sec_kind = normalise_section_kind(current_heading)
                    if sec_kind == "abstract" and not abstract:
                        abstract = list(current_paras)
                    else:
                        sections.append(Section(current_heading, sec_kind, list(current_paras)))
                    current_paras.clear()
                in_bibliography = True
                current_heading = heading_text
                continue
            else:
                if in_bibliography:
                    in_bibliography = False
                    current_paras.clear()
                elif current_paras:
                    sec_kind = normalise_section_kind(current_heading)
                    if sec_kind == "abstract" and not abstract:
                        abstract = list(current_paras)
                    else:
                        sections.append(Section(current_heading, sec_kind, list(current_paras)))
                    current_paras.clear()

                current_heading = heading_text
                continue

        if in_bibliography:
            continue

        # Standard text paragraph or list item
        text = (getattr(item, "text", "") or "").strip()
        if text:
            current_paras.append(Paragraph(text, page_0))

    if current_paras and not in_bibliography:
        sec_kind = normalise_section_kind(current_heading)
        if sec_kind == "abstract" and not abstract:
            abstract = list(current_paras)
        else:
            sections.append(Section(current_heading, sec_kind, list(current_paras)))

    # If no sections were created (e.g. flat text), wrap all paragraphs into one
    if not sections and current_paras:
        sections.append(Section("", "other", list(current_paras)))

    # ─── 2. Map Figures and Save Clean Visual Crops ───────────────────────────
    figures: list[Figure] = []
    min_px = config.DOCLING_MIN_FIG_PX
    padding = getattr(config, "DOCLING_FIGURE_PADDING", 14.0)
    scale = getattr(config, "DOCLING_IMAGE_SCALE", 2.0)

    pdf_doc = None
    if os.path.isfile(pdf_path):
        try:
            import fitz
            pdf_doc = fitz.open(pdf_path)
        except Exception as exc:
            logger.debug("Could not open %s with PyMuPDF: %s", pdf_path, exc)
            pdf_doc = None

    # Index texts by self_ref for fast caption lookup
    texts_by_ref = {}
    for t_item in getattr(doc, "texts", []):
        s_ref = getattr(t_item, "self_ref", None)
        if s_ref:
            texts_by_ref[s_ref] = t_item

    # Collect raw picture items
    raw_pics = []
    for i, pic in enumerate(getattr(doc, "pictures", [])):
        fid = getattr(pic, "self_ref", None) or f"fig_{i}"
        caption = pic.caption_text(doc) or ""
        page_0 = 0
        b_raw = None
        if getattr(pic, "prov", None) and len(pic.prov) > 0:
            prov = pic.prov[0]
            page_0 = max(0, prov.page_no - 1)
            b_obj = getattr(prov, "bbox", None)
            if b_obj:
                b_raw = {
                    "l": float(b_obj.l),
                    "t": float(b_obj.t),
                    "r": float(b_obj.r),
                    "b": float(b_obj.b),
                    "coord_origin": getattr(b_obj, "coord_origin", "BOTTOMLEFT"),
                }

        # Look up caption bbox
        cap_b = None
        for cap_ref in getattr(pic, "captions", []):
            cref = getattr(cap_ref, "cref", None)
            t_obj = texts_by_ref.get(cref)
            if t_obj and getattr(t_obj, "prov", None) and len(t_obj.prov) > 0:
                cb = getattr(t_obj.prov[0], "bbox", None)
                if cb:
                    cap_b = {
                        "l": float(cb.l),
                        "t": float(cb.t),
                        "r": float(cb.r),
                        "b": float(cb.b),
                        "coord_origin": getattr(cb, "coord_origin", "BOTTOMLEFT"),
                    }
                    break

        raw_pics.append({
            "idx": i,
            "fid": fid,
            "caption": caption,
            "page_0": page_0,
            "b": b_raw,
            "cap_b": cap_b,
            "pic_obj": pic,
        })

    # Group adjacent sub-panel pictures on the same page
    by_page: dict[int, list[dict]] = {}
    for rp in raw_pics:
        by_page.setdefault(rp["page_0"], []).append(rp)

    merged_pics = []
    for p0, page_items in by_page.items():
        used = set()
        for idx1, it1 in enumerate(page_items):
            if idx1 in used:
                continue
            cur = dict(it1)
            used.add(idx1)
            if cur["b"]:
                for idx2, it2 in enumerate(page_items):
                    if idx2 in used or not it2["b"]:
                        continue
                    b1, b2 = cur["b"], it2["b"]
                    w1, h1 = abs(b1["r"] - b1["l"]), abs(b1["t"] - b1["b"])
                    w2, h2 = abs(b2["r"] - b2["l"]), abs(b2["t"] - b2["b"])

                    # Vertical arrangement (one above the other)
                    h_overlap = min(b1["r"], b2["r"]) - max(b1["l"], b2["l"])
                    v_dist = max(0.0, max(b2["b"] - b1["t"], b1["b"] - b2["t"]))
                    stacked = (h_overlap >= 0.5 * min(w1, w2)) and (v_dist <= 14.0)

                    # Horizontal arrangement (side-by-side)
                    v_overlap = min(b1["t"], b2["t"]) - max(b1["b"], b2["b"])
                    h_dist = max(0.0, max(b2["l"] - b1["r"], b1["l"] - b2["r"]))
                    side_by_side = (v_overlap >= 0.5 * min(h1, h2)) and (h_dist <= 14.0)

                    if stacked or side_by_side:
                        cur["b"] = {
                            "l": min(b1["l"], b2["l"]),
                            "r": max(b1["r"], b2["r"]),
                            "b": min(b1["b"], b2["b"]),
                            "t": max(b1["t"], b2["t"]),
                            "coord_origin": b1.get("coord_origin", "BOTTOMLEFT"),
                        }
                        if not cur["caption"].strip() and it2["caption"].strip():
                            cur["caption"] = it2["caption"]
                        if not cur["cap_b"] and it2["cap_b"]:
                            cur["cap_b"] = it2["cap_b"]
                        used.add(idx2)
            merged_pics.append(cur)

    # Render and filter figures
    try:
        for out_idx, mp in enumerate(merged_pics):
            fid = mp["fid"]
            caption = mp["caption"]
            page_0 = mp["page_0"]
            b = mp["b"]
            cap_b = mp["cap_b"]
            pic_obj = mp["pic_obj"]

            bbox = (b["l"], b["t"], b["r"], b["b"]) if b else None
            image_filename = ""
            im = None

            if pdf_doc is not None and 0 <= page_0 < len(pdf_doc) and b:
                try:
                    page_obj = pdf_doc[page_0]
                    page_w, page_h = page_obj.rect.width, page_obj.rect.height
                    coord_origin = b.get("coord_origin", "BOTTOMLEFT")
                    is_bottom_left = "BOTTOMLEFT" in str(coord_origin).upper()
                    if is_bottom_left:
                        x0 = b["l"]
                        y0 = page_h - b["t"]
                        x1 = b["r"]
                        y1 = page_h - b["b"]
                    else:
                        x0 = b["l"]
                        y0 = b["t"]
                        x1 = b["r"]
                        y1 = b["b"]

                    x0, x1 = min(x0, x1), max(x0, x1)
                    y0, y1 = min(y0, y1), max(y0, y1)

                    # Filter false positives:
                    # Check text inside unpadded box
                    test_rect = fitz.Rect(x0, y0, x1, y1)
                    text_in_rect = page_obj.get_text("text", clip=test_rect).strip()
                    bw = abs(x1 - x0)
                    bh = abs(y1 - y0)

                    # 1. Captionless text block (e.g. author profile box, header banner)
                    if not caption.strip() and len(text_in_rect) > 60:
                        logger.debug("Skipping uncaptioned text-block crop (%d chars) on page %d for %s", len(text_in_rect), page_0 + 1, key)
                        continue

                    # 2. Captionless full-page artifact (e.g. references page)
                    if not caption.strip() and (bw >= 0.75 * page_w or bh >= 0.75 * page_h) and (bw * bh >= 0.40 * page_w * page_h):
                        logger.debug("Skipping full-page false-positive crop on page %d for %s", page_0 + 1, key)
                        continue

                    px0 = max(0.0, x0 - padding)
                    py0 = max(0.0, y0 - padding)
                    px1 = min(page_w, x1 + padding)
                    py1 = min(page_h, y1 + padding)

                    # Avoid caption boundary intrusion
                    if cap_b:
                        cap_origin = cap_b.get("coord_origin", "BOTTOMLEFT")
                        if "BOTTOMLEFT" in str(cap_origin).upper():
                            cx0, cy0 = cap_b["l"], page_h - cap_b["t"]
                            cx1, cy1 = cap_b["r"], page_h - cap_b["b"]
                        else:
                            cx0, cy0 = cap_b["l"], cap_b["t"]
                            cx1, cy1 = cap_b["r"], cap_b["b"]
                        cx0, cx1 = min(cx0, cx1), max(cx0, cx1)
                        cy0, cy1 = min(cy0, cy1), max(cy0, cy1)

                        # If caption is directly below the figure
                        if cy0 >= y1 - 4.0 and cy0 <= y1 + 35.0:
                            py1 = min(py1, max(y1, cy0 - 2.0))
                        # If caption is directly above the figure
                        elif cy1 <= y0 + 4.0 and cy1 >= y0 - 35.0:
                            py0 = max(py0, min(y0, cy1 + 2.0))

                    crop_rect = fitz.Rect(px0, py0, px1, py1)
                    pix = page_obj.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=crop_rect, alpha=False)
                    from PIL import Image
                    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    bbox = (px0, py0, px1, py1)
                except Exception as exc:
                    logger.debug("PyMuPDF padded crop failed for %s pic %d: %s", key, out_idx, exc)
                    im = None

            if im is None:
                try:
                    im = pic_obj.get_image(doc)
                except Exception as exc:
                    logger.debug("Figure crop extraction failed for %s pic %d: %s", key, out_idx, exc)
                    im = None

            if im is not None:
                w, h = im.size
                aspect = w / max(1, h)
                if w >= min_px and h >= min_px and 0.08 < aspect < 12.0:
                    image_filename = f"{key}_p{page_0}_f{out_idx}.png"
                    out_path = os.path.join(images_dir, image_filename)
                    im.save(out_path, format="PNG")
                else:
                    bbox = None
            else:
                bbox = None

            m = re.match(r"^(Figure|Fig\.?|FIG\.?)\s*(\d+[a-zA-Z]*)", caption, re.IGNORECASE)
            label = m.group(0).strip() if m else f"Figure {out_idx + 1}"

            figures.append(
                Figure(
                    id=fid,
                    kind="figure",
                    label=label,
                    caption=caption,
                    page=page_0,
                    bbox=bbox,
                    image_path=image_filename,
                )
            )
    finally:
        if pdf_doc is not None:
            try:
                pdf_doc.close()
            except Exception:
                pass

    # ─── 3. Map Tables (Markdown Only, Never Image Crops) ─────────────────────
    for j, tbl in enumerate(getattr(doc, "tables", [])):
        tid = getattr(tbl, "self_ref", None) or f"tab_{j}"
        caption = tbl.caption_text(doc) or ""
        
        page_0 = 0
        if getattr(tbl, "prov", None) and len(tbl.prov) > 0:
            page_0 = max(0, tbl.prov[0].page_no - 1)

        m = re.match(r"^(Table|Tab\.?|TAB\.?)\s*(\d+[a-zA-Z]*)", caption, re.IGNORECASE)
        label = m.group(0).strip() if m else f"Table {j + 1}"

        try:
            table_md = tbl.export_to_markdown(doc) or ""
        except Exception:
            table_md = ""

        # Tables are text/markdown representations: image_path is ALWAYS empty
        figures.append(
            Figure(
                id=tid,
                kind="table",
                label=label,
                caption=caption,
                page=page_0,
                bbox=None,
                table_markdown=table_md,
                image_path="",
            )
        )

    # ─── 4. Map Formulas ──────────────────────────────────────────────────────
    formulas: list[Formula] = []
    for k, item in enumerate(getattr(doc, "texts", [])):
        lbl = str(getattr(item, "label", "")).lower()
        if "formula" in lbl:
            f_text = getattr(item, "text", "") or ""
            if f_text.strip():
                p0 = 0
                if getattr(item, "prov", None) and len(item.prov) > 0:
                    p0 = max(0, item.prov[0].page_no - 1)
                formulas.append(
                    Formula(
                        id=f"formula_{k}",
                        text=f_text,
                        page=p0,
                    )
                )

    return Document(
        key=key,
        title=title,
        abstract=abstract,
        sections=sections,
        figures=figures,
        extraction=extraction_mode,
        n_bib=0,
        formulas=formulas,
    )
