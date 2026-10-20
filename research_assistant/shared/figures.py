# research_assistant/shared/figures.py
"""Figures and tables in the ingest: crop each one from GROBID's box.

The crop is embedded as an image (embeddinggemma-2) so a text query can find
the figure. Nothing describes it in words, and it is never judge evidence: the
types in EXCLUDED_FROM_EVIDENCE are left out of every evidence search.
"""

from __future__ import annotations

import os

from research_assistant.config import EXCLUDED_FROM_EVIDENCE  # noqa: F401  (re-exported)
from research_assistant.shared.extract import Figure
from research_assistant.shared.log import get_logger

logger = get_logger("figures")

# The stored text of a figure_image entry. It differs from the caption entry's
# text so the two are not dropped as duplicates of each other.
FIGURE_IMAGE_PREFIX = "Image of {label}: "
_PAD_PT = 10.0


def crop_name(document_key: str, fig: Figure, index: int) -> str:
    """Same shape as the v1 layout path wrote: <doc>_p<page>_f<i>.png."""
    return f"{document_key}_p{fig.page}_f{index}.png"


def crop_figure(pdf_path: str, fig: Figure, out_path: str, dpi: int) -> bool:
    """Render the figure's box (plus a small pad, clamped to the page) to PNG.
    False when there is nothing to crop; never raises for a missing box."""
    if not fig.bbox:
        return False
    import fitz

    pdf = fitz.open(pdf_path)
    if fig.page < 0 or fig.page >= len(pdf):
        return False
    page = pdf[fig.page]
    x0, y0, x1, y1 = fig.bbox
    clip = fitz.Rect(
        max(0.0, x0 - _PAD_PT), max(0.0, y0 - _PAD_PT),
        min(page.rect.width, x1 + _PAD_PT), min(page.rect.height, y1 + _PAD_PT),
    )
    if clip.x1 <= clip.x0 or clip.y1 <= clip.y0:
        return False
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    page.get_pixmap(clip=clip, dpi=dpi).save(out_path)
    return True
