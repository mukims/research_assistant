# research_assistant/shared/figures.py
"""Figures and tables in the v2 ingest: crop from GROBID's box, describe with
the caption and the sentences that cite it.

Nothing here decides *whether* to describe — that is the run's switch,
threaded down from the CLI/UI to process_pdf_v2(). This module only knows how.
"""

from __future__ import annotations

import os

from research_assistant.prompts import FIGURE_DESCRIPTION
from research_assistant.shared.extract import Figure
from research_assistant.shared.log import get_logger

logger = get_logger("figures")

DESCRIPTION_PREFIX = "Auto-generated description of {label} — verify values against the figure: "
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


def figure_context(fig: Figure, max_refs: int = 4) -> str:
    caption = f"{fig.label}: {fig.caption}" if fig.caption else fig.label
    lines = [f"Caption: {caption}"]
    if fig.refs:
        lines.append("Referenced in the text:")
        lines.extend(f"- {r}" for r in fig.refs[:max_refs])
    return "\n".join(lines)


def describe_crop(image_path: str, fig: Figure, context: str) -> str:
    """One model call. Raw text back; the caller adds DESCRIPTION_PREFIX."""
    from research_assistant.shared.llm import chat

    prompt = FIGURE_DESCRIPTION.format(fig_type=fig.kind, context=context)
    return chat([{"role": "user", "content": prompt}], images=[image_path], temperature=0.0).content.strip()
