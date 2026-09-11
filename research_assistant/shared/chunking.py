# research_assistant/shared/chunking.py
"""Sentence windows packed within a section — the v2 chunker.

Pure functions: no I/O, no model calls, no config reads. chunk_document()
(Task 7) turns a Document into Chunks; this half is the sentence splitter
and the packer.

Rules (spec §4.3): close a window at `target` chars; never exceed `hard_max`
(a lone sentence over the max stands alone rather than being cut); prefer to
close at a paragraph boundary once ≥ 70% of target; the last sentence of one
window opens the next; never cross a section (the caller packs one section
at a time).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from research_assistant.shared.extract import Document, Section, Paragraph
from research_assistant.shared.log import get_logger

logger = get_logger("chunking")

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[“\"])")
# A piece ending in one of these did not end a sentence. Applied after pysbd
# too: it splits "Phys. Rev." and "Ref. [3]" on physics prose.
_ABBR = re.compile(
    r"(\bet al|\bFig|\bFigs|\bRef|\bRefs|\bEq|\bEqs|\bSec|\bSecs|\bTab|\bvs|\bcf|\be\.g|\bi\.e"
    r"|\bPhys|\bRev|\bLett|\bNat|\bAdv|\bMater|\bAppl|\bJ|\bSci|\bChem|\bOpt|\bVol|\bpp"
    r"|\b[A-Z])\.$"
)


def _merge_abbreviations(pieces: list[str]) -> list[str]:
    out: list[str] = []
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        if out and _ABBR.search(out[-1]):
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out


def _regex_sentences(text: str) -> list[str]:
    return _merge_abbreviations(_SPLIT.split(text))


_segmenter = None


def sentences(text: str) -> list[str]:
    """pysbd when available, the regex splitter otherwise; the abbreviation
    merge runs on both."""
    global _segmenter
    text = " ".join((text or "").split())
    if not text:
        return []
    if _segmenter is None:
        try:
            import pysbd
            _segmenter = pysbd.Segmenter(language="en", clean=False)
        except Exception as exc:                        # noqa: BLE001
            logger.debug("pysbd unavailable (%s) — regex sentence splitter in use.", exc)
            _segmenter = False
    if _segmenter:
        try:
            return _merge_abbreviations(_segmenter.segment(text))
        except Exception as exc:                        # noqa: BLE001
            logger.debug("pysbd failed on a paragraph (%s) — regex fallback.", exc)
    return _regex_sentences(text)


def pack_windows(paragraphs, target: int, hard_max: int) -> list[tuple[str, int, int]]:
    """Pack sentences into windows. Returns (text, page_first, page_last)."""
    out: list[tuple[str, int, int]] = []
    cur: list[str] = []          # sentences in the open window
    cur_pages: list[int] = []
    fresh = 0                    # sentences added since the last close, excluding the carried overlap

    def cur_len():
        return sum(len(x) for x in cur) + max(0, len(cur) - 1)

    def close(overlap=True):
        nonlocal cur, cur_pages, fresh
        if not cur:
            return
        out.append((" ".join(cur), min(cur_pages), max(cur_pages)))
        # one-sentence overlap: the last sentence opens the next window
        cur, cur_pages = ([cur[-1]], [cur_pages[-1]]) if overlap else ([], [])
        fresh = 0

    for para in paragraphs:
        for s in sentences(para.text):
            if len(s) > hard_max:
                # A giant sentence stands alone; whatever is pending closes
                # first (without overlap), a bare carried overlap is dropped.
                if fresh:
                    close(overlap=False)
                else:
                    cur, cur_pages, fresh = [], [], 0
                out.append((s, para.page, para.page))
                continue
            if cur and cur_len() + 1 + len(s) > hard_max:
                close()
                if cur and len(cur[0]) + 1 + len(s) > hard_max:   # the overlap itself is too long to carry
                    cur, cur_pages = [], []
            cur.append(s)
            cur_pages.append(para.page)
            fresh += 1
            if cur_len() >= target:
                close()
        # paragraph boundary: a good place to stop once the window is big enough
        if fresh and cur_len() >= 0.7 * target:
            close()

    # Flush only new content: after close() the open window is just the
    # carried overlap, and emitting that alone would duplicate the last window's tail.
    if fresh:
        close(overlap=False)
    return out


# ─── Filters ─────────────────────────────────────────────────────────────────

DROP_KINDS = frozenset({"acknowledgements"})


def alpha_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(c.isalpha() or c.isspace() for c in s) / len(s)


def filter_document(doc: Document, min_alpha: float = 0.6, min_section_chars: int = 200) -> Document:
    """Drop what should never be indexed, then merge sections too small to
    stand alone into their neighbour (forward; the trailing one backward)."""
    kept: list[Section] = []
    for sec in doc.sections:
        if sec.kind in DROP_KINDS:
            continue
        paras = [p for p in sec.paragraphs if alpha_ratio(p.text) >= min_alpha]
        if not paras:
            continue
        kept.append(Section(sec.heading, sec.kind, paras))

    merged: list[Section] = []
    pending: list[Paragraph] = []
    for sec in kept:
        if sum(len(p.text) for p in sec.paragraphs) < min_section_chars:
            pending.extend(sec.paragraphs)
            continue
        merged.append(Section(sec.heading, sec.kind, pending + sec.paragraphs))
        pending = []
    if pending:
        if merged:
            merged[-1].paragraphs.extend(pending)
        else:
            merged.append(Section("", "other", pending))

    abstract = [p for p in doc.abstract if alpha_ratio(p.text) >= min_alpha]
    return Document(doc.key, doc.title, abstract, merged, doc.figures, doc.extraction, doc.n_bib)


# ─── Document → chunks ───────────────────────────────────────────────────────

@dataclass
class Chunk:
    text: str
    type: str                  # "text" | "caption"
    section: str
    section_raw: str
    page_first: int
    page_last: int
    seq: int
    figure_id: str = ""
    figure_kind: str = ""
    figure_label: str = ""


def chunk_document(doc: Document, target: int, hard_max: int,
                   min_chars: int = 200, min_alpha: float = 0.6) -> list[Chunk]:
    doc = filter_document(doc, min_alpha=min_alpha)
    chunks: list[Chunk] = []

    def emit_text(paragraphs, kind, heading):
        for text, p0, p1 in pack_windows(paragraphs, target, hard_max):
            if len(text) < min_chars or alpha_ratio(text) < min_alpha:
                continue
            chunks.append(Chunk(text, "text", kind, heading, p0, p1, len(chunks)))

    if doc.abstract:
        emit_text(doc.abstract, "abstract", "")
    for sec in doc.sections:
        emit_text(sec.paragraphs, sec.kind, sec.heading)

    for fig in doc.figures:
        text = f"{fig.label}: {fig.caption}" if fig.caption else fig.label
        if not text.strip():
            continue
        chunks.append(Chunk(text, "caption", "caption", fig.label, fig.page, fig.page, len(chunks),
                            figure_id=fig.id, figure_kind=fig.kind, figure_label=fig.label))
    return chunks
