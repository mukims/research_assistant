# research_assistant/shared/extract.py
"""PDF → Document: the structure the v2 chunker consumes.

Two producers, one shape. parse_tei() reads GROBID's processFulltextDocument
output — sections with headings, paragraphs with page numbers, figures with
captions and bitmap boxes, and the bibliography kept apart. extract_pymupdf()
(Task 5) is the fallback and produces the same shape with one section of
kind "other" and no figures.

Pages are 0-based throughout, matching what v1 stored from PyMuPDF; GROBID's
coords are 1-based and converted here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import etree

from research_assistant.shared.log import get_logger

logger = get_logger("extract")

TEI = "{http://www.tei-c.org/ns/1.0}"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

SECTION_KINDS = (
    "abstract", "introduction", "background", "methods", "results",
    "discussion", "conclusion", "acknowledgements", "other",
)


@dataclass
class Paragraph:
    text: str
    page: int


@dataclass
class Figure:
    id: str
    kind: str                       # "figure" | "table"
    label: str                      # "Figure 1", "Table 2"
    caption: str
    page: int
    bbox: tuple | None              # (x0, y0, x1, y1) in PDF points, or None
    refs: list = field(default_factory=list)   # body sentences that cite it


@dataclass
class Section:
    heading: str
    kind: str
    paragraphs: list


@dataclass
class Document:
    key: str
    title: str
    abstract: list
    sections: list
    figures: list
    extraction: str                 # "grobid" | "pymupdf"
    n_bib: int = 0


# ─── Section kinds ───────────────────────────────────────────────────────────

_KIND_PATTERNS = [
    ("acknowledgements", re.compile(r"acknowledg|funding|author contribution|competing interest|conflict of interest|data availability", re.I)),
    ("introduction",     re.compile(r"\bintroduction\b", re.I)),
    ("background",       re.compile(r"\b(background|related work|prior work|preliminar)", re.I)),
    ("conclusion",       re.compile(r"\b(conclusion|summary|outlook)", re.I)),
    ("results",          re.compile(r"\bresults?\b", re.I)),
    ("discussion",       re.compile(r"\bdiscussion\b", re.I)),
    ("methods",          re.compile(r"\b(method|experimental|materials|setup|procedure|model|theory)", re.I)),
]


def normalise_section_kind(heading: str) -> str:
    """Map a raw heading to one of SECTION_KINDS. Order matters: 'Results and
    discussion' is results; 'Summary and outlook' is conclusion."""
    text = heading or ""
    for kind, pat in _KIND_PATTERNS:
        if pat.search(text):
            return kind
    return "other"


_RUNNING = [
    re.compile(r"^\(\d+ of \d+\)$"),
    re.compile(r"\bet al\.?$", re.I),
    re.compile(r"^\d+$"),
    re.compile(r"\b(19|20)\d{2}\b.*\b\d{3,}\b"),      # "Adv. Mater. 2023, 35, 2211157"
]


def is_running_header(heading: str) -> bool:
    """GROBID sometimes emits a page header/footer as a <head> without an n=.
    Those must not start a section — their paragraphs belong to the previous one."""
    h = (heading or "").strip()
    return bool(h) and any(p.search(h) for p in _RUNNING)


# ─── TEI helpers ─────────────────────────────────────────────────────────────

def _text(el) -> str:
    # "".join, not " ".join: GROBID's text nodes carry their own spacing, and
    # inserting one at every element boundary turns "(Figure 1b)" into
    # "( Figure 1b )".
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def _first_page(coords: str | None) -> int:
    """'3,48.0,653.0,496.0,8.0;3,...' → 2 (0-based). Unknown → 0."""
    if not coords:
        return 0
    try:
        return max(0, int(coords.split(";")[0].split(",")[0]) - 1)
    except (ValueError, IndexError):
        return 0


def _union_box(coords: str | None):
    """Union of every (x,y,w,h) group in a coords string → (x0,y0,x1,y1)."""
    boxes = []
    for group in (coords or "").split(";"):
        parts = group.split(",")
        if len(parts) != 5:
            continue
        try:
            _, x, y, w, h = (float(v) for v in parts)
        except ValueError:
            continue
        boxes.append((x, y, x + w, y + h))
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _sentence_containing(ref) -> str:
    """The <s> holding a <ref>, else its <p>, as single-spaced text (≤ 400 chars)."""
    node = ref.getparent()
    while node is not None and node.tag not in (f"{TEI}s", f"{TEI}p"):
        node = node.getparent()
    return _text(node)[:400] if node is not None else ""


# ─── Parser ──────────────────────────────────────────────────────────────────

def parse_tei(xml: bytes, key: str) -> Document:
    root = etree.fromstring(xml)

    title_el = root.find(f".//{TEI}titleStmt/{TEI}title")
    title = _text(title_el)

    abstract = [
        Paragraph(_text(p), _first_page(p.get("coords")))
        for p in root.findall(f".//{TEI}abstract//{TEI}p")
        if _text(p)
    ]

    body = root.find(f".//{TEI}body")
    sections: list[Section] = []
    if body is not None:
        for div in body.findall(f"{TEI}div"):
            head = div.find(f"{TEI}head")
            heading = _text(head)
            paras = [
                Paragraph(_text(p), _first_page(p.get("coords")))
                for p in div.findall(f"{TEI}p")
                if _text(p)
            ]
            unnumbered = head is None or head.get("n") is None
            if unnumbered and is_running_header(heading) and sections:
                sections[-1].paragraphs.extend(paras)
                continue
            if not paras and not heading:
                continue
            sections.append(Section(heading, normalise_section_kind(heading), paras))

    figures: list[Figure] = []
    by_id: dict[str, Figure] = {}
    if body is not None:
        for i, fig in enumerate(body.findall(f".//{TEI}figure")):
            fid = fig.get(XML_ID) or f"fig_{i}"
            kind = "table" if fig.get("type") == "table" else "figure"
            head = _text(fig.find(f"{TEI}head")).rstrip(" .")
            label_num = _text(fig.find(f"{TEI}label"))
            label = head or (f"{'Table' if kind == 'table' else 'Figure'} {label_num}".strip())
            caption = _text(fig.find(f"{TEI}figDesc"))
            graphic = fig.find(f"{TEI}graphic")
            coords = graphic.get("coords") if graphic is not None and graphic.get("coords") else fig.get("coords")
            f = Figure(fid, kind, label, caption, _first_page(coords), _union_box(coords))
            figures.append(f)
            by_id[fid] = f

        for ref in body.iter(f"{TEI}ref"):
            if ref.get("type") != "figure":
                continue
            target = (ref.get("target") or "").lstrip("#")
            f = by_id.get(target)
            if f is None:
                continue
            sentence = _sentence_containing(ref)
            if sentence and sentence not in f.refs:
                f.refs.append(sentence)

    n_bib = len(root.findall(f".//{TEI}listBibl/{TEI}biblStruct"))

    return Document(key=key, title=title, abstract=abstract, sections=sections,
                    figures=figures, extraction="grobid", n_bib=n_bib)
