# research_assistant/shared/tei_structure.py
"""Where a paragraph sits in the citing paper and what it points at.

Two facts GROBID's TEI states but does not hand over cleanly. Sections are
flat: "2." and "2.1." are sibling <div>s, so a heading's parent is the
nearest earlier heading with a shallower number. Figures and tables are
<figure> nodes a paragraph reaches through <ref type="figure"
target="#fig_0"> — when GROBID linked the reference; about one in six it
leaves untargeted, and "Fig. 3" in the text is the only witness.

Both walks are deterministic and cost no model call. Both are hygiene
first: GROBID emits page headers as <head> and as <figure>, and neither
may become a breadcrumb or an artifact.
"""

from __future__ import annotations

import re

from research_assistant.shared.claim_text import clean_text
from research_assistant.shared.extract import is_running_header

_NUMBERED_RE = re.compile(r"^\d+(\.\d+)*$")


def _depth(n) -> int:
    """'2.1.' → 2, '2.' → 1; unnumbered or roman → 0."""
    n = (n or "").strip().rstrip(".")
    return n.count(".") + 1 if n and _NUMBERED_RE.match(n) else 0


def _label(head) -> str:
    return f"{(head.get('n') or '').strip()} {clean_text(head)}".strip()


def section_breadcrumb(p) -> str:
    """'2. Results and Discussion > 2.1. Terahertz Spectral Analysis' for a
    paragraph in the 2.1 div; the heading alone at the top; '' before any
    heading. The parent of a heading is the nearest earlier sibling with a
    shallower number, and the climb stops at depth 1 — an unnumbered
    heading ("Conclusions", "I. INVERSION PROCEDURE") is only ever a
    paragraph's own crumb, never a parent, because GROBID also files the
    journal's page header as an unnumbered <head>. A headless or
    page-header div inherits the heading before it."""
    div = p.find_parent("div")
    if div is None:
        return ""
    crumbs: list[str] = []
    depth = None
    for cand in [div, *div.find_previous_siblings("div")]:
        head = cand.find("head")
        if head is None or is_running_header(clean_text(head)):
            continue
        d = _depth(head.get("n"))
        if depth is None or d < depth:
            crumbs.append(_label(head))
            depth = d
        if depth <= 1:
            break
    return " > ".join(reversed(crumbs))
