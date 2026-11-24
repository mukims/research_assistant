"""
pdf_refs.py — Seed PDF bibliography block extraction and multi-bibliography alignment.

Academic manuscripts (especially arXiv physics/CS papers with appended
Supplemental Material, Nature/Science articles with separate Methods
references, and papers with Data Availability mini-bibliographies) often:
  1. Omit printed reference numbers ("[25]" or "5.") from GROBID's
     <note type="raw_reference">, causing every citation after a skipped
     footnote to disagree with 1-based <biblStruct> list positions.
  2. Mix non-bibliographic endnotes ("[21] See Supplemental Material at ...")
     into the numbered reference list.
  3. Restart reference numbering at "[1]" inside Supplemental Material or
     Appendices while GROBID concatenates all <biblStruct> items into one
     flat <listBibl>.

This module extracts the actual printed numbered reference blocks from the
seed PDF (when available on disk), aligns GROBID <biblStruct> entries to
their exact (bib_group, printed_index) via monotonic dynamic programming,
identifies printed endnotes, and maps body paragraphs to their corresponding
bibliography group.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from research_assistant import config
from research_assistant.shared.log import get_logger

logger = get_logger("pdf_refs")

_RE_BRACKET = re.compile(r"^\s*\[(\d{1,4})\]\s*(.*)$")
_RE_DOT = re.compile(r"^\s*(\d{1,4})\.\s*(.*)$")
_RE_BARE = re.compile(r"^\s*(\d{1,3})\s+([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ\.\-']+.*)$")

_BIB_SIGNAL_RE = re.compile(
    r"\b(1[89]\d{2}|20\d{2})\b"
    r"|et\s+al\."
    r"|arXiv"
    r"|doi[:\s/]"
    r"|10\.\d{4,9}/"
    r"|PMID"
    r"|Press\b"
    r"|Univ(?:ersity|\.)"
    r"|Supplemental\s+Material"
    r"|URL\s+will\s+be\s+inserted"
    r"|Phys\.|Lett\.|Rev\.|J\.\s*[A-Z]|Nat\.|Sci\.|Chem\.|Mater\.| Proc\.|IEEE|ACM",
    re.IGNORECASE,
)

_REF_HEADING_RE = re.compile(
    r"(?m)^\s*(?:references|bibliography|literature\s+cited|works\s+cited)\s*$",
    re.IGNORECASE,
)

_ENDNOTE_TEXT_RE = re.compile(
    r"^\s*(?:see\s+(?:supplemental|supplementary|sm\b)|"
    r"supplemental\s+material\s+at\b|"
    r"url\s+will\s+be\s+inserted|"
    r"an?\s+alternative\s+definition\b|"
    r"the\s+applicability\s+of\s+the\s+method\b|"
    r"note\s+that\b|"
    r"here\s+we\b|"
    r"in\s+this\s+(?:work|paper|manuscript|note)\b)",
    re.IGNORECASE,
)

_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "that", "this", "are", "was",
    "were", "using", "based", "under", "between", "through", "over", "after",
    "via", "their", "its", "have", "has", "can", "will", "not", "clear", "yet",
}


def is_endnote_text(text: Optional[str]) -> bool:
    """Return True if *text* is clearly an explanatory footnote/endnote rather than a citation."""
    if not text:
        return False
    s = " ".join(str(text).split())
    if _ENDNOTE_TEXT_RE.search(s):
        return True
    if "url will be inserted" in s.lower():
        return True
    return False


def locate_seed_pdf(seed_path: Optional[str], tei_source: Optional[str] = None) -> Optional[str]:
    """Locate the seed PDF on disk across RAW_DIR, RAW_DIR/processed, and TEI sibling dirs."""
    candidates: list[str] = []
    if seed_path and isinstance(seed_path, str):
        candidates.append(seed_path)
        base = os.path.basename(seed_path)
        raw_dir = getattr(config, "RAW_DIR", os.path.join(getattr(config, "DATA_DIR", "data"), "raw"))
        candidates.append(os.path.join(raw_dir, base))
        candidates.append(os.path.join(raw_dir, "processed", base))

    if tei_source and isinstance(tei_source, str) and os.path.exists(tei_source):
        tei_base = os.path.basename(tei_source)
        stem = re.sub(r"\.(?:grobid|references|fulltext)?\.?tei\.xml$", "", tei_base, flags=re.IGNORECASE)
        if stem and stem != tei_base:
            tei_dir = os.path.dirname(os.path.abspath(tei_source))
            parent_raw = os.path.dirname(tei_dir)
            candidates.append(os.path.join(parent_raw, f"{stem}.pdf"))
            candidates.append(os.path.join(parent_raw, "processed", f"{stem}.pdf"))

    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def extract_pdf_bibliography_groups(pdf_path: str) -> tuple[list[dict], list[str]]:
    """Extract sequential numbered reference blocks and per-page text from *pdf_path*."""
    try:
        import pymupdf
    except ImportError:
        return [], []

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:
        logger.debug("Could not open seed PDF %s for reference numbering: %s", pdf_path, exc)
        return [], []

    try:
        pages_text = [doc[i].get_text("text") for i in range(len(doc))]
    finally:
        doc.close()

    raw_items: list[dict] = []
    for pno, ptxt in enumerate(pages_text):
        lines = ptxt.splitlines()
        cur: Optional[dict] = None
        pending_num: Optional[tuple[int, str]] = None
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            if pending_num is not None:
                k_pend, st_pend = pending_num
                pending_num = None
                if not re.match(r"^\d+$", s):
                    cur = {"page": pno, "k": k_pend, "style": st_pend, "text": s}
                    raw_items.append(cur)
                    continue
            m_b = _RE_BRACKET.match(ln)
            if m_b:
                k = int(m_b.group(1))
                rest = m_b.group(2).strip()
                if rest:
                    cur = {"page": pno, "k": k, "style": "bracket", "text": rest}
                    raw_items.append(cur)
                else:
                    pending_num = (k, "bracket")
                continue
            m_d = _RE_DOT.match(ln)
            if m_d and int(m_d.group(1)) <= 1500:
                k = int(m_d.group(1))
                rest = m_d.group(2).strip()
                if rest:
                    cur = {"page": pno, "k": k, "style": "dot", "text": rest}
                    raw_items.append(cur)
                else:
                    pending_num = (k, "dot")
                continue
            m_bare = _RE_BARE.match(ln)
            if m_bare:
                k = int(m_bare.group(1))
                rest = m_bare.group(2).strip()
                cur = {"page": pno, "k": k, "style": "bare", "text": rest}
                raw_items.append(cur)
                continue
            if cur is not None and len(cur["text"]) < 600:
                cur["text"] += " " + s

    blocks: list[dict] = []
    i = 0
    while i < len(raw_items):
        it = raw_items[i]
        if it["k"] in (1, 2):
            chain = [it]
            st_allowed = {it["style"]} if it["style"] == "bracket" else {"dot", "bare"}
            for j in range(i + 1, len(raw_items)):
                cand = raw_items[j]
                prev = chain[-1]
                if cand["page"] - prev["page"] > 4:
                    break
                if cand["k"] == 1 and prev["k"] >= 3 and cand["page"] >= prev["page"]:
                    break
                if cand["style"] in st_allowed and prev["k"] < cand["k"] <= prev["k"] + 5:
                    chain.append(cand)

            has_ref_heading = bool(_REF_HEADING_RE.search(pages_text[chain[0]["page"]]))
            min_len = 2 if (it["k"] == 1 and has_ref_heading) else 4
            bib_hits = sum(1 for c in chain if _BIB_SIGNAL_RE.search(c["text"]))
            if len(chain) >= min_len and (bib_hits / len(chain)) >= 0.5:
                blocks.append(
                    {
                        "start_page": chain[0]["page"],
                        "end_page": chain[-1]["page"],
                        "items": {c["k"]: c["text"] for c in chain},
                        "pages": {c["k"]: c["page"] for c in chain},
                    }
                )
                last_used = chain[-1]
                while i < len(raw_items) and raw_items[i] is not last_used:
                    i += 1
        i += 1

    return blocks, pages_text


def _norm_tokens(text: Optional[str]) -> set[str]:
    if not text:
        return set()
    clean = re.sub(r"-\s+", "", str(text).lower())
    words = re.findall(r"[a-z0-9]{3,}", clean)
    return {w for w in words if w not in _STOPWORDS}


def _entry_pdf_similarity(entry: dict, pdf_text: str) -> float:
    """Similarity in [0.0, 1.0] between a parsed TEI <biblStruct> dict and a PDF reference line."""
    pdf_norm = re.sub(r"-\s+", "", pdf_text.lower())
    pdf_toks = _norm_tokens(pdf_text)
    if not pdf_toks:
        return 0.0

    ref_parts = [
        entry.get("raw_reference") or "",
        entry.get("title") or "",
        " ".join(entry.get("authors") or []),
        entry.get("venue") or "",
    ]
    ref_toks = _norm_tokens(" ".join(ref_parts))
    if not ref_toks:
        return 0.0

    inter = len(ref_toks & pdf_toks)
    recall_ref = inter / max(1, min(len(ref_toks), len(pdf_toks)))
    jaccard = inter / max(1, len(ref_toks | pdf_toks))
    score = 0.65 * recall_ref + 0.35 * jaccard

    # First-author surname check
    authors = entry.get("authors") or []
    if authors:
        first_a = authors[0].strip()
        surname = first_a.split(",")[0].strip() if "," in first_a else first_a.split()[-1].strip()
        surname_clean = re.sub(r"[^a-z]", "", surname.lower())
        if len(surname_clean) >= 3:
            if surname_clean in re.sub(r"[^a-z]", " ", pdf_norm):
                score += 0.20
            else:
                score -= 0.15

    # Year check
    yr = entry.get("year")
    if yr and str(yr) in pdf_norm:
        score += 0.10

    return max(0.0, min(1.0, score))


def align_biblstructs_to_pdf_groups(
    bib_entries: list[dict], pdf_groups: list[dict]
) -> dict[int, dict[int, str]]:
    """Monotonically align *bib_entries* to *pdf_groups* in-place and return endnotes_by_group.

    Sets ``bib_group``, ``printed_index``, and ``index`` on matched entries, and
    returns ``{group_idx: {k: endnote_text}}`` for numbered items in the PDF
    that are explanatory footnotes/endnotes rather than bibliographic entries.
    """
    from research_assistant.shared.source_key import extract_arxiv_from_text, extract_doi_from_text

    endnotes_by_group: dict[int, dict[int, str]] = {g: {} for g in range(len(pdf_groups))}
    if not bib_entries or not pdf_groups:
        return endnotes_by_group

    slots: list[tuple[int, int, str]] = []
    for g_idx, grp in enumerate(pdf_groups):
        for k in sorted(grp["items"]):
            slots.append((g_idx, k, grp["items"][k]))

    n_b = len(bib_entries)
    n_s = len(slots)
    if n_s == 0:
        return endnotes_by_group

    # Precompute similarity matrix
    sim = [[_entry_pdf_similarity(bib_entries[i], slots[j][2]) for j in range(n_s)] for i in range(n_b)]

    # Monotonic DP: dp[i][j] = best total score aligning prefix bib_entries[:i] and slots[:j]
    dp = [[0.0] * (n_s + 1) for _ in range(n_b + 1)]
    choice = [[0] * (n_s + 1) for _ in range(n_b + 1)]  # 1: match (i-1, j-1), 2: skip i-1, 3: skip j-1

    min_match_sim = 0.28
    for i in range(1, n_b + 1):
        for j in range(1, n_s + 1):
            best = dp[i - 1][j]
            ch = 2
            if dp[i][j - 1] > best:
                best = dp[i][j - 1]
                ch = 3
            s_ij = sim[i - 1][j - 1]
            if s_ij >= min_match_sim:
                cand = dp[i - 1][j - 1] + s_ij
                if cand >= best:
                    best = cand
                    ch = 1
            dp[i][j] = best
            choice[i][j] = ch

    matched_b_to_s: dict[int, int] = {}
    matched_s_to_b: dict[int, int] = {}
    i, j = n_b, n_s
    while i > 0 and j > 0:
        ch = choice[i][j]
        if ch == 1:
            matched_b_to_s[i - 1] = j - 1
            matched_s_to_b[j - 1] = i - 1
            i -= 1
            j -= 1
        elif ch == 2:
            i -= 1
        else:
            j -= 1

    # Assign bib_group and printed_index to matched entries
    for b_idx, s_idx in matched_b_to_s.items():
        g_idx, k, pdf_txt = slots[s_idx]
        entry = bib_entries[b_idx]
        entry["bib_group"] = g_idx
        entry["printed_index"] = k
        entry["index"] = k
        if not entry.get("raw_reference"):
            entry["raw_reference"] = pdf_txt
        if not entry.get("doi"):
            entry["doi"] = extract_doi_from_text(pdf_txt)
        if not entry.get("arxiv_id"):
            entry["arxiv_id"] = extract_arxiv_from_text(pdf_txt)

    # Interpolate bib_group for any unmatched bib_entries based on neighbors
    last_group = 0
    for b_idx, entry in enumerate(bib_entries):
        if "bib_group" in entry:
            last_group = entry["bib_group"]
        else:
            entry["bib_group"] = last_group

    # Record unmatched PDF slots that are explanatory footnotes/endnotes (or note-only biblStructs)
    for s_idx, (g_idx, k, pdf_txt) in enumerate(slots):
        if s_idx not in matched_s_to_b:
            if is_endnote_text(pdf_txt) or not re.search(r"\b(1[89]\d{2}|20\d{2})\b", pdf_txt):
                endnotes_by_group[g_idx][k] = pdf_txt
        else:
            entry = bib_entries[matched_s_to_b[s_idx]]
            if is_endnote_text(pdf_txt) and not (entry.get("authors") or entry.get("year") or entry.get("doi")):
                entry["is_endnote"] = True
                endnotes_by_group[g_idx][k] = pdf_txt

    return endnotes_by_group


def assign_paragraph_bib_group(
    p_text: str,
    pdf_pages_text: list[str],
    pdf_groups: list[dict],
    min_page: int = 0,
) -> tuple[int, int]:
    """Return ``(bib_group_index, matched_page)`` for a body paragraph."""
    if len(pdf_groups) <= 1 or not pdf_pages_text:
        return 0, min_page

    p_toks = _norm_tokens(p_text)
    best_page = min_page
    best_score = -1.0

    if p_toks:
        for pno in range(len(pdf_pages_text)):
            page_toks = _norm_tokens(pdf_pages_text[pno])
            if not page_toks:
                continue
            overlap = len(p_toks & page_toks) / max(1, len(p_toks))
            # Mild preference for forward monotonic progression from min_page
            bonus = 0.05 if pno >= min_page else 0.0
            score = overlap + bonus
            if score > best_score and overlap >= 0.35:
                best_score = score
                best_page = pno

    # Assign to the first bibliography group whose end_page >= best_page
    for g_idx, grp in enumerate(pdf_groups):
        if best_page <= grp["end_page"]:
            return g_idx, best_page
    return len(pdf_groups) - 1, best_page
