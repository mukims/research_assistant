"""
extractor.py — Extract in-text citation claims and bibliography references from TEI XML.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Optional

from bs4 import BeautifulSoup

from research_assistant.config import RAW_DIR
from research_assistant.shared.claim_text import (
    CITE_TOKEN_RE,
    SKIP_ROLES,
    cite_token,
    claim_quality,
    classify_citation_role,
    clean_text,
    is_numeric_cite,
    paragraph_sentences,
    reattach_leading_markers,
    render_claim,
    render_sentence,
    sentence_context,
    tidy_punctuation,
)
from research_assistant.shared.extract import normalise_section_kind
from research_assistant.shared.log import get_logger
from research_assistant.shared.source_key import normalise_doi
from research_assistant.shared.tei_structure import artifact_registry, paragraph_artifacts, section_breadcrumb

logger = get_logger("audit_extractor")

XML_OUTPUT_DIR = os.path.join(RAW_DIR, "grobid_output")
_NUM_RE = re.compile(r"\d+")
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


def _clean(node) -> str:
    """Collapse line breaks and spaces."""
    if node is None:
        return ""
    text = node.get_text() if hasattr(node, "get_text") else str(node)
    return " ".join(text.split()).strip()


def _clean_claim_punctuation(text: str) -> str:
    """Tidy punctuation spaces left behind by stripped citation markers."""
    return tidy_punctuation(text)


def _person_name(pers) -> str:
    parts = [_clean(f) for f in pers.find_all("forename")]
    surname = _clean(pers.find("surname"))
    if surname:
        parts.append(surname)
    return " ".join(p for p in parts if p)


def find_tei_for_seed(seed_path: str) -> Optional[str]:
    """Find GROBID TEI XML corresponding to a seed PDF."""
    if not seed_path:
        return None
    stem = os.path.splitext(os.path.basename(seed_path))[0]
    candidates = [
        os.path.join(XML_OUTPUT_DIR, f"{stem}.grobid.tei.xml"),
        os.path.join(XML_OUTPUT_DIR, f"{stem}.tei.xml"),
        os.path.join(XML_OUTPUT_DIR, f"{stem}.references.tei.xml"),
        os.path.join(XML_OUTPUT_DIR, f"{stem}.fulltext.tei.xml"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    matches = glob.glob(os.path.join(XML_OUTPUT_DIR, f"*{stem}*.tei.xml"))
    return matches[0] if matches else None


def _paragraph_section(p) -> str:
    """Section kind of the <div> the paragraph sits in."""
    div = p.find_parent("div")
    head = div.find("head") if div is not None else None
    if head is not None:
        return normalise_section_kind(clean_text(head))
    if div is not None and not any(
        sibling.find("head") is not None for sibling in div.find_previous_siblings("div")
    ):
        return "introduction"
    return "other"


def _build_number_map(body, bib_by_id: dict) -> dict[int, dict]:
    """What GROBID itself linked: every targeted ref with a numeric text teaches '[k] means this entry'."""
    number_map: dict[int, dict] = {}
    for ref in body.find_all("ref", type="bibr"):
        target = (ref.get("target") or "").lstrip("#")
        txt = clean_text(ref)
        if target in bib_by_id and is_numeric_cite(txt):
            nums = _NUM_RE.findall(txt)
            if len(nums) == 1:
                number_map.setdefault(int(nums[0]), bib_by_id[target])
    return number_map


def _looks_like_reference(entry: Optional[dict]) -> bool:
    """True if entry has enough signal to be a real reference."""
    if not entry:
        return False
    if entry.get("title") and len(entry["title"].split()) >= 3:
        return True
    if entry.get("doi"):
        return True
    return False


def _unknown_ref(target: str, txt: str) -> dict:
    """Fallback reference object for citations that could not be dereferenced."""
    return {
        "xml_id": target,
        "title": f"Unresolved reference ({txt})" if txt else "Unresolved reference",
        "authors": [],
        "year": None,
        "doi": None,
        "raw": txt,
    }


def extract_seed_citation_claims(tei_source: str | BeautifulSoup) -> list[dict]:
    """Extract citation claims from TEI XML body paragraphs."""
    from research_assistant.audit.resolver import _resolve_ref

    if isinstance(tei_source, BeautifulSoup):
        soup = tei_source
    else:
        soup = BeautifulSoup(tei_source, "xml")

    body = soup.find("body")
    if not body:
        return []

    # 1. Build bibliography index
    bib_by_id: dict[str, dict] = {}
    bib_by_index: list[dict] = []
    back = soup.find("back") or soup
    for b in back.find_all("biblStruct"):
        xid = b.get("xml:id") or b.get("id") or ""
        t_node = b.find("title", level="a") or b.find("title")
        t_str = _clean(t_node)
        doi_node = b.find("idno", type="DOI") or b.find("idno", type="doi")
        doi_str = normalise_doi(_clean(doi_node)) if doi_node else None
        dt = b.find("date")
        yr_val = None
        if dt and dt.get("when"):
            m = _YEAR_RE.search(dt["when"])
            if m:
                yr_val = int(m.group(1))
        if yr_val is None:
            m = _YEAR_RE.search(_clean(b))
            if m:
                yr_val = int(m.group(1))

        auth_list = [_person_name(p) for p in b.find_all("author") if _person_name(p)]

        entry = {
            "xml_id": xid,
            "title": t_str,
            "authors": auth_list,
            "year": yr_val,
            "doi": doi_str,
            "raw": _clean(b),
        }
        if xid:
            bib_by_id[xid] = entry
        bib_by_index.append(entry)

    number_map = _build_number_map(body, bib_by_id)
    artifacts = artifact_registry(soup)

    claims: list[dict] = []
    claim_idx = 0

    for p_tag in body.find_all("p"):
        p_arts = paragraph_artifacts(p_tag, artifacts)
        sec_name = _paragraph_section(p_tag)
        sec_bc = section_breadcrumb(p_tag)

        sents = paragraph_sentences(p_tag)
        for s_idx, s_node in enumerate(sents):
            s_ctx = sentence_context(s_node, sents, s_idx)
            refs = s_node.find_all("ref", type="bibr")
            if not refs:
                continue

            for r in refs:
                tgt = (r.get("target") or "").lstrip("#")
                r_txt = clean_text(r)
                resolved, how, note = _resolve_ref(tgt, r_txt, bib_by_id, bib_by_index, number_map)
                if not resolved:
                    resolved = _unknown_ref(tgt, r_txt)

                c_text = render_claim(s_node, r)
                c_clean = _clean_claim_punctuation(c_text)
                if not c_clean:
                    continue

                role = classify_citation_role(c_clean, r_txt)
                qual = claim_quality(c_clean, resolved)

                claim_idx += 1
                claims.append({
                    "claim_id": f"claim_{claim_idx:04d}",
                    "claim_text": c_clean,
                    "target_ref": resolved,
                    "citation_marker": r_txt,
                    "section": sec_name,
                    "section_breadcrumb": sec_bc,
                    "role": role,
                    "quality": qual,
                    "artifacts": p_arts,
                    "sentence_context": s_ctx,
                })

    return claims
