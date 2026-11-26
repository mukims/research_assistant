"""
seed_audit.py — Extract in-text citation claims from uploaded seed paper TEI XML
and verify them against downloaded/ingested reference PDFs using the Agent 8 judgement rubric.
"""

import glob
import hashlib
import json
import os
import re
import time
from datetime import datetime
from typing import Optional

from bs4 import BeautifulSoup

from research_assistant.shared.claim_text import (
    CITE_TOKEN_RE,
    SKIP_ROLES,
    cite_token,
    claim_quality,
    classify_citation_role,
    clean_text,
    describe_citation_locus,
    expand_numeric_cite_markers,
    is_non_citation_ref,
    is_numeric_cite,
    paragraph_sentences,
    reattach_leading_markers,
    render_claim,
    render_sentence,
    sentence_context,
    split_author_year_cites,
    tidy_punctuation,
)
from research_assistant.shared.extract import normalise_section_kind
from research_assistant.judgement.evidence import (
    assemble_evidence,
    needs_escalation,
)
from research_assistant.judgement.judge import (
    _judge_once,
    _judgement_model,
    build_citing_paper_note,
    compose_context,
    filter_relevant_artifacts,
    guard_abstract_only_verdict,
    infer_evidence_tier,
)
from research_assistant.judgement.policy import evaluate_reliability
from research_assistant.judgement.source_assessor import assess_source
from research_assistant.config import (
    EXCLUDED_FROM_EVIDENCE,
    DOWNLOADED_JSON_PATH,
    JUDGEMENT_ESCALATE_TOP_K,
    JUDGEMENT_EVIDENCE_MAX_CHARS,
    JUDGEMENT_NEIGHBOUR_WINDOW,
    JUDGEMENT_TOP_K,
    PULLED_PDFS_DIR,
    RAW_DIR,
    CITATION_AUDIT_CONTEXTUALIZE_QUERIES,
    CITATION_AUDIT_CHECK_UNCITED,
    CITATION_AUDIT_MAX_UNCITED_SENTENCES,
    CITATION_AUDIT_MAX_CLAIMS,
    CITATION_AUDIT_MAX_PAIRS_PER_SENTENCE,
    CITATION_AUDIT_DEFERRED_PAYWALL_THRESHOLD,
    CITATION_JUDGEMENT_SECOND_PASS,
)
from research_assistant.shared import pipeline_status
from research_assistant.shared.atomic import atomic_write_json
import concurrent.futures
from research_assistant.shared.log import get_logger
from research_assistant.shared.search import (
    expand_neighbours,
    extract_salient_claim_terms,
    hybrid_search,
    rerank_candidates,
    supplement_intra_doc_lexical_hits,
)
from research_assistant.shared.source_key import (
    extract_arxiv_from_text,
    extract_doi_from_text,
    normalise_doi,
    pdf_key,
)
from research_assistant.shared.tei_structure import artifact_registry, paragraph_artifacts, section_breadcrumb

logger = get_logger("seed_audit")

AUDIT_DIR = os.path.join(RAW_DIR, "seed_audits")
XML_OUTPUT_DIR = os.path.join(RAW_DIR, "grobid_output")


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
    # Check exact known names
    candidates = [
        os.path.join(XML_OUTPUT_DIR, f"{stem}.grobid.tei.xml"),
        os.path.join(XML_OUTPUT_DIR, f"{stem}.tei.xml"),
        os.path.join(XML_OUTPUT_DIR, f"{stem}.references.tei.xml"),
        os.path.join(XML_OUTPUT_DIR, f"{stem}.fulltext.tei.xml"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Fallback pattern match
    matches = glob.glob(os.path.join(XML_OUTPUT_DIR, f"*{stem}*.tei.xml"))
    return matches[0] if matches else None


_NUM_RE = re.compile(r"\d+")
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


def _paragraph_section(p) -> str:
    """Section kind of the <div> the paragraph sits in. GROBID often leaves
    the introduction headless: a headless div before any headed one is the
    introduction; a headless div after a headed one is unknown ("other")."""
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
    """What GROBID itself linked: every targeted ref with a numeric text
    teaches "[k] means this entry". Consulted for the refs it left untargeted."""
    number_map: dict[int, dict] = {}
    for ref in body.find_all("ref", type="bibr"):
        target = (ref.get("target") or "").lstrip("#")
        txt = clean_text(ref)
        if target in bib_by_id and is_numeric_cite(txt):
            nums = _NUM_RE.findall(txt)
            if len(nums) == 1:
                number_map.setdefault(int(nums[0]), bib_by_id[target])
    return number_map


AUDIT_SCHEMA_VERSION = 2

_GENERIC_TITLE_LEAD_WORDS = frozenset({
    "a", "an", "the", "on", "in", "of", "to", "for", "towards", "toward", "beyond", "rethinking",
    "reimagining", "understanding", "learning", "deep", "quantum", "experimental", "theoretical",
    "direct", "high", "low", "new", "novel", "scalable", "efficient", "automated", "general",
    "introduction", "methods", "results", "discussion", "review", "survey", "overview",
})
_ABSTRACT_SELF_INTRO_RE = re.compile(
    r"\b(?:we\s+(?:introduce|present|propose|develop|design|build|report)|"
    r"(?:called|named|termed|dubbed))\s+([A-Z][A-Za-z0-9_\-]{2,24})\b"
)


def extract_seed_self_identity(soup: BeautifulSoup) -> tuple[str | None, list[str]]:
    """Extract the citing seed manuscript's own title and self-coined framework/system names."""
    seed_title = None
    ts = soup.find("titleStmt")
    if ts:
        t_node = ts.find("title", attrs={"type": "main"}) or ts.find("title")
        if t_node:
            t_clean = _clean(t_node)
            if t_clean:
                seed_title = t_clean

    self_names: list[str] = []
    seen_low: set[str] = set()

    def _add_name(cand: str) -> None:
        cl = cand.strip(".,;:()[]\"'")
        low = cl.lower()
        if len(cl) < 3 or low in _GENERIC_TITLE_LEAD_WORDS or low in seen_low:
            return
        has_special_shape = (
            any(c.isdigit() for c in cl)
            or any(c.isupper() for c in cl[1:])
            or "-" in cl
        )
        if not has_special_shape and not cl[0].isupper():
            return
        seen_low.add(low)
        self_names.append(cl)

    if seed_title and ":" in seed_title:
        lead = seed_title.split(":", 1)[0].strip()
        if re.match(r"^[A-Z][A-Za-z0-9_\-]{2,24}$", lead):
            _add_name(lead)

    ab = soup.find("abstract")
    if ab:
        ab_txt = _clean(ab)
        for m in _ABSTRACT_SELF_INTRO_RE.finditer(ab_txt):
            _add_name(m.group(1))

    return seed_title, self_names


def extract_seed_citation_claims(
    tei_source: str | BeautifulSoup,
    seed_pdf_path: Optional[str] = None,
) -> list[dict]:
    """Extract in-text citation claims from TEI XML.

    Returns a list of dicts with keys:
        - sentence: display sentence with [citation] markers
        - claim: cleaned sentence as plain prose (citations removed or narrative normalized)
        - cite_text: raw in-text citation text (e.g. "[14]", "Smith et al. (2020)")
        - target: xml:id target (e.g. "b13")
        - ref: structured reference dict (xml_id, index, title, authors, year, doi, venue, is_monograph, raw)
        - resolved: bool, whether ref was matched to bibliography
        - resolution: 'target' | 'number_map' | 'position' | 'surname' | 'unresolved'
        - role: 'software' | 'pointer' | 'method' | 'evidential'
        - claim_quality: list of issues flagged
        - context: 3-sentence window with «focus»
        - section: normalised section kind
        - cite_count: number of citations in sentence
        - sentence_index: 0-based index of sentence in paragraph
        - paragraph_id: identifier of the body paragraph (e.g. "p_0" or xml:id)
        - paragraph_index: 0-based integer index of the body paragraph
        - paragraph_refs: list of unique evidential resolved reference dicts cited in this paragraph
        - section_heading: breadcrumb of numbered headings ("2. Results > 2.1. THz analysis"), "" if unknown
        - artifacts: figures/tables the paragraph points at: [{id, kind, label, caption}]
    """
    from research_assistant.audit.resolver import _resolve_ref, _unknown_ref

    if isinstance(tei_source, BeautifulSoup):
        soup = tei_source
    else:
        if not os.path.exists(tei_source):
            logger.warning("TEI file not found: %s", tei_source)
            return []
        with open(tei_source, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "xml")

    seed_title, seed_self_names = extract_seed_self_identity(soup)

    # 1. Parse bibliography from <listBibl>
    from research_assistant.audit.pdf_refs import (
        align_biblstructs_to_pdf_groups,
        assign_paragraph_bib_group,
        extract_pdf_bibliography_groups,
        is_endnote_text,
        locate_seed_pdf,
    )

    def _first_nonempty(nodes) -> Optional[str]:
        for nd in nodes:
            if nd is not None:
                val = _clean(nd)
                if val:
                    return val
        return None

    list_bibl = soup.find("listBibl")
    bib_entries: list[dict] = []
    bib_by_id = {}
    bib_by_index = {}

    if list_bibl:
        for idx, b in enumerate(list_bibl.find_all("biblStruct")):
            xid = b.get("xml:id") or b.get("id") or f"b{idx}"
            analytic = b.find("analytic")
            monogr = b.find("monogr")

            raw_node = b.find("note", type="raw_reference")
            raw_reference = _clean(raw_node) if raw_node else None

            title = None
            if analytic:
                title = _first_nonempty([analytic.find("title", type="main"), *analytic.find_all("title")])
            # GROBID sometimes emits an empty <analytic><title/> on long author lists and puts the title in an untyped <note>
            if not title and analytic:
                for nt in b.find_all("note"):
                    if nt.get("type"):
                        continue
                    nt_txt = _clean(nt)
                    if (
                        nt_txt
                        and len(nt_txt) >= 10
                        and not nt_txt.lower().startswith(("http", "arxiv", "doi"))
                        and not is_endnote_text(nt_txt)
                    ):
                        title = nt_txt
                        break
            if not title and monogr:
                title = _first_nonempty([
                    monogr.find("title", level="m"),
                    *(t for t in monogr.find_all("title") if t.get("level") != "j"),
                ])

            authors = []
            scope = analytic or b
            for a in scope.find_all("author"):
                pers = a.find("persName")
                if pers:
                    pname = _person_name(pers)
                    if pname:
                        authors.append(pname)
            # Recover surname dropped by GROBID when only 1-letter initials were parsed (e.g. "P O" for "P. O. Dral")
            if (
                len(authors) == 1
                and raw_reference
                and all(len(tok.strip(".")) <= 1 for tok in authors[0].split())
            ):
                m_lead = re.match(
                    r"^\s*(?:\[\d+\]\s*|\d+\.\s*)?((?:[A-Z]\.\s*)+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ\-']+)\s*,",
                    raw_reference,
                )
                if m_lead:
                    authors = [m_lead.group(1).strip()]

            date_node = b.find("date", type="published") or b.find("date")
            year = None
            if date_node:
                when = date_node.get("when") or _clean(date_node)
                ym = re.search(r"(1[89]\d{2}|20\d{2})", when)
                if ym:
                    year = int(ym.group(1))

            doi_node = b.find("idno", type="DOI")
            doi = normalise_doi(doi_node.text) if doi_node and doi_node.text else None
            if not doi and raw_reference:
                doi = extract_doi_from_text(raw_reference)

            arxiv_id = None
            for idno in b.find_all("idno"):
                if (idno.get("type") or "").strip().lower() == "arxiv":
                    id_txt = _clean(idno)
                    if id_txt:
                        arxiv_id = extract_arxiv_from_text(f"arXiv:{id_txt}") or id_txt.lower().removeprefix("arxiv:").strip()
                        break
            ptr_url = None
            for ptr in b.find_all("ptr"):
                tgt = (ptr.get("target") or "").strip()
                if tgt:
                    ptr_url = tgt
                    break
            if not arxiv_id:
                for text_cand in (doi, ptr_url, raw_reference, title):
                    arxiv_id = extract_arxiv_from_text(text_cand)
                    if arxiv_id:
                        break

            journal = monogr.find("title", level="j") if monogr else None
            book = monogr.find("title", level="m") if monogr else None
            j_clean = _clean(journal) if journal else None
            b_clean = _clean(book) if book else None
            has_analytic_title = bool(analytic and _first_nonempty(analytic.find_all("title")))
            is_monograph = bool(b_clean) and not has_analytic_title
            # Do not treat an authorless, yearless, ID-less prose footnote as a book venue
            if is_monograph and b_clean == title and not (authors or year or doi or arxiv_id or ptr_url):
                venue = None
            else:
                venue = j_clean or b_clean

            if not raw_reference and not title:
                synth_parts = [p for p in (", ".join(authors), venue, str(year or ""), doi, arxiv_id, ptr_url) if p]
                if synth_parts:
                    raw_reference = ". ".join(synth_parts)

            is_endnote = bool(
                not (authors or year or doi or arxiv_id or ptr_url or j_clean)
                and is_endnote_text(raw_reference or title)
            )

            # Printed reference label/number extraction:
            # 1. Check <biblStruct n="...">
            # 2. Check <label> tag
            # 3. Check leading printed label in raw_reference, e.g. "[1]" or "1." or "1 "
            printed_num = None
            if b.get("n") and b.get("n").strip().isdigit():
                printed_num = int(b.get("n").strip())
            if printed_num is None:
                lbl_tag = b.find("label")
                if lbl_tag:
                    lbl_txt = _clean(lbl_tag)
                    m_lbl = re.match(r"^\[?(\d+)\]?[\.\:\s]*$", lbl_txt)
                    if m_lbl:
                        printed_num = int(m_lbl.group(1))
            if printed_num is None and raw_reference:
                m_raw = re.match(r"^\s*\[?(\d+)\]?[\.\:\s]", raw_reference)
                if m_raw:
                    printed_num = int(m_raw.group(1))

            effective_index = printed_num if printed_num is not None else (idx + 1)
            ref_info = {
                "xml_id": xid,
                "index": effective_index,
                "position_index": idx + 1,
                "printed_index": printed_num,
                "title": title,
                "authors": authors,
                "year": year,
                "doi": doi,
                "arxiv_id": arxiv_id,
                "url": ptr_url,
                "venue": venue,
                "is_monograph": is_monograph,
                "is_endnote": is_endnote,
                "raw_reference": raw_reference,
            }
            bib_entries.append(ref_info)
            bib_by_id[xid] = ref_info
            if printed_num is not None:
                bib_by_index[printed_num] = ref_info
            bib_by_index.setdefault(idx + 1, ref_info)

    # 2. Parse body paragraphs (with seed PDF printed-label & multi-bibliography alignment if available)
    body = soup.find("body")
    if not body:
        return []

    physical_pdf = locate_seed_pdf(seed_pdf_path, tei_source if isinstance(tei_source, str) else None)
    pdf_groups: list[dict] = []
    pdf_pages_text: list[str] = []
    endnotes_by_group: dict[int, dict[int, str]] = {}
    if physical_pdf and bib_entries:
        orig_indices = [(e.get("printed_index"), e.get("index")) for e in bib_entries]
        pdf_groups, pdf_pages_text = extract_pdf_bibliography_groups(physical_pdf)
        if pdf_groups:
            endnotes_by_group = align_biblstructs_to_pdf_groups(bib_entries, pdf_groups)
            aligned_count = sum(1 for e in bib_entries if e.get("printed_index") is not None)
            # Guard against unnumbered author-year manuscripts/textbooks whose body text
            # contains numbered exercise lists mistaken for small bibliography blocks.
            if len(bib_entries) >= 8 and aligned_count < max(3, int(0.25 * len(bib_entries))):
                pdf_groups = []
                endnotes_by_group = {}
                for entry, (orig_pi, orig_idx) in zip(bib_entries, orig_indices):
                    entry["printed_index"] = orig_pi
                    entry["index"] = orig_idx
                    entry.pop("bib_group", None)

    if pdf_groups:
        group_bib_by_id: dict[int, dict] = {g: {} for g in range(len(pdf_groups))}
        group_bib_by_index: dict[int, dict] = {g: {} for g in range(len(pdf_groups))}
        group_pos_counter: dict[int, int] = {g: 0 for g in range(len(pdf_groups))}
        for entry in bib_entries:
            g = entry.get("bib_group", 0)
            if g not in group_bib_by_id:
                g = 0
            group_pos_counter[g] = group_pos_counter.get(g, 0) + 1
            g_pos = group_pos_counter[g]
            group_bib_by_id[g][entry["xml_id"]] = entry
            if entry.get("printed_index") is not None:
                group_bib_by_index[g][entry["printed_index"]] = entry
            elif g_pos not in endnotes_by_group.get(g, {}):
                group_bib_by_index[g].setdefault(g_pos, entry)
        # Rebuild single-group index as well when len(pdf_groups) == 1
        if len(pdf_groups) == 1:
            bib_by_index = group_bib_by_index[0]

    number_map = _build_number_map(body, bib_by_id)
    group_number_maps: dict[int, dict] = {}
    if pdf_groups:
        for g in range(len(pdf_groups)):
            group_number_maps[g] = _build_number_map(body, group_bib_by_id[g])

    registry = artifact_registry(soup)

    claims = []
    seen_pairs = set()
    cur_min_page = 0

    def _stitch_adjacent_refs(p_node) -> None:
        """Merge GROBID-split author-year and lifespan <ref type='bibr'> siblings in place."""
        from bs4 import NavigableString

        cur_refs = list(p_node.find_all("ref", type="bibr"))
        idx_r = 0
        while idx_r + 1 < len(cur_refs):
            ra = cur_refs[idx_r]
            rb = cur_refs[idx_r + 1]
            if ra.parent is None or rb.parent is None or ra.parent != rb.parent:
                idx_r += 1
                continue
            between_nodes = []
            sib = ra.next_sibling
            contiguous = True
            while sib is not None and sib is not rb:
                if isinstance(sib, NavigableString) and sib.strip() in {"", "and", "&", ","}:
                    between_nodes.append(sib)
                    sib = sib.next_sibling
                else:
                    contiguous = False
                    break
            if not contiguous or sib is not rb:
                idx_r += 1
                continue

            ta = clean_text(ra)
            tb = clean_text(rb)
            mid = " ".join(s.strip() for s in between_nodes if s.strip())
            has_year_a = bool(re.search(r"\b(?:1[89]\d{2}|20\d{2})\b", ta))
            has_year_b = bool(re.search(r"\b(?:1[89]\d{2}|20\d{2})\b", tb))

            # Case 1: Split birth-death lifespan, e.g. 'Fisher (1890' + 'Fisher ( -1962) )'
            is_split_lifespan = bool(
                re.search(r"\(\s*1[5-9]\d{2}\s*$", ta)
                and re.search(r"[-–—]\s*(?:1[6-9]\d{2}|20\d{2})", tb)
            )
            # Case 2: Split co-author citation, e.g. '(Kingma and' + 'Welling, 2014, ...)'
            is_split_author = bool(
                not (ra.get("target") or "").strip()
                and not is_numeric_cite(ta)
                and not is_numeric_cite(tb)
                and not has_year_a
                and has_year_b
                and re.search(r"(?:\band\b|&|,\s*$)", ta, re.IGNORECASE)
            )
            # Case 3: Split numeric range, e.g. '[13–' + '15]' or '41–' + '43'
            is_split_num_range = bool(
                not mid
                and re.search(r"\d+\s*[-–—]\s*$", ta)
                and re.match(r"^\s*\d+\s*[\]\)]?\s*$", tb)
            )
            if is_split_lifespan or is_split_author or is_split_num_range:
                merged = f"{ta}{tb}" if is_split_num_range else " ".join(x for x in (ta, mid, tb) if x)
                ra.clear()
                ra.append(merged)
                if is_split_lifespan or is_split_num_range:
                    if "target" in ra.attrs:
                        del ra.attrs["target"]
                elif rb.get("target") and len(split_author_year_cites(merged)) == 1:
                    ra["target"] = rb["target"]
                for bn in between_nodes:
                    bn.extract()
                rb.decompose()
                cur_refs.pop(idx_r + 1)
                continue
            idx_r += 1

    for p_idx, p in enumerate(body.find_all("p")):
        if not p.find("ref", type="bibr"):
            continue
        _stitch_adjacent_refs(p)
        refs = p.find_all("ref", type="bibr")
        if not refs:
            continue

        p_id = p.get("xml:id") or p.get("id") or f"p_{p_idx}"
        section = _paragraph_section(p)
        heading = section_breadcrumb(p)
        p_artifacts = paragraph_artifacts(p, registry)

        active_bib_by_id = bib_by_id
        active_bib_by_index = bib_by_index
        active_number_map = number_map
        active_endnotes: dict[int, str] = {}
        grp_idx = 0
        if pdf_groups:
            p_plain = p.get_text(" ", strip=True)
            grp_idx, cur_min_page = assign_paragraph_bib_group(
                p_plain, pdf_pages_text, pdf_groups, min_page=cur_min_page
            )
            active_bib_by_id = group_bib_by_id.get(grp_idx, bib_by_id)
            active_bib_by_index = group_bib_by_index.get(grp_idx, bib_by_index)
            active_number_map = group_number_maps.get(grp_idx, number_map)
            active_endnotes = endnotes_by_group.get(grp_idx, {})

        # Citations become period-free tokens; what the old placeholder
        # embedded lives in this table instead.
        cites = {}
        tok_counter = 0
        for ref in refs:
            target = (ref.get("target") or "").lstrip("#")
            txt = clean_text(ref)
            has_valid_target = bool(target and target in bib_by_id)

            if is_non_citation_ref(txt, has_valid_target=has_valid_target):
                ref.replace_with(f" {txt} " if txt else " ")
                continue

            if not is_numeric_cite(txt):
                sub_cites = split_author_year_cites(txt)
            else:
                cand_num_cites = expand_numeric_cite_markers(txt)
                if len(cand_num_cites) > 1:
                    # Expand only when bibliography holds entries for the expanded indices
                    known_hits = 0
                    for nc in cand_num_cites:
                        ns = _NUM_RE.findall(nc)
                        if ns:
                            kv = int(ns[0])
                            if (
                                kv in active_bib_by_index
                                or kv in active_endnotes
                                or kv in active_number_map
                            ):
                                known_hits += 1
                    sub_cites = cand_num_cites if known_hits >= 2 else [txt]
                else:
                    sub_cites = [txt]

            replacement_tokens = []
            for sub_txt in sub_cites:
                n = tok_counter
                tok_counter += 1
                sub_target = target if len(sub_cites) == 1 else ""
                if len(sub_cites) > 1 and target and target in bib_by_id:
                    if is_numeric_cite(sub_txt):
                        ns = _NUM_RE.findall(sub_txt)
                        if ns and bib_by_id[target].get("index") == int(ns[0]):
                            sub_target = target
                    else:
                        # Retain target only for the sub-citation whose author/year matches bib_by_id[target]
                        cand_match, _, _ = _resolve_ref("", sub_txt, {target: bib_by_id[target]}, {}, {})
                        if cand_match is not None:
                            sub_target = target

                # Check if numeric marker points to a printed explanatory endnote in this bibliography group
                endnote_hit = None
                if active_endnotes and is_numeric_cite(sub_txt):
                    nums_txt = _NUM_RE.findall(sub_txt)
                    if nums_txt:
                        k_num = int(nums_txt[0])
                        if k_num in active_endnotes and (
                            k_num not in active_bib_by_index
                            or active_bib_by_index[k_num].get("is_endnote")
                            or active_bib_by_index[k_num].get("printed_index") != k_num
                        ):
                            note_text = active_endnotes[k_num]
                            endnote_hit = {
                                "xml_id": f"endnote_g{grp_idx}_{k_num}",
                                "index": k_num,
                                "printed_index": k_num,
                                "title": note_text[:120],
                                "authors": [],
                                "year": None,
                                "doi": None,
                                "venue": None,
                                "is_monograph": False,
                                "is_endnote": True,
                                "raw_reference": note_text,
                            }

                if endnote_hit is not None:
                    ref_info, resolution, note = endnote_hit, "endnote", ""
                elif is_numeric_cite(sub_txt):
                    scoped_target = sub_target if (not pdf_groups or sub_target in active_bib_by_id) else ""
                    ref_info, resolution, note = _resolve_ref(
                        scoped_target, sub_txt, active_bib_by_id, active_bib_by_index, active_number_map
                    )
                else:
                    # Author-year citations do not restart at [1] in appendices: search full bib_by_id
                    ref_info, resolution, note = _resolve_ref(
                        sub_target, sub_txt, bib_by_id, bib_by_index, number_map
                    )
                cites[n] = {
                    "target": sub_target,
                    "txt": sub_txt,
                    "ref": ref_info,
                    "resolution": resolution,
                    "note": note,
                }
                replacement_tokens.append(cite_token(n))

            ref.replace_with(" " + " ".join(replacement_tokens) + " ")

        sentences = reattach_leading_markers(paragraph_sentences(p), cites)
        display = [render_sentence(s, cites) for s in sentences]

        # The deferral ratio counts the references a paragraph leans on for
        # evidence — not the software it used or the review it points to.
        p_refs_dict = {}
        paragraph_claims = []
        for s_idx, sent in enumerate(sentences):
            tokens = [int(m) for m in CITE_TOKEN_RE.findall(sent)]
            if not tokens:
                continue

            claim_text = render_claim(sent, cites)
            if len(claim_text) < 15:
                continue
            quality = claim_quality(claim_text)
            context = sentence_context(display, s_idx)

            sent_refs = []
            for m in tokens:
                m_cite = cites[m]
                m_ref = m_cite["ref"] or _unknown_ref(m_cite["target"], m_cite["txt"])
                sent_refs.append(m_ref)

            for n in tokens:
                cite = cites[n]
                resolved = cite["ref"] is not None
                ref_info = cite["ref"] or _unknown_ref(cite["target"], cite["txt"])
                if ref_info.get("is_endnote"):
                    role = "pointer"
                else:
                    role = classify_citation_role(sent, cite_token(n), ref_info)

                ref_key = ref_info.get("xml_id") or cite["target"] or cite["txt"]
                dedup_key = (claim_text, ref_key)
                if dedup_key in seen_pairs:
                    continue
                seen_pairs.add(dedup_key)

                if resolved and role == "evidential":
                    p_refs_dict.setdefault(ref_key, ref_info)

                locus = describe_citation_locus(sent, n, cites)
                if locus["attribution_note"]:
                    focal_display = list(display)
                    focal_display[s_idx] = locus["focal_sentence"]
                    cite_context = f"{sentence_context(focal_display, s_idx)}\n{locus['attribution_note']}"
                else:
                    cite_context = context

                paragraph_claims.append({
                    "sentence": display[s_idx],
                    "claim": claim_text,
                    "target_clause": locus["target_clause"],
                    "attribution_mode": locus["mode"],
                    "cite_text": cite["txt"],
                    "target": cite["target"],
                    "ref": ref_info,
                    "resolved": resolved,
                    "resolution": cite["resolution"],
                    "resolution_note": cite["note"],
                    "role": role,
                    "claim_quality": quality,
                    "context": cite_context,
                    "section": section,
                    "section_heading": heading,
                    "artifacts": p_artifacts,
                    "cite_count": len(tokens),
                    "sentence_index": s_idx,
                    "paragraph_id": p_id,
                    "paragraph_index": p_idx,
                    "sentence_refs": sent_refs,
                    **({"seed_title": seed_title} if seed_title else {}),
                    **({"seed_self_names": seed_self_names} if seed_self_names else {}),
                })

        p_unique_refs = list(p_refs_dict.values())
        for c in paragraph_claims:
            c["paragraph_refs"] = p_unique_refs
        claims.extend(paragraph_claims)

    return claims


from research_assistant.audit.resolver import (
    _title_key,
    _match_downloaded_paper,
    attach_cited_summaries,
    enrich_item_bibliometrics,
)
from research_assistant.audit import resolver as _audit_resolver


def _load_downloaded_manifest() -> dict:
    return _audit_resolver._load_downloaded_manifest(DOWNLOADED_JSON_PATH, PULLED_PDFS_DIR)


def _is_ref_available(dl_entry: Optional[dict]) -> bool:
    """True if reference has a downloaded file, canonical corpus entry, or abstract fallback available."""
    if not dl_entry:
        return False
    path = dl_entry.get("path")
    if path and os.path.exists(path):
        return True
    if dl_entry.get("canonical_document"):
        return True
    if dl_entry.get("evidence_tier") == "abstract_only" and dl_entry.get("abstract"):
        return True
    return False


def _relocate_manifest_paths(manifest: dict) -> dict:
    return _audit_resolver._relocate_manifest_paths(manifest, PULLED_PDFS_DIR)


def contextualize_citation_queries(claims: list[dict], model: str | None = None) -> None:
    return _audit_resolver.contextualize_citation_queries(
        claims, model=model, enabled=CITATION_AUDIT_CONTEXTUALIZE_QUERIES
    )


def _extract_claim_keywords(claim: str, ref_title: str = "") -> str:
    """Extract informative keywords from a claim (using cited title only when claim is sparse)."""
    terms = extract_salient_claim_terms(claim, max_terms=12)
    if len(terms) < 3 and ref_title:
        extra = extract_salient_claim_terms(ref_title, max_terms=6)
        seen = {t.lower() for t in terms}
        for e in extra:
            if e.lower() not in seen:
                terms.append(e)
                seen.add(e.lower())
    return " ".join(terms[:12])


def _get_document_fallback_chunks(
    doc_name: str,
    collection,
    texts,
    metadatas,
    k: int = 5,
) -> list[dict]:
    """Retrieve fallback chunks directly from a document when search queries return 0 hits."""
    if not doc_name:
        return []

    # 1. Try collection.get() if it returns real dict with documents
    try:
        if hasattr(collection, "get"):
            res = collection.get(where={"document": doc_name}, limit=k)
            if isinstance(res, dict):
                docs = res.get("documents")
                metas = res.get("metadatas") or []
                if isinstance(docs, list) and docs:
                    hits = []
                    for i, d in enumerate(docs):
                        if not d or not str(d).strip():
                            continue
                        m = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {"document": doc_name}
                        if m.get("type") in EXCLUDED_FROM_EVIDENCE:
                            continue
                        hits.append({
                            "chunk_index": -1,
                            "text": str(d),
                            "metadata": m,
                            "rrf_score": 0.001,
                        })
                        if len(hits) >= k:
                            break
                    if hits:
                        return hits
    except Exception as exc:
        logger.debug("Collection fallback get failed: %s", exc)

    # 2. Try scanning metadatas / texts
    if metadatas and texts:
        hits = []
        try:
            total = len(metadatas)
            for i in range(total):
                meta = metadatas[i]
                if isinstance(meta, dict) and meta.get("document") == doc_name:
                    if meta.get("type") in EXCLUDED_FROM_EVIDENCE:
                        continue
                    text = texts[i] if i < len(texts) else ""
                    if text and str(text).strip():
                        hits.append({
                            "chunk_index": i,
                            "text": str(text),
                            "metadata": meta,
                            "rrf_score": 0.001,
                        })
                        if len(hits) >= k:
                            break
            if hits:
                return hits
        except Exception as exc:
            logger.debug("Metadata scan fallback failed: %s", exc)

    return []


def _judge_claim_entry(
    item: dict,
    collection,
    bm25,
    texts,
    metadatas,
    top_k: int = JUDGEMENT_TOP_K,
    escalate_k: int = JUDGEMENT_ESCALATE_TOP_K,
) -> dict:
    """Evaluate a single claim against its cited document in the vector store."""
    from research_assistant.judgement.judge import (
        DERIVED_FIELDS,
        PROVENANCE_FIELDS,
        REQUIRED_FIELDS,
        JudgementParseError,
    )

    retrieve_k = max(top_k, escalate_k if escalate_k > top_k else 0)
    pool_k = max(12, retrieve_k) if top_k >= 2 else retrieve_k
    search_query = item.get("search_query") or item.get("target_clause") or item["claim"]
    item["search_query"] = search_query

    try:
        hits = hybrid_search(
            search_query,
            collection,
            bm25,
            texts,
            metadatas,
            top_k=pool_k,
            doc_filter={item["document"]},
            exclude_types=EXCLUDED_FROM_EVIDENCE,
        )
    except Exception as exc:
        logger.warning("Retrieval failed for %s: %s", item.get("document"), exc)
        item["outcome"] = "retrieval_failed"
        item["judgement"] = None
        item["reason"] = f"Retrieval failed: {exc}"
        return item

    # Query relaxation fallback 1: If search_query was contextualized, try raw claim text
    if not hits and search_query != item.get("claim") and item.get("claim"):
        try:
            hits = hybrid_search(
                item["claim"],
                collection,
                bm25,
                texts,
                metadatas,
                top_k=retrieve_k,
                doc_filter={item["document"]},
                exclude_types=EXCLUDED_FROM_EVIDENCE,
            )
            if hits:
                item["search_query_relaxed"] = item["claim"]
        except Exception as exc:
            logger.debug("Fallback raw claim search failed for %s: %s", item.get("document"), exc)

    # Query relaxation fallback 2: Salient keywords from claim + cited reference title
    if not hits:
        kw = _extract_claim_keywords(item.get("claim", ""), (item.get("ref") or {}).get("title", ""))
        if kw and kw != search_query and kw != item.get("claim"):
            try:
                hits = hybrid_search(
                    kw,
                    collection,
                    bm25,
                    texts,
                    metadatas,
                    top_k=retrieve_k,
                    doc_filter={item["document"]},
                    exclude_types=EXCLUDED_FROM_EVIDENCE,
                )
                if hits:
                    item["search_query_relaxed"] = kw
            except Exception as exc:
                logger.debug("Fallback keyword search failed for %s: %s", item.get("document"), exc)

    # Query relaxation fallback 3: Document-level fallback chunks (including normalized pdf_key)
    if not hits:
        doc_candidates = [item.get("document")]
        if item.get("document") and ":" not in str(item["document"]):
            norm_k = pdf_key(str(item["document"]))
            if norm_k and norm_k not in doc_candidates:
                doc_candidates.append(norm_k)
        for doc_cand in doc_candidates:
            if doc_cand != item.get("document"):
                try:
                    hits = hybrid_search(
                        search_query,
                        collection,
                        bm25,
                        texts,
                        metadatas,
                        top_k=retrieve_k,
                        doc_filter={doc_cand},
                        exclude_types=EXCLUDED_FROM_EVIDENCE,
                    )
                    if hits:
                        item["document"] = doc_cand
                        break
                except Exception:
                    pass
            doc_hits = _get_document_fallback_chunks(
                doc_cand, collection, texts, metadatas, k=retrieve_k
            )
            if doc_hits:
                hits = doc_hits
                item["document"] = doc_cand
                item["search_query_relaxed"] = "[document fallback chunks]"
                break

    if not hits:
        item["outcome"] = "no_evidence"
        item["judgement"] = None
        item["reason"] = "No relevant passages found in the cited document."
        return item

    focal_claim = item.get("target_clause") or item.get("claim") or ""
    hits = supplement_intra_doc_lexical_hits(
        [focal_claim, item.get("claim", ""), search_query],
        {item["document"]},
        collection,
        texts,
        metadatas,
        hits,
        exclude_types=EXCLUDED_FROM_EVIDENCE,
    )
    hits = rerank_candidates(
        search_query,
        hits,
        top_k=max(pool_k, len(hits)),
        claim_text=focal_claim,
        neighbour_window=JUDGEMENT_NEIGHBOUR_WINDOW,
        primary_k=top_k,
    )

    expand_neighbours(hits, texts, metadatas, window=JUDGEMENT_NEIGHBOUR_WINDOW)
    for h in hits[:top_k]:
        if isinstance(h.get("metadata"), dict):
            enrich_item_bibliometrics(item, hit_meta=h["metadata"])
    item["evidence_tier"] = infer_evidence_tier(hits[:top_k], prior_tier=item.get("evidence_tier"))
    item["evidence"] = assemble_evidence(hits[:top_k], JUDGEMENT_EVIDENCE_MAX_CHARS)

    def _provenance(k):
        used = hits[:k]
        item["evidence_hits"] = len(used)
        item["evidence_sections"] = sorted({(h.get("metadata") or {}).get("section") for h in used} - {None})
        item["evidence_pages"] = sorted({(h.get("metadata") or {}).get("page") for h in used} - {None})

    _provenance(top_k)
    item["escalated"] = False

    claim_context = item.get("context")
    if claim_context and not str(claim_context).startswith("[Section:"):
        rel_artifacts = filter_relevant_artifacts(
            item.get("sentence") or item.get("claim"), item.get("artifacts")
        )
        citing_note = build_citing_paper_note(
            item.get("claim"),
            seed_title=item.get("seed_title"),
            seed_self_names=item.get("seed_self_names"),
        )
        sec_label = item.get("section_heading") or (
            str(item["section"]).replace("_", " ").title() if item.get("section") else None
        )
        if sec_label or rel_artifacts or citing_note:
            claim_context = compose_context(
                claim_context,
                section=sec_label,
                artifacts=rel_artifacts,
                citing_paper_note=citing_note,
            )

    try:
        verdict = _judge_once(item["claim"], item["evidence"], context=claim_context)
        if escalate_k and needs_escalation(verdict):
            # Also merge keyword-relaxed chunks on escalation so body-section evidence is not missed
            kw_esc = _extract_claim_keywords(item.get("claim", ""), (item.get("ref") or {}).get("title", ""))
            if kw_esc and kw_esc != search_query:
                try:
                    kw_hits = hybrid_search(
                        kw_esc,
                        collection,
                        bm25,
                        texts,
                        metadatas,
                        top_k=retrieve_k,
                        doc_filter={item["document"]},
                        exclude_types=EXCLUDED_FROM_EVIDENCE,
                    )
                    if kw_hits:
                        expand_neighbours(kw_hits, texts, metadatas, window=JUDGEMENT_NEIGHBOUR_WINDOW)
                        seen_cidx = {h.get("chunk_index") for h in hits if h.get("chunk_index") is not None}
                        seen_txts = {h.get("text") for h in hits}
                        for kh in kw_hits:
                            cidx = kh.get("chunk_index")
                            if (cidx is not None and cidx in seen_cidx) or kh.get("text") in seen_txts:
                                continue
                            hits.append(kh)
                            if cidx is not None:
                                seen_cidx.add(cidx)
                            seen_txts.add(kh.get("text"))
                except Exception:
                    pass
            eff_esc_k = max(escalate_k, min(len(hits), 8))
            if len(hits) > top_k:
                item["first_judgement"] = verdict.get("judgement")
                item["evidence"] = assemble_evidence(
                    hits[:eff_esc_k], max(JUDGEMENT_EVIDENCE_MAX_CHARS, 9000)
                )
                _provenance(eff_esc_k)
                item["escalated"] = True
                verdict = _judge_once(item["claim"], item["evidence"], context=claim_context)

        # Second Judge Pass: for Unclear verdicts or Low confidence
        if CITATION_JUDGEMENT_SECOND_PASS and (
            verdict.get("judgement") == "Unclear / insufficient evidence"
            or verdict.get("confidence") == "Low"
        ):
            if len(hits) > top_k and not item.get("escalated"):
                expanded_k = escalate_k if escalate_k else min(len(hits), top_k + 2)
                item["evidence"] = assemble_evidence(
                    hits[:expanded_k], JUDGEMENT_EVIDENCE_MAX_CHARS
                )
                _provenance(expanded_k)
                item["escalated"] = True

            adjudication_context = claim_context or ""
            adjudication_note = (
                "\n[Second Judge Review: Primary evaluation was unclear or low-confidence. "
                "Carefully inspect the passages for direct or implicit support, methodology, or contradictions.]"
            )
            second_ctx = f"{adjudication_context}{adjudication_note}".strip()
            try:
                second_verdict = _judge_once(item["claim"], item["evidence"], context=second_ctx)
                if isinstance(second_verdict, dict) and second_verdict.get("judgement"):
                    item["second_judge_verdict"] = second_verdict.get("judgement")
                    item["second_judge_confidence"] = second_verdict.get("confidence")
                    item["second_judge_reason"] = second_verdict.get("reason")

                    first_conf = {"Low": 0, "Medium": 1, "High": 2}.get(verdict.get("confidence"), 0)
                    second_conf = {"Low": 0, "Medium": 1, "High": 2}.get(second_verdict.get("confidence"), 0)

                    if (
                        verdict.get("judgement") == "Unclear / insufficient evidence"
                        and second_verdict.get("judgement") != "Unclear / insufficient evidence"
                    ) or (second_conf > first_conf):
                        logger.info(
                            "Second judge resolved '%s' (%s) -> '%s' (%s)",
                            verdict.get("judgement"),
                            verdict.get("confidence"),
                            second_verdict.get("judgement"),
                            second_verdict.get("confidence"),
                        )
                        verdict = second_verdict
            except Exception as exc:
                logger.debug("Second judge pass skipped: %s", exc)

        verdict = guard_abstract_only_verdict(verdict, item.get("evidence_tier"))
        item["outcome"] = "judged"
        for k in REQUIRED_FIELDS:
            if k in verdict:
                item[k] = verdict[k]
        for k in (*DERIVED_FIELDS, *PROVENANCE_FIELDS):
            if k in verdict:
                item[k] = verdict[k]
        logger.info(
            " -> %s (%s confidence)",
            verdict.get("judgement"),
            verdict.get("confidence"),
        )
    except JudgementParseError as exc:
        logger.warning("Judgement parse failed: %s", exc)
        item["outcome"] = "parse_failed"
        item["judgement"] = None
        item["reason"] = "Model returned unparseable response."
    except Exception as exc:
        logger.warning("Judgement call failed: %s", exc)
        item["outcome"] = "call_failed"
        item["judgement"] = None
        item["reason"] = f"Model evaluation error: {exc}"

    return item


def get_cached_seed_audit(seed_path: str, include_failed: bool = False) -> dict | None:
    """Retrieve an existing audit report for a seed paper if available on disk.

    Checks by filename stem, canonical arXiv/DOI/content-hash keys, and returns
    the parsed report dict, or None if no valid audit exists. A report the
    judge never ran on (the backend was down) is not returned unless
    *include_failed* — the caller then knows the corpus is ready and only the
    audit needs re-running.
    """
    if not seed_path:
        return None

    stem = os.path.splitext(os.path.basename(seed_path))[0]
    candidates = [
        os.path.join(AUDIT_DIR, f"{stem}_audit.json"),
        os.path.join(AUDIT_DIR, f"arxiv_{stem}_audit.json"),
    ]

    # Try inspecting PDF on disk to resolve canonical key (e.g. if uploaded with a temp name)
    if os.path.exists(seed_path):
        try:
            from research_assistant.shared.pdf_metadata import _inspect_pdf
            from research_assistant.shared.fetch import filename_for
            _, doi, arxiv_id = _inspect_pdf(seed_path)
            if arxiv_id:
                k = f"arxiv:{arxiv_id.strip().lower()}"
                s = os.path.splitext(filename_for(k))[0]
                candidates.append(os.path.join(AUDIT_DIR, f"{s}_audit.json"))
            elif doi:
                norm_d = normalise_doi(doi) or doi.lower().strip()
                k = f"doi:{norm_d}"
                s = os.path.splitext(filename_for(k))[0]
                candidates.append(os.path.join(AUDIT_DIR, f"{s}_audit.json"))
            else:
                with open(seed_path, "rb") as f:
                    digest = hashlib.sha1(f.read()).hexdigest()[:16]
                k = f"file:{digest}"
                s = os.path.splitext(filename_for(k))[0]
                candidates.append(os.path.join(AUDIT_DIR, f"{s}_audit.json"))
        except Exception:
            pass

    for candidate in candidates:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    report = json.load(f)
                if isinstance(report, dict) and report.get("results") is not None:
                    if _audit_never_ran(report) and not include_failed:
                        logger.info("Ignoring cached audit %s: the judge never ran (backend was down).", candidate)
                        return None
                    return report
            except Exception as exc:
                logger.warning("Failed to load cached audit from %s: %s", candidate, exc)

    return None


BACKEND_FAILURE_OUTCOMES_ALL = ("retrieval_failed", "call_failed", "not_attempted")


def _audit_never_ran(report: dict) -> bool:
    """A report the judge never got to work on — the run tripped the
    backend-failure breaker, or produced no verdict and only backend
    failures. Serving it from cache repeats an outage that is over."""
    if report.get("aborted"):
        return True
    results = report.get("results") or []
    judged = any(r.get("outcome") == "judged" for r in results)
    failed = any(r.get("outcome") in BACKEND_FAILURE_OUTCOMES_ALL for r in results)
    return failed and not judged


VERDICTS = (
    "Supports", "Partially supports", "Contradicts", "Does not support",
    "Unclear / insufficient evidence",
)

# Every way a (sentence, citation) pair ends without a verdict. Spec §3.
NOT_ASSESSED_OUTCOMES = (
    "not_downloaded", "deferred_paywalled", "cap_exceeded", "cluster_skipped",
    "not_attempted", "no_evidence", "retrieval_failed", "parse_failed",
    "call_failed", "not_a_claim", "malformed_claim", "unresolved_ref",
)
_ATTEMPTED_OUTCOMES = ("judged", "no_evidence", "retrieval_failed", "parse_failed", "call_failed")
_UNAVAILABLE_OUTCOMES = ("not_downloaded", "deferred_paywalled")
_NON_EVIDENTIAL_OUTCOMES = ("not_a_claim", "malformed_claim")
_RELIABILITY_KEYS = (
    ("high", "HIGH"), ("moderate", "MODERATE"), ("low", "LOW"),
    ("contradicted", "CONTRADICTED"), ("unsupported", "UNSUPPORTED"), ("unresolved", "UNRESOLVED"),
)


def compute_totals(results: list[dict]) -> dict:
    """The one place counts come from. Verdict counts are over judged items
    only; everything else is in not_assessed, keyed by outcome, so
    judged + sum(not_assessed) == total always holds."""
    judged = [r for r in results if r.get("outcome") == "judged"]
    non_evidential = sum(1 for r in results if r.get("outcome") in _NON_EVIDENTIAL_OUTCOMES)
    totals = {
        "total": len(results),
        "evidential_citations": len(results) - non_evidential,
        "non_evidential": non_evidential,
        "downloaded": sum(1 for r in results if r.get("downloaded")),
        "judged": len(judged),
    }
    for verdict in VERDICTS:
        totals[verdict] = sum(1 for r in judged if r.get("judgement") == verdict)
    totals["not_downloaded"] = sum(1 for r in results if r.get("outcome") == "not_downloaded")
    totals["deferred_paywalled"] = sum(1 for r in results if r.get("outcome") == "deferred_paywalled")
    not_assessed = {}
    for outcome in NOT_ASSESSED_OUTCOMES:
        n = sum(1 for r in results if r.get("outcome") == outcome)
        if n:
            not_assessed[outcome] = n
    totals["not_assessed"] = not_assessed
    totals["coverage"] = {
        "downloaded": totals["downloaded"],
        "attempted": sum(1 for r in results if r.get("outcome") in _ATTEMPTED_OUTCOMES),
        "judged": len(judged),
    }
    totals["reliability"] = {
        key: sum(1 for r in results if r.get("reliability") == rating)
        for key, rating in _RELIABILITY_KEYS
    }
    unavailable_n = sum(
        1 for r in results
        if r.get("outcome") in _UNAVAILABLE_OUTCOMES
        and (r.get("reliability") in (None, "UNRESOLVED"))
    )
    non_evid_rel_n = sum(
        1 for r in results
        if r.get("outcome") in _NON_EVIDENTIAL_OUTCOMES
        and (r.get("reliability") in (None, "UNRESOLVED"))
    )
    inconclusive_n = sum(
        1 for r in results
        if r.get("outcome") not in (*_UNAVAILABLE_OUTCOMES, *_NON_EVIDENTIAL_OUTCOMES)
        and (
            r.get("reliability") == "UNRESOLVED"
            or (
                r.get("reliability") is None
                and (r.get("outcome") != "judged" or r.get("judgement") == "Unclear / insufficient evidence")
            )
        )
    )
    totals["reliability_breakdown"] = {
        "high": totals["reliability"]["high"],
        "moderate": totals["reliability"]["moderate"],
        "low": totals["reliability"]["low"],
        "contradicted": totals["reliability"]["contradicted"],
        "unsupported": totals["reliability"]["unsupported"],
        "inconclusive": inconclusive_n,
        "unavailable_paywalled": unavailable_n,
        "non_evidential_skipped": non_evid_rel_n,
    }
    return totals


SECTION_RANK = {"results": 0, "discussion": 0, "conclusion": 0, "methods": 1}
MAX_PAIRS_PER_SENTENCE = CITATION_AUDIT_MAX_PAIRS_PER_SENTENCE
MAX_CONSECUTIVE_BACKEND_FAILURES = 3
BACKEND_FAILURE_OUTCOMES = ("retrieval_failed", "call_failed")


def prioritise_claims(
    claims: list[dict], max_pairs_per_sentence: Optional[int] = None
) -> tuple[list[dict], list[dict]]:
    """The order the budget is spent in, and the pairs it never reaches.

    Results and discussion before methods before introduction; sentences
    with few citations before "[13]–[27] have been proposed"; then, because
    most physics headings are topical ("DFT-based tight-binding Hamiltonian")
    and map to no rank, later paragraphs before earlier ones — the opening of
    a paper is background, its later pages are where results are compared
    with prior work.

    If max_pairs_per_sentence > 0 (or CITATION_AUDIT_MAX_PAIRS_PER_SENTENCE > 0),
    at most that many pairs per sentence are judged — the rest are cluster_skipped.
    By default (0), all citations in a sentence are audited.
    """
    ordered = sorted(
        claims,
        key=lambda c: (
            SECTION_RANK.get(c.get("section", "other"), 2),
            c.get("cite_count", 1),
            -c.get("paragraph_index", 0),
            c.get("sentence_index", 0),
        ),
    )
    max_pairs = (
        max_pairs_per_sentence
        if max_pairs_per_sentence is not None
        else MAX_PAIRS_PER_SENTENCE
    )
    per_sentence: dict[tuple, int] = {}
    to_judge, skipped = [], []
    for c in ordered:
        key = (c.get("paragraph_index", 0), c.get("sentence_index", 0))
        per_sentence[key] = per_sentence.get(key, 0) + 1
        if max_pairs > 0 and per_sentence[key] > max_pairs:
            c["outcome"] = "cluster_skipped"
            c["judgement"] = None
            c["reason"] = f"Sentence cites {c.get('cite_count')} papers; only {max_pairs} judged."
            skipped.append(c)
        else:
            to_judge.append(c)
    return to_judge, skipped


def cross_check_seed_audit(
    seed_path: str,
    cached_report: dict,
    search_resources=None,
    max_new_claims: int = 0,
) -> tuple[dict, dict]:
    """Cross-check an existing audit report against the current knowledge base.

    Verifies whether previously unretrieved / deferred citations are now available,
    evaluates any newly available references if max_new_claims > 0, refreshes the
    Source Assessor and Reliability Policy across all items in memory, recomputes
    totals, and persists the updated report.
    """
    t0 = time.perf_counter()
    report = dict(cached_report)
    results = [dict(r) for r in report.get("results", [])]
    seed_pdf_name = report.get("seed_name") or os.path.basename(seed_path)
    stem = os.path.splitext(os.path.basename(seed_path))[0] if seed_path else "unknown"

    downloaded_manifest = _load_downloaded_manifest()
    newly_judged_count = 0
    initial_judged_count = sum(
        1 for r in report.get("results", []) if r.get("outcome") == "judged"
    )

    # 1. Identify claims whose reference PDFs became available (or upgraded from abstract to full_text)
    unresolved_claims = [
        r for r in results
        if r.get("outcome") in (
            "not_downloaded",
            "deferred_paywalled",
            "cap_exceeded",
            "cluster_skipped",
            "no_evidence",
            "retrieval_failed",
            "call_failed",
            "not_attempted",
        )
        or (
            r.get("outcome") == "judged"
            and r.get("evidence_tier") == "abstract_only"
            and r.get("judgement") == "Unclear / insufficient evidence"
        )
    ]

    # Rows that failed on the backend already know their document; a
    # re-judge costs one call each and is what turns an outage into
    # verdicts. They do not count against max_new_claims, which budgets
    # references that only became available since the last run.
    backend_failed = [
        item for item in unresolved_claims
        if item.get("outcome") in BACKEND_FAILURE_OUTCOMES_ALL and item.get("document")
    ]
    newly_available_pairs = []
    for item in unresolved_claims:
        if item in backend_failed:
            continue
        ref_info = item.get("ref")
        matched = _match_downloaded_paper(ref_info, seed_pdf_name, downloaded_manifest)
        if not matched:
            continue
        has_fulltext = bool(matched.get("path") or matched.get("canonical_document"))
        if item.get("outcome") == "judged" and item.get("evidence_tier") == "abstract_only":
            # Only re-judge an abstract_only Unclear item when full_text is now available
            if has_fulltext:
                newly_available_pairs.append((item, matched))
        elif has_fulltext or (matched.get("evidence_tier") == "abstract_only" and matched.get("abstract")):
            newly_available_pairs.append((item, matched))

    claims_to_rejudge = []
    selected_pairs = newly_available_pairs if max_new_claims < 0 else newly_available_pairs[:max_new_claims]
    overflow_pairs = [] if max_new_claims <= 0 else newly_available_pairs[max_new_claims:]

    for item, matched in selected_pairs:
        enrich_item_bibliometrics(item, dl_entry=matched)
        item["downloaded"] = True
        doc_file = matched.get("canonical_document") or (
            os.path.basename(matched["path"])
            if matched.get("path")
            else (matched.get("key") or matched.get("title") or "unknown.pdf")
        )
        item["document"] = doc_file
        item["citation_source"] = (
            matched.get("title")
            or matched.get("raw_reference")
            or matched.get("key")
            or doc_file
        )
        if matched.get("canonical_document"):
            item["evidence_tier"] = "full_text"
        elif matched.get("evidence_tier"):
            item["evidence_tier"] = matched["evidence_tier"]
        claims_to_rejudge.append(item)

    for item, matched in overflow_pairs:
        enrich_item_bibliometrics(item, dl_entry=matched)
        item["downloaded"] = True
        item["outcome"] = "cap_exceeded"
        item["judgement"] = None
        item["reason"] = "Maximum claims evaluation budget reached during cross-check."

    to_judge = backend_failed + claims_to_rejudge
    if to_judge:
        logger.info(
            "Cross-check: re-judging %d backend-failed citation(s) and %d newly available (cap %d).",
            len(backend_failed), min(len(claims_to_rejudge), max_new_claims), max_new_claims,
        )
        if search_resources is None:
            from research_assistant.shared.db import load_search_resources

            search_resources = load_search_resources()
        collection, bm25, texts, metadatas = search_resources

        for item in to_judge:
            _judge_claim_entry(item, collection, bm25, texts, metadatas)
            newly_judged_count += 1
        # The outage the old record describes is over once anything was
        # re-judged; the banner must not outlive it.
        report["aborted"] = None

    # 2. In-memory refresh of Source Assessor and Reliability Policy across all claims
    for r in results:
        if r.get("ref"):
            dl_match = _match_downloaded_paper(r.get("ref"), seed_pdf_name, downloaded_manifest)
            if dl_match:
                enrich_item_bibliometrics(r, dl_entry=dl_match)
        source_eval = assess_source(metadata=r.get("metadata") or {}, ref_info=r.get("ref"))
        r["source_grade"] = source_eval["grade"]
        r["source_assessment"] = source_eval
        rel_eval = evaluate_reliability(
            relation=r.get("judgement"),
            source_grade=source_eval["grade"],
            confidence=r.get("confidence"),
            span_verified=r.get("span_verified"),
            rubric_violations=r.get("rubric_violations"),
            rubric_mismatch=r.get("rubric_mismatch", False),
            outcome=r.get("outcome") or "judged",
            span_cites_others=r.get("span_cites_others"),
            evidence_tier=r.get("evidence_tier", "full_text"),
        )
        r["reliability"] = rel_eval["rating"]
        r["reliability_badge"] = rel_eval["badge"]
        r["reliability_label"] = rel_eval["rating_label"]
        r["reliability_explanation"] = rel_eval["explanation"]

    # 3. Recalculate totals
    totals = compute_totals(results)

    report["totals"] = totals
    report["results"] = results
    report["cross_checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # 4. Save refreshed report
    os.makedirs(AUDIT_DIR, exist_ok=True)
    out_file = os.path.join(AUDIT_DIR, f"{stem}_audit.json")
    atomic_write_json(out_file, report)
    try:
        md_content = generate_seed_audit_markdown(report)
        out_md = os.path.join(AUDIT_DIR, f"{stem}_audit.md")
        with open(out_md, "w", encoding="utf-8") as fh:
            fh.write(md_content)
    except Exception as exc:
        logger.warning("Could not update markdown audit report: %s", exc)

    elapsed = time.perf_counter() - t0
    summary = {
        "duration_seconds": round(elapsed, 2),
        "total_claims": len(results),
        "already_judged_count": initial_judged_count,
        "newly_available_count": len(newly_available_pairs),
        "newly_judged_count": newly_judged_count,
        "from_cache": True,
    }
    return report, summary


_AUTHOR_DISCOURSE_RE = re.compile(
    r"^(?:in\s+(?:this\s+(?:paper|work|study|section|letter|article)|section\s+\d|sec\.\s*\d|appendix)|"
    r"here\s*,?\s*we\b|we\s+(?:present|introduce|propose|develop|investigate|measure|demonstrate|"
    r"show|find|observe|organize|conclude|summarize|define|consider|denote|assume|derive)\b|"
    r"our\s+(?:results|approach|method|analysis|experiments|simulations|findings)\b|"
    r"the\s+remainder\s+of\s+this\b|section\s+\d+\s+describes\b|fig(?:ure)?\.?\s*\d+|table\s+\d+)",
    re.IGNORECASE,
)


def extract_uncited_candidate_sentences(tei_source, max_sentences: int = CITATION_AUDIT_MAX_UNCITED_SENTENCES) -> list[dict]:
    """Extract clean, uncited body sentences that may state prior-work or empirical facts without attribution."""
    if isinstance(tei_source, BeautifulSoup):
        soup = tei_source
    else:
        if not tei_source or not os.path.exists(tei_source):
            return []
        with open(tei_source, "r", encoding="utf-8") as fh:
            soup = BeautifulSoup(fh, "xml")

    body = soup.find("body")
    if not body:
        return []

    candidates: list[dict] = []
    seen_sents: set[str] = set()
    for p_idx, p in enumerate(body.find_all("p")):
        p_copy = BeautifulSoup(str(p), "xml").find("p")
        if p_copy is None:
            continue
        p_id = p.get("xml:id") or p.get("id") or f"p_{p_idx}"
        section = _paragraph_section(p)
        if section in {"acknowledgements", "references", "appendix"}:
            continue
        heading = section_breadcrumb(p)

        # Mark any bibr ref that is a real citation so sentences carrying citations are skipped
        for ref in p_copy.find_all("ref", type="bibr"):
            txt = clean_text(ref)
            target = (ref.get("target") or "").lstrip("#")
            if is_non_citation_ref(txt, has_valid_target=bool(target)):
                ref.replace_with(f" {txt} " if txt else " ")
            else:
                ref.replace_with(" __HAS_CITE__ ")

        for sent in paragraph_sentences(p_copy):
            if "__HAS_CITE__" in sent:
                continue
            cleaned = tidy_punctuation(sent)
            words = cleaned.split()
            if len(words) < 8 or len(words) > 70:
                continue
            if "fragment" in claim_quality(cleaned):
                continue
            if _AUTHOR_DISCOURSE_RE.search(cleaned):
                continue
            if cleaned in seen_sents:
                continue
            seen_sents.add(cleaned)
            candidates.append({
                "sentence": cleaned,
                "section": section,
                "section_heading": heading,
                "paragraph_id": p_id,
            })
            if max_sentences > 0 and len(candidates) >= max_sentences:
                return candidates
    return candidates


def detect_uncited_claims(
    tei_source,
    max_sentences: int = CITATION_AUDIT_MAX_UNCITED_SENTENCES,
    batch_checker=None,
) -> list[dict]:
    """Run batched citation-worthiness check over uncited candidate sentences in *tei_source*."""
    from research_assistant.config import CITATION_CHECK_BATCH_SIZE

    candidates = extract_uncited_candidate_sentences(tei_source, max_sentences=max_sentences)
    if not candidates:
        return []

    if batch_checker is None:
        from research_assistant.agents.agent5_batch_citer import _batch_needs_citation
        batch_checker = _batch_needs_citation

    batch_size = max(1, int(CITATION_CHECK_BATCH_SIZE or 15))
    flagged: list[dict] = []
    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start:start + batch_size]
        try:
            needs = batch_checker([c["sentence"] for c in chunk])
            for cand, need in zip(chunk, needs):
                if need:
                    flagged.append(cand)
        except Exception as exc:
            logger.warning("Uncited-claim check failed on batch at %d: %s", start, exc)
    return flagged


def audit_seed_citations(
    seed_path: str,
    search_resources=None,
    max_claims: int = CITATION_AUDIT_MAX_CLAIMS,
    top_k: int = JUDGEMENT_TOP_K,
    force: bool = False,
    skip_if_cached: bool = True,
    concurrency: int | None = None,
    check_uncited: bool | None = None,
) -> dict:
    """Audit the in-text citation claims of an uploaded seed paper against the local corpus.

    Args:
        seed_path: Path to the seed PDF.
        search_resources: Tuple of (collection, bm25, texts, metadatas) or None to load.
        max_claims: Maximum number of claims citing downloaded papers to judge with the LLM.
            Defaults to CITATION_AUDIT_MAX_CLAIMS; the rest are reported cap_exceeded.
        top_k: Number of hits to retrieve per claim.
        force: If True, bypasses any cached audit and runs from scratch.
        skip_if_cached: If True and an audit already exists, cross-checks and returns it.

    Returns:
        Structured audit report dict containing totals and detailed item results.
    """
    if not force and skip_if_cached:
        cached = get_cached_seed_audit(seed_path)
        if cached:
            logger.info("Found cached audit report for %s — running fast cross-check", seed_path)
            cc_cap = max_claims if max_claims > 0 else -1
            report, summary = cross_check_seed_audit(
                seed_path, cached, search_resources=search_resources, max_new_claims=cc_cap
            )
            report["from_cache"] = True
            report["cross_check_summary"] = summary
            return report
    stem = os.path.splitext(os.path.basename(seed_path))[0] if seed_path else "unknown"
    seed_pdf_name = os.path.basename(seed_path) if seed_path else ""

    tei_path = find_tei_for_seed(seed_path)
    if not tei_path:
        logger.info("No TEI XML found for seed PDF %s — skipping citation audit.", seed_path)
        return {
            "seed_path": seed_path,
            "seed_name": seed_pdf_name,
            "error": "No GROBID TEI XML found for seed PDF.",
            "totals": {
                "total": 0,
                "downloaded": 0,
                "judged": 0,
                "Supports": 0,
                "Partially supports": 0,
                "Contradicts": 0,
                "Does not support": 0,
                "Unclear / insufficient evidence": 0,
                "not_downloaded": 0,
                "deferred_paywalled": 0,
            },
            "results": [],
        }

    claims = extract_seed_citation_claims(tei_path, seed_pdf_path=seed_path)
    if not claims:
        logger.info("No in-text citation claims found in %s.", tei_path)
        return {
            "audit_schema_version": AUDIT_SCHEMA_VERSION,
            "seed_path": seed_path,
            "seed_name": seed_pdf_name,
            "generated": datetime.now().isoformat(timespec="seconds"),
            "model": _judgement_model(),
            "totals": {
                "total": 0,
                "downloaded": 0,
                "judged": 0,
                "Supports": 0,
                "Partially supports": 0,
                "Contradicts": 0,
                "Does not support": 0,
                "Unclear / insufficient evidence": 0,
                "not_downloaded": 0,
                "deferred_paywalled": 0,
            },
            "results": [],
        }

    # Pairs that are not claims about a paper's findings never reach the
    # judge, the budget, or the missing-references table.
    skipped_claims = []
    judgeable = []
    for c in claims:
        c["judgement"] = None
        if not c.get("resolved", True):
            c["outcome"] = "unresolved_ref"
            c["reason"] = c.get("resolution_note") or "Citation marker could not be matched to a bibliography entry."
        elif c.get("role") in SKIP_ROLES:
            c["outcome"] = "not_a_claim"
            c["reason"] = f"{c['role']} citation — not a verifiable claim about the cited paper."
        elif {"placeholder_residue", "fragment"} & set(c.get("claim_quality") or []):
            c["outcome"] = "malformed_claim"
            c["reason"] = f"Extracted sentence is not judgeable: {', '.join(c['claim_quality'])}."
        else:
            judgeable.append(c)
            continue
        c["downloaded"] = False
        skipped_claims.append(c)
    claims = judgeable

    downloaded_manifest = _load_downloaded_manifest()

    # Group claims by paragraph_id to evaluate paywall ratio per paragraph
    paragraphs = {}
    for c in claims:
        p_id = c.get("paragraph_id", "p_0")
        paragraphs.setdefault(p_id, []).append(c)

    downloaded_claims = []
    undownloaded_claims = []
    deferred_claims = []

    for p_id, p_claims in paragraphs.items():
        # Get unique paragraph refs
        p_refs = p_claims[0].get("paragraph_refs", [])
        if not p_refs:
            seen_r = {}
            for c in p_claims:
                r = c.get("ref")
                if r:
                    rk = r.get("xml_id") or normalise_doi(r.get("doi")) or r.get("title")
                    seen_r[rk] = r
            p_refs = list(seen_r.values())

        total_p_refs = len(p_refs)
        missing_p_refs = []

        for r in p_refs:
            dl_entry = _match_downloaded_paper(r, seed_pdf_name, downloaded_manifest)
            if not _is_ref_available(dl_entry):
                missing_p_refs.append(r)

        missing_count = len(missing_p_refs)
        paywall_ratio = (missing_count / total_p_refs) if total_p_refs > 0 else 0.0

        pct_threshold = CITATION_AUDIT_DEFERRED_PAYWALL_THRESHOLD
        if total_p_refs > 0 and paywall_ratio > pct_threshold:
            # Paragraph is >90% paywalled: defer evaluation of claims
            for c in p_claims:
                dl_entry = _match_downloaded_paper(c.get("ref"), seed_pdf_name, downloaded_manifest)
                if dl_entry:
                    enrich_item_bibliometrics(c, dl_entry=dl_entry)
                c["downloaded"] = False
                c["outcome"] = "deferred_paywalled"
                c["judgement"] = "Deferred (pending paywalled evidence)"
                pct_str = f"{int(pct_threshold * 100)}%"
                c["reason"] = (
                    f"Evaluation deferred: {missing_count}/{total_p_refs} references cited in this "
                    f"paragraph are missing from the corpus (>{pct_str} paywalled)."
                )
                c["paragraph_missing_refs"] = missing_p_refs
                c["paywall_ratio"] = paywall_ratio
                deferred_claims.append(c)
        else:
            # Under threshold: evaluate available citations directly ("work it out")
            for c in p_claims:
                ref = c.get("ref")
                dl_entry = _match_downloaded_paper(ref, seed_pdf_name, downloaded_manifest)
                if dl_entry:
                    enrich_item_bibliometrics(c, dl_entry=dl_entry)
                is_dl = _is_ref_available(dl_entry)
                c["paragraph_missing_refs"] = missing_p_refs
                c["paywall_ratio"] = paywall_ratio

                # Identify if other references cited in this exact same sentence are missing from the corpus
                curr_key = (ref.get("xml_id") or normalise_doi(ref.get("doi")) or ref.get("title")) if ref else None
                compound_missing = []
                for s_ref in c.get("sentence_refs", []):
                    s_key = s_ref.get("xml_id") or normalise_doi(s_ref.get("doi")) or s_ref.get("title")
                    if s_key and s_key != curr_key:
                        s_dl = _match_downloaded_paper(s_ref, seed_pdf_name, downloaded_manifest)
                        if not _is_ref_available(s_dl):
                            compound_missing.append(s_ref)
                if compound_missing:
                    c["compound_missing_refs"] = compound_missing

                if is_dl:
                    c["downloaded"] = True
                    c["document"] = dl_entry.get("canonical_document") or (
                        os.path.basename(dl_entry["path"])
                        if dl_entry.get("path")
                        else (dl_entry.get("key") or dl_entry.get("title") or "unknown.pdf")
                    )
                    c["citation_source"] = (
                        dl_entry.get("title")
                        or dl_entry.get("raw_reference")
                        or dl_entry.get("key")
                    )
                    if dl_entry.get("canonical_document"):
                        c["evidence_tier"] = "full_text"
                    elif dl_entry.get("evidence_tier"):
                        c["evidence_tier"] = dl_entry["evidence_tier"]
                    downloaded_claims.append(c)
                else:
                    c["downloaded"] = False
                    c["outcome"] = "not_downloaded"
                    c["judgement"] = "Not downloaded"
                    c["reason"] = "Reference PDF was not available or could not be downloaded."
                    undownloaded_claims.append(c)

    logger.info(
        "Found %d in-text citation claims (%d cite downloaded references, %d not in corpus, %d deferred paywalled).",
        len(claims),
        len(downloaded_claims),
        len(undownloaded_claims),
        len(deferred_claims),
    )

    ordered, cluster_skipped = prioritise_claims(downloaded_claims)
    if max_claims <= 0:
        claims_to_judge = ordered
        overflow = []
    else:
        claims_to_judge = ordered[:max_claims]
        overflow = ordered[max_claims:]
    aborted = None

    if claims_to_judge:
        pipeline_status.update_progress(
            detail="Reading cited summaries; contextualizing search queries",
            item_current=0,
            item_total=len(claims_to_judge),
        )
        attach_cited_summaries(claims_to_judge)
        contextualize_citation_queries(claims_to_judge)

        if search_resources is None:
            from research_assistant.shared.db import load_search_resources

            search_resources = load_search_resources()
        collection, bm25, texts, metadatas = search_resources

        escalate_k = JUDGEMENT_ESCALATE_TOP_K if JUDGEMENT_ESCALATE_TOP_K > top_k else 0
        consecutive_failures = 0

        default_concurrency = int(os.environ.get("CITATION_VERIFY_CONCURRENCY", "1"))
        workers = concurrency if concurrency is not None else default_concurrency
        workers = max(1, min(workers, len(claims_to_judge))) if claims_to_judge else 1

        if workers <= 1:
            for i, item in enumerate(claims_to_judge, 1):
                if pipeline_status.is_cancel_requested():
                    logger.info("Citation audit halted by cancellation request at item %d/%d", i, len(claims_to_judge))
                    aborted = {"reason": "User requested cancellation", "after_attempted": i - 1}
                if aborted:
                    item["outcome"] = "not_attempted"
                    item["judgement"] = None
                    item["reason"] = f"Audit stopped after {aborted['after_attempted']} attempts: {aborted['reason']}"
                    continue
                ref_info = item.get("ref") or {}
                ref_lbl = f"[{ref_info.get('index') or '?'}] {ref_info.get('title') or item.get('cite_text', '')}"
                pipeline_status.update_progress(
                    item_current=i,
                    item_total=len(claims_to_judge),
                    current_item_name=ref_lbl[:60],
                    detail=f"Judging citation ({i}/{len(claims_to_judge)}): {ref_lbl[:40]}",
                )
                logger.info("[%d/%d] Auditing citation %s: %s", i, len(claims_to_judge), ref_lbl[:40], item["claim"][:80])
                _judge_claim_entry(item, collection, bm25, texts, metadatas, top_k=top_k, escalate_k=escalate_k)

                # A backend that is down fails every call the same way; three in a
                # row is that, not three unlucky citations. Stop, say so, keep
                # the budget for a run that can use it.
                if item.get("outcome") in BACKEND_FAILURE_OUTCOMES:
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_BACKEND_FAILURES:
                        aborted = {"reason": item.get("reason", item["outcome"]), "after_attempted": i}
                        logger.warning("Audit stopped after %d consecutive backend failures: %s", i, aborted["reason"])
                        pipeline_status.add_event(f"⚠️ Citation audit stopped: {aborted['reason'][:80]}")
                else:
                    consecutive_failures = 0
        else:
            def _judge_task(i_item):
                i, item = i_item
                if pipeline_status.is_cancel_requested():
                    item["outcome"] = "not_attempted"
                    item["judgement"] = None
                    item["reason"] = "Audit stopped: User requested cancellation"
                    return i, item
                ref_info = item.get("ref") or {}
                ref_lbl = f"[{ref_info.get('index') or '?'}] {ref_info.get('title') or item.get('cite_text', '')}"
                logger.info("[%d/%d] Auditing citation %s: %s", i, len(claims_to_judge), ref_lbl[:40], item["claim"][:80])
                _judge_claim_entry(item, collection, bm25, texts, metadatas, top_k=top_k, escalate_k=escalate_k)
                return i, item

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_idx = {
                    executor.submit(_judge_task, (i, item)): i
                    for i, item in enumerate(claims_to_judge, 1)
                }
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_idx):
                    completed_count += 1
                    idx, item = future.result()
                    ref_info = item.get("ref") or {}
                    ref_lbl = f"[{ref_info.get('index') or '?'}] {ref_info.get('title') or item.get('cite_text', '')}"
                    pipeline_status.update_progress(
                        item_current=completed_count,
                        item_total=len(claims_to_judge),
                        current_item_name=ref_lbl[:60],
                        detail=f"Judged ({completed_count}/{len(claims_to_judge)}): {ref_lbl[:40]}",
                    )

    for c in overflow:
        if not c.get("outcome"):
            c["outcome"] = "cap_exceeded"
            c["judgement"] = None
            c["reason"] = "Maximum claims evaluation budget reached."

    all_results = claims_to_judge + overflow + cluster_skipped + undownloaded_claims + deferred_claims + skipped_claims

    for r in all_results:
        source_eval = assess_source(metadata=r.get("metadata") or {}, ref_info=r.get("ref"))
        r["source_grade"] = source_eval["grade"]
        r["source_assessment"] = source_eval
        rel_eval = evaluate_reliability(
            relation=r.get("judgement"),
            source_grade=source_eval["grade"],
            confidence=r.get("confidence"),
            span_verified=r.get("span_verified"),
            rubric_violations=r.get("rubric_violations"),
            rubric_mismatch=r.get("rubric_mismatch", False),
            outcome=r.get("outcome") or "judged",
            span_cites_others=r.get("span_cites_others"),
            evidence_tier=r.get("evidence_tier", "full_text"),
        )
        r["reliability"] = rel_eval["rating"]
        r["reliability_badge"] = rel_eval["badge"]
        r["reliability_label"] = rel_eval["rating_label"]
        r["reliability_explanation"] = rel_eval["explanation"]

    totals = compute_totals(all_results)

    run_uncited = CITATION_AUDIT_CHECK_UNCITED if check_uncited is None else check_uncited
    uncited_claims: list[dict] = []
    if run_uncited and not aborted:
        uncited_claims = detect_uncited_claims(tei_path)
        totals["uncited_claims_flagged"] = len(uncited_claims)

    report = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "seed_path": seed_path,
        "seed_name": seed_pdf_name,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "model": _judgement_model(),
        "totals": totals,
        "results": all_results,
        "aborted": aborted,
        **({"uncited_claims": uncited_claims} if run_uncited else {}),
    }

    os.makedirs(AUDIT_DIR, exist_ok=True)
    out_file = os.path.join(AUDIT_DIR, f"{stem}_audit.json")
    atomic_write_json(out_file, report)
    logger.info("Saved seed citation audit to %s", out_file)

    # Every run is kept: the latest overwrites <stem>_audit.json as before,
    # and a timestamped copy accumulates real (claim, evidence, verdict)
    # triples for the judge evaluation set.
    history_dir = os.path.join(AUDIT_DIR, "history")
    os.makedirs(history_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    atomic_write_json(os.path.join(history_dir, f"{stem}_{stamp}_audit.json"), report)

    try:
        md_content = generate_seed_audit_markdown(report)
        out_md = os.path.join(AUDIT_DIR, f"{stem}_audit.md")
        with open(out_md, "w", encoding="utf-8") as fh:
            fh.write(md_content)
        logger.info("Saved seed citation audit report to %s", out_md)
    except Exception as exc:
        logger.warning("Could not write markdown audit report: %s", exc)

    return report


def get_deferred_missing_references(audit_report: dict) -> list[dict]:
    """Aggregates all unique missing references blocking deferred paragraphs.

    Returns a list of dicts with keys:
        - xml_id: str
        - index: Optional[int]
        - title: str
        - authors: list[str]
        - year: Optional[int]
        - doi: Optional[str]
        - raw_reference: Optional[str]
        - affected_claims: list[str] (unique claim sentences requiring this reference)
    """
    if not audit_report:
        return []

    results = audit_report.get("results", [])
    missing_results = [
        r for r in results
        if r.get("outcome") in ("deferred_paywalled", "not_downloaded") or not r.get("downloaded")
    ]

    missing_map = {}

    for item in missing_results:
        sent = item.get("sentence") or item.get("claim") or ""
        missing_refs = item.get("paragraph_missing_refs") or []
        if not missing_refs and item.get("ref"):
            missing_refs = [item["ref"]]

        for m_ref in missing_refs:
            if not isinstance(m_ref, dict):
                continue
            doi = normalise_doi(m_ref.get("doi"))
            xml_id = m_ref.get("xml_id") or ""
            title = (m_ref.get("title") or "").strip()

            key = xml_id or doi or title.lower()
            if not key:
                continue

            if key not in missing_map:
                missing_map[key] = {
                    "xml_id": xml_id,
                    "index": m_ref.get("index"),
                    "title": title or "Unknown Title",
                    "authors": m_ref.get("authors") or [],
                    "year": m_ref.get("year"),
                    "doi": doi or m_ref.get("doi"),
                    "raw_reference": m_ref.get("raw_reference"),
                    "affected_claims": [],
                }

            if sent and sent not in missing_map[key]["affected_claims"]:
                missing_map[key]["affected_claims"].append(sent)

    return list(missing_map.values())
 
 
get_missing_references = get_deferred_missing_references


def save_and_register_reference_pdf(
    pdf_bytes: bytes,
    ref_info: dict,
    seed_pdf_name: str,
    original_filename: str = "reference.pdf",
    pulled_pdfs_dir: str = PULLED_PDFS_DIR,
    downloaded_manifest_path: str = DOWNLOADED_JSON_PATH,
    ingest: bool = True,
) -> dict:
    """Save an uploaded reference PDF, register it in downloaded.json atomically, and ingest into ChromaDB.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF file.
        ref_info: Reference metadata dict (xml_id, doi, title, authors, year, etc.).
        seed_pdf_name: Filename of the seed paper that cited this reference.
        original_filename: Original name of the uploaded PDF file.
        pulled_pdfs_dir: Directory where reference PDFs are stored.
        downloaded_manifest_path: Path to downloaded.json manifest.
        ingest: Whether to invoke ChromaDB ingestion immediately.

    Returns:
        Metadata dict of the registered paper.
    """
    if not pdf_bytes:
        raise ValueError("pdf_bytes must be non-empty bytes")

    ref_info = ref_info or {}
    doi = normalise_doi(ref_info.get("doi"))
    xml_id = ref_info.get("xml_id")
    title = ref_info.get("title") or ""

    # Sanitize / determine file name
    if original_filename and original_filename != "reference.pdf":
        clean_base = re.sub(r"[^\w\.\-]", "_", os.path.basename(original_filename))
    elif doi:
        clean_base = f"{doi.replace('/', '_')}.pdf"
    elif xml_id:
        clean_base = f"{xml_id}.pdf"
    elif title:
        clean_title = re.sub(r"[^\w\-]", "_", title)[:40].strip("_")
        clean_base = f"{clean_title}.pdf"
    else:
        clean_base = "reference.pdf"

    if not clean_base.lower().endswith(".pdf"):
        clean_base += ".pdf"

    os.makedirs(pulled_pdfs_dir, exist_ok=True)
    dest = os.path.join(pulled_pdfs_dir, clean_base)
    with open(dest, "wb") as fh:
        fh.write(pdf_bytes)
    logger.info("Saved reference PDF to %s (%d bytes)", dest, len(pdf_bytes))

    manifest = {}
    if os.path.exists(downloaded_manifest_path):
        try:
            with open(downloaded_manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except Exception as exc:
            logger.warning("Could not read manifest at %s: %s", downloaded_manifest_path, exc)

    if doi:
        key = f"doi:{doi}"
    elif xml_id and seed_pdf_name:
        key = f"xml:{xml_id}:{seed_pdf_name}"
    elif xml_id:
        key = f"xml:{xml_id}"
    else:
        key = clean_base

    entry = {
        "key": key,
        "path": dest,
        "provider": "manual_upload",
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "title": ref_info.get("title"),
        "raw_reference": ref_info.get("raw_reference"),
        "doi": doi,
        "xml_id": xml_id,
        "cited_by": seed_pdf_name,
    }
    manifest[key] = entry

    os.makedirs(os.path.dirname(os.path.abspath(downloaded_manifest_path)), exist_ok=True)
    atomic_write_json(downloaded_manifest_path, manifest, ensure_ascii=False)
    logger.info("Registered %s in manifest %s", key, downloaded_manifest_path)

    if ingest:
        try:
            from research_assistant.agents import agent6_manual_ingestor

            citation_lbl = ref_info.get("title") or os.path.splitext(clean_base)[0]
            ingest_res = agent6_manual_ingestor.ingest_manual_pdf(dest, citation_string=citation_lbl)
            entry["ingest_result"] = ingest_res
            entry["ingested"] = True
            logger.info("Ingested %s: %s", dest, ingest_res)
        except Exception as exc:
            logger.warning("Ingestion of %s failed: %s", dest, exc)
            entry["ingest_result"] = None
            entry["ingested"] = False
            entry["ingest_error"] = str(exc)
    else:
        entry["ingested"] = False

    return entry


from research_assistant.audit.report import (
    _GUARD_VIOLATION,
    _SPAN_VIOLATION,
    _slot,
    explain_verdict_steps,
    explain_rubric_verdict,
    _audit_item_badge,
    unassessed_bucket,
    get_deferred_missing_references,
    generate_seed_audit_markdown,
)
