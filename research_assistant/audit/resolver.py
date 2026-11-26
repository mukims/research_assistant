"""
resolver.py — Reference resolution, manifest matching, query contextualization, and summary attachment.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Optional

from research_assistant.config import (
    CITATION_AUDIT_CONTEXTUALIZE_QUERIES,
    DOWNLOADED_JSON_PATH,
    PULLED_PDFS_DIR,
)
from research_assistant.shared.claim_text import clean_text, is_numeric_cite
from research_assistant.shared.log import get_logger
from research_assistant.shared import pipeline_status
from research_assistant.shared.paper_index import normalise_arxiv
from research_assistant.shared.source_key import (
    extract_arxiv_from_text,
    normalise_doi,
    source_key,
)

logger = get_logger("audit_resolver")

_NUM_RE = re.compile(r"\d+")
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})([a-z])?\b")
_CHAR_FOLD_MAP = str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "ß": "ss", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "đ": "d", "Đ": "D"})


def _fold_accents(text: str) -> str:
    """Strip diacritics while preserving case and word boundaries (e.g. 'Bollobás' -> 'Bollobas')."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFKD", str(text).translate(_CHAR_FOLD_MAP))
    return "".join(c for c in norm if not unicodedata.combining(c))


def _looks_like_reference(entry: Optional[dict]) -> bool:
    """A bibliography entry with an author, a year, a DOI or a venue. A
    footnote GROBID swept into the list has none of them."""
    return bool(entry) and bool(
        entry.get("authors") or entry.get("year") or entry.get("doi") or entry.get("venue")
    )


def _unknown_ref(target: str, txt: str) -> dict:
    return {
        "xml_id": target or f"unknown_{txt}",
        "index": None,
        "title": txt or "Unknown reference",
        "authors": [],
        "year": None,
        "doi": None,
        "raw_reference": txt,
        "venue": None,
        "is_monograph": False,
    }


_ROMAN_TO_INT = {
    "i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000,
}


def _roman_to_int(s: str) -> Optional[int]:
    """Convert Roman numeral string to integer."""
    s = s.strip().lower()
    if not s or not all(c in _ROMAN_TO_INT for c in s):
        return None
    val = 0
    prev = 0
    for c in reversed(s):
        curr = _ROMAN_TO_INT[c]
        if curr >= prev:
            val += curr
        else:
            val -= curr
        prev = curr
    return val if val > 0 else None


def _int_to_roman(num: int) -> str:
    """Convert integer to lowercase Roman numeral string."""
    val_map = [
        (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"),
        (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
        (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")
    ]
    roman = ""
    for val, sym in val_map:
        while num >= val:
            roman += sym
            num -= val
    return roman


def _ref_matches_number(entry: Optional[dict], k: Optional[int]) -> bool:
    if not entry or k is None:
        return False
    if entry.get("printed_index") == k:
        return True
    raw = (entry.get("raw_reference") or "").strip()
    if re.match(rf"^\s*\[?{k}\]?[\.\:\s]", raw):
        return True
    # Match Roman numeral prefix in raw_reference if 1 <= k <= 100
    if 1 <= k <= 100:
        r_str = _int_to_roman(k)
        if re.match(rf"^\s*\[?{re.escape(r_str)}\]?[\.\:\s]", raw, re.IGNORECASE):
            return True
    return False


def _resolve_ref(target, txt, bib_by_id, bib_by_index, number_map) -> tuple[Optional[dict], str, str]:
    """(reference, how, note). Resolve in-text citation marker against bibliography.
    Resolves against printed reference labels/numbers and author-year strings in raw_reference
    before falling back to positional index to eliminate GROBID index shift.
    """
    tgt = bib_by_id.get(target)
    if not is_numeric_cite(txt):
        if tgt is not None:
            return tgt, "target", ""
        txt_folded = _fold_accents(txt or "")
        year_m = _YEAR_RE.search(txt_folded)
        req_year = int(year_m.group(1)) if year_m else None
        has_et_al = bool(re.search(r"\bet\s+al\b", txt_folded, re.IGNORECASE))
        cite_cap_words = re.findall(r"\b(?:[A-Z]{2,}|[A-Z][a-zA-Z\-']{1,})\b", txt_folded)
        first_cite_surname = cite_cap_words[0].lower() if cite_cap_words else None

        best_entry = None
        best_score = -1
        for info in bib_by_id.values():
            surnames = []
            for a in info.get("authors", []):
                a_clean = _fold_accents(a).strip()
                if not a_clean:
                    continue
                if "," in a_clean:
                    surnames.append(a_clean.split(",")[0].strip())
                else:
                    surnames.append(a_clean.split()[-1].strip())
            raw = _fold_accents(info.get("raw_reference") or "")
            title = _fold_accents(info.get("title") or "")

            matched_surnames = 0
            first_author_matched = False
            for idx_s, s in enumerate(surnames):
                cands = [s] + [tok for tok in s.split() if len(tok) > 2]
                if any(len(c) > 1 and re.search(rf"\b{re.escape(c)}\b", txt_folded, re.IGNORECASE) for c in cands):
                    matched_surnames += 1
                    if idx_s == 0:
                        first_author_matched = True

            fallback_hit = False
            if matched_surnames == 0 and cite_cap_words:
                # Match Titlecase names or all-caps institutional acronyms (e.g. IUPAC, NASA)
                fallback_hit = any(
                    len(auth) > 2 and (
                        re.search(rf"\b{re.escape(auth)}\b", raw, re.IGNORECASE)
                        or re.search(rf"\b{re.escape(auth)}\b", title, re.IGNORECASE)
                    )
                    for auth in cite_cap_words
                    if auth.lower() not in {"and", "the", "for", "see", "ref", "refs", "vol", "chap"}
                )

            if matched_surnames == 0 and not fallback_hit:
                continue

            info_year = info.get("year")
            year_score = 0
            if req_year is not None:
                if info_year == req_year or re.search(rf"\b{req_year}[a-z]?\b", raw):
                    year_score = 20
                elif info_year is not None and abs(int(info_year) - req_year) == 1 and matched_surnames >= 1 and first_author_matched:
                    # Preprint-to-journal 1-year lag fallback when first author agrees
                    year_score = 4
                else:
                    continue
            else:
                # Without a year in the citation marker, require strong author match
                if not first_author_matched and matched_surnames == 0:
                    continue
                year_score = 2

            score = year_score + (matched_surnames * 5)
            if first_author_matched:
                score += 8
                if first_cite_surname and surnames and surnames[0].lower() == first_cite_surname:
                    score += 4
            if has_et_al and len(surnames) >= 3:
                score += 2
            elif not has_et_al and surnames and matched_surnames == len(surnames):
                score += 3

            if score > best_score:
                best_score = score
                best_entry = info

        if best_entry is not None:
            return best_entry, "surname", ""
        return None, "unresolved", ""

    k = None
    nums = _NUM_RE.findall(txt)
    if nums:
        k = int(nums[0])
    else:
        m_roman = re.search(r"\b([ivxlcdm]+)\b", txt, re.IGNORECASE)
        if m_roman:
            k = _roman_to_int(m_roman.group(1))

    # Check printed reference match (printed_index or raw_reference starting with [k] or k.)
    if k is not None:
        if _ref_matches_number(tgt, k):
            return tgt, "target", ""
        for entry in bib_by_id.values():
            if _ref_matches_number(entry, k):
                return entry, "printed_label", ""

    at_position = bib_by_index.get(k) if k is not None else None
    pos = at_position if _looks_like_reference(at_position) else None
    linked = number_map.get(k) if k is not None else None

    def _name(entry):
        return (entry.get("title") or entry.get("raw_reference") or entry.get("xml_id") or "?")[:60]

    if tgt is not None and pos is not None:
        if tgt is pos:
            return tgt, "target", ""
        # Two real, different entries. Only the printed label the citation names can
        # break the tie; "position_index == k" cannot, because that is the very
        # position witness that disagrees with the link (2026-11-02: it silently
        # overrode the link and judged the claim against a guess).
        tgt_labelled, pos_labelled = _ref_matches_number(tgt, k), _ref_matches_number(pos, k)
        if tgt_labelled != pos_labelled:
            return (tgt if tgt_labelled else pos), "printed_label", ""
        return None, "ambiguous", (
            f"GROBID links {txt} to '{_name(tgt)}' but list position {k} is '{_name(pos)}'; not judged against a guess."
        )
    if tgt is not None:
        return tgt, "target", ""
    if pos is not None and linked is not None and linked is not pos:
        if _ref_matches_number(pos, k):
            return pos, "position", ""
        if _ref_matches_number(linked, k):
            return linked, "number_map", ""
        return None, "ambiguous", (
            f"{txt} was linked elsewhere to '{_name(linked)}' but list position {k} is '{_name(pos)}'; not judged against a guess."
        )
    if pos is not None:
        return pos, "position", ""
    if linked is not None:
        return linked, "number_map", ""
    if at_position is not None:
        return None, "unresolved", f"List position {k} is a note, not a reference: '{_name(at_position)}'."
    return None, "unresolved", ""


def _load_downloaded_manifest(manifest_path: str | None = None, pulled_dir: str | None = None) -> dict:
    path = manifest_path or DOWNLOADED_JSON_PATH
    manifest = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                manifest = loaded
        except Exception as e:
            logger.warning("Could not read %s: %s", path, e)
            manifest = {}

    manifest = _relocate_manifest_paths(manifest, pulled_dir=pulled_dir)

    # Enrich from sibling paper_index.json (aliases + corpus papers not in downloaded.json)
    idx_path = os.path.join(os.path.dirname(os.path.abspath(path)), "paper_index.json")
    if os.path.exists(idx_path):
        try:
            from research_assistant.shared.academic_apis import verify_candidate_match
            from research_assistant.shared.paper_index import PaperIndex
            from research_assistant.shared.source_key import pdf_key

            idx = PaperIndex.load(idx_path)
            known_docs = set()
            for k, entry in manifest.items():
                if not isinstance(entry, dict):
                    continue
                doc_cand = pdf_key(entry["path"]) if entry.get("path") else (entry.get("key") or k)
                resolved = idx.resolve(doc_cand)
                if resolved is not None:
                    known_docs.add(resolved.document)
                    # Guard against legacy pre-verification PDFs in paper_index whose title disagrees
                    if entry.get("title") and resolved.title:
                        if not verify_candidate_match(entry["title"], resolved.title, max_year_diff=4):
                            continue
                    if resolved.document != doc_cand:
                        entry["canonical_document"] = resolved.document
                        if resolved.path and os.path.exists(resolved.path):
                            entry["path"] = resolved.path
                            entry["evidence_tier"] = "full_text"
                else:
                    known_docs.add(doc_cand)

            for doc_name, p_entry in idx._papers.items():
                if doc_name in known_docs:
                    continue
                if not p_entry.path or not os.path.exists(p_entry.path):
                    continue
                synth_key = (
                    f"doi:{p_entry.dois[0]}"
                    if p_entry.dois
                    else (f"arxiv:{p_entry.arxiv_ids[0]}" if p_entry.arxiv_ids else f"corpus:{doc_name}")
                )
                if synth_key not in manifest:
                    manifest[synth_key] = {
                        "key": synth_key,
                        "path": p_entry.path,
                        "canonical_document": doc_name,
                        "evidence_tier": "full_text",
                        "provider": "corpus",
                        "title": p_entry.title,
                        "doi": p_entry.dois[0] if p_entry.dois else None,
                        "arxiv_id": p_entry.arxiv_ids[0] if p_entry.arxiv_ids else None,
                    }
        except Exception as exc:
            logger.debug("Could not enrich manifest from %s: %s", idx_path, exc)

    return manifest


def _relocate_manifest_paths(manifest: dict, pulled_dir: str | None = None) -> dict:
    """downloaded.json records absolute paths. When the data directory moves
    to another mount or machine the PDFs are still there under
    PULLED_PDFS_DIR by basename, and the index still holds their chunks —
    but every existence check on the recorded path fails and the audit
    defers every paragraph as "missing". Resolve by basename before that
    happens; keep the recorded path alongside so the relocation is visible."""
    p_dir = pulled_dir or PULLED_PDFS_DIR
    relocated = 0
    for entry in manifest.values():
        recorded = entry.get("path") if isinstance(entry, dict) else None
        if not recorded or os.path.exists(recorded):
            continue
        local = os.path.join(p_dir, os.path.basename(recorded))
        if os.path.exists(local):
            entry["recorded_path"] = recorded
            entry["path"] = local
            relocated += 1
    if relocated:
        logger.info("Resolved %d manifest path(s) under %s by basename.", relocated, p_dir)
    return manifest


def _title_key(title) -> str:
    """Case, markup and punctuation folded out, so two spellings of one title compare equal."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", re.sub(r"<[^>]+>", " ", (title or "").lower())).split())


def _match_downloaded_paper(ref: Optional[dict], seed_pdf_name: str, downloaded_manifest: dict) -> Optional[dict]:
    """Find downloaded paper entry matching a cited reference.

    None means unresolved, and an unresolved reference is reported as such. A
    wrong match is worse: the claim is then judged, confidently, against a
    different paper."""
    if not ref or not downloaded_manifest:
        return None

    ref_xml_id = ref.get("xml_id")
    ref_doi = normalise_doi(ref.get("doi"))
    ref_arxiv = normalise_arxiv(
        ref.get("arxiv_id")
        or extract_arxiv_from_text(ref.get("doi"))
        or extract_arxiv_from_text(ref.get("url"))
        or extract_arxiv_from_text(ref.get("raw_reference"))
    )
    ref_title = _title_key(ref.get("title"))

    # 1. Match by xml_id and cited_by (including multi-paper cited_refs registry)
    if ref_xml_id and seed_pdf_name:
        cref_tag = f"{seed_pdf_name}:{ref_xml_id}"
        for entry in downloaded_manifest.values():
            if not isinstance(entry, dict):
                continue
            if entry.get("xml_id") == ref_xml_id and entry.get("cited_by") == seed_pdf_name:
                return entry
            cited_refs = entry.get("cited_refs")
            if isinstance(cited_refs, list) and cref_tag in cited_refs:
                return entry

    # 2. Match by DOI
    if ref_doi:
        for entry in downloaded_manifest.values():
            if isinstance(entry, dict) and normalise_doi(entry.get("doi")) == ref_doi:
                return entry

    # 2b. Match by arXiv ID
    if ref_arxiv:
        arxiv_key = f"arxiv:{ref_arxiv}"
        if isinstance(downloaded_manifest.get(arxiv_key), dict):
            return downloaded_manifest[arxiv_key]
        for entry in downloaded_manifest.values():
            if not isinstance(entry, dict):
                continue
            ent_arxiv = normalise_arxiv(
                entry.get("arxiv_id")
                or extract_arxiv_from_text(entry.get("doi"))
                or extract_arxiv_from_text(entry.get("key"))
            )
            if ent_arxiv and ent_arxiv == ref_arxiv:
                return entry

    # 2c. Match by deterministic source_key(ref)
    s_key = source_key(ref)
    if s_key and isinstance(downloaded_manifest.get(s_key), dict):
        # Guard against matching a completely different DOI when ref carried an explicit DOI
        cand_entry = downloaded_manifest[s_key]
        cand_doi = normalise_doi(cand_entry.get("doi"))
        if not (ref_doi and cand_doi and ref_doi != cand_doi):
            return cand_entry

    # 3. Match by title. Exactly equal wins. Otherwise only near-equal: one title
    # inside the other with the shorter at least 80% of the longer (a subtitle, an
    # erratum suffix). The manifest holds entries titled "rev. mod. phys",
    # "science", "deep learning", and a bare substring test matched those to any
    # reference containing the words. Two different papers that both qualify are
    # not guessed between.
    from research_assistant.shared.academic_apis import title_variants

    candidate_ref_titles = []
    for tv in title_variants(ref.get("title")):
        tk = _title_key(tv)
        if len(tk) > 10 and tk not in candidate_ref_titles:
            candidate_ref_titles.append(tk)
    if len(ref_title) > 10 and ref_title not in candidate_ref_titles:
        candidate_ref_titles.insert(0, ref_title)

    for r_title in candidate_ref_titles:
        exact, near = [], []
        for entry in downloaded_manifest.values():
            if not isinstance(entry, dict):
                continue
            entry_title = _title_key(entry.get("title"))
            if not entry_title:
                continue
            if entry_title == r_title:
                exact.append((entry_title, entry))
                continue
            shorter, longer = sorted((entry_title, r_title), key=len)
            if len(shorter) > 10 and shorter in longer and len(shorter) >= 0.8 * len(longer):
                near.append((entry_title, entry))
        for group in (exact, near):
            if group:
                if len({t for t, _ in group}) == 1:      # one title, however many copies
                    return group[0][1]
                logger.warning("Reference %r matches %d different downloaded titles; leaving it unresolved",
                               r_title[:60], len({t for t, _ in group}))
                return None

    return None


def enrich_item_bibliometrics(
    item: dict,
    dl_entry: Optional[dict] = None,
    hit_meta: Optional[dict] = None,
) -> dict:
    """Merge bibliometric and retraction signals from downloaded.json / chunk metadata into *item*.

    Copies only bibliographic fields (never chunk-level storage fields such as
    ``type='text_chunk'``) into ``item['metadata']`` and backfills missing
    ``doi`` / ``arxiv_id`` / ``venue`` onto ``item['ref']`` so ``assess_source()``
    sees verified retractions, citation counts, DOIs, and venues.
    """
    meta = dict(item.get("metadata") or {})

    if isinstance(dl_entry, dict):
        if dl_entry.get("is_retracted") is not None or dl_entry.get("retracted") is not None:
            meta["is_retracted"] = bool(
                meta.get("is_retracted")
                or dl_entry.get("is_retracted")
                or dl_entry.get("retracted")
            )
        if dl_entry.get("citation_count") is not None and meta.get("citation_count") is None:
            meta["citation_count"] = dl_entry["citation_count"]
        cand_venue = dl_entry.get("venue") or dl_entry.get("journal") or dl_entry.get("container")
        if cand_venue and not meta.get("venue"):
            meta["venue"] = cand_venue
        raw_doi = dl_entry.get("doi")
        cand_doi = normalise_doi(raw_doi) or (str(raw_doi).strip() if raw_doi else None)
        if cand_doi and not meta.get("doi"):
            meta["doi"] = cand_doi
        cand_arxiv = (
            dl_entry.get("arxiv_id")
            or extract_arxiv_from_text(dl_entry.get("doi"))
            or extract_arxiv_from_text(dl_entry.get("key"))
        )
        if cand_arxiv and not meta.get("arxiv_id"):
            meta["arxiv_id"] = cand_arxiv
        if dl_entry.get("publication_type") and not meta.get("publication_type"):
            meta["publication_type"] = dl_entry["publication_type"]
        if dl_entry.get("provider") and not meta.get("provider"):
            meta["provider"] = dl_entry["provider"]
        if dl_entry.get("title") and not meta.get("title"):
            meta["title"] = dl_entry["title"]
        if dl_entry.get("is_monograph") and "is_monograph" not in meta:
            meta["is_monograph"] = bool(dl_entry["is_monograph"])

    if isinstance(hit_meta, dict):
        if hit_meta.get("is_retracted") is not None or hit_meta.get("retracted") is not None:
            meta["is_retracted"] = bool(
                meta.get("is_retracted")
                or hit_meta.get("is_retracted")
                or hit_meta.get("retracted")
            )
        if hit_meta.get("citation_count") is not None and meta.get("citation_count") is None:
            meta["citation_count"] = hit_meta["citation_count"]
        cand_venue = hit_meta.get("venue") or hit_meta.get("journal") or hit_meta.get("container")
        if cand_venue and not meta.get("venue"):
            meta["venue"] = cand_venue
        raw_doi = hit_meta.get("doi")
        cand_doi = normalise_doi(raw_doi) or (str(raw_doi).strip() if raw_doi else None)
        if cand_doi and not meta.get("doi"):
            meta["doi"] = cand_doi
        cand_arxiv = hit_meta.get("arxiv_id") or extract_arxiv_from_text(hit_meta.get("doi"))
        if cand_arxiv and not meta.get("arxiv_id"):
            meta["arxiv_id"] = cand_arxiv
        if hit_meta.get("publication_type") and not meta.get("publication_type"):
            meta["publication_type"] = hit_meta["publication_type"]
        if hit_meta.get("title") and not meta.get("title"):
            meta["title"] = hit_meta["title"]

    if meta:
        item["metadata"] = meta

    ref = item.get("ref")
    if isinstance(ref, dict):
        if meta.get("doi") and not ref.get("doi"):
            ref["doi"] = meta["doi"]
        if meta.get("arxiv_id") and not ref.get("arxiv_id"):
            ref["arxiv_id"] = meta["arxiv_id"]
        if meta.get("venue") and not ref.get("venue"):
            ref["venue"] = meta["venue"]

    return item


def match_manifest_for_verifier(entry: dict, downloaded_manifest: dict) -> Optional[dict]:
    """Match an Agent 8 verification entry against ``downloaded_manifest`` by
    canonical document, PDF filename, DOI, arXiv ID, or citation_source title."""
    if not entry or not downloaded_manifest:
        return None

    doc = entry.get("document")
    meta = entry.get("metadata") or {}
    ref = entry.get("ref") or {}
    cite_src = entry.get("citation_source") or ""

    if doc:
        doc_base = os.path.basename(str(doc))
        for m_entry in downloaded_manifest.values():
            if not isinstance(m_entry, dict):
                continue
            if m_entry.get("canonical_document") == doc or m_entry.get("key") == doc:
                return m_entry
            m_path = m_entry.get("path")
            if m_path and os.path.basename(str(m_path)) in (str(doc), doc_base):
                return m_entry

    synth_ref = {
        "doi": meta.get("doi") or ref.get("doi"),
        "arxiv_id": meta.get("arxiv_id") or ref.get("arxiv_id") or extract_arxiv_from_text(doc) or extract_arxiv_from_text(cite_src),
        "title": meta.get("title") or ref.get("title") or cite_src,
        "raw_reference": ref.get("raw_reference") or cite_src,
    }
    return _match_downloaded_paper(synth_ref, "", downloaded_manifest)


def attach_cited_summaries(claims: list[dict]) -> None:
    """Read the ingest-time summary of each cited paper into claim['cited_summary'].

    One store read across the whole batch. If the store is down or has no
    summaries yet, nothing is set; the audit continues without them.
    """
    from research_assistant.shared.retrieve import document_summaries

    docs = {c["document"] for c in claims if c.get("document")}
    if not docs:
        return
    try:
        sums = document_summaries(docs)
    except Exception as exc:
        logger.warning("Could not read cited summaries (%s) — continuing without", exc)
        return
    for c in claims:
        doc = c.get("document")
        if doc and doc in sums:
            c["cited_summary"] = sums[doc]


def contextualize_citation_queries(claims: list[dict], model: str | None = None, enabled: bool | None = None) -> None:
    """Batch-contextualize search queries for claims using paragraph context.

    Groups claims by paragraph and invokes LLM to formulate self-contained, keyword-rich
    queries that resolve pronouns ('this approach', 'they', 'such methods') into explicit
    technical concepts. Updates each claim in-place with 'search_query'.
    """
    if not claims:
        return

    def _fallback_query(c: dict) -> str:
        ref = c.get("ref") or {}
        ref_title = ref.get("title") or ""
        authors = ref.get("authors") or []
        first_author = authors[0] if authors else ""
        year = str(ref.get("year") or "")
        claim = c.get("target_clause") or c.get("claim") or ""
        keywords = [t for t in (first_author, year, ref_title) if t]
        if keywords:
            return f"{' '.join(keywords)}: {claim}"
        return claim

    is_enabled = CITATION_AUDIT_CONTEXTUALIZE_QUERIES if enabled is None else enabled
    if not is_enabled:
        for c in claims:
            if not c.get("search_query"):
                c["search_query"] = _fallback_query(c)
        return

    by_paragraph = {}
    for c in claims:
        p_id = c.get("paragraph_id") or "p_0"
        by_paragraph.setdefault(p_id, []).append(c)

    import concurrent.futures
    from research_assistant.shared.llm import chat

    def _process_paragraph(p_id, p_claims):
        if pipeline_status.is_cancel_requested():
            for c in p_claims:
                if not c.get("search_query"):
                    c["search_query"] = _fallback_query(c)
            return

        unqueried = [c for c in p_claims if not c.get("search_query")]
        if not unqueried:
            return

        first = unqueried[0]
        context_text = first.get("context") or first.get("sentence") or ""
        section = first.get("section_heading") or ""
        artifact_lines = "\n".join(
            f"- {a['label']}: {a['caption']}" for a in (first.get("artifacts") or []) if a.get("caption")
        )

        claim_entries = []
        summarised: dict[str, int] = {}
        for idx, c in enumerate(unqueried):
            ref = c.get("ref") or {}
            ref_str = f"{', '.join((ref.get('authors') or [])[:2])} ({ref.get('year') or 'n.d.'}) - '{ref.get('title') or ''}'"
            entry = f"[{idx}] Cited Reference: {ref_str}\n    Claim Sentence: \"{c.get('claim', '')}\""
            if c.get("target_clause"):
                entry += f"\n    Target Sub-Clause Anchored by This Citation: \"{c['target_clause']}\""
            summary, doc = c.get("cited_summary"), c.get("document")
            if summary:
                if doc in summarised:
                    entry += f"\n    What the cited paper is about: as for [{summarised[doc]}]"
                else:
                    summarised[doc] = idx
                    entry += f"\n    What the cited paper is about: {summary}"
            claim_entries.append(entry)

        prompt = (
            "You are a scientific retrieval assistant. For each citation claim extracted from the "
            "paragraph below, generate a focused, standalone search query to find the supporting "
            "passage in the cited paper.\n\n"
            "Rules:\n"
            "1. Resolve pronouns ('this method', 'they', 'the authors', 'this result') to the specific "
            "technique, theory, or findings described in the paragraph.\n"
            "2. Where the sentence points at a figure or table, say what that figure shows (from its "
            "caption below) instead of its number — the cited paper has its own figure numbers.\n"
            "3. Phrase the query in the cited paper's own vocabulary: its summary, where given, says "
            "what it calls its system and method.\n"
            "4. Include the cited author's name, publication year, and essential domain keywords.\n"
            "5. Keep each query concise (10-25 words), focused on concrete technical search terms.\n"
            "6. Return ONLY a valid JSON array of strings in the exact same order as the inputs, e.g.:\n"
            '["query for 0", "query for 1"]\n\n'
            + (f"Section of the citing paper: {section}\n\n" if section else "")
            + f"Paragraph Context:\n\"\"\"\n{context_text}\n\"\"\"\n\n"
            + (f"Figures and tables the paragraph refers to:\n{artifact_lines}\n\n" if artifact_lines else "")
            + "Citations to Contextualize:\n" + "\n".join(claim_entries) + "\n\n"
            "JSON array of queries:"
        )

        try:
            res = chat([{"role": "user", "content": prompt}], model=model, temperature=0.0)
            raw = res.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            if isinstance(parsed, list) and len(parsed) == len(unqueried):
                for c, q in zip(unqueried, parsed):
                    if isinstance(q, str) and len(q.strip()) > 5:
                        c["search_query"] = q.strip()
                    else:
                        c["search_query"] = _fallback_query(c)
                return
        except Exception as exc:
            logger.debug("LLM query contextualization failed for paragraph %s: %s", p_id, exc)

        for c in unqueried:
            if not c.get("search_query"):
                c["search_query"] = _fallback_query(c)

    workers = min(4, len(by_paragraph)) if by_paragraph else 1
    if workers <= 1:
        for p_id, p_claims in by_paragraph.items():
            _process_paragraph(p_id, p_claims)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(lambda item: _process_paragraph(item[0], item[1]), by_paragraph.items()))
