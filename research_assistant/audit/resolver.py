"""
resolver.py — Reference resolution, manifest matching, query contextualization, and summary attachment.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from research_assistant.config import (
    CITATION_AUDIT_CONTEXTUALIZE_QUERIES,
    DOWNLOADED_JSON_PATH,
    PULLED_PDFS_DIR,
)
from research_assistant.shared.claim_text import clean_text, is_numeric_cite
from research_assistant.shared.log import get_logger
from research_assistant.shared import pipeline_status
from research_assistant.shared.source_key import normalise_doi

logger = get_logger("audit_resolver")

_NUM_RE = re.compile(r"\d+")
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


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
        year = _YEAR_RE.search(txt or "")
        req_year = int(year.group(1)) if year else None
        for info in bib_by_id.values():
            surnames = []
            for a in info.get("authors", []):
                a_clean = a.strip()
                if not a_clean:
                    continue
                if "," in a_clean:
                    surnames.append(a_clean.split(",")[0].strip())
                else:
                    surnames.append(a_clean.split()[-1].strip())
            raw = info.get("raw_reference") or ""
            title = info.get("title") or ""
            surname_tokens = []
            for s in surnames:
                surname_tokens.append(s)
                for tok in s.split():
                    if len(tok) > 2:
                        surname_tokens.append(tok)
            surname_hit = any(len(s) > 1 and re.search(rf"\b{re.escape(s)}\b", txt, re.IGNORECASE) for s in surname_tokens)
            if not surname_hit:
                # Match Titlecase names or all-caps institutional acronyms (e.g. IUPAC, NASA)
                m_authors = re.findall(r"\b(?:[A-Z]{2,}|[A-Z][a-z]{2,})\b", txt)
                surname_hit = any(
                    re.search(rf"\b{re.escape(auth)}\b", raw, re.IGNORECASE) or
                    re.search(rf"\b{re.escape(auth)}\b", title, re.IGNORECASE)
                    for auth in m_authors
                )
            if surname_hit:
                if req_year is None or info.get("year") == req_year or str(req_year) in raw:
                    return info, "surname", ""
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
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        logger.warning("Could not read %s: %s", path, e)
        return {}
    return _relocate_manifest_paths(manifest, pulled_dir=pulled_dir)


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
    ref_title = _title_key(ref.get("title"))

    # 1. Match by xml_id and cited_by
    if ref_xml_id and seed_pdf_name:
        for entry in downloaded_manifest.values():
            if (
                entry.get("xml_id") == ref_xml_id
                and entry.get("cited_by") == seed_pdf_name
            ):
                return entry

    # 2. Match by DOI
    if ref_doi:
        for entry in downloaded_manifest.values():
            if normalise_doi(entry.get("doi")) == ref_doi:
                return entry

    # 3. Match by title. Exactly equal wins. Otherwise only near-equal: one title
    # inside the other with the shorter at least 80% of the longer (a subtitle, an
    # erratum suffix). The manifest holds entries titled "rev. mod. phys",
    # "science", "deep learning", and a bare substring test matched those to any
    # reference containing the words. Two different papers that both qualify are
    # not guessed between.
    if len(ref_title) > 10:
        exact, near = [], []
        for entry in downloaded_manifest.values():
            entry_title = _title_key(entry.get("title"))
            if not entry_title:
                continue
            if entry_title == ref_title:
                exact.append((entry_title, entry))
                continue
            shorter, longer = sorted((entry_title, ref_title), key=len)
            if len(shorter) > 10 and shorter in longer and len(shorter) >= 0.8 * len(longer):
                near.append((entry_title, entry))
        for group in (exact, near):
            if group:
                if len({t for t, _ in group}) == 1:      # one title, however many copies
                    return group[0][1]
                logger.warning("Reference %r matches %d different downloaded titles; leaving it unresolved",
                               ref_title[:60], len({t for t, _ in group}))
                return None

    return None


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
        claim = c.get("claim") or ""
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
