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

from research_assistant.agents.agent5_batch_citer import split_into_sentences
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
from research_assistant.agents.agent8_verifier import (
    _judge_once,
    _judgement_model,
    assemble_evidence,
    needs_escalation,
)
from research_assistant.judgement.policy import evaluate_reliability
from research_assistant.judgement.source_assessor import assess_source
from research_assistant.config import (
    DOWNLOADED_JSON_PATH,
    JUDGEMENT_ESCALATE_TOP_K,
    JUDGEMENT_EVIDENCE_MAX_CHARS,
    JUDGEMENT_NEIGHBOUR_WINDOW,
    JUDGEMENT_TOP_K,
    PULLED_PDFS_DIR,
    RAW_DIR,
    CITATION_AUDIT_CONTEXTUALIZE_QUERIES,
)
from research_assistant.shared import pipeline_status
from research_assistant.shared.atomic import atomic_write_json
from research_assistant.shared.log import get_logger
from research_assistant.shared.search import expand_neighbours, hybrid_search
from research_assistant.shared.source_key import normalise_doi
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


def _resolve_ref(target, txt, bib_by_id, bib_by_index, number_map) -> tuple[Optional[dict], str, str]:
    """(reference, how, note). Two witnesses for a numeric marker: the entry
    GROBID linked it to, and the entry at that list position. On one paper
    the links were off by one and position was right; elsewhere a footnote
    in the list shifts every position after it. So: agreement resolves;
    disagreement between two real entries is "ambiguous" and never judged;
    a footnote-like entry at the position yields to the link. A surname
    match needs the whole word and, when both sides have one, the year."""
    tgt = bib_by_id.get(target)
    if not is_numeric_cite(txt):
        if tgt is not None:
            return tgt, "target", ""
        year = _YEAR_RE.search(txt or "")
        for info in bib_by_id.values():
            surnames = [a.split()[-1] for a in info.get("authors", []) if a.split()]
            if any(len(s) > 2 and re.search(rf"\b{re.escape(s)}\b", txt) for s in surnames):
                if year is None or info.get("year") is None or int(year.group(1)) == info["year"]:
                    return info, "surname", ""
        return None, "unresolved", ""

    nums = _NUM_RE.findall(txt)
    k = int(nums[0]) if nums else None
    at_position = bib_by_index.get(k) if k is not None else None
    pos = at_position if _looks_like_reference(at_position) else None
    linked = number_map.get(k) if k is not None else None

    def _name(entry):
        return (entry.get("title") or entry.get("raw_reference") or entry.get("xml_id") or "?")[:60]

    if tgt is not None and pos is not None:
        if tgt is pos:
            return tgt, "target", ""
        return None, "ambiguous", (
            f"GROBID links {txt} to '{_name(tgt)}' but list position {k} is '{_name(pos)}'; not judged against a guess."
        )
    if tgt is not None:
        return tgt, "target", ""
    if pos is not None and linked is not None and linked is not pos:
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


def extract_seed_citation_claims(tei_source: str | BeautifulSoup) -> list[dict]:
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
    if isinstance(tei_source, BeautifulSoup):
        soup = tei_source
    else:
        if not os.path.exists(tei_source):
            logger.warning("TEI file not found: %s", tei_source)
            return []
        with open(tei_source, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "xml")

    # 1. Parse bibliography from <listBibl>
    list_bibl = soup.find("listBibl")
    bib_by_id = {}
    bib_by_index = {}

    if list_bibl:
        for idx, b in enumerate(list_bibl.find_all("biblStruct")):
            xid = b.get("xml:id") or b.get("id") or f"b{idx}"
            analytic = b.find("analytic")
            monogr = b.find("monogr")
            t_node = (
                (analytic.find("title", type="main") or analytic.find("title"))
                if analytic
                else None
            ) or ((monogr.find("title", level="m") or monogr.find("title")) if monogr else None)
            title = _clean(t_node) if t_node else None

            authors = []
            scope = analytic or b
            for a in scope.find_all("author"):
                pers = a.find("persName")
                if pers:
                    pname = _person_name(pers)
                    if pname:
                        authors.append(pname)

            date_node = b.find("date", type="published") or b.find("date")
            year = None
            if date_node:
                when = date_node.get("when") or _clean(date_node)
                ym = re.search(r"(1[89]\d{2}|20\d{2})", when)
                if ym:
                    year = int(ym.group(1))

            doi_node = b.find("idno", type="DOI")
            doi = normalise_doi(doi_node.text) if doi_node and doi_node.text else None
            raw_node = b.find("note", type="raw_reference")
            raw_reference = _clean(raw_node) if raw_node else None

            journal = monogr.find("title", level="j") if monogr else None
            book = monogr.find("title", level="m") if monogr else None
            venue = _clean(journal) if journal else (_clean(book) if book else None)
            is_monograph = bool(book) and not (analytic and analytic.find("title"))

            ref_info = {
                "xml_id": xid,
                "index": idx + 1,
                "title": title,
                "authors": authors,
                "year": year,
                "doi": doi,
                "venue": venue,
                "is_monograph": is_monograph,
                "raw_reference": raw_reference,
            }
            bib_by_id[xid] = ref_info
            bib_by_index[idx + 1] = ref_info

    # 2. Parse body paragraphs
    body = soup.find("body")
    if not body:
        return []

    number_map = _build_number_map(body, bib_by_id)
    registry = artifact_registry(soup)

    claims = []
    seen_pairs = set()

    for p_idx, p in enumerate(body.find_all("p")):
        refs = p.find_all("ref", type="bibr")
        if not refs:
            continue

        p_id = p.get("xml:id") or p.get("id") or f"p_{p_idx}"
        section = _paragraph_section(p)
        heading = section_breadcrumb(p)
        p_artifacts = paragraph_artifacts(p, registry)

        # Citations become period-free tokens; what the old placeholder
        # embedded lives in this table instead.
        cites = {}
        for n, ref in enumerate(refs):
            target = (ref.get("target") or "").lstrip("#")
            txt = clean_text(ref)
            ref_info, resolution, note = _resolve_ref(
                target, txt, bib_by_id, bib_by_index, number_map
            )
            cites[n] = {"target": target, "txt": txt, "ref": ref_info, "resolution": resolution, "note": note}
            ref.replace_with(f" {cite_token(n)} ")

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
                role = classify_citation_role(sent, cite_token(n), ref_info)

                ref_key = ref_info.get("xml_id") or cite["target"] or cite["txt"]
                dedup_key = (claim_text, ref_key)
                if dedup_key in seen_pairs:
                    continue
                seen_pairs.add(dedup_key)

                if resolved and role == "evidential":
                    p_refs_dict.setdefault(ref_key, ref_info)

                paragraph_claims.append({
                    "sentence": display[s_idx],
                    "claim": claim_text,
                    "cite_text": cite["txt"],
                    "target": cite["target"],
                    "ref": ref_info,
                    "resolved": resolved,
                    "resolution": cite["resolution"],
                    "resolution_note": cite["note"],
                    "role": role,
                    "claim_quality": quality,
                    "context": context,
                    "section": section,
                    "section_heading": heading,
                    "artifacts": p_artifacts,
                    "cite_count": len(tokens),
                    "sentence_index": s_idx,
                    "paragraph_id": p_id,
                    "paragraph_index": p_idx,
                    "sentence_refs": sent_refs,
                })

        p_unique_refs = list(p_refs_dict.values())
        for c in paragraph_claims:
            c["paragraph_refs"] = p_unique_refs
        claims.extend(paragraph_claims)

    return claims


def _load_downloaded_manifest() -> dict:
    if not os.path.exists(DOWNLOADED_JSON_PATH):
        return {}
    try:
        with open(DOWNLOADED_JSON_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        logger.warning("Could not read %s: %s", DOWNLOADED_JSON_PATH, e)
        return {}
    return _relocate_manifest_paths(manifest)


def _relocate_manifest_paths(manifest: dict) -> dict:
    """downloaded.json records absolute paths. When the data directory moves
    to another mount or machine the PDFs are still there under
    PULLED_PDFS_DIR by basename, and the index still holds their chunks —
    but every existence check on the recorded path fails and the audit
    defers every paragraph as "missing". Resolve by basename before that
    happens; keep the recorded path alongside so the relocation is visible."""
    relocated = 0
    for entry in manifest.values():
        recorded = entry.get("path") if isinstance(entry, dict) else None
        if not recorded or os.path.exists(recorded):
            continue
        local = os.path.join(PULLED_PDFS_DIR, os.path.basename(recorded))
        if os.path.exists(local):
            entry["recorded_path"] = recorded
            entry["path"] = local
            relocated += 1
    if relocated:
        logger.info("Resolved %d manifest path(s) under %s by basename.", relocated, PULLED_PDFS_DIR)
    return manifest


def _match_downloaded_paper(ref: Optional[dict], seed_pdf_name: str, downloaded_manifest: dict) -> Optional[dict]:
    """Find downloaded paper entry matching a cited reference."""
    if not ref or not downloaded_manifest:
        return None

    ref_xml_id = ref.get("xml_id")
    ref_doi = normalise_doi(ref.get("doi"))
    ref_title = (ref.get("title") or "").lower().strip()

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

    # 3. Match by exact or high-overlap title
    if len(ref_title) > 10:
        for entry in downloaded_manifest.values():
            entry_title = (entry.get("title") or "").lower().strip()
            if entry_title and (ref_title in entry_title or entry_title in ref_title):
                return entry

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


def contextualize_citation_queries(claims: list[dict], model: str | None = None) -> None:
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

    if not CITATION_AUDIT_CONTEXTUALIZE_QUERIES:
        for c in claims:
            if not c.get("search_query"):
                c["search_query"] = _fallback_query(c)
        return

    by_paragraph = {}
    for c in claims:
        p_id = c.get("paragraph_id") or "p_0"
        by_paragraph.setdefault(p_id, []).append(c)

    from research_assistant.shared.llm import chat

    for p_id, p_claims in by_paragraph.items():
        unqueried = [c for c in p_claims if not c.get("search_query")]
        if not unqueried:
            continue

        first = unqueried[0]
        context_text = first.get("context") or first.get("sentence") or ""
        section = first.get("section_heading") or ""
        artifact_lines = "\n".join(
            f"- {a['label']}: {a['caption']}" for a in (first.get("artifacts") or []) if a.get("caption")
        )

        # One summary per cited paper per paragraph: the second claim on the
        # same paper points back at the first instead of repeating 180 words.
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
                    if isinstance(q, str) and q.strip():
                        c["search_query"] = q.strip()
                    else:
                        c["search_query"] = _fallback_query(c)
                continue
        except Exception as exc:
            logger.debug("LLM query contextualization failed for paragraph %s: %s", p_id, exc)

        for c in unqueried:
            if not c.get("search_query"):
                c["search_query"] = _fallback_query(c)


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
        REQUIRED_FIELDS,
        JudgementParseError,
    )

    retrieve_k = max(top_k, escalate_k if escalate_k > top_k else 0)
    search_query = item.get("search_query") or item["claim"]
    item["search_query"] = search_query

    try:
        hits = hybrid_search(
            search_query,
            collection,
            bm25,
            texts,
            metadatas,
            top_k=retrieve_k,
            doc_filter={item["document"]},
            exclude_types={"figure_description"},
        )
    except Exception as exc:
        logger.warning("Retrieval failed for %s: %s", item.get("document"), exc)
        item["outcome"] = "retrieval_failed"
        item["judgement"] = None
        item["reason"] = f"Retrieval failed: {exc}"
        return item

    if not hits:
        item["outcome"] = "no_evidence"
        item["judgement"] = None
        item["reason"] = "No relevant passages found in the cited document."
        return item

    expand_neighbours(hits, texts, metadatas, window=JUDGEMENT_NEIGHBOUR_WINDOW)
    item["evidence"] = assemble_evidence(hits[:top_k], JUDGEMENT_EVIDENCE_MAX_CHARS)

    def _provenance(k):
        used = hits[:k]
        item["evidence_hits"] = len(used)
        item["evidence_sections"] = sorted({(h.get("metadata") or {}).get("section") for h in used} - {None})
        item["evidence_pages"] = sorted({(h.get("metadata") or {}).get("page") for h in used} - {None})

    _provenance(top_k)
    item["escalated"] = False

    # The breadcrumb when the TEI gave one, the kind otherwise; the captions
    # of the figures the paragraph points at; never the cited summary.
    from research_assistant.judgement.judge import compose_context

    section = item.get("section_heading") or (
        str(item["section"]).replace("_", " ").title() if item.get("section") else None
    )
    claim_context = compose_context(item.get("context"), section=section, artifacts=item.get("artifacts"))

    try:
        verdict = _judge_once(item["claim"], item["evidence"], context=claim_context)
        if escalate_k and len(hits) > top_k and needs_escalation(verdict):
            item["first_judgement"] = verdict.get("judgement")
            item["evidence"] = assemble_evidence(
                hits[:escalate_k], JUDGEMENT_EVIDENCE_MAX_CHARS
            )
            _provenance(escalate_k)
            item["escalated"] = True
            verdict = _judge_once(item["claim"], item["evidence"], context=claim_context)

        item["outcome"] = "judged"
        for k in REQUIRED_FIELDS:
            if k in verdict:
                item[k] = verdict[k]
        for k in DERIVED_FIELDS:
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
            from research_assistant.agents.agent0_discoverer import _inspect_pdf
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
_RELIABILITY_KEYS = (
    ("high", "HIGH"), ("moderate", "MODERATE"), ("low", "LOW"),
    ("contradicted", "CONTRADICTED"), ("unsupported", "UNSUPPORTED"), ("unresolved", "UNRESOLVED"),
)


def compute_totals(results: list[dict]) -> dict:
    """The one place counts come from. Verdict counts are over judged items
    only; everything else is in not_assessed, keyed by outcome, so
    judged + sum(not_assessed) == total always holds."""
    judged = [r for r in results if r.get("outcome") == "judged"]
    totals = {
        "total": len(results),
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
    return totals


SECTION_RANK = {"results": 0, "discussion": 0, "conclusion": 0, "methods": 1}
MAX_PAIRS_PER_SENTENCE = 3
MAX_CONSECUTIVE_BACKEND_FAILURES = 3
BACKEND_FAILURE_OUTCOMES = ("retrieval_failed", "call_failed")


def prioritise_claims(claims: list[dict]) -> tuple[list[dict], list[dict]]:
    """The order the budget is spent in, and the pairs it never reaches.

    Results and discussion before methods before introduction; sentences
    with few citations before "[13]–[27] have been proposed"; then, because
    most physics headings are topical ("DFT-based tight-binding Hamiltonian")
    and map to no rank, later paragraphs before earlier ones — the opening of
    a paper is background, its later pages are where results are compared
    with prior work. At most MAX_PAIRS_PER_SENTENCE pairs per sentence — the
    rest are cluster_skipped, which a re-run with a larger budget does not
    revisit.
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
    per_sentence: dict[tuple, int] = {}
    to_judge, skipped = [], []
    for c in ordered:
        key = (c.get("paragraph_index", 0), c.get("sentence_index", 0))
        per_sentence[key] = per_sentence.get(key, 0) + 1
        if per_sentence[key] > MAX_PAIRS_PER_SENTENCE:
            c["outcome"] = "cluster_skipped"
            c["judgement"] = None
            c["reason"] = f"Sentence cites {c.get('cite_count')} papers; only {MAX_PAIRS_PER_SENTENCE} judged."
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

    # 1. Identify claims whose reference PDFs became available since previous audit
    unresolved_claims = [
        r for r in results
        if r.get("outcome") in (
            "not_downloaded",
            "deferred_paywalled",
            "cap_exceeded",
            "no_evidence",
            "retrieval_failed",
            "call_failed",
            "not_attempted",
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
    claims_to_rejudge = []
    for item in unresolved_claims:
        if item in backend_failed:
            continue
        ref_info = item.get("ref")
        matched = _match_downloaded_paper(ref_info, seed_pdf_name, downloaded_manifest)
        if matched:
            item["downloaded"] = True
            doc_file = (
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
            claims_to_rejudge.append(item)

    to_judge = backend_failed + claims_to_rejudge[:max_new_claims]
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
        "newly_available_count": len(claims_to_rejudge),
        "newly_judged_count": newly_judged_count,
        "from_cache": True,
    }
    return report, summary


def audit_seed_citations(
    seed_path: str,
    search_resources=None,
    max_claims: int = 20,
    top_k: int = JUDGEMENT_TOP_K,
    force: bool = False,
    skip_if_cached: bool = True,
) -> dict:
    """Audit the in-text citation claims of an uploaded seed paper against the local corpus.

    Args:
        seed_path: Path to the seed PDF.
        search_resources: Tuple of (collection, bm25, texts, metadatas) or None to load.
        max_claims: Maximum number of claims citing downloaded papers to judge with the LLM.
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
            report, summary = cross_check_seed_audit(
                seed_path, cached, search_resources=search_resources
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

    claims = extract_seed_citation_claims(tei_path)
    if not claims:
        logger.info("No in-text citation claims found in %s.", tei_path)
        return {
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
            if not (dl_entry and dl_entry.get("path") and os.path.exists(dl_entry["path"])):
                missing_p_refs.append(r)

        missing_count = len(missing_p_refs)
        paywall_ratio = (missing_count / total_p_refs) if total_p_refs > 0 else 0.0
        for c in p_claims:
            ref = c.get("ref")
            dl_entry = _match_downloaded_paper(ref, seed_pdf_name, downloaded_manifest)
            is_dl = bool(dl_entry and dl_entry.get("path") and os.path.exists(dl_entry["path"]))
            c["paragraph_missing_refs"] = missing_p_refs
            c["paywall_ratio"] = paywall_ratio

            # Identify if other references cited in this exact same sentence are missing from the corpus
            curr_key = (ref.get("xml_id") or normalise_doi(ref.get("doi")) or ref.get("title")) if ref else None
            compound_missing = []
            for s_ref in c.get("sentence_refs", []):
                s_key = s_ref.get("xml_id") or normalise_doi(s_ref.get("doi")) or s_ref.get("title")
                if s_key and s_key != curr_key:
                    s_dl = _match_downloaded_paper(s_ref, seed_pdf_name, downloaded_manifest)
                    if not (s_dl and s_dl.get("path") and os.path.exists(s_dl["path"])):
                        compound_missing.append(s_ref)
            if compound_missing:
                c["compound_missing_refs"] = compound_missing

            if is_dl:
                c["downloaded"] = True
                c["document"] = os.path.basename(dl_entry["path"])
                c["citation_source"] = (
                    dl_entry.get("title")
                    or dl_entry.get("raw_reference")
                    or dl_entry.get("key")
                )
                downloaded_claims.append(c)
            else:
                c["downloaded"] = False
                c["outcome"] = "not_downloaded"
                c["judgement"] = "Not downloaded"
                c["reason"] = "Reference PDF was not available or could not be downloaded."
                undownloaded_claims.append(c)

    logger.info(
        "Found %d in-text citation claims (%d cite downloaded references, %d not in corpus).",
        len(claims),
        len(downloaded_claims),
        len(undownloaded_claims),
    )

    ordered, cluster_skipped = prioritise_claims(downloaded_claims)
    claims_to_judge = ordered[:max_claims]
    overflow = ordered[max_claims:]
    aborted = None

    if claims_to_judge:
        attach_cited_summaries(claims_to_judge)
        pipeline_status.update_progress(detail="Contextualizing search queries from paragraph context")
        contextualize_citation_queries(claims_to_judge)

        if search_resources is None:
            from research_assistant.shared.db import load_search_resources

            search_resources = load_search_resources()
        collection, bm25, texts, metadatas = search_resources

        from research_assistant.judgement.judge import DERIVED_FIELDS, REQUIRED_FIELDS, JudgementParseError

        escalate_k = JUDGEMENT_ESCALATE_TOP_K if JUDGEMENT_ESCALATE_TOP_K > top_k else 0
        consecutive_failures = 0

        for i, item in enumerate(claims_to_judge, 1):
            if aborted:
                item["outcome"] = "not_attempted"
                item["judgement"] = None
                item["reason"] = f"Audit stopped after {aborted['after_attempted']} attempts: {aborted['reason']}"
                continue
            ref_info = item.get("ref") or {}
            ref_lbl = f"[{ref_info.get('index') or '?'}] {ref_info.get('title') or item.get('cite_text', '')}"
            pipeline_status.update_progress(
                detail=f"Judging citation ({i}/{len(claims_to_judge)}): {ref_lbl[:40]}"
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
        )
        r["reliability"] = rel_eval["rating"]
        r["reliability_badge"] = rel_eval["badge"]
        r["reliability_label"] = rel_eval["rating_label"]
        r["reliability_explanation"] = rel_eval["explanation"]

    totals = compute_totals(all_results)

    report = {
        "seed_path": seed_path,
        "seed_name": seed_pdf_name,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "model": _judgement_model(),
        "totals": totals,
        "results": all_results,
        "aborted": aborted,
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


def explain_rubric_verdict(item: dict) -> str:
    """Explains why a citation received its verdict based on the 3-slot rubric."""
    judgement = item.get("judgement") or ""
    outcome = item.get("outcome", "")
    slots = item.get("slots") or {}

    if outcome == "deferred_paywalled" or "Deferred" in judgement:
        reason = item.get("reason")
        if reason:
            return f"{reason} Upload the missing reference PDF(s) to verify this claim."
        return (
            "Evaluation deferred: More than 50% of the references cited in this paragraph are missing from the corpus. "
            "Upload the missing reference PDF(s) to enable empirical verification."
        )
    if outcome == "not_downloaded":
        return (
            "This reference paper was paywalled, a book, or otherwise unavailable for open-access download. "
            "Its full text is not in the corpus, so claims citing it could not be empirically verified."
        )
    if outcome == "no_evidence":
        return "The cited reference is in the corpus, but semantic and keyword search found no passages discussing this specific assertion."
    if outcome in ("retrieval_failed", "call_failed", "parse_failed"):
        return f"Evaluation could not complete due to a processing issue: {item.get('reason', outcome)}."
    if outcome == "cap_exceeded":
        return "Not assessed: the per-paper claim budget was reached before this citation. Re-run with a higher budget to judge it."
    if outcome == "not_attempted":
        return "Not assessed: the audit stopped early because the model backend was unreachable. Re-run once the backend is up."
    if outcome == "cluster_skipped":
        return "Not assessed: this sentence cites many papers; only the first few were judged."
    if outcome == "not_a_claim":
        role = item.get("role", "non-evidential")
        return f"Not assessed: this is a {role} citation, not a verifiable claim about the cited paper's findings."
    if outcome == "malformed_claim":
        return f"Not assessed: the extracted sentence is a fragment ({', '.join(item.get('claim_quality') or [])}) and could not be judged."
    if outcome == "unresolved_ref":
        note = item.get("resolution_note") or item.get("reason")
        return f"Not assessed: {note}" if note else "Not assessed: the citation marker could not be matched to a bibliography entry."
    if outcome != "judged":
        return item.get("reason") or f"Not assessed ({outcome})."

    def _val(name):
        s = slots.get(name)
        if isinstance(s, dict):
            return s.get("verdict", "")
        return str(s or "")

    f_v = _val("finding")
    s_v = _val("scope")
    st_v = _val("strength")

    if judgement == "Supports":
        return (
            "Fully supported: The cited evidence explicitly verifies the asserted finding, matches the tested scope/conditions, and corroborates the claim's strength."
        )

    if judgement == "Contradicts":
        return (
            "Contradiction flagged: The cited paper investigated the same scope and reported an empirical finding directly opposite or inconsistent with the claim."
        )

    if judgement == "Does not support":
        if s_v in ("Does not support", "Insufficient"):
            return (
                "Scope mismatch: The cited paper investigated a different material system, experimental condition, or domain than asserted in the claim."
            )
        if f_v in ("Does not support", "Insufficient"):
            return (
                "Finding mismatch: While the cited paper is in a related area, it does not report the specific effect or relationship asserted."
            )
        return "Does not support: The evidence does not validate the asserted finding or tested conditions."

    if judgement == "Partially supports":
        hedges = []
        if s_v in ("Partially supports", "Insufficient"):
            hedges.append("the paper tested only a narrower subset of the asserted conditions or scope")
        if st_v in ("Partially supports", "Insufficient"):
            hedges.append("the paper proposed a hedged or specific mechanism whereas the claim asserted a definitive or generalized effect")
        detail = " and ".join(hedges) if hedges else "certain slots were qualified"
        return f"Partial support: The core finding is substantiated in the cited text, but {detail}."

    if judgement == "Unclear / insufficient evidence":
        return (
            "Insufficient evidence: The retrieved passages from the cited paper are too fragmentary or ambiguous to determine whether the claim is supported."
        )

    return item.get("reason") or "Evaluation complete."


_VERDICT_BADGES = {
    "Supports": "🟢 Supports",
    "Partially supports": "🟡 Partially Supports",
    "Contradicts": "🔴 Contradicts",
    "Does not support": "🟠 Does Not Support",
    "Unclear / insufficient evidence": "⚪ Unclear / Insufficient Evidence",
}

_NOT_ASSESSED_BADGES = {
    "deferred_paywalled": "⏳ Deferred (Pending Evidence)",
    "not_downloaded": "🔒 Paywalled / Not In Corpus",
    "cap_exceeded": "⏸ Not Assessed (budget)",
    "cluster_skipped": "⏸ Not Assessed (cluster)",
    "not_attempted": "⏸ Not Assessed (backend down)",
    "retrieval_failed": "⏸ Not Assessed (retrieval failed)",
    "call_failed": "⏸ Not Assessed (model call failed)",
    "parse_failed": "⏸ Not Assessed (unparseable reply)",
    "no_evidence": "⏸ Not Assessed (no passages found)",
    "malformed_claim": "⏸ Not Assessed (sentence fragment)",
    "unresolved_ref": "⏸ Not Assessed (unmatched reference)",
}


def _audit_item_badge(item: dict) -> str:
    """The label a citation row wears. Outcome first: a verdict label is only
    ever shown for an item the judge produced a verdict for."""
    outcome = item.get("outcome")
    if outcome == "judged":
        return _VERDICT_BADGES.get(item.get("judgement") or "", "⚪ Unclear / Insufficient Evidence")
    if outcome == "not_a_claim":
        return f"🔧 Not a Claim ({item.get('role') or 'non-evidential'})"
    return _NOT_ASSESSED_BADGES.get(outcome or "", f"⏸ Not Assessed ({outcome or 'unknown'})")


def generate_seed_audit_markdown(report: dict, seed_title: str | None = None) -> str:
    """Generates a complete, comprehensive Markdown audit report for the given report dict."""
    title = (
        seed_title
        or report.get("seed_name")
        or os.path.basename(report.get("seed_path", "Seed Paper"))
    )
    totals = report.get("totals", {})
    results = report.get("results", [])
    generated = report.get("generated", datetime.now().isoformat(timespec="seconds"))
    model = report.get("model", "judgement-engine")

    total_citations = totals.get("total", len(results))
    pct = lambda val: f"{round((val / total_citations) * 100, 1)}%" if total_citations else "0%"

    lines = [
        f"# 🔍 In-Text Citation Audit Report",
        "",
        f"**Document:** `{title}`  ",
        f"**Generated:** `{generated}`  ",
        f"**Evaluation Model:** `{model}`  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "| Category | Count | Percentage | Description |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Total In-Text Citations** | **{total_citations}** | 100% | Unique in-text reference instances |",
        f"| **Downloaded / In Corpus** | **{totals.get('downloaded', 0)}** | {pct(totals.get('downloaded', 0))} | Reference PDFs available in library |",
        f"| 🟢 **Supports** | **{totals.get('Supports', 0)}** | {pct(totals.get('Supports', 0))} | Evidence directly validates claim |",
        f"| 🟡 **Partially Supports** | **{totals.get('Partially supports', 0)}** | {pct(totals.get('Partially supports', 0))} | Core finding matches; scope or strength hedged |",
        f"| 🔴 **Contradicts** | **{totals.get('Contradicts', 0)}** | {pct(totals.get('Contradicts', 0))} | Evidence directly opposes claim |",
        f"| 🟠 **Does Not Support** | **{totals.get('Does not support', 0)}** | {pct(totals.get('Does not support', 0))} | Scope or finding mismatch |",
        f"| ⚪ **Unclear / Insufficient** | **{totals.get('Unclear / insufficient evidence', 0)}** | {pct(totals.get('Unclear / insufficient evidence', 0))} | Fragile or inconclusive evidence |",
        f"| ⏳ **Deferred (Pending Evidence)** | **{totals.get('deferred_paywalled', 0)}** | {pct(totals.get('deferred_paywalled', 0))} | Paragraph >50% paywalled; evaluation deferred |",
        f"| 🔒 **Paywalled / Unchecked** | **{totals.get('not_downloaded', 0)}** | {pct(totals.get('not_downloaded', 0))} | Non-OA reference; unavailable |",
        "",
        "### Scientific Evidence Reliability Rating",
        "",
        "| Reliability Tier | Count | Percentage | Standard & Policy |",
        "| :--- | :--- | :--- | :--- |",
        f"| 🟢 **High Reliability** | **{totals.get('reliability', {}).get('high', 0)}** | {pct(totals.get('reliability', {}).get('high', 0))} | Peer-reviewed primary research with verified verbatim evidence span |",
        f"| 🟡 **Moderate Reliability** | **{totals.get('reliability', {}).get('moderate', 0)}** | {pct(totals.get('reliability', {}).get('moderate', 0))} | Supported by unreviewed preprints, secondary surveys, or hedged findings |",
        f"| 🟠 **Low / Flagged** | **{totals.get('reliability', {}).get('low', 0)}** | {pct(totals.get('reliability', {}).get('low', 0))} | Unverified spans, retracted/flawed sources, or rubric violations |",
        f"| 🔴 **Contradicted** | **{totals.get('reliability', {}).get('contradicted', 0)}** | {pct(totals.get('reliability', {}).get('contradicted', 0))} | Cited peer-reviewed literature directly refutes the claim |",
        f"| 🟠 **Unsupported** | **{totals.get('reliability', {}).get('unsupported', 0)}** | {pct(totals.get('reliability', {}).get('unsupported', 0))} | The cited paper was read and does not report this |",
        f"| ⏳ **Unresolved / Pending** | **{totals.get('reliability', {}).get('unresolved', 0)}** | {pct(totals.get('reliability', {}).get('unresolved', 0))} | Judged but undecidable, or not assessed (see coverage) |",
        "",
    ]

    coverage = totals.get("coverage", {})
    not_assessed = totals.get("not_assessed", {})
    coverage_lines = [
        "### Assessment coverage",
        "",
        "| | Count |",
        "| :--- | :--- |",
        f"| Citations whose reference is in the corpus | **{coverage.get('downloaded', 0)}** |",
        f"| Attempted (retrieval + model call) | **{coverage.get('attempted', 0)}** |",
        f"| Judged by the model | **{coverage.get('judged', 0)}** |",
    ]
    for outcome, n in sorted(not_assessed.items(), key=lambda kv: -kv[1]):
        coverage_lines.append(f"| Not assessed — {outcome.replace('_', ' ')} | {n} |")
    coverage_lines.append("")
    if report.get("aborted"):
        ab = report["aborted"]
        coverage_lines += [
            f"> ⚠️ **Audit stopped early** after {ab.get('after_attempted')} attempt(s): {ab.get('reason')}  ",
            "> Every citation after that point is *not attempted*. Re-run once the backend is reachable.",
            "",
        ]
    lines.extend(coverage_lines)

    lines.extend([
        "---",
        "",
        "## 2. Verification Rubric & Methodology",
        "",
        "Every in-text citation was audited by extracting the exact sentence in the paper, isolating the referenced source, and retrieving the top matching passages from that source. The evidence is evaluated against a formal **three-slot decomposition**:",
        "",
        "- **Finding Slot**: Does the cited evidence assert the specific phenomenon, relationship, or effect claimed?",
        "- **Scope Slot**: Did the cited paper test the same system, material, conditions, or environment?",
        "- **Strength Slot**: Does the evidence establish causation or generality, or only an isolated observation or hypothesis?",
        "",
        "### Paragraph Context & Compound Citation Policy",
        "",
        "Every in-text citation whose reference PDF is available in the local library is audited directly against the 3-slot rubric. When a sentence cites multiple references and some are missing from the corpus, the audit notes that the available reference may only support part of the compound assertion. Citations for unavailable references are cataloged in Section 3 so they can be uploaded directly.",
        "",
        "---",
        "",
    ])

    missing_refs = get_deferred_missing_references(report)
    if totals.get("deferred_paywalled", 0) > 0:
        lines.append(f"## 3. Missing References Required for Deferred Paragraphs ({len(missing_refs)})\n")
    else:
        lines.append(f"## 3. Missing References Needed for Full Verification ({len(missing_refs)})\n")
    if missing_refs:
        req_col = "Required By Deferred Claim(s)" if totals.get("deferred_paywalled", 0) > 0 else "Required By Claim(s)"
        lines.append(f"| Reference / Title | Authors | Year | DOI | {req_col} |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for ref in missing_refs:
            ref_num = f"[{ref.get('index') or '?'}]"
            title_s = f"{ref_num} {ref.get('title') or 'Unknown Title'}"
            authors_list = ref.get("authors", [])
            auth_s = (
                ", ".join(authors_list[:2]) + (" et al." if len(authors_list) > 2 else "")
                if authors_list
                else "Unknown authors"
            )
            yr_s = str(ref.get("year") or "N/A")
            doi_val = ref.get("doi")
            doi_s = f"[`{doi_val}`](https://doi.org/{doi_val})" if doi_val else "N/A"
            affected = ref.get("affected_claims", [])
            claims_s = "<br>• ".join(f'"{c}"' for c in affected[:3]) if affected else "—"
            if len(affected) > 3:
                claims_s += f"<br>*(+{len(affected) - 3} more)*"
            lines.append(f"| {title_s} | {auth_s} | {yr_s} | {doi_s} | {claims_s} |")
        lines.append("")
    else:
        lines.append("*All references cited in the paper are available in the local corpus.*\n")

    needs_review = [
        r for r in results
        if r.get("outcome") == "judged"
        and r.get("judgement") in ("Contradicts", "Does not support", "Unclear / insufficient evidence")
    ]
    supported = [
        r for r in results
        if r.get("outcome") == "judged"
        and r.get("judgement") in ("Supports", "Partially supports")
    ]
    not_assessed = [
        r for r in results
        if r.get("outcome") not in ("judged", "not_downloaded", "deferred_paywalled")
    ]
    deferred = [r for r in results if r.get("outcome") == "deferred_paywalled"]
    not_downloaded = [r for r in results if r.get("outcome") == "not_downloaded"]

    def _format_entry(item, index):
        ref_info = item.get("ref") or {}
        ref_num = f"[{ref_info.get('index') or '?'}]"
        ref_title = ref_info.get("title") or item.get("cite_text") or "Unknown Reference"
        ref_year = f" ({ref_info.get('year')})" if ref_info.get("year") else ""
        ref_authors = (
            ", ".join(ref_info.get("authors", [])[:3])
            + (" et al." if len(ref_info.get("authors", [])) > 3 else "")
            if ref_info.get("authors")
            else "Unknown authors"
        )
        doi = ref_info.get("doi")
        doi_str = f"https://doi.org/{doi}" if doi else "N/A"

        badge = _audit_item_badge(item)

        rel_line = (
            f"- **Scientific Reliability:** `{item.get('reliability_badge', '⚪ Unresolved')}` — *{item.get('reliability_explanation', '')}*"
            if item.get("reliability_badge")
            else None
        )
        source_ass = item.get("source_assessment") or {}
        src_line = (
            f"- **Source Quality:** `{source_ass.get('badge', '❓ Unknown')}` ({source_ass.get('grade_label', '')}) · *{source_ass.get('rationale', '')}*"
            if source_ass.get("badge")
            else None
        )

        judged = item.get("outcome") == "judged"
        out_lines = [f"### Citation {index}: {ref_num} {ref_title}{ref_year}", ""]
        if judged:
            out_lines.append(
                f"- **Verdict:** `{badge}` · **Confidence:** `{item.get('confidence')}` · "
                f"**Evidence Sufficiency:** `{item.get('evidence_sufficiency')}`"
            )
        else:
            out_lines.append(f"- **Status:** `{badge}` — *{item.get('reliability_explanation') or explain_rubric_verdict(item)}*")
        if judged and rel_line:
            out_lines.append(rel_line)
        if src_line:
            out_lines.append(src_line)
        role = item.get("role"); section = item.get("section_heading") or item.get("section")
        if role or section:
            out_lines.append(f"- **Cited in:** {section or '—'} · **Citation role:** {role or '—'}")
        out_lines.extend([
            f"- **Statement in Paper:**  \n  > \"{item.get('sentence') or item.get('claim', '')}\"",
            (
                f"- **Cited Reference:** {ref_authors}{ref_year}. *{ref_title}*. DOI: [`{doi}`]({doi_str})"
                if doi
                else f"- **Cited Reference:** {ref_authors}{ref_year}. *{ref_title}*"
            ),
        ])
        if judged:
            secs = ", ".join(item.get("evidence_sections") or []) or "—"
            pages = ", ".join(str(p) for p in (item.get("evidence_pages") or []))
            out_lines.append(f"- **Evidence from:** {secs}" + (f" (p. {pages})" if pages else "") +
                             f" · {item.get('evidence_hits', 0)} passage(s)")
            if item.get("escalated"):
                out_lines.append(f"- ↻ **Escalated:** first verdict {item.get('first_judgement')}; re-judged on more evidence.")
            if item.get("span_verified") is False:
                out_lines.append("- ⚠ **Supporting span not found verbatim in the evidence** — treat it as a paraphrase.")
            if item.get("rubric_mismatch"):
                out_lines.append(f"- ⚠ **Rubric:** model said {item.get('model_judgement')}; the rules derive {item.get('judgement')}.")
            if "too_long" in (item.get("claim_quality") or []):
                out_lines.append("- ⚠ **Long sentence:** the claim is over 80 words; the verdict is about the whole sentence.")

        slots = item.get("slots")
        if slots and isinstance(slots, dict):
            out_lines.append("")
            out_lines.append("**Slot Decomposition Analysis:**")
            out_lines.append("")
            out_lines.append("| Slot | Assertion Extracted from Claim | Evaluation |")
            out_lines.append("| :--- | :--- | :--- |")
            for slot_name in ("finding", "scope", "strength"):
                sdata = slots.get(slot_name) or {}
                if isinstance(sdata, dict):
                    asrt = sdata.get("assertion", "—")
                    v = sdata.get("verdict", "—")
                else:
                    asrt = "—"
                    v = str(sdata or "—")
                out_lines.append(f"| **`{slot_name}`** | {asrt} | `{v}` |")

        if item.get("supporting_span"):
            out_lines.append("")
            out_lines.append(f"- **Verbatim Evidence from Cited Paper:**  \n  > \"{item['supporting_span']}\"")
        elif item.get("evidence"):
            out_lines.append("")
            out_lines.append(f"- **Evidence Passages Considered:**  \n  > \"{item['evidence'][:350].strip()}...\"")

        out_lines.append("")
        if judged and item.get("reason"):
            out_lines.append(f"- **Assessment:** {item['reason']}")
        out_lines.append(f"- **Why this verdict?** {explain_rubric_verdict(item)}")
        out_lines.append("")
        out_lines.append("---")
        out_lines.append("")
        return "\n".join(out_lines)

    current_sec = 4
    if needs_review:
        lines.append(f"## {current_sec}. Citations Needing Review ({len(needs_review)})\n")
        for i, item in enumerate(needs_review, 1):
            lines.append(_format_entry(item, i))
        current_sec += 1

    if supported:
        lines.append(f"## {current_sec}. Supported Citations ({len(supported)})\n")
        for i, item in enumerate(supported, 1):
            lines.append(_format_entry(item, i))
        current_sec += 1

    if not_assessed:
        lines.append(f"## {current_sec}. Not Assessed ({len(not_assessed)})\n")
        for i, item in enumerate(sorted(not_assessed, key=lambda r: str(r.get("outcome") or "")), 1):
            lines.append(_format_entry(item, i))
        current_sec += 1

    if deferred:
        lines.append(f"## {current_sec}. Deferred Paragraphs ({len({r.get('paragraph_id') for r in deferred})})\n")
        by_para: dict = {}
        for r in deferred:
            by_para.setdefault(r.get("paragraph_id", "p_?"), []).append(r)
        for p_id, items in by_para.items():
            first = items[0]
            missing = first.get("paragraph_missing_refs") or []
            lines.append(f"### Paragraph {p_id} ({first.get('section', '—')}) — {len(missing)} missing reference(s)\n")
            for r in items:
                lines.append(f"> {r.get('sentence')}\n")
            for m in missing:
                if not isinstance(m, dict):
                    continue
                doi_val = m.get("doi")
                doi_s = f" · [`{doi_val}`](https://doi.org/{doi_val})" if doi_val else ""
                lines.append(f"- [{m.get('index') or '?'}] {m.get('title') or 'Unknown title'} ({m.get('year') or 'n.d.'}){doi_s}")
            lines.append("\n---\n")
        current_sec += 1

    if not_downloaded:
        lines.append(f"## {current_sec}. Paywalled or Unavailable Citations ({len(not_downloaded)})\n")
        for i, item in enumerate(not_downloaded, 1):
            lines.append(_format_entry(item, i))
        current_sec += 1

    return "\n".join(lines)
