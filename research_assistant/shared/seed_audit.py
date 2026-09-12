"""
seed_audit.py — Extract in-text citation claims from uploaded seed paper TEI XML
and verify them against downloaded/ingested reference PDFs using the Agent 8 judgement rubric.
"""

import glob
import json
import os
import re
from datetime import datetime
from typing import Optional

from bs4 import BeautifulSoup

from research_assistant.agents.agent5_batch_citer import split_into_sentences
from research_assistant.agents.agent8_verifier import (
    _judge_once,
    _judgement_model,
    assemble_evidence,
    needs_escalation,
)
from research_assistant.config import (
    DOWNLOADED_JSON_PATH,
    JUDGEMENT_ESCALATE_TOP_K,
    JUDGEMENT_EVIDENCE_MAX_CHARS,
    JUDGEMENT_NEIGHBOUR_WINDOW,
    JUDGEMENT_TOP_K,
    RAW_DIR,
)
from research_assistant.shared import pipeline_status
from research_assistant.shared.atomic import atomic_write_json
from research_assistant.shared.log import get_logger
from research_assistant.shared.search import expand_neighbours, hybrid_search
from research_assistant.shared.source_key import normalise_doi

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
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\[\s*\]", "", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return " ".join(text.split()).strip()


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


def extract_seed_citation_claims(tei_source: str | BeautifulSoup) -> list[dict]:
    """Extract in-text citation claims from TEI XML.

    Returns a list of dicts with keys:
        - sentence: original sentence with [citation] marker
        - claim: cleaned sentence as plain prose (citations removed)
        - cite_text: raw in-text citation text (e.g. "[14]", "Smith et al. (2020)")
        - target: xml:id target (e.g. "b13")
        - ref: structured reference dict (xml_id, index, title, authors, year, doi, raw)
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

            ref_info = {
                "xml_id": xid,
                "index": idx + 1,
                "title": title,
                "authors": authors,
                "year": year,
                "doi": doi,
                "raw_reference": raw_reference,
            }
            bib_by_id[xid] = ref_info
            bib_by_index[idx + 1] = ref_info

    # 2. Parse body paragraphs
    body = soup.find("body")
    if not body:
        return []

    claims = []
    seen_pairs = set()

    for p in body.find_all("p"):
        refs = p.find_all("ref", type="bibr")
        if not refs:
            continue

        # Replace ref tags with identifiable tokens
        for idx, ref in enumerate(refs):
            target = (ref.get("target") or "").lstrip("#")
            txt = _clean(ref)
            ref.replace_with(f" __CITE_{idx}_{target}_{txt}__ ")

        clean_p = _clean(p)
        sentences = split_into_sentences(clean_p)

        for sent in sentences:
            matches = re.findall(r"__CITE_\d+_([^_]*)_([^_]*)__", sent)
            if not matches:
                continue

            # Produce clean claim
            claim_text = re.sub(r"__CITE_\d+_[^_]*_[^_]*__", "", sent)
            claim_text = _clean_claim_punctuation(claim_text)
            if len(claim_text) < 15:
                continue

            # Readable sentence with citations
            human_sent = sent
            for target, txt in matches:
                display_tag = f"[{txt.strip('[]')}]" if txt else ""
                human_sent = re.sub(
                    rf"__CITE_\d+_{re.escape(target)}_{re.escape(txt)}__",
                    display_tag,
                    human_sent,
                    count=1,
                )
            human_sent = " ".join(human_sent.split()).strip()

            for target, txt in matches:
                ref_info = bib_by_id.get(target)
                if not ref_info:
                    # Try numeric citation e.g. [14]
                    nums = re.findall(r"\d+", txt)
                    if nums:
                        try:
                            ref_info = bib_by_index.get(int(nums[0]))
                        except (ValueError, TypeError):
                            pass

                # If still not found and txt has author name, try matching surname
                if not ref_info and txt:
                    for r in bib_by_id.values():
                        if any(
                            author.split()[-1].lower() in txt.lower()
                            for author in r.get("authors", [])
                            if author.split()
                        ):
                            ref_info = r
                            break

                ref_key = ref_info.get("xml_id") if ref_info else (target or txt)
                dedup_key = (claim_text, ref_key)
                if dedup_key in seen_pairs:
                    continue
                seen_pairs.add(dedup_key)

                claims.append({
                    "sentence": human_sent,
                    "claim": claim_text,
                    "cite_text": txt,
                    "target": target,
                    "ref": ref_info,
                })

    return claims


def _load_downloaded_manifest() -> dict:
    if not os.path.exists(DOWNLOADED_JSON_PATH):
        return {}
    try:
        with open(DOWNLOADED_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not read %s: %s", DOWNLOADED_JSON_PATH, e)
        return {}


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


def audit_seed_citations(
    seed_path: str,
    search_resources=None,
    max_claims: int = 20,
    top_k: int = JUDGEMENT_TOP_K,
) -> dict:
    """Audit the in-text citation claims of an uploaded seed paper against the local corpus.

    Args:
        seed_path: Path to the seed PDF.
        search_resources: Tuple of (collection, bm25, texts, metadatas) or None to load.
        max_claims: Maximum number of claims citing downloaded papers to judge with the LLM.
        top_k: Number of hits to retrieve per claim.

    Returns:
        Structured audit report dict containing totals and detailed item results.
    """
    stem = os.path.splitext(os.path.basename(seed_path))[0] if seed_path else "unknown"
    seed_pdf_name = os.path.basename(seed_path) if seed_path else ""

    tei_path = find_tei_for_seed(seed_path)
    if not tei_path:
        logger.info("No TEI XML found for seed PDF %s — skipping citation audit.", seed_path)
        return {
            "seed_path": seed_path,
            "error": "No GROBID TEI XML found for seed PDF.",
            "totals": {"total": 0, "judged": 0},
            "results": [],
        }

    claims = extract_seed_citation_claims(tei_path)
    if not claims:
        logger.info("No in-text citation claims found in %s.", tei_path)
        return {
            "seed_path": seed_path,
            "totals": {"total": 0, "judged": 0},
            "results": [],
        }

    downloaded_manifest = _load_downloaded_manifest()

    # Enrich claims with downloaded status
    downloaded_claims = []
    undownloaded_claims = []

    for c in claims:
        ref = c.get("ref")
        dl_entry = _match_downloaded_paper(ref, seed_pdf_name, downloaded_manifest)
        if dl_entry and dl_entry.get("path") and os.path.exists(dl_entry["path"]):
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

    claims_to_judge = downloaded_claims[:max_claims]

    if claims_to_judge:
        if search_resources is None:
            from research_assistant.shared.db import load_search_resources

            search_resources = load_search_resources()
        collection, bm25, texts, metadatas = search_resources

        from research_assistant.judgement.judge import DERIVED_FIELDS, REQUIRED_FIELDS, JudgementParseError

        escalate_k = JUDGEMENT_ESCALATE_TOP_K if JUDGEMENT_ESCALATE_TOP_K > top_k else 0
        retrieve_k = max(top_k, escalate_k)

        for i, item in enumerate(claims_to_judge, 1):
            ref_info = item.get("ref") or {}
            ref_lbl = (
                f"[{ref_info.get('index') or '?'}] {ref_info.get('title') or item.get('cite_text', '')}"
            )
            pipeline_status.update_progress(
                detail=f"Judging citation ({i}/{len(claims_to_judge)}): {ref_lbl[:40]}"
            )
            logger.info(
                "[%d/%d] Auditing citation %s: %s",
                i,
                len(claims_to_judge),
                ref_lbl[:40],
                item["claim"][:80],
            )

            try:
                hits = hybrid_search(
                    item["claim"],
                    collection,
                    bm25,
                    texts,
                    metadatas,
                    top_k=retrieve_k,
                    doc_filter={item["document"]},
                    exclude_types={"figure_description"},
                )
            except Exception as exc:
                logger.warning("Retrieval failed for %s: %s", item["document"], exc)
                item["outcome"] = "retrieval_failed"
                item["judgement"] = "Unclear / insufficient evidence"
                item["reason"] = f"Retrieval failed: {exc}"
                continue

            if not hits:
                item["outcome"] = "no_evidence"
                item["judgement"] = "Unclear / insufficient evidence"
                item["reason"] = "No relevant passages found in the cited document."
                continue

            expand_neighbours(hits, texts, metadatas, window=JUDGEMENT_NEIGHBOUR_WINDOW)
            item["evidence"] = assemble_evidence(hits[:top_k], JUDGEMENT_EVIDENCE_MAX_CHARS)

            try:
                verdict = _judge_once(item["claim"], item["evidence"])
                if escalate_k and len(hits) > top_k and needs_escalation(verdict):
                    item["first_judgement"] = verdict.get("judgement")
                    item["evidence"] = assemble_evidence(
                        hits[:escalate_k], JUDGEMENT_EVIDENCE_MAX_CHARS
                    )
                    verdict = _judge_once(item["claim"], item["evidence"])

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
                item["judgement"] = "Unclear / insufficient evidence"
                item["reason"] = "Model returned unparseable response."
            except Exception as exc:
                logger.warning("Judgement call failed: %s", exc)
                item["outcome"] = "call_failed"
                item["judgement"] = "Unclear / insufficient evidence"
                item["reason"] = f"Model evaluation error: {exc}"

    all_results = claims_to_judge + downloaded_claims[max_claims:] + undownloaded_claims

    totals = {
        "total": len(all_results),
        "downloaded": len(downloaded_claims),
        "judged": sum(1 for r in all_results if r.get("outcome") == "judged"),
        "Supports": sum(1 for r in all_results if r.get("judgement") == "Supports"),
        "Partially supports": sum(
            1 for r in all_results if r.get("judgement") == "Partially supports"
        ),
        "Contradicts": sum(1 for r in all_results if r.get("judgement") == "Contradicts"),
        "Does not support": sum(
            1 for r in all_results if r.get("judgement") == "Does not support"
        ),
        "Unclear / insufficient evidence": sum(
            1 for r in all_results if r.get("judgement") == "Unclear / insufficient evidence"
        ),
        "not_downloaded": sum(1 for r in all_results if r.get("outcome") == "not_downloaded"),
    }

    report = {
        "seed_path": seed_path,
        "seed_name": seed_pdf_name,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "model": _judgement_model(),
        "totals": totals,
        "results": all_results,
    }

    os.makedirs(AUDIT_DIR, exist_ok=True)
    out_file = os.path.join(AUDIT_DIR, f"{stem}_audit.json")
    atomic_write_json(out_file, report)
    logger.info("Saved seed citation audit to %s", out_file)

    return report
