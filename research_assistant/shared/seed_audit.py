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
        - paragraph_id: identifier of the body paragraph (e.g. "p_0" or xml:id)
        - paragraph_index: 0-based integer index of the body paragraph
        - paragraph_refs: list of all unique reference dicts cited in this paragraph
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

    for p_idx, p in enumerate(body.find_all("p")):
        refs = p.find_all("ref", type="bibr")
        if not refs:
            continue

        p_id = p.get("xml:id") or p.get("id") or f"p_{p_idx}"

        # Resolve all unique reference dicts cited in this paragraph
        p_refs_dict = {}
        for idx, ref in enumerate(refs):
            target = (ref.get("target") or "").lstrip("#")
            txt = _clean(ref)
            ref_info = bib_by_id.get(target)
            if not ref_info:
                # Try numeric citation e.g. [14]
                nums = re.findall(r"\d+", txt)
                if nums:
                    try:
                        ref_info = bib_by_index.get(int(nums[0]))
                    except (ValueError, TypeError):
                        pass

            if not ref_info and txt:
                for r in bib_by_id.values():
                    if any(
                        author.split()[-1].lower() in txt.lower()
                        for author in r.get("authors", [])
                        if author.split()
                    ):
                        ref_info = r
                        break

            if not ref_info:
                ref_info = {
                    "xml_id": target or f"unknown_{txt}",
                    "index": None,
                    "title": txt or "Unknown reference",
                    "authors": [],
                    "year": None,
                    "doi": None,
                    "raw_reference": txt,
                }

            rk = ref_info.get("xml_id") or normalise_doi(ref_info.get("doi")) or ref_info.get("title") or target or txt
            if rk:
                p_refs_dict[rk] = ref_info

            # Replace ref tag with identifiable token
            ref.replace_with(f" __CITE_{idx}_{target}_{txt}__ ")

        p_unique_refs = list(p_refs_dict.values())

        clean_p = _clean(p)
        sentences = split_into_sentences(clean_p)

        paragraph_claims = []
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

                if not ref_info:
                    ref_info = {
                        "xml_id": target or f"unknown_{txt}",
                        "index": None,
                        "title": txt or "Unknown reference",
                        "authors": [],
                        "year": None,
                        "doi": None,
                        "raw_reference": txt,
                    }

                ref_key = ref_info.get("xml_id") if ref_info else (target or txt)
                dedup_key = (claim_text, ref_key)
                if dedup_key in seen_pairs:
                    continue
                seen_pairs.add(dedup_key)

                paragraph_claims.append({
                    "sentence": human_sent,
                    "claim": claim_text,
                    "cite_text": txt,
                    "target": target,
                    "ref": ref_info,
                    "paragraph_id": p_id,
                    "paragraph_index": p_idx,
                    "paragraph_refs": p_unique_refs,
                })

        claims.extend(paragraph_claims)

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
        logger.warning("Retrieval failed for %s: %s", item.get("document"), exc)
        item["outcome"] = "retrieval_failed"
        item["judgement"] = "Unclear / insufficient evidence"
        item["reason"] = f"Retrieval failed: {exc}"
        return item

    if not hits:
        item["outcome"] = "no_evidence"
        item["judgement"] = "Unclear / insufficient evidence"
        item["reason"] = "No relevant passages found in the cited document."
        return item

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

    return item


def get_cached_seed_audit(seed_path: str) -> dict | None:
    """Retrieve an existing audit report for a seed paper if available on disk.

    Checks by filename stem, canonical arXiv/DOI/content-hash keys, and returns
    the parsed report dict, or None if no valid audit exists.
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
                    return report
            except Exception as exc:
                logger.warning("Failed to load cached audit from %s: %s", candidate, exc)

    return None


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
        )
    ]

    claims_to_rejudge = []
    for item in unresolved_claims:
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

    if claims_to_rejudge and max_new_claims > 0:
        logger.info(
            "Found %d previously unresolved citation(s) now available in corpus. Judging up to %d...",
            len(claims_to_rejudge),
            max_new_claims,
        )
        if search_resources is None:
            from research_assistant.shared.db import load_search_resources

            search_resources = load_search_resources()
        collection, bm25, texts, metadatas = search_resources

        for item in claims_to_rejudge[:max_new_claims]:
            _judge_claim_entry(item, collection, bm25, texts, metadatas)
            newly_judged_count += 1

    # 2. In-memory refresh of Source Assessor and Reliability Policy across all claims
    for r in results:
        source_eval = assess_source(metadata=r.get("metadata") or {}, ref_info=r.get("ref"))
        r["source_grade"] = source_eval["grade"]
        r["source_assessment"] = source_eval
        rel_eval = evaluate_reliability(
            relation=r.get("judgement", "Unclear / insufficient evidence"),
            source_grade=source_eval["grade"],
            confidence=r.get("confidence", "Medium"),
            span_verified=r.get("span_verified"),
            rubric_violations=r.get("rubric_violations"),
            rubric_mismatch=r.get("rubric_mismatch", False),
        )
        r["reliability"] = rel_eval["rating"]
        r["reliability_badge"] = rel_eval["badge"]
        r["reliability_label"] = rel_eval["rating_label"]
        r["reliability_explanation"] = rel_eval["explanation"]

    # 3. Recalculate totals
    totals = {
        "total": len(results),
        "downloaded": sum(1 for r in results if r.get("downloaded")),
        "judged": sum(1 for r in results if r.get("outcome") == "judged"),
        "Supports": sum(1 for r in results if r.get("judgement") == "Supports"),
        "Partially supports": sum(
            1 for r in results if r.get("judgement") == "Partially supports"
        ),
        "Contradicts": sum(1 for r in results if r.get("judgement") == "Contradicts"),
        "Does not support": sum(
            1 for r in results if r.get("judgement") == "Does not support"
        ),
        "Unclear / insufficient evidence": sum(
            1 for r in results if r.get("judgement") == "Unclear / insufficient evidence"
        ),
        "not_downloaded": sum(1 for r in results if r.get("outcome") == "not_downloaded"),
        "deferred_paywalled": sum(
            1 for r in results if r.get("outcome") == "deferred_paywalled"
        ),
        "reliability": {
            "high": sum(1 for r in results if r.get("reliability") == "HIGH"),
            "moderate": sum(1 for r in results if r.get("reliability") == "MODERATE"),
            "low": sum(1 for r in results if r.get("reliability") == "LOW"),
            "contradicted": sum(
                1 for r in results if r.get("reliability") == "CONTRADICTED"
            ),
            "unresolved": sum(1 for r in results if r.get("reliability") == "UNRESOLVED"),
        },
    }

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

        if paywall_ratio > 0.50:
            # Mark entire paragraph and all claims as deferred
            for c in p_claims:
                ref = c.get("ref")
                dl_entry = _match_downloaded_paper(ref, seed_pdf_name, downloaded_manifest)
                is_dl = bool(dl_entry and dl_entry.get("path") and os.path.exists(dl_entry["path"]))
                if is_dl:
                    c["document"] = os.path.basename(dl_entry["path"])
                    c["citation_source"] = (
                        dl_entry.get("title")
                        or dl_entry.get("raw_reference")
                        or dl_entry.get("key")
                    )
                c["downloaded"] = is_dl
                c["outcome"] = "deferred_paywalled"
                c["judgement"] = "Deferred (pending paywalled evidence)"
                if is_dl:
                    c["reason"] = (
                        f"Evaluation deferred: Although this reference is in the corpus, "
                        f"{missing_count}/{total_p_refs} references cited in this paragraph are missing (>50% paywalled)."
                    )
                else:
                    c["reason"] = (
                        f"Evaluation deferred: {missing_count}/{total_p_refs} references cited in this "
                        f"paragraph are missing from the corpus (>50% paywalled)."
                    )
                c["paragraph_missing_refs"] = missing_p_refs
                c["paywall_ratio"] = paywall_ratio
                deferred_claims.append(c)
        else:
            # Paragraph is under threshold: evaluate individual claims
            for c in p_claims:
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
                    c["paragraph_missing_refs"] = missing_p_refs
                    c["paywall_ratio"] = paywall_ratio
                    downloaded_claims.append(c)
                else:
                    c["downloaded"] = False
                    c["outcome"] = "not_downloaded"
                    c["judgement"] = "Not downloaded"
                    c["reason"] = "Reference PDF was not available or could not be downloaded."
                    c["paragraph_missing_refs"] = missing_p_refs
                    c["paywall_ratio"] = paywall_ratio
                    undownloaded_claims.append(c)

    logger.info(
        "Found %d in-text citation claims (%d cite downloaded references, %d not in corpus, %d deferred paywalled).",
        len(claims),
        len(downloaded_claims),
        len(undownloaded_claims),
        len(deferred_claims),
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
            _judge_claim_entry(
                item,
                collection,
                bm25,
                texts,
                metadatas,
                top_k=top_k,
                escalate_k=escalate_k,
            )

    for c in downloaded_claims[max_claims:]:
        if not c.get("outcome"):
            c["outcome"] = "cap_exceeded"
            c["judgement"] = "Unclear / insufficient evidence"
            c["reason"] = "Maximum claims evaluation budget reached."

    all_results = claims_to_judge + downloaded_claims[max_claims:] + undownloaded_claims + deferred_claims

    for r in all_results:
        source_eval = assess_source(metadata=r.get("metadata") or {}, ref_info=r.get("ref"))
        r["source_grade"] = source_eval["grade"]
        r["source_assessment"] = source_eval
        rel_eval = evaluate_reliability(
            relation=r.get("judgement", "Unclear / insufficient evidence"),
            source_grade=source_eval["grade"],
            confidence=r.get("confidence", "Medium"),
            span_verified=r.get("span_verified"),
            rubric_violations=r.get("rubric_violations"),
            rubric_mismatch=r.get("rubric_mismatch", False),
        )
        r["reliability"] = rel_eval["rating"]
        r["reliability_badge"] = rel_eval["badge"]
        r["reliability_label"] = rel_eval["rating_label"]
        r["reliability_explanation"] = rel_eval["explanation"]

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
        "deferred_paywalled": sum(1 for r in all_results if r.get("outcome") == "deferred_paywalled"),
        "reliability": {
            "high": sum(1 for r in all_results if r.get("reliability") == "HIGH"),
            "moderate": sum(1 for r in all_results if r.get("reliability") == "MODERATE"),
            "low": sum(1 for r in all_results if r.get("reliability") == "LOW"),
            "contradicted": sum(1 for r in all_results if r.get("reliability") == "CONTRADICTED"),
            "unresolved": sum(1 for r in all_results if r.get("reliability") == "UNRESOLVED"),
        },
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
    deferred_results = [r for r in results if r.get("outcome") == "deferred_paywalled"]

    missing_map = {}

    for item in deferred_results:
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
    judgement = item.get("judgement", "")
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
        return (
            "The cited reference is in the corpus, but semantic and keyword search found no passages discussing this specific assertion."
        )

    if outcome in ("retrieval_failed", "call_failed", "parse_failed"):
        return f"Evaluation could not complete due to a processing issue: {item.get('reason', outcome)}."

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
        f"| ⚪ **Unclear / Insufficient** | **{totals.get('Unclear / insufficient evidence', 0)}** | {pct(totals.get('Unclear / insufficient evidence', 0))} | Fragile or fragmentary evidence |",
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
        f"| ⏳ **Unresolved / Pending** | **{totals.get('reliability', {}).get('unresolved', 0)}** | {pct(totals.get('reliability', {}).get('unresolved', 0))} | Unretrieved, paywalled, or insufficient evidence |",
        "",
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
        "### Paragraph-Level Evidence Threshold & Deferral Policy",
        "",
        "To prevent spurious contradictions or premature negative verdicts when key literature is missing, citation verification operates on paragraph context:",
        "- If **more than 50% (>50%)** of the unique references cited in a paragraph are unavailable (missing/paywalled), evaluation of all claims in that paragraph is **deferred** (`Deferred (pending paywalled evidence)`).",
        "- When **50% or more** of a paragraph's cited references are present in the corpus, claims citing available references are evaluated against the 3-slot rubric, while claims citing missing references are marked as `Not downloaded`.",
        "",
        "---",
        "",
    ]

    deferred_missing = get_deferred_missing_references(report)
    lines.append(f"## 3. Missing References Required for Deferred Paragraphs ({len(deferred_missing)})\n")
    if deferred_missing:
        lines.append("| Reference / Title | Authors | Year | DOI | Required By Deferred Claim(s) |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for ref in deferred_missing:
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
        lines.append("*No paragraphs were deferred; all citations were either available or under the 50% paywall threshold.*\n")

    needs_review = [
        r for r in results
        if r.get("judgement") in ("Contradicts", "Does not support", "Unclear / insufficient evidence")
    ]
    supported = [
        r for r in results
        if r.get("judgement") in ("Supports", "Partially supports")
    ]
    paywalled = [r for r in results if r.get("outcome") in ("not_downloaded", "deferred_paywalled")]
    other = [r for r in results if r not in needs_review and r not in supported and r not in paywalled]

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

        judgement = item.get("judgement", "Unclear")
        confidence = item.get("confidence", "Medium")
        sufficiency = item.get("evidence_sufficiency", "partial")

        badge_map = {
            "Supports": "🟢 Supports",
            "Partially supports": "🟡 Partially Supports",
            "Contradicts": "🔴 Contradicts",
            "Does not support": "🟠 Does Not Support",
            "Unclear / insufficient evidence": "⚪ Unclear / Insufficient Evidence",
            "Deferred (pending paywalled evidence)": "⏳ Deferred (Pending Evidence)",
        }
        if item.get("outcome") == "deferred_paywalled":
            badge = "⏳ Deferred (Pending Evidence)"
        elif item.get("outcome") == "not_downloaded":
            badge = "🔒 Paywalled / Not In Corpus"
        else:
            badge = badge_map.get(judgement, "⚪ Unclear / Insufficient Evidence")

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

        out_lines = [
            f"### Citation {index}: {ref_num} {ref_title}{ref_year}",
            "",
            f"- **Verdict:** `{badge}` · **Confidence:** `{confidence}` · **Evidence Sufficiency:** `{sufficiency}`",
        ]
        if rel_line:
            out_lines.append(rel_line)
        if src_line:
            out_lines.append(src_line)
        out_lines.extend([
            f"- **Statement in Paper:**  \n  > \"{item.get('sentence') or item.get('claim', '')}\"",
            (
                f"- **Cited Reference:** {ref_authors}{ref_year}. *{ref_title}*. DOI: [`{doi}`]({doi_str})"
                if doi
                else f"- **Cited Reference:** {ref_authors}{ref_year}. *{ref_title}*"
            ),
        ])

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
        if item.get("reason"):
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

    if paywalled:
        lines.append(f"## {current_sec}. Paywalled or Unavailable Citations ({len(paywalled)})\n")
        for i, item in enumerate(paywalled, 1):
            lines.append(_format_entry(item, i))
        current_sec += 1

    if other:
        lines.append(f"## {current_sec}. Other Citations ({len(other)})\n")
        for i, item in enumerate(other, 1):
            lines.append(_format_entry(item, i))
        current_sec += 1

    return "\n".join(lines)
