"""
academic_apis.py — Unified client library for academic and bibliometric APIs.

Provides rate-limited, cached, and robust client functions for:
- OpenAlex (metadata, citation counts, retractions, open access)
- CrossRef (DOI resolution, metadata, citations)
- Semantic Scholar (graph API, abstracts, paper recommendations)
- arXiv (E-utilities and arXiv query API)
"""

from __future__ import annotations

import html
import json
import os
import re
import time
from typing import Any, Dict, List, Optional
import urllib.parse

import requests

from research_assistant.config import (
    ARXIV_RATE_LIMIT,
    UNPAYWALL_EMAIL,
    UNPAYWALL_SLEEP,
    USER_AGENT,
)
from research_assistant.shared.log import get_logger

logger = get_logger("academic_apis")

DEFAULT_TIMEOUT = 15
DEFAULT_HEADERS = {"User-Agent": USER_AGENT}

_LAST_CALL_TIMESTAMPS: Dict[str, float] = {}


def _rate_limit(api_name: str, min_interval_seconds: float) -> None:
    """Enforces minimum interval between calls to an API domain."""
    now = time.time()
    last = _LAST_CALL_TIMESTAMPS.get(api_name, 0.0)
    elapsed = now - last
    if elapsed < min_interval_seconds:
        time.sleep(min_interval_seconds - elapsed)
    _LAST_CALL_TIMESTAMPS[api_name] = time.time()


ROMAN_NUMERALS = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"}
_CONSORTIUM_SURNAMES = frozenset({
    "team", "consortium", "collaboration", "group", "project", "authors",
    "openai", "anthropic", "deepmind", "google", "meta", "qwen", "deepseek", "kimi", "gemma",
})

_TRAILING_YEAR_IN_TITLE_RE = re.compile(r"[.,;:\s]+\b(1[89]\d{2}|20\d{2})\b[.,;\s]*$")
_GLUED_AUTHOR_PREFIX_RE = re.compile(r"^([A-Z][a-zA-Z\-']{1,25}),\s+([A-Z].{15,})$")


def clean_reference_title(title: Optional[str]) -> str:
    """Strip common GROBID extraction artefacts (trailing year, trailing venue pipe) from a title."""
    if not title:
        return ""
    t = " ".join(str(title).split()).strip()
    t = _TRAILING_YEAR_IN_TITLE_RE.sub("", t).strip()
    if "|" in t:
        head = t.split("|", 1)[0].strip()
        if len(head) >= 15:
            t = head
    return t.rstrip(".,;:")


def title_variants(title: Optional[str]) -> List[str]:
    """Return candidate clean title variants (e.g. with glued leading author surname stripped)."""
    if not title:
        return []
    raw = " ".join(str(title).split()).strip()
    cleaned = clean_reference_title(raw)
    out: List[str] = []
    for cand in (cleaned, raw):
        if cand and cand not in out:
            out.append(cand)
    m = _GLUED_AUTHOR_PREFIX_RE.match(cleaned)
    if m and len(m.group(2).split()) >= 4:
        stripped = m.group(2).strip()
        if stripped not in out:
            out.append(stripped)
    return out


def _extract_author_surname(author: Optional[str]) -> str:
    """Extract normalized surname from an author string.
    Handles formats:
      - "Last, First" or "Last, Initials": returns "Last"
      - "First Last" or "Initials Last": returns "Last"
      - Single name "Last": returns "Last"
      - Compound names "Van der Waals, J. D." -> "vanderwaals"
    """
    if not author:
        return ""
    cleaned = author.strip()
    if not cleaned:
        return ""
    if "," in cleaned:
        parts = cleaned.split(",")
        first_part = parts[0].strip()
        tokens_before = re.findall(r"\w+", first_part)
        tokens_after = re.findall(r"\w+", parts[1]) if len(parts) > 1 else []
        if len("".join(tokens_before)) <= 2 and len("".join(tokens_after)) > 2:
            surname = tokens_after[0]
        else:
            surname = first_part
    else:
        tokens = cleaned.split()
        if not tokens:
            return ""
        # If every token is a 1-character initial (e.g. GROBID parsed "P. O. Dral" as "P O"),
        # do not treat the middle initial as a surname.
        if all(len(re.sub(r"[^\w]", "", t)) <= 1 for t in tokens):
            return ""
        if len(tokens) > 1 and tokens[-1].lower().rstrip(".") in {"jr", "sr", "ii", "iii", "iv", "v"}:
            surname = tokens[-2]
        else:
            surname = tokens[-1]
    norm_s = re.sub(r"[^\w]", "", surname).lower()
    return norm_s if len(norm_s) > 1 else ""


def _match_surnames(a1: str, a2: str) -> bool:
    """Determine whether two author strings represent the same first author surname."""
    s1 = _extract_author_surname(a1)
    s2 = _extract_author_surname(a2)
    if not s1 or not s2:
        return False
    if s1 == s2:
        return True
    if len(s1) >= 4 and len(s2) >= 4:
        # Support compound prefix / suffix matching (e.g. "vanderwaals" and "waals")
        if s1.endswith(s2) or s2.endswith(s1):
            return True
        import difflib
        ratio = difflib.SequenceMatcher(None, s1, s2).ratio()
        if ratio >= 0.85:
            return True
    return False


def _reconcile_compound_tokens(tokens1: set[str], tokens2: set[str], ordered1: List[str], ordered2: List[str]) -> tuple[set[str], set[str]]:
    """Reconcile PDF line-break word concatenations (e.g. 'spindependent' vs 'spin' + 'dependent')."""
    if not (tokens1 - tokens2) or not (tokens2 - tokens1):
        return tokens1, tokens2
    t1 = set(tokens1)
    t2 = set(tokens2)
    diff1 = t1 - t2
    diff2 = t2 - t1
    for i in range(len(ordered2) - 1):
        a, b = ordered2[i], ordered2[i + 1]
        joined = a + b
        if joined in diff1 and a in diff2 and b in diff2:
            t1.discard(joined)
            t1.update((a, b))
    diff1 = t1 - t2
    diff2 = t2 - t1
    for i in range(len(ordered1) - 1):
        a, b = ordered1[i], ordered1[i + 1]
        joined = a + b
        if joined in diff2 and a in diff1 and b in diff1:
            t2.discard(joined)
            t2.update((a, b))
    return t1, t2


def verify_candidate_match(
    target_title: str,
    candidate_title: str,
    target_author: Optional[str] = None,
    candidate_author: Optional[str] = None,
    target_year: Optional[int] = None,
    candidate_year: Optional[int] = None,
    candidate_authors: Optional[List[str]] = None,
    max_year_diff: int = 1,
) -> bool:
    """Enforce bidirectional title, author, and year similarity between target reference and candidate.
    Requires:
      - Conjunctive title similarity: Title token Jaccard >= 0.70 AND SequenceMatcher ratio >= 0.80
      - Congruent part/volume identifiers: No conflicting Roman numerals, digits, or layer modifiers
      - First author surname match (if both authors provided)
      - Publication year within 1 year (if both years provided)
    """
    if not target_title or not candidate_title:
        return False

    import difflib

    cand_clean = clean_reference_title(candidate_title) or candidate_title.strip()
    t2_variants = [candidate_title.strip().lower()]
    if cand_clean.lower() not in t2_variants:
        t2_variants.append(cand_clean.lower())

    t1_variants = [v.lower() for v in title_variants(target_title)] or [target_title.strip().lower()]

    title_matched = False
    best_jaccard = 0.0
    best_ratio = 0.0
    for t1 in t1_variants:
        for t2 in t2_variants:
            ordered1 = re.findall(r"[a-z0-9]+", t1)
            ordered2 = re.findall(r"[a-z0-9]+", t2)
            if not ordered1 or not ordered2:
                continue
            tokens1, tokens2 = _reconcile_compound_tokens(set(ordered1), set(ordered2), ordered1, ordered2)

            jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2)
            ratio = difflib.SequenceMatcher(None, t1, t2).ratio()
            if jaccard < 0.70 or ratio < 0.80:
                continue

            # Check for differing numbers or Roman numerals (e.g. Part I vs Part II, Vol 1 vs Vol 2)
            diff1 = tokens1 - tokens2
            diff2 = tokens2 - tokens1

            digits1 = {t for t in diff1 if t.isdigit()}
            digits2 = {t for t in diff2 if t.isdigit()}
            if digits1 and digits2 and digits1 != digits2:
                continue

            roman1 = {t for t in diff1 if t in ROMAN_NUMERALS}
            roman2 = {t for t in diff2 if t in ROMAN_NUMERALS}
            if roman1 and roman2 and roman1 != roman2:
                continue

            layer_mods = {"monolayer", "bilayer", "trilayer", "multilayer", "fewlayer"}
            if (diff1 & layer_mods) and (diff2 & layer_mods):
                continue

            title_matched = True
            best_jaccard = max(best_jaccard, jaccard)
            best_ratio = max(best_ratio, ratio)
            break
        if title_matched:
            break

    if not title_matched:
        return False

    # 2. Author verification
    if target_author and candidate_author:
        if not _match_surnames(target_author, candidate_author):
            s_tgt = _extract_author_surname(target_author)
            s_cand = _extract_author_surname(candidate_author)
            consortium_case = (s_tgt in _CONSORTIUM_SURNAMES or s_cand in _CONSORTIUM_SURNAMES)
            matched_alt = False
            if consortium_case and candidate_authors:
                matched_alt = any(_match_surnames(target_author, ca) for ca in candidate_authors if ca)
            if not matched_alt and consortium_case and best_jaccard >= 0.90 and best_ratio >= 0.92:
                matched_alt = True
            if not matched_alt:
                return False

    # 3. Year verification
    if target_year is not None and candidate_year is not None:
        try:
            if abs(int(target_year) - int(candidate_year)) > max_year_diff:
                return False
        except (ValueError, TypeError):
            pass

    return True



def call_crossref_api(
    query: str,
    rows: int = 3,
    mailto: Optional[str] = UNPAYWALL_EMAIL,
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[Dict[str, Any]]:
    """Queries CrossRef REST API for works matching a query string.

    With no contact address the request carries no mailto and uses the shared pool.
    """
    _rate_limit("crossref", 0.1)
    params: Dict[str, Any] = {
        "query.bibliographic": query,
        "rows": rows,
    }
    if mailto:
        params["mailto"] = mailto
    url = "https://api.crossref.org/works"
    headers = dict(DEFAULT_HEADERS)

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        logger.debug("CrossRef returned HTTP %d for query %s", resp.status_code, query[:40])
    except Exception as exc:
        logger.debug("CrossRef request failed for %s: %s", query[:40], exc)
    return None


def call_openalex_api(
    query_or_doi: str,
    mailto: Optional[str] = UNPAYWALL_EMAIL,
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[Dict[str, Any]]:
    """Queries OpenAlex API by DOI or search string.

    With no contact address the request carries no mailto and uses the shared pool.
    """
    _rate_limit("openalex", 0.1)
    clean_target = query_or_doi.strip()
    headers = dict(DEFAULT_HEADERS)
    contact = f"mailto={urllib.parse.quote(mailto)}" if mailto else ""

    if clean_target.startswith("10."):
        url = f"https://api.openalex.org/works/https://doi.org/{clean_target}"
        if contact:
            url += f"?{contact}"
    elif "doi.org" in clean_target:
        url = f"https://api.openalex.org/works/{clean_target}"
        if contact:
            url += f"?{contact}"
    else:
        encoded = urllib.parse.quote(clean_target)
        url = f"https://api.openalex.org/works?search={encoded}"
        if contact:
            url += f"&{contact}"

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        logger.debug("OpenAlex returned HTTP %d for %s", resp.status_code, query_or_doi[:40])
    except Exception as exc:
        logger.debug("OpenAlex request failed for %s: %s", query_or_doi[:40], exc)
    return None


def call_semantic_scholar_api(
    query_or_id: str,
    fields: str = "title,abstract,authors,year,citationCount,isOpenAccess,openAccessPdf,externalIds",
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[Dict[str, Any]]:
    """Queries Semantic Scholar Academic Graph API."""
    _rate_limit("semanticscholar", 0.5)
    clean_target = query_or_id.strip()
    headers = dict(DEFAULT_HEADERS)

    if clean_target.startswith("10."):
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{clean_target}?fields={fields}"
    elif clean_target.lower().startswith("arxiv:"):
        aid = re.sub(r"v\d+$", "", clean_target[6:].strip())
        url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{aid}?fields={fields}"
    else:
        encoded = urllib.parse.quote(clean_target)
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded}&limit=5&fields={fields}"

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        logger.debug("Semantic Scholar returned HTTP %d for %s", resp.status_code, query_or_id[:40])
    except Exception as exc:
        logger.debug("Semantic Scholar request failed for %s: %s", query_or_id[:40], exc)
    return None


def fetch_open_access_pdf_url(doi: str, mailto: Optional[str] = UNPAYWALL_EMAIL, timeout: int = DEFAULT_TIMEOUT) -> Optional[str]:
    """Queries Unpaywall for direct open-access PDF download URL for a DOI.

    Unpaywall refuses requests without a contact email, so none is sent without one.
    """
    if not mailto:
        return None
    _rate_limit("unpaywall", UNPAYWALL_SLEEP)
    clean_doi = doi.strip()
    if clean_doi.startswith("https://doi.org/"):
        clean_doi = clean_doi[16:]
    elif clean_doi.startswith("http://doi.org/"):
        clean_doi = clean_doi[15:]

    url = f"https://api.unpaywall.org/v2/{clean_doi}?email={mailto}"
    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            best_oa = data.get("best_oa_location")
            if best_oa and best_oa.get("url_for_pdf"):
                return best_oa["url_for_pdf"]
            if best_oa and best_oa.get("url"):
                return best_oa["url"]
    except Exception as exc:
        logger.debug("Unpaywall request failed for %s: %s", doi, exc)
    return None


def reconstruct_openalex_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """Reconstruct linear text from OpenAlex abstract_inverted_index."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return None
    pos_word = []
    for word, positions in inverted_index.items():
        if isinstance(positions, list):
            for pos in positions:
                pos_word.append((pos, word))
    if not pos_word:
        return None
    pos_word.sort(key=lambda x: x[0])
    return " ".join(w for _, w in pos_word)


def _collect_openalex_oa_urls(work: Dict[str, Any]) -> List[str]:
    """Collect ordered open-access PDF URLs from an OpenAlex work record (repositories first)."""
    from research_assistant.shared.fetch import candidate_pdf_urls

    repo_urls: List[str] = []
    pub_urls: List[str] = []
    landing_rewrites: List[str] = []

    def _add(lst: List[str], url: Optional[str]):
        for cand in candidate_pdf_urls(url):
            if cand not in repo_urls and cand not in pub_urls and cand not in landing_rewrites:
                lst.append(cand)

    all_locs: List[Dict[str, Any]] = []
    if isinstance(work.get("best_oa_location"), dict):
        all_locs.append(work["best_oa_location"])
    for loc in work.get("locations") or []:
        if isinstance(loc, dict):
            all_locs.append(loc)
    if isinstance(work.get("primary_location"), dict):
        all_locs.append(work["primary_location"])

    for loc in all_locs:
        src = loc.get("source") or {}
        is_repo = (src.get("type") == "repository") or (loc.get("version") in ("submittedVersion", "acceptedVersion"))
        pdf_u = loc.get("pdf_url")
        if pdf_u:
            _add(repo_urls if is_repo else pub_urls, pdf_u)
        land_u = loc.get("landing_page_url")
        if land_u:
            expanded = candidate_pdf_urls(land_u)
            for c in expanded:
                if c != land_u or c.lower().endswith(".pdf"):
                    _add(landing_rewrites, c)

    oa_info = work.get("open_access") or {}
    if oa_info.get("oa_url"):
        _add(pub_urls, oa_info["oa_url"])

    ids = work.get("ids") or {}
    pmcid_raw = str(ids.get("pmcid") or "")
    m_pmc = re.search(r"(PMC\d+)", pmcid_raw, re.IGNORECASE)
    if m_pmc:
        _add(repo_urls, f"https://www.ebi.ac.uk/europepmc/webservices/rest/{m_pmc.group(1).upper()}/fullTextPdf")

    return repo_urls + landing_rewrites + pub_urls


def fetch_bibliometrics_and_abstract(target: str, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """
    Fetch bibliometrics, open-access PDF URLs, identifiers, and abstract from OpenAlex and Semantic Scholar.
    """
    from research_assistant.shared.fetch import candidate_pdf_urls
    from research_assistant.shared.source_key import extract_arxiv_from_text, normalise_doi

    res: Dict[str, Any] = {
        "title": None,
        "abstract": None,
        "is_retracted": False,
        "citation_count": None,
        "venue": None,
        "year": None,
        "doi": None,
        "arxiv_id": None,
        "pmcid": None,
        "pmid": None,
        "oa_pdf_url": None,
        "oa_pdf_urls": [],
    }
    if not target or not str(target).strip():
        return res

    clean_t = str(target).strip()
    if clean_t.startswith(("title:", "raw:", "url:")):
        return res

    is_id_lookup = clean_t.startswith("10.") or "doi.org" in clean_t.lower() or clean_t.lower().startswith("arxiv:")
    oa_urls: List[str] = []

    def _push_url(u: Optional[str]):
        for cand in candidate_pdf_urls(u):
            if cand and cand not in oa_urls:
                oa_urls.append(cand)

    # 1. Try OpenAlex
    try:
        oa_query = clean_t
        if clean_t.lower().startswith("arxiv:"):
            aid_clean = re.sub(r"v\d+$", "", clean_t[6:].strip())
            oa_query = f"10.48550/arxiv.{aid_clean}"
        oa_data = call_openalex_api(oa_query, timeout=timeout)
        if oa_data:
            work = None
            if "results" in oa_data and isinstance(oa_data["results"], list):
                for cand_w in oa_data["results"][:5]:
                    if not isinstance(cand_w, dict):
                        continue
                    if is_id_lookup or verify_candidate_match(clean_t, str(cand_w.get("title") or "")):
                        work = cand_w
                        break
            elif isinstance(oa_data, dict) and ("title" in oa_data or "id" in oa_data):
                work = oa_data

            if isinstance(work, dict) and ("title" in work or "id" in work):
                res["title"] = work.get("title")
                res["abstract"] = reconstruct_openalex_abstract(work.get("abstract_inverted_index"))
                res["is_retracted"] = bool(work.get("is_retracted", False))
                res["citation_count"] = work.get("cited_by_count")
                res["year"] = work.get("publication_year")
                loc = work.get("primary_location") or {}
                source = loc.get("source") or {}
                res["venue"] = source.get("display_name")
                res["doi"] = normalise_doi(work.get("doi")) or work.get("doi")

                ids = work.get("ids") or {}
                if ids.get("pmcid"):
                    m_pmc = re.search(r"(PMC\d+)", str(ids["pmcid"]), re.IGNORECASE)
                    if m_pmc:
                        res["pmcid"] = m_pmc.group(1).upper()
                if ids.get("pmid"):
                    m_pm = re.search(r"(\d+)", str(ids["pmid"]))
                    if m_pm:
                        res["pmid"] = m_pm.group(1)
                for loc_item in (work.get("locations") or []):
                    if isinstance(loc_item, dict) and not res["arxiv_id"]:
                        res["arxiv_id"] = extract_arxiv_from_text(loc_item.get("landing_page_url") or loc_item.get("pdf_url"))
                if not res["arxiv_id"]:
                    res["arxiv_id"] = extract_arxiv_from_text(str(res["doi"] or ""))

                for u in _collect_openalex_oa_urls(work):
                    _push_url(u)
    except Exception as exc:
        logger.debug("Error fetching from OpenAlex for %s: %s", clean_t[:40], exc)

    # 2. Query Semantic Scholar when abstract, citation count, or direct OA PDF URL is missing
    if not res["abstract"] or res["citation_count"] is None or not oa_urls:
        try:
            s2_target = normalise_doi(res["doi"]) or clean_t
            s2_data = call_semantic_scholar_api(s2_target, timeout=timeout)
            if s2_data:
                paper = None
                if "data" in s2_data and isinstance(s2_data["data"], list):
                    for cand_p in s2_data["data"][:5]:
                        if not isinstance(cand_p, dict):
                            continue
                        if is_id_lookup or verify_candidate_match(clean_t, str(cand_p.get("title") or "")):
                            paper = cand_p
                            break
                elif isinstance(s2_data, dict) and ("title" in s2_data or "paperId" in s2_data):
                    paper = s2_data

                if isinstance(paper, dict):
                    if not res["abstract"] and paper.get("abstract"):
                        res["abstract"] = paper.get("abstract")
                    if res["citation_count"] is None and paper.get("citationCount") is not None:
                        res["citation_count"] = paper.get("citationCount")
                    if not res["title"] and paper.get("title"):
                        res["title"] = paper.get("title")
                    if not res["year"] and paper.get("year"):
                        res["year"] = paper.get("year")
                    ext_ids = paper.get("externalIds") or {}
                    if isinstance(ext_ids, dict):
                        if not res["doi"] and ext_ids.get("DOI"):
                            res["doi"] = normalise_doi(ext_ids["DOI"])
                        if not res["arxiv_id"] and ext_ids.get("ArXiv"):
                            res["arxiv_id"] = str(ext_ids["ArXiv"]).strip().lower()
                            _push_url(f"https://arxiv.org/pdf/{res['arxiv_id']}.pdf")
                        if not res["pmcid"] and ext_ids.get("PubMedCentral"):
                            pmc_val = str(ext_ids["PubMedCentral"]).strip().upper()
                            if not pmc_val.startswith("PMC"):
                                pmc_val = f"PMC{pmc_val}"
                            res["pmcid"] = pmc_val
                            _push_url(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc_val}/fullTextPdf")
                        if not res["pmid"] and ext_ids.get("PubMed"):
                            res["pmid"] = str(ext_ids["PubMed"]).strip()
                    oa_pdf = paper.get("openAccessPdf")
                    if isinstance(oa_pdf, dict) and oa_pdf.get("url"):
                        _push_url(oa_pdf["url"])
        except Exception as exc:
            logger.debug("Error fetching from Semantic Scholar for %s: %s", clean_t[:40], exc)

    # 3. Fallback to Crossref work metadata if abstract is still missing for a known DOI
    effective_doi = normalise_doi(res["doi"]) or (normalise_doi(clean_t) if is_id_lookup and not clean_t.lower().startswith("arxiv:") else None)
    if not res["abstract"] and effective_doi:
        try:
            _rate_limit("crossref_biblio", 0.1)
            cr_url = f"https://api.crossref.org/works/{urllib.parse.quote(effective_doi, safe='/')}"
            cr_resp = requests.get(cr_url, headers=DEFAULT_HEADERS, timeout=timeout)
            if cr_resp.status_code == 200:
                cr_msg = (cr_resp.json() or {}).get("message") or {}
                raw_abs = cr_msg.get("abstract")
                if isinstance(raw_abs, str) and raw_abs.strip():
                    cleaned_abs = re.sub(r"<[^>]+>", " ", html.unescape(raw_abs))
                    cleaned_abs = re.sub(r"^\s*abstract[\s:.\-]*", "", cleaned_abs, flags=re.IGNORECASE)
                    cleaned_abs = " ".join(cleaned_abs.split())
                    if len(cleaned_abs) >= 40:
                        res["abstract"] = cleaned_abs
                        res["doi"] = res["doi"] or effective_doi
        except Exception as exc:
            logger.debug("Crossref abstract fallback failed for %s: %s", effective_doi[:40], exc)

    if res["arxiv_id"]:
        _push_url(f"https://arxiv.org/pdf/{res['arxiv_id']}.pdf")
    if res["pmcid"]:
        _push_url(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{res['pmcid']}/fullTextPdf")

    res["oa_pdf_urls"] = oa_urls
    res["oa_pdf_url"] = oa_urls[0] if oa_urls else None
    return res

