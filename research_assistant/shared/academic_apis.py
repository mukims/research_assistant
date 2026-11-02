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
        if len(tokens) > 1 and tokens[-1].lower().rstrip(".") in {"jr", "sr", "ii", "iii", "iv", "v"}:
            surname = tokens[-2]
        else:
            surname = tokens[-1]
    return re.sub(r"[^\w]", "", surname).lower()


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


def verify_candidate_match(
    target_title: str,
    candidate_title: str,
    target_author: Optional[str] = None,
    candidate_author: Optional[str] = None,
    target_year: Optional[int] = None,
    candidate_year: Optional[int] = None,
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

    t1 = target_title.strip().lower()
    t2 = candidate_title.strip().lower()

    # 1. Title verification
    tokens1 = set(re.findall(r"[a-z0-9]+", t1))
    tokens2 = set(re.findall(r"[a-z0-9]+", t2))
    if not tokens1 or not tokens2:
        return False

    jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2)

    import difflib
    ratio = difflib.SequenceMatcher(None, t1, t2).ratio()

    # Conjunctive requirement: both metrics must meet thresholds
    if jaccard < 0.70 or ratio < 0.80:
        return False

    # Check for differing numbers or Roman numerals (e.g. Part I vs Part II, Vol 1 vs Vol 2)
    diff1 = tokens1 - tokens2
    diff2 = tokens2 - tokens1

    digits1 = {t for t in diff1 if t.isdigit()}
    digits2 = {t for t in diff2 if t.isdigit()}
    if digits1 and digits2 and digits1 != digits2:
        return False

    roman1 = {t for t in diff1 if t in ROMAN_NUMERALS}
    roman2 = {t for t in diff2 if t in ROMAN_NUMERALS}
    if roman1 and roman2 and roman1 != roman2:
        return False

    layer_mods = {"monolayer", "bilayer", "trilayer", "multilayer", "fewlayer"}
    if (diff1 & layer_mods) and (diff2 & layer_mods):
        return False

    # 2. Author verification
    if target_author and candidate_author:
        if not _match_surnames(target_author, candidate_author):
            return False

    # 3. Year verification
    if target_year is not None and candidate_year is not None:
        try:
            if abs(int(target_year) - int(candidate_year)) > 1:
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
    fields: str = "title,abstract,authors,year,citationCount,isOpenAccess,openAccessPdf",
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[Dict[str, Any]]:
    """Queries Semantic Scholar Academic Graph API."""
    _rate_limit("semanticscholar", 0.5)
    clean_target = query_or_id.strip()
    headers = dict(DEFAULT_HEADERS)

    if clean_target.startswith("10."):
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{clean_target}?fields={fields}"
    elif clean_target.lower().startswith("arxiv:"):
        url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{clean_target[6:]}?fields={fields}"
    else:
        encoded = urllib.parse.quote(clean_target)
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded}&limit=3&fields={fields}"

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


def fetch_bibliometrics_and_abstract(target: str, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """
    Fetch bibliometrics and abstract from OpenAlex and Semantic Scholar.
    Returns:
        dict: {
            "title": str | None,
            "abstract": str | None,
            "is_retracted": bool,
            "citation_count": int | None,
            "venue": str | None,
            "year": int | None,
            "doi": str | None,
            "oa_pdf_url": str | None,
        }
    """
    res = {
        "title": None,
        "abstract": None,
        "is_retracted": False,
        "citation_count": None,
        "venue": None,
        "year": None,
        "doi": None,
        "oa_pdf_url": None,
    }

    # 1. Try OpenAlex
    try:
        oa_data = call_openalex_api(target, timeout=timeout)
        if oa_data:
            work = oa_data
            if "results" in oa_data and isinstance(oa_data["results"], list) and oa_data["results"]:
                work = oa_data["results"][0]
            if isinstance(work, dict) and ("title" in work or "id" in work):
                res["title"] = work.get("title")
                res["abstract"] = reconstruct_openalex_abstract(work.get("abstract_inverted_index"))
                res["is_retracted"] = bool(work.get("is_retracted", False))
                res["citation_count"] = work.get("cited_by_count")
                res["year"] = work.get("publication_year")
                loc = work.get("primary_location") or {}
                source = loc.get("source") or {}
                res["venue"] = source.get("display_name")
                res["doi"] = work.get("doi")
                oa_info = work.get("open_access") or {}
                res["oa_pdf_url"] = oa_info.get("oa_url")
    except Exception as exc:
        logger.debug("Error fetching from OpenAlex for %s: %s", target[:40], exc)

    # 2. Fall back to Semantic Scholar if abstract or citation count missing
    if not res["abstract"] or res["citation_count"] is None:
        try:
            s2_data = call_semantic_scholar_api(target, timeout=timeout)
            if s2_data:
                paper = s2_data
                if "data" in s2_data and isinstance(s2_data["data"], list) and s2_data["data"]:
                    paper = s2_data["data"][0]
                if isinstance(paper, dict):
                    if not res["abstract"] and paper.get("abstract"):
                        res["abstract"] = paper.get("abstract")
                    if res["citation_count"] is None and paper.get("citationCount") is not None:
                        res["citation_count"] = paper.get("citationCount")
                    if not res["title"] and paper.get("title"):
                        res["title"] = paper.get("title")
                    if not res["year"] and paper.get("year"):
                        res["year"] = paper.get("year")
                    oa_pdf = paper.get("openAccessPdf")
                    if not res["oa_pdf_url"] and isinstance(oa_pdf, dict) and oa_pdf.get("url"):
                        res["oa_pdf_url"] = oa_pdf.get("url")
        except Exception as exc:
            logger.debug("Error fetching from Semantic Scholar for %s: %s", target[:40], exc)

    return res

