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
)
from research_assistant.shared.log import get_logger

logger = get_logger("academic_apis")

DEFAULT_TIMEOUT = 15
DEFAULT_HEADERS = {
    "User-Agent": f"research_assistant/0.2 (mailto:{UNPAYWALL_EMAIL}; https://github.com/google-deepmind/marvin)"
}

_LAST_CALL_TIMESTAMPS: Dict[str, float] = {}


def _rate_limit(api_name: str, min_interval_seconds: float) -> None:
    """Enforces minimum interval between calls to an API domain."""
    now = time.time()
    last = _LAST_CALL_TIMESTAMPS.get(api_name, 0.0)
    elapsed = now - last
    if elapsed < min_interval_seconds:
        time.sleep(min_interval_seconds - elapsed)
    _LAST_CALL_TIMESTAMPS[api_name] = time.time()


def call_crossref_api(
    query: str,
    rows: int = 3,
    mailto: str = UNPAYWALL_EMAIL,
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[Dict[str, Any]]:
    """Queries CrossRef REST API for works matching a query string."""
    _rate_limit("crossref", 0.1)
    params = {
        "query.bibliographic": query,
        "rows": rows,
        "mailto": mailto,
    }
    url = "https://api.crossref.org/works"
    headers = dict(DEFAULT_HEADERS)
    headers["User-Agent"] = f"research_assistant/0.2 (mailto:{mailto})"

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
    mailto: str = UNPAYWALL_EMAIL,
    timeout: int = DEFAULT_TIMEOUT,
) -> Optional[Dict[str, Any]]:
    """Queries OpenAlex API by DOI or search string."""
    _rate_limit("openalex", 0.1)
    clean_target = query_or_doi.strip()
    headers = dict(DEFAULT_HEADERS)
    headers["User-Agent"] = f"research_assistant/0.2 (mailto:{mailto})"

    if clean_target.startswith("10."):
        url = f"https://api.openalex.org/works/https://doi.org/{clean_target}?mailto={mailto}"
    elif "doi.org" in clean_target:
        url = f"https://api.openalex.org/works/{clean_target}?mailto={mailto}"
    else:
        encoded = urllib.parse.quote(clean_target)
        url = f"https://api.openalex.org/works?search={encoded}&mailto={mailto}"

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


def fetch_open_access_pdf_url(doi: str, mailto: str = UNPAYWALL_EMAIL, timeout: int = DEFAULT_TIMEOUT) -> Optional[str]:
    """Queries Unpaywall for direct open-access PDF download URL for a DOI."""
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

