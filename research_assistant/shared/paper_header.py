"""Extract paper title, first-author surnames, DOI and arXiv ID via GROBID and Docling."""

from dataclasses import dataclass
import json
import logging
import os
import re

from bs4 import BeautifulSoup
import requests

from research_assistant import config
from research_assistant.shared import docling_extract
from research_assistant.shared.paper_index import normalise_arxiv, surname_key
from research_assistant.shared.source_key import fold_name, fold_title, normalise_doi

logger = logging.getLogger(__name__)

_GENERIC_SECTION_HEADERS = frozenset({
    "abstract",
    "introduction",
    "contents",
    "article",
    "research article",
    "original article",
    "review",
    "letter",
})

_VENUE_WORDS = re.compile(
    r"\b(journal|proceedings|transactions|bulletin|magazine|repository|portal)\b",
    re.IGNORECASE,
)

_FORBIDDEN_AUTHOR_WORDS = re.compile(
    r"\b(university|department|institute|laboratory|school|college|centre|center|faculty|abstract|doi|"
    r"received|accepted|published|article|journal|vol|volume|citations|reads|keywords|email|"
    r"correspondence|copyright)\b",
    re.IGNORECASE,
)

_AUTHOR_DELIMITERS = re.compile(r"[,;·&]|\band\b", re.IGNORECASE)


@dataclass(frozen=True)
class PaperHeader:
    title: str | None
    first_authors: tuple[str, ...]   # first-author surnames, GROBID's first; () when none
    doi: str | None                  # normalise_doi() form
    arxiv_id: str | None             # normalise_arxiv() form


def header_from_grobid_tei(tei_xml: str) -> PaperHeader:
    """Parse GROBID processHeaderDocument TEI XML into a PaperHeader."""
    soup = BeautifulSoup(tei_xml, "xml")

    # Title: teiHeader fileDesc titleStmt title
    title_tag = soup.select_one("teiHeader fileDesc titleStmt title")
    title = title_tag.get_text(strip=True) if title_tag else None
    if not title:
        title = None

    # Surname: first sourceDesc biblStruct analytic author persName surname
    surname_tag = soup.select_one("sourceDesc biblStruct analytic author persName surname")
    surname = surname_tag.get_text(strip=True) if surname_tag else None
    first_authors = (surname,) if surname else ()

    # DOI: idno[type="DOI"]
    doi_tag = soup.select_one('idno[type="DOI"]')
    doi_val = doi_tag.get_text(strip=True) if doi_tag else None
    doi = normalise_doi(doi_val)

    # arXiv: idno[type="arXiv"]
    arxiv_tag = soup.select_one('idno[type="arXiv"]')
    arxiv_val = arxiv_tag.get_text(strip=True) if arxiv_tag else None
    arxiv_id = normalise_arxiv(arxiv_val)

    return PaperHeader(
        title=title,
        first_authors=first_authors,
        doi=doi,
        arxiv_id=arxiv_id,
    )


def grobid_header(
    pdf_path: str,
    server: str = config.GROBID_SERVER,
    timeout: float = 60,
) -> PaperHeader | None:
    """Call GROBID's processHeaderDocument API to extract header metadata."""
    url = f"{server.rstrip('/')}/api/processHeaderDocument"
    try:
        with open(pdf_path, "rb") as f:
            files = {"input": (os.path.basename(pdf_path), f, "application/pdf")}
            resp = requests.post(
                url,
                files=files,
                data={"consolidateHeader": "0"},
                headers={"Accept": "application/xml"},
                timeout=timeout,
            )
        if resp.status_code != 200 or not resp.text or not resp.text.strip():
            return None
        return header_from_grobid_tei(resp.text)
    except Exception as exc:
        logger.debug("GROBID header extraction failed for %s: %s", pdf_path, exc)
        return None


def header_from_docling(doc: dict) -> PaperHeader:
    """Extract header metadata from a cached DoclingDocument JSON."""
    p1_items = [
        it for it in doc.get("texts", [])
        if it.get("prov") and it["prov"][0].get("page_no") == 1
    ]

    # --- Title ---
    title = None
    title_idx = None

    for i, it in enumerate(p1_items):
        if it.get("label") == "title":
            t = it.get("text", "").strip()
            if t:
                title = t
                title_idx = i
                break

    if title is None:
        for i, it in enumerate(p1_items):
            if it.get("label") == "section_header":
                text = it.get("text", "").strip()
                words = text.split()
                if len(words) < 4:
                    continue
                folded_t = fold_name(text)
                if folded_t in _GENERIC_SECTION_HEADERS:
                    continue
                if _VENUE_WORDS.search(text):
                    continue
                title = text
                title_idx = i
                break

    # --- First author ---
    first_authors: tuple[str, ...] = ()
    if title_idx is not None:
        subsequent_texts = [
            it for it in p1_items[title_idx + 1:]
            if it.get("label") == "text"
        ][:12]

        for it in subsequent_texts:
            raw_text = it.get("text", "")
            # Remove * ∗ † ‡ § ¶ and digits
            cleaned = re.sub(r"[*∗†‡§¶0-9]", "", raw_text)
            lowered = cleaned.lower()
            if "@" in lowered or "http" in lowered or "www." in lowered:
                continue
            if _FORBIDDEN_AUTHOR_WORDS.search(cleaned):
                continue

            # First name segment before delimiter
            segment = _AUTHOR_DELIMITERS.split(cleaned)[0].strip()
            tokens = segment.split()
            if not (2 <= len(tokens) <= 4):
                continue
            if not all(tok[0].isupper() for tok in tokens):
                continue

            # Surname: last token with leading and trailing non-letters removed
            last_token = tokens[-1]
            surname = re.sub(r"^[\W\d_]+|[\W\d_]+$", "", last_token)
            if sum(c.isalpha() for c in surname) < 2:
                # Bare initial, no surname
                first_authors = ()
                break
            first_authors = (surname,)
            break

    # --- DOI and arXiv from page 1 text ---
    p1_text = "\n".join(it.get("text", "") for it in p1_items)
    doi = None
    doi_match = re.search(r"\b10\.\d{4,9}/[^\s\"<>]+", p1_text)
    if doi_match:
        raw_doi = doi_match.group(0).rstrip(".,;)")
        doi = normalise_doi(raw_doi)

    arxiv_id = None
    arxiv_match = re.search(r"arXiv:\s*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", p1_text, re.IGNORECASE)
    if arxiv_match:
        raw_arxiv = arxiv_match.group(1)
        arxiv_id = normalise_arxiv(raw_arxiv)

    return PaperHeader(
        title=title,
        first_authors=first_authors,
        doi=doi,
        arxiv_id=arxiv_id,
    )


def docling_header(pdf_path: str, convert_if_missing: bool = False) -> PaperHeader | None:
    """Extract header using Docling cache, or optionally convert pages 1–2."""
    cache_path = docling_extract._docling_cache_path(pdf_path)
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
            if isinstance(doc, dict):
                return header_from_docling(doc)
            return None
        except Exception as exc:
            logger.debug("Failed to read Docling cache %s: %s", cache_path, exc)
            return None

    if convert_if_missing:
        try:
            converter = docling_extract.get_digital_converter()
            res = converter.convert(pdf_path, page_range=(1, 2))
            doc = res.document.model_dump(mode="json")
            return header_from_docling(doc)
        except Exception as exc:
            logger.debug("Docling conversion failed for %s: %s", pdf_path, exc)
            return None

    return None


def read_header(pdf_path: str, convert_if_missing: bool = False) -> PaperHeader | None:
    """Read header from GROBID, falling back to / merging with Docling."""
    g = grobid_header(pdf_path)
    d = docling_header(pdf_path, convert_if_missing=convert_if_missing)
    if not g and not d:
        return None

    g_title = g.title if g else None
    d_title = d.title if d else None
    title = g_title or d_title
    if not title:
        return None

    g_authors = g.first_authors if g else ()
    d_authors = d.first_authors if d else ()
    merged_authors: list[str] = []
    seen_keys: set[str] = set()

    for author in (*g_authors, *d_authors):
        k = surname_key(author)
        if k and k not in seen_keys:
            seen_keys.add(k)
            merged_authors.append(author)

    if not merged_authors:
        return None

    doi = (g.doi if g and g.doi else None) or (d.doi if d and d.doi else None)
    arxiv_id = (g.arxiv_id if g and g.arxiv_id else None) or (d.arxiv_id if d and d.arxiv_id else None)

    return PaperHeader(
        title=title,
        first_authors=tuple(merged_authors),
        doi=doi,
        arxiv_id=arxiv_id,
    )
