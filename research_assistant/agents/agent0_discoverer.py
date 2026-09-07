"""
Agent 0 — Seed paper discoverer.

Turns a free-text research idea into the first PDF the rest of the pipeline
needs. It searches a scholarly index (OpenAlex, then Semantic Scholar — see
SEARCH_PROVIDERS), whose results come back ranked by relevance, picks the top
result that actually has an open-access PDF, downloads it into RAW_DIR under a
source_key-derived name, and records the choice in seed_papers.json.

Nothing downstream knows Agent 0 exists: it just leaves a PDF in RAW_DIR, which
is exactly what Agent 1 already reads. The returned path also lets an
orchestrator hand the seed straight to shared.ingestion.ingest_pdfs() so the
seed paper itself is indexed, not only the papers it cites.

    python agent0_discoverer.py --query "topological protection in disordered wires"
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import time

import requests

from research_assistant.config import (
    RAW_DIR,
    SEED_PAPERS_PATH,
    SEARCH_PROVIDERS,
    SEARCH_LIMIT,
    OPENALEX_URL,
    OPENALEX_MAILTO,
    S2_API_KEY,
    S2_SEARCH_URL,
)
from research_assistant.schemas import SeedPaper
from research_assistant.shared.fetch import HEADERS, download_pdf, filename_for, _is_pdf
from research_assistant.shared.log import get_logger
from research_assistant.shared.source_key import source_key, normalise_doi

logger = get_logger("agent0")


# ─── Provider searches ───────────────────────────────────────────────────────
# Each returns a list of normalised candidate dicts, most relevant first:
#   {title, doi, arxiv_id, pmid, authors: [str], year, pdf_url}
# pdf_url may be None; pick_best() skips those.


def _search_openalex(query, limit):
    res = requests.get(
        OPENALEX_URL,
        params={"search": query, "per_page": limit, "mailto": OPENALEX_MAILTO},
        headers=HEADERS,
        timeout=20,
    )
    res.raise_for_status()
    out = []
    for w in res.json().get("results") or []:
        loc = w.get("best_oa_location") or w.get("primary_location") or {}
        ids = w.get("ids") or {}
        out.append({
            "title": w.get("display_name") or w.get("title"),
            "doi": w.get("doi"),
            "arxiv_id": None,
            "pmid": (ids.get("pmid") or "").rsplit("/", 1)[-1] or None,
            "authors": [
                a["author"]["display_name"]
                for a in w.get("authorships") or []
                if a.get("author", {}).get("display_name")
            ],
            "year": w.get("publication_year"),
            "pdf_url": loc.get("pdf_url") or (w.get("open_access") or {}).get("oa_url"),
        })
    return out


def _search_semanticscholar(query, limit, retries=4):
    headers = dict(HEADERS)
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY

    fields = "title,year,authors,externalIds,openAccessPdf"
    for attempt in range(retries):
        res = requests.get(
            S2_SEARCH_URL,
            params={"query": query, "limit": limit, "fields": fields},
            headers=headers,
            timeout=20,
        )
        # The keyless pool 429s constantly; back off on those, but not on any
        # other HTTP error.
        if res.status_code == 429 and attempt < retries - 1:
            wait = float(res.headers.get("Retry-After", 2 ** (attempt + 1)))
            logger.warning("Semantic Scholar rate limited; retrying in %.0fs…", wait)
            time.sleep(wait)
            continue
        res.raise_for_status()
        out = []
        for p in res.json().get("data") or []:
            ext = p.get("externalIds") or {}
            out.append({
                "title": p.get("title"),
                "doi": ext.get("DOI"),
                "arxiv_id": ext.get("ArXiv"),
                "pmid": str(ext["PubMed"]) if ext.get("PubMed") else None,
                "authors": [a["name"] for a in p.get("authors") or [] if a.get("name")],
                "year": p.get("year"),
                "pdf_url": (p.get("openAccessPdf") or {}).get("url"),
            })
        return out
    return []


def _search_arxiv(query, limit):
    import xml.etree.ElementTree as ET

    res = requests.get(
        "http://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "max_results": limit,
                "sortBy": "relevance"},
        headers=HEADERS,
        timeout=20,
    )
    res.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(res.content)

    out = []
    for e in root.findall("a:entry", ns):
        raw_id = (e.findtext("a:id", "", ns) or "").rsplit("/abs/", 1)[-1]
        arxiv_id = raw_id.split("v")[0] or None

        pdf = None
        for link in e.findall("a:link", ns):
            if link.get("title") == "pdf":
                pdf = link.get("href", "").replace("http://", "https://")
        if pdf and not pdf.endswith(".pdf"):
            pdf += ".pdf"

        out.append({
            "title": " ".join((e.findtext("a:title", "", ns) or "").split()),
            "doi": None,
            "arxiv_id": arxiv_id,
            "pmid": None,
            "authors": [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)],
            "year": (e.findtext("a:published", "", ns) or "")[:4] or None,
            "pdf_url": pdf,
        })
    return out


_PROVIDERS = {
    "openalex": _search_openalex,
    "arxiv": _search_arxiv,
    "semanticscholar": _search_semanticscholar,
}


# ─── Selection ───────────────────────────────────────────────────────────────


def find_and_fetch_seed(query, force=False, limit=SEARCH_LIMIT):
    """Walk providers × their ranked candidates until a PDF *actually downloads*.

    A candidate whose ``pdf_url`` 404s or serves an HTML paywall page (common
    for publisher links surfaced by OpenAlex) is skipped, and the search moves
    on to the next candidate and then the next provider — so a physics query
    that OpenAlex only has paywalled still gets picked up from arXiv.

    Returns ``(candidate, key, dest_path)`` or ``(None, None, None)``.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    seen = set()

    for name in SEARCH_PROVIDERS:
        pname = name.strip()
        provider = _PROVIDERS.get(pname)
        if not provider:
            logger.warning("Unknown search provider %r — skipping.", pname)
            continue
        try:
            results = provider(query, limit)
        except requests.RequestException as exc:
            logger.warning("%s search failed: %s", pname, exc)
            continue

        n_links = 0
        for cand in results:
            if not cand.get("pdf_url"):
                continue
            key = source_key(cand)
            if not key or key in seen:
                continue
            seen.add(key)
            n_links += 1

            dest = os.path.join(RAW_DIR, filename_for(key))
            if os.path.exists(dest) and not force:
                logger.info("%s: [%s] already on disk", pname, key)
                return cand, key, dest

            ok, reason = download_pdf(cand["pdf_url"], dest)
            if ok:
                logger.info("%s: [%s] %s", pname, key, os.path.basename(dest))
                return cand, key, dest
            logger.info("%s: [%s] not downloadable — %s", pname, key, reason)

        logger.info(
            "%s: %d result(s), %d with a PDF link, none downloadable.",
            pname, len(results), n_links,
        )
    return None, None, None


# ─── Manifest ────────────────────────────────────────────────────────────────


def _load_seeds():
    if not os.path.exists(SEED_PAPERS_PATH):
        return {}
    try:
        with open(SEED_PAPERS_PATH, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.warning("%s is unreadable; starting fresh.", SEED_PAPERS_PATH)
        return {}


def _save_seeds(seeds):
    with open(SEED_PAPERS_PATH, "w") as f:
        json.dump(seeds, f, indent=2, ensure_ascii=False)


def get_seed(query):
    """Return Agent 0's manifest record for *query* (key, title, path, …) or None."""
    return _load_seeds().get(query)


def get_seed_by_path(path):
    """Return (record, query) for a seed record with matching path, or (None, None).
    Iterates in reverse so that the most recent record for a path is returned."""
    if not path:
        return None, None
    norm = os.path.abspath(path)
    for q, rec in reversed(list(_load_seeds().items())):
        p = rec.get("path")
        if p and os.path.abspath(p) == norm:
            return rec, q
    return None, None


# ─── Direct-URL fallback ─────────────────────────────────────────────────────

_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?|[a-z-]+/[0-9]{7})")


def _from_arxiv_url(url):
    """(pdf_url, arxiv_id) for an arxiv.org link, else (url, None)."""
    m = _ARXIV_RE.search(url or "")
    if not m:
        return url, None
    arxiv_id = m.group(1)
    return f"https://arxiv.org/pdf/{arxiv_id}", arxiv_id


def _download_and_record(query, key, pdf_url, seeds, force, arxiv_id=None, source="manual-url"):
    """Download *pdf_url* into RAW_DIR, write the seed manifest, return the path."""
    os.makedirs(RAW_DIR, exist_ok=True)
    dest = os.path.join(RAW_DIR, filename_for(key))

    if os.path.exists(dest) and not force:
        logger.info("PDF already on disk: %s", os.path.basename(dest))
    else:
        ok, reason = download_pdf(pdf_url, dest)
        if not ok:
            logger.error("Could not download %s: %s", pdf_url, reason)
            return None
        logger.info("Saved -> %s", os.path.basename(dest))

    seeds[query] = SeedPaper(
        key=key,
        url=pdf_url,
        path=dest,
        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        arxiv_id=arxiv_id,
        source=source,
    ).to_dict()
    _save_seeds(seeds)
    return dest


def discover_from_url(query, url, force=False):
    """Seed from a PDF link the user supplied — the fallback when the search
    turns up nothing open-access. arXiv abstract links are rewritten to the PDF.

    Returns the local PDF path, or None if the download failed.
    """
    seeds = _load_seeds()
    prior = seeds.get(query)
    if prior and not force and os.path.exists(prior.get("path", "")):
        logger.info("Already seeded for this query: %s", prior["path"])
        return prior["path"]

    pdf_url, arxiv_id = _from_arxiv_url(url.strip())
    key = f"arxiv:{arxiv_id}" if arxiv_id else (source_key({"url": pdf_url}) or "url:manual")
    logger.info("Seeding from supplied link [%s]: %s", key, pdf_url)
    return _download_and_record(
        query, key, pdf_url, seeds, force, arxiv_id=arxiv_id, source="manual-url"
    )


# ─── Direct-file fallback ────────────────────────────────────────────────────

_DOI_RE = re.compile(rb"(?i)(?:doi[:\s/]|https?://doi\.org/|doi\.org\\/)(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)")
_ARXIV_RE_BYTES = re.compile(rb"(?i)arxiv[:\s/]+([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)")
_PDF_TITLE_RE = re.compile(rb"/Title\s*(?:\((.*?)\)|<([0-9a-fA-F]+)>)", re.DOTALL)
_XMP_TITLE_RE = re.compile(rb"<dc:title>[\s\S]*?<rdf:li[^>]*>([\s\S]*?)</rdf:li>", re.IGNORECASE)
_XMP_TITLE_FALLBACK_RE = re.compile(rb"<dc:title>([\s\S]*?)</dc:title>", re.IGNORECASE)
_XMP_DOI_RE = re.compile(rb"<(?:prism:doi|dc:identifier)>\s*(?:doi:)?\s*(10\.\d{4,9}/[^<\s]+)\s*</", re.IGNORECASE)

_HEADER_JUNK_RE = re.compile(
    r"(?i)^(?:arxiv[:\s/]|https?://|doi[:\s/]|10\.\d{4,9}/|"
    r"journal of|physical review|nature\b|science\b|proceedings of|"
    r"ieee\b|springer|elsevier|volume\b|vol\.\b|issue\b|no\.\b|pp\.\b|pages\b|"
    r"published|accepted|received|issn\b|typeset|draft version)",
)


def _decode_pdf_title(raw_str: bytes | str) -> str | None:
    """Decode PDF title string from literal bytes/str or hex string."""
    if isinstance(raw_str, str):
        raw_str = raw_str.encode("utf-8", "ignore")
    if not raw_str:
        return None
    if raw_str.startswith(b"\xfe\xff"):
        try:
            return raw_str.decode("utf-16-be").strip()
        except Exception:
            pass
    elif raw_str.startswith(b"\xff\xfe"):
        try:
            return raw_str.decode("utf-16-le").strip()
        except Exception:
            pass
    decoded = (
        raw_str.replace(b"\\(", b"(")
        .replace(b"\\)", b")")
        .replace(b"\\\\", b"\\")
        .replace(b"\\r", b" ")
        .replace(b"\\n", b" ")
        .replace(b"\\t", b" ")
    )
    try:
        text = decoded.decode("utf-8", "ignore").strip()
    except Exception:
        text = decoded.decode("latin-1", "ignore").strip()
    return re.sub(r"\s+", " ", text) if text else None


def _clean_title(title: str | None) -> str | None:
    if not title:
        return None
    t = title.strip()
    t = (
        t.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > 3 and t.lower() not in ("untitled", "default", "none", "nan", "untitled document"):
        return t
    return None


def _clean_doi(raw_doi: str | None) -> str | None:
    if not raw_doi:
        return None
    d = raw_doi.strip().rstrip(".,;)>]").strip()
    m = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", d)
    if m:
        d = m.group(0).rstrip(".,;)>]")
        return normalise_doi(d) or d.lower()
    return None


def _is_pdf_content(first_chunk: bytes) -> bool:
    """PDF header (%PDF-) anywhere in first 1024 bytes per ISO 32000-1 §7.5.2."""
    return b"%PDF-" in first_chunk[:1024]


def _inspect_pdf(file_path: str, raw_bytes: bytes | None = None) -> tuple[str | None, str | None, str | None]:
    """Extract (title, doi, arxiv_id) from PDF metadata or text if available."""
    title, doi, arxiv_id = None, None, None

    # 1. Try PyMuPDF if installed
    doc = None
    try:
        import fitz
        if file_path and os.path.exists(file_path):
            doc = fitz.open(file_path)
        elif raw_bytes:
            doc = fitz.open(stream=raw_bytes, filetype="pdf")

        if doc:
            meta = doc.metadata or {}
            t = _clean_title(meta.get("title"))
            if t:
                title = t

            for page_num in range(min(2, len(doc))):
                text = doc[page_num].get_text("text")
                if not arxiv_id:
                    m_arx = re.search(r"(?i)arxiv[:\s/]+([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)", text)
                    if m_arx:
                        arxiv_id = m_arx.group(1)
                if not doi:
                    m_doi = re.search(r"(?i)(?:doi[:\s/]|https?://doi\.org/)(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", text)
                    if m_doi:
                        doi = _clean_doi(m_doi.group(1))
                if not title and page_num == 0:
                    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 3]
                    for candidate_line in lines[:10]:
                        if _HEADER_JUNK_RE.search(candidate_line):
                            continue
                        cand_title = _clean_title(candidate_line)
                        if cand_title and len(cand_title) > 5:
                            title = cand_title
                            break
    except Exception:
        pass
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass

    # 2. Fallback regex on raw bytes
    if raw_bytes is None and file_path and os.path.exists(file_path):
        try:
            size = os.path.getsize(file_path)
            with open(file_path, "rb") as f:
                if size <= 1024 * 1024:
                    raw_bytes = f.read()
                else:
                    head = f.read(512 * 1024)
                    f.seek(max(0, size - 512 * 1024))
                    tail = f.read(512 * 1024)
                    raw_bytes = head + b"\n" + tail
        except OSError:
            raw_bytes = b""
    elif raw_bytes is None:
        raw_bytes = b""

    # Look for arXiv ID
    if not arxiv_id and raw_bytes:
        m_arx = _ARXIV_RE_BYTES.search(raw_bytes[:128 * 1024])
        if m_arx:
            try:
                arxiv_id = m_arx.group(1).decode("utf-8", "ignore")
            except Exception:
                pass

    # Look for DOI
    if not doi and raw_bytes:
        m_xmp_doi = _XMP_DOI_RE.search(raw_bytes)
        if m_xmp_doi:
            try:
                doi = _clean_doi(m_xmp_doi.group(1).decode("utf-8", "ignore"))
            except Exception:
                pass
        if not doi:
            m_doi = _DOI_RE.search(raw_bytes[:128 * 1024])
            if m_doi:
                try:
                    doi = _clean_doi(m_doi.group(1).decode("utf-8", "ignore"))
                except Exception:
                    pass

    # Look for Title
    if not title and raw_bytes:
        m_xmp = _XMP_TITLE_RE.search(raw_bytes) or _XMP_TITLE_FALLBACK_RE.search(raw_bytes)
        if m_xmp:
            try:
                cand = _clean_title(m_xmp.group(1).decode("utf-8", "ignore"))
                if cand:
                    title = cand
            except Exception:
                pass

        if not title:
            m_title = _PDF_TITLE_RE.search(raw_bytes)
            if m_title:
                raw_t = m_title.group(1)
                hex_t = m_title.group(2)
                if hex_t:
                    try:
                        raw_bytes_t = bytes.fromhex(hex_t.decode("ascii"))
                        cand = _decode_pdf_title(raw_bytes_t)
                        title = _clean_title(cand)
                    except Exception:
                        pass
                elif raw_t:
                    cand = _decode_pdf_title(raw_t)
                    title = _clean_title(cand)

    return title, doi, arxiv_id


def discover_from_file(
    query: str = "",
    file_input: str | bytes | os.PathLike = "",
    filename: str | None = None,
    force: bool = False,
) -> str | None:
    """Seed directly from an uploaded or local PDF file.

    Validates that the file is a PDF, extracts or infers metadata (title, DOI, arXiv ID),
    computes a deterministic key, copies/writes it to RAW_DIR, and records the seed paper.
    If *query* is empty, the paper's title or cleaned filename is used as the query.

    Returns the local path in RAW_DIR, or None if the file is invalid or cannot be read.
    """
    os.makedirs(RAW_DIR, exist_ok=True)

    if isinstance(file_input, (str, os.PathLike)):
        file_path = str(file_input)
        if not os.path.exists(file_path):
            logger.error("Seed file does not exist: %s", file_path)
            return None
        filename = filename or os.path.basename(file_path)
        try:
            with open(file_path, "rb") as f:
                content = f.read()
        except OSError as exc:
            logger.error("Could not read seed file %s: %s", file_path, exc)
            return None
    elif hasattr(file_input, "read"):
        filename = filename or getattr(file_input, "name", "uploaded_paper.pdf")
        content = file_input.read()
        file_path = None
    elif isinstance(file_input, (bytes, bytearray)):
        content = bytes(file_input)
        filename = filename or "uploaded_paper.pdf"
        file_path = None
    else:
        logger.error("Unsupported file input type: %s", type(file_input))
        return None

    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    if not _is_pdf_content(content[:1024]):
        logger.error("File is not a valid PDF (header does not contain %%PDF-): %s", filename)
        return None

    stem = os.path.splitext(os.path.basename(filename))[0]
    clean_stem = re.sub(r"[_\-]+", " ", stem).strip()
    title, doi, arxiv_id = _inspect_pdf(file_path or "", raw_bytes=content)

    if arxiv_id:
        key = f"arxiv:{arxiv_id}"
    elif doi:
        key = f"doi:{normalise_doi(doi) or doi.lower()}"
    else:
        digest = hashlib.sha1(content).hexdigest()[:16]
        key = f"file:{digest}"

    dest = os.path.join(RAW_DIR, filename_for(key))

    if os.path.exists(dest) and not force:
        logger.info("PDF already on disk in RAW_DIR: %s", os.path.basename(dest))
    else:
        tmp_path = dest + ".part"
        try:
            with open(tmp_path, "wb") as f:
                f.write(content)
            os.replace(tmp_path, dest)
        except OSError as exc:
            logger.error("Failed writing seed PDF to %s: %s", dest, exc)
            return None
        logger.info("Saved seed PDF -> %s", os.path.basename(dest))

    seeds = _load_seeds()
    effective_query = (query or "").strip() or title or clean_stem or "Uploaded seed paper"
    seeds[effective_query] = SeedPaper(
        key=key,
        path=dest,
        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source="upload",
        title=title or clean_stem,
        doi=doi,
        arxiv_id=arxiv_id,
    ).to_dict()
    _save_seeds(seeds)

    return dest


# ─── Entry point ─────────────────────────────────────────────────────────────


def discover(query, force=False):
    """Find the top relevant open-access paper for *query* and pull it to RAW_DIR.

    Returns the local PDF path, or None when nothing usable was found.
    """
    seeds = _load_seeds()
    prior = seeds.get(query)
    if prior and not force and os.path.exists(prior.get("path", "")):
        logger.info("Already seeded for this query: %s", prior["path"])
        return prior["path"]

    logger.info("Searching for: %s", query)
    cand, key, dest = find_and_fetch_seed(query, force=force)
    if not dest:
        logger.error("No downloadable open-access PDF for query: %s", query)
        return None

    seeds[query] = SeedPaper(
        key=key,
        path=dest,
        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source="search",
        title=cand.get("title"),
        url=cand["pdf_url"],
        doi=cand.get("doi"),
        arxiv_id=cand.get("arxiv_id"),
    ).to_dict()
    _save_seeds(seeds)
    return dest


def main():
    parser = argparse.ArgumentParser(description="Agent 0 — Seed paper discoverer")
    parser.add_argument("--query", default="", help="Research idea to seed from.")
    parser.add_argument(
        "--url",
        help="Seed directly from this PDF / arXiv link instead of searching.",
    )
    parser.add_argument(
        "--file",
        help="Seed directly from a local PDF file instead of searching.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-search and re-download even if this query was already seeded.",
    )
    args = parser.parse_args()

    if args.file:
        path = discover_from_file(args.query, args.file, force=args.force)
    elif args.url:
        path = discover_from_url(args.query, args.url, force=args.force)
    else:
        if not args.query:
            parser.error("Must provide --query, --file, or --url")
        path = discover(args.query, force=args.force)
    if not path:
        raise SystemExit(1)
    print(path)


if __name__ == "__main__":
    main()
