"""
Agent 1 — reference extractor.

GROBID primary, regex fallback, per PDF. tech_ireland's GROBID + TEI parse
produces structured references — title, authors, year, DOI, and a confidence
score for each consolidated DOI against the printed reference — but it needs
a running Java service; when that service was down the old pipeline
dead-ended. citation_builder's pdftotext + reference-numbering regex needs
nothing but a system binary, at the cost of returning only a raw string per
reference with no structured fields.

Merged: try GROBID first, fall back to regex per PDF (not per run), so one
paper GROBID could not parse is still covered by the weaker path instead of
being silently dropped, and a completely dead GROBID server degrades the
whole run instead of stopping it.

Provides:
    grobid_alive()                    — probe the server once per run
    extract_references()              — grobid → regex → none, per PDF
    run_extractor()                   — batch entry point for the orchestrator
    parse_tei_file() / parse_reference() / parse_article_metadata()
                                       — TEI parsing, used by the grobid path
    summarise()                       — counts worth logging after a run
    _unique_destination()             — collision-safe filing into
                                         raw/processed/ and raw/failed/
"""

import copy
import glob
import json
import os
import re
import shutil
import subprocess
from difflib import SequenceMatcher
from functools import lru_cache

import requests

from research_assistant.config import (
    EXTRACTED_CITATIONS_PATH,
    GROBID_BATCH_CONCURRENCY,
    GROBID_SERVER,
    PAPER_INDEX_PATH,
    RAW_DIR,
)
from research_assistant.schemas import Reference
from research_assistant.shared.atomic import atomic_write_json
from research_assistant.shared.log import get_logger
from research_assistant.shared.paper_index import edit
from research_assistant.shared import pipeline_status

logger = get_logger("agent1")

# GROBID's TEI output. Kept alongside the PDFs so a run can be inspected
# and re-parsed without hitting the server again.
XML_OUTPUT_DIR = os.path.join(RAW_DIR, "grobid_output")

TEI_SUFFIXES = (
    ".references.tei.xml",
    ".fulltext.tei.xml",
    ".grobid.tei.xml",
    ".tei.xml",
)

# Reference types that rarely have a real DOI. A consolidated DOI on one of
# these is a likely false match rather than a find.
GREY_MARKERS = (
    "available online",
    "accessed on",
    "http://",
    "https://",
    "www.",
    "technical report",
    "white paper",
    "thesis",
    "dissertation",
    "standard",
    "patent",
    "datasheet",
)


# ─── GROBID batch + TEI parsing ──────────────────────────────────────────────
# Ported from tech_ireland/agent1_extractor.py, imports rewritten only.


def run_grobid_batch(pdf_dir: str, output_dir: str) -> None:
    """Send every PDF in pdf_dir to GROBID at GROBID_SERVER, once, as a batch.

    grobid_client is imported here, not at module scope, so this module still
    imports — and grobid_alive()/extract_references()/the regex fallback all
    still work — with the package absent. Same reasoning as bs4 being
    imported inside parse_tei_file below rather than at the top of the file:
    neither is installed in every environment this module needs to import in.
    """
    from grobid_client.grobid_client import GrobidClient

    logger.info("Sending PDFs in %s to GROBID at %s…", pdf_dir, GROBID_SERVER)
    # check_server=False: a serverless GROBID (Cloud Run/HF Space) may be
    # cold-starting; the platform holds the request while it boots, and any
    # real failure still surfaces per-file in the batch below.
    client = GrobidClient(grobid_server=GROBID_SERVER, check_server=False)

    # processFulltextDocument returns header, body and bibliography, and tags
    # in-text citation markers with the reference they point to.
    client.process(
        service="processFulltextDocument",
        input_path=pdf_dir,
        output=output_dir,
        n=GROBID_BATCH_CONCURRENCY,
        consolidate_citations=0,
        consolidate_header=True,
        include_raw_citations=True,
        force=False,
    )
    logger.info("GROBID batch complete.")


extract_tei = run_grobid_batch


def _clean(node):
    """Collapse GROBID's line wrapping into single-spaced text."""
    if node is None:
        return None
    text = " ".join(node.text.split())
    return text or None


def _stem(path):
    name = os.path.basename(path)
    for suffix in TEI_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def _person_name(pers):
    parts = [_clean(f) for f in pers.find_all("forename")]
    parts.append(_clean(pers.find("surname")))
    name = " ".join(p for p in parts if p)
    return name or None


def _normalise(text):
    """Lowercase alphanumeric tokens, for loose title comparison."""
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _doi_confidence(title, raw_reference):
    """
    Rate how well a consolidated title matches the reference as printed.

    GROBID's consolidation can match grey literature to an unrelated DOI and
    overwrite the title with the matched record. Comparing the returned title
    against the raw string catches most of those.

    Returns one of: "high", "medium", "low", "unknown".
    """
    if not raw_reference:
        return "unknown"

    raw_lower = raw_reference.lower()
    is_grey = any(marker in raw_lower for marker in GREY_MARKERS)

    if not title:
        return "low" if is_grey else "unknown"

    title_tokens = _normalise(title)
    raw_tokens = _normalise(raw_reference)
    if not title_tokens:
        return "unknown"

    overlap = len(title_tokens & raw_tokens) / len(title_tokens)
    ratio = SequenceMatcher(None, title.lower(), raw_lower).ratio()

    if overlap >= 0.8:
        return "medium" if is_grey else "high"
    if overlap >= 0.5 or ratio >= 0.4:
        return "medium"
    return "low"


def parse_reference(bibl, source_file=None):
    """Parse a single <biblStruct> from a TEI reference list."""
    analytic = bibl.find("analytic")
    monogr = bibl.find("monogr")

    # Article title lives in <analytic>; for books and reports the title is
    # in <monogr> instead, so fall back rather than returning None.
    title = None
    if analytic:
        title = _clean(analytic.find("title", type="main")) or _clean(analytic.find("title"))
    if not title and monogr:
        title = _clean(monogr.find("title", level="m")) or _clean(monogr.find("title"))

    container = None
    if monogr:
        journal = monogr.find("title", level="j")
        container = _clean(journal) if journal else None
        if container == title:
            container = None

    doi_node = bibl.find("idno", type="DOI")
    doi = doi_node.text.strip().lower() if doi_node and doi_node.text else None

    authors = []
    scope = analytic or bibl
    for author in scope.find_all("author"):
        pers = author.find("persName")
        if pers:
            name = _person_name(pers)
            if name:
                authors.append(name)

    year = None
    date = bibl.find("date", type="published")
    if date:
        when = date.get("when") or _clean(date) or ""
        match = re.search(r"(1[89]\d{2}|20\d{2})", when)
        if match:
            year = int(match.group(1))

    raw_node = bibl.find("note", type="raw_reference")
    raw_reference = _clean(raw_node)

    return {
        "source_file": source_file,
        "xml_id": bibl.get("xml:id") or bibl.get("{http://www.w3.org/XML/1998/namespace}id") or bibl.get("id"),
        "title": title,
        "container": container,
        "authors": authors,
        "year": year,
        "doi": doi,
        "doi_confidence": _doi_confidence(title, raw_reference) if doi else None,
        "raw_reference": raw_reference,
    }


def parse_article_metadata(soup, source_file=None):
    """Parse the citing paper's own title, authors and DOI from the TEI header."""
    header = soup.find("teiHeader")
    analytic = None
    if header:
        source_desc = header.find("sourceDesc")
        if source_desc:
            bibl = source_desc.find("biblStruct")
            if bibl:
                analytic = bibl.find("analytic")

    if analytic is None:
        return {"source_file": source_file, "title": None, "doi": None, "authors": []}

    doi_node = analytic.find("idno", type="DOI")
    authors = []
    for author in analytic.find_all("author"):
        pers = author.find("persName")
        if pers:
            name = _person_name(pers)
            if name:
                authors.append(name)

    return {
        "source_file": source_file,
        "title": _clean(analytic.find("title", type="main")) or _clean(analytic.find("title")),
        "doi": doi_node.text.strip().lower() if doi_node and doi_node.text else None,
        "authors": authors,
    }


def parse_tei_file(tei_file_path):
    """Parse one TEI file into its article metadata and its reference list.

    bs4 is imported here, not at module scope — see run_grobid_batch's
    docstring; the same reasoning applies.
    """
    from bs4 import BeautifulSoup

    with open(tei_file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "xml")

    source_file = f"{_stem(tei_file_path)}.pdf"
    article = parse_article_metadata(soup, source_file=source_file)

    list_bibl = soup.find("listBibl")
    references = []
    if list_bibl is not None:
        references = [
            parse_reference(bibl, source_file=source_file)
            for bibl in list_bibl.find_all("biblStruct")
        ]

    return article, references


@lru_cache(maxsize=None)
def _parse_tei_file_cached(tei_file_path: str):
    """parse_tei_file(), memoized per path.

    _references_from_grobid and _article_from_grobid both read the same TEI
    file for any PDF that took the GROBID path; gating Finding 1's fix on
    grobid_ok (rather than method) makes _article_from_grobid run for every
    GROBID'd PDF, not just the ones whose references also came from GROBID —
    so the duplicate parse this avoids is now more frequent, not less.

    lru_cache hands back the *same* (article, references) tuple to every
    caller, and both are built from mutable dicts/lists — so this function
    is never called directly; go through _parse_tei_file() below, which
    deep-copies before returning, so one caller mutating its copy can never
    affect another.
    """
    return parse_tei_file(tei_file_path)


def _parse_tei_file(tei_file_path: str):
    """Cached parse_tei_file(), safe for callers to mutate their own copy."""
    article, references = _parse_tei_file_cached(tei_file_path)
    return copy.deepcopy(article), copy.deepcopy(references)


def summarise(references):
    """Counts worth logging after a run. Takes dicts (Reference.to_dict())."""
    total = len(references)
    with_doi = [r for r in references if r["doi"]]
    suspect = [r for r in with_doi if r["doi_confidence"] in ("low", "unknown")]
    return {
        "references": total,
        "with_doi": len(with_doi),
        "doi_coverage": round(len(with_doi) / total, 3) if total else 0.0,
        "suspect_doi": len(suspect),
        "suspect_ids": [(r["source_file"], r["xml_id"]) for r in suspect],
    }


def _tei_path_for(pdf_path: str) -> str:
    """The cached TEI path for pdf_path, under XML_OUTPUT_DIR.

    Falls back to the first TEI_SUFFIXES candidate (which will not exist)
    when GROBID has not processed this PDF, so callers can check with a
    single os.path.exists rather than handling None separately.
    """
    stem = _stem(pdf_path)
    for suffix in TEI_SUFFIXES:
        candidate = os.path.join(XML_OUTPUT_DIR, stem + suffix)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(XML_OUTPUT_DIR, stem + TEI_SUFFIXES[0])


def _references_from_grobid(pdf_path: str) -> list[Reference]:
    """Read and parse this PDF's cached TEI, if GROBID already produced one.

    run_extractor() calls run_grobid_batch() once, up front, for every PDF in
    the run — that is the only thing that talks to the server, with
    GROBID_BATCH_CONCURRENCY's worth of parallelism. This function never
    uploads anything itself; it only reads whatever run_grobid_batch already
    wrote to XML_OUTPUT_DIR. Calling GROBID once per PDF from here would lose
    that batch concurrency and be materially slower against a shared/public
    server.
    """
    tei_path = _tei_path_for(pdf_path)
    if not os.path.exists(tei_path):
        return []

    try:
        _article, raw_references = _parse_tei_file(tei_path)
    except Exception as exc:  # noqa: BLE001 — malformed TEI means fall back
        logger.warning("Could not parse TEI for %s: %s", pdf_path, exc)
        return []

    references = []
    for raw in raw_references:
        if not (raw.get("raw_reference") or raw.get("title")):
            continue
        references.append(
            Reference(
                raw_reference=raw.get("raw_reference") or raw.get("title") or "",
                source_file=raw.get("source_file") or os.path.basename(pdf_path),
                title=raw.get("title"),
                container=raw.get("container"),
                authors=tuple(raw.get("authors") or ()),
                year=raw.get("year"),
                doi=raw.get("doi"),
                doi_confidence=raw.get("doi_confidence"),
                xml_id=raw.get("xml_id"),
                extraction_method="grobid",
            )
        )
    return references


def _article_from_grobid(pdf_path: str) -> dict | None:
    """The citing paper's own metadata, when GROBID has a cached TEI for it.

    A PDF that fell back to the regex path never reaches here and so
    contributes no article record — expected, since pdftotext + regex
    recovers only the reference list, never the TEI header.
    """
    tei_path = _tei_path_for(pdf_path)
    if not os.path.exists(tei_path):
        return None
    try:
        article, _references = _parse_tei_file(tei_path)
    except Exception as exc:  # noqa: BLE001 — malformed TEI means skip it
        logger.warning("Could not parse article metadata for %s: %s", pdf_path, exc)
        return None
    return article


# ─── Regex fallback ──────────────────────────────────────────────────────────
# Weaker than GROBID — a raw string per reference, no structured fields —
# but it needs no server, which is the whole point: an unreachable GROBID used to
# end the pipeline.
#
# A naive line-by-line regex without section or sequence tracking fails
# spectacularly: equations like (1) and floats/axis labels like 0.0 or 7.99
# get classified as citations, turning entire bodies of prose into hundreds of
# fake references.
#
# Robust extraction:
# 1. Detect explicit "References" / "Bibliography" section headings if present.
# 2. Extract strictly ascending sequential citation runs ([1], [2]... or 1., 2....)
#    to ignore isolated inline mentions, equation numbers, and floats.
# 3. Stop before appendices / supplementary information blocks.

REF_HEADING_RE = re.compile(
    r"^(?:[0-9IVX]+\.?\s*)?(?:references|bibliography|literature cited|references and notes)\s*$",
    re.IGNORECASE,
)
SECTION_STOP_RE = re.compile(
    r"^(?:SI-[IVX0-9]+|Supplementary|Supporting Information|Extended Data|Acknowledgements|Author Contributions|Appendix|Appendices)\b",
    re.IGNORECASE,
)
CITATION_PATTERNS = [
    re.compile(r"^\[(\d{1,4})\](?:\s*(.*))?$"),       # [1] Author...
    re.compile(r"^(\d{1,4})\.(?:\s+(.*))?$"),         # 1. Author...
]


def _match_citation_line(line):
    """Return (id, rest_of_line) for a reference-list line, else None."""
    for pattern in CITATION_PATTERNS:
        match = pattern.match(line)
        if match:
            return int(match.group(1)), match.group(2)
    return None


def _pymupdf_text(pdf_path: str) -> str:
    """Raw text via PyMuPDF, a hard dependency, for hosts without poppler."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return ""
    try:
        with fitz.open(pdf_path) as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as exc:  # noqa: BLE001 — a corrupt PDF just yields no text
        logger.warning("PyMuPDF could not read %s: %s", pdf_path, exc)
        return ""


def _pdftotext(pdf_path: str) -> str:
    """Extract raw text with poppler's pdftotext, else PyMuPDF. "" if both fail.

    The regex fallback exists for when GROBID is down; it must not also depend
    on a binary most machines (this project's Mac included) do not have.
    """
    try:
        result = subprocess.run(
            ["pdftotext", pdf_path, "-"],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("pdftotext unavailable for %s (%s); using PyMuPDF", pdf_path, exc)
        return _pymupdf_text(pdf_path)
    if result.returncode != 0:
        logger.warning("pdftotext failed on %s: %s; using PyMuPDF", pdf_path, result.stderr.strip())
        return _pymupdf_text(pdf_path)
    return result.stdout


def _references_from_regex(pdf_path: str) -> list[Reference]:
    """Reference strings recovered by numbering patterns and sequential context."""
    text = _pdftotext(pdf_path)
    if not text.strip():
        return []

    source_file = os.path.basename(pdf_path)
    lines = [l.strip() for l in text.splitlines()]

    # 1. Search for a section heading
    heading_idx = None
    for i, line in enumerate(lines):
        if REF_HEADING_RE.match(line):
            heading_idx = i
            break

    search_lines = lines[heading_idx + 1:] if heading_idx is not None else lines
    best_cits: list[tuple[int, str]] = []

    for pat in CITATION_PATTERNS:
        matches = []
        for i, line in enumerate(search_lines):
            m = pat.match(line)
            if m:
                rest = m.group(2) or ""
                # If dot-pattern with empty rest (e.g. "1." on its own line),
                # ensure the subsequent line starts with an uppercase author/title letter
                if pat == CITATION_PATTERNS[1] and not rest:
                    if i + 1 < len(search_lines) and not re.match(r"^[A-Z]", search_lines[i + 1]):
                        continue
                matches.append((int(m.group(1)), i, rest, line))

        # Look for sequential ascending runs starting at 1 (or 2/3 in case 1 was missed)
        for start_idx, (num, line_i, rest, _) in enumerate(matches):
            if num in (1, 2, 3):
                seq = [(num, line_i, rest)]
                expected = num + 1
                prev_line_i = line_i
                for next_num, next_line_i, next_rest, _ in matches[start_idx + 1:]:
                    # References in a bibliography are close together (e.g. within 30 lines)
                    if next_line_i - prev_line_i > 30:
                        break
                    if next_num == expected:
                        seq.append((next_num, next_line_i, next_rest))
                        expected += 1
                        prev_line_i = next_line_i
                    elif next_num == expected + 1 and expected > 2:
                        seq.append((next_num, next_line_i, next_rest))
                        expected = next_num + 1
                        prev_line_i = next_line_i

                if len(seq) > len(best_cits):
                    cits = []
                    for idx in range(len(seq)):
                        n, start_l, r = seq[idx]
                        end_l = (
                            seq[idx + 1][1]
                            if idx + 1 < len(seq)
                            else min(len(search_lines), start_l + 15)
                        )
                        collected = [r] if r else []
                        for l_idx in range(start_l + 1, end_l):
                            l = search_lines[l_idx]
                            if not l:
                                continue
                            if SECTION_STOP_RE.match(l):
                                break
                            collected.append(l)
                        full_txt = " ".join(collected).strip()
                        if len(full_txt) >= 10:
                            cits.append((n, full_txt))
                    if len(cits) > len(best_cits):
                        best_cits = cits

    seen = set()
    references: list[Reference] = []
    for _, raw_ref in best_cits:
        if raw_ref not in seen:
            seen.add(raw_ref)
            references.append(
                Reference(
                    raw_reference=raw_ref,
                    source_file=source_file,
                    extraction_method="regex",
                )
            )

    return references


def _unique_destination(directory: str, filename: str) -> str:
    """Return a path in *directory* for *filename* that overwrites nothing.

    Two source papers can share a basename, and a plain shutil.move would
    replace the earlier file without a word. Ported from citation_builder's
    agent1_extractor.py, which files PDFs into processed/failed the same way
    this module does.
    """
    stem, ext = os.path.splitext(filename)
    candidate = os.path.join(directory, filename)
    n = 2
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{stem}_{n}{ext}")
        n += 1
    return candidate


# ─── Strategy selection ──────────────────────────────────────────────────────


def _file_away(pdf_path: str, directory: str, name: str) -> None:
    """Move a finished PDF out of RAW_DIR, tolerating a concurrent filer.

    run_extractor() globs the whole of RAW_DIR, so two pipelines — app.py and
    watch.py, or two orchestrator runs — can be filing the same directory at
    once. Losing that race is not a reason to abandon the batch: the references
    are already extracted by this point, and an uncaught error here would
    discard every other PDF's results along with this one's.
    """
    os.makedirs(directory, exist_ok=True)
    destination = _unique_destination(directory, name)
    try:
        shutil.move(pdf_path, destination)
    except (FileNotFoundError, shutil.Error) as exc:
        logger.warning(
            "%s could not be filed under %s (%s) — another run likely moved it "
            "first. Its references were still extracted.",
            name, os.path.basename(directory), exc,
        )
        return

    try:
        with edit(PAPER_INDEX_PATH) as idx:
            idx.repoint_path(pdf_path, destination)
    except Exception as exc:
        logger.warning(
            "Could not repoint paper index path for %s -> %s: %s",
            pdf_path, destination, exc,
        )


def grobid_alive() -> bool:
    """Probe the GROBID server once per run."""
    try:
        response = requests.get(f"{GROBID_SERVER}/api/isalive", timeout=10)
    except requests.RequestException as exc:
        logger.warning("GROBID at %s is unreachable (%s).", GROBID_SERVER, exc)
        return False
    return response.ok and "true" in response.text.lower()


def extract_references(pdf_path: str, grobid_ok: bool) -> tuple[list[Reference], str]:
    """Return (references, method) for one PDF.

    method is "grobid", "regex" or "none". GROBID is preferred whenever it is
    up and productive; the regex path covers both a dead server and a PDF that
    GROBID parsed without finding a reference list.
    """
    if grobid_ok:
        references = _references_from_grobid(pdf_path)
        if references:
            return references, "grobid"
        logger.info(
            "GROBID found no references in %s — falling back to pattern matching.",
            os.path.basename(pdf_path),
        )

    references = _references_from_regex(pdf_path)
    if references:
        return references, "regex"
    return [], "none"


def extract_citations_from_pdf(
    pdf_path: str, *, grobid_ok: bool | None = None
) -> list[dict]:
    """Extract references for a single PDF non-destructively.

    Does not move or modify the source PDF file on disk.
    Returns a list of reference dictionaries.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if grobid_ok is None:
        grobid_ok = grobid_alive()
    refs, _method = extract_references(pdf_path, grobid_ok)
    return [r.to_dict() for r in refs]


def run_extractor() -> dict:
    """Extract references from every PDF in RAW_DIR.

    GROBID is tried first via a single batch call (run_grobid_batch), so the
    whole run gets GROBID_BATCH_CONCURRENCY's worth of parallelism against the
    server rather than one request per PDF. Each PDF then falls back to the
    regex path individually — a dead server, or one PDF GROBID could not
    parse, degrades only that PDF instead of ending the run.

    Files each source PDF by outcome so a paper is never silently swallowed:
    raw/processed/ when references were read, raw/failed/ when neither
    strategy produced any.
    """
    os.makedirs(XML_OUTPUT_DIR, exist_ok=True)

    standalone = not pipeline_status.get_status().get("active")
    with pipeline_status.track_stage(
        "extract",
        "Extracting reference list",
        current_step=3,
        total_steps=5,
        detail="Extracting reference list from papers",
        mark_idle_on_exit=standalone,
        last_summary="Extracted references",
    ):
        pdfs = sorted(glob.glob(os.path.join(RAW_DIR, "*.pdf")))
        if not pdfs:
            logger.info("No PDFs found in %s.", RAW_DIR)
            pipeline_status.update_progress(item_current=0, item_total=0, detail="No PDFs found in raw directory")
            return {"reference_count": 0, "processed": 0, "failed": 0}

        pipeline_status.update_progress(
            item_current=0,
            item_total=len(pdfs),
            detail=f"Checking GROBID server for {len(pdfs)} PDF(s)…",
        )
        grobid_ok = grobid_alive()
        if grobid_ok:
            pipeline_status.add_event(f"📖 Querying GROBID server for reference extraction ({len(pdfs)} PDF(s))…")
            pipeline_status.update_progress(detail="Querying GROBID batch endpoint…")
            try:
                run_grobid_batch(RAW_DIR, XML_OUTPUT_DIR)
            except Exception as exc:  # noqa: BLE001 — any client failure means fall back
                logger.warning(
                    "GROBID batch failed (%s) — falling back to pattern-based "
                    "extraction for all %d PDF(s).", exc, len(pdfs),
                )
                grobid_ok = False
                pipeline_status.add_event("⚠️ GROBID batch failed, falling back to regex extraction")
        else:
            logger.warning(
                "GROBID at %s is not responding — falling back to pattern-based "
                "extraction for all %d PDF(s). References will lack DOIs, authors "
                "and years; Agent 2 will resolve what it can via Crossref.",
                GROBID_SERVER, len(pdfs),
            )
            pipeline_status.add_event("⚠️ GROBID offline, using pattern-based reference extraction")

        processed_dir = os.path.join(RAW_DIR, "processed")
        failed_dir = os.path.join(RAW_DIR, "failed")
        os.makedirs(processed_dir, exist_ok=True)

        articles, all_references = [], []
        by_method = {"grobid": 0, "regex": 0, "none": 0}

        for i, pdf in enumerate(pdfs, 1):
            name = os.path.basename(pdf)
            pipeline_status.update_progress(
                item_current=i,
                item_total=len(pdfs),
                current_item_name=name,
                detail=f"Extracting references [{i}/{len(pdfs)}]: {name}",
            )
            references, method = extract_references(pdf, grobid_ok)
            by_method[method] += 1

            if grobid_ok:
                # method describes only where the *references* came from. GROBID
                # may have processed this PDF fine (TEI header, title, authors,
                # DOI all present) while finding no reference list, in which case
                # extract_references reports "regex" or "none" here — but the
                # citing paper's own metadata is still sitting on disk and must
                # not be dropped just because this PDF's method wasn't "grobid".
                # _article_from_grobid() itself checks os.path.exists(tei_path),
                # so this is a no-op — not a wasted lookup — when GROBID never
                # produced a TEI for this PDF at all.
                article = _article_from_grobid(pdf)
                if article:
                    articles.append(article)

            if references:
                all_references.extend(references)
                logger.info("%s: %d reference(s) via %s.", name, len(references), method)
                _file_away(pdf, processed_dir, name)
            else:
                # Nothing came out: a scan with no text layer, an unrecognised
                # reference format, or no reference list at all. Filing it under
                # processed/ would claim a success it did not have.
                logger.warning("%s: no references extracted — filing under failed/.", name)
                _file_away(pdf, failed_dir, name)

        reference_dicts = [r.to_dict() for r in all_references]

        # Merge with existing extracted citations to prevent wiping earlier seed papers
        merged_refs = list(reference_dicts)
        merged_articles = list(articles)
        if os.path.exists(EXTRACTED_CITATIONS_PATH):
            try:
                with open(EXTRACTED_CITATIONS_PATH, "r", encoding="utf-8") as f:
                    old_payload = json.load(f)
                    seen_ref_keys = {
                        (r.get("source_file"), r.get("raw_reference"), r.get("doi"))
                        for r in reference_dicts
                    }
                    for old_ref in old_payload.get("references") or []:
                        k = (old_ref.get("source_file"), old_ref.get("raw_reference"), old_ref.get("doi"))
                        if k not in seen_ref_keys:
                            merged_refs.append(old_ref)
                            seen_ref_keys.add(k)
                    seen_art_dois = {a.get("doi") for a in articles if a.get("doi")}
                    for old_art in old_payload.get("articles") or []:
                        if old_art.get("doi") not in seen_art_dois:
                            merged_articles.append(old_art)
            except Exception as e:
                logger.warning("Could not read existing extracted citations to merge: %s", e)

        payload = {
            "articles": merged_articles,
            "references": merged_refs,
            "summary": summarise(merged_refs),
            "extraction": by_method,
        }
        # Atomic: Agent 2 reads this back as its work list, and a truncated write
        # would strand the whole reference chain behind a JSONDecodeError.
        atomic_write_json(EXTRACTED_CITATIONS_PATH, payload, ensure_ascii=False)

        pipeline_status.add_event(
            f"📖 Extracted {len(all_references)} references ({by_method['grobid']} via GROBID, {by_method['regex']} via regex)"
        )
        pipeline_status.update_progress(
            item_current=len(pdfs),
            item_total=len(pdfs),
            detail=f"Extracted {len(all_references)} references across {len(pdfs)} paper(s)",
        )

        logger.info(
            "Wrote %d reference(s) from %d PDF(s) (%d via GROBID, %d via patterns, "
            "%d yielded nothing) to %s.",
            len(all_references), len(pdfs), by_method["grobid"], by_method["regex"],
            by_method["none"], EXTRACTED_CITATIONS_PATH,
        )
        return {
            "reference_count": len(all_references),
            "processed": by_method["grobid"] + by_method["regex"],
            "failed": by_method["none"],
        }



if __name__ == "__main__":
    run_extractor()
