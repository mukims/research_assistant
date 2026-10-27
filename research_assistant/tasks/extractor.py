"""
Scientific Protocol Miner: extracts operational protocols and synthesizes Task Graphs from papers.
"""

import glob
import os
import re
from typing import Any, Dict, Optional

from bs4 import BeautifulSoup

from research_assistant import config
from research_assistant.shared.llm import chat
from research_assistant.shared.log import get_logger
from research_assistant.shared.seed_audit import find_tei_for_seed
from research_assistant.tasks.models import (
    AgentRole,
    ArtifactContract,
    ArtifactType,
    Task,
    TaskCategory,
    TaskGraph,
)

logger = get_logger("task_extractor")

METHOD_KEYWORDS = (
    "method", "model", "algorithm", "architecture", "experiment",
    "setup", "implementation", "dataset", "benchmark", "evaluation",
    "simulation", "results", "analysis", "theory", "approach",
    "framework", "design", "protocol", "pipeline", "workflow",
    "hamiltonian", "disorder", "spectrum", "spectral", "correlation",
    "hyperuniform", "potential", "inverse", "determination", "conductance",
    "transport", "lattice", "solver", "greens", "green", "formalism",
    "formulation", "derivation", "calculation", "computation", "tight-binding",
    "optimization", "numerical",
)

ROMAN_NUMS = (
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
)
ROMAN_PAT = "|".join(ROMAN_NUMS)

AFFILIATION_RE = re.compile(
    r"(?i)(?:@|email|university|universidade|department|departamento|institute|instituto|"
    r"school|college|laboratory|laborat|center|centre|author|correspond|received|"
    r"accepted|published|arxiv|doi|license|copyright|springer|elsevier|wiley|ieee)"
)

SECTION_PREFIX_RE = re.compile(
    rf"^(?:(?:\d+(?:\.\d+)*\.?|(?:{ROMAN_PAT})\.?)[\.\s]+)[A-Z]"
)


def _is_binary_content(data: bytes) -> bool:
    """Returns True if the byte slice appears to be binary rather than plain text."""
    if not data:
        return False
    if data.startswith(b"%PDF-"):
        return True
    if data.startswith((b"\xff\xfe", b"\xfe\xff", b"\xef\xbb\xbf", b"\x00\x00\xfe\xff")):
        return False
    if b"\x00" in data:
        return True
    try:
        decoded = data[:4096].decode("utf-8")
        non_printable = sum(1 for c in decoded if ord(c) < 32 and c not in "\n\r\t")
        return (non_printable / max(1, len(decoded))) > 0.05
    except UnicodeDecodeError:
        return True


def _extract_from_tei_xml(xml_content: str, fallback_title: str = "Scientific Paper") -> Dict[str, Any]:
    """Extracts title, abstract, methodology sections, and tables/figures from TEI XML."""
    if not xml_content or not isinstance(xml_content, (str, bytes)):
        return {
            "title": fallback_title or "Scientific Paper",
            "abstract": "",
            "sections": [],
            "tables": [],
        }

    soup = BeautifulSoup(xml_content, "xml")

    # 1. Title
    title_tag = soup.find("title", type="main") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else fallback_title

    # 2. Abstract
    abstract_tag = soup.find("abstract")
    abstract = abstract_tag.get_text(separator=" ", strip=True) if abstract_tag else ""

    # 3. Focus on Methodological and Experimental Sections (exclude teiHeader/abstract divs)
    sections = []
    body = soup.find("body") or soup.find("text") or soup
    divs = [
        d for d in body.find_all("div")
        if not d.find_parent("abstract") and not d.find_parent("teiHeader")
    ]
    top_divs = [d for d in divs if d.parent == body or (d.parent and d.parent.name in ("body", "text"))]
    candidate_divs = top_divs if top_divs else divs

    non_method_re = re.compile(
        r"(?i)\b(acknowledg|funding|competing interest|conflict of interest|author contribution|data availability|code availability)\b"
    )

    for div in candidate_divs:
        head = div.find("head")
        head_text = head.get_text(strip=True) if head else ""
        text = div.get_text(separator=" ", strip=True)
        if not text or len(text) < 50:
            continue

        if non_method_re.search(head_text):
            continue

        lower_head = head_text.lower()
        lower_body = text[:600].lower()

        if any(k in lower_head for k in METHOD_KEYWORDS) or any(k in lower_body for k in METHOD_KEYWORDS):
            sections.append(f"## {head_text or 'Methodology'}\n{text[:4000]}")
        elif len(sections) < 3 and len(text) > 200:
            # Include introductory context if early
            sections.append(f"## {head_text or 'Introduction'}\n{text[:4000]}")
        elif len(text) > 300:
            # Retain other substantive sections
            sections.append(f"## {head_text or 'Section'}\n{text[:4000]}")

    # 4. Tables and Figures
    tables = []
    for fig in soup.find_all(["figure", "table"]):
        fig_type = fig.get("type") or ""
        cap = fig.find("figDesc")
        cap_text = cap.get_text(strip=True) if cap else ""
        head = fig.find("head")
        head_text = head.get_text(strip=True) if head else ""
        full_cap = f"{head_text} {cap_text}".strip() if head_text else cap_text
        if not fig_type:
            if re.search(r"(?i)\btable\b", full_cap) or fig.name == "table":
                fig_type = "Table"
            elif re.search(r"(?i)\bfig(?:ure)?\b", full_cap):
                fig_type = "Figure"
            else:
                fig_type = "Figure/Table"
        if full_cap:
            tables.append(f"[{fig_type}]: {full_cap[:300]}")

    return {
        "title": title or fallback_title,
        "abstract": abstract,
        "sections": sections[:12],
        "tables": tables[:8],
    }


def _find_cached_tei(pdf_path: Any) -> Optional[str]:
    """
    Checks if cached GROBID TEI XML already exists on disk for a PDF.
    Searches via find_tei_for_seed() as well as data/raw/grobid_output/ and data/audit/tei/.
    """
    if not pdf_path or not isinstance(pdf_path, (str, bytes, os.PathLike)):
        return None
    if isinstance(pdf_path, bytes):
        try:
            pdf_path = pdf_path.decode("utf-8", errors="replace")
        except Exception:
            return None
    if isinstance(pdf_path, os.PathLike):
        pdf_path = os.fspath(pdf_path)

    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    if not stem:
        return None

    # 1. Check via find_tei_for_seed from seed_audit
    try:
        found = find_tei_for_seed(pdf_path)
        if found and os.path.exists(found):
            found_base = os.path.basename(found)
            if found_base.startswith(f"{stem}.") or (len(stem) >= 6 and stem in found_base):
                return found
    except Exception as exc:
        logger.debug("find_tei_for_seed error for %s: %s", pdf_path, exc)

    # 2. Check search directories

    search_dirs = [
        getattr(config, "GROBID_TEI_DIR", os.path.join(config.DATA_DIR, "raw", "grobid_output")),
        os.path.join(config.DATA_DIR, "raw", "grobid_output"),
        os.path.join(config.DATA_DIR, "audit", "tei"),
        os.path.join(config.DATA_DIR, "audit"),
    ]
    for sdir in search_dirs:
        if not os.path.exists(sdir):
            continue
        for suffix in [".grobid.tei.xml", ".tei.xml", ".references.tei.xml", ".fulltext.tei.xml", ".xml"]:
            candidate = os.path.join(sdir, f"{stem}{suffix}")
            if os.path.exists(candidate):
                return candidate
        # Only fuzzy glob if stem is long enough to avoid false-positive single/double-letter matches
        if len(stem) >= 6:
            escaped_stem = glob.escape(stem)
            matches = glob.glob(os.path.join(sdir, f"*{escaped_stem}*.tei.xml"))
            if matches:
                return matches[0]

    return None


def _extract_from_pdf(source: Any, fallback_title: str = "Scientific Paper") -> Dict[str, Any]:
    """
    Extracts text blocks, title, abstract, section headings, and table/figure captions
    from a PDF file, stream, or bytes using PyMuPDF.
    """
    import fitz

    heading_re = re.compile(
        rf"^(?:(?:\d+(?:\.\d+)*|(?:{ROMAN_PAT}))[\.\s]+)?([A-Z][A-Za-z0-9\s,\-–:]{{2,60}})$"
    )
    caption_re = re.compile(
        r"(?i)\b((?:Fig(?:ure)?|Table)\s*[\dIVX]+)[:\.\-\s]+"
    )

    doc = None
    opened_here = True
    try:
        if hasattr(source, "page_count") and hasattr(source, "load_page"):
            doc = source
            opened_here = False
        elif isinstance(source, (bytes, bytearray)):
            if len(source) == 0:
                logger.warning("PDF source bytes are empty.")
                return {
                    "title": fallback_title or "Scientific Paper",
                    "abstract": "",
                    "sections": [],
                    "tables": [],
                }
            doc = fitz.open(stream=source, filetype="pdf")
        elif isinstance(source, str):
            if os.path.isfile(source):
                if os.path.getsize(source) == 0:
                    logger.warning("PDF file %s is empty (0 bytes).", source)
                    return {
                        "title": fallback_title or os.path.basename(source),
                        "abstract": "",
                        "sections": [],
                        "tables": [],
                    }
                doc = fitz.open(source)
            else:
                raw_bytes = source.encode("latin1", errors="replace")
                if len(raw_bytes) == 0:
                    return {
                        "title": fallback_title or "Scientific Paper",
                        "abstract": "",
                        "sections": [],
                        "tables": [],
                    }
                doc = fitz.open(stream=raw_bytes, filetype="pdf")
        else:
            raise ValueError(f"Unsupported PDF source: {type(source)}")

        # Check encryption / password protection
        if doc.needs_pass or (doc.is_encrypted and not doc.authenticate("")):
            logger.warning("PDF document is encrypted or requires password; cannot extract text.")
            return {
                "title": fallback_title or "Encrypted PDF",
                "abstract": "",
                "sections": [],
                "tables": [],
            }

        # 1. Title: Metadata or first prominent block / filename
        title = ""
        meta = doc.metadata or {}
        meta_title = (meta.get("title") or "").strip()
        if (
            meta_title
            and len(meta_title) > 3
            and not re.search(r"(?i)(\.eps|\.pdf|\.ps|\.dvi|\.tex|\.doc|\.png|\.tif|arxiv|doi|untitled|default|ieee|springer|gnuplot|latex)", meta_title)
        ):
            title = meta_title

        if not title and len(doc) > 0:
            for page_idx in range(min(2, len(doc))):
                blocks = doc[page_idx].get_text("blocks")
                for b in blocks:
                    if b[6] != 0:
                        continue
                    tb = b[4].strip()
                    lines = [l.strip() for l in tb.splitlines() if l.strip()]
                    if not lines:
                        continue
                    first_line = lines[0]
                    if re.search(r"(?i)^(arxiv|doi|ieee|springer|elsevier|nature|biorxiv|proceedings|volume|issn|hal|oist)", first_line):
                        continue
                    cand = " ".join(lines)
                    if AFFILIATION_RE.search(cand):
                        continue
                    if first_line.lower().startswith("title:"):
                        title = cand[6:].strip()
                        break
                    if 5 <= len(cand) <= 250 and not cand.lower().startswith(("abstract", "summary")):
                        title = cand
                        break
                if title:
                    break

        if not title:
            title = fallback_title or "Scientific Paper"

        # 2. Abstract or Summary
        abstract = ""

        # 2a. Priority: Block-level search on first 2 pages
        if len(doc) > 0:
            for page_idx in range(min(2, len(doc))):
                for b in doc[page_idx].get_text("blocks"):
                    if b[6] != 0:
                        continue
                    tb = b[4].strip()
                    if re.match(r"(?i)^(?:abstract|summary|a\s*b\s*s\s*t\s*r\s*a\s*c\s*t|s\s*u\s*m\s*m\s*a\s*r\s*y)\b", tb):
                        cand = re.sub(r"(?i)^(?:abstract|summary|a\s*b\s*s\s*t\s*r\s*a\s*c\s*t|s\s*u\s*m\s*m\s*a\s*r\s*y)[\s:—–\-]*", "", tb).strip()
                        cand = " ".join(cand.split())
                        if len(cand) > 15:
                            abstract = cand[:3000]
                            break
                if abstract:
                    break

        # 2b. Secondary: Regex search on first 3 pages if block search did not match
        if not abstract and len(doc) > 0:
            first_pages_text = "\n".join(doc[i].get_text("text") for i in range(min(3, len(doc))))
            abs_match = re.search(
                r"(?i)\b(?:abstract|summary|a\s+b\s+s\s+t\s+r\s+a\s+c\s+t|s\s+u\s+m\s+m\s+a\s+r\s+y)\b[\s:—–\-]*(.+?)(?=\n\s*(?:(?:\d+\.?\d*|(?:(?:" + ROMAN_PAT + r"))[\.\s]+)[A-Z]|introduction|keywords|key\s*words|index terms|categories|\#\#|\n\n|\Z))",
                first_pages_text,
                re.DOTALL,
            )
            if abs_match:
                cand = " ".join(abs_match.group(1).split()).strip()
                if len(cand) > 20:
                    abstract = cand[:3000]

        # 3. Section Headings and Text
        NUM_PREFIX_RE = re.compile(
            rf"^(?:(?:\d+(?:\.\d+)*|(?:{ROMAN_PAT}))\.?|[A-Z]\.|Section\s+\d+\.?)$",
            re.IGNORECASE,
        )
        raw_sections: list[tuple[str, list[str]]] = []
        current_head: Optional[str] = None
        current_paras: list[str] = []
        tables: list[str] = []
        all_body_blocks: list[str] = []

        for page in doc:
            for b in page.get_text("blocks"):
                if b[6] != 0:
                    continue
                text = b[4].strip()
                if not text or len(text) < 3:
                    continue

                # Check if this text block is a caption
                m_cap = caption_re.search(text)
                if m_cap:
                    label = m_cap.group(1).strip()
                    caption = " ".join(text[m_cap.end():].split()).strip()
                    tag = "Table" if "table" in label.lower() else "Figure"
                    full_caption = f"{label}: {caption}" if caption else label
                    if len(full_caption) > 5 and len(tables) < 5:
                        tables.append(f"[{tag}]: {full_caption[:300]}")
                    continue

                lines = [line.strip() for line in text.splitlines() if line.strip()]
                first_line = lines[0] if lines else ""

                # Skip affiliations / author metadata lines before headings
                if current_head is None and AFFILIATION_RE.search(text) and len(lines) <= 6:
                    continue

                norm_p = " ".join(text.split())
                if len(norm_p) > 20 and not AFFILIATION_RE.search(norm_p):
                    all_body_blocks.append(norm_p)

                is_head = False
                heading_text = ""

                if len(lines) == 1 and len(first_line) < 80:
                    m = heading_re.match(first_line)
                    if m:
                        lower_h = first_line.lower()
                        if (
                            any(k in lower_h for k in METHOD_KEYWORDS)
                            or any(k in lower_h for k in ("introduction", "background", "related work", "overview", "discussion", "conclusion"))
                            or SECTION_PREFIX_RE.match(first_line)
                        ):
                            if not AFFILIATION_RE.search(first_line):
                                is_head = True
                                heading_text = first_line

                elif len(lines) >= 2:
                    if NUM_PREFIX_RE.match(first_line) and len(first_line) < 20:
                        cand_head = f"{first_line} {lines[1]}"
                        if SECTION_PREFIX_RE.match(cand_head) and not AFFILIATION_RE.search(cand_head):
                            is_head = True
                            heading_text = cand_head
                            if len(lines) > 2:
                                rest_title = " ".join(lines[1:])
                                if len(rest_title) < 80 and rest_title.isupper():
                                    heading_text = f"{first_line} {rest_title}"
                                    rest_text = ""
                                else:
                                    rest_text = " ".join(lines[2:]).strip()
                            else:
                                rest_text = ""
                            if current_head is not None and current_paras:
                                raw_sections.append((current_head, current_paras))
                            current_head = heading_text
                            current_paras = [rest_text] if rest_text else []
                            continue
                    elif len(first_line) < 60:
                        lower_h = first_line.lower()
                        if (
                            SECTION_PREFIX_RE.match(first_line)
                            or any(k in lower_h for k in ("methodology", "model architecture", "experimental setup"))
                        ):
                            if not AFFILIATION_RE.search(first_line):
                                is_head = True
                                heading_text = first_line
                                rest_text = " ".join(lines[1:]).strip()
                                if current_head is not None and current_paras:
                                    raw_sections.append((current_head, current_paras))
                                current_head = heading_text
                                current_paras = [rest_text] if rest_text else []
                                continue

                if is_head:
                    if current_head is not None and current_paras:
                        raw_sections.append((current_head, current_paras))
                    current_head = heading_text
                    current_paras = []
                else:
                    if len(norm_p) > 10 and current_head is not None:
                        current_paras.append(norm_p)

        if current_head is not None and current_paras:
            raw_sections.append((current_head, current_paras))

        # 4. Fallback abstract: prominent paragraph on page 0 before first heading
        if not abstract and len(doc) > 0:
            for b in doc[0].get_text("blocks"):
                if b[6] != 0:
                    continue
                tb = b[4].strip()
                norm_b = " ".join(tb.split())
                if (
                    len(norm_b) >= 60
                    and ". " in norm_b
                    and norm_b != title
                    and not AFFILIATION_RE.search(norm_b)
                    and not caption_re.search(tb)
                    and norm_b.count(",") < max(3, len(norm_b.split()) // 4)
                    and not (raw_sections and norm_b.startswith(raw_sections[0][0]))
                ):
                    abstract = norm_b[:3000]
                    break

        sections = []
        for head, paras in raw_sections:
            body_text = "\n\n".join(paras).strip()
            if not body_text or len(body_text) < 15:
                continue
            head_lower = head.lower()
            body_sample = body_text[:400].lower()
            if any(k in head_lower for k in METHOD_KEYWORDS) or any(k in body_sample for k in METHOD_KEYWORDS):
                sections.append(f"## {head}\n{body_text[:2500]}")
            elif len(sections) < 3 and len(body_text) > 15:
                sections.append(f"## {head}\n{body_text[:2500]}")
            elif len(body_text) > 200:
                sections.append(f"## {head}\n{body_text[:2500]}")

        # Fallback if no sections detected
        if not sections:
            source_blocks = []
            if raw_sections:
                source_blocks = ["\n\n".join(p) for _, p in raw_sections]
            elif all_body_blocks:
                source_blocks = all_body_blocks

            if source_blocks:
                all_text = "\n\n".join(source_blocks)
                chunk_size = 2000
                for i in range(0, min(len(all_text), 12000), chunk_size):
                    chunk = all_text[i : i + chunk_size].strip()
                    if chunk:
                        sections.append(f"## Section {len(sections)+1}\n{chunk}")

        return {
            "title": title,
            "abstract": abstract,
            "sections": sections[:12],
            "tables": tables[:8],
        }

    except Exception as exc:
        logger.warning("PyMuPDF extraction failed (%s); falling back to plain text.", exc)
        raw_preview = ""
        is_file = isinstance(source, str) and os.path.isfile(source)
        if is_file:
            try:
                with open(source, "rb") as f:
                    sample = f.read(4096)
                is_corrupt_bin = sample.startswith(b"%PDF-") or (source.lower().endswith(".pdf") and _is_binary_content(sample))
                if not is_corrupt_bin:
                    with open(source, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read(12000)
                    clean_str = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", content)
                    raw_preview = " ".join(clean_str.split())[:12000]
            except Exception:
                raw_preview = ""
        elif isinstance(source, str):
            if not source.startswith("%PDF-"):
                clean_str = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", source)
                raw_preview = " ".join(clean_str.split())[:12000]
        elif isinstance(source, (bytes, bytearray)):
            if not _is_binary_content(source[:4096]):
                try:
                    decoded = source.decode("utf-8", errors="replace")
                    raw_preview = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", decoded)
                    raw_preview = " ".join(raw_preview.split())[:12000]
                except Exception:
                    raw_preview = ""
        return {
            "title": fallback_title or "Scientific Paper",
            "abstract": "",
            "sections": [raw_preview] if raw_preview else [],
            "tables": [],
        }
    finally:
        if doc and opened_here:
            try:
                doc.close()
            except Exception:
                pass


def _extract_from_plain_text(text: Any, fallback_title: str = "Scientific Paper") -> Dict[str, Any]:
    """Extracts planning context from plain text."""
    if not text or not isinstance(text, str):
        text = str(text) if text is not None else ""
    clean_text = text.strip()
    return {
        "title": fallback_title or "Scientific Paper",
        "abstract": "",
        "sections": [clean_text[:12000]] if clean_text else [],
        "tables": [],
    }


def extract_paper_planning_context(tei_source: Any) -> Dict[str, Any]:
    """
    Extracts high-value methodological sections, abstract, equations, and tables
    from a paper's TEI XML, PDF, or plain text.
    Handles file paths, PathLike objects, file-like streams, raw text, and binary streams safely without decoding crashes.
    """
    # 0. Handle None or empty
    if tei_source is None:
        return _extract_from_plain_text("", fallback_title="Scientific Paper")

    # Handle fitz.Document object directly
    if hasattr(tei_source, "page_count") and hasattr(tei_source, "load_page"):
        return _extract_from_pdf(tei_source, fallback_title="Scientific Paper")

    # Capture origin filename if tei_source is a file-like object with .name attribute
    origin_name = getattr(tei_source, "name", None)
    if not isinstance(origin_name, (str, os.PathLike)):
        origin_name = None
    else:
        origin_name = os.fspath(origin_name)

    # 1. Normalize PathLike to str
    if isinstance(tei_source, os.PathLike):
        tei_source = os.fspath(tei_source)

    # 2. Handle file-like objects (e.g. io.BytesIO, io.BufferedReader, Streamlit UploadedFile)
    if hasattr(tei_source, "getvalue") and callable(tei_source.getvalue):
        try:
            tei_source = tei_source.getvalue()
        except Exception:
            pass
    elif hasattr(tei_source, "read") and callable(tei_source.read):
        try:
            if hasattr(tei_source, "seek") and callable(tei_source.seek):
                tei_source.seek(0)
            tei_source = tei_source.read()
        except Exception:
            pass

    # 3. Handle in-memory bytes/bytearray directly
    if isinstance(tei_source, (bytes, bytearray)):
        stream_fallback_title = os.path.basename(origin_name) if origin_name else "Uploaded Document"
        if len(tei_source) == 0:
            return _extract_from_plain_text("", fallback_title=stream_fallback_title)

        # Check if origin_name has a cached TEI XML
        if origin_name and origin_name.lower().endswith(".pdf"):
            cached_tei = _find_cached_tei(origin_name)
            if cached_tei and os.path.exists(cached_tei):
                try:
                    with open(cached_tei, "r", encoding="utf-8", errors="replace") as f:
                        tei_content = f.read()
                    context = _extract_from_tei_xml(tei_content, fallback_title=stream_fallback_title)
                    if context and (context.get("abstract") or context.get("sections")):
                        logger.info("Found cached TEI XML for stream %s: %s", origin_name, cached_tei)
                        return context
                except Exception as exc:
                    logger.warning("Failed parsing cached TEI %s (%s); proceeding with stream.", cached_tei, exc)

        if tei_source.startswith(b"%PDF-"):
            return _extract_from_pdf(tei_source, fallback_title=stream_fallback_title)
        # Check UTF BOMs / encodings
        if tei_source.startswith(b"\xef\xbb\xbf"):
            try:
                decoded = tei_source.decode("utf-8-sig")
            except Exception:
                decoded = tei_source.decode("utf-8", errors="replace")
        elif tei_source.startswith(b"\xff\xfe\x00\x00") or tei_source.startswith(b"\x00\x00\xfe\xff"):
            try:
                decoded = tei_source.decode("utf-32")
            except Exception:
                decoded = tei_source.decode("utf-8", errors="replace")
        elif tei_source.startswith(b"\xff\xfe") or tei_source.startswith(b"\xfe\xff"):
            try:
                decoded = tei_source.decode("utf-16")
            except Exception:
                decoded = tei_source.decode("utf-8", errors="replace")
        elif b"\x00" in tei_source:
            # Binary stream or PDF without standard header
            return _extract_from_pdf(tei_source, fallback_title=stream_fallback_title)
        else:
            decoded = tei_source.decode("utf-8", errors="replace")

        if "<TEI" in decoded[:500] or decoded.strip().startswith("<"):
            return _extract_from_tei_xml(decoded, fallback_title=stream_fallback_title)
        return _extract_from_plain_text(decoded, fallback_title=stream_fallback_title)

    # 4. Check if tei_source is a file path on disk
    is_file = False
    if isinstance(tei_source, str) and "\x00" not in tei_source and "\n" not in tei_source and len(tei_source) < 4096:
        try:
            is_file = os.path.isfile(tei_source)
        except Exception:
            is_file = False

    if is_file:
        fallback_title = os.path.basename(tei_source)
        is_pdf = tei_source.lower().endswith(".pdf")
        if not is_pdf:
            # Peek first 1024 bytes for PDF magic or null bytes
            try:
                with open(tei_source, "rb") as f:
                    magic = f.read(1024)
                if magic.startswith(b"%PDF-") or (
                    b"\x00" in magic
                    and not magic.startswith((b"\xff\xfe", b"\xfe\xff", b"\xef\xbb\xbf", b"\x00\x00\xfe\xff"))
                ):
                    is_pdf = True
            except Exception:
                pass

        if is_pdf:
            # 4a. Check if cached GROBID TEI XML already exists
            cached_tei = _find_cached_tei(tei_source)
            if cached_tei and os.path.exists(cached_tei):
                try:
                    with open(cached_tei, "r", encoding="utf-8", errors="replace") as f:
                        tei_content = f.read()
                    context = _extract_from_tei_xml(tei_content, fallback_title=fallback_title)
                    if context and (context.get("abstract") or context.get("sections")):
                        logger.info("Found cached TEI XML for %s: %s", tei_source, cached_tei)
                        return context
                except Exception as exc:
                    logger.warning("Failed parsing cached TEI %s (%s); falling back to PyMuPDF.", cached_tei, exc)

            # 4b. If no TEI XML exists, extract text safely using PyMuPDF
            return _extract_from_pdf(tei_source, fallback_title=fallback_title)

        # Non-PDF file (e.g. .xml, .txt): detect BOM / encoding and NEVER open without errors="replace"
        try:
            with open(tei_source, "rb") as f:
                raw_head = f.read(4)
            if raw_head.startswith(b"\xef\xbb\xbf"):
                enc = "utf-8-sig"
            elif raw_head.startswith(b"\xff\xfe\x00\x00") or raw_head.startswith(b"\x00\x00\xfe\xff"):
                enc = "utf-32"
            elif raw_head.startswith(b"\xff\xfe") or raw_head.startswith(b"\xfe\xff"):
                enc = "utf-16"
            else:
                enc = "utf-8"
        except Exception:
            enc = "utf-8"

        with open(tei_source, "r", encoding=enc, errors="replace") as f:
            content = f.read()

        if content.startswith("%PDF-") or ("\x00" in content and enc not in ("utf-16", "utf-32")):
            logger.info("File %s contains binary/PDF signature; routing to PyMuPDF.", tei_source)
            return _extract_from_pdf(tei_source, fallback_title=fallback_title)

        if tei_source.endswith(".xml") or "<TEI" in content[:500]:
            return _extract_from_tei_xml(content, fallback_title=fallback_title)

        # Plain text file
        return _extract_from_plain_text(content, fallback_title=fallback_title)

    # 5. Not a file path on disk: check in-memory signatures first
    raw_str = str(tei_source) if tei_source is not None else ""
    if raw_str.startswith("%PDF-") or "\x00" in raw_str:
        logger.info("Raw string contains binary/PDF signature; routing to PyMuPDF.")
        return _extract_from_pdf(raw_str, fallback_title="Scientific Paper")

    if "<TEI" in raw_str[:500] or raw_str.strip().startswith("<"):
        return _extract_from_tei_xml(raw_str, fallback_title="Scientific Paper")

    # 6. String input that looked like a file path but file does not exist on disk
    if isinstance(tei_source, str) and "\n" not in tei_source and len(tei_source) < 500:
        lower = tei_source.lower()
        if not lower.startswith("<") and (
            lower.endswith((".pdf", ".xml", ".tei.xml", ".txt", ".text", ".md"))
            or ("/" in tei_source and "<" not in tei_source)
            or ("\\" in tei_source and "<" not in tei_source)
        ):
            logger.warning("Specified file path '%s' does not exist on disk.", tei_source)
            return {
                "title": os.path.basename(tei_source) or "Scientific Paper",
                "abstract": "",
                "sections": [],
                "tables": [],
            }

    # Plain raw text
    return _extract_from_plain_text(raw_str, fallback_title="Scientific Paper")


def extract_grounded_excerpts_and_domain(
    paper_title: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extracts a concise domain label and high-relevance, 1-2 sentence verbatim anchors
    (under 250 characters each) across experimental phases, preventing paragraph dumps.
    """
    ctx = context or {}
    raw_title = paper_title or ctx.get("title") or "Scientific Paper"
    t = raw_title.strip()
    for prefix in (
        "A ", "An ", "The ", "Towards ", "On ", "Inferring ", "Investigating ",
        "Benchmarking ", "Deep Learning for ", "A Study of ",
    ):
        if t.lower().startswith(prefix.lower()):
            t = t[len(prefix):].strip()
            break
    if ":" in t:
        t = t.split(":")[0].strip()
    elif "—" in t:
        t = t.split("—")[0].strip()
    elif " - " in t:
        t = t.split(" - ")[0].strip()
    if len(t) > 48:
        words = t[:48].split()
        t = " ".join(words[:-1]) if len(words) > 1 else t[:48]
    t = re.sub(r"\s+(?:from|using|with|via|in|for|and|of)$", "", t, flags=re.IGNORECASE).strip()
    domain_label = (t.title() if t.islower() else t) or "Scientific Target"

    # Keywords for phase-specific sentence matching
    phase_keywords = {
        "env": ("environment", "python", "package", "setup", "cuda", "runtime", "dependency", "library"),
        "data": (
            "disorder", "dataset", "split", "sample", "configuration", "structure factor",
            "s(k)", "parameter", "synthetic", "lead", "lattice", "fourier", "cutoff",
            "heaviside", "onsite", "stealthy", "gaussian", "amplitudes", "w_k",
        ),
        "model": ("model", "hamiltonian", "green", "rgf", "solver", "architecture", "equation", "formula", "operator", "tight-binding", "matrix"),
        "smoke": ("smoke", "gradient", "forward", "check", "finite", "dimension", "step", "verification"),
        "experiment": ("simulation", "experiment", "transport", "conductance", "transmittance", "sweep", "training", "ensemble", "realization"),
        "evaluation": ("misfit", "metric", "error", "benchmark", "accuracy", "recovery", "target", "comparison", "table", "figure", "fig"),
        "audit": ("audit", "scorecard", "checksum", "verification", "replicate", "reproducibility", "tolerance"),
    }

    excerpts: Dict[str, List[str]] = {}

    sections = ctx.get("sections") or []
    tables = ctx.get("tables") or []
    abstract = ctx.get("abstract") or ""

    # Abstract first sentence for env/overview
    abstract_sents = [
        " ".join(s.split()).strip()
        for s in re.split(r"(?<=[.!?])\s+", abstract)
        if 25 <= len(" ".join(s.split()).strip()) <= 240
    ]
    primary_abstract_sent = abstract_sents[0] if abstract_sents else f"Replication methodology for {domain_label}."

    for phase, kws in phase_keywords.items():
        best_sent = ""
        best_score = -1

        # Search tables for evaluation/audit
        if phase in ("evaluation", "audit") and tables:
            for tbl in tables:
                clean_tbl = " ".join(tbl.split()).strip()
                if 25 <= len(clean_tbl) <= 240:
                    score = sum(2 for kw in kws if kw in clean_tbl.lower())
                    if score > best_score:
                        best_score = score
                        best_sent = clean_tbl

        for sec in sections:
            lines = sec.split("\n", 1)
            head = lines[0].replace("#", "").strip().lower()
            body = lines[1] if len(lines) > 1 else ""
            head_match = sum(2 for kw in kws if kw in head)
            for s in re.split(r"(?<=[.!?])\s+", body):
                s_clean = " ".join(s.split()).strip()
                s_clean = re.sub(r"^#+\s*", "", s_clean).strip()
                if 30 <= len(s_clean) <= 250:
                    s_lower = s_clean.lower()
                    score = head_match + sum(1 for kw in kws if kw in s_lower)
                    if phase == "data" and any(m in s_lower for m in ("s(k)", "fourier", "heaviside", "gaussian", "w k", "w_k", "cutoff", "w j", "w_j")):
                        score += 3
                    if score > best_score:
                        best_score = score
                        best_sent = s_clean

        if not best_sent:
            FALLBACK_DEFAULTS = {
                "env": [primary_abstract_sent] if abstract_sents else [],
                "data": ["Dataset and preprocessing details."],
                "model": ["Model architecture and methodology."],
                "smoke": ["Model architecture and methodology."],
                "experiment": ["Experimental setup and training protocol."],
                "evaluation": ["Evaluation results and comparison."],
                "audit": ["Evaluation results and comparison."],
            }
            if not sections and not abstract and not tables:
                best_sent = FALLBACK_DEFAULTS.get(phase, [""])[0] if FALLBACK_DEFAULTS.get(phase) else ""
            elif phase == "env":
                best_sent = primary_abstract_sent if abstract_sents else ""
            elif phase == "smoke":
                best_sent = excerpts.get("model", [FALLBACK_DEFAULTS["smoke"][0]])[0]
            elif phase == "audit":
                best_sent = excerpts.get("evaluation", [FALLBACK_DEFAULTS["audit"][0]])[0]
            else:
                best_sent = primary_abstract_sent if abstract_sents else FALLBACK_DEFAULTS.get(phase, [""])[0]

        excerpts[phase] = [best_sent] if best_sent else []

    return {
        "domain_label": domain_label,
        "excerpts": excerpts,
    }


def _heuristic_task_graph(
    paper_id: str,
    paper_title: str,
    context: Optional[Dict[str, Any]] = None,
    full_phases: bool = False,
) -> TaskGraph:
    """
    Deterministic fallback task graph builder for offline testing or when LLM is unreachable.
    When full_phases is True (or context has 'full_phases': True, or TASK_GRAPH_7_PHASES=1),
    builds an authentic 7-task DAG covering all 6 mandatory scientific phases.
    When full_phases is False, builds the standard 5-task DAG for compatibility with existing tests.
    """
    ctx = context or {}
    use_7_phases = (
        full_phases
        or bool(ctx.get("full_phases"))
        or bool(ctx.get("six_phases"))
        or os.environ.get("TASK_GRAPH_7_PHASES") == "1"
    )

    if use_7_phases:
        from research_assistant.tasks.iteration import _build_heuristic_7phase_fallback
        return _build_heuristic_7phase_fallback(paper_id, paper_title, ctx)

    domain_data = extract_grounded_excerpts_and_domain(paper_title, ctx)
    domain = domain_data["domain_label"]
    excerpts = domain_data["excerpts"]

    is_hyperuniform = any(
        k in s.lower()
        for k in ("hyperuniform", "stealthy", "tight-binding disorder")
        for s in [paper_title, domain, str(ctx.get("abstract", "")), str(ctx.get("title", ""))]
    )

    graph = TaskGraph(paper_id=paper_id, paper_title=paper_title)

    t1 = Task(
        task_id="task_01_env",
        title=f"Provision Environment for {domain}",
        category=TaskCategory.ENVIRONMENT_SETUP,
        description=f"Set up Python environment, install required computational packages for {domain}, and verify CUDA/CPU limits (OMP_NUM_THREADS=1).",
        assigned_role=AgentRole.DEVOPS_AGENT,
        dependencies=[],
        inputs=["requirements.txt", "Dockerfile"],
        outputs=["env_ready.lock"],
        acceptance_criteria=["Environment builds cleanly with exit code 0", "Key scientific libraries import without error"],
        paper_context_excerpts=excerpts.get("env", []),
        artifact_contracts=[
            ArtifactContract(
                path="env_ready.lock",
                artifact_type=ArtifactType.LOCKFILE,
                description=f"Locked environment and dependencies for {domain}",
            ),
        ],
    )

    if is_hyperuniform:
        t2 = Task(
            task_id="task_02_data",
            title="Synthesize 1D Stealthy Hyperuniform Disorder Configurations & Structure Factor S(k)",
            category=TaskCategory.DATA_PREPROCESSING,
            description="Synthesize 1D stealthy hyperuniform disorder configurations w_j via Fourier filtering: (1) Discretize tight-binding chain of N=10^4 sites with wavevectors k_n = 2*pi*n/N for n in [-N/2, N/2 - 1]; (2) Impose spectral cutoff K = 2*pi*chi (chi in (0, 0.5)) with structure factor S(k) = Theta(|k| - K); (3) Generate random complex Gaussian Fourier amplitudes w_k = W * sqrt(S(k)) * eta_k with Hermitian symmetry w_{-k} = w_k* and w_0 = 0 to enforce real-space potential; (4) Apply inverse FFT to obtain real-space on-site disorder potential w_j = (1/sqrt(N)) sum_n w_{kn} exp(i k_n j) for N_conf disorder realizations.",
            assigned_role=AgentRole.DATA_ENGINEER,
            dependencies=["task_01_env"],
            inputs=["env_ready.lock", "data_spec.yaml"],
            outputs=["data/disorder_configurations.h5", "data/structure_factor.pt"],
            acceptance_criteria=[
                "Structure factor S(k) strictly vanishes (S(k) = 0) for all |k| < 2*pi*chi",
                "Generated on-site potential w_j is strictly real-valued with imaginary component norm < 1e-12",
                "Disorder configurations tensor verified non-empty with zero NaN/Inf entries",
            ],
            paper_context_excerpts=excerpts.get("data", []),
            artifact_contracts=[
                ArtifactContract(
                    path="data/disorder_configurations.h5",
                    artifact_type=ArtifactType.TENSOR,
                    shape=[1000, 10000],
                    dtype="float32",
                    description="Synthesized stealthy hyperuniform disorder realizations",
                ),
                ArtifactContract(
                    path="data/structure_factor.pt",
                    artifact_type=ArtifactType.TENSOR,
                    shape=[10000],
                    dtype="float32",
                    description="Stealthy hyperuniform structure factor S(k)",
                ),
            ],
        )

        t3 = Task(
            task_id="task_03_model",
            title="Implement Tight-Binding Hamiltonian & Recursive Green's Function (RGF) Solver",
            category=TaskCategory.MODEL_IMPLEMENTATION,
            description="Construct the 1D tight-binding Hamiltonian H = H_C + H_coupling + H_L + H_R with nearest-neighbor hopping amplitude t=1 and on-site energies w_j. Implement the Recursive Green's Function (RGF) algorithm and Caroli-Meir-Wingreen formula to compute surface Green's functions and energy-dependent transmittance T(E) with O(N) complexity.",
            assigned_role=AgentRole.MODEL_ARCHITECT,
            dependencies=["task_01_env"],
            inputs=["env_ready.lock", "model_config.yaml"],
            outputs=["src/tight_binding.py", "src/rgf_solver.py"],
            acceptance_criteria=[
                "Tight-binding Hamiltonian matrix matches open-system lead boundary conditions",
                "RGF solver forward pass computes transmittance T(E) in [0, 1] without NaN values",
            ],
            paper_context_excerpts=excerpts.get("model", []),
            artifact_contracts=[
                ArtifactContract(
                    path="src/tight_binding.py",
                    artifact_type=ArtifactType.CODE,
                    description="Tight-binding Hamiltonian and lead coupling operators",
                ),
                ArtifactContract(
                    path="src/rgf_solver.py",
                    artifact_type=ArtifactType.CODE,
                    description="Recursive Green's Function (RGF) transport solver",
                ),
            ],
        )

        t4 = Task(
            task_id="task_04_experiment",
            title="Simulate Ensemble Transmittance Spectra T(E; W, chi) & Map Misfit Cost Landscape F(W, chi)",
            category=TaskCategory.TRAINING_OR_SIMULATION,
            description="Compute disorder-averaged transmittance spectra T(E; W, chi) = (1/N_conf) sum_i T_i(E) over N=10^4 chains. Generate target spectrum T_tar(E) at (W_tar=0.10, chi_tar=0.30) with N_conf=1000. Sweep trial parameters (W, chi) with N_conf=200 and compute misfit cost landscape F(W, chi) = (1/N_E) sum_m [T(E_m; W, chi) - T_tar(E_m)]^2 over energy window [E-, E+] with single-thread capping (torch.set_num_threads(1), OMP_NUM_THREADS=1).",
            assigned_role=AgentRole.EXPERIMENT_RUNNER,
            dependencies=["task_02_data", "task_03_model"],
            inputs=["data/disorder_configurations.h5", "src/rgf_solver.py"],
            outputs=["results/transmittance_spectra.h5", "results/misfit_landscape.csv"],
            acceptance_criteria=[
                "Disorder ensemble averaging completed across target and trial parameter grid",
                "Misfit cost landscape F(W, chi) contains non-empty finite float entries across all trial pairs",
            ],
            paper_context_excerpts=excerpts.get("experiment", []),
            artifact_contracts=[
                ArtifactContract(
                    path="results/transmittance_spectra.h5",
                    artifact_type=ArtifactType.TENSOR,
                    shape=["N_trials", "N_energies"],
                    dtype="float32",
                    description="Ensemble-averaged transmittance spectra T(E)",
                ),
                ArtifactContract(
                    path="results/misfit_landscape.csv",
                    artifact_type=ArtifactType.TABULAR,
                    columns={"W": "float", "chi": "float", "misfit": "float"},
                    description="Misfit cost function landscape F(W, chi)",
                ),
            ],
        )

        t5 = Task(
            task_id="task_05_eval",
            title="Extract Inversion Parameters (W_opt, chi_opt) & Benchmark Against Paper Spectra",
            category=TaskCategory.METRIC_EVALUATION,
            description="Locate global minimum of misfit cost landscape F(W, chi) to extract recovered parameters (W_opt, chi_opt). Compare against reference target parameters (W_tar=0.10, chi_tar=0.30) within 5% error and benchmark transmittance curves against Fig. 2 and Fig. 3.",
            assigned_role=AgentRole.EVALUATION_AGENT,
            dependencies=["task_04_experiment"],
            inputs=["results/misfit_landscape.csv", "results/transmittance_spectra.h5"],
            outputs=["results/optimal_parameters.json", "results/benchmark_report.json"],
            acceptance_criteria=[
                "Parameter recovery error |W_opt - W_tar|/W_tar < 0.05 and |chi_opt - chi_tar|/chi_tar < 0.05",
                "Benchmark report cross-references Fig. 2 and Fig. 3 transmittance step locations",
            ],
            paper_context_excerpts=excerpts.get("evaluation", []),
            artifact_contracts=[
                ArtifactContract(
                    path="results/optimal_parameters.json",
                    artifact_type=ArtifactType.METRICS_JSON,
                    required_keys=["W_opt", "chi_opt", "relative_error_W", "relative_error_chi"],
                    description="Recovered stealthiness and disorder parameters",
                ),
                ArtifactContract(
                    path="results/benchmark_report.json",
                    artifact_type=ArtifactType.METRICS_JSON,
                    required_keys=["benchmark_status", "energy_window"],
                    description="Transmittance curve benchmark evaluation",
                ),
            ],
        )
    else:
        t2 = Task(
            task_id="task_02_data",
            title=f"Acquire & Preprocess {domain} Data",
            category=TaskCategory.DATA_PREPROCESSING,
            description=f"Download raw datasets and synthesize configurations for {domain}, apply normalization, and generate canonical splits.",
            assigned_role=AgentRole.DATA_ENGINEER,
            dependencies=["task_01_env"],
            inputs=["env_ready.lock", "data_spec.yaml"],
            outputs=["data/processed_data.h5"],
            acceptance_criteria=["Dataset configurations verified non-empty", "Zero missing values or NaN entries detected"],
            paper_context_excerpts=excerpts.get("data", []),
            artifact_contracts=[
                ArtifactContract(
                    path="data/processed_data.h5",
                    artifact_type=ArtifactType.TENSOR,
                    shape=[1000, 128],
                    dtype="float32",
                    description=f"Processed dataset tensor for {domain}",
                ),
            ],
        )

        t3 = Task(
            task_id="task_03_model",
            title=f"Implement {domain} Architecture & Operators",
            category=TaskCategory.MODEL_IMPLEMENTATION,
            description=f"Implement mathematical model architecture, solvers, and loss formulations for {domain}.",
            assigned_role=AgentRole.MODEL_ARCHITECT,
            dependencies=["task_01_env"],
            inputs=["env_ready.lock", "model_config.yaml"],
            outputs=["src/model.py", "src/loss.py"],
            acceptance_criteria=["Forward pass completes with correct tensor dimensions", "Loss and gradients are differentiable and finite"],
            paper_context_excerpts=excerpts.get("model", []),
            artifact_contracts=[
                ArtifactContract(
                    path="src/model.py",
                    artifact_type=ArtifactType.CODE,
                    description=f"Core model implementation for {domain}",
                ),
                ArtifactContract(
                    path="src/loss.py",
                    artifact_type=ArtifactType.CODE,
                    description=f"Loss and objective formulation for {domain}",
                ),
            ],
        )

        t4 = Task(
            task_id="task_04_experiment",
            title=f"Execute {domain} Simulation & Experiments",
            category=TaskCategory.TRAINING_OR_SIMULATION,
            description=f"Execute simulations and training runs according to paper hyperparameters for {domain} with thread capping (torch.set_num_threads(1), OMP_NUM_THREADS=1).",
            assigned_role=AgentRole.EXPERIMENT_RUNNER,
            dependencies=["task_02_data", "task_03_model"],
            inputs=["data/processed_data.h5", "src/model.py"],
            outputs=["checkpoints/model_best.pt", "logs/train_metrics.csv"],
            acceptance_criteria=["Loss and objective converge over iterations", "Simulation checkpoint saved successfully"],
            paper_context_excerpts=excerpts.get("experiment", []),
            artifact_contracts=[
                ArtifactContract(
                    path="checkpoints/model_best.pt",
                    artifact_type=ArtifactType.TENSOR,
                    shape=["parameters"],
                    dtype="float32",
                    description=f"Trained checkpoint weights for {domain}",
                ),
                ArtifactContract(
                    path="logs/train_metrics.csv",
                    artifact_type=ArtifactType.TABULAR,
                    columns={"epoch": "int", "loss": "float"},
                    description=f"Training and simulation metrics log for {domain}",
                ),
            ],
        )

        t5 = Task(
            task_id="task_05_eval",
            title=f"Compute Metrics & Validate Claims for {domain}",
            category=TaskCategory.METRIC_EVALUATION,
            description=f"Compute test evaluation metrics for {domain} and verify if results replicate values reported in the paper.",
            assigned_role=AgentRole.EVALUATION_AGENT,
            dependencies=["task_04_experiment"],
            inputs=["checkpoints/model_best.pt", "data/processed_data.h5"],
            outputs=["results/metrics.json"],
            acceptance_criteria=["Replication metrics computed within target tolerances", "Comparison table generated against reported numbers"],
            paper_context_excerpts=excerpts.get("evaluation", []),
            artifact_contracts=[
                ArtifactContract(
                    path="results/metrics.json",
                    artifact_type=ArtifactType.METRICS_JSON,
                    required_keys=["replicated_metric", "baseline_error"],
                    description=f"Replication metric scores for {domain}",
                ),
            ],
        )

    for t in [t1, t2, t3, t4, t5]:
        graph.add_task(t)

    return graph


def deconstruct_paper_into_task_graph(
    tei_or_path: Any,
    paper_id: Optional[str] = None,
    paper_title: Optional[str] = None,
    model: Optional[str] = None,
    force_heuristic: bool = False,
    full_phases: bool = False,
    build_refs: bool = False,
    workspace_dir: Optional[str] = None,
) -> TaskGraph:
    """
    Mines a scientific paper and synthesizes an executable TaskGraph DAG.
    Builds a paper-scoped reference subgraph, dereferences methodological citations,
    retrieves grounded evidence across 4 methodological facets, executes the
    ProposerCriticEngine refinement loop with fallback to heuristic synthesis,
    and compiles a grounded `TaskContextPack` for every task in the DAG.
    """
    if isinstance(tei_or_path, os.PathLike):
        tei_or_path = os.fspath(tei_or_path)

    context = extract_paper_planning_context(tei_or_path)
    title = paper_title or context.get("title") or "Scientific Paper"
    pid = paper_id
    if not pid:
        path_str = tei_or_path if isinstance(tei_or_path, str) else getattr(tei_or_path, "name", None)
        if isinstance(path_str, (str, os.PathLike)) and "\x00" not in str(path_str) and "\n" not in str(path_str) and len(str(path_str)) < 4096:
            try:
                base_name = os.path.basename(os.fspath(path_str))
                for ext in (".grobid.tei.xml", ".tei.xml", ".references.tei.xml", ".fulltext.tei.xml", ".pdf", ".xml", ".txt", ".text", ".md"):
                    if base_name.lower().endswith(ext):
                        base_name = base_name[:-len(ext)]
                        break
                base_clean = base_name.lstrip(".")
                pid = re.sub(r"[^\w\-]", "_", base_clean).strip("_") if base_clean else "paper"
            except Exception:
                pid = "paper"
        else:
            pid = "paper"
    else:
        pid = re.sub(r"[^\w\-]", "_", str(pid).lstrip(".")).strip("_")
    pid = pid or "paper"

    subgraph: Optional[Dict[str, Any]] = None
    dereferenced_citations = []
    search_resources = None

    # Wire Paper-Scoped Reference Subgraph, Citation Dereferencing & Multi-Facet Retrieval
    try:
        from research_assistant.tasks.retrieval import (
            build_paper_reference_subgraph,
            dereference_method_citations,
            format_dereferenced_citations,
            get_ingested_documents,
            load_search_resources,
            pdf_key,
            populate_graph_context_packs,
            retrieve_faceted_paper_context,
        )

        raw_target = tei_or_path if isinstance(tei_or_path, (str, bytes, os.PathLike)) else getattr(tei_or_path, "name", "")
        if raw_target:
            subgraph = build_paper_reference_subgraph(
                raw_target,
                paper_title=title,
                auto_fetch_refs=build_refs,
            )
            context["subgraph"] = subgraph

        path_str = tei_or_path if isinstance(tei_or_path, (str, os.PathLike)) else getattr(tei_or_path, "name", None)
        if path_str and isinstance(path_str, (str, os.PathLike)) and "\n" not in str(path_str) and not str(path_str).strip().startswith("<"):
            d_key = pdf_key(os.fspath(path_str))
            ingested = get_ingested_documents()
            cited_keys = set((subgraph or {}).get("cited_doc_keys") or [])
            has_grounded_docs = (d_key in ingested) or bool(cited_keys & ingested)

            if d_key in ingested:
                faceted = retrieve_faceted_paper_context(
                    d_key,
                    query_hint=title,
                    fallback_doc_filter=(cited_keys if cited_keys else None),
                )
                if faceted and faceted.get("formatted_context"):
                    context["formatted_context"] = faceted["formatted_context"]
                    context["faceted_results"] = faceted
            elif os.environ.get("TASK_ENABLE_AUTO_INGEST") == "1" or build_refs:
                from research_assistant.tasks.retrieval import check_or_ingest_paper
                doc_key = check_or_ingest_paper(tei_or_path, title=title)
                if doc_key:
                    has_grounded_docs = True
                    faceted = retrieve_faceted_paper_context(
                        doc_key,
                        query_hint=title,
                        fallback_doc_filter=(cited_keys if cited_keys else None),
                    )
                    if faceted and faceted.get("formatted_context"):
                        context["formatted_context"] = faceted["formatted_context"]
                        context["faceted_results"] = faceted

            if has_grounded_docs and subgraph and subgraph.get("method_citations"):
                try:
                    search_resources = load_search_resources()
                except Exception:
                    search_resources = None
                dereferenced_citations = dereference_method_citations(
                    subgraph,
                    search_resources=search_resources,
                )
                if dereferenced_citations:
                    context["dereferenced_citations"] = dereferenced_citations
                    deref_md = format_dereferenced_citations(dereferenced_citations)
                    if deref_md:
                        existing_fmt = context.get("formatted_context") or ""
                        context["formatted_context"] = f"{existing_fmt}\n\n{deref_md}".strip()
    except Exception as exc:
        logger.debug("Subgraph/faceted retrieval skipped or failed (%s); proceeding with local context.", exc)

    def _finalize_graph(g: TaskGraph) -> TaskGraph:
        try:
            from research_assistant.tasks.retrieval import populate_graph_context_packs
            populate_graph_context_packs(
                g,
                subgraph=subgraph,
                dereferenced_citations=dereferenced_citations,
                search_resources=search_resources,
            )
        except Exception as e:
            logger.debug("Context pack population skipped: %s", e)

        # Automatically make root dir for paper and drop all markdown files
        try:
            from research_assistant.tasks.workspace import materialize_deconstructed_paper_bundle
            materialize_deconstructed_paper_bundle(
                graph=g,
                workspace_dir=workspace_dir,
                context=context,
                tei_or_path=tei_or_path,
            )
        except Exception as e:
            logger.warning("Paper markdown bundle materialization skipped: %s", e)

        return g

    if force_heuristic or os.environ.get("CITATION_TASK_HEURISTIC_ONLY") == "1":
        logger.info("Synthesizing task graph via heuristic mode.")
        return _finalize_graph(_heuristic_task_graph(pid, title, context, full_phases=full_phases))

    # Execute ProposerCriticEngine refinement loop
    try:
        from research_assistant.tasks.iteration import ProposerCriticEngine
        engine = ProposerCriticEngine(model=model, chat_fn=chat, full_phases=full_phases)
        graph, scorecards = engine.run_refinement_loop(pid, title, context, max_turns=3)
        errors = graph.validate()
        if errors:
            logger.warning("Synthesized graph had validation errors (%s); falling back to heuristic.", errors)
            return _finalize_graph(_heuristic_task_graph(pid, title, context, full_phases=full_phases))
        logger.info("Successfully synthesized TaskGraph with %d tasks via ProposerCriticEngine.", len(graph.tasks))
        return _finalize_graph(graph)
    except Exception as exc:
        logger.warning("LLM task deconstruction failed (%s); using heuristic fallback.", exc)
        return _finalize_graph(_heuristic_task_graph(pid, title, context, full_phases=full_phases))
