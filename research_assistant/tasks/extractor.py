"""
Scientific Protocol Miner: extracts operational protocols and synthesizes Task Graphs from papers.
"""

import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Union

from bs4 import BeautifulSoup

from research_assistant import config
from research_assistant.config import LLM_MODEL
from research_assistant.shared.llm import chat
from research_assistant.shared.log import get_logger
from research_assistant.shared.seed_audit import find_tei_for_seed
from research_assistant.tasks.models import (
    AgentRole,
    Task,
    TaskCategory,
    TaskGraph,
    TaskStatus,
)
from research_assistant.tasks.prompts import (
    PROTOCOL_DECONSTRUCTION_SYSTEM,
    PROTOCOL_DECONSTRUCTION_USER,
)

logger = get_logger("task_extractor")

METHOD_KEYWORDS = (
    "method", "model", "algorithm", "architecture", "experiment",
    "setup", "implementation", "dataset", "benchmark", "evaluation",
    "simulation", "results", "analysis", "theory", "approach",
    "framework", "design", "protocol", "pipeline", "workflow",
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

    for div in candidate_divs:
        head = div.find("head")
        head_text = head.get_text(strip=True) if head else ""
        text = div.get_text(separator=" ", strip=True)
        if not text or len(text) < 50:
            continue

        if any(k in head_text.lower() for k in METHOD_KEYWORDS):
            sections.append(f"## {head_text}\n{text[:4000]}")
        elif len(sections) < 3 and len(text) > 200:
            # Include introductory context if early
            sections.append(f"## {head_text or 'Introduction'}\n{text[:4000]}")

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
        "sections": sections[:6],
        "tables": tables[:5],
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
            if any(k in head_lower for k in METHOD_KEYWORDS):
                sections.append(f"## {head}\n{body_text[:2500]}")
            elif len(sections) < 3 and len(body_text) > 15:
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
            "sections": sections[:6],
            "tables": tables[:5],
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


def _heuristic_task_graph(paper_id: str, paper_title: str, context: Dict[str, Any]) -> TaskGraph:
    """
    Deterministic fallback task graph builder for offline testing or when LLM is unreachable.
    Builds an authentic 5-task scientific replication DAG.
    """
    graph = TaskGraph(paper_id=paper_id, paper_title=paper_title)

    abstract_excerpt = [context["abstract"][:300]] if context.get("abstract") else []
    section_excerpt = (context.get("sections") or ["Dataset and preprocessing details."])[:1]
    model_excerpt = (context.get("sections") or ["Model architecture and methodology."])[:1]
    exp_excerpt = (context.get("sections") or ["Experimental setup and training protocol."])[:1]
    table_excerpt = (context.get("tables") or ["Evaluation results and comparison."])[:1]

    t1 = Task(
        task_id="task_01_env",
        title="Provision Environment & Hardware Dependencies",
        category=TaskCategory.ENVIRONMENT_SETUP,
        description="Set up Python environment, install required computational packages, and verify CUDA support.",
        assigned_role=AgentRole.DEVOPS_AGENT,
        dependencies=[],
        inputs=["requirements.txt", "Dockerfile"],
        outputs=["env_ready.lock"],
        acceptance_criteria=["Environment builds cleanly", "Key libraries import without error"],
        paper_context_excerpts=abstract_excerpt,
    )

    t2 = Task(
        task_id="task_02_data",
        title="Acquire and Preprocess Evaluation Datasets",
        category=TaskCategory.DATA_PREPROCESSING,
        description="Download raw datasets described in paper, apply normalization, and generate canonical splits.",
        assigned_role=AgentRole.DATA_ENGINEER,
        dependencies=["task_01_env"],
        inputs=["env_ready.lock", "data_spec.yaml"],
        outputs=["data/processed_data.h5"],
        acceptance_criteria=["Train/Val/Test splits verified", "Zero missing values or NaN entries"],
        paper_context_excerpts=section_excerpt,
    )

    t3 = Task(
        task_id="task_03_model",
        title="Implement Core Model Architecture & Algorithms",
        category=TaskCategory.MODEL_IMPLEMENTATION,
        description="Write the mathematical model architecture, loss functions, and optimization step.",
        assigned_role=AgentRole.MODEL_ARCHITECT,
        dependencies=["task_01_env"],
        inputs=["env_ready.lock", "model_config.yaml"],
        outputs=["src/model.py", "src/loss.py"],
        acceptance_criteria=["Forward pass completes with correct tensor dimensions", "Loss is differentiable"],
        paper_context_excerpts=model_excerpt,
    )

    t4 = Task(
        task_id="task_04_experiment",
        title="Execute Baseline and Proposed Experiments",
        category=TaskCategory.TRAINING_OR_SIMULATION,
        description="Train model according to paper hyperparameters and record training/simulation logs.",
        assigned_role=AgentRole.EXPERIMENT_RUNNER,
        dependencies=["task_02_data", "task_03_model"],
        inputs=["data/processed_data.h5", "src/model.py"],
        outputs=["checkpoints/model_best.pt", "logs/train_metrics.csv"],
        acceptance_criteria=["Loss converges over epochs", "Model weights checkpointed"],
        paper_context_excerpts=exp_excerpt,
    )

    t5 = Task(
        task_id="task_05_eval",
        title="Compute Evaluation Metrics & Replication Audit",
        category=TaskCategory.METRIC_EVALUATION,
        description="Compute test set metrics and verify if results replicate values reported in the paper.",
        assigned_role=AgentRole.EVALUATION_AGENT,
        dependencies=["task_04_experiment"],
        inputs=["checkpoints/model_best.pt", "data/processed_data.h5"],
        outputs=["results/evaluation_report.json"],
        acceptance_criteria=["Primary performance metric computed on test set", "Comparison table generated"],
        paper_context_excerpts=table_excerpt,
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
) -> TaskGraph:
    """
    Mines a scientific paper and synthesizes an executable TaskGraph DAG.
    Uses LLM with fallback to heuristic synthesis for offline environments.
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

    if force_heuristic or os.environ.get("CITATION_TASK_HEURISTIC_ONLY") == "1":
        logger.info("Synthesizing task graph via heuristic mode.")
        return _heuristic_task_graph(pid, title, context)

    # Format planning prompt
    paper_text_parts = [
        f"Abstract:\n{context.get('abstract', 'N/A')}\n",
        "\n".join(context.get("sections", [])),
    ]
    if context.get("tables"):
        paper_text_parts.append("\nKey Figures / Tables:\n" + "\n".join(context["tables"]))

    paper_text = "\n\n".join(paper_text_parts)[:10000]

    prompt = PROTOCOL_DECONSTRUCTION_USER.format(
        paper_title=title,
        paper_text=paper_text,
    )

    try:
        res = chat(
            [
                {"role": "system", "content": PROTOCOL_DECONSTRUCTION_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            model=model or LLM_MODEL,
            temperature=0.1,
        )
        raw = res.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)
        tasks_data = parsed.get("tasks", [])

        if not tasks_data or not isinstance(tasks_data, list):
            logger.warning("LLM returned no tasks; falling back to heuristic.")
            return _heuristic_task_graph(pid, title, context)

        graph = TaskGraph(paper_id=pid, paper_title=parsed.get("paper_title") or title)
        for t_dict in tasks_data:
            # Map values gracefully to enum
            try:
                cat_val = t_dict.get("category", "").lower()
                category = next(
                    (c for c in TaskCategory if c.value == cat_val),
                    TaskCategory.MODEL_IMPLEMENTATION,
                )
                role_val = t_dict.get("assigned_role", "").lower()
                role = next(
                    (r for r in AgentRole if r.value == role_val),
                    AgentRole.MODEL_ARCHITECT,
                )

                task = Task(
                    task_id=t_dict.get("task_id", f"task_{len(graph.tasks)+1}"),
                    title=t_dict.get("title", "Execution Task"),
                    category=category,
                    assigned_role=role,
                    dependencies=t_dict.get("dependencies", []),
                    description=t_dict.get("description", ""),
                    inputs=t_dict.get("inputs", []),
                    outputs=t_dict.get("outputs", []),
                    acceptance_criteria=t_dict.get("acceptance_criteria", []),
                    paper_context_excerpts=t_dict.get("paper_context_excerpts", []),
                    tools_required=t_dict.get("tools_required", []),
                )
                graph.add_task(task)
            except Exception as e:
                logger.warning("Failed to parse task %s: %s", t_dict.get("task_id"), e)

        # Validate DAG
        errors = graph.validate()
        if errors:
            logger.warning("Synthesized graph had validation errors (%s); falling back to heuristic.", errors)
            return _heuristic_task_graph(pid, title, context)

        logger.info("Successfully synthesized TaskGraph with %d tasks via LLM.", len(graph.tasks))
        return graph

    except Exception as exc:
        logger.warning("LLM task deconstruction failed (%s); using heuristic fallback.", exc)
        return _heuristic_task_graph(pid, title, context)
